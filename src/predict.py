"""
predict.py  —  Stock Market Trend Prediction  |  Phase 6 Demo
=============================================================
Usage:
    python src/predict.py --ticker AAPL
    python src/predict.py --ticker TCS          # Indian stock → nearest US proxy
    python src/predict.py --ticker NVDA --verbose

Author : <your name>
Project: Final Year CS Project — Parul University
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings("ignore")

# ── make sure src/ is importable when called from the project root ───────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch
import pandas as pd
import yfinance as yf

# ── import everything directly from your own modules ─────────────────────────
from model import LSTMClassifier
from config import (
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "models"),
    FEATURE_COLS,       # the exact 28 columns, in training order
    LOOKBACK,           # 30
    INPUT_SIZE,         # 28
    HIDDEN_SIZE,        # 64
    DROPOUT,            # 0.2
    BIDIRECTIONAL,      # False
    DEVICE,
    US_STOCKS,          # ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
    INDIAN_STOCKS,      # ["TCS", "RELIANCE", "HDFCBANK"]
)

# ─────────────────────────────────────────────────────────────────────────────
#  PROXY MAP  —  Indian stock → closest US trained model
#  (sector / market-cap rationale shown in comments)
# ─────────────────────────────────────────────────────────────────────────────

INDIAN_TO_US_PROXY: dict[str, str] = {
    "TCS":       "MSFT",    # large-cap IT services
    "RELIANCE":  "AMZN",    # diversified conglomerate (energy + retail + telecom)
    "HDFCBANK":  "MSFT",    # large-cap financial → closest stable mega-cap
}

NSE_SUFFIX = ".NS"


# ─────────────────────────────────────────────────────────────────────────────
#  FEATURE ENGINEERING
#  Matches FEATURE_COLS from config.py exactly — same names, same order.
# ─────────────────────────────────────────────────────────────────────────────

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-creates the 28 features from config.FEATURE_COLS on raw OHLCV data.
    Column order matches training exactly so the model receives the right input.
    """
    close  = df["Close"].squeeze()
    high   = df["High"].squeeze()
    low    = df["Low"].squeeze()
    open_  = df["Open"].squeeze()
    volume = df["Volume"].squeeze()

    feat = pd.DataFrame(index=df.index)

    # ── Raw OHLCV ─────────────────────────────────────────────────────────────
    feat["Open"]   = open_
    feat["High"]   = high
    feat["Low"]    = low
    feat["Close"]  = close
    feat["Volume"] = volume

    # ── Moving Averages ───────────────────────────────────────────────────────
    feat["SMA_20"] = close.rolling(20).mean()
    feat["SMA_50"] = close.rolling(50).mean()
    feat["EMA_12"] = close.ewm(span=12, adjust=False).mean()
    feat["EMA_26"] = close.ewm(span=26, adjust=False).mean()

    # ── RSI (14) ──────────────────────────────────────────────────────────────
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    feat["RSI"] = 100 - (100 / (1 + rs))

    # ── MACD ──────────────────────────────────────────────────────────────────
    feat["MACD"]           = feat["EMA_12"] - feat["EMA_26"]
    feat["MACD_Signal"]    = feat["MACD"].ewm(span=9, adjust=False).mean()
    feat["MACD_Histogram"] = feat["MACD"] - feat["MACD_Signal"]

    # ── Stochastic Oscillator (%K, %D) ────────────────────────────────────────
    low14  = low.rolling(14).min()
    high14 = high.rolling(14).max()
    feat["Stoch_K"] = 100 * (close - low14) / (high14 - low14 + 1e-9)
    feat["Stoch_D"] = feat["Stoch_K"].rolling(3).mean()

    # ── Bollinger Bands ───────────────────────────────────────────────────────
    bb_mid             = close.rolling(20).mean()
    bb_std             = close.rolling(20).std()
    feat["BB_Upper"]   = bb_mid + 2 * bb_std
    feat["BB_Middle"]  = bb_mid
    feat["BB_Lower"]   = bb_mid - 2 * bb_std
    feat["BB_Width"]   = (feat["BB_Upper"] - feat["BB_Lower"]) / (bb_mid + 1e-9)
    feat["BB_Pct"]     = (close - feat["BB_Lower"]) / (feat["BB_Upper"] - feat["BB_Lower"] + 1e-9)

    # ── ATR (14) ──────────────────────────────────────────────────────────────
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    feat["ATR"] = tr.rolling(14).mean()

    # ── OBV ───────────────────────────────────────────────────────────────────
    feat["OBV"] = (np.sign(close.diff()) * volume).fillna(0).cumsum()

    # ── Volume indicators ─────────────────────────────────────────────────────
    feat["Volume_SMA"]   = volume.rolling(10).mean()
    feat["Volume_Ratio"] = volume / (feat["Volume_SMA"] + 1e-9)

    # ── Return features ───────────────────────────────────────────────────────
    feat["Daily_Return"] = close.pct_change()
    feat["Log_Return"]   = np.log(close / close.shift(1))

    # ── Spread features ───────────────────────────────────────────────────────
    feat["HL_Spread"] = high - low
    feat["OC_Spread"] = open_ - close

    # ── Drop NaN rows (SMA_50 is the binding constraint — needs 50 rows) ──────
    feat.dropna(inplace=True)

    # ── Enforce exact column order from config.FEATURE_COLS ──────────────────
    feat = feat[FEATURE_COLS]

    assert feat.shape[1] == INPUT_SIZE, (
        f"Feature count mismatch: got {feat.shape[1]}, expected {INPUT_SIZE}."
    )

    return feat


# ─────────────────────────────────────────────────────────────────────────────
#  DATA DOWNLOAD
# ─────────────────────────────────────────────────────────────────────────────

def download_recent_data(yf_ticker: str) -> pd.DataFrame:
    """
    Download enough history to cover LOOKBACK trading days after feature
    engineering. SMA_50 consumes 50 rows, so we fetch ~150 calendar days
    worth of data to be safe.
    """
    fetch_days = (LOOKBACK + 80) * 2      # calendar days ≈ 2× trading days
    print(f"  Downloading {yf_ticker} ...", end=" ", flush=True)
    df = yf.download(yf_ticker, period=f"{fetch_days}d",
                     auto_adjust=True, progress=False)
    if df.empty:
        raise ValueError(
            f"No data returned for '{yf_ticker}'.\n"
            "  US stocks  : use plain ticker, e.g. AAPL\n"
            "  Indian stocks: bare name works too, e.g. TCS  (script adds .NS)"
        )
    # yfinance can return MultiIndex columns — flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    print(f"got {len(df)} trading days.")
    return df


# ─────────────────────────────────────────────────────────────────────────────
#  NORMALISATION  —  column-wise min-max on the lookback window
#
#  If you saved a fitted scaler in Phase 2/3 (e.g. AAPL_scaler.pkl), swap in:
#      import joblib
#      scaler = joblib.load(os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "models"), f"{model_ticker}_scaler.pkl"))
#      return scaler.transform(window)
# ─────────────────────────────────────────────────────────────────────────────

def normalise(window: np.ndarray) -> np.ndarray:
    """Column-wise min-max normalisation on a (LOOKBACK, INPUT_SIZE) array."""
    col_min = window.min(axis=0, keepdims=True)
    col_max = window.max(axis=0, keepdims=True)
    denom   = col_max - col_min
    denom[denom == 0] = 1e-9
    return (window - col_min) / denom


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_ticker: str) -> LSTMClassifier:
    """Load the best checkpoint for `model_ticker` from MODEL_DIR."""
    path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "models"), f"{model_ticker}_best.pt")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Checkpoint not found: {path}\n"
            f"Make sure Phase 3 training completed successfully for {model_ticker}."
        )
    model = LSTMClassifier(
        input_size    = INPUT_SIZE,
        hidden_size   = HIDDEN_SIZE,
        dropout       = DROPOUT,
        bidirectional = BIDIRECTIONAL,
    )
    state = torch.load(path, map_location=DEVICE, weights_only=True)
    # handle both a raw state_dict and a wrapped checkpoint dict
    if isinstance(state, dict) and "model_state" in state:
     state = state["model_state"]
    elif isinstance(state, dict) and "model_state_dict" in state:
     state = state["model_state_dict"] 
    model.load_state_dict(state)
    model.to(DEVICE)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
#  CORE PREDICTION PIPELINE
# ─────────────────────────────────────────────────────────────────────────────

def predict(ticker: str, verbose: bool = False) -> dict:
    ticker_upper = ticker.upper().strip()

    # ── 1. Resolve yfinance ticker & proxy model ──────────────────────────────
    bare_indian  = ticker_upper.removesuffix(NSE_SUFFIX)
    is_indian    = (ticker_upper in INDIAN_STOCKS or
                    bare_indian   in INDIAN_STOCKS)

    if ticker_upper in US_STOCKS:
        yf_ticker    = ticker_upper
        model_ticker = ticker_upper
        proxy_used   = False

    elif is_indian:
        yf_ticker    = bare_indian + NSE_SUFFIX
        model_ticker = INDIAN_TO_US_PROXY.get(bare_indian, "MSFT")
        proxy_used   = True

    else:
        yf_ticker    = ticker_upper
        model_ticker = "AAPL"
        proxy_used   = True
        print(f"  Warning: '{ticker_upper}' not in trained list. Using AAPL as proxy.")

    if verbose:
        print(f"\n  Ticker      : {ticker_upper}  (yfinance: {yf_ticker})")
        print(f"  Model       : {model_ticker}" + ("  [proxy]" if proxy_used else ""))
        print(f"  Device      : {DEVICE}")
        print(f"  Lookback    : {LOOKBACK} days  |  Features: {INPUT_SIZE}")

    # ── 2. Download raw OHLCV ─────────────────────────────────────────────────
    raw_df  = download_recent_data(yf_ticker)

    # ── 3. Compute the 28 features ────────────────────────────────────────────
    feat_df = compute_features(raw_df)

    if len(feat_df) < LOOKBACK:
        raise ValueError(
            f"Only {len(feat_df)} rows after feature engineering — need {LOOKBACK}.\n"
            "Try a ticker with longer trading history."
        )

    # ── 4. Grab the most recent LOOKBACK rows → shape (LOOKBACK, INPUT_SIZE) ──
    window = feat_df.iloc[-LOOKBACK:].values.astype(np.float32)

    if verbose:
        print(f"  Window      : rows {len(feat_df)-LOOKBACK}–{len(feat_df)-1} "
              f"of feat_df  →  shape {window.shape}")

    # ── 5. Normalise ──────────────────────────────────────────────────────────
    window_norm = normalise(window)                           # (30, 28)

    # ── 6. Build input tensor  (batch=1, seq=30, features=28) ────────────────
    x = torch.tensor(window_norm, dtype=torch.float32).unsqueeze(0).to(DEVICE)

    # ── 7. Load model & run forward pass ─────────────────────────────────────
    model = load_model(model_ticker)
    with torch.no_grad():
        prob_up = model(x).item()     # Sigmoid lives inside classifier → [0, 1]

    # ── 8. Interpret result ───────────────────────────────────────────────────
    direction  = "UP"  if prob_up >= 0.5 else "DOWN"
    confidence = prob_up if direction == "UP" else (1.0 - prob_up)

    return {
        "ticker":      ticker_upper,
        "yf_ticker":   yf_ticker,
        "direction":   direction,
        "probability": confidence * 100,
        "model_used":  model_ticker,
        "proxy_used":  proxy_used,
        "raw_prob":    prob_up,
    }


# ─────────────────────────────────────────────────────────────────────────────
#  OUTPUT FORMATTING
# ─────────────────────────────────────────────────────────────────────────────

def print_result(r: dict, verbose: bool = False) -> None:
    GREEN = "\033[92m"
    RED   = "\033[91m"
    BOLD  = "\033[1m"
    RESET = "\033[0m"

    color = GREEN if r["direction"] == "UP" else RED
    arrow = "▲"   if r["direction"] == "UP" else "▼"

    print()
    print("━" * 52)
    print(f"  Prediction for  {r['ticker']}")
    print("━" * 52)
    if r["proxy_used"]:
        print(f"  Proxy model  :  {r['model_used']}  "
              f"(no trained model for {r['ticker']})")
        print()
    print(f"  Trend        :  {BOLD}{color}{arrow}  {r['direction']}{RESET}")
    print(f"  Confidence   :  {BOLD}{r['probability']:.1f}%{RESET}")
    if verbose:
        print()
        print(f"  Raw sigmoid  :  {r['raw_prob']:.6f}")
        print(f"  Model file   :  {r['model_used']}_best.pt")
        print(f"  yf ticker    :  {r['yf_ticker']}")
    print("━" * 52)
    print()


# ─────────────────────────────────────────────────────────────────────────────
#  CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Stock Market Trend Predictor — Parul University FYP Phase 6"
    )
    p.add_argument("--ticker", "-t", required=True,
                   help="Ticker symbol, e.g.  AAPL  NVDA  TCS  RELIANCE")
    p.add_argument("--verbose", "-v", action="store_true",
                   help="Show debug info: window shape, raw sigmoid, device")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print()
    print("  Stock Market Trend Prediction  |  Parul University FYP")
    print(f"  Ticker : {args.ticker.upper()}")
    print()

    try:
        result = predict(args.ticker, verbose=args.verbose)
        print_result(result, verbose=args.verbose)
    except FileNotFoundError as e:
        print(f"\n  ERROR  {e}\n")
        sys.exit(1)
    except ValueError as e:
        print(f"\n  ERROR  {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n  Unexpected error: {e}")
        if args.verbose:
            import traceback; traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()