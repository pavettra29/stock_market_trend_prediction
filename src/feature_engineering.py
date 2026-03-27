"""
feature_engineering.py
=======================
Adds technical indicators to cleaned stock data.

What are technical indicators?
  Formulas applied to price/volume data that help a model
  detect patterns — like momentum, trend direction, and volatility.

Indicators added:
  Trend     : SMA_20, SMA_50, EMA_12, EMA_26
  Momentum  : RSI, MACD, MACD_Signal, MACD_Histogram, Stochastic %K/%D
  Volatility: Bollinger Bands (Upper, Lower, Width, %B), ATR
  Volume    : OBV, Volume_SMA, Volume_Ratio
  Price     : Daily_Return, Log_Return, HL_Spread, OC_Spread
  Target    : 1 if price goes UP next day, 0 if DOWN

"""

import pandas as pd
import numpy as np
import os

# We use pandas-ta library for indicators (simpler than ta-lib to install)
# Install with: pip install pandas-ta
import pandas_ta as ta

ALL_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
              "TCS", "RELIANCE", "HDFCBANK"]


def add_features(file_path):
    """
    Loads a cleaned CSV and adds all technical indicators.
    Returns a DataFrame with new feature columns.
    """

    print(f"\n  Adding features to: {file_path}")

    # ── FIX 1: Use set_index("Date") NOT drop(columns=["Date"]) ────
    # Why? The Date column is NOT useless — it's your time axis.
    # set_index makes Date the row label (index) so it's available
    # for charts and time-based operations, but not passed as a
    # model feature.
    df = pd.read_csv(file_path, index_col="Date", parse_dates=True)

    # Print available columns so you know what you're working with
    print(f"  Columns: {list(df.columns)}")
    print(f"  Rows   : {len(df)}")

    # Convert Close to numeric just in case (handles any stray text)
    df["Close"]  = pd.to_numeric(df["Close"],  errors="coerce")
    df["Open"]   = pd.to_numeric(df["Open"],   errors="coerce")
    df["High"]   = pd.to_numeric(df["High"],   errors="coerce")
    df["Low"]    = pd.to_numeric(df["Low"],    errors="coerce")
    df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
    df.dropna(subset=["Close"], inplace=True)

    # ── TREND INDICATORS ───────────────────────────────────────────
    # SMA = Simple Moving Average: plain average of last N closing prices
    # EMA = Exponential Moving Average: like SMA but recent days weigh more
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["EMA_12"] = ta.ema(df["Close"], length=12)
    df["EMA_26"] = ta.ema(df["Close"], length=26)

    # ── MOMENTUM INDICATORS ────────────────────────────────────────
    # RSI: "how overbought or oversold is this stock right now?"
    #   > 70 = overbought (might fall soon)
    #   < 30 = oversold   (might rise soon)
    df["RSI"] = ta.rsi(df["Close"], length=14)

    # MACD: difference between fast EMA and slow EMA
    #   Positive = upward momentum, Negative = downward momentum
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    macd_col  = [c for c in macd.columns if c.startswith("MACD_")][0]
    macds_col = [c for c in macd.columns if c.startswith("MACDs")][0]
    macdh_col = [c for c in macd.columns if c.startswith("MACDh")][0]
    df["MACD"]           = macd[macd_col]
    df["MACD_Signal"]    = macd[macds_col]
    df["MACD_Histogram"] = macd[macdh_col]

    # Stochastic Oscillator: where is today's price relative to recent range?
    #   > 80 = overbought, < 20 = oversold
    stoch = ta.stoch(df["High"], df["Low"], df["Close"])
    stochk_col = [c for c in stoch.columns if c.startswith("STOCHk")][0]
    stochd_col = [c for c in stoch.columns if c.startswith("STOCHd")][0]
    df["Stoch_K"] = stoch[stochk_col]
    df["Stoch_D"] = stoch[stochd_col]

    # ── VOLATILITY INDICATORS ──────────────────────────────────────
    # Bollinger Bands: a "channel" around the moving average
    #   Price near upper band = high volatility upward
    #   Price near lower band = high volatility downward
    bbands = ta.bbands(df["Close"], length=20, std=2)
    # Auto-detect column names (they vary slightly across pandas_ta versions)
    # e.g. "BBU_20_2.0" or "BBU_20_2" depending on version
    bb_upper_col  = [c for c in bbands.columns if c.startswith("BBU")][0]
    bb_middle_col = [c for c in bbands.columns if c.startswith("BBM")][0]
    bb_lower_col  = [c for c in bbands.columns if c.startswith("BBL")][0]
    df["BB_Upper"]  = bbands[bb_upper_col]
    df["BB_Middle"] = bbands[bb_middle_col]
    df["BB_Lower"]  = bbands[bb_lower_col]
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Pct"]    = (df["Close"] - df["BB_Lower"]) / (
                       df["BB_Upper"] - df["BB_Lower"] + 1e-10)

    # ATR = Average True Range: how much does the price move per day?
    #   High ATR = volatile market  |  Low ATR = calm market
    df["ATR"] = ta.atr(df["High"], df["Low"], df["Close"], length=14)

    # ── VOLUME INDICATORS ─────────────────────────────────────────
    # OBV = On-Balance Volume: are people buying or selling more?
    #   Rising OBV with rising price = strong uptrend
    df["OBV"] = ta.obv(df["Close"], df["Volume"])

    # Volume SMA: average volume over 20 days
    # Volume Ratio: today's volume vs average — high ratio = unusual activity
    df["Volume_SMA"]   = ta.sma(df["Volume"], length=20)
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_SMA"] + 1e-10)

    # ── PRICE-DERIVED FEATURES ────────────────────────────────────
    # Daily Return: how much did the price change today in %?
    df["Daily_Return"] = df["Close"].pct_change() * 100

    # Log Return: mathematically cleaner version of daily return
    df["Log_Return"]   = np.log(df["Close"] / df["Close"].shift(1))

    # HL Spread: how wide was today's price range?
    df["HL_Spread"]    = (df["High"] - df["Low"]) / df["Close"] * 100

    # OC Spread: did price close higher or lower than it opened?
    df["OC_Spread"]    = (df["Close"] - df["Open"]) / df["Open"] * 100

    # ── FIX 2: ADD TARGET COLUMN ──────────────────────────────────
    # This is the label your model will learn to predict!
    #   1 = "tomorrow's Close is HIGHER than today's" (price goes UP)
    #   0 = "tomorrow's Close is LOWER or same"       (price goes DOWN)
    #
    # shift(-1) means "look at the NEXT row's Close"
    df["Target"] = (df["Close"].shift(-1) > df["Close"]).astype(int)

    # ── FIX 3: Drop NaN rows AFTER adding all indicators ─────────
    # Why? SMA_50 needs 50 days to calculate — the first 49 rows are NaN.
    # Also the last row's Target is NaN (no "next day" to compare to).
    # Drop all rows with ANY NaN value.
    rows_before = len(df)
    df.dropna(inplace=True)
    print(f"  Dropped {rows_before - len(df)} NaN rows "
          f"(from indicator warmup + last target row)")
    print(f"  Final  : {len(df)} clean rows, {df.shape[1]} features")

    return df


def engineer_all():
    """Adds features to all cleaned stock files."""

    os.makedirs("data", exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  FEATURE ENGINEERING")
    print(f"{'='*50}")

    for name in ALL_STOCKS:
        cleaned_path  = f"data/{name}_cleaned.csv"
        featured_path = f"data/{name}_featured.csv"

        if not os.path.exists(cleaned_path):
            print(f"\n  ⚠️  Skipping {name} — cleaned file not found.")
            print(f"       Run data_cleaning.py first!")
            continue

        df = add_features(cleaned_path)
        df.to_csv(featured_path)
        print(f"  ✓ Saved → {featured_path}")

    print("\n✅ Feature engineering complete!\n")


if __name__ == "__main__":
    engineer_all()