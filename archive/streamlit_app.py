import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pandas_ta as ta
import yfinance as yf
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stock Trend Predictor",
    page_icon="📈",
    layout="centered"
)

# ── Model ────────────────────────────────────────────────────────────────────
class LSTMClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=28, hidden_size=64,
                            num_layers=1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid()
        )
    def forward(self, x):
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1, :]).squeeze(1)

MODELS_DIR   = os.path.join(os.path.dirname(__file__), "models")
US_STOCKS    = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
INDIAN_MAP   = {"TCS": "MSFT", "TCS.NS": "MSFT",
                "RELIANCE": "AMZN", "RELIANCE.NS": "AMZN",
                "HDFCBANK": "MSFT", "HDFCBANK.NS": "MSFT"}
FEATURE_COLS = [
    "Open","High","Low","Close","Volume",
    "SMA_20","SMA_50","EMA_12","EMA_26",
    "RSI","MACD","MACD_Signal","MACD_Histogram",
    "Stoch_K","Stoch_D",
    "BB_Upper","BB_Middle","BB_Lower","BB_Width","BB_Pct",
    "ATR","OBV","Volume_SMA","Volume_Ratio",
    "Daily_Return","Log_Return","HL_Spread","OC_Spread"
]

@st.cache_resource
def load_model(key):
    path  = os.path.join(MODELS_DIR, f"{key}_best.pt")
    model = LSTMClassifier()
    ckpt  = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model

def compute_features(df):
    df = df.copy()
    df["SMA_20"] = ta.sma(df["Close"], length=20)
    df["SMA_50"] = ta.sma(df["Close"], length=50)
    df["EMA_12"] = ta.ema(df["Close"], length=12)
    df["EMA_26"] = ta.ema(df["Close"], length=26)
    df["RSI"]    = ta.rsi(df["Close"], length=14)
    macd = ta.macd(df["Close"], fast=12, slow=26, signal=9)
    df["MACD"]           = macd[[c for c in macd.columns if c.startswith("MACD_")][0]]
    df["MACD_Signal"]    = macd[[c for c in macd.columns if c.startswith("MACDs")][0]]
    df["MACD_Histogram"] = macd[[c for c in macd.columns if c.startswith("MACDh")][0]]
    stoch = ta.stoch(df["High"], df["Low"], df["Close"])
    df["Stoch_K"] = stoch[[c for c in stoch.columns if c.startswith("STOCHk")][0]]
    df["Stoch_D"] = stoch[[c for c in stoch.columns if c.startswith("STOCHd")][0]]
    bb = ta.bbands(df["Close"], length=20, std=2)
    df["BB_Upper"]  = bb[[c for c in bb.columns if c.startswith("BBU")][0]]
    df["BB_Middle"] = bb[[c for c in bb.columns if c.startswith("BBM")][0]]
    df["BB_Lower"]  = bb[[c for c in bb.columns if c.startswith("BBL")][0]]
    df["BB_Width"]  = (df["BB_Upper"] - df["BB_Lower"]) / df["BB_Middle"]
    df["BB_Pct"]    = (df["Close"] - df["BB_Lower"]) / (df["BB_Upper"] - df["BB_Lower"] + 1e-10)
    df["ATR"]          = ta.atr(df["High"], df["Low"], df["Close"], length=14)
    df["OBV"]          = ta.obv(df["Close"], df["Volume"])
    df["Volume_SMA"]   = ta.sma(df["Volume"], length=20)
    df["Volume_Ratio"] = df["Volume"] / (df["Volume_SMA"] + 1e-10)
    df["Daily_Return"] = df["Close"].pct_change() * 100
    df["Log_Return"]   = np.log(df["Close"] / df["Close"].shift(1))
    df["HL_Spread"]    = (df["High"] - df["Low"]) / df["Close"] * 100
    df["OC_Spread"]    = (df["Close"] - df["Open"]) / df["Open"] * 100
    df.dropna(inplace=True)
    return df

def predict(ticker):
    t = ticker.upper().strip()
    if t in US_STOCKS:
        model_key = t
        sym       = t
    elif t in INDIAN_MAP:
        model_key = INDIAN_MAP[t]
        sym       = t if "." in t else t + ".NS"
    else:
        return None, "Ticker not supported. Try AAPL, MSFT, GOOGL, AMZN, NVDA, TCS, RELIANCE, HDFCBANK"

    raw = yf.download(sym, period="6mo", auto_adjust=True, progress=False)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)
    raw = raw[["Open","High","Low","Close","Volume"]].dropna()

    if len(raw) < 60:
        return None, f"Not enough data for {ticker}"

    featured = compute_features(raw)
    if len(featured) < 30:
        return None, "Not enough data after feature computation"

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(featured[FEATURE_COLS].values)
    seq    = torch.tensor(scaled[-30:], dtype=torch.float32).unsqueeze(0)

    model = load_model(model_key)
    with torch.no_grad():
        prob = model(seq).item()

    return {
        "ticker":      t,
        "direction":   "UP" if prob >= 0.5 else "DOWN",
        "probability": prob,
        "close":       round(float(raw["Close"].iloc[-1]), 2),
        "date":        str(raw.index[-1].date()),
        "model":       f"{model_key} model" + (" (OOD transfer)" if t not in US_STOCKS else ""),
    }, None

# ── UI ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; color: #e2e8f0; }
</style>
""", unsafe_allow_html=True)

st.title("📈 Stock Price Trend Predictor")
st.caption("LSTM Deep Learning Model · Final Year CS Project · Parul University")
st.divider()

# Quick select buttons
st.write("**Quick select:**")
cols = st.columns(8)
tickers = ["AAPL","MSFT","GOOGL","AMZN","NVDA","TCS","RELIANCE","HDFCBANK"]
quick_pick = None
for i, t in enumerate(tickers):
    if cols[i].button(t, use_container_width=True):
        quick_pick = t

# Text input
ticker_input = st.text_input(
    "Or type any ticker:",
    value=quick_pick or "",
    placeholder="e.g. AAPL",
).upper().strip()

predict_btn = st.button("🔮 Predict", type="primary", use_container_width=True)

if predict_btn and ticker_input:
    with st.spinner(f"Downloading latest data for {ticker_input} and running model..."):
        result, err = predict(ticker_input)

    if err:
        st.error(err)
    else:
        is_up = result["direction"] == "UP"
        prob  = result["probability"]
        pct   = round(prob * 100 if is_up else (1 - prob) * 100, 1)

        # Big direction display
        if is_up:
            st.success(f"## ▲ {result['direction']}  —  {pct}% confidence")
        else:
            st.error(f"## ▼ {result['direction']}  —  {pct}% confidence")

        # Probability bar
        st.progress(prob, text=f"UP probability: {round(prob*100,1)}%")

        # Metrics row
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Ticker",       result["ticker"])
        c2.metric("Latest Close", f"${result['close']}")
        c3.metric("As of",        result["date"])
        c4.metric("Model",        result["model"].split("(")[0].strip())

        st.caption(f"Model used: {result['model']}")
        st.divider()
        st.caption("⚠️ For educational purposes only. Not financial advice.")

elif predict_btn:
    st.warning("Please enter a ticker symbol first.")
