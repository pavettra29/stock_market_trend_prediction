import os
import torch

# === MODEL PATH - Critical for Render deployment ===
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "models")

US_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
INDIAN_STOCKS = ["TCS", "RELIANCE", "HDFCBANK"]

# Training parameters (adjust if your model uses different values)
LOOKBACK = 30
FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "EMA_20",
    "RSI", "MACD", "MACD_signal", "MACD_hist",
    "BB_upper", "BB_lower", "BB_width",
    "Returns", "Volatility_20",
    "Lag_1", "Lag_2", "Lag_3", "Lag_5",
    "DayOfWeek", "IsMonthEnd"
]

INPUT_SIZE = len(FEATURE_COLS)
HIDDEN_SIZE = 64
DROPOUT = 0.2
BIDIRECTIONAL = False
DEVICE = torch.device("cpu")

print(f"✅ Config loaded. MODEL_DIR = {MODEL_DIR}")
