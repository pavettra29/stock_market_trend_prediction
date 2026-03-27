# config.py
import os

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR   = os.path.join(BASE_DIR, "..", "models")
RESULTS_DIR = os.path.join(BASE_DIR, "..", "results")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

US_STOCKS     = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
INDIAN_STOCKS = ["TCS", "RELIANCE", "HDFCBANK"]

FEATURE_COLS = [
    "Open", "High", "Low", "Close", "Volume",
    "SMA_20", "SMA_50", "EMA_12", "EMA_26",
    "RSI", "MACD", "MACD_Signal", "MACD_Histogram",
    "Stoch_K", "Stoch_D",
    "BB_Upper", "BB_Middle", "BB_Lower", "BB_Width", "BB_Pct",
    "ATR", "OBV",
    "Volume_SMA", "Volume_Ratio",
    "Daily_Return", "Log_Return",
    "HL_Spread", "OC_Spread",
]
TARGET_COL = "Target"

LOOKBACK   = 30       # reduced from 60 — shorter window trains faster
                      # and avoids losing too many samples
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15

INPUT_SIZE    = 28
HIDDEN_SIZE   = 64    # reduced from 128 — less capacity = less overfitting
NUM_LAYERS    = 1     # single layer
DROPOUT       = 0.2   # reduced from 0.3
BIDIRECTIONAL = False

BATCH_SIZE     = 32
MAX_EPOCHS     = 200
LEARNING_RATE  = 1e-3
WEIGHT_DECAY   = 0.0
EARLY_STOP_PAT = 20
LR_SCHEDULER   = True
LR_FACTOR      = 0.5
LR_PATIENCE    = 10

SEED = 42

import torch
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"