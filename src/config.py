# config.py
# Central configuration for Phase 3: LSTM Model Design & Training

import os

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR        = os.path.join(BASE_DIR, "..", "data")
MODEL_DIR       = os.path.join(BASE_DIR, "..", "models")
RESULTS_DIR     = os.path.join(BASE_DIR, "..", "results")

os.makedirs(MODEL_DIR,   exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# ── Stocks ───────────────────────────────────────────────────────────────────
US_STOCKS     = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]   # training
INDIAN_STOCKS = ["TCS", "RELIANCE", "HDFCBANK"]              # OOD test (no .NS)

# ── Feature columns (must match your _featured.csv exactly) ──────────────────
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
TARGET_COL = "Target"   # binary: 1 = price up, 0 = price down

# ── Sequence / split ─────────────────────────────────────────────────────────
LOOKBACK   = 60
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.15
# TEST_FRAC  = 0.15  (remainder)

# ── Model architecture ───────────────────────────────────────────────────────
INPUT_SIZE    = 28        # matches len(FEATURE_COLS)
HIDDEN_SIZE   = 64
NUM_LAYERS    = 1
DROPOUT       = 0.2
BIDIRECTIONAL = False

# ── Training ─────────────────────────────────────────────────────────────────
BATCH_SIZE      = 64
MAX_EPOCHS      = 100
LEARNING_RATE   = 1e-3
WEIGHT_DECAY    = 1e-5
EARLY_STOP_PAT  = 10
LR_SCHEDULER    = True
LR_FACTOR       = 0.5
LR_PATIENCE     = 5

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED = 42

# ── Device (Apple Silicon MPS if available, else CPU) ────────────────────────
import torch
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"