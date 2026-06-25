import os
import torch

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "models")

US_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
INDIAN_STOCKS = ["TCS", "RELIANCE", "HDFCBANK"]

LOOKBACK = 30
FEATURE_COLS = ["Open", "High", "Low", "Close", "Volume"]  # Minimal for now - expand later
INPUT_SIZE = len(FEATURE_COLS)
HIDDEN_SIZE = 64
DROPOUT = 0.2
BIDIRECTIONAL = False
DEVICE = torch.device("cpu")
