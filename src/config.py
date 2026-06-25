import os

# === MODEL PATH - Critical for Render deployment ===
# When running from api/ folder, models are in api/models/
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api", "models")

US_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]
INDIAN_STOCKS = ["TCS", "RELIANCE", "HDFCBANK"]

# Add your other config variables here (keep them as they were)
LOOKBACK = 30
# FEATURE_COLS, INPUT_SIZE, HIDDEN_SIZE, etc. — copy from your backup if needed
