"""
data_collection.py
==================
Downloads historical stock data for:
  - US stocks  : used for MODEL TRAINING  (AAPL, MSFT, GOOGL, AMZN, NVDA)
  - Indian stocks : used for TESTING only (TCS.NS, RELIANCE.NS, HDFCBANK.NS)

"""

import yfinance as yf
import pandas as pd
import os

# ──────────────────────────────────────────────
# STEP 1 — Define your tickers
# ──────────────────────────────────────────────

# US stocks for training the LSTM model
US_TICKERS = {
    "AAPL":  "AAPL",
    "MSFT":  "MSFT",
    "GOOGL": "GOOGL",
    "AMZN":  "AMZN",
    "NVDA":  "NVDA",
}

# Indian stocks for testing generalizability
# .NS = NSE (National Stock Exchange) — required by Yahoo Finance
INDIAN_TICKERS = {
    "TCS":       "TCS.NS",
    "RELIANCE":  "RELIANCE.NS",
    "HDFCBANK":  "HDFCBANK.NS",
}

# Combine both
ALL_TICKERS = {**US_TICKERS, **INDIAN_TICKERS}

# ──────────────────────────────────────────────
# STEP 2 — Date range
# ──────────────────────────────────────────────
START_DATE = "2015-01-01"
END_DATE   = "2024-12-31"   # FIX: was "2024-01-01" — now includes full 2024

# ──────────────────────────────────────────────
# STEP 3 — Download function
# ──────────────────────────────────────────────

def download_stock_data(name, ticker, start, end):
    """
    Downloads OHLCV data for one stock.

    What is OHLCV?
      O = Open price at start of day
      H = Highest price during the day
      L = Lowest price during the day
      C = Close price at end of day
      V = Volume (number of shares traded)

    auto_adjust=True:
      Automatically adjusts prices for stock splits and dividends.
      Without this, a 2-for-1 split makes the price look like it
      dropped 50% overnight — which would confuse your model badly.
    """
    print(f"  Downloading {name} ({ticker})...")

    try:
        # FIX: added auto_adjust=True to correct for splits/dividends
        df = yf.download(ticker, start=start, end=end,
                         progress=False, auto_adjust=True)

        # FIX: check if download actually returned data
        if df.empty:
            print(f"  ⚠️  No data for {name}. Check ticker symbol: {ticker}")
            return None

        # Keep only the 5 OHLCV columns we need
        df = df[["Open", "High", "Low", "Close", "Volume"]]
        df.index.name = "Date"

        print(f"  ✓ {name}: {len(df)} rows "
              f"({df.index[0].date()} → {df.index[-1].date()})")
        return df

    except Exception as e:
        # FIX: catch any network or API errors gracefully
        print(f"  ✗ Failed to download {name}: {e}")
        return None


# ──────────────────────────────────────────────
# STEP 4 — Save to CSV
# ──────────────────────────────────────────────

def collect_all():
    """Downloads and saves data for all tickers."""

    # Create the data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  Downloading {len(ALL_TICKERS)} stocks")
    print(f"  Period: {START_DATE} → {END_DATE}")
    print(f"{'='*50}\n")

    print("📦 US Training Stocks:")
    for name, ticker in US_TICKERS.items():
        df = download_stock_data(name, ticker, START_DATE, END_DATE)
        if df is not None:
            path = f"data/{name}.csv"
            df.to_csv(path)
            print(f"     Saved → {path}")

    print("\n🇮🇳 Indian Test Stocks:")
    for name, ticker in INDIAN_TICKERS.items():
        df = download_stock_data(name, ticker, START_DATE, END_DATE)
        if df is not None:
            path = f"data/{name}.csv"
            df.to_csv(path)
            print(f"     Saved → {path}")

    print("\n✅ Data collection complete!\n")


# ──────────────────────────────────────────────
# Run when you execute: python data_collection.py
# ──────────────────────────────────────────────
if __name__ == "__main__":
    collect_all()