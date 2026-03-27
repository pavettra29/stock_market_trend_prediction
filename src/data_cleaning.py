"""
data_cleaning.py
================
Cleans raw stock CSVs downloaded by data_collection.py.

ABOUT THE yfinance CSV FORMAT (latest version):
  The newest yfinance saves CSVs with 3 metadata rows at the top:

  Row 0 → Price, AAPL, AAPL, AAPL...     (metadata)
  Row 1 → Ticker, AAPL, AAPL, AAPL...    (metadata)
  Row 2 → (blank or extra info)           (metadata)
  Row 3 → Date, Close, High, Low,...      (REAL header)
  Row 4 → 2015-01-02, 182.3,...           (data starts)

  This file auto-detects how many rows to skip.
"""

import pandas as pd
import os

ALL_STOCKS = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
              "TCS", "RELIANCE", "HDFCBANK"]


def read_yfinance_csv(file_path):
    """
    Reads a yfinance CSV regardless of how many metadata rows it has.

    Strategy: scan the first 6 rows to find which row contains
    'Date' or 'Open' or 'Close' — that's the real header row.
    Then skip everything above it.
    """

    # Read first 6 rows with no assumptions
    peek = pd.read_csv(file_path, header=None, nrows=6)

    # Search each row to find the real header
    # The real header contains column names like 'Date', 'Close', 'Open' etc.
    header_row = None
    for i, row in peek.iterrows():
        row_values = [str(v).strip().lower() for v in row.tolist()]
        if "date" in row_values or "close" in row_values or "open" in row_values:
            header_row = i
            print(f"  Found real header at row {i}: {row.tolist()}")
            break

    if header_row is None:
        raise ValueError(
            f"Could not find a header row (Date/Close/Open) in the first 6 rows.\n"
            f"  File: {file_path}\n"
            f"  Please open it in VS Code and check the structure.\n"
            f"  First 6 rows were:\n{peek.to_string()}"
        )

    # Now read the CSV properly — skip all rows above the header
    df = pd.read_csv(
        file_path,
        skiprows=header_row,   # skip metadata rows
        index_col=0,           # first column = Date
        parse_dates=True
    )

    df.index.name = "Date"

    # Standardise column names to Title Case (close→Close, open→Open)
    df.columns = [str(c).strip().title() for c in df.columns]

    # Keep only the 5 OHLCV columns
    needed   = ["Open", "High", "Low", "Close", "Volume"]
    available = [c for c in needed if c in df.columns]

    if not available:
        raise ValueError(
            f"Still could not find OHLCV columns after skip.\n"
            f"  Columns found: {list(df.columns)}\n"
            f"  File: {file_path}\n"
            f"  First 6 rows of file:\n{peek.to_string()}"
        )

    df = df[available]

    # Remove any leftover non-date rows (extra metadata that snuck through)
    valid = pd.to_datetime(df.index, errors="coerce").notna()
    removed = (~valid).sum()
    if removed > 0:
        print(f"  Removed {removed} non-date row(s)")
    df = df[valid]
    df.index = pd.to_datetime(df.index)

    return df


def clean_data(file_path):
    """Full cleaning pipeline for one stock CSV."""

    print(f"\n  Cleaning: {file_path}")

    df = read_yfinance_csv(file_path)

    print(f"  Loaded   : {len(df)} rows | columns: {list(df.columns)}")

    # Sort by date — oldest row first
    df.sort_index(inplace=True)

    # Remove duplicate dates
    dupes = df.index.duplicated().sum()
    if dupes > 0:
        print(f"  Removing {dupes} duplicate date(s)")
    df = df[~df.index.duplicated(keep="first")]

    # Forward-fill gaps (holidays / weekends) instead of dropping rows
    df = df.ffill()

    # Convert all price/volume columns to numeric
    # yfinance sometimes saves values as strings — this fixes that
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Remove corrupted rows (Close = 0 or still NaN)
    df = df[df["Close"] > 0]
    df.dropna(subset=["Close"], inplace=True)

    print(f"  Cleaned  : {len(df)} rows")
    print(f"  Range    : {df.index[0].date()} → {df.index[-1].date()}")

    return df


def clean_all():
    os.makedirs("data", exist_ok=True)

    print(f"\n{'='*50}")
    print(f"  DATA CLEANING")
    print(f"{'='*50}")

    success, failed = [], []

    for name in ALL_STOCKS:
        raw_path     = f"data/{name}.csv"
        cleaned_path = f"data/{name}_cleaned.csv"

        if not os.path.exists(raw_path):
            print(f"\n  ⚠️  Skipping {name} — {raw_path} not found.")
            failed.append(name)
            continue

        try:
            df = clean_data(raw_path)
            df.to_csv(cleaned_path)
            print(f"  ✓ Saved → {cleaned_path}")
            success.append(name)
        except Exception as e:
            print(f"\n  ✗ Error cleaning {name}:\n  {e}")
            failed.append(name)

    print(f"\n{'─'*50}")
    print(f"  ✅ Cleaned: {len(success)} stocks  →  {success}")
    if failed:
        print(f"  ❌ Failed : {len(failed)} stocks  →  {failed}")
    print(f"{'─'*50}\n")


if __name__ == "__main__":
    clean_all()