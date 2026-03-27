# dataset.py
# Builds chronological sequences from featured CSVs.
# Scaler is fit on TRAIN split only — then reused for val/test.

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler

from config import (
    DATA_DIR, FEATURE_COLS, TARGET_COL,
    LOOKBACK, TRAIN_FRAC, VAL_FRAC,
)


# ─────────────────────────────────────────────────────────────────────────────
def load_featured_csv(ticker: str) -> pd.DataFrame:
    """Load the _featured.csv produced in Phase 2."""
    fname = ticker.replace(".", "_") + "_featured.csv"
    path  = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Featured file not found: {path}")
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    df.sort_index(inplace=True)
    return df


def chronological_split(df: pd.DataFrame):
    """Return (train_df, val_df, test_df) with no data leakage."""
    n      = len(df)
    t_end  = int(n * TRAIN_FRAC)
    v_end  = int(n * (TRAIN_FRAC + VAL_FRAC))
    return df.iloc[:t_end], df.iloc[t_end:v_end], df.iloc[v_end:]


def build_sequences(features: np.ndarray, targets: np.ndarray, lookback: int):
    """
    Slide a window of `lookback` rows across features.
    X shape: (N, lookback, n_features)
    y shape: (N,)
    """
    X, y = [], []
    for i in range(lookback, len(features)):
        X.append(features[i - lookback: i])
        y.append(targets[i])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
class StockSequenceDataset(Dataset):
    """
    PyTorch Dataset for a single stock, single split.

    Usage
    -----
    scaler = None                         # first call creates & fits it
    train_ds = StockSequenceDataset("AAPL", split="train")
    scaler   = train_ds.scaler           # reuse for val & test
    val_ds   = StockSequenceDataset("AAPL", split="val",  scaler=scaler)
    test_ds  = StockSequenceDataset("AAPL", split="test", scaler=scaler)
    """

    def __init__(self, ticker: str, split: str = "train",
                 scaler: MinMaxScaler | None = None,
                 lookback: int = LOOKBACK):
        assert split in ("train", "val", "test"), "split must be train/val/test"

        df = load_featured_csv(ticker)

        # ── drop rows with NaN in features or target ──────────────────────
        cols_needed = FEATURE_COLS + [TARGET_COL]
        df = df[cols_needed].dropna()

        # ── chronological split ───────────────────────────────────────────
        train_df, val_df, test_df = chronological_split(df)
        split_map = {"train": train_df, "val": val_df, "test": test_df}
        split_df  = split_map[split]

        raw_features = split_df[FEATURE_COLS].values
        raw_targets  = split_df[TARGET_COL].values

        # ── scaling: fit only on train ────────────────────────────────────
        if scaler is None:
            if split != "train":
                raise ValueError("Pass a fitted scaler for val/test splits.")
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(train_df[FEATURE_COLS].values)  # fit on FULL train set
        self.scaler = scaler

        scaled_features = scaler.transform(raw_features)

        # ── build sequences ───────────────────────────────────────────────
        self.X, self.y = build_sequences(scaled_features, raw_targets, lookback)

        self.ticker = ticker
        self.split  = split

        print(f"  [{ticker}] {split:5s} → X={self.X.shape}  "
              f"pos={int(self.y.sum())}  neg={int((1-self.y).sum())}")

    # ── PyTorch interface ─────────────────────────────────────────────────
    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return (torch.tensor(self.X[idx]),          # (lookback, n_features)
                torch.tensor(self.y[idx]))           # scalar


# ─────────────────────────────────────────────────────────────────────────────
def get_dataloaders(ticker: str, batch_size: int = 64, lookback: int = LOOKBACK,
                    num_workers: int = 0):
    """
    Convenience function — returns (train_loader, val_loader, test_loader, scaler).
    num_workers=0 is safest on macOS (MPS).
    """
    from torch.utils.data import DataLoader

    train_ds = StockSequenceDataset(ticker, split="train", lookback=lookback)
    val_ds   = StockSequenceDataset(ticker, split="val",   scaler=train_ds.scaler, lookback=lookback)
    test_ds  = StockSequenceDataset(ticker, split="test",  scaler=train_ds.scaler, lookback=lookback)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers)

    return train_loader, val_loader, test_loader, train_ds.scaler


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Quick smoke-test
    print("Testing dataset pipeline on AAPL …")
    tl, vl, tel, sc = get_dataloaders("AAPL", batch_size=64)
    xb, yb = next(iter(tl))
    print(f"  Batch X: {xb.shape}  y: {yb.shape}  dtype: {xb.dtype}")
    print("Dataset OK ✓")