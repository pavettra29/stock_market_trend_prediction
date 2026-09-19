# train.py — Final stable version
# Key settings vs previous versions:
#   • Patience = 15 (was 10) — gives model more time to learn
#   • MAX_EPOCHS = 150       — more room to converge
#   • LR = 1e-3, no decay    — let Adam find its own rhythm
#   • Plain BCELoss + Sigmoid in model
#   • No pos_weight          — classes are ~50/50

import os
import time
import csv
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from config import (
    US_STOCKS, MODEL_DIR, RESULTS_DIR, DEVICE, SEED,
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT, BIDIRECTIONAL,
    BATCH_SIZE, LOOKBACK, WEIGHT_DECAY,
)
from dataset import get_dataloaders
from model import LSTMClassifier

MAX_EPOCHS     = 150
LEARNING_RATE  = 1e-3
EARLY_STOP_PAT = 15
LR_PATIENCE    = 8
LR_FACTOR      = 0.5


def set_seed(seed: int = SEED):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def run_epoch(model, loader, criterion, optimizer=None, device=DEVICE):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    total_loss, correct, total = 0.0, 0, 0
    ctx = torch.enable_grad() if is_train else torch.no_grad()

    with ctx:
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            probs   = model(X_batch)
            loss    = criterion(probs, y_batch)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            predicted   = (probs >= 0.5).float()
            correct    += (predicted == y_batch).sum().item()
            total_loss += loss.item() * len(y_batch)
            total      += len(y_batch)

    return total_loss / total, correct / total


def train_ticker(ticker: str, verbose: bool = True):
    set_seed()
    print(f"\n{'═'*55}")
    print(f"  Training on: {ticker}   device={DEVICE}")
    print(f"{'═'*55}")

    train_loader, val_loader, _, _ = get_dataloaders(
        ticker, batch_size=BATCH_SIZE, lookback=LOOKBACK)

    criterion = nn.BCELoss()

    model = LSTMClassifier(
        hidden_size=HIDDEN_SIZE, num_layers=NUM_LAYERS,
        dropout=DROPOUT, bidirectional=BIDIRECTIONAL,
    ).to(DEVICE)
    model.summary()

    optimizer = Adam(model.parameters(), lr=LEARNING_RATE,
                     weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode="min",
                                  factor=LR_FACTOR, patience=LR_PATIENCE)

    log_path  = os.path.join(RESULTS_DIR, f"{ticker}_training_log.csv")
    ckpt_path = os.path.join(MODEL_DIR,   f"{ticker}_best.pt")

    with open(log_path, "w", newline="") as f:
        csv.writer(f).writerow(["epoch", "train_loss", "train_acc",
                                 "val_loss", "val_acc", "lr"])

    best_val_loss  = float("inf")
    patience_count = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer)
        vl_loss, vl_acc = run_epoch(model, val_loader,   criterion)
        scheduler.step(vl_loss)
        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        with open(log_path, "a", newline="") as f:
            csv.writer(f).writerow(
                [epoch, tr_loss, tr_acc, vl_loss, vl_acc, lr_now])

        if verbose:
            print(f"  Ep {epoch:3d}/{MAX_EPOCHS}  "
                  f"tr_loss={tr_loss:.4f}  tr_acc={tr_acc:.4f}  "
                  f"vl_loss={vl_loss:.4f}  vl_acc={vl_acc:.4f}  "
                  f"lr={lr_now:.2e}  ({elapsed:.1f}s)")

        if vl_loss < best_val_loss:
            best_val_loss  = vl_loss
            patience_count = 0
            torch.save({
                "epoch":       epoch,
                "model_state": model.state_dict(),
                "val_loss":    vl_loss,
                "val_acc":     vl_acc,
                "ticker":      ticker,
            }, ckpt_path)
            if verbose:
                print(f"    ✓ checkpoint saved (val_loss={vl_loss:.4f})")
        else:
            patience_count += 1
            if patience_count >= EARLY_STOP_PAT:
                print(f"\n  Early stopping at epoch {epoch}.")
                break

    print(f"\n  Best val_loss for {ticker}: {best_val_loss:.4f}")
    print(f"  Log   → {log_path}")
    print(f"  Model → {ckpt_path}")
    return best_val_loss


def train_all(tickers=US_STOCKS):
    results = {}
    for ticker in tickers:
        results[ticker] = train_ticker(ticker)
    print(f"\n{'═'*55}")
    print("  Training complete — summary:")
    for t, v in results.items():
        print(f"    {t:<15} best val_loss = {v:.4f}")
    print(f"{'═'*55}\n")
    return results


if __name__ == "__main__":
    train_all()