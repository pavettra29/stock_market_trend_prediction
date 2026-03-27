# evaluate.py
# Loads best saved checkpoints, evaluates on test sets.
# Model outputs probabilities directly (Sigmoid in model head).

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report,
    precision_score, recall_score, roc_curve,
)

from config import (
    US_STOCKS, INDIAN_STOCKS, MODEL_DIR, RESULTS_DIR, DEVICE,
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT, BIDIRECTIONAL,
    BATCH_SIZE, LOOKBACK,
)
from dataset import get_dataloaders
from model import LSTMClassifier


def load_model(ticker: str) -> LSTMClassifier:
    ckpt_path = os.path.join(MODEL_DIR, f"{ticker}_best.pt")
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No checkpoint found: {ckpt_path}")
    model = LSTMClassifier(
    hidden_size=HIDDEN_SIZE,
    dropout=DROPOUT,
    bidirectional=BIDIRECTIONAL,
).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  Loaded: {ckpt_path}  (epoch={ckpt['epoch']}, "
          f"val_loss={ckpt['val_loss']:.4f})")
    return model


def predict(model, loader, device=DEVICE):
    """Returns (y_true, y_proba) — model already applies Sigmoid."""
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            probs = model(X_batch.to(device))   # already probabilities
            all_probs.append(probs.cpu().numpy())
            all_labels.append(y_batch.numpy())
    return np.concatenate(all_labels), np.concatenate(all_probs)


def compute_metrics(y_true, y_proba, threshold=0.5):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "accuracy":  round(accuracy_score(y_true, y_pred),               4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_true, y_pred,    zero_division=0), 4),
        "f1":        round(f1_score(y_true, y_pred,        zero_division=0), 4),
        "roc_auc":   round(roc_auc_score(y_true, y_proba),                4),
    }


def plot_confusion_matrix(y_true, y_pred, ticker, split):
    cm  = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 3))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Down", "Up"], yticklabels=["Down", "Up"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"{ticker} — {split} confusion matrix")
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{ticker}_{split}_confusion.png")
    plt.savefig(out, dpi=150); plt.close()
    return out


def plot_training_curve(ticker):
    log_path = os.path.join(RESULTS_DIR, f"{ticker}_training_log.csv")
    if not os.path.exists(log_path):
        return
    df = pd.read_csv(log_path)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="Train loss")
    axes[0].plot(df["epoch"], df["val_loss"],   label="Val loss")
    axes[0].set_title(f"{ticker} — Loss"); axes[0].legend()
    axes[1].plot(df["epoch"], df["train_acc"], label="Train acc")
    axes[1].plot(df["epoch"], df["val_acc"],   label="Val acc")
    axes[1].set_title(f"{ticker} — Accuracy"); axes[1].legend()
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, f"{ticker}_training_curve.png")
    plt.savefig(out, dpi=150); plt.close()
    return out


def plot_roc_curves_all(tickers=None):
    if tickers is None:
        tickers = US_STOCKS
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]
    fig, ax = plt.subplots(figsize=(7, 6))
    for ticker, color in zip(tickers, colors):
        try:
            model = load_model(ticker)
            _, _, test_loader, _ = get_dataloaders(ticker, batch_size=BATCH_SIZE)
            y_true, y_proba = predict(model, test_loader)
            fpr, tpr, _ = roc_curve(y_true, y_proba)
            auc = roc_auc_score(y_true, y_proba)
            ax.plot(fpr, tpr, color=color, lw=2,
                    label=f"{ticker}  (AUC = {auc:.3f})")
        except Exception as e:
            print(f"  Skipping {ticker} for ROC: {e}")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random baseline")
    ax.set_xlim([0, 1]); ax.set_ylim([0, 1.05])
    ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
    ax.set_title("Figure 7.3 — ROC curves, all US stocks (test set)")
    ax.legend(loc="lower right"); ax.grid(alpha=0.3)
    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, "all_us_roc_curves.png")
    plt.savefig(out, dpi=150); plt.close()
    print(f"\n  ROC curves saved → {out}")
    return out


def evaluate_ticker(train_ticker, eval_ticker=None, split="test"):
    if eval_ticker is None:
        eval_ticker = train_ticker
    model = load_model(train_ticker)
    _, _, test_loader, _ = get_dataloaders(
        eval_ticker, batch_size=BATCH_SIZE, lookback=LOOKBACK)
    y_true, y_proba = predict(model, test_loader)
    metrics = compute_metrics(y_true, y_proba)
    y_pred  = (y_proba >= 0.5).astype(int)
    plot_confusion_matrix(y_true, y_pred, eval_ticker, split)
    label = (f"{train_ticker}→{eval_ticker}"
             if train_ticker != eval_ticker else train_ticker)
    print(f"\n  [{label}]  {split}")
    print(f"    Acc={metrics['accuracy']}  F1={metrics['f1']}  "
          f"AUC={metrics['roc_auc']}  "
          f"Prec={metrics['precision']}  Rec={metrics['recall']}")
    print(classification_report(y_true, y_pred,
                                 target_names=["Down", "Up"], zero_division=0))
    return metrics


def evaluate_all():
    all_results = []

    print("\n" + "═"*55)
    print("  IN-DISTRIBUTION (US stocks)")
    print("═"*55)
    for ticker in US_STOCKS:
        try:
            m = evaluate_ticker(ticker)
            plot_training_curve(ticker)
            all_results.append({"train": ticker, "eval": ticker,
                                 "type": "in-dist", **m})
        except Exception as e:
            print(f"  Skipping {ticker}: {e}")

    plot_roc_curves_all()

    print("\n" + "═"*55)
    print("  OUT-OF-DISTRIBUTION (Indian stocks)")
    print("═"*55)
    for ind_ticker in INDIAN_STOCKS:
        for us_ticker in US_STOCKS:
            try:
                m = evaluate_ticker(us_ticker, eval_ticker=ind_ticker)
                all_results.append({"train": us_ticker, "eval": ind_ticker,
                                     "type": "OOD", **m})
            except Exception as e:
                print(f"  Skipping {us_ticker}→{ind_ticker}: {e}")

    summary_df   = pd.DataFrame(all_results)
    summary_path = os.path.join(RESULTS_DIR, "evaluation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\n  Summary saved → {summary_path}")
    print(summary_df.to_string(index=False))
    return summary_df


if __name__ == "__main__":
    evaluate_all()