# model.py — Simplified single-layer LSTM
# Simpler architecture is more stable on small financial datasets.
# Two stacked layers with dropout was over-regularising and collapsing.

import torch
import torch.nn as nn
from config import INPUT_SIZE, HIDDEN_SIZE, DROPOUT, BIDIRECTIONAL


class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size:    int   = INPUT_SIZE,
        hidden_size:   int   = HIDDEN_SIZE,
        dropout:       float = DROPOUT,
        bidirectional: bool  = BIDIRECTIONAL,
    ):
        super().__init__()
        self.hidden_size    = hidden_size
        self.bidirectional  = bidirectional
        self.num_directions = 2 if bidirectional else 1

        # Single LSTM layer — simpler = more stable on ~1600 samples
        self.lstm = nn.LSTM(
            input_size    = input_size,
            hidden_size   = hidden_size,
            num_layers    = 1,
            batch_first   = True,
            bidirectional = bidirectional,
        )

        fc_input = hidden_size * self.num_directions

        self.classifier = nn.Sequential(
            nn.Linear(fc_input, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.lstm.flatten_parameters()
        out, _ = self.lstm(x)
        last    = out[:, -1, :]
        return self.classifier(last).squeeze(1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        d = "Bi-LSTM" if self.bidirectional else "LSTM"
        print(f"\n{'─'*50}")
        print(f"  {d}Classifier  (1 layer, simplified)")
        print(f"  Hidden: {self.hidden_size}")
        print(f"  Trainable params: {self.count_parameters():,}")
        print(f"{'─'*50}\n")


if __name__ == "__main__":
    from config import DEVICE, LOOKBACK
    m = LSTMClassifier().to(DEVICE)
    m.summary()
    x = torch.randn(8, LOOKBACK, INPUT_SIZE).to(DEVICE)
    o = m(x)
    print(f"  Output: {o.shape}  range=[{o.min():.3f}, {o.max():.3f}]")
    print("Model OK ✓")