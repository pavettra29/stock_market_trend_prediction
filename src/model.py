# model.py — matches the architecture used to train the saved checkpoints
# Single LSTM layer -> Linear(hidden,32) -> ReLU -> Dropout -> Linear(32,1) -> Sigmoid

import torch
import torch.nn as nn
from config import INPUT_SIZE, HIDDEN_SIZE, NUM_LAYERS, DROPOUT


class LSTMClassifier(nn.Module):

    def __init__(
        self,
        input_size:  int   = INPUT_SIZE,
        hidden_size: int   = HIDDEN_SIZE,
        num_layers:  int   = NUM_LAYERS,
        dropout:      float = DROPOUT,
        bidirectional: bool  = False,   # accepted for compatibility, unused (checkpoints are unidirectional)
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param.data)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param.data)
            elif "bias" in name:
                param.data.fill_(0)
                n = param.size(0)
                param.data[n // 4: n // 2].fill_(1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_out    = lstm_out[:, -1, :]
        out         = self.classifier(last_out)
        return out.squeeze(1)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        print(f"\n{'─'*50}")
        print(f"  LSTMClassifier")
        print(f"  Hidden:  {self.hidden_size}  |  Layers: {self.num_layers}")
        print(f"  Trainable params: {self.count_parameters():,}")
        print(f"{'─'*50}\n")


if __name__ == "__main__":
    from config import DEVICE, LOOKBACK
    model = LSTMClassifier().to(DEVICE)
    model.summary()
    dummy = torch.randn(8, LOOKBACK, INPUT_SIZE).to(DEVICE)
    out   = model(dummy)
    print(f"  Output shape : {out.shape}")
    print(f"  Output range : [{out.min():.3f}, {out.max():.3f}]")
    print("Model OK ✓")
