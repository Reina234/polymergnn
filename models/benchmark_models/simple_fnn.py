import torch.nn as nn


class SimpleFNN(nn.Module):
    def __init__(self, input_dim: int, dropout=0.2, hidden_dim=128, output_dim=1):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, batch):
        x = batch["fingerprint"]  # Extract fingerprints_tensor from batch
        x = x.view(x.shape[0], -1)  # Flatten for FNN
        return self.model(x)
