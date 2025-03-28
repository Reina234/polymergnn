import torch.nn as nn
import torch


class SimpleCNN(nn.Module):
    def __init__(self, input_dim: int, num_molecules, output_dim=1):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels=input_dim, out_channels=64, kernel_size=3, padding=1
        )
        self.conv2 = nn.Conv1d(
            in_channels=64, out_channels=32, kernel_size=3, padding=1
        )
        self.fc = nn.Linear(32 * num_molecules, output_dim)

    def forward(self, batch):
        features = batch["polymer_feats"]
        fingerprints = batch["fingerprints_tensor"]
        x = torch.cat([features, fingerprints], dim=1)
        x = x.permute(0, 2, 1)  # Reshape for CNN: (batch, features, sequence)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(x.shape[0], -1)  # Flatten
        return self.fc(x)
