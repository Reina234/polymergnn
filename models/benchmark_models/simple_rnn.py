import torch.nn as nn
import torch


class SimpleRNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dim=128, output_dim=1):
        super().__init__()
        self.rnn = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, batch):
        features = batch["polymer_feats"]
        fingerprints = batch["fingerprints_tensor"]
        x = torch.cat([features, fingerprints], dim=1)
        _, (hn, _) = self.rnn(x)  # Get last hidden state
        return self.fc(hn[-1])  # Output final hidden state
