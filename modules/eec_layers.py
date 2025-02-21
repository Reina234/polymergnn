import torch
import torch.nn as nn
import torch.nn.functional as F


class ECCLayer(nn.Module):
    """
    Edge-Conditioned Convolutions
    """

    def __init__(self, in_channels, out_channels, edge_in_channels, dropout_rate=0.0):
        super().__init__()
        self.dropout_rate = dropout_rate
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_in_channels, in_channels * out_channels),
            nn.ReLU(),
            nn.Linear(in_channels * out_channels, in_channels * out_channels),
        )
        self.root = nn.Linear(in_channels, out_channels)
        self.dropout_layer = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index, edge_attr):
        # x: (N, in_channels)
        # edge_index: (2, E)
        # edge_attr: (E, edge_in_channels)
        E = edge_index.size(1)
        # Compute weight matrices for each edge from its features.
        W = self.edge_mlp(edge_attr)  # (E, in_channels*out_channels)
        W = W.view(E, x.size(1), -1)  # (E, in_channels, out_channels)

        source, target = edge_index  # Each is (E,)
        # For each edge, transform the source node feature:
        messages = torch.bmm(x[source].unsqueeze(1), W).squeeze(1)  # (E, out_channels)
        # Apply dropout to the messages:
        messages = self.dropout_layer(messages)
        # Aggregate messages at the target nodes (using sum aggregation):
        out = torch.zeros(x.size(0), messages.size(1), device=x.device)
        out = out.index_add_(0, target, messages)
        # Combine with the node’s own feature:
        out = out + self.root(x)
        return F.relu(out)
