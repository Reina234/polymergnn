import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


# assumes batch structure as in PolymerGNNDataloader
class PolymerGATNN(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1,
        num_heads: int = 4,  # Multi-head attention for better representation
    ):
        super().__init__()

        # First GAT layer
        self.conv1 = GATConv(input_dim, hidden_dim, heads=num_heads, edge_dim=2)
        self.conv2 = GATConv(hidden_dim * num_heads, output_dim, edge_dim=2)

        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, batch):

        x = batch["node_features"]  # (N, D)

        # 2. Edge indices and attributes
        edge_index = batch["edge_index"]  # (2, N_edges)
        edge_attr = batch["edge_attr"]  # (N_edges, 2)

        # 3. GNN layers
        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)

        # 4. Aggregate node embeddings into polymer-level embeddings
        polymer_mapping = batch["polymer_mapping"]
        graph_embedding = global_mean_pool(x, polymer_mapping)

        return graph_embedding
