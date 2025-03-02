import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, global_mean_pool


class ShapGATModule(nn.Module):
    def __init__(
        self,
        input_dim: int,  # embedding dim
        hidden_dim: int,
        output_dim: int,  # GAT output dim
        dropout_rate: float = 0.1,
        num_heads: int = 4,  # Multi-head attention
    ):
        """
        Graph Attention Network (GAT) for polymer embeddings.

        Args:
        - input_dim (int): Initial node embedding dimension (MPNN output).
        - hidden_dim (int): Hidden layer size for GAT.
        - output_dim (int): Output embedding dimension.
        - dropout_rate (float): Dropout rate for regularization.
        - num_heads (int): Number of attention heads for GAT.
        """
        super().__init__()
        gat_output_dim = output_dim // 2
        # GAT layers
        self.conv1 = GATConv(input_dim + 1, hidden_dim, heads=num_heads, edge_dim=2)
        self.conv2 = GATConv(hidden_dim * num_heads, gat_output_dim, edge_dim=2)

        # Regularization
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, batch):
        x = batch["node_features"]  # [N, input_dim]
        edge_index = batch["edge_index"]  # [2, num_edges]
        edge_attr = batch["edge_attr"]  # [num_edges, 2]
        polymer_mapping = batch["polymer_mapping"]  # [N] Mapping for aggregation

        solvent_tags = self._generate_solvent_tags(polymer_mapping).to(x.device)
        x = torch.cat([x, solvent_tags.unsqueeze(-1)], dim=-1)  # [N, input_dim + 1]

        x = self.conv1(x, edge_index, edge_attr)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index, edge_attr)

        # Separate monomer and solvent nodes
        monomer_mask = solvent_tags == 0
        solvent_mask = solvent_tags == 1

        monomer_embeddings = x[monomer_mask]
        solvent_embeddings = x[solvent_mask]

        # Pool separately
        monomer_pooled = global_mean_pool(
            monomer_embeddings, polymer_mapping[monomer_mask]
        )
        solvent_pooled = global_mean_pool(
            solvent_embeddings, polymer_mapping[solvent_mask]
        )

        # Combine representations
        graph_embedding = torch.cat([monomer_pooled, solvent_pooled], dim=-1)

        return graph_embedding

    @staticmethod
    def _generate_solvent_tags(polymer_mapping: torch.Tensor) -> torch.Tensor:
        solvent_tags = torch.zeros_like(polymer_mapping, dtype=torch.float)

        # Identify solvent nodes (last node of each polymer)
        _, _, counts = torch.unique_consecutive(
            polymer_mapping, return_inverse=True, return_counts=True
        )
        solvent_indices = (
            torch.cumsum(counts, dim=0) - 1
        )  # Last molecule in each polymer

        solvent_tags[solvent_indices] = 1  # Set solvent tags to 1
        return solvent_tags
