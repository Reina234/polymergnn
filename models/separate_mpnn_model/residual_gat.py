import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GlobalAttention


class ResidualGatedGATModule(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        dropout_rate: float = 0.1,
        num_heads: int = 4,
    ):
        super().__init__()
        edge_attr_dim = 2
        augmented_dim = input_dim + 2  # + solvent tag and density

        self.conv1 = GATConv(
            augmented_dim, hidden_dim, heads=num_heads, edge_dim=edge_attr_dim
        )
        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)

        self.conv2 = GATConv(
            hidden_dim * num_heads, augmented_dim, heads=1, edge_dim=edge_attr_dim
        )
        self.norm2 = nn.LayerNorm(augmented_dim)

        self.gate_fc = nn.Sequential(
            nn.Linear(augmented_dim, 1),
            nn.Sigmoid(),
        )

        self.attention_pooling = GlobalAttention(
            gate_nn=nn.Sequential(nn.Linear(augmented_dim, 1), nn.Sigmoid())
        )

        self.final_projection = nn.Linear(augmented_dim, output_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, batch):
        x = batch["node_features"]
        edge_index = batch["edge_index"]
        edge_attr = batch["edge_attr"]
        polymer_mapping = batch["polymer_mapping"]
        polymer_feats = batch["polymer_feats"]  # [B, 3]

        density = polymer_feats[:, 0]
        solvent_tags = self._generate_solvent_tags(polymer_mapping).to(x.device)
        density_per_node = density[polymer_mapping]

        x = torch.cat(
            [x, solvent_tags.unsqueeze(-1), density_per_node.unsqueeze(-1)], dim=-1
        )

        x_res = x
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)

        gate = self.gate_fc(x_res)
        x = gate * x + (1 - gate) * x_res

        pooled = self.attention_pooling(x, polymer_mapping)  # [B, augmented_dim]
        return self.final_projection(pooled)

    @staticmethod
    def _generate_solvent_tags(polymer_mapping: torch.Tensor) -> torch.Tensor:
        solvent_tags = torch.zeros_like(polymer_mapping, dtype=torch.float)
        _, _, counts = torch.unique_consecutive(
            polymer_mapping, return_inverse=True, return_counts=True
        )
        solvent_indices = torch.cumsum(counts, dim=0) - 1
        solvent_tags[solvent_indices] = 1
        return solvent_tags
