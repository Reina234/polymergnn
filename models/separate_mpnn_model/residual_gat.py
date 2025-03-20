import torch
import torch.nn as nn
from torch_geometric.nn import GATConv, GlobalAttention


class ResidualGatedGATModule(nn.Module):
    def __init__(
        self,
        input_dim: int,  # Embedding dim
        hidden_dim: int,
        output_dim: int,  # Final GAT output dim
        dropout_rate: float = 0.1,
        num_heads: int = 4,  # Multi-head attention
    ):
        super().__init__()
        edge_attr_dim = 2

        # First GAT layer
        self.conv1 = GATConv(
            input_dim + 2, hidden_dim, heads=num_heads, edge_dim=edge_attr_dim
        )
        self.norm1 = nn.LayerNorm(hidden_dim * num_heads)

        # Second GAT layer (final message passing step)
        self.conv2 = GATConv(
            hidden_dim * num_heads, input_dim + 2, heads=1, edge_dim=edge_attr_dim
        )
        self.norm2 = nn.LayerNorm(
            input_dim + 2
        )  # Normalize to match residual connection

        # Gated residual connection
        self.gate_fc = nn.Sequential(
            nn.Linear(input_dim + 2, 1),  # Gate scalar per node
            nn.Sigmoid(),  # Output between 0 and 1
        )

        # Attention pooling
        self.attention_pooling = GlobalAttention(
            gate_nn=nn.Sequential(
                nn.Linear(input_dim + 2, 1), nn.Sigmoid()  # Gate network
            )
        )

        # Final projection layer (ensures `output_dim` consistency)
        self.final_projection = nn.Linear(2 * (input_dim + 2), output_dim)

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

    def forward(self, batch):
        x = batch["node_features"]  # [N, input_dim]
        edge_index = batch["edge_index"]  # [2, num_edges]
        edge_attr = batch["edge_attr"]  # [num_edges, 2]
        polymer_mapping = batch["polymer_mapping"]  # [N] Mapping for aggregation
        polymer_feats = batch["polymer_feats"]  # [B, 3] → (density, N, T)

        density = polymer_feats[:, 0]  # Shape: [B] (per polymer)

        # Generate solvent/polymer identity tags
        solvent_tags = self._generate_solvent_tags(polymer_mapping).to(x.device)
        monomer_mask = solvent_tags == 0  # Polymer nodes
        solvent_mask = solvent_tags == 1  # Solvent nodes

        # Expand density to match per-node assignment
        density_per_node = density[polymer_mapping]  # Shape: [N]

        # Append solvent and density features
        x = torch.cat(
            [x, solvent_tags.unsqueeze(-1), density_per_node.unsqueeze(-1)], dim=-1
        )

        # First GAT Layer
        x_res = x  # Store residual
        x = self.conv1(x, edge_index, edge_attr)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Second GAT Layer
        x = self.conv2(x, edge_index, edge_attr)
        x = self.norm2(x)

        # Apply gated residual connection
        gate = self.gate_fc(x_res)  # Compute gating factor
        x = gate * x + (1 - gate) * x_res  # Blend with residual

        # Separate monomer and solvent nodes
        monomer_embeddings = x[monomer_mask]
        solvent_embeddings = x[solvent_mask]

        # Pool separately
        monomer_pooled = self.attention_pooling(
            monomer_embeddings, polymer_mapping[monomer_mask]
        )
        solvent_pooled = self.attention_pooling(
            solvent_embeddings, polymer_mapping[solvent_mask]
        )

        # Combine representations
        graph_embedding = torch.cat([monomer_pooled, solvent_pooled], dim=-1)

        # Ensure final output is `output_dim`
        graph_embedding = self.final_projection(graph_embedding)

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
