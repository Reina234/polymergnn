import torch
import torch.nn as nn


class OneOutputFNN(nn.Module):
    def __init__(
        self,
        input_dim: int,  # Polymer embedding + N, T
        shared_layer_dim: int,
        hidden_dim: int,
        dropout_rate: float = 0.2,
    ):
        """
        Multi-task FNN for predicting polymer properties.

        Args:
        - input_dim (int): Input size (GNN output + polymer features N, T).
        - hidden_dim (int): Size of hidden layers.
        - dropout_rate (float): Dropout for regularization.
        """
        super().__init__()

        input_dim_fnn = input_dim + 3  # Add 2 for N and temp
        # Shared representation
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim_fnn, shared_layer_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(shared_layer_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, batch):

        polymer_embedding = batch["polymer_embedding"]  # [B, embedding_dim]
        # polymer_feats = batch["polymer_feats"][:, 2]  # [B, 2] (N, T) -> (T)
        # normalized_T = (polymer_feats / 100).unsqueeze(-1)  # Shape: [B, 1]
        polymer_feats = batch["polymer_feats"][:, 0:3]  # [B, 2] (D (?), N, T)
        scaling_factors = torch.tensor([1, 10.0, 100.0], device=polymer_feats.device)
        normalized_feats = polymer_feats / scaling_factors
        # Concatenate polymer features with embeddings
        combined_input = torch.cat([polymer_embedding, normalized_feats], dim=-1)
        # Shared representation
        output = self.shared_layer(combined_input)

        return output
