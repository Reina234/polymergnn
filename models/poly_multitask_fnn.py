import torch
import torch.nn as nn


class PolymerMultiTaskFNN(nn.Module):
    def __init__(
        self,
        input_dim: int,  # Polymer embedding + N, T
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

        # Shared representation
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )

        # Heads for individual property predictions
        self.sasa_head = nn.Linear(hidden_dim, 2)  # Mean & Std Dev
        self.log_rg_head = nn.Linear(hidden_dim, 2)  # Mean & Std Dev
        self.log_ree_head = nn.Linear(hidden_dim, 2)  # Mean & Std Dev
        self.log_diffusion_head = nn.Linear(
            hidden_dim + 1, 1
        )  # Only mean, conditioned on SASA

    def forward(self, batch):
        """
        Forward pass through the multi-task FNN.

        Args:
        - batch (dict): Contains:
            - "polymer_embedding": Polymer representation from GATNN [batch, embedding_dim]
            - "polymer_feats": Additional polymer properties (N, T) [batch, 2]

        Returns:
        - Dict of predicted polymer properties
        """
        polymer_embedding = batch["polymer_embedding"]  # [B, embedding_dim]
        polymer_feats = batch["polymer_feats"]  # [B, 2] (N, T)

        # Concatenate polymer features with embeddings
        combined_input = torch.cat([polymer_embedding, polymer_feats], dim=-1)

        # Shared representation
        shared_repr = self.shared_layer(combined_input)

        # Predict independent properties
        sasa = self.sasa_head(shared_repr)  # [B, 2] (mean, std)
        log_rg = self.log_rg_head(shared_repr)  # [B, 2] (mean, std)

        # Dependent properties
        log_diffusion = self.log_diffusion_head(
            torch.cat([shared_repr, sasa[:, 0].unsqueeze(-1)], dim=-1)
        )  # [B, 1]
        log_ree = self.log_ree_head(
            torch.cat([shared_repr, log_rg], dim=-1)
        )  # [B, 2] (mean, std)

        return {
            "sasa": sasa,  # [B, 2]
            "log_rg": log_rg,  # [B, 2]
            "log_diffusion": log_diffusion.squeeze(-1),  # [B]
            "log_ree": log_ree,  # [B, 2]
        }
