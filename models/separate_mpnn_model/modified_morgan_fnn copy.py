import torch
import torch.nn as nn


class ModifiedMorganPolymerMultiTaskFNNNoT(nn.Module):
    def __init__(
        self,
        input_dim: int,  # Polymer embedding + N, T
        shared_layer_dim: int,
        hidden_dim: int,
        n_bits: int,
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

        input_dim_fnn = input_dim + 3  # Add 1 for T
        # Shared representation
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim_fnn, shared_layer_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(shared_layer_dim, shared_layer_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        # Heads for individual property predictions
        self.sasa_head = nn.Sequential(
            nn.Linear(shared_layer_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )
        self.log_rg_head = nn.Sequential(
            nn.Linear(shared_layer_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )
        self.log_ree_head = nn.Sequential(
            nn.Linear(shared_layer_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_diffusion_head = nn.Sequential(
            nn.Linear(2 * n_bits + input_dim_fnn + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        # self.log_diffusion_head = nn.Sequential(
        #    nn.Linear(shared_layer_dim, hidden_dim),
        #    nn.SiLU(),
        #    nn.Dropout(dropout_rate),
        #    nn.Linear(hidden_dim, hidden_dim),
        #    nn.SiLU(),
        #    nn.Linear(hidden_dim,x 1),
        # )

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
        n_bits = batch["fingerprints_tensor"]

        polymer_feats = batch["polymer_feats"][:, 0:3]  # [B, 2] (D (?), N, T)
        scaling_factors = torch.tensor([1, 10.0, 100.0], device=polymer_feats.device)
        normalized_feats = polymer_feats / scaling_factors

        combined_input = torch.cat([polymer_embedding, normalized_feats], dim=-1)
        shared_repr = self.shared_layer(combined_input)

        sasa = self.sasa_head(shared_repr)  # [B, 2] (mean, std)

        log_rg = self.log_rg_head(shared_repr)  # [B, 2] (mean, std)

        diff_input = torch.cat(
            [n_bits, combined_input, sasa[:, 0].unsqueeze(-1)], dim=-1
        )

        log_diffusion = self.log_diffusion_head(diff_input)  # [B, 1]
        # log_diffusion = self.log_diffusion_head(shared_repr)
        log_ree_input = torch.cat([shared_repr, log_rg[:, 0].unsqueeze(-1)], dim=-1)

        log_ree = self.log_ree_head(log_ree_input)

        output = self.process_outputs(
            sasa=sasa, log_rg=log_rg, log_diffusion=log_diffusion, log_ree=log_ree
        )

        return output

    def process_outputs(self, sasa, log_rg, log_diffusion, log_ree):
        """
        Processes the model outputs into a single concatenated tensor
        with the correct ordering for MSE loss.

        Args:
            sasa (torch.Tensor): Shape [B, 2]
            log_rg (torch.Tensor): Shape [B, 2]
            log_diffusion (torch.Tensor): Shape [B, 1]
            log_ree (torch.Tensor): Shape [B, 2]

        Returns:
            torch.Tensor: A single tensor with the ordering [log_rg, log_diffusion, sasa, log_ree]
        """

        # Concatenate in the correct order
        output_tensor = torch.cat(
            [log_rg, log_diffusion, sasa, log_ree], dim=-1
        )  # [B, 7]

        return output_tensor  # Shape [B, 7]
