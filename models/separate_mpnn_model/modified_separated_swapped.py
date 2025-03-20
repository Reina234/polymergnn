import torch
import torch.nn as nn


class SwappedMoreLayerMultiTaskFNNV2(nn.Module):
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
            nn.Linear(shared_layer_dim, shared_layer_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )
        # Heads for individual property predictions
        self.sasa_head = nn.Sequential(
            nn.Linear(shared_layer_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, 2)
        )
        self.log_rg_head = nn.Sequential(
            nn.Linear(shared_layer_dim + 1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2),
        )
        self.log_ree_head = nn.Sequential(
            nn.Linear(shared_layer_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_diffusion_head = nn.Sequential(
            nn.Linear(input_dim_fnn + 1, hidden_dim),
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

        polymer_embedding = batch["polymer_embedding"]  # [B, embedding_dim]
        # polymer_feats = batch["polymer_feats"][:, 2]  # [B, 2] (N, T) -> (T)
        # normalized_T = (polymer_feats / 100).unsqueeze(-1)  # Shape: [B, 1]
        polymer_feats = batch["polymer_feats"][:, 0:3]  # [B, 2] (D (?), N, T)
        scaling_factors = torch.tensor([1, 10.0, 100.0], device=polymer_feats.device)
        normalized_feats = polymer_feats / scaling_factors
        # Concatenate polymer features with embeddings
        combined_input = torch.cat([polymer_embedding, normalized_feats], dim=-1)
        # Shared representation
        shared_repr = self.shared_layer(combined_input)

        sasa = self.sasa_head(shared_repr)  # [B, 2] (mean, std)

        diff_input = torch.cat([combined_input, sasa[:, 0].unsqueeze(-1)], dim=-1)

        log_diffusion = self.log_diffusion_head(diff_input)  # [B, 1]
        # log_diffusion = self.log_diffusion_head(shared_repr)
        log_ree = self.log_ree_head(shared_repr)
        log_rg_input = torch.cat([shared_repr, log_ree[:, 0].unsqueeze(-1)], dim=-1)

        log_rg = self.log_rg_head(log_rg_input)  # [B, 2] (mean, std)
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
