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

        super().__init__()

        input_dim_fnn = input_dim + 3  # Add 1 for T
        # Shared representation
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim_fnn + 2 * n_bits, shared_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Heads for individual property predictions
        self.rg_mu_sasa_re_head = nn.Sequential(
            nn.Linear(shared_layer_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 4),
        )
        self.log_rg_re_sigma_head = nn.Sequential(
            nn.Linear(shared_layer_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

        self.log_diffusion_head = nn.Sequential(
            nn.Linear(shared_layer_dim, hidden_dim),
            nn.ReLU(),
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
        n_bits = batch["fingerprints_tensor"]
        polymer_feats = batch["polymer_feats"][:, 0:3]  # [B, 2] (D (?), N, T)
        scaling_factors = torch.tensor([1, 10.0, 100.0], device=polymer_feats.device)
        normalized_feats = polymer_feats / scaling_factors
        combined_input = torch.cat(
            [polymer_embedding, normalized_feats, n_bits], dim=-1
        )

        shared_repr = self.shared_layer(combined_input)

        log_rg_re_sigma = self.log_rg_re_sigma_head(shared_repr)  # [B, 2] (mean, std)

        rg_mu_sasa_re = self.rg_mu_sasa_re_head(shared_repr)  # [B, 2] (mean, std)

        log_diffusion = self.log_diffusion_head(shared_repr)  # [B, 1]
        # log_diffusion = self.log_diffusion_head(shared_repr)

        output = self.process_outputs(log_rg_re_sigma, rg_mu_sasa_re, log_diffusion)

        return output

    def process_outputs(self, log_rg_re_sigma, rg_mu_sasa_re, log_diffusion):

        log_rg_sigma = log_rg_re_sigma[:, 0].unsqueeze(-1)
        log_re_sigma = log_rg_re_sigma[:, 1].unsqueeze(-1)

        rg_mu = rg_mu_sasa_re[:, 0].unsqueeze(-1)
        sasa = rg_mu_sasa_re[:, 1:3]
        re = rg_mu_sasa_re[:, 3].unsqueeze(-1)

        output_tensor = torch.cat(
            [rg_mu, log_rg_sigma, log_diffusion, sasa, re, log_re_sigma], dim=-1
        )

        return output_tensor
