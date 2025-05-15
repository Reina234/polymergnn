import torch
import torch.nn as nn


class OldPolymerMultiTaskFNN(nn.Module):
    def __init__(
        self,
        input_dim: int,  # Polymer embedding + N, T
        shared_layer_dim: int,
        hidden_dim: int,
        dropout_rate: float = 0.2,
    ):
        super().__init__()

        # Shared representation
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, shared_layer_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
        )
        # Heads for individual property predictions
        self.sasa_head = nn.Sequential(
            nn.Linear(shared_layer_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)
        )
        self.log_rg_head = nn.Sequential(
            nn.Linear(shared_layer_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 2)
        )
        self.log_ree_head = nn.Sequential(
            nn.Linear(shared_layer_dim + 1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_diffusion_head = nn.Sequential(
            nn.Linear(input_dim + 1, hidden_dim),
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
        polymer_feats = batch["polymer_feats"]  # [B, 2] (N, T)
        scaling_factors = torch.tensor(
            [10.0, 290.0], device=polymer_feats.device
        )  # Ensure the same device
        normalized_feats = polymer_feats / scaling_factors
        # Concatenate polymer features with embeddings
        combined_input = torch.cat([polymer_embedding, normalized_feats], dim=-1)
        # Shared representation
        shared_repr = self.shared_layer(combined_input)

        sasa = self.sasa_head(shared_repr)  # [B, 2] (mean, std)

        log_rg = self.log_rg_head(shared_repr)  # [B, 2] (mean, std)

        diff_input = torch.cat([combined_input, sasa[:, 0].unsqueeze(-1)], dim=-1)

        log_diffusion = self.log_diffusion_head(diff_input)  # [B, 1]
        # log_diffusion = self.log_diffusion_head(shared_repr)
        log_ree_input = torch.cat([shared_repr, log_rg[:, 0].unsqueeze(-1)], dim=-1)

        log_ree = self.log_ree_head(log_ree_input)

        output = self.process_outputs(
            sasa=sasa, log_rg=log_rg, log_diffusion=log_diffusion, log_ree=log_ree
        )

        return output

    def process_outputs(self, sasa, log_rg, log_diffusion, log_ree):
        
        output_tensor = torch.cat(
            [log_rg, log_diffusion, sasa, log_ree], dim=-1
        )  

        return output_tensor  
