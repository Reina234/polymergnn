import torch
import torch.nn as nn
from torch_geometric.nn import global_mean_pool


class MoreLayerMultiTaskFNNV4(nn.Module):
    def __init__(
        self,
        input_dim: int,  # GNN embedding (monomer + solvent)
        shared_layer_dim: int,
        hidden_dim: int,
        dropout_rate: float = 0.2,
    ):
        """
        Multi-task FNN for predicting polymer properties.

        Args:
        - input_dim (int): Input size (GNN output + polymer features N, T).
        - shared_layer_dim (int): Size of shared representation layers.
        - hidden_dim (int): Hidden layer size for property heads.
        - dropout_rate (float): Dropout for regularization.
        """
        super().__init__()

        # Polymer features (D, N, T) are concatenated later
        input_dim_fnn = input_dim + 3

        # Shared representation for non-diffusion tasks
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim_fnn, shared_layer_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(shared_layer_dim, shared_layer_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
        )

        # Property prediction heads
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

        # Diffusion head (Uses pooled monomer embeddings + solvent embeddings + SASA)
        self.log_diffusion_head = nn.Sequential(
            nn.Linear(
                input_dim + input_dim // 2 + 1, hidden_dim
            ),  # MPNN pooled + solvent + SASA
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, batch):
        """
        Forward pass for the FNN.

        Args:
        - batch (dict): Contains:
            - "node_features": [N_monomers, embedding_dim] (Direct MPNN output)
            - "solvent_embedding": [B, embedding_dim] (single solvent per polymer)
            - "polymer_mapping": [N_monomers] (mapping of monomers to batches)
            - "polymer_feats": Additional polymer properties (D, N, T) [B, 3]

        Returns:
        - Dict with predicted polymer properties
        """
        # Extract features
        monomer_embeddings = batch[
            "monomer_node_features"
        ]  # [N_monomers, embedding_dim]
        solvent_embedding = batch["solvent_node_features"]  # [B, embedding_dim]
        polymer_mapping = batch["polymer_mapping"]  # [N_monomers] → batch index

        # Normalize polymer features (D, N, T)
        polymer_feats = batch["polymer_feats"][:, 0:3]  # [B, 3]
        scaling_factors = torch.tensor([1, 10.0, 100.0], device=polymer_feats.device)
        normalized_feats = polymer_feats / scaling_factors

        # Shared representation for non-diffusion tasks
        combined_input = torch.cat(
            [solvent_embedding, normalized_feats], dim=-1
        )  # [B, input_dim_fnn]
        shared_repr = self.shared_layer(combined_input)

        # Property predictions
        sasa = self.sasa_head(shared_repr)  # [B, 2]
        log_rg = self.log_rg_head(shared_repr)  # [B, 2]

        # Pool monomer embeddings **only for diffusion**
        monomer_pooled = global_mean_pool(
            monomer_embeddings, polymer_mapping
        )  # [B, embedding_dim]
        diffusion_input = torch.cat(
            [monomer_pooled, solvent_embedding, sasa[:, 0].unsqueeze(-1)], dim=-1
        )  # [B, input_dim + input_dim//2 + 1]

        log_diffusion = self.log_diffusion_head(diffusion_input)  # [B, 1]

        log_ree_input = torch.cat([shared_repr, log_rg[:, 0].unsqueeze(-1)], dim=-1)
        log_ree = self.log_ree_head(log_ree_input)  # [B, 1]

        output = self.process_outputs(
            sasa=sasa, log_rg=log_rg, log_diffusion=log_diffusion, log_ree=log_ree
        )

        return output

    def process_outputs(self, sasa, log_rg, log_diffusion, log_ree):
        """
        Processes the model outputs into a single concatenated tensor.

        Args:
            sasa (torch.Tensor): Shape [B, 2]
            log_rg (torch.Tensor): Shape [B, 2]
            log_diffusion (torch.Tensor): Shape [B, 1]
            log_ree (torch.Tensor): Shape [B, 2]

        Returns:
            torch.Tensor: A single tensor with the ordering [log_rg, log_diffusion, sasa, log_ree]
        """
        output_tensor = torch.cat(
            [log_rg, log_diffusion, sasa, log_ree], dim=-1
        )  # [B, 7]
        return output_tensor
