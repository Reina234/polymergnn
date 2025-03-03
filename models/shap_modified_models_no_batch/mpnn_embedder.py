import torch
import torch.nn as nn
from typing import List, Optional
from featurisers.molecule_featuriser import RDKitFeaturizer


class ShapMoleculeEmbeddingModel(nn.Module):

    def __init__(
        self,  # ConfiguredMPNN instance
        mpnn_output_dim,
        rdkit_featurizer: RDKitFeaturizer,
        selected_rdkit_features: List[str],
        hidden_dim: int,
        output_dim: Optional[int] = None,
    ):
        super().__init__()
        if not output_dim:
            output_dim = hidden_dim

        self.rdkit_featurizer = rdkit_featurizer
        self.selected_rdkit_features = selected_rdkit_features
        self.rdkit_dim = len(selected_rdkit_features)

        total_in = mpnn_output_dim + self.rdkit_dim

        self.rdkit_norm = nn.LayerNorm(self.rdkit_dim) if self.rdkit_dim > 0 else None

        self.fusion = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, mpnn_out, full_rdkit_tensor):
        selected_rdkit = self.rdkit_featurizer.select_features(
            full_rdkit_tensor, self.selected_rdkit_features
        )
        rdkit_emb = (
            self.rdkit_norm(selected_rdkit) if self.rdkit_norm else selected_rdkit
        )

        fused_input = torch.cat([mpnn_out, rdkit_emb], dim=-1)
        molecule_embs = self.fusion(fused_input)
        return molecule_embs
