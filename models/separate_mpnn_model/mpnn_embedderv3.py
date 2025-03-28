import torch
import torch.nn as nn
from typing import List
from models.modules.configured_mpnn import ConfiguredMPNN
from featurisers.molecule_featuriser import RDKitFeaturizer


class NormalizedFusionMoleculeEmbeddingModel(nn.Module):
    def __init__(
        self,
        chemprop_mpnn: ConfiguredMPNN,  # ChemProp-based MPNN instance
        rdkit_featurizer: RDKitFeaturizer,  # RDKit feature extractor
        selected_rdkit_features: List[str],
        hidden_dim: int,
        output_dim: int,  # Final embedding output size
        projection_ratio: float = 0.5,  # Ratio of MPNN size for RDKit projection
    ):
        super().__init__()

        self.mpnn = chemprop_mpnn
        self.rdkit_featurizer = rdkit_featurizer
        self.selected_rdkit_features = selected_rdkit_features

        # Feature dimensions
        self.mpnn_dim = self.mpnn.output_dim  # e.g., 96
        self.rdkit_dim = len(selected_rdkit_features)  # e.g., 7

        # RDKit feature projection
        self.projected_rdkit_dim = int(self.mpnn_dim * projection_ratio)
        self.rdkit_projection = nn.Linear(self.rdkit_dim, self.projected_rdkit_dim)

        # Layer Normalization for stability
        # self.mpnn_norm = nn.LayerNorm(self.mpnn_dim)
        # self.rdkit_norm = nn.LayerNorm(self.projected_rdkit_dim)

        # Fusion Layer
        fused_dim = self.mpnn_dim + self.projected_rdkit_dim
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, molgraphs, rdkit_tensor):
        """
        Forward pass for molecule embedding.

        Args:
        - molgraphs: Molecular graph batch (for MPNN)
        - rdkit_tensor: Precomputed RDKit descriptors

        Returns:
        - Molecule embeddings after normalization-based fusion
        """
        # Compute embeddings from MPNN
        mpnn_out = self.mpnn(molgraphs)  # [batch, mpnn_dim]
        # mpnn_out = self.mpnn_norm(mpnn_out)

        # Extract and project RDKit features
        selected_rdkit = self.rdkit_featurizer.select_features(
            rdkit_tensor, self.selected_rdkit_features
        )  # [batch, rdkit_dim]
        projected_rdkit = self.rdkit_projection(selected_rdkit)
        # projected_rdkit = self.rdkit_norm(projected_rdkit)

        # Concatenate features
        fused_input = torch.cat([mpnn_out, projected_rdkit], dim=-1)

        # Final processing via fusion network
        molecule_embs = self.fusion(fused_input)

        return molecule_embs
