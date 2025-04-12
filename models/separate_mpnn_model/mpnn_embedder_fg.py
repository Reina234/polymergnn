import torch
import torch.nn as nn
from typing import List
from models.modules.configured_mpnn import ConfiguredMPNN
from featurisers.molecule_featuriser import RDKitFeaturizer


class FGFusionMoleculeEmbeddingModel(nn.Module):

    def __init__(
        self,
        chemprop_mpnn: ConfiguredMPNN,
        rdkit_featurizer: RDKitFeaturizer,  # RDKit feature extractor
        selected_rdkit_features: List[str],
        hidden_dim: int,
        output_dim: int,  # Final embedding output size
        fg_n_bits_dim: int = 64,
        projection_ratio: float = 0.5,
    ):
        super().__init__()

        self.mpnn = chemprop_mpnn
        self.rdkit_featurizer = rdkit_featurizer
        self.selected_rdkit_features = selected_rdkit_features

        self.mpnn_dim = self.mpnn.output_dim  # e.g., 96
        self.rdkit_dim = len(selected_rdkit_features)  # e.g., 7

        # RDKit feature projection
        projection_dim = int(self.mpnn_dim * projection_ratio)
        self.rdkit_projection = nn.Linear(self.rdkit_dim, projection_dim)

        # Layer Normalization for stability
        self.mpnn_norm = nn.LayerNorm(self.mpnn_dim)
        self.rdkit_norm = nn.LayerNorm(projection_dim)

        self.fg_projection = nn.Linear(fg_n_bits_dim, projection_dim)

        # Fusion Layer
        fused_dim = self.mpnn_dim + projection_dim * 2
        self.fusion = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, molgraphs, rdkit_tensor, fg_tensor):  # CONTINUEEE
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
        mpnn_out = self.mpnn_norm(mpnn_out)

        # Extract and project RDKit features
        selected_rdkit = self.rdkit_featurizer.select_features(
            rdkit_tensor, self.selected_rdkit_features
        )  # [batch, rdkit_dim]
        projected_rdkit = self.rdkit_projection(selected_rdkit)
        projected_rdkit = self.rdkit_norm(projected_rdkit)

        projected_fg = self.fg_projection(fg_tensor)

        # Concatenate features
        fused_input = torch.cat([mpnn_out, projected_rdkit, projected_fg], dim=-1)

        # Final processing via fusion network
        molecule_embs = self.fusion(fused_input)

        return molecule_embs
