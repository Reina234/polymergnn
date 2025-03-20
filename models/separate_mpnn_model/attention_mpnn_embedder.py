import torch
import torch.nn as nn
from typing import List
from models.modules.configured_mpnn import ConfiguredMPNN
from featurisers.molecule_featuriser import RDKitFeaturizer


class AttentionFusionMoleculeEmbeddingModel(nn.Module):
    def __init__(
        self,
        chemprop_mpnn: ConfiguredMPNN,  # ChemProp-based MPNN instance
        rdkit_featurizer: RDKitFeaturizer,  # RDKit feature extractor
        selected_rdkit_features: List[str],
        hidden_dim: int,
        output_dim: int,  # Final embedding output size
    ):
        super().__init__()

        self.mpnn = chemprop_mpnn
        self.rdkit_featurizer = rdkit_featurizer
        self.selected_rdkit_features = selected_rdkit_features

        # RDKit feature dimension based on selected descriptors
        self.rdkit_dim = len(selected_rdkit_features)

        # Compute max feature size
        self.mpnn_dim = self.mpnn.output_dim  # 96 in your case
        self.feature_dim = max(
            self.mpnn_dim, self.rdkit_dim
        )  # Ensure consistent feature size

        # Attention mechanism per feature dimension
        self.attn_fc = nn.Sequential(
            nn.Linear(self.mpnn_dim + self.rdkit_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(
                hidden_dim, (self.mpnn_dim + self.rdkit_dim)
            ),  # Per-feature attention
            nn.Softmax(dim=-1),  # Ensure values sum to 1
        )

        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(self.mpnn_dim + self.rdkit_dim, hidden_dim),
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
        - Molecule embeddings after attention-based fusion
        """
        # Compute embeddings from MPNN
        mpnn_out = self.mpnn(molgraphs)  # [batch, 96]

        # Extract selected RDKit features
        selected_rdkit = self.rdkit_featurizer.select_features(
            rdkit_tensor, self.selected_rdkit_features
        )  # [batch, 7]

        combined_input = torch.cat([mpnn_out, selected_rdkit], dim=-1)
        attn_scores = self.attn_fc(combined_input)
        fused_embedding = attn_scores * combined_input
        # Final processing via fusion network
        molecule_embs = self.fusion(fused_embedding)

        return molecule_embs
