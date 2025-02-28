import torch
import torch.nn as nn
from typing import List, Optional
from modules.configured_mpnn import ConfiguredMPNN
from featurisers.molecule_featuriser import RDKitFeaturizer


class MoleculeEmbeddingModel(nn.Module):

    def __init__(
        self,
        chemprop_mpnn: ConfiguredMPNN,  # ConfiguredMPNN instance
        rdkit_featurizer: RDKitFeaturizer,
        selected_rdkit_features: List[str],
        chemberta_dim: int,
        hidden_dim: int,
        output_dim: Optional[int] = None,  # mpnn output dim
        use_rdkit: bool = True,
        use_chembert: bool = True,
    ):
        super().__init__()
        if not output_dim:
            output_dim = hidden_dim
        self.mpnn = chemprop_mpnn
        self.rdkit_featurizer = rdkit_featurizer if use_rdkit else None
        self.selected_rdkit_features = selected_rdkit_features if use_rdkit else None
        self.use_rdkit = use_rdkit
        self.use_chembert = use_chembert

        # If not using ChemBERTa, we set its dimension to zero.
        self.chemberta_dim = chemberta_dim if use_chembert else 0

        # RDKit dimension based on number of selected features.
        self.rdkit_dim = len(selected_rdkit_features) if use_rdkit else 0

        # The total input dimension is now:
        # MP-embedding + (ChemBERTa if used) + (RDKit features if used)
        total_in = self.mpnn.output_dim + self.chemberta_dim + self.rdkit_dim

        self.bert_norm = nn.LayerNorm(chemberta_dim) if use_chembert else None
        self.rdkit_norm = (
            nn.LayerNorm(self.rdkit_dim) if (use_rdkit and self.rdkit_dim > 0) else None
        )

        self.fusion = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, batch):
        mpnn_out = self.mpnn(batch["batch_mol_graph"])

        if self.use_chembert:
            chemberta_emb = self.bert_norm(batch["chemberta_tensor"])
            chemberta_emb = chemberta_emb.squeeze(1)
        else:
            chemberta_emb = torch.empty(mpnn_out.size(0), 0, device=mpnn_out.device)

        if self.use_rdkit and "rdkit_tensor" in batch:
            full_rdkit_tensor = batch["rdkit_tensor"]
            selected_rdkit = self.rdkit_featurizer.select_features(
                full_rdkit_tensor, self.selected_rdkit_features
            )
            rdkit_emb = (
                self.rdkit_norm(selected_rdkit) if self.rdkit_norm else selected_rdkit
            )
        else:
            rdkit_emb = torch.empty(mpnn_out.size(0), 0, device=mpnn_out.device)

        fused_input = torch.cat([mpnn_out, chemberta_emb, rdkit_emb], dim=-1)
        molecule_embs = self.fusion(fused_input)
        return molecule_embs, mpnn_out, chemberta_emb, rdkit_emb
