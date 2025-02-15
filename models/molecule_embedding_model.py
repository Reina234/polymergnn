import torch
import torch.nn as nn
from typing import List
from modules.configured_mpnn import ConfiguredMPNN
from featurisers.molecule_featuriser import RDKitFeaturizer


class MoleculeEmbeddingModel(nn.Module):
    def __init__(
        self,
        chemprop_mpnn: ConfiguredMPNN,
        rdkit_featurizer: RDKitFeaturizer,
        selected_rdkit_features: List[str] = [
            "MolWt",
            "MolLogP",
            "MolMR",
            "TPSA",
            "NumRotatableBonds",
            "RingCount",
            "FractionCSP3",
        ],
        chemberta_dim: int = 600,
        hidden_dim: int = 256,
    ):
        super().__init__()
        self.mpnn = chemprop_mpnn
        self.rdkit_featurizer = rdkit_featurizer
        self.selected_rdkit_features = selected_rdkit_features

        self.rdkit_dim = len(selected_rdkit_features)

        self.bert_norm = nn.LayerNorm(chemberta_dim)
        self.rdkit_norm = nn.LayerNorm(self.rdkit_dim) if self.rdkit_dim > 0 else None

        total_in = self.mpnn.output_dim + chemberta_dim + self.rdkit_dim

        self.fusion = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, batch):

        # 1) MPNN embeddings
        mpnn_out = self.mpnn(batch["batch_mol_graph"])

        # 2) ChemBERTa embeddings
        chemberta_emb = self.bert_norm(batch["chemberta_tensor"])
        chemberta_emb = chemberta_emb.squeeze(1)

        # 3) RDKit feature selection and normalization (if applicable)
        if self.selected_rdkit_features and "rdkit_tensor" in batch:
            full_rdkit_tensor = batch["rdkit_tensor"]
            selected_rdkit = self.rdkit_featurizer.select_features(
                full_rdkit_tensor, self.selected_rdkit_features
            )
            rdkit_emb = self.rdkit_norm(selected_rdkit) if self.rdkit_norm else None

            fused_input = torch.cat([mpnn_out, chemberta_emb, rdkit_emb], dim=-1)
        else:
            fused_input = torch.cat([mpnn_out, chemberta_emb], dim=-1)

        molecule_embs = self.fusion(fused_input)
        return molecule_embs
