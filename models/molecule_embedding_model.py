import torch
import torch.nn as nn
from chemprop.nn import BondMessagePassing, NormAggregation, RegressionFFN
from chemprop.models.model import MPNN
from typing import List, Dict, Any
from chemprop.data import BatchMolGraph


class ConfiguredMPNN(nn.Module):
    def __init__(
        self,
        output_dim: int,
        aggregation_method=NormAggregation(),
        d_h: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        undirected: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        mp = BondMessagePassing(
            d_h=d_h, depth=depth, dropout=dropout, undirected=undirected
        )
        agg = aggregation_method
        fnn = RegressionFFN(hidden_dim=d_h, n_tasks=output_dim)
        self.model = MPNN(mp, agg, fnn)

    def forward(self, batch_mol_graph: BatchMolGraph):
        return self.model(batch_mol_graph)


class MoleculeEmbeddingModel(nn.Module):
    def __init__(
        self,
        chemprop_mpnn: ConfiguredMPNN,
        fusion_layer: nn.Sequential,
        mpnn_dim: int = 300,  # dimension of MPNN output
        chemberta_dim: int = 600,  # dimension of ChemBERTa embedding
        rdkit_dim: int = 10,  # if you don't want to use, pass in 0
        hidden_dim: int = 256,  # dimension for final molecule embedding
    ):
        super().__init__()

        self.mpnn = chemprop_mpnn

        self.bert_norm = nn.LayerNorm(chemberta_dim)
        if rdkit_dim > 0:
            self.rdkit_norm = nn.LayerNorm(rdkit_dim)
        else:
            self.rdkit_norm = None

        total_in = mpnn_dim + chemberta_dim + rdkit_dim
??? configure below?
        self.fusion = nn.Sequential(
            nn.Linear(total_in, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.hidden_dim = hidden_dim

    def forward(self, batch):
        """
        batch: dictionary from your collate_fn:
            {
              "batch_mol_graph": ...
              "chemberta_tensor": [N_mols, chemberta_dim]
              "rdkit_tensor": [N_mols, rdkit_dim] (optional)
              "system_indices": ...
              "polymer_feats": ...
              "polymer_mapping": ...
              "labels": ...
            }
        Returns:
          molecule_embs: [N_mols, hidden_dim]
            A final embedding for each molecule in the batch.
        """

        # 1) MPNN
        mpnn_out = self.mpnn.encoder(batch["batch_mol_graph"])
        # shape => [N_mols, mpnn_dim]

        # 2) LayerNorm on ChemBERTa
        chemberta_emb = batch["chemberta_tensor"]
        chemberta_emb = self.bert_norm(chemberta_emb)  # [N_mols, chemberta_dim]

        # 3) Optional: handle RDKit
        if (
            self.use_rdkit
            and self.rdkit_norm is not None
            and batch["rdkit_tensor"] is not None
        ):
            rdkit_feats = self.rdkit_norm(batch["rdkit_tensor"])  # [N_mols, rdkit_dim]
            # 4) Concat all
            fused_input = torch.cat([mpnn_out, chemberta_emb, rdkit_feats], dim=-1)
        else:
            fused_input = torch.cat([mpnn_out, chemberta_emb], dim=-1)

        # 5) Fusion MLP
        molecule_embs = self.fusion(fused_input)  # [N_mols, hidden_dim]

        return molecule_embs
