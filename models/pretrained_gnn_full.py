import torch
import torch.nn as nn
from models.molecule_embedding_model import MoleculeEmbeddingModel
from models.modules.gat_module import GATModuleMod
from models.poly_multitask_fnn import PolymerMultiTaskFNN


class PretrainedMPNNGNN(nn.Module):
    def __init__(
        self,
        pretrained_embedding_model: MoleculeEmbeddingModel,
        embedding_dim: int = 256,
        gnn_hidden_dim: int = 256,
        gnn_output_dim: int = 128,
        gnn_dropout: float = 0.1,
        gnn_num_heads: int = 4,
        multitask_fnn_shared_layer_dim: int = 128,
        multitask_fnn_hidden_dim: int = 128,
        multitask_fnn_dropout: float = 0.1,
    ):
        super().__init__()

        self.molecule_embedding = pretrained_embedding_model

        self.polymer_gnn = GATModuleMod(
            input_dim=embedding_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            dropout_rate=gnn_dropout,
            num_heads=gnn_num_heads,
        )

        self.polymer_fnn = PolymerMultiTaskFNN(
            input_dim=gnn_output_dim + 2,  # +2 for the N and T from polymer feats
            shared_layer_dim=multitask_fnn_shared_layer_dim,
            hidden_dim=multitask_fnn_hidden_dim,
            dropout_rate=multitask_fnn_dropout,
        )

    def forward(self, batch, return_intermediates=False):
        batch["node_features"], mpnn_out, chemberta_emb, rdkit_emb = (
            self.molecule_embedding(batch)
        )

        batch["polymer_embedding"] = self.polymer_gnn(batch)

        predictions = self.polymer_fnn(batch)

        if return_intermediates:
            return (
                predictions,
                mpnn_out,
                chemberta_emb,
                rdkit_emb,
                batch["polymer_embedding"],
            )
        return predictions
