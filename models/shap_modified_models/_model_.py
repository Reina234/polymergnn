import torch
import torch.nn as nn
from models.shap_modified_models.mpnn_config import ShapMPNN
from models.shap_modified_models.mpnn_embedder import ShapMoleculeEmbeddingModel
from models.shap_modified_models.gat import ShapGATModule
from featurisers.molecule_featuriser import RDKitFeaturizer
from models.shap_modified_models.multitask_fnn import ShapMorganPolymerMultiTaskFNN


class PolymerGNNNoMPNNsSystem(nn.Module):

    def __init__(
        self,
        d_e: int = 50,
        d_v: int = 50,
        n_bits: int = 2048,
        mpnn_output_dim: int = 300,
        mpnn_hidden_dim: int = 300,
        mpnn_depth: int = 3,
        mpnn_dropout: float = 0.1,
        rdkit_selection_tensor: torch.Tensor = None,
        molecule_embedding_hidden_dim: int = 512,
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
        rdkit_features = [
            "MolWt",
            "MolLogP",
            "MolMR",
            "TPSA",
            "NumRotatableBonds",
            "RingCount",
            "FractionCSP3",
        ]

        if rdkit_selection_tensor is None:
            rdkit_selection_tensor = torch.ones(len(rdkit_features))
        elif rdkit_selection_tensor.shape[0] != len(rdkit_features):
            raise ValueError(
                f"rdkit_selection_tensor must have {len(rdkit_features)} elements!"
            )

        selected_rdkit_features = [
            feat
            for feat, select in zip(rdkit_features, rdkit_selection_tensor.tolist())
            if select == 1
        ]

        self.mpnn = ShapMPNN(
            d_e=d_e,
            d_v=d_v,
            output_dim=mpnn_output_dim,
            d_h=mpnn_hidden_dim,
            depth=mpnn_depth,
            dropout=mpnn_dropout,
        )

        self.molecule_embedding = ShapMoleculeEmbeddingModel(
            rdkit_featurizer=RDKitFeaturizer(),
            selected_rdkit_features=selected_rdkit_features,
            hidden_dim=molecule_embedding_hidden_dim,
            output_dim=embedding_dim,
        )

        self.polymer_gnn = ShapGATModule(
            input_dim=embedding_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            dropout_rate=gnn_dropout,
            num_heads=gnn_num_heads,
        )

        self.polymer_fnn = ShapMorganPolymerMultiTaskFNN(
            input_dim=gnn_output_dim + 2,  # +2 for the N and T from polymer feats
            shared_layer_dim=multitask_fnn_shared_layer_dim,
            hidden_dim=multitask_fnn_hidden_dim,
            n_bits=n_bits,
            dropout_rate=multitask_fnn_dropout,
        )

    def forward(self, batch):
        batch["mpnn_out"] = self.mpnn(batch)
        batch["node_features"] = self.molecule_embedding(batch)

        batch["polymer_embedding"] = self.polymer_gnn(batch)

        predictions = self.polymer_fnn(batch)

        return predictions
