import torch
import torch.nn as nn
from models.molecule_embedding_model import MoleculeEmbeddingModel
from models.modules.configured_mpnn import ConfiguredMPNN
from models.temperature_aware_gnn.no_morgan_fnn import (
    DensityNoSharedLayerPolymerMultiTaskFNN,
)
from models.temperature_aware_gnn.gat_with_node_features import GATModuleNT
from featurisers.molecule_featuriser import RDKitFeaturizer
from chemprop.nn import NormAggregation


class DensityPolymerGNNSystem(nn.Module):
    def __init__(
        self,
        mpnn_output_dim: int = 300,
        mpnn_hidden_dim: int = 300,
        mpnn_depth: int = 3,
        mpnn_dropout: float = 0.1,
        rdkit_selection_tensor: torch.Tensor = None,
        molecule_embedding_hidden_dim: int = 512,
        embedding_dim: int = 256,
        use_rdkit: bool = True,
        use_chembert: bool = True,
        gnn_hidden_dim: int = 256,
        gnn_output_dim: int = 128,
        gnn_dropout: float = 0.1,
        gnn_num_heads: int = 4,
        multitask_fnn_shared_layer_dim: int = 128,
        multitask_fnn_hidden_dim: int = 128,
        multitask_fnn_dropout: float = 0.1,
    ):
        super().__init__()

        # Select RDKit Features
        rdkit_features = [
            "MolWt",
            "MolLogP",
            "MolMR",
            "TPSA",
            "NumRotatableBonds",
            "RingCount",
            "FractionCSP3",
        ]

        # Validate rdkit_selection_tensor
        if rdkit_selection_tensor is None:
            rdkit_selection_tensor = torch.ones(len(rdkit_features))  # Default: Use all
        elif rdkit_selection_tensor.shape[0] != len(rdkit_features):
            raise ValueError(
                f"rdkit_selection_tensor must have {len(rdkit_features)} elements!"
            )

        selected_rdkit_features = [
            feat
            for feat, select in zip(rdkit_features, rdkit_selection_tensor.tolist())
            if select == 1
        ]

        # Initialize MPNN
        mpnn = ConfiguredMPNN(
            output_dim=mpnn_output_dim,
            aggregation_method=NormAggregation(),
            d_h=mpnn_hidden_dim,
            depth=mpnn_depth,
            dropout=mpnn_dropout,
            undirected=True,
        )

        # Molecule Embedding Model
        self.molecule_embedding = MoleculeEmbeddingModel(
            chemprop_mpnn=mpnn,
            rdkit_featurizer=RDKitFeaturizer(),
            selected_rdkit_features=selected_rdkit_features,
            chemberta_dim=600,
            hidden_dim=molecule_embedding_hidden_dim,
            output_dim=embedding_dim,
            use_rdkit=use_rdkit,
            use_chembert=use_chembert,
        )

        self.polymer_gnn = GATModuleNT(
            input_dim=embedding_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            dropout_rate=gnn_dropout,
            num_heads=gnn_num_heads,
        )

        self.polymer_fnn = DensityNoSharedLayerPolymerMultiTaskFNN(
            input_dim=gnn_output_dim,
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
