import torch
import torch.nn as nn
from models.shap_modified_models_no_batch.mpnn_embedder import (
    ShapMoleculeEmbeddingModel,
)
from models.shap_modified_models_no_batch.gat import ShapGATModule
from featurisers.molecule_featuriser import RDKitFeaturizer
from models.shap_modified_models_no_batch.multitask_fnn import (
    ShapMorganPolymerMultiTaskFNN,
)


class PolymerGNNNoMPNNsSystem(nn.Module):

    def __init__(
        self,
        mpnn,
        n_bits: int = 2048,
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

        self.mpnn = mpnn

        self.molecule_embedding = ShapMoleculeEmbeddingModel(
            mpnn_output_dim=mpnn.output_dim,
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

    def forward(
        self,
        mpnn_out,
        full_rdkit_tensor,
        polymer_feats,
        fingerprints,
        edge_index,
        edge_attr,
        polymer_mapping,
    ):

        molecule_embedding = self.molecule_embedding(mpnn_out, full_rdkit_tensor)

        polymer_embedding = self.polymer_gnn(
            molecule_embedding, edge_index, edge_attr, polymer_mapping
        )

        predictions = self.polymer_fnn(polymer_embedding, polymer_feats, fingerprints)
        print("forward pass happened")
        return predictions


# need to now try shap, load up a pre-loaded mpnn as the starting point to create batch[mpnn_out]
# then will need to split rdkit tensor, so one from batch, into our individual features, to then reconstruct in perturbation
# extract the .mpnn after loading the weights/state dict into one of the pretrained one, then reconstruct into wrapper
