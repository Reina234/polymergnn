import torch
import torch.nn as nn
from models.separate_mpnn_model.mpnn_embedderv2 import (
    NormalizedFusionMoleculeEmbeddingModel,
)
from models.separate_mpnn_model.modified_configured_mpnn import AttentiveConfiguredMPNN
from models.separate_mpnn_model.modified_fnn_v3 import NewConfigFNNV2
from models.separate_mpnn_model.skip_gat import SkipGatedGATModule
from featurisers.molecule_featuriser import RDKitFeaturizer


class SeparatedGNNSystemV3(nn.Module):

    def __init__(
        self,
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

        mpnn = AttentiveConfiguredMPNN(
            output_dim=mpnn_output_dim,
            d_h=mpnn_hidden_dim,
            depth=mpnn_depth,
            dropout=mpnn_dropout,
            undirected=True,
        )

        self.monomer_embedding = NormalizedFusionMoleculeEmbeddingModel(
            chemprop_mpnn=mpnn,
            rdkit_featurizer=RDKitFeaturizer(),
            selected_rdkit_features=selected_rdkit_features,
            hidden_dim=molecule_embedding_hidden_dim,
            output_dim=embedding_dim,
        )
        self.solvent_embedding = NormalizedFusionMoleculeEmbeddingModel(
            chemprop_mpnn=mpnn,
            rdkit_featurizer=RDKitFeaturizer(),
            selected_rdkit_features=selected_rdkit_features,
            hidden_dim=molecule_embedding_hidden_dim,
            output_dim=embedding_dim,
        )

        self.polymer_gnn = SkipGatedGATModule(
            input_dim=embedding_dim,
            hidden_dim=gnn_hidden_dim,
            output_dim=gnn_output_dim,
            dropout_rate=gnn_dropout,
            num_heads=gnn_num_heads,
        )

        self.polymer_fnn = NewConfigFNNV2(
            input_dim=gnn_output_dim,
            shared_layer_dim=multitask_fnn_shared_layer_dim,
            hidden_dim=multitask_fnn_hidden_dim,
            dropout_rate=multitask_fnn_dropout,
        )

    def forward(self, batch):
        batch["monomer_node_features"] = self.monomer_embedding(
            batch["batch_monomer_graph"],
            batch["monomer_rdkit_tensor"],
        )

        batch["solvent_node_features"] = self.solvent_embedding(
            batch["batch_solvent_graph"],
            batch["solvent_rdkit_tensor"],
        )

        batch["node_features"] = self.recombine_node_features(
            batch["monomer_node_features"],
            batch["solvent_node_features"],
            batch["polymer_mapping"],
        )

        batch["polymer_embedding"] = self.polymer_gnn(batch)

        predictions = self.polymer_fnn(batch)

        return predictions

    @staticmethod
    def recombine_node_features(monomer_features, solvent_features, polymer_mapping):
        device = monomer_features.device  # Ensure correct tensor placement
        feature_dim = monomer_features.shape[1]

        num_molecules = len(polymer_mapping)  # Total number of molecules
        num_polymers = (
            polymer_mapping[-1] + 1
        )  # Number of polymer systems (since it's sorted)

        full_features = torch.zeros((num_molecules, feature_dim), device=device)

        monomer_idx, solvent_idx = 0, 0  # Track indices in monomer & solvent tensors
        start_idx = 0  # Track where each polymer system starts

        for i in range(num_polymers):
            # Get the number of molecules in the current polymer system
            num_molecules_in_polymer = (polymer_mapping == i).sum().item()

            # The last molecule is always the solvent
            num_monomers = num_molecules_in_polymer - 1

            # Get indices of this polymer's molecules in full_features
            polymer_indices = torch.arange(
                start_idx, start_idx + num_molecules_in_polymer, device=device
            )

            # Assign monomer features
            full_features[polymer_indices[:-1]] = monomer_features[
                monomer_idx : monomer_idx + num_monomers
            ]
            monomer_idx += num_monomers

            # Assign solvent feature (last molecule in system)
            full_features[polymer_indices[-1]] = solvent_features[solvent_idx]
            solvent_idx += 1

            # Move to the next polymer system
            start_idx += num_molecules_in_polymer

        return full_features
