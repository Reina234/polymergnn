import torch
import torch.nn as nn
from models.separate_mpnn_model.modified_configured_mpnn import AttentiveConfiguredMPNN


class JustMPNN(nn.Module):

    def __init__(
        self,
        mpnn_output_dim: int = 300,
        mpnn_hidden_dim: int = 300,
        mpnn_depth: int = 3,
        mpnn_dropout: float = 0.1,
        multitask_fnn_hidden_dim: int = 128,
    ):
        super().__init__()

        self.solvent_mpnn = AttentiveConfiguredMPNN(
            output_dim=mpnn_output_dim,
            d_h=mpnn_hidden_dim,
            depth=mpnn_depth,
            dropout=mpnn_dropout,
            undirected=True,
        )

        self.monomer_mpnn = AttentiveConfiguredMPNN(
            output_dim=mpnn_output_dim,
            d_h=mpnn_hidden_dim,
            depth=mpnn_depth,
            dropout=mpnn_dropout,
            undirected=True,
        )

        self.fnn = nn.Sequential(
            nn.Linear(mpnn_output_dim, multitask_fnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(multitask_fnn_hidden_dim, multitask_fnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(multitask_fnn_hidden_dim, multitask_fnn_hidden_dim),
            nn.ReLU(),
            nn.Linear(multitask_fnn_hidden_dim, 7),
        )

    def average_monomer_add_solvent(
        self, monomer_features, solvent_features, polymer_mapping
    ):
        """
        Averages variable-length monomer features per polymer and adds the corresponding solvent feature.
        """
        device = monomer_features.device
        feature_dim = monomer_features.shape[1]
        num_polymers = polymer_mapping.max().item() + 1

        combined_features = torch.zeros((num_polymers, feature_dim), device=device)

        monomer_idx = 0
        for i in range(num_polymers):
            count = (polymer_mapping == i).sum().item()
            num_monomers = count - 1
            avg = monomer_features[monomer_idx : monomer_idx + num_monomers].mean(dim=0)
            combined_features[i] = avg + solvent_features[i]
            monomer_idx += num_monomers

        return combined_features

    def forward(self, batch):

        monomer_results = self.monomer_mpnn(
            batch["batch_monomer_graph"],
        )

        solvent_results = self.solvent_mpnn(
            batch["batch_solvent_graph"],
        )

        combined = self.average_monomer_add_solvent(
            monomer_results, solvent_results, batch["polymer_mapping"]
        )

        predictions = self.fnn(combined)

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
