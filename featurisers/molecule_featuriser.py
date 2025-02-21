import torch
from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem.rdchem import Mol


class RDKitFeaturizer:
    def __init__(self):
        """Initialize RDKitFeaturizer with all feature names and their indices."""
        self.feature_names = [
            "NumHDonors",
            "NumHAcceptors",
            "MolWt",
            "MolLogP",
            "MolMR",
            "TPSA",
            "NumRotatableBonds",
            "RingCount",
            "FractionCSP3",
        ]
        self.feature_map = {name: i for i, name in enumerate(self.feature_names)}

    def featurise(self, mol: Mol) -> torch.Tensor:
        """
        Extracts molecular descriptors from an RDKit Mol object.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

        Returns:
            torch.Tensor: Tensor of computed molecular properties.
        """
        if mol is None:
            return torch.tensor([float("nan")] * len(self.feature_names))

        features = [
            Lipinski.NumHDonors(mol),
            Lipinski.NumHAcceptors(mol),
            Descriptors.MolWt(mol),
            Crippen.MolLogP(mol),
            Crippen.MolMR(mol),
            rdMolDescriptors.CalcTPSA(mol),
            Lipinski.NumRotatableBonds(mol),
            Lipinski.RingCount(mol),
            rdMolDescriptors.CalcFractionCSP3(mol),
        ]

        return torch.tensor(features, dtype=torch.float32)

    def select_features(
        self, rdkit_tensor: torch.Tensor, features_to_keep: list
    ) -> torch.Tensor:
        """
        Select specific RDKit features from the full feature tensor.

        Args:
            rdkit_tensor (torch.Tensor): Full RDKit feature tensor [N, total_features].
            features_to_keep (list): List of feature names to extract.

        Returns:
            torch.Tensor: Tensor containing only the selected features [N, len(features_to_keep)].
        """
        indices = [
            self.feature_map[feat]
            for feat in features_to_keep
            if feat in self.feature_map
        ]
        if not indices:
            raise ValueError(f"No valid features selected from {features_to_keep}")
        return rdkit_tensor[:, indices]

    def get_feature_map(self) -> dict:
        """Return the RDKit feature name-to-index mapping."""
        return self.feature_map

    def compute_hbond_matrix(self, rdkit_features_list):
        """
        torch.Tensor: A square matrix of shape (n, n), where n = len(rdkit_features_list).
        """
        donor_idx = self.feature_map["NumHDonors"]
        acceptor_idx = self.feature_map["NumHAcceptors"]
        n = len(rdkit_features_list)
        hbond_matrix = torch.zeros(n, n, dtype=torch.float32)
        for i in range(n):
            donors_i = rdkit_features_list[i][donor_idx].item()
            acceptors_i = rdkit_features_list[i][acceptor_idx].item()
            for j in range(n):
                donors_j = rdkit_features_list[j][donor_idx].item()
                acceptors_j = rdkit_features_list[j][acceptor_idx].item()
                hbond_value = max(acceptors_i, donors_j) + max(donors_i, acceptors_j)
                hbond_matrix[i, j] = hbond_value
        return hbond_matrix
