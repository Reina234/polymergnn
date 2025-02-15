from rdkit.Chem import Descriptors, Lipinski, Crippen, rdMolDescriptors
from rdkit.Chem.rdchem import Mol
import torch


class RDKitFeaturizer:
    def __init__(self):
        pass

    def featurise(self, mol: Mol) -> dict:
        """
        Extracts molecular descriptors from an RDKit Mol object.

        Args:
            mol (rdkit.Chem.rdchem.Mol): RDKit molecule object.

        Returns:
            list: List of computed molecular properties.
        """
        if mol is None:
            return None

        features = {
            "NumHDonors": Lipinski.NumHDonors(mol),
            "NumHAcceptors": Lipinski.NumHAcceptors(mol),
            "MolWt": Descriptors.MolWt(mol),
            "MolLogP": Crippen.MolLogP(mol),
            "MolMR": Crippen.MolMR(mol),
            "TPSA": rdMolDescriptors.CalcTPSA(mol),
            "NumRotatableBonds": Lipinski.NumRotatableBonds(mol),
            "RingCount": Lipinski.RingCount(mol),
            "FractionCSP3": rdMolDescriptors.CalcFractionCSP3(mol),
        }

        return list(features.values())


def select_rdkit_features(
    rdkit_tensor: torch.Tensor, feature_map: dict, features_to_keep: list
) -> torch.Tensor:
    """
    Selects specific RDKit features from a tensor.

    Args:
        rdkit_tensor (torch.Tensor): Full RDKit feature tensor [N, total_features].
        feature_map (dict): A mapping of feature names to their tensor indices.
        features_to_keep (list): List of feature names to extract.

    Returns:
        torch.Tensor: Tensor containing only the selected features [N, len(features_to_keep)].
    """
    indices = [feature_map[feat] for feat in features_to_keep if feat in feature_map]
    if not indices:
        raise ValueError(f"No valid features selected from {features_to_keep}")

    return rdkit_tensor[:, indices]
