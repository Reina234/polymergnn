from typing import List, Tuple
import torch


class BondMatrixCreator:
    def __init__(self):
        self.n = None

    def set_n_from_smiles(self, smiles_list: List[str]):
        self.n = len(smiles_list)

    def set_n_from_rdkit_list(self, rdkit_list: List):
        self.n = len(rdkit_list)

    def set_n(self, n: int):
        self.n = n

    def create_ABC_matrix(self):
        n = self.n
        if n is None:
            raise ValueError("n is not set")
        if n <= 2:
            return torch.zeros(n, n)

        block = torch.full((n - 1, n - 1), 1 / (n - 1))
        block.fill_diagonal_(0)

        matrix = torch.zeros(n, n)
        matrix[: n - 1, : n - 1] = block
        return matrix


class HBondMatrixCreator:
    def __init__(self):
        self.feature_map = None

    def set_feature_map_from_featuriser(self, featuriser):
        self.feature_map = featuriser.feature_map

    def create_hbond_matrix(
        self, rdkit_list: List[torch.tensor]
    ) -> Tuple[torch.tensor, int]:
        rdkit_tensors = torch.stack(rdkit_list)  # Ensures proper batching

        donor_idx = self.feature_map["NumHDonors"]
        acceptor_idx = self.feature_map["NumHAcceptors"]

        num_h_donors = rdkit_tensors[:, donor_idx]
        num_h_acceptors = rdkit_tensors[:, acceptor_idx]

        n = len(rdkit_list)

        hbond_matrix = torch.zeros((n, n), dtype=torch.float32)

        for i in range(n):
            for j in range(n):
                hbond_matrix[i, j] = max(num_h_acceptors[i], num_h_donors[j]) + max(
                    num_h_donors[i], num_h_acceptors[j]
                )

        return hbond_matrix, n
