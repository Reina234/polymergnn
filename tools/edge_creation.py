import torch
from abc import ABC, abstractmethod
from tools.edge_matrix_gens import HBondMatrixCreator, BondMatrixCreator
from featurisers.molecule_featuriser import RDKitFeaturizer
from typing import List, Tuple


class EdgeCreator(ABC):
    @staticmethod
    def create_fully_connected_edges(n: int) -> torch.Tensor:

        row_idx = torch.arange(n).repeat(n)  # row indices (source)
        col_idx = torch.arange(n).repeat_interleave(n)  # column indices (target)
        edge_index = torch.stack([row_idx, col_idx], dim=0)  # Shape (2, N*N)
        return edge_index

    @abstractmethod
    def create_edge_indeces_and_attributes(self, rdkit_list: List[torch.tensor]):
        pass


class BondHBondEdgeCreator(EdgeCreator):

    def __init__(
        self,
        bond_matrix_creator=BondMatrixCreator(),
        hbond_matrix_creator=HBondMatrixCreator(),
    ):
        super().__init__()
        self.bond_matrix_creator = bond_matrix_creator
        self.hbond_matrix_creator = hbond_matrix_creator
        self.n = None

    def initialise_creators(self, featuriser=RDKitFeaturizer()):
        self.hbond_matrix_creator.set_feature_map_from_featuriser(featuriser)

    def create_matrices(
        self, rdkit_list: List[torch.tensor], featuriser=RDKitFeaturizer()
    ) -> Tuple[torch.tensor, torch.tensor]:
        self.hbond_matrix_creator.set_feature_map_from_featuriser(featuriser=featuriser)
        h_bond_matrix, n = self.hbond_matrix_creator.create_hbond_matrix(
            rdkit_list=rdkit_list
        )
        self.n = n
        self.bond_matrix_creator.set_n(n)
        a_b_c_matrix = self.bond_matrix_creator.create_ABC_matrix()
        return a_b_c_matrix, h_bond_matrix

    def create_ABC_matrix(self, smiles_list: List[str]):
        self.bond_matrix_creator.set_n_from_smiles(smiles_list)
        return self.bond_matrix_creator.create_ABC_matrix()

    def create_hbond_matrix(self, rdkit_list: List[torch.tensor]):
        return self.hbond_matrix_creator.create_hbond_matrix(rdkit_list)

    @staticmethod
    def _convert_matrices_to_edge_features(
        a_b_c_matrix: torch.Tensor, h_bond_matrix: torch.Tensor
    ) -> torch.Tensor:
        assert a_b_c_matrix.shape == h_bond_matrix.shape, "Matrix size mismatch"

        bond_probs = a_b_c_matrix.flatten()
        h_bond_counts = h_bond_matrix.flatten()

        # Concatenate into (N*N, 2) edge attribute tensor
        edge_attr = torch.stack([bond_probs, h_bond_counts], dim=1)
        return edge_attr

    def create_edge_indeces_and_attributes(
        self, rdkit_list: List[torch.tensor], featuriser=RDKitFeaturizer()
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        a_b_c_matrix, hbond_matrix = self.create_matrices(
            rdkit_list=rdkit_list, featuriser=featuriser
        )
        edge_attr = self._convert_matrices_to_edge_features(
            a_b_c_matrix=a_b_c_matrix, h_bond_matrix=hbond_matrix
        )
        edge_indeces = self.create_fully_connected_edges(n=self.n)
        return edge_indeces, edge_attr
