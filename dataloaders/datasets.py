import pandas as pd  # type: ignore
from typing import List, Optional, Any, Tuple
from config.data_models import IFG
import torch
from rdkit.Chem.rdchem import Mol
from torch.utils.data import Dataset
from tools.ertl_algorithm import ErtlAlgorithm
from tools.dataset_transformer import (
    DatasetTransformer,
    NoDataTransform,
    StandardScalerTransform,
)
from chemprop.data import BatchMolGraph, MolGraph
from tools.smiles_transformers import SmilesTransformer, NoSmilesTransform
from tools.smiles_to_mol import Smiles2Mol
from abc import ABC, abstractmethod
from dataloaders.mol_to_molgraph import FGMembershipMol2MolGraph


class MolecularDataset(ABC, Dataset):
    fg_detector = ErtlAlgorithm()

    def __init__(
        self,
        data: pd.DataFrame,
        smiles_transformer: SmilesTransformer,
        smiles_column: int = 0,
        feature_columns: Optional[List[int]] = None,
        target_columns: Optional[List[int]] = None,
        feature_transformer: DatasetTransformer = NoDataTransform(),
        target_transformer: DatasetTransformer = StandardScalerTransform(),
    ):
        self.data = data
        self.smiles2mol = Smiles2Mol(smiles_transformer=smiles_transformer)
        self.feature_transformer = feature_transformer
        self.target_transformer = target_transformer

        # extract SMILES
        self.smiles_lists = (
            self.data.iloc[:, smiles_column]
            .apply(lambda x: [s.strip() for s in x.split(",")])
            .tolist()
        )

        # target Column Selection
        if target_columns is None:
            target_columns = [self.data.columns[-1]]
        else:
            target_columns = [
                self.data.columns[i] if isinstance(i, int) else i
                for i in target_columns
            ]
        self.target_columns = target_columns

        if feature_columns is None:
            all_cols = list(self.data.columns)
            first_target_idx = self.data.columns.get_loc(self.target_columns[0])
            if first_target_idx > smiles_column + 1:
                feature_columns = all_cols[smiles_column + 1 : first_target_idx]
            else:
                feature_columns = []
        else:
            feature_columns = [
                self.data.columns[i] if isinstance(i, int) else i
                for i in feature_columns
            ]
        self.feature_columns = feature_columns

        self.features = self._apply_transform(
            data=self.data,
            transformer=self.feature_transformer,
            columns=self.feature_columns,
        )

        self.targets = self._apply_transform(
            data=self.data,
            transformer=self.target_transformer,
            columns=self.target_columns,
        )
        self.mols = [
            [self.smiles2mol.convert(smi) for smi in smiles]
            for smiles in self.smiles_lists
        ]

        self.molgraphs = self._convert_mols_to_molgraph()

    @abstractmethod
    def _convert_mols_to_molgraph(self) -> List[List[MolGraph]]:
        pass

    def _apply_transform(
        self,
        data: pd.DataFrame,
        transformer: DatasetTransformer,
        columns: Optional[List[int]],
    ):
        if columns:
            return transformer.fit_transform(data[columns].values)
        return None

    def __len__(self):
        return len(self.mols)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def _get_default_items(
        self, idx
    ) -> Tuple[List[Mol], List[MolGraph], torch.tensor, torch.tensor]:
        molgraphs = self.molgraphs[idx]
        batch_molgraph = BatchMolGraph(molgraphs)
        features = self._retrieve_tensor_from_column(idx=idx, column=self.features)
        targets = self._retrieve_tensor_from_column(idx=idx, column=self.targets)

        return batch_molgraph, features, targets

    def _molgraphs_to_batchmolgraph(self, molgraph_list: List[MolGraph]):
        return [BatchMolGraph(molgraph) for molgraph in molgraph_list]

    @staticmethod
    def _retrieve_tensor_from_column(idx: int, column: List[List[Any]]) -> torch.tensor:
        if column is not None:
            return torch.tensor(column[idx], dtype=torch.float32)

    def _retrieve_fgs_from_mols(self, mols: List[Mol]) -> List[List[IFG]]:
        fg_list = []
        for mol in mols:
            fg_list.append(self.fg_detector.detect(mol))
        return fg_list

    @staticmethod
    @abstractmethod
    def collate_fn(batch):
        pass


class FGFeaturisedMolecularDataset(MolecularDataset):

    def __init__(
        self,
        data: pd.DataFrame,
        smiles_transformer: SmilesTransformer = NoSmilesTransform(),
        smiles_column: int = 0,
        feature_columns: Optional[List[int]] = None,
        target_columns: Optional[List[int]] = None,
        feature_transformer: DatasetTransformer = NoDataTransform(),
        target_transformer: DatasetTransformer = StandardScalerTransform(),
    ):
        self.mol_to_molgraph = FGMembershipMol2MolGraph()
        super().__init__(
            data=data,
            smiles_transformer=smiles_transformer,
            smiles_column=smiles_column,
            feature_columns=feature_columns,
            target_columns=target_columns,
            feature_transformer=feature_transformer,
            target_transformer=target_transformer,
        )

    def _convert_mols_to_molgraph(self):
        molgraphs = [
            [self.mol_to_molgraph.convert(mol) for mol in mols] for mols in self.mols
        ]

        return molgraphs

    def __getitem__(self, idx):
        return self._get_default_items(idx=idx)

    @staticmethod
    def collate_fn(batch):
        molgraph_batches, features, targets = zip(*batch)

        return {
            "molgraph_batches": list(molgraph_batches),
            "features": (
                torch.stack(features)
                if isinstance(features[0], torch.Tensor)
                else list(features)
            ),
            "targets": (
                torch.stack(targets)
                if isinstance(targets[0], torch.Tensor)
                else list(targets)
            ),
        }
