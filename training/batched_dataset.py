import pandas as pd  # type: ignore
from typing import List, Optional, Any, Tuple
from config.data_models import IFG
import torch
from rdkit.Chem.rdchem import Mol
from torch.utils.data import Dataset
from featurisers.ertl_algorithm import ErtlAlgorithm
from tools.dataset_transformer import (
    DatasetTransformer,
    NoDataTransform,
    StandardScalerTransform,
)
from tools.edge_creation import BondHBondEdgeCreator
from chemprop.data import BatchMolGraph, MolGraph
from tools.smiles_transformers import SmilesTransformer, NoSmilesTransform
from tools.smiles_to_mol import Smiles2Mol
from abc import ABC, abstractmethod
from tools.mol_to_molgraph import Mol2MolGraph, FGMembershipMol2MolGraph
from featurisers.molecule_featuriser import RDKitFeaturizer
from featurisers.chemberta_tokeniser import ChemBERTaEmbedder
from tools.utils import stack_tensors


class PolymerDataset(ABC, Dataset):
    fg_detector = ErtlAlgorithm()

    def __init__(
        self,
        data: pd.DataFrame,
        monomer_smiles_transformer: SmilesTransformer,
        solvent_smiles_transformer: SmilesTransformer,
        mol_to_molgraph: Mol2MolGraph,
        monomer_smiles_column: int = 0,
        solvent_smiles_column: Optional[int] = None,
        feature_columns: Optional[List[int]] = None,
        target_columns: Optional[List[int]] = None,
        feature_transformer: DatasetTransformer = NoDataTransform(),
        target_transformer: DatasetTransformer = StandardScalerTransform(),
        is_train: bool = False,
    ):
        self.data = data
        self.mol_to_molgraph = mol_to_molgraph
        self.monomer_smiles_lists = (
            self.data.iloc[:, monomer_smiles_column]
            .apply(lambda x: [s.strip() for s in x.split(";")])
            .tolist()
        )
        self.solvent_smiles = (
            (self.data.iloc[:, solvent_smiles_column].tolist())
            if solvent_smiles_column
            else None
        )

        self.monomer_smiles2mol = Smiles2Mol(
            smiles_transformer=monomer_smiles_transformer
        )
        self.solvent_smiles2mol = Smiles2Mol(
            smiles_transformer=solvent_smiles_transformer
        )

        self.smiles_lists = None
        self.mols = None

        self._combine_smiles()

        self.feature_transformer = feature_transformer
        self.target_transformer = target_transformer

        self.target_columns = [
            self.data.columns[i] if isinstance(i, int) else i for i in target_columns
        ]

        if feature_columns is not None:
            feature_columns = [
                self.data.columns[i] if isinstance(i, int) else i
                for i in feature_columns
            ]
        self.feature_columns = feature_columns

        if is_train:
            self.feature_transformer, self.target_transformer = (
                self.fit_test_train_transformers()
            )

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

        self.molgraphs = self._convert_mols_to_molgraph()

    def _combine_smiles(self):

        self.smiles_lists = [
            monomer_list + ([self.solvent_smiles[i]] if self.solvent_smiles else [])
            for i, monomer_list in enumerate(self.monomer_smiles_lists)
        ]

        self.mols = [
            [self.monomer_smiles2mol.convert(smi) for smi in monomer_list]
            + (
                [self.solvent_smiles2mol.convert(self.solvent_smiles[i])]
                if self.solvent_smiles
                else []
            )
            for i, monomer_list in enumerate(self.monomer_smiles_lists)
        ]

    def fit_transform(
        self,
        data: pd.DataFrame,
        transformer: DatasetTransformer,
        columns: Optional[List[int]],
    ):
        if not transformer:
            return None
        if columns:
            scaler = transformer.fit(data[columns].values)
            return scaler
        return None

    def fit_test_train_transformers(self):
        feature_scalar = self.fit_transform(
            data=self.data,
            transformer=self.feature_transformer,
            columns=self.feature_columns,
        )
        transform_scalar = self.fit_transform(
            data=self.data,
            transformer=self.target_transformer,
            columns=self.target_columns,
        )
        return feature_scalar, transform_scalar

    @abstractmethod
    def _convert_mols_to_molgraph(self) -> List[List[MolGraph]]:
        pass

    def _apply_transform(
        self,
        data: pd.DataFrame,
        transformer: DatasetTransformer,
        columns: Optional[List[int]],
    ):
        if not transformer:
            return None
        if columns:
            return transformer.transform(data[columns].values)
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

        features = self._retrieve_tensor_from_column(idx=idx, column=self.features)
        targets = self._retrieve_tensor_from_column(idx=idx, column=self.targets)

        return molgraphs, features, targets

    def _molgraphs_to_batchmolgraph(self, molgraph_list: List[MolGraph]):
        return [BatchMolGraph(molgraph) for molgraph in molgraph_list]

    @staticmethod
    def _retrieve_tensor_from_column(idx: int, column: List[List[Any]]) -> torch.tensor:
        if column is not None and column[idx] is not None:
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


class PolymerBertDataset(PolymerDataset):

    def __init__(
        self,
        data: pd.DataFrame,
        monomer_smiles_transformer: SmilesTransformer,
        solvent_smiles_transformer: SmilesTransformer = NoSmilesTransform(),
        mol_to_molgraph: Mol2MolGraph = FGMembershipMol2MolGraph(),
        monomer_smiles_column: int = 0,
        solvent_smiles_column: Optional[int] = None,
        feature_columns: Optional[List[int]] = None,
        target_columns: Optional[List[int]] = None,
        feature_transformer: DatasetTransformer = NoDataTransform(),
        target_transformer: DatasetTransformer = StandardScalerTransform(),
        is_train: bool = False,
    ):
        self.chemberta_embedder = ChemBERTaEmbedder()
        self.rdkit_featuriser = RDKitFeaturizer()
        super().__init__(
            data=data,
            monomer_smiles_transformer=monomer_smiles_transformer,
            solvent_smiles_transformer=solvent_smiles_transformer,
            mol_to_molgraph=mol_to_molgraph,
            monomer_smiles_column=monomer_smiles_column,
            solvent_smiles_column=solvent_smiles_column,
            feature_columns=feature_columns,
            target_columns=target_columns,
            feature_transformer=feature_transformer,
            target_transformer=target_transformer,
            is_train=is_train,
        )

    def _convert_mols_to_molgraph(self):
        molgraphs = [
            [self.mol_to_molgraph.convert(mol) for mol in mols] for mols in self.mols
        ]

        return molgraphs

    def __getitem__(self, idx):
        mols = self.mols[idx]
        smiles_list = self.smiles_lists[idx]
        chemberta_vals = [
            self.chemberta_embedder.embed(smiles) for smiles in smiles_list
        ]
        rdkit_list = [self.rdkit_featuriser.featurise(mol) for mol in mols]
        molgraphs, features, targets = self._get_default_items(idx=idx)
        return idx, molgraphs, chemberta_vals, rdkit_list, features, targets

    @staticmethod
    def collate_fn(batch):
        all_molgraphs = []  # For building a single BatchMolGraph
        chemberta_flat = []  # List of per-monomer ChemBERTa embeddings
        rdkit_flat = []  # List of per-monomer RDKit dicts
        system_indices = []  # e.g. [0,1,2,3,4,...]
        polymer_mapping = []  # e.g. [0,0,0,1,1,...]
        polymer_feats_list = []  # One entry per polymer system
        labels_list = []

        polymer_counter = 0
        for sys_idx, (
            _,
            molgraphs,
            chemberta_vals,
            rdkit_list,
            poly_feats,
            targets,
        ) in enumerate(batch):

            num_monomers = len(molgraphs)
            for m in range(num_monomers):
                all_molgraphs.append(molgraphs[m])
                chemberta_flat.append(chemberta_vals[m])
                rdkit_flat.append(rdkit_list[m])
                system_indices.append(sys_idx)
                polymer_mapping.append(polymer_counter)

            polymer_counter += 1
            polymer_feats_list.append(poly_feats)
            labels_list.append(targets)

        batch_mol_graph = BatchMolGraph(all_molgraphs)

        return {
            "batch_mol_graph": batch_mol_graph,
            "chemberta_tensor": stack_tensors(chemberta_flat),
            "rdkit_tensor": stack_tensors(rdkit_flat),
            "system_indices": system_indices,  # e.g. [0,1,2,3,4,...]
            "polymer_feats": stack_tensors(polymer_feats_list),
            "polymer_mapping": polymer_mapping,  # e.g. [0,0,0,1,1,...]
            "labels": stack_tensors(labels_list),
        }


class PolymerGNNDataset(PolymerDataset):

    def __init__(
        self,
        data: pd.DataFrame,
        monomer_smiles_transformer: SmilesTransformer,
        solvent_smiles_transformer: SmilesTransformer = NoSmilesTransform(),
        mol_to_molgraph: Mol2MolGraph = FGMembershipMol2MolGraph(),
        monomer_smiles_column: int = 0,
        solvent_smiles_column: Optional[int] = None,
        feature_columns: Optional[List[int]] = None,
        target_columns: Optional[List[int]] = None,
        feature_transformer: DatasetTransformer = NoDataTransform(),
        target_transformer: DatasetTransformer = StandardScalerTransform(),
        is_train: bool = False,
    ):
        self.edge_index_creator = BondHBondEdgeCreator()
        self.chemberta_embedder = ChemBERTaEmbedder()
        self.rdkit_featuriser = RDKitFeaturizer()
        super().__init__(
            data=data,
            monomer_smiles_transformer=monomer_smiles_transformer,
            solvent_smiles_transformer=solvent_smiles_transformer,
            mol_to_molgraph=mol_to_molgraph,
            monomer_smiles_column=monomer_smiles_column,
            solvent_smiles_column=solvent_smiles_column,
            feature_columns=feature_columns,
            target_columns=target_columns,
            feature_transformer=feature_transformer,
            target_transformer=target_transformer,
            is_train=is_train,
        )

    def _convert_mols_to_molgraph(self):
        molgraphs = [
            [self.mol_to_molgraph.convert(mol) for mol in mols] for mols in self.mols
        ]

        return molgraphs

    def __getitem__(self, idx):
        mols = self.mols[idx]
        smiles_list = self.smiles_lists[idx]
        chemberta_vals = [
            self.chemberta_embedder.embed(smiles).detach().cpu().numpy()
            for smiles in smiles_list
        ]
        rdkit_list = [self.rdkit_featuriser.featurise(mol) for mol in mols]
        edge_indeces, edge_attr = (
            self.edge_index_creator.create_edge_indeces_and_attributes(
                rdkit_list=rdkit_list, featuriser=self.rdkit_featuriser
            )
        )
        molgraphs, features, targets = self._get_default_items(idx=idx)
        return (
            idx,
            molgraphs,
            chemberta_vals,
            rdkit_list,
            features,
            targets,
            edge_indeces,
            edge_attr,
        )

    @staticmethod
    def collate_fn(batch):
        all_molgraphs = []
        chemberta_flat = []
        rdkit_flat = []
        system_indices = []
        polymer_mapping = []
        polymer_feats_list = []
        labels_list = []

        edge_indices_list = []
        edge_attr_list = []
        solvent_labels_list = []

        node_offset = 0  # Keeps track of node indices across polymers

        for sys_idx, (
            _,
            molgraphs,
            chemberta_vals,
            rdkit_list,
            poly_feats,
            targets,
            edge_indices,
            edge_attr,
        ) in enumerate(batch):

            num_monomers = len(molgraphs)

            # Adjust edge indices so they don't overlap across polymers
            adjusted_edge_indices = edge_indices + node_offset  # Shift edge indices
            edge_indices_list.append(adjusted_edge_indices)
            edge_attr_list.append(edge_attr)

            for m in range(num_monomers):
                all_molgraphs.append(molgraphs[m])
                chemberta_flat.append(chemberta_vals[m])
                rdkit_flat.append(rdkit_list[m])
                system_indices.append(sys_idx)
                polymer_mapping.append(sys_idx)  # Each polymer has a unique index

                # Assign solvent labels: Solvent is always the LAST molecule in each polymer
                solvent_label = 1 if m == (num_monomers - 1) else 0
                solvent_labels_list.append(solvent_label)

            node_offset += num_monomers  # Increment offset for the next polymer

            polymer_feats_list.append(poly_feats)
            labels_list.append(targets)

        batch_mol_graph = BatchMolGraph(all_molgraphs)

        return {
            "batch_mol_graph": batch_mol_graph,
            "chemberta_tensor": stack_tensors(chemberta_flat),
            "rdkit_tensor": stack_tensors(rdkit_flat),
            "system_indices": system_indices,
            "polymer_feats": stack_tensors(polymer_feats_list),
            "polymer_mapping": torch.tensor(polymer_mapping),
            "labels": stack_tensors(labels_list),
            "edge_index": torch.cat(edge_indices_list, dim=1),
            "edge_attr": torch.cat(edge_attr_list, dim=0),
            "solvent_labels": torch.tensor(solvent_labels_list).unsqueeze(
                1
            ),  # Shape (N, 1)
        }
