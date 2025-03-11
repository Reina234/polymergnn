import pandas as pd  # type: ignore
from typing import List, Optional, Tuple
from config.data_models import IFG
import torch
from rdkit.Chem.rdchem import Mol
from rdkit.Chem.rdFingerprintGenerator import GetMorganGenerator
from torch.utils.data import Dataset
from featurisers.ertl_algorithm import ErtlAlgorithm
from tools.edge_creation import BondHBondEdgeCreator
from chemprop.data import BatchMolGraph, MolGraph
from tools.smiles_transformers import SmilesTransformer, NoSmilesTransform
from tools.smiles_to_mol import Smiles2Mol
from abc import ABC, abstractmethod
from tools.mol_to_molgraph import Mol2MolGraph, FGMembershipMol2MolGraph
from featurisers.molecule_featuriser import RDKitFeaturizer
from featurisers.chemberta_tokeniser import ChemBERTaEmbedder
from tools.utils import stack_tensors
from tools.transform_pipeline_manager import TransformPipelineManager
import numpy as np
from torch.utils.data.dataloader import default_collate


class PolymerDataset(ABC, Dataset):
    fg_detector = ErtlAlgorithm()

    def __init__(
        self,
        data: pd.DataFrame,
        pipeline_manager: TransformPipelineManager,
        monomer_smiles_transformer: SmilesTransformer,
        solvent_smiles_transformer: SmilesTransformer,
        mol_to_molgraph: Mol2MolGraph,
        target_columns: Optional[List[int]] = None,
        feature_columns: Optional[List[int]] = None,
        monomer_smiles_column: int = 0,
        solvent_smiles_column: Optional[int] = None,
        is_train: bool = False,
    ):
        self.untransformed_data = data.copy()
        self.untransformed_data, log_indexes = self.duplicate_columns(
            df=self.untransformed_data, col_indices=target_columns
        )
        self.target_columns = target_columns + log_indexes

        self.feature_columns = feature_columns

        self.mol_to_molgraph = mol_to_molgraph
        self.monomer_smiles_lists = (
            self.untransformed_data.iloc[:, monomer_smiles_column]
            .apply(lambda x: [s.strip() for s in x.split(";")])
            .tolist()
        )
        self.solvent_smiles = (
            (self.untransformed_data.iloc[:, solvent_smiles_column].tolist())
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
        self.pipeline_manager = pipeline_manager
        self._combine_smiles()
        if is_train:
            self.pipeline_manager.fit(df=self.untransformed_data)

        self.data = self.pipeline_manager.transform(df=self.untransformed_data)

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

    @staticmethod
    def duplicate_columns(
        df: pd.DataFrame, col_indices: List[int]
    ) -> Tuple[pd.DataFrame, List[int]]:
        duplicated = df.iloc[:, col_indices]
        df_new = pd.concat([df, duplicated], axis=1)
        new_indices = list(range(len(df.columns), len(df_new.columns)))
        return df_new, new_indices

    @staticmethod
    @abstractmethod
    def collate_fn(batch):
        pass

    @abstractmethod
    def _convert_mols_to_molgraph(self) -> List[List[MolGraph]]:
        pass

    def __len__(self):
        return len(self.mols)

    @abstractmethod
    def __getitem__(self, idx):
        pass

    def _get_default_items(self, idx) -> Tuple[torch.tensor, torch.tensor]:
        features = self._retrieve_tensor_from_column(
            idx=idx, df=self.data, column_idx=self.feature_columns
        )
        targets = self._retrieve_tensor_from_column(
            idx=idx, df=self.data, column_idx=self.target_columns
        )

        return features, targets

    def _molgraphs_to_batchmolgraph(self, molgraph_list: List[MolGraph]):
        return [BatchMolGraph(molgraph) for molgraph in molgraph_list]

    @staticmethod
    def _retrieve_tensor_from_column(
        idx: int, df: pd.DataFrame, column_idx: Optional[List[int]]
    ) -> torch.Tensor:
        if df is not None and column_idx is not None and idx < len(df) and column_idx:
            values = df.iloc[idx, column_idx].tolist()
            return torch.tensor(values, dtype=torch.float32)
        return None

    def _retrieve_fgs_from_mols(self, mols: List[Mol]) -> List[List[IFG]]:
        fg_list = []
        for mol in mols:
            fg_list.append(self.fg_detector.detect(mol))
        return fg_list


class PolymerBertDataset(PolymerDataset):

    def __init__(
        self,
        data: pd.DataFrame,
        pipeline_manager: TransformPipelineManager,
        monomer_smiles_transformer: SmilesTransformer,
        solvent_smiles_transformer: SmilesTransformer = NoSmilesTransform(),
        mol_to_molgraph: Mol2MolGraph = FGMembershipMol2MolGraph(),
        monomer_smiles_column: int = 0,
        solvent_smiles_column: Optional[int] = None,
        feature_columns: Optional[List[int]] = None,
        target_columns: Optional[List[int]] = None,
        is_train: bool = False,
    ):
        self.chemberta_embedder = ChemBERTaEmbedder()
        self.rdkit_featuriser = RDKitFeaturizer()
        super().__init__(
            data=data,
            pipeline_manager=pipeline_manager,
            monomer_smiles_transformer=monomer_smiles_transformer,
            solvent_smiles_transformer=solvent_smiles_transformer,
            mol_to_molgraph=mol_to_molgraph,
            target_columns=target_columns,
            feature_columns=feature_columns,
            monomer_smiles_column=monomer_smiles_column,
            solvent_smiles_column=solvent_smiles_column,
            is_train=is_train,
        )

    def _convert_mols_to_molgraph(self):
        molgraphs = [
            [self.mol_to_molgraph.convert(mol) for mol in mols] for mols in self.mols
        ]

        return molgraphs

    def __getitem__(self, idx):
        mols = self.mols[idx]
        molgraphs = self.molgraphs[idx]
        smiles_list = self.smiles_lists[idx]
        chemberta_vals = [
            self.chemberta_embedder.embed(smiles) for smiles in smiles_list
        ]
        rdkit_list = [self.rdkit_featuriser.featurise(mol) for mol in mols]
        features, targets = self._get_default_items(idx=idx)
        return (
            idx,
            molgraphs,
            chemberta_vals,
            rdkit_list,
            features,
            targets,
        )

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
        pipeline_manager: TransformPipelineManager,
        monomer_smiles_transformer: SmilesTransformer,
        solvent_smiles_transformer: SmilesTransformer = NoSmilesTransform(),
        mol_to_molgraph: Mol2MolGraph = FGMembershipMol2MolGraph(),
        monomer_smiles_column: int = 0,
        solvent_smiles_column: Optional[int] = None,
        feature_columns: Optional[List[int]] = None,
        target_columns: Optional[List[int]] = None,
        is_train: bool = False,
    ):
        self.edge_index_creator = BondHBondEdgeCreator()
        self.chemberta_embedder = ChemBERTaEmbedder()
        self.rdkit_featuriser = RDKitFeaturizer()
        self.rdkit_featuriser = RDKitFeaturizer()
        super().__init__(
            data=data,
            pipeline_manager=pipeline_manager,
            monomer_smiles_transformer=monomer_smiles_transformer,
            solvent_smiles_transformer=solvent_smiles_transformer,
            mol_to_molgraph=mol_to_molgraph,
            target_columns=target_columns,
            feature_columns=feature_columns,
            monomer_smiles_column=monomer_smiles_column,
            solvent_smiles_column=solvent_smiles_column,
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
        edge_indeces, edge_attr = (
            self.edge_index_creator.create_edge_indeces_and_attributes(
                rdkit_list=rdkit_list, featuriser=self.rdkit_featuriser
            )
        )
        molgraphs = self.molgraphs[idx]
        features, targets = self._get_default_items(idx=idx)
        return (
            idx,
            molgraphs,
            chemberta_vals,
            rdkit_list,
            features,
            targets,
            edge_indeces,
            edge_attr,
            smiles_list,
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
        smiles_list = []

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
            smiles,
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

            smiles_list.append(smiles)
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
            "smiles_list": smiles_list,
        }


class SimpleDataset(PolymerDataset):
    def __init__(
        self,
        data: pd.DataFrame,
        n_bits: int,
        pipeline_manager: TransformPipelineManager,
        monomer_smiles_transformer: SmilesTransformer,
        solvent_smiles_transformer: SmilesTransformer = NoSmilesTransform(),
        radius: int = 2,
        monomer_smiles_column: int = 0,
        solvent_smiles_column: Optional[int] = None,
        feature_columns: Optional[List[int]] = None,
        target_columns: Optional[List[int]] = None,
        is_train: bool = False,
    ):
        self.n_bits = n_bits
        self.edge_index_creator = BondHBondEdgeCreator()
        self.chemberta_embedder = ChemBERTaEmbedder()
        self.rdkit_featuriser = RDKitFeaturizer()
        self.morgan_generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
        super().__init__(
            data=data,
            pipeline_manager=pipeline_manager,
            monomer_smiles_transformer=monomer_smiles_transformer,
            solvent_smiles_transformer=solvent_smiles_transformer,
            mol_to_molgraph=None,
            target_columns=target_columns,
            feature_columns=feature_columns,
            monomer_smiles_column=monomer_smiles_column,
            solvent_smiles_column=solvent_smiles_column,
            is_train=is_train,
        )

    def _convert_mols_to_molgraph(self):
        pass

    def smiles_to_fingerprint(self, mol: Mol, radius=2):

        if mol is None:
            fingerprint = np.zeros(self.n_bits)  # Zero-vector for invalid SMILES
        else:
            fingerprint = self.morgan_generator.GetCountFingerprintAsNumPy(mol)

        return fingerprint

    def pool_features(self, features_tensor: torch.Tensor) -> torch.Tensor:
        non_solvent_features = features_tensor[:-1]  # Exclude solvent
        solvent_feature = features_tensor[-1]  # Last molecule is always the solvent

        if len(non_solvent_features) > 0:
            pooled_feature = torch.mean(non_solvent_features, dim=0)
        else:
            pooled_feature = (
                solvent_feature.clone()
            )  # Edge case: If only solvent exists

        final_feature = torch.cat([pooled_feature, solvent_feature], dim=0)
        return final_feature  # Shape: (2 * feature_dim,)

    def __getitem__(self, idx):
        mols = self.mols[idx]
        smiles_list = self.smiles_lists[idx]
        fingerprints = [self.smiles_to_fingerprint(mol) for mol in mols]
        fingerprints_tensor = torch.tensor(np.array(fingerprints), dtype=torch.float32)
        pooled_fingerprint = self.pool_features(fingerprints_tensor)

        chemberta_vals = [
            self.chemberta_embedder.embed(smiles).detach().numpy()
            for smiles in smiles_list
        ]
        chemberta_tensor = torch.tensor(chemberta_vals, dtype=torch.float32)
        pooled_chemberta = self.pool_features(chemberta_tensor)

        rdkit_list = [self.rdkit_featuriser.featurise(mol) for mol in mols]
        rdkit_tensor = torch.tensor(np.array(rdkit_list), dtype=torch.float32)
        pooled_rdkit = self.pool_features(rdkit_tensor)

        features, targets = self._get_default_items(idx=idx)

        return {
            "idx": idx,
            "chemberta": pooled_chemberta,
            "rdkit": pooled_rdkit,
            "features": features,
            "targets": targets,
            "fingerprint": pooled_fingerprint,
        }

    @staticmethod
    def collate_fn(batch):
        return default_collate(batch)


class PolymerMorganGNNDataset(PolymerDataset):

    def __init__(
        self,
        data: pd.DataFrame,
        n_bits: int,
        pipeline_manager: TransformPipelineManager,
        monomer_smiles_transformer: SmilesTransformer,
        solvent_smiles_transformer: SmilesTransformer = NoSmilesTransform(),
        mol_to_molgraph: Mol2MolGraph = FGMembershipMol2MolGraph(),
        monomer_smiles_column: int = 0,
        solvent_smiles_column: Optional[int] = None,
        feature_columns: Optional[List[int]] = None,
        target_columns: Optional[List[int]] = None,
        radius: int = 2,
        is_train: bool = False,
    ):
        self.n_bits = n_bits
        self.edge_index_creator = BondHBondEdgeCreator()
        self.chemberta_embedder = ChemBERTaEmbedder()
        self.rdkit_featuriser = RDKitFeaturizer()
        self.morgan_generator = GetMorganGenerator(radius=radius, fpSize=n_bits)
        super().__init__(
            data=data,
            pipeline_manager=pipeline_manager,
            monomer_smiles_transformer=monomer_smiles_transformer,
            solvent_smiles_transformer=solvent_smiles_transformer,
            mol_to_molgraph=mol_to_molgraph,
            target_columns=target_columns,
            feature_columns=feature_columns,
            monomer_smiles_column=monomer_smiles_column,
            solvent_smiles_column=solvent_smiles_column,
            is_train=is_train,
        )

    def smiles_to_fingerprint(self, mol: Mol, radius=2):

        if mol is None:
            fingerprint = np.zeros(self.n_bits)  # Zero-vector for invalid SMILES
        else:
            fingerprint = self.morgan_generator.GetCountFingerprintAsNumPy(mol)

        return fingerprint

    def pool_features(self, features_tensor: torch.Tensor) -> torch.Tensor:
        non_solvent_features = features_tensor[:-1]  # Exclude solvent
        solvent_feature = features_tensor[-1]  # Last molecule is always the solvent

        if len(non_solvent_features) > 0:
            pooled_feature = torch.mean(non_solvent_features, dim=0)
        else:
            pooled_feature = (
                solvent_feature.clone()
            )  # Edge case: If only solvent exists

        final_feature = torch.cat([pooled_feature, solvent_feature], dim=0)
        return final_feature  # Shape: (2 * feature_dim,)

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
        edge_indeces, edge_attr = (
            self.edge_index_creator.create_edge_indeces_and_attributes(
                rdkit_list=rdkit_list, featuriser=self.rdkit_featuriser
            )
        )
        molgraphs = self.molgraphs[idx]
        features, targets = self._get_default_items(idx=idx)

        fingerprints = [self.smiles_to_fingerprint(mol) for mol in mols]
        fingerprints_tensor = torch.tensor(np.array(fingerprints), dtype=torch.float32)
        pooled_fingerprint = self.pool_features(fingerprints_tensor)

        return (
            idx,
            molgraphs,
            chemberta_vals,
            rdkit_list,
            features,
            targets,
            edge_indeces,
            edge_attr,
            smiles_list,
            pooled_fingerprint,
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
        smiles_list = []
        fingerprint_list = []

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
            smiles,
            pooled_fingerprint,
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

            smiles_list.append(smiles)
            polymer_feats_list.append(poly_feats)
            labels_list.append(targets)
            fingerprint_list.append(pooled_fingerprint)

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
            "smiles_list": smiles_list,
            "fingerprints_tensor": stack_tensors(fingerprint_list),
        }


class PolymerSeparatedDataset(PolymerDataset):
    def __init__(
        self,
        data: pd.DataFrame,
        pipeline_manager: TransformPipelineManager,
        monomer_smiles_transformer: SmilesTransformer,
        solvent_smiles_transformer: SmilesTransformer = NoSmilesTransform(),
        mol_to_molgraph: Mol2MolGraph = FGMembershipMol2MolGraph(),
        monomer_smiles_column: int = 0,
        solvent_smiles_column: Optional[int] = None,
        feature_columns: Optional[List[int]] = None,
        target_columns: Optional[List[int]] = None,
        is_train: bool = False,
    ):
        self.edge_index_creator = BondHBondEdgeCreator()
        self.chemberta_embedder = ChemBERTaEmbedder()
        self.rdkit_featuriser = RDKitFeaturizer()
        self.rdkit_featuriser = RDKitFeaturizer()
        super().__init__(
            data=data,
            pipeline_manager=pipeline_manager,
            monomer_smiles_transformer=monomer_smiles_transformer,
            solvent_smiles_transformer=solvent_smiles_transformer,
            mol_to_molgraph=mol_to_molgraph,
            target_columns=target_columns,
            feature_columns=feature_columns,
            monomer_smiles_column=monomer_smiles_column,
            solvent_smiles_column=solvent_smiles_column,
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
        edge_indeces, edge_attr = (
            self.edge_index_creator.create_edge_indeces_and_attributes(
                rdkit_list=rdkit_list, featuriser=self.rdkit_featuriser
            )
        )
        molgraphs = self.molgraphs[idx]
        features, targets = self._get_default_items(idx=idx)
        return (
            idx,
            molgraphs,
            chemberta_vals,
            rdkit_list,
            features,
            targets,
            edge_indeces,
            edge_attr,
            smiles_list,
        )

    @staticmethod
    def collate_fn(batch):
        monomer_molgraphs = []
        solvent_molgraphs = []

        monomer_chemberta_flat = []
        solvent_chemberta_flat = []

        monomer_rdkit_flat = []
        solvent_rdkit_flat = []

        system_indices = []
        polymer_mapping = []
        polymer_feats_list = []
        labels_list = []
        smiles_list = []

        edge_indices_list = []
        edge_attr_list = []
        solvent_labels_list = []

        node_offset = 0  # Keeps track of node indices across monomers

        for sys_idx, (
            _,
            molgraphs,
            chemberta_vals,
            rdkit_list,
            poly_feats,
            targets,
            edge_indices,
            edge_attr,
            smiles,
        ) in enumerate(batch):

            num_molecules = len(molgraphs)

            # Ensure that the solvent is the last molecule
            monomer_graphs = molgraphs[:-1]  # All but last are monomers
            solvent_graph = molgraphs[-1]  # Last molecule is solvent

            # Append monomer and solvent graphs separately
            monomer_molgraphs.extend(monomer_graphs)
            solvent_molgraphs.append(solvent_graph)

            # Split ChemBERTa & RDKit features into monomer & solvent
            monomer_chemberta_flat.extend(chemberta_vals[:-1])  # Monomers only
            solvent_chemberta_flat.append(chemberta_vals[-1])  # Solvent only

            monomer_rdkit_flat.extend(rdkit_list[:-1])  # Monomers only
            solvent_rdkit_flat.append(rdkit_list[-1])  # Solvent only

            # Adjust edge indices to prevent overlaps across different polymers
            adjusted_edge_indices = edge_indices + node_offset
            edge_indices_list.append(adjusted_edge_indices)
            edge_attr_list.append(edge_attr)

            # Update system and polymer mappings
            system_indices.extend([sys_idx] * num_molecules)
            polymer_mapping.extend([sys_idx] * num_molecules)

            # Assign solvent labels (Solvent is always last in each polymer system)
            solvent_labels_list.extend([0] * (num_molecules - 1) + [1])

            node_offset += num_molecules  # Increment offset for next polymer system

            smiles_list.append(smiles)
            polymer_feats_list.append(poly_feats)
            labels_list.append(targets)

        # Create separate batched molecular graphs
        batch_monomer_graph = BatchMolGraph(monomer_molgraphs)
        batch_solvent_graph = BatchMolGraph(solvent_molgraphs)

        # Ensure edge indices and attributes are not empty before concatenation
        if edge_indices_list:
            batch_edge_index = torch.cat(edge_indices_list, dim=1)
        else:
            batch_edge_index = torch.empty((2, 0), dtype=torch.long)

        if edge_attr_list:
            batch_edge_attr = torch.cat(edge_attr_list, dim=0)
        else:
            batch_edge_attr = torch.empty(
                (0, edge_attr_list[0].shape[1]) if edge_attr_list else (0, 0)
            )

        return {
            "batch_monomer_graph": batch_monomer_graph,
            "batch_solvent_graph": batch_solvent_graph,
            "monomer_chemberta_tensor": stack_tensors(monomer_chemberta_flat),
            "solvent_chemberta_tensor": stack_tensors(solvent_chemberta_flat),
            "monomer_rdkit_tensor": stack_tensors(monomer_rdkit_flat),
            "solvent_rdkit_tensor": stack_tensors(solvent_rdkit_flat),
            "system_indices": system_indices,
            "polymer_feats": stack_tensors(polymer_feats_list),
            "polymer_mapping": torch.tensor(polymer_mapping),
            "labels": stack_tensors(labels_list),
            "edge_index": batch_edge_index,
            "edge_attr": batch_edge_attr,
            "solvent_labels": torch.tensor(solvent_labels_list).unsqueeze(
                1
            ),  # Shape (N, 1)
            "smiles_list": smiles_list,
        }
