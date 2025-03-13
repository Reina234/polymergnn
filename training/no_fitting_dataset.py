import pandas as pd  # type: ignore
from typing import List, Optional, Tuple
from config.data_models import IFG
import torch
from tools.edge_creation import BondHBondEdgeCreator
from tools.smiles_transformers import SmilesTransformer, NoSmilesTransform
from tools.mol_to_molgraph import Mol2MolGraph, FGMembershipMol2MolGraph
from featurisers.molecule_featuriser import RDKitFeaturizer
from featurisers.chemberta_tokeniser import ChemBERTaEmbedder
from tools.utils import stack_tensors
from tools.transform_pipeline_manager import TransformPipelineManager
from chemprop.data import BatchMolGraph, MolGraph
from rdkit.Chem.rdchem import Mol
from torch.utils.data import Dataset
from featurisers.ertl_algorithm import ErtlAlgorithm
from tools.smiles_to_mol import Smiles2Mol
from abc import ABC, abstractmethod


class NoFitPolymerDataset(ABC, Dataset):
    fg_detector = ErtlAlgorithm()

    def __init__(
        self,
        data: pd.DataFrame,
        monomer_smiles_transformer: SmilesTransformer,
        solvent_smiles_transformer: SmilesTransformer,
        mol_to_molgraph: Mol2MolGraph,
        feature_columns: Optional[List[int]] = None,
        monomer_smiles_column: int = 0,
        solvent_smiles_column: Optional[int] = None,
    ):
        self.untransformed_data = data.copy()

        self.feature_columns = feature_columns

        self.mol_to_molgraph = mol_to_molgraph
        self.monomer_smiles_lists = (
            self.untransformed_data.iloc[:, monomer_smiles_column]
            .apply(lambda x: [s.strip() for s in x.split(";")])
            .tolist()
        )
        self.solvent_smiles = self.untransformed_data.iloc[
            :, solvent_smiles_column
        ].tolist()
        print(self.solvent_smiles)

        self.monomer_smiles2mol = Smiles2Mol(
            smiles_transformer=monomer_smiles_transformer
        )
        self.solvent_smiles2mol = Smiles2Mol(
            smiles_transformer=solvent_smiles_transformer
        )

        self.smiles_lists = None
        self.mols = None
        self._combine_smiles()
        self.data = self.untransformed_data
        self.molgraphs = self._convert_mols_to_molgraph()

    def _combine_smiles(self):
        # print(self.smiles_lists)
        self.smiles_lists = [
            monomer_list + ([self.solvent_smiles[i]] if self.solvent_smiles else [])
            for i, monomer_list in enumerate(self.monomer_smiles_lists)
        ]
        # print(self.smiles_lists)
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

    def _get_default_items(self, idx) -> Tuple[torch.tensor]:

        features = self._retrieve_tensor_from_column(
            idx=idx, df=self.data, column_idx=self.feature_columns
        )

        return features

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


class NoFitPolymerSeparatedDataset(NoFitPolymerDataset):
    def __init__(
        self,
        data: pd.DataFrame,
        monomer_smiles_transformer: SmilesTransformer,
        solvent_smiles_transformer: SmilesTransformer = NoSmilesTransform(),
        mol_to_molgraph: Mol2MolGraph = FGMembershipMol2MolGraph(),
        monomer_smiles_column: int = 0,
        solvent_smiles_column: Optional[int] = None,
        feature_columns: Optional[List[int]] = None,
    ):
        self.edge_index_creator = BondHBondEdgeCreator()
        self.chemberta_embedder = ChemBERTaEmbedder()
        self.rdkit_featuriser = RDKitFeaturizer()
        self.rdkit_featuriser = RDKitFeaturizer()
        super().__init__(
            data=data,
            monomer_smiles_transformer=monomer_smiles_transformer,
            solvent_smiles_transformer=solvent_smiles_transformer,
            mol_to_molgraph=mol_to_molgraph,
            feature_columns=feature_columns,
            monomer_smiles_column=monomer_smiles_column,
            solvent_smiles_column=solvent_smiles_column,
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
        features = self._get_default_items(idx=idx)

        return (
            idx,
            molgraphs,
            chemberta_vals,
            rdkit_list,
            features,
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
            "edge_index": batch_edge_index,
            "edge_attr": batch_edge_attr,
            "solvent_labels": torch.tensor(solvent_labels_list).unsqueeze(
                1
            ),  # Shape (N, 1)
            "smiles_list": smiles_list,
        }
