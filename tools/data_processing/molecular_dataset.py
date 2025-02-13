import pandas as pd  # type: ignore
from typing import List, Optional

import torch
from torch.utils.data import Dataset

from chemprop.data import BatchMolGraph
from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer

from tools.data_processing.ertyl_algorithm import ErtlAlgorithm
from tools.data_processing.preprocessor import (
    MoleculeProcessor,
    NoPreprocessing,
    Preprocessor,
)

from torch.utils.data import Dataset
from typing import List, Optional
import pandas as pd
import numpy as np
from tools.data_processing.dataset_preprocessor import (
    PreprocessorStrategy,
    NoDataPreprocessing,
    StandardScalerPreprocessor,
)


class MolecularDataset(Dataset):
    """
    A PyTorch Dataset for the ChemProp MPNN.
    Supports preprocessing for features and targets via preprocessing strategies.
    """

    featuriser = SimpleMoleculeMolGraphFeaturizer()
    fg_detector = ErtlAlgorithm()

    def __init__(
        self,
        data: pd.DataFrame,
        smiles_column: int = 0,
        feature_columns: Optional[List[int]] = None,
        target_columns: Optional[List[int]] = None,
        mol_preprocessor: Preprocessor = NoPreprocessing(),
        feature_preprocessor: PreprocessorStrategy = NoDataPreprocessing(),
        target_preprocessor: PreprocessorStrategy = StandardScalerPreprocessor(),
    ):
        """
        Args:
            data: Input pandas DataFrame.
            smiles_column: Index of the SMILES column.
            feature_columns: List of indices for feature columns.
            target_columns: List of indices for target columns.
            feature_preprocessor: Preprocessing strategy for features.
            target_preprocessor: Preprocessing strategy for targets.
        """
        self.data = data
        self.mol_processor = MoleculeProcessor(
            preprocessor=mol_preprocessor
        )  # Assume molecules don't need extra pre-processing
        self.feature_preprocessor = feature_preprocessor
        self.target_preprocessor = target_preprocessor

        # ✅ Extract SMILES
        self.smiles_lists = (
            self.data.iloc[:, smiles_column]
            .apply(lambda x: [s.strip() for s in x.split(",")])
            .tolist()
        )

        # ✅ Target Column Selection
        if target_columns is None:
            target_columns = [self.data.columns[-1]]
        else:
            target_columns = [
                self.data.columns[i] if isinstance(i, int) else i
                for i in target_columns
            ]
        self.target_columns = target_columns

        # ✅ Feature Column Selection
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

        # ✅ Apply Feature Preprocessing
        self.features = (
            self.feature_preprocessor.fit_transform(
                self.data[self.feature_columns].values
            )
            if self.feature_columns
            else None
        )

        # ✅ Apply Target Preprocessing
        self.targets = self.target_preprocessor.fit_transform(
            self.data[self.target_columns].values
        )

        # ✅ Convert SMILES to Mol objects
        self.mols = [
            [self.mol_processor.process(smi) for smi in smiles]
            for smiles in self.smiles_lists
        ]

        # ✅ Convert to Graph Representation
        self.molgraphs = [[self.featuriser(mol) for mol in mols] for mols in self.mols]

    def __len__(self):
        return len(self.mols)

    def __getitem__(self, idx):
        mols = self.mols[idx]
        molgraphs = self.molgraphs[idx]

        # Detect FGs, returning a list of IFG objects
        fg_list = []
        for mol in mols:
            fg_list.append(self.fg_detector.detect(mol))

        batch_molgraph = BatchMolGraph(molgraphs)
        if self.features is not None:
            features = torch.tensor(self.features[idx], dtype=torch.float32)
        else:
            features = None
        targets = torch.tensor(self.targets[idx], dtype=torch.float32)

        return batch_molgraph, mols, fg_list, features, targets


def custom_collate_fn(batch):
    """
    Custom collate function to handle BatchMolGraph objects properly.
    """
    batch_molgraphs = [item[0] for item in batch]  # Extract molecular graphs
    mols = [item[1] for item in batch]  # Extract molecule representations
    batch_fg_lists = [item[2] for item in batch]  # Functional group lists
    others = [item[3:] for item in batch]  # Remaining elements (e.g., targets)

    # Convert molecular graphs into a batch object
    batch_molgraphs = BatchMolGraph(batch_molgraphs)

    # Convert targets to tensors
    batch_targets = torch.tensor([item[-1] for item in others], dtype=torch.float32)

    return batch_molgraphs, mols, batch_fg_lists, None, batch_targets
