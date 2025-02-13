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


# pylint: disable=no-member
class MolecularDataset(Dataset):
    """
    A PyTorch Dataset to generate data for the ChemProp MPNN.
    Expected default column order in the input DataFrame:
      [SMILES, additional features..., targets]

    - smiles_column: the column index holding SMILES strings (default: 0).
      SMILES strings in that column should be comma-separated.
    - feature_columns: a list of column indices for additional features.
      If empty, no additional features are provided.
    - target_columns: a list of column indices for targets.
      If not provided, defaults to the last column.
    - mol_processor: a MolProcessor instance to convert SMILES into Mol objects.
      This allows for modular swapping of preprocessing strategies.
    - fg_detector: a FunctionalGroupDetector instance; defaults to the dummy version.
    """

    featuriser = SimpleMoleculeMolGraphFeaturizer()
    fg_detector = ErtlAlgorithm()

    def __init__(
        self,
        data: pd.DataFrame,
        smiles_column: int = 0,
        feature_columns: Optional[List[int]] = None,
        target_columns: Optional[List[int]] = None,
        preprocessor: Preprocessor = NoPreprocessing,
    ):

        self.data = data
        self.mol_processor = MoleculeProcessor(preprocessor=preprocessor)
        # Use the SMILES column by index; assume each cell is a comma-separated string of SMILES.
        self.smiles_lists = (
            self.data.iloc[:, smiles_column]
            .apply(lambda x: [s.strip() for s in x.split(",")])
            .tolist()
        )

        # If target_columns is not provided, use the last column.
        if target_columns is None:
            target_columns = [self.data.columns[-1]]
        else:
            # If target_columns are provided as indices, convert to column names.
            target_columns = [
                self.data.columns[i] if isinstance(i, int) else i
                for i in target_columns
            ]
        self.target_columns = target_columns

        # If feature_columns is not provided, use all columns between the SMILES column and the first target column.
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

        # Extract features and targets from the DataFrame.
        self.features = (
            self.data[self.feature_columns].values if self.feature_columns else None
        )
        self.targets = self.data[self.target_columns].values

        # Set the FG detector and MolProcessor.

        # Convert SMILES strings to Mol objects.
        # self.mols will be a list (for each row) of lists (for each SMILES in that row).
        self.mols = [
            [self.mol_processor.process(smi) for smi in smiles]
            for smiles in self.smiles_lists
        ]

        # Create datapoints: For each row, for each Mol, create a MoleculeDatapoint.
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
