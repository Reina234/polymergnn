from training.hyperparameter_tuner import HyperparameterTuner
from training.model_factory import MoleculeEmbeddingPredictionModelFactory
import torch
from tools.dataset_transformer import (
    MinMaxScalerTransform,
    NoDataTransform,
    StandardScalerTransform,
)
import torch
import pandas as pd
from training.trainer import MoleculeTrainer
from tools.mol_to_molgraph import SimpleMol2MolGraph, FGMembershipMol2MolGraph
from sklearn.model_selection import train_test_split
from training.batched_dataset import PolymerBertDataset
from tools.smiles_transformers import NoSmilesTransform, PolymerisationSmilesTransform
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("tests/output_2_4.csv")
# df = df.drop(df.columns[[0, 1]], axis=1)  # Drop unnecessary columns

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_dataset = PolymerBertDataset(
    data=train_df,
    monomer_smiles_column=3,
    solvent_smiles_column=1,
    monomer_smiles_transformer=PolymerisationSmilesTransform(),
    mol_to_molgraph=FGMembershipMol2MolGraph(),
    target_columns=[6, 7, 8, 9, 10, 11, 12],
    target_transformer=StandardScalerTransform(),
    feature_columns=[4, 5],
    is_train=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,  # Batching datapoints, but not merging molecules
    shuffle=True,
    collate_fn=train_dataset.collate_fn,  # Use custom function
)

test = next(iter(train_loader))
print(test)
