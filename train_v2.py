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
from tools.smiles_transformers import NoSmilesTransform
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("tests/Lipophilicity.csv")
print(df.head())
print(df.dtypes)
# df = df.drop(df.columns[[0, 1]], axis=1)  # Drop unnecessary columns

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_dataset = PolymerBertDataset(
    data=train_df,
    monomer_smiles_transformer=NoSmilesTransform(),
    mol_to_molgraph=SimpleMol2MolGraph(),
    monomer_smiles_column=2,
    target_columns=[1],
    target_transformer=MinMaxScalerTransform(),
    is_train=True,
)

target_transformer = train_dataset.target_transformer
val_dataset = PolymerBertDataset(
    data=val_df,
    monomer_smiles_transformer=NoSmilesTransform(),
    mol_to_molgraph=SimpleMol2MolGraph(),
    monomer_smiles_column=2,
    target_columns=[1],
    target_transformer=target_transformer,
)
test_dataset = PolymerBertDataset(
    data=test_df,
    monomer_smiles_transformer=NoSmilesTransform(),
    mol_to_molgraph=SimpleMol2MolGraph(),
    monomer_smiles_column=2,
    target_columns=[1],
    target_transformer=target_transformer,
)


train_loader = DataLoader(
    train_dataset,
    batch_size=32,  # Batching datapoints, but not merging molecules
    shuffle=True,
    collate_fn=train_dataset.collate_fn,  # Use custom function
)

val_loader = DataLoader(
    val_dataset, batch_size=32, shuffle=False, collate_fn=val_dataset.collate_fn
)
test_loader = DataLoader(
    test_dataset, batch_size=32, shuffle=False, collate_fn=test_dataset.collate_fn
)

# Hyperparameter search space
search_space = {
    "lr": [0.001, 0.0005],
    "epochs": [50, 80],
    "hidden_dim": [128, 256],
    "d_h": [300, 400],
    "depth": [2, 3],
    "dropout": [0.1, 0.3],
    "weight_decay": [0.0, 0.01],
    "use_rdkit": [True, False],
    "use_chembert": [True, False],
}

# Get output_dim from dataset or dataloader
output_dim = train_loader.dataset.targets.shape[1]  # Example for regression

# Use the new factory with output_dim
molecule_factory = MoleculeEmbeddingPredictionModelFactory(
    output_dim=output_dim, multi_head=True
)

# Run Tuning
tuner = HyperparameterTuner(
    model_factory=molecule_factory,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    trainer=MoleculeTrainer,
    search_space=search_space,
    max_trials=5,
    trial_name="trial/lipophilicity",
    use_tensorboard=True,
    save_results_dir="results/lipophobicity",
    additional_info={"batch_size": 32},
)
results = tuner.run()


# Show the best trial
best_trial = min(results, key=lambda x: x["test_loss"])
print("Best Hyperparameters:", best_trial["params"])
print("Best Test Metrics:", best_trial["metrics"])
