# import optuna

import torch
from tools.transform_pipeline_manager import TransformPipelineManager
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tools.mol_to_molgraph import (
    FGMembershipMol2MolGraph,
    SimpleMol2MolGraph,
)
from sklearn.model_selection import train_test_split
from training.refactored_batched_dataset import PolymerGNNDataset
from tools.smiles_transformers import PolymerisationSmilesTransform
from torch.utils.data import DataLoader
from training.refactored_trainer import PolymerGNNTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("data/output_2_4_2.csv")


# Create Transform Manager
pipeline_manager = TransformPipelineManager([4, 5], [6, 7, 8, 9, 10, 11])

# Apply same transformation to all features & targets
pipeline_manager.set_feature_pipeline(StandardScaler())
pipeline_manager.set_target_pipeline(StandardScaler())


train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


train_dataset = PolymerGNNDataset(
    data=train_df,
    pipeline_manager=pipeline_manager,
    monomer_smiles_column=3,
    solvent_smiles_column=1,
    monomer_smiles_transformer=PolymerisationSmilesTransform(),
    mol_to_molgraph=SimpleMol2MolGraph(),
    target_columns=[6, 7, 8, 9, 10, 11],
    feature_columns=[4, 5],
    is_train=True,
)
fitted_pipeline_manager = train_dataset.pipeline_manager

val_dataset = PolymerGNNDataset(
    data=val_df,
    pipeline_manager=fitted_pipeline_manager,
    monomer_smiles_column=3,
    solvent_smiles_column=1,
    monomer_smiles_transformer=PolymerisationSmilesTransform(),
    mol_to_molgraph=SimpleMol2MolGraph(),
    target_columns=[6, 7, 8, 9, 10, 11],
    feature_columns=[4, 5],
    is_train=False,
)

test_dataset = PolymerGNNDataset(
    data=test_df,
    pipeline_manager=fitted_pipeline_manager,
    monomer_smiles_column=3,
    solvent_smiles_column=1,
    monomer_smiles_transformer=PolymerisationSmilesTransform(),
    mol_to_molgraph=SimpleMol2MolGraph(),
    target_columns=[6, 7, 8, 9, 10, 11],
    feature_columns=[4, 5],
    is_train=False,
)

hyperparams = {
    "batch_size": 32,
    "lr": 0.0025,
    "weight_decay": 1e-6,
    "log_diffusion_factor": 5.0,  # Tune scaling
    "log_rg_factor": 3.0,
    "mpnn_output_dim": 128,
    "mpnn_hidden_dim": 96,
    "mpnn_depth": 2,
    "mpnn_dropout": 0.327396910351,
    "rdkit_selection_tensor": torch.tensor([1, 1, 1, 1, 1, 1, 1]),
    "log_selection_tensor": torch.tensor(
        [1, 1, 1, 0, 0, 1]
    ),  # Only log-transform 2nd label
    "molecule_embedding_hidden_dim": 192,
    "embedding_dim": 100,
    "use_rdkit": True,
    "use_chembert": False,
    "gnn_hidden_dim": 128,
    "gnn_output_dim": 64,
    "gnn_dropout": 0.1,
    "gnn_num_heads": 4,
    "multitask_fnn_hidden_dim": 96,
    "multitask_fnn_shared_layer_dim": 128,
    "multitask_fnn_dropout": 0.1,
    "epochs": 20,
    "weights": torch.tensor([1.0, 1.0, 8.0, 1.0, 1.0, 1.0]),
}

gnn_trainer = PolymerGNNTrainer(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    hyperparams=hyperparams,
    log_dir="logs/full_gnn_trials/",
    track_learning_curve=True,
)
train_loader = DataLoader(
    train_dataset, batch_size=32, collate_fn=train_dataset.collate_fn
)


gnn_trainer.run()


# torch.save(gnn_trainer.model.state_dict(), "polymergnn_full_mod_gat.pth")
