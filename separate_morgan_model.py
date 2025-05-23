# import optuna

import torch
from tools.transform_pipeline_manager import TransformPipelineManager
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tools.mol_to_molgraph import FGMembershipMol2MolGraph
from sklearn.model_selection import train_test_split
from training.refactored_batched_dataset import PolymerMorganSeparatedDataset
from tools.smiles_transformers import PolymerisationSmilesTransform
from torch.utils.data import DataLoader
from training.trainer_3_features import SeparateMorganPolymerGNNTrainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_columns = [7, 8, 9, 10, 11, 12, 13]
feature_columns = [0, 5, 6]
monomer_smiles_column = 4
solvent_smiles_column = 2

df = pd.read_csv("PLACEHOLDER")


n_bits = 2048


pipeline_manager = TransformPipelineManager(
    feature_indexes=feature_columns, target_indexes=target_columns
)


pipeline_manager.set_feature_pipeline(StandardScaler())
pipeline_manager.set_target_pipeline(StandardScaler())


train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


train_dataset = PolymerMorganSeparatedDataset(
    data=train_df,
    n_bits=n_bits,
    pipeline_manager=pipeline_manager,
    monomer_smiles_column=monomer_smiles_column,
    solvent_smiles_column=solvent_smiles_column,
    monomer_smiles_transformer=PolymerisationSmilesTransform(),
    mol_to_molgraph=FGMembershipMol2MolGraph(),
    target_columns=target_columns,
    feature_columns=feature_columns,
    is_train=True,
)
fitted_pipeline_manager = train_dataset.pipeline_manager

val_dataset = PolymerMorganSeparatedDataset(
    data=val_df,
    n_bits=n_bits,
    pipeline_manager=fitted_pipeline_manager,
    monomer_smiles_column=monomer_smiles_column,
    solvent_smiles_column=solvent_smiles_column,
    monomer_smiles_transformer=PolymerisationSmilesTransform(),
    mol_to_molgraph=FGMembershipMol2MolGraph(),
    target_columns=target_columns,
    feature_columns=feature_columns,
    is_train=False,
)

test_dataset = PolymerMorganSeparatedDataset(
    data=test_df,
    n_bits=n_bits,
    pipeline_manager=fitted_pipeline_manager,
    monomer_smiles_column=monomer_smiles_column,
    solvent_smiles_column=solvent_smiles_column,
    monomer_smiles_transformer=PolymerisationSmilesTransform(),
    mol_to_molgraph=FGMembershipMol2MolGraph(),
    target_columns=target_columns,
    feature_columns=feature_columns,
    is_train=False,
)


hyperparams = {
    "batch_size": 32,
    "lr": 0.001,
    "weight_decay": 0,
    "mpnn_output_dim": 128,
    "mpnn_hidden_dim": 256,
    "mpnn_depth": 3,
    "mpnn_dropout": 0.1,
    "rdkit_selection_tensor": torch.tensor([1, 1, 1, 1, 1, 1, 1]),
    "log_selection_tensor": torch.tensor(
        [1, 1, 1, 0, 0, 1, 1]
    ),  # Only log-transform 2nd label
    "molecule_embedding_hidden_dim": 256,
    "embedding_dim": 256,
    "gnn_hidden_dim": 256,
    "gnn_output_dim": 256,
    "gnn_dropout": 0.1,
    "gnn_num_heads": 2,
    "multitask_fnn_hidden_dim": 128,
    "multitask_fnn_shared_layer_dim": 256,
    "multitask_fnn_dropout": 0.1,
    "epochs": 55,
    "weights": torch.tensor([1.0, 2.0, 8.0, 1.0, 1.0, 1.0, 1.0]),
}


gnn_trainer = SeparateMorganPolymerGNNTrainer(
    n_bits=n_bits,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    hyperparams=hyperparams,
    log_dir="logs/full_gnn_trials/with_density/",
    track_learning_curve=True,
)
train_loader = DataLoader(
    train_dataset, batch_size=32, collate_fn=train_dataset.collate_fn
)


gnn_trainer.run()


torch.save(
    gnn_trainer.model.state_dict(), "separate_march_11_full_morgan_gat_min_max.pth"
)
