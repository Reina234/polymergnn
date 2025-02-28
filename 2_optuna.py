import optuna

import torch
from tools.transform_pipeline_manager import TransformPipelineManager
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tools.mol_to_molgraph import FGMembershipMol2MolGraph
from sklearn.model_selection import train_test_split
from training.refactored_batched_dataset import PolymerGNNDataset
from tools.smiles_transformers import PolymerisationSmilesTransform
from training.refactored_trainer import PolymerGNNTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("tests/output_2_4_2.csv")


# Create Transform Manager
pipeline_manager = TransformPipelineManager([4, 5], [6, 7, 8, 9, 10, 11, 12])

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
    mol_to_molgraph=FGMembershipMol2MolGraph(),
    target_columns=[6, 7, 8, 9, 10, 11, 12],
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
    mol_to_molgraph=FGMembershipMol2MolGraph(),
    target_columns=[6, 7, 8, 9, 10, 11, 12],
    feature_columns=[4, 5],
    is_train=False,
)

test_dataset = PolymerGNNDataset(
    data=test_df,
    pipeline_manager=fitted_pipeline_manager,
    monomer_smiles_column=3,
    solvent_smiles_column=1,
    monomer_smiles_transformer=PolymerisationSmilesTransform(),
    mol_to_molgraph=FGMembershipMol2MolGraph(),
    target_columns=[6, 7, 8, 9, 10, 11, 12],
    feature_columns=[4, 5],
    is_train=False,
)


def objective(trial):
    hyperparams_config = {
        "batch_size": trial.suggest_categorical("batch_size", [16, 32, 64]),
        "lr": trial.suggest_loguniform("lr", 1e-5, 1e-2),
        "weight_decay": trial.suggest_loguniform("weight_decay", 1e-6, 1e-2),
        "mpnn_output_dim": trial.suggest_int("mpnn_output_dim", 64, 256, step=32),
        "mpnn_hidden_dim": trial.suggest_int("mpnn_hidden_dim", 64, 256, step=32),
        "mpnn_depth": trial.suggest_int("mpnn_depth", 1, 5),
        "mpnn_dropout": trial.suggest_uniform("mpnn_dropout", 0.0, 0.5),
        "molecule_embedding_hidden_dim": trial.suggest_int(
            "molecule_embedding_hidden_dim", 128, 512, step=64
        ),
        "embedding_dim": trial.suggest_int("embedding_dim", 64, 256, step=32),
        "use_rdkit": trial.suggest_categorical("use_rdkit", [True, False]),
        "use_chembert": trial.suggest_categorical("use_chembert", [True, False]),
        "rdkit_selection_tensor": torch.tensor([0, 0, 1, 1, 1, 1, 1]),
        "log_selection_tensor": torch.tensor([1, 1, 1, 0, 0, 0, 0]),
        "gnn_hidden_dim": trial.suggest_int("gnn_hidden_dim", 64, 256, step=32),
        "gnn_output_dim": trial.suggest_int("gnn_output_dim", 32, 128, step=32),
        "gnn_dropout": trial.suggest_uniform("gnn_dropout", 0.0, 0.5),
        "gnn_num_heads": trial.suggest_int("gnn_num_heads", 2, 8, step=2),
        "multitask_fnn_hidden_dim": trial.suggest_int(
            "multitask_fnn_hidden_dim", 32, 128, step=32
        ),
        "multitask_fnn_dropout": trial.suggest_uniform(
            "multitask_fnn_dropout", 0.0, 0.5
        ),
        "epochs": trial.suggest_int("epochs", 10, 100),
    }

    gnn_trainer = PolymerGNNTrainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        hyperparams=hyperparams_config,
    )
    gnn_trainer.train(epochs=hyperparams_config["epochs"])

    val_loss, _ = gnn_trainer.evaluate(
        gnn_trainer.val_loader, epoch=0, mode="Validation"
    )

    return val_loss  # Optuna minimizes this loss


study = optuna.create_study(
    pruner=optuna.pruners.HyperbandPruner(),
    direction="minimize",
    sampler=optuna.samplers.TPESampler(),
)
study.optimize(objective, n_trials=50)

# Print best hyperparameters
best_hyperparams = study.best_params
print("Best hyperparameters:", best_hyperparams)


print("Decoded rdkit_selection_tensor:", best_hyperparams["rdkit_selection_tensor"])
print("Decoded log_selection_tensor:", best_hyperparams["log_selection_tensor"])
