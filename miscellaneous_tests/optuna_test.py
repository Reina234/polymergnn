import optuna

import torch
from tools.transform_pipeline_manager import TransformPipelineManager
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tools.mol_to_molgraph import FGMembershipMol2MolGraph
from sklearn.model_selection import train_test_split
from training.refactored_batched_dataset import PolymerGNNDataset
from tools.smiles_transformers import PolymerisationSmilesTransform
from torch.utils.data import DataLoader
from training.refactored_trainer import PolymerGNNTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("data/output_2_4_2.csv")


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

hyperparams = {
    "batch_size": 32,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "mpnn_output_dim": 128,
    "mpnn_hidden_dim": 128,
    "mpnn_depth": 2,
    "mpnn_dropout": 0.2,
    "rdkit_selection_tensor": torch.tensor([1, 0, 1, 0, 1, 0, 1]),
    "log_selection_tensor": torch.tensor(
        [0, 1, 0, 0, 0, 1, 1]
    ),  # Only log-transform 2nd label
    "molecule_embedding_hidden_dim": 256,
    "embedding_dim": 128,
    "use_rdkit": True,
    "use_chembert": True,
    "gnn_hidden_dim": 128,
    "gnn_output_dim": 64,
    "gnn_dropout": 0.1,
    "gnn_num_heads": 4,
    "multitask_fnn_hidden_dim": 64,
    "multitask_fnn_dropout": 0.1,
}


# Load a batch from DataLoader
train_loader = DataLoader(
    train_dataset, batch_size=32, collate_fn=train_dataset.collate_fn
)
test_batch = next(iter(train_loader))

# Forward pass test
# print("\nTesting Forward Pass...")
# predictions, labels = trainer.forward_pass(test_batch)
# print(f"Predictions shape: {predictions.shape}")
# print(f"Labels shape: {labels.shape}")

# Loss computation test
# print("\nComputing Loss...")
# loss = trainer.compute_loss(predictions, labels)
# print(f"Loss: {loss.item()}")

# Check if `filter_target_labels` is correctly selecting log-transformed labels
# filtered_labels = trainer.log_transform_helper.filter_target_labels(
#    test_batch["labels"]
# )
# print("\nFiltered Labels (Should match selected log/non-log targets):")
# print(filtered_labels)

# Check inverse transformation
# inv_labels, inv_preds = trainer.inverse_transform(
#    labels.detach().numpy(), predictions.detach().numpy()
# )

# trainer.run()


def encode_tensor(tensor):
    """Convert a binary tensor to an integer."""
    binary_str = "".join(map(str, tensor.tolist()))  # Convert tensor to binary string
    return int(binary_str, 2)  # Convert binary string to integer


def decode_tensor(value, length):
    """Convert an integer back to a binary tensor of given length."""
    binary_str = bin(value)[2:].zfill(length)  # Convert to binary and pad
    return torch.tensor([int(b) for b in binary_str])  # Convert to tensor


def objective(trial):
    """Objective function for Optuna hyperparameter tuning."""

    tensor_length = 7  # Length of the binary selection tensors

    # Define the hyperparameter search space
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
        # Encode selection tensors as integers
        "rdkit_selection_encoded": trial.suggest_int(
            "rdkit_selection_encoded", 0, 2**tensor_length - 1
        ),
        "log_selection_encoded": trial.suggest_int(
            "log_selection_encoded", 0, 2**tensor_length - 1
        ),
        "epochs": trial.suggest_int("epochs", 10, 100),
    }

    # Decode selection tensors back into binary tensors
    hyperparams_config["rdkit_selection_tensor"] = decode_tensor(
        hyperparams_config["rdkit_selection_encoded"], tensor_length
    )
    hyperparams_config["log_selection_tensor"] = decode_tensor(
        hyperparams_config["log_selection_encoded"], tensor_length
    )

    # Initialize trainer with trial hyperparameters
    gnn_trainer = PolymerGNNTrainer(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        hyperparams=hyperparams_config,
    )

    # Train the model
    gnn_trainer.train(epochs=hyperparams_config["epochs"])

    # Evaluate on validation set
    val_loss, _ = gnn_trainer.evaluate(
        gnn_trainer.val_loader, epoch=0, mode="Validation"
    )

    return val_loss  # Optuna minimizes this loss


study = optuna.create_study(
    pruner=optuna.pruners.HyperbandPruner(), direction="minimize"
)
study.optimize(objective, n_trials=50)  # Run 50 trials

# Print best hyperparameters
best_hyperparams = study.best_params
print("Best hyperparameters:", best_hyperparams)

# Decode the best found selection tensors
best_hyperparams["rdkit_selection_tensor"] = decode_tensor(
    best_hyperparams["rdkit_selection_encoded"], 7
)
best_hyperparams["log_selection_tensor"] = decode_tensor(
    best_hyperparams["log_selection_encoded"], 7
)

print("Decoded rdkit_selection_tensor:", best_hyperparams["rdkit_selection_tensor"])
print("Decoded log_selection_tensor:", best_hyperparams["log_selection_tensor"])


# Trial 0 finished with value: 0.41995464265346527 and parameters: {'batch_size': 32, 'lr': 0.0008200682951258308, 'weight_decay': 0.007006149865358475, 'mpnn_output_dim': 256, 'mpnn_hidden_dim': 96, 'mpnn_depth': 1, 'mpnn_dropout': 0.3273969103512452, 'molecule_embedding_hidden_dim': 192, 'embedding_dim': 64, 'use_rdkit': True, 'use_chembert': False, 'gnn_hidden_dim': 64, 'gnn_output_dim': 32, 'gnn_dropout': 0.2873908862729379, 'gnn_num_heads': 2, 'multitask_fnn_hidden_dim': 96, 'multitask_fnn_dropout': 0.373362595357648, 'rdkit_selection_encoded': 65, 'log_selection_encoded': 75, 'epochs': 23}. Best is trial 0 with value: 0.41995464265346527.
