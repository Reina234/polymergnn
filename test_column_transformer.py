import torch
from tools.transform_pipeline_manager import TransformPipelineManager
from sklearn.preprocessing import StandardScaler
import torch
import pandas as pd
from tools.mol_to_molgraph import FGMembershipMol2MolGraph
from sklearn.model_selection import train_test_split
from training.refactored_batched_dataset import PolymerGNNDataset
from tools.smiles_transformers import PolymerisationSmilesTransform
from torch.utils.data import DataLoader
from training.refactored_trainer import PolymerGNNTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("tests/output_2_4.csv")


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

trainer = PolymerGNNTrainer(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    hyperparams=hyperparams,
)

print(f"Trainer Model: {trainer.model}")
print(f"Using device: {trainer.device}")


# Load a batch from DataLoader
train_loader = DataLoader(
    train_dataset, batch_size=32, collate_fn=train_dataset.collate_fn
)
test_batch = next(iter(train_loader))

# Forward pass test
print("\nTesting Forward Pass...")
predictions, labels = trainer.forward_pass(test_batch)
print(f"Predictions shape: {predictions.shape}")
print(f"Labels shape: {labels.shape}")

# Loss computation test
print("\nComputing Loss...")
loss = trainer.compute_loss(predictions, labels)
print(f"Loss: {loss.item()}")

# Check if `filter_target_labels` is correctly selecting log-transformed labels
filtered_labels = trainer.log_transform_helper.filter_target_labels(
    test_batch["labels"]
)
print("\nFiltered Labels (Should match selected log/non-log targets):")
print(filtered_labels)

# Check inverse transformation
inv_labels, inv_preds = trainer.inverse_transform(
    labels.detach().numpy(), predictions.detach().numpy()
)
print("\nInverse Transformed Labels and Predictions:")
print(f"Inverse Labels: {inv_labels}")
print(f"Inverse Predictions: {inv_preds}")

print("\n✅ Trainer setup successful. No shape mismatches detected.")


trainer.run()
