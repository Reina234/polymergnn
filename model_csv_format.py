import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tools.transform_pipeline_manager import TransformPipelineManager
from tools.mol_to_molgraph import FGMembershipMol2MolGraph
from tools.smiles_transformers import PolymerisationSmilesTransform
from training.refactored_batched_dataset import PolymerSeparatedDataset
from training.trainer_3_features import SeparatedGNNTrainer

# Define file paths
model_path = "march_11_model.pth"
input_csv = "solvent_polymer_interactions.csv"
output_csv = "solvent_polymer_predictions.csv"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CSV
df = pd.read_csv(input_csv)
df[df.shape[1]] = 30
df[df.shape[1]] = 298

output_file = "additional_data.csv"
df.to_csv(output_file, index=False)

raise ValueError
# Define dataset column indexes
target_columns = []
feature_columns = [1, 5, 6]
monomer_smiles_column = 2
solvent_smiles_column = 0

# Initialize pipeline manager (no scaling)
pipeline_manager = TransformPipelineManager(
    feature_indexes=feature_columns, target_indexes=target_columns
)
pipeline_manager.set_feature_pipeline(None)
pipeline_manager.set_target_pipeline(None)

# Create dataset
inference_dataset = PolymerSeparatedDataset(
    data=df,
    pipeline_manager=pipeline_manager,
    monomer_smiles_column=monomer_smiles_column,
    solvent_smiles_column=solvent_smiles_column,
    monomer_smiles_transformer=PolymerisationSmilesTransform(),
    mol_to_molgraph=FGMembershipMol2MolGraph(),
    target_columns=target_columns,
    feature_columns=feature_columns,
    is_train=False,
)

# Dataloader
inference_loader = DataLoader(
    inference_dataset, batch_size=32, collate_fn=inference_dataset.collate_fn
)

# Load trained model
hyperparams = {
    "batch_size": 32,
    "lr": 0.0025,
    "weight_decay": 1e-6,
    "log_diffusion_factor": 5.0,
    "log_rg_factor": 3.0,
    "mpnn_output_dim": 128,
    "mpnn_hidden_dim": 96,
    "mpnn_depth": 2,
    "mpnn_dropout": 0.327396910351,
    "rdkit_selection_tensor": torch.tensor([1, 1, 1, 1, 1, 1, 1]),
    "log_selection_tensor": torch.tensor([1, 1, 1, 0, 0, 1]),
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
}

gnn_trainer = SeparatedGNNTrainer(
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    hyperparams=hyperparams,
)

gnn_trainer.model.load_state_dict(torch.load(model_path, map_location=device))
gnn_trainer.model.to(device).eval()

# RDKit headers
rdkit_headers = [
    "NumHDonors",
    "NumHAcceptors",
    "MolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
]

# Run inference
predictions_list = []
monomer_rdkit_list = []
solvent_rdkit_list = []

with torch.no_grad():
    for batch in inference_loader:
        batch_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Model predictions
        predictions = gnn_trainer.model(batch_device).cpu().numpy()
        predictions_list.append(predictions)

        # Collect RDKit features
        monomer_rdkit_list.append(batch_device["monomer_rdkit_tensor"].cpu().numpy())
        solvent_rdkit_list.append(batch_device["solvent_rdkit_tensor"].cpu().numpy())

# Concatenate predictions and RDKit features
preds_numpy = np.concatenate(predictions_list, axis=0)
monomer_rdkit_numpy = np.vstack(monomer_rdkit_list)
solvent_rdkit_numpy = np.vstack(solvent_rdkit_list)

# Add predictions to DataFrame
df["Rg_mean"] = preds_numpy[:, 0]
df["Rg_SD"] = preds_numpy[:, 1]
df["SASA_mean"] = preds_numpy[:, 2]
df["SASA_SD"] = preds_numpy[:, 3]
df["D_mean"] = preds_numpy[:, 4]
df["Re_mean"] = preds_numpy[:, 5]

# Add RDKit features with clear headers
for i, header in enumerate(rdkit_headers):
    df[f"monomer_{header}"] = monomer_rdkit_numpy[:, i]
for i, header in enumerate(rdkit_headers):
    df[f"solvent_{header}"] = solvent_rdkit_numpy[:, i]

# Save updated CSV
df.to_csv(output_csv, index=False)

print(f"Predictions with RDKit features saved to: {output_csv}")
