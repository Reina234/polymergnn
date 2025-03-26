import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tools.transform_pipeline_manager import TransformPipelineManager
from tools.mol_to_molgraph import FGMembershipMol2MolGraph
from tools.smiles_transformers import PolymerisationSmilesTransform
from training.no_fitting_dataset import NoFitPolymerSeparatedDataset
from separated_models import train_dataset, val_dataset, gnn_trainer

# Define file paths
model_path = "/Users/reinazheng/Desktop/polymergnn/march_28_v3.pth"
input_csv = (
    "/Users/reinazheng/Desktop/polymergnn/solvation_prediction/additional_data.csv"
)
output_csv = "solvent_polymer_predictions3.csv"

# Device setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load CSV
solvation_df = pd.read_csv(input_csv)

sol_feature_columns = [1, 5, 6]
sol_monomer_smiles_column = 2
sol_solvent_smiles_column = 0

###################################################################################
# Create dataset
inference_dataset = NoFitPolymerSeparatedDataset(
    data=solvation_df,
    monomer_smiles_column=sol_monomer_smiles_column,
    solvent_smiles_column=sol_solvent_smiles_column,
    monomer_smiles_transformer=PolymerisationSmilesTransform(),
    mol_to_molgraph=FGMembershipMol2MolGraph(),
    feature_columns=sol_feature_columns,
)

inference_loader = DataLoader(
    inference_dataset, batch_size=32, collate_fn=inference_dataset.collate_fn
)

#################################################################################

hyperparams = {
    "batch_size": 32,
    "lr": 0.001,
    "weight_decay": 1e-5,
    "mpnn_output_dim": 128,
    "mpnn_hidden_dim": 128,
    "mpnn_depth": 3,
    "mpnn_dropout": 0.1,
    "rdkit_selection_tensor": torch.tensor([1, 1, 1, 1, 1, 1, 1]),
    "log_selection_tensor": torch.tensor(
        [1, 1, 1, 0, 0, 1]
    ),  # Only log-transform 2nd label
    "molecule_embedding_hidden_dim": 192,
    "embedding_dim": 80,
    "use_rdkit": True,
    "use_chembert": False,
    "gnn_hidden_dim": 128,
    "gnn_output_dim": 64,
    "gnn_dropout": 0.1,
    "gnn_num_heads": 5,
    "multitask_fnn_hidden_dim": 128,
    "multitask_fnn_shared_layer_dim": 256,
    "multitask_fnn_dropout": 0.1,
    "epochs": 100,
    "weights": torch.tensor([1.0, 1.0, 10.0, 1.0, 1.0, 1.0]),
}
target_columns = [7, 8, 9, 10, 11, 12]
feature_columns = [0, 5, 6]
monomer_smiles_column = 4
solvent_smiles_column = 2

df = pd.read_csv("data/output_3_11_with_density.csv")


pipeline_manager = TransformPipelineManager(
    feature_indexes=feature_columns, target_indexes=target_columns
)


idx = train_dataset[0]
batch_mol = idx[1][0]


gnn_trainer.model.monomer_embedding.mpnn.initialize_model(batch_mol)
gnn_trainer.model.solvent_embedding.mpnn.initialize_model(batch_mol)

gnn_trainer.model.load_state_dict(torch.load(model_path, map_location=device))
gnn_trainer.model.to(device).eval()


##################################################


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


preds_numpy = gnn_trainer.infer(inference_loader)

# Collect RDKit features
monomer_rdkit_list = []
solvent_rdkit_list = []

with torch.no_grad():
    for batch in inference_loader:
        batch_device = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        # Collect RDKit features
        monomer_rdkit_list.append(batch_device["monomer_rdkit_tensor"].cpu().numpy())
        solvent_rdkit_list.append(batch_device["solvent_rdkit_tensor"].cpu().numpy())

# Concatenate RDKit features
monomer_rdkit_numpy = np.vstack(monomer_rdkit_list)
solvent_rdkit_numpy = np.vstack(solvent_rdkit_list)

# Add predictions to DataFrame
solvation_df["Rg_mean"] = preds_numpy[:, 0]
solvation_df["Rg_SD"] = preds_numpy[:, 1]
solvation_df["D_mean"] = preds_numpy[:, 2]
solvation_df["SASA_mean"] = preds_numpy[:, 3]
solvation_df["SASA_SD"] = preds_numpy[:, 4]
solvation_df["Re_mean"] = preds_numpy[:, 5]

# Add RDKit features with clear headers
for i, header in enumerate(rdkit_headers):
    solvation_df[f"monomer_{header}"] = monomer_rdkit_numpy[:, i]
for i, header in enumerate(rdkit_headers):
    solvation_df[f"solvent_{header}"] = solvent_rdkit_numpy[:, i]

# Save updated CSV
solvation_df.to_csv(output_csv, index=False)

print(f"Predictions with RDKit features saved to: {output_csv}")
