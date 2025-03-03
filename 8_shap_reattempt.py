import shap
import numpy as np
import torch
from tools.transform_pipeline_manager import TransformPipelineManager
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tools.mol_to_molgraph import FGMembershipMol2MolGraph
from sklearn.model_selection import train_test_split
from training.refactored_batched_dataset import PolymerMorganGNNDataset
from tools.smiles_transformers import PolymerisationSmilesTransform
from torch.utils.data import DataLoader

# from models.polymer_gnn_no_mpnn import PolymerGNNNoMPNNsSystem
from modules.configured_mpnn import ConfiguredMPNN
from chemprop.nn import NormAggregation
from models.shap_modified_models_no_batch.model_ import PolymerGNNNoMPNNsSystem

df = pd.read_csv("tests/output_2_4_2.csv")

feature_columns = [4, 5]
target_columns = [6, 7, 8, 9, 10, 11]
monomer_smiles_column = 3
solvent_smiles_column = 1

pipeline_manager = TransformPipelineManager(feature_columns, target_columns)

pipeline_manager.set_feature_pipeline(StandardScaler())
pipeline_manager.set_target_pipeline(StandardScaler())


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)
shap_df, val_df = train_test_split(df, test_size=0.01, random_state=42)

n_bits = 2048

train_dataset = PolymerMorganGNNDataset(
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

shap_dataset = PolymerMorganGNNDataset(
    data=shap_df,
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


shap_loader = DataLoader(
    shap_dataset, batch_size=1, shuffle=False, collate_fn=shap_dataset.collate_fn
)


train_batch = train_dataset[0]
batch_mol = train_batch[1][0]

########################## CONFIG ##########################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rdkit_selection_tensor = torch.tensor([1, 1, 1, 1, 1, 1, 1])
mpnn_output_dim = 128
mpnn_hidden_dim = 96
mpnn_depth = 2
mpnn_dropout = 0.327396910351
molecule_embedding_hidden_dim = 192
embedding_dim = 100
use_rdkit = True
use_chembert = False
fnn_hidden_dim = 256
output_dim = len(target_columns)
gnn_hidden_dim = 128
gnn_output_dim = 64
gnn_dropout = 0.1
gnn_num_heads = 4
multitask_fnn_hidden_dim = 96
multitask_fnn_shared_layer_dim = 128
multitask_fnn_dropout = 0.1
epochs = 50
weights = torch.tensor([1.0, 1.0, 8.0, 1.0, 1.0, 1.0])

########################## MODEL ##########################
mpnn = ConfiguredMPNN(
    output_dim=mpnn_output_dim,
    aggregation_method=NormAggregation(),
    d_h=mpnn_hidden_dim,
    depth=mpnn_depth,
    dropout=mpnn_dropout,
    undirected=True,
)

mpnn.initialize_model(batch_mol)

model = PolymerGNNNoMPNNsSystem(
    mpnn=mpnn,
    rdkit_selection_tensor=rdkit_selection_tensor,
    molecule_embedding_hidden_dim=molecule_embedding_hidden_dim,
    embedding_dim=embedding_dim,
    gnn_hidden_dim=gnn_hidden_dim,
    gnn_output_dim=gnn_output_dim,
    gnn_dropout=gnn_dropout,
    gnn_num_heads=gnn_num_heads,
    multitask_fnn_hidden_dim=multitask_fnn_hidden_dim,
    multitask_fnn_shared_layer_dim=multitask_fnn_shared_layer_dim,
    multitask_fnn_dropout=multitask_fnn_dropout,
)

pretrained_dict = torch.load("polymergnn_full_mod_morgan2.pth", weights_only=True)

model.load_state_dict(pretrained_dict, strict=False)


######################## MODEL WRAPPER ############################
rdkit_feature_names = [
    "HDonors",
    "HAcceptors",
    "MolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
]


# Collect all 7 batches
all_batches = []
for batch in shap_loader:
    if batch["polymer_mapping"].shape[0] == 2:  # Your validation condition
        all_batches.append(batch)
        if len(all_batches) >= 7:
            break

# Preprocess batches into SHAP-compatible format
shap_inputs = []
for batch in all_batches:
    # Replicate your original preprocessing logic
    polymer_rdkit = batch["rdkit_tensor"][0].unsqueeze(0)
    solvent_rdkit = batch["rdkit_tensor"][1].unsqueeze(0)
    polymer_fingerprint = batch["fingerprints_tensor"][:, :2048]
    solvent_fingerprint = batch["fingerprints_tensor"][:, 2048:]
    polymer_mpnn_out = model.mpnn(batch["batch_mol_graph"])[0].unsqueeze(0)
    solvent_mpnn_out = model.mpnn(batch["batch_mol_graph"])[1].unsqueeze(0)
    polymer_feats = batch["polymer_feats"]

    shap_input = (
        torch.cat(
            [
                polymer_rdkit,
                solvent_rdkit,
                polymer_fingerprint,
                solvent_fingerprint,
                polymer_mpnn_out,
                solvent_mpnn_out,
                polymer_feats,
            ],
            dim=1,
        )
        .detach()
        .cpu()
        .numpy()
    )

    shap_inputs.append(shap_input)

background_data = np.concatenate(shap_inputs, axis=0)


def model_wrapper(perturbed_inputs):
    if isinstance(perturbed_inputs, np.ndarray):
        perturbed_inputs = torch.tensor(perturbed_inputs, dtype=torch.float32)

    # Ensure batch dimension is handled
    batch_size = perturbed_inputs.shape[0]

    outputs = []
    for i in range(batch_size):
        idx = 0
        single_input = perturbed_inputs[i].unsqueeze(0)  # (1, n_features)

        # Split into components (replicate your original logic)
        perturbed_polymer_rdkit = single_input[:, idx : idx + 9]
        idx += 9
        perturbed_solvent_rdkit = single_input[:, idx : idx + 9]
        idx += 9
        perturbed_polymer_fingerprint = single_input[:, idx : idx + 2048]
        idx += 2048
        perturbed_solvent_fingerprint = single_input[:, idx : idx + 2048]
        idx += 2048
        perturbed_polymer_mpnn = single_input[
            :, idx : idx + 128
        ]  # Assuming mpnn_dim=128
        idx += 128
        perturbed_solvent_mpnn = single_input[:, idx : idx + 128]
        idx += 128
        perturbed_polymer_feats = single_input[
            :, idx : idx + 2
        ]  # Assuming N/T features

        # Get static features from first batch (adapt as needed)
        static_args = (
            all_batches[0]["edge_index"],
            all_batches[0]["edge_attr"],
            all_batches[0]["polymer_mapping"],
        )

        output = (
            model(
                torch.cat([perturbed_polymer_mpnn, perturbed_solvent_mpnn], dim=0),
                torch.cat([perturbed_polymer_rdkit, perturbed_solvent_rdkit], dim=0),
                perturbed_polymer_feats,
                torch.cat(
                    [perturbed_polymer_fingerprint, perturbed_solvent_fingerprint],
                    dim=1,
                ),
                *static_args,
            )
            .detach()
            .cpu()
            .numpy(),
        )

        outputs.append(output)

    return np.concatenate(outputs, axis=0)


explainer = shap.KernelExplainer(
    model_wrapper, shap.sample(background_data, 7)  # Use all 7 samples as background
)

# For explanation, use first sample as example
sample_idx = 0
shap_values = explainer.shap_values(background_data[sample_idx : sample_idx + 1])


# Process all samples
all_shap_values = []
for i in range(7):
    shap_values = explainer.shap_values(background_data[i : i + 1])
    all_shap_values.append(shap_values)

# Combine SHAP values (assuming regression task)
shap_values_np = np.stack(all_shap_values, axis=0)  # Shape: (7, n_features, n_outputs)


def aggregate_shap(shap_array):
    polymer_fingerprint = np.mean(shap_array[:, 18:2066, :], axis=1)
    solvent_fingerprint = np.mean(shap_array[:, 2066:4114, :], axis=1)
    polymer_mpnn = np.mean(shap_array[:, 4114:4242, :], axis=1)
    solvent_mpnn = np.mean(shap_array[:, 4242:4370, :], axis=1)

    return np.concatenate(
        [
            shap_array[:, :18, :],  # RDKit features
            polymer_fingerprint[:, None, :],
            solvent_fingerprint[:, None, :],
            polymer_mpnn[:, None, :],
            solvent_mpnn[:, None, :],
            shap_array[:, 4370:, :],  # Polymer features
        ],
        axis=1,
    )


shap_aggregated = aggregate_shap(shap_values_np)

# For summary plot (average across samples)
shap_aggregated_2d = np.mean(shap_aggregated, axis=0)  # (n_features, n_outputs)

feature_names = (
    [f"{name}_polymer" for name in rdkit_feature_names]
    + [f"{name}_solvent" for name in rdkit_feature_names]
    + ["polymer_fingerprint", "solvent_fingerprint", "polymer_mpnn", "solvent_mpnn"]
    + ["N", "T"]
)

shap.summary_plot(
    shap_aggregated_2d.T,  # SHAP expects (n_samples, n_features)
    features=np.zeros((1, len(feature_names))),  # Dummy features for names
    feature_names=feature_names,
    plot_type="bar",
)
