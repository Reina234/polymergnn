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

test_loader = DataLoader(
    train_dataset, batch_size=10, shuffle=False, collate_fn=shap_dataset.collate_fn
)


batch = next(iter(test_loader))
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

# 🚀 Collect All Valid Batches & Extract Features
valid_batches = []
polymer_rdkit_list = []
solvent_rdkit_list = []
polymer_fingerprint_list = []
solvent_fingerprint_list = []
polymer_mpnn_list = []
solvent_mpnn_list = []
polymer_feats_list = []
edge_index_list = []
edge_attr_list = []
polymer_mapping_list = []

for batch in shap_loader:
    if batch["polymer_mapping"].shape[0] == 2:
        valid_batches.append(batch)

        # Extract features
        mpnn_out = model.mpnn(batch["batch_mol_graph"])

        polymer_rdkit_list.append(batch["rdkit_tensor"][0])
        solvent_rdkit_list.append(batch["rdkit_tensor"][1])

        polymer_fingerprint_list.append(batch["fingerprints_tensor"][:, :2048])
        solvent_fingerprint_list.append(batch["fingerprints_tensor"][:, 2048:])

        polymer_mpnn_list.append(mpnn_out[0])
        solvent_mpnn_list.append(mpnn_out[1])

        polymer_feats_list.append(batch["polymer_feats"])

        # Retrieve Edge Attributes & Index
        edge_index_list.append(batch["edge_index"])
        edge_attr_list.append(batch["edge_attr"])
        polymer_mapping_list.append(batch["polymer_mapping"])

# 🚀 Ensure we found valid batches
if len(valid_batches) == 0:
    raise ValueError("No valid batches found in val_loader!")


def combine_shifted_tensors(tensor_list):
    if not tensor_list:
        return torch.tensor([])  # Return empty tensor if list is empty

    combined_tensors = []
    shift = 0  # Keeps track of the cumulative shift

    for tensor in tensor_list:
        if tensor.numel() == 0:
            continue  # Skip empty tensors
        shifted_tensor = tensor + shift
        combined_tensors.append(shifted_tensor)
        shift = shifted_tensor.max().item() + 1  # Update shift for the next tensor

    return torch.cat(combined_tensors, dim=0)


import torch


def combine_shifted_tensors_tuple(tensor_tuple_list):
    """
    Combines a list of tuples of tensors while shifting indices properly.

    Each tuple is treated separately, and shifting is applied using the max index found in any tensor
    inside that tuple. The final result maintains the tuple structure.

    Args:
        tensor_tuple_list (list of tuples of torch.Tensor): List of tuples of tensors.

    Returns:
        tuple of torch.Tensor: Concatenated and shifted tensors, maintaining the tuple structure.
    """
    if not tensor_tuple_list:
        return tuple()  # Return empty tuple if list is empty

    # Initialize lists to store combined tensors per tuple element
    num_elements = len(tensor_tuple_list[0])  # Get the number of elements per tuple
    combined_tensors = [[] for _ in range(num_elements)]
    shifts = [0] * num_elements  # Track shifts per tuple element

    for tensor_tuple in tensor_tuple_list:
        # Find the max index across all tensors in this tuple to apply a uniform shift
        max_index = (
            max(
                (tensor.max().item() for tensor in tensor_tuple if tensor.numel() > 0),
                default=-1,
            )
            + 1
        )

        shifted_tuple = []
        for i, tensor in enumerate(tensor_tuple):
            if tensor.numel() > 0:
                shifted_tensor = tensor.clone() + shifts[i]
                shifted_tuple.append(shifted_tensor)
                combined_tensors[i].append(shifted_tensor)

        # Update shift for next tuple based on the max index found in this one
        for i in range(num_elements):
            shifts[i] += max_index

    # Convert each list to a concatenated tensor
    final_result = tuple(
        torch.cat(tensor_list, dim=0) if tensor_list else torch.tensor([])
        for tensor_list in combined_tensors
    )

    return final_result


import torch

import torch


def combine_shifted_2d_tensors(tensor_list):
    """
    Combines a list of 2D tensors (each of shape [2, N]) into a single 2D tensor
    while shifting indices in both rows to avoid collisions.

    The first row represents node indices and is shifted cumulatively.
    The second row is also shifted using the same offset.

    Args:
        tensor_list (list of torch.Tensor): List of tensors with shape [2, N].

    Returns:
        torch.Tensor: Concatenated tensor with indices shifted properly.
    """
    if not tensor_list:
        return torch.tensor(
            [[], []], dtype=torch.long
        )  # Return empty tensor if list is empty

    combined_tensors = []
    shift = 0  # Keeps track of the cumulative shift

    for tensor in tensor_list:
        if tensor.numel() == 0:
            continue  # Skip empty tensors

        shifted_tensor = tensor.clone()  # Avoid modifying original tensors
        shifted_tensor += shift  # Shift both rows by the same amount
        combined_tensors.append(shifted_tensor)

        shift += tensor.max().item() + 1  # Update shift for the next tensor

    return torch.cat(combined_tensors, dim=1)  # Concatenate along the second dimension


# Example


def concatenate_tensors(tensor_list):
    """
    Concatenates a list of tensors into a single tensor along dim=0.

    Args:
        tensor_list (list of torch.Tensor): List of tensors to concatenate.

    Returns:
        torch.Tensor: A single concatenated tensor.
    """
    if not tensor_list:
        return torch.tensor([])  # Return empty tensor if list is empty

    return torch.cat(tensor_list, dim=0)


edge_indexes = combine_shifted_2d_tensors(edge_index_list)
edge_attrs = concatenate_tensors(edge_attr_list)
polymer_mappings = combine_shifted_tensors(polymer_mapping_list)

polymer_rdkit = torch.stack(polymer_rdkit_list, dim=0)
solvent_rdkit = torch.stack(solvent_rdkit_list, dim=0)


print(edge_index_list)
print(edge_indexes)

polymer_fingerprint = torch.stack(
    polymer_fingerprint_list, dim=0
)  # (num_samples, 2048)
solvent_fingerprint = torch.stack(
    solvent_fingerprint_list, dim=0
)  # (num_samples, 2048)

polymer_mpnn_out = torch.stack(polymer_mpnn_list, dim=0)  # (num_samples, mpnn_dim)
solvent_mpnn_out = torch.stack(solvent_mpnn_list, dim=0)  # (num_samples, mpnn_dim)

polymer_feats = torch.stack(
    polymer_feats_list, dim=0
)  # (num_samples, num_polymer_features)


shap_input = torch.cat(
    [
        polymer_rdkit,  # (num_samples, num_rdkit_features)
        solvent_rdkit,  # (num_samples, num_rdkit_features)
        polymer_fingerprint.squeeze(1),  # (num_samples, 2048)
        solvent_fingerprint.squeeze(1),  # (num_samples, 2048)
        polymer_mpnn_out,  # (num_samples, mpnn_dim)
        solvent_mpnn_out,  # (num_samples, mpnn_dim)
        polymer_feats.squeeze(1),
    ],
    dim=1,  # 🚀 Concatenating along feature axis
)


shap_input = shap_input.detach().cpu().numpy()


def model_wrapper(perturbed_inputs):
    if isinstance(perturbed_inputs, np.ndarray):
        perturbed_inputs = torch.tensor(perturbed_inputs, dtype=torch.float32)

    idx = 0
    perturbed_polymer_rdkit = perturbed_inputs[:, idx : idx + polymer_rdkit.shape[1]]
    idx += polymer_rdkit.shape[1]

    perturbed_solvent_rdkit = perturbed_inputs[:, idx : idx + solvent_rdkit.shape[1]]
    idx += solvent_rdkit.shape[1]

    perturbed_polymer_fingerprint = perturbed_inputs[:, idx : idx + 2048]
    idx += 2048

    perturbed_solvent_fingerprint = perturbed_inputs[:, idx : idx + 2048]
    idx += 2048

    perturbed_polymer_mpnn = perturbed_inputs[:, idx : idx + polymer_mpnn_out.shape[1]]
    idx += polymer_mpnn_out.shape[1]

    perturbed_solvent_mpnn = perturbed_inputs[:, idx : idx + solvent_mpnn_out.shape[1]]
    idx += solvent_mpnn_out.shape[1]

    perturbed_polymer_feats = perturbed_inputs[:, idx : idx + polymer_feats.shape[1]]
    idx += polymer_feats.shape[1]

    concatenated_rdkit = torch.cat(
        [perturbed_polymer_rdkit, perturbed_solvent_rdkit], dim=0
    )
    concatinated_fingerprints = torch.cat(
        [perturbed_polymer_fingerprint, perturbed_solvent_fingerprint], dim=1
    )
    concatenated_mpnn = torch.cat(
        [perturbed_polymer_mpnn, perturbed_solvent_mpnn], dim=0
    )

    model_output = (
        model(
            concatenated_mpnn,
            concatenated_rdkit,
            perturbed_polymer_feats,
            concatinated_fingerprints,
            edge_indexes,
            edge_attrs,
            polymer_mappings,
        )
        .detach()
        .cpu()
        .numpy()
    )

    return model_output


explainer = shap.KernelExplainer(model_wrapper, shap.sample(shap_input, 7))
shap_values = explainer.shap_values(shap_input)

shap_values_np = np.array(shap_values)
polymer_fingerprint_shap = np.mean(shap_values_np[:, 18:2066], axis=1, keepdims=True)
solvent_fingerprint_shap = np.mean(shap_values_np[:, 2066:4114], axis=1, keepdims=True)


polymer_mpnn_shap = np.mean(shap_values_np[:, 4114:4242], axis=1, keepdims=True)
solvent_mpnn_shap = np.mean(shap_values_np[:, 4242:4370], axis=1, keepdims=True)


shap_aggregated_2d = np.concatenate(
    [
        shap_values_np[:, :18].squeeze().T,  # RDKit Features
        polymer_fingerprint_shap.reshape(-1, 1),  # Aggregated Fingerprints
        solvent_fingerprint_shap.reshape(-1, 1),
        polymer_mpnn_shap.reshape(-1, 1),  # Aggregated MPNN
        solvent_mpnn_shap.reshape(-1, 1),
        shap_values_np[:, 4370:].squeeze().T,  # Polymer Features
    ],
    axis=1,
)


feature_names = (
    [f"{name}_polymer" for name in rdkit_feature_names]
    + [f"{name}_solvent" for name in rdkit_feature_names]
    + ["polymer_fingerprint", "solvent_fingerprint", "polymer_mpnn", "solvent_mpnn"]
    + ["N", "T"]
)

shap.summary_plot(
    shap_aggregated_2d,
    feature_names=feature_names,
    plot_type="bar",
)
