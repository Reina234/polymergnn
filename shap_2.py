import shap
import numpy as np
import torch
from tools.transform_pipeline_manager import TransformPipelineManager
from sklearn.preprocessing import StandardScaler
import pandas as pd
from tools.mol_to_molgraph import FGMembershipMol2MolGraph
from sklearn.model_selection import train_test_split
from training.refactored_batched_dataset import PolymerGNNDataset
from tools.smiles_transformers import PolymerisationSmilesTransform
from torch.utils.data import DataLoader

# from models.polymer_gnn_no_mpnn import PolymerGNNNoMPNNsSystem
from modules.configured_mpnn import ConfiguredMPNN
from chemprop.nn import NormAggregation
from models.shap_modified_models_no_batch.model_ import PolymerGNNNoMorganMPNNsSystem

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

train_dataset = PolymerGNNDataset(
    data=train_df,
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

shap_dataset = PolymerGNNDataset(
    data=shap_df,
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

model = PolymerGNNNoMorganMPNNsSystem(
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

pretrained_dict = torch.load("polymergnn_full_mod_gat.pth", weights_only=True)

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


def generate_polymer_mapping(num_samples):
    return torch.arange(num_samples).repeat_interleave(2)


def generate_fully_connected_edge_index(num_nodes):
    row = torch.arange(num_nodes).repeat(num_nodes)  # Repeat each node N times
    col = torch.arange(num_nodes).repeat_interleave(
        num_nodes
    )  # Expand each node N times

    # Remove self-loops (i.e., where row == col)

    return torch.stack([row, col], dim=0)


def generate_edge_attr(perturbed_inputs, edge_index, polymer_rdkit, solvent_rdkit):
    idx = 0
    perturbed_polymer_rdkit = perturbed_inputs[:, idx : idx + polymer_rdkit.shape[1]]
    idx += polymer_rdkit.shape[1]

    perturbed_solvent_rdkit = perturbed_inputs[:, idx : idx + solvent_rdkit.shape[1]]
    idx += solvent_rdkit.shape[1]

    # Merge polymer and solvent RDKit features
    rdkit_features = torch.cat(
        [perturbed_polymer_rdkit, perturbed_solvent_rdkit], dim=0
    )

    # Extract HDonors (Index 0) and HAcceptors (Index 1) from rdkit_features
    h_donors = rdkit_features[:, 0]  # HDonors
    h_acceptors = rdkit_features[:, 1]  # HAcceptors

    num_edges = edge_index.shape[1]
    edge_attr = torch.zeros((num_edges, 2), dtype=torch.float32)  # Initialize edge_attr

    # Iterate over edges to compute edge attributes
    for k in range(num_edges):
        i, j = edge_index[:, k]  # Get nodes for this edge
        edge_attr[k, 1] = max(h_acceptors[i], h_donors[j]) + max(
            h_donors[i], h_acceptors[j]
        )

    return edge_attr


polymer_rdkit = torch.stack(polymer_rdkit_list, dim=0)
solvent_rdkit = torch.stack(solvent_rdkit_list, dim=0)


polymer_mpnn_out = torch.stack(polymer_mpnn_list, dim=0)  # (num_samples, mpnn_dim)
solvent_mpnn_out = torch.stack(solvent_mpnn_list, dim=0)  # (num_samples, mpnn_dim)

polymer_feats = torch.stack(
    polymer_feats_list, dim=0
)  # (num_samples, num_polymer_features)


shap_input = torch.cat(
    [
        polymer_rdkit,  # (num_samples, num_rdkit_features)
        solvent_rdkit,  # (num_samples, num_rdkit_features)
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

    num_samples = perturbed_inputs.shape[0]  # Number of perturbed input samples

    # ======= Extract Features from Input =======
    idx = 0
    perturbed_polymer_rdkit = perturbed_inputs[:, idx : idx + polymer_rdkit.shape[1]]
    idx += polymer_rdkit.shape[1]

    perturbed_solvent_rdkit = perturbed_inputs[:, idx : idx + solvent_rdkit.shape[1]]
    idx += solvent_rdkit.shape[1]

    perturbed_polymer_mpnn = perturbed_inputs[:, idx : idx + polymer_mpnn_out.shape[1]]
    idx += polymer_mpnn_out.shape[1]

    perturbed_solvent_mpnn = perturbed_inputs[:, idx : idx + solvent_mpnn_out.shape[1]]
    idx += solvent_mpnn_out.shape[1]

    perturbed_polymer_feats = perturbed_inputs[:, idx : idx + polymer_feats.shape[1]]
    idx += polymer_feats.shape[1]

    # ======= Construct Dynamic Graph Structures =======
    polymer_mappings = generate_polymer_mapping(num_samples)  # [0, 0, 1, 1, 2, 2, ...]
    edge_index = generate_fully_connected_edge_index(
        num_samples
    )  # Fully connected graph
    edge_attr = generate_edge_attr(
        perturbed_inputs, edge_index, polymer_rdkit, solvent_rdkit
    )  # Edge attributes

    # ======= Concatenate Inputs for Model =======
    concatenated_rdkit = torch.cat(
        [perturbed_polymer_rdkit, perturbed_solvent_rdkit], dim=0
    )
    concatenated_mpnn = torch.cat(
        [perturbed_polymer_mpnn, perturbed_solvent_mpnn], dim=0
    )

    # ======= Pass to Model =======
    model_output = (
        model(
            concatenated_mpnn,
            concatenated_rdkit,
            perturbed_polymer_feats,
            edge_index,
            edge_attr,
            polymer_mappings,
        )
        .detach()
        .cpu()
        .numpy()
    )

    return model_output


explainer = shap.KernelExplainer(model_wrapper, shap.sample(shap_input, 7))
shap_values = explainer.shap_values(shap_input, n_samples=2048)

shap_values_np = np.array(shap_values)
polymer_fingerprint_shap = np.mean(shap_values_np[:, 18:2066], axis=1, keepdims=True)
solvent_fingerprint_shap = np.mean(shap_values_np[:, 2066:4114], axis=1, keepdims=True)


polymer_mpnn_shap = np.mean(shap_values_np[:, 18:146], axis=1, keepdims=True)
solvent_mpnn_shap = np.mean(shap_values_np[:, 146:273], axis=1, keepdims=True)


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
