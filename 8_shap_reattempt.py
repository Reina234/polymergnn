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

val_dataset = PolymerMorganGNNDataset(
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


val_loader = DataLoader(
    val_dataset, batch_size=1, shuffle=False, collate_fn=val_dataset.collate_fn
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


def model_wrapper(perturbed_inputs):
    if isinstance(perturbed_inputs, np.ndarray):
        perturbed_inputs = torch.tensor(perturbed_inputs, dtype=torch.float32)

    idx = 0
    perturbed_polymer_rdkit = perturbed_inputs[:, idx : idx + polymer_rdkit.shape[0]]
    idx += polymer_rdkit.shape[0]

    perturbed_solvent_rdkit = perturbed_inputs[:, idx : idx + solvent_rdkit.shape[0]]
    idx += solvent_rdkit.shape[0]

    perturbed_polymer_fingerprint = perturbed_inputs[:, idx : idx + 2048]
    idx += 2048

    perturbed_solvent_fingerprint = perturbed_inputs[:, idx : idx + 2048]
    idx += 2048

    perturbed_polymer_mpnn = perturbed_inputs[:, idx : idx + polymer_mpnn_out.shape[0]]
    idx += polymer_mpnn_out.shape[0]

    perturbed_solvent_mpnn = perturbed_inputs[:, idx : idx + solvent_mpnn_out.shape[0]]
    idx += solvent_mpnn_out.shape[0]

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
            edge_index,
            edge_attr,
            polymer_mapping,
        )
        .detach()
        .cpu()
        .numpy()
    )

    return model_output


########################## MODEL DECONSTRUCTION ##########################

model.eval()


for batch in val_loader:
    if batch["polymer_mapping"].shape[0] == 2:  # Check if tensor has exactly 2 elements
        sample_batch = batch
        break


mpnn_out = model.mpnn(sample_batch["batch_mol_graph"])
polymer_mpnn_out = mpnn_out[0]
solvent_mpnn_out = mpnn_out[1]

edge_index = sample_batch["edge_index"]
edge_attr = sample_batch["edge_attr"]
polymer_mapping = sample_batch["polymer_mapping"]

polymer_rdkit = sample_batch["rdkit_tensor"][0]
solvent_rdkit = sample_batch["rdkit_tensor"][1]

polymer_fingerprint = sample_batch["fingerprints_tensor"][:, :2048]
solvent_fingerprint = sample_batch["fingerprints_tensor"][:, 2048:]

polymer_feats = sample_batch["polymer_feats"]

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
shap_input = torch.cat(
    [
        polymer_rdkit.unsqueeze(0),  # Shape (1, num_rdkit_features)
        solvent_rdkit.unsqueeze(0),  # Shape (1, num_rdkit_features)
        polymer_fingerprint,  # Shape (1, 2048)
        solvent_fingerprint,  # Shape (1, 2048)
        polymer_mpnn_out.unsqueeze(0),  # Shape (1, mpnn_dim)
        solvent_mpnn_out.unsqueeze(0),  # Shape (1, mpnn_dim)
        polymer_feats,  # Shape (1, num_polymer_features), **not perturbed**
    ],
    dim=1,
)  # Shape: (1, total_features)


shap_input = shap_input.detach().cpu().numpy()
explainer = shap.KernelExplainer(model_wrapper, shap.sample(shap_input, 100))
shap_values = explainer.shap_values(shap_input)

shap_values_np = np.array(shap_values)


polymer_fingerprint_shap = np.mean(shap_values_np[:, 18:2066], axis=1, keepdims=True)
solvent_fingerprint_shap = np.mean(shap_values_np[:, 2066:4114], axis=1, keepdims=True)

polymer_mpnn_shap = np.mean(shap_values_np[:, 4114:4242], axis=1, keepdims=True)
solvent_mpnn_shap = np.mean(shap_values_np[:, 4242:4370], axis=1, keepdims=True)

shap_aggregated_2d = np.concatenate(
    [
        shap_values_np[:, :18].squeeze().T,
        polymer_fingerprint_shap.reshape(-1, 1),
        solvent_fingerprint_shap.reshape(-1, 1),
        polymer_mpnn_shap.reshape(-1, 1),
        solvent_mpnn_shap.reshape(-1, 1),
        shap_values_np[:, 4370:].squeeze().T,
    ],
    axis=1,
)


feature_names = (
    [f"{name}_polymer" for name in rdkit_feature_names]  # Polymer RDKit
    + [f"{name}_solvent" for name in rdkit_feature_names]  # Solvent RDKit
    + [
        "polymer_fingerprint",
        "solvent_fingerprint",  # Aggregated groups
        "polymer_mpnn",
        "solvent_mpnn",
    ]
    + ["N", "T"]  # Polymer features
)


shap.summary_plot(
    shap_aggregated_2d,
    feature_names=feature_names,
    plot_type="bar",  # Use "dot" for detailed view or "bar" for global importance
)
