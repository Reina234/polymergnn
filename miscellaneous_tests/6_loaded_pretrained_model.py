import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from models.modules.configured_mpnn import ConfiguredMPNN
from featurisers.molecule_featuriser import RDKitFeaturizer
from chemprop.nn import NormAggregation
from models.molecule_embedding_model import MoleculeEmbeddingModel
from tools.transform_pipeline_manager import TransformPipelineManager
from sklearn.preprocessing import StandardScaler
from tools.mol_to_molgraph import FGMembershipMol2MolGraph
from training.refactored_trainer import PretrainedGNNTrainer
from training.refactored_batched_dataset import PolymerMorganGNNDataset
from tools.smiles_transformers import PolymerisationSmilesTransform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


df = pd.read_csv("data/output_2_4_2.csv")

feature_columns = [4, 5]
target_columns = [6, 7, 8, 9, 10, 11]
monomer_smiles_column = 3
solvent_smiles_column = 1

pipeline_manager = TransformPipelineManager(feature_columns, target_columns)

pipeline_manager.set_feature_pipeline(StandardScaler())
pipeline_manager.set_target_pipeline(StandardScaler())


train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)


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


################## CONFIG ####################

rdkit_selection_tensor = torch.tensor([0, 0, 1, 1, 1, 1, 1])
mpnn_output_dim = 256
mpnn_hidden_dim = 500
mpnn_depth = 4
mpnn_dropout = 0.1
molecule_embedding_hidden_dim = 256
embedding_dim = 256
use_rdkit = True
use_chembert = False
fnn_hidden_dim = 256
output_dim = len(target_columns)

rdkit_features = [
    "MolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "NumRotatableBonds",
    "RingCount",
    "FractionCSP3",
]

# Validate rdkit_selection_tensor
if rdkit_selection_tensor is None:
    rdkit_selection_tensor = torch.ones(len(rdkit_features))  # Default: Use all
elif rdkit_selection_tensor.shape[0] != len(rdkit_features):
    raise ValueError(
        f"rdkit_selection_tensor must have {len(rdkit_features)} elements!"
    )

selected_rdkit_features = [
    feat
    for feat, select in zip(rdkit_features, rdkit_selection_tensor.tolist())
    if select == 1
]


idx = train_dataset[0]
batch_mol = idx[1][0]
######################## MODEL ############################
mpnn = ConfiguredMPNN(
    output_dim=mpnn_output_dim,
    aggregation_method=NormAggregation(),
    d_h=mpnn_hidden_dim,
    depth=mpnn_depth,
    dropout=mpnn_dropout,
    undirected=True,
)

mpnn.initialize_model(batch_mol)


embedding_model = MoleculeEmbeddingModel(
    chemprop_mpnn=mpnn,
    rdkit_featurizer=RDKitFeaturizer(),
    selected_rdkit_features=selected_rdkit_features,
    chemberta_dim=600,
    hidden_dim=molecule_embedding_hidden_dim,
    output_dim=embedding_dim,
    use_rdkit=use_rdkit,
    use_chembert=use_chembert,
)


################## LOADING ####################

pretrained_mpnn_path = "pretrained_embedding_on_lipophobicity.pth"

pretrained_dict = torch.load(pretrained_mpnn_path, weights_only=True)
model_dict = embedding_model.state_dict()


pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
embedding_model.load_state_dict(pretrained_dict, strict=False)

# embedding_model.load_state_dict(torch.load(pretrained_mpnn_path, weights_only=True))

embedding_model.mpnn.mp.apply(lambda module: module.requires_grad_(False))


hyperparams = {
    "batch_size": 32,
    "lr": 5e-4,
    "weight_decay": 1e-5,
    "log_selection_tensor": torch.tensor([1, 1, 1, 0, 0, 1]),
    "embedding_dim": embedding_dim,
    "gnn_hidden_dim": 128,
    "gnn_output_dim": 120,
    "gnn_dropout": 0.1,
    "gnn_num_heads": 4,
    "multitask_fnn_hidden_dim": 96,
    "multitask_fnn_shared_layer_dim": 128,
    "multitask_fnn_dropout": 0.1,
    "epochs": 50,
    "weights": torch.tensor([1.0, 1.0, 8.0, 1.0, 1.0, 1.0]),
}

################## TRAINING ####################

trainer = PretrainedGNNTrainer(
    model=embedding_model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    hyperparams=hyperparams,
    log_dir="logs/pretraining",
    save_results_dir="results/pretraining/loaded/",
    track_learning_curve=True,
)

trainer.run()
