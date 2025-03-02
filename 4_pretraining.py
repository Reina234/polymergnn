import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from modules.configured_mpnn import ConfiguredMPNN
from featurisers.molecule_featuriser import RDKitFeaturizer
from chemprop.nn import NormAggregation
from models.mpnn_pretrain_wrapper import PretrainingWrapper
from models.molecule_embedding_model import MoleculeEmbeddingModel
from featurisers.molecule_featuriser import RDKitFeaturizer
from modules.configured_mpnn import ConfiguredMPNN
from tools.transform_pipeline_manager import TransformPipelineManager
from sklearn.preprocessing import StandardScaler
from tools.mol_to_molgraph import FGMembershipMol2MolGraph
from training.refactored_batched_dataset import PolymerGNNDataset
from tools.smiles_transformers import NoSmilesTransform
from training.refactored_trainer import PretrainingTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_columns = [1]
monomer_smiles_column = 2
feature_columns = None


df = pd.read_csv("tests/Lipophilicity.csv")
# df = pd.read_csv("tests/freesolv/FreeSolv_SAMPL.csv")
# df = df.drop(df.columns[[0, 1]], axis=1)

pipeline_manager = TransformPipelineManager(
    feature_columns, target_columns, log_transform_targets=False
)
pipeline_manager.set_feature_pipeline(StandardScaler())
pipeline_manager.set_target_pipeline(StandardScaler())

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

train_dataset = PolymerGNNDataset(
    data=train_df,
    pipeline_manager=pipeline_manager,
    monomer_smiles_column=monomer_smiles_column,
    solvent_smiles_column=None,
    monomer_smiles_transformer=NoSmilesTransform(),
    mol_to_molgraph=FGMembershipMol2MolGraph(),
    target_columns=target_columns,
    feature_columns=feature_columns,
    is_train=True,
)

fitted_pipeline_manager = train_dataset.pipeline_manager

val_dataset = PolymerGNNDataset(
    data=val_df,
    pipeline_manager=fitted_pipeline_manager,
    monomer_smiles_column=monomer_smiles_column,
    solvent_smiles_column=None,
    monomer_smiles_transformer=NoSmilesTransform(),
    mol_to_molgraph=FGMembershipMol2MolGraph(),
    target_columns=target_columns,
    feature_columns=feature_columns,
    is_train=False,
)

test_dataset = PolymerGNNDataset(
    data=test_df,
    pipeline_manager=fitted_pipeline_manager,
    monomer_smiles_column=monomer_smiles_column,
    solvent_smiles_column=None,
    monomer_smiles_transformer=NoSmilesTransform(),
    mol_to_molgraph=FGMembershipMol2MolGraph(),
    target_columns=target_columns,
    feature_columns=feature_columns,
    is_train=False,
)


rdkit_selection_tensor = torch.tensor([0, 0, 1, 1, 1, 1, 1])
mpnn_output_dim = 256
mpnn_hidden_dim = 500
mpnn_depth = 4
mpnn_dropout = 0.1
molecule_embedding_hidden_dim = 256
embedding_dim = 256
use_rdkit = True
use_chembert = False


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
mpnn = ConfiguredMPNN(
    output_dim=mpnn_output_dim,
    aggregation_method=NormAggregation(),
    d_h=mpnn_hidden_dim,
    depth=mpnn_depth,
    dropout=mpnn_dropout,
    undirected=True,
)

# Molecule Embedding Model
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


fnn_hidden_dim = 256
output_dim = len(target_columns)
model = PretrainingWrapper(embedding_model, output_dim, embedding_dim, fnn_hidden_dim)


hyperparams = {
    "batch_size": 32,
    "lr": 5e-4,
    "weight_decay": 1e-5,
    "log_selection_tensor": torch.tensor([0]),
    "epochs": 20,
}

trainer = PretrainingTrainer(
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    hyperparams=hyperparams,
    log_dir="logs/pretraining",
    save_results_dir="results/pretraining/",
    track_learning_curve=True,
)

trainer.run()

torch.save(embedding_model.state_dict(), "pretrained_embedding_on_lipophobicity.pth")
