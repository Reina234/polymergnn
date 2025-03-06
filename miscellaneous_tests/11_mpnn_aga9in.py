import torch.nn as nn
from chemprop.nn import BondMessagePassing, NormAggregation, RegressionFFN
from chemprop.models.model import MPNN
from chemprop.data import BatchMolGraph
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from models.modules.configured_mpnn import ConfiguredMPNN
from chemprop.nn import NormAggregation
from models.modules.configured_mpnn import ConfiguredMPNN
from tools.transform_pipeline_manager import TransformPipelineManager
from sklearn.preprocessing import StandardScaler
from tools.mol_to_molgraph import FGMembershipMol2MolGraph
from training.refactored_trainer import MPNNTrainer
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


class ConfigMPNN(nn.Module):
    def __init__(
        self,
        output_dim: int,
        aggregation_method=NormAggregation(),
        d_h: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        undirected: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.aggregation_method = aggregation_method
        self.d_h = d_h
        self.depth = depth
        self.dropout = dropout
        self.undirected = undirected

        self.mp = None  # Will be initialized during the first forward pass
        self.model = None
        self.d_e = None
        self.d_v = None
        self.fnn = RegressionFFN

    def initialize_model(self, batch_mol_graph: BatchMolGraph):
        """Initializes BondMessagePassing using dimensions from the input batch."""
        self.d_e = batch_mol_graph.E.shape[1]  # Edge feature dimension
        self.d_v = batch_mol_graph.V.shape[1]  # Node feature dimension

        self.mp = BondMessagePassing(
            d_h=self.d_h,
            d_e=self.d_e,
            d_v=self.d_v,
            depth=self.depth,
            dropout=self.dropout,
            undirected=self.undirected,
        )
        self.model = MPNN(
            message_passing=self.mp,
            agg=self.aggregation_method,
            predictor=RegressionFFN(
                input_dim=self.d_h, hidden_dim=self.d_h, n_tasks=self.output_dim
            ),
        )
        return self.model

    def forward(self, batch_mol_graph: BatchMolGraph):
        """Runs the forward pass, initializing the model if needed."""
        if self.model is None:
            self.initialize_model(batch_mol_graph)
        return self.model(batch_mol_graph)


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

idx = train_dataset[0]
batch_mol = idx[1][0]
print(batch_mol)
mpnn = ConfiguredMPNN(
    output_dim=7,
    aggregation_method=NormAggregation(),
    d_h=mpnn_hidden_dim,
    depth=mpnn_depth,
    dropout=mpnn_dropout,
    undirected=True,
)

mpnn.initialize_model(batch_mol)

hyperparams = {
    "batch_size": 32,
    "lr": 5e-4,
    "weight_decay": 1e-5,
    "log_selection_tensor": torch.tensor([1, 1, 1, 0, 0, 1]),
    "epochs": 20,
}
trainer = MPNNTrainer(
    model=mpnn,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    hyperparams=hyperparams,
    log_dir="logs/pretraining",
    save_results_dir="results/pretraining/",
    track_learning_curve=True,
)

trainer.train(20)


# torch.save(mpnn.state_dict(), "mpnn_pure.pth")
