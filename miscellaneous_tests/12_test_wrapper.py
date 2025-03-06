from chemprop.data import BatchMolGraph
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
from training.refactored_batched_dataset import PolymerGNNDataset
from tools.smiles_transformers import NoSmilesTransform
from training.refactored_trainer import MPNNTrainer


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
target_columns = [1, 2]
monomer_smiles_column = 0
feature_columns = None

df = pd.read_csv("data/freesolv/FreeSolv_SAMPL.csv")
df = df.drop(df.columns[[0, 1]], axis=1)

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

idx = train_dataset[0]
batch_mol = idx[1][0]
from collections import namedtuple

MolGraph = namedtuple("MolGraph", ["V", "E", "edge_index", "rev_edge_index"])


def create_batch_molgraph(V, E, edge_index, rev_edge_index):
    mol_graph = MolGraph(V=V, E=E, edge_index=edge_index, rev_edge_index=rev_edge_index)
    # Convert single `MolGraph` into `BatchMolGraph`
    batch_mol_graph = BatchMolGraph([mol_graph])

    return batch_mol_graph


test = create_batch_molgraph(
    batch_mol.V, batch_mol.E, batch_mol.edge_index, batch_mol.rev_edge_index
)


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
from training.refactored_batched_dataset import PolymerGNNDataset
from tools.smiles_transformers import NoSmilesTransform
from training.refactored_trainer import MPNNTrainer


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
target_columns = [1, 2]
monomer_smiles_column = 0
feature_columns = None

df = pd.read_csv("data/freesolv/FreeSolv_SAMPL.csv")
df = df.drop(df.columns[[0, 1]], axis=1)

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


mpnn_output_dim = 256
mpnn_hidden_dim = 500
mpnn_depth = 4
mpnn_dropout = 0.1
molecule_embedding_hidden_dim = 256
embedding_dim = 256
use_rdkit = True
use_chembert = False


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

mpnn = ConfiguredMPNN(
    output_dim=2,
    aggregation_method=NormAggregation(),
    d_h=mpnn_hidden_dim,
    depth=mpnn_depth,
    dropout=mpnn_dropout,
    undirected=True,
)

mpnn.initialize_model(batch_mol)


pretrained_dict = torch.load("mpnn_pure.pth", weights_only=True)
model_dict = mpnn.state_dict()

mpnn.eval()


class ChempropWrapper(torch.nn.Module):
    """
    Wrapper to make Chemprop's MPNN compatible with PyG's GNNExplainer.
    """

    def __init__(self, chemprop_mpnn):
        super().__init__()
        self.chemprop_mpnn = chemprop_mpnn  # Chemprop's trained MPNN

    def forward(self, V, edge_index, E, rev_edge_index):
        """
        Converts PyG-style input into Chemprop BatchMolGraph.
        Args:
            x (torch.Tensor): Node features
            edge_index (torch.Tensor): Edge indices
        Returns:
            Model output (predicted property)
        """

        batch_mol_graph = create_batch_molgraph(
            V.detach(), E, edge_index, rev_edge_index
        )  # Convert PyG graph
        return self.chemprop_mpnn(batch_mol_graph)  # Run Chemprop MPNN


model = ChempropWrapper(mpnn)

print(mpnn(test))

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
from torch_geometric.explain.algorithm import GNNExplainer
from torch_geometric.explain import Explainer, ModelConfig

algorithm = GNNExplainer(epochs=100)

config = ModelConfig("regression", "graph")
explainer = Explainer(
    model, algorithm, "phenomenon", config, node_mask_type="attributes"
)

# Convert node features (V) and edge features (E) to float
V_tensor = torch.tensor(batch_mol.V, dtype=torch.float32, device=device)
E_tensor = torch.tensor(batch_mol.E, dtype=torch.float32, device=device)

# Convert edge_index and rev_edge_index to long (they store indices)
edge_index_tensor = torch.tensor(batch_mol.edge_index, dtype=torch.long, device=device)
rev_edge_index_tensor = torch.tensor(
    batch_mol.rev_edge_index, dtype=torch.long, device=device
)

# Convert target to float
target_tensor = torch.tensor(idx[6], dtype=torch.float32, device=device)

# Run GNNExplainer with the corrected inputs
explainer(
    V_tensor,
    edge_index_tensor,
    E=E_tensor,
    rev_edge_index=rev_edge_index_tensor,
    target=target_tensor,  # Ensure `target` is also a tensor if needed
)

# explainer(model=embedding_model, x=batch_mol.V, edge_index=batch_mol.E, target=idx[5])

# get_masked_prediction(batch_mol.V, edge_index=batch_mol.E, node_mask: Optional[Union[Tensor, Dict[str, Tensor]]] = None, edge_mask: Optional[Union[Tensor, Dict[Tuple[str, str, str], Tensor]]] = None, **kwargs)
