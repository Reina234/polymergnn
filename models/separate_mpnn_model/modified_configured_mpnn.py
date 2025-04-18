import torch.nn as nn
from chemprop.nn import BondMessagePassing, AttentiveAggregation, RegressionFFN
from chemprop.models.model import MPNN
from chemprop.data import BatchMolGraph


class AttentiveConfiguredMPNN(nn.Module):
    def __init__(
        self,
        output_dim: int,
        d_h: int = 300,
        depth: int = 3,
        dropout: float = 0.0,
        undirected: bool = True,
    ):
        super().__init__()
        self.output_dim = output_dim
        self.aggregation_method = AttentiveAggregation(output_size=d_h)
        self.d_h = d_h
        self.depth = depth
        self.dropout = dropout
        self.undirected = undirected

        self.mp = None  # Will be initialized during the first forward pass
        self.model = None
        self.d_e = None
        self.d_v = None

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
