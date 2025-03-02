import torch.nn as nn
from chemprop.nn import BondMessagePassing, NormAggregation, RegressionFFN
from chemprop.models.model import MPNN


class ShapMPNN(nn.Module):
    def __init__(
        self,
        d_e: int,
        d_v: int,
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

        self.mp = BondMessagePassing(
            d_h=self.d_h,
            d_e=d_e,
            d_v=d_v,
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

    def forward(self, batch):
        batch_mol_graph = batch["batch_mol_graph"]
        return self.model(batch_mol_graph)
