import torch
from abc import ABC, abstractmethod
from typing import Any, Dict
from models.modules.configured_mpnn import ConfiguredMPNN
from models.molecule_embedding_model import MoleculeEmbeddingModel
from featurisers.molecule_featuriser import RDKitFeaturizer
from models.molecule_prediction_model import (
    MoleculePredictionModel,
    MultiHeadMoleculePredictionModel,
)


class ModelFactory(ABC):
    @abstractmethod
    def create_model(self, params: dict) -> torch.nn.Module:
        pass

    @abstractmethod
    def create_optimizer(
        self, model: torch.nn.Module, params: dict
    ) -> torch.optim.Optimizer:
        pass


class MoleculeEmbeddingModelFactory(ABC):
    def __init__(
        self,
        default_rdkit_features=None,
        default_chemberta_dim=600,
        default_hidden_dim=256,
        default_optimizer_class=torch.optim.Adam,
    ):
        self.default_rdkit_features = default_rdkit_features or [
            "MolWt",
            "MolLogP",
            "MolMR",
            "TPSA",
            "NumRotatableBonds",
            "RingCount",
            "FractionCSP3",
        ]
        self.default_chemberta_dim = default_chemberta_dim
        self.default_hidden_dim = default_hidden_dim
        self.default_optimizer_class = default_optimizer_class

    def create_model(self, params: Dict[str, Any]):
        mpnn = ConfiguredMPNN(
            output_dim=self.default_hidden_dim,
            d_h=params.get("d_h", 300),
            depth=params.get("depth", 3),
            dropout=params.get("dropout", 0.0),
            undirected=True,
        )

        rdkit_featurizer = RDKitFeaturizer()

        # Optional components controlled by hyperparameters.
        use_rdkit = params.get("use_rdkit", True)
        use_chembert = params.get("use_chembert", True)

        model = MoleculeEmbeddingModel(
            chemprop_mpnn=mpnn,
            rdkit_featurizer=rdkit_featurizer,
            selected_rdkit_features=params.get(
                "selected_rdkit_features", self.default_rdkit_features
            ),
            chemberta_dim=params.get("chemberta_dim", self.default_chemberta_dim),
            hidden_dim=params.get("hidden_dim", self.default_hidden_dim),
            use_rdkit=use_rdkit,
            use_chembert=use_chembert,
        )
        return model

    def create_optimizer(self, model, params):
        return self.default_optimizer_class(
            model.parameters(),
            lr=params.get("lr", 0.001),
            weight_decay=params.get("weight_decay", 0.0),
        )


class MoleculeEmbeddingPredictionModelFactory(ABC):
    """Factory for creating MoleculePredictionModel instances."""

    def __init__(
        self,
        output_dim: int,
        multi_head: bool = False,
        default_rdkit_features=None,
        default_chemberta_dim=600,
        default_hidden_dim=256,
        default_optimizer_class=torch.optim.Adam,
    ):
        self.output_dim = output_dim
        self.multi_head = multi_head
        self.default_rdkit_features = default_rdkit_features or [
            "MolWt",
            "MolLogP",
            "MolMR",
            "TPSA",
            "NumRotatableBonds",
            "RingCount",
            "FractionCSP3",
        ]
        self.default_chemberta_dim = default_chemberta_dim
        self.default_hidden_dim = default_hidden_dim
        self.default_optimizer_class = default_optimizer_class

    def create_model(self, params):
        mpnn = ConfiguredMPNN(
            output_dim=self.default_hidden_dim,
            d_h=params.get("d_h", 300),
            depth=params.get("depth", 3),
            dropout=params.get("dropout", 0.0),
            undirected=True,
        )

        rdkit_featurizer = RDKitFeaturizer()

        use_rdkit = params.get("use_rdkit", True)
        use_chembert = params.get("use_chembert", True)

        embedding_model = MoleculeEmbeddingModel(
            chemprop_mpnn=mpnn,
            rdkit_featurizer=rdkit_featurizer,
            selected_rdkit_features=params.get(
                "selected_rdkit_features", self.default_rdkit_features
            ),
            chemberta_dim=params.get("chemberta_dim", self.default_chemberta_dim),
            hidden_dim=params.get("hidden_dim", self.default_hidden_dim),
            use_rdkit=use_rdkit,
            use_chembert=use_chembert,
        )
        if self.output_dim == 1 or not self.multi_head:
            model = MoleculePredictionModel(
                embedding_model=embedding_model,
                output_dim=self.output_dim,
                hidden_dim=params.get("hidden_dim", self.default_hidden_dim),
            )
        else:
            model = MultiHeadMoleculePredictionModel(
                embedding_model=embedding_model,
                output_dim=self.output_dim,
                hidden_dim=params.get("hidden_dim", self.default_hidden_dim),
            )
        return model

    def create_optimizer(self, model, params):
        return self.default_optimizer_class(
            model.parameters(),
            lr=params.get("lr", 0.001),
            weight_decay=params.get("weight_decay", 0.0),
        )
