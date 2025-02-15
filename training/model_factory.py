import torch
from abc import ABC, abstractmethod
from modules.configured_mpnn import ConfiguredMPNN
from models.molecule_embedding_model import MoleculeEmbeddingModel
from featurisers.molecule_featuriser import RDKitFeaturizer
from models.molecule_prediction_model import MoleculePredictionModel


class ModelFactory(ABC):
    """Abstract base class for model factories."""

    @abstractmethod
    def create_model(self, params: dict) -> torch.nn.Module:
        pass

    @abstractmethod
    def create_optimizer(
        self, model: torch.nn.Module, params: dict
    ) -> torch.optim.Optimizer:
        pass


class MoleculeEmbeddingModelFactory(ModelFactory):
    """Factory for creating MoleculeEmbeddingModel instances."""

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

    def create_model(self, params):
        """Creates a MoleculeEmbeddingModel based on hyperparameters."""
        mpnn = ConfiguredMPNN(
            output_dim=self.default_hidden_dim,
            d_h=params.get("d_h", 300),
            depth=params.get("depth", 3),
            dropout=params.get("dropout", 0.0),
            undirected=True,
        )

        rdkit_featurizer = RDKitFeaturizer()

        model = MoleculeEmbeddingModel(
            chemprop_mpnn=mpnn,
            rdkit_featurizer=rdkit_featurizer,
            selected_rdkit_features=params.get(
                "selected_rdkit_features", self.default_rdkit_features
            ),
            chemberta_dim=params.get("chemberta_dim", self.default_chemberta_dim),
            hidden_dim=params.get("hidden_dim", self.default_hidden_dim),
        )

        return model

    def create_optimizer(self, model, params):
        """Creates an optimizer for the MoleculeEmbeddingModel."""
        return self.default_optimizer_class(
            model.parameters(),
            lr=params.get("lr", 0.001),
            weight_decay=params.get("weight_decay", 0.0),
        )


class MoleculeEmbeddingPredictionModelFactory(ModelFactory):
    """Factory for creating MoleculePredictionModel instances."""

    def __init__(
        self,
        output_dim: int,  # From dataloader
        default_rdkit_features=None,
        default_chemberta_dim=600,
        default_hidden_dim=256,
        default_optimizer_class=torch.optim.Adam,
    ):
        self.output_dim = output_dim
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
        """Creates a MoleculePredictionModel with MoleculeEmbeddingModel."""
        # 1) MPNN Configuration
        mpnn = ConfiguredMPNN(
            output_dim=self.default_hidden_dim,
            d_h=params.get("d_h", 300),
            depth=params.get("depth", 3),
            dropout=params.get("dropout", 0.0),
            undirected=True,
        )

        # 2) Embedding Model (Base)
        rdkit_featurizer = RDKitFeaturizer()
        embedding_model = MoleculeEmbeddingModel(
            chemprop_mpnn=mpnn,
            rdkit_featurizer=rdkit_featurizer,
            selected_rdkit_features=params.get(
                "selected_rdkit_features", self.default_rdkit_features
            ),
            chemberta_dim=params.get("chemberta_dim", self.default_chemberta_dim),
            hidden_dim=params.get("hidden_dim", self.default_hidden_dim),
        )

        # 3) Prediction Model (with FNN)
        model = MoleculePredictionModel(
            embedding_model=embedding_model,
            output_dim=self.output_dim,  # From dataloader
            hidden_dim=params.get("hidden_dim", self.default_hidden_dim),
        )

        return model

    def create_optimizer(self, model, params):
        """Creates an optimizer for the MoleculePredictionModel."""
        return self.default_optimizer_class(
            model.parameters(),
            lr=params.get("lr", 0.001),
            weight_decay=params.get("weight_decay", 0.0),
        )
