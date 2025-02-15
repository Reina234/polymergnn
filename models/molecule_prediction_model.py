import torch
import torch.nn as nn
from models.molecule_embedding_model import MoleculeEmbeddingModel


class MoleculePredictionModel(nn.Module):
    def __init__(
        self,
        embedding_model: MoleculeEmbeddingModel,
        output_dim: int = 1,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.embedding_model = embedding_model
        self.output_layer = nn.Sequential(
            nn.Linear(embedding_model.hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, batch):
        """Generate embeddings and predict."""
        embeddings = self.embedding_model(batch)  # Shape: [batch_size, hidden_dim]
        predictions = self.output_layer(embeddings)  # Shape: [batch_size, output_dim]
        return predictions
