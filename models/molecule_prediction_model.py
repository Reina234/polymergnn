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


class MultiHeadMoleculePredictionModel(nn.Module):
    def __init__(
        self,
        embedding_model: MoleculeEmbeddingModel,
        output_dim: int = 1,
        hidden_dim: int = 128,
    ):
        """
        Multi-task version of MoleculePredictionModel.
        Each output dimension is predicted by its own FNN head.

        Args:
            embedding_model: Shared MoleculeEmbeddingModel (e.g., MPNN + ChemBERTa).
            output_dim: Number of tasks (each task gets its own head).
            hidden_dim: Hidden dimension for each head.
        """
        super().__init__()
        self.embedding_model = embedding_model

        # Create a head for each task
        self.task_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(embedding_model.hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),  # 1 output per head
                )
                for _ in range(output_dim)
            ]
        )

    def forward(self, batch):
        """Generates embeddings and returns predictions from all heads."""
        embeddings = self.embedding_model(batch)  # [batch_size, hidden_dim]

        # Each head produces one output
        predictions = [head(embeddings) for head in self.task_heads]

        # Concatenate results: [batch_size, output_dim]
        predictions = torch.cat(predictions, dim=1)

        return predictions
