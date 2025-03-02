import torch.nn as nn
from models.molecule_embedding_model import MoleculeEmbeddingModel


class PretrainingWrapper(nn.Module):

    def __init__(
        self,
        embedding_model: MoleculeEmbeddingModel,
        output_dim: int,
        embedding_dim: int = 256,
        fnn_hidden_dim: int = 256,
    ):
        super().__init__()

        # Select RDKit Features

        self.embedding_model = embedding_model

        self.output_layer = nn.Sequential(
            nn.Linear(embedding_dim, fnn_hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(fnn_hidden_dim, fnn_hidden_dim),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(fnn_hidden_dim, output_dim),
        )

    def forward(self, batch):
        embeddings, _, _, _ = self.embedding_model(batch)  # MPNN generates embeddings
        predictions = self.output_layer(embeddings)  # FNN head makes predictions
        return predictions
