import torch.nn as nn


class SetTransformerPooling(nn.Module):
    """
    Aggregates a set of FG embeddings into a single vector using a lightweight set transformer.
    Given the typically small number of FGs per monomer, a single-layer transformer is used.
    """

    def __init__(self, fg_dim: int, num_heads: int, num_layers: int):
        super(SetTransformerPooling, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=fg_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, fg_embeds):
        # fg_embeds: [num_FGs, FG_DIM] -> add batch dim: [1, num_FGs, FG_DIM]
        if fg_embeds.shape[0] == 0:
            return None
        x = self.transformer(fg_embeds.unsqueeze(0))
        # x: [1, num_FGs, FG_DIM]; perform pooling over FG dimension.
        x = self.pool(x.transpose(1, 2)).squeeze(2)  # [1, FG_DIM]
        return x  # Aggregated FG representation
