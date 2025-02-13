import torch
import torch.nn as nn


class FusionModule(nn.Module):
    """
    Fuses the global (monomer) embedding with the aggregated FG embedding.
    We first project the FG embedding to match the global embedding dimension,
    then compute a learnable gating weight to fuse the two representations.
    Finally, we concatenate the fused representation with the original concatenation and project it.
    """

    def __init__(self, global_dim: int, fg_dim: int, out_dim: int):
        super(FusionModule, self).__init__()
        # Project FG embedding from fg_dim to global_dim
        self.fg_proj = nn.Linear(fg_dim, global_dim)
        self.gate = nn.Sequential(nn.Linear(global_dim + global_dim, 1), nn.Sigmoid())
        self.fc = nn.Linear(global_dim + (global_dim + global_dim), out_dim)

    def forward(self, global_embed, fg_agg):
        # global_embed: [1, global_dim]; fg_agg: [1, fg_dim] (or None)
        if fg_agg is None:
            fg_agg = torch.zeros_like(global_embed)
        else:
            fg_agg = self.fg_proj(fg_agg)  # Now fg_agg becomes [1, global_dim]
        concat = torch.cat([global_embed, fg_agg], dim=1)  # [1, global_dim * 2]
        weight = self.gate(concat)  # [1, 1] learned weight between 0 and 1
        fused = weight * global_embed + (1 - weight) * fg_agg  # [1, global_dim]
        return self.fc(torch.cat([fused, concat], dim=1))  # [1, out_dim]
