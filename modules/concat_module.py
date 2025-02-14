import torch
import torch.nn as nn


# Inside FusionModule
class ConcatModule(nn.Module):
    def __init__(self, global_dim: int, fg_dim: int, out_dim: int):
        super(ConcatModule, self).__init__()
        self.fc = nn.Linear(
            global_dim + fg_dim, out_dim
        )  # Simple concatenation-based fusion

    def forward(self, global_embed, fg_agg):
        if fg_agg is None:
            fg_agg = torch.zeros_like(global_embed)
        fused = torch.cat([global_embed, fg_agg], dim=1)
        return self.fc(fused)
