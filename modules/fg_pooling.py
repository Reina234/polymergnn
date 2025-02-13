import torch
import torch.nn as nn
from config.data_models import IFG


class FGPooling(nn.Module):
    """
    For each detected FG, pools the core atom features (using the indices in ifg.atomIds)
    and pools the environment atom features (using ifg.environment, if provided).
    The two pooled vectors are concatenated and passed through an MLP to yield a FG embedding.
    We use scatter_mean for simple, permutation-invariant pooling.
    """

    def __init__(self, atom_dim: int, fg_dim: int):
        super(FGPooling, self).__init__()
        # Concatenated core and environment vectors: dimension = 2 * atom_dim.
        self.mlp = nn.Sequential(
            nn.Linear(atom_dim * 2, fg_dim), nn.ReLU(), nn.Linear(fg_dim, fg_dim)
        )
        self.fg_dim = fg_dim
        self.atom_dim = atom_dim

    def forward(self, atom_feats, fg_list: IFG):
        fg_embeddings = []
        for ifg in fg_list:
            # Pool core atoms.
            core_indices = list(ifg.atomIds)
            core_emb = atom_feats[core_indices].mean(dim=0)
            # Pool environment atoms (if provided and valid).
            if (
                ifg.envIds
                and (len(ifg.envIds) > 0)
                and (max(ifg.envIds) < atom_feats.size(0))
            ):
                env_indices = list(ifg.envIds)
                env_emb = atom_feats[env_indices].mean(dim=0)
            else:
                env_emb = torch.zeros_like(core_emb)
            # Concatenate core and environment.
            combined = torch.cat([core_emb, env_emb], dim=0)
            fg_embeddings.append(self.mlp(combined))
        if len(fg_embeddings) == 0:
            # If no FGs detected, return an empty tensor.
            return torch.zeros(0, self.fg_dim, device=atom_feats.device)
        return torch.stack(fg_embeddings, dim=0)  # Shape: [num_FGs, FG_DIM]
