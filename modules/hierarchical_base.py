import torch.nn as nn
from modules.fg_pooling import FGPooling
from modules.fusion_module import FusionModule
from modules.global_pooling import GlobalPooling
from modules.set_transformer import SetTransformerPooling


class HierarchicalFGModel(nn.Module):
    def __init__(
        self,
        mpnn_dim: int,
        fg_dim: int,
        global_dim: int,
        out_dim: int,
        st_heads: int,
        st_layers: int,
    ):
        super(HierarchicalFGModel, self).__init__()
        self.global_pool = GlobalPooling(in_dim=mpnn_dim, global_dim=global_dim)
        self.fg_pool = FGPooling(atom_dim=mpnn_dim, fg_dim=fg_dim)
        self.set_transformer = SetTransformerPooling(
            fg_dim=fg_dim, num_heads=st_heads, num_layers=st_layers
        )
        self.fusion = FusionModule(
            global_dim=global_dim, fg_dim=fg_dim, out_dim=out_dim
        )

    def forward(self, atom_feats, fg_list):
        # Global embedding from all atom features.
        global_embed = self.global_pool(atom_feats)  # [1, FINAL_DIM]
        # FG branch: compute per-FG embeddings.
        fg_embeds = self.fg_pool(atom_feats, fg_list)  # [num_FGs, FG_DIM]
        # Aggregate FG embeddings via set transformer pooling.
        fg_agg = self.set_transformer(fg_embeds)  # [1, FG_DIM] or None
        # Fuse global and aggregated FG embeddings.
        final_embed = self.fusion(global_embed, fg_agg)  # [1, FINAL_DIM]
        return final_embed, fg_embeds
