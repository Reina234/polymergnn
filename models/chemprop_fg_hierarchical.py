import torch
import torch.nn as nn
from typing import List, Optional, Any
from chemprop.data import BatchMolGraph
from chemprop.nn import BondMessagePassing
from tools.utils import nt_xent_loss
from models.modules.hierarchical_base import HierarchicalFGModel
from config.data_models import IFG


class ChemPropFGHierarchicalModel(nn.Module):
    """
    End-to-end model.
    1. The ChemProp MPNN produces a tensor of atom embeddings and a batch vector.
    2. We split the atom embeddings by molecule (using BatchMolGraph.batch).
    3. For each molecule, we obtain FG detections (via FunctionalGroupDetector).
    4. The HierarchicalFGModel computes a final embedding per molecule.
    5. Optionally, two augmented views (via dropout) provide a contrastive signal on FG embeddings.
    6. A regression head maps the final embedding to a target.
    """

    def __init__(
        self,
        mpnn_dim: int,
        fg_dim: int,
        global_dim: int,
        final_dim: int,
        st_heads: int,
        st_layers: int,
        temperature: float,
        dropout_prob: float,
        contrastive=True,
        additional_features_dim=0,
        target_dim: int = 1,
    ):
        # NEED TO INTEGRATE ADDITIONAL FEATURES - this can just be concatinated onto reg, head
        # but then reg head dims would be final_dims + additional_features_dims
        super(ChemPropFGHierarchicalModel, self).__init__()
        self.mpnn = BondMessagePassing(d_h=mpnn_dim)
        self.contrastive = contrastive
        self.temperature = temperature
        self.hierarchical_fg_model = HierarchicalFGModel(
            mpnn_dim=mpnn_dim,
            fg_dim=fg_dim,
            global_dim=global_dim,
            out_dim=final_dim,
            st_heads=st_heads,
            st_layers=st_layers,
        )
        self.regression_head = nn.Linear(
            final_dim + additional_features_dim, target_dim
        )
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(
        self,
        batch_molgraph: BatchMolGraph,
        fg_list: List[List[IFG]],
        additional_features: Optional[List[Any]] = None,
    ):
        f_atoms = self.mpnn(batch_molgraph)
        batch_vec = batch_molgraph.batch
        unique_ids = torch.unique(batch_vec)
        monomer_embeds = []
        contrast_losses = []
        for mon_id in unique_ids:
            idxs = (batch_vec == mon_id).nonzero(as_tuple=False).squeeze()
            atom_feats = f_atoms[idxs]  # [num_atoms, ATOM_DIM] for this molecule
            fg_list_per_mol = fg_list[mon_id.item()]  # FG list for this molecule
            # Generate two augmented views for contrastive learning.
            view1 = self.dropout(atom_feats)
            view2 = self.dropout(atom_feats)
            final_embed1, fg_embeds1 = self.hierarchical_fg_model(
                view1, fg_list_per_mol
            )
            _, fg_embeds2 = self.hierarchical_fg_model(view2, fg_list_per_mol)
            if self.contrastive and fg_embeds1.size(0) > 0 and fg_embeds2.size(0) > 0:
                cl_loss = nt_xent_loss(
                    fg_embeds1, fg_embeds2, temperature=self.temperature
                )
            else:
                cl_loss = torch.tensor(0.0, device=f_atoms.device)
            contrast_losses.append(cl_loss)
            monomer_embeds.append(final_embed1)
        if additional_features is not None:
            additional_features = torch.tensor(
                additional_features, dtype=torch.float32, device=f_atoms.device
            )
            final_embeddings = torch.cat(
                [final_embeddings, additional_features], dim=1
            )  # 🔥 Fix: Concatenate on dim=1

        final_embeddings = torch.cat(monomer_embeds, dim=0)  # [B, FINAL_DIM]
        regression_output = self.regression_head(
            final_embeddings
        )  # should be [B, 1], is returning [1, B] for some reason

        # regression_output = regression_output.view(-1, 1)  # fix shape issue
        if self.contrastive:
            contrastive_loss = torch.stack(contrast_losses).mean()
        else:
            contrastive_loss = torch.tensor(0.0, device=f_atoms.device)
        return regression_output, contrastive_loss, final_embeddings
