import torch.nn as nn


class GlobalPooling(nn.Module):
    """
    Computes a global embedding for a monomer by mean pooling atom features
    and then applying a linear projection.
    """

    def __init__(self, in_dim: int, global_dim: int):
        super(GlobalPooling, self).__init__()
        self.fc = nn.Linear(in_dim, global_dim)

    def forward(self, atom_feats):
        pooled = atom_feats.mean(dim=0, keepdim=True)
        return self.fc(pooled)
