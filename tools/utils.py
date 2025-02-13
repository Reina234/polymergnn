import torch
import torch.nn.functional as F


def nt_xent_loss(z1, z2, temperature):
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    sim_matrix = torch.matmul(z1, z2.t()) / temperature
    positives = torch.diag(sim_matrix)
    loss = -torch.log(torch.exp(positives) / torch.sum(torch.exp(sim_matrix), dim=1))
    return loss.mean()
