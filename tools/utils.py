import torch
import torch.nn.functional as F


def stack_tensors(tensor_list, convert_to_tensor=True):

    if not tensor_list:
        return None

    if not convert_to_tensor and any(t is None for t in tensor_list):
        return None  # Return None immediately if any None is found

    tensor_list = [
        torch.tensor(0.0, dtype=torch.float32) if t is None else t for t in tensor_list
    ]

    if convert_to_tensor:
        tensor_list = [
            (
                torch.tensor(t, dtype=torch.float32)
                if not isinstance(t, torch.Tensor)
                else t
            )
            for t in tensor_list
        ]

    return torch.stack(tensor_list, dim=0) if tensor_list else None


def nt_xent_loss(z1, z2, temperature):
    z1 = F.normalize(z1, p=2, dim=1)
    z2 = F.normalize(z2, p=2, dim=1)
    sim_matrix = torch.matmul(z1, z2.t()) / temperature
    positives = torch.diag(sim_matrix)
    loss = -torch.log(torch.exp(positives) / torch.sum(torch.exp(sim_matrix), dim=1))
    return loss.mean()
