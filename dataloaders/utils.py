import torch


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
