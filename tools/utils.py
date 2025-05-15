import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import numpy as np
from typing import Optional


def log_transform(X):
    return np.log(X + 1e-8)


def inverse_log_transform(X):
    return np.exp(X) - 1e-8


def create_column_transformer(
    feature_indices: list,
    target_indices: list,
    num_original_columns: Optional[int] = None,
    feature_transform=StandardScaler(),
    target_transform=StandardScaler(),
    log_target_transform=StandardScaler(),
):
    if not num_original_columns:
        num_original_columns = max(feature_indices + target_indices) + 1

    transformers = []
    column_order = []  
    for i in feature_indices:
        name = f"feature_{i}"
        transformers.append((name, feature_transform, [i]))
        column_order.append(name)


    for i, index in enumerate(target_indices):
        name = f"target_{i}"
        transformers.append((name, target_transform, [index]))
        column_order.append(name)

    for i, index in enumerate(target_indices):
        log_index = num_original_columns + target_indices.index(index)  # Offset index

        name = f"log_{i}"
        log_pipeline = Pipeline(
            [
                (
                    "log_transform",
                    FunctionTransformer(
                        log_transform, inverse_func=inverse_log_transform
                    ),
                ),
                ("scaler", log_target_transform),
            ]
        )
        transformers.append((name, log_pipeline, [log_index]))
        column_order.append(name)

    column_transformer = ColumnTransformer(transformers)

    return column_transformer


def stack_tensors(tensor_list, convert_to_tensor=True):

    if not tensor_list:
        return None

    if not convert_to_tensor and any(t is None for t in tensor_list):
        return None 

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
