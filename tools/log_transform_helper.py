import torch
from sklearn.pipeline import Pipeline
import numpy as np
from typing import List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LogTransformHelper:

    def __init__(
        self, log_selection_tensor: torch.tensor, target_transformers: List[Pipeline]
    ):
        log_selection_tensor = log_selection_tensor.cpu().numpy().flatten()
        self.target_transformers = target_transformers
        self.num_targets = int(len(self.target_transformers) / 2)

        if log_selection_tensor.shape[0] != self.num_targets:
            raise ValueError(
                f"log_selection_tensor should have length {self.num_targets}, but got {log_selection_tensor.shape[0]}."
            )

        self.log_selection_tensor = log_selection_tensor
        self.selected_transforms = self._select_transforms()

    def _select_transforms(self) -> List[Pipeline]:

        selected_transforms = []
        for i, log_flag in enumerate(self.log_selection_tensor):
            if log_flag:
                selected_transforms.append(
                    self.target_transformers[i + self.num_targets]
                )
            else:
                selected_transforms.append(
                    self.target_transformers[i]
                )  # Use normal inverse
        return selected_transforms

    def inverse_transform(self, values_to_transform: torch.tensor) -> torch.tensor:
        if not isinstance(values_to_transform, np.ndarray):
            values_to_transform = np.array(values_to_transform)

        transformed_data = np.zeros_like(values_to_transform)

        for i, transform in enumerate(self.selected_transforms):
            if not transform:
                transformed_data[:, i] = values_to_transform[:, i]
                logger.info("Skipping transform for target %d", i)
                continue
            transformed_data[:, i] = transform.inverse_transform(
                values_to_transform[:, i].reshape(-1, 1)
            ).flatten()

        transformed_data = torch.tensor(transformed_data).float()

        return transformed_data

    def filter_target_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """
        Filters the correct target labels based on the log selection tensor.
        If a target is log-transformed, it selects the corresponding value from
        the second half of the labels (2N columns total).

        Args:
            labels (torch.Tensor): A tensor of shape (batch_size, 2N).

        Returns:
            torch.Tensor: A tensor of shape (batch_size, N), correctly selecting
                        log-transformed or regular values.
        """
        num_targets = self.num_targets  # N
        _, num_cols = labels.shape

        if num_cols != 2 * num_targets:
            raise ValueError(
                f"Expected labels to have {2 * num_targets} columns, got {num_cols}."
            )

        # Ensure log_selection_tensor is a PyTorch tensor and on the same device
        log_selection_tensor = torch.as_tensor(
            self.log_selection_tensor, dtype=torch.bool, device=labels.device
        )

        normal_indices = torch.arange(num_targets, device=labels.device)
        log_indices = normal_indices + num_targets  # Shift by N

        # Ensure correct dtype and shape for indexing
        selected_indices = torch.where(
            log_selection_tensor,  # Condition (bool tensor)
            log_indices,  # If True, use log-transformed index
            normal_indices,  # If False, use normal index
        )

        # Ensure indexing works correctly
        return labels[:, selected_indices.view(-1)]  # Shape: (batch_size, N)
