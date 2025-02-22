import torch
from sklearn.pipeline import Pipeline
import numpy as np
from typing import List


class LogTransformHelper:

    def __init__(
        self, log_selection_tensor: torch.tensor, feature_transformers: List[Pipeline]
    ):

        self.feature_transformers = feature_transformers
        self.num_targets = len(self.feature_transformers) / 2

        if log_selection_tensor.shape[0] != self.num_targets:
            raise ValueError(
                f"log_selection_tensor should have length {self.num_targets}, but got {log_selection_tensor.shape[0]}."
            )

        self.log_selection_tensor = log_selection_tensor.cpu().numpy()
        self.selected_transforms = self._select_transforms()

    def _select_transforms(self) -> List[Pipeline]:

        selected_transforms = []
        for i, log_flag in enumerate(self.log_selection_tensor):
            if log_flag:
                selected_transforms.append(
                    self.feature_transformers[i + self.num_targets]
                )
            else:
                selected_transforms.append(
                    self.feature_transformers[i]
                )  # Use normal inverse
        return selected_transforms

    def inverse_transform(self, predictions: torch.tensor) -> torch.tensor:
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)

        transformed_data = np.zeros_like(predictions)

        for i, transform in enumerate(self.selected_transforms):
            transformed_data[:, i] = transform.inverse_transform(
                predictions[:, i].reshape(-1, 1)
            ).flatten()

        transformed_data = torch.tensor(transformed_data).float()

        return transformed_data

    def filter_target_labels(self, labels: torch.Tensor) -> torch.Tensor:

        if self.log_selection_tensor.shape[0] != self.num_targets:
            raise ValueError(
                f"log_selection_tensor should have length {self.num_targets}, but got {self.log_selection_tensor.shape[0]}."
            )

        selected_indices = torch.arange(self.num_targets) + (
            self.log_selection_tensor * self.num_targets
        )
        return labels[:, selected_indices]
