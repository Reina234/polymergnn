import torch
from sklearn.compose import ColumnTransformer
import numpy as np


class LogTransformHelper:

    def __init__(
        self,
        log_selection_tensor: torch.tensor,
        column_transformer: ColumnTransformer,
    ):
        self.log_selection_tensor = (
            log_selection_tensor.cpu().numpy()
        )  # Convert to numpy for indexing
        self.column_transformer = column_transformer
        self.selected_transforms = self._select_transforms()

    def _select_transforms(self):
        """Creates a list of the correct transform names based on `log_selection_tensor`."""
        selected_transforms = []
        for i, log_flag in enumerate(self.log_selection_tensor):
            if log_flag:
                selected_transforms.append(f"log_{i}")  # Use log-transformed inverse
            else:
                selected_transforms.append(f"target_{i}")  # Use normal inverse
        return selected_transforms

    def inverse_transform(self, predictions: torch.tensor):
        """
        Applies the correct inverse transformations based on `log_selection_tensor`.

        Args:
        - predictions (np.ndarray): Model predictions.

        Returns:
        - np.ndarray: Inverse transformed predictions.
        """
        if not isinstance(predictions, np.ndarray):
            predictions = np.array(predictions)

        transformed_data = np.zeros_like(predictions)

        for i, transform_name in enumerate(self.selected_transforms):
            inverse_pipeline = self.column_transformer.named_transformers_[
                transform_name
            ]
            transformed_data[:, i] = inverse_pipeline.inverse_transform(
                predictions[:, i].reshape(-1, 1)
            ).flatten()

        return transformed_data

    def filter_target_labels(self, log_selection_tensor: torch.Tensor) -> torch.Tensor:

        labels = self.log_selection_tensor
        _, num_total_labels = labels.shape  # num_total_labels = 2n
        n = num_total_labels // 2  # Get the number of distinct target types

        if log_selection_tensor.shape[0] != n:
            raise ValueError(
                f"log_selection_tensor should have length {n}, but got {log_selection_tensor.shape[0]}."
            )

        selected_indices = torch.arange(n) + (log_selection_tensor * n)

        filtered_labels = labels[:, selected_indices]

        return filtered_labels
