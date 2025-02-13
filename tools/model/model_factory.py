import torch.nn as nn
from typing import List, Dict, Any


class ModelFactory:
    """
    Creates models dynamically based on hyperparameters and additional fixed arguments.
    """

    def __init__(
        self,
        model_class: nn.Module,
        model_param_keys: List[str],
        extra_args: Dict[str, Any] = None,
    ):
        """
        Args:
            model_class: The class of the model (e.g., ChemPropFGHierarchicalModel, MPNN, etc.).
            model_param_keys: List of hyperparameters that belong to the model (filters out lr).
            extra_args: Dictionary of fixed parameters required by the model.
        """
        self.model_class = model_class
        self.model_param_keys = model_param_keys
        self.extra_args = extra_args or {}  # 🔥 Default to empty dict if no extra args

    def create_model(self, hyperparams, device):
        """
        Creates and returns a model instance with filtered hyperparameters and extra arguments.
        """
        # Filter out only model-specific hyperparameters
        model_params = {
            k: hyperparams[k] for k in self.model_param_keys if k in hyperparams
        }

        # Merge with fixed extra arguments
        model_params.update(self.extra_args)  # 🔥 Inject additional arguments

        # Instantiate model
        model = self.model_class(**model_params).to(device)
        return model
