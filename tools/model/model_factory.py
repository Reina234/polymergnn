import torch.nn as nn


class ModelFactory:
    """
    Creates models dynamically based on hyperparameters.
    Supports ChemProp, MPNN, and other architectures.
    """

    def __init__(self, model_class: nn.Module):
        self.model_class = model_class

    def create_model(self, hyperparams, device):
        """Creates and returns a model instance with given hyperparameters."""
        model = self.model_class(**hyperparams)
        return model.to(device)
