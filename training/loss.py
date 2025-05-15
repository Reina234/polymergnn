import torch.nn as nn


class LossStrategy:
    def compute_loss(self, predictions, targets):
        raise NotImplementedError


class MSELossStrategy(LossStrategy):

    def __init__(self):
        self.criterion = nn.MSELoss()

    def compute_loss(self, predictions, targets):
        return self.criterion(predictions, targets)


class MSEWithContrastiveLoss(LossStrategy):
    def __init__(self):
        self.criterion = nn.MSELoss()

    def compute_loss(self, regression_output, contrastive_loss, targets):
        mse_loss = self.criterion(regression_output, targets)
        total_loss = mse_loss + contrastive_loss
        return total_loss, mse_loss, contrastive_loss
