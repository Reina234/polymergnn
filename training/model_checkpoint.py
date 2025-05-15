import torch
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelCheckpoint:
    def __init__(self, save_dir="checkpoints"):
        os.makedirs(save_dir, exist_ok=True)
        self.save_dir = save_dir

    def save(self, model, epoch):
        save_path = os.path.join(self.save_dir, f"model_epoch_{epoch}.pth")
        torch.save(model.state_dict(), save_path)
        logger.info("Model saved at %s", save_path)

    def load(self, model, checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path))
        logger.info("Loaded model from %s", checkpoint_path)
        return model
