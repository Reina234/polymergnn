import logging.config
import torch
import os
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenericTrainer:
    def __init__(
        self,
        model,
        optimizer,
        loss_strategy,
        train_loader,
        val_loader,
        test_loader,
        device,
        log_dir="logs",
        use_tensorboard=True,
        save_results_dir="results",
    ):
        """
        Generic trainer that supports different models, loss functions, and data types.
        """
        self.model = model
        self.optimizer = optimizer
        self.loss_strategy = loss_strategy
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.save_results_dir = save_results_dir

        os.makedirs(save_results_dir, exist_ok=True)

        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir)

    def forward_pass(self, batch):
        """Handles different models by checking expected input signatures."""
        batch_molgraph, _, batch_fg_list, _, batch_targets = batch
        batch_targets = batch_targets.to(self.device)

        if hasattr(self.model, "contrastive"):  # ChemPropFGHierarchicalModel
            regression_output, contrastive_loss, _ = self.model(
                batch_molgraph, batch_fg_list
            )
            return regression_output, contrastive_loss, batch_targets

        elif hasattr(self.model, "message_passing"):  # MPNN
            regression_output = self.model(batch_molgraph)
            return regression_output, None, batch_targets

        else:  # Standard feedforward model
            regression_output = self.model(batch_molgraph)
            return regression_output, None, batch_targets

    def train(self, epochs=50):
        self.model.train()
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            total_loss, total_mse_loss, total_cl_loss = 0.0, 0.0, 0.0

            for batch in self.train_loader:
                self.optimizer.zero_grad()

                regression_output, contrastive_loss, batch_targets = self.forward_pass(
                    batch
                )

                # Compute loss
                if contrastive_loss is not None:
                    total_loss_value, mse_loss, cl_loss = (
                        self.loss_strategy.compute_loss(
                            regression_output, contrastive_loss, batch_targets
                        )
                    )
                else:
                    mse_loss = self.loss_strategy.compute_loss(
                        regression_output, batch_targets
                    )
                    total_loss_value, cl_loss = mse_loss, 0.0

                total_loss_value.backward()
                self.optimizer.step()

                total_loss += total_loss_value.item()
                total_mse_loss += mse_loss.item()
                total_cl_loss += cl_loss if isinstance(cl_loss, float) else 0.0

            avg_loss = total_loss / len(self.train_loader)
            avg_mse_loss = total_mse_loss / len(self.train_loader)
            avg_cl_loss = total_cl_loss / len(self.train_loader)

            if self.use_tensorboard:
                self.writer.add_scalar("Loss/Total", avg_loss, epoch)
                self.writer.add_scalar("Loss/MSE", avg_mse_loss, epoch)
                self.writer.add_scalar("Loss/Contrastive", avg_cl_loss, epoch)
                self.writer.flush()

            logger.info(
                f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - MSE Loss: {avg_mse_loss:.4f} - Contrastive Loss: {avg_cl_loss:.4f}"
            )

        if self.use_tensorboard:
            self.writer.close()

    def evaluate(self, data_loader, save_results=False):
        """Runs inference on a dataset and computes MAPE."""
        self.model.eval()
        predictions, actuals = [], []

        with torch.no_grad():
            for batch in data_loader:
                regression_output, _, batch_targets = self.forward_pass(batch)

                predictions.append(regression_output.cpu().numpy())
                actuals.append(batch_targets.cpu().numpy())

        predictions, actuals = np.concatenate(predictions), np.concatenate(actuals)

        # Compute MAPE
        mape = (
            np.mean(np.abs((actuals - predictions) / actuals)) * 100
        )  # Mean Absolute Percentage Error

        if save_results:
            results = {
                "predictions": predictions.tolist(),
                "actuals": actuals.tolist(),
                "mape": mape,
            }
            with open(
                f"{self.save_results_dir}/test_results.json", "w", encoding="utf-8"
            ) as f:
                json.dump(results, f)

        logger.info(f"MAPE on test set: {mape:.4f}")
        return mape
