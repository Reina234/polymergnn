import logging.config
import torch
import os
import json
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import os
import torch
import numpy as np
import json
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GenericTrainer:
    def __init__(
        self,
        model,
        loss_strategy,
        train_loader,
        val_loader,
        test_loader,
        device,
        hyperparams: dict,
        log_dir="logs",
        use_tensorboard=True,
        save_results_dir="results",
        track_learning_curve=False,  # ✅ Add parameter here
    ):
        """
        Generic trainer that initializes the optimizer dynamically.
        """
        self.model = model
        self.loss_strategy = loss_strategy
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.hyperparams = hyperparams
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.save_results_dir = save_results_dir
        self.track_learning_curve = track_learning_curve  # ✅ Store flag

        os.makedirs(save_results_dir, exist_ok=True)

        # 🔥 Dynamically create optimizer from hyperparameters
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyperparams["lr"])

        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir)
            self.writer.add_hparams(hyperparams, {})

        # 🔥 Track training and validation loss if required
        self.train_losses = [] if self.track_learning_curve else None
        self.val_losses = [] if self.track_learning_curve else None

    def train(self, epochs=50):
        """Train model, log training & validation loss, and track learning curves."""
        self.model.train()
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            total_loss, total_mse_loss, total_cl_loss = 0.0, 0.0, 0.0
            num_batches = len(self.train_loader)

            for batch in self.train_loader:
                self.optimizer.zero_grad()
                regression_output, contrastive_loss, batch_targets = self.forward_pass(
                    batch
                )

                # Compute training loss
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

            # ✅ Normalize training losses
            avg_train_loss = total_loss / num_batches
            avg_train_mse_loss = total_mse_loss / num_batches
            avg_train_cl_loss = total_cl_loss / num_batches

            # ✅ Compute validation loss and MAPE
            val_metrics = self.evaluate(self.val_loader)
            avg_val_loss = val_metrics["mse_loss"]
            avg_val_mape = val_metrics["mape"]

            # ✅ Store learning curve losses
            if self.track_learning_curve:
                self.train_losses.append(avg_train_loss)
                self.val_losses.append(avg_val_loss)

            # ✅ Log Training & Validation Loss to TensorBoard
            if self.use_tensorboard:
                self.writer.add_scalar("Loss/Training_Total", avg_train_loss, epoch)
                self.writer.add_scalar("Loss/Training_MSE", avg_train_mse_loss, epoch)
                self.writer.add_scalar(
                    "Loss/Training_Contrastive", avg_train_cl_loss, epoch
                )
                self.writer.add_scalar("Loss/Validation_Total", avg_val_loss, epoch)
                self.writer.add_scalar("Loss/Validation_MAPE", avg_val_mape, epoch)
                self.writer.flush()

            logger.info(
                f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - "
                f"Val MAPE: {avg_val_mape:.2f}% - MSE Loss: {avg_train_mse_loss:.4f} - Contrastive Loss: {avg_train_cl_loss:.4f}"
            )

        if self.use_tensorboard:
            self.writer.add_hparams(
                self.hyperparams,
                {
                    "Final_Validation_Loss": avg_val_loss,
                    "Final_Validation_MAPE": avg_val_mape,
                },
            )
            self.writer.close()

        # ✅ Plot learning curve at the end of training
        if self.track_learning_curve:
            self.plot_learning_curve()

    def forward_pass(self, batch):
        """
        Handles different models by checking expected input signatures.
        Works with:
        - ChemPropFGHierarchicalModel
        - MPNN models
        - Standard FNNs
        """
        batch_molgraph, _, batch_fg_list, _, batch_targets = batch

        if isinstance(batch_targets, tuple) or isinstance(batch_targets, list):
            batch_targets = torch.stack(
                batch_targets
            )  # Only stack if it's a list of tensors

        batch_targets = batch_targets.to(self.device)
        batch_targets = batch_targets.unsqueeze(0)  # 🔥 Fix shape mismatch

        if hasattr(self.model, "contrastive"):  # ChemPropFGHierarchicalModel
            regression_output, contrastive_loss, _ = self.model(
                batch_molgraph, batch_fg_list
            )
            return regression_output, contrastive_loss, batch_targets

        elif hasattr(self.model, "message_passing"):  # MPNN (or similar architectures)
            regression_output = self.model(batch_molgraph)
            return regression_output, None, batch_targets

        else:  # Standard feedforward models (FNN, etc.)
            batch_molgraph = torch.stack(batch_molgraph).to(self.device)
            regression_output = self.model(batch_molgraph)
            return regression_output, None, batch_targets

    def evaluate(self, data_loader, save_results=False):
        """Runs inference on a dataset and computes MAPE and MSE loss."""
        self.model.eval()
        predictions, actuals = [], []
        total_loss = 0.0
        num_batches = len(data_loader)

        with torch.no_grad():
            for batch in data_loader:
                try:
                    regression_output, contrastive_loss, batch_targets = (
                        self.forward_pass(batch)
                    )

                    predictions.append(regression_output.cpu().numpy())
                    actuals.append(batch_targets.cpu().numpy())

                    # ✅ Ensure correct arguments are passed to compute_loss
                    if contrastive_loss is not None:
                        mse_loss = self.loss_strategy.compute_loss(
                            regression_output, contrastive_loss, batch_targets
                        )[
                            1
                        ]  # Extract only MSE loss
                    else:
                        mse_loss = self.loss_strategy.compute_loss(
                            regression_output, batch_targets
                        )  # Fix: Only pass necessary parameters

                    total_loss += mse_loss.item()

                except Exception as e:
                    logger.error(f"Error during evaluation: {e}")
                    continue  # Skip problematic batch

        # ✅ Normalize validation loss
        avg_loss = total_loss / num_batches if num_batches > 0 else total_loss

        # ✅ Convert lists to NumPy arrays
        try:
            predictions, actuals = np.concatenate(predictions), np.concatenate(actuals)
        except ValueError as e:
            logger.error(f"Error concatenating predictions or actuals: {e}")
            return None  # Exit function if data is malformed

        # ✅ Compute MAPE (handle division by zero)
        epsilon = 1e-8
        try:
            mape = np.mean(np.abs((actuals - predictions) / (actuals + epsilon))) * 100
        except Exception as e:
            logger.error(f"Error computing MAPE: {e}")
            mape = None

        # ✅ Save results
        if save_results:
            results = {
                "predictions": predictions.tolist(),
                "actuals": actuals.tolist(),
                "mape": mape,
                "mse_loss": avg_loss,
            }
            with open(
                f"{self.save_results_dir}/test_results.json", "w", encoding="utf-8"
            ) as f:
                json.dump(results, f)

        logger.info(f"MAPE on test set: {mape:.4f} | MSE Loss: {avg_loss:.4f}")

        return {
            "mape": mape,
            "mse_loss": avg_loss,
        }  # ✅ Return dictionary with both metrics

    def plot_learning_curve(self):
        """Plots training vs validation loss over epochs."""
        plt.figure(figsize=(8, 6))
        plt.plot(
            range(1, len(self.train_losses) + 1),
            self.train_losses,
            label="Training Loss",
            marker="o",
        )
        plt.plot(
            range(1, len(self.val_losses) + 1),
            self.val_losses,
            label="Validation Loss",
            marker="s",
        )
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid()
        plt.savefig(f"{self.save_results_dir}/learning_curve.png")  # Save plot
        plt.show()


class GenericTrainer2:
    def __init__(
        self,
        model,
        loss_strategy,
        train_loader,
        val_loader,
        test_loader,
        device,
        hyperparams: dict,
        log_dir="logs",
        use_tensorboard=True,
        save_results_dir="results",
    ):
        """
        Generic trainer that initializes the optimizer dynamically based on lr.
        """
        self.model = model
        self.loss_strategy = loss_strategy
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.hyperparams = hyperparams  # Store hyperparameters for tracking
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.save_results_dir = save_results_dir

        os.makedirs(save_results_dir, exist_ok=True)

        # 🔥 Dynamically create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=hyperparams["lr"])

        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir)
            self.writer.add_hparams(hyperparams, {})

    def train(self, epochs=50):
        """Train model, log metrics, and track learning curves."""
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

            # 🔥 Track training and validation loss for learning curve
            val_loss = self.evaluate(self.val_loader)
            self.train_losses.append(avg_loss)
            self.val_losses.append(val_loss)

            # ✅ Log both Training and Validation loss to TensorBoard
            if self.use_tensorboard:
                self.writer.add_scalar("Loss/Training", avg_loss, epoch)
                self.writer.add_scalar("Loss/Validation", val_loss, epoch)
                self.writer.add_scalar("Loss/MSE", avg_mse_loss, epoch)
                self.writer.add_scalar("Loss/Contrastive", avg_cl_loss, epoch)
                self.writer.flush()

            logger.info(
                f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f} - Val Loss: {val_loss:.4f} - MSE Loss: {avg_mse_loss:.4f} - Contrastive Loss: {avg_cl_loss:.4f}"
            )

        if self.use_tensorboard:
            self.writer.add_hparams(
                self.hyperparams, {"Final_Validation_Loss": avg_loss}
            )
            self.writer.close()

        # 🔥 Plot learning curves at the end of training
        self.plot_learning_curve()

    def forward_pass(self, batch):
        """
        Handles different models by checking expected input signatures.
        Works with:
        - ChemPropFGHierarchicalModel
        - MPNN models
        - Standard FNNs
        """
        batch_molgraph, _, batch_fg_list, _, batch_targets = batch

        if isinstance(batch_targets, tuple) or isinstance(batch_targets, list):
            batch_targets = torch.stack(
                batch_targets
            )  # Only stack if it's a list of tensors

        batch_targets = batch_targets.to(self.device)
        batch_targets = batch_targets.unsqueeze(-1)  # 🔥 Fix shape mismatch

        if hasattr(self.model, "contrastive"):  # ChemPropFGHierarchicalModel
            regression_output, contrastive_loss, _ = self.model(
                batch_molgraph, batch_fg_list
            )
            return regression_output, contrastive_loss, batch_targets

        elif hasattr(self.model, "message_passing"):  # MPNN (or similar architectures)
            regression_output = self.model(
                batch_molgraph
            )  # ✅ Pass `batch_molgraph` explicitly
            return regression_output, None, batch_targets

        else:  # Standard feedforward models (FNN, etc.)
            batch_molgraph = torch.stack(batch_molgraph).to(self.device)
            regression_output = self.model(batch_molgraph)
            return regression_output, None, batch_targets

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
                f"{self.save_results_dir}/test_results.json", "", encoding="utf-8"
            ) as f:
                json.dump(results, f)

        logger.info(f" MAPE on test set: {mape:.4f}")
        return mape
