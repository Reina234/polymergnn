import torch
import os
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    mean_absolute_percentage_error,
    r2_score,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer(ABC):
    """Base Trainer with utilities and improvements for modularity and single responsibility."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        train_loader: torch.utils.data.DataLoader,
        val_loader: Optional[torch.utils.data.DataLoader],
        test_loader: Optional[torch.utils.data.DataLoader],
        device: torch.device,
        loss_strategy=None,
        hyperparams: dict = None,
        log_dir="logs",
        save_results_dir="results",
        use_tensorboard=True,
        track_learning_curve=False,
        evaluation_metrics: List[str] = None,
        figure_size=(8, 6),
        writer: Optional[SummaryWriter] = None,
        flush_tensorboard_each_epoch: bool = False,
        close_writer_on_finish: bool = True,
    ):
        self.model = model
        self.optimiser = optimiser
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.loss_strategy = loss_strategy
        self.hyperparams = hyperparams or {}
        self.evaluation_metrics = evaluation_metrics or ["MAE", "MAPE", "RMSE"]
        self.num_batches = len(train_loader) if train_loader else 0
        self.track_learning_curve = track_learning_curve
        self.figure_size = figure_size

        os.makedirs(save_results_dir, exist_ok=True)
        self.save_results_dir = save_results_dir

        self.use_tensorboard = use_tensorboard
        if self.use_tensorboard:
            if writer is not None:
                self.writer = writer
            else:
                self.writer = SummaryWriter(log_dir)
                self.writer.add_hparams(self.hyperparams, {})
        else:
            self.writer = None

        self.flush_tensorboard_each_epoch = flush_tensorboard_each_epoch
        self.close_writer_on_finish = close_writer_on_finish

        self.train_losses = [] if self.track_learning_curve else None
        self.val_losses = [] if self.track_learning_curve else None

        # Attributes to store final test metrics.
        self.final_test_loss = None
        self.final_test_metrics = {}

    @abstractmethod
    def forward_pass(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def compute_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        pass

    def _train_batch(self, batch) -> Tuple[float, np.ndarray, np.ndarray]:
        """Performs a single training batch and returns loss and detached outputs."""
        self.optimiser.zero_grad()
        predictions, labels = self.forward_pass(batch)
        loss = self.compute_loss(predictions, labels)
        loss.backward()
        self.optimiser.step()
        return (
            loss.item(),
            predictions.detach().cpu().numpy(),
            labels.detach().cpu().numpy(),
        )

    def _train_epoch(self, epoch: int) -> Tuple[float, Optional[Dict[str, float]]]:
        """Trains for one epoch and returns average loss and additional metrics."""
        self.model.train()
        epoch_loss = 0.0
        all_preds, all_labels = [], []

        for batch in self.train_loader:
            batch_loss, preds, labels = self._train_batch(batch)
            epoch_loss += batch_loss
            all_preds.append(preds)
            all_labels.append(labels)

        avg_loss = epoch_loss / self.num_batches
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        train_metrics = self.compute_metrics(all_labels, all_preds)

        metrics_dict = {"Training Loss": avg_loss, **train_metrics}
        self._write_scalar_to_tensorboard(metrics_dict, epoch)
        self._add_to_learning_curve(avg_loss, is_train=True)
        return avg_loss, train_metrics

    def evaluate(
        self, loader: torch.utils.data.DataLoader, epoch: int, mode: str = "Validation"
    ) -> Tuple[float, Dict[str, float]]:
        """Evaluates model and logs metrics."""
        self.model.eval()
        all_preds, all_labels = [], []
        total_loss = 0.0

        with torch.no_grad():
            for batch in loader:
                predictions, labels = self.forward_pass(batch)
                loss = self.compute_loss(predictions, labels)
                total_loss += loss.item()
                all_preds.append(predictions.cpu().numpy())
                all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / len(loader)
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        metrics = self.compute_metrics(all_labels, all_preds)

        metrics_dict = {f"{mode} Loss": avg_loss, **metrics}
        self._write_scalar_to_tensorboard(metrics_dict, epoch)
        self._add_to_learning_curve(avg_loss, is_train=False)
        return avg_loss, metrics

    def compute_metrics(
        self, labels: np.ndarray, preds: np.ndarray
    ) -> Dict[str, float]:
        """Computes common regression metrics."""
        results = {}
        for metric in self.evaluation_metrics:
            if metric == "MAE":
                results["Metrics/MAE"] = mean_absolute_error(labels, preds)
            elif metric == "MAPE":
                results["Metrics/MAPE"] = 100 * mean_absolute_percentage_error(
                    labels, preds
                )
            elif metric == "RMSE":
                results["Metrics/RMSE"] = np.sqrt(mean_squared_error(labels, preds))
            elif metric == "R2":
                results["Metrics/R2"] = r2_score(labels, preds)
        return results

    def _add_to_learning_curve(self, loss: float, is_train: bool):
        """Tracks loss for learning curve plotting."""
        if not self.track_learning_curve:
            return
        if is_train:
            self.train_losses.append(loss)
        else:
            self.val_losses.append(loss)

    def _write_scalar_to_tensorboard(self, metrics: Dict[str, float], epoch: int):
        if self.use_tensorboard and self.writer is not None:
            self.writer.add_scalar("Epoch", epoch, epoch)
            for key, value in metrics.items():
                self.writer.add_scalar(key, value, epoch)

    def _normalise_loss_dict(
        self, loss_dict: Dict[str, float], keys_to_normalise: Optional[List[str]]
    ) -> Dict[str, float]:
        if keys_to_normalise is None:
            keys_to_normalise = loss_dict.keys()
        for key in keys_to_normalise:
            loss_dict[key] /= self.num_batches
        return loss_dict

    def _flush_tensorboard(self):
        if self.use_tensorboard and self.writer is not None:
            self.writer.flush()

    def _write_hparams_to_tensorboard(
        self, train_losses: Dict[str, float], val_losses: Dict[str, float]
    ):
        if self.use_tensorboard and self.writer is not None:
            final_losses = self._make_final_losses(train_losses, val_losses)
            hparams = {**self.hyperparams, "epochs": self.hyperparams.get("epochs", 0)}
            self.writer.add_hparams(hparams, final_losses)
            if "Final Train Loss" in final_losses:
                self.writer.add_scalar(
                    "Final/Train Loss", final_losses["Final Train Loss"]
                )
            if "Final Val Loss" in final_losses:
                self.writer.add_scalar("Final/Val Loss", final_losses["Final Val Loss"])
            self.writer.flush()
            if self.close_writer_on_finish:
                self.writer.close()

    @abstractmethod
    def _make_final_losses(
        self,
        additional_train_losses: Dict[str, float],
        additional_val_losses: Dict[str, float],
    ) -> Dict[str, float]:
        pass

    def train(self, epochs: int):
        """Runs the full training loop."""
        logger.info("Starting Training...")
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            epoch_loss, _ = self._train_epoch(epoch)
            val_loss = None
            if self.val_loader:
                val_loss, _ = self.evaluate(self.val_loader, epoch, mode="Validation")
            if self.flush_tensorboard_each_epoch:
                self._flush_tensorboard()

        self._write_hparams_to_tensorboard(
            train_losses={"Final Train Loss": epoch_loss},
            val_losses={"Final Val Loss": val_loss or 0},
        )

        if self.track_learning_curve:
            self.plot_learning_curve()

        logger.info("Training Completed.")

    def test(self):
        """Tests the model using the test loader and records final metrics."""
        if not self.test_loader:
            logger.warning("No test loader provided.")
            return
        final_loss, test_metrics = self.evaluate(self.test_loader, epoch=0, mode="Test")
        self.final_test_loss = final_loss
        self.final_test_metrics = test_metrics

        results_path = os.path.join(self.save_results_dir, "test_results.txt")
        with open(results_path, "a", encoding="utf-8") as f:
            for k, v in test_metrics.items():
                f.write("{k}: {v:.6f}\n".format(k=k, v=v))
        logger.info("Test results saved to %s", results_path)

    def plot_learning_curve(self):
        """Plots the learning curve."""
        import matplotlib.pyplot as plt

        plt.figure(figsize=self.figure_size)
        plt.plot(self.train_losses, label="Train Loss", marker="o")
        if self.val_losses:
            plt.plot(self.val_losses, label="Validation Loss", marker="s")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Learning Curve")
        plt.legend()
        plt.grid(True)
        plt.savefig(
            os.path.join(self.save_results_dir, self._create_learning_curve_figname())
        )
        plt.show()

    def _create_learning_curve_figname(self):
        figname = "_".join(
            [
                f"{key}={value}"
                for key, value in self.hyperparams.items()
                if key != "device"
            ]
        )
        return f"{figname}_learning_curve.png"


class MoleculeTrainer(Trainer):
    """Trainer specialized for MoleculeEmbeddingModel."""

    def forward_pass(self, batch):
        batch = {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }
        predictions = self.model(batch)
        labels = batch["labels"].to(self.device)
        return predictions, labels

    def compute_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.mse_loss(predictions, labels)

    def _make_final_losses(
        self,
        additional_train_losses: Dict[str, float],
        additional_val_losses: Dict[str, float],
    ) -> Dict[str, float]:
        final_losses = {}
        if additional_train_losses:
            final_losses["Final Train Loss"] = additional_train_losses.get(
                "Final Train Loss", 0
            )
        if additional_val_losses:
            final_losses["Final Val Loss"] = additional_val_losses.get(
                "Final Val Loss", 0
            )
        logger.info("Final Training Loss: %s", final_losses.get("Final Train Loss", 0))
        logger.info("Final Validation Loss: %s", final_losses.get("Final Val Loss", 0))
        return final_losses
