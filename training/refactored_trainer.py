import torch
import os
import logging
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from typing import Dict, Tuple, List, Optional
import json
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# mean_percentage_error() removed due to exploding values
from tools.log_transform_helper import LogTransformHelper
from torch.utils.data import DataLoader
import torch.optim as optim
from models.polymer_gnn_full import PolymerGNNSystem
from training.refactored_batched_dataset import PolymerGNNDataset, PolymerDataset

logger = logging.getLogger(__name__)


class Trainer(ABC):
    """Base Trainer with utilities for modularity and single responsibility."""

    def __init__(
        self,
        train_dataset: PolymerDataset,
        val_dataset: Optional[PolymerDataset],
        test_dataset: Optional[PolymerDataset],
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
        additional_info: Optional[Dict[str, str]] = None,
    ):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.hyperparams = hyperparams or {}
        self.model = self.configure_model()
        self.model.to(self.device)
        batch_size = self.hyperparams.get("batch_size", 32)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=train_dataset.collate_fn,
        )
        self.val_loader = (
            DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=val_dataset.collate_fn,
            )
            if val_dataset
            else None
        )
        self.test_loader = (
            DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=test_dataset.collate_fn,
            )
            if test_dataset
            else None
        )

        self.evaluation_metrics = evaluation_metrics or ["MAE", "MAPE", "RMSE"]
        self.num_batches = len(self.train_loader) if self.train_loader else 0
        self.track_learning_curve = track_learning_curve
        self.figure_size = figure_size
        self.eval_metrics = {}
        self.optimiser = self.configure_optimiser()
        self.summary_line = self.generate_run_summary(
            self,
            model=self.model,
            optimiser=self.optimiser,
            device=self.device,
            num_batches=self.num_batches,
            feature_pipelines=train_dataset.pipeline_manager.feature_pipelines,
            target_pipelines=train_dataset.pipeline_manager.target_pipelines,
            molecule_featuriser=train_dataset.mol_to_molgraph,
        )

        if additional_info:
            json_info = json.dumps(
                additional_info, indent=None, separators=(", ", ": ")
            )
            self.summary_line = {
                **json.loads(self.summary_line),
                **json.loads(json_info),
            }
            self.summary_line = json.dumps(
                self.summary_line, indent=None, separators=(", ", ": ")
            )

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

        self.final_test_loss = None
        self.final_test_metrics = {}

    @abstractmethod
    def forward_pass(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    @abstractmethod
    def configure_model(self) -> torch.nn.Module:
        pass

    @abstractmethod
    def inverse_transform(
        self, labels: np.ndarray, preds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def configure_optimiser(self) -> torch.optim.Optimizer:
        optim_lr = self.hyperparams.get("lr", 1e-3)
        optim_weight_decay = self.hyperparams.get("weight_decay", 0.0)
        optimiser = optim.Adam(
            self.model.parameters(), lr=optim_lr, weight_decay=optim_weight_decay
        )
        return optimiser

    def compute_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:
        base_loss = torch.nn.functional.mse_loss(predictions, labels)
        return base_loss

    def _train_batch(self, batch) -> Tuple[float, np.ndarray, np.ndarray]:
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
        all_labels, all_preds = self.inverse_transform(all_labels, all_preds)
        train_metrics = self.compute_metrics(all_labels, all_preds)
        metrics_dict = {"Training Loss": avg_loss, **train_metrics}
        self._write_scalar_to_tensorboard(metrics_dict, epoch)
        self._add_to_learning_curve(avg_loss, is_train=True)
        return avg_loss, train_metrics

    def evaluate(
        self, loader: torch.utils.data.DataLoader, epoch: int, mode: str = "Validation"
    ) -> Tuple[float, Dict[str, float]]:
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
        all_labels, all_preds = self.inverse_transform(all_labels, all_preds)
        metrics = self.compute_metrics(all_labels, all_preds)
        metrics_dict = {f"{mode} Loss": avg_loss, **metrics}
        self._write_scalar_to_tensorboard(metrics_dict, epoch)
        self._add_to_learning_curve(avg_loss, is_train=False)
        return avg_loss, metrics

    def compute_metrics(
        self, labels: np.ndarray, preds: np.ndarray
    ) -> Dict[str, float]:
        results = {}
        for metric in self.evaluation_metrics:
            if metric == "MAE":
                results["Metrics/MAE"] = mean_absolute_error(labels, preds)
            elif metric == "MAPE":
                results["Metrics/MAPE"] = 100 * np.mean(
                    np.abs(labels - preds) / np.abs(np.max(labels, 10**-10))
                )
            elif metric == "RMSE":
                results["Metrics/RMSE"] = np.sqrt(mean_squared_error(labels, preds))
            elif metric == "R2":
                results["Metrics/R2"] = r2_score(labels, preds)
        return results

    def _add_to_learning_curve(self, loss: float, is_train: bool):
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

    def run(self):
        epochs = self.hyperparams.get("epochs", 50)
        self.train(epochs=epochs)
        self.test()

    def train(self, epochs: int):
        logger.info("Starting Training...")
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            epoch_loss, _ = self._train_epoch(epoch)
            val_loss = None
            if self.val_loader:
                val_loss, eval_metrics = self.evaluate(
                    self.val_loader, epoch, mode="Validation"
                )
                self.eval_metrics[epoch] = eval_metrics
            if self.flush_tensorboard_each_epoch:
                self._flush_tensorboard()
        self._write_hparams_to_tensorboard(
            train_losses={"Final Train Loss": epoch_loss},
            val_losses={"Final Val Loss": val_loss or 0},
        )
        if self.track_learning_curve:
            self._plot_learning_curve()
            self._save_learning_curve_data()
            self._save_eval_metrics()
        logger.info("Training Completed.")

    def _flush_tensorboard(self):
        if self.use_tensorboard and self.writer is not None:
            self.writer.flush()

    def test(self):
        if not self.test_loader:
            logger.warning("No test loader provided.")
            return
        final_loss, test_metrics = self.evaluate(self.test_loader, epoch=0, mode="Test")
        self.final_test_loss = final_loss
        self.final_test_metrics = test_metrics
        summary_lines = []
        summary_dict = json.loads(self.summary_line)
        for key, value in summary_dict.items():
            summary_lines.append(f"{key}: {value}")
        results_path = os.path.join(self.save_results_dir, "test_results.txt")
        with open(results_path, "a", encoding="utf-8") as f:
            f.write("=== Test Run Summary ===\n")
            f.write("\n".join(summary_lines) + "\n\n")
            f.write("--- Test Metrics ---\n")
            for k, v in test_metrics.items():
                f.write(f"{k}: {v:.6f}\n")
            f.write("=" * 50 + "\n\n")
        logger.info("Test results saved to %s", results_path)

    @staticmethod
    def generate_run_summary(instance, **kwargs) -> str:
        summary = {}
        summary["class"] = instance.__class__.__name__
        for attr_name, attr_value in vars(instance).items():
            if hasattr(attr_value, "__class__"):
                summary[attr_name] = attr_value.__class__.__name__
            else:
                summary[attr_name] = repr(attr_value)
        for name, value in kwargs.items():
            if hasattr(value, "__class__"):
                summary[name] = value.__class__.__name__
            else:
                summary[name] = repr(value)
        return json.dumps(summary, indent=None, separators=(", ", ": "))

    def _plot_learning_curve(self):
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

    def _save_learning_curve_data(self):
        losses = {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
        }
        filename = self._create_convergence_data_name()
        logger.info("Saving learning curve data to %s", filename)
        with open(
            os.path.join(self.save_results_dir, filename), "w", encoding="utf-8"
        ) as f:
            json.dump(losses, f)
        logger.info("Learning curve data saved.")

    def _create_run_key(self):
        name = "_".join(
            [
                f"{key}={value}"
                for key, value in self.hyperparams.items()
                if key != "device"
            ]
        )
        return name

    def _create_learning_curve_figname(self):
        figname = self._create_run_key()
        return f"{figname}_learning_curve.png"

    def _create_convergence_data_name(self):
        return f"{self._create_run_key()}_convergence.json"

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

    def _create_metrics_data_name(self):
        return f"{self._create_run_key()}_metrics.json"

    def _save_eval_metrics(self):
        filename = self._create_metrics_data_name()
        file_path = os.path.join(self.save_results_dir, filename)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(self.eval_metrics, f, indent=4)

        logger.info("Saved validation metrics to %s", file_path)


class PolymerGNNTrainer(Trainer):

    def __init__(
        self,
        train_dataset: PolymerGNNDataset,
        val_dataset: Optional[PolymerGNNDataset],
        test_dataset: Optional[PolymerGNNDataset],
        hyperparams: dict,
        log_dir="logs",
        save_results_dir="results",
        use_tensorboard=True,
        track_learning_curve=False,
        evaluation_metrics: List[str] = None,
        figure_size=(8, 6),
        writer: Optional[SummaryWriter] = None,
        flush_tensorboard_each_epoch: bool = False,
        close_writer_on_finish: bool = True,
        additional_info: Optional[Dict[str, str]] = None,
    ):
        """
        Additional arguments:
         - loss_config: Dictionary for configuring loss weights or other loss-related hyperparameters.
         - log_transform_helper: An instance of a log transform helper (like LogTransformHelper) that will
           be used for inverse transforming predictions and labels.
        """
        super().__init__(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            hyperparams=hyperparams,
            log_dir=log_dir,
            save_results_dir=save_results_dir,
            use_tensorboard=use_tensorboard,
            track_learning_curve=track_learning_curve,
            evaluation_metrics=evaluation_metrics,
            figure_size=figure_size,
            writer=writer,
            flush_tensorboard_each_epoch=flush_tensorboard_each_epoch,
            close_writer_on_finish=close_writer_on_finish,
            additional_info=additional_info,
        )

        log_selection_tensor = self.hyperparams.get(
            "log_selection_tensor", torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
        )
        self.log_transform_helper = LogTransformHelper(
            log_selection_tensor=log_selection_tensor,
            target_transformers=train_dataset.pipeline_manager.target_pipelines,
        )

    def configure_model(self) -> torch.nn.Module:
        mpnn_output_dim = self.hyperparams.get("mpnn_output_dim", 300)
        mpnn_hidden_dim = self.hyperparams.get("mpnn_hidden_dim", 300)
        mpnn_depth = self.hyperparams.get("mpnn_depth", 3)
        mpnn_dropout = self.hyperparams.get("mpnn_dropout", 0.1)
        rdkit_selection_tensor = self.hyperparams.get("rdkit_selection_tensor", None)
        molecule_embedding_hidden_dim = self.hyperparams.get(
            "molecule_embedding_hidden_dim", 512
        )
        embedding_dim = self.hyperparams.get("embedding_dim", 256)
        use_rdkit = self.hyperparams.get("use_rdkit", True)
        use_chembert = self.hyperparams.get("use_chembert", True)
        gnn_hidden_dim = self.hyperparams.get("gnn_hidden_dim", 256)
        gnn_output_dim = self.hyperparams.get("gnn_output_dim", 128)
        gnn_dropout = self.hyperparams.get("gnn_dropout", 0.1)
        gnn_num_heads = self.hyperparams.get("gnn_num_heads", 4)
        multitask_fnn_hidden_dim = self.hyperparams.get("multitask_fnn_hidden_dim", 128)
        multitask_fnn_dropout = self.hyperparams.get("multitask_fnn_dropout", 0.1)
        model = PolymerGNNSystem(
            mpnn_output_dim=mpnn_output_dim,
            mpnn_hidden_dim=mpnn_hidden_dim,
            mpnn_depth=mpnn_depth,
            mpnn_dropout=mpnn_dropout,
            rdkit_selection_tensor=rdkit_selection_tensor,
            molecule_embedding_hidden_dim=molecule_embedding_hidden_dim,
            embedding_dim=embedding_dim,
            use_rdkit=use_rdkit,
            use_chembert=use_chembert,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            gnn_dropout=gnn_dropout,
            gnn_num_heads=gnn_num_heads,
            multitask_fnn_hidden_dim=multitask_fnn_hidden_dim,
            multitask_fnn_dropout=multitask_fnn_dropout,
        )
        return model

    def forward_pass(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:
        # Move tensors to device
        batch = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()
        }

        predictions = self.model(batch)

        labels = batch["labels"].to(self.device)

        if torch.isnan(labels).any():
            logger.error(
                "NaN detected in labels! Labels: %s for smiles: %s",
                labels,
                batch["smiles_list"],
            )
        filtered_labels = self.log_transform_helper.filter_target_labels(labels)

        return predictions, filtered_labels

    def inverse_transform(
        self, labels: np.ndarray, preds: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        inv_labels = self.log_transform_helper.inverse_transform(
            values_to_transform=labels
        )
        inv_preds = self.log_transform_helper.inverse_transform(
            values_to_transform=preds
        )
        return inv_labels.numpy(), inv_preds.numpy()
