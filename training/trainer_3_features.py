import torch
import logging
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple, List, Optional
from training.refactored_trainer import Trainer
import json
from tqdm import tqdm

# mean_percentage_error() removed due to exploding values
from tools.log_transform_helper import LogTransformHelper
from models.temperature_aware_gnn.full_model import MorganPolymerGNNSystem3Feature
from models.temperature_aware_gnn.full_model_no_morgan import DensityPolymerGNNSystem
from training.refactored_batched_dataset import PolymerGNNDataset
from models.separate_mpnn_model.full_separate_mpnn_model import SeparatedGNNSystem
from models.separate_mpnn_model.full_morgan_separate_mpnn import (
    MorganSeparatedGNNSystem,
)
from models.separate_mpnn_model.full_separate_mpnn_v2 import SeparatedGNNSystemV2
from models.separate_mpnn_model.full_morgan_mpnn_model import SeparatedGNNSystemV3Morgan
from models.separate_mpnn_model.full_one_output_model import OneOutputGNN
from models.separate_mpnn_model.full_separate_v3 import SeparatedGNNSystemV3
from models.separate_mpnn_model.full_chemberta_model import (
    SeparatedGNNSystemChemberta,
)

logger = logging.getLogger(__name__)


class DensityMorganPolymerGNNTrainer(Trainer):

    def __init__(
        self,
        n_bits: int,
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
        self.n_bits = n_bits
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
        self.loss_weights = self.hyperparams.get(
            "weights", torch.tensor([1, 1, 1, 1, 1, 1, 1])
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
        multitask_fnn_shared_layer_dim = self.hyperparams.get(
            "multitask_fnn_shared_layer_dim", 128
        )
        multitask_fnn_dropout = self.hyperparams.get("multitask_fnn_dropout", 0.1)
        model = MorganPolymerGNNSystem3Feature(
            n_bits=self.n_bits,
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
            multitask_fnn_shared_layer_dim=multitask_fnn_shared_layer_dim,
            multitask_fnn_dropout=multitask_fnn_dropout,
        )
        return model

    def _configure_optimiser(self) -> torch.optim.Optimizer:
        # Base learning rate and weight decay
        optim_lr = self.hyperparams.get("lr", 1e-3)
        optim_weight_decay = self.hyperparams.get("weight_decay", 0.0)

        # Optuna-tuned scaling factors for specific heads
        log_diffusion_factor = self.hyperparams.get(
            "log_diffusion_factor", 2.0
        )  # >1 increases LR
        log_rg_factor = self.hyperparams.get(
            "log_rg_factor", 1.5
        )  # Can be >1, <1, or 1

        # Define parameter groups with different LRs
        optimiser = torch.optim.Adam(
            [
                {
                    "params": self.model.parameters(),
                    "lr": optim_lr,
                    "weight_decay": optim_weight_decay,
                },  # Default
                {
                    "params": self.model.polymer_fnn.log_diffusion_head.parameters(),
                    "lr": optim_lr * log_diffusion_factor,
                    "weight_decay": optim_weight_decay,
                },  # Increased LR for log_diffusion_head
                {
                    "params": self.model.polymer_fnn.log_rg_head.parameters(),
                    "lr": optim_lr * log_rg_factor,
                    "weight_decay": optim_weight_decay,
                },  # Tuned LR for log_rg_head
            ]
        )

        return optimiser

    def compute_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:

        loss_vect = self.loss_fn(predictions, labels)
        weighted_loss = loss_vect * self.loss_weights
        final_loss = torch.mean(weighted_loss)
        return final_loss

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


class DensityPolymerGNNTrainer(Trainer):

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
        self.loss_weights = self.hyperparams.get(
            "weights", torch.tensor([1, 1, 1, 1, 1, 1, 1])
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
        multitask_fnn_shared_layer_dim = self.hyperparams.get(
            "multitask_fnn_shared_layer_dim", 128
        )
        multitask_fnn_dropout = self.hyperparams.get("multitask_fnn_dropout", 0.1)
        model = DensityPolymerGNNSystem(
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
            multitask_fnn_shared_layer_dim=multitask_fnn_shared_layer_dim,
            multitask_fnn_dropout=multitask_fnn_dropout,
        )
        return model

    def _configure_optimiser(self) -> torch.optim.Optimizer:
        # Base learning rate and weight decay
        optim_lr = self.hyperparams.get("lr", 1e-3)
        optim_weight_decay = self.hyperparams.get("weight_decay", 0.0)

        # Optuna-tuned scaling factors for specific heads
        log_diffusion_factor = self.hyperparams.get(
            "log_diffusion_factor", 2.0
        )  # >1 increases LR
        log_rg_factor = self.hyperparams.get(
            "log_rg_factor", 1.5
        )  # Can be >1, <1, or 1

        # Define parameter groups with different LRs
        optimiser = torch.optim.Adam(
            [
                {
                    "params": self.model.parameters(),
                    "lr": optim_lr,
                    "weight_decay": optim_weight_decay,
                },  # Default
                {
                    "params": self.model.polymer_fnn.log_diffusion_head.parameters(),
                    "lr": optim_lr * log_diffusion_factor,
                    "weight_decay": optim_weight_decay,
                },  # Increased LR for log_diffusion_head
                {
                    "params": self.model.polymer_fnn.log_rg_head.parameters(),
                    "lr": optim_lr * log_rg_factor,
                    "weight_decay": optim_weight_decay,
                },  # Tuned LR for log_rg_head
            ]
        )

        return optimiser

    def compute_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:

        loss_vect = self.loss_fn(predictions, labels)
        weighted_loss = loss_vect * self.loss_weights
        final_loss = torch.mean(weighted_loss)
        return final_loss

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


class SeparatedGNNTrainer(Trainer):

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
        self.loss_weights = self.hyperparams.get(
            "weights", torch.tensor([1, 1, 1, 1, 1, 1, 1])
        )
        log_selection_tensor = self.hyperparams.get(
            "log_selection_tensor", torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
        )
        self.log_transform_helper = LogTransformHelper(
            log_selection_tensor=log_selection_tensor,
            target_transformers=train_dataset.pipeline_manager.target_pipelines,
        )

    def infer(self, loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Runs inference on a given DataLoader and returns predictions.

        Args:
            loader (torch.utils.data.DataLoader): The DataLoader for inference data.

        Returns:
            np.ndarray: Inverse-transformed predictions.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Get predictions
                predictions = self.model(batch)  # Shape: (N, num_outputs)

                all_preds.append(predictions.cpu().numpy())

        all_preds_numpy = [pred for pred in all_preds if pred.shape[0] > 0]

        # Concatenate along the first axis (axis=0) to stack batches into (T, 6)
        preds_numpy = np.concatenate(all_preds_numpy, axis=0)

        inv_preds = self.log_transform_helper.inverse_transform(
            values_to_transform=preds_numpy
        )

        return inv_preds

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
        multitask_fnn_shared_layer_dim = self.hyperparams.get(
            "multitask_fnn_shared_layer_dim", 128
        )
        multitask_fnn_dropout = self.hyperparams.get("multitask_fnn_dropout", 0.1)
        model = SeparatedGNNSystem(
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
            multitask_fnn_shared_layer_dim=multitask_fnn_shared_layer_dim,
            multitask_fnn_dropout=multitask_fnn_dropout,
        )
        return model

    def _configure_optimiser(self) -> torch.optim.Optimizer:
        # Base learning rate and weight decay
        optim_lr = self.hyperparams.get("lr", 1e-3)
        optim_weight_decay = self.hyperparams.get("weight_decay", 0.0)

        # Optuna-tuned scaling factors for specific heads
        log_diffusion_factor = self.hyperparams.get(
            "log_diffusion_factor", 2.0
        )  # >1 increases LR
        log_rg_factor = self.hyperparams.get(
            "log_rg_factor", 1.5
        )  # Can be >1, <1, or 1

        # Define parameter groups with different LRs
        optimiser = torch.optim.Adam(
            [
                {
                    "params": self.model.parameters(),
                    "lr": optim_lr,
                    "weight_decay": optim_weight_decay,
                },  # Default
                {
                    "params": self.model.polymer_fnn.log_diffusion_head.parameters(),
                    "lr": optim_lr * log_diffusion_factor,
                    "weight_decay": optim_weight_decay,
                },  # Increased LR for log_diffusion_head
                {
                    "params": self.model.polymer_fnn.log_rg_head.parameters(),
                    "lr": optim_lr * log_rg_factor,
                    "weight_decay": optim_weight_decay,
                },  # Tuned LR for log_rg_head
            ]
        )

        return optimiser

    def compute_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:

        loss_vect = self.loss_fn(predictions, labels)
        weighted_loss = loss_vect * self.loss_weights
        final_loss = torch.mean(weighted_loss)
        return final_loss

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


class SeparateMorganPolymerGNNTrainer(Trainer):

    def __init__(
        self,
        n_bits: int,
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
        self.n_bits = n_bits
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
        self.loss_weights = self.hyperparams.get(
            "weights", torch.tensor([1, 1, 1, 1, 1, 1, 1])
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
        multitask_fnn_shared_layer_dim = self.hyperparams.get(
            "multitask_fnn_shared_layer_dim", 128
        )
        multitask_fnn_dropout = self.hyperparams.get("multitask_fnn_dropout", 0.1)
        model = MorganSeparatedGNNSystem(
            n_bits=self.n_bits,
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
            multitask_fnn_shared_layer_dim=multitask_fnn_shared_layer_dim,
            multitask_fnn_dropout=multitask_fnn_dropout,
        )
        return model

    def _configure_optimiser(self) -> torch.optim.Optimizer:
        # Base learning rate and weight decay
        optim_lr = self.hyperparams.get("lr", 1e-3)
        optim_weight_decay = self.hyperparams.get("weight_decay", 0.0)

        # Optuna-tuned scaling factors for specific heads
        log_diffusion_factor = self.hyperparams.get(
            "log_diffusion_factor", 2.0
        )  # >1 increases LR
        log_rg_factor = self.hyperparams.get(
            "log_rg_factor", 1.5
        )  # Can be >1, <1, or 1

        # Define parameter groups with different LRs
        optimiser = torch.optim.Adam(
            [
                {
                    "params": self.model.parameters(),
                    "lr": optim_lr,
                    "weight_decay": optim_weight_decay,
                },  # Default
                {
                    "params": self.model.polymer_fnn.log_diffusion_head.parameters(),
                    "lr": optim_lr * log_diffusion_factor,
                    "weight_decay": optim_weight_decay,
                },  # Increased LR for log_diffusion_head
                {
                    "params": self.model.polymer_fnn.log_rg_head.parameters(),
                    "lr": optim_lr * log_rg_factor,
                    "weight_decay": optim_weight_decay,
                },  # Tuned LR for log_rg_head
            ]
        )

        return optimiser

    def compute_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:

        loss_vect = self.loss_fn(predictions, labels)
        weighted_loss = loss_vect * self.loss_weights
        final_loss = torch.mean(weighted_loss)
        return final_loss

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


class SeparatedGNNTrainerV2(Trainer):

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
        self.loss_weights = self.hyperparams.get(
            "weights", torch.tensor([1, 1, 1, 1, 1, 1, 1])
        )
        log_selection_tensor = self.hyperparams.get(
            "log_selection_tensor", torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
        )
        self.log_transform_helper = LogTransformHelper(
            log_selection_tensor=log_selection_tensor,
            target_transformers=train_dataset.pipeline_manager.target_pipelines,
        )

    def infer(self, loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Runs inference on a given DataLoader and returns predictions.

        Args:
            loader (torch.utils.data.DataLoader): The DataLoader for inference data.

        Returns:
            np.ndarray: Inverse-transformed predictions.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Get predictions
                predictions = self.model(batch)  # Shape: (N, num_outputs)

                all_preds.append(predictions.cpu().numpy())

        all_preds_numpy = [pred for pred in all_preds if pred.shape[0] > 0]

        # Concatenate along the first axis (axis=0) to stack batches into (T, 6)
        preds_numpy = np.concatenate(all_preds_numpy, axis=0)

        inv_preds = self.log_transform_helper.inverse_transform(
            values_to_transform=preds_numpy
        )

        return inv_preds

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
        gnn_hidden_dim = self.hyperparams.get("gnn_hidden_dim", 256)
        gnn_output_dim = self.hyperparams.get("gnn_output_dim", 128)
        gnn_dropout = self.hyperparams.get("gnn_dropout", 0.1)
        gnn_num_heads = self.hyperparams.get("gnn_num_heads", 4)
        multitask_fnn_hidden_dim = self.hyperparams.get("multitask_fnn_hidden_dim", 128)
        multitask_fnn_shared_layer_dim = self.hyperparams.get(
            "multitask_fnn_shared_layer_dim", 128
        )
        multitask_fnn_dropout = self.hyperparams.get("multitask_fnn_dropout", 0.1)
        model = SeparatedGNNSystemV2(
            mpnn_output_dim=mpnn_output_dim,
            mpnn_hidden_dim=mpnn_hidden_dim,
            mpnn_depth=mpnn_depth,
            mpnn_dropout=mpnn_dropout,
            rdkit_selection_tensor=rdkit_selection_tensor,
            molecule_embedding_hidden_dim=molecule_embedding_hidden_dim,
            embedding_dim=embedding_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            gnn_dropout=gnn_dropout,
            gnn_num_heads=gnn_num_heads,
            multitask_fnn_hidden_dim=multitask_fnn_hidden_dim,
            multitask_fnn_shared_layer_dim=multitask_fnn_shared_layer_dim,
            multitask_fnn_dropout=multitask_fnn_dropout,
        )
        return model

    def _configure_optimiser(self) -> torch.optim.Optimizer:
        # Base learning rate and weight decay
        optim_lr = self.hyperparams.get("lr", 1e-3)
        optim_weight_decay = self.hyperparams.get("weight_decay", 0.0)

        # Optuna-tuned scaling factors for specific heads
        log_diffusion_factor = self.hyperparams.get(
            "log_diffusion_factor", 2.0
        )  # >1 increases LR
        log_rg_factor = self.hyperparams.get(
            "log_rg_factor", 1.5
        )  # Can be >1, <1, or 1

        # Define parameter groups with different LRs
        optimiser = torch.optim.Adam(
            [
                {
                    "params": self.model.parameters(),
                    "lr": optim_lr,
                    "weight_decay": optim_weight_decay,
                },  # Default
                {
                    "params": self.model.polymer_fnn.log_diffusion_head.parameters(),
                    "lr": optim_lr * log_diffusion_factor,
                    "weight_decay": optim_weight_decay,
                },  # Increased LR for log_diffusion_head
                {
                    "params": self.model.polymer_fnn.log_rg_head.parameters(),
                    "lr": optim_lr * log_rg_factor,
                    "weight_decay": optim_weight_decay,
                },  # Tuned LR for log_rg_head
            ]
        )

        return optimiser

    def compute_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:

        loss_vect = self.loss_fn(predictions, labels)
        weighted_loss = loss_vect * self.loss_weights
        final_loss = torch.mean(weighted_loss)
        return final_loss

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


class SeparateMorganMPNNPolymerGNNTrainer(Trainer):

    def __init__(
        self,
        n_bits: int,
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
        self.n_bits = n_bits
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
        self.loss_weights = self.hyperparams.get(
            "weights", torch.tensor([1, 1, 1, 1, 1, 1, 1])
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
        gnn_hidden_dim = self.hyperparams.get("gnn_hidden_dim", 256)
        gnn_output_dim = self.hyperparams.get("gnn_output_dim", 128)
        gnn_dropout = self.hyperparams.get("gnn_dropout", 0.1)
        gnn_num_heads = self.hyperparams.get("gnn_num_heads", 4)
        multitask_fnn_hidden_dim = self.hyperparams.get("multitask_fnn_hidden_dim", 128)
        multitask_fnn_shared_layer_dim = self.hyperparams.get(
            "multitask_fnn_shared_layer_dim", 128
        )
        multitask_fnn_dropout = self.hyperparams.get("multitask_fnn_dropout", 0.1)
        model = SeparatedGNNSystemV3Morgan(
            n_bits=self.n_bits,
            mpnn_output_dim=mpnn_output_dim,
            mpnn_hidden_dim=mpnn_hidden_dim,
            mpnn_depth=mpnn_depth,
            mpnn_dropout=mpnn_dropout,
            rdkit_selection_tensor=rdkit_selection_tensor,
            molecule_embedding_hidden_dim=molecule_embedding_hidden_dim,
            embedding_dim=embedding_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            gnn_dropout=gnn_dropout,
            gnn_num_heads=gnn_num_heads,
            multitask_fnn_hidden_dim=multitask_fnn_hidden_dim,
            multitask_fnn_shared_layer_dim=multitask_fnn_shared_layer_dim,
            multitask_fnn_dropout=multitask_fnn_dropout,
        )
        return model

    def _configure_optimiser(self) -> torch.optim.Optimizer:
        # Base learning rate and weight decay
        optim_lr = self.hyperparams.get("lr", 1e-3)
        optim_weight_decay = self.hyperparams.get("weight_decay", 0.0)

        # Optuna-tuned scaling factors for specific heads
        log_diffusion_factor = self.hyperparams.get(
            "log_diffusion_factor", 2.0
        )  # >1 increases LR
        log_rg_factor = self.hyperparams.get(
            "log_rg_factor", 1.5
        )  # Can be >1, <1, or 1

        # Define parameter groups with different LRs
        optimiser = torch.optim.Adam(
            [
                {
                    "params": self.model.parameters(),
                    "lr": optim_lr,
                    "weight_decay": optim_weight_decay,
                },  # Default
                {
                    "params": self.model.polymer_fnn.log_diffusion_head.parameters(),
                    "lr": optim_lr * log_diffusion_factor,
                    "weight_decay": optim_weight_decay,
                },  # Increased LR for log_diffusion_head
                {
                    "params": self.model.polymer_fnn.log_rg_head.parameters(),
                    "lr": optim_lr * log_rg_factor,
                    "weight_decay": optim_weight_decay,
                },  # Tuned LR for log_rg_head
            ]
        )

        return optimiser

    def compute_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:

        loss_vect = self.loss_fn(predictions, labels)
        weighted_loss = loss_vect * self.loss_weights
        final_loss = torch.mean(weighted_loss)
        return final_loss

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


class SeparatedGNNOneOutput(Trainer):

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
        self.loss_weights = self.hyperparams.get(
            "weights", torch.tensor([1, 1, 1, 1, 1, 1, 1])
        )
        log_selection_tensor = self.hyperparams.get(
            "log_selection_tensor", torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
        )
        self.log_transform_helper = LogTransformHelper(
            log_selection_tensor=log_selection_tensor,
            target_transformers=train_dataset.pipeline_manager.target_pipelines,
        )

    def infer(self, loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Runs inference on a given DataLoader and returns predictions.

        Args:
            loader (torch.utils.data.DataLoader): The DataLoader for inference data.

        Returns:
            np.ndarray: Inverse-transformed predictions.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Get predictions
                predictions = self.model(batch)  # Shape: (N, num_outputs)

                all_preds.append(predictions.cpu().numpy())

        all_preds_numpy = [pred for pred in all_preds if pred.shape[0] > 0]

        # Concatenate along the first axis (axis=0) to stack batches into (T, 6)
        preds_numpy = np.concatenate(all_preds_numpy, axis=0)

        inv_preds = self.log_transform_helper.inverse_transform(
            values_to_transform=preds_numpy
        )

        return inv_preds

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
        gnn_hidden_dim = self.hyperparams.get("gnn_hidden_dim", 256)
        gnn_output_dim = self.hyperparams.get("gnn_output_dim", 128)
        gnn_dropout = self.hyperparams.get("gnn_dropout", 0.1)
        gnn_num_heads = self.hyperparams.get("gnn_num_heads", 4)
        multitask_fnn_hidden_dim = self.hyperparams.get("multitask_fnn_hidden_dim", 128)
        multitask_fnn_shared_layer_dim = self.hyperparams.get(
            "multitask_fnn_shared_layer_dim", 128
        )
        multitask_fnn_dropout = self.hyperparams.get("multitask_fnn_dropout", 0.1)
        model = OneOutputGNN(
            mpnn_output_dim=mpnn_output_dim,
            mpnn_hidden_dim=mpnn_hidden_dim,
            mpnn_depth=mpnn_depth,
            mpnn_dropout=mpnn_dropout,
            rdkit_selection_tensor=rdkit_selection_tensor,
            molecule_embedding_hidden_dim=molecule_embedding_hidden_dim,
            embedding_dim=embedding_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            gnn_dropout=gnn_dropout,
            gnn_num_heads=gnn_num_heads,
            multitask_fnn_hidden_dim=multitask_fnn_hidden_dim,
            multitask_fnn_shared_layer_dim=multitask_fnn_shared_layer_dim,
            multitask_fnn_dropout=multitask_fnn_dropout,
        )
        return model

    def _configure_optimiser(self) -> torch.optim.Optimizer:
        # Base learning rate and weight decay
        optim_lr = self.hyperparams.get("lr", 1e-3)
        optim_weight_decay = self.hyperparams.get("weight_decay", 0.0)

        # Optuna-tuned scaling factors for specific heads
        log_diffusion_factor = self.hyperparams.get(
            "log_diffusion_factor", 2.0
        )  # >1 increases LR
        log_rg_factor = self.hyperparams.get(
            "log_rg_factor", 1.5
        )  # Can be >1, <1, or 1

        # Define parameter groups with different LRs
        optimiser = torch.optim.Adam(
            [
                {
                    "params": self.model.parameters(),
                    "lr": optim_lr,
                    "weight_decay": optim_weight_decay,
                },  # Default
                {
                    "params": self.model.polymer_fnn.log_diffusion_head.parameters(),
                    "lr": optim_lr * log_diffusion_factor,
                    "weight_decay": optim_weight_decay,
                },  # Increased LR for log_diffusion_head
                {
                    "params": self.model.polymer_fnn.log_rg_head.parameters(),
                    "lr": optim_lr * log_rg_factor,
                    "weight_decay": optim_weight_decay,
                },  # Tuned LR for log_rg_head
            ]
        )

        return optimiser

    def compute_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:

        loss_vect = self.loss_fn(predictions, labels)
        weighted_loss = loss_vect * self.loss_weights
        final_loss = torch.mean(weighted_loss)
        return final_loss

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


class SeparatedGNNV3(Trainer):

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
        self.loss_weights = self.hyperparams.get(
            "weights", torch.tensor([1, 1, 1, 1, 1, 1, 1])
        )
        log_selection_tensor = self.hyperparams.get(
            "log_selection_tensor", torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
        )
        self.log_transform_helper = LogTransformHelper(
            log_selection_tensor=log_selection_tensor,
            target_transformers=train_dataset.pipeline_manager.target_pipelines,
        )

    def infer(self, loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Runs inference on a given DataLoader and returns predictions.

        Args:
            loader (torch.utils.data.DataLoader): The DataLoader for inference data.

        Returns:
            np.ndarray: Inverse-transformed predictions.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Get predictions
                predictions = self.model(batch)  # Shape: (N, num_outputs)

                all_preds.append(predictions.cpu().numpy())

        all_preds_numpy = [pred for pred in all_preds if pred.shape[0] > 0]

        # Concatenate along the first axis (axis=0) to stack batches into (T, 6)
        preds_numpy = np.concatenate(all_preds_numpy, axis=0)

        inv_preds = self.log_transform_helper.inverse_transform(
            values_to_transform=preds_numpy
        )

        return inv_preds

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
        gnn_hidden_dim = self.hyperparams.get("gnn_hidden_dim", 256)
        gnn_output_dim = self.hyperparams.get("gnn_output_dim", 128)
        gnn_dropout = self.hyperparams.get("gnn_dropout", 0.1)
        gnn_num_heads = self.hyperparams.get("gnn_num_heads", 4)
        multitask_fnn_hidden_dim = self.hyperparams.get("multitask_fnn_hidden_dim", 128)
        multitask_fnn_shared_layer_dim = self.hyperparams.get(
            "multitask_fnn_shared_layer_dim", 128
        )
        multitask_fnn_dropout = self.hyperparams.get("multitask_fnn_dropout", 0.1)
        model = SeparatedGNNSystemV3(
            mpnn_output_dim=mpnn_output_dim,
            mpnn_hidden_dim=mpnn_hidden_dim,
            mpnn_depth=mpnn_depth,
            mpnn_dropout=mpnn_dropout,
            rdkit_selection_tensor=rdkit_selection_tensor,
            molecule_embedding_hidden_dim=molecule_embedding_hidden_dim,
            embedding_dim=embedding_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            gnn_dropout=gnn_dropout,
            gnn_num_heads=gnn_num_heads,
            multitask_fnn_hidden_dim=multitask_fnn_hidden_dim,
            multitask_fnn_shared_layer_dim=multitask_fnn_shared_layer_dim,
            multitask_fnn_dropout=multitask_fnn_dropout,
        )
        return model

    def _configure_optimiser(self) -> torch.optim.Optimizer:
        # Base learning rate and weight decay
        optim_lr = self.hyperparams.get("lr", 1e-3)
        optim_weight_decay = self.hyperparams.get("weight_decay", 0.0)

        # Optuna-tuned scaling factors for specific heads
        log_diffusion_factor = self.hyperparams.get(
            "log_diffusion_factor", 2.0
        )  # >1 increases LR
        log_rg_factor = self.hyperparams.get(
            "log_rg_factor", 1.5
        )  # Can be >1, <1, or 1

        # Define parameter groups with different LRs
        optimiser = torch.optim.Adam(
            [
                {
                    "params": self.model.parameters(),
                    "lr": optim_lr,
                    "weight_decay": optim_weight_decay,
                },  # Default
                {
                    "params": self.model.polymer_fnn.log_diffusion_head.parameters(),
                    "lr": optim_lr * log_diffusion_factor,
                    "weight_decay": optim_weight_decay,
                },  # Increased LR for log_diffusion_head
                {
                    "params": self.model.polymer_fnn.log_rg_head.parameters(),
                    "lr": optim_lr * log_rg_factor,
                    "weight_decay": optim_weight_decay,
                },  # Tuned LR for log_rg_head
            ]
        )

        return optimiser

    def compute_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:

        loss_vect = self.loss_fn(predictions, labels)
        weighted_loss = loss_vect * self.loss_weights
        final_loss = torch.mean(weighted_loss)
        return final_loss

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


class SeparatedGNNChemBertaTrainer(Trainer):

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
        self.loss_weights = self.hyperparams.get(
            "weights", torch.tensor([1, 1, 1, 1, 1, 1, 1])
        )
        log_selection_tensor = self.hyperparams.get(
            "log_selection_tensor", torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
        )
        self.log_transform_helper = LogTransformHelper(
            log_selection_tensor=log_selection_tensor,
            target_transformers=train_dataset.pipeline_manager.target_pipelines,
        )

    def infer(self, loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Runs inference on a given DataLoader and returns predictions.

        Args:
            loader (torch.utils.data.DataLoader): The DataLoader for inference data.

        Returns:
            np.ndarray: Inverse-transformed predictions.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Get predictions
                predictions = self.model(batch)  # Shape: (N, num_outputs)

                all_preds.append(predictions.cpu().numpy())

        all_preds_numpy = [pred for pred in all_preds if pred.shape[0] > 0]

        # Concatenate along the first axis (axis=0) to stack batches into (T, 6)
        preds_numpy = np.concatenate(all_preds_numpy, axis=0)

        inv_preds = self.log_transform_helper.inverse_transform(
            values_to_transform=preds_numpy
        )

        return inv_preds

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
        gnn_hidden_dim = self.hyperparams.get("gnn_hidden_dim", 256)
        gnn_output_dim = self.hyperparams.get("gnn_output_dim", 128)
        gnn_dropout = self.hyperparams.get("gnn_dropout", 0.1)
        gnn_num_heads = self.hyperparams.get("gnn_num_heads", 4)
        multitask_fnn_hidden_dim = self.hyperparams.get("multitask_fnn_hidden_dim", 128)
        multitask_fnn_shared_layer_dim = self.hyperparams.get(
            "multitask_fnn_shared_layer_dim", 128
        )
        multitask_fnn_dropout = self.hyperparams.get("multitask_fnn_dropout", 0.1)
        model = SeparatedGNNSystemChemberta(
            mpnn_output_dim=mpnn_output_dim,
            mpnn_hidden_dim=mpnn_hidden_dim,
            mpnn_depth=mpnn_depth,
            mpnn_dropout=mpnn_dropout,
            rdkit_selection_tensor=rdkit_selection_tensor,
            molecule_embedding_hidden_dim=molecule_embedding_hidden_dim,
            embedding_dim=embedding_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            gnn_dropout=gnn_dropout,
            gnn_num_heads=gnn_num_heads,
            multitask_fnn_hidden_dim=multitask_fnn_hidden_dim,
            multitask_fnn_shared_layer_dim=multitask_fnn_shared_layer_dim,
            multitask_fnn_dropout=multitask_fnn_dropout,
        )
        return model

    def _configure_optimiser(self) -> torch.optim.Optimizer:
        # Base learning rate and weight decay
        optim_lr = self.hyperparams.get("lr", 1e-3)
        optim_weight_decay = self.hyperparams.get("weight_decay", 0.0)

        # Optuna-tuned scaling factors for specific heads
        log_diffusion_factor = self.hyperparams.get(
            "log_diffusion_factor", 2.0
        )  # >1 increases LR
        log_rg_factor = self.hyperparams.get(
            "log_rg_factor", 1.5
        )  # Can be >1, <1, or 1

        # Define parameter groups with different LRs
        optimiser = torch.optim.Adam(
            [
                {
                    "params": self.model.parameters(),
                    "lr": optim_lr,
                    "weight_decay": optim_weight_decay,
                },  # Default
                {
                    "params": self.model.polymer_fnn.log_diffusion_head.parameters(),
                    "lr": optim_lr * log_diffusion_factor,
                    "weight_decay": optim_weight_decay,
                },  # Increased LR for log_diffusion_head
                {
                    "params": self.model.polymer_fnn.log_rg_head.parameters(),
                    "lr": optim_lr * log_rg_factor,
                    "weight_decay": optim_weight_decay,
                },  # Tuned LR for log_rg_head
            ]
        )

        return optimiser

    def compute_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:

        loss_vect = self.loss_fn(predictions, labels)
        weighted_loss = loss_vect * self.loss_weights
        final_loss = torch.mean(weighted_loss)
        return final_loss

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


class SeparatedGNNTrainerWithJSON(Trainer):

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
        self.loss_weights = self.hyperparams.get(
            "weights", torch.tensor([1, 1, 1, 1, 1, 1, 1])
        )
        log_selection_tensor = self.hyperparams.get(
            "log_selection_tensor", torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
        )
        self.log_transform_helper = LogTransformHelper(
            log_selection_tensor=log_selection_tensor,
            target_transformers=train_dataset.pipeline_manager.target_pipelines,
        )

    def infer(self, loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Runs inference on a given DataLoader and returns predictions.

        Args:
            loader (torch.utils.data.DataLoader): The DataLoader for inference data.

        Returns:
            np.ndarray: Inverse-transformed predictions.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Get predictions
                predictions = self.model(batch)  # Shape: (N, num_outputs)

                all_preds.append(predictions.cpu().numpy())

        all_preds_numpy = [pred for pred in all_preds if pred.shape[0] > 0]

        # Concatenate along the first axis (axis=0) to stack batches into (T, 6)
        preds_numpy = np.concatenate(all_preds_numpy, axis=0)

        inv_preds = self.log_transform_helper.inverse_transform(
            values_to_transform=preds_numpy
        )

        return inv_preds

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
        gnn_hidden_dim = self.hyperparams.get("gnn_hidden_dim", 256)
        gnn_output_dim = self.hyperparams.get("gnn_output_dim", 128)
        gnn_dropout = self.hyperparams.get("gnn_dropout", 0.1)
        gnn_num_heads = self.hyperparams.get("gnn_num_heads", 4)
        multitask_fnn_hidden_dim = self.hyperparams.get("multitask_fnn_hidden_dim", 128)
        multitask_fnn_shared_layer_dim = self.hyperparams.get(
            "multitask_fnn_shared_layer_dim", 128
        )
        multitask_fnn_dropout = self.hyperparams.get("multitask_fnn_dropout", 0.1)
        model = SeparatedGNNSystemV2(
            mpnn_output_dim=mpnn_output_dim,
            mpnn_hidden_dim=mpnn_hidden_dim,
            mpnn_depth=mpnn_depth,
            mpnn_dropout=mpnn_dropout,
            rdkit_selection_tensor=rdkit_selection_tensor,
            molecule_embedding_hidden_dim=molecule_embedding_hidden_dim,
            embedding_dim=embedding_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            gnn_dropout=gnn_dropout,
            gnn_num_heads=gnn_num_heads,
            multitask_fnn_hidden_dim=multitask_fnn_hidden_dim,
            multitask_fnn_shared_layer_dim=multitask_fnn_shared_layer_dim,
            multitask_fnn_dropout=multitask_fnn_dropout,
        )
        return model

    def _configure_optimiser(self) -> torch.optim.Optimizer:
        # Base learning rate and weight decay
        optim_lr = self.hyperparams.get("lr", 1e-3)
        optim_weight_decay = self.hyperparams.get("weight_decay", 0.0)

        # Optuna-tuned scaling factors for specific heads
        log_diffusion_factor = self.hyperparams.get(
            "log_diffusion_factor", 2.0
        )  # >1 increases LR
        log_rg_factor = self.hyperparams.get(
            "log_rg_factor", 1.5
        )  # Can be >1, <1, or 1

        # Define parameter groups with different LRs
        optimiser = torch.optim.Adam(
            [
                {
                    "params": self.model.parameters(),
                    "lr": optim_lr,
                    "weight_decay": optim_weight_decay,
                },  # Default
                {
                    "params": self.model.polymer_fnn.log_diffusion_head.parameters(),
                    "lr": optim_lr * log_diffusion_factor,
                    "weight_decay": optim_weight_decay,
                },  # Increased LR for log_diffusion_head
                {
                    "params": self.model.polymer_fnn.log_rg_head.parameters(),
                    "lr": optim_lr * log_rg_factor,
                    "weight_decay": optim_weight_decay,
                },  # Tuned LR for log_rg_head
            ]
        )

        return optimiser

    def compute_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:

        loss_vect = self.loss_fn(predictions, labels)
        weighted_loss = loss_vect * self.loss_weights
        final_loss = torch.mean(weighted_loss)
        return final_loss

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

    def final_eval(
        self,
        loader: torch.utils.data.DataLoader,
        save_preds_path: Optional[str] = None,
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

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_labels, all_preds = self.inverse_transform(all_labels, all_preds)

        # Optional JSON saving for parity plotting
        if save_preds_path is not None:
            preds_labels = [
                {"true": true.tolist(), "pred": pred.tolist()}
                for true, pred in zip(all_labels, all_preds)
            ]
            with open(save_preds_path, "w") as f:
                json.dump(preds_labels, f, indent=2)

            logger.info("Predictions saved to %s", save_preds_path)

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
            self.final_eval(
                self.val_loader,
                save_preds_path=f"{self.save_results_dir}/val_preds.json",
            )
        logger.info("Training Completed.")


from models.separate_mpnn_model.full_separate_mpnn_v3 import SeparatedGNNSystemFG


class FGGNNTrainerWithJSON(Trainer):

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
        self.loss_weights = self.hyperparams.get(
            "weights", torch.tensor([1, 1, 1, 1, 1, 1, 1])
        )
        log_selection_tensor = self.hyperparams.get(
            "log_selection_tensor", torch.tensor([0, 0, 0, 0, 0, 0, 0, 0])
        )
        self.log_transform_helper = LogTransformHelper(
            log_selection_tensor=log_selection_tensor,
            target_transformers=train_dataset.pipeline_manager.target_pipelines,
        )

    def infer(self, loader: torch.utils.data.DataLoader) -> np.ndarray:
        """
        Runs inference on a given DataLoader and returns predictions.

        Args:
            loader (torch.utils.data.DataLoader): The DataLoader for inference data.

        Returns:
            np.ndarray: Inverse-transformed predictions.
        """
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in loader:
                batch = {
                    k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for k, v in batch.items()
                }

                # Get predictions
                predictions = self.model(batch)  # Shape: (N, num_outputs)

                all_preds.append(predictions.cpu().numpy())

        all_preds_numpy = [pred for pred in all_preds if pred.shape[0] > 0]

        # Concatenate along the first axis (axis=0) to stack batches into (T, 6)
        preds_numpy = np.concatenate(all_preds_numpy, axis=0)

        inv_preds = self.log_transform_helper.inverse_transform(
            values_to_transform=preds_numpy
        )

        return inv_preds

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
        gnn_hidden_dim = self.hyperparams.get("gnn_hidden_dim", 256)
        gnn_output_dim = self.hyperparams.get("gnn_output_dim", 128)
        gnn_dropout = self.hyperparams.get("gnn_dropout", 0.1)
        gnn_num_heads = self.hyperparams.get("gnn_num_heads", 4)
        multitask_fnn_hidden_dim = self.hyperparams.get("multitask_fnn_hidden_dim", 128)
        multitask_fnn_shared_layer_dim = self.hyperparams.get(
            "multitask_fnn_shared_layer_dim", 128
        )
        multitask_fnn_dropout = self.hyperparams.get("multitask_fnn_dropout", 0.1)
        fg_n_bits = self.hyperparams.get("fg_n_bits", 64)
        model = SeparatedGNNSystemFG(
            mpnn_output_dim=mpnn_output_dim,
            mpnn_hidden_dim=mpnn_hidden_dim,
            mpnn_depth=mpnn_depth,
            mpnn_dropout=mpnn_dropout,
            rdkit_selection_tensor=rdkit_selection_tensor,
            molecule_embedding_hidden_dim=molecule_embedding_hidden_dim,
            embedding_dim=embedding_dim,
            gnn_hidden_dim=gnn_hidden_dim,
            gnn_output_dim=gnn_output_dim,
            gnn_dropout=gnn_dropout,
            gnn_num_heads=gnn_num_heads,
            multitask_fnn_hidden_dim=multitask_fnn_hidden_dim,
            multitask_fnn_shared_layer_dim=multitask_fnn_shared_layer_dim,
            multitask_fnn_dropout=multitask_fnn_dropout,
            fg_n_bits=fg_n_bits,
        )
        return model

    def _configure_optimiser(self) -> torch.optim.Optimizer:
        # Base learning rate and weight decay
        optim_lr = self.hyperparams.get("lr", 1e-3)
        optim_weight_decay = self.hyperparams.get("weight_decay", 0.0)

        # Optuna-tuned scaling factors for specific heads
        log_diffusion_factor = self.hyperparams.get(
            "log_diffusion_factor", 2.0
        )  # >1 increases LR
        log_rg_factor = self.hyperparams.get(
            "log_rg_factor", 1.5
        )  # Can be >1, <1, or 1

        # Define parameter groups with different LRs
        optimiser = torch.optim.Adam(
            [
                {
                    "params": self.model.parameters(),
                    "lr": optim_lr,
                    "weight_decay": optim_weight_decay,
                },  # Default
                {
                    "params": self.model.polymer_fnn.log_diffusion_head.parameters(),
                    "lr": optim_lr * log_diffusion_factor,
                    "weight_decay": optim_weight_decay,
                },  # Increased LR for log_diffusion_head
                {
                    "params": self.model.polymer_fnn.log_rg_head.parameters(),
                    "lr": optim_lr * log_rg_factor,
                    "weight_decay": optim_weight_decay,
                },  # Tuned LR for log_rg_head
            ]
        )

        return optimiser

    def compute_loss(
        self, predictions: torch.Tensor, labels: torch.Tensor
    ) -> torch.Tensor:

        loss_vect = self.loss_fn(predictions, labels)
        weighted_loss = loss_vect * self.loss_weights
        final_loss = torch.mean(weighted_loss)
        return final_loss

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

    def final_eval(
        self,
        loader: torch.utils.data.DataLoader,
        save_preds_path: Optional[str] = None,
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

        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        all_labels, all_preds = self.inverse_transform(all_labels, all_preds)

        # Optional JSON saving for parity plotting
        if save_preds_path is not None:
            preds_labels = [
                {"true": true.tolist(), "pred": pred.tolist()}
                for true, pred in zip(all_labels, all_preds)
            ]
            with open(save_preds_path, "w") as f:
                json.dump(preds_labels, f, indent=2)

            logger.info("Predictions saved to %s", save_preds_path)

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
            self.final_eval(
                self.val_loader,
                save_preds_path=f"{self.save_results_dir}/val_preds.json",
            )
        logger.info("Training Completed.")
