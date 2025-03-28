from training.refactored_trainer import Trainer
from training.refactored_batched_dataset import SimpleDataset
from models.benchmark_models.simple_fnn import SimpleFNN
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Tuple, List, Optional
from tools.log_transform_helper import LogTransformHelper
import logging
from abc import ABC, abstractmethod
from models.benchmark_models.mpnn import JustMPNN
from models.benchmark_models.simple_cnn import SimpleCNN
from models.benchmark_models.simple_rnn import SimpleRNN


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BenchmarkTrainer(Trainer, ABC):
    def __init__(
        self,
        train_dataset: SimpleDataset,
        val_dataset: Optional[SimpleDataset],
        test_dataset: Optional[SimpleDataset],
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

        self.output_dim = len(train_dataset.target_columns) // 2
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

    @abstractmethod
    def configure_model(self) -> torch.nn.Module:
        pass

    def forward_pass(self, batch) -> Tuple[torch.Tensor, torch.Tensor]:

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


class MPNNTrainer(BenchmarkTrainer):

    def __init__(
        self,
        train_dataset: SimpleDataset,
        val_dataset: Optional[SimpleDataset],
        test_dataset: Optional[SimpleDataset],
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

    def configure_model(self) -> torch.nn.Module:
        mpnn_output_dim = self.hyperparams.get("mpnn_output_dim", 128)
        mpnn_hidden_dim = self.hyperparams.get("mpnn_hidden_dim", 96)
        mpnn_depth = self.hyperparams.get("mpnn_depth", 2)
        mpnn_dropout = self.hyperparams.get("mpnn_dropout", 0.327396910351)
        multitask_fnn_hidden_dim = self.hyperparams.get("multitask_fnn_hidden_dim", 96)
        model = JustMPNN(
            mpnn_output_dim=mpnn_output_dim,
            mpnn_hidden_dim=mpnn_hidden_dim,
            mpnn_depth=mpnn_depth,
            mpnn_dropout=mpnn_dropout,
            multitask_fnn_hidden_dim=multitask_fnn_hidden_dim,
        )
        return model


class FNNTrainer(BenchmarkTrainer):
    def __init__(
        self,
        train_dataset: SimpleDataset,
        val_dataset: Optional[SimpleDataset],
        test_dataset: Optional[SimpleDataset],
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

    def configure_model(self) -> torch.nn.Module:

        hidden_dim = self.hyperparams.get("hidden_dim", 256)
        dropout = self.hyperparams.get("dropout", 0.2)
        fingerprint_n_bits = self.hyperparams.get("n_bits", 2048)
        model = SimpleFNN(
            input_dim=2 * fingerprint_n_bits + 3,
            dropout=dropout,
            hidden_dim=hidden_dim,
            output_dim=self.output_dim,
        )
        return model


class RNNTrainer(BenchmarkTrainer):
    def __init__(
        self,
        train_dataset: SimpleDataset,
        val_dataset: Optional[SimpleDataset],
        test_dataset: Optional[SimpleDataset],
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

    def configure_model(self) -> torch.nn.Module:

        hidden_dim = self.hyperparams.get("hidden_dim", 256)
        dropout = self.hyperparams.get("dropout", 0.2)
        fingerprint_n_bits = self.hyperparams.get("n_bits", 2048)
        model = SimpleRNN(
            input_dim=fingerprint_n_bits + 3,
            hidden_dim=hidden_dim,
            output_dim=self.output_dim,
        )
        return model


class FNNTrainerRDKit(BenchmarkTrainer):
    def __init__(
        self,
        train_dataset: SimpleDataset,
        val_dataset: Optional[SimpleDataset],
        test_dataset: Optional[SimpleDataset],
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

    def configure_model(self) -> torch.nn.Module:

        hidden_dim = self.hyperparams.get("hidden_dim", 256)
        dropout = self.hyperparams.get("dropout", 0.2)
        fingerprint_n_bits = self.hyperparams.get("n_bits", 2048)
        model = SimpleFNN(
            input_dim=6 * 2 + 3,
            dropout=dropout,
            hidden_dim=hidden_dim,
            output_dim=self.output_dim,
        )
        return model
