import torch
import os
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from typing import List, Optional, Tuple, Dict
import matplotlib.pyplot as plt
import logging
from training.loss import LossStrategy
from abc import ABC, abstractmethod

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Trainer(ABC):
    def __init__(
        self,
        model: torch.nn.Module,
        optimiser: torch.optim.Optimizer,
        loss_strategy: LossStrategy,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        test_loader: torch.utils.data.DataLoader,
        device: torch.device,
        hyperparams: dict,
        log_dir="logs",
        use_tensorboard=True,
        save_results_dir="results",
        track_learning_curve=False,
        figure_size=(8, 6),
    ):
        self.optimizer = optimiser
        self.model = model
        self.loss_strategy = loss_strategy
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        self.hyperparams = hyperparams
        self.save_results_dir = save_results_dir
        self.track_learning_curve = track_learning_curve
        self.num_batches = len(self.train_loader)
        self.figure_size = figure_size

        os.makedirs(save_results_dir, exist_ok=True)

        if self.use_tensorboard:
            self.writer = SummaryWriter(log_dir)
            self.writer.add_hparams(hyperparams, {})

        self.train_losses = [] if self.track_learning_curve else None
        self.val_losses = [] if self.track_learning_curve else None

    @abstractmethod
    def _get_batch_loss(self, batch) -> torch.Tensor:
        pass

    def _train_batch(self, batch) -> torch.Tensor:
        self.optimizer.zero_grad()
        loss = self._get_batch_loss(batch)
        loss.backward()
        self.optimizer.step()
        return loss

    def _train_epoch(
        self, epoch: int, use_tensorboard: bool
    ) -> Dict[Dict[str, float], Dict[str, float]]:
        epoch_loss = 0.0
        for batch in self.train_loader:
            loss = self._train_batch(batch)
            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / self.num_batches
        self._write_scalar_to_tensorboard(
            {"Loss/Training": avg_epoch_loss}, epoch, use_tensorboard=use_tensorboard
        )
        self._add_to_learning_curve(avg_epoch_loss, is_train=True)
        return avg_epoch_loss, None

    def _add_to_learning_curve(self, loss: float, is_train: bool):
        if not self.track_learning_curve:
            return
        if is_train:
            self.train_losses.append(loss)
        else:
            self.val_losses.append(loss)

    def _write_scalar_to_tensorboard(
        self, dict_to_write: Dict[str, float], epoch: int, use_tensorboard: bool
    ) -> None:
        if use_tensorboard:
            for key, value in dict_to_write.items():
                self.writer.add_scalar(key, value, epoch)

    def _normalise_loss_dict(
        self, loss_dict: Dict[str, float], keys_to_normalise: Optional[List[str]]
    ) -> Dict[str, float]:
        if keys_to_normalise is None:
            keys_to_normalise = loss_dict.keys()
        for key in keys_to_normalise:
            loss_dict[key] /= self.num_batches

        return loss_dict

    def train(self, epochs=50):
        self.model.train()
        epoch_losses = []
        val_losses = []
        for epoch in tqdm(range(epochs), desc="Training Progress"):
            epoch_loss, additional_train_losses = self._train_epoch(
                epoch=epoch, use_tensorboard=self.use_tensorboard
            )
            epoch_losses.append(epoch_loss)

            val_loss, additional_val_losses = self.evaluate(
                self.val_loader, epoch=epoch, use_tensorboard=self.use_tensorboard
            )
            val_losses.append(val_loss)

            self._flush_tensorboard(use_tensorboard=self.use_tensorboard)

        self._write_hparams_to_tensorboard(
            additional_train_losses=additional_train_losses,
            additional_val_losses=additional_val_losses,
            use_tensorboard=self.use_tensorboard,
        )
        self.plot_learning_curve(track_curve=self.track_learning_curve)

    def _write_hparams_to_tensorboard(
        self,
        additional_train_losses: Dict[str, float],
        additional_val_losses: Dict[str, float],
        use_tensorboard: bool,
    ) -> None:
        if not use_tensorboard:
            return
        self.writer.add_hparams(
            self.hyperparams,
            self._make_final_losses(
                additional_train_losses=additional_train_losses,
                additional_val_losses=additional_val_losses,
            ),
        )
        self.writer.close()

    @abstractmethod
    def _make_final_losses(
        self,
        additional_train_losses: Dict[str, float],
        additional_val_losses: Dict[str, float],
    ) -> Dict[str, float]:
        pass

    def _flush_tensorboard(self, use_tensorboard: bool):
        if use_tensorboard:
            self.writer.flush()

    @abstractmethod
    def evaluate(
        self, val_loader, epoch: int, use_tensorboard: bool
    ) -> Tuple[float, Dict[str, float]]:
        pass

    @abstractmethod
    def forward_pass(self, batch):
        pass

    def plot_learning_curve(self, track_curve: bool):
        """Plots training vs validation loss over epochs."""
        if not track_curve:
            return
        figname = self._create_learning_curve_figname()
        fig_path = os.path.join(self.save_results_dir, figname)

        plt.figure(figsize=self.figure_size)
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
        plt.savefig(fig_path)  # Save plot
        plt.show()

    def _create_learning_curve_figname(self):

        figname = "_".join(
            [
                f"{key}={value}"
                for key, value in self.hyperparams.items()
                if key not in ["device"]
            ]
        )
        return f"{figname}_learning_curve.png"
