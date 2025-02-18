import torch
import os
import logging
from typing import Optional
import pandas as pd

# from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from training.trainer import Trainer  # Or your Trainer subclass (e.g., MoleculeTrainer)
from training.loss import MSELossStrategy
from training.model_factory import ModelFactory
from torch.utils.tensorboard import SummaryWriter
from training.batched_dataset import PolymerDataset

logger = logging.getLogger(__name__)


class HyperparameterTuner:

    def __init__(
        self,
        model_factory: ModelFactory,
        train_dataset: PolymerDataset,
        val_dataset: PolymerDataset,
        test_dataset: PolymerDataset,
        device: torch.device,
        trainer: Trainer,  # Expecting a Trainer subclass
        search_space: dict,
        max_trials: int = None,
        use_tensorboard: bool = True,
        save_results_dir: str = "results/hparams_tuning",
        trial_name: str = "trial",
        shared_writer: Optional[SummaryWriter] = None,
        additional_info: Optional[dict] = None,
    ):
        """
        Args:
            model_factory: Creates a model instance from hyperparameters.
            train_loader, val_loader, test_loader: DataLoaders.
            device: Torch device.
            trainer: Trainer subclass (e.g., MoleculeTrainer).
            search_space: Hyperparameter grid.
            max_trials: Maximum number of trials.
            use_tensorboard: Whether to log to TensorBoard.
            save_results_dir: Directory for results.
            shared_writer: A shared SummaryWriter for all trials (optional).
        """

        self.output_dim = train_dataset.targets.shape[1]
        self.model_factory = model_factory

        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.device = device
        self.trainer = trainer
        self.search_space = search_space
        self.max_trials = max_trials
        self.use_tensorboard = use_tensorboard
        self.save_results_dir = save_results_dir
        self.trial_name = trial_name
        os.makedirs(save_results_dir, exist_ok=True)
        self.shared_writer = shared_writer
        self.additional_info = additional_info

    def run(self):
        param_grid = list(ParameterGrid(self.search_space))
        if self.max_trials:
            param_grid = param_grid[: self.max_trials]

        results = []

        for i, params in enumerate(param_grid):
            logger.info("Trial %d/%d with params: %s", i + 1, len(param_grid), params)

            # Create the model using the factory.
            model = self.model_factory.create_model(params).to(self.device)

            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params.get("weight_decay", 0.0),
            )
            loss_strategy = MSELossStrategy()

            batch_size = params.get("batch_size", 32)
            if self.additional_info:
                params.update(self.additional_info)
            # Instantiate the Trainer with shared tensorboard writer and updated flush/close settings.
            trainer_instance = self.trainer(
                model=model,
                optimiser=optimizer,
                loss_strategy=loss_strategy,
                train_dataset=self.train_dataset,
                val_dataset=self.val_dataset,
                test_dataset=self.test_dataset,
                batch_size=batch_size,
                device=self.device,
                hyperparams=params,
                log_dir=f"logs/{self.trial_name}",
                use_tensorboard=self.use_tensorboard,
                save_results_dir=self.save_results_dir,
                track_learning_curve=True,
                writer=self.shared_writer,
                flush_tensorboard_each_epoch=False,
                close_writer_on_finish=False,
                additional_info=params,
            )

            trainer_instance.train(epochs=params["epochs"])
            trainer_instance.test()

            trial_result = {
                "params": params,
                "test_loss": trainer_instance.final_test_loss,
                "metrics": trainer_instance.final_test_metrics,
            }
            results.append(trial_result)
            logger.info("Trial %d Results: %s", i + 1, trial_result)

        self._save_results(results)
        logger.info("Hyperparameter tuning complete.")
        return results

    def _save_results(self, results):
        results_df = pd.DataFrame(
            [
                {
                    **r["params"],
                    "test_loss": r["test_loss"],
                    **r["metrics"],
                }
                for r in results
            ]
        )
        results_file = os.path.join(self.save_results_dir, "hparams_results.csv")
        results_df.to_csv(results_file, index=False)
        logger.info("Saved hyperparameter tuning results to %s", results_file)
