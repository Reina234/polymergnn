import torch
import os
import logging
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.model_selection import ParameterGrid
from training.trainer import Trainer  # Or your Trainer subclass (e.g., MoleculeTrainer)
from training.loss import MSELossStrategy
from training.model_factory import ModelFactory

logger = logging.getLogger(__name__)


class HyperparameterTuner:

    def __init__(
        self,
        model_factory: ModelFactory,
        train_loader: DataLoader,
        val_loader: DataLoader,
        test_loader: DataLoader,
        device: torch.device,
        trainer: Trainer,
        search_space: dict,
        max_trials: int = None,
        use_tensorboard: bool = True,
        save_results_dir: str = "results/hparams_tuning",
    ):
        """
        Args:
            model_factory: Function to create a model instance from a hyperparameter dict.
            train_loader, val_loader, test_loader: DataLoaders.
            device: Torch device.
            search_space: Dict where keys are hyperparameter names and values are lists.
            max_trials: Maximum number of trials (optional).
            use_tensorboard: Whether to log to TensorBoard.
            save_results_dir: Directory to save results.
        """
        self.model_factory = model_factory
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.trainer = trainer
        self.search_space = search_space
        self.max_trials = max_trials
        self.use_tensorboard = use_tensorboard
        self.save_results_dir = save_results_dir
        os.makedirs(save_results_dir, exist_ok=True)

    def run(self):
        param_grid = list(ParameterGrid(self.search_space))
        if self.max_trials:
            param_grid = param_grid[: self.max_trials]

        results = []

        for i, params in enumerate(param_grid):
            logger.info("Trial %d/%d with params: %s", i + 1, len(param_grid), params)

            # Create the model using our factory function.
            model = self.model_factory.create_model(params).to(self.device)

            # Configure optimizer with learning rate and weight decay.
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=params["lr"],
                weight_decay=params.get("weight_decay", 0.0),
            )
            loss_strategy = MSELossStrategy()

            # Instantiate your Trainer (replace GenericTrainer with your actual trainer class if needed)
            trainer = self.trainer(
                model=model,
                optimiser=optimizer,
                loss_strategy=loss_strategy,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                test_loader=self.test_loader,
                device=self.device,
                hyperparams=params,
                log_dir=f"logs/trial_{i+1}",
                use_tensorboard=self.use_tensorboard,
                save_results_dir=self.save_results_dir,
                track_learning_curve=True,
            )

            trainer.train(epochs=params["epochs"])
            trainer.test()

            # For demonstration, we assume test loss is available in trainer.test() output.
            trial_result = {
                "params": params,
                "test_loss": getattr(trainer, "final_test_loss", None),
                "metrics": getattr(trainer, "final_test_metrics", {}),
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
