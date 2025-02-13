import itertools
import random
import logging
from tools.model.trainer import GenericTrainer
from tools.model.model_factory import ModelFactory
from tools.model.loss import MSEWithContrastiveLoss

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HyperparameterTuner:
    def __init__(
        self,
        model_factory: ModelFactory,
        param_grid,
        train_loader,
        val_loader,
        test_loader,
        device,
        enable_tuning=True,
        search_type="grid",
        num_samples=5,
        fixed_params=None,
    ):
        """
        Hyperparameter tuner that dynamically creates models and passes only necessary parameters.
        """
        self.model_factory = model_factory
        self.param_grid = param_grid
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = device
        self.enable_tuning = enable_tuning
        self.search_type = search_type
        self.num_samples = num_samples
        self.fixed_params = fixed_params
        self.results = {}

    def generate_configs(self):
        """Generate hyperparameter configurations (Grid or Random Search)."""
        if not self.enable_tuning and self.fixed_params:
            return [self.fixed_params]

        param_keys = self.param_grid.keys()
        param_combinations = list(itertools.product(*self.param_grid.values()))

        if self.search_type == "random":
            param_combinations = random.sample(
                param_combinations, min(self.num_samples, len(param_combinations))
            )

        return [dict(zip(param_keys, values)) for values in param_combinations]

    def tune(self):
        """Train a single model if tuning is disabled, otherwise perform hyperparameter tuning."""
        configs = self.generate_configs()
        best_model = None
        best_loss = float("inf")

        for config in configs:
            logger.info(f"Training model with params: {config}")

            # Create model dynamically
            model = self.model_factory.create_model(config, self.device)

            # Trainer (🔥 Now handles optimizer internally)
            trainer = GenericTrainer(
                model,
                MSEWithContrastiveLoss(),
                self.train_loader,
                self.val_loader,
                self.test_loader,
                self.device,
                hyperparams=config,
                log_dir=f"logs/hparam_{config['mpnn_dim']}",
            )

            # Train & Evaluate
            trainer.train(epochs=50)
            val_loss = trainer.evaluate(self.val_loader)

            self.results[str(config)] = val_loss

            if val_loss < best_loss:
                best_loss = val_loss
                best_model = config

            if (
                not self.enable_tuning
            ):  # 🔥 If tuning is disabled, stop after one iteration
                break

        logger.info(f"Best Model: {best_model} with Validation Loss: {best_loss}")
        return best_model
