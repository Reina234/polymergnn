import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
from tools.data_processing.molecular_dataset import MolecularDataset, custom_collate_fn
from models.chemprop2_hierarchical import ChemPropFGVer2
from tools.model.model_factory import ModelFactory
from tools.model.trainer import GenericTrainer
from tools.model.loss import MSEWithContrastiveLoss
from sklearn.preprocessing import StandardScaler

# ✅ Normalize the target columns

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv("tests/freesolv/FreeSolv_SAMPL.csv")
df = df.drop(df.columns[[0, 1]], axis=1)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)

# Convert datasets to DataLoader
train_loader = MolecularDataset(train_df, target_columns=[1])
val_loader = MolecularDataset(val_df, target_columns=[1])

test_loader = MolecularDataset(test_df, target_columns=[1])

# Define Fixed Hyperparameters
fixed_hyperparams = {
    "mpnn_dim": 256,
    "fg_dim": 128,
    "global_dim": 256,
    "final_dim": 256,
    "st_heads": 4,
    "st_layers": 1,
    "temperature": 0.5,
    "dropout_prob": 0.4,
    "contrastive": False,
    "lr": 1e-3,
}

extra_args = {"target_dim": 2}

# Create Model
chemprop_factory = ModelFactory(
    ChemPropFGVer2,
    model_param_keys=list(fixed_hyperparams.keys()),
    extra_args=extra_args,
)
model = chemprop_factory.create_model(fixed_hyperparams, device)

# Train Model Using GenericTrainer
trainer = GenericTrainer(
    model=model,
    loss_strategy=MSEWithContrastiveLoss(),
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    hyperparams=fixed_hyperparams,
    use_tensorboard=True,
    track_learning_curve=True,
)

trainer.train(epochs=50)  # Train
trainer.evaluate(test_loader, save_results=True)  # Evaluate

print("✅ Training complete. Learning curve saved in results/learning_curve.png")
print("✅ View live metrics in TensorBoard with: tensorboard --logdir logs")
