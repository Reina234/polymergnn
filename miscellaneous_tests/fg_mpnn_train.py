import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import torch.optim as optim
from tools.data_processing.molecular_dataset import MolecularDataset, custom_collate_fn
from models.chemprop_fg_hierarchical import ChemPropFGHierarchicalModel

from tools.model.model_factory import ModelFactory
from tools.model.hyperparameter_tuner import HyperparameterTuner

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("data/freesolv/FreeSolv_SAMPL.csv")
df = df.drop(df.columns[[0, 1]], axis=1)  # Drop unnecessary columns

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
train_loader = MolecularDataset(train_df, target_columns=[1, 2])

val_loader = MolecularDataset(val_df, target_columns=[1, 2])
test_loader = MolecularDataset(test_df, target_columns=[1, 2])
# model = ChemPropFGHierarchicalModel(
#    mpnn_dim=256,
#    fg_dim=128,
#    global_dim=256,
#    final_dim=256,
#    st_heads=4,
#    st_layers=1,
#    temperature=0.5,
#    dropout_prob=0.2,
#    contrastive=True,
# ).to(device)

# Training setup
# optimizer = optim.Adam(model.parameters(), lr=1e-3)
# loss_strategy = MSEWithContrastiveLoss()

# trainer = GenericTrainer(
#    model, optimizer, loss_strategy, train_loader, val_loader, test_loader, device
# )
# trainer.train(epochs=50)
# trainer.evaluate(test_loader, save_results=True)

# Define hyperparameter search space
param_grid = {
    "mpnn_dim": [128, 256],
    "fg_dim": [128, 256],
    "global_dim": [128, 256],
    "final_dim": [128, 256],
    "st_heads": [4, 8],
    "st_layers": [1, 2],
    "temperature": [0.5],
    "dropout_prob": [0.1, 0.2],
    "contrastive": [True, False],
    "lr": [1e-3, 1e-4],
}

extra_args = {"target_dim": 2}

chemprop_factory = ModelFactory(
    ChemPropFGHierarchicalModel,
    model_param_keys=[
        "mpnn_dim",
        "fg_dim",
        "global_dim",
        "final_dim",
        "st_heads",
        "st_layers",
        "temperature",
        "dropout_prob",
        "contrastive",
    ],
    extra_args=extra_args,
)
# chemprop_factory = ModelFactory(
#    model_class=ChemPropFGHierarchicalModel,
#    model_param_keys=list(param_grid.keys()),
#    extra_args=extra_args,
# )
tuner = HyperparameterTuner(
    chemprop_factory,
    param_grid,
    train_loader,
    val_loader,
    test_loader,
    device,
    search_type="grid",
)
best_model_config = tuner.tune()
