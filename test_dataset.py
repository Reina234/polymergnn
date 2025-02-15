from chemprop.nn import BondMessagePassing, NormAggregation, RegressionFFN
import torch
from chemprop.models.model import MPNN
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
from dataloaders.unbatched_V1_datasets import FGFeaturisedMolecularDataset
from tools.model.loss import MSELossStrategy
from tools.model.trainer import GenericTrainer
from tools.model.model_factory import ModelFactory
from dataloaders.unbatched_V1_datasets import PolymerBertDataset
from tools.smiles_transformers import NoSmilesTransform
from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("tests/freesolv/FreeSolv_SAMPL.csv")
df = df.drop(df.columns[[0, 1]], axis=1)  # Drop unnecessary columns

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

train_loader = PolymerBertDataset(
    data=train_df, monomer_smiles_transformer=NoSmilesTransform(), target_columns=[1, 2]
)
print(train_loader[0])

dataloader = DataLoader(
    train_loader,
    batch_size=32,  # Batching datapoints, but not merging molecules
    shuffle=True,
    collate_fn=train_loader.collate_fn,  # Use custom function
)

print(next(iter(dataloader)))
