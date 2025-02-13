from chemprop.nn import BondMessagePassing, NormAggregation, RegressionFFN
import torch
from chemprop.models.model import MPNN
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tools.data_processing.molecular_dataset import MolecularDataset
from tools.model.loss import MSELossStrategy
from tools.model.trainer import GenericTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("tests/freesolv/FreeSolv_SAMPL.csv")
df = df.drop(df.columns[[0, 1]], axis=1)  # Drop unnecessary columns

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
train_loader = MolecularDataset(train_df, target_columns=[1, 2])

val_loader = MolecularDataset(val_df, target_columns=[1, 2])
test_loader = MolecularDataset(test_df, target_columns=[1, 2])

mp = BondMessagePassing()
agg = NormAggregation()
ffn = RegressionFFN()

mpnn_model = MPNN(mp, agg, ffn).to(device)

optimizer = optim.Adam(mpnn_model.parameters(), lr=1e-3)
loss_strategy = MSELossStrategy()

trainer = GenericTrainer(
    mpnn_model, optimizer, loss_strategy, train_loader, val_loader, test_loader, device
)
trainer.train(epochs=50)
trainer.evaluate(test_loader, save_results=True)
