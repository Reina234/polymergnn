from chemprop.nn import BondMessagePassing, NormAggregation, RegressionFFN
import torch
from chemprop.models.model import MPNN
import pandas as pd
from sklearn.model_selection import train_test_split
import torch.optim as optim
from tools.data_processing.molecular_dataset import MolecularDataset
from tools.model.loss import MSELossStrategy
from tools.model.trainer import GenericTrainer
from tools.model.model_factory import ModelFactory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("data/freesolv/FreeSolv_SAMPL.csv")
df = df.drop(df.columns[[0, 1]], axis=1)  # Drop unnecessary columns

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)
train_loader = MolecularDataset(train_df, target_columns=[1, 2])

val_loader = MolecularDataset(val_df, target_columns=[1, 2])
test_loader = MolecularDataset(test_df, target_columns=[1, 2])

mp = BondMessagePassing()
agg = NormAggregation()
ffn = RegressionFFN(n_tasks=2)

mpnn_model = MPNN(mp, agg, ffn).to(device)

optimizer = optim.Adam(mpnn_model.parameters(), lr=1e-3)
loss_strategy = MSELossStrategy()

# Train Model Using GenericTrainer
trainer = GenericTrainer(
    model=mpnn_model,
    loss_strategy=MSELossStrategy(),
    train_loader=train_loader,
    val_loader=val_loader,
    test_loader=test_loader,
    device=device,
    hyperparams={"lr": 1e-3},
    use_tensorboard=True,
    track_learning_curve=True,
)

trainer.train(epochs=50)  # Train
trainer.evaluate(test_loader, save_results=True)  # Evaluate

print("✅ Training complete. Learning curve saved in results/learning_curve.png")
print("✅ View live metrics in TensorBoard with: tensorboard --logdir logs")
