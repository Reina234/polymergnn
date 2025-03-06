import torch
from tools.transform_pipeline_manager import TransformPipelineManager
from sklearn.preprocessing import StandardScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from training.refactored_batched_dataset import SimpleDataset
from tools.smiles_transformers import PolymerisationSmilesTransform
from training.benchmark_trainer import FNNTrainer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load dataset
df = pd.read_csv("data/output_2_4_2.csv")

n_bits = 2048
target_columns = [6, 7, 8, 9, 10, 11]  # Create Transform Manager
feature_columns = [4, 5]

pipeline_manager = TransformPipelineManager(feature_columns, target_columns)

# Apply same transformation to all features & targets
pipeline_manager.set_feature_pipeline(StandardScaler())
pipeline_manager.set_target_pipeline(StandardScaler())


train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42)

val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42)


train_dataset = SimpleDataset(
    data=train_df,
    pipeline_manager=pipeline_manager,
    n_bits=n_bits,
    monomer_smiles_column=3,
    solvent_smiles_column=1,
    monomer_smiles_transformer=PolymerisationSmilesTransform(),
    target_columns=target_columns,
    feature_columns=feature_columns,
    is_train=True,
)
fitted_pipeline_manager = train_dataset.pipeline_manager

val_dataset = SimpleDataset(
    data=val_df,
    pipeline_manager=fitted_pipeline_manager,
    n_bits=n_bits,
    monomer_smiles_column=3,
    solvent_smiles_column=1,
    monomer_smiles_transformer=PolymerisationSmilesTransform(),
    target_columns=target_columns,
    feature_columns=feature_columns,
    is_train=False,
)

test_dataset = SimpleDataset(
    data=test_df,
    pipeline_manager=fitted_pipeline_manager,
    n_bits=n_bits,
    monomer_smiles_column=3,
    solvent_smiles_column=1,
    monomer_smiles_transformer=PolymerisationSmilesTransform(),
    target_columns=target_columns,
    feature_columns=feature_columns,
    is_train=False,
)


config = {
    "batch_size": 32,
    "n_bits": 2048,
    "hidden_dim": 256,
    "dropout": 0.2,
    "log_selection_tensor": torch.tensor([1, 1, 1, 0, 0, 1]),
}
trainer = FNNTrainer(
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    test_dataset=test_dataset,
    hyperparams=config,
    save_results_dir="results/benchmarks/fnn",
    log_dir="logs/benchmarks/fnn",
)


trainer.run()
