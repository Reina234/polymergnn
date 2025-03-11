import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 🔹 Define column indexes and their corresponding names
output_column_indexes = [7, 8, 9, 10, 11, 12]
output_column_names = ["Rg_mean", "Rg_SD", "D", "SASA_mean", "SASA_SD", "Re"]

# 🔹 Load the dataset without headers
csv_path = "/Users/reinazheng/Desktop/polymergnn/data/output_3_11.csv"
df = pd.read_csv(csv_path, header=None)  # No headers in the file

# 🔹 Select only the relevant columns & rename them
df_outputs = df.iloc[:, output_column_indexes]
df_outputs.columns = output_column_names  # Assign correct names

# 🔹 Compute correlation matrix
corr_matrix = df_outputs.corr()

# 🔹 1️⃣ Heatmap of Correlations
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Output Variables")
plt.show()

# 🔹 2️⃣ Pair Plot (Scatter and KDE distributions)
sns.pairplot(df_outputs, diag_kind="kde", corner=True)
plt.show()

# 🔹 3️⃣ Print Strongest Correlations
correlation_values = corr_matrix.abs().unstack().sort_values(ascending=False)
correlation_values = correlation_values[
    correlation_values < 1
]  # Remove self-correlation
print("\n🔍 Strongest Correlations:\n", correlation_values.head(10))
