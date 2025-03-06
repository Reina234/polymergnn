import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use Times New Roman font
plt.rcParams["font.family"] = "Times New Roman"

# Dataset names
datasets = ["Lipophilicity", "Truncated FreeSolv", "Our dataset"]
methods = [
    "(-)FG edges, (-)ChemBERTa",
    "(+)FG edges, (-)ChemBERTa",
    "(-)FG edges, (+)ChemBERTa",
    "(+)FG edges, (+)ChemBERTa",
]

# Errors (mean values for bars)
errors = np.array(
    [
        [43.5, 42, 47, 48],  # Lipophilicity
        [32, 30, 39, 36],  # FreeSolv
        [47.5, 40.1, 106, 65],  # Our Dataset
    ]
)

# Symmetric error values
errors_symmetric = np.array(
    [
        [2.5, 2.0, 5.0, 2.0],  # Lipophilicity
          # FreeSolv
        [13.5, 15.1, 45.0, 6.0], 
         [6.0, 11.0, 12.0, 10.0], # Our Dataset
    ]
)

# Convert to DataFrame for Seaborn
data = []
for i, dataset in enumerate(datasets):
    for j, method in enumerate(methods):
        data.append(
            {
                "Dataset": dataset,
                "Method": method,
                "sMAPE Error (%)": errors[i, j],  # ✅ Mean values (correct y-values)
                "Error": errors_symmetric[i, j],  # ✅ Symmetric error bars
            }
        )

df = pd.DataFrame(data)

# Get Set2 color palette and shift the starting point
full_palette = sns.color_palette("Set2", n_colors=20)  # Generate extra colors
bar_colors = full_palette[15 : 15 + len(methods)]  # Start after the 15th color

# Create figure
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

# Create bar plot with custom colors
ax = sns.barplot(
    data=df,
    x="Dataset",
    y="sMAPE Error (%)",
    hue="Method",
    ci=None,
    palette=bar_colors,
    edgecolor="black",
)

# ✅ FIX: Get correct mapping of dataset names to x-tick positions
dataset_positions = {
    label.get_text(): pos for label, pos in zip(ax.get_xticklabels(), ax.get_xticks())
}

# Define bar width
width = 0.15  # Width of individual bars
num_methods = len(methods)

# Iterate through dataframe to align error bars correctly
for index, row in df.iterrows():
    dataset_index = dataset_positions[row["Dataset"]]  # ✅ Correct dataset position
    method_index = methods.index(row["Method"])  # Get correct method index

    # Compute x position based on method index
    x_position = dataset_index + (method_index - (num_methods - 1) / 2) * width

    # Get error value
    y_value = row["sMAPE Error (%)"]
    error = row["Error"]  # ✅ Symmetric error

    # Add symmetric error bars
    plt.errorbar(
        x=x_position,
        y=y_value,
        yerr=error,  # ✅ Symmetric error bars
        fmt="none",
        ecolor="black",
        capsize=3,
        elinewidth=1,
    )

# Labels and formatting
plt.ylabel("sMAPE Error (%)", fontsize=14)
plt.xlabel("")  # ✅ Remove "Dataset" label

# Adjust x-ticks
plt.xticks(fontsize=14)

# Legend
plt.legend(fontsize=12, title_fontsize=14)
plt.ylim(0, 150)

# Adjust layout
plt.tight_layout()
plt.savefig("seaborn_smape_errors_fixed.png", dpi=300, bbox_inches="tight")
plt.show()
