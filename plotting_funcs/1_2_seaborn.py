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

# Min and Max Errors for Error Bars
min_errors = np.array(
    [
        [41, 40, 42, 46],
        [26, 19, 27, 26],
        [34, 25, 61, 59],
    ]
)

max_errors = np.array(
    [
        [46, 44, 52, 50],
        [38, 41, 51, 46],
        [61, 56, 151, 71],
    ]
)

# Calculate error bar ranges
lower_errors = errors - min_errors  # Distance from mean to min
print(lower_errors)
upper_errors = max_errors - errors  # Distance from mean to max
print(upper_errors)
raise ValueError
# Convert to DataFrame for Seaborn
data = []
for i, dataset in enumerate(datasets):
    for j, method in enumerate(methods):
        data.append(
            {
                "Dataset": dataset,
                "Method": method,
                "sMAPE Error (%)": errors[i, j],  # ✅ Mean values (correct y-values)
                "Lower Error": lower_errors[i, j],  # ✅ Lower error bars
                "Upper Error": upper_errors[i, j],  # ✅ Upper error bars
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

# Correctly position error bars using ax.patches
for bar, (i, row) in zip(ax.patches, df.iterrows()):
    x_position = bar.get_x() + bar.get_width() / 2  # ✅ Center error bar on the bar
    y_value = row["sMAPE Error (%)"]
    lower_err = row["Lower Error"]
    upper_err = row["Upper Error"]

    # Add error bars
    plt.errorbar(
        x=x_position,
        y=y_value,
        yerr=[[lower_err], [upper_err]],  # ✅ Corrected asymmetric error bars
        fmt="none",
        ecolor="black",
        capsize=3,
        elinewidth=1,
    )

# Labels and formatting
plt.ylabel("sMAPE Error (%)", fontsize=14)
plt.xlabel("")  # ✅ Remove "Dataset" label
# plt.title("Ablation Tests Across Datasets", fontsize=14, fontweight="bold")

# Adjust x-ticks
plt.xticks(fontsize=14)

# Legend
plt.legend(fontsize=12, title_fontsize=14)
plt.ylim(0, 150)

# Adjust layout
plt.tight_layout()
plt.savefig("seaborn_smape_errors_fixed.png", dpi=300, bbox_inches="tight")
plt.show()
