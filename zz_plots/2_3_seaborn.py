import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
# plt.rcParams["font.family"] = "Times New Roman"  # Use serif font

datasets = [
    "(+) D shared layer",
    "(-) D shared layer",
    "(+) Morgan fingerprint,\n (-) D shared layer",
    "(+) Pre-trained \n (-) D shared layer",
]
methods = ["Rg mean", "Rg SD", "D mean", "SASA mean", "SASA SD", "Re mean", "Average"]

errors = np.array(
    [
        [10, 25, 75.5, 10.76, 38, 73, 38.4],  # Shared layer
        [7.9, 27, 52, 9.3, 33.5, 45.5, 29.2],  # One less shared layer
        [8.0, 27.6, 36.8, 9.5, 27, 43, 25.9],  # Morgan FP, (-) Shared Layer
        [16.1, 28.9, 71, 15, 32, 47, 37.3],  # fine tuned
    ]
)

# Placeholder for average error bars (replace later)
average_values = errors[:, -1]  # Last column is the average
average_min_error = np.array(
    [33, 28.6, 25, 34.2]
)  # Placeholder: Replace with real values
average_max_error = np.array(
    [44.8, 29.7, 26.8, 40.3]
)  # Placeholder: Replace with real values

# Convert to DataFrame for Seaborn
data = []
for i, dataset in enumerate(datasets):
    for j, method in enumerate(methods[:-1]):  # Exclude "Average"
        data.append(
            {
                "Dataset": dataset,
                "Method": method,
                "sMAPE (%)": errors[i, j],
            }
        )

df = pd.DataFrame(data)

# Create figure
plt.figure(figsize=(8, 6))
sns.set_style("white")
sns.color_palette("Set2")
sns.set_palette("Set2")
# Create bar plot without error bars
ax = sns.barplot(
    data=df,
    x="Dataset",
    y="sMAPE (%)",
    hue="Method",
    palette="Set2",
    edgecolor="black",
)

# Overlay the average as a line plot
sns.lineplot(
    x=np.arange(len(datasets)),
    y=average_values,
    marker="o",
    linestyle="-",
    color="black",
    label="Average",
    markersize=8,
    linewidth=2,
)

# Add shaded region for average error bars
plt.fill_between(
    np.arange(len(datasets)),
    average_min_error,
    average_max_error,
    color="black",
    alpha=0.2,  # Transparency for shaded error region
    label="Average ± Error",
)

# Annotate the average values
for i, avg in enumerate(average_values):
    plt.text(
        i,
        avg + 2,
        f"{avg:.1f}",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="black",
        bbox=dict(facecolor="white", boxstyle="square,pad=0.2"),
    )

# Labels and formatting
plt.ylabel("sMAPE (%)", fontsize=14, fontweight="bold")
plt.xticks(fontsize=12, rotation=0)
plt.ylim(0, 100)
plt.xlabel("")
# Adjust legend
plt.legend(fontsize=11, loc="upper center", ncol=3)

# Adjust layout and save
plt.tight_layout()
plt.savefig(
    "seaborn_grouped_bars_with_average_shaded_error.png", dpi=300, bbox_inches="tight"
)
plt.show()
