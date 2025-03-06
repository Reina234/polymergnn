import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data
plt.rcParams["font.family"] = "Times New Roman"  # Use serif font

datasets = [
    "(+) D shared layer",
    "(+) D shared layer",
    "(+) Morgan fingerprint,\n (-) D shared layer",
    "Fine-tuned",
]
methods = ["Rg mean", "Rg SD", "D mean", "SASA mean", "SASA SD", "Re mean", "Average"]

errors = np.array(
    [
        [10, 25, 75.5, 10.76, 38, 73, 38.4],  # Shared layer
        [7.9, 27, 52, 9.3, 33.5, 45.5, 29.2],  # One less shared layer
        [8.0, 27.6, 36.8, 9.5, 27, 43, 25.9],  # Morgan FP, (-) Shared Layer
        [16.1, 28.9, 71, 15, 32, 61, 37.3],  # fine tuned
    ]
)

# Placeholder for error bars: min and max values (replace later)
min_error = np.array(
    [
        [8, 24, 71, 10.5, 36, 40, 33],  # Min values
        [7.2, 26, 49, 8.6, 32, 43, 28.6],
        [7.8, 27, 35, 9.3, 26, 39, 25],
        [14, 27, 69, 12, 31, 42, 34.2],
    ]
)


max_error = np.array(
    [
        [12, 26, 80, 11, 40, 102, 44.8],  # Max values
        [8.3, 28, 55, 10, 35, 48, 29.7],
        [9.2, 28, 42, 11, 30, 46, 26.8],
        [18, 30, 75, 15, 34, 80, 40.3],
    ]
)

# Extract average values separately
average_values = errors[:, -1]  # Last column is the average
errors = errors[:, :-1]  # Remove "Average" column from bars
min_error = min_error[:, :-1]  # Remove "Average" column
max_error = max_error[:, :-1]  # Remove "Average" column

# Convert to DataFrame for Seaborn
data = []
for i, dataset in enumerate(datasets):
    for j, method in enumerate(methods[:-1]):  # Exclude "Average"
        data.append(
            {
                "Dataset": dataset,
                "Method": method,
                "sMAPE (%)": errors[i, j],
                "Min": min_error[i, j],
                "Max": max_error[i, j],
            }
        )

df = pd.DataFrame(data)

# Create figure
plt.figure(figsize=(8, 6))
sns.set_style("whitegrid")

# Create bar plot with error bars
ax = sns.barplot(
    data=df,
    x="Dataset",
    y="sMAPE (%)",
    hue="Method",
    ci=None,  # Disable built-in CI
    capsize=0.1,
    palette="colorblind",
    edgecolor="black",
)

# Add custom error bars
for i, dataset in enumerate(datasets):
    for j, method in enumerate(methods[:-1]):
        y = errors[i, j]
        ymin = min_error[i, j]
        ymax = max_error[i, j]
        plt.errorbar(
            x=i + (j - len(methods) / 2) * 0.12,  # Offset error bars
            y=y,
            yerr=[[y - ymin], [ymax - y]],
            fmt="none",
            ecolor="black",
            capsize=3,
            elinewidth=1,
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
plt.ylabel("sMAPE (%)", fontsize=14)
plt.xticks(fontsize=11, rotation=0)
plt.ylim(0, 100)

# Adjust legend
plt.legend(fontsize=11, loc="upper center", ncol=3)

# Adjust layout and save
plt.tight_layout()
plt.savefig("seaborn_grouped_bars_with_error_bars.png", dpi=300, bbox_inches="tight")
plt.show()
