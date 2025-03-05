import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Use Times New Roman font
plt.rcParams["font.family"] = "Times New Roman"

# Dataset names
datasets = ["A", "B", "C", "D"]
methods = ["Rg mean", "Rg SD", "D mean", "SASA mean", "SASA SD", "Re mean"]

# Errors
errors = np.array(
    [
        [10, 25, 75.5, 10.76, 38, 73],  # Shared layer
        [7.9, 27, 52, 9.3, 33.5, 45.5],  # One less shared layer
        [8.0, 27.6, 36.8, 9.5, 27, 43],  # Morgan FP, (-) Shared Layer
        [16.1, 28.9, 71, 15, 32, 61],  # Fine-tuned
    ]
)

# Placeholder for error bars (min and max)
min_error = np.array(
    [
        [8, 24, 71, 10.5, 36, 40],
        [7.2, 26, 49, 8.6, 32, 43],
        [7.8, 27, 35, 9.3, 26, 39],
        [14, 27, 69, 12, 31, 42],
    ]
)

max_error = np.array(
    [
        [12, 26, 80, 11, 40, 102],
        [8.3, 28, 55, 10, 35, 48],
        [9.2, 28, 42, 11, 30, 46],
        [18, 30, 75, 15, 34, 80],
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
                "sMAPE (%)": errors[i, j],
                "Min": min_error[i, j],
                "Max": max_error[i, j],
            }
        )

df = pd.DataFrame(data)

# Get Set2 color palette to match the combined plot
property_colors = sns.color_palette("Set2", n_colors=len(methods))

# Create subplots (3 rows, 2 columns)
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(5, 7), sharey=True)
axes = axes.flatten()

# Iterate over each method and create separate plots
for i, method in enumerate(methods):
    ax = axes[i]

    # Filter data for the current method
    df_method = df[df["Method"] == method]

    # Plot bars with Seaborn
    sns.barplot(
        data=df_method,
        x="Dataset",
        y="sMAPE (%)",
        ax=ax,
        color=property_colors[i],  # Ensure colors match the combined plot
        edgecolor="black",
    )

    # Add error bars manually
    for j, dataset in enumerate(datasets):
        y = errors[j, i]
        ymin = min_error[j, i]
        ymax = max_error[j, i]
        ax.errorbar(
            x=j,
            y=y,
            yerr=[[y - ymin], [ymax - y]],  # Asymmetric error bars
            fmt="none",
            ecolor="black",
            capsize=3,
            elinewidth=1,
        )

        # Annotate values
        ax.text(
            j,
            y + 2,  # Adjust position above the bar
            f"{y:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
        )

    # Formatting
    ax.set_title(method, fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(datasets)))
    ax.set_xlabel("")
    if i >= 4:  # Only label x-axis for the last row
        ax.set_xticklabels(datasets, fontsize=8, rotation=0)
    else:
        ax.set_xticklabels([])

    ax.set_ylim(0, 105)
    if i % 2 == 0:  # Add y-label to left-side subplots
        ax.set_ylabel("sMAPE (%)", fontsize=10, fontweight="bold")

# General formatting
fig.suptitle("sMAPE Across Properties", fontsize=14, fontweight="bold")
fig.tight_layout()
fig.subplots_adjust(hspace=0.4, wspace=0.3)  # Adjust spacing

# Save and show
plt.savefig("seaborn_subplots_sMAPE.png", dpi=300, bbox_inches="tight")
plt.show()
