import numpy as np
import matplotlib.pyplot as plt

# Data
datasets = [
    "(+) D shared layer",
    "(+) D shared layer",
    "(+) Morgan fingerprint\n(-) D shared layer",  # Multi-line label
]
methods = ["Rg mean", "Rg SD", "D mean", "SASA mean", "SASA SD", "Re mean", "Average"]

errors = np.array(
    [
        [11, 26, 71, 10, 36, 40, 33],  # Shared layer
        [7.4, 24, 49, 8.6, 32, 43, 27],  # One less shared layer
        [7.8, 28, 38, 9.3, 30, 43, 26],  # Morgan FP, (-) Shared Layer
    ]
)

# Extract average values separately (last column)
average_values = errors[:, -1]  # Last column is the average
errors = errors[:, :-1]  # Remove "Average" column from bars
methods = methods[:-1]  # Remove "Average" from labels

# Use a professional style
plt.style.use("seaborn-v0_8-colorblind")

# Get colors from the style and ensure enough colors
style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
num_methods = len(methods)
num_datasets = len(datasets)

# Ensure enough colors for properties and datasets by cycling colors if needed
property_colors = [
    style_colors[i % len(style_colors)] for i in range(num_methods)
]  # Colors per property
dataset_colors = [
    style_colors[i % len(style_colors)] for i in range(num_datasets)
]  # Colors per dataset

# Generate a figure for each property
for i, method in enumerate(methods):
    fig, ax = plt.subplots(figsize=(6, 4))

    # Plot bars for each dataset condition
    for j, dataset in enumerate(datasets):
        ax.bar(
            dataset,
            errors[j][i],
            color=dataset_colors[j],  # Consistent dataset colors
            edgecolor=property_colors[i],  # Outline color matches main plot
            linewidth=1.5,
            width=0.5,
        )

    # Formatting
    ax.set_ylabel("sMAPE (%)", fontsize=12)
    ax.set_xlabel("Dataset Condition", fontsize=12)
    ax.set_title(f"sMAPE for {method}", fontsize=14, fontweight="bold")
    ax.set_xticklabels(datasets, fontsize=10, rotation=0, ha="center")

    # Save each figure
    plt.tight_layout()
    plt.savefig(f"sMAPE_{method.replace(' ', '_')}.png", dpi=300, bbox_inches="tight")
    plt.show()
