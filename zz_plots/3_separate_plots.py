import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "Times New Roman"  # Use serif font
datasets = ["A", "B", "C", "D"]
methods = ["Rg mean", "Rg SD", "D mean", "SASA mean", "SASA SD", "Re mean", "Average"]

errors = np.array(
    [
        [11, 26, 71, 10, 36, 40, 33],  # Shared layer
        [7.4, 24, 49, 8.6, 32, 43, 27],  # One less shared layer
        [7.8, 28, 38, 9.3, 30, 43, 26],  # Morgan FP, (-) Shared Layer
        [14, 27, 69, 15, 32, 42, 33],  # fine tuned
    ]
)


# Extract average values separately (last column)
average_values = errors[:, -1]  # Last column is the average
errors = errors[:, :-1]  # Remove "Average" column from bars
methods = methods[:-1]  # Remove "Average" from labels

# Use a professional style
plt.style.use("seaborn-v0_8-colorblind")

# Get colors from the main grouped bar plot
style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
num_methods = len(methods)
num_datasets = len(datasets)

# Ensure enough colors for properties (one per property)
property_colors = [
    style_colors[i % len(style_colors)] for i in range(num_methods)
]  # Colors per property

# Create subplots
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(4.5, 6), sharey=True)
axes = axes.flatten()  # Flatten for easy iteration

# Plot each property separately
for i, method in enumerate(methods):
    for j, dataset in enumerate(datasets):
        bar = axes[i].bar(
            j,
            errors[j][i],
            color=property_colors[i],
            width=0.5,
            alpha=0.8,
            label=dataset,
        )

        # Annotate value on top of the bar
        for rect in bar:
            height = rect.get_height()
            axes[i].text(
                rect.get_x() + rect.get_width() / 2,
                height + 2,  # Adjust position above the bar
                f"{height:.1f}",
                ha="center",
                va="bottom",
                fontsize=9,
                fontweight="bold",
            )

    axes[i].set_title(method, fontsize=10, fontweight="bold")
    axes[i].set_xticks(range(len(datasets)))
    if i >= (3 - 1) * 2:  # Last row condition
        axes[i].set_xticklabels(datasets, fontsize=8, rotation=0)
    else:
        axes[i].set_xticklabels([])
    axes[i].set_ylim(0, 100)
    if i % 2 == 0:
        axes[i].set_ylabel(
            "sMAPE (%)", fontsize=10, fontweight="bold"
        )  # Removes text but keeps ticks
# General formatting
fig.suptitle("sMAPE Across Properties", fontsize=14, fontweight="bold")
fig.tight_layout()  # Automatically optimizes layout
fig.subplots_adjust(hspace=0.3, wspace=0.2)  # Adjust vertical and horizontal spacing

fig.savefig("subplots_sMAPE.png", dpi=300, bbox_inches="tight")
plt.show()
