import matplotlib.pyplot as plt
import numpy as np

# Data
plt.rcParams["font.family"] = "Times New Roman"  # Use serif font


datasets = [
    "(+) D shared layer",
    "(+) D shared layer",
    "(+) Morgan fingerprint,\n (-) D shared layer",
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

# Bar width
width = 0.12
x = np.arange(len(datasets))  # Dataset positions

# Use a professional style
plt.style.use("seaborn-v0_8-colorblind")

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Get colors from the style
style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Plot grouped bars (methods across dataset conditions)
for i, method in enumerate(methods[:-1]):  # Exclude "Average"
    ax.bar(
        x + i * width - (len(methods) - 2) * width / 2,  # Centering bars
        errors[:, i],
        width=width,
        label=method,
        color=style_colors[i % len(style_colors)],  # Cycle through colors
    )

# Overlay the average as a line plot
ax.plot(
    x,
    average_values,
    marker="o",
    linestyle="-",
    color="black",
    label="Average",
    markersize=8,
    linewidth=2,
)

# Labels and formatting
# ax.set_xlabel("Dataset Condition", fontsize=14)
ax.set_ylabel("sMAPE (%)", fontsize=14)
ax.set_title("Ablation Study: Effects on sMAPE", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=12, rotation=0)

# Legend
ax.legend(
    title="Property",
    fontsize=11,
    title_fontsize=12,
    loc="upper right",
)

# Adjust layout
plt.tight_layout()
plt.savefig("grouped_bars_with_average.png", dpi=300, bbox_inches="tight")
plt.show()
