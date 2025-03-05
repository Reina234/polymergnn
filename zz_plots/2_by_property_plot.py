import matplotlib.pyplot as plt
import numpy as np

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
        [11, 26, 71, 10, 36, 40, 33],  # Shared layer
        [7.4, 24, 49, 8.6, 32, 43, 27],  # One less shared layer
        [7.8, 28, 38, 9.3, 30, 43, 26],  # Morgan FP, (-) Shared Layer
        [14, 27, 69, 15, 32, 42, 33],  # fine tuned
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
fig, ax = plt.subplots(figsize=(8, 6))

# Get colors from the style
style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

# Plot grouped bars (methods across dataset conditions)
for i, method in enumerate(methods[:-1]):  # Exclude "Average"
    ax.bar(
        x + i * width - (len(methods) - 2) * width / 2,  # Centering bars
        errors[:, i],
        width=width,
        label=method,
        color=style_colors[i % len(style_colors)],
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

# ✅ Annotate the average values with a white background
for i, avg in enumerate(average_values):
    ax.text(
        x[i],
        avg + 2,
        f"{avg:.1f}",
        ha="center",
        fontsize=12,
        fontweight="bold",
        color="black",
        bbox=dict(facecolor="white", boxstyle="square,pad=0.2"),  # White background
    )

# Labels and formatting
ax.set_ylabel("sMAPE (%)", fontsize=14)
# ax.set_title("Ablation Study: Effects on sMAPE", fontsize=14, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=11, rotation=0)
ax.set_ylim(0, 100)
# Legend
ax.legend(
    fontsize=11,
    loc="upper center",
    ncol=3,  # Arrange legend items in 2 columns
)

# Adjust layout
plt.tight_layout()
plt.savefig("grouped_bars_with_average_annotated2.png", dpi=300, bbox_inches="tight")
plt.show()
