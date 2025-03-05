import matplotlib.pyplot as plt
import numpy as np

# Data
plt.rcParams["font.family"] = "Times New Roman"  # Use serif font
datasets = ["Lipophilicity", "Truncated FreeSolv", "Our dataset"]
methods = [
    "(-)FG edges, (-)ChemBERTa",
    "(+)FG edges, (-)ChemBERTa",
    "(-)FG edges, (+)ChemBERTa",
    "(+)FG edges, (+)ChemBERTa",
]
errors = np.array(
    [
        [43.5, 42, 47, 48],  # Lipophilicity
        [32, 30, 39, 36],  # FreeSolv
        [47.5, 40.1, 106, 65],  # Our Dataset
    ]
)

min_errors = np.array(
    [
        [41, 40, 42, 46],  # Lipophilicity
        [26, 19, 27, 26],  # FreeSolv
        [34, 25, 61, 59],  # Our Dataset
    ]
)

max_errors = np.array(
    [
        [46, 44, 52, 50],  # Lipophilicity
        [38, 41, 51, 46],  # FreeSolv
        [61, 56, 151, 71],  # Our Dataset
    ]
)
# Use a colorblind-friendly style
plt.style.use("seaborn-v0_8-colorblind")

# Get colors from the current style, skipping the first 5 colors
style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
bar_colors = [
    style_colors[(i + 15) % len(style_colors)] for i in range(len(methods))
]  # Ensure enough colors

# Create figure
fig, ax = plt.subplots(figsize=(5, 4))
width = 0.12  # Width of bars
x = np.arange(len(datasets))  # Dataset positions

# Plot each method with assigned colors
for i, method in enumerate(methods):
    ax.bar(x + i * width, errors[:, i], width=width, label=method, color=bar_colors[i])

# Formatting
ax.set_ylabel("sMAPE Error (%)", fontsize=14)
ax.set_xlabel("Dataset", fontsize=12)
ax.set_title("Ablation tests across datasets", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(datasets, fontsize=14)

# Legend
ax.legend(fontsize=12, title_fontsize=14)
ax.set_ylim(0, 100)
# Adjust layout
plt.tight_layout()
plt.savefig("smape_errors_shifted_colors2.png", dpi=300, bbox_inches="tight")
plt.show()
