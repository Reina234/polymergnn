import matplotlib.pyplot as plt
import numpy as np

# Data
plt.rcParams["font.family"] = "Times New Roman"  # Use serif font
datasets = ["Lipophilicity", "FreeSolv", "Our Dataset"]
methods = [
    "(-)FG edges, (-)ChemBERTa",
    "(+)FG edges, (-)ChemBERTa",
    "(-)FG edges  (+)ChemBERTa",
    "(+)FG edges  (+)ChemBERTa",
]
errors = np.array(
    [
        [41, 42, 47, 46],  # Lipophilicity
        [24, 19, 27, 26],  # FreeSolv
        [61, 56, 71, 59],  # Our Dataset
    ]
)

# Configurable colors
# colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
plt.style.use("seaborn-v0_8-colorblind")
# Save and show
fig, ax = plt.subplots(figsize=(8, 6))
width = 0.2  # Width of bars
x = np.arange(len(datasets))  # Dataset positions

# Plot each method
for i, method in enumerate(methods):
    ax.bar(x + i * width, errors[:, i], width=width, label=method)

# Formatting
ax.set_ylabel("sMAPE Error (%)", fontsize=12)
ax.set_xlabel("Dataset", fontsize=12)
ax.set_title("Ablation tests across datasets", fontsize=14, fontweight="bold")
ax.set_xticks(x + width)
ax.set_xticklabels(datasets, fontsize=10)

ax.legend(fontsize=10, title_fontsize=11)


plt.tight_layout()
plt.savefig("smape_errors_matplotlib.png", dpi=300, bbox_inches="tight")
plt.show()
