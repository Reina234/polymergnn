import matplotlib.pyplot as plt
import numpy as np

# Data
plt.rcParams["font.family"] = "Times New Roman"  # Use serif font

datasets = [
    "FNN",
    "MPNN",
    "Our model",
]

errors = np.array(
    [
        37,  # FNN
        51,  # MPNN (standard)
        26,  # MPNN + GNN
    ]
)

# Use a colorblind-friendly style
plt.style.use("seaborn-v0_8-colorblind")

# Get colors from the current style
style_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
bar_colors = [
    style_colors[(i + 28) % len(style_colors)] for i in range(len(datasets))
]  # Ensure enough colors

# Create figure
fig, ax = plt.subplots(figsize=(3.5, 2.5))

# Bar chart with reduced width
bars = ax.bar(
    datasets,
    errors,
    color=bar_colors,
    linewidth=1.2,
    width=0.4,  # ✅ Reduced bar width
)

# Labels and title
ax.set_ylabel("sMAPE (%)", fontsize=12)
ax.set_xlabel("Model", fontsize=12)
# ax.set_title("Model Comparison: sMAPE Error", fontsize=14, fontweight="bold")

# Annotate values on top of bars
for bar in bars:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 1,
        f"{yval:.1f}",
        ha="center",
        fontsize=10,
    )


ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig("model_comparison_colored.png", dpi=300, bbox_inches="tight")
plt.show()
