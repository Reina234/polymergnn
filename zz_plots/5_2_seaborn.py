import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data
# plt.rcParams["font.family"] = "Times New Roman"  # Use serif font

datasets = ["FNN", "MPNN", "Our model"]
errors = np.array([37, 46, 26])  # Mean values
errors_symmetric = np.array([3, 16, 1])  # Symmetric error values

# Create a DataFrame for seaborn
df = pd.DataFrame(
    {"Model": datasets, "sMAPE (%)": errors, "Error Bar": errors_symmetric}
)

# Create a seaborn bar plot with error bars
plt.figure(figsize=(4.5, 3))
ax = sns.barplot(
    data=df,
    x="Model",
    y="sMAPE (%)",
    palette="Paired",
    capsize=0.2,
    errcolor="black",
    errwidth=1,
    edgecolor="black",
)

# Add manual error bars to ensure correct alignment
bar_positions = [bar.get_x() + bar.get_width() / 2 for bar in ax.patches]
bar_heights = [bar.get_height() for bar in ax.patches]

# Assign error bars correctly
for x, y, err in zip(bar_positions, bar_heights, df["Error Bar"].values):
    plt.errorbar(x, y, yerr=err, fmt="none", color="black", capsize=5, capthick=1, lw=1)

# Labels and formatting
plt.ylabel("sMAPE (%)", fontsize=12, fontweight="bold")
plt.xlabel("", fontsize=12)
plt.ylim(0, 100)

# Annotate values on top of bars
for bar in ax.patches:
    yval = bar.get_height()
    ax.text(
        bar.get_x() + bar.get_width() / 2,
        yval + 6,
        f"{yval:.1f}",
        ha="center",
        fontsize=10,
        bbox=dict(facecolor="white", boxstyle="square,pad=0", edgecolor="white"),
    )

plt.tight_layout()
plt.savefig("seaborn_2subplots_sMAPE.png", dpi=300, bbox_inches="tight")

plt.show()
