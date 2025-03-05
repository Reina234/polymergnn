# Import necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Define dataset names and methods
dataset_names = ["Lipophilicity", "Truncated FreeSolv", "Our dataset"]
methods = [
    "(-)FG edges, (-)ChemBERTa",
    "(+)FG edges, (-)ChemBERTa",
    "(-)FG edges, (+)ChemBERTa",
    "(+)FG edges, (+)ChemBERTa",
]

# Define the mean error values for each dataset and method
errors = {
    "Lipophilicity": [43.5, 42, 47, 48],
    "Truncated FreeSolv": [32, 30, 39, 36],
    "Our dataset": [47.5, 40.1, 76, 65],
}

# Define the corresponding symmetric error values
errors_symmetric = {
    "Lipophilicity": [2.5, 2.0, 5.0, 2.0],
    "Truncated FreeSolv": [6.0, 11.0, 12.0, 10.0],
    "Our dataset": [13.5, 15.1, 15.0, 6.0],
}

# Convert data into a pandas DataFrame for seaborn plotting
data_list = []
for dataset in dataset_names:
    for method_idx, method in enumerate(methods):
        data_list.append(
            {
                "Dataset": dataset,
                "Method": method,
                "Error": errors[dataset][method_idx],
                "Error Bar": errors_symmetric[dataset][method_idx],
            }
        )

df = pd.DataFrame(data_list)
sns.set_style("white")
# Set up the seaborn plot
plt.figure(figsize=(8, 6))
ax = sns.barplot(
    data=df,
    x="Dataset",
    y="Error",
    hue="Method",
    capsize=0.2,
    errcolor="black",
    palette="pastel",
    errwidth=1,
    edgecolor="black",
)

# Manually assign each error bar to each corresponding bar
bar_positions = [
    bar.get_x() + bar.get_width() / 2 for bar in ax.patches
]  # Get center x positions
bar_heights = [bar.get_height() for bar in ax.patches]  # Get bar heights

# Error bar values in the same order as bars
error_bar_values = [
    2.5,
    6.0,
    13.5,
    2.0,
    11.0,
    15.1,
    5.0,
    12.0,
    18.0,
    2.0,
    10.0,
    6.0,
]

# Assign the error bars manually
for x, y, err in zip(bar_positions, bar_heights, error_bar_values):
    plt.errorbar(
        x,
        y,
        yerr=err,
        fmt="none",
        ecolor="black",
        color="black",
        capsize=5,
        capthick=1,
        lw=1,
    )

# Customize labels and legend
plt.xlabel("")
plt.ylabel("sMAPE Error (%)", fontsize=14, fontweight="bold")
# plt.title("Comparison of Errors Across Datasets and Methods")
plt.xticks(rotation=0, fontsize=12)
plt.legend(fontsize=12, title_fontsize=14, loc="upper left")
plt.ylim(0, 120)

# Show the plot
plt.tight_layout()
plt.savefig(
    "seaborn1_grouped_bars_with_average_shaded_error.png", dpi=300, bbox_inches="tight"
)
plt.show()
