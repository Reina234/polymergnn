import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_multitask_parity(json_path, save_path_prefix="parity_plot"):

    with open(json_path, "r") as f:
        data = json.load(f)

    y_true = np.array([entry["true"] for entry in data])
    y_pred = np.array([entry["pred"] for entry in data])

    n_params = y_true.shape[1]
    param_names = [f"Parameter {i+1}" for i in range(n_params)]

    sns.set_style("white")
    cmap = plt.get_cmap("viridis")

    fig, axs = plt.subplots(3, 2, figsize=(12, 12))
    axs = axs.flatten()

    for i in range(n_params):
        ax = axs[i]
        color = cmap(i / n_params)
        ax.scatter(y_true[:, i], y_pred[:, i], alpha=0.7, s=40, c=color, edgecolor="k")
        ax.plot(
            [min(y_true[:, i]), max(y_true[:, i])],
            [min(y_true[:, i]), max(y_true[:, i])],
            "--",
            color="gray",
        )
        ax.set_title(param_names[i])
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")

    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_grid.png", dpi=300)
    plt.show()

    # Plot average across all parameters
    y_true_mean = y_true.mean(axis=1)
    y_pred_mean = y_pred.mean(axis=1)

    plt.figure(figsize=(6, 6))
    plt.scatter(
        y_true_mean, y_pred_mean, alpha=0.7, s=50, c="darkorange", edgecolor="k"
    )
    min_val, max_val = min(y_true_mean.min(), y_pred_mean.min()), max(
        y_true_mean.max(), y_pred_mean.max()
    )
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="gray")
    plt.xlabel("Mean True")
    plt.ylabel("Mean Predicted")
    plt.title("Average Parity Plot")
    plt.tight_layout()
    plt.savefig(f"{save_path_prefix}_mean.png", dpi=300)
    plt.show()


def plot_param_no_outliers(
    json_path, param_index=2, save_path="param3_no_outliers.png"
):
    # Load data
    with open(json_path, "r") as f:
        data = json.load(f)

    # Extract true and predicted values for parameter 3 (index 2)
    y_true = np.array([entry["true"][param_index] for entry in data])
    y_pred = np.array([entry["pred"][param_index] for entry in data])

    # Calculate IQR to remove outliers
    q1, q3 = np.percentile(y_true, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    # Mask for non-outliers
    mask = (y_true >= lower_bound) & (y_true <= upper_bound)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    # Plot
    sns.set_style("white")
    plt.figure(figsize=(6, 6))
    plt.scatter(
        y_true_filtered, y_pred_filtered, s=50, alpha=0.7, c="royalblue", edgecolor="k"
    )
    min_val = min(y_true_filtered.min(), y_pred_filtered.min())
    max_val = max(y_true_filtered.max(), y_pred_filtered.max())
    plt.plot([min_val, max_val], [min_val, max_val], "--", color="gray")
    plt.xlabel("True (Parameter 3)")
    plt.ylabel("Predicted (Parameter 3)")
    plt.title("Parameter 3 Parity Plot (Outliers Removed)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


plot_multitask_parity("/Users/reinazheng/Desktop/polymergnn/resultsval_preds.json")

plot_param_no_outliers(
    "/Users/reinazheng/Desktop/polymergnn/resultsval_preds.json",
    param_index=2,
    save_path="param3_no_outliers.png",
)
