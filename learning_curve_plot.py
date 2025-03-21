import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.ticker as ticker

# Set plot style

sns.set_style("white")
# plt.rcParams["font.family"] = "Times New Roman"
# Removed as sns.set_style("whitegrid") is sufficient


def plot_losses(json_path, underfit_epoch, overfit_epoch):
    with open(json_path, "r") as f:
        data = json.load(f)

    train_losses = data["train_losses"]
    val_losses = data["val_losses"]
    epochs = list(range(1, len(train_losses) + 1))

    df = pd.DataFrame(
        {
            "Epoch": epochs * 2,
            "Loss": train_losses + val_losses,
            "Split": ["Training set"] * len(epochs) + ["Validation set"] * len(epochs),
        }
    )

    # Plot
    plt.figure(figsize=(5, 4))
    ax = sns.lineplot(
        data=df,
        x="Epoch",
        y="Loss",
        hue="Split",
        linewidth=1.5,
        palette=["royalblue", "tomato"],
    )

    ax.tick_params(which="both", bottom=True, left=True, length=5)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(20))
    # ax.xaxis.set_minor_locator(ticker.MultipleLocator(10))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
    # ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

    ax.set_ylim(0, 0.8)
    ax.set_xlim(1, epochs[-1])
    max_loss = max(max(train_losses), max(val_losses))

    # Shade underfit region
    if underfit_epoch > 1:
        plt.axvspan(1, underfit_epoch, alpha=0.1, color="orange")
        plt.text(
            underfit_epoch / 2,
            max_loss * 0.5,
            "Underfit",
            color="sienna",
            fontsize=9,
            ha="center",
            weight="bold",
        )

    # Shade overfit region
    if overfit_epoch < epochs[-1]:
        plt.axvspan(overfit_epoch, epochs[-1], alpha=0.1, color="red")
        plt.text(
            (overfit_epoch + epochs[-1]) / 2,
            max_loss * 0.5,
            "Overfit",
            color="firebrick",
            fontsize=9,
            ha="center",
            weight="bold",
        )

    # Plot labels

    # plt.title("Training vs Validation Loss", fontsize=16)
    plt.xlabel("Epoch", fontsize=10)
    plt.ylabel("Loss", fontsize=10)
    legend = plt.legend(fontsize=9, loc="upper right")
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(1.0)
    legend.get_frame().set_boxstyle("Square")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser()

    # parser.add_argument(
    #    "json_path", type=str, help="Path to JSON file with train/val losses"
    # )
    # parser.add_argument(
    #    "--split_epoch",
    #    type=int,
    #    default=50,
    #    help="Epoch to split underfit/overfit regions",
    # )
    # args = parser.parse_args()
    filepath = "results/e25d218ac4_convergence.json"
    plot_losses(filepath, 92, 102)
