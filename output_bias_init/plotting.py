import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


if __name__ == "__main__":
    plt.style.use("ggplot")
    # color1 = "#191d40"  # Black
    # color2 = "#00ffc5"   # Green
    # color3 = "#0043ff"   # Blue
    color_1 = "#e63946"
    color_2 = "#1d3557"

    # sns.set_palette([color_1, color_2])
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=str, required=True)
    args = parser.parse_args()

    plotting_df = pd.read_csv(args.input_csv)
    plotting_df["output_bias_value"] = sigmoid(plotting_df["output_bias_value"])

    fig, ax = plt.subplot_mosaic(
        [["loss", "roc_auc"]], figsize=(12, 4), layout="constrained"
    )
    sns.lineplot(
        data=plotting_df,
        x="n_epochs",
        y="loss",
        hue="type",
        style="type",
        ax=ax["loss"],
        markers=True,
        # marker="o",
        legend=False,
        errorbar="ci",
        markersize=10,
    )
    ax["loss"].set_xlabel("Epoch", fontsize=20)
    ax["loss"].set_ylabel("Binary Cross Entropy", fontsize=20)
    # Increase the size of the labels
    ax["loss"].tick_params(axis="both", which="major", labelsize=15)

    sns.lineplot(
        data=plotting_df,
        x="n_epochs",
        y="roc_auc",
        hue="type",
        style="type",
        ax=ax["roc_auc"],
        markers=True,
        # marker="o",
        legend=True,
        errorbar="ci",
        markersize=10,
    )
    ax["roc_auc"].set_ylabel("ROC-AUC", fontsize=20)
    ax["roc_auc"].set_xlabel("Epoch", fontsize=20)
    ax["roc_auc"].tick_params(axis="both", which="major", labelsize=15)

    # Set common legend for both subplots
    handles, labels = ax["roc_auc"].get_legend_handles_labels()
    labels = ["Vanilla", "Positive\nRate"]
    fig.legend(
        handles, labels, loc="center right", title="Bias Intitalization", fontsize=20
    )
    # Remove the original legends
    ax["roc_auc"].get_legend().remove()

    output_path = Path(args.input_csv).with_suffix(".png")
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()
