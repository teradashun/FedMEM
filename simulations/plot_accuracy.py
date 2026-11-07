# ラウンドに対する精度

import json
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


if __package__:
    from .plot_common import (
        METHODS,
        experiment_directories,
        make_title,
        cohort_size,
    )
else:
    from plot_common import (
        METHODS,
        experiment_directories,
        make_title,
        cohort_size,
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alpha", choices=("0.5", "1e10"))
    args = parser.parse_args(argv)

    plt.rcParams.update({
        "font.size": 10,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })

    for alpha, directory in experiment_directories(
        args.root, args.alpha
    ):
        curves = {}
        reference_rounds = None

        for method in METHODS:
            path = directory / method / "round_metrics.csv"
            df = pd.read_csv(path)

            rounds = pd.to_numeric(
                df["round"], errors="raise"
            ).to_numpy()
            accuracy = pd.to_numeric(
                df["accuracy"], errors="raise"
            ).to_numpy()

            if len(df) < 2 or not np.array_equal(
                rounds, np.arange(len(df))
            ):
                raise ValueError(f"ラウンド番号が不正です: {path}")

            if (
                not np.isfinite(accuracy).all()
                or ((accuracy < 0) | (accuracy > 100)).any()
            ):
                raise ValueError(f"精度の値が不正です: {path}")

            if reference_rounds is None:
                reference_rounds = rounds
            elif not np.array_equal(rounds, reference_rounds):
                raise ValueError("手法間でラウンド数が異なります")

            curves[method] = (rounds, accuracy)

        fig, ax = plt.subplots(figsize=(6.4, 4.2))

        for method, (label, color, style) in METHODS.items():
            rounds, accuracy = curves[method]
            ax.plot(
                rounds,
                accuracy,
                label=label,
                color=color,
                linestyle=style,
                linewidth=1.6,
            )

        title = make_title(directory, alpha)

        ax.set(
            title=title,
            xlabel="Global round",
            ylabel="Mean client accuracy (%)",
            xlim=(0, reference_rounds[-1]),
            ylim=(0, 100),
        )
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

        output = directory / "figures"
        output.mkdir(parents=True, exist_ok=True)

        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(
                output / f"accuracy.{extension}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)

        print(f"精度グラフを保存しました: {output}")


if __name__ == "__main__":
    main()