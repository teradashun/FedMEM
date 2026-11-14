# 更新スキップ率

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
        "xtick.labelsize": 8,
        "pdf.fonttype": 42,
    })

    for alpha, directory in experiment_directories(
        args.root, args.alpha
    ):
        rates = []
        reference_rounds = None

        for method in METHODS:
            path = directory / method / "round_metrics.csv"
            df = pd.read_csv(path)
            df = df[df["round"] > 0].copy()

            rounds = df["round"].to_numpy()

            if len(df) == 0 or not np.array_equal(
                rounds, np.arange(1, len(df) + 1)
            ):
                raise ValueError(f"ラウンド番号が不正です: {path}")

            if reference_rounds is None:
                reference_rounds = rounds
            elif not np.array_equal(rounds, reference_rounds):
                raise ValueError("手法間でラウンド数が異なります")

            flags = df["skipped"].astype(str).str.lower()

            if not flags.isin(["true", "false"]).all():
                raise ValueError(f"スキップ記録が不正です: {path}")

            skipped = flags.eq("true")
            rate = 100.0 * skipped.mean()
            rates.append(rate)

            print(
                f"alpha={alpha}, {method}: "
                f"{skipped.sum()}/{len(df)}ラウンド、"
                f"{rate:.2f}%"
            )

        fig, ax = plt.subplots(figsize=(7.6, 4.2))

        positions = np.arange(len(METHODS))
        bars = ax.bar(
            positions,
            rates,
            color=[value[1] for value in METHODS.values()],
        )

        for bar, rate in zip(bars, rates):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                rate + 2,
                f"{rate:.1f}%",
                ha="center",
                fontsize=9,
            )

        title = make_title(directory, alpha)

        ax.set(
            title=title,
            xlabel="Method",
            ylabel="Skipped updates (%)",
            ylim=(0, 110),
        )
        ax.set_xticks(
            positions,
            [
                label.replace(": ", "\n", 1)
                for label, color, style in METHODS.values()
            ],
        )
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

        output = directory / "figures"
        output.mkdir(parents=True, exist_ok=True)

        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(
                output / f"skip_rate.{extension}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)


if __name__ == "__main__":
    main()