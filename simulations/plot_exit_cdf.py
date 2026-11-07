import argparse
import json
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
        reference_cohort = None

        for method in METHODS:
            cohort = cohort_size(directory, method)

            if reference_cohort is None:
                reference_cohort = cohort
            elif cohort != reference_cohort:
                raise ValueError(
                    "手法間で参加端末数が異なります"
                )

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

            accepted = df[flags.eq("false")]
            values = pd.to_numeric(
                accepted["exit_index_sum"],
                errors="raise",
            ).to_numpy(dtype=float)

            if len(values) == 0:
                print(
                    f"alpha={alpha}, {method}: "
                    "更新がないためCDFを省略します"
                )
                continue

            if (
                not np.isfinite(values).all()
                or ((values < cohort) | (values > 3 * cohort)).any()
                or not np.equal(values, np.floor(values)).all()
            ):
                raise ValueError(f"Exit番号の総和が不正です: {path}")

            x, counts = np.unique(values, return_counts=True)
            y = 100.0 * np.cumsum(counts) / len(values)

            curves[method] = (
                np.r_[cohort - 1, x, 3 * cohort + 1],
                np.r_[0, y, 100],
            )

        fig, ax = plt.subplots(figsize=(6.4, 4.2))

        for method in METHODS:
            label, color, style = METHODS[method]

            if method not in curves:
                ax.plot(
                    [], [],
                    color=color,
                    linestyle=style,
                    label=f"{label} (no updates)",
                )
                continue

            x, y = curves[method]

            ax.step(
                x,
                y,
                where="post",
                label=label,
                color=color,
                linestyle=style,
                linewidth=1.6,
            )

        title = make_title(directory, alpha)

        ax.set(
            title=make_title(directory, alpha),
            xlabel="Sum of selected Exit indices",
            ylabel="Empirical CDF (%)",
            xlim=(
                reference_cohort - 1,
                3 * reference_cohort + 1,
            ),
            ylim=(0, 100),
        )

        ax.set_xticks([
            reference_cohort,
            2 * reference_cohort,
            3 * reference_cohort,
        ])

        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)

        if not curves:
            ax.text(
                0.5, 0.5,
                "No accepted updates",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )

        output = directory / "figures"
        output.mkdir(parents=True, exist_ok=True)

        fig.tight_layout()
        for extension in ("pdf", "png"):
            fig.savefig(
                output / f"exit_sum_cdf.{extension}",
                dpi=300,
                bbox_inches="tight",
            )
        plt.close(fig)


if __name__ == "__main__":
    main()