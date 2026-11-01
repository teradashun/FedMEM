import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHODS = {
    "A": ("A: Fixed20 Ideal", "#000000", ":"),
    "B": ("B: Fixed20 constrained", "#009E73", "-."),
    "C": ("C: Fixed10", "#D55E00", "--"),
    "D": ("D: FedMEM", "#0072B2", "-"),
}


def make_title(directory, alpha):
    gammas = {}

    # 遅延制約を適用するB・C・Dの設定を確認する。
    for method in ("B", "C", "D"):
        path = directory / method / "metadata.json"
        with path.open(encoding="utf-8") as file:
            gamma = float(json.load(file)["gamma"])

        if not np.isfinite(gamma) or gamma <= 0:
            raise ValueError(f"gammaが不正です: {path}")

        gammas[method] = gamma

    if len(set(gammas.values())) != 1:
        raise ValueError(f"手法間でgammaが異なります: {gammas}")

    gamma = gammas["D"]
    alpha_tex = "0.5" if alpha == "0.5" else r"10^{10}"

    return (
        rf"$\alpha={alpha_tex},\ "
        rf"\gamma={gamma:g}\,\mathrm{{s}}$"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    plt.rcParams.update({
        "font.size": 10,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
    })

    for alpha in ("0.5", "1e10"):
        directory = args.root / f"alpha_{alpha}"
        curves = {}
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
                or ((values < 10) | (values > 30)).any()
                or not np.equal(values, np.floor(values)).all()
            ):
                raise ValueError(f"Exit番号の総和が不正です: {path}")

            x, counts = np.unique(values, return_counts=True)
            y = 100.0 * np.cumsum(counts) / len(values)

            curves[method] = (
                np.r_[9, x, 31],
                np.r_[0, y, 100],
            )

        if not curves:
            print(f"alpha={alpha}: CDFの対象データがありません")
            continue

        fig, ax = plt.subplots(figsize=(4.4, 3.2))

        for method, (x, y) in curves.items():
            label, color, style = METHODS[method]
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
            title=title,
            xlabel="Sum of selected Exit indices",
            ylabel="Empirical CDF (%)",
            xlim=(9, 31),
            ylim=(0, 100),
        )
        ax.set_xticks([10, 15, 20, 25, 30])
        ax.grid(alpha=0.3)
        ax.legend()

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