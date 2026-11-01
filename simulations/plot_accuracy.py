# ラウンドに対する精度

import json
import argparse
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

        fig, ax = plt.subplots(figsize=(4.4, 3.2))

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
        ax.legend()

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