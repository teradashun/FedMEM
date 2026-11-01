from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results" / "CIFAR10" / "eenet34_1525_quarter"
OUTPUT = RESULTS / "figures"

METHODS = (
    ("resnet34_quarter_fedmem", "FedMEM", "#0072B2", "-"),
    ("resnet34_quarter_fixed15", "Fixed Exit15", "#D55E00", "--"),
    ("resnet34_quarter_fixed34", "Fixed Exit34", "#009E73", "-."),
)

plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# 保存前に全6条件を確認
data = {}

for alpha in ("0.5", "1e10"):
    reference_rounds = None

    for method, _, _, _ in METHODS:
        path = RESULTS / f"alpha_{alpha}" / method / "accuracy.csv"
        df = pd.read_csv(path)

        rounds = pd.to_numeric(df["round"], errors="raise").to_numpy()
        accuracy = pd.to_numeric(
            df["accuracy"], errors="raise"
        ).to_numpy()

        if (
            len(df) == 0
            or not np.isfinite(rounds).all()
            or not np.isfinite(accuracy).all()
            or ((accuracy < 0) | (accuracy > 100)).any()
        ):
            raise ValueError(f"不正な結果があります: {path}")

        if not np.array_equal(rounds, np.arange(1, len(df) + 1)):
            raise ValueError(f"ラウンド番号を確認してください: {path}")

        if reference_rounds is None:
            reference_rounds = rounds
        elif not np.array_equal(rounds, reference_rounds):
            raise ValueError(f"手法間でラウンドが一致しません: {path}")

        data[(alpha, method)] = (rounds, accuracy)

OUTPUT.mkdir(parents=True, exist_ok=True)

for alpha in ("0.5", "1e10"):
    fig, ax = plt.subplots(figsize=(3.5, 2.7))

    for method, label, color, linestyle in METHODS:
        rounds, accuracy = data[(alpha, method)]
        ax.plot(
            rounds,
            accuracy,
            label=label,
            color=color,
            linestyle=linestyle,
            linewidth=1.6,
        )

    title = (
        r"$\alpha = 0.5$"
        if alpha == "0.5"
        else r"$\alpha = 10^{10}$"
    )

    ax.set_title(title)
    ax.set_xlabel("Global round")
    ax.set_ylabel("Mean client accuracy (%)")
    ax.set_xlim(1, rounds[-1])
    ax.set_ylim(0, 100)
    ax.set_xticks([1, 5, 10, 15, 20, 25, 30])
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")

    fig.tight_layout()

    basename = OUTPUT / f"accuracy_alpha_{alpha}"
    fig.savefig(basename.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(basename.with_suffix(".png"), dpi=300,
                bbox_inches="tight")
    plt.close(fig)

    print(f"保存しました: {basename}.pdf / .png")