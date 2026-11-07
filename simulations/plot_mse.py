"""ラウンドごとの平均上り通信MSEを棒グラフにする。"""

"""
python3 simulations/plot_mse.py \
  --root results/all_patterns/20261104_004749_483302572
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__:
    from .plot_common import METHODS, experiment_directories, make_title
else:
    from plot_common import METHODS, experiment_directories, make_title


def load_mse(directory, method):
    path = directory / method / "round_metrics.csv"
    df = pd.read_csv(path)
    df = df.loc[df["round"] > 0].copy()
    rounds = pd.to_numeric(df["round"], errors="raise").to_numpy()
    if len(rounds) == 0 or not np.array_equal(rounds, np.arange(1, len(rounds) + 1)):
        raise ValueError(f"ラウンド番号が不正です: {path}")
    flags = df["skipped"].astype(str).str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError(f"スキップ記録が不正です: {path}")
    skipped = flags.eq("true").to_numpy()
    mse = pd.to_numeric(df["uplink_mse"], errors="raise").to_numpy(
        dtype=float, copy=True
    )
    if not np.isfinite(mse[~skipped]).all() or (mse[~skipped] < 0).any():
        raise ValueError(f"更新ラウンドのMSEが不正です: {path}")
    # 未送信のラウンドをMSE=0として扱わない。
    mse[skipped] = np.nan
    return rounds, mse, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alpha", choices=("0.5", "1e10"))
    parser.add_argument("--method", choices=tuple(METHODS), nargs="+",
                        default=list(METHODS))
    args = parser.parse_args(argv)
    methods = list(dict.fromkeys(args.method))
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})

    for alpha, directory in experiment_directories(args.root, args.alpha):
        data = {method: load_mse(directory, method) for method in methods}
        rounds = data[methods[0]][0]
        if any(not np.array_equal(values[0], rounds) for values in data.values()):
            raise ValueError(f"手法間でラウンド数が異なります: {directory}")

        fig, ax = plt.subplots(figsize=(12, 4.8))
        width = 0.8 / len(methods)
        maximum = 0.0
        for index, method in enumerate(methods):
            _, mse, skipped = data[method]
            positions = rounds + (index - (len(methods) - 1) / 2) * width
            valid = ~skipped
            label, color, _ = METHODS[method]
            if not valid.any():
                label += " (no updates)"
            ax.bar(positions[valid], mse[valid], width=width,
                   color=color, label=label)
            # ゼロMSEと、送信しなかったラウンドを区別する。
            zero = valid & (mse == 0)
            ax.scatter(positions[zero], np.zeros(zero.sum()), marker="_",
                       color=color, s=30, zorder=3)
            ax.scatter(positions[skipped], np.zeros(skipped.sum()), marker="x",
                       color=color, s=14, zorder=3)
            if valid.any():
                maximum = max(maximum, float(mse[valid].max()))

        ax.set(title=make_title(directory, alpha), xlabel="Global round",
               ylabel="Mean client uplink MSE",
               xlim=(0.4, rounds[-1] + 0.6),
               ylim=(0, maximum * 1.2 if maximum > 0 else 1e-6))
        ax.set_xticks(np.unique(np.r_[1, rounds[4::5], rounds[-1]]))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=8, ncol=2)
        fig.text(0.5, 0.01, "x: skipped (MSE undefined); _: zero MSE",
                 ha="center", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        output = directory / "figures"
        output.mkdir(parents=True, exist_ok=True)
        suffix = "" if methods == list(METHODS) else "_" + "_".join(methods)
        for extension in ("png", "pdf"):
            fig.savefig(output / f"uplink_mse{suffix}.{extension}",
                        dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"MSEグラフを保存しました: {output / f'uplink_mse{suffix}'}")


if __name__ == "__main__":
    main()
