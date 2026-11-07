"""端末ごとの帯域幅をラウンド別の積み上げ棒グラフで表示する。"""

# python3 simulations/plot_bandwidth.py   --root results/all_patterns/20261105_052456_112753139

import argparse
import json
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


def load_bandwidth(directory, method):
    """スキップ時の候補帯域は使用量に含めず、実際の割当だけを返す。"""
    folder = directory / method
    with (folder / "metadata.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    num_clients = int(metadata["config"]["FL"]["num_clients"])
    metrics = pd.read_csv(folder / "round_metrics.csv")
    metrics = metrics.loc[metrics["round"] > 0].copy()
    rounds = metrics["round"].to_numpy()
    if len(rounds) == 0 or not np.array_equal(rounds, np.arange(1, len(rounds) + 1)):
        raise ValueError(f"ラウンド番号が不正です: {folder}")
    flags = metrics["skipped"].astype(str).str.lower()
    if not flags.isin(["true", "false"]).all():
        raise ValueError(f"スキップ記録が不正です: {folder}")
    skipped = flags.eq("true").to_numpy()

    df = pd.read_csv(folder / "client_conditions.csv")
    if df.duplicated(["round", "client"]).any():
        raise ValueError(f"端末別の記録が重複しています: {folder}")
    expected = pd.MultiIndex.from_product(
        [rounds, range(num_clients)], names=["round", "client"]
    )
    actual = pd.MultiIndex.from_frame(df[["round", "client"]])
    if len(actual) != len(expected) or not actual.difference(expected).empty:
        raise ValueError(f"端末・ラウンドの記録に欠落または余分があります: {folder}")

    df["bandwidth_hz"] = pd.to_numeric(df["bandwidth_hz"], errors="raise")
    bandwidth = df.pivot(
        index="round", columns="client", values="bandwidth_hz"
    ).reindex(index=rounds, columns=range(num_clients)).to_numpy(dtype=float, copy=True)
    # 実行不能時にはinfや候補値が保存されるため、先にゼロへ置き換える。
    bandwidth[skipped, :] = 0.0
    if method == "E":
        bandwidth[:] = 0.0  # Error Freeには無線帯域の割当がない。
    if np.isinf(bandwidth).any() or (bandwidth < 0).any():
        raise ValueError(f"実行ラウンドの帯域幅が不正です: {folder}")
    # 非参加端末のNaNは帯域割当ゼロを表す。
    bandwidth = np.nan_to_num(bandwidth, nan=0.0)
    return rounds, bandwidth / 1e6, skipped


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--alpha", choices=("0.5", "1e10"))
    parser.add_argument("--method", choices=tuple(METHODS), nargs="+",
                        default=list(METHODS))
    args = parser.parse_args(argv)
    plt.rcParams.update({"font.size": 10, "pdf.fonttype": 42})

    for alpha, directory in experiment_directories(args.root, args.alpha):
        title = make_title(directory, alpha)
        for method in dict.fromkeys(args.method):
            rounds, bandwidth, skipped = load_bandwidth(directory, method)
            fig, ax = plt.subplots(figsize=(10, 5))
            bottom = np.zeros(len(rounds))
            colors = plt.get_cmap("tab20")
            for client in range(bandwidth.shape[1]):
                values = bandwidth[:, client]
                ax.bar(rounds, values, bottom=bottom, width=0.8,
                       color=colors(client % 20), label=f"Client {client}")
                bottom += values
            if skipped.any():
                ax.scatter(rounds[skipped], np.zeros(skipped.sum()),
                           marker="x", color="red", s=25, zorder=3,
                           label="Skipped round")
            if method == "E":
                ax.text(0.5, 0.5, "Error Free: no wireless bandwidth allocation",
                        transform=ax.transAxes, ha="center", va="center")
            ax.set(title=f"{title}\n{METHODS[method][0]}",
                   xlabel="Global round", ylabel="Allocated bandwidth (MHz)",
                   xlim=(0.3, rounds[-1] + 0.7),
                   ylim=(0, max(float(bottom.max()) * 1.1, 0.001)))
            ticks = np.unique(np.r_[1, rounds[4::5], rounds[-1]])
            ax.set_xticks(ticks)
            ax.grid(axis="y", alpha=0.3)
            ax.set_axisbelow(True)
            ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1),
                      fontsize=8, ncol=2)
            output = directory / "figures"
            output.mkdir(parents=True, exist_ok=True)
            fig.tight_layout()
            for extension in ("png", "pdf"):
                fig.savefig(output / f"bandwidth_{method}.{extension}",
                            dpi=300, bbox_inches="tight")
            plt.close(fig)
            print(f"帯域幅グラフを保存しました: {output / f'bandwidth_{method}'}")


if __name__ == "__main__":
    main()
