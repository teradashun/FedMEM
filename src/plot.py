from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

from .utils import load_config


def main():
    parent_dir = Path(__file__).resolve().parent.parent
    config = load_config(str(parent_dir))

    dataset_name = config["training"]["dataset_name"]
    model_name = config["training"]["model_name"]

    result_dir = (
        parent_dir / "results" / dataset_name / model_name
    )

    conditions = [
        ("0.5", r"$\alpha=0.5$"),
        ("1e10", r"$\alpha=10^{10}$"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    expected_rounds = None

    for tag, label in conditions:
        path = result_dir / f"alpha_{tag}" / "accuracy.csv"
        df = pd.read_csv(path)

        rounds = df["round"].tolist()
        if expected_rounds is None:
            expected_rounds = rounds
        elif rounds != expected_rounds:
            raise ValueError("2条件のラウンド数が一致していません")

        ax.plot(
            df["round"],
            df["accuracy"],
            label=label,
        )

    ax.set_xlabel("Training round")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("FedMEM: Dirichlet alpha comparison")
    ax.set_ylim(0, 100)
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    output = result_dir / "alpha_comparison.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)

    print(f"グラフ保存先: {output}")


if __name__ == "__main__":
    main()