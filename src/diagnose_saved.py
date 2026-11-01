import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from .dataset import get_datasets
from .models import MODEL_SPECS


@torch.inference_mode()
def evaluate_all_exits(
    model,
    dataset,
    batch_size,
    device,
    num_classes,
):
    model.eval()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(0),
    )

    exit_ids = tuple(model.exit_ids)
    losses = np.zeros(len(exit_ids), dtype=np.float64)
    confusion = np.zeros(
        (len(exit_ids), num_classes, num_classes),
        dtype=np.int64,
    )
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        if not isinstance(outputs, (tuple, list)):
            raise ValueError("マルチExitの出力ではありません")

        if len(outputs) != len(exit_ids):
            raise ValueError("Exit数と出力数が一致しません")

        for index, logits in enumerate(outputs):
            if logits.shape != (labels.size(0), num_classes):
                raise ValueError("出力形状が想定と異なります")

            losses[index] += F.cross_entropy(
                logits,
                labels,
                reduction="sum",
            ).item()

            predictions = logits.argmax(dim=1)

            counts = torch.bincount(
                labels * num_classes + predictions,
                minlength=num_classes * num_classes,
            )

            confusion[index] += (
                counts.reshape(num_classes, num_classes)
                .cpu()
                .numpy()
            )

        total += labels.size(0)

    if total == 0:
        raise ValueError("評価データが空です")

    metrics = []
    class_metrics = []

    for index, exit_id in enumerate(exit_ids):
        matrix = confusion[index]
        support = matrix.sum(axis=1)
        correct = matrix.diagonal()
        represented = support > 0

        recalls = np.full(num_classes, np.nan)
        recalls[represented] = (
            100.0 * correct[represented] / support[represented]
        )

        metrics.append({
            "exit_layer": exit_id,
            "samples": total,
            "accuracy": 100.0 * correct.sum() / total,
            "loss": losses[index] / total,

            # サンプル数で重み付けしないクラス別正解率の平均
            # サンプルが存在するクラスのみを対象にする
            "mean_class_accuracy": recalls[represented].mean(),
            "represented_classes": int(represented.sum()),
        })

        for class_id in range(num_classes):
            class_metrics.append({
                "exit_layer": exit_id,
                "class_id": class_id,
                "samples": int(support[class_id]),
                "correct": int(correct[class_id]),
                "accuracy": recalls[class_id],
            })

    return metrics, class_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default=None,
    )
    args = parser.parse_args()

    checkpoint_path = args.checkpoint.resolve()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    device = args.device or (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model_name = checkpoint["model_name"]
    spec = MODEL_SPECS[model_name]

    if tuple(spec["exit_ids"]) != tuple(checkpoint["exit_ids"]):
        raise ValueError("保存時と現在のExit構成が異なります")

    # 学習時と同じデータ保存先を使う
    project_root = Path(__file__).resolve().parent.parent

    import os
    os.chdir(project_root)

    train_loader, _, test_loader = get_datasets(
        checkpoint["batch_size"],
        checkpoint["dataset_name"],
        val_ratio=0.0,
    )

    datasets = {
        "train": train_loader.dataset,
        "test": test_loader.dataset,
    }

    model = spec["factory"](
        num_classes=checkpoint["num_classes"],
    ).to(device)

    rows = []
    class_rows = []

    for client_id in range(checkpoint["num_clients"]):
        # strict=Trueで構造と重みの対応を確認する
        model.load_state_dict(
            checkpoint["client_states"][client_id],
            strict=True,
        )

        selected_exit = checkpoint["selected_exits"][client_id]

        for split_name, full_dataset in datasets.items():
            indices = checkpoint[
                f"{split_name}_indices"
            ][client_id].tolist()

            if (
                len(indices) == 0
                or min(indices) < 0
                or max(indices) >= len(full_dataset)
                or len(set(indices)) != len(indices)
            ):
                raise ValueError("保存されたサンプル番号が不正です")

            subset = Subset(full_dataset, indices)

            metrics, per_class = evaluate_all_exits(
                model,
                subset,
                checkpoint["batch_size"],
                device,
                checkpoint["num_classes"],
            )

            common = {
                "alpha": checkpoint["alpha"],
                "seed": checkpoint["seed"],
                "round": checkpoint["round"],
                "client": client_id,
                "split": split_name,
            }

            for metric in metrics:
                row = {
                    **common,
                    "selected_exit": selected_exit,
                    **metric,
                }

                # 学習実行時の評価と再評価が一致するか確認
                if (
                    split_name == "test"
                    and metric["exit_layer"] == selected_exit
                ):
                    reference = checkpoint[
                        "selected_test_accuracy"
                    ][client_id]

                    row["saved_accuracy"] = reference
                    row["accuracy_difference_pp"] = (
                        metric["accuracy"] - reference
                    )

                rows.append(row)

            class_rows.extend(
                {**common, **metric}
                for metric in per_class
            )

    # 再診断時にも前の診断結果を上書きしない
    from datetime import datetime

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    output_dir = (
        checkpoint_path.parent
        / f"diagnosis_{checkpoint_path.stem}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=False)

    df = pd.DataFrame(rows)

    df.to_csv(
        output_dir / "exit_metrics.csv",
        index=False,
    )
    pd.DataFrame(class_rows).to_csv(
        output_dir / "class_metrics.csv",
        index=False,
    )

    summary = df.groupby(
        ["split", "exit_layer"],
        as_index=False,
    )[["accuracy", "loss", "mean_class_accuracy"]].mean()

    summary.to_csv(
        output_dir / "summary.csv",
        index=False,
    )

    print(summary.to_string(index=False))

    checked = df["accuracy_difference_pp"].dropna()
    print(
        "保存時との最大精度差:",
        float(checked.abs().max()),
        "ポイント",
    )
    print(f"診断結果: {output_dir}")


if __name__ == "__main__":
    main()