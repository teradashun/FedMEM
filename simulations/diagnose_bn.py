import argparse
from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor

from src.models import MODEL_SPECS


def make_loader(dataset, indices, batch_size):
    return DataLoader(
        Subset(dataset, indices.tolist()),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        # 診断用の独立した乱数系列
        generator=torch.Generator().manual_seed(0),
    )


@torch.no_grad()
def evaluate(model, loader, device, exit_ids):
    model.eval()

    correct = [0] * len(exit_ids)
    losses = [0.0] * len(exit_ids)
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        if len(outputs) != len(exit_ids):
            raise ValueError("モデル出力数とExit数が一致しません")

        total += labels.numel()

        for k, logits in enumerate(outputs):
            correct[k] += (
                logits.argmax(dim=1) == labels
            ).sum().item()

            losses[k] += F.cross_entropy(
                logits, labels, reduction="sum"
            ).item()

    if total == 0:
        raise ValueError("評価対象のデータが空です")

    return [
        {
            "exit": exit_id,
            "accuracy": 100.0 * correct[k] / total,
            # 損失の重み0.2、0.3、0.5を掛ける前のCE
            "cross_entropy": losses[k] / total,
        }
        for k, exit_id in enumerate(exit_ids)
    ]


@torch.no_grad()
def recalibrate_private_bn(
    model, train_loader, device, shared_prefixes
):
    model.eval()
    targets = []

    for name, module in model.named_modules():
        is_shared = (name + ".").startswith(
            shared_prefixes
        )

        if (
            isinstance(module, torch.nn.BatchNorm2d)
            and not is_shared
            and module.track_running_stats
        ):
            module.reset_running_stats()
            module.momentum = None
            module.train()
            targets.append(module)

    if not targets:
        raise ValueError("個人化層のBNが見つかりません")

    # ラベルは使用しない。逆伝播・パラメータ更新も行わない。
    for images, _ in train_loader:
        model(images.to(device))

    model.eval()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", type=Path, required=True
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--device",
        choices=("cpu", "cuda"),
        default="cpu",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    data_root = (
        args.data_root
        if args.data_root is not None
        else root / "data" / "CIFAR_10"
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDAを使用できません。--device cpuで実行してください。"
        )

    device = torch.device(args.device)

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=True,
    )

    spec = MODEL_SPECS[checkpoint["model_name"]]
    prefixes = tuple(spec["shared_prefixes"])
    exit_ids = tuple(checkpoint["exit_ids"])
    states = checkpoint["client_states"]

    if len(states) != checkpoint["num_clients"]:
        raise ValueError("保存された端末数が一致しません")

    train_dataset = CIFAR10(
        root=str(data_root),
        train=True,
        download=False,
        transform=ToTensor(),
    )
    test_dataset = CIFAR10(
        root=str(data_root),
        train=False,
        download=False,
        transform=ToTensor(),
    )

    rows = []

    for client, state in enumerate(states):
        print(
            f"端末 {client + 1}/{len(states)} を診断中",
            flush=True,
        )

        train_loader = make_loader(
            train_dataset,
            checkpoint["train_indices"][client],
            checkpoint["batch_size"],
        )
        test_loader = make_loader(
            test_dataset,
            checkpoint["test_indices"][client],
            checkpoint["batch_size"],
        )

        model = spec["factory"](
            num_classes=checkpoint["num_classes"]
        ).to(device)
        model.load_state_dict(state, strict=True)

        for variant in (
            "saved_bn",
            "recalibrated_private_bn",
        ):
            if variant == "recalibrated_private_bn":
                recalibrate_private_bn(
                    model,
                    train_loader,
                    device,
                    prefixes,
                )

                # 学習パラメータは変更しない。
                for key, value in model.named_parameters():
                    if not torch.equal(
                        value.detach().cpu(), state[key]
                    ):
                        raise RuntimeError(
                            f"パラメータが変化しました: {key}"
                        )

                # 共有層のBN統計も変更しない。
                for key, value in model.state_dict().items():
                    if key.startswith(prefixes):
                        if not torch.equal(
                            value.detach().cpu(), state[key]
                        ):
                            raise RuntimeError(
                                f"共有層が変化しました: {key}"
                            )

            for split, loader in (
                ("train", train_loader),
                ("test", test_loader),
            ):
                metrics = evaluate(
                    model, loader, device, exit_ids
                )

                for metric in metrics:
                    rows.append({
                        "alpha": checkpoint["alpha"],
                        "method": checkpoint["method"],
                        "round": checkpoint["round"],
                        "client": client,
                        "variant": variant,
                        "split": split,
                        **metric,
                    })

        del model

    frame = pd.DataFrame(rows)

    output = (
        args.output
        if args.output is not None
        else args.checkpoint.parent
        / "bn_diagnostics.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)

    summary = (
        frame.groupby(
            ["variant", "split", "exit"],
            sort=False,
        )[["accuracy", "cross_entropy"]]
        .mean()
    )

    print("\n端末を等重みで平均した診断結果")
    print(summary.to_string())
    print(f"\n保存先: {output}")


if __name__ == "__main__":
    main()