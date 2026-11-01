import argparse
import copy
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .client import train, test_exit, fine_tune_personal
from .dataset import (
    get_datasets,
    split_dataset,
    split_client_tests,
    original_indices,
)
from .models import MODEL_SPECS
from .server import federated_learning
from .utils import load_config, select_optimizer, set_seed
from .round_resources import (
    count_exit_flops,
    make_environment,
    plan_round,
    gamma_boundary,
)


def full_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }


def shared_state(model, prefixes):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
        if key.startswith(prefixes)
    }


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--alpha",
        type=float,
        choices=(0.5, 1e10),
        required=True,
    )
    parser.add_argument(
        "--method",
        choices=("A", "B", "C", "D"),
        required=True,
    )
    parser.add_argument(
        "--gamma", type=float, required=True
    )
    parser.add_argument(
        "--seed", type=int, default=0
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/round_flops"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="学習せずに資源割当とスキップ数を確認する",
    )

    args = parser.parse_args()

    if not math.isfinite(args.gamma) or args.gamma <= 0:
        raise ValueError("gammaは有限の正の値が必要です")

    if args.seed < 0:
        raise ValueError("seedは非負の値が必要です")

    root = Path(__file__).resolve().parent.parent
    os.chdir(root)

    config = load_config(str(root))
    training = config["training"]
    fl = config["FL"]
    communication = config["communication"]

    if training["dataset_name"] != "CIFAR10":
        raise ValueError("この実験ではCIFAR10を使用します")

    if fl["num_clients"] != 20 or fl["cohort"] != 20:
        raise ValueError("全20端末が参加する設定にしてください")

    if fl["ite_num"] != 1:
        raise ValueError("この実行ファイルは1回につき1試行です")

    num_clients = fl["num_clients"]
    rounds = fl["global_rounds"]
    batch_size = training["batch_size"]
    epochs = training["epochs"]
    lr = training["lr"]
    optimizer_name = training["optim_name"]

    model_name = "eenet20_half"
    spec = MODEL_SPECS[model_name]
    exits = tuple(spec["exit_ids"])
    prefixes = spec["shared_prefixes"]

    bandwidth_total = float(
        communication["total_bandwidth"]
    )
    power = 10.0 ** (
        (communication["power_dbm"] - 30.0) / 10.0
    )
    environment_seed = int(
        communication["channel_seed"]
    )

    # 性能倍率  [1e9, 2e9, 5e9, 1e9, 2e9, 5e9, 1e9, 2e9, 5e9, 1e9]
    cpu_hz = np.asarray([
        (1e9, 2e9, 5e9)[i % 3]
        for i in range(num_clients)
    ])
    flops_per_cycle = 5.0

    device = (
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    train_loader, _, test_loader = get_datasets(
        batch_size,
        "CIFAR10",
        val_ratio=0.0,
    )

    set_seed(environment_seed + args.seed)

    global_model = spec["factory"](
        num_classes=10
    ).to(device)

    initial = full_state(global_model)
    flops = count_exit_flops(global_model)

    size_bits = sum(
        tensor.numel() * tensor.element_size() * 8
        for tensor in shared_state(
            global_model, prefixes
        ).values()
    )

    set_seed(args.seed)

    subsets = split_dataset(
        train_loader.dataset,
        num_clients,
        args.alpha,
    )

    test_seed = 200000 + args.seed

    test_subsets, train_counts, test_counts = (
        split_client_tests(
            subsets,
            test_loader.dataset,
            seed=test_seed,
            num_classes=10,
        )
    )

    if any(len(subset) == 0 for subset in subsets):
        raise ValueError("学習データが空の端末があります")

    sample_counts = np.asarray([
        len(subset) for subset in subsets
    ])

    """
    comp = [
        [端末0のExit10の時間, 端末0のExit16の時間, 端末0のExit20の時間],
        [端末1のExit10の時間, 端末1のExit16の時間, 端末1のExit20の時間],
        ...
        [端末9のExit10の時間, 端末9のExit16の時間, 端末9のExit20の時間],
    ]
    """
    comp = (
        sample_counts[:, None]
        * np.asarray([flops[e] for e in exits])[None, :]
        / (cpu_hz[:, None] * flops_per_cycle)
    )

    distances, channels, noises = make_environment(
        num_clients,
        rounds,
        environment_seed,
        args.seed,
    )

    """
    plans = [
        (choices, bandwidth, updated, eta),  # 第1ラウンド
        (choices, bandwidth, updated, eta),  # 第2ラウンド
        ...
    ]
    """
    plans = [
        plan_round(
            args.method,
            comp,
            channels[t],
            float(noises[t]),
            power,
            size_bits,
            bandwidth_total,
            args.gamma,
        )
        for t in range(rounds)
    ]

    print("1画像当たりのFLOPs:", flops)
    print("共有層の送信量:", size_bits, "bit")
    print("CPU周波数:", cpu_hz.tolist())

    for index, exit_id in enumerate(exits):
        print(
            f"Exit{exit_id}: "
            f"最大計算遅延="
            f"{comp[:, index].max():.3f}秒"
        )

    predicted_skips = sum(
        not plan[2] for plan in plans
    )
    print(
        f"更新スキップ予定: {predicted_skips}/{rounds}"
    )

    gamma_rows = []

    print("\n===== 通信＋計算を含む最小gamma =====")

    for t in range(rounds):
        row = {"round": t + 1}

        for index, exit_id in enumerate(exits):
            """
            row = {
                "round": 1,
                "gamma_exit10_s": 12.3,
                "gamma_exit16_s": 18.5,
                "gamma_exit20_s": 25.7,
            }
            """
            row[f"gamma_exit{exit_id}_s"] = gamma_boundary(
                comp_one_exit=comp[:, index],
                gains=channels[t],
                noise=float(noises[t]),
                power=power,
                size_bits=size_bits,
                total_bandwidth=bandwidth_total,
            )

        gamma_rows.append(row)
        """
        gamma_rows = [
            {"round": 1, "gamma_exit10_s": 12.3,
                        "gamma_exit16_s": 18.5, "gamma_exit20_s": 25.7},
            {"round": 2, "gamma_exit10_s": 13.1,
                        "gamma_exit16_s": 19.2, "gamma_exit20_s": 26.4},
            # ... 第30ラウンドまで
        ]
        """

        values = ", ".join(
            f"Exit{exit_id}={row[f'gamma_exit{exit_id}_s']:.3f}秒"
            for exit_id in exits
        )
        # values = "Exit10=12.300秒, Exit16=18.500秒, Exit20=25.700秒"
        print(f"Round {t + 1}: {values}")

    gamma_df = pd.DataFrame(gamma_rows)

    print("\n===== 全ラウンドのgamma境界の分布 =====")

    for exit_id in exits:
        values = gamma_df[f"gamma_exit{exit_id}_s"].to_numpy()
        # exit_id == 10 の場合の例
        # values = np.array([12.3, 13.1, 11.8, ...])


        print(
            f"全員Exit{exit_id}: "
            f"最小={np.min(values):.3f}秒, "
            f"中央値={np.median(values):.3f}秒, "
            f"最大={np.max(values):.3f}秒"
        )

    shallow_max = gamma_df[
        f"gamma_exit{exits[0]}_s"
    ].max()

    deep_max = gamma_df[
        f"gamma_exit{exits[-1]}_s"
    ].max()

    print(
        f"\n全ラウンドで最浅Exitの更新が可能になるgamma: "
        f"{shallow_max:.3f}秒以上"
    )
    print(
        f"全ラウンドで全員が最深Exitを使用可能になるgamma: "
        f"{deep_max:.3f}秒以上"
    )
    
    if args.dry_run:
        return

    alpha_tag = (
        "0.5" if args.alpha == 0.5 else "1e10"
    )
    output = (
        args.output_root
        / f"alpha_{alpha_tag}"
        / args.method
    )

    # 既存結果を上書きしない
    output.mkdir(parents=True, exist_ok=False)

    np.savez_compressed(
        output / "environment.npz",
        distances_m=distances,
        channels=channels,
        noise_w=noises,
        cpu_hz=cpu_hz,
        computation_s=comp,
    )

    split_record = {
        "train_counts": train_counts,
        "test_counts": test_counts,
        "train_split_seed": np.asarray(args.seed),
        "test_split_seed": np.asarray(test_seed),
    }

    for i in range(num_clients):
        split_record[f"train_indices_{i}"] = (
            original_indices(subsets[i])
        )
        split_record[f"test_indices_{i}"] = (
            original_indices(test_subsets[i])
        )

    np.savez_compressed(
        output / "split.npz",
        **split_record,
    )

    with (output / "metadata.json").open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump({
            "model_name": model_name,
            "method": args.method,
            "alpha": args.alpha,
            "seed": args.seed,
            "gamma": args.gamma,
            "rounds": rounds,
            "config": config,
            "cpu_hz": cpu_hz.tolist(),
            "flops_per_cycle": flops_per_cycle,
            "flops_per_image": flops,
            "size_bits": size_bits,
            "environment_seed": environment_seed,
            "noise_reference_w": 1e-10,
            "noise_variation_db": [-3.0, 3.0],
            "device": device,
            "torch_version": str(torch.__version__),
        }, file, indent=2)

    test_loaders = [
        DataLoader(
            subset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            generator=torch.Generator().manual_seed(
                2000 + i
            ),
        )
        for i, subset in enumerate(test_subsets)
    ]

    client_states = [
        copy.deepcopy(initial)
        for _ in range(num_clients)
    ]

    # Cは最浅Exit、A・B・Dは最深Exitで初期モデルを評価する
    evaluation_exits = [
        exits[0] if args.method == "C" else exits[-1]
    ] * num_clients

    def evaluate_clients():
        current_shared = shared_state(
            global_model, prefixes
        )
        evaluation_model = copy.deepcopy(global_model)
        accuracies = []

        for i in range(num_clients):
            evaluation_model.load_state_dict(
                client_states[i],
                strict=True,
            )
            evaluation_model.load_state_dict(
                current_shared,
                strict=False,
            )

            accuracies.append(
                test_exit(
                    evaluation_model,
                    test_loaders[i],
                    device,
                    evaluation_exits[i],
                )
            )

        return accuracies

    last_accuracies = evaluate_clients()

    history = [{
        "seed": args.seed,
        "round": 0,
        "accuracy": float(np.mean(last_accuracies)),
        "updated": False,
        "skipped": False,
        "exit_index_sum": np.nan,
        "noise_w": np.nan,
    }]
    conditions = []

    for t, plan in enumerate(plans, start=1):
        choices, bandwidth, updated, eta = plan

        if updated:
            # 同じ物理ラウンドでは、手法間で学習乱数を揃える
            set_seed(
                300000 + 1000 * args.seed + t
            )

            loaders = [
                DataLoader(
                    subset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                    generator=torch.Generator().manual_seed(
                        1000000 * args.seed + 1000 * t + i
                    ),
                )
                for i, subset in enumerate(subsets)
            ]

            updates = []
            current_shared = shared_state(
                global_model, prefixes
            )

            for i in range(num_clients):
                local_model = copy.deepcopy(global_model)
                local_model.load_state_dict(
                    client_states[i],
                    strict=True,
                )
                local_model.load_state_dict(
                    current_shared,
                    strict=False,
                )

                optimizer = select_optimizer(
                    optimizer_name,
                    local_model,
                    lr,
                )

                for _ in range(epochs):
                    train(
                        optimizer,
                        local_model,
                        loaders[i],
                        device,
                        model_name,
                    )

                client_states[i] = full_state(local_model)

                updates.append(
                    shared_state(local_model, prefixes)
                )
                """
                updates = [
                    端末0の学習後の共有部分,
                    端末1の学習後の共有部分,
                    # ...
                    端末9の学習後の共有部分,
                ]
                """
            federated_learning(
                updates,
                global_model,
                sample_counts.tolist(),
            )

            new_shared = shared_state(
                global_model, prefixes
            )

            for i in range(num_clients):
                personal_model = copy.deepcopy(global_model)
                personal_model.load_state_dict(
                    client_states[i],
                    strict=True,
                )
                personal_model.load_state_dict(
                    new_shared,
                    strict=False,
                )

                optimizer = select_optimizer(
                    optimizer_name,
                    personal_model,
                    lr,
                )

                fine_tune_personal(
                    optimizer,
                    personal_model,
                    loaders[i],
                    device,
                    model_name,
                    prefixes,
                    epochs=1,
                )

                client_states[i] = full_state(
                    personal_model
                )

            evaluation_exits = [
                exits[int(choice) - 1]
                for choice in choices
            ]

            last_accuracies = evaluate_clients()
            exit_sum = int(np.sum(choices))

        else:
            # 学習・集約・微調整を行わず、評価Exitも維持する
            exit_sum = np.nan

        history.append({
            "seed": args.seed,
            "round": t,
            "accuracy": float(np.mean(last_accuracies)),
            "updated": bool(updated),
            "skipped": not bool(updated),
            "exit_index_sum": exit_sum,
            "noise_w": float(noises[t - 1]),
        })

        for i in range(num_clients):
            candidate = (
                int(choices[i])
                if choices is not None
                else None
            )

            b = (
                float(bandwidth[i])
                if bandwidth is not None
                else np.nan
            )

            # 選択Exitでの計算時間
            tc = (
                float(comp[i, candidate - 1])
                if candidate is not None
                else np.nan
            )

            # 通信時間 [s]
            tt = (
                size_bits / (b * eta[i])
                if (
                    np.isfinite(b)
                    and b > 0
                    and eta[i] > 0
                )
                else np.nan
            )

            conditions.append({
                "seed": args.seed,
                "round": t,
                "client": i,
                "updated": bool(updated),
                "distance_m": float(distances[i]),
                "cpu_hz": float(cpu_hz[i]),
                "noise_w": float(noises[t - 1]),
                "channel_real": float(
                    channels[t - 1, i].real
                ),
                "channel_imag": float(
                    channels[t - 1, i].imag
                ),
                "candidate_exit": (
                    exits[candidate - 1]
                    if candidate is not None
                    else np.nan
                ),
                "evaluation_exit": evaluation_exits[i],
                "bandwidth_hz": b,
                "computation_s": tc,
                "transmission_s": tt,
                "total_s": tc + tt,
            })

        pd.DataFrame(history).to_csv(
            output / "round_metrics.csv",
            index=False,
        )
        pd.DataFrame(conditions).to_csv(
            output / "client_conditions.csv",
            index=False,
        )

        status = "更新" if updated else "スキップ"

        print(
            f"Round {t}/{rounds}: {status}, "
            f"Accuracy={np.mean(last_accuracies):.3f}%"
        )

    # 最終ラウンドのみ保存する。
    # 最終ラウンドがスキップされた場合は、維持された重みを保存する。
    torch.save({
        "format_version": 1,
        "model_name": model_name,
        "dataset_name": "CIFAR10",
        "num_classes": 10,
        "num_clients": num_clients,
        "batch_size": batch_size,
        "method": args.method,
        "alpha": args.alpha,
        "seed": args.seed,
        "round": rounds,
        "gamma": args.gamma,
        "config": config,
        "exit_ids": list(exits),
        "selected_exits": list(evaluation_exits),
        "selected_test_accuracy": list(last_accuracies),
        "client_states": client_states,
        "train_indices": [
            torch.tensor(
                original_indices(subset),
                dtype=torch.int64,
            )
            for subset in subsets
        ],
        "test_indices": [
            torch.tensor(
                original_indices(subset),
                dtype=torch.int64,
            )
            for subset in test_subsets
        ],
    }, output / f"seed_{args.seed}_round_{rounds:03d}.pt")

    skip_rate = 100.0 * predicted_skips / rounds

    print(f"更新スキップ率: {skip_rate:.2f}%")
    print(f"結果保存先: {output}")


if __name__ == "__main__":
    main()