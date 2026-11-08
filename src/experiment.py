"""引数の読み込み・評価・記録。準備と学習の流れは FL.py の main() を参照。"""

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from .client import test_exit
from .dataset import original_indices
from .round_resources import gamma_boundary


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


def parse_args(argv=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--alpha",
        type=float,
        choices=(0.5, 1e10),
        required=True,
    )
    parser.add_argument(
        "--method",
        choices=("A", "B", "C", "D", "E", "F"),
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
        default=Path("results"),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="学習せずに資源割当とスキップ数を確認する",
    )
    parser.add_argument(
        "--test-split",
        choices=("train_ratio", "random"),
        default="train_ratio",
        help="クライアント別テストデータの分割方式",
    )

    args = parser.parse_args(argv)

    if not math.isfinite(args.gamma) or args.gamma <= 0:
        raise ValueError("gammaは有限の正の値が必要です")

    if args.seed < 0:
        raise ValueError("seedは非負の値が必要です")


    return args


def print_resource_summary(exits, comp, channels, noises, power, size_bits,
                           bandwidth_total, plans, selected_clients):
    """学習前に、各ラウンドの参加端末の資源割当とgamma境界を確認する。"""
    rounds = len(plans)

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

    print("\n===== 通信 + 計算を含む最小gamma =====")

    for t in range(rounds):
        row = {"round": t + 1}
        participants = selected_clients[t]

        for index, exit_id in enumerate(exits):
            row[f"gamma_exit{exit_id}_s"] = gamma_boundary(
                comp_one_exit=comp[participants, index],
                gains=channels[t, participants],
                noise=float(noises[t]),
                power=power,
                size_bits=size_bits,
                total_bandwidth=bandwidth_total,
            )

        gamma_rows.append(row)

        values = ", ".join(
            f"Exit{exit_id}={row[f'gamma_exit{exit_id}_s']:.3f}秒"
            for exit_id in exits
        )
        print(f"Round {t + 1}: {values}")

    gamma_df = pd.DataFrame(gamma_rows)

    print("\n===== gamma境界の分布 =====")

    for exit_id in exits:
        values = gamma_df[f"gamma_exit{exit_id}_s"].to_numpy()

    shallow_max = gamma_df[
        f"gamma_exit{exits[0]}_s"
    ].max()

    deep_max = gamma_df[
        f"gamma_exit{exits[-1]}_s"
    ].max()


def output_directory(args, seed, ite_num, cohort, num_clients):
    """分割方式・alpha・参加台数・総台数ごとに保存先を分ける。"""
    root = args.output_root
    if ite_num > 1:
        root = root / f"seed_{seed}"
    alpha_tag = "0.5" if args.alpha == 0.5 else "1e10"
    condition = f"{args.test_split}_{alpha_tag}_{cohort}|{num_clients}"
    return root / condition / args.method


def save_split(output, subsets, test_subsets, train_counts, test_counts, seed, test_split):
    """各端末に割り当てたデータの件数・インデックス・分割条件を保存する。"""
    split_record = {
        "train_counts": train_counts,
        "test_counts": test_counts,
        "train_split_seed": np.asarray(seed),
        "test_split_seed": np.asarray(seed),
        "test_split_mode": np.asarray(test_split),
    }
    for i in range(len(subsets)):
        split_record[f"train_indices_{i}"] = original_indices(subsets[i])
        split_record[f"test_indices_{i}"] = original_indices(test_subsets[i])
    np.savez_compressed(output / "split.npz", **split_record)


def evaluate_clients(global_model, client_states, test_loaders, device, evaluation_exits, prefixes):
    """各端末の個人層と最新の共有層を組み合わせ、指定Exitで評価する。"""
    current_shared = shared_state(
        global_model, prefixes
    )
    evaluation_model = copy.deepcopy(global_model)
    accuracies = []

    for i in range(len(client_states)):
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


def make_train_loaders(subsets, batch_size, seed, global_round):
    """手法間で学習乱数を揃え、各ラウンドで同じ順序のデータを使う。"""
    return [
        DataLoader(
            subset, batch_size=batch_size, shuffle=True, drop_last=True,
            generator=torch.Generator().manual_seed(
                1000000 * seed + 1000 * global_round + i
            ),
        )
        for i, subset in enumerate(subsets)
    ]


def record_round(history, conditions, *, seed, global_round, num_clients,
                 exits, comp, distances, cpu_hz, noises, channels, size_bits,
                 plan, participants, evaluation_exits, accuracies,
                 uplink_mses, method):
    choices, bandwidth, updated, eta = plan
    t = global_round
    positions = {client: index for index, client in enumerate(participants)}
    history.append({
        "seed": seed,
        "round": t,
        "accuracy": float(np.mean(accuracies)),
        "updated": bool(updated),
        "skipped": not bool(updated),
        "exit_index_sum": int(np.sum(choices)) if updated else np.nan,
        "noise_w": float(noises[t - 1]),
        "uplink_mse": float(np.mean(list(uplink_mses.values())))
        if uplink_mses else np.nan,
    })

    for i in range(num_clients):
        position = positions.get(i)
        candidate = (
            int(choices[position])
            if position is not None and choices is not None
            else None
        )

        b = (
            float(bandwidth[position])
            if position is not None and bandwidth is not None
            else np.nan
        )

        # tc: 選択Exitでの計算時間
        tc = (
            float(comp[i, candidate - 1])
            if candidate is not None
            else np.nan
        )

        # tt: 通信時間 [s]
        if method == "E":
            tt = 0.0 if position is not None else np.nan

        else:
            tt = (
                size_bits / (b * eta[position])
                if (
                    position is not None
                    and np.isfinite(b)
                    and b > 0
                    and eta[position] > 0
                )
                else np.nan
            )

        conditions.append({
            "seed": seed,
            "round": t,
            "client": i,
            "selected": bool(updated) and position is not None,
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
            "uplink_mse": uplink_mses.get(i, np.nan)
        })


def save_round_metrics(output, history, conditions):
    """中断時にも、それまでのラウンド結果を残す。"""
    pd.DataFrame(history).to_csv(output / "round_metrics.csv", index=False)
    pd.DataFrame(conditions).to_csv(output / "client_conditions.csv", index=False)


def save_checkpoint(output, config, args, seed, model_name, exits,
                    subsets, test_subsets, client_states,
                    evaluation_exits, last_accuracies):
    rounds = config["FL"]["global_rounds"]
    num_clients = config["FL"]["num_clients"]
    batch_size = config["training"]["batch_size"]

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
        "seed": seed,
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
    }, output / f"seed_{seed}_round_{rounds:03d}.pt")

