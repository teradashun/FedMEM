# train_ratio、alpha=0.5
# python -m src.FL --test-split train_ratio --alpha 0.5 --method D --gamma 30

# train_ratio、alpha=1e10
# python -m src.FL --test-split train_ratio --alpha 1e10 --method D --gamma 30

# random、alpha=0.5
# python -m src.FL --test-split random --alpha 0.5 --method D --gamma 30

# random、alpha=1e10
# python -m src.FL --test-split random --alpha 1e10 --method D --gamma 30

import copy
import os
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .client import train, fine_tune_personal
from .communication import transmit_update
from .dataset import get_datasets, split_dataset, split_client_tests
from .models import MODEL_SPECS
from .round_resources import count_exit_flops, make_environment, plan_round
from .experiment import (
    evaluate_clients, full_state, parse_args, output_directory, 
    print_resource_summary, save_checkpoint, save_round_metrics,
    save_split, shared_state, record_round
    )
from .server import federated_learning
from .utils import choose_clients, load_config, select_optimizer, set_seed


def main(argv=None):
    args = parse_args(argv)
    root = Path(__file__).resolve().parent.parent
    os.chdir(root)
    config = load_config(str(root))

    training = config["training"]
    fl              = config["FL"]

    ite_num         = fl["ite_num"]
    global_rounds   = fl["global_rounds"]
    num_clients     = fl["num_clients"]
    cohort          = fl["cohort"]

    batch_size      = training["batch_size"]
    epochs          = training["epochs"]
    lr              = training["lr"]
    optimizer_name  = training["optim_name"]

    model_name = "eenet20_half"
    spec = MODEL_SPECS[model_name]
    exits = spec["exit_ids"]
    prefixes = spec["shared_prefixes"]

    communication = config["communication"]
    bandwidth_total = communication["total_bandwidth"]
    power = 10.0 ** ((communication["power_dbm"] - 30.0) / 10.0)
    cpu_hz = np.asarray([
        (1e9, 2e9, 5e9)[i % 3] for i in range(num_clients)
    ])
    flops_per_cycle = 5.0

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, _, test_loader = get_datasets(batch_size, "CIFAR10", val_ratio=0.0)

    for ite in range(ite_num):
        seed = ite + 100
        print(f"iteration {ite + 1}/{ite_num} (seed={seed})")

        # モデルを初期化する
        set_seed(seed)
        global_model = spec["factory"](num_classes=10).to(device)
        parameter_names = {name for name, _ in global_model.named_parameters()}
        flops = count_exit_flops(global_model)
        size_bits = sum(
            tensor.numel() * tensor.element_size() * 8
            for tensor in shared_state(global_model, prefixes).values()
        )

        # 各端末の学習データとテストデータを準備する
        subsets = split_dataset(train_loader.dataset, num_clients, args.alpha)
        test_subsets, train_counts, test_counts = split_client_tests(
            subsets, test_loader.dataset, seed=seed,
            num_classes=10, mode=args.test_split,
        )
        if any(len(subset) == 0 for subset in subsets):
            raise ValueError("学習データが空の端末があります")

        sample_counts = np.asarray([len(subset) for subset in subsets])

        client_loaders = [
                    DataLoader(
                        subset, batch_size=batch_size, shuffle=True, drop_last=True,
                        generator=torch.Generator().manual_seed(seed + i),
                    )
                    for i, subset in enumerate(subsets)
                ]
        test_loaders = [
            DataLoader(
                subset, batch_size=batch_size, shuffle=False, drop_last=False,
                generator=torch.Generator().manual_seed(seed + i),
            )
            for i, subset in enumerate(test_subsets)
        ]

        # 端末ごとの計算時間と、各ラウンドの通信条件を準備する
        comp = (
            sample_counts[:, None]
            * np.asarray([flops[e] for e in exits])[None, :]
            / (cpu_hz[:, None] * flops_per_cycle)
        )
        distances, channels, noises = make_environment(
            num_clients, global_rounds, bandwidth_total, seed,
        )

        # 参加端末・Exit・帯域幅を各ラウンドについて決める
        set_seed(seed)
        selected_clients = [
            choose_clients(num_clients, cohort) for _ in range(global_rounds)
        ]
        plans = [
            plan_round(
                args.method, comp[participants], channels[t, participants],
                float(noises[t]), power, size_bits, bandwidth_total, args.gamma,
            )
            for t, participants in enumerate(selected_clients)
        ]
        print_resource_summary(
            exits, comp, channels, noises, power, size_bits, bandwidth_total, plans,
            selected_clients,
        )

        if args.dry_run:
            continue

        output = output_directory(args, seed, ite_num, cohort, num_clients)
        output.mkdir(parents=True, exist_ok=False)

        with (output / "metadata.json").open("w", encoding="utf-8") as file:
            json.dump(
                {
                    "gamma": args.gamma,
                    "config": config,
                    "method": args.method,
                    "alpha": args.alpha,
                    "seed": seed,
                },
                file,
                indent=2,
            )

        save_split(output, subsets, test_subsets, train_counts, test_counts, seed, args.test_split)

        initial_state = full_state(global_model)
        client_states = [copy.deepcopy(initial_state) for _ in range(num_clients)]
        initial_exit = {
            "C": exits[0], "F": exits[1],
        }.get(args.method, exits[-1])
        evaluation_exits = [initial_exit] * num_clients

        last_accuracies = evaluate_clients(
            global_model,
            client_states,
            test_loaders,
            device,
            evaluation_exits,
            prefixes,
        )

        history = [{
            "seed": seed,
            "round": 0,
            "accuracy": float(np.mean(last_accuracies)),
            "updated": False,
            "skipped": False,
            "exit_index_sum": np.nan,
            "noise_w": np.nan,
            "uplink_mse": np.nan,
        }]
        conditions = []

        for global_round in range(global_rounds):
            plan = plans[global_round]
            choices, _, updated, _ = plan
            participants = selected_clients[global_round]
            uplink_mses = {}

            if updated:
                set_seed(seed + global_round)
                current_shared = shared_state(global_model, prefixes)
                local_updates = []

                # 参加端末でローカル学習し、共有層を送信する
                for client_idx in participants:
                    client_loaders[client_idx].generator.manual_seed(seed + global_round + client_idx)

                    local_model = copy.deepcopy(global_model)
                    local_model.load_state_dict(client_states[client_idx])
                    local_model.load_state_dict(current_shared, strict=False)
                    optimizer = select_optimizer(optimizer_name, local_model, lr)

                    for _ in range(epochs):
                        train(optimizer, local_model, client_loaders[client_idx], device, model_name)

                    client_states[client_idx] = full_state(local_model)
                    local_shared = shared_state(local_model, prefixes)

                    if args.method == "E":
                        received_shared, client_mse = local_shared, 0.0
                    else:
                        received_shared, client_mse = transmit_update(
                            local_state=local_shared,
                            reference_state=current_shared,
                            parameter_names=parameter_names,
                            channel_gain=channels[global_round, client_idx],
                            power_w=power,
                            noise_w=float(noises[global_round]),
                            seed=[seed, global_round, client_idx],
                        )

                    local_updates.append(received_shared)
                    uplink_mses[client_idx] = client_mse

                # データ数で重み付けして共有層を集約する
                federated_learning(local_updates, global_model, sample_counts[participants].tolist())

                # 更新された共有層を固定し、参加端末の個人層を調整する
                new_shared = shared_state(global_model, prefixes)
                for position, client_idx in enumerate(participants):
                    personal_model = copy.deepcopy(global_model)
                    personal_model.load_state_dict(client_states[client_idx])
                    personal_model.load_state_dict(new_shared, strict=False)
                    optimizer = select_optimizer(optimizer_name, personal_model, lr)

                    fine_tune_personal(
                        optimizer, personal_model, client_loaders[client_idx],
                        device, model_name, prefixes, epochs=1,
                    )

                    client_states[client_idx] = full_state(personal_model)
                    evaluation_exits[client_idx] = exits[int(choices[position]) - 1]

                # 全端末のテストデータで評価する
                last_accuracies = evaluate_clients(
                    global_model, client_states, test_loaders, device, evaluation_exits, prefixes,
                )

                record_round(
                history,
                conditions,
                seed=seed,
                global_round=global_round + 1,
                num_clients=num_clients,
                exits=exits,
                comp=comp,
                distances=distances,
                cpu_hz=cpu_hz,
                noises=noises,
                channels=channels,
                size_bits=size_bits,
                plan=plan,
                participants=participants,
                evaluation_exits=evaluation_exits,
                accuracies=last_accuracies,
                uplink_mses=uplink_mses,
                method=args.method,
            )

            save_round_metrics(output, history, conditions)
            status = "更新" if updated else "スキップ"
            print(
                f"Round {global_round + 1}/{global_rounds}: {status}, "
                f"Accuracy={np.mean(last_accuracies):.3f}%"
            )

        # この試行の最終モデルを保存する
        save_checkpoint(
            output, config, args, seed, model_name, exits, subsets, test_subsets,
            client_states, evaluation_exits, last_accuracies,
        )
        skipped_rounds = sum(row["skipped"] for row in history[1:])
        print(f"更新スキップ率: {100.0 * skipped_rounds / global_rounds:.2f}%")
        print(f"結果保存先: {output}")


if __name__ == "__main__":
    main()
