import math

from .utils import (
    load_config,
    generate_client_distances,
    generate_channel_gain,
)
from .p2 import required_bandwidth
from .performance import compute_time
from .dataset import get_datasets, split_dataset
from .utils import set_seed
from .models import MODEL_SPECS


def find_gamma_boundary(
    clients,
    exit_choice,
    model_name,
    *,
    size_bits,
    total_bandwidth,
    noise_w,
    gamma_low=0.001,
    gamma_high=1000.0,
    tol=1e-5,
):
    """
    全クライアントが指定Exitを使用した場合に、
    総帯域幅制約を満たす最小gammaを二分探索で求める。
    """
    upper_required = sum(
        required_bandwidth(
            client,
            exit_choice,
            model_name=model_name,
            size_bits=size_bits,
            gamma=gamma_high,
            noise_w=noise_w,
        )
        for client in clients.values()
    )

    if upper_required > total_bandwidth:
        raise ValueError(
            f"探索上限gamma={gamma_high}秒でも制約を満たせません"
        )
    
    while gamma_high - gamma_low > tol:
        gamma = (gamma_low + gamma_high) / 2

        required_sum = sum(
            required_bandwidth(
                client,
                exit_choice,
                model_name=model_name,
                size_bits=size_bits,
                gamma=gamma,
                noise_w=noise_w,
            )
            for client in clients.values()
        )

        if required_sum <= total_bandwidth:
            gamma_high = gamma
        else:
            gamma_low = gamma

    return gamma_high


def main():
    import os
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_SPECS),
        required=True,
    )
    parser.add_argument(
        "--alpha",
        type=float,
        choices=(0.5, 1e10),
        required=True,
    )
    args = parser.parse_args()

    model_name = args.model
    spec = MODEL_SPECS[model_name]
    dirichlet_alpha = args.alpha

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)

    config = load_config(parent_dir)

    num_clients = config["FL"]["num_clients"]
    batch_size = config["training"]["batch_size"]
    dataset_name = config["training"]["dataset_name"]

    total_bandwidth = config['communication']['total_bandwidth']
    power_dbm       = config['communication']['power_dbm']
    noise_w         = config['communication']['noise_w']
    channel_seed    = config['communication']['channel_seed']

    power_w = 10.0 ** ((power_dbm - 30.0) / 10.0)
    print(f"送信電力: {power_dbm:.1f} dBm = {power_w:.6f} W")

    train_loader, _, _ = get_datasets(
        batch_size,
        dataset_name,
        val_ratio=0.0,
    )

    set_seed(0)
    client_datasets = split_dataset(
        train_loader.dataset,
        num_clients=num_clients,
        dirichlet_alpha=dirichlet_alpha,
    )

    client_sample_counts = [
        len(client_dataset)
        for client_dataset in client_datasets
    ]

    model = spec["factory"](num_classes=10)

    S = sum(
        tensor.numel() * tensor.element_size() * 8
        for name, tensor in model.state_dict().items()
        if name.startswith(spec["shared_prefixes"])
    )

    print(f"共有部分の送信サイズ: {S:,} bit")

    performance_ratios = [
        float(x)
        for x in config["communication"]["performance_ratios"]
    ]

    if not performance_ratios or any(
        not math.isfinite(r) or r <= 0
        for r in performance_ratios
    ):
        raise ValueError("performance_ratiosには正の倍率を指定してください")

    client_performance_ratios = [
        performance_ratios[i % len(performance_ratios)]
        for i in range(num_clients)
    ]

    # -------------------------
    # 距離・チャネル
    # -------------------------
    set_seed(channel_seed)
    client_distances = generate_client_distances(num_clients)

    client_channel_gains = [
        generate_channel_gain(distance)
        for distance in client_distances
    ]

    clients = {
        i: {
            "performance_ratio": client_performance_ratios[i],
            "distance_m": client_distances[i],
            "channel_gain": client_channel_gains[i],
            "power_w": power_w,
            "num_inference_samples": client_sample_counts[i],
        }
        for i in range(num_clients)
    }

    print("\n===== Computation time =====")
    for i, client in clients.items():
        values = [
            (
                f"Exit{exit_id}="
                f"{compute_time(
                    exit_id,
                    client['performance_ratio'],
                    num_samples=client['num_inference_samples'],
                    model_name=model_name,
                ):.6f}s"
            )
            for exit_id in spec["exit_ids"]
        ]
        print(f"Client {i}: " + ", ".join(values))

    # -------------------------
    # gamma最小値
    # 全員Exit20
    # -------------------------
    gamma_min = find_gamma_boundary(
        clients,
        exit_choice=1,
        model_name=model_name,
        size_bits=S,
        total_bandwidth=total_bandwidth,
        noise_w=noise_w,
    )

    # -------------------------
    # gamma最大側の境界
    # 全員Exit34
    # -------------------------
    gamma_max = find_gamma_boundary(
        clients,
        exit_choice=3,
        model_name=model_name,
        size_bits=S,
        total_bandwidth=total_bandwidth,
        noise_w=noise_w,
    )

    print()
    print("===== Gamma range =====")
    print(
        f"全員Exit{spec['exit_ids'][0]}の最小gamma: "
        f"{gamma_min:.6f} s"
    )
    print(
        f"全員Exit{spec['exit_ids'][-1]}の最小gamma: "
        f"{gamma_max:.6f} s"
    )

    print()
    print(
        f"FedMEM実験候補範囲: "
        f"{gamma_min:.3f} ～ {gamma_max:.3f} s"
    )


if __name__ == "__main__":
    main()