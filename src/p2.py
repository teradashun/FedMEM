import math

from .communication import communication_model
from .performance import compute_time

from .models import MODEL_SPECS


def evaluate_p2(
    clients,
    exit_choices,
    bandwidths,
    *,
    model_name,
    size_bits,
    total_bandwidth,
    gamma,
    noise_w=1e-12,
):
    EXIT_LAYERS = dict(
            enumerate(MODEL_SPECS[model_name]["exit_ids"], start=1)
        )
    
    """Exit・帯域幅の候補について、目的関数と制約を評価する。"""
    if not clients:
        raise ValueError("参加端末が0です")

    if (
        set(clients) != set(exit_choices)
        or set(clients) != set(bandwidths)
    ):
        raise ValueError("各辞書の端末IDを一致させてください")

    if not math.isfinite(total_bandwidth) or total_bandwidth <= 0:
        raise ValueError("総帯域幅は有限の正の値が必要です")

    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gammaは有限の正の値が必要です")

    details = {}

    for client_id, settings in clients.items():
        m = exit_choices[client_id]

        if type(m) is not int or m not in EXIT_LAYERS:
            raise ValueError("Exit番号は整数1・2・3で指定してください")

        # 帯域幅の正値チェックは通信関数内で行う
        rate, t_tran = communication_model(
            size_bits=size_bits,
            bandwidth_hz=bandwidths[client_id],
            power_w=settings["power_w"],
            channel_gain=settings["channel_gain"],
            noise_w=noise_w,
        )

        t_comp = compute_time(
            exit_id=EXIT_LAYERS[m],
            performance_ratio=settings["performance_ratio"],
            num_samples=settings["num_inference_samples"],
            model_name=model_name,
        )

        t_total = t_tran + t_comp

        details[client_id] = {
            "exit_layer": EXIT_LAYERS[m],
            "rate_bps": rate,
            "computation_s": t_comp,
            "transmission_s": t_tran,
            "total_s": t_total,
            "latency_ok": (
                t_total <= gamma
                or math.isclose(
                    t_total, gamma,
                    rel_tol=1e-12, abs_tol=1e-12
                )
            ),
        }

    bandwidth_sum = sum(bandwidths.values())

    bandwidth_ok = (
        bandwidth_sum <= total_bandwidth
        or math.isclose(
            bandwidth_sum, total_bandwidth, rel_tol=1e-12
        )
    )

    return {
        "objective": sum(exit_choices.values()),
        "bandwidth_sum_hz": bandwidth_sum,
        "bandwidth_ok": bandwidth_ok,
        "feasible": bandwidth_ok and all(
            row["latency_ok"] for row in details.values()
        ),
        "clients": details,
    }


def required_bandwidth(
    client,
    exit_choice,
    *,
    model_name,
    size_bits,
    gamma,
    noise_w=1e-12,
):
    EXIT_LAYERS = dict(
        enumerate(MODEL_SPECS[model_name]["exit_ids"], start=1)
    )

    """時間制限を満たす最低帯域幅 [Hz]。不可能ならinf。"""
    if type(exit_choice) is not int or exit_choice not in EXIT_LAYERS:
        raise ValueError("Exit番号は整数1・2・3で指定してください")

    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gammaは有限の正の値が必要です")

    if not math.isfinite(size_bits) or size_bits <= 0:
        raise ValueError("送信サイズは有限の正の値が必要です")

    computation_time = compute_time(
        exit_id=EXIT_LAYERS[exit_choice],
        performance_ratio=client["performance_ratio"],
        num_samples=client["num_inference_samples"],
        model_name=model_name,
    )

    remaining_time = gamma - computation_time

    # 推論だけで時間制限に達する場合、送信時間を確保できない
    if remaining_time <= 0:
        return math.inf

    # 帯域幅1 Hzでの速度 = log2(1 + SNR)
    spectral_efficiency, _ = communication_model(
        size_bits=size_bits,
        bandwidth_hz=1.0,
        power_w=client["power_w"],
        channel_gain=client["channel_gain"],
        noise_w=noise_w,
    )

    return size_bits / (
        remaining_time * spectral_efficiency
    )


def allocate_fedmem(
    clients,
    *,
    model_name,
    size_bits,
    total_bandwidth,
    gamma,
    noise_w=1e-12,
):
    EXIT_LAYERS = dict(
            enumerate(MODEL_SPECS[model_name]["exit_ids"], start=1)
        )
    if not clients:
        raise ValueError("参加端末が0です")

    if not math.isfinite(total_bandwidth) or total_bandwidth <= 0:
        raise ValueError("総帯域幅は有限の正の値が必要です")

    common = {
        "model_name": model_name,
        "size_bits": size_bits,
        "gamma": gamma,
        "noise_w": noise_w,
    }

    # 各端末・各Exitの最低帯域幅
    requirements = {
        i: {
            m: required_bandwidth(settings, m, **common)
            for m in (1, 2, 3)
        }
        for i, settings in clients.items()
    }

    # 最浅Exitでも不可能なら、解なし
    if sum(requirements[i][1] for i in clients) > total_bandwidth:
        raise ValueError("全員が最浅Exitでも制約を満たせません")

    choices = {i: 3 for i in clients}
    steps = []

    # 推論だけで時間制限に達するExitは、先に下げる
    for i in clients:
        while not math.isfinite(requirements[i][choices[i]]):
            old = choices[i]
            choices[i] -= 1
            steps.append((i, old, choices[i]))

    # 総帯域幅を満たすまでExitを下げる
    while (
        sum(requirements[i][choices[i]] for i in clients)
        > total_bandwidth
    ):
        candidates = [
            i for i in clients if choices[i] > 1
        ]

        target = min(
            candidates,
            key=lambda i: (
                compute_time(
                    EXIT_LAYERS[choices[i]],
                    clients[i]["performance_ratio"],
                    num_samples=clients[i]["num_inference_samples"],
                    model_name=model_name,
                ),
                i,
            ),
        )

        old = choices[target]
        choices[target] -= 1
        steps.append((target, old, choices[target]))

    # 選択したExitの最低帯域幅を割り当てる
    bandwidths = {
        i: requirements[i][choices[i]]
        for i in clients
    }

    evaluation = evaluate_p2(
        clients,
        choices,
        bandwidths,
        total_bandwidth=total_bandwidth,
        **common,
    )

    if not evaluation["feasible"]:
        raise RuntimeError("配分結果がP2の制約を満たしていません")

    return choices, bandwidths, steps, evaluation


def find_min_gamma_for_fixed_exit(
    clients,
    exit_choice,
    *,
    size_bits,
    total_bandwidth,
    noise_w,
    gamma_min=0.01,
    gamma_max=1000.0,
    tol=1e-4,
):
    """
    全クライアントが同じExitを使う場合に、
    総帯域幅制約を満たす最小gammaを二分探索で求める。
    """

    low = gamma_min
    high = gamma_max

    while high - low > tol:
        gamma = (low + high) / 2

        required_sum = 0.0

        for client in clients.values():
            bw = required_bandwidth(
                client,
                exit_choice,
                size_bits=size_bits,
                gamma=gamma,
                noise_w=noise_w,
            )

            required_sum += bw

        if required_sum <= total_bandwidth:
            high = gamma
        else:
            low = gamma

    return high