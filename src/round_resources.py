import copy
import math

import numpy as np
import torch
from torch import nn


def count_exit_flops(model):
    """畳み込み・全結合層のみ計数する。1 MAC＝2 FLOPs。"""
    measured = {}
    probe = copy.deepcopy(model).cpu().eval()
    handles = []
    counter = [0]

    def hook(module, inputs, output):
        if isinstance(module, nn.Conv2d):
            macs = (
                output.numel()
                * (module.in_channels // module.groups)
                * math.prod(module.kernel_size)
            )
        else:
            macs = output.numel() * module.in_features

        counter[0] += 2 * macs

    try:
        for module in probe.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                handles.append(
                    module.register_forward_hook(hook)
                )

        with torch.inference_mode():
            x = torch.zeros(1, 3, 32, 32)

            for exit_id in probe.exit_ids:
                counter[0] = 0
                probe.forward_exit(x, exit_id)
                measured[exit_id] = int(counter[0])

    finally:
        for handle in handles:
            handle.remove()

    if any(value <= 0 for value in measured.values()):
        raise ValueError("FLOPsは正の値である必要があります")

    return measured


def make_environment(num_clients, rounds, seed, trial):
    """距離・チャネル・雑音を学習用乱数と独立に生成する。"""
    distance_rng = np.random.default_rng(
        np.random.SeedSequence([seed, trial, 0])
    )
    channel_rng = np.random.default_rng(
        np.random.SeedSequence([seed, trial, 1])
    )
    noise_rng = np.random.default_rng(
        np.random.SeedSequence([seed, trial, 2])
    )

    # 距離は試行開始時に一度だけ生成する
    distances = []

    while len(distances) < num_clients:
        x, y = distance_rng.normal(0.0, 250.0, size=2)
        distance = math.hypot(x, y)

        if 35.0 <= distance <= 500.0:
            distances.append(distance)

    distances = np.asarray(distances)

    path_loss = (
        128.1 + 37.6 * np.log10(distances / 1000.0)
    )

    # ラウンド・端末ごとにRayleighフェージングを生成する
    h = (
        channel_rng.normal(size=(rounds, num_clients))
        + 1j * channel_rng.normal(size=(rounds, num_clients))
    ) / math.sqrt(2.0)

    gains = h * 10.0 ** (-path_loss[None, :] / 20.0)

    # 各ラウンドに1つの共通雑音電力を生成する
    noise_db = noise_rng.uniform(-3.0, 3.0, size=rounds)
    noise = 1e-10 * 10.0 ** (noise_db / 10.0)

    return distances, gains, noise


def plan_round(
    method,
    comp,
    gains,
    noise,
    power,
    size_bits,
    total_bandwidth,
    gamma,
):
    """
    戻り値：
      choices：Exit番号1・2・3。Dが実行不能ならNone。
      bandwidth：割当帯域幅。Dが実行不能ならNone。
      updated：このラウンドで更新を実行するか。
      eta：周波数利用効率 [bit/s/Hz]。
    """
    if method not in ("A", "B", "C", "D", "E"):
        raise ValueError("未対応の手法です")

    if method == "E":
        n = len(comp)
        return (
            np.full(n, 3, dtype=int),  # 最深Exit
            np.full(n, np.nan),       # 帯域割当なし
            True,                    # 毎ラウンド更新
            np.full(n, np.nan),       # 周波数利用効率は不使用
        )

    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("gammaは有限の正の値が必要です")

    if not math.isfinite(noise) or noise <= 0:
        raise ValueError("雑音電力は有限の正の値が必要です")

    comp = np.asarray(comp, dtype=float)
    n, exits = comp.shape

    if (
        exits != 3
        or not np.isfinite(comp).all()
        or (comp < 0).any()
    ):
        raise ValueError("計算遅延の値が不正です")

    if (np.diff(comp, axis=1) < 0).any():
        raise ValueError(
            "深いExitほど計算遅延が大きい必要があります"
        )

    eta = np.log1p(
        power * np.abs(gains) ** 2 / noise
    ) / math.log(2.0)

    remaining = gamma - comp
    required = np.full_like(comp, np.inf)

    valid = (remaining > 0) & (eta[:, None] > 0)

    np.divide(
        size_bits,
        remaining * eta[:, None],
        out=required,
        where=valid,
    )

    def within_budget(values):
        value = float(np.sum(values))

        return math.isfinite(value) and (
            value <= total_bandwidth
            or math.isclose(
                value,
                total_bandwidth,
                rel_tol=1e-12,
                abs_tol=1e-8,
            )
        )

    if method == "A":
        # Ideal：最深Exit、帯域幅等分、制約によらず更新
        choices = np.full(n, 3, dtype=int)
        bandwidth = np.full(n, total_bandwidth / n)
        updated = True

    else:
        if method == "B":
            choices = np.full(n, 3, dtype=int)

        elif method == "C":
            choices = np.ones(n, dtype=int)

        else:
            # D：従来のExitを浅くする規則を維持する
            choices = np.full(n, 3, dtype=int)

            # 最浅Exitでも実行不能ならスキップする
            if not within_budget(required[:, 0]):
                return None, None, False, eta

            for i in range(n):
                while not math.isfinite(
                    required[i, choices[i] - 1]
                ):
                    choices[i] -= 1

            while not within_budget(
                required[np.arange(n), choices - 1]
            ):
                candidates = np.flatnonzero(choices > 1)

                if len(candidates) == 0:
                    raise RuntimeError(
                        "資源割当処理の整合性が失われています"
                    )

                target = min(
                    candidates,
                    key=lambda i: (
                        comp[i, choices[i] - 1],
                        int(i),
                    ),
                )
                choices[target] -= 1

        bandwidth = required[
            np.arange(n), choices - 1
        ]
        updated = within_budget(bandwidth)

    return choices, bandwidth, updated, eta


def gamma_boundary(
    comp_one_exit,
    gains,
    noise,
    power,
    size_bits,
    total_bandwidth,
    tol=1e-6,
):
    """全端末が指定Exitを使う場合の最小実行可能gamma [秒]。"""
    comp_one_exit = np.asarray(comp_one_exit, dtype=float)
    gains = np.asarray(gains)

    if (
        comp_one_exit.ndim != 1
        or gains.shape != comp_one_exit.shape
        or comp_one_exit.size == 0
        or not np.isfinite(comp_one_exit).all()
        or (comp_one_exit < 0).any()
        or not np.isfinite(gains).all()
    ):
        raise ValueError("端末の計算遅延・チャネルが不正です")

    if any(
        not math.isfinite(value) or value <= 0
        for value in (
            noise,
            power,
            size_bits,
            total_bandwidth,
            tol,
        )
    ):
        raise ValueError(
            "通信条件と許容誤差は有限の正の値が必要です"
        )

    eta = np.log1p(
        power * np.abs(gains) ** 2 / noise
    ) / math.log(2.0)

    if (eta <= 0).any() or not np.isfinite(eta).all():
        return math.inf

    # 通信時間は正なので、最大計算遅延以下では実行不能
    low = float(comp_one_exit.max())

    # 帯域幅等分時の最大合計遅延を、実行可能な探索上限にする
    equal_total = comp_one_exit + size_bits / (
        (total_bandwidth / len(comp_one_exit)) * eta
    )
    high = float(equal_total.max())

    if not math.isfinite(high):
        return math.inf

    # 丸め誤差によって上限が実行不能になることを避ける
    high = high * (1.0 + 1e-10) + tol

    def feasible(gamma):
        remaining = gamma - comp_one_exit

        if (remaining <= 0).any():
            return False

        required = size_bits / (remaining * eta)
        return bool(np.sum(required) <= total_bandwidth)

    if not feasible(high):
        raise RuntimeError(
            "gamma探索の上限が実行可能になっていません"
        )

    for _ in range(200):
        if high - low <= tol:
            break

        mid = low + (high - low) / 2.0

        if mid == low or mid == high:
            break

        if feasible(mid):
            high = mid
        else:
            low = mid

    return high