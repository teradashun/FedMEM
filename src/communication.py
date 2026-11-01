import math
import numpy as np
import torch


def communication_model(
    size_bits,
    bandwidth_hz,
    power_w,
    channel_gain,
    noise_w,
):
    """通信速度[bit/s]とモデル送信時間[秒]を返す。"""

    if bandwidth_hz <= 0 or power_w <= 0 or noise_w <= 0:
        raise ValueError("通信条件は正の値にしてください")

    # SNR = P * |g|^2 / N
    snr = power_w * (np.abs(channel_gain) ** 2) / noise_w

    # Shannon capacity
    rate_bps = bandwidth_hz * math.log2(1.0 + snr)

    # モデル送信時間
    time_s = size_bits / rate_bps

    return rate_bps, time_s


def transmit_update(
    local_state,
    reference_state,
    parameter_names,
    channel_gain,
    power_w,
    noise_w,
    seed,
):
    """
    1端末の共有パラメータ更新量を、個別の通信路で送受信する。

    前提:
      ・チャネル係数はサーバーで正確に既知
      ・上りのみ通信誤差を考慮
      ・正規化係数と形状は正確に共有される
      ・学習パラメータ以外のバッファは正確に届く

    戻り値:
      received_state: サーバーが受信・復元した共有モデル
      mse: 送信前後の学習パラメータの平均二乗誤差
    """
    h = complex(channel_gain)

    if not np.isfinite(h) or abs(h) == 0:
        raise ValueError("channel_gainは有限かつ非ゼロにしてください")

    if not math.isfinite(power_w) or power_w <= 0:
        raise ValueError("power_wは有限の正の値にしてください")

    if not math.isfinite(noise_w) or noise_w < 0:
        raise ValueError("noise_wは有限の非負値にしてください")

    rng = np.random.default_rng(seed)

    sqrt_power = math.sqrt(power_w)
    noise_std = math.sqrt(noise_w / 2.0)

    received_state = {}
    squared_error = 0.0
    parameter_count = 0

    for name, local_tensor in local_state.items():
        # 端末が保持するモデルは変更しない。
        received_state[name] = local_tensor.detach().clone()

        # BN統計量・整数バッファなどは通信誤差の対象外。
        if (
            name not in parameter_names
            or not local_tensor.is_floating_point()
        ):
            continue

        reference = (
            reference_state[name]
            .detach()
            .cpu()
            .to(torch.float64)
        )

        local = (
            local_tensor.detach()
            .cpu()
            .to(torch.float64)
        )

        # 学習前後の共有パラメータの差分。
        delta = (local - reference).numpy().reshape(-1)

        if not np.isfinite(delta).all():
            raise ValueError(f"{name}の更新量にNaNまたはInfがあります")

        length = delta.size
        if length == 0:
            continue

        # 隣接する実数2個を複素シンボル1個へ割り当てる。
        padded_delta = delta
        if length % 2:
            padded_delta = np.pad(delta, (0, 1))

        symbols = (
            padded_delta[0::2]
            + 1j * padded_delta[1::2]
        )

        # このテンソルの平均複素シンボル電力を1へ正規化。
        scale = float(np.sqrt(
            np.mean(np.abs(symbols) ** 2)
        ))

        # 全要素ゼロの更新は、ゼロであることを正確に通知し、
        # このテンソルの信号送信を省略する仮定。
        if scale == 0.0:
            parameter_count += length
            continue

        transmitted = symbols / scale

        # 各端末の受信信号に独立な複素ガウス雑音を追加。
        # E[|noise|^2] = noise_w
        noise = noise_std * (
            rng.standard_normal(symbols.shape)
            + 1j * rng.standard_normal(symbols.shape)
        )

        # sqrt(P)は振幅調整。hには減衰と位相が含まれる。
        received_signal = (
            sqrt_power * h * transmitted + noise
        )

        # チャネル既知：受信側で減衰・位相を補償する。
        recovered_symbols = (
            received_signal / (sqrt_power * h)
        ) * scale

        recovered_delta = np.empty(
            padded_delta.size,
            dtype=np.float64,
        )
        recovered_delta[0::2] = recovered_symbols.real
        recovered_delta[1::2] = recovered_symbols.imag
        recovered_delta = recovered_delta[:length]

        if not np.isfinite(recovered_delta).all():
            raise FloatingPointError(
                f"{name}の受信更新量にNaNまたはInfがあります"
            )

        restored_delta = torch.from_numpy(
            recovered_delta.copy()
        ).reshape(local_tensor.shape)

        # 受信モデル = 学習開始時の共有モデル + 受信更新量
        restored_model = (
            reference + restored_delta
        ).to(
            device=local_tensor.device,
            dtype=local_tensor.dtype,
        )

        if not torch.isfinite(restored_model).all():
            raise FloatingPointError(
                f"{name}の受信モデルにNaNまたはInfがあります"
            )

        received_state[name] = restored_model

        error = (
            restored_model.detach().cpu().to(torch.float64)
            - local
        )
        squared_error += error.square().sum().item()
        parameter_count += length

    mse = (
        squared_error / parameter_count
        if parameter_count > 0
        else 0.0
    )

    return received_state, mse