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
    local_state, reference_state, parameter_names, channel_gain, power_w, noise_w, seed
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
    h = complex(channel_gain)   # チャネル
    P = float(power_w)          # 電力
    sigma2 = float(noise_w)     # ノイズ

    # 位相補償だけを行う係数。絶対値は1。
    p = np.exp(-1j * np.angle(h))

    choices=("A", "B", "C", "D", "E", "F")

    rng = np.random.default_rng(seed)

    sqrt_power = math.sqrt(P)
    noise_std = math.sqrt(sigma2 / 2.0)

    received_state = {}
    squared_error = 0.0
    parameter_count = 0

    for name, local_tensor in local_state.items():
        # 端末が保持するモデルは変更しない。
        received_state[name] = local_tensor.detach().clone()

        # BN統計量・整数バッファなどは通信誤差の対象外。
        if (name not in parameter_names
            or not local_tensor.is_floating_point()):
            continue

        reference = (reference_state[name].detach().cpu().to(torch.float64))

        local = (local_tensor.detach().cpu().to(torch.float64))

        # 学習前後の共有パラメータの差分。
        delta = (local - reference).numpy().reshape(-1)

        if not np.isfinite(delta).all():
            raise ValueError(f"{name}の更新量にNaNまたはInfがあります")

        length = delta.size
        if length == 0:
            continue

        # 隣接する実数2個を複素シンボル1個へ変換。
        padded_delta = delta
        if length % 2:
            padded_delta = np.pad(delta, (0, 1))

        symbols = (padded_delta[0::2]+ 1j * padded_delta[1::2])

        # 端末・テンソルごとの平均を引く。
        symbol_mean = np.mean(symbols)
        centered_symbols = symbols - symbol_mean

        # 平均複素シンボル電力を1にする。
        scale = float(np.sqrt(np.mean(np.abs(centered_symbols) ** 2)))

        # 定数信号は平均値だけで復元できる。
        # 平均値は正確に通知されると仮定し、信号送信を省略。
        # received_stateには既にlocal_tensorのコピーが入っている。
        if scale == 0.0:
            parameter_count += length
            continue

        transmitted = centered_symbols / scale

        # xに送信電力を含める。
        x = sqrt_power * transmitted

        # 複素AWGN：E[|noise|^2] = sigma2
        noise = noise_std * (
            rng.standard_normal(transmitted.shape)
            + 1j * rng.standard_normal(transmitted.shape)
        )

        # pは位相補償のみ。|p| = 1。
        received_signal = p * h * x + noise

        # 位相補償後の実効振幅。
        a = sqrt_power * abs(h)

        # 正規化信号に対するMSE最小化回復係数。
        m = a / (a * a + sigma2)

        # 信号回復後、正規化を戻す。
        recovered_symbols = (m * received_signal) * scale + symbol_mean

        recovered_delta = np.empty(padded_delta.size,dtype=np.float64,)
        recovered_delta[0::2] = recovered_symbols.real
        recovered_delta[1::2] = recovered_symbols.imag
        recovered_delta = recovered_delta[:length]

        if not np.isfinite(recovered_delta).all():
            raise FloatingPointError(f"{name}の受信更新量にNaNまたはInfがあります")

        restored_delta = torch.from_numpy(recovered_delta.copy()).reshape(local_tensor.shape)

        # 受信モデル = 学習開始時の共有モデル + 受信更新量
        restored_model = (reference + restored_delta).to(
            device=local_tensor.device,dtype=local_tensor.dtype
            )

        if not torch.isfinite(restored_model).all():
            raise FloatingPointError(f"{name}の受信モデルにNaNまたはInfがあります")

        received_state[name] = restored_model

        error = (restored_model.detach().cpu().to(torch.float64) - local)
        squared_error += error.square().sum().item()
        parameter_count += length

    mse = (squared_error / parameter_count if parameter_count > 0 else 0.0)

    return received_state, mse