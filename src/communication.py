import math
import numpy as np


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