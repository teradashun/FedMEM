import json
import math
from pathlib import Path


profile_path = Path(__file__).with_name(
    "inference_profile.json"
)

with profile_path.open(encoding="utf-8") as file:
    profile = json.load(file)

REFERENCE_CPU_SPEED = profile["metadata"][
    "reference_cpu_speed"
]

F_EQ_REF = {
    int(exit_id): values["equivalent_cycles"]
    for exit_id, values in profile["exits"].items()
}


def compute_time(
    exit_id,
    cpu_speed,
    cycles_by_exit,
    num_samples=1,
):
    """推論時間の推定値 [秒]。"""
    if exit_id not in cycles_by_exit:
        raise ValueError(f"未定義のExit: {exit_id}")
    if not math.isfinite(cpu_speed) or cpu_speed <= 0:
        raise ValueError("cpu_speedは有限の正の値が必要です")
    if num_samples <= 0:
        raise ValueError("num_samplesは正の値が必要です")

    total_cycles = cycles_by_exit[exit_id] * num_samples

    return total_cycles / cpu_speed