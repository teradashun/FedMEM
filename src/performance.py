import json
import math
from functools import lru_cache
from pathlib import Path

from .models import MODEL_SPECS


@lru_cache(maxsize=None)
def load_reference_times(model_name):
    spec = MODEL_SPECS[model_name]
    path = Path(__file__).with_name(spec["profile_file"])

    with path.open(encoding="utf-8") as file:
        profile = json.load(file)

    if profile["metadata"].get("model_name") != model_name:
        raise ValueError(f"測定モデルが一致しません: {path}")

    times = {
        int(exit_id): float(values["inference_time_s"])
        for exit_id, values in profile["exits"].items()
    }

    if set(times) != set(spec["exit_ids"]):
        raise ValueError(f"測定Exitが一致しません: {path}")

    if any(
        not math.isfinite(t) or t <= 0
        for t in times.values()
    ):
        raise ValueError("推論時間は有限の正の値が必要です")

    return times


def compute_time(
    exit_id,
    performance_ratio,
    *,
    num_samples=1,
    model_name,
):
    times = load_reference_times(model_name)

    if exit_id not in times:
        raise ValueError(f"未対応のExit: {exit_id}")

    if not math.isfinite(performance_ratio) or performance_ratio <= 0:
        raise ValueError("性能倍率は有限の正の値が必要です")

    if not math.isfinite(num_samples) or num_samples < 0:
        raise ValueError("サンプル数は有限の非負値が必要です")

    return times[exit_id] * num_samples / performance_ratio