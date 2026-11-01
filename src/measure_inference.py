import json
import platform
import statistics
import time
import argparse
from pathlib import Path

import torch
from .models import MODEL_SPECS


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_SPECS),
        required=True,
    )
    args = parser.parse_args()

    model_name = args.model
    spec = MODEL_SPECS[model_name]
    
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.manual_seed(0)

    model = spec["factory"](num_classes=10).cpu().eval()

    # CIFAR10と同じ形状：1枚、3チャンネル、32×32
    x = torch.randn(1, 3, 32, 32)

    reference_cpu_speed = 2e9  # 換算用の仮定値
    warmup = 30
    repeats = 200

    results = {}

    with torch.inference_mode():
        for exit_id in model.exit_ids:
            # 初回実行などの影響を除くための準備
            for _ in range(warmup):
                model.forward_exit(x, exit_id)

            times = []

            for _ in range(repeats):
                start = time.perf_counter_ns()

                model.forward_exit(x, exit_id)

                elapsed_s = (
                    time.perf_counter_ns() - start
                ) / 1e9
                times.append(elapsed_s)

            median_s = statistics.median(times)

            results[str(exit_id)] = {
                "inference_time_s": median_s,
                "equivalent_cycles": (
                    median_s * reference_cpu_speed
                ),
            }

            print(
                f"Exit{exit_id}: "
                f"推論={median_s * 1000:.3f} ms, "
                f"等価cycles="
                f"{median_s * reference_cpu_speed:.6e}"
            )

    output = {
        "metadata": {
            "model_name": model_name,
            "device": "cpu",
            "threads": 1,
            "input_shape": list(x.shape),
            "warmup": warmup,
            "repeats": repeats,
            "statistic": "median",
            "reference_cpu_speed": reference_cpu_speed,
            "torch_version": str(torch.__version__),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "exits": results,
    }

    path = Path(__file__).with_name(spec["profile_file"])
    path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )

    print(f"測定結果を保存しました: {path}")


if __name__ == "__main__":
    main()