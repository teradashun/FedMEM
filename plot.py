
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ===== 設定 =====
csv_files = [
    "csv_data/fedavg_1e10.csv",
    "csv_data/fedmem_1e10.csv",
]

labels = ["fedavg", "error_free"]
output_path = "accuracy_comparison.png"

# ===== グラフ作成 =====
plt.figure(figsize=(8, 5))

for file, label in zip(csv_files, labels):
    df = pd.read_csv(file)

    # ラウンド順に並べ替え
    df = df.sort_values("round")

    plt.plot(
        df["round"],
        df["accuracy"],
        marker="o",
        markersize=3,
        linewidth=1.8,
        label=label,
    )

plt.xlabel("Communication Round")
plt.ylabel("Accuracy (%)")
plt.title("α=0.5")

plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.ylim(0, 100)
plt.tight_layout()

# 保存
plt.savefig(output_path, dpi=300, bbox_inches="tight")

print(f"Saved: {Path(output_path).resolve()}")
