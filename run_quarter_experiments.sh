#!/usr/bin/env bash
set -euo pipefail

# このスクリプトが置かれているFedMEMへ移動
cd "$(dirname "${BASH_SOURCE[0]}")"

read -rp "gamma [秒]: " GAMMA

SEED=0
RUN_ROOT="results/all_patterns/$(date +%Y%m%d_%H%M%S_%N)"

mkdir -p "$RUN_ROOT/logs"

TOTAL=24
COUNT=0

for test_split in train_ratio random; do
    # 分割方式ごとに出力先を分ける
    OUTPUT_ROOT="$RUN_ROOT/$test_split"

    for alpha in 0.5 1e10; do
        for method in A B C D E F; do
            COUNT=$((COUNT + 1))

            LOG_FILE="$RUN_ROOT/logs/${test_split}_alpha_${alpha}_${method}_seed_${SEED}.log"

            echo
            echo "[$COUNT/$TOTAL] split=$test_split alpha=$alpha method=$method seed=$SEED"

            python3 -u -m src.FL \
                --alpha "$alpha" \
                --method "$method" \
                --gamma "$GAMMA" \
                --seed "$SEED" \
                --test-split "$test_split" \
                --output-root "$OUTPUT_ROOT" \
                2>&1 | tee "$LOG_FILE"
        done
    done
done

echo
echo "全${TOTAL}パターンが完了しました。"
echo "結果保存先: $RUN_ROOT"