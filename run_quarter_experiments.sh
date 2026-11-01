(
    set -euo pipefail

    cd ~/BranchyNet

    read -rp "gamma [秒]: " GAMMA
    RUN_ROOT="results/resnet20_half_flops/$(date +%Y%m%d_%H%M%S)"

    mkdir -p "$RUN_ROOT/logs"

    for alpha in 0.5 1e10; do
        for method in A B C D; do
            python -u -m src.FL \
                --alpha "$alpha" \
                --method "$method" \
                --gamma "$GAMMA" \
                --output-root "$RUN_ROOT" \
                2>&1 | tee \
                "$RUN_ROOT/logs/alpha_${alpha}_${method}.log"
        done
    done

    python simulations/plot_accuracy.py --root "$RUN_ROOT"
    python simulations/plot_skip_rate.py --root "$RUN_ROOT"
    python simulations/plot_exit_cdf.py --root "$RUN_ROOT"

    echo "結果保存先: $RUN_ROOT"
)