# python3 simulations/plot_all.py   --root results/all_patterns/20261105_052456_112753139

import argparse
from pathlib import Path

if __package__:
    from . import plot_accuracy, plot_exit_cdf, plot_skip_rate
    from .plot_common import experiment_directories
else:
    import plot_accuracy
    import plot_exit_cdf
    import plot_skip_rate
    from plot_common import experiment_directories


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()

    for alpha, directory in experiment_directories(args.root):
        print(
            f"\n描画中: {directory.parent.name}, alpha={alpha}",
            flush=True,
        )

        options = [
            "--root", str(directory.parent),
            "--alpha", alpha,
        ]

        plot_accuracy.main(options)
        plot_skip_rate.main(options)
        plot_exit_cdf.main(options)


if __name__ == "__main__":
    main()