# python -m simulations.plot_all --root "results/train_ratio_0.5_10|20"

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
            "--root", str(directory),
            "--alpha", alpha,
        ]

        plot_accuracy.main(options)
        plot_skip_rate.main(options)
        plot_exit_cdf.main(options)


if __name__ == "__main__":
    main()