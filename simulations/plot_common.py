import json
import math
from pathlib import Path


METHODS = {
    "A": ("A: Fixed20, equal bandwidth", "#000000", ":"),
    "B": ("B: Fixed20, constrained", "#009E73", "-."),
    "C": ("C: Fixed10", "#D55E00", "--"),
    "D": ("D: FedMEM", "#0072B2", "-"),
    "E": ("E: Error Free", "#CC79A7", (0, (5, 2))),
    "F": ("F: Fixed16", "#E69F00", (0, (3, 1, 1, 1))),
}


def experiment_directories(root, alpha=None):
    root = Path(root)
    alphas = (alpha,) if alpha else ("0.5", "1e10")

    if any(
        (root / split).is_dir()
        for split in ("train_ratio", "random")
    ):
        roots = [
            root / "train_ratio",
            root / "random",
        ]
    else:
        roots = [root]

    for split_root in roots:
        for value in alphas:
            directory = split_root / f"alpha_{value}"

            if not directory.is_dir():
                raise FileNotFoundError(
                    f"結果フォルダがありません: {directory}"
                )

            yield value, directory


def make_title(directory, alpha):
    gammas = {}

    for method in ("B", "C", "D", "F"):
        path = directory / method / "metadata.json"

        with path.open(encoding="utf-8") as file:
            gamma = float(json.load(file)["gamma"])

        if not math.isfinite(gamma) or gamma <= 0:
            raise ValueError(f"gammaが不正です: {path}")

        gammas[method] = gamma

    if len(set(gammas.values())) != 1:
        raise ValueError(
            f"手法間でgammaが異なります: {gammas}"
        )

    alpha_tex = "0.5" if alpha == "0.5" else r"10^{10}"
    split = directory.parent.name

    prefix = (
        f"{split}\n"
        if split in ("train_ratio", "random")
        else ""
    )

    return prefix + (
        rf"$\alpha={alpha_tex},\ "
        rf"\gamma={gammas['D']:g}\,\mathrm{{s}}$"
    )


def cohort_size(directory, method):
    path = directory / method / "metadata.json"

    with path.open(encoding="utf-8") as file:
        metadata = json.load(file)

    count = int(metadata["config"]["FL"]["cohort"])

    if count < 1:
        raise ValueError(f"参加端末数が不正です: {path}")

    return count