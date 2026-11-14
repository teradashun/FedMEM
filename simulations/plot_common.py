import json
import math
from pathlib import Path


METHODS = {
    # "A": ("A: Fixed20, equal bandwidth", "#000000", ":"),
    "E": (f"A: error free", "#CC79A7", (0, (5, 2))),
    "D": (f"B: Joint optimization", "#0072B2", "-"),
    "B": (f"C: FixedExit20", "#009E73", "-."),
    "F": (f"D: FixedExit16", "#E69F00", (0, (3, 1, 1, 1))),
    "C": (f"E: FixedExit10", "#D55E00", "--"),
}


def experiment_directories(root, alpha=None):
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"結果フォルダがありません: {root}")

    split_roots = [
        root / split
        for split in ("train_ratio", "random")
        if (root / split).is_dir()
    ]
    roots = split_roots or [root]

    for directory in roots:
        recorded_alphas = {}
        for method in METHODS:
            path = directory / method / "metadata.json"
            if path.is_file():
                with path.open(encoding="utf-8") as file:
                    value = float(json.load(file)["alpha"])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"alphaが不正です: {path}")
                recorded_alphas[method] = value

        if recorded_alphas:
            if len(set(recorded_alphas.values())) != 1:
                raise ValueError(
                    f"手法間でalphaが異なります: {recorded_alphas}"
                )
            value = next(iter(recorded_alphas.values()))
            if alpha is not None and float(alpha) != value:
                raise ValueError(
                    f"指定alpha={alpha}と結果のalpha={value:g}が異なります: {directory}"
                )
            # 直下に手法別結果がある場合は、実際のalphaで一度だけ描画する。
            yield ("1e10" if value == 1e10 else f"{value:g}"), directory
            continue

        # 従来の train_ratio/alpha_0.5/B/... 形式にも対応する。
        alphas = (alpha,) if alpha is not None else ("0.5", "1e10")
        children = [directory / f"alpha_{value}" for value in alphas]
        existing = [child for child in children if child.is_dir()]
        if not existing:
            raise FileNotFoundError(
                f"結果のmetadata.jsonまたはalpha別フォルダがありません: {directory}"
            )
        for child in existing:
            yield from experiment_directories(child, child.name.removeprefix("alpha_"))


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

    alpha_value = float(alpha)
    alpha_tex = r"10^{10}" if alpha_value == 1e10 else f"{alpha_value:g}"
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