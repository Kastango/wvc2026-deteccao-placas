"""Statistical analysis of the sample-selection training grid.

Required CSV columns:
technique,fraction,selection_repeat,selection_hash,train_seed,map50_95

Validates N methods x 2 fractions x 8 selections x 2 train seeds, then computes
paired differences against random sampling, hierarchical bootstrap confidence
intervals, exact sign randomization and Holm correction within each fraction.
"""

from argparse import ArgumentParser
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd


TECHNIQUES = {
    "random",
    "kmeans_dinov2",
    "kmeans_clip",
    "kmeans_shallow",
    "opf_dinov2",
    "typiclust_dinov2",
    "kcenter_dinov2",
    "probcover_dinov2",
    "facility_dinov2",
    "freesel_dino",
}
FRACTIONS = {0.05, 0.10}
REPEATS = set(range(1, 9))
TRAIN_SEEDS = {41, 42}
BOOTSTRAPS = 10_000


def validate(data: pd.DataFrame) -> None:
    required = {
        "technique", "fraction", "selection_repeat",
        "selection_hash", "train_seed", "map50_95",
    }
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    present = set(data["technique"])
    if "random" not in present:
        raise ValueError("The CSV must contain the 'random' control")
    unknown = present - TECHNIQUES
    if unknown:
        raise ValueError(f"Unknown methods: {sorted(unknown)}")
    if set(data["fraction"].round(2)) != FRACTIONS:
        raise ValueError("The CSV must contain only fractions 0.05 and 0.10")
    if set(data["selection_repeat"]) != REPEATS:
        raise ValueError("Each cell must contain repeats 1..8")
    if set(data["train_seed"]) != TRAIN_SEEDS:
        raise ValueError("Each selection must use train seeds 41 and 42")

    keys = ["technique", "fraction", "selection_repeat", "train_seed"]
    if data.duplicated(keys).any():
        raise ValueError("Duplicate training runs found for the same cell")
    counts = data.groupby(["technique", "fraction"]).size()
    if not counts.eq(16).all():
        raise ValueError("Each method x fraction must contain 8 selections x 2 train seeds")

    unique = data.groupby(["technique", "fraction"])["selection_hash"].nunique()
    deterministic = unique[~unique.eq(8)]
    if not deterministic.empty:
        print(
            "WARNING: duplicate deterministic selections; variability in these cells "
            f"comes only from train seeds: {deterministic.to_dict()}"
        )


def exact_randomization(differences: np.ndarray) -> float:
    """Exact two-sided sign randomization over the paired differences."""
    observed = abs(float(differences.mean()))
    signs = np.asarray(list(product((-1.0, 1.0), repeat=len(differences))))
    statistics = abs((signs * differences).mean(axis=1))
    return float(np.mean(statistics >= observed - 1e-12))


def hierarchical_ci(seed_differences: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """Resample selections and their paired train seeds hierarchically."""
    n_selections, n_seeds = seed_differences.shape
    estimates = np.empty(BOOTSTRAPS)
    for index in range(BOOTSTRAPS):
        chosen = rng.integers(0, n_selections, n_selections)
        seed_draws = rng.integers(0, n_seeds, (n_selections, n_seeds))
        sampled = seed_differences[chosen[:, None], seed_draws]
        estimates[index] = sampled.mean()
    return tuple(np.quantile(estimates, [0.025, 0.975]))


def holm(p_values: pd.Series) -> pd.Series:
    order = p_values.sort_values().index
    adjusted = pd.Series(index=p_values.index, dtype=float)
    running = 0.0
    total = len(order)
    for rank, name in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p_values[name]))
        running = max(running, candidate)
        adjusted[name] = running
    return adjusted


def analyze(data: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(2026)
    rows = []
    index = ["fraction", "selection_repeat", "train_seed"]
    scores = data.set_index(index + ["technique"])["map50_95"].unstack()

    for fraction in sorted(FRACTIONS):
        block = scores.loc[fraction]
        for technique in sorted(set(block.columns) - {"random"}):
            paired = (block[technique] - block["random"]).unstack("train_seed")
            paired = paired.loc[sorted(REPEATS), sorted(TRAIN_SEEDS)]
            by_selection = paired.mean(axis=1).to_numpy()
            low, high = hierarchical_ci(paired.to_numpy(), rng)
            rows.append({
                "fraction": fraction,
                "technique": technique,
                "mean_gain": float(paired.to_numpy().mean()),
                "median_gain_by_selection": float(np.median(by_selection)),
                "ci95_low": low,
                "ci95_high": high,
                "p_exact": exact_randomization(by_selection),
            })

    result = pd.DataFrame(rows)
    result["p_holm"] = np.nan
    for fraction, positions in result.groupby("fraction").groups.items():
        values = result.loc[positions].set_index("technique")["p_exact"]
        adjusted = holm(values)
        result.loc[positions, "p_holm"] = result.loc[positions, "technique"].map(adjusted)
    result["practically_relevant"] = result["mean_gain"] >= 0.02
    result["supported"] = (
        (result["ci95_low"] > 0)
        & (result["p_holm"] < 0.05)
        & result["practically_relevant"]
    )
    return result.sort_values(["fraction", "mean_gain"], ascending=[True, False])


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("triage_analysis.csv"))
    args = parser.parse_args()

    data = pd.read_csv(args.input_csv)
    validate(data)
    result = analyze(data)
    result.to_csv(args.output, index=False)
    print(result.to_string(index=False))
    print(f"\nsaved to: {args.output}")


if __name__ == "__main__":
    main()
