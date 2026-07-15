"""Aggregate per-run metrics into per-technique tables.

Reads the per-run training metrics (``triage_results.csv``) and the selection
diagnostics (``selections_summary.csv``), then writes one row per
technique x fraction with mean and standard deviation for every numeric
metric: quality (precision, recall, F1, mAP), time (train, validation,
inference, CPU) and resource use (RAM, GPU memory/utilization), plus the
selection-stage costs. The output CSV feeds the README result tables.
"""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path

import pandas as pd

from dataset_config import spec


# Per-class AP columns (``ap50_<classe>``) are detected dynamically, so the
# same summary works for the 2-class BVTSLD and the 3-class TT100K.
TRAINING_METRICS = [
    "precision", "recall", "f1", "map50", "map75", "map50_95",
    "train_time_s", "val_time_s", "infer_ms_per_img", "cpu_time_s",
    "peak_rss_mb", "gpu_mem_avg_mb", "gpu_mem_peak_mb", "gpu_util_avg_pct",
]
SELECTION_METRICS = [
    "coverage_mean", "coverage_max", "stability_jaccard",
    "selection_seconds_total", "rss_peak_mb",
]


def summarize_training(runs: pd.DataFrame) -> pd.DataFrame:
    present = [
        name for name in runs.columns
        if name in TRAINING_METRICS or name.startswith("ap50_")
    ]
    numeric = runs[present].apply(pd.to_numeric, errors="coerce")
    numeric[["technique", "fraction"]] = runs[["technique", "fraction"]]
    grouped = numeric.groupby(["technique", "fraction"])
    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std().add_suffix("_std")
    counts = grouped.size().rename("runs")
    return pd.concat([counts, means, stds], axis=1)


def summarize_selection(selections: pd.DataFrame) -> pd.DataFrame:
    present = [name for name in SELECTION_METRICS if name in selections.columns]
    numeric = selections[present].apply(pd.to_numeric, errors="coerce")
    numeric[["technique", "fraction"]] = selections[["technique", "fraction"]]
    summary = numeric.groupby(["technique", "fraction"]).mean()
    return summary.add_prefix("selection_").rename(columns={
        "selection_selection_seconds_total": "selection_seconds_total",
        "selection_rss_peak_mb": "selection_rss_peak_mb",
    })


def main() -> None:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="bvtsld", choices=("bvtsld", "tt100k"))
    parser.add_argument("--training-csv", type=Path)
    parser.add_argument("--selections-csv", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output_dir = spec(args.dataset).output_dir
    args.training_csv = args.training_csv or output_dir / "triage_results.csv"
    args.selections_csv = args.selections_csv or output_dir / "selections_summary.csv"
    args.output = args.output or output_dir / "metrics_summary.csv"

    parts = []
    if args.training_csv.exists():
        parts.append(summarize_training(pd.read_csv(args.training_csv)))
    else:
        print(f"training CSV not found, skipping: {args.training_csv}")
    if args.selections_csv.exists():
        parts.append(summarize_selection(pd.read_csv(args.selections_csv)))
    else:
        print(f"selections CSV not found, skipping: {args.selections_csv}")
    if not parts:
        raise SystemExit("no input CSV found")

    summary = pd.concat(parts, axis=1).sort_index()
    summary.to_csv(args.output)
    with pd.option_context("display.width", 200, "display.max_columns", 8):
        print(summary.round(4).to_string())
    print(f"\nsaved to: {args.output}")


if __name__ == "__main__":
    main()
