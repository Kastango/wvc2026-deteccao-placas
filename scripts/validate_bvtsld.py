"""Validate every artifact required by the BVTSLD YOLO training grid."""
from __future__ import annotations

import csv
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "bvtsld"
YOLO = OUTPUT / "yolo_bvtsld"
METHODS = {
    "random", "kmeans_dinov2", "kmeans_clip", "kmeans_shallow",
    "opf_dinov2", "typiclust_dinov2", "kcenter_dinov2",
    "probcover_dinov2", "facility_dinov2", "freesel_dino",
}
FRACTIONS = {0.05, 0.10}
REPEATS = set(range(1, 9))
SPLIT_DIRS = {"pool": "train", "validation": "val", "test": "test"}


def load_json(path: Path):
    return json.loads(path.read_text())


def check_yolo_labels(split: dict[str, list[str]]) -> tuple[dict, list[str]]:
    report, errors = {}, []
    for split_name, yolo_name in SPLIT_DIRS.items():
        image_dir = YOLO / "images" / yolo_name
        label_dir = YOLO / "labels" / yolo_name
        images = sorted(image_dir.glob("*.jpg"))
        labels = sorted(label_dir.glob("*.txt"))
        expected = len(split[split_name])
        report[yolo_name] = {"images": len(images), "labels": len(labels), "expected": expected}
        if len(images) != expected or len(labels) != expected:
            errors.append(f"YOLO {yolo_name} count mismatch")
        if {p.stem for p in images} != {p.stem for p in labels}:
            errors.append(f"YOLO {yolo_name} image/label stems differ")
        for path in labels:
            for line_number, line in enumerate(path.read_text().splitlines(), 1):
                fields = line.split()
                if len(fields) != 5:
                    errors.append(f"Malformed label: {path}:{line_number}")
                    continue
                cls, *coords = fields
                if cls not in {"0", "1", "2"} or any(not 0 <= float(v) <= 1 for v in coords):
                    errors.append(f"Invalid label value: {path}:{line_number}")
    return report, errors


def check_selections(pool: set[str]) -> tuple[dict, list[str]]:
    errors, counts = [], Counter()
    paths = sorted((OUTPUT / "selections").glob("*.json"))
    for path in paths:
        item = load_json(path)
        key = (item.get("technique"), float(item.get("fraction", -1)), int(item.get("repeat", -1)))
        counts[key] += 1
        images = item.get("images", [])
        budget = round(key[1] * len(pool))
        if len(images) != budget or len(set(images)) != budget:
            errors.append(f"Invalid selection budget or duplicates: {path.name}")
        if not set(images) <= pool:
            errors.append(f"Selection outside train pool: {path.name}")
    expected = {(m, f, r) for m in METHODS for f in FRACTIONS for r in REPEATS}
    present = set(counts)
    if present != expected or any(value != 1 for value in counts.values()):
        errors.append("Selection grid is not exactly 10 methods x 2 fractions x 8 repeats")
    return {"files": len(paths), "expected": len(expected), "complete": not errors}, errors


def main() -> None:
    errors: list[str] = []
    required = [
        OUTPUT / "records.json", OUTPUT / "split.json", OUTPUT / "selections_summary.csv",
        OUTPUT / "oracle_results.json", YOLO / "data.yaml",
        OUTPUT / "runs" / "oracle" / "weights" / "best.pt", ROOT / "yolov8n.pt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    errors.extend(f"Missing artifact: {path}" for path in missing)
    if missing:
        report = {"ready": False, "errors": errors}
    else:
        records = load_json(OUTPUT / "records.json")
        split = load_json(OUTPUT / "split.json")
        record_ids = {item["id"] for item in records}
        split_sets = {name: set(ids) for name, ids in split.items()}
        if set().union(*split_sets.values()) != record_ids:
            errors.append("Fixed split does not cover records exactly")
        for left, right in (("pool", "validation"), ("pool", "test"), ("validation", "test")):
            if split_sets[left] & split_sets[right]:
                errors.append(f"Split leakage: {left}/{right}")

        yolo_report, yolo_errors = check_yolo_labels(split)
        selection_report, selection_errors = check_selections(split_sets["pool"])
        errors.extend(yolo_errors + selection_errors)

        embeddings = {}
        for path in sorted(OUTPUT.glob("embeddings_bvtsld_*.npy")):
            shape = list(np.load(path, mmap_mode="r").shape)
            embeddings[path.name] = shape
            if shape[0] != len(split_sets["pool"]):
                errors.append(f"Embedding row mismatch: {path.name}")

        with (OUTPUT / "selections_summary.csv").open(newline="") as handle:
            summary_rows = sum(1 for _ in csv.DictReader(handle))
        if summary_rows != 160:
            errors.append(f"Expected 160 selection summary rows, found {summary_rows}")

        oracle = load_json(OUTPUT / "oracle_results.json")
        oracle_ok = all([
            oracle.get("dataset") == "bvtsld", oracle.get("epochs") == 40,
            oracle.get("train_seed") == 42,
            oracle.get("imgsz") == 640, oracle.get("patience") == 0,
            oracle.get("train_images") == len(split_sets["pool"]),
            oracle.get("validation", {}).get("map50_95") is not None,
        ])
        if not oracle_ok:
            errors.append("Oracle protocol or validation metrics are incomplete")

        triage_csv = OUTPUT / "triage_results.csv"
        triage_runs = 0
        if triage_csv.exists():
            with triage_csv.open(newline="") as handle:
                triage_runs = sum(1 for _ in csv.DictReader(handle))
        smoke_csv = OUTPUT / "triage_smoke.csv"
        smoke_checkpoint = OUTPUT / "runs" / "smoke" / "random_frac05_rep1_seed41" / "weights" / "best.pt"
        smoke_ok = smoke_csv.exists() and smoke_checkpoint.exists()
        if not smoke_ok:
            errors.append("Smoke training result or checkpoint is missing")
        training_lists = YOLO / "triage_lists"
        list_count = len(list(training_lists.glob("*.txt")))
        yaml_count = len(list(training_lists.glob("*.yaml")))
        training_configs_ready = list_count == 160 and yaml_count == 160
        if not training_configs_ready:
            errors.append("Expected 160 materialized YOLO training lists and configs")

        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "ready": not errors,
            "errors": errors,
            "dataset": {"records": len(records), "split": {k: len(v) for k, v in split.items()}},
            "yolo_dataset": yolo_report,
            "embeddings": embeddings,
            "selections": selection_report,
            "oracle": {
                "checkpoint": "outputs/bvtsld/runs/oracle/weights/best.pt",
                "validation": oracle["validation"],
                "protocol_valid": oracle_ok,
            },
            "triage": {"completed_runs": triage_runs, "expected_runs": 320},
            "training_configs": {
                "ready": training_configs_ready,
                "lists": list_count,
                "yamls": yaml_count,
            },
            "smoke_test": {
                "passed": smoke_ok,
                "results": "outputs/bvtsld/triage_smoke.csv",
                "checkpoint": "outputs/bvtsld/runs/smoke/random_frac05_rep1_seed41/weights/best.pt",
            },
        }

    output_path = OUTPUT / "project_status.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
