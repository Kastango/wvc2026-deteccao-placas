"""Train YOLOv8n on the saved BVTSLD sample selections.

The full grid contains 160 selections and two train seeds. Completed cells in
``triage_results.csv`` are skipped, so the command is safe to resume.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "bvtsld"
YOLO_DIR = OUTPUT / "yolo_bvtsld"
RESULTS_CSV = OUTPUT / "triage_results.csv"

ULTRALYTICS_VERSION = "8.3.0"
MODEL_WEIGHTS = ROOT / "yolov8n.pt"
EPOCHS, IMG_SIZE, PATIENCE, BATCH_SIZE = 40, 640, 0, 16
TRAIN_SEEDS = (41, 42)
TARGET_CLASSES = ["regulatory", "warning", "information"]
AUGMENTATION = dict(
    hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, degrees=0.0, translate=0.1,
    scale=0.5, shear=0.0, perspective=0.0, flipud=0.0, fliplr=0.5,
    mosaic=1.0, close_mosaic=10, mixup=0.0, copy_paste=0.0, erasing=0.4,
)
FIELDS = [
    "technique", "fraction", "selection_repeat", "selection_hash",
    "train_seed", "map50_95", "map50", "train_time_s", "device", "run_name",
]


def selection_hash(image_ids: list[str]) -> str:
    return hashlib.sha1(",".join(sorted(image_ids)).encode()).hexdigest()[:12]


def load_done(results_csv: Path) -> set[tuple]:
    if not results_csv.exists():
        return set()
    with results_csv.open() as handle:
        return {
            (r["technique"], float(r["fraction"]), int(r["selection_repeat"]), int(r["train_seed"]))
            for r in csv.DictReader(handle)
        }


def append_row(results_csv: Path, row: dict) -> None:
    new_file = not results_csv.exists()
    with results_csv.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--technique", help="Run only this selection method")
    parser.add_argument("--fraction", type=float, choices=(0.05, 0.10))
    parser.add_argument("--repeat", type=int, choices=range(1, 9))
    parser.add_argument("--train-seed", type=int, choices=TRAIN_SEEDS)
    parser.add_argument("--device", help="Ultralytics device override (for example mps, cuda, cpu)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and list the selected grid")
    parser.add_argument("--smoke", action="store_true", help="Run one cell for 2 epochs in a separate CSV")
    args = parser.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"protocolo requer ultralytics=={ULTRALYTICS_VERSION}; encontrado {ultralytics.__version__}"
        )
    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

    required = [
        OUTPUT / "records.json",
        YOLO_DIR / "data.yaml",
        YOLO_DIR / "images" / "train",
        YOLO_DIR / "labels" / "train",
        YOLO_DIR / "images" / "val",
        YOLO_DIR / "labels" / "val",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise RuntimeError(f"Missing required artifacts: {missing}")
    records = {r["id"]: r for r in json.loads((OUTPUT / "records.json").read_text())}

    results_csv = OUTPUT / "triage_smoke.csv" if args.smoke else RESULTS_CSV
    runs_dir = OUTPUT / "runs" / ("smoke" if args.smoke else "triage")
    epochs = 2 if args.smoke else EPOCHS
    train_seeds = (args.train_seed,) if args.train_seed else TRAIN_SEEDS

    selections = sorted((OUTPUT / "selections").glob("*.json"))
    artifacts = [(path, json.loads(path.read_text())) for path in selections]
    if args.technique:
        artifacts = [item for item in artifacts if item[1]["technique"] == args.technique]
    if args.fraction is not None:
        artifacts = [item for item in artifacts if float(item[1]["fraction"]) == args.fraction]
    if args.repeat is not None:
        artifacts = [item for item in artifacts if int(item[1]["repeat"]) == args.repeat]
    if args.smoke:
        artifacts = artifacts[:1]
        train_seeds = train_seeds[:1]
    if not artifacts:
        raise RuntimeError("No saved selections match the requested filters")

    known_techniques = {item[1]["technique"] for item in artifacts}
    if args.technique and args.technique not in known_techniques:
        raise RuntimeError(f"Unknown technique: {args.technique}")

    done = load_done(results_csv)
    lists_dir = YOLO_DIR / "triage_lists"
    lists_dir.mkdir(exist_ok=True)
    runs_dir.mkdir(parents=True, exist_ok=True)

    total = len(artifacts) * len(train_seeds)
    pending = sum(
        (a["technique"], float(a["fraction"]), int(a["repeat"]), seed) not in done
        for _, a in artifacts for seed in train_seeds
    )
    print(
        f"selections={len(artifacts)} train_seeds={list(train_seeds)} "
        f"cells={total} pending={pending} device={device} epochs={epochs}"
    )

    for path, artifact in artifacts:
        technique, fraction, repeat = artifact["technique"], artifact["fraction"], artifact["repeat"]
        unknown_ids = sorted(set(artifact["images"]) - set(records))
        if unknown_ids:
            raise RuntimeError(f"{path.name} contains unknown image IDs: {unknown_ids[:5]}")
        image_paths = [
            str(YOLO_DIR / "images" / "train" / f"{Path(records[i]['image']).stem}.jpg")
            for i in artifact["images"]
        ]
        missing_images = [item for item in image_paths if not Path(item).exists()]
        if missing_images:
            raise RuntimeError(f"{path.name} references missing train images: {missing_images[:5]}")
        cell_hash = selection_hash(artifact["images"])
        list_file = lists_dir / f"{path.stem}.txt"
        list_file.write_text("\n".join(image_paths) + "\n")
        data_yaml = lists_dir / f"{path.stem}.yaml"
        data_yaml.write_text(
            f"path: {YOLO_DIR.resolve()}\n"
            f"train: {list_file.resolve()}\n"
            "val: images/val\n"
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(TARGET_CLASSES))
        )

        for seed in train_seeds:
            key = (technique, float(fraction), int(repeat), seed)
            if key in done:
                continue
            if args.dry_run:
                print(f"PENDING {path.stem} train_seed={seed}")
                continue
            run_name = f"{path.stem}_seed{seed}"
            started = time.time()
            model = YOLO(str(MODEL_WEIGHTS) if MODEL_WEIGHTS.exists() else MODEL_WEIGHTS.name)
            model.train(
                data=str(data_yaml), epochs=epochs, imgsz=IMG_SIZE, patience=PATIENCE,
                batch=BATCH_SIZE, seed=seed, deterministic=True, pretrained=True,
                optimizer="SGD", device=device, project=str(runs_dir.resolve()),
                name=run_name, exist_ok=True, verbose=False, plots=False, **AUGMENTATION,
            )
            metrics = YOLO(runs_dir / run_name / "weights" / "best.pt").val(
                data=str(data_yaml), split="val", device=device, verbose=False,
                project=str(runs_dir.resolve()), name=f"{run_name}_eval", exist_ok=True,
            )
            append_row(results_csv, {
                "technique": technique, "fraction": fraction, "selection_repeat": repeat,
                "selection_hash": cell_hash, "train_seed": seed,
                "map50_95": round(float(metrics.box.map), 4),
                "map50": round(float(metrics.box.map50), 4),
                "train_time_s": round(time.time() - started, 1),
                "device": device, "run_name": run_name,
            })
            done.add(key)
            print(f"{run_name} mAP50-95={float(metrics.box.map):.4f}")

    if args.dry_run:
        print(f"dry-run complete: {pending} pending cells")
    else:
        print(f"training complete: results saved to {results_csv}")


if __name__ == "__main__":
    main()
