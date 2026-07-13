"""Train and validate the frozen full-pool BVTSLD YOLOv8n oracle."""
from __future__ import annotations

import argparse
import json
import time

from run_local_triage import (
    AUGMENTATION,
    BATCH_SIZE,
    EPOCHS,
    IMG_SIZE,
    MODEL_WEIGHTS,
    OUTPUT,
    PATIENCE,
    TARGET_CLASSES,
    ULTRALYTICS_VERSION,
    YOLO_DIR,
)


TRAIN_SEED = 42


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", help="Ultralytics device: cuda, mps or cpu")
    args = parser.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO

    if ultralytics.__version__ != ULTRALYTICS_VERSION:
        raise RuntimeError(
            f"protocolo requer ultralytics=={ULTRALYTICS_VERSION}; "
            f"encontrado {ultralytics.__version__}"
        )
    device = args.device or (
        "cuda" if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
    data_yaml = YOLO_DIR / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"materialize o dataset YOLO primeiro: {data_yaml}")

    runs = OUTPUT / "runs"
    model = YOLO(str(MODEL_WEIGHTS) if MODEL_WEIGHTS.exists() else MODEL_WEIGHTS.name)
    started = time.time()
    model.train(
        data=str(data_yaml), epochs=EPOCHS, imgsz=IMG_SIZE, patience=PATIENCE,
        batch=BATCH_SIZE, seed=TRAIN_SEED, deterministic=True, pretrained=True,
        optimizer="SGD", device=device, project=str(runs.resolve()), name="oracle",
        exist_ok=True, verbose=False, plots=True, **AUGMENTATION,
    )
    train_time = time.time() - started
    metrics = YOLO(runs / "oracle" / "weights" / "best.pt").val(
        data=str(data_yaml), split="val", device=device, verbose=False,
        project=str(runs.resolve()), name="oracle_eval", exist_ok=True,
    )
    ap50 = {name: 0.0 for name in TARGET_CLASSES}
    for class_index, value in zip(metrics.box.ap_class_index, metrics.box.ap50):
        ap50[TARGET_CLASSES[int(class_index)]] = round(float(value), 4)

    split = json.loads((OUTPUT / "split.json").read_text())
    result = {
        "dataset": "bvtsld",
        "device": device,
        "ultralytics_version": ultralytics.__version__,
        "weights": MODEL_WEIGHTS.name,
        "epochs": EPOCHS,
        "imgsz": IMG_SIZE,
        "patience": PATIENCE,
        "unknown_object_policy": "quarantine",
        "auxiliary_boxes": 0,
        "batch": BATCH_SIZE,
        "train_seed": TRAIN_SEED,
        "augmentation": AUGMENTATION,
        "train_images": len(split["pool"]),
        "train_time_s": round(train_time, 1),
        "validation": {
            "map50": round(float(metrics.box.map50), 4),
            "map50_95": round(float(metrics.box.map), 4),
            "ap50_per_class": ap50,
        },
    }
    (OUTPUT / "oracle_results.json").write_text(
        json.dumps(result, indent=1, ensure_ascii=False) + "\n"
    )
    print(json.dumps(result["validation"], indent=2))


if __name__ == "__main__":
    main()
