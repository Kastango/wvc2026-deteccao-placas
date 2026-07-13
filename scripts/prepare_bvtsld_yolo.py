"""Materialize the frozen BVTSLD split in Ultralytics YOLO format."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "outputs" / "bvtsld"
DEFAULT_OUTPUT = SOURCE / "yolo_bvtsld"
SPLIT_DIRS = {"pool": "train", "validation": "val", "test": "test"}
CLASS_NAMES = ["regulatory", "warning", "information"]
JPEG_QUALITY = 92


def label_text(boxes: list[list[float]]) -> str:
    lines = [f"{int(cls)} {x:.6f} {y:.6f} {w:.6f} {h:.6f}" for cls, x, y, w, h in boxes]
    return "\n".join(lines)


def expected_counts(split: dict[str, list[str]]) -> dict[str, int]:
    return {directory: len(split[name]) for name, directory in SPLIT_DIRS.items()}


def is_complete(output: Path, split: dict[str, list[str]]) -> bool:
    return output.joinpath("data.yaml").exists() and all(
        len(list((output / "images" / directory).glob("*.jpg"))) == count
        and len(list((output / "labels" / directory).glob("*.txt"))) == count
        for directory, count in expected_counts(split).items()
    )


def materialize(output: Path, force: bool = False) -> None:
    records = {item["id"]: item for item in json.loads((SOURCE / "records.json").read_text())}
    split = json.loads((SOURCE / "split.json").read_text())

    if is_complete(output, split) and not force:
        print(f"already complete: {output}")
        return
    if output.exists():
        if not force:
            raise RuntimeError(f"incomplete output exists; rerun with --force: {output}")
        shutil.rmtree(output)

    for split_name, directory in SPLIT_DIRS.items():
        image_dir = output / "images" / directory
        label_dir = output / "labels" / directory
        image_dir.mkdir(parents=True, exist_ok=True)
        label_dir.mkdir(parents=True, exist_ok=True)
        for image_id in split[split_name]:
            record = records[image_id]
            source_image = ROOT / record["image"]
            if not source_image.exists():
                raise FileNotFoundError(source_image)
            # The frozen local YOLO artifact was encoded as RGB JPEG quality 92.
            with Image.open(source_image) as image:
                image.convert("RGB").save(image_dir / f"{image_id}.jpg", quality=JPEG_QUALITY)
            (label_dir / f"{image_id}.txt").write_text(label_text(record["boxes"]))

    names = "".join(f"  {index}: {name}\n" for index, name in enumerate(CLASS_NAMES))
    (output / "data.yaml").write_text(
        f"path: {output.resolve()}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n{names}"
    )
    print(f"materialized: {output}")
    print(f"counts: {expected_counts(split)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    materialize(args.output, args.force)


if __name__ == "__main__":
    main()
