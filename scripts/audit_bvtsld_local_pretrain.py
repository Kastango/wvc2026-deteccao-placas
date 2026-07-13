from __future__ import annotations

import csv
import importlib.metadata
import json
import math
import statistics
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

try:
    import numpy as np
except ImportError:  # pragma: no cover - dependency check is reported below
    np = None


ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = ROOT / "datasets" / "bvtsld" / "Brazilian Vertical Traffic Signs and Lights Dataset"
OUTPUT = ROOT / "outputs" / "bvtsld"

TARGET_CLASSES = ["regulatory", "warning", "information"]
UNKNOWN_OBJECT_POLICY = "quarantine"
BVTSLD_CODE_MAP = {
    "000000": 0,
    "000001": 0,
    "000003": 0,
    "000004": 0,
    "000007": 0,
    "000008": 0,
    "000009": 0,
    "000023": 0,
    "000028": 0,
    "000042": 0,
    "000025": 1,
    "000035": 2,
    "000040": 2,
}
UNMAPPED_TRAFFIC_LIGHT_CODES = {"000051", "000052", "000053"}
EXPECTED_TECHNIQUES = {
    "random",
    "kmeans_dinov2",
    "opf_dinov2",
    "typiclust_dinov2",
    "probcover_dinov2",
    "freesel_dino",
}
EXPECTED_FRACTIONS = {0.05, 0.10, 0.20, 0.50}


def expected_repeats() -> dict[str, int]:
    repeats = {technique: 8 for technique in EXPECTED_TECHNIQUES}
    repeats["opf_dinov2"] = 1
    return repeats


EXPECTED_ULTRALYTICS = "8.3.0"


def read_json(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def base_id(path: Path) -> str:
    return path.stem.split("@", 1)[0]


def parse_objects(xml_file: Path) -> list[dict]:
    tree = ET.parse(xml_file)
    width = float(tree.findtext(".//size/width") or 800)
    height = float(tree.findtext(".//size/height") or 450)
    out = []
    for obj in tree.findall(".//object"):
        code = (obj.findtext("name") or "").strip()
        box = obj.find("bndbox")
        if box is None:
            continue
        xmin = float(box.findtext("xmin"))
        ymin = float(box.findtext("ymin"))
        xmax = float(box.findtext("xmax"))
        ymax = float(box.findtext("ymax"))
        out.append(
            {
                "code": code,
                "bbox_xyxy": [xmin, ymin, xmax, ymax],
                "bbox_yolo": [
                    (xmin + xmax) / 2 / width,
                    (ymin + ymax) / 2 / height,
                    (xmax - xmin) / width,
                    (ymax - ymin) / height,
                ],
            }
        )
    return out


def audit_raw_dataset() -> tuple[dict, list[dict]]:
    image_files = sorted((DATASET_ROOT / "images").glob("*"))
    xml_files = sorted((DATASET_ROOT / "annotations").glob("*.xml"))
    original_xmls = [p for p in xml_files if "@" not in p.stem]

    code_boxes_all = Counter()
    code_images_all = Counter()
    code_boxes_original = Counter()
    code_images_original = Counter()
    records_from_xml = []
    quarantine = []

    for xml_file in xml_files:
        objects = parse_objects(xml_file)
        codes = [obj["code"] for obj in objects]
        code_boxes_all.update(codes)
        code_images_all.update(set(codes))

    for xml_file in original_xmls:
        objects = parse_objects(xml_file)
        codes = [obj["code"] for obj in objects]
        code_boxes_original.update(codes)
        code_images_original.update(set(codes))
        unknown = sorted({code for code in codes if code not in BVTSLD_CODE_MAP})
        if unknown:
            quarantine.append(
                {
                    "id": xml_file.stem,
                    "reason": "unmapped_or_unverified_code",
                    "source_categories": unknown,
                    "image": str(
                        (DATASET_ROOT / "images" / f"{xml_file.stem}.jpg").relative_to(ROOT)
                    ),
                }
            )
            continue
        boxes = [
            [BVTSLD_CODE_MAP[obj["code"]], *obj["bbox_yolo"]]
            for obj in objects
        ]
        records_from_xml.append(
            {
                "id": xml_file.stem,
                "image": str(DATASET_ROOT / "images" / f"{xml_file.stem}.jpg"),
                "boxes": boxes,
                "source_categories": sorted(set(codes)),
            }
        )

    image_bases = {base_id(p) for p in image_files}
    xml_bases = {base_id(p) for p in xml_files}
    raw = {
        "image_files_total": len(image_files),
        "image_base_ids": len(image_bases),
        "original_image_files": sum(1 for p in image_files if "@" not in p.stem),
        "augmented_image_files": sum(1 for p in image_files if "@" in p.stem),
        "xml_files_total": len(xml_files),
        "xml_base_ids": len(xml_bases),
        "original_xml_files": len(original_xmls),
        "augmented_xml_files": sum(1 for p in xml_files if "@" in p.stem),
        "base_images_without_xml": sorted(image_bases - xml_bases),
        "base_xml_without_image": sorted(xml_bases - image_bases),
        "observed_codes_all_boxes": dict(sorted(code_boxes_all.items())),
        "observed_codes_all_images": dict(sorted(code_images_all.items())),
        "observed_codes_original_boxes": dict(sorted(code_boxes_original.items())),
        "observed_codes_original_images": dict(sorted(code_images_original.items())),
        "mapped_codes": dict(sorted(BVTSLD_CODE_MAP.items())),
        "unmapped_codes_expected": sorted(UNMAPPED_TRAFFIC_LIGHT_CODES),
    }
    return raw, records_from_xml, quarantine


def audit_records(records: list[dict], records_from_xml: list[dict]) -> dict:
    box_counts = Counter()
    image_counts = Counter()
    for record in records:
        seen = set()
        for box in record["boxes"]:
            cls = int(box[0])
            box_counts[TARGET_CLASSES[cls]] += 1
            seen.add(TARGET_CLASSES[cls])
        image_counts.update(seen)

    generated_ids = {r["id"] for r in records_from_xml}
    current_ids = {r["id"] for r in records}
    return {
        "records_json_count": len(records),
        "records_from_original_xml_count": len(records_from_xml),
        "records_match_xml_rebuild": generated_ids == current_ids,
        "missing_from_records_json": sorted(generated_ids - current_ids),
        "extra_in_records_json": sorted(current_ids - generated_ids),
        "class_box_counts": dict(box_counts),
        "class_image_counts": dict(image_counts),
        "total_boxes": sum(box_counts.values()),
    }


def audit_split(split: dict, records: list[dict]) -> dict:
    ids = {r["id"] for r in records}
    sets = {name: set(values) for name, values in split.items()}
    intersections = {
        f"{a}_{b}": len(sets[a] & sets[b])
        for i, a in enumerate(sets)
        for b in list(sets)[i + 1 :]
    }
    covered = set().union(*sets.values()) if sets else set()
    return {
        "counts": {name: len(values) for name, values in split.items()},
        "unique_counts": {name: len(set(values)) for name, values in split.items()},
        "intersections": intersections,
        "covers_all_records": covered == ids,
        "missing_record_ids": sorted(ids - covered),
        "unknown_split_ids": sorted(covered - ids),
        "test_closed_for_local_pretrain": True,
    }


def audit_embeddings(split: dict) -> dict:
    out = {}
    pool_count = len(split.get("pool", []))
    for path in sorted(OUTPUT.glob("embeddings_bvtsld_*.npy")):
        item = {"exists": True}
        if np is None:
            item["readable"] = False
            item["error"] = "numpy_missing"
        else:
            arr = np.load(path, mmap_mode="r")
            item.update(
                {
                    "shape": list(arr.shape),
                    "dtype": str(arr.dtype),
                    "matches_pool_count": int(arr.shape[0]) == pool_count,
                }
            )
            if arr.ndim == 2 and arr.shape[0] > 0:
                sample = np.asarray(arr[: min(len(arr), 2048)], dtype=np.float32)
                norms = np.linalg.norm(sample, axis=1)
                item["sample_l2_norm_min"] = round(float(norms.min()), 6)
                item["sample_l2_norm_max"] = round(float(norms.max()), 6)
                item["sample_l2_norm_mean"] = round(float(norms.mean()), 6)
        out[path.name] = item
    return out


def load_selection_summary() -> list[dict]:
    path = OUTPUT / "selections_summary.csv"
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def audit_selections(split: dict) -> dict:
    selection_dir = OUTPUT / "selections"
    files = sorted(selection_dir.glob("*.json"))
    pool_ids = set(split.get("pool", []))
    repeats_by_technique = expected_repeats()
    groups = defaultdict(list)
    subset_ok = True
    budget_ok = True
    selection_seed_ok = True
    malformed = []

    for path in files:
        try:
            item = read_json(path)
            technique = item["technique"]
            fraction = float(item["fraction"])
            repeat = int(item["repeat"])
            images = tuple(sorted(item["images"]))
        except Exception as exc:  # pragma: no cover - diagnostic path
            malformed.append({"file": str(path), "error": str(exc)})
            continue
        subset_ok = subset_ok and set(images) <= pool_ids
        budget = max(1, round(fraction * len(pool_ids)))
        budget_ok = budget_ok and len(images) == budget
        if "selection_seed" in item:
            selection_seed_ok = selection_seed_ok and item["selection_seed"] == 42 + repeat - 1
        else:
            selection_seed_ok = False
        groups[(technique, fraction)].append(images)

    group_status = []
    for technique, fraction in sorted(groups):
        selections = groups[(technique, fraction)]
        unique_count = len(set(selections))
        group_status.append(
            {
                "technique": technique,
                "fraction": fraction,
                "repeats": len(selections),
                "unique_selections": unique_count,
                "expected_current_protocol": technique in EXPECTED_TECHNIQUES
                and fraction in EXPECTED_FRACTIONS,
                "complete_for_current_protocol": technique in EXPECTED_TECHNIQUES
                and fraction in EXPECTED_FRACTIONS
                and len(selections) == repeats_by_technique.get(technique, 0),
            }
        )

    current_groups = [
        g
        for g in group_status
        if g["expected_current_protocol"]
    ]
    complete = (
        len(current_groups) == len(EXPECTED_TECHNIQUES) * len(EXPECTED_FRACTIONS)
        and all(g["complete_for_current_protocol"] for g in current_groups)
    )
    extra_groups = [
        g for g in group_status if not g["expected_current_protocol"]
    ]
    return {
        "selection_files": len(files),
        "summary_rows": len(load_selection_summary()),
        "subset_of_pool": subset_ok,
        "budget_ok": budget_ok,
        "selection_seed_ok": selection_seed_ok,
        "malformed_files": malformed,
        "groups": group_status,
        "current_protocol_complete": complete,
        "extra_historical_groups": extra_groups,
        "expected_current_protocol_files": sum(repeats_by_technique.values())
        * len(EXPECTED_FRACTIONS),
    }


def audit_oracle() -> dict:
    data = read_json(OUTPUT / "oracle_results.json")
    if not data:
        return {"exists": False}
    return {
        "exists": True,
        "dataset": data.get("dataset"),
        "device": data.get("device"),
        "epochs": data.get("epochs"),
        "train_images": data.get("train_images"),
        "train_time_s": data.get("train_time_s"),
        "validation": data.get("validation"),
        "test_single_final_eval_present": "test_SINGLE_FINAL_EVAL" in data,
        "note": "frozen BVTSLD oracle; validation only; test split remains closed",
    }


def audit_environment() -> dict:
    try:
        ultralytics_version = importlib.metadata.version("ultralytics")
    except importlib.metadata.PackageNotFoundError:
        ultralytics_version = None
    return {
        "ultralytics_version": ultralytics_version,
        "expected_ultralytics_version": EXPECTED_ULTRALYTICS,
        "ultralytics_matches_protocol": ultralytics_version == EXPECTED_ULTRALYTICS,
    }


def status_from_checks(audit: dict) -> dict:
    checks = {
        "taxonomy_report_materialized": (OUTPUT / "taxonomy_report.json").exists(),
        "quarantine_materialized": (OUTPUT / "quarantine.json").exists(),
        "records_match_xml_rebuild": audit["records"]["records_match_xml_rebuild"],
        "split_has_no_intersections": all(
            value == 0 for value in audit["split"]["intersections"].values()
        ),
        "split_covers_records": audit["split"]["covers_all_records"],
        "dinov2_embedding_matches_pool": audit["embeddings"]
        .get("embeddings_bvtsld_dinov2_full.npy", {})
        .get("matches_pool_count", False),
        "current_selection_protocol_complete": audit["selections"][
            "current_protocol_complete"
        ],
        "ultralytics_matches_protocol": audit["environment"][
            "ultralytics_matches_protocol"
        ],
    }
    blockers = [
        name for name, ok in checks.items() if not ok
    ]
    return {
        "checks": checks,
        "ready_for_local_training": not blockers,
        "blockers": blockers,
        "interpretation": (
            "ready for local training"
            if not blockers
            else "fix blockers before local training"
        ),
    }


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)
    raw, records_from_xml, quarantine = audit_raw_dataset()
    records = read_json(OUTPUT / "records.json") or []
    split = read_json(OUTPUT / "split.json") or {}

    taxonomy_report = {
        "dataset": "bvtsld",
        "mapping_status": "local_machine_audited_human_review_pending",
        "review": {
            "status": "machine_audited",
            "reviewers": [],
            "map_version": "bvtsld-visual-v1",
            "human_review_required_for_main_evidence": True,
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        },
        "target_classes": TARGET_CLASSES,
        "code_map": BVTSLD_CODE_MAP,
        "unknown_object_policy": UNKNOWN_OBJECT_POLICY,
        "auxiliary_boxes": 0,
        "observed_categories": raw["observed_codes_original_boxes"],
        "quarantine_count": len(quarantine),
        "quarantine_by_code": dict(
            sorted(Counter(code for item in quarantine for code in item["source_categories"]).items())
        ),
        "raw_dataset": raw,
    }
    write_json(OUTPUT / "taxonomy_report.json", taxonomy_report)
    write_json(OUTPUT / "quarantine.json", quarantine)

    audit = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_dataset": raw,
        "taxonomy_report": taxonomy_report,
        "quarantine": {
            "count": len(quarantine),
            "by_code": taxonomy_report["quarantine_by_code"],
        },
        "records": audit_records(records, records_from_xml),
        "split": audit_split(split, records),
        "embeddings": audit_embeddings(split),
        "selections": audit_selections(split),
        "oracle": audit_oracle(),
        "environment": audit_environment(),
    }
    audit["status"] = status_from_checks(audit)
    write_json(OUTPUT / "local_pretrain_audit.json", audit)
    print(json.dumps(audit["status"], indent=2, ensure_ascii=False))
    print(f"wrote {OUTPUT / 'taxonomy_report.json'}")
    print(f"wrote {OUTPUT / 'quarantine.json'}")
    print(f"wrote {OUTPUT / 'local_pretrain_audit.json'}")


if __name__ == "__main__":
    main()
