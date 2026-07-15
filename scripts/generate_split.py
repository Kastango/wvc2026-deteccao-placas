"""Generate a frozen pool/validation/test split for a dataset.

Deterministic: shuffles the eligible record ids with ``random.Random(42)`` and
takes 15% for validation and 15% for test; the remaining 70% form the training
pool. The output ``split.json`` is the frozen artifact consumed by the whole
pipeline; regenerating it with the same records reproduces it byte for byte.
"""
from __future__ import annotations

import argparse
import json
import random

from dataset_config import spec


SEED = 42
VALIDATION_SHARE = 0.15
TEST_SHARE = 0.15


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="bvtsld", choices=("bvtsld", "tt100k"))
    parser.add_argument("--force", action="store_true", help="Overwrite the frozen split.json")
    args = parser.parse_args()

    output = spec(args.dataset).output_dir
    target = output / "split.json"
    if target.exists() and not args.force:
        raise SystemExit(f"split already frozen; rerun with --force: {target}")

    ids = sorted(r["id"] for r in json.loads((output / "records.json").read_text()))
    rng = random.Random(SEED)
    rng.shuffle(ids)
    n = len(ids)
    n_validation = round(n * VALIDATION_SHARE)
    n_test = round(n * TEST_SHARE)
    n_pool = n - n_validation - n_test
    split = {
        "pool": sorted(ids[:n_pool]),
        "validation": sorted(ids[n_pool : n_pool + n_validation]),
        "test": sorted(ids[n_pool + n_validation :]),
    }
    target.write_text(json.dumps(split, indent=1, ensure_ascii=False) + "\n")
    print({name: len(values) for name, values in split.items()}, f"seed={SEED}")


if __name__ == "__main__":
    main()
