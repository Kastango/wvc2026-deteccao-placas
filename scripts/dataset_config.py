"""Shared dataset registry and selection protocol constants.

Every pipeline script resolves paths, target classes and code maps from here,
so the same code runs on the 2-class BVTSLD and on the 3-class TT100K by
switching ``--dataset``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# Selection protocol, shared by every dataset.
FRACTIONS = (0.05, 0.10, 0.20, 0.50)
STOCHASTIC_REPEATS = 8
DETERMINISTIC_TECHNIQUES = frozenset({"opf_dinov2"})
TECHNIQUES = (
    "random", "kmeans_dinov2", "opf_dinov2",
    "typiclust_dinov2", "probcover_dinov2", "freesel_dino",
)

# TT100K's own category identifiers aggregate by their initial letter.
TT100K_PREFIX_MAP = {"p": "regulatory", "w": "warning", "i": "information"}


def expected_repeats() -> dict[str, int]:
    return {
        technique: 1 if technique in DETERMINISTIC_TECHNIQUES else STOCHASTIC_REPEATS
        for technique in TECHNIQUES
    }


def expected_selections() -> int:
    return sum(expected_repeats().values()) * len(FRACTIONS)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    target_classes: tuple[str, ...]
    dataset_root: Path
    map_version: str
    # Frozen code map (BVTSLD); TT100K maps by prefix instead.
    code_map: dict[str, int] = field(default_factory=dict)
    quarantined_codes: frozenset[str] = frozenset()

    @property
    def output_dir(self) -> Path:
        return ROOT / "outputs" / self.key

    @property
    def yolo_dir(self) -> Path:
        return self.output_dir / f"yolo_{self.key}"

    @property
    def embeddings_path(self) -> Path:
        return self.output_dir / f"embeddings_{self.key}_dinov2_full.npy"

    @property
    def patterns_path(self) -> Path:
        return self.output_dir / f"patterns_{self.key}_freesel.npz"

    def class_index_for_code(self, code: str) -> int | None:
        """Target class index for a source category, or None when unmapped."""
        if self.code_map:
            return self.code_map.get(code)
        group = TT100K_PREFIX_MAP.get(code[:1].lower())
        return self.target_classes.index(group) if group in self.target_classes else None


DATASETS = {
    # BVTSLD 2-class protocol: every CONTRAN R-* code observed in the original
    # images -> regulatory; the three traffic-light codes -> traffic_light.
    # The only excluded code is 000025 (A-18, warning): images that contain it
    # are quarantined. The 3-class taxonomy remains reserved for TT100K.
    "bvtsld": DatasetSpec(
        key="bvtsld",
        target_classes=("regulatory", "traffic_light"),
        dataset_root=ROOT / "datasets" / "bvtsld"
        / "Brazilian Vertical Traffic Signs and Lights Dataset",
        map_version="bvtsld-2class-v3",
        code_map={
            "000000": 0, "000001": 0, "000003": 0, "000004": 0, "000007": 0,
            "000008": 0, "000009": 0, "000023": 0, "000028": 0, "000035": 0,
            "000040": 0, "000042": 0,
            "000051": 1, "000052": 1, "000053": 1,
        },
        quarantined_codes=frozenset({"000025"}),
    ),
    "tt100k": DatasetSpec(
        key="tt100k",
        target_classes=("regulatory", "warning", "information"),
        dataset_root=ROOT / "datasets" / "tt100k" / "data",
        map_version="tt100k-prefix-code-review-v1",
    ),
}


def spec(key: str) -> DatasetSpec:
    if key not in DATASETS:
        raise KeyError(f"unknown dataset {key!r}; expected one of {sorted(DATASETS)}")
    return DATASETS[key]
