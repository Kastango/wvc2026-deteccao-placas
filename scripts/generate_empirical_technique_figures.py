"""Generate empirical/conceptual figures for sample-selection techniques.

The previous technique figures are hand-built diagrams. This script keeps the
figures explanatory, but anchors them in the actual experiment artifacts:
embeddings from the unlabeled pool and saved selections from each technique.

The 2D projection is only a visual aid. Selections are loaded from the original
pipeline outputs, where the methods operated in the full embedding space.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.lines import Line2D
from scipy.stats import gaussian_kde
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET = "bvtsld"
DEFAULT_FRACTION = 0.10
DEFAULT_REPEAT = 1
DEFAULT_PROJECTION = "tsne"
SELECTION_SEED = 42

FIGURE_BG = "#FAFBFC"
PANEL_BG = "#FFFFFF"
POOL_COLOR = "#9AA7B3"
SELECTED_COLOR = "#1565C0"
TEXT_COLOR = "#102A43"
MUTED_COLOR = "#52616B"
GRID_COLOR = "#E9EEF3"
DENSITY_COLOR = "#D7E3F4"


@dataclass(frozen=True)
class TechniqueSpec:
    key: str
    title: str
    slug: str
    note: str
    space: str


@dataclass(frozen=True)
class PanelData:
    xy: np.ndarray
    selected_points: np.ndarray
    selected_images: int
    pool_items: int
    space_label: str
    draw_pool_density: bool = True


TECHNIQUES: tuple[TechniqueSpec, ...] = (
    TechniqueSpec(
        "random",
        "Random sampling",
        "method_01_random",
        "control: samples without using the pool geometry",
        "no embedding · index grid",
    ),
    TechniqueSpec(
        "kmeans_dinov2",
        "k-means + nearest-to-centroid sample",
        "method_02_kmeans",
        "one real sample per cluster in the DINOv2 embedding",
        "DINOv2 · image embedding",
    ),
    TechniqueSpec(
        "opf_dinov2",
        "OPF root + quota",
        "method_03_opf",
        "OPF roots and clusters completed by quota to a fixed budget",
        "DINOv2 · image embedding",
    ),
    TechniqueSpec(
        "typiclust_dinov2",
        "TypiClust",
        "method_04_typiclust",
        "typical samples in budget-defined clusters",
        "DINOv2 · image embedding",
    ),
    TechniqueSpec(
        "probcover_dinov2",
        "ProbCover",
        "method_06_probcover",
        "coverage of neighbors not yet covered",
        "DINOv2 · image embedding",
    ),
    TechniqueSpec(
        "freesel_dino",
        "FreeSel",
        "method_08_freesel",
        "selects diverse local patterns instead of only whole scenes",
        "DINO · local patterns",
    ),
)

TECHNIQUE_BY_KEY = {spec.key: spec for spec in TECHNIQUES}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate technique figures from real embeddings and saved selections."
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--fraction", type=float, default=DEFAULT_FRACTION)
    parser.add_argument("--repeat", type=int, default=DEFAULT_REPEAT)
    parser.add_argument(
        "--projection",
        choices=("pca", "tsne"),
        default=DEFAULT_PROJECTION,
        help="2D projection used only for visualization.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "figs")
    parser.add_argument(
        "--techniques",
        nargs="*",
        default=[spec.key for spec in TECHNIQUES],
        help="Technique keys to render. Defaults to every known saved technique.",
    )
    parser.add_argument(
        "--grid",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also render a comparative small-multiples panel.",
    )
    parser.add_argument(
        "--individual",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Render one additional figure per method. Disabled by default to avoid duplicates.",
    )
    return parser.parse_args()


def fraction_tag(fraction: float) -> str:
    return f"frac{int(round(fraction * 100)):02d}"


def load_json(path: Path) -> object:
    with path.open(encoding="utf-8") as fh:
        return json.load(fh)


def load_pool_ids(output: Path) -> list[str]:
    split = load_json(output / "split.json")
    if not isinstance(split, dict) or not isinstance(split.get("pool"), list):
        raise ValueError(f"Invalid split file: {output / 'split.json'}")
    return [str(item) for item in split["pool"]]


def load_embeddings(output: Path, dataset: str, embedding: str) -> np.ndarray:
    path = output / f"embeddings_{dataset}_{embedding}_full.npy"
    if not path.exists():
        path = output / f"embeddings_{dataset}_{embedding}.npy"
    if not path.exists():
        raise FileNotFoundError(
            f"Embedding file not found for '{embedding}' in {output}"
        )
    emb = np.load(path)
    if emb.ndim != 2:
        raise ValueError(f"Expected a 2D embedding matrix, got shape {emb.shape}")
    return emb.astype(np.float32, copy=False)


def project_embeddings(emb: np.ndarray, method: str, seed: int) -> np.ndarray:
    if method == "pca":
        return PCA(n_components=2, random_state=seed).fit_transform(emb)

    init = PCA(n_components=2, random_state=seed).fit_transform(emb)
    perplexity = min(30, max(5, (len(emb) - 1) // 3))
    return TSNE(
        n_components=2,
        init=init,
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
        max_iter=1_000,
    ).fit_transform(emb)


def normalize_xy(xy: np.ndarray) -> np.ndarray:
    lo = np.percentile(xy, 1, axis=0)
    hi = np.percentile(xy, 99, axis=0)
    span = np.maximum(hi - lo, 1e-9)
    out = (xy - lo) / span
    return np.clip(out, -0.05, 1.05)


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return values / (np.linalg.norm(values, axis=1, keepdims=True) + 1e-9)


def index_grid(n_items: int) -> np.ndarray:
    """Arbitrary layout for methods that do not use a representation space."""
    columns = math.ceil(math.sqrt(n_items))
    index = np.arange(n_items)
    xy = np.column_stack((index % columns, index // columns)).astype(np.float32)
    return normalize_xy(xy)


def selection_path(output: Path, technique: str, fraction: float, repeat: int) -> Path:
    return output / "selections" / f"{technique}_{fraction_tag(fraction)}_rep{repeat}.json"


def load_selection_indices(
    output: Path,
    technique: str,
    fraction: float,
    repeat: int,
    pool_ids: list[str],
) -> np.ndarray:
    path = selection_path(output, technique, fraction, repeat)
    if not path.exists():
        raise FileNotFoundError(f"Selection file not found: {path}")

    data = load_json(path)
    if not isinstance(data, dict) or not isinstance(data.get("images"), list):
        raise ValueError(f"Invalid selection file: {path}")

    pool_index = {image_id: i for i, image_id in enumerate(pool_ids)}
    missing = [image_id for image_id in data["images"] if image_id not in pool_index]
    if missing:
        sample = ", ".join(missing[:5])
        raise ValueError(f"{path} contains images outside the pool: {sample}")

    return np.array([pool_index[str(image_id)] for image_id in data["images"]], dtype=int)


def available_specs(techniques: Iterable[str]) -> list[TechniqueSpec]:
    specs: list[TechniqueSpec] = []
    for key in techniques:
        if key in TECHNIQUE_BY_KEY:
            specs.append(TECHNIQUE_BY_KEY[key])
            continue

        title = re.sub(r"[_-]+", " ", key).strip()
        slug = re.sub(r"[^a-z0-9]+", "_", key.lower()).strip("_")
        specs.append(
            TechniqueSpec(
                key, title, slug, "selection saved by the pipeline",
                "unknown representation",
            )
        )
    return specs


def setup_axis(ax: plt.Axes, title: str, note: str, selected_count: int, pool_count: int) -> None:
    ax.set_title(title, loc="left", fontsize=15, fontweight="bold", color=TEXT_COLOR, pad=10)
    ax.text(
        0,
        -0.075,
        f"{selected_count}/{pool_count} selected - {note}",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.5,
        color=MUTED_COLOR,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_facecolor(PANEL_BG)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_density(ax: plt.Axes, xy: np.ndarray) -> None:
    """Draw a subtle density field so the pool shape reads before the points."""
    try:
        values = gaussian_kde(xy.T)
    except np.linalg.LinAlgError:
        return

    grid_x, grid_y = np.mgrid[-0.08:1.08:130j, -0.08:1.08:130j]
    coords = np.vstack([grid_x.ravel(), grid_y.ravel()])
    density = values(coords).reshape(grid_x.shape)
    density = density / max(float(density.max()), 1e-12)

    cmap = LinearSegmentedColormap.from_list(
        "pool_density",
        [(1, 1, 1, 0), (*to_rgb(DENSITY_COLOR), 0.62)],
    )
    ax.imshow(
        density.T,
        extent=(-0.08, 1.08, -0.08, 1.08),
        origin="lower",
        cmap=cmap,
        interpolation="bicubic",
        zorder=0,
    )


def draw_selection(
    ax: plt.Axes,
    panel: PanelData,
    title: str,
    note: str,
    compact: bool = False,
) -> None:
    xy = panel.xy
    selected = panel.selected_points
    if panel.draw_pool_density:
        draw_density(ax, xy)
    dense_pattern_space = len(xy) > 1_500
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=(5 if dense_pattern_space else 14) if compact else (8 if dense_pattern_space else 19),
        c=POOL_COLOR,
        edgecolors="none",
        alpha=0.26 if dense_pattern_space else 0.34,
        zorder=1,
    )
    ax.scatter(
        xy[selected, 0],
        xy[selected, 1],
        s=(42 if dense_pattern_space else 92) if compact else (58 if dense_pattern_space else 124),
        facecolors="none",
        edgecolors=SELECTED_COLOR,
        linewidths=1.15,
        alpha=0.22,
        zorder=2,
    )
    ax.scatter(
        xy[selected, 0],
        xy[selected, 1],
        s=(20 if dense_pattern_space else 48) if compact else (28 if dense_pattern_space else 66),
        c=SELECTED_COLOR,
        edgecolors="white",
        linewidths=1.15,
        alpha=0.98,
        zorder=3,
    )
    if compact:
        ax.text(
            0.5, 1.14, title, transform=ax.transAxes, ha="center", va="bottom",
            fontsize=11.5, fontweight="bold", color=TEXT_COLOR,
        )
        ax.text(
            0.5, 1.045, textwrap.fill(note, width=38), transform=ax.transAxes,
            ha="center", va="bottom", fontsize=7.5, color=MUTED_COLOR,
        )
        ax.text(
            0.5, 0.04, panel.space_label, transform=ax.transAxes,
            ha="center", va="bottom", fontsize=6.8, fontweight="bold",
            color=MUTED_COLOR,
            bbox={"boxstyle": "round,pad=0.3", "facecolor": "#F4F7FA",
                  "edgecolor": GRID_COLOR, "alpha": 0.94},
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-0.08, 1.08)
        ax.set_aspect("equal", adjustable="box")
        ax.set_facecolor(PANEL_BG)
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(GRID_COLOR)
            spine.set_linewidth(0.9)
    else:
        setup_axis(ax, title, note, panel.selected_images, panel.pool_items)


def trace_freesel_patterns(
    patterns: np.ndarray,
    pattern_ids: np.ndarray,
    n_images: int,
    fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce FreeSel and retain the pattern that triggers each image choice."""
    budget = max(1, int(round(fraction * n_images)))
    rng = np.random.RandomState(seed)
    first_image = int(rng.randint(n_images))
    first_patterns = np.flatnonzero(pattern_ids == first_image)
    chosen_images = [first_image]
    # The first image is sampled before any farthest pattern exists. Use its
    # first local pattern as its single visual representative.
    trigger_patterns = [int(first_patterns[0])]
    min_distance = (1.0 - patterns @ patterns[first_patterns].T).min(axis=1)

    while len(chosen_images) < budget:
        for pattern_index in np.argsort(-min_distance):
            image_index = int(pattern_ids[pattern_index])
            if image_index not in chosen_images:
                break
        chosen_images.append(image_index)
        trigger_patterns.append(int(pattern_index))
        image_patterns = np.flatnonzero(pattern_ids == image_index)
        min_distance = np.minimum(
            min_distance,
            (1.0 - patterns @ patterns[image_patterns].T).min(axis=1),
        )
    return np.asarray(chosen_images), np.asarray(trigger_patterns)


def build_panels(
    output: Path,
    dataset: str,
    pool_ids: list[str],
    selections: dict[str, np.ndarray],
    specs: list[TechniqueSpec],
    projection: str,
    seed: int,
    fraction: float,
    repeat: int,
) -> dict[str, PanelData]:
    """Project each method in the representation space it actually uses."""
    image_spaces = {
        "dinov2": load_embeddings(output, dataset, "dinov2"),
    }
    for name, values in image_spaces.items():
        if len(values) != len(pool_ids):
            raise ValueError(f"{name} rows ({len(values)}) != pool size ({len(pool_ids)})")

    projected = {
        name: normalize_xy(project_embeddings(normalize_rows(values), projection, seed))
        for name, values in image_spaces.items()
    }

    freesel_path = output / f"patterns_{dataset}_freesel.npz"
    if not freesel_path.exists():
        raise FileNotFoundError(f"FreeSel local-pattern artifact not found: {freesel_path}")
    freesel = np.load(freesel_path)
    patterns = normalize_rows(freesel["patterns"])
    pattern_ids = np.asarray(freesel["ids"], dtype=int)
    if pattern_ids.min() != 0 or pattern_ids.max() != len(pool_ids) - 1:
        raise ValueError("FreeSel pattern IDs do not span the image pool")
    projected_patterns = normalize_xy(project_embeddings(patterns, projection, seed))

    panels: dict[str, PanelData] = {}
    for spec in specs:
        selected_images = selections[spec.key]
        if spec.key == "random":
            panels[spec.key] = PanelData(
                xy=index_grid(len(pool_ids)),
                selected_points=selected_images,
                selected_images=len(selected_images),
                pool_items=len(pool_ids),
                space_label=spec.space,
                draw_pool_density=False,
            )
        elif spec.key == "freesel_dino":
            traced_images, selected_patterns = trace_freesel_patterns(
                patterns,
                pattern_ids,
                len(pool_ids),
                fraction,
                SELECTION_SEED + repeat - 1,
            )
            if set(traced_images.tolist()) != set(selected_images.tolist()):
                raise ValueError("Reconstructed FreeSel trace differs from saved selection")
            panels[spec.key] = PanelData(
                xy=projected_patterns,
                selected_points=selected_patterns,
                selected_images=len(selected_images),
                pool_items=len(pool_ids),
                space_label=spec.space,
            )
        else:
            panels[spec.key] = PanelData(
                xy=projected["dinov2"],
                selected_points=selected_images,
                selected_images=len(selected_images),
                pool_items=len(pool_ids),
                space_label=spec.space,
            )
    return panels


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=200,
        bbox_inches="tight",
        facecolor="white",
        metadata={"Creator": "generate_empirical_technique_figures.py"},
    )


def render_individual(
    panels: dict[str, PanelData],
    specs: list[TechniqueSpec],
    args: argparse.Namespace,
) -> None:
    for spec in specs:
        fig, ax = plt.subplots(figsize=(8.6, 6.0), constrained_layout=True)
        fig.patch.set_facecolor(FIGURE_BG)
        draw_selection(ax, panels[spec.key], spec.title, spec.note)
        fig.suptitle(
            f"{args.dataset.upper()} - {spec.space} - {args.projection.upper()} "
            f"- fraction {args.fraction:.0%}, repeat {args.repeat}",
            x=0.0,
            ha="left",
            fontsize=9,
            color=MUTED_COLOR,
        )
        save_figure(
            fig,
            args.output_dir,
            f"{spec.slug}_selection_space_{args.dataset}_{args.projection}_{fraction_tag(args.fraction)}_rep{args.repeat}",
        )
        plt.close(fig)


def render_grid(
    panels: dict[str, PanelData],
    specs: list[TechniqueSpec],
    args: argparse.Namespace,
) -> None:
    cols = 3
    rows = math.ceil(len(specs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(12.4, 9.3), constrained_layout=False)
    fig.patch.set_facecolor(FIGURE_BG)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.76, bottom=0.10, hspace=0.48, wspace=0.12)
    axes_arr = np.array(axes, dtype=object).reshape(rows, cols)

    for ax, spec in zip(axes_arr.flat, specs):
        draw_selection(ax, panels[spec.key], spec.title, spec.note, compact=True)

    for ax in axes_arr.flat[len(specs):]:
        ax.axis("off")

    title = "How each method represents and samples the unlabeled pool"
    fig.suptitle(
        title, x=0.5, ha="center", fontsize=21, fontweight="bold",
        color=TEXT_COLOR, y=0.98,
    )
    fig.text(
        0.5,
        0.94,
        f"{args.dataset.upper()} · method-specific {args.projection.upper()} projections · "
        f"{next(iter(panels.values())).selected_images} "
        f"images selected per method · repeat {args.repeat}",
        ha="center",
        va="top",
        fontsize=10.5,
        color=MUTED_COLOR,
    )

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=POOL_COLOR,
               markeredgewidth=0, markersize=7, alpha=0.42, label="pool item in the method's space"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SELECTED_COLOR,
               markeredgecolor="white", markeredgewidth=1.0, markersize=9,
               label="selected image (representative pattern for FreeSel)"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncols=2,
        frameon=False,
        fontsize=10,
        bbox_to_anchor=(0.5, 0.89),
    )
    fig.text(
        0.5, 0.025,
        "Each panel projects the representation actually used by that method; selection runs in the full space.",
        ha="center", va="bottom", fontsize=9.5, color=MUTED_COLOR,
    )
    save_figure(
        fig,
        args.output_dir,
        f"methods_selection_spaces_{args.dataset}_{args.projection}_{fraction_tag(args.fraction)}_rep{args.repeat}",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = ROOT / "outputs" / args.dataset
    pool_ids = load_pool_ids(output)
    specs = available_specs(args.techniques)
    selections = {
        spec.key: load_selection_indices(output, spec.key, args.fraction, args.repeat, pool_ids)
        for spec in specs
    }
    panels = build_panels(
        output, args.dataset, pool_ids, selections, specs, args.projection,
        args.seed, args.fraction, args.repeat,
    )

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": PANEL_BG,
            "figure.facecolor": FIGURE_BG,
            "axes.edgecolor": GRID_COLOR,
            "savefig.facecolor": FIGURE_BG,
        }
    )

    if args.individual:
        render_individual(panels, specs, args)
    if args.grid:
        render_grid(panels, specs, args)

    if args.individual:
        print(f"Generated {len(specs)} individual figures in {args.output_dir}")
    if args.grid:
        print("Generated comparative grid")


if __name__ == "__main__":
    main()
