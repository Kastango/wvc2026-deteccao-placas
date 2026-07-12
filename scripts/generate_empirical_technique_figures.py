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
DEFAULT_EMBEDDING = "dinov2"
DEFAULT_PROJECTION = "tsne"

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


TECHNIQUES: tuple[TechniqueSpec, ...] = (
    TechniqueSpec(
        "random",
        "Random sampling",
        "method_01_random",
        "control: samples without using the pool geometry",
    ),
    TechniqueSpec(
        "kmeans_dinov2",
        "k-means + medoid",
        "method_02_kmeans",
        "one real sample per cluster in the embedding space",
    ),
    TechniqueSpec(
        "kmeans_clip",
        "k-means · CLIP",
        "method_02b_kmeans_clip",
        "same k-means selection using CLIP embeddings",
    ),
    TechniqueSpec(
        "kmeans_shallow",
        "k-means · shallow features",
        "method_02c_kmeans_shallow",
        "same k-means selection using color and texture features",
    ),
    TechniqueSpec(
        "opf_dinov2",
        "OPF root + quota",
        "method_03_opf",
        "OPF roots and clusters completed by quota to a fixed budget",
    ),
    TechniqueSpec(
        "typiclust_dinov2",
        "TypiClust",
        "method_04_typiclust",
        "typical samples in budget-defined clusters",
    ),
    TechniqueSpec(
        "kcenter_dinov2",
        "k-center greedy",
        "method_05_kcenter",
        "coverage by samples far from the current selection",
    ),
    TechniqueSpec(
        "probcover_dinov2",
        "ProbCover",
        "method_06_probcover",
        "coverage of neighbors not yet covered",
    ),
    TechniqueSpec(
        "facility_dinov2",
        "Facility location",
        "method_07_facility",
        "maximizes global similarity to the nearest prototype",
    ),
    TechniqueSpec(
        "freesel_dino",
        "FreeSel",
        "method_08_freesel",
        "selects diverse local patterns instead of only whole scenes",
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
    parser.add_argument("--embedding", default=DEFAULT_EMBEDDING)
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
        raise FileNotFoundError(f"Embedding file not found: {path}")
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
        specs.append(TechniqueSpec(key, title, slug, "selection saved by the pipeline"))
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
    xy: np.ndarray,
    selected: np.ndarray,
    title: str,
    note: str,
    compact: bool = False,
) -> None:
    draw_density(ax, xy)
    ax.scatter(
        xy[:, 0],
        xy[:, 1],
        s=14 if compact else 19,
        c=POOL_COLOR,
        edgecolors="none",
        alpha=0.34,
        zorder=1,
    )
    ax.scatter(
        xy[selected, 0],
        xy[selected, 1],
        s=92 if compact else 124,
        facecolors="none",
        edgecolors=SELECTED_COLOR,
        linewidths=1.15,
        alpha=0.22,
        zorder=2,
    )
    ax.scatter(
        xy[selected, 0],
        xy[selected, 1],
        s=48 if compact else 66,
        c=SELECTED_COLOR,
        edgecolors="white",
        linewidths=1.15,
        alpha=0.98,
        zorder=3,
    )
    if compact:
        ax.text(
            0.0, 1.14, title, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=15, fontweight="bold", color=TEXT_COLOR,
        )
        ax.text(
            0.0, 1.055, textwrap.fill(note, width=62), transform=ax.transAxes,
            ha="left", va="bottom", fontsize=9.5, color=MUTED_COLOR,
        )
        ax.text(
            0.965, 0.96, f"{len(selected)} selected", transform=ax.transAxes,
            ha="right", va="top", fontsize=8.5, fontweight="bold",
            color=SELECTED_COLOR,
            bbox={"boxstyle": "round,pad=0.28", "facecolor": "white",
                  "edgecolor": GRID_COLOR, "alpha": 0.92},
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
        setup_axis(ax, title, note, len(selected), len(xy))


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
    xy: np.ndarray,
    selections: dict[str, np.ndarray],
    specs: list[TechniqueSpec],
    args: argparse.Namespace,
) -> None:
    for spec in specs:
        fig, ax = plt.subplots(figsize=(8.6, 6.0), constrained_layout=True)
        fig.patch.set_facecolor(FIGURE_BG)
        draw_selection(ax, xy, selections[spec.key], spec.title, spec.note)
        fig.suptitle(
            f"{args.dataset.upper()} - {args.embedding.upper()} - {args.projection.upper()} "
            f"- fraction {args.fraction:.0%}, repeat {args.repeat}",
            x=0.0,
            ha="left",
            fontsize=9,
            color=MUTED_COLOR,
        )
        save_figure(
            fig,
            args.output_dir,
            f"{spec.slug}_empirical_{args.dataset}_{args.embedding}_{args.projection}_{fraction_tag(args.fraction)}_rep{args.repeat}",
        )
        plt.close(fig)


def render_grid(
    xy: np.ndarray,
    selections: dict[str, np.ndarray],
    specs: list[TechniqueSpec],
    args: argparse.Namespace,
) -> None:
    cols = 2
    rows = math.ceil(len(specs) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(10.5, 4.55 * rows), constrained_layout=False)
    fig.patch.set_facecolor(FIGURE_BG)
    fig.subplots_adjust(left=0.055, right=0.975, top=0.89, bottom=0.045, hspace=0.42, wspace=0.16)
    axes_arr = np.array(axes, dtype=object).reshape(rows, cols)

    for ax, spec in zip(axes_arr.flat, specs):
        draw_selection(ax, xy, selections[spec.key], spec.title, spec.note, compact=True)

    for ax in axes_arr.flat[len(specs):]:
        ax.axis("off")

    title = "How each method samples the same unlabeled pool"
    fig.suptitle(title, fontsize=22, fontweight="bold", color=TEXT_COLOR, y=0.985)
    fig.text(
        0.5,
        0.965,
        f"{args.dataset.upper()} · shared {args.embedding.upper()} {args.projection.upper()} projection · "
        f"label fraction {args.fraction:.0%} · repeat {args.repeat}",
        ha="center",
        va="top",
        fontsize=11,
        color=MUTED_COLOR,
    )

    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=POOL_COLOR,
               markeredgewidth=0, markersize=7, alpha=0.42, label="unlabeled pool"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=SELECTED_COLOR,
               markeredgecolor="white", markeredgewidth=1.0, markersize=9,
               label="selected image"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncols=2,
        frameon=False,
        fontsize=10.5,
        bbox_to_anchor=(0.5, 0.944),
    )
    fig.text(
        0.5, 0.014,
        "The 2D projection is for visualization only; selection runs in the full embedding space.",
        ha="center", va="bottom", fontsize=9.5, color=MUTED_COLOR,
    )
    save_figure(
        fig,
        args.output_dir,
        f"methods_empirical_grid_{args.dataset}_{args.embedding}_{args.projection}_{fraction_tag(args.fraction)}_rep{args.repeat}",
    )
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output = ROOT / "outputs" / args.dataset
    pool_ids = load_pool_ids(output)
    emb = load_embeddings(output, args.dataset, args.embedding)
    if len(pool_ids) != len(emb):
        raise ValueError(
            f"Pool size ({len(pool_ids)}) and embedding rows ({len(emb)}) do not match."
        )

    specs = available_specs(args.techniques)
    xy = normalize_xy(project_embeddings(emb, args.projection, args.seed))
    selections = {
        spec.key: load_selection_indices(output, spec.key, args.fraction, args.repeat, pool_ids)
        for spec in specs
    }

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
        render_individual(xy, selections, specs, args)
    if args.grid:
        render_grid(xy, selections, specs, args)

    if args.individual:
        print(f"Generated {len(specs)} individual figures in {args.output_dir}")
    if args.grid:
        print("Generated comparative grid")


if __name__ == "__main__":
    main()
