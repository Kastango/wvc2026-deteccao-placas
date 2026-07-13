"""Generate the frozen pool representations used by the selection methods.

Produces, for every image in the train pool (order of ``split.json``):

- ``embeddings_{dataset}_dinov2_full.npy``: one global DINOv2 ViT-S/14
  embedding per image (384-d, CLS token, 224 x 224 input).
- ``patterns_{dataset}_freesel.npz``: five local DINO ViT-S/16 patterns per
  image (384-d), obtained from the penultimate block by retaining the patches
  responsible for 50% of the CLS attention and clustering them into five
  centroids. This follows FreeSel's official dense semantic pooling recipe.

``--verify`` compares freshly computed representations of a sample of pool
images against the stored artifacts (cosine similarity) instead of writing
anything. Use it to confirm that this script reproduces the frozen artifacts
before regenerating them on a new machine.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.cluster import KMeans
from torchvision import transforms


ROOT = Path(__file__).resolve().parents[1]
PATTERNS_PER_IMAGE = 5
FREESEL_ATTENTION_MASS = 0.5
SEED = 42

IMAGENET_NORM = transforms.Normalize(
    mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
)


def device_name() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load_pool(output: Path) -> list[Path]:
    split = json.loads((output / "split.json").read_text())
    records = {r["id"]: r for r in json.loads((output / "records.json").read_text())}
    return [Path(records[image_id]["image"]) for image_id in split["pool"]]


def preprocess(path: Path, side: int = 224) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = transforms.functional.to_tensor(
        transforms.functional.resize(image, (side, side))
    )
    return IMAGENET_NORM(tensor)


@torch.no_grad()
def dinov2_embeddings(paths: list[Path], device: str, batch: int = 16) -> np.ndarray:
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14")
    model.eval().to(device)
    # The frozen BVTSLD artifact was produced by square-resizing to 224 px.
    side = 224
    out = []
    for start in range(0, len(paths), batch):
        stack = torch.stack([preprocess(p, side) for p in paths[start : start + batch]])
        out.append(model(stack.to(device)).cpu().float().numpy())
    return np.concatenate(out)


@torch.no_grad()
def freesel_patterns(
    paths: list[Path], device: str, batch: int = 16
) -> tuple[np.ndarray, np.ndarray]:
    model = torch.hub.load("facebookresearch/dino:main", "dino_vits16")
    model.eval().to(device)
    side = 224  # 14 x 14 patches of 16 px
    patterns, ids = [], []
    for start in range(0, len(paths), batch):
        stack = torch.stack([preprocess(p, side) for p in paths[start : start + batch]])
        device_stack = stack.to(device)
        # FreeSel uses the earlier of the last two blocks, not the final block.
        tokens = model.get_intermediate_layers(device_stack, n=2)[0][:, 1:]
        attention = model.get_last_selfattention(device_stack).mean(dim=1)[:, 0, 1:]
        attention = attention / attention.sum(dim=1, keepdim=True)
        for offset, (image_patches, image_attention) in enumerate(
            zip(tokens.cpu().float().numpy(), attention.cpu().float())
        ):
            # Official FreeSel filtering: retain the most-attended patches whose
            # cumulative attention accounts for ATTENTION_MASS of the image.
            order = torch.argsort(image_attention)
            keep_sorted = torch.cumsum(image_attention[order], dim=0) > (
                1.0 - FREESEL_ATTENTION_MASS
            )
            keep = torch.zeros_like(keep_sorted, dtype=torch.bool)
            keep[order] = keep_sorted
            filtered = image_patches[keep.numpy()]
            if len(filtered) < PATTERNS_PER_IMAGE:
                filtered = image_patches
            km = KMeans(
                n_clusters=PATTERNS_PER_IMAGE, n_init=10, random_state=SEED
            ).fit(filtered)
            patterns.append(km.cluster_centers_.astype(np.float32))
            ids.extend([start + offset] * PATTERNS_PER_IMAGE)
    return np.concatenate(patterns), np.asarray(ids, dtype=np.int64)


def cosine_rowwise(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return (a * b).sum(axis=1)


def verify(output: Path, dataset: str, paths: list[Path], device: str, sample: int) -> None:
    rng = np.random.default_rng(SEED)
    picked = np.sort(rng.choice(len(paths), min(sample, len(paths)), replace=False))
    sample_paths = [paths[i] for i in picked]

    stored = np.load(output / f"embeddings_{dataset}_dinov2_full.npy")[picked]
    fresh = dinov2_embeddings(sample_paths, device)
    sim = cosine_rowwise(stored, fresh)
    print(f"dinov2: cosine min={sim.min():.4f} mean={sim.mean():.4f} n={len(sim)}")
    if sim.min() < 0.999:
        raise RuntimeError("fresh DINOv2 embeddings do not reproduce the frozen artifact")

    archive = np.load(output / f"patterns_{dataset}_freesel.npz")
    stored_patterns, stored_ids = archive["patterns"], archive["ids"]
    fresh_patterns, _ = freesel_patterns(sample_paths, device)
    sims = []
    for row, pool_index in enumerate(picked):
        stored_set = stored_patterns[stored_ids == pool_index]
        fresh_set = fresh_patterns[
            row * PATTERNS_PER_IMAGE : (row + 1) * PATTERNS_PER_IMAGE
        ]
        norm_stored = stored_set / (np.linalg.norm(stored_set, axis=1, keepdims=True) + 1e-9)
        norm_fresh = fresh_set / (np.linalg.norm(fresh_set, axis=1, keepdims=True) + 1e-9)
        # Cluster order is arbitrary: match each fresh pattern to its best stored one.
        sims.append(float((norm_fresh @ norm_stored.T).max(axis=1).mean()))
    sims = np.asarray(sims)
    print(f"freesel: matched cosine min={sims.min():.4f} mean={sims.mean():.4f} n={len(sims)}")
    # K-means centroid order and initialization vary across compatible sklearn
    # versions, so semantic equivalence is checked rather than byte identity.
    if sims.mean() < 0.85 or sims.min() < 0.75:
        raise RuntimeError("fresh FreeSel patterns do not reproduce the frozen artifact")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="bvtsld")
    parser.add_argument("--device", default=device_name())
    parser.add_argument(
        "--verify", action="store_true",
        help="Compare a sample against the stored artifacts instead of writing.",
    )
    parser.add_argument("--sample", type=int, default=32, help="Sample size for --verify.")
    args = parser.parse_args()

    output = ROOT / "outputs" / args.dataset
    paths = load_pool(output)
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        raise RuntimeError(f"Missing pool images: {missing[:5]}")

    if args.verify:
        verify(output, args.dataset, paths, args.device, args.sample)
        return

    embeddings = dinov2_embeddings(paths, args.device)
    np.save(output / f"embeddings_{args.dataset}_dinov2_full.npy", embeddings)
    print(f"wrote embeddings {embeddings.shape}")

    patterns, ids = freesel_patterns(paths, args.device)
    np.savez(
        output / f"patterns_{args.dataset}_freesel.npz", patterns=patterns, ids=ids
    )
    print(f"wrote patterns {patterns.shape}")


if __name__ == "__main__":
    main()
