from __future__ import annotations

import argparse
import json
import os
import resource
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.neighbors import NearestNeighbors

from dataset_config import FRACTIONS, STOCHASTIC_REPEATS, spec


SEED = 42
REPEATS = STOCHASTIC_REPEATS
DATASET = spec("bvtsld")
OUTPUT = DATASET.output_dir
SELECTIONS_DIR = OUTPUT / "selections"
CLASSES = list(DATASET.target_classes)
KNN_K = 64
KNN_QUERY_BATCH = 2_048
KMEANS_EXACT_MAX_POINTS = 10_000
KNN_BACKENDS_USED: set[str] = set()

METHOD_FIDELITY = {
    "random": "random sampling control",
    "kmeans_dinov2": "k-means + sample nearest to centroid on DINOv2 embeddings",
    "opf_dinov2": "OPF root + quota on the full pool",
    "typiclust_dinov2": "TypiClust",
    "probcover_dinov2": "ProbCover; unlabeled delta at purity >= 0.95",
    "freesel_dino": "FreeSel; FDS on local DINO patterns",
}


def normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def rss_mb():
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024**2) if os.uname().sysname == "Darwin" else value / 1024


def cpu_seconds():
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return usage.ru_utime + usage.ru_stime


def budget_for(fraction, n):
    return max(1, int(round(fraction * n)))


def fit_kmeans(x, n_clusters, n_init, seed):
    cls = KMeans if len(x) <= KMEANS_EXACT_MAX_POINTS else MiniBatchKMeans
    kwargs = dict(n_clusters=n_clusters, n_init=n_init, random_state=seed)
    if cls is MiniBatchKMeans:
        kwargs.update(batch_size=2_048, max_iter=100)
    return cls(**kwargs).fit(x)


def knn_cosine(x, k=KNN_K):
    x = normalize_rows(x)
    k = min(k + 1, len(x))
    try:
        import faiss

        index = faiss.IndexFlatIP(x.shape[1])
        index.add(x)
        sim, idx = index.search(x, k)
        KNN_BACKENDS_USED.add("faiss_flatip")
        return (1.0 - sim).astype(np.float32), idx.astype(np.int32), "faiss_flatip"
    except ImportError:
        nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute", n_jobs=-1)
        nn.fit(x)
        dist, idx = nn.kneighbors(x, return_distance=True)
        KNN_BACKENDS_USED.add("sklearn_cosine_knn")
        return dist.astype(np.float32), idx.astype(np.int32), "sklearn_cosine_knn"


def chunked_cosine_distance(x, selected, chunk=KNN_QUERY_BATCH):
    x, s = normalize_rows(x), normalize_rows(x[selected])
    out = np.empty(len(x), dtype=np.float32)
    chunk = max(1, min(chunk, 16_000_000 // max(1, len(s))))
    for start in range(0, len(x), chunk):
        out[start : start + chunk] = 1.0 - (x[start : start + chunk] @ s.T).max(axis=1)
    return out


def group_quotas(group_labels, budget):
    groups, sizes = np.unique(group_labels, return_counts=True)
    quotas = np.floor(budget * sizes / sizes.sum()).astype(int)
    while quotas.sum() < budget:
        quotas[np.argmax(sizes - quotas)] += 1
    return dict(zip(groups.tolist(), quotas.tolist()))


def sample_near_centroid(indices, emb, center):
    d_center = np.linalg.norm(emb[indices] - center, axis=1)
    return int(indices[np.argmin(d_center)])


def random_selections(n, fraction, seed=SEED):
    size = budget_for(fraction, n)
    return [
        np.sort(np.random.RandomState(seed + rep).choice(n, size, replace=False))
        for rep in range(REPEATS)
    ]


def kmeans_selections(emb, fraction, seed=SEED):
    budget, out = budget_for(fraction, len(emb)), []
    for rep in range(REPEATS):
        k = min(budget, len(emb))
        km = fit_kmeans(emb, n_clusters=k, n_init=10, seed=seed + rep)
        sel = [
            sample_near_centroid(np.where(km.labels_ == group)[0], emb, km.cluster_centers_[group])
            for group in range(k)
        ]
        if len(sel) != budget or len(set(sel)) != budget:
            raise RuntimeError("kmeans generated duplicate or wrong-size selection")
        out.append(np.sort(np.array(sel, dtype=int)))
    return out


def opf_selections(emb, fraction):
    import logging
    from opfython.models import UnsupervisedOPF

    for name in list(logging.Logger.manager.loggerDict):
        if name.startswith("opfython"):
            logging.getLogger(name).setLevel(logging.ERROR)

    budget = budget_for(fraction, len(emb))
    opf = UnsupervisedOPF(max_k=20)
    opf.fit(emb.tolist())
    labels = np.array([node.cluster_label for node in opf.subgraph.nodes])
    group_root = {
        node.cluster_label: node.idx for node in opf.subgraph.nodes if node.root == node.idx
    }
    selection = []
    for group, quota in group_quotas(labels, budget).items():
        if quota == 0:
            continue
        idx = np.where(labels == group)[0]
        root = int(group_root.get(group, idx[0]))
        ranked = idx[np.argsort(np.linalg.norm(emb[idx] - emb[root], axis=1))]
        picked = [root] + [int(i) for i in ranked if int(i) != root]
        selection.extend(picked[:quota])
    return [np.sort(np.unique(selection))[:budget]]


def typicality(emb, indices, k_nn=20):
    if len(indices) == 1:
        return np.array([1.0])
    dist, _, _ = knn_cosine(emb[indices], k=min(k_nn, len(indices) - 1))
    return 1.0 / (dist[:, 1:].mean(axis=1) + 1e-9)


def typiclust_selections(emb, fraction, seed=SEED):
    budget, out = budget_for(fraction, len(emb)), []
    for rep in range(REPEATS):
        k = min(budget, len(emb) - 1)
        km = fit_kmeans(emb, n_clusters=k, n_init=4, seed=seed + rep)
        sel = []
        for group in range(k):
            idx = np.where(km.labels_ == group)[0]
            if len(idx):
                sel.append(int(idx[np.argmax(typicality(emb, idx))]))
        rnd = np.random.RandomState(seed + rep)
        while len(set(sel)) < budget:
            sel.append(int(rnd.randint(len(emb))))
        out.append(np.sort(np.unique(sel))[:budget])
    return out


def probcover_selections(emb, fraction, seed=SEED, purity_target=0.95):
    x = normalize_rows(emb)
    n = len(x)
    budget = budget_for(fraction, n)
    dist = 1.0 - x @ x.T
    tri = dist[np.triu_indices(n, 1)]
    out = []
    for rep in range(REPEATS):
        rnd = np.random.RandomState(seed + rep)
        km = fit_kmeans(x, n_clusters=min(budget, n), n_init=4, seed=seed + rep)
        same_label = km.labels_[None, :] == km.labels_[:, None]
        delta = float(np.quantile(tri, 0.02))
        for q in np.linspace(0.4, 0.02, 20):
            cand = float(np.quantile(tri, q))
            ball = dist <= cand
            purity = (~ball | same_label).all(axis=1).mean()
            if purity >= purity_target:
                delta = cand
                break
        cover = dist <= delta
        covered = np.zeros(n, dtype=bool)
        sel = []
        while len(sel) < budget:
            gains = cover[:, ~covered].sum(axis=1).astype(float)
            if sel:
                gains[np.array(sel, dtype=int)] = -1.0
            if gains.max() <= 0:
                covered[:] = False
                continue
            best = np.flatnonzero(gains == gains.max())
            nxt = int(rnd.choice(best))
            sel.append(nxt)
            covered |= cover[nxt]
        out.append(np.sort(np.array(sel, dtype=int)))
    return out


def freesel_selections(patterns, pattern_ids, n_images, fraction, seed=SEED):
    x = normalize_rows(patterns)
    budget = budget_for(fraction, n_images)
    out = []
    for rep in range(REPEATS):
        rnd = np.random.RandomState(seed + rep)
        first = int(rnd.randint(n_images))
        chosen = [first]
        dmin = (1.0 - x @ x[pattern_ids == first].T).min(axis=1)
        while len(chosen) < budget:
            for p in np.argsort(-dmin):
                img = int(pattern_ids[p])
                if img not in chosen:
                    break
            chosen.append(img)
            dmin = np.minimum(dmin, (1.0 - x @ x[pattern_ids == img].T).min(axis=1))
        out.append(np.sort(np.array(chosen, dtype=int)))
    return out


def coverage(emb, selection):
    d = chunked_cosine_distance(emb, selection)
    return float(d.mean()), float(d.max())


def jaccard(a, b):
    a, b = set(a.tolist()), set(b.tolist())
    return len(a & b) / len(a | b)


def selection_classes(pool, selection):
    return dict(Counter(CLASSES[int(c)] for i in selection for (c, *_) in pool[i]["boxes"]))


def archive_previous_selections():
    if not SELECTIONS_DIR.exists() or not any(SELECTIONS_DIR.glob("*.json")):
        SELECTIONS_DIR.mkdir(exist_ok=True)
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = OUTPUT / f"selections_legacy_{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    for path in SELECTIONS_DIR.glob("*.json"):
        shutil.move(str(path), archive / path.name)
    return archive


def configure(dataset_key: str) -> None:
    global DATASET, OUTPUT, SELECTIONS_DIR, CLASSES
    DATASET = spec(dataset_key)
    OUTPUT = DATASET.output_dir
    SELECTIONS_DIR = OUTPUT / "selections"
    CLASSES = list(DATASET.target_classes)


def load_inputs():
    records = json.loads((OUTPUT / "records.json").read_text())
    split = json.loads((OUTPUT / "split.json").read_text())
    by_id = {record["id"]: record for record in records}
    pool = [by_id[image_id] for image_id in split["pool"]]
    emb = normalize_rows(np.load(DATASET.embeddings_path))
    freesel = np.load(DATASET.patterns_path)
    patterns, pattern_ids = freesel["patterns"], freesel["ids"]
    if len(emb) != len(pool):
        raise RuntimeError(f"embedding dinov2 rows {len(emb)} != pool size {len(pool)}")
    if int(pattern_ids.max()) != len(pool) - 1:
        raise RuntimeError("freesel pattern ids do not span the pool")
    return pool, emb, patterns, pattern_ids


def job_table(pool, emb, patterns, pattern_ids):
    return {
        "random": lambda fraction: random_selections(len(pool), fraction),
        "kmeans_dinov2": lambda fraction: kmeans_selections(emb, fraction),
        "opf_dinov2": lambda fraction: opf_selections(emb, fraction),
        "typiclust_dinov2": lambda fraction: typiclust_selections(emb, fraction),
        "probcover_dinov2": lambda fraction: probcover_selections(emb, fraction),
        "freesel_dino": lambda fraction: freesel_selections(
            patterns, pattern_ids, len(pool), fraction
        ),
    }


def run_job(technique: str, fraction: float, job_output: Path) -> None:
    """Execute one technique x fraction in this dedicated process.

    Wall time, CPU time and peak RSS are measured inside a process that runs
    nothing else, so no other technique interferes with the measurement.
    """
    pool, emb, patterns, pattern_ids = load_inputs()
    job = job_table(pool, emb, patterns, pattern_ids)[technique]
    KNN_BACKENDS_USED.clear()
    rss_before, started = rss_mb(), time.perf_counter()
    cpu_before = cpu_seconds()
    selections = job(fraction)
    wall = time.perf_counter() - started
    payload = {
        "technique": technique,
        "fraction": fraction,
        "selections": [[int(i) for i in selection] for selection in selections],
        "telemetry": {
            "selection_seconds_total": wall,
            "selection_seconds_per_repeat": wall / len(selections),
            "selection_cpu_seconds_total": cpu_seconds() - cpu_before,
            "rss_before_mb": rss_before,
            "rss_peak_mb": rss_mb(),
            "knn_backend": "+".join(sorted(KNN_BACKENDS_USED)) or "not_applicable",
            "opf_scope": "full_pool" if technique == "opf_dinov2" else "not_applicable",
            "measurement_isolation": "dedicated_process",
        },
    }
    job_output.write_text(json.dumps(payload))


def run_isolated(technique: str, fraction: float) -> dict:
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as handle:
        job_output = Path(handle.name)
    try:
        subprocess.run(
            [
                sys.executable, str(Path(__file__).resolve()),
                "--dataset", DATASET.key,
                "--job", technique, "--fraction", f"{fraction}",
                "--job-output", str(job_output),
            ],
            check=True,
        )
        return json.loads(job_output.read_text())
    finally:
        job_output.unlink(missing_ok=True)


def main():
    pool, emb, patterns, pattern_ids = load_inputs()

    archive = archive_previous_selections()
    rows = []

    for fraction in FRACTIONS:
        for technique in METHOD_FIDELITY:
            result = run_isolated(technique, fraction)
            selections = [np.asarray(s, dtype=int) for s in result["selections"]]
            telemetry = result["telemetry"]
            overlaps = [
                jaccard(selections[i], selections[j])
                for i in range(len(selections))
                for j in range(i + 1, len(selections))
            ]
            for rep, selection in enumerate(selections):
                selection_seed = SEED + rep
                cov_mean, cov_max = coverage(emb, selection)
                name = f"{technique}_frac{int(fraction * 100):02d}_rep{rep + 1}"
                artifact = {
                    "technique": technique,
                    "implementation": METHOD_FIDELITY[technique],
                    "fraction": fraction,
                    "repeat": rep + 1,
                    "selection_seed": selection_seed,
                    "images": [pool[i]["id"] for i in selection],
                }
                (SELECTIONS_DIR / f"{name}.json").write_text(
                    json.dumps(artifact, indent=1, ensure_ascii=False) + "\n"
                )
                rows.append(
                    {
                        "technique": technique,
                        "fraction": fraction,
                        "repeat": rep + 1,
                        "selection_seed": selection_seed,
                        "size": len(selection),
                        "coverage_mean": round(cov_mean, 4),
                        "coverage_max": round(cov_max, 4),
                        "stability_jaccard": (
                            round(float(np.mean(overlaps)), 3) if overlaps else 1.0
                        ),
                        "classes": selection_classes(pool, selection),
                        "implementation": METHOD_FIDELITY[technique],
                        **telemetry,
                    }
                )
            print(f"generated {technique} fraction={fraction:.2f}")

    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT / "selections_summary.csv", index=False)
    print(f"wrote {len(rows)} rows to {OUTPUT / 'selections_summary.csv'}")
    if archive:
        print(f"archived previous selections in {archive}")


if __name__ == "__main__":
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", default="bvtsld", choices=("bvtsld", "tt100k"))
    cli.add_argument("--job", choices=sorted(METHOD_FIDELITY))
    cli.add_argument("--fraction", type=float, choices=FRACTIONS)
    cli.add_argument("--job-output", type=Path)
    cli_args = cli.parse_args()
    configure(cli_args.dataset)
    if cli_args.job:
        if cli_args.fraction is None or cli_args.job_output is None:
            cli.error("--job requires --fraction and --job-output")
        run_job(cli_args.job, cli_args.fraction, cli_args.job_output)
    else:
        main()
