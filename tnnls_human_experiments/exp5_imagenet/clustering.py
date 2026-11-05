"""Experiment 5 clustering: standardize discovery features, PHATE 10-d
(seed 7000), k-means k=100 (n_init=50, seed 7001); evaluation-half routing.

Evaluation routing uses the spec's frozen fallback: nearest-neighbor transfer
in the standardized 512-d space (the evaluation image takes the discovery
cluster of its nearest discovery representation). The PHATE library's
out-of-sample transform is not used because it is not a stable fixed mapping
for 25k training points in this implementation; the method actually used is
recorded in every assignment row and in the run report (freeze F1).

Also builds the per-cluster evaluation cohorts (in-cluster evaluation images
+ an equally sized stratified off-cluster sample), deterministically and
independently of any factor — the same observations serve the Local CorEx
support and all of its controls (spec + freeze F3).
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import config, seeds
from common.store import ResultStore

PHATE_SEED = 7000
KMEANS_SEED = 7001
N_CLUSTERS = 100
ASSIGNMENT_METHOD = "nearest_discovery_neighbor_std512"
MAX_EVAL_IMAGES = 1000
MIN_EVAL_IMAGES = 50


def _read_cache(cache_path: Path) -> dict:
    with np.load(cache_path) as z:
        return {"features": z["features"], "logits": z["logits"],
                "labels": z["labels"]}


def standardize_with_discovery_stats(features: np.ndarray, is_disc: np.ndarray):
    """Standardization statistics from the discovery half only (freeze F1),
    applied to all images."""
    f = np.asarray(features, dtype=np.float64)
    mean = f[is_disc].mean(axis=0)
    std = f[is_disc].std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    return (f - mean) / std_safe, mean, std_safe


def route_eval_to_clusters(h_disc_std: np.ndarray, disc_cluster: np.ndarray,
                           h_eval_std: np.ndarray) -> np.ndarray:
    """Nearest-neighbor transfer (spec fallback, frozen in F1): evaluation
    representation assigned the discovery cluster of its nearest discovery
    point in standardized 512-d space. Deterministic."""
    from sklearn.neighbors import NearestNeighbors

    nn = NearestNeighbors(n_neighbors=1, algorithm="brute").fit(h_disc_std)
    _, idx = nn.kneighbors(h_eval_std)
    return disc_cluster[idx[:, 0]]


def run_clustering(store: ResultStore, cfg: dict, cache_path: Path,
                   n_jobs: int = 8):
    """PHATE-10d + k-means(k=100) on discovery representations; evaluation
    images routed by nearest discovery neighbor in standardized 512-d space.

    Returns (cluster_of_image (50k,), info dict)."""
    import phate
    from sklearn.cluster import KMeans

    cond = {"stage": "clusters", "k": cfg["n_clusters"],
            "phate_seed": PHATE_SEED, "kmeans_seed": KMEANS_SEED}
    cache = _read_cache(cache_path)
    labels = cache["labels"]
    features = cache["features"]
    is_disc, is_eval = _split_from_cache_labels(labels)
    disc_idx = np.flatnonzero(is_disc)
    eval_idx = np.flatnonzero(is_eval)

    H_std_all, mean, std = standardize_with_discovery_stats(features, is_disc)
    Hs_disc = H_std_all[is_disc]

    t0 = time.time()
    ph = phate.PHATE(n_components=10, random_state=PHATE_SEED, n_pca=100,
                     verbose=False, n_jobs=n_jobs)
    emb = ph.fit_transform(Hs_disc)
    t_phate = time.time() - t0

    km = KMeans(n_clusters=cfg["n_clusters"], n_init=50,
                random_state=KMEANS_SEED)
    disc_cluster = km.fit_predict(emb)
    t_kmeans = time.time() - t0 - t_phate

    eval_cluster = route_eval_to_clusters(Hs_disc, disc_cluster,
                                          H_std_all[is_eval])

    rows = []
    for i, ci in enumerate(disc_idx):
        rows.append({"image_index": int(ci), "cluster": int(disc_cluster[i]),
                     "true_class": int(labels[ci]),
                     "split": "discovery",
                     "assignment_method": "phate10d_kmeans_fit"})
    for i, ci in enumerate(eval_idx):
        rows.append({"image_index": int(ci), "cluster": int(eval_cluster[i]),
                     "true_class": int(labels[ci]),
                     "split": "evaluation",
                     "assignment_method": ASSIGNMENT_METHOD})
    store.write(cond, "clusters", rows)

    # composition + size rows (F2: actual per-cluster sizes, written before
    # any systematic table is computed)
    comp_rows, size_rows = [], []
    for c in range(cfg["n_clusters"]):
        d_members = disc_idx[disc_cluster == c]
        e_members = eval_idx[eval_cluster == c]
        d_counts = np.bincount(labels[d_members], minlength=1000)
        e_counts = np.bincount(labels[e_members], minlength=1000)
        comp_rows.append({
            "cluster": c,
            "n_discovery": int(len(d_members)),
            "n_evaluation": int(len(e_members)),
            "low_sample_flag": bool(len(e_members) < MIN_EVAL_IMAGES),
            "n_disc_classes": int((d_counts > 0).sum()),
            "n_eval_classes": int((e_counts > 0).sum()),
            "discovery_class_counts_json": json.dumps(
                {int(k): int(v) for k, v in enumerate(d_counts) if v > 0}),
            "evaluation_class_counts_json": json.dumps(
                {int(k): int(v) for k, v in enumerate(e_counts) if v > 0}),
            "largest_discovery_class": int(np.argmax(d_counts)),
            "largest_discovery_class_share":
                float(d_counts.max() / max(len(d_members), 1)),
        })
        size_rows.append({
            "cluster": c, "n_discovery": int(len(d_members)),
            "n_evaluation": int(len(e_members)),
        })
    store.write(cond, "cluster_composition", comp_rows)
    store.write(cond, "cluster_sizes", size_rows)

    np.savez_compressed(
        config.RAW_DIR / "imagenet" / "cluster_artifacts.npz",
        centroids=km.cluster_centers_, embedding=emb.astype(np.float32),
        disc_cluster=disc_cluster.astype(np.int16),
        eval_cluster=eval_cluster.astype(np.int16),
        phate_runtime_s=t_phate, kmeans_runtime_s=t_kmeans)
    dist = size_distribution_report(size_rows)
    info = {"n_discovery": int(is_disc.sum()), "n_evaluation": int(is_eval.sum()),
            "phate_runtime_s": t_phate, "kmeans_runtime_s": t_kmeans,
            "assignment_method": ASSIGNMENT_METHOD,
            "low_sample_clusters": [r["cluster"] for r in comp_rows
                                    if r["low_sample_flag"]]}
    return ClusteringResult(assignments=rows, cluster_size_report=dist,
                            info=info)


def _split_from_cache_labels(labels: np.ndarray):
    from data import imagenet_split
    return imagenet_split(labels)


class ClusteringResult:
    """Bundle: assignment rows, cluster-size report, run info."""

    def __init__(self, assignments, cluster_size_report, info):
        self.assignments = assignments
        self.cluster_size_report = cluster_size_report
        self.info = info


def _assemble(assignments, cluster_size_report, info):
    return ClusteringResult(assignments, cluster_size_report, info)


def size_distribution_report(size_rows: list[dict]) -> dict:
    """F2: actual per-cluster size distribution (before systematic tables)."""
    df = pd.DataFrame(size_rows)
    out = {}
    for col, name in (("n_discovery", "discovery"), ("n_evaluation", "evaluation")):
        v = np.sort(df[col].to_numpy())
        deciles = np.percentile(v, np.arange(0, 101, 10), method="lower")
        out[name] = {
            "n_clusters": int(len(v)),
            "min": int(v.min()),
            "max": int(v.max()),
            "mean": float(v.mean()),
            "deciles": {f"p{int(p)}": int(d) for p, d in
                        zip(np.arange(0, 101, 10), deciles)},
            "n_below_50": int((df[col].to_numpy() < MIN_EVAL_IMAGES).sum())
            if col == "n_evaluation" else None,
        }
    return out


# ---------------------------------------------------------------------------
# Evaluation cohorts (identical for a factor and all of its controls)
# ---------------------------------------------------------------------------

def even_stratified_sample(pool: np.ndarray, labels: np.ndarray, n: int,
                           rng: np.random.Generator) -> np.ndarray:
    """Deterministic stratified round-robin sample, as even as possible
    across the true classes present in `pool`."""
    by_class = []
    for c in np.unique(labels[pool]):  # ascending class ids (deterministic)
        members = pool[labels[pool] == c].copy()
        rng.shuffle(members)
        by_class.append(members)
    out = []
    i = 0
    while len(out) < n:
        added = False
        for members in by_class:
            if i < len(members):
                out.append(members[i])
                added = True
                if len(out) >= n:
                    break
        if not added:
            break
        i += 1
    return np.asarray(out[:n], dtype=np.int64)


def cohort_for_cluster(cluster_of_image: np.ndarray, eval_idx: np.ndarray,
                       labels: np.ndarray,
                       cluster: int) -> tuple[np.ndarray, np.ndarray, bool]:
    """Pure per-cluster cohort builder (unit-testable; deterministic; free of
    any factor information).

    In-cluster: all evaluation images routed to the cluster unless more than
    MAX_EVAL_IMAGES, in which case a deterministic stratified subsample of
    MAX_EVAL_IMAGES. Off-cluster: an equally sized stratified sample from
    evaluation images routed to other clusters.
    Returns (eval_imgs, is_source_mask, low_sample_flag)."""
    in_cluster = eval_idx[cluster_of_image[eval_idx] == cluster]
    off_pool = eval_idx[(cluster_of_image[eval_idx] != cluster)]
    is_low = len(in_cluster) < MIN_EVAL_IMAGES
    if len(in_cluster) > MAX_EVAL_IMAGES:
        rng = np.random.default_rng(seeds.derive_seed(cluster, "eval_incluster"))
        in_cluster = even_stratified_sample(in_cluster, labels,
                                            MAX_EVAL_IMAGES, rng)
    rng = np.random.default_rng(seeds.derive_seed(cluster, "eval_offcluster"))
    n_off = min(len(in_cluster), len(off_pool))
    off = even_stratified_sample(off_pool, labels, n_off, rng)
    eval_imgs = np.concatenate([in_cluster, off]).astype(np.int64)
    is_source = np.zeros(len(eval_imgs), dtype=bool)
    is_source[: len(in_cluster)] = True
    return eval_imgs, is_source, bool(is_low)


def build_eval_cohorts(store: ResultStore, cfg: dict, cache_path: Path):
    """Per-cluster evaluation cohorts, built once and reused by every factor
    and every control in that cluster (freeze F3 / spec: identical cohorts).

    In-cluster: all evaluation images routed to the cluster unless more than
    MAX_EVAL_IMAGES, in which case a deterministic stratified subsample of
    1000. Off-cluster: an equally sized stratified sample from evaluation
    images routed to other clusters.

    Writes an npz cohort cache and a cohort-metadata shard; returns
    (cohorts: dict cluster -> (eval_ids, is_source_mask), low_sample set).
    """
    cache = _read_cache(cache_path)
    labels = cache["labels"]
    cluster_of_image = load_cluster_assignments(store, cfg)
    from data import imagenet_split
    is_disc, _ = imagenet_split(labels)
    eval_idx = np.flatnonzero(~is_disc)
    cohorts = {}
    low_sample = []
    cohort_meta_rows = []
    for k in range(cfg["n_clusters"]):
        eval_imgs, is_source, is_low = cohort_for_cluster(
            cluster_of_image, eval_idx, labels, k)
        cohorts[k] = (eval_imgs, is_source)
        if is_low:
            low_sample.append(k)
        cohort_meta_rows.append({
            "cluster": k, "n_eval_images": int(len(eval_imgs)),
            "n_source": int(is_source.sum()), "n_off": int((~is_source).sum()),
            "low_sample": bool(is_low)})
    np.savez_compressed(
        config.RAW_DIR / "imagenet" / "eval_cohorts.npz",
        **{f"cluster_{k:03d}_images": v[0] for k, v in cohorts.items()},
        **{f"cluster_{k:03d}_source": v[1] for k, v in cohorts.items()})
    store.write({"stage": "eval_cohorts", "phate_seed": PHATE_SEED,
                 "kmeans_seed": KMEANS_SEED, "k": cfg["n_clusters"]},
                "cohort_meta", cohort_meta_rows)
    return cohorts, set(low_sample), cohort_meta_rows


def load_cluster_assignments(store: ResultStore, cfg: dict) -> np.ndarray:
    cond = {"stage": "clusters", "k": cfg["n_clusters"],
            "phate_seed": PHATE_SEED, "kmeans_seed": KMEANS_SEED}
    z = pd.read_parquet(store.shard_path(cond, "clusters"))
    clusters = np.zeros(50_000, dtype=int)
    clusters[z["image_index"].to_numpy()] = z["cluster"].to_numpy()
    return clusters


def cluster_size_distribution_report(size_rows: list[dict]) -> dict:
    return size_distribution_report(size_rows)
