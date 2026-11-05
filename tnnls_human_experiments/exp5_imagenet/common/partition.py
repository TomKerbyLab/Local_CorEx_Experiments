"""Partition methods: PHATE+k-means, raw-space k-means, PCA+k-means, random.

All partitioners return estimated cluster labels (0..k-1) plus runtime, and
share a uniform signature so runners can treat them interchangeably.
"""

from __future__ import annotations

import time

import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

import phate


def phate_kmeans_partition(
    X: np.ndarray,
    k: int,
    seed: int,
    phate_dim: int = 10,
    n_init: int = 50,
    n_pca: int | None = 100,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """PHATE embedding + k-means. Returns (labels, info dict)."""
    X = np.asarray(X, dtype=float)
    t0 = time.time()
    ph = phate.PHATE(
        n_components=phate_dim,
        random_state=seed,
        n_pca=n_pca,
        verbose=verbose,
    )
    emb = ph.fit_transform(X)
    t_phate = time.time() - t0
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = km.fit_predict(emb)
    info = {
        "phate_runtime_s": t_phate,
        "kmeans_runtime_s": time.time() - t0 - t_phate,
        "embedding": emb,
        "kmeans_centroids": km.cluster_centers_,
        "phate_model": ph,
        "kmeans_model": km,
    }
    return labels, info


def raw_kmeans_partition(
    X: np.ndarray, k: int, seed: int, n_init: int = 50
) -> tuple[np.ndarray, dict]:
    X = np.asarray(X, dtype=float)
    t0 = time.time()
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = km.fit_predict(X)
    return labels, {"runtime_s": time.time() - t0, "kmeans_model": km}


def pca_kmeans_partition(
    X: np.ndarray,
    k: int,
    seed: int,
    pca_dim: int = 10,
    n_init: int = 50,
) -> tuple[np.ndarray, dict]:
    X = np.asarray(X, dtype=float)
    t0 = time.time()
    pca = PCA(n_components=pca_dim, random_state=seed)
    Z = pca.fit_transform(X)
    km = KMeans(n_clusters=k, n_init=n_init, random_state=seed)
    labels = km.fit_predict(Z)
    return labels, {
        "runtime_s": time.time() - t0,
        "pca_model": pca,
        "kmeans_model": km,
        "explained_variance_ratio": pca.explained_variance_ratio_,
    }


def balanced_random_partition(
    labels_template: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, dict]:
    """Balanced random partition with the same group sizes as the template.

    Cluster sizes are taken from the template, but memberships are assigned
    randomly (a random subset of the requested size per cluster).
    """
    labels_template = np.asarray(labels_template)
    sizes = [int(np.sum(labels_template == c)) for c in np.unique(labels_template)]
    return balanced_random_partition_from_sizes(len(labels_template), sizes, seed)


def balanced_random_partition_from_sizes(
    n: int, sizes: list[int], seed: int
) -> tuple[np.ndarray, dict]:
    """Balanced random partition of n items into clusters with given sizes."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    labels = np.zeros(n, dtype=int)
    start = 0
    for c, size in enumerate(sizes):
        labels[perm[start : start + size]] = c
        start += size
    assert start == n
    return labels, {"runtime_s": 0.0}