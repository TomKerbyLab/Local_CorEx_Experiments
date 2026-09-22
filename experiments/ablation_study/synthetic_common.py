from __future__ import annotations

from dataclasses import dataclass
from time import time
from typing import Dict, List, Literal, Optional

import numpy as np
from scipy.spatial.distance import cosine
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, precision_recall_curve, auc

import phate
from local_corex import BioCorex, LinearCorex


ModelKind = Literal["linear", "bio"]
PartitionMethod = Literal["oracle", "phate", "raw_kmeans", "random"]


def rng_for(rep: int, *tags: object) -> np.random.Generator:
    """
    Deterministic RNG keyed by rep and arbitrary tags.
    """
    seed = 10_000 + rep
    for t in tags:
        seed = (seed * 1315423911 + hash(t)) % (2**32 - 1)
    return np.random.default_rng(seed)


def finite_mean(values) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    return float(arr.mean())


def weighted_nanmean(values, weights) -> float:
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = np.isfinite(v) & np.isfinite(w)
    if not np.any(mask):
        return np.nan
    v = v[mask]
    w = w[mask]
    if np.sum(w) == 0:
        return np.nan
    return float(np.sum(v * w) / np.sum(w))


def pad_group_masks(groups: List[np.ndarray], n_total: int) -> List[np.ndarray]:
    padded = []
    for g in groups:
        g = np.asarray(g, dtype=float).ravel()
        if g.shape[0] > n_total:
            raise ValueError(f"Group mask has length {g.shape[0]} but n_total is {n_total}.")
        if g.shape[0] < n_total:
            g = np.concatenate([g, np.zeros(n_total - g.shape[0])])
        padded.append(g)
    return padded


def align_binary_partition(y_true: np.ndarray, y_pred: np.ndarray, positive_label: str) -> np.ndarray:
    """
    Align predicted binary labels so that class 0 corresponds to positive_label
    whenever that improves agreement.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    left_match = np.sum((y_true == positive_label) & (y_pred == 0))
    right_match = np.sum((y_true == positive_label) & (y_pred == 1))

    if right_match > left_match:
        return 1 - y_pred
    return y_pred


def safe_cosine(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return np.nan

    return cosine(a, b)


def safe_auc_prc(y_true, y_score):
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score, dtype=float)

    n_pos = int(y_true.sum())
    if n_pos == 0 or n_pos == len(y_true):
        return np.nan

    prec, rec, _ = precision_recall_curve(y_true, y_score)
    return auc(rec, prec)


def get_learned_mi_matrix(corex_model, linear: bool = True) -> np.ndarray:
    if linear:
        moments = getattr(corex_model, "moments", None)
        if moments is None or "MI" not in moments:
            raise AttributeError("Linear Corex model does not expose moments['MI'].")
        learned = moments["MI"]
    else:
        learned = getattr(corex_model, "mis", None)
        if learned is None:
            raise AttributeError("Bio Corex model does not expose mis.")

    learned = np.asarray(learned, dtype=float)
    if learned.ndim == 1:
        learned = learned[None, :]
    return learned


def fit_corex(X: np.ndarray, model_kind: ModelKind, n_hidden: int, seed: int = 42):
    st = time()
    if model_kind == "linear":
        model = LinearCorex(n_hidden=n_hidden, seed=seed, gaussianize="outliers")
    elif model_kind == "bio":
        model = BioCorex(
            n_hidden=n_hidden,
            dim_hidden=2,
            marginal_description="gaussian",
            smooth_marginals=False,
        )
    else:
        raise ValueError(f"Unknown model_kind: {model_kind}")

    _ = model.fit_transform(X)
    et = time()
    return model, et - st


def score_group_coverage(
    groups: List[np.ndarray],
    corex_model,
    linear: bool = True,
    verbose: bool = False,
    plot: bool = False,
    dist: str = "cosine",
) -> float:
    if dist == "auc_prc":
        score_fn = safe_auc_prc
    else:
        score_fn = safe_cosine

    learned = get_learned_mi_matrix(corex_model, linear=linear)
    scores = []

    for group in groups:
        group = np.asarray(group, dtype=float).ravel()
        if np.linalg.norm(group) == 0:
            scores.append(np.nan)
            continue

        best_score = -np.inf if dist == "auc_prc" else np.inf

        for lf in learned:
            lf = np.asarray(lf, dtype=float).ravel()
            if np.linalg.norm(lf) == 0:
                continue

            score = score_fn(group, lf)
            if np.isnan(score):
                continue

            if dist == "auc_prc":
                if score > best_score:
                    best_score = score
            else:
                if score < best_score:
                    best_score = score

        scores.append(best_score if np.isfinite(best_score) else np.nan)

    return finite_mean(scores)


def score_lf_quality(
    groups: List[np.ndarray],
    corex_model,
    linear: bool = True,
    verbose: bool = False,
    plot: bool = False,
    dist: str = "cosine",
) -> List[float]:
    if dist == "auc_prc":
        score_fn = safe_auc_prc
    else:
        score_fn = safe_cosine

    learned = get_learned_mi_matrix(corex_model, linear=linear)
    scores = []

    for lf in learned:
        lf = np.asarray(lf, dtype=float).ravel()
        if np.linalg.norm(lf) == 0:
            scores.append(np.nan)
            continue

        best_score = -np.inf if dist == "auc_prc" else np.inf

        for group in groups:
            group = np.asarray(group, dtype=float).ravel()
            if np.linalg.norm(group) == 0:
                continue

            score = score_fn(group, lf)
            if np.isnan(score):
                continue

            if dist == "auc_prc":
                if score > best_score:
                    best_score = score
            else:
                if score < best_score:
                    best_score = score

        scores.append(best_score if np.isfinite(best_score) else np.nan)

    return scores


def score_model_against_groups(
    groups: List[np.ndarray],
    corex_model,
    linear: bool = True,
) -> Dict[str, float]:
    group_cosine = score_group_coverage(groups, corex_model, linear=linear, dist="cosine")
    group_auc = score_group_coverage(groups, corex_model, linear=linear, dist="auc_prc")
    lf_cosine = score_lf_quality(groups, corex_model, linear=linear, dist="cosine")
    lf_auc = score_lf_quality(groups, corex_model, linear=linear, dist="auc_prc")

    return {
        "ave_group_cosine": float(group_cosine),
        "ave_auc_group_prc": float(group_auc),
        "ave_lf_cosine": float(finite_mean(lf_cosine)),
        "ave_auc_lf_prc": float(finite_mean(lf_auc)),
    }


def partition_data(
    X: np.ndarray,
    y_true: np.ndarray,
    positive_label: str,
    method: PartitionMethod,
    rep: int,
    n_components: int = 5,
) -> Dict[str, object]:
    st = time()

    if method == "oracle":
        y_pred = np.where(np.asarray(y_true) == positive_label, 0, 1)

    elif method == "random":
        y_pred = rng_for(rep, "random_partition").integers(low=0, high=2, size=X.shape[0])

    else:
        if method == "phate":
            emb = phate.PHATE(
                n_components=n_components,
                n_jobs=4,
                random_state=42,
            ).fit_transform(X)
        elif method == "raw_kmeans":
            emb = X
        else:
            raise ValueError(f"Unknown partition method: {method}")

        km = KMeans(n_clusters=2, random_state=42, n_init="auto")
        y_pred = km.fit_predict(emb)

    et = time()
    y_pred_aligned = align_binary_partition(y_true, y_pred, positive_label=positive_label)
    ari = adjusted_rand_score(y_true, y_pred)

    return {
        "y_pred": y_pred,
        "y_pred_aligned": y_pred_aligned,
        "ari": float(ari),
        "partition_time": float(et - st),
    }


def evaluate_global_setting(
    X: np.ndarray,
    all_groups: List[np.ndarray],
    model_kind: ModelKind,
    n_hidden: int,
) -> Dict[str, object]:
    model, corex_time = fit_corex(X, model_kind=model_kind, n_hidden=n_hidden)
    scores = score_model_against_groups(all_groups, model, linear=(model_kind == "linear"))

    return {
        "corex_time": float(corex_time),
        "total_time": float(corex_time),
        **scores,
    }


def evaluate_local_from_partition(
    X: np.ndarray,
    y_true: np.ndarray,
    true_groups: Dict[str, List[np.ndarray]],
    partition_result: Dict[str, object],
    cluster_names: List[str],
    model_kind: ModelKind,
    n_hidden: int,
) -> Dict[str, object]:
    y_pred_aligned = np.asarray(partition_result["y_pred_aligned"])
    partition_time = float(partition_result["partition_time"])
    partition_ari = float(partition_result["ari"])

    cluster_rows = []
    for cluster_id, cluster_name in enumerate(cluster_names):
        idx = y_pred_aligned == cluster_id
        n_cluster = int(np.sum(idx))
        if n_cluster == 0:
            continue

        X_sub = X[idx]
        model, corex_time = fit_corex(X_sub, model_kind=model_kind, n_hidden=n_hidden)
        scores = score_model_against_groups(true_groups[cluster_name], model, linear=(model_kind == "linear"))

        cluster_rows.append(
            {
                "cluster_name": cluster_name,
                "cluster_size": n_cluster,
                "cluster_purity": float(np.mean(np.asarray(y_true)[idx] == cluster_name)),
                "corex_time": float(corex_time),
                **scores,
            }
        )

    if not cluster_rows:
        return {
            "partition_ari": partition_ari,
            "partition_time": partition_time,
            "corex_time": np.nan,
            "total_time": np.nan,
            "cluster_purity": np.nan,
            "ave_group_cosine": np.nan,
            "ave_auc_group_prc": np.nan,
            "ave_lf_cosine": np.nan,
            "ave_auc_lf_prc": np.nan,
        }

    df = np.asarray([r["cluster_size"] for r in cluster_rows], dtype=float)
    weights = df / df.sum()

    row = {
        "partition_ari": partition_ari,
        "partition_time": partition_time,
        "corex_time": float(np.sum([r["corex_time"] for r in cluster_rows])),
        "total_time": float(np.sum([r["corex_time"] for r in cluster_rows]) + partition_time),
        "cluster_purity": weighted_nanmean([r["cluster_purity"] for r in cluster_rows], weights),
        "ave_group_cosine": weighted_nanmean([r["ave_group_cosine"] for r in cluster_rows], weights),
        "ave_auc_group_prc": weighted_nanmean([r["ave_auc_group_prc"] for r in cluster_rows], weights),
        "ave_lf_cosine": weighted_nanmean([r["ave_lf_cosine"] for r in cluster_rows], weights),
        "ave_auc_lf_prc": weighted_nanmean([r["ave_auc_lf_prc"] for r in cluster_rows], weights),
    }
    return row