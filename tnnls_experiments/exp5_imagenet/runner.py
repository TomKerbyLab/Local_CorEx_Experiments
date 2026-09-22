"""Experiment 5 runner: ResNet-18 ImageNet validation.

Stages:
  1. split        per-class 25/25 discovery/evaluation split (row order)
  2. verify       cached features vs fresh forward passes
  3. clusters     PHATE-10d + k-means k=100 on discovery representations;
                  evaluation images assigned to nearest centroid in
                  standardized 512-d space (recorded method)
  4. fits         Linear CorEx per discovery cluster on [H | L], M=10,
                  top-3 factors by TC
  5. ablations    q=0.8 supports among the 512 pooled units; sparse-update
                  logits l^A = l - W[:,S] h_S; 5 matched random controls +
                  univariate control; evaluation on in-cluster + matched
                  off-cluster images
  6. case study   African-grey cluster (deterministic discovery-only rule)
                  and DeepDream objective config

The convolutional backbone is never rerun during ablations; logits come from
the cached features and the final-layer weight matrix.
"""

from __future__ import annotations

import json
import logging
import time

import numpy as np
import pandas as pd

from ..common import config, corex, metrics, seeds
from ..common.io import FailureLog, write_run_report
from ..common.runtime import setup_logging
from ..common.store import ResultStore
from . import data as exp5_data

EXPERIMENT = "imagenet"
logger = logging.getLogger(f"tnnls.{EXPERIMENT}")

PHATE_SEED = 7000
KMEANS_SEED = 7001
N_CLUSTERS = 100
N_FACTORS = 10
N_RETAINED = 3
PRIMARY_Q = 0.80
N_RANDOM_CONTROLS = 5
MIN_EVAL_IMAGES = 50
MAX_EVAL_IMAGES = 1000
MIN_CLASSWISE_OBS = 5
TOP_K_SHIFTS = 20

DEFAULTS = {
    "n_clusters": N_CLUSTERS,
    "n_factors": N_FACTORS,
    "n_retained": N_RETAINED,
    "primary_q": PRIMARY_Q,
    "n_random_controls": N_RANDOM_CONTROLS,
    "extra_thresholds": [0.60, 0.70, 0.90],  # case study only
}


# ---------------------------------------------------------------------------
# Stages 1-2: split + cache verification
# ---------------------------------------------------------------------------

def run_split_and_verify(store: ResultStore, n_check: int = 256) -> dict:
    cond = {"stage": "split_verify", "n_check": n_check}
    if store.is_done(cond, ("split",)):
        z = store.read_all("split")
        return {"verified": True, "rows": len(z)}
    cache = exp5_data.load_cached()
    labels = cache["labels"]
    is_disc, is_eval = exp5_data.imagenet_split(labels)
    report = exp5_data.verify_cache(n_check=n_check)
    if not report["verified"]:
        raise RuntimeError(f"cache verification failed: {report}")
    # per-class integrity: exactly 25/25
    for c in range(1000):
        d = is_disc[labels == c].sum()
        e = is_eval[labels == c].sum()
        assert d == 25 and e == 25, f"class {c}: {d}/{e}"
    rows = [{"image_index": i, "true_class": int(labels[i]),
             "split": "discovery" if is_disc[i] else "evaluation",
             "integrity_ok": True}
            for i in range(len(labels))]
    store.write(cond, "split", rows)
    store.write(cond, "verify", [report])
    return {"verified": True, "rows": len(rows), "report": report}


# ---------------------------------------------------------------------------
# Stage 3: clustering
# ---------------------------------------------------------------------------

def run_clustering(store: ResultStore, cfg: dict) -> dict:
    """PHATE on the 25k discovery features + k-means k=100; evaluation images
    assigned to the nearest k-means centroid in standardized feature space."""
    cond = {"stage": "clusters", "k": cfg["n_clusters"],
            "phate_seed": PHATE_SEED, "kmeans_seed": KMEANS_SEED}
    if store.is_done(cond, ("clusters",)):
        return {"cached": True}

    import phate
    from sklearn.cluster import KMeans

    cache = exp5_data.load_cached()
    labels = cache["labels"]
    features = cache["features"]
    is_disc, is_eval = exp5_data.imagenet_split(labels)

    H_disc = features[is_disc]
    H_std_all, mean, std = _standardize_global(features)
    Hs_disc = H_std_all[is_disc]

    t0 = time.time()
    ph = phate.PHATE(n_components=10, random_state=PHATE_SEED, n_pca=100,
                     verbose=False, n_jobs=8)
    emb = ph.fit_transform(Hs_disc)
    t_phate = time.time() - t0

    km = KMeans(n_clusters=cfg["n_clusters"], n_init=50,
                random_state=KMEANS_SEED)
    disc_cluster = km.fit_predict(emb)

    # evaluation assignment: nearest centroid in standardized 512-d space
    eval_idx = np.flatnonzero(is_eval)
    Hs_eval = H_std_all[is_eval]
    # standardized-space centroids: mean/std of discovery members per cluster
    centroids = np.zeros((cfg["n_clusters"], Hs_disc.shape[1]))
    for c in range(cfg["n_clusters"]):
        members = Hs_disc[disc_cluster == c]
        if len(members):
            centroids[c] = members.mean(axis=0)
    dists = ((Hs_eval[:, None, :] - centroids[None, :, :]) ** 2).sum(-1)
    eval_cluster = dists.argmin(axis=1)

    rows = []
    disc_idx = np.flatnonzero(is_disc)
    for i, ci in enumerate(disc_idx):
        rows.append({"image_index": int(ci), "cluster": int(disc_cluster[i]),
                     "split": "discovery"})
    for i, ci in enumerate(eval_idx):
        rows.append({"image_index": int(ci), "cluster": int(eval_cluster[i]),
                     "split": "evaluation", "assignment_method": "nearest_centroid_std512"})
    store.write(cond, "clusters", rows)
    # save centroids + embedding for figures
    np.savez_compressed(
        config.RAW_DIR / EXPERIMENT / "cluster_artifacts.npz",
        centroids=centroids, embedding=emb,
        disc_cluster=disc_cluster, eval_cluster=eval_cluster,
        phate_runtime_s=t_phate)
    return {"n_disc": int(is_disc.sum()), "n_eval": int(is_eval.sum()),
            "phate_runtime_s": t_phate}


def _standardize_global(features: np.ndarray):
    mean = features.mean(axis=0)
    std = features.std(axis=0)
    std_safe = np.where(std < 1e-12, 1.0, std)
    return (features - mean) / std_safe, mean, std


# ---------------------------------------------------------------------------
# Stage 4: CorEx fits
# ---------------------------------------------------------------------------

def run_fits(store: ResultStore, cfg: dict, clusters: dict | None = None) -> dict:
    """Linear CorEx per discovery cluster on [H | L]; top-3 factors by TC."""
    if clusters is None:
        clusters = _load_clusters(store, cfg)
    cache = exp5_data.load_cached()
    labels, features, logits = cache["labels"], cache["features"], cache["preds"]
    is_disc, _ = exp5_data.imagenet_split(labels)
    disc_idx = np.flatnonzero(is_disc)
    cluster_of_image = clusters  # image_index -> cluster

    n_done = 0
    for k in range(cfg["n_clusters"]):
        cond = {"stage": "fit", "cluster": k}
        if store.is_done(cond, ("factors", "supports")):
            n_done += 1
            continue
        members = disc_idx[cluster_of_image[disc_idx] == k]
        if len(members) < 10:
            logger.warning("cluster %d has %d discovery images; skipping", k,
                           len(members))
            continue
        H = features[members]
        L = logits[members]
        H_std, valid_h, _ = corex.standardize_columns(H)
        L_std, valid_l, _ = corex.standardize_columns(L)
        s_fit = seeds.derive_seed(k, "corex")
        V = np.hstack([H_std, L_std])
        fit = corex.fit_linear_corex(V, cfg["n_factors"], seed=s_fit)
        order = np.argsort(-fit.tcs)[: cfg["n_retained"]]
        mis = fit.association_scores()
        n_h = H.shape[1]
        factor_rows, support_rows = [], []
        for rank, j in enumerate(order):
            lmis = mis[j, n_h:]
            dominant = int(np.argmax(lmis))
            factor_rows.append({
                "cluster": k, "factor_index": int(j), "factor_rank": rank,
                "factor_tc": float(fit.tcs[j]),
                "corex_seed": s_fit, "corex_converged": fit.converged,
                "dominant_logit": dominant,
                "logit_mi_gap": float(np.sort(lmis)[-1] - np.sort(lmis)[-2]),
                "n_valid_units": int(valid_h.sum()),
                "corex_runtime_s": fit.runtime_s,
            })
            umis = mis[j, :n_h]
            sup = metrics.cumulative_mi_support(umis, cfg["primary_q"], valid_h)
            for r, (unit, score) in enumerate(zip(sup.indices, sup.scores)):
                support_rows.append({
                    "cluster": k, "factor_index": int(j), "threshold":
                    cfg["primary_q"], "mi_rank": r, "unit": int(unit),
                    "mi_score": float(score), "support_size": sup.size,
                    "support_density": sup.density,
                })
        store.write(cond, "factors", factor_rows)
        store.write(cond, "supports", support_rows)
        n_done += 1
        if n_done % 10 == 0:
            logger.info("fits: %d/%d clusters", n_done, cfg["n_clusters"])
    return {"clusters_fitted": n_done}


def _load_clusters(store: ResultStore, cfg: dict) -> np.ndarray:
    cond = {"stage": "clusters", "k": cfg["n_clusters"],
            "phate_seed": PHATE_SEED, "kmeans_seed": KMEANS_SEED}
    z = pd.read_parquet(store.shard_path(cond, "clusters"))
    clusters = np.zeros(50_000, dtype=int)
    clusters[z["image_index"].to_numpy()] = z["cluster"].to_numpy()
    return clusters


# ---------------------------------------------------------------------------
# Stage 5: ablations (sparse logit update)
# ---------------------------------------------------------------------------

def load_final_layer(device: str = "cuda"):
    """ResNet-18 final-layer W (1000x512), b (1000), cached to disk once."""
    import torch

    path = config.CHECKPOINTS_DIR / "resnet18_fc.pt"
    if path.exists():
        d = torch.load(path, map_location=device)
        return d["W"].to(device), d["b"].to(device)
    m = exp5_data.resnet18_model(device)
    W = m.fc.weight.detach().clone()
    b = m.fc.bias.detach().clone()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"W": W.cpu(), "b": b.cpu()}, path)
    return W.to(device), b.to(device)


def ablated_logits_sparse(logits: np.ndarray, H: np.ndarray, W: np.ndarray,
                          support: list[int]) -> np.ndarray:
    """l^A = l - W[:, S] @ h_S (sparse update; unit-tested vs full pass)."""
    import torch

    S = np.asarray(support, dtype=np.int64)
    if len(S) == 0:
        return logits
    Ws = W[:, torch.tensor(S, device=W.device)].cpu().numpy()
    return logits - H[:, S] @ Ws.T


def run_ablations(store: ResultStore, cfg: dict, device: str = "cuda") -> dict:
    import torch

    W, b = load_final_layer(device)
    cache = exp5_data.load_cached()
    labels = cache["labels"]
    features, logits = cache["features"], cache["preds"]
    is_disc, is_eval = exp5_data.imagenet_split(labels)
    clusters = _load_clusters(store, cfg)
    eval_idx = np.flatnonzero(is_eval)

    factors = store.read_all("factors")
    supports = store.read_all("supports")
    probs0 = _softmax_np(logits)

    summary_rows, shift_rows = [], []
    for (k, fi), grp in factors.groupby(["cluster", "factor_index"]):
        cond = {"stage": "ablation", "cluster": int(k), "factor_index": int(fi)}
        if store.is_done(cond, ("ablation_summary",)):
            continue
        sup = supports[(supports["cluster"] == k)
                       & (supports["factor_index"] == fi)]
        sup_list = sup.sort_values("mi_rank")["unit"].tolist()
        if not sup_list:
            continue
        frow = grp.iloc[0]
        dominant = int(frow["dominant_logit"])
        s = len(sup_list)

        # evaluation set: in-cluster + matched off-cluster control
        in_cluster = eval_idx[clusters[eval_idx] == k]
        low_sample = len(in_cluster) < MIN_EVAL_IMAGES
        if len(in_cluster) > MAX_EVAL_IMAGES:
            # deterministic stratified sample of 1000 by true class
            rng = np.random.default_rng(seeds.derive_seed(k, fi, "evalsample"))
            strata = []
            for c in np.unique(labels[in_cluster]):
                idx_c = in_cluster[labels[in_cluster] == c]
                take = max(1, int(round(MAX_EVAL_IMAGES * len(idx_c) / len(in_cluster))))
                strata.append(idx_c[:take])
            in_cluster = np.concatenate(strata)
        # off-cluster control: same size, stratified across other clusters' classes
        off_pool = eval_idx[clusters[eval_idx] != k]
        rng = np.random.default_rng(seeds.derive_seed(k, fi, "offcluster"))
        n_off = min(len(in_cluster), len(off_pool))
        off = _stratified_sample(off_pool, labels, n_off, rng)

        eval_imgs = np.concatenate([in_cluster, off])
        is_source = np.zeros(len(eval_imgs), dtype=bool)
        is_source[: len(in_cluster)] = True

        # interventions: LC, 5 random, univariate
        H_disc_cluster = features[np.flatnonzero(is_disc)[
            clusters[np.flatnonzero(is_disc)] == k]]
        L_disc_cluster = logits[np.flatnonzero(is_disc)[
            clusters[np.flatnonzero(is_disc)] == k]]
        valid_h = np.ones(features.shape[1], dtype=bool)
        uni_sup = _univariate_support(H_disc_cluster,
                                      L_disc_cluster[:, dominant], s)

        interventions = [("local_corex", 0, sup_list)]
        for draw in range(cfg["n_random_controls"]):
            s_rand = seeds.derive_seed(k, fi, "random", draw)
            rng2 = np.random.default_rng(s_rand)
            rs = rng2.choice(512, size=s, replace=False).tolist()
            interventions.append((f"random", draw, rs))
        interventions.append(("univariate", 0, uni_sup))

        factor_rows, factor_shift_rows = [], []
        for method, draw, support in interventions:
            p_abl = _softmax_np(ablated_logits_sparse(
                logits[eval_imgs], features[eval_imgs], W, support))
            row, top_rows = _summarize_imagenet_intervention(
                k, fi, method, draw, support, p_abl, probs0[eval_imgs],
                labels[eval_imgs], is_source, dominant, low_sample)
            factor_rows.append(row)
            factor_shift_rows.extend(top_rows)
        store.write(cond, "ablation_summary", factor_rows)
        store.write(cond, "top_shifts", factor_shift_rows)
        summary_rows.extend(factor_rows)
        shift_rows.extend(factor_shift_rows)

    return {"n_factor_ablations": len(summary_rows)}


def _stratified_sample(pool, labels, n, rng):
    """Stratify as evenly as possible across true classes."""
    by_class = [pool[labels[pool] == c] for c in np.unique(labels[pool])]
    rng.shuffle(by_class)
    out = []
    i = 0
    while len(out) < n:
        added = False
        for arr in by_class:
            if i < len(arr):
                out.append(arr[i])
                added = True
                if len(out) >= n:
                    break
        if not added:
            break
        i += 1
    return np.array(out[:n])


def _univariate_support(H, logit_col, size):
    if logit_col.std() < 1e-12 or H.shape[0] < 3:
        return list(range(size))
    Hc = H - H.mean(0)
    lc = logit_col - logit_col.mean()
    denom = np.sqrt((Hc ** 2).sum(0)) * np.sqrt((lc ** 2).sum()) + 1e-12
    corr = np.abs((Hc * lc[:, None]).sum(0) / denom)
    return np.argsort(-corr, kind="stable")[:size].tolist()


def _softmax_np(x):
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def _summarize_imagenet_intervention(
    k, fi, method, draw, support, p_abl, p_base, y, is_source, dominant,
    low_sample,
):
    """Image-level TV + classwise metrics + localization contrast."""
    tv = 0.5 * np.abs(p_abl - p_base).sum(1)
    tv_source = float(tv[is_source].mean())
    tv_off = float(tv[~is_source].mean())
    L_contrast = ((tv_source - tv_off) / (tv_source + tv_off + 1e-12))
    # true-class prob change
    dp_true = p_abl[np.arange(len(y)), y] - p_base[np.arange(len(y)), y]
    dp_dom = p_abl[:, dominant] - p_base[:, dominant]
    top1_0 = p_base.argmax(1)
    top1_a = p_abl.argmax(1)
    correct_0 = top1_0 == y
    correct_a = top1_a == y
    # classwise shifts for classes with >= MIN_CLASSWISE_OBS
    represented, T_list = [], []
    delta_rows = []
    for c in np.unique(y):
        m = y == c
        if m.sum() < MIN_CLASSWISE_OBS:
            continue
        delta = p_abl[m].mean(0) - p_base[m].mean(0)
        T_c = 0.5 * np.abs(delta).sum()
        represented.append(c)
        T_list.append(T_c)
        if method == "local_corex":
            # top 20 decreased/increased classes
            dec = np.argsort(delta)[:TOP_K_SHIFTS]
            inc = np.argsort(delta)[-TOP_K_SHIFTS:]
            for rank, cc in enumerate(dec):
                delta_rows.append({"cluster": k, "factor_index": fi,
                                   "method": method, "direction": "decrease",
                                   "rank": rank, "true_class": int(c),
                                   "output_class": int(cc),
                                   "shift": float(delta[cc])})
            for rank, cc in enumerate(inc[::-1]):
                delta_rows.append({"cluster": k, "factor_index": fi,
                                   "method": method, "direction": "increase",
                                   "rank": rank, "true_class": int(c),
                                   "output_class": int(cc),
                                   "shift": float(delta[cc])})
    T = np.array(T_list)
    if T.sum() > 0 and len(represented):
        r = T / T.sum()
        H_input = float(-(r * np.log(r, where=r > 0)).sum() / np.log(len(represented)))
        D_input = float(r.max())
    else:
        H_input = D_input = float("nan")
    row = {
        "cluster": k, "factor_index": fi, "method": method,
        "random_draw": draw, "support_size": len(support),
        "n_eval": int(len(y)), "n_source": int(is_source.sum()),
        "low_sample": bool(low_sample), "n_represented_classes": len(represented),
        "tv_source": tv_source, "tv_off": tv_off,
        "localization_contrast": float(L_contrast),
        "mean_dp_true_class": float(dp_true.mean()),
        "mean_dp_dominant_class": float(dp_dom.mean()),
        "top1_change": float(correct_a.mean() - correct_0.mean()),
        "H_input": H_input, "D_input": D_input,
        "dominant_logit": int(dominant),
    }
    # concordance: is c* among the 5 most decreased for the most affected class?
    if len(represented) and method == "local_corex":
        c_star_cls = int(represented[int(np.argmax(T))])
        m = y == c_star_cls
        delta = p_abl[m].mean(0) - p_base[m].mean(0)
        top5_dec = set(np.argsort(delta)[:5].tolist())
        row["dominant_in_top5_decreased"] = bool(dominant in top5_dec)
        y_star = int(represented[int(np.argmax(T))])
        row["most_affected_true_class"] = y_star
    return row, delta_rows


# ---------------------------------------------------------------------------
# Stage 6: African-grey case study (deterministic, discovery-only)
# ---------------------------------------------------------------------------

def select_african_grey_cluster(store: ResultStore, cfg: dict) -> dict:
    """Spec rule: among clusters with >=100 discovery images, the one with
    the largest proportion of African-grey images; ties by count, then lower
    cluster id. Uses discovery data only."""
    cache = exp5_data.load_cached()
    labels = cache["labels"]
    is_disc, _ = exp5_data.imagenet_split(labels)
    disc_idx = np.flatnonzero(is_disc)
    clusters = _load_clusters(store, cfg)
    best = None
    for k in range(cfg["n_clusters"]):
        members = disc_idx[clusters[disc_idx] == k]
        n = len(members)
        if n < 100:
            continue
        n_grey = int((labels[members] == exp5_data.AFRICAN_GREY_HF_IDX).sum())
        prop = n_grey / n
        key = (prop, n_grey, -k)
        if best is None or key > best[0]:
            best = (key, k, n, n_grey, prop)
    if best is None:
        raise RuntimeError("no cluster with >=100 discovery images")
    return {"cluster": best[1], "n_discovery": best[2],
            "n_african_grey": best[3], "proportion": best[4],
            "note": "cluster IDs are run-specific"}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg: dict, stages=("verify", "clusters", "fits", "ablations")) -> dict:
    defaults = {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS}}
    run_id = config.make_run_id(EXPERIMENT, defaults, 0)
    setup_logging(EXPERIMENT, run_id)
    if config.is_complete(run_id):
        return {"run_id": run_id, "status": "already_complete"}
    config.save_resolved_config(run_id, defaults, extra={"stages": list(stages)})
    store = ResultStore(EXPERIMENT)
    out = {}
    if "verify" in stages:
        out["split_verify"] = run_split_and_verify(store, int(cfg.get("n_check", 256)))
    if "clusters" in stages:
        out["clusters"] = run_clustering(store, defaults)
    if "fits" in stages:
        out["fits"] = run_fits(store, defaults)
    if "ablations" in stages:
        out["ablations"] = run_ablations(store, defaults)
    if "case_study" in stages:
        out["case_study"] = select_african_grey_cluster(store, defaults)
    config.mark_complete(run_id, {k: str(v) for k, v in out.items()})
    write_run_report(EXPERIMENT, run_id, 1, 0, {"stages": ",".join(stages)})
    return {"run_id": run_id, **out}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--stages", type=str, default="verify,clusters,fits,ablations")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = config.load_config(args.config) if args.config else {}
    if args.smoke:
        cfg = {"n_clusters": 5, "n_factors": 6, "n_retained": 2,
               "n_random_controls": 2, "n_check": 64}
    main(cfg, stages=tuple(args.stages.split(",")))