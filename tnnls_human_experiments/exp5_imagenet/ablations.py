"""Experiment 5 stage 5: exact pooled-feature ablation survey.

For each retained support S_j the ablated logits are computed by the exact
sparse update (spec, unit-tested against a full final-layer forward pass):

    l^A(x) = l(x) - W[:, S] h_S(x)

No bias change, no BN re-estimation, no backbone rerun, no evaluation data
anywhere in support selection, no DeepDream involvement (DeepDream is out of
scope for this quantitative port).

Interventions per factor: the Local CorEx support, 5 matched random supports
(seeds derived from the complete factor key), and a univariate
logit-association support of equal size (evaluated for every factor, so the
spec's LC-vs-univariate summary covers the whole survey and F3 is satisfied
for the case-study cluster). All interventions are evaluated on the
identical per-cluster cohort, and every row carries the source/off-cluster
contrast L_j so localization is always control-relative (F3).

Per-cell full arrays (per-image metrics, classwise mean-shift vectors) are
preserved in a compressed npz next to the tidy shards; the tidy tables keep
the summarized metrics and the top-20 decreased/increased classes.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from common import config, seeds
from common.io import FailureLog
from common.runtime import run_parallel
from common.store import ResultStore

MIN_EVAL_IMAGES = 50
MAX_EVAL_IMAGES = 1000
MIN_CLASSWISE_OBS = 5
TOP_K_SHIFTS = 20
N_RANDOM_CONTROLS = 5


# ---------------------------------------------------------------------------
# Final layer + exact sparse update
# ---------------------------------------------------------------------------

def load_final_layer_path() -> Path:
    return config.CHECKPOINTS_DIR / "resnet18_fc.pt"


def build_final_layer_cache() -> Path:
    """Extract W (1000x512), b (1000) from the torchvision checkpoint once."""
    from data import resnet18_model

    path = load_final_layer_path()
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        m = resnet18_model("cuda:0")
        torch.save({"W": m.fc.weight.detach().cpu(),
                    "b": m.fc.bias.detach().cpu()}, path)
    return path


def load_final_layer() -> tuple[np.ndarray, np.ndarray]:
    path = load_final_layer_path()
    if not path.exists():
        build_final_layer_cache()
    d = torch.load(load_final_layer_path(), map_location="cpu")
    return (d["W"].numpy().astype(np.float64),
            d["b"].numpy().astype(np.float64))


def ablated_logits_sparse(logits: np.ndarray, H: np.ndarray,
                          W: np.ndarray, support) -> np.ndarray:
    """l^A = l - W[:, S] @ h_S. Exact sparse update; empty support returns
    the baseline logits unchanged."""
    S = np.asarray(support, dtype=np.int64).ravel()
    if len(S) == 0:
        return logits
    return logits - H[:, S] @ W[:, S].T


def ablated_logits_full(H: np.ndarray, W: np.ndarray, b: np.ndarray,
                        support) -> np.ndarray:
    """Full final-layer forward on zeroed coordinates (reference path)."""
    H_abl = H.copy()
    H_abl[:, np.asarray(support, dtype=np.int64).ravel()] = 0.0
    return H_abl @ W.T + b


def softmax_np(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def univariate_support(H: np.ndarray, logit_col: np.ndarray, size: int) -> list[int]:
    """Top-|corr| pooled coordinates with a specified logit column,
    within the discovery cluster (spec control)."""
    if logit_col.std() < 1e-12 or H.shape[0] < 3:
        return list(range(size))
    Hc = H - H.mean(0)
    lc = logit_col - logit_col.mean()
    denom = np.sqrt((Hc ** 2).sum(0)) * np.sqrt((lc ** 2).sum()) + 1e-12
    corr = np.abs((Hc * lc[:, None]).sum(0) / denom)
    return np.argsort(-corr, kind="stable")[:size].tolist()


def random_support(size: int, seed_key: tuple) -> list[int]:
    """Uniform random pooled coordinates; deterministic in the factor key."""
    rng = np.random.default_rng(seeds.derive_seed(*seed_key))
    return rng.choice(512, size=size, replace=False).tolist()


# ---------------------------------------------------------------------------
# Evaluation machinery
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def _cached_arrays(cache_path: Path) -> dict:
    if "cache" not in _CACHE:
        with np.load(cache_path) as z:
            _CACHE["cache"] = {"features": z["features"].astype(np.float64),
                               "logits": z["logits"].astype(np.float64),
                               "labels": z["labels"]}
    return _CACHE["cache"]


def _cohorts_arrays() -> dict:
    if "cohorts" not in _CACHE:
        with np.load(config.RAW_DIR / "imagenet" / "eval_cohorts.npz") as z:
            _CACHE["cohorts"] = {k: z[k] for k in z.files}
    return _CACHE["cohorts"]


def eval_cohort(cluster: int) -> tuple[np.ndarray, np.ndarray]:
    z = _cohorts_arrays()
    return z[f"cluster_{cluster:03d}_images"], z[f"cluster_{cluster:03d}_source"].astype(bool)


class CellArrays:
    """npz payload builder for one (cluster, factor) cell."""

    def __init__(self):
        self.d = {}

    def add(self, method, draw, arrays: dict):
        for key, arr in arrays.items():
            self.d[f"{method}_{draw}_{key}"] = arr


def run_ablation_cell(cond: dict) -> dict:
    """One (cluster, factor) cell: Local CorEx support, matched random
    supports, and the univariate support; each evaluated on the cluster's
    cohort. Writes parquet shards + a compressed npz."""
    store = ResultStore("imagenet")
    cluster = int(cond["cluster"])
    factor_index = int(cond["factor_index"])
    threshold = float(cond.get("threshold", 0.80))
    cond_full = {"stage": "ablation", "cluster": cluster,
                 "factor_index": factor_index, "threshold": threshold}
    if store.is_done(cond_full, ("ablation_summary", "top_shifts")):
        return {"cluster": cluster, "factor_index": factor_index, "cached": True}

    cache_path = Path(cond["cache_path"])
    cache = _cached_arrays(cache_path)
    features, logits, labels = (cache["features"], cache["logits"],
                                cache["labels"])
    W, b = load_final_layer()

    eval_imgs, is_source = eval_cohort(cluster)
    H_eval = features[eval_imgs]
    L_eval = logits[eval_imgs]
    y_eval = labels[eval_imgs]
    p_base = softmax_np(L_eval)

    factors = store.read_all("factors")
    frow = factors[(factors["cluster"] == cluster)
                   & (factors["factor_index"] == factor_index)].iloc[0]
    if frow.get("fit_failed", False) == True:  # noqa: E712
        store.write(cond_full, "ablation_summary",
                    [{"cluster": cluster, "factor_index": factor_index,
                      "method": "local_corex", "fit_failed": True,
                      "failure_reason": str(frow.get(
                          "failure_reason", "corex_fit_failed"))}])
        store.write(cond_full, "top_shifts", [])
        return {"cluster": cluster, "factor_index": factor_index,
                "status": "fit_failed"}

    supports_table = store.read_all("supports")
    if "support_override" in cond and cond["support_override"] is not None:
        # case-study cells: support computed from the stored discovery MI
        # vector (case_study.py), never from evaluation data
        support = sorted(int(u) for u in cond["support_override"])
        sup_rows = None
    else:
        sup_rows = supports_table[(supports_table["cluster"] == cluster)
                                  & (supports_table["factor_index"] == factor_index)
                                  & (supports_table["threshold"] == threshold)]
    if sup_rows is not None and len(sup_rows) == 0:
        # threshold outside of the stored primary support (case-study only
        # cell) should not happen here; record as failure-as-data row
        store.write(cond_full, "ablation_summary",
                    [{"cluster": cluster, "factor_index": factor_index,
                      "method": "local_corex", "fit_failed": False,
                      "no_support_at_threshold": True, "threshold": threshold}])
        store.write(cond_full, "top_shifts", [])
        return {"cluster": cluster, "factor_index": factor_index,
                "status": "no_support", "threshold": threshold}
    elif sup_rows is not None:
        support = np.sort(sup_rows.sort_values("mi_rank")["unit"].to_numpy()).tolist()

    # discovery-cluster matrices for the univariate control (cluster data only;
    # evaluation images never enter support construction)
    from data import imagenet_split
    disc_all = store.read_all("clusters")
    disc_member_idx = disc_all[(disc_all["split"] == "discovery")
                               & (disc_all["cluster"] == cluster)][
        "image_index"].to_numpy()
    H_disc = features[disc_member_idx]
    L_disc = logits[disc_member_idx]
    dominant = int(frow["dominant_logit"])
    s = len(support)
    uni_sup = univariate_support(H_disc, L_disc[:, dominant], s)

    interventions = [("local_corex", 0, support)]
    for draw in range(int(cond.get("n_random_controls", N_RANDOM_CONTROLS))):
        interventions.append(("random", draw,
                              random_support(s, (cluster, factor_index,
                                                 "random", draw))))
    interventions.append(("univariate", 0, uni_sup))

    payload: dict[str, np.ndarray] = {}
    summary_rows, shift_rows = [], []
    for method, draw, sup in interventions:
        t0 = time.time()
        p_abl = softmax_np(ablated_logits_sparse(L_eval, H_eval, W, sup))
        row, top_rows, arrays = summarize_intervention(
            cluster, factor_index, threshold, method, draw, sup, p_abl,
            p_base, y_eval, is_source, dominant, cond["wnid_by_index"])
        row["runtime_s"] = time.time() - t0
        summary_rows.append(row)
        shift_rows.extend(top_rows)
        arrays.pop("delta_all", None)
        for key, arr in arrays.items():
            payload[f"{method}_{draw}_{key}"] = arr

    shift_dir = config.RAW_DIR / "imagenet" / "classwise_shifts"
    shift_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        shift_dir / f"cluster_{cluster:03d}_factor_{factor_index:02d}"
        f"_q{int(round(threshold * 100)):02d}.npz", **payload)
    store.write_multiple(cond_full, {"ablation_summary": summary_rows,
                                     "top_shifts": shift_rows})
    return {"cluster": cluster, "factor_index": factor_index,
            "status": "ok", "n_rows": len(summary_rows)}


def summarize_intervention(cluster, factor_index, threshold, method, draw,
                           support, p_abl, p_base, y, is_source, dominant,
                           wnb):
    """Image-level TV, classwise shifts, T_y / r_y / H_input / dominant
    fraction, accuracy changes, localization contrast L_j, and top shifted
    classes (with class names and WordNet ids)."""
    tv = 0.5 * np.abs(p_abl - p_base).sum(1)
    tv_source = float(tv[is_source].mean())
    tv_off = float(tv[~is_source].mean())
    L_contrast = (tv_source - tv_off) / (tv_source + tv_off + 1e-12)

    idx = np.arange(len(y))
    dp_true = p_abl[idx, y] - p_base[idx, y]
    dp_dom = p_abl[:, dominant] - p_base[:, dominant]
    top1_0 = p_base.argmax(1)
    top1_a = p_abl.argmax(1)
    correct_0 = top1_0 == y
    correct_a = top1_a == y

    top5_0 = np.sort(np.argsort(-p_base, axis=1, kind="stable")[:, :5], axis=1)
    top5_a = np.sort(np.argsort(-p_abl, axis=1, kind="stable")[:, :5], axis=1)
    top5_same = (top5_0 == top5_a).all(axis=1)

    represented, T_list, deltas = [], [], {}
    for c in np.unique(y):
        m = y == c
        if m.sum() < MIN_CLASSWISE_OBS:
            continue
        delta = p_abl[m].mean(0) - p_base[m].mean(0)
        deltas[int(c)] = delta
        represented.append(int(c))
        T_list.append(0.5 * np.abs(delta).sum())
    T = np.array(T_list)
    if T.sum() > 0 and len(represented):
        r = T / T.sum()
        H_input = float(-(r * np.log(r, where=r > 0)).sum()
                        / np.log(len(represented)))
        D_input = float(r.max())
        most_affected = int(represented[int(np.argmax(T))])
    else:
        H_input = D_input = float("nan")
        most_affected = int(represented[0]) if represented else -1

    # dominant-class concordance: c*_j among the 5 most-decreased classes
    # for the most-affected true class
    dominant_in_top5 = None
    H_output = float("nan")
    D_output = float("nan")
    delta_star = deltas.get(most_affected)
    if delta_star is not None:
        top5_dec = np.argsort(delta_star, kind="stable")[:5].tolist()
        dominant_in_top5 = bool(int(dominant) in top5_dec)
        q = np.abs(delta_star)
        if q.sum() > 0:
            qn = q / q.sum()
            H_output = float(-(qn * np.log(qn, where=qn > 0)).sum()
                             / np.log(p_base.shape[1]))
            D_output = float(qn.max())

    row = {
        "cluster": cluster, "factor_index": factor_index,
        "method": method, "random_draw": draw, "threshold": threshold,
        "support_json": json.dumps(sorted(int(i) for i in support)),
        "support_size": int(len(support)),
        "support_density": float(len(support) / 512.0),
        "dominant_logit": int(dominant),
        "n_eval": int(len(y)), "n_source": int(is_source.sum()),
        "n_off": int((~is_source).sum()),
        "low_sample": bool(int(is_source.sum()) < MIN_EVAL_IMAGES),
        "n_represented_classes": len(represented),
        "tv_source": tv_source, "tv_off": tv_off,
        "localization_contrast": float(L_contrast),
        "mean_dp_true_class": float(dp_true.mean()),
        "mean_dp_dominant_class": float(dp_dom.mean()),
        "top1_acc_change": float(correct_a.mean() - correct_0.mean()),
        "top1_flip_rate": float((top1_0 != top1_a).mean()),
        "top5_set_change_rate": float((~top5_same).mean()),
        "H_input": H_input, "dominant_input_fraction": D_input,
        "H_output_redistribution": H_output,
        "dominant_output_shift_fraction": D_output,
        "most_affected_true_class": most_affected,
        "dominant_in_top5_decreased": dominant_in_top5,
        "represented_class_tsy_json": json.dumps(
            {c: T_list[i] for i, c in enumerate(represented)}),
    }

    # tidy top-20 decreased/increased classes for every represented group
    shift_rows = []
    for c in represented:
        full = deltas[c]
        dec = np.argsort(full, kind="stable")[:TOP_K_SHIFTS]
        inc = np.argsort(-full, kind="stable")[:TOP_K_SHIFTS]
        for direction, idxs in (("decrease", dec), ("increase", inc)):
            for rank, cc in enumerate(idxs):
                shift_rows.append({
                    "cluster": cluster, "factor_index": factor_index,
                    "method": method, "random_draw": draw,
                    "threshold": threshold,
                    "true_class": int(c), "output_class": int(cc),
                    "output_wnid": wnb[int(cc)]["wnid"],
                    "output_class_name": wnb[int(cc)]["class_name"],
                    "direction": direction, "shift_rank": int(rank),
                    "signed_shift": float(full[cc]),
                })

    arrays = {
        "tv": tv.astype(np.float32),
        "dp_true": dp_true.astype(np.float32),
        "dp_dominant": dp_dom.astype(np.float32),
        "top5_same": top5_same,
        "top1_correct": correct_a.astype(np.bool_),
        "classwise_group_ids": np.asarray(represented, dtype=np.int64),
    }
    for c, d in deltas.items():
        arrays[f"delta_{c}"] = d.astype(np.float32)
    return row, shift_rows, arrays


def wnids_by_index() -> list[dict]:
    from data import class_mapping
    return class_mapping()


def run_ablations(store: ResultStore, cfg: dict, cache_path: Path,
                  failure_log: FailureLog, max_workers: int | None = 16) -> dict:
    factors = store.read_all("factors")
    if "fit_failed" in factors.columns:
        factors = factors[~factors["fit_failed"].astype(bool)]
    cells = [{"stage": "ablation", "cluster": int(k), "factor_index": int(fi),
              "threshold": float(cfg["primary_q"]),
              "n_random_controls": int(cfg["n_random_controls"]),
              "cache_path": str(cache_path), "wnid_by_index": wnids_by_index()}
             for (k, fi) in factors.groupby(["cluster", "factor_index"]
                                            ).groups.keys()]
    t0 = time.time()
    run_parallel(_run_ablation_condition, cells, max_workers, failure_log,
                 "imagenet", stage_name="ablation")
    return {"n_cells": len(cells), "runtime_s": time.time() - t0}


def _run_ablation_condition(cond: dict) -> dict:
    return run_ablation_cell(cond)
