"""Two-cluster class-agnostic case study (freeze change-log entries of
2026-11-05, written BEFORE any result of these clusters was read).

Pre-designated panels (see `paper/exp5_protocol_freeze.md`, change log):
  primary:   the cluster with the highest dominant-class purity among
             clusters with >= 60 combined (discovery + evaluation) images,
             ties by purity, then combined size, then lower cluster id
             (resolved to cluster-21, great grey owl, 47/73 = 64.4%;
             runner-up cluster-69 at 57.8%).
  secondary: the largest fully-pure cluster, ties by lower cluster id
             (resolved to cluster-85, African grey parrot, 46/46 = 100%;
             cluster-41, planetarium, 34/34, excluded by the size
             tie-break). Cluster-85 is `low_sample` (evaluation cohort
             below the F2 30-image floor) and is reported as illustrative.

Both panels are computed and REPORTED REGARDLESS OF OUTCOME; neither may
be promoted, demoted, or dropped after its results are read.

Selection is outcome-blind: only the per-image assignment table and true
labels enter. Factor selection: the two retained factors with the largest
mutual information with the cluster's dominant-class logit (label-space
class identity, not a fitted outcome). Supports at q in
{0.60, 0.70, 0.80, 0.90}: the q=0.80 cells come from the main ablation
stage (every retained factor was ablated there), the remaining thresholds
are evaluated here with matched random + univariate controls under the
identical cell machinery as the frozen case study (stage `ablation`,
support_override from the stored discovery MI vectors).
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from common import config
from common.io import FailureLog
from common.runtime import run_parallel
from common.store import ResultStore

from ablations import run_ablation_cell, wnids_by_index

PANEL_THRESHOLDS = (0.60, 0.70, 0.80, 0.90)
MIN_COMBINED_FOR_PRIMARY = 60
N_SELECTED_FACTORS = 2

# Pre-designation recorded in the freeze change log BEFORE results were
# read; the rules below must resolve exactly to these clusters, whose ids
# are run-specific (run commit 3b1313c, cache labels_sha256 8771f37f...).
EXPECTED_PRIMARY = 21
EXPECTED_SECONDARY = 85


# ---------------------------------------------------------------------------
# Outcome-blind selection rules (unit-testable; no fits, no ablations)
# ---------------------------------------------------------------------------

def combined_composition(assign: pd.DataFrame) -> dict[int, dict]:
    """Per-cluster dominant-class purity over discovery + evaluation.

    assign: assignment table with columns (cluster, true_class, split).
    Outcome-blind: labels only."""
    out = {}
    per = assign.groupby("cluster")
    for k, g in per:
        counts = g["true_class"].value_counts()
        dom_class = int(counts.index[0])
        dom_count = int(counts.iloc[0])
        n = int(counts.sum())
        # split the dominant-class count by half for provenance
        dom_splits = (g[g["true_class"] == dom_class]["split"]
                      .value_counts().to_dict())
        out[int(k)] = {
            "n_combined": n,
            "dominant_class": dom_class,
            "dominant_count": dom_count,
            "purity": dom_count / n,
            "dominant_count_discovery": int(dom_splits.get("discovery", 0)),
            "dominant_count_evaluation": int(dom_splits.get("evaluation", 0)),
            "n_distinct_classes": int(len(counts)),
        }
    return out


def primary_rule(comp: dict[int, dict]) -> int:
    """Highest dominant-class purity among clusters with >= 60 combined
    images; ties by purity, then combined size, then lower cluster id."""
    cands = [(v["purity"], v["n_combined"], -k) for k, v in comp.items()
             if v["n_combined"] >= MIN_COMBINED_FOR_PRIMARY]
    if not cands:
        raise RuntimeError("primary rule failed: no cluster with >= "
                           f"{MIN_COMBINED_FOR_PRIMARY} combined images")
    best = max(cands)
    return -best[2]


def secondary_rule(comp: dict[int, dict]) -> int:
    """Largest fully-pure (purity == 1) cluster; ties by lower cluster id."""
    cands = [(v["n_combined"], -k) for k, v in comp.items()
             if v["purity"] == 1.0]
    if not cands:
        raise RuntimeError("secondary rule failed: no fully-pure cluster")
    best = max(cands)
    return -best[1]


def selection_records(store: ResultStore, assign: pd.DataFrame) -> list[dict]:
    comp = combined_composition(assign)
    primary = primary_rule(comp)
    secondary = secondary_rule(comp)
    assert primary == EXPECTED_PRIMARY, \
        f"primary rule resolved to {primary}, change log fixed {EXPECTED_PRIMARY}"
    assert secondary == EXPECTED_SECONDARY, \
        f"secondary rule resolved to {secondary}, change log fixed {EXPECTED_SECONDARY}"
    runner_up = max([(v["purity"], v["n_combined"], -k)
                     for k, v in comp.items()
                     if v["n_combined"] >= MIN_COMBINED_FOR_PRIMARY
                     and k != primary])
    pure_other = sorted(k for k, v in comp.items()
                        if v["purity"] == 1.0 and k != secondary)
    recs = []
    for panel, cid, rule_text in (
            ("primary", primary,
             "highest dominant-class purity among clusters with >= 60 "
             "combined images, ties by purity, then size, then lower id"),
            ("secondary", secondary,
             "largest fully-pure cluster, ties by lower cluster id")):
        c = comp[cid]
        recs.append({
            "panel": panel, "selected_cluster": int(cid),
            "rule": rule_text,
            "dominant_class": c["dominant_class"],
            "n_combined": c["n_combined"],
            "dominant_count": c["dominant_count"],
            "purity": c["purity"],
            "dominant_count_discovery": c["dominant_count_discovery"],
            "dominant_count_evaluation": c["dominant_count_evaluation"],
            "cluster_ids_run_specific": True,
            "uses_evaluation_labels": True,
            "uses_evaluation_results": False,
            "pre_designated_in_freeze_change_log":
                "paper/exp5_protocol_freeze.md change log 2026-11-05",
        })
    runner_up = max(v["purity"] for k, v in comp.items()
                    if k != primary and v["n_combined"] >= MIN_COMBINED_FOR_PRIMARY)
    recs[0]["runner_up_purity"] = float(runner_up)
    pure_other = sorted(k for k, v in comp.items()
                        if v["purity"] == 1.0 and k != secondary)
    recs[1]["excluded_fully_pure_clusters"] = json.dumps(
        [{"cluster": int(k), "n_combined": comp[k]["n_combined"],
          "dominant_class": comp[k]["dominant_class"]} for k in pure_other])
    return recs


# ---------------------------------------------------------------------------
# Factor / support selection (stored discovery MI only)
# ---------------------------------------------------------------------------

def select_cluster_factors(cluster: int, dom_class: int) -> list[dict]:
    """Two retained factors with the largest MI with the dominant-class
    logit (L[:, dom_class]), from the stored discovery fit MI matrix."""
    import case_study as cs
    fit = cs.load_fit_mis(cluster)
    mis = fit["mis_clipped"]
    n_h = int(fit["n_h"])
    order = fit["order"].tolist()
    col = mis[:, n_h + dom_class]
    cand = sorted(order, key=lambda j: (-col[j], j))
    rows = []
    for rank, j in enumerate(cand[:N_SELECTED_FACTORS]):
        rows.append({
            "panel_dom_class": dom_class, "cluster": cluster,
            "factor_index": int(j), "selection_rank": rank,
            "factor_to_dominant_logit_mi_nats": float(col[j]),
            "factor_to_dominant_logit_mi_bits": float(col[j] / np.log(2.0)),
            "unit_mi_rank_within_cluster":
                int(np.where(np.asarray(order) == j)[0][0]),
        })
    return rows


def _cluster_dominant_class(assign: pd.DataFrame, cluster: int) -> int:
    g = assign[assign["cluster"] == cluster]
    counts = g["true_class"].value_counts()
    return int(counts.index[0])


# ---------------------------------------------------------------------------
# Ablation cells (same machinery as the frozen case study)
# ---------------------------------------------------------------------------

def _cell(cond: dict) -> dict:
    return run_ablation_cell(cond)


# ---------------------------------------------------------------------------
# Secondary-cluster report (generalized from case_study.secondary_cluster_report)
# ---------------------------------------------------------------------------

def secondary_cluster_report(store: ResultStore, cluster: int,
                             factors: list[dict]) -> list[dict]:
    import case_study as cs
    from clustering import load_cluster_assignments
    clusters = load_cluster_assignments(store, {
        "n_clusters": 100, "phate_seed": 7000, "kmeans_seed": 7001})
    with np.load(config.RAW_DIR / "imagenet" / "eval_cohorts.npz") as z:
        eval_imgs = z[f"cluster_{cluster:03d}_images"]
        is_source = z[f"cluster_{cluster:03d}_source"].astype(bool)
    rows = []
    for f in factors:
        fi = f["factor_index"]
        q_main = int(round(0.80 * 100))
        with np.load(config.RAW_DIR / "imagenet" / "classwise_shifts" /
                     f"cluster_{cluster:03d}_factor_{fi:02d}"
                     f"_q{q_main}.npz") as z:
            tv = z["local_corex_0_tv"]
        off = ~is_source
        off_imgs = eval_imgs[off]
        tv_off = tv[off]
        off_cluster_ids = clusters[off_imgs]
        df = pd.DataFrame({"cluster": off_cluster_ids, "tv": tv_off})
        per = df.groupby("cluster")["tv"].agg(["mean", "count"])\
            .sort_values("mean", ascending=False)
        for rank, (k, agg) in enumerate(per.head(10).iterrows()):
            rows.append({"cluster": cluster, "factor_index": fi,
                         "off_cluster": int(k),
                         "mean_tv_off_cluster": float(agg["mean"]),
                         "n_images": int(agg["count"]), "rank": rank,
                         "is_most_affected_secondary": rank == 0})
    return rows


# ---------------------------------------------------------------------------
# Dominant-class focus metrics (from stored classwise delta arrays)
# ---------------------------------------------------------------------------

def focus_rows(cluster: int, factors: list[dict], dom_class: int,
               n_random_draws: int = 5) -> list[dict]:
    """The cluster's dominant-class probability change per intervention
    (Local CorEx, univariate, mean over the matched random supports),
    from the stored classwise mean-shift arrays. delta_dom is the mean
    logit-space probability change over the eval-cohort images whose true
    class equals the cluster's dominant class (>= 5 such images required
    by the cell machinery, which holds for both clusters)."""
    rows = []
    dom_key = f"delta_{dom_class}"
    for f in factors:
        fi = f["factor_index"]
        for q in PANEL_THRESHOLDS:
            path = (config.RAW_DIR / "imagenet" / "classwise_shifts" /
                    f"cluster_{cluster:03d}_factor_{fi:02d}"
                    f"_q{int(round(q * 100)):02d}.npz")
            if not path.exists():
                rows.append({"cluster": cluster, "factor_index": fi,
                             "threshold": q, "missing_npz": str(path)})
                continue
            with np.load(path) as z:
                keys = set(z.files)
                per_method = {}
                draw_deltas = []
                for key in sorted(keys):
                    if not key.endswith(dom_key):
                        continue
                    delta = z[key]
                    head = key[: -(len(dom_key) + 1)]
                    if head.startswith("random_"):
                        draw_deltas.append((head, delta))
                    else:
                        per_method[head] = delta
                def _stats(delta):
                    return {"dom_class_prob_change": float(delta[dom_class]),
                            "dom_class_total_variation":
                                float(0.5 * np.abs(delta).sum())}
                for head, delta in per_method.items():
                    rows.append({"cluster": cluster, "factor_index": fi,
                                 "threshold": q,
                                 "method": head.rsplit("_", 1)[0],
                                 "random_draw": 0, **_stats(delta)})
                if draw_deltas:
                    stack = np.stack([d for _, d in sorted(draw_deltas)])
                    rows.append({"cluster": cluster, "factor_index": fi,
                                 "threshold": q, "method": "random",
                                 "random_draw": "mean_over_draws",
                                 "dom_class_prob_change":
                                     float(stack[:, dom_class].mean()),
                                 "dom_class_total_variation":
                                     float(0.5 * np.abs(
                                         stack.mean(axis=0)).sum())})
    return rows


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def run_two_cluster_case_study(store: ResultStore, cache_path: Path,
                               max_workers: int | None = 8) -> dict:
    import case_study as cs

    assign = store.read_all("clusters")
    labels = None
    with np.load(cache_path) as z:
        labels = z["labels"]
    assert (assign["true_class"].to_numpy()
            == labels[assign["image_index"].to_numpy()]).all(), \
        "assignment table disagrees with cache labels"

    sel_recs = selection_records(store, assign)
    conditions = {"stage": "case_study_two_selection"}
    if not store.is_done(conditions, ("case_study_two_selection",)):
        store.write(conditions, "case_study_two_selection", sel_recs)

    wnid = wnids_by_index()
    cells, panel_meta = [], []
    for rec in sel_recs:
        cluster, dom = int(rec["selected_cluster"]), int(rec["dominant_class"])
        factors = select_cluster_factors(cluster, dom)
        fac_cond = {"stage": "case_study_two_factors", "cluster": cluster}
        if not store.is_done(fac_cond, ("case_study_two_factors",)):
            store.write(fac_cond, "case_study_two_factors", factors)
        panel_meta.append({"panel": rec["panel"], "cluster": cluster,
                           "dominant_class": dom, "factors": factors})
        for f in factors:
            sup_rows = cs.case_support_rows(cluster, f["factor_index"])
            for q in PANEL_THRESHOLDS:
                support = np.sort(
                    pd.DataFrame([r for r in sup_rows
                                  if r["threshold"] == q]
                                 ).sort_values("mi_rank")["unit"]
                    .to_numpy()).tolist()
                cells.append({"stage": "ablation", "cell_stage":
                              "case_two_ablation",
                              "kinds": ("case_two_ablation_summary",
                                        "case_two_top_shifts"),
                              "cluster": cluster,
                              "factor_index": f["factor_index"],
                              "threshold": q, "support_override": support,
                              "n_random_controls": 5,
                              "cache_path": str(cache_path),
                              "wnid_by_index": wnid})
    t0 = time.time()
    run_parallel(_cell, cells, max_workers, FailureLog("imagenet"),
                 "imagenet", stage_name="case_study_two_ablation")

    # secondary-cluster report + focus metrics + tidy summary assembly
    sec_rows, focus = [], []
    for meta in panel_meta:
        cluster = meta["cluster"]
        factors = meta["factors"]
        sec_rows.extend(secondary_cluster_report(store, cluster,
                                                 [{'factor_index': f['factor_index']}
                                                  for f in factors]))
        focus.extend(focus_rows(cluster, factors, meta["dominant_class"]))

    abl = store.read_all("case_two_ablation_summary")
    keys = {(int(c["cluster"]), int(c["factor_index"]), float(c["threshold"]))
            for c in cells}
    mask = [(int(k), int(fi), float(q)) in keys
            for k, fi, q in zip(abl["cluster"], abl["factor_index"],
                                abl["threshold"])]
    case_abl = abl[mask].copy()
    doms = {int(m["cluster"]): int(m["dominant_class"]) for m in panel_meta}
    panels = {int(m["cluster"]): m["panel"] for m in panel_meta}
    case_abl["panel"] = case_abl["cluster"].map(panels)
    case_abl["cluster_dominant_class"] = case_abl["cluster"].map(doms)
    case_abl["git_commit"] = _head()

    out_dir = config.SUMMARIES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(sel_recs).to_csv(out_dir / "imagenet_case_study_two_selection.csv",
                                  index=False)
    pd.concat([pd.DataFrame(m["factors"]).assign(panel=m["panel"])
               for m in panel_meta], ignore_index=True).to_csv(
        out_dir / "imagenet_case_study_two_factors.csv", index=False)
    sup_all = []
    for meta in panel_meta:
        for f in meta["factors"]:
            sup_all.extend(cs.case_support_rows(meta["cluster"],
                                                f["factor_index"]))
    pd.DataFrame(sup_all).to_csv(out_dir / "imagenet_case_study_two_supports.csv",
                                 index=False)
    case_abl.to_csv(out_dir / "imagenet_case_study_two.csv", index=False)
    pd.DataFrame(sec_rows).to_csv(
        out_dir / "imagenet_case_study_two_secondary.csv", index=False)
    pd.DataFrame(focus).to_csv(out_dir / "imagenet_case_study_two_focus.csv",
                               index=False)

    return {"panels": panel_meta, "n_cells": len(cells),
            "runtime_s": time.time() - t0,
            "summaries": [out_dir / f for f in (
                "imagenet_case_study_two_selection.csv",
                "imagenet_case_study_two_factors.csv",
                "imagenet_case_study_two_supports.csv",
                "imagenet_case_study_two.csv",
                "imagenet_case_study_two_secondary.csv",
                "imagenet_case_study_two_focus.csv")]}


def _head() -> str:
    import subprocess
    return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                          text=True).stdout.strip()


if __name__ == "__main__":
    from common.config import RAW_DIR
    store = ResultStore("imagenet")
    summary = run_two_cluster_case_study(
        store, RAW_DIR / "imagenet" / "resnet18_imagenet_val_cache.npz")
    print(json.dumps({k: v for k, v in summary.items() if k != "panels"},
                     indent=2, default=str))
    for meta in summary["panels"]:
        print(meta["panel"], "cluster", meta["cluster"], "dominant class",
              meta["dominant_class"],
              [(f["factor_index"], round(f["factor_to_dominant_logit_mi_bits"], 3))
               for f in meta["factors"]])
