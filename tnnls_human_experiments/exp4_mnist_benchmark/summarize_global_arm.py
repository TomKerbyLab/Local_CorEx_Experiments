"""Summarize the global CorEx comparison arm from stored shards (fb#2 arm).

Reads the ResultStore tables written by global_correx_arm.py and writes:

- `global_arm_contrasts.csv`: per (budget, layer, input_condition,
  threshold) paired contrasts — Local CorEx minus the mean of its matched
  random draws for T_max, H_input, and accuracy change; factor-paired
  mean, SE, and 95% t-Interval over the 25 retained factors (5 seeds x
  top-5). Spatial concentration (normalized entropy over the 20 fixed
  test clusters, and the dominant-cluster fraction) compared LC vs random.
- `global_arm_summary.csv`: the same cells' LC and random levels.
- `global_arm_coverage.csv`: fit counts, converged flags, degenerate
  factors, and per-(cond, layer, budget) support-size stats.

Reads raw shards only; never refits.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from common.store import ResultStore

RUN_NAME = "global_corex_arm"
N_CLUSTERS = 20
T_CRIT = {25: 2.060, 30: 2.042, 40: 2.023, 50: 2.009, 60: 2.000, 100: 1.984}


def _t_crit(df):
    n = max(len(df), 2)
    key = min(T_CRIT, key=lambda k: abs(k - n)) if n > 25 else 2.060
    return T_CRIT.get(key, 2.060) if n >= 25 else 2.069


def spatial_metrics(cluster_tv_json: str) -> dict:
    vals = np.asarray(json.loads(cluster_tv_json))
    vals = vals[vals >= 0]
    if len(vals) == 0 or vals.sum() <= 0:
        return {"tv_entropy": np.nan, "tv_dominant_frac": np.nan}
    r = vals / vals.sum()
    r = r[r > 0]
    e = -(r * np.log(r)).sum() / np.log(len(vals))
    return {"tv_entropy": float(e), "tv_dominant_frac": float(r.max())}


def _expand(df: pd.DataFrame) -> pd.DataFrame:
    sp = pd.DataFrame(list(df["cluster_tv"].map(spatial_metrics)))
    return pd.concat([df.reset_index(drop=True), sp], axis=1)


def paired_contrasts(store: ResultStore) -> pd.DataFrame:
    tv = store.read_all("global_cluster_tv")
    tv = _expand(tv)
    rng = tv[tv.method == "random"].groupby(
        ["model_seed", "input_condition", "layer", "budget", "factor_rank",
         "threshold"]) [["T_max", "H_input", "accuracy_change", "tv_entropy",
                         "tv_dominant_frac"]].mean().reset_index()
    lc = tv[tv.method == "global_corex"]
    p = lc.merge(rng, on=["model_seed", "input_condition", "layer", "budget",
                          "factor_rank", "threshold"],
                 suffixes=("", "_rand"))
    p["d_T_max"] = p.T_max - p.T_max_rand
    p["d_H_input"] = p.H_input - p.H_input_rand
    p["d_acc"] = p.accuracy_change - p.accuracy_change_rand
    p["d_entropy"] = p.tv_entropy - p.tv_entropy_rand
    p["d_dominant"] = p.tv_dominant_frac - p.tv_dominant_frac_rand
    out = []
    for key, g in p.groupby(["input_condition", "layer", "budget",
                             "threshold"]):
        row = {"input_condition": key[0], "layer": key[1], "budget": key[2],
               "threshold": key[3], "n_factors": len(g)}
        for m in ("d_T_max", "d_H_input", "d_acc", "d_entropy",
                  "d_dominant"):
            v = g[m].to_numpy()
            se = v.std(ddof=1) / np.sqrt(len(v))
            tc = _t_crit(len(v))
            row[f"{m}_mean"] = float(v.mean())
            row[f"{m}_se"] = float(se)
            row[f"{m}_ci_lo"] = float(v.mean() - tc * se)
            row[f"{m}_ci_hi"] = float(v.mean() + tc * se)
        out.append(row)
    pd.DataFrame(out).sort_values(
        ["input_condition", "layer", "budget", "threshold"]).to_csv(
        "run_results/global_arm_contrasts.csv", index=False)
    return pd.DataFrame(out)


def levels(store: ResultStore) -> pd.DataFrame:
    tv = store.read_all("global_cluster_tv")
    tv = _expand(tv)
    lv = tv.groupby(["method", "input_condition", "layer", "budget",
                     "threshold"]).agg(
        T_max=("T_max", "mean"), H_input=("H_input", "mean"),
        accuracy_change=("accuracy_change", "mean"),
        tv_entropy=("tv_entropy", "mean"),
        tv_dominant_frac=("tv_dominant_frac", "mean"),
        n=("T_max", "size"),
    ).reset_index()
    lv.to_csv("run_results/global_arm_summary.csv", index=False)
    return lv


def coverage(store: ResultStore) -> pd.DataFrame:
    fac = store.read_all("global_factors")
    cov = fac.groupby(["input_condition", "layer", "budget"]).agg(
        n_factors=("factor_rank", "size"),
        n_seeds=fac.model_seed.nunique,
        converged_frac=("corex_converged", "mean"),
        mean_tc=("factor_tc", "mean"),
        mean_runtime_s=("corex_runtime_s", "mean"),
    ).reset_index()
    sv = store.read_all("global_supports")
    sv = sv.groupby(["input_condition", "layer", "budget",
                     "threshold"]).support_size.agg(
        ["mean", "min", "max"]).reset_index()
    sv.columns = ["input_condition", "layer", "budget", "threshold",
                  "support_mean", "support_min", "support_max"]
    out = cov.merge(sv, on=["input_condition", "layer", "budget"], how="left")
    out.to_csv("run_results/global_arm_coverage.csv", index=False)
    return out


def main():
    store = ResultStore("mnist")
    if store.read_all("global_cluster_tv").empty:
        print("global_cluster_tv empty or missing — run the arm first")
        return
    print("contrasts:\n", paired_contrasts(store).to_string(index=False))
    print("coverage:\n", coverage(store).to_string(index=False))
    print("levels written to run_results/global_arm_summary.csv")


if __name__ == "__main__":
    main()
