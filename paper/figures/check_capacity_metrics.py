"""Round-3 targeted analyses on committed stores.

1. Gaussian equal-budget grid: decompose each configuration's score into
   (i) fixed-collection coverage (sum of matched top-k F1 over the pooled
   true groups; unmatched groups contribute 0) and (ii) matched-pair
   precision (mean over matched pairs, the previously plotted statistic),
   to expose the denominator change between M = 5 and M = 10.
2. MNIST matched-volume local/global comparison at three support-fraction
   pairs (global threshold vs local threshold chosen so pooled support
   fractions agree), using committed absolute tables on both arms.

Run with the project venv from the repository root:
    .venv/bin/python paper/figures/check_capacity_metrics.py
Writes outputs/capacity_check/.

Also verifies the E4 split accounting from the committed split manifest
(45k train + 5k val + 10k discovery partition the 60k pool, per class).
"""

import json
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "outputs/capacity_check"
MAIN = ROOT  # committed result files live inside the repo tree


def gaussian_denominators():
    g = pd.read_csv(ROOT / "tnnls_human_experiments/gaussian/capacity_data"
                         "/global_capacity/global_capacity_combined.csv")
    g = g[g.covariance_setup != "homogeneous"]
    g["n_groups_pooled"] = (g.num_matched_factors
                            + g.num_unmatched_true_groups)
    g["coverage_f1"] = (g.matched_top_k_f1 * g.num_matched_factors
                        / g.n_groups_pooled)
    g["n_learned"] = g.num_matched_factors + g.num_unmatched_learned_factors
    g["precision_f1"] = g.matched_top_k_f1

    sel = g[g.scope == "global_capacity"]
    loc = g[g.scope == "local_pooled_capacity"]
    rows = []
    for setup in ["disjoint", "nondisjoint"]:
        for b in [5, 10]:
            s = sel[(sel.covariance_setup == setup) & (sel.budget == b)]
            l = loc[(loc.covariance_setup == setup) & (loc.budget == b)]
            for arm, df, nfac in (("global M=%d" % b, s, f"{b}"),
                                  ("local stack", l, None)):
                if len(df) == 0:
                    continue
                rows.append(dict(
                    setup=setup, arm=arm,
                    n_groups=df.n_groups_pooled.unique().tolist(),
                    n_matched=df.num_matched_factors.unique().tolist(),
                    n_learned=df.n_learned.unique().tolist(),
                    matched_pair_f1=df.precision_f1.mean(),
                    coverage_f1=df.coverage_f1.mean(),
                    n_reps=len(df)))
    t = pd.DataFrame(rows)
    print("== Gaussian equal-budget grid: coverage vs matched-pair scoring ==")
    print(t.round(3).to_string(index=False))
    return t


def matched_volume_neural():
    loc = pd.read_csv(ROOT / "tnnls_human_experiments/exp4_mnist_benchmark/"
                           "results/mnist_tc_density_magnitude.csv")
    gsum = pd.read_csv(ROOT / "tnnls_human_experiments/exp4_mnist_benchmark/"
                            "results/global_arm_summary.csv")
    gcov = pd.read_csv(ROOT / "tnnls_human_experiments/exp4_mnist_benchmark/"
                            "results/global_arm_coverage.csv")
    units = {1: 500, 2: 400, 3: 300}
    layers = [1, 2, 3]
    gl = gsum[(gsum.method == "global_corex")
              & (gsum.input_condition == "output_aware")
              & (gsum.budget == 10)]

    pairs = [(0.6, 0.7), (0.7, 0.8), (0.8, 0.9)]  # (global q, local q)
    rows = []
    for gq, lq in pairs:
        for l in layers:
            gv_ = gl[(gl.layer == l) & (gl.threshold == gq)].iloc[0]
            gfrac = gcov[(gcov.input_condition == "output_aware")
                         & (gcov.budget == 10) & (gcov.layer == l)
                         & (gcov.threshold == gq)].support_mean.iloc[0] \
                / units[l]
            lc = loc[(loc.layer == l) & (loc.threshold == lq)]
            perseed = lc.groupby("model_seed")[
                ["T_max", "H_input", "support_size"]].mean()
            lfrac = perseed.support_size.mean() / units[l]
            rows.append(dict(
                global_q=gq, local_q=lq, layer=l,
                global_T_max=gv_.T_max, global_H_input=gv_.H_input,
                global_frac=gfrac,
                local_T_max=perseed.T_max.mean(),
                local_T_max_se=perseed.T_max.std(ddof=1)
                / np.sqrt(len(perseed)),
                local_H_input=perseed.H_input.mean(),
                local_H_input_se=perseed.H_input.std(ddof=1)
                / np.sqrt(len(perseed)),
                local_frac=lfrac,
                frac_diff=gfrac - lfrac,
                n_seeds=len(perseed)))
    t = pd.DataFrame(rows)
    print("\n== MNIST matched-volume local/global comparison (pooled means "
          "on both arms; local side also carries per-seed uncertainty) ==")
    print(t.round(3).to_string(index=False))
    return t


def split_accounting():
    meta = json.load(open(ROOT / "tnnls_human_experiments/exp4_mnist_benchmark"
                               "/results/mnist_splits_manifest.json"))
    sizes = meta["sizes"]
    per_class = meta["class_counts"]
    ok_sizes = sizes["train"] + sizes["val"] + sizes["discovery"] == 60000
    print("\n== E4 split accounting from the committed manifest ==")
    print("sizes partition the 60k pool (45k + 5k + 10k):", ok_sizes)
    print("val profile sums to 5,000:",
          sum(per_class["val"]) == sizes["val"])
    print("discovery profile sums to 10,000:",
          sum(per_class["discovery"]) == sizes["discovery"])
    print("note: the manifest's 'train' class row carries the full 60k-pool "
          "profile (sums to 60,000), so train = pool minus val minus "
          "discovery; discovery is disjoint from classifier training by "
          "construction of the recorded stratified split (seed 5000).")


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    t1 = gaussian_denominators()
    t2 = matched_volume_neural()
    split_accounting()
    t1.to_csv(OUT / "gaussian_capacity_coverage_vs_precision.csv", index=False)
    t2.to_csv(OUT / "mnist_matched_volume.csv", index=False)
    print("written:", OUT)
