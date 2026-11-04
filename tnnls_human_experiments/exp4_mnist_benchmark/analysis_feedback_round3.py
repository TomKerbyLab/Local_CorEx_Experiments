"""Round-3 feedback recomputations (exp4.md §E4.14-§E4.14.2, [rev3] items).

Every number quoted in the round-3 revision of paper/exp4.md is reproduced
here from the committed re-run store (`run_results/raw/mnist`) and the
committed results tables (`results/`). Read-only: no store writes, no
model runs. Optional determinism/precision check (refits all 300
output-aware fit conditions; needs the classifier checkpoints and a GPU/
CPU forward pass) behind `--refit`.

Analyses:
  A. grid composition (corrected component list, §E4.13)
  B. cancellation reconciliation (Finding 1: ratio definitions, per-class
     splits, heavy-cancellation fractions, largest-gap cases)
  C. q=0.99 Jaccard vs same-size random reference (§E4.14.2, decision #19)
  D. TC-magnitude: raw / residualized / partial Spearman / size-bands
     (Finding 6, decision #20)
  E. concordance vs matched-random reference (Finding 7)
  F. activity-matching ratios per stratum (Finding 1 / sensitivity items)
  G. factor heterogeneity fractions + association links (new display)
  H. profile cohorts: levels vs excesses, paired own-vs-other contrast
  I. within-digit contrast for mixed clusters
  J. quote checks: per-seed excess, max single-cell shift, density ranges
  K. extension aggregates + univariate paired diffs (§E4.14.2)
"""
from __future__ import annotations

import glob
import json
import sys
import collections
import numpy as np
import pandas as pd

HERE = __file__ and __import__("os").path.dirname(__file__)
RAW = HERE + "/run_results/raw/mnist"
RES = HERE + "/results"


def load_all(pattern, cols=None):
    return pd.concat([pd.read_parquet(f, columns=cols)
                      for f in sorted(glob.glob(pattern))], ignore_index=True)


def sp(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return np.nan
    return float(pd.Series(a).corr(pd.Series(b), method="spearman"))


def partial_spearman(x, y, z):
    """Spearman correlation of the rank-regression residuals after removing
    z from both variables (standard partial Spearman)."""
    rx, ry, rz = (pd.Series(v).rank().values for v in (x, y, z))
    def resid(t, s):
        A = np.column_stack([s, np.ones(len(s))])
        coef, *_ = np.linalg.lstsq(A, t, rcond=None)
        return t - A @ coef
    return sp(resid(rx, rz), resid(ry, rz))


def jaccard_expected(s1, s2, pool):
    e = s1 * s2 / pool
    return e / (s1 + s2 - e)


def main():
    abl = load_all(RAW + "/*.ablation_summary.parquet")
    fac = load_all(RAW + "/*.factors.parquet")
    ext = load_all(RAW + "/*.extension_summary.parquet")
    comps = json.load(open(RAW + "/cluster_compositions.json"))
    dom_digit = {int(c): int(np.argmax(v)) for c, v in comps.items()}
    lc_all = abl[abl.method == "local_corex"]
    key6 = ["model_seed", "cluster", "layer", "input_condition",
            "factor_index", "threshold"]
    key4 = ["model_seed", "cluster", "layer", "input_condition", "factor_index"]

    print("== A. grid composition (§E4.13 corrected list)")
    print(abl.groupby(["input_condition", "method"]).size().to_string())
    print("total ablation rows:", len(abl), "| extension rows:", len(ext))

    print("\n== B. cancellation reconciliation (Finding 1)")
    for th in (0.6, 0.7, 0.8, 0.9):
        sub = lc_all[(lc_all.input_condition == "output_aware")
                     & (lc_all.threshold == th)]
        print(f"q={th}: mean T_max/U_max={(sub.T_max/sub.U_max).mean():.4f} "
              f"(ratio of maxima), mean T_w/U_w="
              f"{(sub.T_weighted/sub.U_weighted).mean():.4f} (weighted averages)")
    lc8 = lc_all[(lc_all.input_condition == "output_aware")
                 & (lc_all.threshold == 0.8)]
    rows = []
    for _, r in lc8.iterrows():
        delta = np.array(json.loads(r["delta_matrix"]))
        U = np.array(json.loads(r["U_per_class"]))
        T = 0.5 * np.abs(delta).sum(axis=1)
        for y in range(10):
            rows.append(dict(seed=r["model_seed"], layer=int(r["layer"]),
                             cluster=int(r["cluster"]), fact=int(r["factor_index"]),
                             y=y, T_y=T[y], U_y=U[y],
                             is_assoc=(y == int(r["dominant_logit"])),
                             is_dom=(y == dom_digit[int(r["cluster"])])))
    TD = pd.DataFrame(rows)
    TD["gap"] = TD.U_y - TD.T_y
    TD["ratio"] = TD.T_y / TD.U_y
    for name, sel in (("assoc-class rows", TD.is_assoc),
                      ("dominant-digit rows", TD.is_dom),
                      ("other classes", ~TD.is_assoc & ~TD.is_dom & (TD.T_y > 1e-9))):
        s = TD[sel]
        print(f"{name:22s} n={len(s):5d} median T_y/U_y={s.ratio.median():.3f} "
              f"median gap={s.gap.median():.4f} p90 gap={s.gap.quantile(.9):.4f}")
    D = TD[TD.is_assoc].groupby("layer").ratio.median()
    print("assoc-class median T_y/U_y by layer:",
          ", ".join(f"L{l}={v:.3f}" for l, v in D.items()))
    da = TD[TD.is_assoc]
    print(f"assoc-class fraction ratio<0.8: {(da.ratio<0.8).mean():.3f}, "
          f"<0.5: {(da.ratio<0.5).mean():.3f}")
    top = TD.nlargest(3, "gap")
    print("largest absolute gaps (U_y-T_y):")
    print(top[["seed", "cluster", "layer", "fact", "y", "T_y", "U_y",
               "gap"]].round(4).to_string(index=False))

    print("\n== C. q=0.99 Jaccard vs same-size random reference (§E4.14.2)")
    diag = pd.read_csv(RES + "/extension_support_diag.csv")
    print(f"ext diag (extension-store masks): cross-factor J median="
          f"{diag.cross_factor_jaccard_q99.median():.3f} p90="
          f"{diag.cross_factor_jaccard_q99.quantile(.9):.3f}; self-J median="
          f"{diag.self_jaccard_95_99.median():.3f}; mean size 0.95->0.99: "
          f"{diag['size_q0.95'].mean():.0f}->{diag['size_q0.99'].mean():.0f}, "
          f"median density {diag['density_q0.95'].median():.3f}->"
          f"{diag['density_q0.99'].median():.3f} "
          f"(means {diag['density_q0.95'].mean():.3f}->"
          f"{diag['density_q0.99'].mean():.3f})")
    try:
        from common import metrics as M
        from common import seeds as S
        RC = []
        for _, row in fac[fac.input_condition == "output_aware"].iterrows():
            hmi = np.array(json.loads(row["hidden_mi_vector"]))
            vm = np.array(json.loads(row["valid_mask"]), dtype=bool)
            s_tie = int(S.derive_seed(int(row["corex_seed"]), "support_ties"))
            sup99 = M.nested_supports(hmi, [0.99], vm)[0]
            sup95 = M.nested_supports(hmi, [0.95], vm)[0]
            RC.append(dict(seed=int(row.model_seed), cluster=int(row.cluster),
                           layer=int(row.layer), nvalid=int(vm.sum()),
                           s99=len(sup99.indices),
                           s95=len(sup95.indices),
                           set99=frozenset(int(i) for i in sup99.indices),
                           set95=frozenset(int(i) for i in sup95.indices)))
        RC = pd.DataFrame(RC)
        selfj = RC.apply(lambda r: len(r.set95 & r.set99) / len(r.set99), axis=1)
        print(f"self-J (0.95 vs 0.99, exact masks): median {selfj.median():.3f}; "
              f"size-ratio s95/s99 median {(RC.s95/RC.s99).median():.3f}")
        obs, exp = [], []
        for _, g in RC.groupby(["seed", "cluster", "layer"]):
            rr = list(g.itertuples())
            for a in range(len(rr)):
                for b in range(a + 1, len(rr)):
                    ra, rb = rr[a], rr[b]
                    inter = len(ra.set99 & rb.set99)
                    uni = len(ra.set99 | rb.set99)
                    obs.append(inter / uni if uni else np.nan)
                    exp.append(jaccard_expected(ra.s99, rb.s99, ra.nvalid))
        print(f"within-fit cross-factor J: observed median {np.nanmedian(obs):.3f} "
              f"p90 {np.nanpercentile(obs, 90):.3f} | same-size random reference "
              f"median {np.nanmedian(exp):.3f} (n={len(obs)} pairs)")
        by_sl = collections.defaultdict(list)
        for r in RC.itertuples():
            by_sl[(r.seed, r.layer)].append(r)
        ccobs, ccexp = [], []
        for rr in by_sl.values():
            for a in range(len(rr)):
                for b in range(a + 1, len(rr)):
                    ra, rb = rr[a], rr[b]
                    if ra.cluster == rb.cluster:
                        continue
                    inter = len(ra.set99 & rb.set99)
                    uni = len(ra.set99 | rb.set99)
                    ccobs.append(inter / uni if uni else np.nan)
                    ccexp.append(jaccard_expected(ra.s99, rb.s99, ra.nvalid))
        print(f"cross-cluster same-layer J: observed median {np.nanmedian(ccobs):.3f} "
              f"| reference median {np.nanmedian(ccexp):.3f} (n={len(ccobs)})")
    except FileNotFoundError:
        print("  (valid_mask columns not present in this store — reference "
              "from extension_support_diag only)")

    print("\n== D. TC-magnitude (Finding 6, corrected)")
    j = lc_all.merge(fac[key4 + ["factor_tc"]], on=key4, how="left")
    out = []
    for layer in (1, 2, 3):
        for th in (0.6, 0.7, 0.8, 0.9):
            g = j[(j.layer == layer) & (j.threshold == th)]
            for seed, gs in g.groupby("model_seed"):
                rr = dict(layer=layer, th=th, seed=seed,
                          raw=sp(gs.factor_tc, gs.T_max),
                          part=partial_spearman(gs.factor_tc, gs.T_max,
                                                gs.support_size))
                out.append(rr)
            out.append(dict(layer=layer, th=th, seed=-1,
                            raw=sp(g.factor_tc, g.T_max),
                            part=partial_spearman(g.factor_tc, g.T_max,
                                                  g.support_size)))
        g = j[(j.layer == layer) & (j.threshold == th)]
    PO = pd.DataFrame(out)
    pooled = PO[PO.seed == -1]
    print("pooled per stratum (raw / partial Spearman):")
    for _, r in pooled.iterrows():
        print(f"  L{int(r.layer)} q={r.th}: raw={r.raw:+.3f} partial={r.part:+.3f}")
    per_seed = PO[PO.seed != -1].groupby(["layer", "th"])
    print("raw range:", round(PO[PO.seed == -1].raw.min(), 3), "-",
          round(PO[PO.seed == -1].raw.max(), 3),
          "| partial range:", round(PO[PO.seed == -1].part.min(), 3), "-",
          round(PO[PO.seed == -1].part.max(), 3))
    # size-quartile bands (pooled)
    print("within-size-quartile-band Spearman (pooled, median across bands):")
    for layer in (1, 2, 3):
        for th in (0.6, 0.7, 0.8, 0.9):
            g = j[(j.layer == layer) & (j.threshold == th)]
            q = pd.qcut(g.support_size, 4, duplicates="drop")
            vals = [sp(sub.factor_tc, sub.T_max)
                    for _, sub in g.groupby(q, observed=True)]
            print(f"  L{layer} q={th}: {np.nanmedian(vals):.3f}")

    print("\n== E. concordance reference (Finding 7)")
    rnd_all = abl[abl.method == "random"].merge(
        lc_all[key6 + ["dominant_logit"]].rename(columns={"dominant_logit":
                                                          "cstar"}),
        on=key6)
    for cond in ("output_aware", "hidden_only"):
        lrate = lc_all[lc_all.input_condition == cond].groupby(
            "layer").concordant.mean()
        rrate = rnd_all[rnd_all.input_condition == cond].groupby(
            "layer").concordant.mean()
        print(f"{cond}: LC " + ", ".join(f"L{l}={v:.3f}" for l, v in lrate.items())
              + " | matched-random-vs-c* " +
              ", ".join(f"L{l}={v:.3f}" for l, v in rrate.items()))

    print("\n== F. activity matching (removed magnitude ratios)")
    rmean = rnd_all.groupby(key6).mean_removed_activation_mag.mean().rename(
        "rnd_mag")
    am = lc_all.merge(rmean, on=key6)
    am["ratio"] = am.mean_removed_activation_mag / am.rnd_mag
    med = am.groupby(["input_condition", "layer", "threshold"]).ratio.median()
    print("stratum medians range:", round(med.min(), 3), "-", round(med.max(), 3))

    print("\n== G. factor heterogeneity (round-3 display)")
    fkey = fac.set_index(key4)
    for th in (0.8, 0.9):
        s = lc_all[(lc_all.input_condition == "output_aware")
                   & (lc_all.threshold == th)].copy()
        s["assoc_bits"] = [float(fkey.loc[(r.model_seed, r.cluster, r.layer,
                                           "output_aware", r.factor_index)]
                                 [f"logit_mi_{int(r.dominant_logit)}"])
                           for r in s.itertuples()]
        strong = s.T_max >= 0.10
        sel = s.H_input <= 0.50
        print(f"q={th} (n={len(s)}): strong&selective {(strong&sel).mean():.1%}, "
              f"strong&broad {(strong&~sel).mean():.1%}, weak {(~strong).mean():.1%}")
        for layer, g in s.groupby("layer"):
            st = g.T_max >= 0.10
            sg = g.H_input <= 0.50
            print(f"  L{layer}: s&sel {(st&sg).mean():.1%}, s&broad "
                  f"{(st&~sg).mean():.1%}, weak {(~st).mean():.1%}")
        if th == 0.8:
            print("  Spearman(assoc, T_max) =", round(sp(s.assoc_bits, s.T_max), 3),
                  "| Spearman(assoc, H_input) =", round(sp(s.assoc_bits, s.H_input), 3))
            s["q4"] = pd.qcut(s.assoc_bits, 4, labels=["Q1", "Q2", "Q3", "Q4"])
            print(s.groupby("q4", observed=True)[["T_max", "H_input"]]
                  .mean().round(3).to_string())

    print("\n== H. profile cohorts (levels vs excesses, paired contrast)")
    pc = pd.read_csv(RES + "/mnist_profile_cohorts.csv")
    for stat in ("a_mean", "flip_rate"):
        s = pc[(pc.stat == stat) & (pc.input_condition == "output_aware")]
        for c in ("own_cluster", "same_digit_other_clusters", "other_digits"):
            g = s[s.cohort == c].groupby("layer")
            print(f"{stat} {c}: LC level {g.lc.mean().min():.4f}-"
                  f"{g.lc.mean().max():.4f}, excess {g['diff'].mean().min():.4f}-"
                  f"{g['diff'].mean().max():.4f}")
    w = pc[pc.stat == "a_mean"].pivot_table(
        index=["model_seed", "cluster", "layer", "input_condition",
               "factor_index"], columns="cohort", values="diff")
    w = w.dropna(subset=["own_cluster", "same_digit_other_clusters"])
    w["contrast"] = w.own_cluster - w.same_digit_other_clusters
    for l in (1, 2, 3):
        s = w[(w.index.get_level_values("layer") == l)
              & (w.index.get_level_values("input_condition") == "output_aware")]
        ps = s.groupby(level=0).contrast.mean()
        print(f"own-vs-same-digit-other contrast L{l}: {s.contrast.mean():+.4f} "
              f"(across-seed SE {ps.std(ddof=1)/np.sqrt(5):.4f})")

    print("\n== I. within-digit contrast, mixed clusters")
    prof = load_all(RAW + "/*.profiles.parquet")
    plc = prof[prof.method == "local_corex"]
    prm = prof[prof.method == "random_mean"]
    m = plc.merge(prm, on=["model_seed", "cluster", "layer", "input_condition",
                           "eval_cluster", "true_class", "factor_index"],
                  suffixes=("_lc", "_rm"))
    m["diffc"] = m.a_mean_lc - m.a_mean_rm
    mix = {int(c) for c, v in comps.items() if np.sort(v)[::-1][1] >= 40}
    for c in sorted(mix):
        dd = dom_digit[c]
        for layer in (1, 2, 3):
            sub = m[(m.cluster == c) & (m.layer == layer)
                    & (m.input_condition == "output_aware")]
            parts = []
            for dgt in range(10):
                inm = (sub.eval_cluster == c) & (sub.true_class == dgt)
                otm = (~inm) & (sub.true_class == dgt)
                if not inm.any():
                    continue
                nin = sub[inm].n_images_lc.sum()
                dinm = (sub[inm].diffc * sub[inm].n_images_lc).sum() / max(nin, 1)
                doutm = (sub[otm].diffc * sub[otm].n_images_lc).sum() \
                    / max(sub[otm].n_images_lc.sum(), 1) if otm.any() else np.nan
                parts.append(f"d{dgt}: in={dinm:.3f} out={doutm:.3f}")
            print(f"cluster {c} (dom {dd}) L{layer}: " + "; ".join(parts))

    print("\n== J. quote checks (Findings 1-2)")
    key = key6
    lcH2 = lc_all[(lc_all.input_condition == "output_aware") & (lc_all.layer == 2)
                  & (lc_all.threshold == 0.9)]
    rm = abl[(abl.method == "random") & (abl.layer == 2) & (abl.threshold == 0.9)
             & (abl.input_condition == "output_aware")].groupby(
        key).T_max.mean().rename("rnd").reset_index()
    mm = lcH2.merge(rm, on=key)
    per_seed = mm.groupby("model_seed").apply(lambda x: (x.T_max - x.rnd).mean())
    print("per-seed T_max excess H2 q=0.9:",
          [round(v, 3) for v in per_seed.values])
    mx = 0.0
    for f in sorted(glob.glob(RAW + "/*.ablation_summary.parquet")):
        d = pd.read_parquet(f, columns=["method", "delta_matrix"])
        for _, r in d[d.method == "local_corex"].iterrows():
            mx = max(mx, float(np.abs(np.array(json.loads(r["delta_matrix"]))).max()))
    print("largest single-cell |shift| over LC rows:", round(mx, 4))
    sup = load_all(RAW + "/*.supports.parquet",
                   ["layer", "threshold", "support_size", "support_density"])
    g = sup.groupby(["threshold", "layer"]).agg(dens=("support_density", "mean"),
                                                sz=("support_size", "mean"))
    print(g.round(3).to_string())

    print("\n== K. extension aggregates + univariate diffs (§E4.14.2)")
    e = ext[(ext.method == "local_corex") & (ext.input_condition == "output_aware")]
    print(e.groupby(["threshold", "layer"]).agg(
        T_max=("T_max", "mean"), H_in=("H_input", "mean"),
        prev=("substantial_class_concentrated", "mean")).round(3).to_string())
    uv = ext[(ext.method == "univariate") & (ext.input_condition == "output_aware")]
    mk = ["model_seed", "cluster", "layer", "factor_index", "threshold"]
    d = e.set_index(mk)[["T_max", "H_input"]].rename(
        columns={"T_max": "lc_T", "H_input": "lc_H"}).join(
        uv.set_index(mk)[["T_max", "H_input"]].rename(
            columns={"T_max": "uv_T", "H_input": "uv_H"})).dropna().reset_index()
    gg = d.groupby(["threshold", "layer"]).apply(
        lambda x: pd.Series({"dT": (x.lc_T - x.uv_T).mean(),
                             "dH": (x.lc_H - x.uv_H).mean()}))
    print(gg.round(4).to_string())


def refit_check():
    """Deterministic refit of the 300 output-aware fit conditions: (1) the
    stored 6-decimal hidden-MI vectors reproduce exactly; (2) supports at
    q in {0.95, 0.99} from full-precision MI vs the stored rounded vectors."""
    import warnings
    import torch
    from benchmark import (iter_cluster_layer_contexts, mapped_supports,
                           fit_and_rank)
    from common import seeds as S
    from data import load_splits
    from train import load_model
    from common.store import ResultStore
    from common.config import RAW_DIR
    store = ResultStore("mnist")
    splits = load_splits(save_dir=RAW_DIR / "mnist")
    labels_cluster = pd.read_parquet(store.shard_path(
        {"stage": "clusters", "phate_seed": 5200, "kmeans_seed": 5201,
         "k": 20}, "clusters")).set_index("discovery_index")["cluster"] \
        .sort_index().to_numpy()
    cfg = {"n_clusters": 20, "layers": [1, 2, 3],
           "support_thresholds": [0.95, 0.99]}
    device = "cuda" if torch.cuda.is_available() else "cpu"
    fidx = {}
    for f in sorted(glob.glob(RAW + "/*.factors.parquet")):
        dd = pd.read_parquet(f)
        for _, r in dd[dd.input_condition == "output_aware"].iterrows():
            fidx[(int(r.model_seed), int(r.cluster), int(r.layer),
                  int(r.factor_index))] = json.loads(r["hidden_mi_vector"])
    n_cond = n_fac = n_sup_diff = fac_mism = 0
    max_vdiff = 0.0
    diffs = []
    for seed in (5100, 5101, 5102, 5103, 5104):
        model = load_model(seed, device=device)
        for cluster, layer, ctx in iter_cluster_layer_contexts(
                model, splits, labels_cluster, cfg, device):
            n_cond += 1
            s_fit = int(S.derive_seed(seed, "corex", cluster, layer,
                                      "output_aware"))
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("ignore")
                fr = fit_and_rank(ctx.H_std, ctx.L_std, 10, s_fit, 5,
                                  valid_h=ctx.valid_h)
            s_tie = int(S.derive_seed(s_fit, "support_ties"))
            for j in fr["order"]:
                n_fac += 1
                hmis = fr["hidden_mis"][j]
                stored = np.array(fidx[(seed, cluster, layer, int(j))])
                vd = float(np.abs(np.round(hmis, 6) - stored).max())
                if vd > 1e-9:
                    fac_mism += 1
                max_vdiff = max(max_vdiff, vd)
                sf = mapped_supports(hmis, [0.95, 0.99], ctx.valid_h, s_tie)
                sr = mapped_supports(np.round(hmis, 6), [0.95, 0.99],
                                     ctx.valid_h, s_tie)
                for q, a, b in zip((0.95, 0.99), sf, sr):
                    if set(a.indices) != set(b.indices):
                        n_sup_diff += 1
                        diffs.append((seed, cluster, layer, int(j), q,
                                      len(set(a.indices) ^ set(b.indices))))
    print(f"refit: conditions={n_cond} factors={n_fac}")
    print(f"stored-vector mismatches: {fac_mism} "
          f"(max |rounded refit - stored| = {max_vdiff:.2e})")
    print(f"supports differing full vs 6-dec-rounded: {n_sup_diff}")
    for dset in diffs:
        print("  ", dset)


if __name__ == "__main__":
    _args = sys.argv[1:]
    if any(a in ("--refit", "--refit-only") for a in _args):
        refit_check()
    if "--refit-only" not in _args:
        main()