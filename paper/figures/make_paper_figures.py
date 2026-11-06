"""Generate the manuscript figures (figure_1.png ... figure_8.png).

Run with the project venv from the repository root:
    .venv/bin/python paper/figures/make_paper_figures.py

Every quantitative panel is computed from committed result files. The
commit-frozen E2 primary-convention absolutes and the E5 case-study panels
reuse numbers fixed in the committed accounts of record
(`paper/claim_evidence_table.md`, `paper/exp5.md` Sec. E5.6) and are labeled
as such. The dissertation-example panels in figure_7 come from the
dissertation chapter PDF (Chapter 2), shipped as
`paper/figures/assets/diss_*.png`.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent
ASSETS = OUT / "assets"

plt.rcParams.update({
    "figure.dpi": 200,
    "savefig.bbox": "tight",
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
})

C = {"oracle": "#1b9e77", "phate": "#7570b3", "random": "#8c8c8c",
     "global": "#d95f02", "local": "#1b9e77", "lc": "#1b9e77",
     "uni": "#e7298a", "r": "#8c8c8c", "gl": "#d95f02"}


# ---------------------------------------------------------------- figure 1
def figure_1():
    fig, ax = plt.subplots(figsize=(11.0, 3.7))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 33)

    # zone backgrounds: discovery (selection) vs evaluation (outcomes)
    ax.add_patch(plt.Rectangle((0.0, 8), 79.0, 23.0, fc="#f4f6f1",
                               ec="0.5", lw=0.7, ls="--", zorder=0))
    ax.add_patch(plt.Rectangle((80.0, 8), 19.0, 23.0, fc="#fdf3ef",
                               ec="0.5", lw=0.7, ls="--", zorder=0))
    ax.text(39.5, 32.3, "SELECTION — discovery data only", ha="center",
            fontsize=7.6, color="0.3", weight="bold")
    ax.text(89.5, 32.3, "EVALUATION — held-out cohorts", ha="center",
            fontsize=7.6, color="0.3", weight="bold")

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="0.2",
                                   lw=1.0, zorder=2))
        ax.text(x + w / 2, y + h - 2.0, title, ha="center", va="top",
                fontsize=8.6, weight="bold", zorder=3)
        ax.text(x + w / 2, y + h - 6.8, body, ha="center", va="top",
                fontsize=6.8, zorder=3)

    def arrow(x0, x1, y=20):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color="0.2", lw=1.3))

    box(1, 8, 17.5, 22, "Observations & variables",
        "held-out activations h\n(rows = observations,\ncolumns = variables)\n(+ logits y = Wh + b)", "#eef6ef")
    box(20.9, 8, 17.5, 22, "Partition\n(partitions rows)",
        "PHATE embedding +\nk-means on observations:\nregions / discovery\nclusters", "#eeeaf6")
    box(40.8, 8, 17.5, 22, "Local CorEx\n(groups columns)",
        "one fit per region\ngroups VARIABLES into\nfactors (top-k kept);\nglobal fit for comparison", "#fdeee2")
    box(60.7, 8, 17.5, 22, "Factor supports",
        "per factor, cumulative-MI\nthreshold q over variables;\nrandom / univariate ctl.", "#e2e9f3")
    box(80.6, 8, 18.4, 22, "Interventions",
        "zero the support columns\n(exact masked update;\nImageNet: Δz = −W[:,S]h_S)\non held-out cohorts", "#f6e2e2")

    for x0, x1 in [(19.2, 20.7), (39.1, 40.6), (59.0, 60.5), (78.9, 80.4)]:
        arrow(x0, x1)
    down = ("Readout: per-input-class TV shift of the class-mean prediction vector\n"
            "(T$_{max}$ = max over input classes), input-class concentration\n"
            "H$_{input}$, localization contrast L$_j$, vs. size-matched random & univariate controls")
    ax.annotate(down, xy=(89.8, 8), xytext=(47, -6),
                ha="center", va="top", fontsize=7.2,
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.0))
    fig.savefig(OUT / "figure_1.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def figure_2():
    """E1: per-condition localization deltas (the operating conditions),
    plus the fixed-collection coverage / matched-pair precision breakdown
    of the equal-budget comparison."""
    s = pd.read_csv(ROOT / "tnnls_human_experiments/gaussian/ablation_data"
                         "/summaries/gaussian_summary.csv")
    p = s[s.experiment == "primary"]

    # per-condition F1 by method, then paired deltas vs global
    cond = p.pivot_table(
        index=["covariance_setup", "cluster_size", "alpha"],
        columns="method", values="matched_top_k_f1")
    setups = ["homogeneous", "nondisjoint", "disjoint"]
    sizes = [50, 500, 5000]
    alphas = [0.0, 0.5, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(12.2, 4.0),
                             gridspec_kw={"width_ratios": [1, 1, 1.25]})

    def heat(ax, comp):
        mat = np.full((len(setups) * len(sizes), len(alphas)), np.nan)
        labels = []
        for i, (su, sz) in enumerate([(su, sz) for su in setups
                                      for sz in sizes]):
            labels.append(f"{su[:5]}\nn={sz}")
            for j, a in enumerate(alphas):
                if (su, sz, a) in cond.index:
                    mat[i, j] = cond.loc[(su, sz, a), comp] - \
                        cond.loc[(su, sz, a), "global"]
        im = ax.imshow(mat, cmap="RdBu_r", aspect="auto",
                       norm=matplotlib.colors.TwoSlopeNorm(0, -0.5, 0.5))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    v = mat[i, j]
                    ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                            fontsize=6.5,
                            color="white" if v > 0.35 else "0.1")
        ax.set_xticks(range(len(alphas)))
        ax.set_xticklabels([f"α={a:g}" for a in alphas])
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=6.5)
        return im

    im = heat(axes[0], "phate")
    axes[0].set_title("a  PHATE − global", fontsize=9.5)
    heat(axes[1], "oracle")
    axes[1].set_title("b  true-region reference − global", fontsize=9.5)
    axes[1].set_yticklabels([])

    cax = fig.colorbar(im, ax=axes[1], fraction=0.04, pad=0.03)
    cax.set_label("paired ΔF1 (centered at 0)", fontsize=7.5)
    cax.ax.tick_params(labelsize=7)

    # (c) coverage (fixed pooled collection) vs matched-pair precision,
    # recomputed from the store
    cap = pd.read_csv(ROOT / "tnnls_human_experiments/gaussian/capacity_data"
                           "/global_capacity/global_capacity_combined.csv")
    cap = cap[cap.covariance_setup != "homogeneous"]
    cap["n_groups"] = cap.num_matched_factors + cap.num_unmatched_true_groups
    cap["coverage_f1"] = (cap.matched_top_k_f1 * cap.num_matched_factors
                          / cap.n_groups)

    ax = axes[2]
    setups2 = ["disjoint", "nondisjoint"]
    bw = 0.26
    offs = [-0.75, -0.45, 0.0, 0.30, 0.62]
    centers = [0.0, 1.95]
    kind_map = [
        ("gM5cov", "coverage (fixed 10 groups)", C["gl"], ""),
        ("gM5prc", "matched-pair precision\n(of the M=5 average)", "#f0b27a",
         "//"),
        ("gM10cov", None, C["gl"], ""),
        ("gM10prc", None, "#f0b27a", "//"),
        ("loccov", "local stack (2 × M=5)\n10 candidates", C["local"], ""),
    ]
    seen = set()
    for i, su in enumerate(setups2):
        cx = centers[i]
        for (kind, lbl, col, hatch), off in zip(kind_map, offs):
            row = cap[(cap.covariance_setup == su)
                      & (cap.scope == ("local_pooled_capacity"
                                       if kind.startswith("loc")
                                       else "global_capacity"))
                      & (cap.budget == (5 if kind.endswith(("gM5cov",
                                                            "gM5prc",
                                                            "loccov"))
                                        else 10))]
            val = (row.coverage_f1.mean() if kind.endswith("cov")
                   else row.matched_top_k_f1.mean())
            plot_lbl = lbl if lbl and lbl not in seen else None
            if plot_lbl:
                seen.add(lbl)
            xpos = cx + off
            ax.bar(xpos, val, width=bw * 0.9, color=col, ec="0.2", lw=0.6,
                   hatch=hatch, label=plot_lbl)
            va_txt = f"{val:.2f}".replace("0.", ".")
            if kind.endswith("prc") or kind.startswith("loc"):
                inside = val > 0.75
                ax.text(xpos, val - 0.03 if inside else val + 0.02,
                        va_txt, ha="center", fontsize=6.0, color="0.15",
                        va="top" if inside else "bottom")
            else:
                ax.text(xpos, val + 0.02, va_txt, ha="center", fontsize=6.0)
        ax.text(cx, 1.1, su + "\n(10 pooled groups)", ha="center",
                fontsize=7, va="bottom")
    ax.set_xticks([])
    ax.set_ylim(0, 1.09)
    ax.set_ylabel("top-d recovery F1")
    ax.set_title("c  equal-candidate comparison:\n"
                 "coverage vs matched-pair scoring", fontsize=9)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), fontsize=6.4,
              ncol=3)
    ax.set_xlim(-1.05, 2.65)

    fig.tight_layout(rect=[0.02, 0.07, 1, 1])
    fig.savefig(OUT / "figure_2.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def _e2_paired(path, gen, corex_input=None):
    """Per-(true-group) best top-d-match F1 convention, recomputed from a
    committed combined CSV: per replicate, each true group takes its best
    candidate factor; per-replicate value = mean over groups; paired
    partition-minus-global deltas across the 16 seeds of each condition.
    corex_input filters to ONE preprocessing variant before matching
    (round-3 review: pooling variants would let each group take its best
    of two variants' factors)."""
    m = pd.read_csv(path)
    m = m[m.generator == gen]
    if corex_input is not None:
        m = m[m.corex_input == corex_input]
    per = m.groupby(["n_per_region", "alpha", "noise", "n_nuisance",
                     "estimator", "root_seed", "partition_method",
                     "region", "true_group"], as_index=False)["topd_f1"].max()
    piv = per.pivot_table(
        index=["n_per_region", "alpha", "noise", "estimator", "root_seed"],
        columns="partition_method", values="topd_f1", aggfunc="mean")
    rows = []
    for key, grp in piv.groupby(level=[0, 1, 2, 3]):
        n, alpha, noise, est = key
        for comp in ["oracle", "phate"]:
            d = grp[comp] - grp["global"]
            se = d.std(ddof=1) / np.sqrt(len(d))
            rows.append(dict(gen=gen, n=n, alpha=alpha, noise=noise,
                             est=est, comp=comp, d=d.mean(), se=se,
                             lo=d.mean() - 1.96 * se,
                             hi=d.mean() + 1.96 * se, pos=(d > 0).sum(),
                             nseed=len(d)))
    return pd.DataFrame(rows)


def _e2_frozen():
    """Reproduce the notebook's frozen PRIMARY convention exactly
    (`analysis_core_grids.ipynb`, cell 15): per-(seed, alpha, noise,
    region, true_group) best top-d match over ALL candidate factors;
    averaged per condition within seed across all alpha x noise levels;
    seed-level pairing with t(15) = 2.131. Bio rows are scored on the
    primary input variant; linear rows are unfiltered (reference only)."""
    cont = pd.read_csv(ROOT / "tnnls_human_experiments/AI_implemented/"
                            "exp2_nonlinear/nonlinear_results/"
                            "continuous_core/continuous_core_matches_combined.csv")
    disc = pd.read_csv(ROOT / "tnnls_human_experiments/AI_implemented/"
                            "exp2_nonlinear/nonlinear_results/"
                            "discrete_core/discrete_core_matches_combined.csv")
    # explicit generator filter (round-3 review): never rely on the table
    # holding a single generator
    cont = cont[cont.generator == "sine_sum"]
    disc = disc[disc.generator == "parity"]

    def bestmatch(df, pm, est, corex_input, n):
        s = df[(df.partition_method == pm) & (df.estimator == est)]
        if corex_input is not None:
            s = s[s.corex_input == corex_input]
        s = s[s.n_per_region == n]
        keys = ["root_seed", "alpha", "noise", "region", "true_group"]
        best = s.groupby(keys).agg(topd_f1=("topd_f1", "max"),
                                   cosine=("cosine", "max")).reset_index()
        agg = (best.groupby(["root_seed", "alpha", "noise"])
                   .agg(topd_f1=("topd_f1", "mean"),
                        cosine=("cosine", "mean"))
                   .groupby("root_seed").mean())
        return agg

    out = []
    for gen, df, inp in (("sine_sum", cont, "binned_with_routing"),
                         ("parity", disc, "with_routing")):
        for n in (500, 5000):
            o = bestmatch(df, "oracle", "bio", inp, n)
            gl = bestmatch(df, "global", "bio", inp, n)
            li = bestmatch(df, "oracle", "linear", None, n)
            for metric in ("topd_f1", "cosine"):
                d = (o[metric] - gl[metric]).dropna()
                se = d.std(ddof=1) / np.sqrt(len(d))
                t = 2.131
                out.append(dict(gen=gen, n=n, metric=metric,
                                oracle=o[metric].mean(),
                                global_=gl[metric].mean(),
                                linear_oracle=li[metric].mean(),
                                d=d.mean(), lo=d.mean() - t * se,
                                hi=d.mean() + t * se, se=se,
                                nseed=len(d)))
    return pd.DataFrame(out)


def figure_3():
    """E2: per-condition recalculation (both estimators); the frozen
    PRIMARY-convention paired deltas, now computed rather than hard-coded
    and labeled by estimator; the EM capability diagnostic."""
    cont = _e2_paired(ROOT / "tnnls_human_experiments/AI_implemented/"
                           "exp2_nonlinear/nonlinear_results/continuous_core/"
                           "continuous_core_matches_combined.csv",
                      "sine_sum", corex_input="binned_with_routing")
    disc = _e2_paired(ROOT / "tnnls_human_experiments/AI_implemented/"
                           "exp2_nonlinear/nonlinear_results/discrete_core/"
                           "discrete_core_matches_combined.csv", "parity",
                      corex_input="with_routing")
    frz = _e2_frozen()

    lp = pd.read_csv(ROOT / "tnnls_human_experiments/exp3_redundancy/"
                          "redundancy_results/parity_capability/"
                          "parity_capability_summary.csv")

    fig = plt.figure(figsize=(11.5, 6.6))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.35, 1], hspace=0.55,
                          wspace=0.28)

    # -- (a) 2x2 grid: per-condition paired deltas (n=5000, noise 0.05)
    gs_a = gs[0, 0:2].subgridspec(2, 2, wspace=0.25, hspace=0.55)
    for gi, (gen, genname) in enumerate([("sine_sum", "sine-sum"),
                                         ("parity", "parity")]):
        for ei, est in enumerate(["Linear", "Bio"]):
            ax = fig.add_subplot(gs_a[gi, ei])
            src = cont if gen == "sine_sum" else disc
            sub = src[(src.n == 5000) & (src.noise == 0.05)
                      & (src.est == est.lower())]
            xs = np.arange(3)
            w = 0.38
            t95 = 2.131  # 95% t CI, df = 15
            for ci, (comp, col, lbl) in enumerate(
                    [("oracle", C["oracle"], "true-region − global"),
                     ("phate", C["phate"], "PHATE − global")]):
                s2 = sub[sub.comp == comp].sort_values("alpha")
                xs_ = xs + (ci - 0.5) * w
                se = s2.se
                ax.bar(xs_, s2.d, yerr=t95 * se, width=w, color=col,
                       ec="0.2", lw=0.6, capsize=3,
                       label=lbl if (gi == 0 and ei == 0) else None)
                for x_, (_, r) in zip(xs_, s2.iterrows()):
                    npos = f"{int(r.pos)}/{int(r.nseed)}"
                    ax.text(x_, (r.d + t95 * r.se if r.d >= 0
                                 else r.d - t95 * r.se)
                            + (0.010 if r.d >= 0 else -0.026),
                            npos, ha="center", fontsize=5.4, color="0.35")
            ax.axhline(0, color="0.45", lw=0.8)
            ax.set_xticks(xs)
            ax.set_xticklabels(["α=0", "α=0.5", "α=1"])
            ax.set_title(f"{genname} · {est}", fontsize=8.5)
            if ei == 0:
                ax.set_ylabel("paired ΔF1 vs global\n(95% CI, t$_{15}$)")

    # shared partition legend for the (a) grid (round-3 review)
    h_a, l_a = fig.axes[0].get_legend_handles_labels()
    fig.legend(h_a, l_a, loc="upper left", bbox_to_anchor=(0.075, 0.965),
               fontsize=6.8, ncol=2, frameon=False)

    # -- (b) the frozen PRIMARY convention, now computed from the notebook's
    # definition (Bio candidate pool; equal-condition weighting; t(15)).
    ax = fig.add_subplot(gs[0, 2])
    sel = frz[(frz.n == 5000)].copy().reset_index(drop=True)
    order = {"sine_sum": 0, "parity": 1}
    sel["ord"] = sel.gen.map(order)
    sel = sel.sort_values(["ord", "metric"], ascending=[True, False])
    sel = sel.reset_index(drop=True)
    ys = np.arange(len(sel))
    ax.barh(ys - 0.18, sel.d, xerr=[sel.d - sel.lo, sel.hi - sel.d],
            height=0.34, color="#1b9e77", ec="0.2", lw=0.6, capsize=2,
            label="paired Δ (Bio, 95% CI, t$_{15}$)")
    for y, (_, r) in zip(ys, sel.iterrows()):
        lbl = (f"{r.gen}, n=5000, "
               f"{'F1' if r.metric == 'topd_f1' else 'cosine'} : "
               f"{r.oracle:.2f} vs {r.global_:.2f}")
        ax.text(-0.128, y, lbl, ha="left", va="bottom", fontsize=6.4)
    ax.axvline(0, color="0.45", lw=0.8)
    ax.set_yticks([])
    ax.set_xlim(-0.13, 0.30)
    ax.set_xlabel("paired Δ: true-region − global (Bio)")
    ax.set_title("b  primary convention\n(Bio; t$_{15}$)", fontsize=9)
    ax.legend(fontsize=6.4, loc="center right")

    # -- (c) capability diagnostic
    ax = fig.add_subplot(gs[1, :])
    w = 0.36
    grp_x = {500: [0.0, 1.0, 2.0], 5000: [3.4, 4.4, 5.4]}
    for k, (nsv, lbl) in enumerate([(500, "n = 500"),
                                    (5000, "n = 5000")]):
        sb2 = lp[lp.n_samples == nsv].sort_values("d")
        base = np.array(grp_x[nsv])
        ax.bar(base - w / 2, sb2.seeded_init_tc_bits,
               yerr=sb2.seeded_init_se, width=w, color="#3b6fb6",
               ec="0.2", lw=0.6, capsize=2.5,
               label="atom-seeded init (uses ground truth)" if k == 0
               else None)
        ax.bar(base + w / 2, sb2.std_init_tc_bits, yerr=sb2.std_init_se,
               width=w, color=C["gl"], ec="0.2", lw=0.6, capsize=2.5,
               label="default EM init" if k == 0 else None)
        for j, (_, r) in enumerate(sb2.iterrows()):
            ax.text(base[j] - w / 2, r.seeded_init_tc_bits + 0.015,
                    f"{r.seeded_init_tc_bits:.2f}", ha="center", fontsize=6)
            ax.text(base[j] + w / 2, max(r.std_init_tc_bits, 0.02) / 2,
                    f"{r.std_init_tc_bits:.2f}", ha="center", fontsize=6,
                    color="white" if r.std_init_tc_bits > 0.25 else "0.25")
        ax.text(base.mean(), 0.685, lbl, fontsize=8.5, weight="bold",
                ha="center")
    ax.axhline(0.5, color="0.4", ls=":", lw=1)
    ax.text(-0.6, 0.522, "analytic TC = 0.500", fontsize=7, ha="left",
            color="0.3", va="bottom")
    ax.set_xticks([0, 1, 2, 3.4, 4.4, 5.4])
    ax.set_xticklabels(["d = 3", "d = 4", "d = 5", "d = 3", "d = 4",
                        "d = 5"])
    ax.set_ylabel("fitted top-factor TC (bits)")
    ax.set_ylim(0, 0.76)
    ax.set_title("c  EM-fitted parity capability diagnostic: the model family represents"
                 " the structure; default-EM optimization collapses at n = 5000",
                 fontsize=9)
    ax.legend(fontsize=7, loc="upper center", ncol=2)
    ax.set_xlim(-0.7, 6.1)

    fig.savefig(OUT / "figure_3.png")
    plt.close(fig)


def figure_4():
    c = pd.read_csv(ROOT / "tnnls_human_experiments/exp3_redundancy/"
                         "redundancy_results/core/core_info_summary.csv")
    main = c[(c.n_samples == 5000) & (c.dim_hidden > 0)]
    red = main[main.generator == "redundant"].sort_values("d")
    par = main[main.generator == "synergistic"].sort_values("d")

    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.0))
    ax = axes[0]
    xs = np.arange(len(red))
    w = 0.36
    ax.bar(xs - w / 2, red.top_factor_tc_mean,
           yerr=[red.top_factor_tc_mean - red.top_factor_tc_ci_low,
                 red.top_factor_tc_ci_high - red.top_factor_tc_mean],
           width=w, color="#1b9e77", ec="0.2", lw=0.6, capsize=3,
           label="redundancy-dominated")
    ax.bar(xs + w / 2, par.top_factor_tc_mean,
           yerr=[par.top_factor_tc_mean - par.top_factor_tc_ci_low,
                 par.top_factor_tc_ci_high - par.top_factor_tc_mean],
           width=w, color="#d95f02", ec="0.2", lw=0.6, capsize=3,
           label="synergy-dominated (parity)")
    ax.axhline(0.5, color="0.4", ls=":", lw=1)
    ax.text(len(red) - 0.4, 0.512, "matched population TC = 0.500",
            fontsize=7, ha="right", color="0.3")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"d = {d}" for d in red.d])
    ax.set_ylabel("fitted top-factor TC (bits)")
    ax.set_ylim(0, 0.62)
    ax.legend(loc="center right", fontsize=7)
    ax.set_title("a  same population TC, different fitted TC\n"
                 "(fitted TC tracks discoverability, not type)")

    ax = axes[1]
    xs = np.arange(len(red))
    ax.bar(xs - w / 2, red.omega_on_support_char_mean, width=w,
           color="#1b9e77", ec="0.2", lw=0.6,
           label="redundant: $\\Omega$ on discovered support")
    ax.bar(xs + w / 2, par.omega_on_support_char_mean, width=w,
           color="#d95f02", ec="0.2", lw=0.6,
           label="parity: $\\Omega$ on discovered support")
    for x, (_, r), (_, p_) in zip(xs, red.iterrows(), par.iterrows()):
        ax.plot(x - w / 2, r.omega_true_group_char_mean, "v", color="#0b5d44",
                ms=5, zorder=4)
        ax.plot(x + w / 2, p_.omega_true_group_char_mean, "v", color="#8a3c00",
                ms=5, zorder=4)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.plot([], [], "v", color="0.2", label="$\\Omega$ on true group")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"d = {d}" for d in red.d])
    ax.set_ylabel("$\\Omega$ (bits, signed)")
    ax.legend(loc="lower left", fontsize=6.5)
    ax.set_title("b  $\\Omega$ instrument: population-level right where\n"
                 "recovery works; support-recovery error propagates\n"
                 "into the characterization (no failure warning)")

    fig.tight_layout()
    fig.savefig(OUT / "figure_4.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def figure_5():
    rnd = pd.read_csv(ROOT / "tnnls_human_experiments/exp4_mnist_benchmark/"
                           "results/mnist_lc_vs_random.csv")
    uni = pd.read_csv(ROOT / "tnnls_human_experiments/exp4_mnist_benchmark/"
                           "results/mnist_lc_vs_univariate.csv")
    prev = pd.read_csv(ROOT / "tnnls_human_experiments/exp4_mnist_benchmark/"
                            "results/mnist_prevalence.csv")

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.1))
    qs = sorted(rnd.threshold.unique())
    layers = [1, 2, 3]
    layer_names = {1: "H1", 2: "H2", 3: "H3"}
    lcolors = {1: "#1b9e77", 2: "#7570b3", 3: "#d95f02"}

    spec = [
        (rnd, "T_max", "a  magnitude vs matched random",
         "paired $\\Delta$T$_{max}$ (LC $-$ random)"),
        (prev, "prevalence", "b  concentrated effects by layer",
         "conjunctive prevalence"),
        (uni, "T_max", "c  vs univariate control",
         "paired $\\Delta$T$_{max}$ (LC $-$ univariate)"),
        (rnd, "flip_rate", "d  per-image flip rate (all test images)",
         "paired $\\Delta$flip rate (LC $-$ random)"),
    ]
    for e, (df, metric, title, ylabel) in enumerate(spec):
        ax = axes[e]
        for l in layers:
            if df is prev:
                sub = prev[prev.layer == l].sort_values("threshold")
                y, err = sub.prevalence, sub.se
            else:
                sub = df[(df.layer == l) & (df.metric == metric)].sort_values(
                    "threshold")
                y, err = sub.mean_diff, sub.se
            ax.errorbar(sub.threshold, y, yerr=2 * err, marker="o", ms=4,
                        color=lcolors[l], label=layer_names[l], capsize=2,
                        lw=1.3)
        if df is not prev:
            ax.axhline(0, color="0.5", lw=0.7)
        else:
            ax.set_ylim(-0.02, 0.45)
        ax.set_xticks(qs)
        ax.set_xlabel("support threshold q")
        ax.set_ylabel(ylabel)
        ax.legend(fontsize=7)
        ax.set_title(title, fontsize=8.5)

    fig.tight_layout()
    fig.savefig(OUT / "figure_5.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 6
def figure_6():
    """Direct local vs global comparison, same measurements, output-aware,
    global M = 10. Local absolutes: mnist_tc_density_magnitude.csv; global
    absolutes: global_arm_summary.csv."""
    loc = pd.read_csv(ROOT / "tnnls_human_experiments/exp4_mnist_benchmark/"
                           "results/mnist_tc_density_magnitude.csv")
    gsum = pd.read_csv(ROOT / "tnnls_human_experiments/exp4_mnist_benchmark/"
                            "results/global_arm_summary.csv")
    gcov = pd.read_csv(ROOT / "tnnls_human_experiments/exp4_mnist_benchmark/"
                            "results/global_arm_coverage.csv")

    gl = gsum[(gsum.method == "global_corex")
              & (gsum.input_condition == "output_aware") & (gsum.budget == 10)]
    grnd = gsum[(gsum.method == "random")
                & (gsum.input_condition == "output_aware")
                & (gsum.budget == 10)]
    layers = [1, 2, 3]
    layer_names = {1: "H1 (500 units)", 2: "H2 (400 units)",
                   3: "H3 (300 units)"}
    layer_units = {1: 500, 2: 400, 3: 300}
    lcolors = {1: "#1b9e77", 2: "#7570b3", 3: "#d95f02"}

    fig, axes = plt.subplots(1, 4, figsize=(13.5, 3.1))

    qs = [0.6, 0.7, 0.8, 0.9]

    # (a) T_max absolute at q = 0.9
    ax = axes[0]
    xs = np.arange(3)
    w = 0.36

    def loc_seed_err(col):
        """Local-arm per-seed means (5 network seeds) -> mean and 2*SE."""
        perseed = loc[loc.threshold == 0.9].groupby(
            ["layer", "model_seed"])[col].mean().reset_index()
        means, errs = [], []
        for l in layers:
            v = perseed[perseed.layer == l][col].to_numpy()
            means.append(v.mean())
            errs.append(2 * v.std(ddof=1) / np.sqrt(len(v)))
        return np.array(means), np.array(errs)

    lv, lv_err = loc_seed_err("T_max")
    gv = [gl[(gl.threshold == 0.9) & (gl.layer == l)].T_max.iloc[0]
          for l in layers]
    rv = [grnd[(grnd.threshold == 0.9) & (grnd.layer == l)].T_max.iloc[0]
          for l in layers]
    ax.bar(xs - w / 2, lv, yerr=lv_err, width=w, color=C["lc"], ec="0.2",
           lw=0.6, capsize=3, label="local CorEx (5 seeds)")
    ax.bar(xs + w / 2, gv, width=w, color=C["gl"], ec="0.2", lw=0.6,
           label="global CorEx (M = 10)")
    for i, l in enumerate(layers):
        ax.plot(i - w / 2, 0.0, "none")
        ax.plot(i + w / 2, rv[i], "v", color="0.25", ms=5, zorder=4,
                label="size-matched random" if i == 0 else None)
    for x, v in zip(xs - w / 2, lv):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.6)
    for x, v in zip(xs + w / 2, gv):
        ax.text(x, v + 0.01, f"{v:.2f}", ha="center", fontsize=6.6)
    ax.set_xticks(xs)
    ax.set_xticklabels([layer_names[l] for l in layers], fontsize=7)
    ax.set_ylabel("T$_{max}$ (class-mean prediction TV,\nmax over input classes)")
    ax.set_title("a  T$_{max}$, q = 0.9", fontsize=9)
    ax.legend(fontsize=6.4)
    ax.set_ylim(0, 0.9)

    # (b) H_input absolute at q = 0.9
    ax = axes[1]
    lv, lv_err_h = loc_seed_err("H_input")
    gv = [gl[(gl.threshold == 0.9) & (gl.layer == l)].H_input.iloc[0]
          for l in layers]
    rv = [grnd[(grnd.threshold == 0.9) & (grnd.layer == l)].H_input.iloc[0]
          for l in layers]
    ax.bar(xs - w / 2, lv, yerr=lv_err_h, width=w, color=C["lc"], ec="0.2",
           lw=0.6, capsize=3, label="local CorEx")
    ax.bar(xs + w / 2, gv, width=w, color=C["gl"], ec="0.2", lw=0.6,
           label="global CorEx (M = 10)")
    for i, l in enumerate(layers):
        ax.plot(i + w / 2, rv[i], "v", color="0.25", ms=5, zorder=4,
                label="size-matched random" if i == 0 else None)
    for x, v in zip(xs - w / 2, lv):
        ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.6)
    for x, v in zip(xs + w / 2, gv):
        ax.text(x, v + 0.012, f"{v:.2f}", ha="center", fontsize=6.6)
    ax.set_xticks(xs)
    ax.set_xticklabels(["H1", "H2", "H3"])
    ax.set_ylabel("H$_{input}$ (lower = more\ninput-class-concentrated)")
    ax.set_title("b  H$_{input}$, q = 0.9", fontsize=9)
    ax.legend(fontsize=6.4, loc="center right")
    ax.set_ylim(0, 1.08)

    # (c) support size as a fraction of the layer, by threshold
    ax = axes[2]
    for l in layers:
        sub = loc[loc.layer == l].groupby("threshold")["support_size"].mean()
        fr = [sub[q] / layer_units[l] for q in qs]
        ax.plot(qs, fr, marker="o", ms=4, color=lcolors[l],
                label=f"local {layer_names[l].split(' ')[0]}" if True else None,
                lw=1.3)
        gcov_l = gcov[(gcov.input_condition == "output_aware")
                      & (gcov.budget == 10) & (gcov.layer == l)].sort_values(
            "threshold")
        frg = gcov_l.support_mean.to_numpy() / layer_units[l]
        ax.plot(gcov_l.threshold, frg, marker="s", ms=4, ls="--",
                color=lcolors[l], alpha=0.75, lw=1.1)
    ax.plot([], [], "s", color="0.4", label="global (dashed)")
    ax.set_xticks(qs)
    ax.set_xticklabels([str(q) for q in qs])
    ax.set_xlabel("support threshold q")
    ax.set_ylabel("support size / layer units")
    ax.set_title("c  support size, fraction of layer\n(solid local, dashed global)",
                 fontsize=9)
    ax.legend(fontsize=6.4, loc="upper left")
    ax.set_ylim(0, 1.0)

    # (d) matched-volume comparison: global q vs local q chosen so pooled
    # support fractions agree (committed tables on both arms)
    ax = axes[3]
    pairs = [(0.7, 0.8), (0.8, 0.9)]
    xs = np.arange(6, dtype=float)
    w = 0.38
    for k, (gq, lq) in enumerate(pairs):
        for j, l in enumerate(layers):
            gv_ = gl[(gl.layer == l) & (gl.threshold == gq)].iloc[0]
            lc = loc[(loc.layer == l) & (loc.threshold == lq)]
            perseed = lc.groupby("model_seed")[["T_max", "H_input"]].mean()
            x = xs[k * 3 + j]
            ax.bar(x - w / 2, gv_.T_max, width=w, color=C["gl"], ec="0.2",
                   lw=0.6,
                   label="global (q=%.1f)" % gq if j == 0 and k == 0
                   else None)
            ax.bar(x + w / 2, perseed.T_max.mean(),
                   yerr=2 * perseed.T_max.std(ddof=1) / np.sqrt(5),
                   width=w, color=lcolors[l], ec="0.2", lw=0.6, capsize=2.5,
                   label="local (q=%.1f)" % lq if j == 0 and k == 0
                   else None)
            ax.text(x - w / 2, gv_.T_max + 0.008,
                    f"{gv_.T_max:.2f}", ha="center", fontsize=5.6)
            ax.text(x + w / 2, perseed.T_max.mean() + 0.008,
                    f"{perseed.T_max.mean():.2f}", ha="center", fontsize=5.6)
    # layer labels on the bottom axis
    labels = [f"H{l}" for _ in pairs for l in layers]
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=7)
    ax.axvline(2.5, color="0.8", lw=0.8, ls=":")
    ax.set_ylim(0, 0.46)
    ax.set_ylabel("T$_{max}$ at matched volume")
    ax.set_title("d  matched support-fraction pairs\n"
                 "(global q=0.7↔local q=0.8; global q=0.8↔local q=0.9)",
                 fontsize=8.4)
    ax.legend(fontsize=6.0, loc="upper left")

    fig.tight_layout()
    fig.savefig(OUT / "figure_6.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 7
def figure_7():
    """The single strongest worked example from the earlier analysis
    (dissertation Ch. 2, Fig. 2.12a). The other historical panels
    (cluster-deletion accuracy bars, DeepDream, African-grey logit
    shifts) move to the supplement / Appendix A assets."""
    path = ASSETS / "diss_fig2_12a.png"
    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    ax.axis("off")
    ax.imshow(plt.imread(path))
    ax.set_title("a  EARLIER 200-unit model, MNIST cluster-16, H1:\n"
                 "decoder reconstruction after MI-weighted perturbation\n"
                 "(top: factor 1, bottom: factor 2); right: per-logit MI",
                 fontsize=8.4, pad=6)
    fig.tight_layout()
    fig.savefig(OUT / "figure_7.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 8
def figure_8():
    def load(control):
        """Prefer the corrected (q = 0.80 matched-pairing) canonical tables
        written by check_e5_pairing_fix.py; fall back to the committed
        convention tables if absent."""
        fixed = ROOT / ("outputs/summaries/"
                        f"imagenet_systematic_lc_vs_{control}_q080.csv")
        base = ROOT / ("outputs/summaries/"
                       f"imagenet_systematic_lc_vs_{control}.csv")
        return pd.read_csv(fixed if fixed.exists() else base), \
            fixed.exists()

    sys_r, ok_r = load("random")
    sys_u, ok_u = load("univariate")
    loc = pd.read_csv(ROOT / "outputs/summaries/imagenet_localization_contrasts.csv")
    assert ok_r and ok_u, \
        "corrected pairing tables missing; run check_e5_pairing_fix.py"

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2))

    ax = axes[0]

    def get(df, stat):
        r = df[df.statistic == stat].iloc[0]
        return r.mean_diff, r.ci_low, r.ci_high

    stats = [("tv_source", "$\\Delta$TV\n(discovery region)"),
             ("localization_contrast",
              "$\\Delta$localization contrast\nL$_j$")]
    xs = np.arange(len(stats))
    w = 0.38
    for i, (s_, lbl) in enumerate(stats):
        m0, lo0, hi0 = get(sys_r, s_)
        m1, lo1, hi1 = get(sys_u, s_)
        ax.bar(i - w / 2, m0, yerr=np.array([[m0 - lo0], [hi0 - m0]]),
               width=w, color="#3b6fb6", ec="0.2", lw=0.6, capsize=3,
               label="LC $-$ matched random" if i == 0 else None)
        ax.bar(i + w / 2, m1, yerr=np.array([[m1 - lo1], [hi1 - m1]]),
               width=w, color="#e7298a", ec="0.2", lw=0.6, capsize=3,
               label="LC $-$ univariate" if i == 0 else None)
        ax.text(i - w / 2, hi0 + 0.008, f"{m0:+.2f}", ha="center", fontsize=7)
        ax.text(i + w / 2, m1 - 0.036, f"{m1:+.2f}", ha="center", fontsize=7,
                va="top")
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([lbl for _, lbl in stats], fontsize=7.5)
    ax.set_ylabel("paired difference (cluster bootstrap)")
    ax.set_ylim(-0.05, 0.335)
    ax.set_title("a  systematic survey: paired contrasts\n(282 cells, 94 clusters)", fontsize=9)
    ax.legend(fontsize=6.5, loc="upper left")

    ax = axes[1]
    lc = loc[loc.method == "local_corex"].localization_contrast
    rr = loc[loc.method == "random_mean_of_5"].localization_contrast
    ax.violinplot([rr.to_numpy(), lc.to_numpy()], showmedians=True,
                  widths=0.7)
    ax.set_xticks([1, 2])
    ax.set_xticklabels([f"matched random\n(median {rr.median():.3f})",
                        f"local CorEx\n(median {lc.median():.3f})"])
    ax.set_ylabel("localization contrast L$_j$")
    ax.set_title("b  effects concentrate near the\nfactor's discovery region", fontsize=9)

    # case-study panels (committed account-of-record values, exp5.md E5.6)
    ax = axes[2]
    STYLES = {"lc_a": ("#1b9e77", "LC factor A"),
              "lc_b": ("#1b9e77", "LC factor B"),
              "rand": ("#8c8c8c", "matched random (A)"),
              "rand2": ("#8c8c8c", "matched random (B)"),
              "uni": ("#e7298a", "univariate (A)"),
              "uni2": ("#e7298a", "univariate (B)")}
    owl = [(-0.86, "lc_a"), (-0.85, "lc_b"), (-0.05, "rand"),
           (-0.09, "rand2"), (-0.85, "uni"), (-0.70, "uni2")]
    grey = [(-0.28, "lc_a"), (-0.35, "lc_b"), (-0.08, "rand")]
    owl_off = np.linspace(-0.63, 0.63, 6)
    grey_off = [-0.45, -0.20, 0.30]
    used = set()
    for i, (vals, offs, center) in enumerate([(owl, owl_off, 0.0),
                                              (grey, grey_off, 1.9)]):
        tiny_count = 0
        for (v, style), pos_off in zip(vals, offs):
            pos = center + pos_off
            label = STYLES[style][1] if style not in used else None
            used.add(style)
            ax.bar(pos, v, width=0.22, color=STYLES[style][0], ec="0.2",
                   lw=0.4, label=label)
            if abs(v) < 0.12:
                ax.text(pos, v + 0.025, f"{v:.2f}", ha="center",
                        fontsize=6, va="bottom")
            else:
                ty = v - 0.03 - 0.05 * (tiny_count % 2)
                tiny_count += 1
                ax.text(pos, ty, f"{v:.2f}", ha="center", fontsize=6)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks([0.0, 1.9])
    ax.set_xticklabels(["great grey owl\n(cluster 21)",
                        "African grey\n(cluster 85, illustr.)"])
    ax.set_ylabel("$\\Delta$ class probability at q = 0.80")
    ax.set_xlim(-1.05, 3.05)
    ax.set_ylim(-1.02, 0.35)
    ax.set_title("c  case-study panels\n(record values; see §A)", fontsize=9)
    ax.text(1.9 + 0.30, -0.14, "univariate control\nnot recorded for\nthis panel",
            ha="center", fontsize=5.8, color="0.45")
    ax.legend(fontsize=6.2, loc="upper left", ncol=2, columnspacing=0.9)

    fig.tight_layout()
    fig.savefig(OUT / "figure_8.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
    figure_7()
    figure_8()
    print("figures written to", OUT)
