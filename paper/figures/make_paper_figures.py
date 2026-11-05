"""Generate the draft-manuscript figures (figure_1.png ... figure_6.png).

Run from the repository root with the project venv:
    .venv/bin/python paper/figures/make_paper_figures.py

Every panel is computed from committed result files. Panels whose underlying
per-replicate values were not committed as CSVs (E2 primary-convention
numbers, E5 case-study panels) reuse the numbers fixed in the committed
accounts of record (`paper/claim_evidence_table.md`, `paper/exp2 verdict`,
`paper/exp5.md` Sec. E5.6) and are labeled as such in the captions.
"""

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = pathlib.Path(__file__).resolve().parent

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
     "uni": "#e7298a", "r": "#8c8c8c"}


# ---------------------------------------------------------------- figure 1
def figure_1():
    fig, ax = plt.subplots(figsize=(11.0, 3.0))
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 30)

    def box(x, y, w, h, title, body, fc):
        ax.add_patch(plt.Rectangle((x, y), w, h, fc=fc, ec="0.2",
                                   lw=1.0, zorder=2))
        ax.text(x + w / 2, y + h - 2.4, title, ha="center", va="top",
                fontsize=9, weight="bold", zorder=3)
        ax.text(x + w / 2, y + h - 7.0, body, ha="center", va="top",
                fontsize=7.0, zorder=3)

    def arrow(x0, x1, y=20):
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color="0.2", lw=1.3))

    box(1, 8, 17.5, 22, "Neural activations",
        "held-out test images,\nhidden activations h\n(+ logits y = Wh + b)", "#eef6ef")
    box(20.9, 8, 17.5, 22, "Partition",
        "PHATE embedding +\nk-means: regions /\ndiscovery clusters", "#eeeaf6")
    box(40.8, 8, 17.5, 22, "Local CorEx",
        "one fit per region\n(top factors);\nglobal fit for comparison", "#fdeee2")
    box(60.7, 8, 17.5, 22, "Factor supports",
        "cumulative-MI\nthreshold q per factor;\nrandom / univariate ctl.", "#e2e9f3")
    box(80.6, 8, 18.4, 22, "Interventions",
        "delete support S\n(exact masked update)\non held-out cohorts", "#f6e2e2")

    for x0, x1 in [(19.2, 20.7), (39.1, 40.6), (59.0, 60.5), (78.9, 80.4)]:
        arrow(x0, x1)
    down = ("Readout: max per-class prob. change (effect size),\n"
            "class concentration, localization contrast L$_j$,\n"
            "vs. size-matched random & univariate controls")
    ax.annotate(down, xy=(89.8, 8), xytext=(47, -6),
                ha="center", va="top", fontsize=7.4,
                arrowprops=dict(arrowstyle="-|>", color="0.3", lw=1.0))
    fig.savefig(OUT / "figure_1.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 2
def _method_bars(ax, means, order, key, labels=None):
    xs = np.arange(len(order))
    for x, m in zip(xs, order):
        mu = means[m]
        err = means[f"{m}_se"]
        ax.bar(x, mu, yerr=err, width=0.65,
               color=C[key(m)], ec="0.2", lw=0.6, capsize=3)
        ax.text(x, mu + 0.02, f"{mu:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels if labels is not None else order)


def figure_2():
    s = pd.read_csv(ROOT / "tnnls_human_experiments/gaussian/ablation_data"
                         "/summaries/gaussian_summary.csv")
    p = s[s.experiment == "primary"]
    cond = p.pivot_table(index=["covariance_setup", "cluster_size", "alpha"],
                         columns="method", values="matched_top_k_f1")
    se = cond.std(ddof=1) / np.sqrt(len(cond))
    means = {m: cond[m].mean() for m in cond}
    for m in cond:
        means[f"{m}_se"] = se[m]

    cap = pd.read_csv(ROOT / "tnnls_human_experiments/gaussian/capacity_data"
                           "/global_capacity/global_capacity_combined.csv")
    capm = (cap.groupby(["covariance_setup", "budget", "scope"])
               ["matched_top_k_f1"].agg(["mean", "sem"]).reset_index())

    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

    ax = axes[0]
    _method_bars(ax, means, ["global", "random", "phate", "oracle"],
                 lambda m: "oracle" if m in ("oracle", "phate") else m,
                 labels=["global", "random", "data-driven\n(PHATE)", "oracle"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("top-d support recovery F1")
    ax.set_title("a  Linear Gaussian grids (27 conditions)")

    ax = axes[1]
    setups = ["disjoint", "nondisjoint"]
    budgets = [5, 10]
    xs = np.arange(len(setups) * len(budgets), dtype=float)
    w = 0.36
    for i, (su, b) in enumerate([(s_, b_) for s_ in setups for b_ in budgets]):
        row_g = capm[(capm.covariance_setup == su) & (capm.budget == b)
                     & (capm.scope == "global_capacity")].iloc[0]
        row_l = capm[(capm.covariance_setup == su) & (capm.budget == b)
                     & (capm.scope == "local_pooled_capacity")].iloc[0]
        ax.bar(xs[i] - w / 2, row_g["mean"], yerr=row_g["sem"], width=w,
               color=C["global"], ec="0.2", lw=0.6, capsize=3,
               label="global fit" if i == 0 else None)
        ax.bar(xs[i] + w / 2, row_l["mean"], yerr=row_l["sem"], width=w,
               color=C["local"], ec="0.2", lw=0.6, capsize=3,
               label="local pooled (budget-matched)" if i == 0 else None)
        ax.text(xs[i] - w / 2, row_g["mean"] + 0.02, f"{row_g['mean']:.2f}",
                ha="center", fontsize=7)
        ax.text(xs[i] + w / 2, row_l["mean"] + 0.02, f"{row_l['mean']:.2f}",
                ha="center", fontsize=7)
    ax.set_xticks(xs)
    ax.set_xticklabels(["disj.\nM=5", "disj.\nM=10", "non-disj.\nM=5",
                        "non-disj.\nM=10"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("top-d support recovery F1")
    ax.set_title("b  Capacity control: equal factor budget")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.24),
              fontsize=7, ncol=2)

    ax = axes[2]
    bio = s[(s.experiment == "bio_check")]
    cond_b = bio.pivot_table(index=["covariance_setup", "cluster_size",
                                    "alpha"], columns="method",
                             values="matched_top_k_f1")
    means_b = {m: cond_b[m].mean() for m in cond_b}
    means_b[f"{m}_se"] = 0
    for m in cond_b:
        means_b[f"{m}_se"] = cond_b[m].std(ddof=1) / np.sqrt(len(cond_b))
    _method_bars(ax, means_b, ["global", "phate", "oracle"],
                 lambda m: "oracle",
                 labels=["global", "data-driven\n(PHATE)", "oracle"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("top-d support recovery F1")
    ax.set_title("c  Bio CorEx estimator\n(single check condition)")

    fig.tight_layout()
    fig.savefig(OUT / "figure_2.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 3
def figure_3():
    """E2 headline. Absolute sine-sum n=5000 numbers and parity deltas are
    the committed primary-convention values (analysis notebook +
    claim_evidence_table.md R1.2/R1.3)."""
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.0))

    ax = axes[0]
    metrics = ["F1", "cosine"]
    oracle = [0.337, 0.383]
    glob = [0.268, 0.319]
    xs = np.arange(len(metrics))
    w = 0.35
    for x, o, g in zip(xs, oracle, glob):
        ax.bar(x - w / 2, o, width=w, color=C["oracle"], ec="0.2", lw=0.6,
               label="oracle partition" if x == 0 else None)
        ax.bar(x + w / 2, g, width=w, color=C["global"], ec="0.2", lw=0.6,
               label="global" if x == 0 else None)
    ax.set_xticks(xs)
    ax.set_xticklabels(["top-d F1", "support cosine"])
    ax.set_ylim(0, 0.55)
    ax.annotate("$\\Delta$F1 +0.069\nCI [0.043, 0.094]\n(15/16 seeds)",
                xy=(0.5, 0.40), xycoords="data", ha="center", fontsize=7.5,
                bbox=dict(fc="#fdf3d8", ec="0.6", lw=0.5))
    ax.set_ylabel("score")
    ax.set_title("a  sine-sum, n = 5000 (primary convention)")

    ax = axes[1]
    deltas = [0.161, 0.179]
    ci = [[0.161 - 0.123, 0.196 - 0.161],
          [0.179 - 0.162, 0.062 - 0.179]]
    ci = [[0.161 - 0.123, 0.200 - 0.161],
          [0.179 - 0.162, 0.196 - 0.179]]
    xs = np.arange(2)
    ax.bar(xs, deltas, yerr=np.array(ci).T, width=0.5, color=C["local"],
           ec="0.2", lw=0.6, capsize=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(["parity, n = 5000\n(16/16 seeds)",
                        "parity, n = 500\n(16/16 seeds)"])
    ax.set_ylabel("paired $\\Delta$F1 (oracle $-$ global)")
    ax.set_ylim(0, 0.23)
    ax.set_title("b  parity stress case: localization gain")

    ax = axes[2]
    ax.axis("off")
    txt = (
        "Failure modes kept visible:\n"
        "\n" + "$\\bullet$ " + "support precision lags: cosine 0.383 at\n"
        "   n = 5000 stays below the strict 0.5 bar\n"
        "\n" + "$\\bullet$ " + "Linear CorEx on parity: oracle F1 0.412\n"
        "   is below global 0.464 ($\\Delta$ $-$0.052,\n"
        "   CI [$-$0.093, $-$0.010])\n"
        "\n" + "$\\bullet$ " + "capability diagnostic: default-EM init\n"
        "   collapses at n = 5000 (0.0000 bits,\n"
        "   8/8 replicates) while a ground-truth\n"
        "   atom-seeded init reaches the analytic\n"
        "   TC (0.498$-$0.522 vs exact 0.500)\n"
        "\n" + "$\\bullet$ " + "random partitions explain more TC than\n"
        "   oracle (2.287 vs 1.439 bits): TC is not\n"
        "   a recovery proxy")
    ax.text(0.02, 0.98, txt, va="top", ha="left", fontsize=7.6,
            transform=ax.transAxes)
    ax.set_title("c  qualifications")

    fig.tight_layout()
    fig.savefig(OUT / "figure_3.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 4
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
    ax.set_title("a  same TC, different fitted TC\n(detectability, not type)")

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
    ax.set_title("b  $\\Omega$ instrument: equal where recovery\nworks, collapses where it fails")

    fig.tight_layout()
    fig.savefig(OUT / "figure_4.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 5
def figure_5():
    def contrast_frame(path):
        d = pd.read_csv(ROOT / "tnnls_human_experiments/exp4_mnist_benchmark"
                             "/results" / path)
        return d
    rnd = contrast_frame("mnist_lc_vs_random.csv")
    uni = contrast_frame("mnist_lc_vs_univariate.csv")
    prev = pd.read_csv(ROOT / "tnnls_human_experiments/exp4_mnist_benchmark/"
                            "results/mnist_prevalence.csv")
    garm = pd.read_csv(ROOT / "tnnls_human_experiments/exp4_mnist_benchmark/"
                            "results/global_arm_contrasts.csv")

    fig, axes = plt.subplots(2, 2, figsize=(9.6, 6.0))
    qs = sorted(rnd.threshold.unique())
    layers = [1, 2, 3]
    layer_names = {1: "H1", 2: "H2", 3: "H3"}
    lcolors = {1: "#1b9e77", 2: "#7570b3", 3: "#d95f02"}

    def by_q(df, metric):
        out = {}
        for l in layers:
            sub = df[(df.layer == l) & (df.metric == metric)].sort_values(
                "threshold")
            out[l] = sub
        return out

    ax = axes[0, 0]
    for l in layers:
        sub = by_q(rnd, "T_max")[l]
        ax.errorbar(sub.threshold, sub.mean_diff, yerr=2 * sub.se,
                    marker="o", ms=4, color=lcolors[l],
                    label=layer_names[l], capsize=2, lw=1.3)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xticks(qs)
    ax.set_xlabel("support threshold q")
    ax.set_ylabel("paired $\\Delta$T$_{max}$ (LC $-$ matched random)")
    ax.legend(fontsize=7)
    ax.set_title("a  local CorEx vs matched random")

    ax = axes[0, 1]
    for l in layers:
        sub = prev[prev.layer == l].sort_values("threshold")
        ax.errorbar(sub.threshold, sub.prevalence, yerr=2 * sub.se,
                    marker="s", ms=4, color=lcolors[l],
                    label=layer_names[l], capsize=2, lw=1.3)
    ax.set_xticks(qs)
    ax.set_ylim(-0.02, 0.46)
    ax.set_xlabel("support threshold q")
    ax.set_ylabel("conjunctive prevalence\n(T$_{max}\\geq$0.10, H$_{input}\\leq$0.50)")
    ax.legend(fontsize=7)
    ax.set_title("b  class-concentrated effects by layer")

    ax = axes[1, 0]
    for l in layers:
        sub = by_q(uni, "T_max")[l]
        ax.errorbar(sub.threshold, sub.mean_diff, yerr=2 * sub.se,
                    marker="^", ms=4, color=lcolors[l],
                    label=layer_names[l], capsize=2, lw=1.3)
    ax.axhline(0, color="0.5", lw=0.7)
    ax.set_xticks(qs)
    ax.set_xlabel("support threshold q")
    ax.set_ylabel("paired $\\Delta$T$_{max}$ (LC $-$ univariate)")
    ax.legend(fontsize=7)
    ax.set_title("c  local CorEx vs univariate logit-association")

    ax = axes[1, 1]
    sub = garm[(garm.input_condition == "output_aware")
               & (garm.budget == 10) & (garm.threshold == 0.9)]
    xs = np.arange(len(sub))
    ax.bar(xs, sub.d_T_max_mean,
           yerr=[sub.d_T_max_mean - sub.d_T_max_ci_lo,
                 sub.d_T_max_ci_hi - sub.d_T_max_mean],
           width=0.6, color="#d95f02", ec="0.2", lw=0.6, capsize=3)
    ax.bar(xs - 0, -0, width=0.6, color="none")
    ax.set_xticks(xs)
    ax.set_xticklabels([layer_names[int(v)] for v in sub.layer])
    ax.set_ylabel("paired $\\Delta$T$_{max}$ (global $-$ matched random)")
    for x, (_, r) in zip(xs, sub.iterrows()):
        ax.text(x, r.d_T_max_mean + 0.01, f"{r.d_T_max_mean:.2f}",
                ha="center", fontsize=7)
    ax.set_title("d  global CorEx arm (M = 10, q = 0.9,\noutput-aware): potent but blunt at H3")

    fig.tight_layout()
    fig.savefig(OUT / "figure_5.png")
    plt.close(fig)


# ---------------------------------------------------------------- figure 6
def figure_6():
    sys_r = pd.read_csv(ROOT / "outputs/summaries/imagenet_systematic_lc_vs_random.csv")
    sys_u = pd.read_csv(ROOT / "outputs/summaries/imagenet_systematic_lc_vs_univariate.csv")
    loc = pd.read_csv(ROOT / "outputs/summaries/imagenet_localization_contrasts.csv")

    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.2))

    ax = axes[0]

    def get(df, stat):
        r = df[df.statistic == stat].iloc[0]
        return r.mean_diff, r.ci_low, r.ci_high

    stats = [("tv_source", "TV on discovery region"),
             ("localization_contrast", "localization contrast L$_j$")]
    xs = np.arange(len(stats))
    w = 0.38
    for i, (s_, lbl) in enumerate(stats):
        m0, lo0, hi0 = get(sys_r, s_)
        m1, lo1, hi1 = get(sys_u, s_)
        ax.bar(i - w / 2, m0, yerr=np.array([[m0 - lo0], [hi0 - m0]]), width=w,
               color="#3b6fb6", ec="0.2", lw=0.6, capsize=3,
               label="LC $-$ matched random" if i == 0 else None)
        ax.bar(i + w / 2, m1, yerr=np.array([[m1 - lo1], [hi1 - m1]]), width=w,
               color="#e7298a", ec="0.2", lw=0.6, capsize=3,
               label="LC $-$ univariate" if i == 0 else None)
        ax.text(i - w / 2, hi0 + 0.008, f"{m0:+.2f}", ha="center", fontsize=7)
        ax.text(i + w / 2, hi1 + (0.008 if m1 > 0 else -0.022),
                f"{m1:+.2f}", ha="center", fontsize=7)
    ax.axhline(0, color="0.4", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(["$\\Delta$TV\n(discovery region)",
                        "$\\Delta$localization contrast\nL$_j$"])
    ax.set_ylabel("paired difference (cluster bootstrap)")
    ax.set_ylim(-0.045, 0.335)
    ax.set_title("a  systematic survey: paired contrasts")
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
    ax.set_title("b  effects concentrate near the\nfactor's discovery region")

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
    ax.set_title("c  case-study panels (committed\naccount-of-record values)")
    ax.text(1.9 + 0.30, -0.14, "univariate control\nnot recorded for\nthis panel",
            ha="center", fontsize=5.8, color="0.45")
    ax.legend(fontsize=6.2, loc="upper left", ncol=2, columnspacing=0.9)

    fig.tight_layout()
    fig.savefig(OUT / "figure_6.png")
    plt.close(fig)


if __name__ == "__main__":
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    figure_5()
    figure_6()
    print("figures written to", OUT)
