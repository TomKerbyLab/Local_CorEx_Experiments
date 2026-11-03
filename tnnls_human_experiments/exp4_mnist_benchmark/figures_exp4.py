"""Experiment 4 display set: renders the predeclared figures from the
committed summary tables in results/ (no models rerun).

Palette: validated reference instance (dataviz); layers carry fixed
categorical slots (blue/orange/aqua — all-pairs validated for 3 series),
the cumulative-MI threshold q is an ordinal blue ramp, and method
entities keep one color per figure: Local CorEx = blue, matched random
= orange, univariate = aqua. Claims are influence-shaped per the
pre-registration (paper/exp4.md §E4.12); these are display artifacts,
not new analyses.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
FIGURES = HERE / "figures"

LAYERS = (1, 2, 3)
LAYER_NAMES = {1: "H1 (500)", 2: "H2 (400)", 3: "H3 (300)"}
LAYER_COLOR = {1: "#2a78d6", 2: "#eb6834", 3: "#1baf7a"}
COND_NAMES = {"output_aware": "output-aware", "hidden_only": "hidden-only"}
COND_COLOR = {"output_aware": "#2a78d6", "hidden_only": "#eb6834"}
QS = (0.6, 0.7, 0.8, 0.9)
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"


def _rc():
    plt.rcParams.update({
        "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb",
        "savefig.facecolor": "#fcfcfb", "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.axisbelow": True, "axes.spines.top": False,
        "axes.spines.right": False, "font.size": 10,
        "axes.titlesize": 10.5, "font.family": "DejaVu Sans",
    })


def _load(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / f"mnist_{name}.csv")


def _style(ax, title, ylabel, xlabel=None):
    ax.set_title(title, color=INK, loc="left")
    ax.set_ylabel(ylabel, color=INK)
    if xlabel is not None:
        ax.set_xlabel(xlabel, color=INK)


def _save(fig, name) -> Path:
    FIGURES.mkdir(exist_ok=True)
    out = FIGURES / name
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", out.relative_to(HERE))
    return out


def fig01_prevalence():
    """Readings 1+3: prevalence of substantial-and-class-concentrated LC
    interventions, by layer x threshold (500 LC per stratum)."""
    prev = _load("prevalence")
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    qs = np.arange(len(QS))
    w = 0.25
    for i, layer in enumerate(LAYERS):
        g = prev[prev.layer == layer].sort_values("threshold")
        ax.bar(qs + (i - 1) * w, g.prevalence, width=w * 0.9,
               color=LAYER_COLOR[layer], label=LAYER_NAMES[layer])
        ax.errorbar(qs + (i - 1) * w, g.prevalence, yerr=g["se"], fmt="none",
                    ecolor=INK, elinewidth=0.8, capsize=2, alpha=0.35)
        for x, y in zip(qs + (i - 1) * w, g.prevalence):
            if y >= 0.01:
                ax.text(x, y + 0.012, f"{y:.2f}", ha="center",
                        fontsize=7.5, color=INK)
    ax.set_xticks(qs)
    ax.set_xticklabels([f"q = {q}" for q in QS])
    ax.set_ylim(0, 0.46)
    ax.legend(frameon=False, ncols=3, loc="upper left", fontsize=8.5)
    _style(ax, "Substantial & class-concentrated prevalence (LC supports)",
           "prevalence", "cumulative-MI threshold q")
    fig.tight_layout()
    return _save(fig, "fig01_prevalence_by_layer_threshold.png")


def fig02_lc_vs_random_dose_response():
    """Readings 1+4: paired LC - random differences across q, by layer.
    LC lines ride far above the zero line at every stratum."""
    r = _load("lc_vs_random")
    r = r[r.control == "random"]
    fig, axes = plt.subplots(2, 3, figsize=(9.6, 5.2),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.32})
    for k, metric in enumerate(("T_max", "H_input")):
        for i, layer in enumerate(LAYERS):
            ax = axes[k, i]
            g = r[(r.metric == metric) & (r.layer == layer)].sort_values(
                "threshold")
            color = "#2a78d6" if metric == "T_max" else "#eb6834"
            ax.plot(g.threshold, g["mean_diff"], color=color,
                    marker="o", ms=5, lw=2)
            ax.axhline(0, color=MUTED, lw=0.8)
            if k == 0:
                ax.set_title(LAYER_NAMES[layer], color=INK)
            if i == 0:
                ax.set_ylabel(f"Δ {metric}\n(LC − random)", color=INK)
            if k == 1:
                ax.set_xlabel("cumulative-MI threshold q", color=INK)
    fig.suptitle("Local CorEx − matched random: paired differences",
                 x=0.055, ha="left", fontsize=12, color=INK)
    fig.text(0.055, 0.002,
             "n = 42,500 pairs per stratum; point = mean over seeds of "
             "matched pairs; H_input shown negative (LC more concentrated)",
             fontsize=7.5, color=MUTED, ha="left")
    fig.tight_layout(rect=(0, 0.035, 1, 0.96))
    return _save(fig, "fig02_lc_vs_random_dose_response.png")


def fig03_density_vs_effect():
    """Reading 3: support density vs T_max over the 6,000 LC interventions;
    layer = color + marker shape (secondary encoding for CVD)."""
    td = _load("tc_density_magnitude")
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    shapes = {1: "o", 2: "s", 3: "^"}
    for layer in LAYERS:
        g = td[(td.layer == layer) & (td.T_max > 0)]
        ax.scatter(g.support_density, g.T_max, s=14, alpha=0.22,
                   color=LAYER_COLOR[layer], marker=shapes[layer],
                   linewidths=0, rasterized=True)
        ax.scatter([], [], s=40, color=LAYER_COLOR[layer],
                   marker=shapes[layer], label=LAYER_NAMES[layer])
    ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    _style(ax, "Support density vs intervention effect (6,000 LC supports)",
           "T_max", "support density")
    fig.tight_layout()
    return _save(fig, "fig03_density_vs_effect.png")


def fig04_concordance():
    """Reading 6 (descriptive): reduction-concordance rate + absolute shift
    at the associated class, by layer, both input conditions."""
    c = _load("concordance")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4),
                             gridspec_kw={"wspace": 0.32})
    x = np.arange(len(LAYERS))
    w = 0.32
    conds = ("output_aware", "hidden_only")
    for i, cond in enumerate(conds):
        g = c[c.input_condition == cond].set_index("layer")
        vals = [g.loc[l, "reduction_concordance_rate"] for l in LAYERS]
        ses = [g.loc[l, "se"] for l in LAYERS]
        axes[0].bar(x + (i - 0.5) * w, vals, width=w * 0.9,
                    color=COND_COLOR[cond], label=COND_NAMES[cond])
        axes[0].errorbar(x + (i - 0.5) * w, vals, yerr=ses, fmt="none",
                         ecolor=INK, elinewidth=0.8, capsize=2, alpha=0.35)
        vals = [g.loc[l, "mean_abs_shift_at_associated_class"] for l in LAYERS]
        axes[1].bar(x + (i - 0.5) * w, vals, width=w * 0.9,
                    color=COND_COLOR[cond])
    axes[0].axhline(0.10, color=MUTED, lw=0.8)
    axes[0].text(2.38, 0.115, "chance 0.10", fontsize=7.5, color=MUTED,
                 ha="right")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([LAYER_NAMES[l] for l in LAYERS])
    axes[0].set_ylim(0, 0.92)
    axes[0].legend(frameon=False, fontsize=8.5)
    _style(axes[0], "Reduction concordance (associated = most reduced)",
           "rate")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([LAYER_NAMES[l] for l in LAYERS])
    axes[1].set_ylim(0, 0.085)
    _style(axes[1], "Absolute shift at the associated class",
           "mean |Δ_{y*, c*}|")
    fig.tight_layout()
    return _save(fig, "fig04_concordance_by_layer_condition.png")


def fig05_tc_magnitude():
    """Reading 3: Spearman(TC, T_max), raw (faded) vs support-size-adjusted
    (solid), by layer across q."""
    t = _load("tc_magnitude")
    fig, ax = plt.subplots(figsize=(6.6, 3.3))
    qs = np.arange(len(QS))
    off = 0.22
    for i, layer in enumerate(LAYERS):
        g = t[t.layer == layer].sort_values("threshold")
        x = qs + (i - 1) * off
        ax.plot(x, g["spearman_tc_Tmax"], color=LAYER_COLOR[layer],
                lw=1.0, marker="o", ms=3.5, alpha=0.30)
        ax.plot(x, g["spearman_tc_Tmax_resid_size"], color=LAYER_COLOR[layer],
                lw=2.0, marker="o", ms=5)
    ax.axhline(0, color=MUTED, lw=0.8)
    ax.set_xticks(qs)
    ax.set_xticklabels([f"q = {q}" for q in QS])
    handles = [plt.Line2D([], [], color=LAYER_COLOR[l], lw=2,
                          label=LAYER_NAMES[l]) for l in LAYERS]
    handles += [
        plt.Line2D([], [], color=MUTED, lw=1.0, label="raw"),
        plt.Line2D([], [], color=INK, lw=2.0, label="size-adjusted"),
    ]
    ax.legend(frameon=False, fontsize=8, ncols=5,
              handles=handles, loc="lower right")
    _style(ax, "Factor TC ↔ intervention magnitude (faded = raw; "
           "solid = support-size-adjusted)",
           "Spearman ρ (TC ↔ T_max)")
    fig.tight_layout()
    return _save(fig, "fig05_tc_magnitude_link.png")


def fig06_effective_units():
    """Reading 3: effective number of units per retained factor (entropy of
    normalized hidden-MI weights), by layer. One dot = one factor-rank row
    (seed- and cluster-averaged), condition = color with side-by-side
    clusters; dash = the layer-condition row mean."""
    e = _load("effective_units")
    rng = np.random.default_rng(7)
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    xs = np.arange(len(LAYERS))
    offs = {"output_aware": -0.16, "hidden_only": +0.16}
    for cond in ("output_aware", "hidden_only"):
        for i, layer in enumerate(LAYERS):
            g = e[(e.layer == layer) & (e.input_condition == cond)]
            x = xs[i] + offs[cond] + rng.uniform(-0.055, 0.055, size=len(g))
            ax.scatter(x, g["mean"], s=26, color=COND_COLOR[cond], alpha=0.8)
            ax.scatter([xs[i] + offs[cond]], [g["mean"].mean()],
                       s=90, marker="_", color=INK)
    ax.set_xticks(xs)
    ax.set_xticklabels([LAYER_NAMES[l] for l in LAYERS])
    ax.set_xlim(-0.55, len(LAYERS) - 0.45)
    ax.legend(frameon=False, fontsize=8.5,
              handles=[plt.Line2D([], [], color=COND_COLOR[k], marker="o",
                                  ls="", label=v)
                       for k, v in COND_NAMES.items()])
    _style(ax, "Effective number of units per retained factor: broad "
           "associations, not one dominant unit",
           "effective # units")
    fig.tight_layout()
    return _save(fig, "fig06_effective_units_by_layer.png")


def main() -> list[Path]:
    _rc()
    return [
        fig01_prevalence(),
        fig02_lc_vs_random_dose_response(),
        fig03_density_vs_effect(),
        fig04_concordance(),
        fig05_tc_magnitude(),
        fig06_effective_units(),
    ]


if __name__ == "__main__":
    main()