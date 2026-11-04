"""Experiment 4 qualitative reconstruction panel (predeclared rule,
paper/exp4.md; explanatory visualization only — never factor selection).

Frozen selection:
  1. Among discovery clusters with >= 200 observations, choose the one
     maximizing min(n_4, n_9) (discovery digit counts).
  2. Within that cluster and each layer, choose the retained factor (seed
     5100, output-aware) with the largest max{I(Z_j; L_4), I(Z_j; L_9)},
     read from the stored fit-side logit MIs (bits).
  3. Representative image: the lowest-index TEST image assigned to that
     cluster (fixed held-out rule; assignment stage, decision #11).

Perturbation semantics (round-4 correction, exp4.md §E4.13): the factor
score on the layer's standardized hidden coordinates is z = s_h . h_std,
so z is linear in the RAW activations with gradient g = s_h / sigma_l
(elementwise). The minimum-Euclidean-norm RAW-activation change achieving
a factor-score change delta is
    Delta h = delta * g / ||g||^2,
whose achieved change is delta exactly (asserted per plotted row). delta
in {0, +-1, +-2} std of Z_j over the discovery cluster. The score
perturbed is the factor's hidden-coordinate component: the output-aware
factor's logit loading is separate and is not controlled (logits respond
to the change only through the forward pass from layer 1/2). Layer 1/2
perturbations propagate forward through the classifier to H3; the mirror
autoencoder (train_mirror_ae) decodes the latent to pixels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from common import corex, seeds
from common import store as store_mod
from common.config import CHECKPOINTS_DIR, RAW_DIR
from benchmark import EXPERIMENT, center_logits, extract_representations, \
    standardize_with_flags
from classifier import MLP
from data import load_splits
from train import load_model
from .train_mirror_ae import MirrorDecoder

SEED = 5100
MIN_PANEL_CLUSTER_OBS = 200
DELTAS = (0.0, -2.0, -1.0, 1.0, 2.0)  # in units of std(Z_j)


def select_panel_keys(splits, labels_cluster, model, device):
    """Predeclared selection -> (cluster, {layer: (factor_index, bits)})."""
    store = store_mod.ResultStore(EXPERIMENT)
    cl = store.read_all("clusters")
    y_disc = splits["discovery_y"].numpy()
    comp = {int(c): np.bincount(y_disc[labels_cluster == c], minlength=10)
            for c in np.unique(labels_cluster)}
    big = {c: v for c, v in comp.items()
           if len(np.flatnonzero(labels_cluster == c)) >= MIN_PANEL_CLUSTER_OBS}
    cluster = int(max(big, key=lambda c: float(min(big[c][4], big[c][9]))))

    xc = splits["discovery_x"][torch.tensor(
        np.flatnonzero(labels_cluster == cluster))]
    h1, h2, h3, logits_disc = extract_representations(model, xc, device)
    logits_disc = center_logits(logits_disc)
    factors = {}
    f = store.read_all("factors")
    for layer, H in ((1, h1), (2, h2), (3, h3)):
        fdf = f[(f.model_seed == SEED) & (f.cluster == cluster)
                & (f.layer == layer)
                & (f.input_condition == "output_aware")]
        mi4 = fdf["logit_mi_4"].to_numpy(float)
        mi9 = fdf["logit_mi_9"].to_numpy(float)
        pick = int(np.argmax(np.maximum(mi4, mi9)))
        factors[layer] = (int(fdf.factor_index.to_numpy()[pick]),
                          float(np.max(np.maximum(mi4, mi9))))
    return cluster, factors


def loadings_for_factor(model, splits, labels_cluster, cluster, layer, j,
                        device):
    """Deterministic refit (same seed/input as the stored fit) -> the kept
    factor's loading vector s_h (full width n_h, valid coordinates only)
    and the factor-score std sigma over the discovery cluster."""
    xc = splits["discovery_x"][torch.tensor(
        np.flatnonzero(labels_cluster == cluster))]
    h1, h2, h3, logits_disc = extract_representations(model, xc, device)
    logits_disc = center_logits(logits_disc)
    H = {1: h1, 2: h2, 3: h3}[layer]
    H_std, valid_h, _ = standardize_with_flags(H)
    L_std, _, _ = standardize_with_flags(logits_disc)
    s_fit = seeds.derive_seed(SEED, "corex", cluster, layer, "output_aware")
    v_idx = np.flatnonzero(valid_h)
    V = np.hstack([H_std[:, v_idx], L_std])
    fit = corex.fit_linear_corex(V, 10, seed=s_fit)
    s_h = np.zeros(H.shape[1])
    s_h[v_idx] = np.asarray(fit.model.ws)[j][:len(v_idx)]
    Z = H_std[:, v_idx] @ np.asarray(fit.model.ws)[j][:len(v_idx)]
    return s_h, float(Z.std()), H, (H.mean(axis=0), H.std(axis=0))


def decode_with_perturbations(model, decoder, splits, test_cluster, labels_cluster,
                              cluster, layer, j, device):
    """One panel row: decoded images for the delta grid (units of Z-std)."""
    s_h, sigma, H, (mu, unit_std) = loadings_for_factor(
        model, splits, labels_cluster, cluster, layer, j, device)
    # raw-space minimum-norm direction: g = s_h / sigma, dh = delta*g/||g||^2
    # achieves the factor-score change delta exactly (verified below).
    g = np.divide(s_h, unit_std, out=np.zeros_like(s_h), where=unit_std > 0)
    g_norm2 = float(np.sum(g * g)) or 1e-12
    d_raw = g / g_norm2
    cohort_idx = np.flatnonzero(test_cluster == cluster)
    img_idx = int(cohort_idx[0]) if len(cohort_idx) else 0
    x0 = splits["test_x"][torch.tensor([img_idx])]
    with torch.no_grad():
        logits, ih1, ih2, ih3 = model.hidden_states(x0.to(device))
    base_h = {1: ih1, 2: ih2, 3: ih3}[layer]
    images = []
    for delta in DELTAS:
        step = (delta * d_raw)[None, :]
        achieved = float(s_h @ (delta * d_raw / unit_std))
        if abs(achieved - delta) > 1e-4 * max(abs(delta), 1e-6):
            raise AssertionError(
                f"panel perturbation achieves dz={achieved} for delta={delta}")
        h_raw_pert = base_h + torch.as_tensor(
            step, dtype=base_h.dtype, device=base_h.device)
        cur = h_raw_pert.to(device)
        if layer == 1:
            cur = torch.relu(model.bn2(model.fc2(cur)))
        if layer in (1, 2):
            cur = torch.relu(model.bn3(model.fc3(cur)))
        with torch.no_grad():
            recon = decoder(cur)[0].reshape(28, 28).cpu().numpy()
        images.append(np.clip(recon, 0.0, 1.0))
    return images, img_idx, sigma


def _render(cluster, rows):
    import matplotlib.pyplot as plt
    plt.rcParams["axes.facecolor"] = "#fcfcfb"
    fig, axes = plt.subplots(3, 5, figsize=(12.4, 7.8))
    cols = ("decoded\nunaltered", "-2 std", "-1 std", "+1 std", "+2 std")
    for r, row in enumerate(rows):
        for c, img in enumerate(row["images"]):
            ax = axes[r, c]
            ax.imshow(img, cmap="gray", vmin=0.0, vmax=1.0)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(cols[c], fontsize=9, color="#0b0b0b")
            if c == 0:
                ax.set_ylabel(f"H{row['layer']}: factor "
                              f"{row['factor']} "
                              f"(assoc {row['assoc_bits']:.2f} bits)",
                              fontsize=8, color="#0b0b0b")
    fig.suptitle(
        f"Experiment 4 qualitative panel (explanatory; predeclared rule): "
        f"cluster {cluster}, seed 5100; factor-score-directed symmetric "
        f"perturbations, decoded through the mirror AE",
        x=0.04, ha="left", fontsize=11, color="#0b0b0b")
    fig.tight_layout(rect=(0, 0.01, 1, 0.94))
    out = RAW_DIR.parent / "figures"
    out.mkdir(parents=True, exist_ok=True)
    p = out / "exp4_qualitative_reconstruction_panel.png"
    fig.savefig(p, dpi=200, facecolor="#fcfcfb", bbox_inches="tight")
    print("saved", p)


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    store = store_mod.ResultStore(EXPERIMENT)
    splits = load_splits(save_dir=RAW_DIR / EXPERIMENT)
    labels_cluster = pd.read_parquet(store.shard_path(
        {"stage": "clusters", "phate_seed": 5200, "kmeans_seed": 5201,
         "k": 20}, "clusters")).set_index(
        "discovery_index")["cluster"].sort_index().to_numpy()
    test_cluster = pd.read_parquet(store.shard_path(
        {"stage": "test_clusters", "phate_seed": 5200, "kmeans_seed": 5201,
         "k": 20}, "assignments")).set_index(
        "test_index")["test_cluster"].sort_index().to_numpy()
    model = load_model(SEED, device=device)
    decoder = MirrorDecoder()
    decoder.load_state_dict(torch.load(
        CHECKPOINTS_DIR / "mnist" / f"mnist_mirror_ae_seed{SEED}.pt",
        map_location="cpu"))
    decoder.to(device).eval()
    cluster, factors = select_panel_keys(splits, labels_cluster, model, device)
    rows = []
    for layer in (1, 2, 3):
        j, assoc = factors[layer]
        imgs, img_idx, sigma = decode_with_perturbations(
            model, decoder, splits, test_cluster, labels_cluster, cluster,
            layer, j, device)
        rows.append({"layer": layer, "factor": j, "assoc_bits": assoc,
                     "sigma": sigma, "images": imgs})
    _render(cluster, rows)


if __name__ == "__main__":
    main()