"""Experiment 4 benchmark runner.

Stages (each resumable independently):
  1. splits      fixed 45k/5k/10k/10k split (seed 5000)
  2. clusters    PHATE-10d + k-means k=20 on discovery images (seeds 5200/5201)
  3. train       5 classifiers (seeds 5100-5104)  [run via train.py]
  4. fits       Linear CorEx per (seed, cluster, layer, input_condition),
                 M=10 factors, top-5 retained by TC
  5. supports   cumulative-MI supports q in {0.6,0.7,0.8,0.9} (nested)
  6. ablations  interventions on the 10k test set: Local CorEx, 20 matched
                 random controls, univariate logit-association support;
                 hidden-only condition at q=0.8 with 5 random controls
  7. summarize   reads raw shards; never reruns models

Data discipline: the discovery set drives PHATE/CorEx/ranking/supports and
qualitative selection; the test set is used ONLY for final intervention
evaluation. Test labels/predictions never enter factor fitting or support
construction.

Intervention efficiency: for layer l, the test-set h_l is cached once per
model seed; each intervention runs only the remaining layers from masked
h_l. Numerical equivalence with the canonical model-mask path is unit tested
(test_trunk_equivalence). Dropout is identity in eval mode, so the trunk is
deterministic.
"""

from __future__ import annotations

import functools
import json
import logging
import time

import numpy as np
import pandas as pd
import torch

from ..common import config, corex, metrics, seeds
from ..common.io import FailureLog, write_run_report
from ..common.runtime import setup_logging
from ..common.store import ResultStore
from .classifier import MLP, LAYER_DIMS
from .data import load_splits
from .interventions import (
    accuracy_metrics, classwise_mean_probs, classwise_shift_matrix,
    effect_metrics, individual_effect_metrics, predict_proba,
    random_control_probability, substantial_class_concentrated)

EXPERIMENT = "mnist"
logger = logging.getLogger(f"tnnls.{EXPERIMENT}")

CLUSTER_PHATE_SEED = 5200
CLUSTER_KMEANS_SEED = 5201
N_CLUSTERS = 20
N_FACTORS = 10
N_RETAINED = 5
SUPPORT_THRESHOLDS = (0.60, 0.70, 0.80, 0.90)
N_RANDOM_CONTROLS = 20
N_RANDOM_CONTROLS_HIDDEN_ONLY = 5
HIDDEN_ONLY_FRACTION = 0.80
MODEL_SEEDS = (5100, 5101, 5102, 5103, 5104)
LAYERS = (1, 2, 3)
VAR_FLOOR = 1e-12
NATS_TO_BITS = 1.0 / np.log(2.0)  # Linear CorEx reports TC/MI in nats

INPUT_CONDITIONS = ("output_aware", "hidden_only")

DEFAULTS = {
    "n_factors": N_FACTORS,
    "n_retained": N_RETAINED,
    "support_thresholds": list(SUPPORT_THRESHOLDS),
    "n_random_controls": N_RANDOM_CONTROLS,
    "n_random_controls_hidden_only": N_RANDOM_CONTROLS_HIDDEN_ONLY,
    "input_conditions": list(INPUT_CONDITIONS),
    "model_seeds": list(MODEL_SEEDS),
    "layers": list(LAYERS),
    "n_clusters": N_CLUSTERS,
}


# ---------------------------------------------------------------------------
# Stage 2: fixed discovery clusters
# ---------------------------------------------------------------------------

def build_discovery_clusters(splits: dict, store: ResultStore,
                             n_clusters: int = N_CLUSTERS) -> np.ndarray:
    """PHATE-10d + k-means(k=20, n_init=50, seed 5201) on discovery pixels.

    Pixels scaled to [0,1] and flattened (per spec). Digit labels are not
    used. Saves PHATE coords, labels, sizes, compositions.
    """
    cond = {"stage": "clusters", "phate_seed": CLUSTER_PHATE_SEED,
            "kmeans_seed": CLUSTER_KMEANS_SEED, "k": n_clusters}
    if store.is_done(cond, ("clusters",)):
        z = pd.read_parquet(store.shard_path(cond, "clusters"))
        return z.set_index("discovery_index")["cluster"].sort_index().to_numpy()

    import phate
    from sklearn.cluster import KMeans

    X = splits["discovery_x"].numpy()  # already [0,1] flattened
    ph = phate.PHATE(n_components=10, random_state=CLUSTER_PHATE_SEED,
                     n_pca=100, verbose=False, n_jobs=8)
    emb = ph.fit_transform(X)
    km = KMeans(n_clusters=n_clusters, n_init=50,
                random_state=CLUSTER_KMEANS_SEED)
    labels = km.fit_predict(emb)
    rows = []
    for i, (c, y) in enumerate(zip(labels, splits["discovery_y"].numpy())):
        rows.append({"discovery_index": i, "cluster": int(c),
                     "true_label": int(y),
                     **{f"phate_{j}": float(emb[i, j]) for j in range(10)}})
    store.write(cond, "clusters", rows)
    comp = {}
    for c in range(n_clusters):
        yc = splits["discovery_y"].numpy()[labels == c]
        comp[c] = np.bincount(yc, minlength=10).tolist()
    with open(config.RAW_DIR / EXPERIMENT / "cluster_compositions.json", "w") as f:
        json.dump(comp, f, indent=2)
    return labels


# ---------------------------------------------------------------------------
# Representations
# ---------------------------------------------------------------------------

def extract_representations(model: MLP, x: torch.Tensor,
                            device: str) -> tuple[np.ndarray, ...]:
    """Run the classifier on x (eval mode, no masks); return H1,H2,H3,logits."""
    model.eval()
    model.clear_masks()
    with torch.no_grad():
        xs = x.to(device)
        logits, h1, h2, h3 = model.hidden_states(xs)
    return (h1.cpu().numpy(), h2.cpu().numpy(), h3.cpu().numpy(),
            logits.cpu().numpy())


def standardize_with_flags(M: np.ndarray, var_floor: float = VAR_FLOOR):
    mean = M.mean(axis=0)
    std = M.std(axis=0)
    valid = std > np.sqrt(var_floor)
    Ms = np.zeros_like(M)
    Ms[:, valid] = (M[:, valid] - mean[valid]) / std[valid]
    return Ms, valid, ~valid


def center_logits(logits: np.ndarray) -> np.ndarray:
    """Row-center the logit block: L_c - (1/10) sum_u L_u per image.

    Softmax is invariant to adding an image-dependent constant to every
    logit, so this common component is behaviorally irrelevant to the
    model, yet raw-logit associations can shift with it. Removed before
    standardization for every discovery-side use of the logits (fit input,
    post-hoc associations, univariate control). Note: standardizing each
    column afterwards scales the residual (differential) part by that
    column's spread, which is intended.
    """
    return logits - logits.mean(axis=1, keepdims=True)


def effective_n_units(scores: np.ndarray, valid: np.ndarray) -> float:
    """Effective number of units carrying the factor's association: exp of
    the entropy of the normalized positive hidden-unit MI weights (invalid
    coordinates carry zero weight). One dominant unit -> ~1; k equal units
    -> ~k. Distinguishes broad association from dominant-unit-plus-weak-tail.
    """
    w = np.clip(np.asarray(scores, dtype=float), 0.0, None).copy()
    w[~valid] = 0.0
    tot = w.sum()
    if tot <= 0:
        return float("nan")
    p = w / tot
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def tc_rank_order(tcs: np.ndarray, n_retained: int, tie_seed: int) -> np.ndarray:
    """Top-n_retained among factors with positive explained TC.

    Ties in TC are broken by the seeded permutation (the program's neutral
    tie rule), never by column layout. Conditions with no positive-TC
    factor yield an empty order: no interventions are defined there, and
    the shortfall shows up in coverage reporting, never as silent padding
    with noise factors.
    """
    rng = np.random.default_rng(int(tie_seed))
    pos = np.flatnonzero(np.asarray(tcs, dtype=float) > 0)
    if len(pos) == 0:
        return np.array([], dtype=int)
    perm = rng.permutation(len(pos))
    tpos = np.asarray(tcs, dtype=float)[pos[perm]]
    return pos[perm][np.argsort(-tpos, kind="stable")][:n_retained]


def fit_and_rank(H_std, L_std, n_factors, seed, n_retained=N_RETAINED,
                 valid_h=None):
    """Fit Linear CorEx on [H | L] (or H alone); rank by TC; extract MIs.

    Columns flagged invalid (near-zero variance) are EXCLUDED from the fit
    per spec; their MI entries are zero in the returned full-width arrays.
    """
    valid = np.ones(H_std.shape[1], dtype=bool) if valid_h is None else valid_h
    v_idx = np.flatnonzero(valid)
    H_fit = H_std[:, v_idx]
    V = np.hstack([H_fit, L_std]) if L_std is not None else H_fit
    fit = corex.fit_linear_corex(V, n_factors, seed=seed)
    order = tc_rank_order(fit.tcs, n_retained, seeds.derive_seed(seed, "tc_order"))
    n_h = H_std.shape[1]
    mis = fit.association_scores()
    # map fit-side columns back to full-width hidden indices
    hidden_mis = np.zeros((mis.shape[0], n_h))
    hidden_mis[:, v_idx] = mis[:, :len(v_idx)]
    logit_mis = mis[:, len(v_idx):] if L_std is not None else None
    dominant = (np.argmax(logit_mis, axis=1).astype(int) if logit_mis is not None
                else np.full(mis.shape[0], -1, dtype=int))
    gaps = []
    for j in range(mis.shape[0]):
        row = logit_mis[j] if logit_mis is not None else np.array([])
        gaps.append(float(np.sort(row)[-1] - np.sort(row)[-2])
                    if len(row) >= 2 else float("nan"))
    return {"fit": fit, "order": order, "hidden_mis": hidden_mis,
            "logit_mis": logit_mis, "dominant": dominant, "gaps": gaps}


def posthoc_logit_mi(fit, H_fit: np.ndarray,
                     L_std: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
    """Post-hoc factor-to-logit association for hidden-only fits (spec:
    'associate each factor with logits by estimating I(Z_j; L_c) on the
    discovery observations').

    Factor scores are the linear latents Z = transform(H) of the fitted
    model on the same standardized discovery sample used for the fit; MI is
    the Gaussian estimate -0.5*log(1 - rho^2), the same estimator family as
    LinearCorex's internal association scores. Dead (near-zero-variance)
    factors or logits contribute MI 0; ties in argmax resolve to the lower
    class index. Downstream rows carry logit_mi_source='post_hoc' so these
    estimates are never confused with fit-side scores.
    """
    Z = fit.model.transform(H_fit)  # (n, m)
    n, m = Z.shape
    mis = np.zeros((m, L_std.shape[1]))
    L_active = L_std.std(axis=0) > 1e-12
    z_std = Z.std(axis=0)
    for j in range(m):
        if z_std[j] < 1e-12:
            continue
        zc = Z[:, j]
        for c in np.flatnonzero(L_active):
            rho = float(np.corrcoef(zc, L_std[:, c])[0, 1])
            if np.isfinite(rho):
                mis[j, c] = -0.5 * np.log(max(1.0 - rho * rho, 1e-12))
    dominant = np.argmax(mis, axis=1).astype(int)
    gaps = [float(np.sort(mis[j])[-1] - np.sort(mis[j])[-2]) for j in range(m)]
    return mis, dominant, gaps


# ---------------------------------------------------------------------------
# Trunk evaluator: interventions from cached pre-layer activations
# ---------------------------------------------------------------------------

class TrunkEvaluator:
    """Evaluates layer-l interventions by masking cached test-set h_l and
    running the remaining layers. Equivalence with model-mask forward passes
    is unit tested."""

    def __init__(self, model: MLP, x_test: torch.Tensor, device: str):
        self.model = model.to(device).eval()
        self.model.clear_masks()
        self.device = device
        with torch.no_grad():
            xs = x_test.to(device)
            logits, h1, h2, h3 = self.model.hidden_states(xs)
        self.h = {1: h1, 2: h2, 3: h3}
        self.baseline_logits = logits
        self.baseline_probs = torch.softmax(logits, dim=1).cpu().numpy()
        self.test_x = x_test
        # mean |activation| per unit on the test set, per layer: the
        # activity-removal magnitude baseline for the support diagnostics
        self.unit_magnitude = {l: np.abs(hi.cpu().numpy()).mean(axis=0)
                               for l, hi in self.h.items()}
        self.mean_removed_activation = lambda layer, support: (
            float(self.unit_magnitude[layer][list(support)].mean())
            if len(support) else 0.0)

    @torch.no_grad()
    def ablated_probs(self, layer: int, support: list[int]) -> np.ndarray:
        h = self.h[layer]
        if support:
            h = h.clone()
            h[:, list(support)] = 0.0
        cur = h
        for l in range(layer + 1, 4):
            fc = {2: self.model.fc2, 3: self.model.fc3}[l]
            bn = {2: self.model.bn2, 3: self.model.bn3}[l]
            cur = torch.relu(bn(fc(cur)))
        logits = self.model.fc4(cur)
        return torch.softmax(logits, dim=1).cpu().numpy()

    @torch.no_grad()
    def probs_from_model_mask(self, layer: int, support: list[int]) -> np.ndarray:
        """Canonical path: set the mask on the model and forward."""
        self.model.set_mask(layer, support)
        try:
            return predict_proba(self.model, self.test_x.to(self.device)).cpu().numpy()
        finally:
            self.model.clear_masks()


# ---------------------------------------------------------------------------
# Stage 4/5: fits + supports
# ---------------------------------------------------------------------------

def factor_condition(seed, cluster, layer, input_condition):
    return {"stage": "fit", "model_seed": seed, "cluster": cluster,
            "layer": layer, "input_condition": input_condition}


def run_fit_stage(model_seed, splits, labels_cluster, model, device,
                  store, cfg) -> None:
    """Fit CorEx for every (cluster, layer, input_condition); write factor and
    support shards."""
    x_disc = splits["discovery_x"]
    for cluster in range(int(cfg["n_clusters"])):
        idx = np.flatnonzero(labels_cluster == cluster)
        if len(idx) == 0:
            continue
        xc = x_disc[torch.tensor(idx)]
        h1, h2, h3, logits = extract_representations(model, xc, device)
        for layer in cfg["layers"]:
            H = {1: h1, 2: h2, 3: h3}[layer]
            H_std, valid_h, excluded_h = standardize_with_flags(H)
            L_std, _, _ = standardize_with_flags(center_logits(logits))
            for cond_name in cfg["input_conditions"]:
                fcond = factor_condition(model_seed, cluster, layer, cond_name)
                if store.is_done(fcond, ("factors", "supports")):
                    continue
                s_fit = seeds.derive_seed(model_seed, "corex", cluster, layer,
                                          cond_name)
                L_for_fit = L_std if cond_name == "output_aware" else None
                fr = fit_and_rank(H_std, L_for_fit, int(cfg["n_factors"]), s_fit,
                                  int(cfg["n_retained"]), valid_h=valid_h)
                if fr["logit_mis"] is None:
                    # hidden-only: post-hoc factor-to-logit association (spec)
                    v_idx = np.flatnonzero(valid_h)
                    fr["logit_mis"], fr["dominant"], fr["gaps"] = \
                        posthoc_logit_mi(fr["fit"], H_std[:, v_idx], L_std)
                    logit_source = "post_hoc"
                else:
                    logit_source = "fit"
                factor_rows = []
                support_rows = []
                for rank, j in enumerate(fr["order"]):
                    hmis = fr["hidden_mis"][j]
                    row = {
                        "model_seed": model_seed, "cluster": cluster,
                        "layer": layer, "input_condition": cond_name,
                        "factor_index": int(j), "factor_rank": rank,
                        "factor_tc": float(fr["fit"].tcs[j]) / NATS_TO_BITS,  # bits
                        "corex_seed": s_fit,
                        "corex_converged": fr["fit"].converged,
                        "n_negative_mi": fr["fit"].n_negative_mi,
                        "n_valid_hidden": int(valid_h.sum()),
                        "n_excluded_hidden": int((~valid_h).sum()),
                        "corex_runtime_s": fr["fit"].runtime_s,
                        "logit_mi_source": logit_source,
                        "effective_n_units": effective_n_units(hmis, valid_h),
                        # full-width hidden-MI vector (zeros at excluded
                        # coordinates) for the TC/distributedness analyses
                        "hidden_mi_vector": json.dumps(
                            np.round(hmis, 6).tolist()),
                    }
                    if fr["logit_mis"] is not None:
                        lmis = fr["logit_mis"][j]
                        row["dominant_logit"] = int(fr["dominant"][j])
                        row["logit_mi_gap"] = float(fr["gaps"][j])
                        # association scores are recorded in bits at this
                        # boundary (fit-side and post-hoc alike; Linear CorEx
                        # reports nats — the Experiment 3 unit lesson)
                        for c in range(10):
                            row[f"logit_mi_{c}"] = float(lmis[c]) / NATS_TO_BITS
                    factor_rows.append(row)
                    # hidden-unit supports (nested) from hidden MI only;
                    # MI-score ties broken by seeded permutation (neutral
                    # tie rule), mapped back to original unit indices
                    s_tie = seeds.derive_seed(s_fit, "support_ties")
                    perm = np.random.default_rng(s_tie).permutation(
                        H_std.shape[1])
                    sups_p = metrics.nested_supports(
                        hmis[perm], list(cfg["support_thresholds"]),
                        valid_h[perm])
                    sups = []
                    for sup in sups_p:
                        sup.indices = [int(perm[i]) for i in sup.indices]
                        sup.scores = [float(hmis[i]) for i in sup.indices]
                        sups.append(sup)
                    for q, sup in zip(cfg["support_thresholds"], sups):
                        for rank_in_sup, (unit, score) in enumerate(
                                zip(sup.indices, sup.scores)):
                            support_rows.append({
                                "model_seed": model_seed, "cluster": cluster,
                                "layer": layer, "input_condition": cond_name,
                                "factor_index": int(j), "factor_rank": rank,
                                "threshold": float(q),
                                "mi_rank": rank_in_sup,
                                "hidden_unit": int(unit),
                                "mi_score": float(score) / NATS_TO_BITS,
                                "achieved_fraction": sup.achieved_fraction,
                                "support_size": sup.size,
                                "support_density": sup.density,
                                "support_degenerate": sup.degenerate,
                            })
                store.write(fcond, "factors", factor_rows)
                store.write(fcond, "supports", support_rows)


# ---------------------------------------------------------------------------
# Stage 6: ablations
# ---------------------------------------------------------------------------

def univariate_support(H: np.ndarray, logit_col: np.ndarray, size: int,
                       valid: np.ndarray) -> list[int]:
    """Top-s hidden units by |corr(H_i, L_{c*})| within the cluster; ties by
    lower index. Excludes invalid coordinates."""
    v_idx = np.flatnonzero(valid)
    if size >= len(v_idx):
        return list(v_idx[:size])
    Hv = H[:, valid]
    if logit_col.std() < 1e-12 or Hv.shape[0] < 3:
        return list(v_idx[:size])
    Hc = Hv - Hv.mean(0)
    lc = logit_col - logit_col.mean()
    denom = np.sqrt((Hc ** 2).sum(0)) * np.sqrt((lc ** 2).sum()) + 1e-12
    corr = np.abs((Hc * lc[:, None]).sum(0) / denom)
    order = np.argsort(-corr, kind="stable")  # ties -> lower index
    return [int(v_idx[i]) for i in order[:size]]


def _summarize_intervention(base, method, draw, p_abl, p_base, labels,
                            dominant_logit, removed_act_mag=None):
    delta = classwise_shift_matrix(p_abl, p_base, labels)
    n_per_class = np.bincount(labels, minlength=10).astype(float)
    m = effect_metrics(delta, n_per_class)
    acc = accuracy_metrics(p_abl, p_base, labels)
    indiv = individual_effect_metrics(p_abl, p_base, labels)
    row = {
        **base, "method": method, "random_draw": draw,
        "substantial_class_concentrated": substantial_class_concentrated(
            m["T_max"], m["H_input"]),
        **m,
        "accuracy_change": acc["accuracy_change"],
        "mean_ce_change": acc["mean_ce_change"],
        # cancel-free individual-effect companions (U_y >= T_y)
        "U_max": indiv["U_max"], "U_weighted": indiv["U_weighted"],
        "y_U_star": indiv["y_U_star"],
        "U_per_class": json.dumps(np.round(indiv["U_per_class"], 6).tolist()),
        "flip_rate": indiv["flip_rate"],
        "flip_rate_per_class": json.dumps(
            np.round(indiv["flip_rate_per_class"], 6).tolist()),
        # removed-activation magnitude diagnostic (size matching controls
        # unit count, not removed activation)
        "mean_removed_activation_mag": removed_act_mag,
        "pbar_baseline_matrix": json.dumps(
            np.round(classwise_mean_probs(p_base, labels), 6).tolist()),
        "pbar_ablated_matrix": json.dumps(
            np.round(classwise_mean_probs(p_abl, labels), 6).tolist()),
    }
    if dominant_logit is not None and dominant_logit >= 0:
        row["concordant"] = bool(m["c_minus"] == dominant_logit)
        order = np.argsort(delta[m["y_star"]], kind="stable")
        row["dominant_logit_rank"] = int(np.where(order == dominant_logit)[0][0])
        # MI association is sign-blind (rho^2) and masking is not a directed
        # latent change: also record the absolute shift at the associated
        # class; a low reduction-concordance rate is not by itself evidence
        # against a factor's interpretability
        row["associated_class_shift"] = float(delta[m["y_star"], dominant_logit])
        row["associated_class_abs_shift"] = float(
            abs(row["associated_class_shift"]))
    else:
        row["concordant"] = None
        row["dominant_logit_rank"] = None
        row["associated_class_shift"] = None
        row["associated_class_abs_shift"] = None
    row["delta_matrix"] = json.dumps(np.round(delta, 6).tolist())
    return row


def run_ablation_stage_for_factor(
    model_seed, cluster, layer, input_condition, factor_index, factor_rank,
    supports, H_disc, logits_disc, valid_h, dominant_logit,
    trunk: TrunkEvaluator, labels_test, store, cfg,
) -> None:
    """Evaluate LC + random + univariate interventions for one factor.

    supports: dict threshold -> ordered support indices (nested by construction).
    """
    acond = {"stage": "ablation", "model_seed": model_seed, "cluster": cluster,
             "layer": layer, "input_condition": input_condition,
             "factor_index": factor_index}
    if store.is_done(acond, ("ablation_summary",)):
        return
    n_controls = (int(cfg["n_random_controls_hidden_only"])
                  if input_condition == "hidden_only"
                  else int(cfg["n_random_controls"]))
    thresholds = (list(cfg["support_thresholds"])
                  if input_condition == "output_aware"
                  else [HIDDEN_ONLY_FRACTION])
    rows: list[dict] = []
    for q in thresholds:
        sup = supports.get(q) or []
        if not sup:
            continue
        s = len(sup)
        base_int = {
            "model_seed": model_seed, "cluster": cluster, "layer": layer,
            "input_condition": input_condition, "factor_index": factor_index,
            "factor_rank": factor_rank, "threshold": float(q),
            "support_size": s,
            "dominant_logit": int(dominant_logit)
            if dominant_logit is not None else -1,
        }
        # --- Local CorEx intervention -------------------------------------
        p_abl = trunk.ablated_probs(layer, sup)
        row_lc = _summarize_intervention(
            base_int, "local_corex", 0, p_abl, trunk.baseline_probs,
            labels_test, dominant_logit,
            removed_act_mag=trunk.mean_removed_activation(layer, sup))
        rows.append(row_lc)
        # --- matched random supports --------------------------------------
        rand_T, rand_H = [], []
        for draw in range(n_controls):
            s_rand = seeds.derive_seed(model_seed, "random_ablation", cluster,
                                       layer, factor_index, q, draw)
            rng = np.random.default_rng(s_rand)
            valid_idx = np.flatnonzero(valid_h)
            rand_sup = rng.choice(valid_idx, size=s, replace=False).tolist()
            p_abl = trunk.ablated_probs(layer, rand_sup)
            row_r = _summarize_intervention(
                base_int, "random", draw, p_abl, trunk.baseline_probs,
                labels_test, dominant_logit,
                removed_act_mag=trunk.mean_removed_activation(layer, rand_sup))
            rand_T.append(row_r["T_max"])
            rand_H.append(row_r["H_input"])
            rows.append(row_r)
        row_lc["p_vs_random_Tmax"] = random_control_probability(
            row_lc["T_max"], rand_T, larger_is_extreme=True)
        row_lc["p_vs_random_Hinput"] = random_control_probability(
            row_lc["H_input"], rand_H, larger_is_extreme=False)
        # --- univariate logit-association support -------------------------
        if (dominant_logit is not None and dominant_logit >= 0
                and input_condition == "output_aware"):
            uni = univariate_support(H_disc, logits_disc[:, dominant_logit],
                                     s, valid_h)
            p_abl = trunk.ablated_probs(layer, uni)
            row_u = _summarize_intervention(
                base_int, "univariate", 0, p_abl, trunk.baseline_probs,
                labels_test, dominant_logit,
                removed_act_mag=trunk.mean_removed_activation(layer, uni))
            rows.append(row_u)
    store.write(acond, "ablation_summary", rows)


def run_ablation_stage(model_seed, splits, labels_cluster, model, device,
                       store, cfg) -> None:
    """All interventions for one model seed."""
    labels_test = splits["test_y"].numpy()
    x_disc = splits["discovery_x"]
    trunk = TrunkEvaluator(model, splits["test_x"], device)
    for cluster in range(int(cfg["n_clusters"])):
        idx = np.flatnonzero(labels_cluster == cluster)
        if len(idx) == 0:
            continue
        xc = x_disc[torch.tensor(idx)]
        h1, h2, h3, logits_disc = extract_representations(model, xc, device)
        # row-centered logits: softmax-relevant, behaviorally invariant part
        logits_disc = center_logits(logits_disc)
        for layer in cfg["layers"]:
            H = {1: h1, 2: h2, 3: h3}[layer]
            H_std, valid_h, _ = standardize_with_flags(H)
            L_std, _, _ = standardize_with_flags(logits_disc)
            for cond_name in cfg["input_conditions"]:
                fcond = factor_condition(model_seed, cluster, layer, cond_name)
                if not store.is_done(fcond, ("factors", "supports")):
                    continue  # fits stage must have run
                factors = store.shard_path(fcond, "factors")
                import pandas as pd

                fdf = pd.read_parquet(factors)
                sups_df = pd.read_parquet(store.shard_path(fcond, "supports"))
                for _, frow in fdf.iterrows():
                    fi = int(frow["factor_index"])
                    thresholds = (list(cfg["support_thresholds"])
                                  if cond_name == "output_aware"
                                  else [HIDDEN_ONLY_FRACTION])
                    sup_dict = {}
                    for q in thresholds:
                        sub = sups_df[(sups_df["factor_index"] == fi)
                                      & (sups_df["threshold"] == q)]
                        sup_dict[q] = sub.sort_values("mi_rank")[
                            "hidden_unit"].tolist()
                    dominant = frow.get("dominant_logit", -1)
                    dominant = None if (dominant is None or dominant < 0) else int(dominant)
                    run_ablation_stage_for_factor(
                        model_seed, cluster, layer, cond_name, fi,
                        int(frow["factor_rank"]), sup_dict,
                        H_std, logits_disc, valid_h,
                        dominant, trunk, labels_test, store, cfg)


# ---------------------------------------------------------------------------
# Probability-shift storage (large; per intervention 100 rows)
# ---------------------------------------------------------------------------

def write_probability_shifts(model_seed, store, cfg):
    """Expand saved delta matrices into the tidy mnist_probability_shifts table.

    Reads ablation_summary shards (never reruns models) and writes one parquet
    per model seed. Each row carries the class-mean baseline probability,
    ablated probability, and signed change per (true class, output class) —
    the spec-required contents of mnist_probability_shifts. Per-image
    probabilities are not stored individually (they are reconstructible from
    checkpoints); the class means are stored in full.
    """
    import pandas as pd

    rows = []
    df = store.read_all("ablation_summary")
    if df.empty:
        return None
    for _, r in df.iterrows():
        if r["model_seed"] != model_seed:
            continue
        delta = np.array(json.loads(r["delta_matrix"]))
        pbar_base = np.array(json.loads(r["pbar_baseline_matrix"]))
        pbar_abl = np.array(json.loads(r["pbar_ablated_matrix"]))
        for y in range(10):
            for c in range(10):
                rows.append({
                    "model_seed": r["model_seed"], "cluster": r["cluster"],
                    "layer": r["layer"],
                    "input_condition": r["input_condition"],
                    "factor_index": r["factor_index"],
                    "method": r["method"], "random_draw": r["random_draw"],
                    "threshold": r["threshold"],
                    "true_class": y, "output_class": c,
                    "baseline_probability": float(pbar_base[y, c]),
                    "ablated_probability": float(pbar_abl[y, c]),
                    "signed_change": float(delta[y, c]),
                })
    out = pd.DataFrame(rows)
    path = config.RAW_DIR / EXPERIMENT / f"mnist_probability_shifts_seed{model_seed}.parquet"
    out.to_parquet(path, index=False)
    return path


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(cfg: dict, stages: tuple[str, ...] = ("clusters", "fits", "ablations")) -> dict:
    from .data import load_splits
    from .train import load_model

    defaults = {**DEFAULTS, **{k: v for k, v in cfg.items() if k in DEFAULTS}}
    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    run_id = config.make_run_id(EXPERIMENT, defaults, 0)
    setup_logging(EXPERIMENT, run_id)
    if config.is_complete(run_id):
        logger.info("Run %s already complete; skipping", run_id)
        return {"run_id": run_id, "status": "already_complete"}
    config.save_resolved_config(run_id, defaults, extra={"stages": list(stages)})

    store = ResultStore(EXPERIMENT)
    failure_log = FailureLog(EXPERIMENT)

    splits = load_splits(save_dir=config.RAW_DIR / EXPERIMENT)

    if "clusters" in stages:
        labels_cluster = build_discovery_clusters(splits, store,
                                                   int(defaults["n_clusters"]))
    else:
        cond = {"stage": "clusters", "phate_seed": CLUSTER_PHATE_SEED,
                "kmeans_seed": CLUSTER_KMEANS_SEED, "k": int(defaults["n_clusters"])}
        z = pd.read_parquet(store.shard_path(cond, "clusters"))
        labels_cluster = z.set_index("discovery_index")["cluster"].sort_index().to_numpy()

    summary = {}
    for model_seed in defaults["model_seeds"]:
        try:
            model = load_model(model_seed, device=device)
        except FileNotFoundError:
            logger.warning("Model seed %s checkpoint missing; run train.py first",
                           model_seed)
            continue
        t0 = time.time()
        if "fits" in stages:
            run_fit_stage(model_seed, splits, labels_cluster, model, device,
                          store, defaults)
        if "ablations" in stages:
            run_ablation_stage(model_seed, splits, labels_cluster, model,
                               device, store, defaults)
        if "shifts" in stages:
            write_probability_shifts(model_seed, store, defaults)
        summary[model_seed] = time.time() - t0
        logger.info("model seed %s stages %s done in %.0f s", model_seed,
                    stages, summary[model_seed])

    record = {"stages": list(stages), "model_seeds_done": list(summary),
              "elapsed_per_seed_s": summary}
    config.mark_complete(run_id, record)
    write_run_report(EXPERIMENT, run_id, len(summary), 0,
                     {"stages": ",".join(stages)})
    return {"run_id": run_id, **record}


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, default=None)
    ap.add_argument("--stages", type=str, default="clusters,fits,ablations")
    ap.add_argument("--smoke", action="store_true",
                    help="reduced grid: 2 seeds, 3 clusters, 2 factors, "
                         "5 random controls (validates mechanics only)")
    args = ap.parse_args()
    cfg = config.load_config(args.config) if args.config else {}
    if args.smoke:
        cfg = {
            "model_seeds": [5100],
            "n_clusters": 3,
            "layers": [3],
            "n_factors": 4,
            "n_retained": 2,
            "n_random_controls": 3,
            "n_random_controls_hidden_only": 2,
            "support_thresholds": [0.6, 0.8],
            "input_conditions": ["output_aware", "hidden_only"],
        }
    stages = tuple(args.stages.split(","))
    main(cfg, stages=stages)