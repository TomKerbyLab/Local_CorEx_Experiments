"""End-to-end mechanics tests for the Exp 4 benchmark."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tnnls_experiments.common import metrics, seeds
from tnnls_experiments.exp4_mnist.benchmark import (
    TrunkEvaluator, build_discovery_clusters, effective_n_units, center_logits,
    fit_and_rank, posthoc_logit_mi, run_ablation_stage_for_factor,
    standardize_with_flags, tc_rank_order, univariate_support)
from tnnls_experiments.exp4_mnist.classifier import MLP
from tnnls_experiments.exp4_mnist.interventions import (
    classwise_mean_probs, classwise_shift_matrix, individual_effect_metrics,
    predict_proba)


def _trained_tiny_model(n=2000, epochs=3, seed=0):
    """Tiny quick-trained model with realistic hidden structure."""
    torch.manual_seed(seed)
    model = MLP(dropout_p=0.5)
    xs = torch.rand(n, 784)
    ys = (xs[:, :10].sum(1) > 5).long() * 5  # crude signal in first pixels
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    lossf = torch.nn.CrossEntropyLoss()
    for _ in range(epochs):
        logits = model(xs)
        loss = lossf(logits, ys)
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    return model, xs, ys


def test_trunk_equivalence_with_model_mask():
    # The cached-activation trunk must reproduce the canonical model-mask
    # forward pass exactly (this is validation test 1/3/4/6 in aggregate).
    model, xs, ys = _trained_tiny_model()
    trunk = TrunkEvaluator(model, xs[:500], device="cpu")
    for layer in (1, 2, 3):
        sup = [1, 5, 10, 50, 120]
        if layer == 1:
            sup = [1, 5, 10, 50, 120, 400]
        p_trunk = trunk.ablated_probs(layer, sup)
        p_model = trunk.probs_from_model_mask(layer, sup)
        assert np.allclose(p_trunk, p_model, atol=1e-6), f"layer {layer}"
    # empty support = baseline
    p_empty = trunk.ablated_probs(2, [])
    assert np.allclose(p_empty, trunk.baseline_probs, atol=1e-7)


def test_baseline_probs_match_model():
    model, xs, ys = _trained_tiny_model()
    trunk = TrunkEvaluator(model, xs[:300], device="cpu")
    p = predict_proba(model, xs[:300]).numpy()
    assert np.allclose(trunk.baseline_probs, p, atol=1e-7)


def test_fit_and_rank_shapes_and_ranking():
    rng = np.random.default_rng(1)
    n = 400
    # correlated hidden coords + one logit
    z = rng.standard_normal((n, 1))
    H = np.hstack([z + 0.3 * rng.standard_normal((n, 12)) for _ in range(4)])
    L = (z + rng.standard_normal((n, 1)) * 0.5)
    H_std, valid, _ = standardize_with_flags(H)
    L_std, _, _ = standardize_with_flags(L)
    fr = fit_and_rank(H_std, L_std, n_factors=4, seed=3, n_retained=2)
    assert fr["order"].shape == (2,)
    # order is descending in TC
    tcs = fr["fit"].tcs
    assert tcs[fr["order"][0]] >= tcs[fr["order"][1]]
    assert fr["logit_mis"].shape == (4, 1)
    assert fr["dominant"].shape == (4,)


def test_univariate_support_size_and_exclusion():
    rng = np.random.default_rng(2)
    H = rng.standard_normal((300, 50))
    H[:, 3] = 1.0  # degenerate column
    valid = np.ones(50, bool)
    valid[3] = False
    l = H[:, 7] * 2 + rng.standard_normal(300) * 0.1
    sup = univariate_support(H, l, size=10, valid=valid)
    assert len(sup) == 10
    assert 3 not in sup           # excluded coordinates never selected
    assert len(set(sup)) == 10     # unique
    assert 7 in sup                # the strongest correlate is included


def test_factor_supports_match_intervention_sizes():
    # supports built by cumulative MI have recorded sizes equal to the
    # support used in interventions (checked via nested_supports)
    scores = np.abs(np.random.default_rng(4).normal(size=300))
    valid = np.ones(300, bool)
    valid[299] = False
    sups = metrics.nested_supports(scores, [0.6, 0.8], valid)
    assert all(299 not in s.indices for s in sups)
    assert sups[0].size <= sups[1].size


def test_no_test_data_in_fit_functions():
    # fit_and_rank and univariate_support accept only discovery arrays by
    # construction; verify they never touch torch test tensors: static check
    import inspect
    from tnnls_experiments.exp4_mnist import benchmark

    src_fit = inspect.getsource(benchmark.fit_and_rank)
    assert "test" not in src_fit.replace("latest", "")
    src_uni = inspect.getsource(benchmark.univariate_support)
    assert "test" not in src_uni.replace("latest", "")


def test_smoke_benchmark_end_to_end(tmp_path, monkeypatch):
    # Reduced-grid end-to-end: train tiny model, build clusters, fit, ablate.
    import pandas as pd

    from tnnls_experiments.common import config as cfg_mod, store as store_mod
    monkeypatch.setattr(cfg_mod, "RAW_DIR", tmp_path)
    monkeypatch.setattr(store_mod, "RAW_DIR", tmp_path)
    (tmp_path / "mnist").mkdir(parents=True, exist_ok=True)

    model, xs, ys = _trained_tiny_model(n=1200, epochs=2)
    # fake splits: discovery = first 1000, test = last 200
    import torch as T

    splits = {
        "discovery_x": xs[:1000], "discovery_y": T.tensor(ys[:1000]),
        "test_x": xs[1000:], "test_y": T.tensor(ys[1000:]),
    }
    store = store_mod.ResultStore("mnist")

    # cluster labels via kmeans directly (PHATE too slow for a unit test;
    # build_discovery_clusters is exercised in the smoke run)
    from sklearn.cluster import KMeans

    km = KMeans(n_clusters=2, n_init=3, random_state=1)
    labels_cluster = km.fit_predict(splits["discovery_x"].numpy())

    cfg = {
        "n_factors": 3, "n_retained": 2, "support_thresholds": [0.6, 0.8],
        "n_random_controls": 2, "n_random_controls_hidden_only": 1,
        "input_conditions": ["output_aware"], "layers": [3],
        "n_clusters": 2,
    }
    from tnnls_experiments.exp4_mnist.benchmark import run_fit_stage
    run_fit_stage(99, splits, labels_cluster, model, "cpu", store, cfg)
    factors = store.read_all("factors")
    sups = store.read_all("supports")
    assert len(factors) == 4          # 2 clusters x 2 retained factors
    assert set(factors["factor_rank"]) == {0, 1}
    # nested supports
    for (_, _, _, fi), grp in sups.groupby(
            ["model_seed", "cluster", "layer", "factor_index"]):
        s6 = set(grp[grp["threshold"] == 0.6]["hidden_unit"])
        s8 = set(grp[grp["threshold"] == 0.8]["hidden_unit"])
        assert s6 <= s8

    from tnnls_experiments.exp4_mnist.benchmark import run_ablation_stage
    run_ablation_stage(99, splits, labels_cluster, model, "cpu", store, cfg)
    abl = store.read_all("ablation_summary")
    # per factor-threshold: 1 LC + 2 random + 1 univariate
    n_expected = len(sups.groupby(["cluster", "factor_index", "threshold"]))
    assert len(abl) == n_expected * (1 + 2 + 1)
    # every intervention has a full delta matrix
    import json as J

    for _, r in abl.iterrows():
        delta = np.array(J.loads(r["delta_matrix"]))
        assert delta.shape == (10, 10)
        assert np.abs(delta.sum(axis=1)).max() < 1e-5
    # support sizes match exactly across methods (validation test 8)
    for _, grp in abl.groupby(
            ["model_seed", "cluster", "layer", "input_condition",
             "factor_index", "threshold"]):
        assert grp["support_size"].nunique() == 1


def _latent_block_data(n=400, n_h=40, seed=1):
    """Hidden coords driven by 3 latents; logit 3 correlates with latent 0."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3))
    cols = [z[:, j % 3] + 0.5 * rng.standard_normal(n) for j in range(n_h)]
    H = np.stack(cols, axis=1)
    L = np.stack([rng.standard_normal(n) for _ in range(10)], axis=1)
    L[:, 3] = z[:, 0] + 0.2 * rng.standard_normal(n)
    return H, L


def test_tiny_cluster_excluded_from_grid(tmp_path, monkeypatch):
    # Predeclared tiny-cluster rule: clusters with < MIN_CLUSTER_OBS
    # discovery observations define no fits and no interventions, and the
    # exclusion is visible, never silent padding.
    from tnnls_experiments.common import config as cfg_mod, store as store_mod
    monkeypatch.setattr(cfg_mod, "RAW_DIR", tmp_path)
    monkeypatch.setattr(store_mod, "RAW_DIR", tmp_path)
    (tmp_path / "mnist").mkdir(parents=True, exist_ok=True)

    model, xs, ys = _trained_tiny_model(n=1200, epochs=2)
    import torch as T
    splits = {
        "discovery_x": xs[:1000], "discovery_y": T.tensor(ys[:1000]),
        "test_x": xs[1000:], "test_y": T.tensor(ys[1000:]),
    }
    store = store_mod.ResultStore("mnist")
    labels_cluster = np.array([0] * 940 + [1] * 60)  # cluster 1 is tiny
    cfg = {
        "n_factors": 3, "n_retained": 2, "support_thresholds": [0.6],
        "n_random_controls": 2, "n_random_controls_hidden_only": 1,
        "input_conditions": ["output_aware"], "layers": [3],
        "n_clusters": 2,
    }
    from tnnls_experiments.exp4_mnist.benchmark import (MIN_CLUSTER_OBS,
                                                        run_fit_stage,
                                                        run_ablation_stage)
    assert MIN_CLUSTER_OBS == 100
    run_fit_stage(99, splits, labels_cluster, model, "cpu", store, cfg)
    factors = store.read_all("factors")
    assert set(factors["cluster"]) == {0}   # tiny cluster has no fits
    run_ablation_stage(99, splits, labels_cluster, model, "cpu", store, cfg)
    from tnnls_experiments.exp4_mnist.summarize import coverage
    cov = coverage(store.read_all("ablation_summary"), factors,
                   cluster_sizes={0: 940, 1: 60})
    assert cov.attrs["tiny_clusters_excluded"] == {1: 60}
    assert "excluded [1]" in cov.attrs["note"]


def test_assignment_checks_synthetic():
    # Machinery check on synthetic blobs with an identity PHATE stub: the
    # self-transform agreement and kNN-consistency rates are computed and
    # aligned (blobs separated; k-means recovers the blocks).
    from sklearn.cluster import KMeans

    from tnnls_experiments.exp4_mnist.benchmark import (
        TEST_ASSIGNMENT_CHECK_SUBSET, assignment_checks)

    class _IdentityPHATE:
        def transform(self, X):
            return np.asarray(X, dtype=float)

    rng = np.random.default_rng(0)
    centers = rng.standard_normal((3, 10)) * 8
    emb_disc = np.vstack([centers[c] + rng.standard_normal((200, 10))
                          for c in range(3)])
    km = KMeans(3, n_init=3, random_state=0).fit(emb_disc)
    labels_fit = km.labels_
    emb_test = np.vstack([centers[c] + rng.standard_normal((50, 10))
                          for c in range(3)])
    # the assignment under test: k-means labels from the fitted object
    labels_test = km.predict(emb_test)
    checks, dist, ind = assignment_checks(
        emb_disc, labels_fit, _IdentityPHATE(), km, emb_disc, emb_test,
        labels_test, emb_test)
    assert checks["self_transform_agreement"] == pytest.approx(1.0)
    assert checks["knn_consistency"] > 0.9
    assert checks["n_self_check"] == len(emb_disc) // TEST_ASSIGNMENT_CHECK_SUBSET
    assert dist.shape == (150, 1) and ind.shape == (150, 1)


def test_posthoc_logit_mi_identifies_associated_logit():
    # Hidden-only fit + post-hoc association: every factor built from latent
    # 0-family tracks logit 3, and that logit dominates for those factors.
    H, L = _latent_block_data()
    H_std, valid, _ = standardize_with_flags(H)
    L_std, _, _ = standardize_with_flags(L)
    fr = fit_and_rank(H_std, None, n_factors=5, seed=7,
                      n_retained=3, valid_h=valid)
    assert fr["logit_mis"] is None and fr["dominant"][0] == -1  # pre-posthoc
    v_idx = np.flatnonzero(valid)
    mis, dominant, gaps = posthoc_logit_mi(fr["fit"], H_std[:, v_idx], L_std)
    assert mis.shape == (5, 10) and len(gaps) == 5
    for j in fr["order"]:
        assert mis[j].min() >= 0.0 and gaps[j] >= -1e-9
        if mis[j, 3] >= 0.05:  # substantive association, not sampling noise
            assert dominant[j] == 3
            assert mis[j, 3] == pytest.approx(mis[j].max())
    assert (mis[:, 3] >= 0.05).sum() >= 1  # at least one z0-driven factor


def test_fit_excludes_invalid_hidden_columns():
    # Spec: near-zero-variance coordinates are excluded from CorEx fitting.
    H, L = _latent_block_data()
    H[:, 5] = 2.0  # constant column
    H_std, valid, _ = standardize_with_flags(H)
    L_std, _, _ = standardize_with_flags(L)
    assert valid.sum() == H.shape[1] - 1 and not valid[5]
    fr = fit_and_rank(H_std, L_std, n_factors=5, seed=3,
                      n_retained=3, valid_h=valid)
    assert all(np.isfinite(fr["fit"].tcs))
    assert np.abs(fr["hidden_mis"][:, 5]).max() == 0.0  # zero, not fit
    sups = metrics.nested_supports(fr["hidden_mis"][int(fr["order"][0])],
                                   [0.6, 0.8], valid)
    for s in sups:
        assert 5 not in s.indices


def test_pbar_matrices_consistent_with_delta():
    # pbar_abl = pbar_base + delta, and baseline rows sum to 1.
    rng = np.random.default_rng(9)
    n = 2000
    labels = rng.integers(0, 10, n)
    p_base_np = rng.dirichlet(np.ones(10), size=n)
    p_abl_np = rng.dirichlet(np.ones(10), size=n)  # same marginals family
    delta = classwise_shift_matrix(p_abl_np, p_base_np, labels)
    pb0 = classwise_mean_probs(p_base_np, labels)
    pbA = classwise_mean_probs(p_abl_np, labels)
    assert np.allclose(pbA, pb0 + delta, atol=1e-9)
    assert np.allclose(pb0.sum(axis=1), 1.0, atol=1e-9)


def test_centered_logits_remove_behaviorally_irrelevant_component():
    # Softmax invariance: adding a per-image constant leaves predictions.
    torch.manual_seed(0)
    m = MLP(dropout_p=0.5).eval()
    x = torch.rand(128, 784)
    base = m(x)
    c = torch.rand(len(x), 1)
    with torch.no_grad():
        shifted = torch.softmax(base + c, dim=1)
    assert torch.allclose(shifted, torch.softmax(base, dim=1), atol=1e-6)
    # Discovery association: raw logit correlation is polluted by the common
    # mode; centering removes exactly it.
    rng = np.random.default_rng(4)
    n = 400
    common = rng.standard_normal(n)
    # H correlates ONLY with the common logit mode (behaviorally irrelevant
    # under softmax) and with no differential component
    H = np.stack([common + 0.3 * rng.standard_normal(n)
                  for _ in range(5)], axis=1)
    L = np.stack([common + 0.1 * rng.standard_normal(n)
                  for _ in range(10)], axis=1)
    corr_raw = np.corrcoef(H[:, 0], L[:, 3])[0, 1]
    Lc = center_logits(L)  # centering removes the pure common mode
    assert np.abs(Lc[:, 3]).std() < 0.15  # only the small residual remains
    corr_cen = np.abs(np.corrcoef(H[:, 0], Lc[:, 3])[0, 1])
    assert abs(corr_raw) > 0.5 and corr_cen < 0.2


def test_individual_effect_metrics_bounds():
    # U_y >= T_y (triangle inequality); flip rates in [0, 1]; equality case
    # when every image shifts the same direction.
    rng = np.random.default_rng(11)
    n = 1000
    labels = rng.integers(0, 10, n)
    p0 = rng.dirichlet(np.ones(10), size=n)
    pA = rng.dirichlet(np.ones(10), size=n)
    delta = classwise_shift_matrix(pA, p0, labels)
    T = 0.5 * np.abs(delta).sum(axis=1)
    m = individual_effect_metrics(pA, p0, labels)
    assert np.all(np.array(m["U_per_class"]) >= T - 1e-9)
    assert 0.0 <= m["flip_rate"] <= 1.0
    assert m["U_max"] == pytest.approx(max(m["U_per_class"]))
    # same-direction case: U == T on the affected class
    p0b = np.full((100, 10), 0.1)
    pAb = p0b.copy()
    pAb[:, 3] = 0.05
    pAb[:, 7] = 0.15
    labels_b = np.full(100, 4)
    delta_b = classwise_shift_matrix(pAb, p0b, labels_b)
    T_b = 0.5 * np.abs(delta_b[4]).sum()
    mb = individual_effect_metrics(pAb, p0b, labels_b)
    assert mb["U_per_class"][4] == pytest.approx(T_b, abs=1e-9)


def test_tc_rank_order_positive_and_neutral():
    # Nonpositive-TC factors are never retained; TC ties are broken by the
    # seeded permutation, deterministically, not by column layout.
    tcs = np.array([0.0, 0.5, 0.5, -1.0, 0.3, 0.2])
    order = tc_rank_order(tcs, n_retained=5, tie_seed=42)
    assert 0 not in order and 3 not in order
    assert len(order) == 4  # only positive-TC factors
    order2 = tc_rank_order(tcs, n_retained=5, tie_seed=42)
    assert np.array_equal(order, order2)       # deterministic given seed
    assert tcs[order[0]] == pytest.approx(0.5)  # strongest first
    assert tc_rank_order(np.array([0.0, -0.1]), 5, 1).shape == (0,)


def test_effective_n_units_concentrated_vs_uniform():
    scores = np.zeros(50)
    valid = np.ones(50, bool)
    scores[7] = 3.0                      # single dominant unit -> ~1
    assert effective_n_units(scores, valid) == pytest.approx(1.0, abs=1e-9)
    scores = np.zeros(50)
    scores[:10] = 2.0                    # 10 equal units -> ~10
    assert effective_n_units(scores, valid) == pytest.approx(10.0, rel=1e-9)
    assert np.isnan(effective_n_units(np.zeros(10), np.ones(10, bool)))


def test_removed_activation_magnitude_diagnostic():
    # The trunk caches per-unit mean |activation|; a large-|h| support has a
    # larger removal magnitude than a random one of the same size.
    model, xs, ys = _trained_tiny_model(n=2000, epochs=2)
    trunk = TrunkEvaluator(model, xs[:500], device="cpu")
    vals = trunk.unit_magnitude[3]
    top = list(np.argsort(-vals)[:20])
    low = list(np.argsort(vals)[:20])  # 20 smallest
    assert trunk.mean_removed_activation(3, top) > \
        trunk.mean_removed_activation(3, low)