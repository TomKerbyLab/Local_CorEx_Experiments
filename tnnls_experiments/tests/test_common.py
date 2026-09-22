"""Unit tests for shared infrastructure (00_README shared conventions)."""

from __future__ import annotations

import numpy as np
import pytest

from tnnls_experiments.common import config, corex, information, io, metrics, seeds


# --- seeds ------------------------------------------------------------------

def test_seed_derivation_matches_spec_formula():
    # s_{r,j} = hash(s_r, j) mod (2^32 - 1): deterministic, distinct per component
    for root in (1000, 2000, 5100):
        s1 = seeds.derive_seed(root, "data")
        s2 = seeds.derive_seed(root, "corex")
        assert s1 == seeds.derive_seed(root, "data")
        assert s1 != s2
        assert 0 < s1 < 2**32 - 1


def test_seed_derivation_type_stability():
    # different tag types must hash differently (no silent collisions)
    assert seeds.derive_seed(1000, 1) != seeds.derive_seed(1000, "1")
    assert seeds.derive_seed(1000, (1, 2)) != seeds.derive_seed(1000, (2, 1))
    # floats are hashed by repr: stable across runs
    assert seeds.derive_seed(1000, 0.5) == seeds.derive_seed(1000, 0.5)


# --- metrics ----------------------------------------------------------------

def test_summarize_replicates():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    s = metrics.summarize_replicates(x)
    assert s["mean"] == pytest.approx(2.5)
    assert s["n_valid"] == 4
    assert s["sd"] == pytest.approx(np.std(x, ddof=1))
    assert s["ci_low"] < s["mean"] < s["ci_high"]


def test_paired_difference_summary():
    x = np.arange(1.0, 11.0)
    y = x - 0.5
    s = metrics.paired_difference_summary(x, y)
    assert s["mean_diff"] == pytest.approx(0.5)
    assert s["se_diff"] == pytest.approx(0.0, abs=1e-12)
    assert s["n_pairs"] == 10


def test_topk_tie_break_by_lower_index():
    scores = np.array([0.5, 0.5, 0.1])
    truth = np.array([1, 0, 0], dtype=bool)
    m = metrics.topk_metrics(scores, truth, k=1)
    assert m["precision"] == 1.0  # index 0 wins the tie


def test_cumulative_support_valid_mask():
    scores = np.array([0.5, 0.0, 0.3, 0.2])
    valid = np.array([True, True, True, False])
    sup = metrics.cumulative_mi_support(scores, 0.9, valid_mask=valid)
    assert 3 not in sup.indices  # excluded coordinate never selected


def test_nested_supports():
    scores = np.abs(np.random.default_rng(1).normal(size=30))
    fracs = [0.6, 0.7, 0.8, 0.9]
    sups = metrics.nested_supports(scores, fracs)
    for a, b in zip(sups, sups[1:]):
        assert set(a.indices) <= set(b.indices)
    assert sups[0].size <= sups[-1].size


def test_hungarian_square_and_rectangular():
    m = np.array([[0.9, 0.1], [0.2, 0.8], [0.7, 0.3]])
    r, c, v = metrics.hungarian_match(m)
    assert len(r) == 2 and len(set(r)) == 2 and len(set(c)) == 2
    # 3 learned x 2 true: best is learned 0 -> true 0, learned 1 -> true 1
    assert list(c) == [0, 1]
    assert v.sum() == pytest.approx(0.9 + 0.8)


# --- information ---------------------------------------------------------------

def test_parity_tc_is_one_bit():
    # noiseless 3-way parity: pairwise MI = 0, TC = 1 bit,
    # DTC = H(X) - sum H(X_i|X_{-i}) = 2 - 0 = 2 bits (each bit determined
    # by the others), so Omega = -1: synergy-dominated.
    rng = np.random.default_rng(2)
    n = 200_000
    b1 = rng.integers(0, 2, n)
    b2 = rng.integers(0, 2, n)
    b3 = b1 ^ b2
    X = np.stack([b1, b2, b3], axis=1)
    assert information.total_correlation_plugin(X) == pytest.approx(1.0, abs=0.01)
    assert information.dual_total_correlation_plugin(X) == pytest.approx(2.0, abs=0.01)
    assert information.o_information_plugin(X) == pytest.approx(-1.0, abs=0.01)


def test_redundant_common_cause_o_information_positive():
    # X_i = Z xor E_i with SMALL flip noise eta_R (spec grid {0.01..0.20}):
    # redundancy-dominated, Omega > 0. (Fair-coin noise would destroy all
    # dependence: xor with Bern(1/2) fully randomizes.)
    rng = np.random.default_rng(3)
    n = 100_000
    Z = rng.integers(0, 2, n)
    E = (rng.random((n, 3)) < 0.05).astype(int)  # eta_R = 0.05
    X = (Z[:, None] + E) % 2
    tc = information.total_correlation_plugin(X)
    omega = information.o_information_plugin(X)
    assert tc > 1.0  # near 2 bits: all three X_i copy Z with small noise
    assert omega > 0


def test_hand_computed_entropy_three_vars():
    # validation test 4 (exp3): hand-computed distributions, noiseless
    # X1,X2 ~ Bern(1/2), X3 = X1 xor X2
    # H(X1)=H(X2)=H(X3)=1, H(X)=2, TC=1, DTC=1, Omega=0
    X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 0]])
    assert information.entropy_plugin(X[:, 0]) == pytest.approx(1.0)
    h_joint, k = information.joint_entropy_plugin(X)
    assert h_joint == pytest.approx(2.0)
    assert k == 4
    assert information.total_correlation_plugin(X) == pytest.approx(1.0)


def test_miller_madow_reduces_upward_entropy_bias():
    # plug-in entropy is biased low; MM corrects upward by (K-1)/(2n ln2)
    rng = np.random.default_rng(4)
    x = rng.integers(0, 2, 200)
    h_mm, h = information.entropy_plugin_mm(x)
    assert h_mm == pytest.approx(h + (2 - 1) / (2 * 200 * np.log(2)), rel=1e-6)
    # at n=30 the correction is more visible and always upward
    small = rng.integers(0, 2, 30)
    h_mm_s, h_s = information.entropy_plugin_mm(small)
    assert h_mm_s > h_s


def test_tc_mm_direction():
    # For 3-variable parity, K_i = 2 for each marginal and K_J = 4 occupied
    # joint cells, so the MM corrections cancel: TC_MM == TC_plugin.
    rng = np.random.default_rng(7)
    n = 200
    b1 = rng.integers(0, 2, n); b2 = rng.integers(0, 2, n); b3 = b1 ^ b2
    X = np.stack([b1, b2, b3], 1)
    tc_mm, tc_p = information.total_correlation_plugin_mm(X)
    assert tc_mm == pytest.approx(tc_p, rel=1e-9)
    # Redundant common cause X_i = Z xor E_i with fair noise: the X joint
    # has 8 occupied cells, K_i = 2 each, so
    # TC_MM = TC_plugin + (3*(K_i-1) - (K_J-1)) / (2 n ln2) = TC_p - 4/(2n ln2).
    Z = rng.integers(0, 2, n)
    E = rng.integers(0, 2, size=(n, 3))
    Xr = (Z[:, None] + E) % 2
    tc_mm_r, tc_p_r = information.total_correlation_plugin_mm(Xr)
    expected = tc_p_r + (3 * (2 - 1) - (8 - 1)) / (2 * n * np.log(2))
    assert tc_mm_r == pytest.approx(expected, rel=1e-9)


def test_gaussian_tc_zero_for_independent():
    rng = np.random.default_rng(9)
    X = rng.standard_normal((5000, 10))
    R = np.corrcoef(X.T)
    assert information.gaussian_total_correlation(R) == pytest.approx(0.0, abs=0.05)


def test_gaussian_tc_increases_with_correlation():
    rng = np.random.default_rng(10)
    n = 5000
    f = rng.standard_normal(n)
    X = np.stack([f, f + rng.standard_normal(n) * 0.2], 1)
    R = np.corrcoef(X.T)
    tc = information.gaussian_total_correlation(R)
    assert tc > 1.0
    with pytest.raises(ValueError):
        information.gaussian_total_correlation(np.array([[1.0, 2.0], [2.0, 1.0]]))


# --- corex wrappers -----------------------------------------------------------

def test_linear_corex_fit_record():
    rng = np.random.default_rng(11)
    n = 400
    Z = rng.standard_normal((n, 1))
    X = np.hstack([Z + rng.standard_normal((n, 1)) * 0.3 for _ in range(3)])
    rec = corex.fit_linear_corex(X, n_factors=2, seed=123)
    assert rec.mis.shape == (2, 3)
    assert rec.tcs.shape == (2,)
    assert rec.total_tc() > 0
    assert np.all(rec.association_scores() >= 0)


def test_standardize_columns_flags_low_variance():
    X = np.random.default_rng(12).normal(size=(100, 3))
    X[:, 2] = 1.5
    Xs, valid, excluded = corex.standardize_columns(X)
    assert excluded.tolist() == [False, False, True]
    assert valid.tolist() == [True, True, False]
    assert np.allclose(Xs[:, :2].mean(0), 0, atol=1e-12)
    assert np.allclose(Xs[:, :2].std(0), 1, atol=1e-12)


def test_discretize_equal_frequency():
    rng = np.random.default_rng(13)
    X = rng.normal(size=(1000, 2))
    Xb, edges = corex.discretize_equal_frequency(X, 10)
    assert Xb.min() >= 0 and Xb.max() <= 9
    # equal-frequency: each bin has ~100 members
    counts = np.bincount(Xb[:, 0])
    assert np.all(np.abs(counts - 100) <= 2)


# --- store ---------------------------------------------------------------------

def test_result_store_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "RAW_DIR", tmp_path)
    from tnnls_experiments.common import store as store_mod

    monkeypatch.setattr(store_mod, "RAW_DIR", tmp_path)
    store = store_mod.ResultStore("unit")
    cond = {"a": 1, "b": "x"}
    assert not store.is_done(cond, ("rows",))
    store.write(cond, "rows", [{"a": 1, "v": 2.0}])
    assert store.is_done(cond, ("rows",))
    out = store.read_all("rows")
    assert len(out) == 1 and out["v"][0] == 2.0
    # never overwrite: writing same condition again raises
    with pytest.raises(FileExistsError):
        store.write(cond, "rows", [{"a": 1, "v": 3.0}])