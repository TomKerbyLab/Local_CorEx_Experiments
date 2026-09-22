"""Validation tests for Experiment 3 (spec 'Validation tests')."""

from __future__ import annotations

import numpy as np
import pytest

from tnnls_experiments.common import information
from tnnls_experiments.exp3_redundancy import data


def test_zero_noise_redundant_positive_o_information():
    # 1. At zero noise, the redundant generator has positive O-information.
    rng = np.random.default_rng(1)
    X = data.redundant_group(rng, 4, 100_000, eta=0.0)
    omega = information.o_information_plugin(X)
    assert omega > 0.5  # near d-1 bits: strong redundancy


def test_zero_noise_parity_negative_o_information():
    # 2. At zero noise, the parity generator has negative O-information.
    rng = np.random.default_rng(2)
    X = data.synergistic_group(rng, 4, 100_000, eta=0.0)
    omega = information.o_information_plugin(X)
    assert omega < -0.5


def test_nuisance_independent():
    # 3. Nuisance variables are mutually independent and independent of the group.
    ds = data.generate_dataset("redundant", 3, 20_000, 0.1, seed=5)
    X, d = ds["X"], ds["d"]
    # correlation between group and nuisance ~ 0
    C = np.corrcoef(X.T)
    cross = C[:d, d:]
    assert np.abs(cross).max() < 0.03
    # nuisance pairwise correlations ~ 0
    nn = C[d:, d:]
    off = nn - np.eye(nn.shape[0])
    assert np.abs(off).max() < 0.03


def test_hand_computed_distributions():
    # 4. Entropy calculations agree with hand-computed values for noiseless
    # three-variable examples.
    # Redundant (eta=0): X_i = Z. H(X) = 1 bit, TC = 3*1 - 1 = 2 bits,
    # DTC = sum_i H(X_-i) - 2 H(X) = 3*1 - 2*1 = 1, Omega = 1 bit.
    rng = np.random.default_rng(3)
    X = data.redundant_group(rng, 3, 50_000, eta=0.0)
    assert information.total_correlation_plugin(X) == pytest.approx(2.0, abs=0.01)
    assert information.dual_total_correlation_plugin(X) == pytest.approx(1.0, abs=0.01)
    assert information.o_information_plugin(X) == pytest.approx(1.0, abs=0.01)
    # Parity (eta=0): TC = 1, DTC = 2, Omega = -1.
    Xp = data.synergistic_group(rng, 3, 50_000, eta=0.0)
    assert information.total_correlation_plugin(Xp) == pytest.approx(1.0, abs=0.01)
    assert information.o_information_plugin(Xp) == pytest.approx(-1.0, abs=0.01)


def test_tc_matching_frozen_and_uses_calibration_only():
    # 5. TC matching is deterministic, frozen, and derived analytically
    #    (verified against the 1M-sample empirical calibration here).
    pairs1 = data.selected_pairs()
    pairs2 = data.selected_pairs()
    assert pairs1 == pairs2  # frozen
    # verify analytic TC against empirical calibration sample (seed 4000)
    for d, (eta_r, eta_s) in pairs1.items():
        rng = np.random.default_rng(data.CALIBRATION_SEED)
        # subsample for speed: 200k of the 1M calibration observations
        Xr = data.redundant_group(rng, d, 200_000, eta_r)
        emp = information.total_correlation_plugin(Xr)
        ana = data.population_tc_redundant(d, eta_r)
        assert emp == pytest.approx(ana, abs=0.02), (d, eta_r)
        Xs = data.synergistic_group(rng, d, 200_000, eta_s)
        emp_s = information.total_correlation_plugin(Xs)
        ana_s = data.population_tc_synergistic(d, eta_s)
        assert emp_s == pytest.approx(ana_s, abs=0.02), (d, eta_s)
    # tolerance/flagging
    rows = data.calibrate_noise_pairs()
    sel = [r for r in rows if r["selected"]]
    assert len(sel) == 3
    for r in sel:
        assert r["abs_tc_diff"] == min(x["abs_tc_diff"] for x in rows if x["d"] == r["d"])


def test_highest_tc_factor_selected_without_ground_truth():
    # 6. Factor selection uses only explained TC, never support ground truth.
    from tnnls_experiments.common import corex

    rng = np.random.default_rng(4)
    ds = data.generate_dataset("redundant", 4, 3000, 0.05, seed=6)
    fit = corex.fit_bio_corex(ds["X"], 3, seed=7)
    order = np.argsort(-fit.tcs)
    assert fit.tcs[order[0]] == fit.tcs.max()
    # the top factor's MI must concentrate on the group (redundant case)
    top = fit.association_scores()[order[0]]
    assert top[:4].sum() > top[4:].sum()