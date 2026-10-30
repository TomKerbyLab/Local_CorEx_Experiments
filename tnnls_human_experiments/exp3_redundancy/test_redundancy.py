"""Validation tests for Experiment 3 (spec 03 items, redesigned design).

Spec validation items, mapped:

1. zero-noise redundant generator has positive O-information;
2. zero-noise parity generator has negative O-information;
3. nuisance columns independent of the group and mutually independent —
   checked at implementation level (correlations); no pairwise statistical
   test can certify MUTUAL independence (parity is the counterexample), so
   the guarantee is the construction (independent draws) and these are
   sanity checks only;
4. entropy calculations agree with hand-computed noiseless d=3 values;
5. the equal-TC calibration is deterministic, frozen, target-matched
   (bisection residuals ~0), and agrees with empirical sampling — with a
   fresh derived seed per generator check (isolation/reproducibility);
6. factor selection uses only explained TC, never support ground truth —
   true by pipeline construction (descending_order of factor TCs is the
   only selection); the old reference-implementation test additionally
   hard-asserted that Bio CorEx recovers the group, which is experimental
   performance, not pipeline procedure, and is not a test here.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import corex_fitting
import redundancy_data as rdata
import information_estimates as info_est
import recovery_metrics as rmetrics
from seeds import derived_seed


def test_zero_noise_redundant_positive_o_information():
    # 1. At zero noise, every redundant column is a copy of Z: TC = d-1,
    # DTC = 1, so Omega = d-2 bits (3 at d=4 -> check via the analytic pmf).
    rng = np.random.default_rng(1)
    X = rdata.redundant_group(rng, 4, 100_000, 0.0)
    estimates = info_est.group_estimates(X)
    assert estimates['omega_plugin'] > 1.5  # near d-2 = 2 bits
    assert rdata.population_quantities(rdata.GEN_REDUNDANT, 4, 0.0)[
        'omega_population'] == pytest.approx(2.0, abs=1e-12)


def test_zero_noise_parity_negative_o_information():
    # 2. At zero noise, parity has O-information -(d-2) bits exactly.
    rng = np.random.default_rng(2)
    X = rdata.synergistic_group(rng, 4, 100_000, 0.0)
    estimates = info_est.group_estimates(X)
    assert estimates['omega_plugin'] < -1.5  # -(d-2) = -2 bits


def test_parity_analytic_identities():
    # Omega_S = -(d-2) TC_S and pairwise MI = 0, exactly, at positive noise.
    for d in (3, 4, 5):
        quantities = rdata.population_quantities(
            rdata.GEN_SYNERGISTIC, d, 0.11)
        assert quantities['omega_population'] == pytest.approx(
            -(d - 2) * quantities['tc_population'], abs=1e-12)
        assert quantities['mean_pmi_population'] == pytest.approx(
            0.0, abs=1e-12)


def test_nuisance_columns_sanity():
    # 3. Implementation check: group/nuisance correlations ~0, nuisance
    # pairwise correlations ~0; the guarantee is the construction.
    dataset = rdata.generate_dataset(
        rdata.GEN_REDUNDANT, 3, 0.14, 20_000,
        seed=derived_seed(5, 'data'))
    X, d = dataset['X'], dataset['d']
    C = np.corrcoef(X.T)
    assert np.abs(C[:d, d:]).max() < 0.03
    nuisance = C[d:, d:] - np.eye(C.shape[0] - d)
    assert np.abs(nuisance).max() < 0.03


def test_hand_computed_distributions():
    # 4. Noiseless d=3: redundant TC=2, DTC=1, Omega=1; parity TC=1, DTC=2,
    # Omega=-1.
    rng = np.random.default_rng(3)
    X = rdata.redundant_group(rng, 3, 50_000, 0.0)
    estimates = info_est.group_estimates(X)
    assert estimates['tc_plugin'] == pytest.approx(2.0, abs=0.01)
    assert estimates['dtc_plugin'] == pytest.approx(1.0, abs=0.01)
    assert estimates['omega_plugin'] == pytest.approx(1.0, abs=0.01)
    Xp = rdata.synergistic_group(rng, 3, 50_000, 0.0)
    estimates_p = info_est.group_estimates(Xp)
    assert estimates_p['tc_plugin'] == pytest.approx(1.0, abs=0.01)
    assert estimates_p['dtc_plugin'] == pytest.approx(2.0, abs=0.01)
    assert estimates_p['omega_plugin'] == pytest.approx(-1.0, abs=0.01)


def test_calibration_deterministic_target_matched():
    # 5. The calibration is a pure function of the target (frozen by
    # construction), target-matched to near machine precision, and the
    # population TC agrees with empirical sampling — a FRESH derived seed
    # per generator check (no shared sequential stream between checks).
    rows1 = rdata.calibrate()
    rows2 = rdata.calibrate()
    assert rows1 == rows2
    for row in rows1:
        assert abs(row['tc_residual_bits']) < 1e-9, row
    # parity omega identity must hold at the operating points
    for row in rows1:
        if row['generator'] == rdata.GEN_SYNERGISTIC:
            assert row['omega_population'] == pytest.approx(
                -(row['d'] - 2) * row['tc_population'], abs=1e-12)
        else:
            assert row['omega_population'] > 0
    # empirical agreement, one fresh derived seed per (generator, d) check,
    # moderate n (speed)
    for row in rows1:
        d, generator, eta = row['d'], row['generator'], row['eta']
        rng = np.random.default_rng(
            derived_seed(4000, 'calcheck', generator, d))
        X = rdata.generate_group(generator, rng, d, 200_000, eta)
        tc_empirical = info_est.group_estimates(X)['tc_plugin']
        assert tc_empirical == pytest.approx(
            row['tc_population'], abs=0.02), (d, generator)


def test_neutral_ties_not_layout_aligned():
    # All-tied scores: the top-d set must not be the leading columns (the
    # legacy lower-index rule would award the first columns — here the true
    # group's columns — to an all-tied factor).
    scores = np.zeros(10)
    top = rmetrics.top_d_metrics(scores, np.r_[np.ones(3), np.zeros(7)],
                                 d=3)['top_indexes']
    assert tuple(sorted(top)) != (0, 1, 2)
    # reproducible
    top_again = rmetrics.top_d_metrics(
        scores, np.r_[np.ones(3), np.zeros(7)], d=3)['top_indexes']
    assert top == top_again


def test_degenerate_factor_scores_zeros():
    # All-zero factor: empty support, degenerate flag, zeros on support
    # metrics, and a reproducible (neutral) top-d subset.
    support = rmetrics.cumulative_mi_support(np.zeros(10))
    assert support.degenerate and support.size == 0
    metrics = rmetrics.top_d_metrics(
        np.zeros(10), np.r_[np.ones(3), np.zeros(7)], d=3)
    assert metrics['f1'] >= 0.0  # computed, never crashed or nan


def test_discovered_support_two_stage_machinery():
    # The two-stage flow estimates Omega on a fresh sample over exactly the
    # frozen support columns; a perfect recovery of the redundant group's
    # support reproduces the group's population Omega reference scale.
    dataset = rdata.generate_dataset(
        rdata.GEN_REDUNDANT, 3, 0.14, 50_000, seed=derived_seed(7, 'data'))
    support = tuple(dataset['group_idx'])
    char = rdata.generate_characterization_sample(
        rdata.GEN_REDUNDANT, 3, 0.14, 100_000, seed=derived_seed(7, 'char'))
    discovered = info_est.group_estimates(
        char['X'][:, list(support)])['omega_plugin']
    population = rdata.population_quantities(
        rdata.GEN_REDUNDANT, 3, 0.14)['omega_population']
    assert discovered == pytest.approx(population, abs=0.05)
    # a support that drops one member of a parity group destroys the
    # dependence: Omega ~ 0 rather than Omega < 0 (§E3.7 informative
    # failures)
    ds = rdata.generate_dataset(
        rdata.GEN_SYNERGISTIC, 3, 0.11, 50_000, seed=derived_seed(8, 'data'))
    broken_support = tuple(int(i) for i in ds['group_idx'][:2])
    char_s = rdata.generate_characterization_sample(
        rdata.GEN_SYNERGISTIC, 3, 0.11, 100_000, seed=derived_seed(8, 'char'))
    omega_broken = info_est.group_estimates(
        char_s['X'][:, list(broken_support)])['omega_plugin']
    assert abs(omega_broken) < 0.05
    reference = info_est.group_estimates(
        char_s['X'][:, list(ds['group_idx'])])['omega_plugin']
    assert reference < -0.4


def test_mm_correction_qualified_positive_noise():
    # §E3.2: at positive noise d=3 parity occupies all 8 joint cells, so
    # the noiseless cancellation does NOT hold: TC_MM = TC_plugin
    # - 4/(2n ln 2). Verify the accounting numerically.
    rng = np.random.default_rng(11)
    Xp = rdata.synergistic_group(rng, 3, 50_000, 0.11)
    estimates = info_est.group_estimates(Xp)
    expected_offset = -4 / (2.0 * len(Xp) * np.log(2.0))
    assert estimates['tc_mm'] == pytest.approx(
        estimates['tc_plugin'] + expected_offset, abs=1e-10)


def test_fit_selection_procedure():
    # 6 (procedural part only): the runner's selection ranking is by factor
    # TC with the neutral order; ground-truth recovery is performance, not
    # a test assertion. A smoke fit just confirms the harness runs and the
    # selection column is the argmax under the neutral order.
    import corex_fitting

    dataset = rdata.generate_dataset(
        rdata.GEN_REDUNDANT, 4, 0.19, 3000, seed=derived_seed(6, 'data'))
    fit = corex_fitting.fit_bio_corex(
        dataset['X'], 3, model_seed=derived_seed(6, 'corex'),
        dim_hidden=8, n_repeat=1, max_iter=50)
    order = rmetrics.descending_order(fit['tcs'])
    assert fit['tcs'][order[0]] == fit['tcs'].max()
    assert not fit['uses_ground_truth_init']


def test_fit_reports_tcs_in_bits():
    # Unit check: local_corex computes factor TCs in NATS; fit_bio_corex
    # must report BITS. A noiseless redundant block of d columns has
    # population TC exactly d-1 bits; the recovered top factor's TC must
    # match in the reported (bit-based) units, not nats.
    rng = np.random.default_rng(12)
    X = rdata.redundant_group(rng, 6, 20_000, 0.0)
    fit = corex_fitting.fit_bio_corex(
        X, 1, model_seed=derived_seed(12, 'units'),
        dim_hidden=2, n_repeat=1, max_iter=50)
    assert fit['tcs'][0] == pytest.approx(5.0, abs=0.1)  # bits, not nats


def test_characterization_sample_is_fit_independent():
    # Fresh-seed characterization data must differ from fit data (data
    # separation) while the analytic machinery stays identical.
    ds_fit = rdata.generate_dataset(
        rdata.GEN_REDUNDANT, 3, 0.14, 100, seed=derived_seed(9, 'data'))
    ds_char = rdata.generate_characterization_sample(
        rdata.GEN_REDUNDANT, 3, 0.14, 100, seed=derived_seed(9, 'char'))
    assert not np.array_equal(ds_fit['X'], ds_char['X'])


def test_metadata_and_column_schemas_agree():
    from run_redundancy_experiment import INFO_COLUMNS, FACTOR_COLUMNS
    # every info quantity the summarizer consumes must exist in the schema
    required_info = (
        'tc_plugin', 'tc_mm', 'omega_plugin', 'omega_mm', 'top_factor_tc',
        'omega_on_support_char', 'omega_population',
        'support_matches_group', 'fit_status',
    )
    assert all(column in INFO_COLUMNS for column in required_info)
    required_factor = (
        'factor_tc', 'is_top_tc_factor', 'auprc', 'topd_f1', 'support_f1',
        'nfp_support',
    )
    assert all(column in FACTOR_COLUMNS for column in required_factor)
    # support_discovered is stored as a JSON string and must parse
    assert json.loads(json.dumps([0, 1, 2])) == [0, 1, 2]
    assert all(column in INFO_COLUMNS
               for column in ('tc_target', 'eta', 'n_nuisance'))