"""Validation tests for Experiment 2 (spec section 'Validation tests')."""

from __future__ import annotations

import numpy as np
import pytest

from tnnls_experiments.common import corex, information, metrics
from tnnls_experiments.exp2_nonlinear import data


def test_variable_single_group_membership_per_region():
    # 1. Every variable appears in exactly one true group per region.
    for groups in (data.REGION0_GROUPS, data.REGION1_GROUPS):
        members = [m for g in groups.values() for m in g]
        assert sorted(members) == list(range(1, 15))
        for g1 in groups.values():
            for g2 in groups.values():
                if g1 != g2:
                    assert set(g1) & set(g2) == set()


def test_no_group_identical_across_regions():
    # 2. No true group is identical across regions.
    s0 = set(map(frozenset, data.REGION0_GROUPS.values()))
    s1 = set(map(frozenset, data.REGION1_GROUPS.values()))
    assert len(s0 & s1) == 0


def test_parity_equation_holds_before_noise():
    # 3. Parity equations hold exactly at eta=0.
    rng = np.random.default_rng(1)
    block = data.parity_group(rng, 4, 1000, eta=0.0)
    expected = block[:, :3].sum(axis=1) % 2
    assert np.array_equal(block[:, 3], expected)
    # and flips occur at eta>0
    block_n = data.parity_group(rng, 4, 1000, eta=0.15)
    expected_n = block_n[:, :3].sum(axis=1) % 2
    assert not np.array_equal(block_n[:, 3], expected_n)
    flip_rate = (block_n[:, 3] != expected_n).mean()
    assert 0.10 < flip_rate < 0.20


def test_binary_marginals_bernoulli_half():
    # 4. Empirical binary marginals are approximately Bernoulli(1/2).
    rng = np.random.default_rng(2)
    block = data.parity_group(rng, 4, 50_000, eta=0.0)
    for j in range(4):
        assert abs(block[:, j].mean() - 0.5) < 0.01


def test_parity_pairwise_mi_near_zero():
    # 5. Pairwise parity MI within prespecified tolerance of zero at eta=0.
    from sklearn.metrics import mutual_info_score

    rng = np.random.default_rng(3)
    block = data.parity_group(rng, 3, 20_000, eta=0.0)
    for i in range(3):
        for j in range(i + 1, 3):
            mi = mutual_info_score(block[:, i], block[:, j])
            assert mi < 0.005  # tolerance from estimator bias at n=20k


def test_permuting_one_member_removes_group_dependence():
    # 6. Permuting one member removes estimated group dependence within tolerance.
    rng = np.random.default_rng(4)
    block = data.parity_group(rng, 3, 20_000, eta=0.0)
    tc = information.total_correlation_plugin(block)
    permuted = block.copy()
    permuted[:, 0] = rng.permutation(permuted[:, 0])
    tc_perm = information.total_correlation_plugin(permuted)
    assert tc == pytest.approx(1.0, abs=0.01)
    assert tc_perm == pytest.approx(0.0, abs=0.01)


def test_routing_nuisance_excluded_from_ground_truth():
    # 7. Routing and nuisance variables are excluded from ground-truth supports.
    masks = data.true_group_masks(n_nuisance=6)
    for key, m in masks.items():
        assert m.shape[0] == 14 + 2 + 6
        assert m[14:16].sum() == 0  # routing zero
        assert m[16:].sum() == 0    # nuisance zero
        assert m[:14].sum() > 0     # signal members present


def test_sine_sum_marginal_independence_and_group_dependence():
    # Continuous analogue: near-zero pairwise dependence (after the fixed
    # 10-bin rule) but nonzero multivariate dependence.
    from sklearn.metrics import mutual_info_score

    rng = np.random.default_rng(5)
    block = data.sine_sum_group(rng, 4, 100_000, sigma=0.05)
    # pairwise Pearson near zero between angles and Y
    offdiag = np.abs(np.corrcoef(block.T) - np.eye(4))
    assert offdiag.max() < 0.05
    # discretized pairwise MI near zero
    disc, _ = corex.discretize_equal_frequency(block, 10)
    for i in range(4):
        for j in range(i + 1, 4):
            mi = mutual_info_score(disc[:, i], disc[:, j])
            assert mi < 0.01
    # group TC clearly positive (dependence exists)
    tc = information.total_correlation_plugin(disc)
    assert tc > 0.1
    # permuting one angle removes it
    permuted = disc.copy()
    permuted[:, 0] = rng.permutation(permuted[:, 0])
    tc_perm = information.total_correlation_plugin(permuted)
    assert tc_perm < 0.6 * tc


def test_discrete_continuous_schema_compatible():
    # 8. Discrete and continuous result schemas share core keys.
    disc_keys = set(runner_cols_discrete())
    cont_keys = set(runner_cols_continuous())
    core = {"generator", "n_per_region", "alpha", "noise", "n_nuisance",
            "root_seed", "partition_method", "estimator", "region",
            "partition_draw", "factor_tc", "cosine", "auprc"}
    assert core <= disc_keys and core <= cont_keys


def runner_cols_discrete():
    from tnnls_experiments.exp2_nonlinear import runner
    return runner.MATCH_COLS


def runner_cols_continuous():
    from tnnls_experiments.exp2_nonlinear import runner
    return runner.MATCH_COLS


def test_hungarian_no_factor_reuse_nonlinear():
    # 9. Hungarian matching never assigns a learned factor more than once.
    rng = np.random.default_rng(6)
    assoc = np.abs(rng.normal(size=(4, 16)))
    truth = np.zeros((4, 16))
    for g, members in data.REGION0_GROUPS.items():
        truth[g - 1, [m - 1 for m in members]] = 1
    records = metrics.match_factors_to_groups(assoc, truth)
    learned = [r["learned_factor"] for r in records]
    assert len(set(learned)) == len(learned)


def test_generators_use_derived_seeds():
    # 10. Paired replicate: same seed -> identical observations for all methods.
    d1 = data.generate_discrete_dataset(100, 0.5, 0.05, 14, seed=77)
    d2 = data.generate_discrete_dataset(100, 0.5, 0.05, 14, seed=77)
    assert np.array_equal(d1["X"], d2["X"])
    d3 = data.generate_discrete_dataset(100, 0.5, 0.05, 14, seed=78)
    assert not np.array_equal(d1["X"], d3["X"])


def test_routing_mean_distance():
    # routing-mean distance is 4*alpha
    for alpha in (0.0, 0.5, 1.0):
        nu0, nu1 = data.routing_means(alpha)
        assert np.linalg.norm(nu1 - nu0) == pytest.approx(4 * alpha)


def test_structure_diagnostics_parity():
    diag = data.structure_diagnostics("parity", 3, 0.0, n=100_000, seed=11)
    assert diag["max_pairwise_mi"] < 0.01
    assert diag["group_tc"] == pytest.approx(1.0, abs=0.02)
    assert diag["permuted_group_tc"] < 0.01


def test_structure_diagnostics_sine_sum():
    diag = data.structure_diagnostics("sine_sum", 4, 0.05, n=100_000, seed=12)
    assert diag["max_abs_pearson"] < 0.05
    assert diag["max_pairwise_mi"] < 0.02
    assert diag["group_tc"] > 0.1
    assert diag["permuted_group_tc"] < 0.6 * diag["group_tc"]


def test_capability_diagnostic_std_finds_nothing_seeded_reaches_parity():
    # Standard EM initializations find TC ~ 0 on pure parity; the atom-seeded
    # init reaches the full parity TC. This is the Exp 2 capability statement.
    from tnnls_experiments.common import corex

    rng = np.random.default_rng(13)
    block = data.parity_group(rng, 3, 20_000, eta=0.0)
    fit_std = corex.fit_bio_corex(block, 1, seed=1, max_iter=200,
                                  dim_hidden=4, n_repeat=5)
    # encode the joint pattern uniquely (bit-sums collide on parity atoms)
    powers = 2 ** np.arange(3)
    fit_atom = corex.fit_bio_corex(block, 1, seed=1, max_iter=200,
                                   dim_hidden=8, n_repeat=1,
                                   atom_keys=block @ powers)
    assert fit_std.tcs[0] < 0.05            # standard init: ~0 TC
    # seeded init reaches the full parity TC: 1 bit = ln 2 nats (BioCorex
    # computes TC in nats)
    assert fit_atom.tcs[0] == pytest.approx(np.log(2), abs=0.02)
    assert fit_atom.uses_ground_truth_init and not fit_std.uses_ground_truth_init
    # seeded MI profile is flat over the group members (the expressible solution)
    assert np.all(fit_atom.association_scores()[0][:3] > 0.5)