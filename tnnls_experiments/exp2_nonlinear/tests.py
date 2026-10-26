"""Validation tests for Experiment 2 (spec section 'Validation tests')."""

from __future__ import annotations

import numpy as np
import pytest

from tnnls_experiments.common import corex, information, metrics
from tnnls_experiments.exp2_nonlinear import data, runner


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


def test_cluster_selection_uses_raw_labels():
    # Regression (raw-vs-aligned): when the estimated k-means labels are
    # swapped relative to the true regions, the cluster selected for a region
    # must be the raw cluster that *overlaps* the region, so that slicing raw
    # labels selects the region's own observations.
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])   # swapped: raw 1 = region 0
    region = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    sel_by_region = runner._select_clusters_by_region(labels, region, k=2)
    assert sel_by_region == {0: 1, 1: 0}
    for c in (0, 1):
        sel = sel_by_region[c]
        assert (region[labels == sel] == c).all()
    # collapsed partition: both regions select the same raw cluster
    collapsed = np.zeros(8, dtype=int)
    sel_by_region = runner._select_clusters_by_region(collapsed, region, k=2)
    assert sel_by_region[0] == sel_by_region[1]
    # regression (regions_served): when BOTH regions select one cluster, the
    # single cell serving them must record regions {0, 1}, not just the loop
    # variable of the first region (a {c for c2 ...} comprehension yields
    # only c, silently dropping region 1's matches)
    assert runner.served_regions({0: 1, 1: 1}, 1) == {0, 1}
    assert runner.served_regions({0: 1, 1: 0}, 1) == {0}
    assert runner.served_regions({0: 1, 1: 0}, 0) == {1}
    assert runner.served_regions({0: None, 1: None}, None) == {0, 1}


def test_no_routing_scores_align_to_full_width():
    # Regression (no_routing mismatch): association scores from a fit with
    # routing columns dropped have fewer columns than the full-width truth
    # masks; _score_fit must expand them back so matching works and FP
    # indices stay in the original column space, with routing FPs reported
    # as None.
    p_total = 14 + 2 + 4
    assoc = np.zeros((4, p_total - 2))
    assoc[0, [0, 1, 2]] = 1.0      # region 0, group 1 = {1,2,3}
    assoc[1, [3, 4, 5]] = 1.0      # region 0, group 2 = {4,5,6}
    assoc[2, [6, 7, 8]] = 1.0      # region 0, group 3 = {7,8,9,10}
    assoc[3, [16 - 2, 17 - 2]] = 5.0  # nuisance columns (strong leakage)
    fit = corex.CorexFitRecord(
        estimator="bio", n_factors_requested=4, n_factors_valid=4,
        tcs=np.ones(4), mis=assoc, converged=True, runtime_s=0.0,
        seed=1, n_samples=100, n_visible=p_total - 2)
    ds = data.generate_discrete_dataset(50, 0.5, 0.05, 4, seed=9)
    rows = runner._score_fit(
        {"root_seed": 9}, "phate", 0, 0, "bio", "no_routing", fit,
        ds["true_groups"], ds["routing_idx"], ds["nuisance_idx"],
        p_total, 0.80, dropped=np.asarray(ds["routing_idx"]))
    assert len(rows) == 4
    matched_groups = {r["true_group"] for r in rows}
    assert matched_groups == {1, 2, 3, 4}
    for row in rows:
        assert row["routing_fp"] is None
        assert row["routing_fp_support"] is None
    leakage_rows = [r for r in rows if r["learned_factor"] == 3]
    assert leakage_rows and leakage_rows[0]["nuisance_fp"] > 0.0
    assert leakage_rows[0]["true_group"] in (1, 2, 3, 4)


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