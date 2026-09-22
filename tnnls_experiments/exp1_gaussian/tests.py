"""Validation tests for Experiment 1 (spec section 'Validation tests')."""

from __future__ import annotations

import numpy as np
import pytest

from tnnls_experiments.common import metrics, partition, seeds
from tnnls_experiments.exp1_gaussian import data, runner


def test_covariances_symmetric_pd():
    # 1. Every covariance matrix is symmetric and positive definite.
    for groups in (data.REGION0_GROUPS,
                   data.REGION1_GROUPS_SHARED,
                   data.REGION1_GROUPS_DISJOINT):
        S = data.covariance_matrix(groups)
        assert np.allclose(S, S.T)
        assert np.linalg.eigvalsh(S).min() > 0


def test_signal_variable_single_group_membership():
    # 2. Each signal variable belongs to exactly one true group in each region.
    for regime in data.REGIMES:
        for groups in (data.REGION0_GROUPS, data.region1_groups(regime)):
            all_members = [m for g in groups.values() for m in g]
            assert sorted(all_members) == list(range(1, 21))


def test_regime_shared_group_counts():
    # 3. Shared regime has 3 identical groups; disjoint has 0.
    assert data.count_shared_groups("mostly_shared") == 3
    assert data.count_shared_groups("disjoint") == 0


def test_empirical_within_group_correlation_converges():
    # 4. Empirical within-group correlations converge to rho.
    rng = np.random.default_rng(7)
    S = data.covariance_matrix(data.REGION0_GROUPS)
    X = rng.multivariate_normal(np.zeros(20), S, size=200_000)
    C = np.corrcoef(X.T)
    for members in data.REGION0_GROUPS.values():
        idx = [m - 1 for m in members]
        sub = C[np.ix_(idx, idx)]
        offdiag = sub[np.triu_indices(4, k=1)]
        assert np.allclose(offdiag, data.RHO, atol=0.02)
    # cross-group correlations near zero
    g1 = [m - 1 for m in data.REGION0_GROUPS[1]]
    g4 = [m - 1 for m in data.REGION0_GROUPS[4]]
    cross = C[np.ix_(g1, g4)]
    assert np.allclose(cross, 0.0, atol=0.02)


def test_nuisance_zero_population_covariance():
    # 5. Nuisance variables have zero population covariance with signal vars.
    ds = data.generate_dataset(2_000, 0.5, "mostly_shared", n_nuisance=5, seed=11)
    C = np.corrcoef(ds["X"].T)
    cross = C[:20, 20:]
    assert np.abs(cross).max() < 0.06


def test_mean_separation_norm():
    # 6. ||mu_1 - mu_0||_2 == 4*alpha numerically.
    for alpha in (0.0, 0.25, 0.5, 1.0):
        mu0, mu1 = data.region_means(alpha)
        assert np.isclose(np.linalg.norm(mu1 - mu0), 4 * alpha)


def test_partition_alignment_label_swap_invariance():
    # 7. Hungarian alignment is invariant to swapping raw k-means labels.
    rng = np.random.default_rng(3)
    true = (rng.random(500) < 0.5).astype(int)
    est = true.copy()
    est[:50] = 1 - est[:50]  # some noise
    a1, acc1 = metrics.align_partition(est, true)
    a2, acc2 = metrics.align_partition(1 - est, true)  # swapped labels
    assert np.array_equal(a1, a2)
    assert acc1 == acc2


def test_hungarian_no_factor_reuse():
    # 8. Hungarian factor scoring does not reuse a learned factor.
    assoc = np.abs(np.random.default_rng(5).normal(size=(6, 20)))
    truth = np.zeros((5, 20))
    for g, members in data.REGION0_GROUPS.items():
        truth[g - 1, [m - 1 for m in members]] = 1
    records = metrics.match_factors_to_groups(assoc, truth)
    learned = [r["learned_factor"] for r in records]
    assert len(set(learned)) == len(learned)
    groups_matched = [r["true_group"] for r in records]
    assert len(set(groups_matched)) == len(groups_matched)


def test_perfect_association_recovers_all_ones():
    # 10. A deliberately perfect association matrix produces recovery scores of one.
    truth = np.zeros((5, 25))
    for g, members in data.REGION0_GROUPS.items():
        truth[g - 1, [m - 1 for m in members]] = 1
    # perfect association: exact group patterns, zeros elsewhere
    assoc = truth.copy()
    records = metrics.match_factors_to_groups(assoc, truth)
    for r in records:
        assert r["cosine"] == pytest.approx(1.0)
        assert r["auprc"] == pytest.approx(1.0)
        assert r["topk_precision"] == pytest.approx(1.0)
        assert r["topk_f1"] == pytest.approx(1.0)


def test_cumulative_support_definition():
    scores = np.array([0.4, 0.3, 0.2, 0.1])
    sup = metrics.cumulative_mi_support(scores, 0.8)
    # cumsum fractions: 0.4, 0.7, 0.9 -> first >= 0.8 is at k=3
    assert sup.size == 3
    assert sup.achieved_fraction == pytest.approx(0.9)
    assert sup.indices == (0, 1, 2)
    assert not sup.degenerate
    # degenerate
    sup = metrics.cumulative_mi_support(np.zeros(4), 0.8)
    assert sup.degenerate and sup.size == 0


def test_balanced_random_partition_sizes():
    true = np.array([0] * 30 + [1] * 30)
    labels, _ = partition.balanced_random_partition(true, seed=1)
    vals, counts = np.unique(labels, return_counts=True)
    assert sorted(counts) == [30, 30]
    # different seeds give different memberships
    l2, _ = partition.balanced_random_partition(true, seed=2)
    assert not np.array_equal(labels, l2)


def test_seeds_deterministic_and_distinct():
    assert seeds.derive_seed(1000, "data") == seeds.derive_seed(1000, "data")
    assert seeds.derive_seed(1000, "data") != seeds.derive_seed(1000, "corex")
    assert 0 < seeds.derive_seed(1000, "data") < 2**32 - 1