"""Experiment 1 data generator: Gaussian grouped-covariance recovery.

Spec (01_gaussian_partition_recovery.md):
- p_s = 20 signal variables, m = 5 true groups of size 4.
- Two equally likely regions C in {0,1}; X_s | C=c ~ N(mu_c, Sigma_c).
- Within-group equicorrelation rho = 0.7, zero covariance elsewhere.
- Regimes: "mostly_shared" (3 of 5 groups shared) and "disjoint" (no shared group).
- Mean separation: mu_0 = -(alpha*Delta/2) v, mu_1 = +(alpha*Delta/2) v,
  v = (1,...,1)/sqrt(p_s), Delta = 4, so ||mu_1 - mu_0||_2 = 4*alpha.
- p_n independent N(0,1) nuisance variables appended.
- n denotes observations per region (total 2n), exactly n per region.
"""

from __future__ import annotations

import numpy as np

PS = 20           # signal variables
M_GROUPS = 5       # true groups
GROUP_SIZE = 4
RHO = 0.7
DELTA = 4.0

# Region-0 groups: contiguous blocks
REGION0_GROUPS: dict[int, tuple[int, ...]] = {
    1: (1, 2, 3, 4),
    2: (5, 6, 7, 8),
    3: (9, 10, 11, 12),
    4: (13, 14, 15, 16),
    5: (17, 18, 19, 20),
}

# Region-1 groups, mostly_shared regime: groups 1-3 unchanged,
# variables 13..20 repartitioned
REGION1_GROUPS_SHARED: dict[int, tuple[int, ...]] = {
    1: (1, 2, 3, 4),
    2: (5, 6, 7, 8),
    3: (9, 10, 11, 12),
    4: (13, 14, 17, 18),
    5: (15, 16, 19, 20),
}

# Region-1 groups, disjoint regime: no group identical to any region-0 group
REGION1_GROUPS_DISJOINT: dict[int, tuple[int, ...]] = {
    1: (1, 5, 9, 13),
    2: (2, 6, 10, 17),
    3: (3, 7, 14, 18),
    4: (4, 11, 15, 19),
    5: (8, 12, 16, 20),
}

REGIMES = ("mostly_shared", "disjoint")


def region1_groups(regime: str) -> dict[int, tuple[int, ...]]:
    if regime == "mostly_shared":
        return REGION1_GROUPS_SHARED
    if regime == "disjoint":
        return REGION1_GROUPS_DISJOINT
    raise ValueError(f"unknown regime {regime!r}")


def region_means(alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """mu_0, mu_1 with ||mu_1 - mu_0||_2 == 4*alpha."""
    v = np.ones(PS) / np.sqrt(PS)
    return -(alpha * DELTA / 2) * v, (alpha * DELTA / 2) * v


def covariance_matrix(groups: dict[int, tuple[int, ...]], rho: float = RHO) -> np.ndarray:
    """p_s x p_s covariance with equicorrelated blocks; assert positive definite."""
    S = np.eye(PS)
    for members in groups.values():
        for i in members:
            for j in members:
                if i != j:
                    S[i - 1, j - 1] = rho
    # assert symmetry and positive definiteness
    assert np.allclose(S, S.T)
    eig = np.linalg.eigvalsh(S)
    assert eig.min() > 0, f"covariance not PD: min eig {eig.min()}"
    return S


def true_group_masks(
    regime: str, n_nuisance: int
) -> dict[int, np.ndarray]:
    """Binary (p_s + p_n) masks per (region, group): 1 for in-group signal vars."""
    masks = {}
    g0 = REGION0_GROUPS
    g1 = region1_groups(regime)
    for c, groups in ((0, g0), (1, g1)):
        for g, members in groups.items():
            m = np.zeros(PS + n_nuisance)
            m[[i - 1 for i in members]] = 1.0
            masks[(c, g)] = m
    return masks


def count_shared_groups(regime: str) -> int:
    """Number of groups identical between regions (spec validation test 3)."""
    s0 = set(map(frozenset, REGION0_GROUPS.values()))
    s1 = set(map(frozenset, region1_groups(regime).values()))
    return len(s0 & s1)


def generate_dataset(
    n_per_region: int,
    alpha: float,
    regime: str,
    n_nuisance: int,
    seed: int,
) -> dict:
    """Generate one paired dataset replicate.

    Returns dict with keys:
      X: (2n, PS + n_nuisance) unstandardized data
      region: (2n,) true region labels
      true_groups: {(region, group): binary mask}
      nuisance_idx: indices of nuisance columns
    """
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    rng = np.random.default_rng(seed)
    mu0, mu1 = region_means(alpha)
    S0 = covariance_matrix(REGION0_GROUPS)
    S1 = covariance_matrix(region1_groups(regime))

    X0 = rng.multivariate_normal(mu0, S0, size=n_per_region)
    X1 = rng.multivariate_normal(mu1, S1, size=n_per_region)
    Xs = np.vstack([X0, X1])
    region = np.concatenate([np.zeros(n_per_region, dtype=int), np.ones(n_per_region, dtype=int)])

    if n_nuisance > 0:
        Nn = rng.standard_normal((2 * n_per_region, n_nuisance))
        X = np.hstack([Xs, Nn])
    else:
        X = Xs
    nuisance_idx = np.arange(PS, PS + n_nuisance)

    return {
        "X": X,
        "region": region,
        "true_groups": true_group_masks(regime, n_nuisance),
        "nuisance_idx": nuisance_idx,
        "n_per_region": n_per_region,
        "alpha": alpha,
        "regime": regime,
        "n_nuisance": n_nuisance,
        "seed": seed,
    }