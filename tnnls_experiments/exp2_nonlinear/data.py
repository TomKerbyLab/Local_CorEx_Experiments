"""Experiment 2 data generators: discrete parity and continuous sine-sum.

Shared region construction (02_nonlinear_higher_order_recovery.md):
- 14 signal variables; two equally sized regions with different group
  structures: region 0 groups {1,2,3},{4,5,6},{7..10},{11..14};
  region 1 groups {1,4,7},{2,8,11},{3,5,9,12},{6,10,13,14}.
- Two routing variables R | C=c ~ N(nu_c, I_2), nu_0 = (-2a,0), nu_1 = (2a,0).
- p_n nuisance variables (Bernoulli(1/2) in the discrete part, N(0,1) in the
  continuous part), mutually independent.
- Routing variables are included in PHATE partitioning, excluded from
  factor-recovery scoring (a secondary condition also excludes them from
  CorEx fitting).

Part A (discrete parity): group of size d, B_1..B_{d-1} ~ Bern(1/2),
B_d = xor of the first d-1 bits and flip-noise bit E ~ Bern(eta).
At eta=0 every proper subset is mutually independent but the full group has
TC = (d-1) bits... (for d=3, TC = 1 bit).

Part B (continuous sine-sum): Theta_1..Theta_{d-1} ~ Uniform(-pi,pi),
Y = sin(sum Theta_r) + eps, eps ~ N(0, sigma^2). The noiseless response is
marginally independent of each angle but the full group is dependent.
"""

from __future__ import annotations

import numpy as np

N_SIGNAL = 14
N_GROUPS = 4

REGION0_GROUPS: dict[int, tuple[int, ...]] = {
    1: (1, 2, 3),
    2: (4, 5, 6),
    3: (7, 8, 9, 10),
    4: (11, 12, 13, 14),
}
REGION1_GROUPS: dict[int, tuple[int, ...]] = {
    1: (1, 4, 7),
    2: (2, 8, 11),
    3: (3, 5, 9, 12),
    4: (6, 10, 13, 14),
}


def routing_means(alpha: float) -> tuple[np.ndarray, np.ndarray]:
    return np.array([-2 * alpha, 0.0]), np.array([2 * alpha, 0.0])


def true_group_masks(n_nuisance: int) -> dict[tuple[int, int], np.ndarray]:
    """Binary masks over [signal(14) | routing(2) | nuisance(p_n)] per
    (region, group). Routing and nuisance are always zero in truth vectors."""
    p = N_SIGNAL + 2 + n_nuisance
    masks = {}
    for c, groups in ((0, REGION0_GROUPS), (1, REGION1_GROUPS)):
        for g, members in groups.items():
            m = np.zeros(p)
            m[[i - 1 for i in members]] = 1.0
            masks[(c, g)] = m
    return masks


def group_members(regime_unused=None) -> dict[int, tuple[int, ...]]:
    return REGION0_GROUPS


# ---------------------------------------------------------------------------
# Part A: discrete parity
# ---------------------------------------------------------------------------

def parity_group(rng: np.random.Generator, d: int, n: int, eta: float) -> np.ndarray:
    """One parity group: (n, d) 0/1 ints. B_d = xor(B_1..B_{d-1}) xor E."""
    bits = rng.integers(0, 2, size=(n, d - 1))
    parity = bits.sum(axis=1) % 2
    flips = (rng.random(n) < eta).astype(int)
    last = (parity + flips) % 2
    return np.column_stack([bits, last])


def generate_discrete_dataset(
    n_per_region: int,
    alpha: float,
    eta: float,
    n_nuisance: int,
    seed: int,
) -> dict:
    """Full discrete dataset: 4 parity groups per region + routing + nuisance.

    Layout: columns [0..13] signal, [14..15] routing, [16..] nuisance.
    """
    rng = np.random.default_rng(seed)
    p = N_SIGNAL + 2 + n_nuisance
    X = np.zeros((2 * n_per_region, p), dtype=np.int64)
    region = np.concatenate([np.zeros(n_per_region, int), np.ones(n_per_region, int)])

    for c in (0, 1):
        rows = slice(c * n_per_region, (c + 1) * n_per_region)
        groups = REGION0_GROUPS if c == 0 else REGION1_GROUPS
        for g, members in groups.items():
            d = len(members)
            block = parity_group(rng, d, n_per_region, eta)
            for j, m in enumerate(members):
                X[rows, m - 1] = block[:, j]

    # routing variables
    nu0, nu1 = routing_means(alpha)
    X[:n_per_region, N_SIGNAL:N_SIGNAL + 2] = rng.normal(nu0, 1.0, size=(n_per_region, 2))
    X[n_per_region:, N_SIGNAL:N_SIGNAL + 2] = rng.normal(nu1, 1.0, size=(n_per_region, 2))

    # nuisance: Bernoulli(1/2)
    if n_nuisance > 0:
        X[:, N_SIGNAL + 2:] = rng.integers(0, 2, size=(2 * n_per_region, n_nuisance))

    return {
        "X": X,
        "region": region,
        "true_groups": true_group_masks(n_nuisance),
        "signal_idx": np.arange(N_SIGNAL),
        "routing_idx": np.arange(N_SIGNAL, N_SIGNAL + 2),
        "nuisance_idx": np.arange(N_SIGNAL + 2, p),
        "n_per_region": n_per_region,
        "alpha": alpha,
        "eta": eta,
        "n_nuisance": n_nuisance,
        "generator": "parity",
        "seed": seed,
    }


# ---------------------------------------------------------------------------
# Part B: continuous sine-sum
# ---------------------------------------------------------------------------

def sine_sum_group(rng: np.random.Generator, d: int, n: int, sigma: float) -> np.ndarray:
    """One continuous group: (n, d) floats [Theta_1..Theta_{d-1}, Y]."""
    theta = rng.uniform(-np.pi, np.pi, size=(n, d - 1))
    eps = rng.normal(0, sigma, n)
    y = np.sin(theta.sum(axis=1)) + eps
    return np.column_stack([theta, y])


def generate_continuous_dataset(
    n_per_region: int,
    alpha: float,
    sigma: float,
    n_nuisance: int,
    seed: int,
    product_groups: bool = False,
) -> dict:
    """Full continuous dataset: 4 sine-sum groups per region + routing +
    Gaussian nuisance. Standardized before CorEx by the runner."""
    rng = np.random.default_rng(seed)
    p = N_SIGNAL + 2 + n_nuisance
    X = np.zeros((2 * n_per_region, p))
    region = np.concatenate([np.zeros(n_per_region, int), np.ones(n_per_region, int)])

    for c in (0, 1):
        rows = slice(c * n_per_region, (c + 1) * n_per_region)
        groups = REGION0_GROUPS if c == 0 else REGION1_GROUPS
        for g, members in groups.items():
            d = len(members)
            if product_groups:
                block = _product_group(rng, d, n_per_region, sigma)
            else:
                block = sine_sum_group(rng, d, n_per_region, sigma)
            for j, m in enumerate(members):
                X[rows, m - 1] = block[:, j]

    nu0, nu1 = routing_means(alpha)
    X[:n_per_region, N_SIGNAL:N_SIGNAL + 2] = rng.normal(nu0, 1.0, size=(n_per_region, 2))
    X[n_per_region:, N_SIGNAL:N_SIGNAL + 2] = rng.normal(nu1, 1.0, size=(n_per_region, 2))

    if n_nuisance > 0:
        X[:, N_SIGNAL + 2:] = rng.normal(0, 1, size=(2 * n_per_region, n_nuisance))

    return {
        "X": X,
        "region": region,
        "true_groups": true_group_masks(n_nuisance),
        "signal_idx": np.arange(N_SIGNAL),
        "routing_idx": np.arange(N_SIGNAL, N_SIGNAL + 2),
        "nuisance_idx": np.arange(N_SIGNAL + 2, p),
        "n_per_region": n_per_region,
        "alpha": alpha,
        "sigma": sigma,
        "n_nuisance": n_nuisance,
        "generator": "product" if product_groups else "sine_sum",
        "seed": seed,
    }


def _product_group(rng: np.random.Generator, d: int, n: int, sigma: float) -> np.ndarray:
    """Supplementary generator: Y = prod(X_r) + eps (nonlinear but NOT
    pairwise independent; must not be described as such without qualification)."""
    xs = rng.normal(0, 1, size=(n, d - 1))
    eps = rng.normal(0, sigma, n)
    y = xs.prod(axis=1) + eps
    return np.column_stack([xs, y])


# ---------------------------------------------------------------------------
# Structure diagnostics (independent large samples)
# ---------------------------------------------------------------------------

def structure_diagnostics(
    generator: str,
    d: int,
    noise: float,
    n: int,
    seed: int,
) -> dict:
    """Diagnostic table quantities (spec: at least 100k observations):
    max |pearson corr| within group, max pairwise MI, group TC, permuted TC.

    Pairwise MI for continuous variables is estimated after 10-bin
    equal-frequency discretization on the diagnostic sample itself (fixed
    rule, documented). For parity, variables are already binary and the raw
    bits are used.
    """
    from ..common import corex, information
    from sklearn.metrics import mutual_info_score

    rng = np.random.default_rng(seed)
    if generator == "parity":
        block = parity_group(rng, d, n, noise)
        disc = block.astype(np.int64)
    elif generator == "sine_sum":
        block = sine_sum_group(rng, d, n, noise)
        disc, _ = corex.discretize_equal_frequency(block, 10)
    elif generator == "product":
        block = _product_group(rng, d, n, noise)
        disc, _ = corex.discretize_equal_frequency(block, 10)
    else:
        raise ValueError(generator)

    # max |pearson| within group
    if generator == "parity":
        max_abs_corr = 0.0  # binary pairwise correlations estimated via MI below
        max_corr = float(np.abs(np.corrcoef(block.T)).max() if d > 1 else 0.0)
        offdiag = np.abs(np.corrcoef(block.T) - np.eye(d))
        max_corr = float(offdiag.max())
    else:
        offdiag = np.abs(np.corrcoef(block.T) - np.eye(d))
        max_corr = float(offdiag.max())

    # max pairwise MI (on the discretization used for CorEx)
    max_mi = 0.0
    for i in range(d):
        for j in range(i + 1, d):
            mi = mutual_info_score(disc[:, i], disc[:, j])
            max_mi = max(max_mi, float(mi))

    # group TC and permuted TC (permute one member)
    tc = information.total_correlation_plugin(disc)
    rng_perm = np.random.default_rng(seed + 1)
    perm = disc.copy()
    perm[:, 0] = rng_perm.permutation(perm[:, 0])
    tc_perm = information.total_correlation_plugin(perm)

    return {
        "generator": generator,
        "group_size": d,
        "noise": noise,
        "n": n,
        "max_abs_pearson": max_corr,
        "max_pairwise_mi": max_mi,
        "group_tc": tc,
        "permuted_group_tc": tc_perm,
    }