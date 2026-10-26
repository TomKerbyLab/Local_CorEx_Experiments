"""Data generators for the nonlinear higher-order recovery experiments.

Implements the region construction and generators of
agent_instructions/02_nonlinear_higher_order_recovery.md:

- 14 signal variables; two equally sized regions with different group
  structures (no group is identical between regions):
    region 0: {1,2,3}, {4,5,6}, {7,8,9,10}, {11,12,13,14}
    region 1: {1,4,7}, {2,8,11}, {3,5,9,12}, {6,10,13,14}
- Two routing variables R | C=c ~ N(nu_c, I_2) with
  nu_0 = (-2*alpha, 0) and nu_1 = (2*alpha, 0); the routing-mean distance is
  4*alpha. Routing variables are used for PHATE partitioning and excluded
  from ground-truth scoring.
- p_n nuisance variables: Bernoulli(1/2) for the discrete generator,
  standard normal for the continuous generator.

Part A (discrete parity): for a group of size d, B_1..B_{d-1} are iid
Bernoulli(1/2) and B_d = B_1 xor ... xor B_{d-1} xor E, E ~ Bernoulli(eta).
At eta = 0 every proper subset is mutually independent while the full group
has TC = 1 bit (for every d >= 2: the parity constraint leaves 2^(d-1)
equiprobable atoms).

Part B (continuous sine-sum): Theta_1..Theta_{d-1} are iid Uniform(-pi, pi)
and Y = sin(sum Theta_r) + eps, eps ~ N(0, sigma^2). The noiseless response
is marginally independent of each individual angle.

Part B supplementary (product): Y = prod(X_r) + eps, X_r iid N(0,1). This
group is nonlinear but NOT pairwise independent; results must never be
reported as pairwise-independent higher-order dependence without
qualification.

Column layout of every generated dataset:
    [0..13] signal, [14..15] routing, [16..] nuisance.
"""

import numpy as np


N_SIGNAL = 14
ROUTING_DIM = 2
N_GROUPS = 4

REGION0_GROUPS = {
    1: (1, 2, 3),
    2: (4, 5, 6),
    3: (7, 8, 9, 10),
    4: (11, 12, 13, 14),
}
REGION1_GROUPS = {
    1: (1, 4, 7),
    2: (2, 8, 11),
    3: (3, 5, 9, 12),
    4: (6, 10, 13, 14),
}
REGION_GROUPS = {0: REGION0_GROUPS, 1: REGION1_GROUPS}

GENERATOR_NOISE_NAME = {
    'parity': 'eta',
    'sine_sum': 'sigma',
    'product': 'sigma',
}


def routing_means(alpha):
    """Return (nu_0, nu_1); the routing-mean distance is 4*alpha."""
    return np.array([-2.0 * alpha, 0.0]), np.array([2.0 * alpha, 0.0])


def true_group_masks(n_nuisance):
    """Binary ground-truth masks over [signal | routing | nuisance] columns.

    One mask per (region, group). Routing and nuisance columns are always
    zero: they are excluded from ground-truth supports by construction.
    """
    n_variables = N_SIGNAL + ROUTING_DIM + n_nuisance
    masks = {}
    for region, groups in REGION_GROUPS.items():
        for group, members in groups.items():
            mask = np.zeros(n_variables)
            mask[[member - 1 for member in members]] = 1.0
            masks[(region, group)] = mask
    return masks


def parity_group(rng, group_size, n_observations, eta):
    """One parity group as an (n, d) integer block of 0/1 columns.

    B_d = xor(B_1..B_{d-1}) xor E with E ~ Bernoulli(eta). At eta = 0 every
    proper subset of columns is mutually independent.
    """
    bits = rng.integers(0, 2, size=(n_observations, group_size - 1))
    parity = bits.sum(axis=1) % 2
    flips = (rng.random(n_observations) < eta).astype(int)
    last_bit = (parity + flips) % 2
    return np.column_stack([bits, last_bit])


def sine_sum_group(rng, group_size, n_observations, sigma):
    """One continuous sine-sum group as an (n, d) float block.

    Columns are (Theta_1..Theta_{d-1}, Y) with
    Y = sin(sum Theta_r) + eps, eps ~ N(0, sigma^2).
    """
    theta = rng.uniform(-np.pi, np.pi, size=(n_observations, group_size - 1))
    noise = rng.normal(0.0, sigma, n_observations)
    response = np.sin(theta.sum(axis=1)) + noise
    return np.column_stack([theta, response])


def product_group(rng, group_size, n_observations, sigma):
    """Supplementary generator: columns (X_r, Y) with Y = prod(X_r) + eps.

    Nonlinear dependence, but pairwise dependencies exist: this generator is
    a comparison point, not pairwise-independent ground truth.
    """
    xs = rng.normal(0.0, 1.0, size=(n_observations, group_size - 1))
    noise = rng.normal(0.0, sigma, n_observations)
    response = xs.prod(axis=1) + noise
    return np.column_stack([xs, response])


def _group_block(generator, rng, group_size, n_observations, noise):
    if generator == 'parity':
        return parity_group(rng, group_size, n_observations, eta=noise)
    if generator == 'sine_sum':
        return sine_sum_group(rng, group_size, n_observations, sigma=noise)
    if generator == 'product':
        return product_group(rng, group_size, n_observations, sigma=noise)
    raise ValueError(
        f"Unknown generator '{generator}'. "
        "Choose 'parity', 'sine_sum', or 'product'."
    )


def generate_discrete_dataset(n_per_region, alpha, eta, n_nuisance, seed):
    """Discrete parity dataset for both regions plus routing and nuisance.

    Signal columns are 0/1 integers; routing columns are continuous normals
    (binned before discrete CorEx fitting by the runner); nuisance columns
    are Bernoulli(1/2).
    """
    rng = np.random.default_rng(seed)
    n_variables = N_SIGNAL + ROUTING_DIM + n_nuisance
    X = np.zeros((2 * n_per_region, n_variables), dtype=np.int64)
    region = np.concatenate([
        np.zeros(n_per_region, int),
        np.ones(n_per_region, int),
    ])

    for region_id in (0, 1):
        rows = slice(region_id * n_per_region, (region_id + 1) * n_per_region)
        for group, members in REGION_GROUPS[region_id].items():
            block = parity_group(rng, len(members), n_per_region, eta)
            for member_position, member in enumerate(members):
                X[rows, member - 1] = block[:, member_position]

    nu_0, nu_1 = routing_means(alpha)
    X[:n_per_region, N_SIGNAL:N_SIGNAL + ROUTING_DIM] = rng.normal(
        nu_0, 1.0, size=(n_per_region, ROUTING_DIM))
    X[n_per_region:, N_SIGNAL:N_SIGNAL + ROUTING_DIM] = rng.normal(
        nu_1, 1.0, size=(n_per_region, ROUTING_DIM))

    if n_nuisance > 0:
        X[:, N_SIGNAL + ROUTING_DIM:] = rng.integers(
            0, 2, size=(2 * n_per_region, n_nuisance))

    return {
        'X': X,
        'region': region,
        'true_groups': true_group_masks(n_nuisance),
        'signal_idx': np.arange(N_SIGNAL),
        'routing_idx': np.arange(N_SIGNAL, N_SIGNAL + ROUTING_DIM),
        'nuisance_idx': np.arange(N_SIGNAL + ROUTING_DIM, n_variables),
        'n_per_region': n_per_region,
        'alpha': alpha,
        'noise': eta,
        'n_nuisance': n_nuisance,
        'generator': 'parity',
        'seed': seed,
    }


def generate_continuous_dataset(n_per_region, alpha, sigma, n_nuisance, seed,
                                product_groups=False):
    """Continuous sine-sum (or product) dataset plus routing and nuisance.

    The runner standardizes the data before CorEx fitting; the routing
    columns are continuous normals and nuisance columns are standard
    normal.
    """
    rng = np.random.default_rng(seed)
    n_variables = N_SIGNAL + ROUTING_DIM + n_nuisance
    X = np.zeros((2 * n_per_region, n_variables))
    region = np.concatenate([
        np.zeros(n_per_region, int),
        np.ones(n_per_region, int),
    ])

    generator = 'product' if product_groups else 'sine_sum'
    for region_id in (0, 1):
        rows = slice(region_id * n_per_region, (region_id + 1) * n_per_region)
        for group, members in REGION_GROUPS[region_id].items():
            block = _group_block(
                'product' if product_groups else 'sine_sum',
                rng,
                len(members),
                n_per_region,
                sigma,
            )
            for member_position, member in enumerate(members):
                X[rows, member - 1] = block[:, member_position]

    nu_0, nu_1 = routing_means(alpha)
    X[:n_per_region, N_SIGNAL:N_SIGNAL + ROUTING_DIM] = rng.normal(
        nu_0, 1.0, size=(n_per_region, ROUTING_DIM))
    X[n_per_region:, N_SIGNAL:N_SIGNAL + ROUTING_DIM] = rng.normal(
        nu_1, 1.0, size=(n_per_region, ROUTING_DIM))

    if n_nuisance > 0:
        X[:, N_SIGNAL + ROUTING_DIM:] = rng.normal(
            0.0, 1.0, size=(2 * n_per_region, n_nuisance))

    return {
        'X': X,
        'region': region,
        'true_groups': true_group_masks(n_nuisance),
        'signal_idx': np.arange(N_SIGNAL),
        'routing_idx': np.arange(N_SIGNAL, N_SIGNAL + ROUTING_DIM),
        'nuisance_idx': np.arange(N_SIGNAL + ROUTING_DIM, n_variables),
        'n_per_region': n_per_region,
        'alpha': alpha,
        'noise': sigma,
        'n_nuisance': n_nuisance,
        'generator': generator,
        'seed': seed,
    }


def generate_dataset(generator, n_per_region, alpha, noise, n_nuisance, seed,
                     product_groups=False):
    """Dispatch to the requested generator with a uniform interface."""
    if generator == 'parity':
        return generate_discrete_dataset(
            n_per_region, alpha, eta=noise, n_nuisance=n_nuisance, seed=seed)
    if generator in ('sine_sum', 'product'):
        return generate_continuous_dataset(
            n_per_region, alpha, sigma=noise, n_nuisance=n_nuisance, seed=seed,
            product_groups=(generator == 'product'))
    raise ValueError(
        f"Unknown generator '{generator}'. "
        "Choose 'parity', 'sine_sum', or 'product'."
    )