from __future__ import annotations

from typing import Dict

import numpy as np

from synthetic_common import rng_for


def parity_group(rng: np.random.Generator, arity: int, n: int, flip_prob: float = 0.05) -> np.ndarray:
    if arity < 3:
        raise ValueError("Use arity >= 3.")

    bits = rng.integers(0, 2, size=(n, arity - 1))
    parity = np.mod(bits.sum(axis=1), 2).astype(int)

    last = parity.copy()
    flips = rng.random(n) < flip_prob
    last[flips] = 1 - last[flips]

    bits = np.concatenate([bits, last[:, None]], axis=1)
    x = 2.0 * bits - 1.0
    return x + rng.normal(scale=0.05, size=x.shape)


def continuous_group(rng: np.random.Generator, arity: int, n: int, noise_sd: float = 0.12) -> np.ndarray:
    if arity < 3:
        raise ValueError("Use arity >= 3.")

    base = rng.normal(size=(n, arity - 1))
    if arity == 3:
        last = base[:, 0] * base[:, 1] + rng.normal(scale=noise_sd, size=n)
    elif arity == 4:
        last = np.sin(base[:, 0] + base[:, 1] + base[:, 2]) + rng.normal(scale=noise_sd, size=n)
    else:
        last = np.tanh(base.sum(axis=1)) + rng.normal(scale=noise_sd, size=n)

    return np.concatenate([base, last[:, None]], axis=1)


def generate_nonlinear_dataset(
    rep: int,
    cluster_size: int,
    n_noise: int = 0,
    n_groups: int = 3,
    arity_a: int = 4,
    arity_b: int = 4,
) -> Dict[str, object]:
    """
    Default output is 24 signal variables:
      - cluster A active in dimensions 0:12
      - cluster B active in dimensions 12:24
    """
    rng = rng_for(rep, "nonlinear", cluster_size, n_noise, n_groups, arity_a, arity_b)

    signal_width_a = n_groups * arity_a
    signal_width_b = n_groups * arity_b
    if signal_width_a != signal_width_b:
        raise ValueError(
            f"Cluster A and B signal widths must match. Got {signal_width_a} and {signal_width_b}."
        )

    block_width = signal_width_a

    a_parts = [parity_group(rng, arity_a, cluster_size) for _ in range(n_groups)]
    b_parts = [continuous_group(rng, arity_b, cluster_size) for _ in range(n_groups)]

    a_signal = np.concatenate(a_parts, axis=1)   # (cluster_size, block_width)
    b_signal = np.concatenate(b_parts, axis=1)   # (cluster_size, block_width)

    a_noise = rng.normal(size=(cluster_size, block_width))
    b_noise = rng.normal(size=(cluster_size, block_width))

    # 24 signal dimensions before any extra nuisance variables.
    A = np.concatenate([a_signal, a_noise], axis=1)
    B = np.concatenate([b_noise, b_signal], axis=1)

    X = np.concatenate([A, B], axis=0)
    y_true = np.concatenate(
        [
            np.repeat("clust_A", cluster_size),
            np.repeat("clust_B", cluster_size),
        ]
    )

    if n_noise > 0:
        X = np.concatenate([X, rng.normal(size=(X.shape[0], n_noise))], axis=1)

    n_total = X.shape[1]
    groups_A = []
    groups_B = []

    for g in range(n_groups):
        mask_a = np.zeros(n_total)
        lo_a = g * arity_a
        hi_a = lo_a + arity_a
        mask_a[lo_a:hi_a] = 1.0
        groups_A.append(mask_a)

        mask_b = np.zeros(n_total)
        lo_b = block_width + g * arity_b
        hi_b = lo_b + arity_b
        mask_b[lo_b:hi_b] = 1.0
        groups_B.append(mask_b)

    return {
        "X": X,
        "y_true": y_true,
        "true_groups": {
            "clust_A": groups_A,
            "clust_B": groups_B,
        },
    }