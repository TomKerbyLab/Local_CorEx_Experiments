"""Empirical information-theoretic estimators (Experiment 3 and diagnostics).

All logarithms are base 2 (quantities in bits).

Plug-in entropy estimation for discrete data: H(X) = -sum p log p over the
empirical distribution. The plug-in entropy is biased DOWNWARD; the
Miller-Madow correction adds (K - 1) / (2 n ln 2) for K occupied cells:

    H_MM = H_plugin + (K - 1) / (2 n ln 2)

For total correlation,
    TC = sum_i H(X_i) - H(X),
the correction applies to each marginal entropy and to the joint entropy
separately, each with its own occupied-cell count, so

    TC_MM = TC_plugin + [ sum_i (K_i - 1) - (K_J - 1) ] / (2 n ln 2)

(note the joint-entropy correction enters with a minus sign because the
joint entropy is subtracted in TC). Dual total correlation uses the identity

    DTC = H(X) - sum_i H(X_i | X_{-i})
        = sum_i H(X_{-i}) - (d - 1) H(X).
"""

from __future__ import annotations

import numpy as np

_LN2 = np.log(2.0)


def _row_keys(X: np.ndarray) -> np.ndarray:
    """Map each row of an integer matrix to a unique integer key."""
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[1] == 0:
        return np.zeros(X.shape[0], dtype=np.int64)
    max_vals = X.max(axis=0).astype(np.int64) + 1
    mult = np.ones(X.shape[1], dtype=np.int64)
    for i in range(X.shape[1] - 2, -1, -1):
        mult[i] = mult[i + 1] * max_vals[i + 1]
    return (X.astype(np.int64) * mult).sum(axis=1)


def entropy_from_counts(counts: np.ndarray) -> float:
    """Entropy (bits) from a count vector."""
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def entropy_plugin(x: np.ndarray) -> float:
    """Plug-in entropy (bits) of a discrete vector."""
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    _, counts = np.unique(x, return_counts=True)
    return entropy_from_counts(counts)


def entropy_plugin_mm(x: np.ndarray) -> tuple[float, float]:
    """(H_MM, H_plugin) for a single discrete variable.

    H_MM = H_plugin + (K - 1) / (2 n ln 2): upward correction for the
    downward-biased plug-in entropy.
    """
    x = np.asarray(x)
    n = x.size
    if n == 0:
        return 0.0, 0.0
    _, counts = np.unique(x, return_counts=True)
    h = entropy_from_counts(counts)
    k = int((counts > 0).sum())
    return float(h + (k - 1) / (2.0 * n * _LN2)), float(h)


def joint_entropy_plugin(X: np.ndarray) -> tuple[float, int]:
    """(H_plugin(bits), occupied_cells) of the joint distribution of columns."""
    X = np.asarray(X, dtype=np.int64)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    if n == 0:
        return 0.0, 0
    _, counts = np.unique(_row_keys(X), return_counts=True)
    return entropy_from_counts(counts), int(len(counts))


def joint_entropy_mm(X: np.ndarray) -> tuple[float, float]:
    """(H_MM, H_plugin) for the joint entropy over columns."""
    h, k = joint_entropy_plugin(X)
    n = np.asarray(X).shape[0]
    if n == 0:
        return 0.0, 0.0
    return float(h + (k - 1) / (2.0 * n * _LN2)), float(h)


def total_correlation_plugin(X: np.ndarray) -> float:
    """TC(X) = sum_i H(X_i) - H(X), bits. X is an integer matrix (n x d)."""
    X = np.asarray(X, dtype=np.int64)
    if X.ndim == 1:
        X = X[:, None]
    sum_h = sum(entropy_plugin(X[:, i]) for i in range(X.shape[1]))
    h_joint, _ = joint_entropy_plugin(X)
    return float(sum_h - h_joint)


def total_correlation_plugin_mm(X: np.ndarray) -> tuple[float, float]:
    """(TC_MM, TC_plugin).

    TC_MM = TC_plugin + [ sum_i (K_i - 1) - (K_J - 1) ] / (2 n ln 2).
    The joint-entropy bias enters with a minus because H(X) is subtracted
    in TC. For 3-variable parity (K_i = 2, K_J = 4) the biases cancel and
    TC_MM == TC_plugin; the correction matters when the joint support is
    larger than the sum of marginal supports minus one.
    """
    X = np.asarray(X, dtype=np.int64)
    if X.ndim == 1:
        X = X[:, None]
    n = X.shape[0]
    marginal = [entropy_plugin_mm(X[:, i]) for i in range(X.shape[1])]
    sum_h_mm = sum(h_mm for h_mm, h_plugin in marginal)
    h_joint, k = joint_entropy_plugin(X)
    h_joint_mm = h_joint + (k - 1) / (2.0 * n * _LN2)
    tc_plugin = sum(h_plugin for h_mm, h_plugin in marginal) - h_joint
    return float(sum_h_mm - h_joint_mm), float(tc_plugin)


def dual_total_correlation_plugin(X: np.ndarray) -> float:
    """DTC(X) = sum_i H(X_{-i}) - (d - 1) H(X), bits."""
    X = np.asarray(X, dtype=np.int64)
    if X.ndim == 1:
        X = X[:, None]
    d = X.shape[1]
    if d < 2:
        return 0.0
    h_joint, _ = joint_entropy_plugin(X)
    sum_reduced = 0.0
    for i in range(d):
        h_red, _ = joint_entropy_plugin(np.delete(X, i, axis=1))
        sum_reduced += h_red
    return float(sum_reduced - (d - 1) * h_joint)


def o_information_plugin(X: np.ndarray) -> float:
    """Omega = TC - DTC, bits. Positive = redundancy-dominated."""
    return total_correlation_plugin(X) - dual_total_correlation_plugin(X)


def gaussian_total_correlation(R: np.ndarray) -> float:
    """TC_G = -0.5 log det(R) for a correlation matrix, reported in bits."""
    R = np.asarray(R, dtype=float)
    sign, logdet = np.linalg.slogdet(R)
    if sign <= 0:
        raise ValueError("Correlation matrix is not positive definite")
    return float(-0.5 * logdet / _LN2)