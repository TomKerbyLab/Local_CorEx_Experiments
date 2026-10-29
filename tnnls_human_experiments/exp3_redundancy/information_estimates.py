"""Empirical information-theoretic estimators for Experiment 3.

All logarithms base 2 (quantities in bits). Plug-in entropy estimation for
discrete data; the plug-in entropy is biased DOWNWARD and the Miller-Madow
correction adds (K - 1) / (2 n ln 2) for K occupied cells:

    H_MM = H_plugin + (K - 1) / (2 n ln 2)

Every entropy entering TC or DTC receives its own correction from its own
occupied-cell count, combined with the identity signs:

    TC  = sum_i H(X_i) - H(X)
        => TC_MM  = TC_plugin + [ sum_i (K_i - 1) - (K_J - 1) ] / (2 n ln 2)
    DTC = sum_i H(X_-i) - (d - 1) H(X)
        => DTC_MM = DTC_plugin + [ sum_i (K_-i - 1) - (d-1)(K_J - 1) ]
                    / (2 n ln 2)
    Omega = TC - DTC, Omega_MM = TC_MM - DTC_MM

The MM correction enters with opposite signs on the two sides of each
identity, so no single-entropy shortcut can produce the corrected TC or DTC:
both reduced-joint and full-joint corrections are needed (e.g. at positive
noise the d=3 parity joint occupies all 8 cells, K_J = 8, while the
noiseless case occupies 4 and the biases cancel exactly there — the
qualified statement in paper exp3.md §E3.2).
"""

from __future__ import annotations

import numpy as np

_LN2 = np.log(2.0)


def _row_keys(X):
    """Map each row of an integer matrix to a unique integer key."""
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[1] == 0:
        return np.zeros(X.shape[0], dtype=np.int64)
    max_vals = X.max(axis=0).astype(np.int64) + 1
    multipliers = np.ones(X.shape[1], dtype=np.int64)
    for i in range(X.shape[1] - 2, -1, -1):
        multipliers[i] = multipliers[i + 1] * max_vals[i + 1]
    return (X.astype(np.int64) * multipliers).sum(axis=1)


def _entropy_from_counts(counts):
    counts = np.asarray(counts, dtype=float)
    p = counts / counts.sum()
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def joint_entropy_plugin(X):
    """(H_plugin bits, occupied cells K) of the joint over columns."""
    X = np.asarray(X, dtype=np.int64)
    if X.ndim == 1:
        X = X[:, None]
    if X.shape[0] == 0:
        return 0.0, 0
    keys, counts = np.unique(_row_keys(X), return_counts=True)
    return _entropy_from_counts(counts), int(len(counts))


def joint_entropy_mm(X):
    """(H_MM, H_plugin) of the joint entropy over columns."""
    h, k = joint_entropy_plugin(X)
    n = np.asarray(X).shape[0]
    if n == 0:
        return 0.0, 0.0
    return float(h + (k - 1) / (2.0 * n * _LN2)), float(h)


def total_correlation_plugin_mm(X):
    """(TC_MM, DTC_MM, TC_plugin, DTC_plugin) in bits, all MM-corrected.

    TC and DTC and their MM variants share the entropies computed here, so
    one call per group produces the full estimator block; Omega is
    TC - DTC by either variant (caller-side subtraction).
    """
    X = np.asarray(X, dtype=np.int64)
    if X.ndim == 1:
        X = X[:, None]
    n, d = X.shape
    if d < 2:
        return 0.0, 0.0, 0.0, 0.0
    h_joint, k_joint = joint_entropy_plugin(X)
    correction_joint = (k_joint - 1) / (2.0 * n * _LN2)

    sum_h = 0.0
    correction_marginals = 0.0
    for i in range(d):
        h_i, k_i = joint_entropy_plugin(X[:, i])
        sum_h += h_i
        correction_marginals += k_i - 1
    tc_plugin = sum_h - h_joint
    tc_mm = tc_plugin + (
        correction_marginals - (k_joint - 1)) / (2.0 * n * _LN2)

    sum_reduced = 0.0
    correction_reduced = 0.0
    for i in range(d):
        h_red, k_red = joint_entropy_plugin(
            np.delete(X, i, axis=1))
        sum_reduced += h_red
        correction_reduced += k_red - 1
    dtc_plugin = sum_reduced - (d - 1) * h_joint
    dtc_mm = dtc_plugin + (
        correction_reduced - (d - 1) * (k_joint - 1)) / (2.0 * n * _LN2)
    return float(tc_mm), float(dtc_mm), float(tc_plugin), float(dtc_plugin)


def group_estimates(X):
    """The full estimator block for one group: plug-in and MM TC/DTC/Omega."""
    tc_mm, dtc_mm, tc_plugin, dtc_plugin = total_correlation_plugin_mm(X)
    return {
        'tc_plugin': tc_plugin,
        'dtc_plugin': dtc_plugin,
        'omega_plugin': tc_plugin - dtc_plugin,
        'tc_mm': tc_mm,
        'dtc_mm': dtc_mm,
        'omega_mm': tc_mm - dtc_mm,
    }