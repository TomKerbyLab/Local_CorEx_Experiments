"""CorEx fitting wrappers.

Wraps local_corex LinearCorex and BioCorex behind a uniform interface and
protects the global numpy RNG: BioCorex calls np.random.seed(seed) in its
__init__ and uses the global RNG throughout, so fits run inside
np.random.get_state()/set_state() to keep the rest of the pipeline
deterministic.

Association scores: the spec requires a_{ji} = I(Z_j; X_i).
LinearCorex.mis already equals -0.5 log(1 - rho^2), the MI of Gaussian
variables with correlation rho, and is nonnegative for |rho| <= 1.
BioCorex.mis is a discrete MI estimate and may be slightly negative from
estimator noise. Per the README, negative values are never silently clipped:
they are recorded in the fit record and the raw returned value is stored
alongside the nonnegative score used for matching/supports.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from local_corex import BioCorex, LinearCorex


class BioCorexSeeded(BioCorex):
    """BioCorex with a hard one-hot latent initialization.

    If `_atom_keys` (per-sample integer atom identities) is set before fit,
    the latent is initialized one-hot on atom identity instead of the
    default soft Dirichlet. This exists ONLY for the Experiment 2 capability
    diagnostic: it demonstrates that the variational family can represent a
    pairwise-independent structure that standard EM initializations never
    find. Any fit using this initialization must be labeled
    uses_ground_truth_init=true and must never enter recovery-metric tables.
    """

    def initialize_representation(self):
        super().initialize_representation()
        keys = getattr(self, "_atom_keys", None)
        if keys is None:
            return
        uniq, assign = np.unique(keys, return_inverse=True)
        if len(uniq) > self.dim_hidden:
            assign = assign % self.dim_hidden
        hard = np.zeros((self.n_hidden, self.n_samples, self.dim_hidden))
        for j in range(self.n_hidden):
            hard[j, np.arange(self.n_samples), assign] = 1.0
        self.p_y_given_x = hard


@dataclass
class CorexFitRecord:
    """Everything the specs require recording about one CorEx fit."""

    estimator: str                 # "linear" | "bio"
    n_factors_requested: int
    n_factors_valid: int
    tcs: np.ndarray                # per-factor explained TC
    mis: np.ndarray                # (n_factors, n_visible) raw association scores
    converged: bool
    runtime_s: float
    seed: int
    n_samples: int
    n_visible: int
    n_negative_mi: int = 0         # count of raw scores < -1e-12
    min_raw_mi: float = 0.0
    notes: list = field(default_factory=list)
    transformed_score: str = "none"  # documentation of any transformation applied
    uses_ground_truth_init: bool = False  # capability diagnostic only
    model: Optional[object] = None

    def association_scores(self) -> np.ndarray:
        """Nonnegative scores used for matching/supports (documented transform)."""
        return np.clip(self.mis, 0.0, None)

    def total_tc(self) -> float:
        return float(np.sum(self.tcs))


def fit_linear_corex(
    X: np.ndarray,
    n_factors: int,
    seed: int,
    max_iter: int = 10000,
    tol: float = 1e-5,
    gaussianize: str = "standard",
    verbose: bool = False,
) -> CorexFitRecord:
    """Fit LinearCorex on standardized continuous input X (n x p)."""
    X = np.asarray(X, dtype=float)
    t0 = time.time()
    model = LinearCorex(
        n_hidden=n_factors,
        max_iter=max_iter,
        tol=tol,
        gaussianize=gaussianize,
        seed=int(seed),
        verbose=verbose,
    )
    model.fit(X)
    runtime = time.time() - t0
    mis = np.asarray(model.mis, dtype=float)
    tcs = np.asarray(model.tcs, dtype=float)
    rec = CorexFitRecord(
        estimator="linear",
        n_factors_requested=n_factors,
        n_factors_valid=int(np.sum(tcs > 0)),
        tcs=tcs,
        mis=mis,
        converged=len(model.history.get("TC", [])) < model.max_iter,
        runtime_s=runtime,
        seed=int(seed),
        n_samples=int(X.shape[0]),
        n_visible=int(X.shape[1]),
        n_negative_mi=int(np.sum(mis < -1e-12)),
        min_raw_mi=float(mis.min()) if mis.size else 0.0,
        model=model,
    )
    return rec


def fit_bio_corex(
    X: np.ndarray,
    n_factors: int,
    seed: int,
    marginal_description: str = "discrete",
    dim_hidden: int = 2,
    max_iter: int = 200,
    max_samples: int = 100000,
    n_repeat: int = 1,
    verbose: bool = False,
    n_cpu: int = 1,
    atom_keys: np.ndarray | None = None,
) -> CorexFitRecord:
    """Fit BioCorex. X must be nonnegative integers for 'discrete' marginals.

    n_repeat reruns EM from fresh random initializations and keeps the
    highest-TC solution (BioCorex's built-in restart mechanism).

    atom_keys: optional per-sample atom identities for a hard one-hot latent
    init (capability diagnostic ONLY; see BioCorexSeeded docstring).

    Global RNG protection: BioCorex seeds the global numpy RNG; the state is
    saved and restored around construction and fitting.
    """
    X = np.asarray(X)
    if marginal_description == "discrete":
        X = X.astype(np.int64)
        assert X.min() >= 0, "discrete BioCorex requires nonnegative integers"
        assert X.max() <= 31, "BioCorex discrete values must be in 0..31"
    t0 = time.time()
    state = np.random.get_state()
    try:
        model = BioCorexSeeded(
            n_hidden=n_factors,
            dim_hidden=dim_hidden,
            marginal_description=marginal_description,
            max_iter=max_iter,
            max_samples=max_samples,
            n_repeat=n_repeat,
            seed=int(seed),
            verbose=verbose,
            n_cpu=n_cpu,
        )
        if atom_keys is not None:
            model._atom_keys = np.asarray(atom_keys)
        model.fit(X)
    finally:
        np.random.set_state(state)
    runtime = time.time() - t0
    mis = np.asarray(model.mis, dtype=float)
    tcs = np.asarray(model.tcs, dtype=float)
    hist = getattr(model, "tc_history", None) or []
    rec = CorexFitRecord(
        estimator="bio",
        n_factors_requested=n_factors,
        n_factors_valid=int(np.sum(tcs > 0)),
        tcs=tcs,
        mis=mis,
        converged=len(hist) < model.max_iter if hist else True,
        runtime_s=runtime,
        seed=int(seed),
        n_samples=int(X.shape[0]),
        n_visible=int(X.shape[1]),
        n_negative_mi=int(np.sum(mis < -1e-12)),
        min_raw_mi=float(mis.min()) if mis.size else 0.0,
        uses_ground_truth_init=atom_keys is not None,
        model=model,
    )
    return rec


# ---------------------------------------------------------------------------
# Standardization and discretization
# ---------------------------------------------------------------------------

def standardize_columns(
    X: np.ndarray, var_floor: float = 1e-12
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Standardize columns of X by mean/std; flag near-zero-variance columns.

    Returns (X_std, valid_mask, excluded_mask). Columns with variance below
    var_floor are set to zero and flagged invalid; callers must exclude them
    from CorEx fitting while retaining their original indices in metadata.
    """
    X = np.asarray(X, dtype=float)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    valid = std > np.sqrt(var_floor)
    X_std = np.zeros_like(X)
    X_std[:, valid] = (X[:, valid] - mean[valid]) / std[valid]
    return X_std, valid, ~valid


def discretize_equal_frequency(
    X: np.ndarray, n_bins: int
) -> tuple[np.ndarray, np.ndarray]:
    """Equal-frequency binning per column, bins estimated on the given data.

    Returns (binned ints in 0..n_bins-1, bin_edges (p, n_bins+1)). Columns
    with ties collapse to fewer effective bins; the saved edges reflect this.
    """
    X = np.asarray(X, dtype=float)
    n, p = X.shape
    binned = np.zeros_like(X, dtype=np.int64)
    edges = np.zeros((p, n_bins + 1))
    qs = np.linspace(0.0, 1.0, n_bins + 1)
    for i in range(p):
        e = np.unique(np.quantile(X[:, i], qs))
        binned[:, i] = np.searchsorted(e[1:-1], X[:, i], side="right")
        edges[i, : len(e)] = e
        edges[i, len(e):] = e[-1]
    return binned, edges