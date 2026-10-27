"""Construction and fitting of Local CorEx estimators.

Adapted from ../../gaussian/corex_fitting.py with two experiment-2
hardening changes:

- Every construction receives an explicit integer seed and every Bio CorEx
  construction/fit runs inside a saved/restored global numpy RNG state:
  BioCorex reseeds the global RNG from OS entropy when ``seed=None``, which
  made fits irreproducible in the earlier Gaussian pipeline.
- ``atom_keys`` support via a BioCorex subclass that initializes the latent
  one-hot on per-sample atom identities. This exists ONLY for the group
  total correlation (TC)-capability diagnostic: it shows the variational
  family can represent a pairwise-independent structure standard EM
  initializations never find. Any fit using it must be labeled
  ``uses_ground_truth_init=True`` and must never enter recovery-metric
  summaries.

Estimators: ``linear`` (covariance-based baseline) and ``bio`` (primary;
pass ``marginal_description='gaussian'`` in bio_options for native
continuous fits).
"""

from time import time

import numpy as np
from local_corex import BioCorex, LinearCorex


class BioCorexAtomSeeded(BioCorex):
    """BioCorex with a hard one-hot latent initialization on atom identity.

    Set ``_atom_keys`` to a per-sample integer array before ``fit``; the
    latent representation is then initialized one-hot on the atom identity
    instead of the default soft initialization. Diagnostic use only (see
    module docstring).
    """

    def initialize_representation(self):
        super().initialize_representation()
        atom_keys = getattr(self, '_atom_keys', None)
        if atom_keys is None:
            return
        _, assignment = np.unique(atom_keys, return_inverse=True)
        if len(np.unique(atom_keys)) > self.dim_hidden:
            assignment = assignment % self.dim_hidden
        hard = np.zeros((self.n_hidden, self.n_samples, self.dim_hidden))
        for factor in range(self.n_hidden):
            hard[factor, np.arange(self.n_samples), assignment] = 1.0
        self.p_y_given_x = hard


def fit_corex_model(data, num_factors, estimator, model_seed=None,
                    bio_options=None, linear_options=None):
    """Fit one CorEx estimator and return the fitted model and elapsed time.

    Returns ``(model, linear, model_seed, elapsed_s)``. ``model_seed`` is the
    integer seed actually used (np.nan only if a caller forces seed=None,
    which is not reproducible and is therefore discouraged).
    """
    bio_options = {} if bio_options is None else dict(bio_options)
    linear_options = {} if linear_options is None else dict(linear_options)

    if model_seed is None:
        raise ValueError(
            "Pass an explicit integer model_seed: Bio CorEx reseeds the "
            "global numpy RNG from OS entropy when seed is None."
        )

    start_time = time()
    # Both estimators seed the global numpy RNG internally (LinearCorex and
    # BioCorex call np.random.seed in their constructors); save and restore
    # the state so construction and fitting cannot perturb the pipeline's
    # RNG-dependent code, and fits with identical inputs stay reproducible.
    rng_state = np.random.get_state()
    try:
        if estimator == 'linear':
            options = {
                'n_hidden': num_factors,
                'seed': int(model_seed),
                'gaussianize': 'outliers',
            }
            options.update(linear_options)
            model = LinearCorex(**options)
            linear = True
            used_seed = options.get('seed', np.nan)
            model.fit(data)
        elif estimator == 'bio':
            options = {
                'n_hidden': num_factors,
                'dim_hidden': 2,
                'marginal_description': 'discrete',
                'smooth_marginals': False,
            }
            options.update(bio_options)
            options.setdefault('seed', int(model_seed))
            atom_keys = options.pop('atom_keys', None)
            model_class = (
                BioCorexAtomSeeded if atom_keys is not None else BioCorex)
            if marginal_is_discrete(options.get('marginal_description')):
                integer_data = np.asarray(data)
                if integer_data.min() < 0:
                    raise ValueError(
                        "Discrete Bio CorEx requires nonnegative integer data.")
                data = integer_data.astype(np.int64)
            model = model_class(**options)
            if atom_keys is not None:
                model._atom_keys = np.asarray(atom_keys)
            model.fit(data)
            linear = False
            used_seed = options.get('seed', np.nan)
        else:
            raise ValueError("estimator must be 'linear' or 'bio'.")
    finally:
        np.random.set_state(rng_state)

    return model, linear, used_seed, time() - start_time


def marginal_is_discrete(description):
    """True if a Bio CorEx marginal description needs nonnegative integers."""
    return description in (None, 'discrete')


def factor_variable_scores(model, linear=True):
    """Extract the factors-by-variables association-score matrix.

    Linear CorEx: ``moments['MI']`` equals -0.5 log(1 - rho^2), the Gaussian
    MI, nonnegative for |rho| <= 1. Bio CorEx: ``mis`` is a discrete MI
    estimate that may be slightly negative from estimator noise; raw values
    are preserved and the caller may clip for nonnegative scores.
    """
    if linear:
        scores = model.moments['MI']
    else:
        scores = model.mis
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("Factor-variable scores must be a two-dimensional array.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Factor-variable scores contain non-finite values.")
    return scores


def standardize_columns(data, variance_floor=1e-12):
    """Standardize columns by mean/std; near-constant columns pass zeros.

    The single standardization rule for the pipeline (spec 02: continuous
    inputs are standardized before binning or native-Gaussian fitting).
    """
    data = np.asarray(data, dtype=float)
    means = data.mean(axis=0)
    stds = data.std(axis=0)
    valid = stds > np.sqrt(variance_floor)
    standardized = np.zeros_like(data)
    standardized[:, valid] = (
        (data[:, valid] - means[valid]) / stds[valid]
    )
    return standardized


def discretize_columns(data, n_bins):
    """Equal-frequency binning per column; bins estimated on the given data.

    Returns ``(binned ints 0..n_bins-1, bin_edges (p, n_bins + 1))``. Columns
    with ties collapse to fewer effective bins; the saved edges reflect
    this. The discretization rule must be fixed before the core run (spec
    02) and the edges saved with the condition outputs.
    """
    data = np.asarray(data, dtype=float)
    n_variables = data.shape[1]
    binned = np.zeros_like(data, dtype=np.int64)
    edges = np.zeros((n_variables, n_bins + 1))
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)
    for variable in range(n_variables):
        unique_edges = np.unique(np.quantile(data[:, variable], quantiles))
        binned[:, variable] = np.searchsorted(
            unique_edges[1:-1], data[:, variable], side='right')
        edges[variable, :len(unique_edges)] = unique_edges
        edges[variable, len(unique_edges):] = unique_edges[-1]
    return binned, edges


def validate_corex_fitting():
    """Fast helper checks (no model fits)."""
    standardized = standardize_columns(
        np.column_stack([np.arange(10, dtype=float), np.zeros(10)]))
    assert np.isclose(standardized[:, 0].std(), 1.0)
    assert not standardized[:, 1].any()

    binned, edges = discretize_columns(
        np.arange(100, dtype=float).reshape(-1, 1), 10)
    assert binned.max() == 9
    assert np.isclose(edges[0, -1], 99.0)

    for seed in (None, np.nan):
        try:
            fit_corex_model(np.zeros((4, 3)), 1, 'bio', model_seed=seed)
        except ValueError:
            continue
        raise AssertionError("fit_corex_model must reject non-integer seeds")

    return {
        'seed_required_checked': True,
        'standardization_checked': True,
        'equal_frequency_binning_checked': True,
    }