"""Shared construction and fitting of Local CorEx estimators."""

from time import time

import numpy as np
from local_corex import BioCorex, LinearCorex


def _fit_is_finite(model, data):
    """Return True when a fitted model's TCs and factor scores are finite."""
    tcs = np.asarray(model.moments['TCs'], dtype=float)
    if not np.all(np.isfinite(tcs)):
        return False
    scores = np.asarray(model.transform(data), dtype=float)
    return bool(np.all(np.isfinite(scores)))


FIT_SEED_ATTEMPTS = 10
FIT_BASE_SEED = 42


def fit_corex_model(data, num_factors, estimator, bio_options=None,
                    linear_options=None):
    """Fit one configured CorEx estimator and return elapsed time.

    The default options preserve the Gaussian experiment's current behavior.
    A linear fit whose TCs diverge to nan (rare init-weight divergence for a
    given dataset/factor-count) is retried with the next seed up to
    FIT_SEED_ATTEMPTS times, and the seed that produced the finite fit is
    reported as model_seed. Bio fits are not retried (BioCorEx takes no seed).
    """
    bio_options = {} if bio_options is None else dict(bio_options)
    linear_options = {} if linear_options is None else dict(linear_options)

    start_time = time()
    if estimator == 'linear':
        options = {
            'n_hidden': num_factors,
            'seed': FIT_BASE_SEED,
            'gaussianize': 'outliers',
        }
        options.update(linear_options)
        requested_seed = options.get('seed', FIT_BASE_SEED)
        model_seed = np.nan
        model = None
        for attempt in range(FIT_SEED_ATTEMPTS):
            options['seed'] = requested_seed + attempt
            model = LinearCorex(**options)
            linear = True
            model.fit_transform(data)
            if _fit_is_finite(model, data):
                model_seed = options['seed']
                break
        else:
            raise ValueError(
                f"Linear CorEx diverged (non-finite TCs) for all "
                f"{FIT_SEED_ATTEMPTS} seeds starting at {requested_seed}.")
    elif estimator == 'bio':
        options = {
            'n_hidden': num_factors,
            'dim_hidden': 2,
            'marginal_description': 'gaussian',
            'smooth_marginals': False,
        }
        options.update(bio_options)
        model = BioCorex(**options)
        linear = False
        model_seed = options.get('seed', np.nan)
        model.fit_transform(data)
        if not _fit_is_finite(model, data):
            raise ValueError(
                "Bio CorEx fit diverged (non-finite TCs) with no seed retry "
                "available.")
    else:
        raise ValueError("estimator must be 'linear' or 'bio'.")

    return model, linear, model_seed, time() - start_time
