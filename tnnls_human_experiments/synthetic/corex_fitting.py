"""Shared construction and fitting of Local CorEx estimators."""

from time import time

import numpy as np
from local_corex import BioCorex, LinearCorex


def fit_corex_model(data, num_factors, estimator, bio_options=None,
                    linear_options=None):
    """Fit one configured CorEx estimator and return elapsed time.

    The default options preserve the Gaussian experiment's current behavior.
    Experiment-specific options can be supplied for later estimators, such as
    Bio CorEx marginal descriptions used by the nonlinear experiments.
    """
    bio_options = {} if bio_options is None else dict(bio_options)
    linear_options = {} if linear_options is None else dict(linear_options)

    start_time = time()
    if estimator == 'linear':
        options = {
            'n_hidden': num_factors,
            'seed': 42,
            'gaussianize': 'outliers',
        }
        options.update(linear_options)
        model = LinearCorex(**options)
        linear = True
        model_seed = options.get('seed', np.nan)
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
    else:
        raise ValueError("estimator must be 'linear' or 'bio'.")

    model.fit_transform(data)
    return model, linear, model_seed, time() - start_time
