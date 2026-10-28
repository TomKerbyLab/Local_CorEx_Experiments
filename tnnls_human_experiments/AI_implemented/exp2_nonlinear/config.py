"""Experimental configuration: the knobs you edit between runs.

One place for everything machine-readable (spec 02 grids and options).
This is plain Python so it is versioned, importable, and validated by
tests like the rest of the pipeline; the runner reads it at import time.
The grid builders themselves (the loops over these levels and the seed
offsets) stay in run_nonlinear_experiments.py.
"""

import partitioning
import recovery_metrics

# ---------------------------------------------------------------------------
# Fit and estimation defaults (shared by every experiment)
# ---------------------------------------------------------------------------

DEFAULTS = {
    'n_factors': 4,             # M = 4 requested factors (spec 02)
    'phate_dimensions': partitioning.PHATE_DIMENSIONS,
    'kmeans_n_init': partitioning.KMEANS_N_INIT,
    'n_random_draws': partitioning.N_RANDOM_DRAWS,
    'n_bins': 10,               # fixed pre-run discretization rule
    'support_fraction': recovery_metrics.SUPPORT_FRACTION,
    'bio_max_iter': 200,
    'bio_dim_hidden': 4,        # states per latent factor (main grid; the
                                # 2^d-state comparison is the capability
                                # diagnostic's, not this grid's)
    'bio_n_repeat': 5,          # EM restarts (escape trivial basins)
    'linear_max_iter': 2000,
    'native_gaussian_bio': True,  # continuous secondary variant
}

# ---------------------------------------------------------------------------
# Option tables per experiment (grid builders live in the runner)
# ---------------------------------------------------------------------------

BASE_ESTIMATORS = ('bio', 'linear')
BASE_COREX_INPUTS = ('with_routing',)

EXPERIMENTS = {
    'smoke': {
        'estimators': BASE_ESTIMATORS,
        'corex_inputs': BASE_COREX_INPUTS,
        'native_gaussian_bio': False,
    },
    # repetitions lives here (not per-invocation): the spec'd grid is
    # 16 paired replicates per condition (§4.3); --repetitions still
    # overrides it. The smoke builder hard-codes its 2 conditions and
    # ignores this key.
    'discrete_core': {
        'repetitions': 16,
        'estimators': BASE_ESTIMATORS,
        'corex_inputs': BASE_COREX_INPUTS,
        'native_gaussian_bio': False,
    },
    'continuous_core': {
        'repetitions': 16,
        'estimators': BASE_ESTIMATORS,
        'corex_inputs': BASE_COREX_INPUTS,
        'native_gaussian_bio': True,
    },
    'discrete_supplementary': {
        'repetitions': 16,
        'estimators': BASE_ESTIMATORS,
        'corex_inputs': BASE_COREX_INPUTS,
        'native_gaussian_bio': False,
    },
    'continuous_supplementary': {
        'repetitions': 16,
        'estimators': BASE_ESTIMATORS,
        'corex_inputs': BASE_COREX_INPUTS,
        'native_gaussian_bio': False,
    },
}

# global-capacity control grid (capacity_experiment.py); deliberately kept
# out of EXPERIMENTS so the runner's argparse choices stay unchanged
CAPACITY_CONFIG = {
    'estimators': BASE_ESTIMATORS,
    'corex_inputs': BASE_COREX_INPUTS,
    'native_gaussian_bio': False,
}

# ---------------------------------------------------------------------------
# Grid levels (consumed by the runner's condition builders)
# ---------------------------------------------------------------------------

CORE_LEVELS = {
    'n_per_region': (500, 5000),
    'alpha': (0.0, 0.5, 1.0),
    'eta': (0.0, 0.05, 0.15),   # parity flip rate
    'sigma': (0.05, 0.20, 0.50),  # sine-sum / product noise
    'nuisance': 14,             # nuisance dimensions in the core grid
}

# sensitivity axes for the supplementary grids (one axis at a time)
SUPPLEMENTARY_LEVELS = {
    'small_n': 50,
    'extra_eta': (0.10, 0.20),
    'nuisance_sweep': (0, 56),
}

ROOT_SEED_BASE = {'parity': 2000, 'sine_sum': 3000, 'product': 3000}

GENERATOR_NOISE_NAMES = {
    'parity': 'eta', 'sine_sum': 'sigma', 'product': 'sigma',
}