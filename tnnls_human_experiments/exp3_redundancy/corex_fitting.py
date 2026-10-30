"""Construction and fitting of Bio CorEx for Experiment 3.

Adapted from ../AI_implemented/exp2_nonlinear/corex_fitting.py (bio discrete
fits only — this experiment's data are binary):

- Every construction receives an explicit integer seed, and every Bio CorEx
  construction/fit runs inside a saved/restored global numpy RNG state:
  BioCorex reseeds the global RNG from OS entropy when ``seed=None``, which
  made fits irreproducible in the earlier Gaussian pipeline.
- ``dim_hidden`` is a CONDITION axis for this experiment (paper exp3.md
  §E3.5): the main arm sets 2^(d-1) (at least sufficient for parity,
  harmlessly overspecified for redundancy), and the sensitivity arm sets 2
  (exactly sufficient for the redundant generator's binary common cause).
- ``atom_keys`` support via a BioCorex subclass that initializes the latent
  one-hot on per-sample atom identities. Contingency for the favorable-pilot
  gate (§E3.5): if ordinary EM fails on parity even at sufficient capacity,
  the Experiment 2 pattern — a ground-truth-informed initialization as a
  capability statement ONLY, labeled ``uses_ground_truth_init=True`` and
  permanently excluded from recovery summaries — is available here.
"""

from time import time

import numpy as np
from local_corex import BioCorex

_LN_2 = np.log(2.0)


class BioCorexAtomSeeded(BioCorex):
    """BioCorex with a hard one-hot latent initialization on atom identity.

    Set ``_atom_keys`` to a per-sample integer array before ``fit``; the
    latent representation is then initialized one-hot on the atom identity
    instead of the default soft initialization. Capability diagnostic only
    (see module docstring); never enter recovery sums with these fits.
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


def fit_bio_corex(data, num_factors, model_seed, dim_hidden=2,
                  max_iter=200, n_repeat=1, smooth_marginals=False,
                  atom_keys=None):
    """Fit one Bio CorEx model and return the fit summary dict.

    The caller supplies the seed; irreproducible fits are not permitted.
    With ``atom_keys`` this runs the capability-diagnostic initialization:
    the returned summary carries ``uses_ground_truth_init=True`` and such
    fits are permanently excluded from recovery summaries.

    Returns fields for the replicate's fit-diagnostic columns; raises on
    failure (the runner catches, records ``failed_fit``, and keeps the
    replicate's ground-truth-side rows — a failed fit costs none of its
    information content).
    """
    if model_seed is None:
        raise ValueError(
            "Pass an explicit integer model_seed: Bio CorEx reseeds the "
            "global numpy RNG from OS entropy when seed is None.")
    data = np.asarray(data).astype(np.int64)
    if data.min() < 0:
        raise ValueError("Discrete Bio CorEx requires nonnegative integers.")

    start_time = time()
    rng_state = np.random.get_state()
    try:
        model_class = BioCorexAtomSeeded if atom_keys is not None else BioCorex
        model = model_class(
            n_hidden=num_factors,
            dim_hidden=int(dim_hidden),
            marginal_description='discrete',
            smooth_marginals=smooth_marginals,
            max_iter=int(max_iter),
            n_repeat=int(n_repeat),
            seed=int(model_seed),
        )
        if atom_keys is not None:
            model._atom_keys = np.asarray(atom_keys)
        model.fit(data)
    finally:
        np.random.set_state(rng_state)
    elapsed = time() - start_time

    # local_corex computes factor TCs and MIs in NATS (exp2's
    # _tcs_to_bits convention); report in BITS so the fit summary matches
    # population quantities and the equivalence margins, which are bits
    # with log-base-2 identities.
    tcs = np.asarray(model.tcs, dtype=float) / _LN_2
    mis = np.asarray(model.mis, dtype=float) / _LN_2
    history = getattr(model, 'tc_history', None) or []
    return {
        'model': model,
        'tcs': tcs,
        'mis': mis,
        'converged': len(history) < int(max_iter) if history else True,
        'fit_runtime_s': elapsed,
        'n_factors_valid': int(np.sum(tcs > 0)),
        'n_negative_mi': int(np.sum(mis < -1e-12)),
        'min_raw_mi': float(mis.min()) if mis.size else 0.0,
        'uses_ground_truth_init': atom_keys is not None,
    }


def factor_variable_scores(model):
    """Factors-by-variables raw association scores (may carry small
    negative entries; callers clip for nonnegative supports, keep raw for
    AUPRC)."""
    scores = np.asarray(model.mis, dtype=float)
    if scores.ndim != 2:
        raise ValueError("Factor-variable scores must be 2-dimensional.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Factor-variable scores contain non-finite values.")
    return scores