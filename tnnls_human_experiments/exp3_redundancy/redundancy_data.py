"""Experiment 3 data: TC-matched redundant and synergistic generators.

Redundant common cause: Z ~ Bern(1/2) hidden; X_i = Z xor E_i with
E_i ~ Bern(eta_R) independent. Every pair carries visible dependence.

Synergistic parity: X_1..X_{d-1} ~ Bern(1/2) iid; X_d = xor of them xor E
with E ~ Bern(eta_S). Any proper subset is mutually independent at eta_S=0.

Equal-TC design (paper exp3.md §E3.4): a single population-TC target
(TC_TARGET bits) is fixed; for each group size d each generator's flip rate
is solved exactly from the analytic population TC by bisection (both
TC(eta) are monotone decreasing), and the calibration — target, roots,
residuals, exact population quantities — is frozen before any replicate.
This replaces the reference implementation's coarse grid matching, whose
best pairs missed the tolerance at d=4,5 (parity's population TC is capped
at 1 bit at every d, TC = 1 - h2(eta), so the TC ranges barely overlap).

Analytic facts used here (verified exactly, see test_redundancy.py):
- parity: TC = 1 - h2(eta), DTC = (d-1)*TC, hence Omega = -(d-2)*TC,
  and every pairwise mutual information is exactly zero.
- redundant: TC in (0, d-1), Omega > 0, mean pairwise MI > 0.
"""

from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

TC_TARGET = 0.5          # bits; must be < 1 for parity to be solvable
NUISANCE_MULTIPLIER = 5  # p_n = 5d nuisance columns
BISECTION_ITERS = 80

GEN_REDUNDANT = 'redundant'
GEN_SYNERGISTIC = 'synergistic'
GENERATORS = (GEN_REDUNDANT, GEN_SYNERGISTIC)


def redundant_group(rng, d, n, eta):
    """X_i = Z xor E_i with hidden common cause Z and flip noise eta."""
    z = rng.integers(0, 2, n)
    flips = (rng.random((n, d)) < eta).astype(np.int64)
    return (z[:, None] + flips) % 2


def synergistic_group(rng, d, n, eta):
    """X_d = xor(X_1..X_{d-1}) xor E with flip noise eta."""
    bits = rng.integers(0, 2, size=(n, d - 1)).astype(np.int64)
    parity = bits.sum(axis=1) % 2
    flips = (rng.random(n) < eta).astype(np.int64)
    return np.column_stack([bits, (parity + flips) % 2])


def generate_group(generator, rng, d, n, eta):
    if generator == GEN_REDUNDANT:
        return redundant_group(rng, d, n, eta)
    if generator == GEN_SYNERGISTIC:
        return synergistic_group(rng, d, n, eta)
    raise ValueError(f"Unknown generator: {generator!r}")


# ---------------------------------------------------------------------------
# Exact population distributions and quantities
# ---------------------------------------------------------------------------

def pmf_redundant(d, eta):
    """Exact population pmf of the redundant generator: {tuple: prob}."""
    pmf = defaultdict(float)
    for z in (0, 1):
        for e in itertools.product((0, 1), repeat=d):
            x = tuple((e[i] ^ z) for i in range(d))
            pmf[x] += 0.5 * eta ** e.count(1) * (1 - eta) ** e.count(0)
    return dict(pmf)


def pmf_synergistic(d, eta):
    """Exact population pmf of the parity generator: {tuple: prob}."""
    pmf = defaultdict(float)
    norm = 2 ** (d - 1)
    for b in itertools.product((0, 1), repeat=d - 1):
        parity = sum(b) % 2
        pmf[b + (parity,)] += (1 - eta) / norm
        pmf[b + (parity ^ 1,)] += eta / norm
    return {key: value for key, value in pmf.items() if value > 0}


GENERATOR_PMF = {
    GEN_REDUNDANT: pmf_redundant,
    GEN_SYNERGISTIC: pmf_synergistic,
}


def _entropy(pmf):
    return -sum(p * np.log2(p) for p in pmf.values() if p > 0)


def _marginal_pmf(pmf, indexes):
    out = defaultdict(float)
    for x, p in pmf.items():
        out[tuple(x[i] for i in indexes)] += p
    return dict(out)


def population_quantities(generator, d, eta):
    """Exact TC, DTC, Omega, mean pairwise MI (bits) for one generator."""
    pmf = GENERATOR_PMF[generator](d, eta)
    h_joint = _entropy(pmf)
    tc = sum(
        _entropy(_marginal_pmf(pmf, (i,))) for i in range(d)) - h_joint
    dtc = sum(
        _entropy(_marginal_pmf(pmf, tuple(j for j in range(d) if j != i)))
        for i in range(d)) - (d - 1) * h_joint
    mean_pmi = 0.0
    pairwise = 0
    for a in range(d):
        for b in range(a + 1, d):
            mean_pmi += (
                _entropy(_marginal_pmf(pmf, (a,)))
                + _entropy(_marginal_pmf(pmf, (b,)))
                - _entropy(_marginal_pmf(pmf, (a, b)))
            )
            pairwise += 1
    if pairwise:
        mean_pmi /= pairwise
    return {
        'tc_population': float(tc),
        'dtc_population': float(dtc),
        'omega_population': float(tc - dtc),
        'mean_pmi_population': float(mean_pmi),
    }


def _tc_population(generator, d, eta):
    return population_quantities(generator, d, eta)['tc_population']


# ---------------------------------------------------------------------------
# Equal-TC calibration: bisection to the target, frozen before any replicate
# ---------------------------------------------------------------------------

def solve_eta(generator, d, target=TC_TARGET):
    """Solve eta so the population TC equals the target (bisection).

    TC(eta) is monotone decreasing on (0, 1): the redundant generator's TC
    falls from d-1 (noiseless copies) toward 0, the parity generator's from
    1 bit toward 0. The target must be < 1 bit for parity to be solvable.
    """
    if not 0.0 < target < 1.0:
        raise ValueError(
            f"TC target must be in (0, 1) bits, got {target}. (Parity's "
            "population TC is 1 - h2(eta) <= 1 bit at every d.)")
    lo, hi = 1e-9, 0.95
    for _ in range(BISECTION_ITERS):
        mid = 0.5 * (lo + hi)
        if _tc_population(generator, d, mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def calibrate(tc_target=TC_TARGET):
    """Frozen equal-TC calibration: one row per (d, generator).

    Records the target, the solved flip rate, the TC residual, and the
    exact population quantities. Deterministic: identical inputs always
    produce identical rows, and reproduce runs are expected to reproduce
    the table exactly. The selected (d, generator) -> eta table is the
    only object replicate generation consults — the calibration is frozen
    by construction before any replicate, because it is a pure function
    of the target and d.
    """
    rows = []
    for d in (3, 4, 5):
        for generator in GENERATORS:
            eta = solve_eta(generator, d, tc_target)
            quantities = population_quantities(generator, d, eta)
            rows.append({
                'd': d,
                'generator': generator,
                'tc_target': float(tc_target),
                'eta': float(eta),
                'tc_residual_bits': float(
                    quantities['tc_population'] - tc_target),
                **quantities,
            })
    return rows


def calibration_table():
    """{(d, generator): eta} — the frozen flip-rate table."""
    return {
        (row['d'], row['generator']): row['eta']
        for row in calibrate()
    }


# ---------------------------------------------------------------------------
# Dataset construction
# ---------------------------------------------------------------------------

def generate_dataset(generator, d, eta, n_samples, seed):
    """One dependent group (d columns) + 5d independent Bernoulli(1/2)
    nuisance columns.

    Nuisance independence is guaranteed by construction (independent coin
    draws); implementation-level checks live in test_redundancy.py.
    """
    rng = np.random.default_rng(seed)
    group = generate_group(generator, rng, d, n_samples, eta)
    nuisance = rng.integers(
        0, 2, size=(n_samples, NUISANCE_MULTIPLIER * d))
    return {
        'X': np.hstack([group, nuisance]).astype(np.int64),
        'group_idx': np.arange(d),
        'nuisance_idx': np.arange(d, d + NUISANCE_MULTIPLIER * d),
        'generator': generator,
        'd': d,
        'eta': float(eta),
        'n_samples': n_samples,
        'seed': seed,
    }


def generate_characterization_sample(generator, d, eta, n_samples, seed):
    """Fresh characterization sample (fit-data independent): group + 5d
    nuisance columns, same shape as the fit data.

    Paper exp3.md §E3.7: the two-stage demonstration freezes the discovered
    support on the fit sample and estimates Omega on exactly those columns
    of THIS sample, so no selection information leaks into the
    characterization estimate.
    """
    return generate_dataset(generator, d, eta, n_samples, seed)