"""Structure diagnostics for the nonlinear experiments (spec 02, table 5).

Two independent diagnostic blocks, both run on >= 100k freshly generated
observations (independent of the fitting data; data separation):

1. ``structure_diagnostics``: per (generator, group size d, noise level),
   the quantities of the diagnostic table in bits:
   - max |Pearson correlation| within the group,
   - max pairwise MI (10-bin equal-frequency discretization for continuous
     generators; raw bits for parity),
   - group TC (plug-in, bits),
   - permuted TC (plug-in TC after independently permuting one member,
     bits): an independence check that removes the higher-order structure
     while preserving marginals, and must be ~ 0.

2. ``capability_diagnostic``: shows that the failure of standard CorEx EM
   to find parity structure is an optimization failure, not an
   expressiveness failure of the variational family:
   (a) standard random inits find TC ~ 0 on pure parity groups, while
   (b) a latent initialized one-hot on the joint atom (the ground-truth
       grouping of samples) reaches the full parity TC (1 bit for every
       d >= 2: the parity constraint leaves 2^(d-1) equiprobable atoms).
   Rows from (b) are labeled uses_ground_truth_init and must NEVER enter
   recovery-metric summaries: they are a capability statement only.

Unit convention: the plugin TC here is computed in bits. local_corex
reports model.tcs in NATS, so the capability diagnostic converts model TCs
to bits (divide by ln 2) before comparing to the d - 1 bit parity truth.
"""

import numpy as np
from pathlib import Path
from sklearn.metrics import mutual_info_score

from corex_fitting import fit_corex_model, discretize_columns
from nonlinear_data import parity_group, product_group, sine_sum_group
from seeds import derived_seed

LN_2 = np.log(2.0)
GROUP_SIZES = (3, 4)
DIAGNOSTIC_N = 100_000
BIO_MAX_ITER = 200
BIO_DIM_HIDDEN = 4
BIO_N_REPEAT = 5

DIAGNOSTIC_CONDITIONS = (
    [('parity', d, eta) for d in GROUP_SIZES for eta in (0.0, 0.05, 0.15)]
    + [('sine_sum', d, sigma)
       for d in GROUP_SIZES for sigma in (0.05, 0.20, 0.50)]
    + [('product', 4, 0.20)]
)

STRUCTURE_COLUMNS = [
    'generator', 'group_size', 'noise', 'n', 'max_abs_pearson',
    'max_pairwise_mi_bits', 'group_tc_bits', 'permuted_group_tc_bits',
    'permuted_max_pairwise_mi_bits',
]

CAPABILITY_COLUMNS = [
    'group_size', 'eta', 'n', 'true_tc_bits', 'std_init_tc_bits',
    'seeded_init_tc_bits', 'uses_ground_truth_init', 'note',
]


def total_correlation_bits(disc):
    """Plug-in total correlation TC = sum_i H(X_i) - H(X_1..X_d), in bits.

    Uses the observed relative frequencies; no smoothing. For continuous
    generators this estimates the TC of the discretized distribution, which
    is the object CorEx with discrete marginals actually models.
    """
    disc = np.asarray(disc, dtype=np.int64)
    d = disc.shape[1]
    tc = 0.0
    for variable in range(d):
        _, counts = np.unique(disc[:, variable], return_counts=True)
        p = counts / disc.shape[0]
        tc += float(-(p * np.log2(p)).sum())
    _, joint_counts = np.unique(disc, axis=0, return_counts=True)
    p_joint = joint_counts / disc.shape[0]
    tc -= float(-(p_joint * np.log2(p_joint)).sum())
    return tc


def structure_diagnostics(generator, group_size, noise, n, seed):
    """Diagnostic-table quantities for one (generator, d, noise) cell."""
    rng = np.random.default_rng(seed)
    if generator == 'parity':
        block = parity_group(rng, group_size, n, noise)
        disc = block.astype(np.int64)
    elif generator == 'sine_sum':
        block = sine_sum_group(rng, group_size, n, noise)
        disc, _ = discretize_columns(block, 10)
    elif generator == 'product':
        block = product_group(rng, group_size, n, noise)
        disc, _ = discretize_columns(block, 10)
    else:
        raise ValueError(
            f"Unknown generator '{generator}'. "
            "Choose 'parity', 'sine_sum', or 'product'.")

    correlations = np.corrcoef(block.T)
    offdiagonal = np.abs(correlations - np.eye(group_size))
    max_abs_pearson = float(offdiagonal.max())

    max_pairwise_mi = max(
        float(mutual_info_score(disc[:, i], disc[:, j]))
        for i in range(group_size) for j in range(i + 1, group_size)
    )
    group_tc = total_correlation_bits(disc)

    # permuted: break all higher-order structure while preserving marginals
    rng_perm = np.random.default_rng(seed + 1)
    permuted = disc.copy()
    permuted[:, 0] = rng_perm.permutation(permuted[:, 0])
    permuted_tc = total_correlation_bits(permuted)
    permuted_max_mi = max(
        float(mutual_info_score(permuted[:, i], permuted[:, j]))
        for i in range(group_size) for j in range(i + 1, group_size)
    )

    return {
        'generator': generator,
        'group_size': group_size,
        'noise': float(noise),
        'n': int(n),
        'max_abs_pearson': max_abs_pearson,
        'max_pairwise_mi_bits': max_pairwise_mi,
        'group_tc_bits': group_tc,
        'permuted_group_tc_bits': permuted_tc,
        'permuted_max_pairwise_mi_bits': permuted_max_mi,
    }


def _tcs_to_bits(tcs):
    """Convert local_corex per-factor TCs (nats) to bits."""
    return np.asarray(tcs, dtype=float) / LN_2


def capability_diagnostic(group_size, eta, n, seed):
    """Standard-init vs atom-seeded-init Bio CorEx TC on one parity group.

    Returns a row dict. The atom-seeded fit must never enter recovery
    summaries; callers record its row with uses_ground_truth_init=True.
    """
    rng = np.random.default_rng(seed)
    block = parity_group(rng, group_size, n, eta)
    # one latent variable, multiple EM restarts for the standard init
    std_model, _, _, _ = fit_corex_model(
        block, 1, 'bio', model_seed=seed,
        bio_options={
            'max_iter': BIO_MAX_ITER,
            'dim_hidden': BIO_DIM_HIDDEN,
            'n_repeat': BIO_N_REPEAT,
        },
    )
    # atom-seeded init: encode each observed joint pattern uniquely. Bit
    # sums collide under parity ((0,0,1) and (1,1,0) share a sum), so the
    # row's binary powers are used, and dim_hidden covers 2^d atoms.
    powers = 2 ** np.arange(group_size)
    atom_keys = block @ powers
    seeded_model, _, _, _ = fit_corex_model(
        block, 1, 'bio', model_seed=seed + 1,
        bio_options={
            'max_iter': BIO_MAX_ITER,
            'dim_hidden': max(BIO_DIM_HIDDEN, 2 ** group_size),
            'n_repeat': 1,
            'atom_keys': atom_keys,
        },
    )
    return {
        'group_size': group_size,
        'eta': float(eta),
        'n': int(n),
        # parity TC is exactly 1 bit for every d >= 2: the parity constraint
        # leaves 2^(d-1) equiprobable atoms, H(X) = d-1 bits, sum H(X_i) = d
        'true_tc_bits': 1.0 if eta == 0 else np.nan,
        'std_init_tc_bits': float(_tcs_to_bits(std_model.tcs)[0]),
        'seeded_init_tc_bits': float(_tcs_to_bits(seeded_model.tcs)[0]),
        'uses_ground_truth_init': True,
        'note': ('standard EM finds ~0 TC on pairwise-independent parity; '
                 'the latent family CAN represent full parity TC when '
                 'seeded with the joint atom (optimization failure, not '
                 'an expressiveness failure).'),
    }


def validate_structure_diagnostics():
    """Fast checks on the TC-in-bits and permutation logic (no model fits)."""

    # pure parity of 3 bits: TC = 1 bit, all pairwise MI = 0
    rng = np.random.default_rng(0)
    parity = parity_group(rng, 3, 50_000, 0.0)
    parity_tc = total_correlation_bits(parity.astype(np.int64))
    pairwise = [
        float(mutual_info_score(parity[:, i], parity[:, j]))
        for i in range(3) for j in range(i + 1, 3)
    ]
    assert abs(parity_tc - 1.0) < 0.02, parity_tc
    assert max(pairwise) < 0.005, pairwise

    # independent columns: both TC and pairwise MI ~ 0
    independent = rng.integers(0, 2, size=(50_000, 3))
    assert abs(total_correlation_bits(independent)) < 0.02

    # permuting one column of parity destroys the parity relation
    permuted = parity.copy()
    rng_perm = np.random.default_rng(1)
    permuted[:, 0] = rng_perm.permutation(permuted[:, 0])
    assert total_correlation_bits(permuted.astype(np.int64)) < 0.02

    # continuous generator diagnostics run and return finite values
    diag = structure_diagnostics('sine_sum', 3, 0.20, 20_000, seed=7)
    assert np.isfinite(diag['group_tc_bits'])
    assert diag['permuted_group_tc_bits'] < diag['group_tc_bits'] + 1e-6
    diag = structure_diagnostics('product', 4, 0.20, 20_000, seed=8)
    assert np.isfinite(diag['max_pairwise_mi_bits'])

    return {
        'parity_tc_bits_checked': True,
        'pairwise_independence_checked': True,
        'permutation_destroys_parity_checked': True,
        'continuous_diagnostics_checked': True,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_all_diagnostics(output_dir, n=DIAGNOSTIC_N, capability=True):
    """Compute and save the two diagnostics tables as CSVs.

    structure_diagnostics.csv: plugin TC-family quantities in bits per
    (generator, d, noise). capability_diagnostics.csv: standard-init vs
    atom-seeded-init Bio CorEx TC on parity; seeded rows carry
    uses_ground_truth_init=True and must never enter recovery summaries.
    """
    import pandas as pd

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    structure_rows = []
    for generator, group_size, noise in DIAGNOSTIC_CONDITIONS:
        seed = derived_seed(generator, group_size, noise)
        row = structure_diagnostics(generator, group_size, noise, n, seed)
        structure_rows.append(row)
        print(
            f"diagnostic {generator} d={group_size} noise={noise}: "
            f"TC={row['group_tc_bits']:.3f} bits, "
            f"permuted TC={row['permuted_group_tc_bits']:.4f} bits, "
            f"max pairwise MI={row['max_pairwise_mi_bits']:.4f} bits"
        )
    structure_path = output_dir / 'structure_diagnostics.csv'
    pd.DataFrame(structure_rows).to_csv(structure_path, index=False)
    print(f"Structure diagnostics written: {structure_path}")

    if not capability:
        return structure_path
    capability_rows = []
    for group_size in GROUP_SIZES:
        for eta in (0.0, 0.05, 0.15):
            row = capability_diagnostic(
                group_size, eta, n, derived_seed('capability', group_size,
                                                 eta))
            capability_rows.append(row)
            print(
                f"capability d={group_size} eta={eta}: "
                f"std-init TC={row['std_init_tc_bits']:.4f} bits, "
                f"atom-seeded TC={row['seeded_init_tc_bits']:.4f} bits "
                f"(true {row['true_tc_bits']})"
            )
    capability_path = output_dir / 'capability_diagnostics.csv'
    pd.DataFrame(capability_rows).to_csv(capability_path, index=False)
    print(f"Capability diagnostics written: {capability_path}")
    return structure_path, capability_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Run the >= 100k structure and capability diagnostics.',
    )
    parser.add_argument(
        '--output-dir', type=Path, default=Path('nonlinear_results'),
        help='Directory for the diagnostics CSVs.',
    )
    parser.add_argument(
        '--n', type=int, default=DIAGNOSTIC_N,
        help='Independent sample count for every diagnostic cell.',
    )
    parser.add_argument(
        '--skip-capability', action='store_true',
        help='Skip the Bio CorEx capability diagnostic (model fits).',
    )
    arguments = parser.parse_args()
    run_all_diagnostics(
        arguments.output_dir, n=arguments.n,
        capability=not arguments.skip_capability,
    )