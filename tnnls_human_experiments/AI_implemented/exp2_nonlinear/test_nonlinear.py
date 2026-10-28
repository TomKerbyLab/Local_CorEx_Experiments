"""Validation tests for the nonlinear higher-order experiments (spec 02).

Ports the ten spec validation tests from the reference implementation,
adapted to this pipeline's modules, plus the raw-vs-aligned and
no-routing regression tests and the capability diagnostic test.
"""

import hashlib
import json

import numpy as np
import pytest
from sklearn.metrics import mutual_info_score

import nonlinear_data
import partitioning
import recovery_metrics
import run_nonlinear_experiments as runner
import corex_fitting
import structure_diagnostics
from structure_diagnostics import total_correlation_bits


def test_variable_single_group_membership_per_region():
    # 1. Every signal variable appears in exactly one true group per region.
    for groups in (nonlinear_data.REGION0_GROUPS,
                   nonlinear_data.REGION1_GROUPS):
        members = [m for g in groups.values() for m in g]
        assert sorted(members) == list(range(1, nonlinear_data.N_SIGNAL + 1))
        for first in groups.values():
            for second in groups.values():
                if first != second:
                    assert set(first) & set(second) == set()


def test_no_group_identical_across_regions():
    # 2. No true group is identical across the two regions.
    region0 = set(map(frozenset, nonlinear_data.REGION0_GROUPS.values()))
    region1 = set(map(frozenset, nonlinear_data.REGION1_GROUPS.values()))
    assert len(region0 & region1) == 0


def test_parity_equation_holds_before_noise():
    # 3. Parity equations hold exactly at eta = 0; flips occur at eta > 0.
    rng = np.random.default_rng(1)
    block = nonlinear_data.parity_group(rng, 4, 1000, eta=0.0)
    expected = block[:, :3].sum(axis=1) % 2
    assert np.array_equal(block[:, 3], expected)

    noisy = nonlinear_data.parity_group(rng, 4, 1000, eta=0.15)
    expected_noisy = noisy[:, :3].sum(axis=1) % 2
    assert not np.array_equal(noisy[:, 3], expected_noisy)
    flip_rate = (noisy[:, 3] != expected_noisy).mean()
    assert 0.10 < flip_rate < 0.20


def test_binary_marginals_bernoulli_half():
    # 4. Empirical binary marginals are approximately Bernoulli(1/2).
    rng = np.random.default_rng(2)
    block = nonlinear_data.parity_group(rng, 4, 50_000, eta=0.0)
    for column in range(4):
        assert abs(block[:, column].mean() - 0.5) < 0.01


def test_parity_pairwise_mi_near_zero():
    # 5. Pairwise parity MI within tolerance of zero at eta = 0.
    rng = np.random.default_rng(3)
    block = nonlinear_data.parity_group(rng, 3, 20_000, eta=0.0)
    for i in range(3):
        for j in range(i + 1, 3):
            mi = mutual_info_score(block[:, i], block[:, j])
            assert mi < 0.005  # estimator-bias tolerance at n = 20k


def test_permuting_one_member_removes_group_dependence():
    # 6. Permuting one member removes group TC while marginals are preserved.
    rng = np.random.default_rng(4)
    block = nonlinear_data.parity_group(rng, 3, 20_000, eta=0.0)
    tc_bits = total_correlation_bits(block.astype(np.int64))
    assert tc_bits == pytest.approx(1.0, abs=0.01)
    permuted = block.copy()
    permuted[:, 0] = rng.permutation(permuted[:, 0])
    permuted_bits = total_correlation_bits(permuted.astype(np.int64))
    assert permuted_bits == pytest.approx(0.0, abs=0.01)


def test_routing_nuisance_excluded_from_ground_truth():
    # 7. Routing and nuisance variables are excluded from ground truth.
    masks = nonlinear_data.true_group_masks(n_nuisance=6)
    for (region, group), mask in masks.items():
        assert mask.shape == (14 + 2 + 6,)
        assert mask[14:16].sum() == 0  # routing zero
        assert mask[16:].sum() == 0    # nuisance zero
        assert mask[:14].sum() > 0     # signal members present


def test_sine_sum_marginal_independence_and_group_dependence():
    # Continuous analogue: near-zero pairwise dependence after the fixed
    # 10-bin rule, but clearly nonzero group TC that permution destroys.
    rng = np.random.default_rng(5)
    block = nonlinear_data.sine_sum_group(rng, 4, 100_000, sigma=0.05)
    offdiagonal = np.abs(np.corrcoef(block.T) - np.eye(4))
    assert offdiagonal.max() < 0.05

    disc, _ = corex_fitting.discretize_columns(block, 10)
    for i in range(4):
        for j in range(i + 1, 4):
            mi = mutual_info_score(disc[:, i], disc[:, j])
            assert mi < 0.01
    tc_bits = total_correlation_bits(disc)
    assert tc_bits > 0.1
    permuted = disc.copy()
    permuted[:, 0] = rng.permutation(permuted[:, 0])
    assert total_correlation_bits(permuted) < 0.6 * tc_bits


def test_discrete_continuous_schema_compatible():
    # 8. Discrete and continuous result rows share the core schema keys.
    core = {
        'generator', 'n_per_region', 'alpha', 'noise', 'n_nuisance',
        'root_seed', 'partition_method', 'estimator', 'region',
        'partition_draw', 'factor_tc', 'cosine', 'auprc',
    }
    run_core = {
        'generator', 'n_per_region', 'alpha', 'noise', 'n_nuisance',
        'root_seed', 'partition_method', 'estimator', 'region',
        'partition_draw',
    }
    assert core <= set(runner.MATCH_COLUMNS)
    assert run_core <= set(runner.RUN_COLUMNS)


def test_hungarian_no_factor_reuse_nonlinear():
    # 9. Hungarian matching never assigns a learned factor twice.
    rng = np.random.default_rng(6)
    association = np.abs(rng.normal(size=(4, 16)))
    truth = np.zeros((4, 16))
    for group, members in nonlinear_data.REGION0_GROUPS.items():
        truth[group - 1, [m - 1 for m in members]] = 1
    result = recovery_metrics.match_factors_to_groups(association, truth)
    learned = [
        assignment['learned_factor']
        for assignment in result['assignments']
    ]
    assert len(set(learned)) == len(learned)


def test_generators_use_derived_seeds():
    # 10. Paired replicates: identical seeds -> identical observations.
    first = nonlinear_data.generate_discrete_dataset(100, 0.5, 0.05, 14, 77)
    second = nonlinear_data.generate_discrete_dataset(100, 0.5, 0.05, 14, 77)
    assert np.array_equal(first['X'], second['X'])
    third = nonlinear_data.generate_discrete_dataset(100, 0.5, 0.05, 14, 78)
    assert not np.array_equal(first['X'], third['X'])


def test_routing_mean_distance():
    # Routing-mean distance is 4 * alpha.
    for alpha in (0.0, 0.5, 1.0):
        nu_0, nu_1 = nonlinear_data.routing_means(alpha)
        assert np.linalg.norm(nu_1 - nu_0) == pytest.approx(4 * alpha)


def test_partition_masks_select_true_region():
    # Regression (raw-vs-aligned): partition masks must select the region's
    # own observations even when the estimated labels are swapped, and the
    # estimator cells must use RAW cluster ids.
    labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])  # swapped: raw 1 = region 0
    region = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    partition = partitioning.partition_data(
        np.arange(16, dtype=float).reshape(8, 2),
        region,
        method='oracle',
        random_state=0,
    )
    assert np.array_equal(
        np.flatnonzero(partition['indexes']['0']),
        np.flatnonzero(region == 0),
    )

    fake_partition = {
        'predicted_labels': labels,
        'ari': 1.0, 'purity': 1.0, 'mapped_accuracy': 1.0,
    }
    cells = runner.estimated_cells(fake_partition, region, method='raw_kmeans')
    clusters = {cell['region_id'] for cell in cells}
    assert clusters == {0, 1}
    for cell in cells:
        assert cell['method'] == 'raw_kmeans'
        assert cell['name'] == f"cluster{cell['region_id']}"
        assert cell['draw'] == 0
        assert all(
            (region == served_region)[cell['rows']].all()
            for served_region in cell['regions']
        )
    # a pure label swap: the Hungarian and max-overlap choices coincide
    assert all(
        cell['fields']['max_overlap_matches_hungarian'] for cell in cells)
    # Hungarian primary cluster selection: every region is served by a
    # distinct raw cluster even when unconstrained max-overlap would pick
    # the same one for both (the retired rule survives as two diagnostic
    # fields); a collapsed single-cluster partition can never reach this
    # code path (match_partition_to_truth raises earlier), so it is not
    # simulated here.
    tied_labels = np.array([0, 0, 1, 1, 0, 0, 1, 1])
    tied_partition = {
        'predicted_labels': tied_labels,
        'ari': 0.0, 'purity': 0.5, 'mapped_accuracy': 0.5,
    }
    cells = runner.estimated_cells(
        tied_partition, region, method='raw_kmeans')
    assert len(cells) == 2
    assert sorted(
        (cell['region_id'], cell['regions']) for cell in cells
    ) == [(0, (0,)), (1, (1,))]
    assert cells[0]['fields']['max_overlap_same_cluster'] is True
    assert cells[0]['fields']['max_overlap_matches_hungarian'] is False
    assert cells[1]['fields'] is cells[0]['fields'] or (
        cells[1]['fields'] == cells[0]['fields'])
    # all cell builders share the uniform shape
    dataset_labels = region
    for builder_cells in (
        runner.global_cells({'region': dataset_labels}),
        runner.oracle_cells({'region': dataset_labels}),
    ):
        for cell in builder_cells:
            assert set(cell) == {
                'method', 'draw', 'name', 'region_id', 'rows', 'regions',
                'fields',
            }


def test_no_routing_scores_align_to_full_width():
    # Regression (no_routing mismatch): association scores from a fit with
    # routing columns dropped are re-expanded to full width before scoring;
    # routing FPs are None and nuisance leakage is still detected while the
    # matched factor set covers every true group.
    dataset = nonlinear_data.generate_discrete_dataset(50, 0.5, 0.05, 4, 9)
    routing_idx = np.asarray(dataset['routing_idx'], dtype=int)
    p_total = dataset['X'].shape[1]  # 14 signal + 2 routing + 4 nuisance

    # narrow width = full width minus the two dropped routing columns, so
    # narrow column 14 is original nuisance column 16
    narrow = np.zeros((4, p_total - 2))
    narrow[0, [0, 1, 2]] = 1.0      # region 0, group 1 = {1,2,3}
    narrow[1, [3, 4, 5]] = 1.0      # region 0, group 2 = {4,5,6}
    narrow[2, [6, 7, 8]] = 1.0      # group 3 = {7,8,9,10}, one member short
    narrow[3, [14, 15]] = 5.0       # strong nuisance leakage
    association = runner.expand_to_full_width(narrow, routing_idx, p_total)
    assert association.shape == (4, p_total)
    assert association[3, list(routing_idx)].sum() == 0

    fit = {
        'model': None, 'linear': False, 'association': association,
        'tcs': np.ones(4), 'dropped': routing_idx,
        'true_groups': dataset['true_groups'],
    }
    rows = runner.score_region_fit(
        {'generator': 'parity', 'partition_method': 'phate',
         'partition_draw': 0, 'estimator': 'bio',
         'corex_input': 'no_routing'},
        fit, region=0,
    )
    assert len(rows) == 4
    matched_groups = {row['true_group'] for row in rows}
    assert matched_groups == {1, 2, 3, 4}
    matched_factors = {row['learned_factor'] for row in rows}
    assert len(matched_factors) == 4
    for row in rows:
        assert row['routing_fp'] is None
        assert row['routing_fp_support'] is None
        assert row['region'] == 0
    leakage_rows = [row for row in rows if row['learned_factor'] == 3]
    assert leakage_rows and leakage_rows[0]['nuisance_fp'] > 0.0


def test_structure_diagnostics_parity():
    diagnostics = structure_diagnostics.structure_diagnostics(
        'parity', 3, 0.0, n=100_000, seed=11)
    assert diagnostics['max_pairwise_mi_bits'] < 0.01
    assert diagnostics['group_tc_bits'] == pytest.approx(1.0, abs=0.02)
    assert diagnostics['analytic_tc_bits'] == pytest.approx(1.0)
    assert diagnostics['permuted_one_tc_mean_bits'] < 0.01
    assert diagnostics['permuted_all_tc_mean_bits'] < 0.01
    assert diagnostics['n_permutations'] == (
        structure_diagnostics.PERMUTATION_NULL_REPEATS)


def test_structure_diagnostics_sine_sum():
    diagnostics = structure_diagnostics.structure_diagnostics(
        'sine_sum', 4, 0.05, n=100_000, seed=12)
    assert diagnostics['max_abs_pearson'] < 0.05
    assert diagnostics['max_pairwise_mi_bits'] < 0.02
    assert diagnostics['group_tc_bits'] > 0.1
    assert np.isnan(diagnostics['analytic_tc_bits'])
    assert diagnostics['permuted_one_tc_mean_bits'] < (
        0.6 * diagnostics['group_tc_bits'])


def test_product_structure_alive_under_one_column_permutation():
    # Product keeps pairwise structure when one member is permuted (its
    # remaining generators still share marginals pairwise), so its proper
    # null is the all-column permutation, which drops the TC to ~0.
    diagnostics = structure_diagnostics.structure_diagnostics(
        'product', 4, 0.20, n=50_000, seed=13)
    assert diagnostics['permuted_all_tc_mean_bits'] < (
        0.5 * diagnostics['permuted_one_tc_mean_bits'])
    # the one-column null keeps the pairwise-alive TC partly intact
    assert diagnostics['permuted_one_tc_mean_bits'] > 0.1


def test_capability_diagnostic_std_finds_nothing_seeded_reaches_parity():
    # Capability statement: standard EM inits find TC ~ 0 on pure parity
    # while the atom-seeded init reaches the near-parity TC, at the SAME
    # sufficient cardinality (2^d states) for both inits — so the failure
    # is optimization, not expressiveness or capacity. Seeded rows never
    # enter recovery summaries.
    row = structure_diagnostics.capability_diagnostic(3, 0.0, 20_000, seed=1)
    standard_tc = row['std_init_tc_bits']
    seeded_tc = row['seeded_init_tc_bits']
    assert standard_tc < 0.05
    assert seeded_tc > 0.5  # 1 bit target; tolerance for EM drift
    assert row['uses_ground_truth_init'] is True
    assert row['true_tc_bits'] == 1.0
    assert row['dim_hidden_states'] == 8
    assert row['analytic_tc_bits'] == pytest.approx(1.0)


def test_degenerate_factor_neutral_tie_handling():
    # The old lower-index stable tie-break handed an all-zero factor a
    # perfect top-d result on the leading true group; neutral (seeded)
    # ties make its top-d a reproducible, layout-free subset instead.
    groups = np.zeros((4, 15))
    groups[0, [0, 1, 2]] = 1.0   # leading block: legacy F1 = 1.0
    groups[1, [3, 4, 5]] = 1.0
    groups[2, [6, 7, 8, 9]] = 1.0
    groups[3, [10, 11, 12, 13]] = 1.0
    zero_factors = np.zeros((4, 15))
    legacy = recovery_metrics.match_factors_to_groups(
        zero_factors, groups, tie_seed=None)
    assert legacy['matched_top_d_f1'] == 0.25  # exactly the leading group
    neutral = recovery_metrics.match_factors_to_groups(zero_factors, groups)
    assert all(
        assignment['top_d_f1'] < 1.0
        for assignment in neutral['assignments']
    )
    repeat = recovery_metrics.match_factors_to_groups(zero_factors, groups)
    assert repeat['matched_top_d_f1'] == neutral['matched_top_d_f1']


def test_capacity_condition_rows_and_seed_reuse():
    # The capacity grid runs end to end at tiny scale, records every fit
    # (kind/budget), and reuses the primary grid's global M=4 seed so the
    # global fit matches the primary grid's exactly.
    import capacity_experiment as capacity
    condition = {
        'generator': 'parity', 'n_per_region': 100, 'alpha': 0.5,
        'noise': 0.05, 'n_nuisance': 0, 'root_seed': 2000,
    }
    tiny_defaults = {
        **runner.DEFAULTS, 'bio_max_iter': 50, 'bio_n_repeat': 1,
        'linear_max_iter': 500,
    }
    original_budgets = capacity.BUDGETS
    capacity.BUDGETS = (2, 4)
    try:
        fit_rows, match_rows = capacity.run_capacity_condition(
            condition, capacity.CAPACITY_CONFIG, tiny_defaults)
    finally:
        capacity.BUDGETS = original_budgets
    kinds = {row['capacity_kind'] for row in fit_rows}
    assert kinds == {'global', 'local_pooled'}
    global_fits = [row for row in fit_rows if row['capacity_kind'] == 'global']
    assert {row['budget'] for row in global_fits} == {2, 4}
    pooled_fits = [
        row for row in fit_rows
        if row['capacity_kind'] == 'local_pooled'
        and row['estimator'] == 'bio']
    assert len(pooled_fits) == 2  # one per region
    # matches cover both regions' groups under every candidate
    match_kinds = {row['capacity_kind'] for row in match_rows}
    assert match_kinds == {'global', 'local_pooled'}
    for row in match_rows:
        assert row['region'] in (0, 1)
        assert 1 <= row['true_group'] <= runner.nonlinear_data.N_GROUPS
    # the pooled local fits reuse the primary grid's oracle seeds
    assert all(
        row['fit_status'] == 'ok' for row in pooled_fits)
    # seed identity: the M=4 global seed equals the primary grid's
    bio_global_b4 = [
        row for row in global_fits
        if row['estimator'] == 'bio' and row['budget'] == 4]
    assert bio_global_b4[0]['model_seed'] == runner.derived_seed(
        2000, 'corex', 0, 'global', 'global', 'bio', 'with_routing')


def test_condition_grids_match_spec():
    # Core grid sizes: discrete 2n x 3a x 3eta x 16 reps; continuous
    # 2n x 3a x 3sigma x 16 reps; smoke = 2 conditions.
    assert len(runner.discrete_core_conditions(16)) == 288
    assert len(runner.continuous_core_conditions(16)) == 288
    smoke = runner.smoke_conditions(1)
    generators = {condition['generator'] for condition in smoke}
    assert generators == {'parity', 'sine_sum'}
    root_seeds = {condition['root_seed'] for condition
                  in runner.discrete_core_conditions(16)}
    assert root_seeds == set(range(2000, 2016))
    root_seeds = {condition['root_seed'] for condition
                  in runner.continuous_core_conditions(16)}
    assert root_seeds == set(range(3000, 3016))


def test_derived_seed_is_stable_and_sensitive():
    # Spec 00: derived seeds must be process-independent (no builtin hash)
    # and must change when any condition component changes.
    first = runner.derived_seed(2000, 'data')
    second = runner.derived_seed(2000, 'data')
    other = runner.derived_seed(2001, 'data')
    assert first == second
    assert first != other
    payload = json.dumps([2000, 'data']).encode()
    assert first == int.from_bytes(
        hashlib.sha256(payload).digest()[:4], 'big')


def test_save_condition_metadata_is_serializable(tmp_path):
    # Regression (metadata circular reference): the experiment config holds
    # the builder function; the metadata JSON must survive serialization.
    condition = runner.smoke_conditions(1)[0]
    config = dict(runner.EXPERIMENTS['smoke'])
    paths = runner.condition_output_paths(tmp_path, 'smoke', condition)
    runner.save_condition_metadata(condition, config, runner.DEFAULTS, paths)
    metadata_path = paths['runs'].with_suffix('.meta.json')
    restored = json.loads(metadata_path.read_text())
    assert restored['condition']['root_seed'] == condition['root_seed']
    assert 'builder' not in restored['config']


def test_recovery_metrics_match_on_cosine():
    # The matching objective is cosine similarity per the spec; an exact
    # factor/group correspondence must yield cosine 1.0 and identity
    # assignment.
    groups = np.zeros((2, 6))
    groups[0, [0, 1]] = 1.0
    groups[1, [2, 3, 4]] = 1.0
    factors = groups.copy()
    result = recovery_metrics.match_factors_to_groups(factors, groups)
    assert result['matched_cosine_similarity'] == pytest.approx(1.0)
    assert result['factor_assignment'] == '0:0;1:1'


def test_match_argument_order_learned_first():
    # Regression (argument order): the first argument must be the learned
    # factor scores, second the true groups (reference convention). With
    # swapped labels the matcher would report group/factor labels from the
    # wrong matrix and compute AUPRC against meaningless labels.
    groups = np.zeros((2, 6))
    groups[0, [0, 1]] = 1.0
    groups[1, [2, 3, 4]] = 1.0
    factors = np.zeros((2, 6))
    factors[0, [0, 1]] = 2.0          # exact copy of group 0, scaled
    factors[1, [2, 3, 4, 5]] = 1.0    # group 1 plus one false positive
    result = recovery_metrics.match_factors_to_groups(factors, groups)
    assignment = result['assignments'][0]
    assert assignment['true_group'] == 0
    assert assignment['learned_factor'] == 0
    assert assignment['cosine_similarity'] == pytest.approx(1.0)
    assert assignment['average_precision'] == pytest.approx(1.0)
    assert result['factor_assignment'] == '0:0;1:1'
    second = result['assignments'][1]
    assert second['true_group'] == 1 and second['learned_factor'] == 1


def test_module_validators_pass():
    assert partitioning.validate_partitioning()['collapsed_partition_checked']
    assert corex_fitting.validate_corex_fitting()
    assert recovery_metrics.validate_recovery_metrics()


def test_shard_specs_parse_and_validate():
    assert runner.parse_shard('0/4') == (0, 4)
    with pytest.raises(ValueError):
        runner.parse_shard('4/4')
    with pytest.raises(ValueError):
        runner.parse_shard('-1/4')
    with pytest.raises(ValueError):
        runner.parse_shard('3')
    with pytest.raises(ValueError):
        runner.parse_shard('0/0')


def test_shard_partition_is_exact_disjoint_cover():
    # Sharding is an order-preserving stride selection: the union of all
    # N shards equals the unfiltered condition list exactly (as a
    # multiset — shard k keeps every list element whose index % N == k),
    # so parallel shards reproduce the serial run's outputs exactly.
    conditions = runner.CONDITION_BUILDERS['discrete_core'](3)
    n_shards = 7  # deliberately not a divisor of the condition count
    shards = [conditions[k::n_shards] for k in range(n_shards)]
    rebuilt = [json.dumps(c, sort_keys=True) for shard in shards
               for c in shard]
    reference = [json.dumps(c, sort_keys=True) for c in conditions]
    assert len(rebuilt) == len(reference)
    assert sorted(rebuilt) == sorted(reference)
    # no condition is attempted twice across shards
    assert len(set(rebuilt)) == len(rebuilt)