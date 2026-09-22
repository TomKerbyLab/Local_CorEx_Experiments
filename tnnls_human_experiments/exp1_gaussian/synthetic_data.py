"""Synthetic Local CorEx experiments, stage 5.

This staged revision now includes the Gaussian data-generating process,
partition controls and diagnostics, and one-to-one factor recovery metrics.
The three covariance conditions are represented as follows:

* homogeneous: ``same_dist=True, disjoint=False``
* nondisjoint: ``same_dist=False, disjoint=False``
* disjoint: ``same_dist=False, disjoint=True``

This stage adds the nuisance-dimension stress test. Independent Gaussian
nuisance variables are appended after the 25 signal variables, and recovery
metrics explicitly measure how much learned factor weight leaks onto them.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phate
from local_corex import BioCorex, LinearCorex
from sklearn.metrics import adjusted_rand_score, average_precision_score
from sklearn.cluster import KMeans
from scipy.spatial.distance import cosine
from scipy.optimize import linear_sum_assignment
from time import time

ALPHA_LIST = [0.0, 0.5, 1.0]
NUM_FACTOR_LIST = [x for x in range(3, 8)]
N_VARIABLES = 25
WITHIN_GROUP_CORRELATION = 0.7

RECOVERY_COLUMNS = [
    'matched_average_precision',
    'matched_adjusted_average_precision',
    'matched_cosine_distance',
    'matched_cosine_similarity',
    'matched_top_k_f1',
    'matched_nuisance_score_fraction',
    'matched_top_k_nuisance_fraction',
    'true_group_recovery_average_precision',
    'learned_factor_precision_average_precision',
    'balanced_average_precision',
    'true_group_recovery_cosine_similarity',
    'learned_factor_precision_cosine_similarity',
    'balanced_cosine_similarity',
    'num_matched_factors',
    'num_unmatched_true_groups',
    'num_unmatched_learned_factors',
    'group_best_average_precision',
    'group_best_cosine_distance',
    'factor_best_average_precision',
    'factor_best_cosine_distance',
    'factor_assignment',
]

# The marginal variances and mean vectors come from the original experiment.
# Keeping them fixed across covariance regimes means that the regimes differ
# only in which variables belong to the same dependence group.
VARIABLE_VARIANCES = np.repeat([7.0, 3.0, 10.0, 5.0, 8.0], 5)
CLUSTER_MEAN_1 = np.repeat([5.0, 10.0, 1.0, 10.0, 3.0], 5)
CLUSTER_MEAN_2 = np.repeat([2.5, 6.0, 3.0, 1.0, 10.0], 5)

# Every regime has five disjoint groups of five variables. "Nondisjoint"
# describes overlap between the groupings in the two regions, not overlap
# between factors within one region.
CLUSTER_1_GROUPS = (
    (0, 1, 2, 3, 4),
    (5, 6, 7, 8, 9),
    (10, 11, 12, 13, 14),
    (15, 16, 17, 18, 19),
    (20, 21, 22, 23, 24),
)

# Each cluster-2 group retains three members of one cluster-1 group and takes
# two members from the preceding group. Thus the structures remain related but
# no cluster-2 group is identical to a cluster-1 group.
NONDISJOINT_CLUSTER_2_GROUPS = (
    (0, 1, 2, 23, 24),
    (3, 4, 5, 6, 7),
    (8, 9, 10, 11, 12),
    (13, 14, 15, 16, 17),
    (18, 19, 20, 21, 22),
)

# Each cluster-2 group contains one member from every cluster-1 group.
DISJOINT_CLUSTER_2_GROUPS = (
    (0, 5, 10, 15, 20),
    (1, 6, 11, 16, 21),
    (2, 7, 12, 17, 22),
    (3, 8, 13, 18, 23),
    (4, 9, 14, 19, 24),
)


def validate_group_definition(groups, n_variables=N_VARIABLES):
    """Check that groups form a partition of all variables."""
    flattened = [variable for group in groups for variable in group]
    if len(groups) != 5:
        raise ValueError(f"Expected 5 groups, found {len(groups)}.")
    if any(len(group) != 5 for group in groups):
        raise ValueError("Every group must contain exactly 5 variables.")
    if sorted(flattened) != list(range(n_variables)):
        raise ValueError("Groups must contain every variable exactly once.")


def create_covariance_matrix(groups, visualize=False):
    """Create a covariance matrix with correlation 0.7 within each group."""
    validate_group_definition(groups)
    mat = np.diag(VARIABLE_VARIANCES.copy())

    for group in groups:
        for i in group:
            for j in group:
                if i != j:
                    mat[i, j] = (
                        WITHIN_GROUP_CORRELATION
                        * np.sqrt(VARIABLE_VARIANCES[i] * VARIABLE_VARIANCES[j])
                    )

    if visualize:
        plt.imshow(mat)
    return mat


def create_cov_mat_clust_1(visualize=False):
    return create_covariance_matrix(CLUSTER_1_GROUPS, visualize=visualize)


def create_nondisjoint_cov_mat_clust_2(visualize=False):
    return create_covariance_matrix(
        NONDISJOINT_CLUSTER_2_GROUPS,
        visualize=visualize,
    )


def create_disjoint_cov_mat_clust_2(visualize=False):
    return create_covariance_matrix(
        DISJOINT_CLUSTER_2_GROUPS,
        visualize=visualize,
    )


def create_cluster_2_mean(alpha):
    """Interpolate from cluster 1's mean to cluster 2's mean."""
    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1.")
    mean_difference = CLUSTER_MEAN_2 - CLUSTER_MEAN_1
    return CLUSTER_MEAN_1 + alpha * mean_difference


def groups_to_indicator_vectors(groups, n_variables=N_VARIABLES):
    """Convert explicit variable groups to binary ground-truth vectors."""
    indicators = []
    for group in groups:
        indicator = np.zeros(n_variables)
        indicator[list(group)] = 1.0
        indicators.append(indicator)
    return indicators


def get_true_groups(cluster, disjoint=False, same_dist=False,
                    nuisance_dimensions=0):
    """Return explicit factor supports for one population."""
    nuisance_variances(nuisance_dimensions)
    if same_dist and disjoint:
        raise ValueError("same_dist and disjoint cannot both be True.")
    if cluster not in {'clust_1', 'clust_2'}:
        raise ValueError("cluster must be 'clust_1' or 'clust_2'.")
    if cluster == 'clust_1' or same_dist:
        groups = CLUSTER_1_GROUPS
    elif disjoint:
        groups = DISJOINT_CLUSTER_2_GROUPS
    else:
        groups = NONDISJOINT_CLUSTER_2_GROUPS
    return groups_to_indicator_vectors(
        groups,
        n_variables=N_VARIABLES + nuisance_dimensions,
    )


def nuisance_variances(nuisance_dimensions):
    """Return sensible marginal variances for independent nuisance variables."""
    if not isinstance(nuisance_dimensions, (int, np.integer)):
        raise TypeError("nuisance_dimensions must be an integer.")
    if nuisance_dimensions < 0:
        raise ValueError("nuisance_dimensions cannot be negative.")
    return np.resize(VARIABLE_VARIANCES, nuisance_dimensions)


def append_nuisance_variables(cluster_1, cluster_2, nuisance_dimensions, rng):
    """Append independent, zero-mean nuisance variables to both populations.

    Nuisance variables have the same distribution in both populations. Their
    variances cycle through the signal-variable variances so that their scale
    is realistic without adding covariance structure or cluster information.
    """
    if nuisance_dimensions == 0:
        return cluster_1, cluster_2

    standard_deviations = np.sqrt(nuisance_variances(nuisance_dimensions))
    nuisance_1 = rng.normal(
        loc=0.0,
        scale=standard_deviations,
        size=(len(cluster_1), nuisance_dimensions),
    )
    nuisance_2 = rng.normal(
        loc=0.0,
        scale=standard_deviations,
        size=(len(cluster_2), nuisance_dimensions),
    )
    return (
        np.concatenate([cluster_1, nuisance_1], axis=1),
        np.concatenate([cluster_2, nuisance_2], axis=1),
    )


def gen_clustered_sample(same_dist=True, alpha=0, cluster_size=100,
                         disjoint=False, nuisance_dimensions=0,
                         random_state=None):
    """Generate two Gaussian populations with controlled local structure."""
    nuisance_variances(nuisance_dimensions)
    if same_dist and disjoint:
        raise ValueError("same_dist and disjoint cannot both be True.")
    if isinstance(random_state, np.random.Generator):
        rng = random_state
    else:
        rng = np.random.default_rng(random_state)

    mat_1 = create_cov_mat_clust_1(visualize=False)
    clust_1 = rng.multivariate_normal(
        CLUSTER_MEAN_1,
        mat_1,
        size=cluster_size,
    )
    if same_dist:
        mat_2 = mat_1
    else:
        if disjoint:
            mat_2 = create_disjoint_cov_mat_clust_2(visualize=False)
        else:
            mat_2 = create_nondisjoint_cov_mat_clust_2(visualize=False)
    clust_2 = rng.multivariate_normal(
        create_cluster_2_mean(alpha),
        mat_2,
        size=cluster_size,
    )
    return append_nuisance_variables(
        clust_1,
        clust_2,
        nuisance_dimensions,
        rng,
    )


def match_partition_to_truth(true_labels, predicted_labels):
    """Align arbitrary partition labels to the true population names.

    The Hungarian assignment maximizes the total number of correctly aligned
    observations. It changes only the names of the estimated clusters, not
    their membership.
    """
    true_labels = np.asarray(true_labels)
    predicted_labels = np.asarray(predicted_labels)
    true_names = np.unique(true_labels)
    predicted_names = np.unique(predicted_labels)

    if len(true_names) != len(predicted_names):
        raise ValueError(
            "The estimated and true partitions must have the same number "
            "of groups for one-to-one matching."
        )

    contingency = np.zeros((len(true_names), len(predicted_names)), dtype=int)
    for i, true_name in enumerate(true_names):
        for j, predicted_name in enumerate(predicted_names):
            contingency[i, j] = np.sum(
                (true_labels == true_name)
                & (predicted_labels == predicted_name)
            )

    true_indexes, predicted_indexes = linear_sum_assignment(-contingency)
    label_mapping = {
        predicted_names[predicted_index]: true_names[true_index]
        for true_index, predicted_index in zip(true_indexes, predicted_indexes)
    }
    mapped_labels = np.array([
        label_mapping[predicted_label]
        for predicted_label in predicted_labels
    ])

    aligned_confusion = np.zeros((len(true_names), len(true_names)), dtype=int)
    for i, true_name in enumerate(true_names):
        for j, mapped_name in enumerate(true_names):
            aligned_confusion[i, j] = np.sum(
                (true_labels == true_name)
                & (mapped_labels == mapped_name)
            )

    return mapped_labels, true_names, contingency, aligned_confusion


def partition_data(data, true_labels, method='phate', random_state=42):
    """Partition observations and calculate ground-truth diagnostics.

    ``oracle`` returns the true regions. ``random`` preserves the true region
    sizes but randomly assigns observations. ``phate`` embeds the observations
    and applies K-means. Estimated labels are aligned to the true labels only
    after the partition has been created.
    """
    true_labels = np.asarray(true_labels)
    true_names, true_counts = np.unique(true_labels, return_counts=True)
    n_clusters = len(true_names)
    rng = np.random.default_rng(random_state)
    start_time = time()

    if method == 'oracle':
        predicted_labels = true_labels.copy()
    elif method == 'random':
        predicted_labels = np.concatenate([
            np.repeat(cluster_number, cluster_size)
            for cluster_number, cluster_size in enumerate(true_counts)
        ])
        rng.shuffle(predicted_labels)
    elif method == 'phate':
        phate_seed = int(rng.integers(0, 2**31 - 1))
        kmeans_seed = int(rng.integers(0, 2**31 - 1))
        phate_operator = phate.PHATE(
            n_components=5,
            n_jobs=4,
            random_state=phate_seed,
        )
        embedded_data = phate_operator.fit_transform(data)
        predicted_labels = KMeans(
            n_clusters=n_clusters,
            random_state=kmeans_seed,
            n_init='auto',
        ).fit_predict(embedded_data)
    else:
        raise ValueError(
            f"Unknown partition method '{method}'. "
            "Choose 'oracle', 'phate', or 'random'."
        )

    partition_time = time() - start_time
    mapped_labels, label_order, contingency, aligned_confusion = (
        match_partition_to_truth(true_labels, predicted_labels)
    )
    partition_purity = contingency.max(axis=0).sum() / len(true_labels)
    mapped_accuracy = np.mean(mapped_labels == true_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)

    indexes = {
        str(label): mapped_labels == label
        for label in label_order
    }
    estimated_region_sizes = {
        str(label): int(np.sum(mapped_labels == label))
        for label in label_order
    }
    region_purity = {
        str(label): float(np.mean(true_labels[indexes[str(label)]] == label))
        for label in label_order
    }

    return {
        'method': method,
        'indexes': indexes,
        'predicted_labels': predicted_labels,
        'mapped_labels': mapped_labels,
        'label_order': [str(label) for label in label_order],
        'confusion_matrix': aligned_confusion,
        'ari': float(ari),
        'purity': float(partition_purity),
        'mapped_accuracy': float(mapped_accuracy),
        'estimated_region_sizes': estimated_region_sizes,
        'region_purity': region_purity,
        'partition_time': partition_time,
    }


def generate_and_partition_data(alpha, cluster_size, disjoint=False,
                                same_dist=False, partition_method='phate',
                                nuisance_dimensions=0,
                                data_random_state=None,
                                partition_random_state=42):
    clust_1, clust_2 = gen_clustered_sample(
        same_dist=same_dist,
        alpha=alpha,
        cluster_size=cluster_size,
        disjoint=disjoint,
        nuisance_dimensions=nuisance_dimensions,
        random_state=data_random_state,
    )
    full_clust = np.concatenate([clust_1, clust_2], axis=0)
    labels = np.concatenate([
        np.repeat('clust_1', cluster_size),
        np.repeat('clust_2', cluster_size),
    ])
    partition = partition_data(
        full_clust,
        labels,
        method=partition_method,
        random_state=partition_random_state,
    )
    return full_clust, labels, partition


def validate_gaussian_generator():
    """Run fast checks on the data-generating process.

    This function does not fit PHATE, K-means, or CorEx. It verifies only the
    mathematical definitions and reproducibility of the Gaussian generator.
    It raises an AssertionError if a check fails and otherwise returns a short
    summary that is convenient to print in an interactive session.
    """
    group_definitions = {
        'cluster_1': CLUSTER_1_GROUPS,
        'nondisjoint_cluster_2': NONDISJOINT_CLUSTER_2_GROUPS,
        'disjoint_cluster_2': DISJOINT_CLUSTER_2_GROUPS,
    }
    covariance_matrices = {
        'cluster_1': create_cov_mat_clust_1(),
        'nondisjoint_cluster_2': create_nondisjoint_cov_mat_clust_2(),
        'disjoint_cluster_2': create_disjoint_cov_mat_clust_2(),
    }

    minimum_eigenvalues = {}
    for name, groups in group_definitions.items():
        validate_group_definition(groups)
        mat = covariance_matrices[name]
        assert np.allclose(mat, mat.T), f'{name} covariance is not symmetric.'
        assert np.allclose(np.diag(mat), VARIABLE_VARIANCES), (
            f'{name} does not preserve the intended marginal variances.'
        )
        minimum_eigenvalue = np.linalg.eigvalsh(mat).min()
        assert minimum_eigenvalue > 0, (
            f'{name} covariance is not positive definite.'
        )
        minimum_eigenvalues[name] = float(minimum_eigenvalue)

        standard_deviations = np.sqrt(np.diag(mat))
        correlation = mat / np.outer(standard_deviations, standard_deviations)
        group_membership = {
            variable: group_number
            for group_number, group in enumerate(groups)
            for variable in group
        }
        for i in range(N_VARIABLES):
            for j in range(N_VARIABLES):
                if i == j:
                    expected_correlation = 1.0
                elif group_membership[i] == group_membership[j]:
                    expected_correlation = WITHIN_GROUP_CORRELATION
                else:
                    expected_correlation = 0.0
                assert np.isclose(correlation[i, j], expected_correlation)

    assert np.allclose(create_cluster_2_mean(0.0), CLUSTER_MEAN_1)
    assert np.allclose(create_cluster_2_mean(1.0), CLUSTER_MEAN_2)
    assert np.allclose(
        create_cluster_2_mean(0.5),
        (CLUSTER_MEAN_1 + CLUSTER_MEAN_2) / 2,
    )

    first_sample = gen_clustered_sample(
        same_dist=False,
        alpha=0.5,
        cluster_size=10,
        disjoint=False,
        random_state=1234,
    )
    repeated_sample = gen_clustered_sample(
        same_dist=False,
        alpha=0.5,
        cluster_size=10,
        disjoint=False,
        random_state=1234,
    )
    assert np.array_equal(first_sample[0], repeated_sample[0])
    assert np.array_equal(first_sample[1], repeated_sample[1])

    sample_with_nuisance = gen_clustered_sample(
        same_dist=False,
        alpha=0.5,
        cluster_size=10,
        disjoint=False,
        nuisance_dimensions=25,
        random_state=1234,
    )
    assert sample_with_nuisance[0].shape == (10, 50)
    assert sample_with_nuisance[1].shape == (10, 50)
    assert np.array_equal(first_sample[0], sample_with_nuisance[0][:, :25])
    assert np.array_equal(first_sample[1], sample_with_nuisance[1][:, :25])
    assert len(get_true_groups(
        'clust_1',
        nuisance_dimensions=25,
    )[0]) == 50

    nondisjoint_overlap = [
        len(set(group_1).intersection(group_2))
        for group_1, group_2 in zip(
            CLUSTER_1_GROUPS,
            NONDISJOINT_CLUSTER_2_GROUPS,
        )
    ]
    assert nondisjoint_overlap == [3, 3, 3, 3, 3]

    disjoint_overlap = np.array([
        [
            len(set(group_1).intersection(group_2))
            for group_2 in DISJOINT_CLUSTER_2_GROUPS
        ]
        for group_1 in CLUSTER_1_GROUPS
    ])
    assert np.all(disjoint_overlap == 1)

    return {
        'minimum_eigenvalues': minimum_eigenvalues,
        'nondisjoint_corresponding_overlap': nondisjoint_overlap,
        'disjoint_pairwise_overlap': disjoint_overlap.tolist(),
        'mean_interpolation_checked': True,
        'sampling_reproducibility_checked': True,
        'nuisance_dimensions_checked': True,
        'paired_signal_samples_checked': True,
    }


def validate_partitioning():
    """Run fast checks on label matching and non-PHATE partitions."""
    true_labels = np.array([
        'clust_1', 'clust_1', 'clust_1', 'clust_1',
        'clust_2', 'clust_2', 'clust_2', 'clust_2',
    ])
    reversed_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    mapped_labels, label_order, _, confusion = match_partition_to_truth(
        true_labels,
        reversed_labels,
    )
    assert np.array_equal(mapped_labels, true_labels)
    assert label_order.tolist() == ['clust_1', 'clust_2']
    assert np.array_equal(confusion, np.diag([4, 4]))

    dummy_data = np.arange(16).reshape(8, 2)
    oracle = partition_data(
        dummy_data,
        true_labels,
        method='oracle',
        random_state=1234,
    )
    assert oracle['ari'] == 1.0
    assert oracle['purity'] == 1.0
    assert oracle['mapped_accuracy'] == 1.0
    assert oracle['estimated_region_sizes'] == {'clust_1': 4, 'clust_2': 4}

    random_1 = partition_data(
        dummy_data,
        true_labels,
        method='random',
        random_state=1234,
    )
    random_2 = partition_data(
        dummy_data,
        true_labels,
        method='random',
        random_state=1234,
    )
    assert np.array_equal(
        random_1['predicted_labels'],
        random_2['predicted_labels'],
    )
    assert random_1['estimated_region_sizes'] == {'clust_1': 4, 'clust_2': 4}

    return {
        'label_permutation_checked': True,
        'oracle_partition_checked': True,
        'random_partition_reproducibility_checked': True,
        'random_partition_size_matching_checked': True,
    }


PARTITION_SEED_CODES = {
    'oracle': 1,
    'phate': 2,
    'random': 3,
}


def covariance_setup_name(same_dist, disjoint):
    """Return the readable name of one covariance condition."""
    if same_dist and disjoint:
        raise ValueError("same_dist and disjoint cannot both be True.")
    if same_dist:
        return 'homogeneous'
    if disjoint:
        return 'disjoint'
    return 'nondisjoint'


def deterministic_seed(*values):
    """Create one reproducible integer seed from condition identifiers."""
    return int(np.random.SeedSequence(values).generate_state(1)[0])


def fit_corex_model(data, num_factors, estimator):
    """Fit one CorEx estimator and return the fitted model and elapsed time."""
    start_time = time()
    if estimator == 'linear':
        model = LinearCorex(
            n_hidden=num_factors,
            seed=42,
            gaussianize='outliers',
        )
        linear = True
        model_seed = 42
    elif estimator == 'bio':
        model = BioCorex(
            n_hidden=num_factors,
            dim_hidden=2,
            marginal_description='gaussian',
            smooth_marginals=False,
        )
        linear = False
        model_seed = np.nan
    else:
        raise ValueError("estimator must be 'linear' or 'bio'.")
    model.fit_transform(data)
    return model, linear, model_seed, time() - start_time


def recovery_result_fields(recovery):
    """Convert recovery results to named output fields."""
    values = factor_recovery_output_values(recovery)
    return dict(zip(RECOVERY_COLUMNS, values))


def empty_partition_fields():
    """Return partition fields for a global model that uses no partition."""
    return {
        'partition_method': 'none',
        'partition_seed': np.nan,
        'partition_ari': np.nan,
        'partition_purity': np.nan,
        'partition_accuracy': np.nan,
        'region_purity': np.nan,
        'true_1_est_1': np.nan,
        'true_1_est_2': np.nan,
        'true_2_est_1': np.nan,
        'true_2_est_2': np.nan,
        'partition_time': 0.0,
    }


def local_partition_fields(partition, cluster, partition_seed):
    """Return diagnostics for one mapped local region."""
    confusion = partition['confusion_matrix']
    return {
        'partition_method': partition['method'],
        'partition_seed': partition_seed,
        'partition_ari': partition['ari'],
        'partition_purity': partition['purity'],
        'partition_accuracy': partition['mapped_accuracy'],
        'region_purity': partition['region_purity'][cluster],
        'true_1_est_1': int(confusion[0, 0]),
        'true_1_est_2': int(confusion[0, 1]),
        'true_2_est_1': int(confusion[1, 0]),
        'true_2_est_2': int(confusion[1, 1]),
        'partition_time': partition['partition_time'],
    }


def score_fitted_model(model, linear, groups, row, verbose=False, plot=False):
    """Evaluate one fitted model against one region's true factor groups."""
    recovery = evaluate_factor_recovery(
        groups,
        model,
        linear=linear,
        verbose=verbose,
        plot=plot,
    )
    row.update(recovery_result_fields(recovery))
    return row


def run_simulation(rep,
                   cluster_size=100,
                   alphas=ALPHA_LIST,
                   num_factors=NUM_FACTOR_LIST,
                   estimators=('linear',),
                   partition_methods=('oracle', 'phate', 'random'),
                   disjoint=False,
                   same_dist=False,
                   nuisance_dimensions=0,
                   visualize_vars=False,
                   verbose=False,
                   progress=True):
    """Run all requested methods on shared datasets and partitions.

    One dataset is generated for each alpha. Every requested partition is
    computed once and reused across factor counts and estimators. Each global
    model is fit once and evaluated against both regional ground truths.
    """
    setup = covariance_setup_name(same_dist, disjoint)
    unknown_partitions = set(partition_methods) - set(PARTITION_SEED_CODES)
    if unknown_partitions:
        raise ValueError(f"Unknown partition methods: {sorted(unknown_partitions)}")
    unknown_estimators = set(estimators) - {'linear', 'bio'}
    if unknown_estimators:
        raise ValueError(f"Unknown estimators: {sorted(unknown_estimators)}")

    total_fits = (
        len(alphas)
        * len(num_factors)
        * len(estimators)
        * (1 + 2 * len(partition_methods))
    )
    completed_fits = 0
    rows = []

    for alpha in alphas:
        condition_values = (
            int(rep),
            int(cluster_size),
            int(round(alpha * 1000)),
            int(disjoint),
            int(same_dist),
        )
        data_seed = deterministic_seed(*condition_values, 0)
        generation_start = time()
        cluster_1, cluster_2 = gen_clustered_sample(
            same_dist=same_dist,
            alpha=alpha,
            cluster_size=cluster_size,
            disjoint=disjoint,
            nuisance_dimensions=nuisance_dimensions,
            random_state=data_seed,
        )
        full_data = np.concatenate([cluster_1, cluster_2], axis=0)
        true_labels = np.concatenate([
            np.repeat('clust_1', cluster_size),
            np.repeat('clust_2', cluster_size),
        ])
        generation_time = time() - generation_start

        partitions = {}
        partition_seeds = {}
        for partition_method in partition_methods:
            partition_seed = deterministic_seed(
                *condition_values,
                PARTITION_SEED_CODES[partition_method],
            )
            partition_seeds[partition_method] = partition_seed
            partitions[partition_method] = partition_data(
                full_data,
                true_labels,
                method=partition_method,
                random_state=partition_seed,
            )

        true_groups = {
            cluster: get_true_groups(
                cluster,
                disjoint=disjoint,
                same_dist=same_dist,
                nuisance_dimensions=nuisance_dimensions,
            )
            for cluster in ('clust_1', 'clust_2')
        }

        for factor_count in num_factors:
            for estimator in estimators:
                completed_fits += 1
                if progress:
                    print(
                        f"Fit {completed_fits}/{total_fits}: global {estimator}, "
                        f"alpha={alpha}, factors={factor_count}"
                    )
                model, linear, model_seed, corex_time = fit_corex_model(
                    full_data,
                    factor_count,
                    estimator,
                )
                fit_id = (
                    f"rep{rep}_n{cluster_size}_z{nuisance_dimensions}_"
                    f"a{alpha}_{setup}_"
                    f"m{factor_count}_{estimator}_global"
                )
                for cluster in ('clust_1', 'clust_2'):
                    row = {
                        'rep_num': rep,
                        'cluster_size': cluster_size,
                        'nuisance_dimensions': nuisance_dimensions,
                        'num_variables': N_VARIABLES + nuisance_dimensions,
                        'alpha': alpha,
                        'covariance_setup': setup,
                        'num_factors': factor_count,
                        'estimator': estimator,
                        'scope': 'global',
                        'groundtruth': cluster,
                        'fit_id': fit_id,
                        'data_seed': data_seed,
                        'model_seed': model_seed,
                        'model_sample_size': len(full_data),
                        'data_generation_time': generation_time,
                        'corex_time': corex_time,
                        'total_time': corex_time,
                    }
                    row.update(empty_partition_fields())
                    rows.append(score_fitted_model(
                        model,
                        linear,
                        true_groups[cluster],
                        row,
                        verbose=verbose,
                        plot=visualize_vars,
                    ))

                for partition_method in partition_methods:
                    partition = partitions[partition_method]
                    for cluster in ('clust_1', 'clust_2'):
                        completed_fits += 1
                        if progress:
                            print(
                                f"Fit {completed_fits}/{total_fits}: local {estimator}, "
                                f"partition={partition_method}, region={cluster}, "
                                f"alpha={alpha}, factors={factor_count}"
                            )
                        local_data = full_data[partition['indexes'][cluster]]
                        model, linear, model_seed, corex_time = fit_corex_model(
                            local_data,
                            factor_count,
                            estimator,
                        )
                        fit_id = (
                            f"rep{rep}_n{cluster_size}_z{nuisance_dimensions}_"
                            f"a{alpha}_{setup}_"
                            f"m{factor_count}_{estimator}_{partition_method}_{cluster}"
                        )
                        row = {
                            'rep_num': rep,
                            'cluster_size': cluster_size,
                            'nuisance_dimensions': nuisance_dimensions,
                            'num_variables': N_VARIABLES + nuisance_dimensions,
                            'alpha': alpha,
                            'covariance_setup': setup,
                            'num_factors': factor_count,
                            'estimator': estimator,
                            'scope': 'local',
                            'groundtruth': cluster,
                            'fit_id': fit_id,
                            'data_seed': data_seed,
                            'model_seed': model_seed,
                            'model_sample_size': len(local_data),
                            'data_generation_time': generation_time,
                            'corex_time': corex_time,
                            'total_time': corex_time + partition['partition_time'],
                        }
                        row.update(local_partition_fields(
                            partition,
                            cluster,
                            partition_seeds[partition_method],
                        ))
                        rows.append(score_fitted_model(
                            model,
                            linear,
                            true_groups[cluster],
                            row,
                            verbose=verbose,
                            plot=visualize_vars,
                        ))

    return pd.DataFrame(rows)

def get_factor_variable_scores(corex_model, linear=True):
    """Extract a factors-by-variables score matrix from either estimator."""
    if linear:
        scores = corex_model.moments['MI']
    else:
        scores = corex_model.mis
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("Factor-variable scores must be a two-dimensional array.")
    if not np.all(np.isfinite(scores)):
        raise ValueError("Factor-variable scores contain non-finite values.")
    return scores


def safe_cosine_distance(first, second):
    """Return worst-case distance when either vector has zero magnitude."""
    if np.linalg.norm(first) == 0 or np.linalg.norm(second) == 0:
        return 1.0
    return float(cosine(first, second))


def create_factor_score_matrices(groups, factor_scores):
    """Score every true-group and learned-factor pairing."""
    groups = np.asarray(groups, dtype=float)
    factor_scores = np.asarray(factor_scores, dtype=float)
    if groups.ndim != 2 or factor_scores.ndim != 2:
        raise ValueError("Groups and factor scores must both be matrices.")
    if groups.shape[1] != factor_scores.shape[1]:
        raise ValueError(
            "True groups and learned factors must describe the same variables."
        )

    average_precision = np.zeros((len(groups), len(factor_scores)))
    adjusted_average_precision = np.zeros_like(average_precision)
    cosine_distance = np.zeros_like(average_precision)
    for group_index, group in enumerate(groups):
        prevalence = np.mean(group)
        for factor_index, factor in enumerate(factor_scores):
            ap_score = (
                average_precision_score(group, factor)
            )
            average_precision[group_index, factor_index] = ap_score
            adjusted_average_precision[group_index, factor_index] = (
                (ap_score - prevalence) / (1.0 - prevalence)
            )
            cosine_distance[group_index, factor_index] = safe_cosine_distance(
                group,
                factor,
            )
    return average_precision, adjusted_average_precision, cosine_distance


def matched_support_diagnostics(groups, factor_scores,
                                group_indexes, factor_indexes):
    """Measure support recovery and nuisance leakage for matched pairs."""
    groups = np.asarray(groups, dtype=float)
    factor_scores = np.asarray(factor_scores, dtype=float)
    nuisance_mask = np.sum(groups, axis=0) == 0

    top_k_f1 = []
    nuisance_score_fraction = []
    top_k_nuisance_fraction = []
    for group_index, factor_index in zip(group_indexes, factor_indexes):
        group = groups[group_index]
        factor = factor_scores[factor_index]
        group_size = int(np.sum(group > 0))
        top_indexes = np.argsort(-factor, kind='stable')[:group_size]
        recovered_count = np.sum(group[top_indexes] > 0)
        top_k_f1.append(float(recovered_count / group_size))
        top_k_nuisance_fraction.append(float(
            np.mean(nuisance_mask[top_indexes])
        ))

        nonnegative_scores = np.maximum(factor, 0.0)
        total_score = np.sum(nonnegative_scores)
        if total_score == 0:
            nuisance_score_fraction.append(0.0)
        else:
            nuisance_score_fraction.append(float(
                np.sum(nonnegative_scores[nuisance_mask]) / total_score
            ))

    return {
        'top_k_f1': np.asarray(top_k_f1),
        'nuisance_score_fraction': np.asarray(nuisance_score_fraction),
        'top_k_nuisance_fraction': np.asarray(top_k_nuisance_fraction),
    }


def harmonic_mean(first, second):
    """Calculate a harmonic mean, returning zero when both inputs are zero."""
    if first + second == 0:
        return 0.0
    return 2 * first * second / (first + second)


def match_factors_to_groups(groups, factor_scores):
    """Match factors to groups once using maximum average precision.

    The resulting assignment is reused for cosine scores. Recovery divides the
    sum of matched scores by the number of true groups, so missing groups are
    penalized. Precision divides by the number of learned factors, so extra
    factors are penalized. Their harmonic mean provides a balanced summary.
    """
    (average_precision,
     adjusted_average_precision,
     cosine_distance) = create_factor_score_matrices(
        groups,
        factor_scores,
    )
    group_indexes, factor_indexes = linear_sum_assignment(-average_precision)
    matched_average_precision = average_precision[group_indexes, factor_indexes]
    matched_adjusted_average_precision = adjusted_average_precision[
        group_indexes,
        factor_indexes,
    ]
    matched_cosine_distance = cosine_distance[group_indexes, factor_indexes]
    matched_cosine_similarity = np.maximum(0.0, 1.0 - matched_cosine_distance)
    support_diagnostics = matched_support_diagnostics(
        groups,
        factor_scores,
        group_indexes,
        factor_indexes,
    )

    num_true_groups = len(groups)
    num_learned_factors = len(factor_scores)
    true_group_ap = matched_average_precision.sum() / num_true_groups
    learned_factor_ap = matched_average_precision.sum() / num_learned_factors
    true_group_cosine = matched_cosine_similarity.sum() / num_true_groups
    learned_factor_cosine = matched_cosine_similarity.sum() / num_learned_factors

    unmatched_groups = sorted(
        set(range(num_true_groups)) - set(group_indexes.tolist())
    )
    unmatched_factors = sorted(
        set(range(num_learned_factors)) - set(factor_indexes.tolist())
    )
    assignments = [
        {
            'true_group': int(group_index),
            'learned_factor': int(factor_index),
            'average_precision': float(ap_score),
            'adjusted_average_precision': float(adjusted_ap_score),
            'cosine_distance': float(cosine_score),
            'top_k_f1': float(top_k_f1),
            'nuisance_score_fraction': float(nuisance_fraction),
            'top_k_nuisance_fraction': float(top_k_nuisance),
        }
        for (group_index, factor_index, ap_score, adjusted_ap_score,
             cosine_score, top_k_f1, nuisance_fraction,
             top_k_nuisance) in zip(
            group_indexes,
            factor_indexes,
            matched_average_precision,
            matched_adjusted_average_precision,
            matched_cosine_distance,
            support_diagnostics['top_k_f1'],
            support_diagnostics['nuisance_score_fraction'],
            support_diagnostics['top_k_nuisance_fraction'],
        )
    ]

    return {
        'average_precision_matrix': average_precision,
        'adjusted_average_precision_matrix': adjusted_average_precision,
        'cosine_distance_matrix': cosine_distance,
        'assignments': assignments,
        'matched_average_precision': float(np.mean(matched_average_precision)),
        'matched_adjusted_average_precision': float(np.mean(
            matched_adjusted_average_precision
        )),
        'matched_cosine_distance': float(np.mean(matched_cosine_distance)),
        'matched_cosine_similarity': float(np.mean(matched_cosine_similarity)),
        'matched_top_k_f1': float(np.mean(support_diagnostics['top_k_f1'])),
        'matched_nuisance_score_fraction': float(np.mean(
            support_diagnostics['nuisance_score_fraction']
        )),
        'matched_top_k_nuisance_fraction': float(np.mean(
            support_diagnostics['top_k_nuisance_fraction']
        )),
        'true_group_recovery_average_precision': float(true_group_ap),
        'learned_factor_precision_average_precision': float(learned_factor_ap),
        'balanced_average_precision': float(harmonic_mean(
            true_group_ap,
            learned_factor_ap,
        )),
        'true_group_recovery_cosine_similarity': float(true_group_cosine),
        'learned_factor_precision_cosine_similarity': float(learned_factor_cosine),
        'balanced_cosine_similarity': float(harmonic_mean(
            true_group_cosine,
            learned_factor_cosine,
        )),
        'num_matched_factors': len(group_indexes),
        'unmatched_true_groups': unmatched_groups,
        'unmatched_learned_factors': unmatched_factors,
    }


def evaluate_factor_recovery(groups, corex_model, linear=True,
                             verbose=False, plot=False):
    """Calculate matched primary metrics and directional diagnostics."""
    factor_scores = get_factor_variable_scores(corex_model, linear=linear)
    matched = match_factors_to_groups(groups, factor_scores)
    average_precision = matched['average_precision_matrix']
    cosine_distance = matched['cosine_distance_matrix']

    matched['group_best_average_precision'] = float(
        np.mean(np.max(average_precision, axis=1))
    )
    matched['group_best_cosine_distance'] = float(
        np.mean(np.min(cosine_distance, axis=1))
    )
    matched['factor_best_average_precision'] = float(
        np.mean(np.max(average_precision, axis=0))
    )
    matched['factor_best_cosine_distance'] = float(
        np.mean(np.min(cosine_distance, axis=0))
    )

    if verbose:
        for assignment in matched['assignments']:
            print(
                "Group {true_group} matched factor {learned_factor}: "
                "AP={average_precision:.3f}, cosine distance={cosine_distance:.3f}"
                .format(**assignment)
            )

    if plot:
        num_matches = len(matched['assignments'])
        _, axes = plt.subplots(num_matches, 1, figsize=(8, num_matches))
        axes = np.atleast_1d(axes)
        for axis, assignment in zip(axes, matched['assignments']):
            group = groups[assignment['true_group']]
            factor = factor_scores[assignment['learned_factor']]
            group_total = np.sum(group)
            factor_total = np.sum(factor)
            normalized_group = group / group_total
            normalized_factor = (
                factor / factor_total
                if factor_total != 0
                else np.zeros_like(factor)
            )
            axis.imshow(np.vstack([normalized_group, normalized_factor]))
            axis.set_ylabel(f"G{assignment['true_group']} / F{assignment['learned_factor']}")
        plt.show()

    return matched


def factor_recovery_output_values(recovery):
    """Return recovery results in the same order as RECOVERY_COLUMNS."""
    assignment_text = ';'.join(
        f"{item['true_group']}:{item['learned_factor']}"
        for item in recovery['assignments']
    )
    return [
        recovery['matched_average_precision'],
        recovery['matched_adjusted_average_precision'],
        recovery['matched_cosine_distance'],
        recovery['matched_cosine_similarity'],
        recovery['matched_top_k_f1'],
        recovery['matched_nuisance_score_fraction'],
        recovery['matched_top_k_nuisance_fraction'],
        recovery['true_group_recovery_average_precision'],
        recovery['learned_factor_precision_average_precision'],
        recovery['balanced_average_precision'],
        recovery['true_group_recovery_cosine_similarity'],
        recovery['learned_factor_precision_cosine_similarity'],
        recovery['balanced_cosine_similarity'],
        recovery['num_matched_factors'],
        len(recovery['unmatched_true_groups']),
        len(recovery['unmatched_learned_factors']),
        recovery['group_best_average_precision'],
        recovery['group_best_cosine_distance'],
        recovery['factor_best_average_precision'],
        recovery['factor_best_cosine_distance'],
        assignment_text,
    ]


def validate_factor_matching():
    """Check matching behavior for correct and incorrect factor counts."""
    groups = np.asarray(groups_to_indicator_vectors(CLUSTER_1_GROUPS))

    perfect = match_factors_to_groups(groups, groups.copy())
    assert np.isclose(perfect['matched_average_precision'], 1.0)
    assert np.isclose(perfect['matched_adjusted_average_precision'], 1.0)
    assert np.isclose(perfect['balanced_average_precision'], 1.0)
    assert np.isclose(perfect['balanced_cosine_similarity'], 1.0)
    assert np.isclose(perfect['matched_top_k_f1'], 1.0)
    assert np.isclose(perfect['matched_nuisance_score_fraction'], 0.0)
    assert np.isclose(perfect['matched_top_k_nuisance_fraction'], 0.0)
    assert perfect['unmatched_true_groups'] == []
    assert perfect['unmatched_learned_factors'] == []

    permutation = [3, 1, 4, 0, 2]
    permuted = match_factors_to_groups(groups, groups[permutation])
    assert np.isclose(permuted['matched_average_precision'], 1.0)
    assert np.isclose(permuted['balanced_average_precision'], 1.0)

    missing = match_factors_to_groups(groups, groups[:3])
    assert np.isclose(missing['matched_average_precision'], 1.0)
    assert np.isclose(
        missing['true_group_recovery_average_precision'],
        3 / 5,
    )
    assert np.isclose(
        missing['learned_factor_precision_average_precision'],
        1.0,
    )
    assert len(missing['unmatched_true_groups']) == 2
    assert missing['unmatched_learned_factors'] == []

    extra_scores = np.vstack([groups, groups[:2]])
    extra = match_factors_to_groups(groups, extra_scores)
    assert np.isclose(
        extra['true_group_recovery_average_precision'],
        1.0,
    )
    assert np.isclose(
        extra['learned_factor_precision_average_precision'],
        5 / 7,
    )
    assert extra['unmatched_true_groups'] == []
    assert len(extra['unmatched_learned_factors']) == 2

    duplicate_scores = np.repeat(groups[[0]], 5, axis=0)
    duplicated = match_factors_to_groups(groups, duplicate_scores)
    assert duplicated['balanced_average_precision'] < 1.0
    assert duplicated['true_group_recovery_average_precision'] < 1.0

    extended_groups = np.asarray(groups_to_indicator_vectors(
        CLUSTER_1_GROUPS,
        n_variables=50,
    ))
    extended_perfect = match_factors_to_groups(
        extended_groups,
        extended_groups.copy(),
    )
    assert np.isclose(
        extended_perfect['matched_adjusted_average_precision'],
        1.0,
    )
    assert np.isclose(extended_perfect['matched_top_k_f1'], 1.0)
    assert np.isclose(
        extended_perfect['matched_nuisance_score_fraction'],
        0.0,
    )

    leaky_scores = extended_groups.copy()
    leaky_scores[:, 25:] = 0.5
    leaky = match_factors_to_groups(extended_groups, leaky_scores)
    assert np.isclose(leaky['matched_average_precision'], 1.0)
    assert leaky['matched_nuisance_score_fraction'] > 0.0
    assert np.isclose(leaky['matched_top_k_nuisance_fraction'], 0.0)

    badly_leaked_scores = extended_groups.copy()
    badly_leaked_scores[:, 25:30] = 2.0
    badly_leaked = match_factors_to_groups(
        extended_groups,
        badly_leaked_scores,
    )
    assert badly_leaked['matched_top_k_nuisance_fraction'] > 0.0
    assert badly_leaked['matched_top_k_f1'] < 1.0

    return {
        'perfect_matching_checked': True,
        'factor_permutation_checked': True,
        'missing_factor_penalty_checked': True,
        'extra_factor_penalty_checked': True,
        'duplicate_factor_penalty_checked': True,
        'adjusted_average_precision_checked': True,
        'nuisance_score_leakage_checked': True,
        'top_k_nuisance_leakage_checked': True,
    }
