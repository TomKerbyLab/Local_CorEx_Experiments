"""Gaussian data generation and ground truth for Local CorEx experiments."""

import matplotlib.pyplot as plt
import numpy as np


N_VARIABLES = 25
WITHIN_GROUP_CORRELATION = 0.7

# These marginal variances and means come from the original experiment.
VARIABLE_VARIANCES = np.repeat([7.0, 3.0, 10.0, 5.0, 8.0], 5)
CLUSTER_MEAN_1 = np.repeat([5.0, 10.0, 1.0, 10.0, 3.0], 5)
CLUSTER_MEAN_2 = np.repeat([2.5, 6.0, 3.0, 1.0, 10.0], 5)

# Each population contains five disjoint groups of five signal variables.
CLUSTER_1_GROUPS = (
    (0, 1, 2, 3, 4),
    (5, 6, 7, 8, 9),
    (10, 11, 12, 13, 14),
    (15, 16, 17, 18, 19),
    (20, 21, 22, 23, 24),
)

# Each group retains three members of one cluster-1 group and takes two
# members from the preceding group.
NONDISJOINT_CLUSTER_2_GROUPS = (
    (0, 1, 2, 23, 24),
    (3, 4, 5, 6, 7),
    (8, 9, 10, 11, 12),
    (13, 14, 15, 16, 17),
    (18, 19, 20, 21, 22),
)

# Each group contains one member from every cluster-1 group.
DISJOINT_CLUSTER_2_GROUPS = (
    (0, 5, 10, 15, 20),
    (1, 6, 11, 16, 21),
    (2, 7, 12, 17, 22),
    (3, 8, 13, 18, 23),
    (4, 9, 14, 19, 24),
)


def validate_group_definition(groups, n_variables=N_VARIABLES):
    """Check that groups form a partition of all signal variables."""
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
    matrix = np.diag(VARIABLE_VARIANCES.copy())

    for group in groups:
        for first_variable in group:
            for second_variable in group:
                if first_variable != second_variable:
                    matrix[first_variable, second_variable] = (
                        WITHIN_GROUP_CORRELATION
                        * np.sqrt(
                            VARIABLE_VARIANCES[first_variable]
                            * VARIABLE_VARIANCES[second_variable]
                        )
                    )

    if visualize:
        plt.imshow(matrix)
    return matrix


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


def nuisance_variances(nuisance_dimensions):
    """Return marginal variances for independent nuisance variables."""
    if not isinstance(nuisance_dimensions, (int, np.integer)):
        raise TypeError("nuisance_dimensions must be an integer.")
    if nuisance_dimensions < 0:
        raise ValueError("nuisance_dimensions cannot be negative.")
    return np.resize(VARIABLE_VARIANCES, nuisance_dimensions)


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


def append_nuisance_variables(cluster_1, cluster_2, nuisance_dimensions, rng):
    """Append independent, zero-mean nuisance variables to both populations.

    Nuisance variables have the same distribution in both populations. Their
    variances cycle through the signal-variable variances so their scale is
    realistic without adding covariance structure or cluster information.
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

    covariance_1 = create_cov_mat_clust_1(visualize=False)
    cluster_1 = rng.multivariate_normal(
        CLUSTER_MEAN_1,
        covariance_1,
        size=cluster_size,
    )

    if same_dist:
        covariance_2 = covariance_1
    elif disjoint:
        covariance_2 = create_disjoint_cov_mat_clust_2(visualize=False)
    else:
        covariance_2 = create_nondisjoint_cov_mat_clust_2(visualize=False)

    cluster_2 = rng.multivariate_normal(
        create_cluster_2_mean(alpha),
        covariance_2,
        size=cluster_size,
    )
    return append_nuisance_variables(
        cluster_1,
        cluster_2,
        nuisance_dimensions,
        rng,
    )


def validate_gaussian_generator():
    """Run fast mathematical and reproducibility checks on the generator."""
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
        matrix = covariance_matrices[name]
        assert np.allclose(matrix, matrix.T)
        assert np.allclose(np.diag(matrix), VARIABLE_VARIANCES)
        minimum_eigenvalue = np.linalg.eigvalsh(matrix).min()
        assert minimum_eigenvalue > 0
        minimum_eigenvalues[name] = float(minimum_eigenvalue)

        standard_deviations = np.sqrt(np.diag(matrix))
        correlation = matrix / np.outer(
            standard_deviations,
            standard_deviations,
        )
        memberships = {
            variable: group_number
            for group_number, group in enumerate(groups)
            for variable in group
        }
        for first_variable in range(N_VARIABLES):
            for second_variable in range(N_VARIABLES):
                if first_variable == second_variable:
                    expected = 1.0
                elif memberships[first_variable] == memberships[second_variable]:
                    expected = WITHIN_GROUP_CORRELATION
                else:
                    expected = 0.0
                assert np.isclose(
                    correlation[first_variable, second_variable],
                    expected,
                )

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
    sample_with_nuisance = gen_clustered_sample(
        same_dist=False,
        alpha=0.5,
        cluster_size=10,
        disjoint=False,
        nuisance_dimensions=25,
        random_state=1234,
    )
    assert np.array_equal(first_sample[0], repeated_sample[0])
    assert np.array_equal(first_sample[1], repeated_sample[1])
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
