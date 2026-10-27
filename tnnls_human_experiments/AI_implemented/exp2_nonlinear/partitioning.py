"""Partition methods for the nonlinear experiments.

Adapted from ../../gaussian/partitioning.py. Methods:

- ``oracle``: the true region labels.
- ``phate``: PHATE embedding (10 dimensions by default) followed by k-means
  with k = 2, n_init = 50.
- ``raw``: k-means directly on the supplied data.
- ``random``: random relabelings that preserve the true region sizes
  (balanced random partitions; the runner draws 10 per dataset).

Labels are aligned to the true regions only after partitioning, and the
result stores boolean observation masks per aligned region name rather than
raw cluster ids. This is deliberate: an aligned region id must never be
used to index the raw estimated labels, because the two differ whenever the
k-means label map is a swap.
"""

from time import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

PHATE_DIMENSIONS = 10
KMEANS_N_INIT = 50
N_RANDOM_DRAWS = 10


def match_partition_to_truth(true_labels, predicted_labels):
    """Align estimated labels to true region names (Hungarian assignment).

    The assignment changes only the names of the estimated clusters, never
    their membership. Requires the same number of distinct labels on both
    sides; a collapsed partition (single estimated cluster) raises
    ValueError, which the runner records as a failed partition.
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
    for true_index, true_name in enumerate(true_names):
        for predicted_index, predicted_name in enumerate(predicted_names):
            contingency[true_index, predicted_index] = np.sum(
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
    for true_index, true_name in enumerate(true_names):
        for mapped_index, mapped_name in enumerate(true_names):
            aligned_confusion[true_index, mapped_index] = np.sum(
                (true_labels == true_name)
                & (mapped_labels == mapped_name)
            )

    return mapped_labels, true_names, contingency, aligned_confusion


def _region_names(true_labels):
    """Region names as strings (the runner slices by region id 0/1)."""
    return [str(label) for label in np.unique(true_labels)]


def partition_data(data, true_labels, method='phate', random_state=42,
                   phate_dimensions=PHATE_DIMENSIONS,
                   kmeans_n_init=KMEANS_N_INIT):
    """Partition observations and compute partition diagnostics.

    Returns a dict with per-region boolean masks (``indexes``) that can be
    applied to the raw data directly, ARI/purity/aligned accuracy, the
    aligned confusion matrix, and the time spent.
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
    elif method in {'phate', 'raw'}:
        phate_seed = int(rng.integers(0, 2**31 - 1))
        kmeans_seed = int(rng.integers(0, 2**31 - 1))
        if method == 'phate':
            import phate

            clustering_data = phate.PHATE(
                n_components=phate_dimensions,
                n_jobs=4,
                random_state=phate_seed,
            ).fit_transform(data)
        else:
            clustering_data = data
        predicted_labels = KMeans(
            n_clusters=n_clusters,
            random_state=kmeans_seed,
            n_init=kmeans_n_init,
        ).fit_predict(clustering_data)
    else:
        raise ValueError(
            f"Unknown partition method '{method}'. "
            "Choose 'oracle', 'phate', 'raw', or 'random'."
        )

    partition_time = time() - start_time
    mapped_labels, _mapped_names, contingency, aligned_confusion = (
        match_partition_to_truth(true_labels, predicted_labels)
    )
    partition_purity = contingency.max(axis=0).sum() / len(true_labels)
    mapped_accuracy = np.mean(mapped_labels == true_labels)
    ari = adjusted_rand_score(true_labels, predicted_labels)

    region_masks = {
        str(label): mapped_labels == label
        for label in np.unique(true_labels)
    }
    estimated_region_sizes = {
        str(label): int(np.sum(mapped_labels == label))
        for label in np.unique(true_labels)
    }
    region_purity = {
        str(label): float(np.mean(true_labels[region_masks[str(label)]] == label))
        for label in np.unique(true_labels)
    }

    return {
        'method': method,
        'indexes': region_masks,
        'predicted_labels': predicted_labels,
        'mapped_labels': mapped_labels,
        'label_order': _region_names(true_labels),
        'confusion_matrix': aligned_confusion,
        'ari': float(ari),
        'purity': float(partition_purity),
        'mapped_accuracy': float(mapped_accuracy),
        'estimated_region_sizes': estimated_region_sizes,
        'region_purity': region_purity,
        'partition_time': partition_time,
    }


def validate_partitioning():
    """Fast self-checks on label matching and non-PHATE partitions."""
    true_labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    swapped_labels = np.array([1, 1, 1, 1, 0, 0, 0, 0])
    mapped_labels, _, _, confusion = match_partition_to_truth(
        true_labels,
        swapped_labels,
    )
    assert np.array_equal(mapped_labels, true_labels)
    assert np.array_equal(confusion, np.diag([4, 4]))

    # the aligned labels of a region select that region's own observations,
    # even under a label swap (the raw-vs-aligned regression case)
    partition = partition_data(
        np.arange(16).reshape(8, 2),
        true_labels,
        method='oracle',
        random_state=1234,
    )
    assert np.array_equal(
        np.flatnonzero(partition['indexes']['0']),
        np.flatnonzero(true_labels == 0),
    )

    dummy_data = np.arange(16).reshape(8, 2)
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
    assert random_1['estimated_region_sizes'] == {'0': 4, '1': 4}

    collapsed_hit = False
    try:
        match_partition_to_truth(true_labels, np.zeros(8, dtype=int))
    except ValueError:
        collapsed_hit = True
    assert collapsed_hit

    return {
        'label_permutation_checked': True,
        'oracle_partition_checked': True,
        'random_partition_reproducibility_checked': True,
        'random_partition_size_matching_checked': True,
        'collapsed_partition_checked': True,
    }