"""Partition controls and diagnostics for Local CorEx experiments."""

from time import time

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

PARTITION_SEED_CODES = {
    'oracle': 1,
    'phate': 2,
    'random': 3,
    'raw': 4,
}


def match_partition_to_truth(true_labels, predicted_labels):
    """Align arbitrary partition labels to the true population names.

    The Hungarian assignment changes only the names of the estimated clusters,
    not their membership.
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


def partition_data(data, true_labels, method='phate', random_state=42,
                   phate_dimensions=5, kmeans_n_init='auto'):
    """Partition observations and calculate ground-truth diagnostics.

    ``oracle`` returns the true regions. ``random`` preserves their sizes but
    randomly assigns observations. ``phate`` embeds the observations and then
    applies K-means. ``raw`` applies K-means directly to the supplied data.
    Labels are aligned only after partitioning.
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
    assert random_1['estimated_region_sizes'] == {
        'clust_1': 4,
        'clust_2': 4,
    }

    return {
        'label_permutation_checked': True,
        'oracle_partition_checked': True,
        'random_partition_reproducibility_checked': True,
        'random_partition_size_matching_checked': True,
    }
