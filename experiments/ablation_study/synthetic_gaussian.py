from __future__ import annotations

from typing import Dict, List

import numpy as np

from synthetic_common import rng_for, pad_group_masks


def create_cov_mat_clust_1(visualize: bool = False) -> np.ndarray:
    mat = np.diag(
        np.hstack(
            [
                np.ones(5) * 7,
                np.ones(5) * 3,
                np.ones(5) * 10,
                np.ones(5) * 5,
                np.ones(5) * 8,
            ]
        )
    ).reshape(25, 25)

    first_block = [(x, y) for x in range(5) for y in range(5) if x != y]
    second_block = [(12, x) for x in range(5, 10)]
    second_block.extend([(x, y) for x in range(5, 10) for y in range(5, 10) if x != y])
    third_block = [(x, y) for x in range(10, 15) if x != 12 for y in range(10, 15) if y != 12 if y != x]
    fourth_block = [(x, y) for x in range(15, 21) for y in range(15, 21) if x != y]
    fifth_block = [(x, y) for x in range(21, 25) for y in range(21, 25) if x != y]

    for edge in first_block:
        mat[edge] = 5
    for edge in second_block:
        mat[edge] = 2
        mat[edge[::-1]] = 2
    for edge in third_block:
        mat[edge] = 2
    for edge in fourth_block:
        mat[edge] = 4
        mat[edge[::-1]] = 4
    for edge in fifth_block:
        mat[edge] = 3

    if visualize:
        import matplotlib.pyplot as plt
        plt.imshow(mat)

    return mat


def create_cov_mat_clust_2(visualize: bool = False) -> np.ndarray:
    mat = np.diag(
        np.hstack(
            [
                np.ones(5) * 7,
                np.ones(5) * 3,
                np.ones(5) * 10,
                np.ones(5) * 5,
                np.ones(5) * 8,
            ]
        )
    ).reshape(25, 25)

    first_block = [(x, y) for x in range(5) for y in range(5) if x != y]
    second_block = [(12, x) for x in range(5, 10)]
    second_block.extend([(x, y) for x in range(5, 10) for y in range(5, 10) if x != y])
    second_block.extend([(x, y) for x in range(15, 20) for y in range(5, 10) if x != y])
    third_block = [(x, y) for x in range(10, 15) if x != 12 for y in range(10, 15) if y != 12 if y != x]
    fourth_block = [(12, x) for x in range(15, 20)]
    fourth_block.extend([(x, y) for x in range(15, 20) for y in range(15, 20) if x != y])
    fifth_block = [(x, y) for x in range(20, 25) for y in range(20, 25) if x != y]

    for edge in first_block:
        mat[edge] = 5
    for edge in second_block:
        mat[edge] = 2
        mat[edge[::-1]] = 2
    for edge in third_block:
        mat[edge] = 2
    for edge in fourth_block:
        mat[edge] = 4
        mat[edge[::-1]] = 4
    for edge in fifth_block:
        mat[edge] = 3

    if visualize:
        import matplotlib.pyplot as plt
        plt.imshow(mat)

    return mat


def create_cov_mat_clust_4(visualize: bool = False) -> np.ndarray:
    mat = np.diag(
        np.hstack(
            [
                np.ones(5) * 7,
                np.ones(5) * 6,
                np.ones(5) * 10,
                np.ones(5) * 6,
                np.ones(5) * 8,
            ]
        )
    ).reshape(25, 25)

    first_block = [(5 * x, 5 * y) for x in range(5) for y in range(5) if x != y]
    second_block = [(5 * x + 1, 5 * y + 1) for x in range(5) for y in range(5) if x != y]
    third_block = [(5 * x + 2, 5 * y + 2) for x in range(5) for y in range(5) if x != y]
    fourth_block = [(5 * x + 3, 5 * y + 3) for x in range(5) for y in range(5) if x != y]
    fifth_block = [(5 * x + 4, 5 * y + 4) for x in range(5) for y in range(5) if x != y]

    for edge in first_block:
        mat[edge] = 5
    for edge in second_block:
        mat[edge] = 2
        mat[edge[::-1]] = 2
    for edge in third_block:
        mat[edge] = 2
    for edge in fourth_block:
        mat[edge] = 4
        mat[edge[::-1]] = 4
    for edge in fifth_block:
        mat[edge] = 3

    if visualize:
        import matplotlib.pyplot as plt
        plt.imshow(mat)

    return mat


def generate_gaussian_dataset(
    rep: int,
    cluster_size: int,
    alpha: float,
    disjoint: bool,
    n_noise: int = 0,
) -> Dict[str, object]:
    rng = rng_for(rep, "gaussian", cluster_size, alpha, disjoint, n_noise)

    clust_means_1 = np.hstack(
        [
            np.ones(5) * 5,
            np.ones(5) * 10,
            np.ones(5),
            np.ones(5) * 10,
            np.ones(5) * 3,
        ]
    )
    clust_means_2 = np.hstack(
        [
            np.ones(5) * 2.5,
            np.ones(5) * 6,
            np.ones(5) * 3,
            np.ones(5),
            np.ones(5) * 10,
        ]
    )
    diff_means = clust_means_1 - clust_means_2

    cov1 = create_cov_mat_clust_1()
    cov2 = create_cov_mat_clust_4() if disjoint else create_cov_mat_clust_2()

    X1 = rng.multivariate_normal(clust_means_1, cov1, size=cluster_size)
    X2 = rng.multivariate_normal(clust_means_1 + alpha * diff_means, cov2, size=cluster_size)

    X = np.concatenate([X1, X2], axis=0)
    y_true = np.concatenate(
        [
            np.repeat("clust_1", cluster_size),
            np.repeat("clust_2", cluster_size),
        ]
    )

    if n_noise > 0:
        X = np.concatenate([X, rng.normal(size=(X.shape[0], n_noise))], axis=1)

    groups_1 = pad_group_masks([((row != 0).astype(float)) for row in cov1], X.shape[1])
    groups_2 = pad_group_masks([((row != 0).astype(float)) for row in cov2], X.shape[1])

    return {
        "X": X,
        "y_true": y_true,
        "true_groups": {
            "clust_1": groups_1,
            "clust_2": groups_2,
        },
    }