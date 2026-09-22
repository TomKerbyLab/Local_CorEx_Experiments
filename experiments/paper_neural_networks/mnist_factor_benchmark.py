from __future__ import annotations

import copy
from typing import Iterable

import numpy as np
import pandas as pd
import torch

from local_corex.utils.nn_utils import prune_model_node


def get_top_factor_indices(corex_model, top_k=5):
    """
    Returns indices of the top-k factors ranked by TC explained.
    """
    tcs = np.asarray(corex_model.tcs)
    order = np.argsort(tcs)[::-1]
    return order[:top_k]


def compute_factor_support(
    mi_values,
    support_levels=(0.60, 0.70, 0.80, 0.90),
):
    """
    Parameters
    ----------
    mi_values : np.ndarray
        MI values for a single factor restricted to hidden nodes.

    Returns
    -------
    dict

    Example:

    {
        0.7 : {
            "nodes": [...],
            "support_size": 15,
            "support_density": 0.03,
            "cumulative_mi": 0.73
        }
    }
    """

    mi_values = np.asarray(mi_values).copy()

    abs_mi = np.abs(mi_values)

    total_mi = abs_mi.sum()

    if total_mi <= 0:
        return {
            level: {
                "nodes": [],
                "support_size": 0,
                "support_density": 0.0,
                "cumulative_mi": 0.0,
            }
            for level in support_levels
        }

    order = np.argsort(abs_mi)[::-1]

    sorted_mi = abs_mi[order]

    cumulative = np.cumsum(sorted_mi) / total_mi

    results = {}

    for level in support_levels:

        idx = np.searchsorted(cumulative, level)

        selected = order[: idx + 1]

        results[level] = {
            "nodes": selected.tolist(),
            "support_size": len(selected),
            "support_density": len(selected) / len(mi_values),
            "cumulative_mi": cumulative[idx],
        }

    return results


def evaluate_model_probabilities(
    model,
    inputs,
):
    """
    Returns probabilities.

    Shape:
        (n_samples, n_classes)
    """

    model.eval()

    device = next(model.parameters()).device

    with torch.no_grad():

        input_tensor = torch.as_tensor(
            inputs,
            dtype=torch.float32,
            device=device,
        )

        logits = model(input_tensor)

        probs = torch.nn.functional.softmax(logits, dim=1)

    return probs.cpu().numpy()


def prune_factor_nodes(
    model,
    hidden_layer_idx,
    node_indexes,
):
    """
    Clone model and prune selected nodes.
    """

    model_clone = copy.deepcopy(model)

    prune_model_node(
        model_clone,
        hidden_layer_idx,
        node_indexes,
    )

    return model_clone


def compute_delta_probabilities(
    base_probs,
    altered_probs,
    sample_idx=None,
):
    """
    Computes mean probability shift.

    Returns vector of length n_classes.
    """

    if sample_idx is not None:
        base_probs = base_probs[sample_idx]
        altered_probs = altered_probs[sample_idx]

    return altered_probs.mean(axis=0) - base_probs.mean(axis=0)


def compute_total_variation(delta_probs):
    """
    TV distance derived from probability shift vector.
    """

    return 0.5 * np.abs(delta_probs).sum()


def compute_normalized_entropy(delta_probs):
    """
    Entropy of shifted probability mass.

    Returns value in [0,1].
    """

    mass = np.abs(delta_probs)

    total_mass = mass.sum()

    if total_mass <= 1e-12:
        return 1.0

    p = mass / total_mass

    entropy = -(p * np.log(p + 1e-12)).sum()

    return entropy / np.log(len(p))


def compute_dominant_shift_fraction(delta_probs):
    """
    Fraction of shifted probability mass captured
    by largest class change.
    """

    mass = np.abs(delta_probs)

    denom = mass.sum()

    if denom <= 1e-12:
        return 0.0

    return mass.max() / denom


def compute_class_accuracy_changes(
    labels,
    base_probs,
    altered_probs,
):
    """
    Returns accuracy changes per class.

    Output:
        length n_classes vector
    """

    labels = np.asarray(labels)

    base_pred = base_probs.argmax(axis=1)
    alt_pred = altered_probs.argmax(axis=1)

    n_classes = int(labels.max()) + 1

    result = np.zeros(n_classes)

    for c in range(n_classes):

        idx = np.where(labels == c)[0]

        if len(idx) == 0:
            continue

        base_acc = np.mean(base_pred[idx] == labels[idx])

        alt_acc = np.mean(alt_pred[idx] == labels[idx])

        result[c] = alt_acc - base_acc

    return result


def is_class_associated(
    total_variation,
    entropy,
    tv_threshold=0.05,
    entropy_threshold=0.45,
):
    """
    Automatic factor classification.
    """

    return (
        total_variation > tv_threshold
        and entropy < entropy_threshold
    )


def run_factor_ablation(
    model,
    inputs,
    labels,
    cluster_indices,
    hidden_layer_idx,
    factor_id,
    support_level,
    support_info,
    base_probs,
):
    """
    Runs a single factor/support-level ablation.

    Returns dict suitable for dataframe row.
    """

    altered_model = prune_factor_nodes(
        model=model,
        hidden_layer_idx=hidden_layer_idx,
        node_indexes=support_info["nodes"],
    )

    altered_probs = evaluate_model_probabilities(
        altered_model,
        inputs,
    )

    global_delta = compute_delta_probabilities(
        base_probs,
        altered_probs,
    )

    local_delta = compute_delta_probabilities(
        base_probs,
        altered_probs,
        sample_idx=cluster_indices,
    )

    global_tv = compute_total_variation(global_delta)
    local_tv = compute_total_variation(local_delta)

    global_entropy = compute_normalized_entropy(global_delta)
    local_entropy = compute_normalized_entropy(local_delta)

    global_dom = compute_dominant_shift_fraction(global_delta)
    local_dom = compute_dominant_shift_fraction(local_delta)

    global_class_acc_change = compute_class_accuracy_changes(
        labels,
        base_probs,
        altered_probs,
    )

    local_class_acc_change = compute_class_accuracy_changes(
        labels[cluster_indices],
        base_probs[cluster_indices],
        altered_probs[cluster_indices],
    )

    return {
        "factor_id": factor_id,
        "support_level": support_level,
        "support_size": support_info["support_size"],
        "support_density": support_info["support_density"],
        "cumulative_mi": support_info["cumulative_mi"],
        "global_tv": global_tv,
        "local_tv": local_tv,
        "global_entropy": global_entropy,
        "local_entropy": local_entropy,
        "global_dom_fraction": global_dom,
        "local_dom_fraction": local_dom,
        "class_associated_global": is_class_associated(
            global_tv,
            global_entropy,
        ),
        "class_associated_local": is_class_associated(
            local_tv,
            local_entropy,
        ),
        "global_delta_probs": global_delta.tolist(),
        "local_delta_probs": local_delta.tolist(),
        "global_class_accuracy_change": global_class_acc_change.tolist(),
        "local_class_accuracy_change": local_class_acc_change.tolist(),
    }