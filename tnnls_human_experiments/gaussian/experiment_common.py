"""Utilities and data contracts shared by synthetic experiments."""

from dataclasses import dataclass, field

import numpy as np


def deterministic_seed(*values):
    """Create one reproducible integer seed from condition identifiers."""
    return int(np.random.SeedSequence(values).generate_state(1)[0])


def groups_to_indicator_vectors(groups, n_variables=None):
    """Convert explicit variable groups to binary ground-truth vectors."""
    groups = tuple(tuple(group) for group in groups)
    if n_variables is None:
        n_variables = max(variable for group in groups for variable in group) + 1
    indicators = []
    for group in groups:
        indicator = np.zeros(n_variables)
        indicator[list(group)] = 1.0
        indicators.append(indicator)
    return indicators


@dataclass
class SyntheticDataset:
    """Shared representation of a generated dataset.

    ``partition_data`` may contain variables that are excluded from
    ``model_data``. This supports the nonlinear routing-variable ablation
    while ensuring that every partition method receives the same observations.
    """

    partition_data: np.ndarray
    model_data: np.ndarray
    true_labels: np.ndarray
    true_groups: dict
    nuisance_mask: np.ndarray
    routing_mask: np.ndarray
    metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        self.partition_data = np.asarray(self.partition_data)
        self.model_data = np.asarray(self.model_data)
        self.true_labels = np.asarray(self.true_labels)
        self.nuisance_mask = np.asarray(self.nuisance_mask, dtype=bool)
        self.routing_mask = np.asarray(self.routing_mask, dtype=bool)

        if self.partition_data.ndim != 2 or self.model_data.ndim != 2:
            raise ValueError("Dataset arrays must be two-dimensional.")
        if self.partition_data.shape[0] != len(self.true_labels):
            raise ValueError("Data rows and true labels must have matching lengths.")
        if self.model_data.shape[0] != len(self.true_labels):
            raise ValueError("Model data rows and true labels must match.")
        n_model_variables = self.model_data.shape[1]
        if self.nuisance_mask.shape != (n_model_variables,):
            raise ValueError("nuisance_mask must match the model-data columns.")
        if self.routing_mask.shape != (n_model_variables,):
            raise ValueError("routing_mask must match the model-data columns.")
        for region, groups in self.true_groups.items():
            groups = np.asarray(groups)
            if groups.ndim != 2 or groups.shape[1] != n_model_variables:
                raise ValueError(
                    f"True groups for {region!r} must match model-data columns."
                )
