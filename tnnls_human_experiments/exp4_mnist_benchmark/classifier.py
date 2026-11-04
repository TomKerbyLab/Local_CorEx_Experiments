"""Experiment 4 classifier: spec-exact architecture and activation-mask
interventions.

Architecture (04_mnist_factor_ablation_benchmark.md):
    h1 = ReLU(BN1(W1 x + b1)),           dim 500
    h1~ = Dropout0.5(h1)
    h2 = ReLU(BN2(W2 h1~ + b2)),         dim 400
    h2~ = Dropout0.5(h2)
    h3 = ReLU(BN3(W3 h2~ + b3)),         dim 300
    h3~ = Dropout0.5(h3)
    l  = W4 h3~ + b4                     (pre-softmax logits)
    p  = softmax(l)

H1/H2/H3 are h1/h2/h3 — after BN and ReLU, BEFORE dropout. At inference
dropout is disabled. Cross-entropy on logits (no LogSoftmax quirk).

Interventions insert a deterministic mask immediately after the layer's
BN+ReLU and before dropout or the next affine layer:

    h_l^abl_i = 0        if i in S
                h_l_i    otherwise

The model is in evaluation mode; BN running stats are untouched; no
rescaling. Activation masking is canonical; equivalence with outgoing-weight
deletion is unit-tested (test_masking_equivalence).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

LAYER_DIMS = (500, 400, 300)
N_CLASSES = 10


class MLP(nn.Module):
    """Spec-exact MNIST classifier with activation masking support."""

    def __init__(self, dropout_p: float = 0.5):
        super().__init__()
        self.dropout_p = dropout_p
        d1, d2, d3 = LAYER_DIMS
        self.fc1 = nn.Linear(784, d1)
        self.bn1 = nn.BatchNorm1d(d1)
        self.fc2 = nn.Linear(d1, d2)
        self.bn2 = nn.BatchNorm1d(d2)
        self.fc3 = nn.Linear(d2, d3)
        self.bn3 = nn.BatchNorm1d(d3)
        self.fc4 = nn.Linear(d3, N_CLASSES)
        self.drop = nn.Dropout(p=dropout_p)
        # running masks: None = no intervention
        self._masks: dict[int, torch.Tensor | None] = {1: None, 2: None, 3: None}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.hidden_forward(x, 1)
        h1 = self.drop(h1)
        h2 = self.hidden_forward(h1, 2)
        h2 = self.drop(h2)
        h3 = self.hidden_forward(h2, 3)
        h3 = self.drop(h3)
        return self.fc4(h3)

    def hidden_forward(self, x: torch.Tensor, layer: int) -> torch.Tensor:
        """Compute h_layer from the previous layer's (already dropped) input.
        Applies BN + ReLU, then the intervention mask if set."""
        if layer == 1:
            h = F.relu(self.bn1(self.fc1(x)))
        elif layer == 2:
            h = F.relu(self.bn2(self.fc2(x)))
        elif layer == 3:
            h = F.relu(self.bn3(self.fc3(x)))
        else:
            raise ValueError(f"layer must be 1..3, got {layer}")
        mask = self._masks.get(layer)
        if mask is not None:
            h = h * mask
        return h

    def hidden_states(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Return (logits, h1, h2, h3) with dropout disabled (eval mode)."""
        h1 = self.hidden_forward(x, 1)
        h2 = self.hidden_forward(h1, 2)
        h3 = self.hidden_forward(h2, 3)
        logits = self.fc4(h3)
        return logits, h1, h2, h3

    # -- intervention API ---------------------------------------------------

    def set_mask(self, layer: int, indices: torch.Tensor | list[int] | None) -> None:
        """Set the ablation mask for `layer` (unit indices -> zeroed).

        The stored mask is 1.0 at KEPT coordinates and 0.0 at ablated ones;
        activations are multiplied by it. None or an empty list clears the
        intervention (mask == all ones).
        """
        if layer not in (1, 2, 3):
            raise ValueError("layer must be 1, 2, or 3")
        if indices is None or len(indices) == 0:
            self._masks[layer] = None
            return
        d = LAYER_DIMS[layer - 1]
        mask = torch.ones(d, device=next(self.parameters()).device)
        idx = torch.as_tensor(list(indices), dtype=torch.long, device=mask.device)
        assert idx.min() >= 0 and idx.max() < d, "mask index out of range"
        assert len(set(idx.tolist())) == len(idx), "duplicate mask indices"
        mask[idx] = 0.0
        self._masks[layer] = mask

    def clear_masks(self) -> None:
        for layer in (1, 2, 3):
            self._masks[layer] = None

    def active_masks(self) -> dict[int, list[int]]:
        """Ablated unit indices per active layer (recovered from the mask)."""
        out = {}
        for layer, m in self._masks.items():
            if m is not None:
                out[layer] = torch.nonzero(m == 0).flatten().tolist()
        return out


def zero_out_weight_rows_clone(model: MLP, layer: int, indices: list[int]) -> MLP:
    """Equivalent intervention via cloned weights: zero outgoing weight rows.

    For layer l in {1,2,3}, zeroing W_{l+1}[:, i] for i in S equals activation
    masking h_l[i] = 0 ONLY when no BN/dropout intervenes between them — the
    next affine layer consumes h_l AFTER dropout, and dropout is disabled at
    evaluation, so at eval time the two are numerically equivalent. This is
    verified by unit test on a deterministic network; used only as a
    cross-check, not as the primary implementation.
    """
    import copy

    m = copy.deepcopy(model)
    d = LAYER_DIMS[layer - 1]
    fc = {1: m.fc2, 2: m.fc3, 3: m.fc4}[layer]
    with torch.no_grad():
        fc.weight[:, list(indices)] = 0.0
    return m


def weight_rows_deletion_logits(model: MLP, x: torch.Tensor, layer: int,
                                indices: list[int]) -> torch.Tensor:
    """Forward pass on a clone with outgoing weight columns zeroed.

    The equivalence with activation masking holds at eval mode (dropout off):
    layer l's masked activations feed fc_{l+1} linearly.
    """
    m = zero_out_weight_rows_clone(model, layer, indices)
    m.clear_masks()
    m.eval()
    with torch.no_grad():
        return m(x)