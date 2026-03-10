import math
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt


# --------- helpers ---------

def get_module_by_name(model: nn.Module, name: str) -> nn.Module:
    """Find a submodule by its dotted name from model.named_modules()."""
    modules = dict(model.named_modules())
    if name not in modules:
        raise KeyError(
            f"Layer '{name}' not found. Available examples: "
            f"{list(modules.keys())[:20]} ... (total {len(modules)})"
        )
    return modules[name]


def _normalize_grad(grad: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return grad / (grad.abs().mean() + eps)


def _jitter_roll(x: torch.Tensor, ox: int, oy: int) -> torch.Tensor:
    # x: (1, C, H, W)
    return torch.roll(torch.roll(x, shifts=ox, dims=-1), shifts=oy, dims=-2)


# --------- activation specs ---------

@dataclass
class TargetSpec:
    layer: str
    weight: float = 1.0

    channels: Optional[Sequence[int]] = None
    node_weights: Optional[Sequence[float]] = None  # NEW

    neurons: Optional[Sequence[Tuple[int, int, int]]] = None
    mask: Optional[torch.Tensor] = None
    objective: str = "l2"


def _apply_selection(act: torch.Tensor, spec: TargetSpec) -> torch.Tensor:
    """
    act shape is usually (N, C, H, W) for conv nets.
    We return a tensor containing only the selected values.
    """
    x = act

    if spec.channels is not None:
        ch = torch.tensor(spec.channels, device=x.device, dtype=torch.long)
        x = x.index_select(dim=1, index=ch)

    if spec.neurons is not None:
        # Gather specific (c, y, x) positions from the original activation.
        vals = []
        for (c, y, xpix) in spec.neurons:
            vals.append(act[:, c, y, xpix])
        x = torch.stack(vals, dim=0)  # (K, N)
        x = x.reshape(-1)             # flatten

    if spec.mask is not None:
        m = spec.mask.to(x.device, x.dtype)
        x = x * m

    return x


def build_objective(specs: List[TargetSpec]) -> Callable[[Dict[str, torch.Tensor]], torch.Tensor]:
    def objective_fn(acts: Dict[str, torch.Tensor]) -> torch.Tensor:
        total = 0.0
        eps = 1e-8

        for spec in specs:
            if spec.layer not in acts:
                raise KeyError(
                    f"Activation for layer '{spec.layer}' not found. "
                    "Check the layer name or whether its hook fired."
                )

            act = acts[spec.layer]
            selected = _apply_selection(act, spec)

            if spec.objective.lower() == "l2":
                # Weighted nodes case: channels selected and node_weights provided
                if spec.channels is not None and spec.node_weights is not None:
                    w = torch.tensor(spec.node_weights, device=selected.device, dtype=selected.dtype)

                    # selected is usually (N, K, H, W) or (N, K)
                    if selected.ndim == 4:
                        per_node = (selected ** 2).mean(dim=(0, 2, 3))  # (K,)
                    elif selected.ndim == 2:
                        per_node = (selected ** 2).mean(dim=0)          # (K,)
                    else:
                        raise ValueError(f"Unexpected selected shape {tuple(selected.shape)} for weighted nodes.")

                    if w.numel() != per_node.numel():
                        raise ValueError(
                            f"node_weights length ({w.numel()}) must match number of selected channels ({per_node.numel()})"
                        )

                    # Optional normalization so weight scale does not blow up step sizes
                    score = (w * per_node).sum() / (w.abs().sum() + eps)
                else:
                    score = (selected ** 2).mean()

            elif spec.objective.lower() == "mean":
                if spec.channels is not None and spec.node_weights is not None:
                    w = torch.tensor(spec.node_weights, device=selected.device, dtype=selected.dtype)
                    if selected.ndim == 4:
                        per_node = selected.mean(dim=(0, 2, 3))  # (K,)
                    elif selected.ndim == 2:
                        per_node = selected.mean(dim=0)          # (K,)
                    else:
                        raise ValueError(f"Unexpected selected shape {tuple(selected.shape)} for weighted nodes.")

                    if w.numel() != per_node.numel():
                        raise ValueError(
                            f"node_weights length ({w.numel()}) must match number of selected channels ({per_node.numel()})"
                        )

                    score = (w * per_node).sum() / (w.abs().sum() + eps)
                else:
                    score = selected.mean()

            else:
                raise ValueError(f"Unknown objective '{spec.objective}'. Use 'l2' or 'mean'.")

            total = total + spec.weight * score

        return total
    return objective_fn


# --------- main deepdream ---------

class DeepDreamer:
    def __init__(
        self,
        model: nn.Module,
        preprocess: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        device: Optional[Union[str, torch.device]] = None,
    ):
        """
        preprocess: function mapping input tensor (1,C,H,W) in [0,1] to model input space
                    (for ImageNet models this is usually normalization).
        """
        self.model = model
        self.model.eval()

        # Freeze model params to avoid storing their grads
        for p in self.model.parameters():
            p.requires_grad_(False)

        self.device = torch.device(device) if device is not None else next(model.parameters()).device
        self.model.to(self.device)

        self.preprocess = preprocess if preprocess is not None else (lambda x: x)

    @torch.no_grad()
    def _octave_pyramid(self, base: torch.Tensor, octave_n: int, octave_scale: float) -> List[torch.Tensor]:
        """Return list of images from small -> large."""
        octaves = [base]
        for _ in range(octave_n - 1):
            prev = octaves[-1]
            h, w = prev.shape[-2], prev.shape[-1]
            nh = max(8, int(h / octave_scale))
            nw = max(8, int(w / octave_scale))
            smaller = F.interpolate(prev, size=(nh, nw), mode="bilinear", align_corners=False)
            octaves.append(smaller)
        return octaves[::-1]  # smallest first

    def dream(
        self,
        image: torch.Tensor,
        specs: List[TargetSpec],
        steps: int = 20,
        step_size: float = 0.02,
        jitter: int = 16,
        octaves: int = 4,
        octave_scale: float = 1.4,
        clamp_range: Tuple[float, float] = (0.0, 1.0),
        grad_normalize: bool = True,
        custom_objective: Optional[Callable[[Dict[str, torch.Tensor]], torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        image: float tensor (1,C,H,W) in [0,1] on any device
        specs: list of TargetSpec, one per target layer (can be multiple)
        returns: dreamed image tensor (1,C,H,W) in [0,1]
        """
        x0 = image.detach().to(self.device).float()
        if x0.ndim != 4 or x0.shape[0] != 1:
            raise ValueError("image must have shape (1, C, H, W)")

        # Prepare objective
        objective_fn = custom_objective if custom_objective is not None else build_objective(specs)

        # Register hooks
        acts: Dict[str, torch.Tensor] = {}
        hooks = []
        try:
            for spec in specs:
                module = get_module_by_name(self.model, spec.layer)

                def _make_hook(layer_name: str):
                    def hook(_module, _inp, out):
                        acts[layer_name] = out
                    return hook

                hooks.append(module.register_forward_hook(_make_hook(spec.layer)))

            # Octave processing
            octaves_list = self._octave_pyramid(x0, octave_n=octaves, octave_scale=octave_scale)
            detail = torch.zeros_like(octaves_list[0])

            for octave_idx, octave_base in enumerate(octaves_list):
                if octave_idx == 0:
                    x = octave_base.clone()
                else:
                    # Upsample detail to current octave size and add
                    detail = F.interpolate(detail, size=octave_base.shape[-2:], mode="bilinear", align_corners=False)
                    x = (octave_base + detail).clamp(*clamp_range)

                x.requires_grad_(True)

                for _ in range(steps):
                    # Jitter
                    if jitter > 0:
                        ox = int(torch.randint(-jitter, jitter + 1, (1,), device=self.device).item())
                        oy = int(torch.randint(-jitter, jitter + 1, (1,), device=self.device).item())
                        x_j = _jitter_roll(x, ox, oy)
                    else:
                        ox = oy = 0
                        x_j = x

                    acts.clear()
                    self.model.zero_grad(set_to_none=True)

                    xin = self.preprocess(x_j)
                    _ = self.model(xin)

                    score = objective_fn(acts)
                    # We want to maximize score, so minimize -score
                    loss = -score
                    loss.backward()

                    with torch.no_grad():
                        g = x.grad
                        if grad_normalize:
                            g = _normalize_grad(g)

                        x.add_(-step_size * g)  # because grad is for loss=-score
                        x.clamp_(*clamp_range)
                        x.grad.zero_()

                    # Un-jitter (roll back)
                    if jitter > 0:
                        with torch.no_grad():
                            x.copy_(_jitter_roll(x, -ox, -oy))

                # Update detail for next octave
                with torch.no_grad():
                    detail = (x.detach() - octave_base)

            return (octaves_list[-1] + detail).clamp(*clamp_range)

        finally:
            for h in hooks:
                h.remove()

def dream_images_and_plot(
    images,
    dreamer,
    specs,
    steps=40,
    step_size=0.02,
    jitter=0,
    octaves=4,
    octave_scale=1.4,
    resize=(224, 224),
    cols=4,
    figsize_scale=4,
    show_original_above=False,
):
    dreamed_images = []
    original_images = []

    for img in images:
        if not isinstance(img, Image.Image):
            raise TypeError("All elements in images must be PIL.Image.Image objects")

        rgb_img = img.convert("RGB")
        if resize is not None:
            rgb_img = rgb_img.resize(resize)
        original_images.append(rgb_img)

        x = torch.from_numpy(np.array(rgb_img)).float() / 255.0
        x = x.permute(2, 0, 1).unsqueeze(0)

        out = dreamer.dream(
            image=x,
            specs=specs,
            steps=steps,
            step_size=step_size,
            jitter=jitter,
            octaves=octaves,
            octave_scale=octave_scale,
        )

        out_img = (
            out.squeeze(0).permute(1, 2, 0).detach().cpu().numpy() * 255.0
        ).astype(np.uint8)
        dreamed_images.append(Image.fromarray(out_img))

    if len(dreamed_images) == 0:
        raise ValueError("images must contain at least one image")

    pair_rows = math.ceil(len(dreamed_images) / cols)

    if show_original_above:
        total_rows = pair_rows * 2
        fig, axes = plt.subplots(
            total_rows,
            cols,
            figsize=(figsize_scale * cols, figsize_scale * total_rows),
        )
        axes = np.array(axes).reshape(total_rows, cols)

        for idx in range(len(dreamed_images)):
            r = idx // cols
            c = idx % cols

            ax_original = axes[2 * r, c]
            ax_dream = axes[2 * r + 1, c]

            ax_original.imshow(original_images[idx])
            ax_original.set_title(f"Original {idx + 1}")
            ax_original.axis("off")

            ax_dream.imshow(dreamed_images[idx])
            ax_dream.set_title(f"Dream {idx + 1}")
            ax_dream.axis("off")

        for idx in range(len(dreamed_images), pair_rows * cols):
            r = idx // cols
            c = idx % cols
            axes[2 * r, c].axis("off")
            axes[2 * r + 1, c].axis("off")
    else:
        fig, axes = plt.subplots(
            pair_rows,
            cols,
            figsize=(figsize_scale * cols, figsize_scale * pair_rows),
        )
        axes = np.array(axes).reshape(-1)

        for idx, ax in enumerate(axes):
            if idx < len(dreamed_images):
                ax.imshow(dreamed_images[idx])
                ax.set_title(f"Dream {idx + 1}")
                ax.axis("off")
            else:
                ax.axis("off")

    plt.tight_layout()
    plt.show()

    return dreamed_images