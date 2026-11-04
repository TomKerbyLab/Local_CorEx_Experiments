"""Experiment 4 qualitative-panel machinery: mirror autoencoder.

Encoder: the trained MNIST classifier (seed 5100), frozen — the latent is
h3 (post-BN, post-ReLU, 300-d), the pre-dropout representation the H3
CorEx factors live in. Decoder: freshly initialized mirror of the encoder
(784-500-400-300 chain reversed, BN + ReLU, sigmoid reconstruction head),
trained to reconstruct the training pixels (MSE). This is the legacy
program's decoder mechanism (build_decoder_from_classifier + AE training)
in a self-contained form; the panel it feeds is an explanatory
visualization whose factor selection must follow the predeclared rule
(paper/exp4.md) — never the reverse.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from ..common.config import CHECKPOINTS_DIR
from ..common.seeds import derive_seed
from .classifier import MLP, LAYER_DIMS
from .data import load_splits

AE_TRAIN_SEED = 5300
AE_EPOCHS = 10
AE_LR = 1e-3
AE_BATCH = 128


class MirrorDecoder(nn.Module):
    """784-500-400-300 mirrored: FC(300-400) BN ReLU, FC(400-500) BN ReLU,
    FC(500-784) sigmoid."""

    def __init__(self):
        super().__init__()
        d3, d2, d1 = LAYER_DIMS[2], LAYER_DIMS[1], LAYER_DIMS[0]
        self.fc1 = nn.Linear(d3, d2)
        self.bn1 = nn.BatchNorm1d(d2)
        self.fc2 = nn.Linear(d2, d1)
        self.bn2 = nn.BatchNorm1d(d1)
        self.fc3 = nn.Linear(d1, 784)

    def forward(self, z):
        h = torch.relu(self.bn1(self.fc1(z)))
        h = torch.relu(self.bn2(self.fc2(h)))
        return torch.sigmoid(self.fc3(h))


def train_mirror_ae(seed=5100, epochs=AE_EPOCHS, device=None) -> dict:
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    splits = load_splits(save_dir=CHECKPOINTS_DIR / "mnist")
    encoder = MLP(dropout_p=0.5)
    encoder.load_state_dict(torch.load(
        CHECKPOINTS_DIR / "mnist" / f"mnist_seed{seed}.pt",
        map_location="cpu"))
    encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad_(False)
    torch.manual_seed(derive_seed(AE_TRAIN_SEED, "mirror_ae", seed))
    decoder = MirrorDecoder().to(device)
    opt = torch.optim.Adam(decoder.parameters(), lr=AE_LR)
    x = splits["train_x"].to(device)
    n = len(x)
    history = []
    t0 = time.time()
    for epoch in range(epochs):
        g = torch.Generator().manual_seed(derive_seed(AE_TRAIN_SEED,
                                                      "ae_batch", epoch))
        perm = torch.randperm(n, generator=g).to(device)
        tot = 0.0
        for i in range(0, n, AE_BATCH):
            xb = x[perm[i:i + AE_BATCH]]
            _, _, _, _ = None, None, None, None
            with torch.no_grad():
                logits, _, _, h3 = encoder.hidden_states(xb)
            recon = decoder(h3)
            loss = torch.nn.functional.mse_loss(recon, xb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
        history.append(tot / n)
    # held-out reconstruction MSE on the untouched test set (flag, not acc)
    with torch.no_grad():
        tx = splits["test_x"].to(device)
        rec_mse = []
        for i in range(0, len(tx), 2048):
            xb = tx[i:i + 2048]
            _, _, _, h3 = encoder.hidden_states(xb)
            rec_mse.append(torch.nn.functional.mse_loss(
                decoder(h3), xb, reduction="none").mean().item())
    out = {"test_recon_mse": float(np.mean(rec_mse)), "epochs": epochs,
           "history": history, "runtime_s": time.time() - t0}
    ckpt_dir = CHECKPOINTS_DIR / "mnist"
    path = ckpt_dir / f"mnist_mirror_ae_seed{seed}.pt"
    torch.save(decoder.state_dict(), path)
    with open(ckpt_dir / f"mnist_mirror_ae_seed{seed}_meta.json", "w") as f:
        json.dump(out, f, indent=2)
    return out


if __name__ == "__main__":
    print(train_mirror_ae())