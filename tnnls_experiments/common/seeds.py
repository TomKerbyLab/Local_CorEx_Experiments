"""Deterministic seed derivation.

Convention (agent_instructions/00_README_experiment_program.md): every run uses a
single root seed and derives component seeds deterministically,

    s_{r,j} = hash(s_r, j) mod (2^32 - 1),

where j identifies the component (data generation, partitioning, CorEx
initialization, random ablation, ...). No global RNG state is relied upon.
"""

from __future__ import annotations

import hashlib
from typing import Sequence

MAX_SEED = 2**32 - 1


def derive_seed(root_seed: int, *tags: object) -> int:
    """Deterministically derive a uint32 seed from a root seed and component tags.

    Tags may be any JSON-serializable scalars/strings (ints, strs, bools, None,
    tuples/lists of these). The hash is stable across runs and platforms.
    """
    payload = repr(_canonical(tags))
    digest = hashlib.sha256(f"{int(root_seed)}|{payload}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big") % MAX_SEED
    # 0 is a valid numpy seed; avoid it anyway for uniformity with human-set seeds.
    return value or 1


def _canonical(obj):
    if isinstance(obj, (list, tuple)):
        return [_canonical(x) for x in obj]
    if isinstance(obj, bool) or obj is None:
        return obj
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        # Use repr for stable float formatting.
        return repr(float(obj))
    if isinstance(obj, str):
        return obj
    return repr(obj)


def seed_seq(root_seed: int, tags: Sequence[object]) -> list[int]:
    """Derive several seeds at once (one per tag element)."""
    return [derive_seed(root_seed, t) for t in tags]