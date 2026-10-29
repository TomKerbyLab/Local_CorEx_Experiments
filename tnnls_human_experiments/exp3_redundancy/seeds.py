"""Derived seeds: stable integers from condition identifiers (spec 00).

Same convention as ../AI_implemented/exp2_nonlinear/seeds.py: every
experiment component — dataset generation, each model fit, the
characterization sample — draws its seed by serializing the identifiers
that name it (root seed, purpose) and hashing them with sha256. The
builtin ``hash()`` is salted per process, so the hash here is the one
process-independent rule for turning identifiers into seeds.
"""

import hashlib
import json


def derived_seed(*parts):
    """Stable 32-bit integer seed from arbitrary identifiers.

    ``json.dumps(parts, sort_keys=True, default=str)`` makes the payload a
    pure function of the parts (numbers, strings, tuples); sha256 keeps it
    stable across processes and machines, unlike the builtin ``hash()``.
    """
    payload = json.dumps(parts, sort_keys=True, default=str).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'big')