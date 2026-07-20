"""Stable seed derivation for independent map-generation random streams."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def derive_seed(root_seed: int, *labels: Any) -> int:
    """Derive a deterministic unsigned 64-bit seed from a root seed and labels.

    The encoding intentionally avoids Python's process-randomized ``hash()`` so
    results are stable across interpreter restarts and call order.
    """

    payload = {"root_seed": int(root_seed), "labels": list(labels)}
    try:
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise TypeError("derive_seed labels must be JSON-serializable") from exc
    digest = hashlib.blake2b(encoded, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)
