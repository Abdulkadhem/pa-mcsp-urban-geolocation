"""Deterministic group-level pilot sampling that never splits a sequence."""

from __future__ import annotations

import hashlib


def stable_group_fraction(group_id: str, seed: int) -> float:
    """Map a stable group identifier to a deterministic value in [0, 1)."""
    if not group_id:
        raise ValueError("group_id must be non-empty")
    digest = hashlib.sha256(f"{seed}:{group_id}".encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return integer / float(1 << 64)


def stable_group_selected(group_id: str, fraction: float, seed: int) -> bool:
    """Select complete groups at the requested expected sampling fraction."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must lie in (0, 1]")
    return stable_group_fraction(group_id, seed) < fraction

