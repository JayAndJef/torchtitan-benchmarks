"""Which engine flags a ``--torchtitan-arg`` or ``--megatron-arg`` may carry.

A perf flag passes; a flag that would change a fact the manifest records is
refused and names its owner. Each engine keeps its own tables. A pattern
ending in ``*`` is a prefix.
"""

from __future__ import annotations

from collections.abc import Mapping


FlagTable = Mapping[str, tuple[str, ...]]
"""One engine's flag patterns, by the owner or the reason of each row."""


def matches(name: str, pattern: str) -> bool:
    """Whether ``name`` is ``pattern``, or starts with it before a ``*``."""
    if pattern.endswith("*"):
        return name.startswith(pattern[:-1])
    return name == pattern


def row_for(name: str, table: FlagTable) -> str | None:
    """The key of the first ``table`` row that covers ``name``, or ``None``."""
    for key, patterns in table.items():
        if any(matches(name, pattern) for pattern in patterns):
            return key
    return None


def ownership(name: str, owned: FlagTable, pinned: FlagTable) -> str | None:
    """``owned by <row>`` or ``pinned by <row>`` for ``name``, or ``None`` when no row covers it."""
    owner = row_for(name, owned)
    if owner is not None:
        return f"owned by {owner}"
    reason = row_for(name, pinned)
    if reason is not None:
        return f"pinned by {reason}"
    return None
