"""Read a schema 19 manifest as the schema 20 manifest of the same run."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


V19_SCHEMA_VERSION = 19

V20_FIELDS: dict[str, dict[str, Any]] = {"torchtitan": {"packed_offsets": False}}
"""The config fields that schema 20 adds, by engine, with the value that every older arm held."""


def upgrade_v19(manifest: Mapping[str, Any]) -> dict[str, Any]:
    """The schema 20 form of a schema 19 manifest; each arm config gains the fields of ``V20_FIELDS``."""
    if manifest.get("schema_version") != V19_SCHEMA_VERSION:
        raise ValueError(
            f"the manifest records schema {manifest.get('schema_version')!r}, "
            f"not {V19_SCHEMA_VERSION}"
        )
    arms = []
    for record in manifest["arms"]:
        added = V20_FIELDS.get(record["engine"], {})
        present = sorted(set(added) & set(record["config"]))
        if present:
            raise ValueError(
                f"schema 19 arm {record['name']!r} records {present}, which "
                "schema 20 adds"
            )
        arms.append({**record, "config": {**record["config"], **added}})
    return {**manifest, "schema_version": 20, "arms": arms}
