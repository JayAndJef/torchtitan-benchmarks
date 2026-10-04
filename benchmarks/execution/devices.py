"""Parse the ``<gpu>`` positional into the devices of one run."""

from __future__ import annotations

import re


DEVICE_LIST = re.compile(r"^\d+(,\d+)*$")
"""Decimal GPU indices separated by single commas."""


def parse_devices(gpu: str) -> tuple[str, ...]:
    """The devices that ``gpu`` names, as typed; a malformed list or a repeated device raises ``ValueError``."""
    if not DEVICE_LIST.match(gpu):
        raise ValueError(
            f"device list {gpu!r} is not one or more comma-separated GPU "
            "indices, for example '0' or '0,1'"
        )
    devices = tuple(gpu.split(","))
    values = [int(device) for device in devices]
    duplicates = sorted({value for value in values if values.count(value) > 1})
    if duplicates:
        raise ValueError(
            f"device list {gpu!r} names the same device more than once: "
            + ", ".join(str(value) for value in duplicates)
        )
    return devices
