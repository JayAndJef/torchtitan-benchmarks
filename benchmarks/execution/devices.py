"""The set of GPUs one run occupies, parsed from the ``<gpu>`` positional.

``parse_devices`` is what decides whether that argument is a legal device
set, and what splits it. Three consumers take the split: ``affinity.py``
resolves a NUMA node per device, ``benchmarks/e2e/runner.py`` counts it for
``benchmarks/e2e/parallelism.py``'s rule 1, and ``benchmarks/cli/kernel.py``
refuses a set of more than one. Two more read the string without splitting
it: ``environment.py`` passes it to ``CUDA_VISIBLE_DEVICES``, and
``provenance.py`` hands it to ``nvidia-smi --id=`` whole and counts the
members by their commas. Those two never call this function, so it is not
the *only* reader -- it is the one that says what a legal string is, and the
grammar below is what makes the comma count agree with the split.

**The string itself is never rewritten.** ``RunRequest.gpu`` keeps exactly
what the operator typed, because every manifest under ``out/`` records it as
``hardware_metadata.requested_gpu`` and ``CUDA_VISIBLE_DEVICES`` is set from
the same value. A normalizing parser would make a resumed run record a
different string than the run it continues.

The accepted spelling is therefore narrow on purpose: decimal indices
separated by single commas, and nothing else. No spaces, no UUIDs, no
negative numbers, no empty entries. Every one of those either reaches
``CUDA_VISIBLE_DEVICES`` unchanged, where the driver would read it
differently than this module does, or reaches ``nvidia-smi --id=`` and
selects a device the manifest does not name. ``"0, 1"`` is the measured
case: ``nvidia-smi`` accepts the space and answers for two devices, where
``CUDA_VISIBLE_DEVICES`` stops at the space and shows one. A duplicate is
refused for the same reason: it would make ``NGPU`` claim more ranks than
the run has devices.

**Two spellings this narrowing gives up, and neither has ever been used.**
Every manifest under ``out/`` records a single decimal index. A
``GPU-<uuid>`` -- which both ``nvidia-smi --id=`` and
``CUDA_VISIBLE_DEVICES`` accept, and which is the only spelling that selects
a MIG instance -- is now refused, so a MIG host needs this grammar widened
before it can run. And ``run 0,1`` used to run **one** GPU: ``NGPU`` was 1,
so the training process took the first visible device while the manifest
recorded ``"0,1"``. That is a wrong recorded fact rather than a missing one,
and rule 1 refusing it is the repair.

This function checks the **spelling** and nothing else. It does not ask
whether a device exists: ``nvidia-smi --id=0,99`` exits 0 and answers for
one device on a two-GPU box, so a nonexistent index reaches the manifest as
a short ``nvidia_smi`` block. The world-size refusal in
``benchmarks/e2e/runner.py`` stops every multi-device run today, and the
stage that lifts it owns that check.

Kept apart from ``paths.py`` because a device set is not a filesystem
location, and apart from both CLI modules because ``kernel-bench`` reads it
too -- it accepts one device only, and says so through this parser rather
than through a second spelling of the rule.
"""

from __future__ import annotations

import re


# Decimal indices, single commas, nothing else. See the module docstring for
# why every rejected spelling is rejected.
DEVICE_LIST = re.compile(r"^\d+(,\d+)*$")


def parse_devices(gpu: str) -> tuple[str, ...]:
    """Split the ``<gpu>`` positional into the devices it names.

    Returns the tokens exactly as they were typed, so ``"0"`` gives
    ``("0",)`` and ``"0,1"`` gives ``("0", "1")``. The caller keeps the
    original string for the manifest and for ``CUDA_VISIBLE_DEVICES``.

    Raises ``ValueError`` on anything the pattern above refuses, and on a
    repeated device. The duplicate test compares integer values, so ``"0,00"``
    is one device written two ways and is refused.
    """
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
