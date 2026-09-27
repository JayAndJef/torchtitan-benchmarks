"""The CPU pinning of a run: a ``numactl`` prefix that binds the processes to the NUMA node of their GPUs.

A failure to resolve the node returns an unpinned result that names the reason; it never raises.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from benchmarks.execution.devices import parse_devices
from benchmarks.execution.provenance import run_text


PCI_BUS_ID = re.compile(
    r"^([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])$"
)


@dataclass(frozen=True)
class CpuPinning:
    """The pinning of one run: a command prefix, and the description that the manifest records."""

    prefix: tuple[str, ...]
    description: str


def _numa_node(gpu: str, sysfs_root: Path) -> int | CpuPinning:
    """The NUMA node of one device, or the unpinned result that names the reason."""
    bus_id = run_text(
        [
            "nvidia-smi",
            "--id=" + gpu,
            "--query-gpu=pci.bus_id",
            "--format=csv,noheader",
        ]
    ).strip()
    match = PCI_BUS_ID.match(bus_id)
    if match is None:
        return CpuPinning((), f"none: cannot resolve PCI bus id ({bus_id})")
    if int(match.group(1), 16) > 0xFFFF:
        return CpuPinning((), f"none: unsupported PCI domain in {bus_id}")
    device = f"{match.group(1)[-4:]}:{match.group(2)}".lower()
    node_path = sysfs_root / "bus/pci/devices" / device / "numa_node"
    try:
        node = int(node_path.read_text())
    except (OSError, ValueError):
        return CpuPinning((), f"none: cannot read {node_path}")
    if node < 0:
        return CpuPinning((), f"none: {device} reports no NUMA affinity")
    return node


def resolve_cpu_pinning(gpu: str, *, sysfs_root: Path = Path("/sys")) -> CpuPinning:
    """The pinning of the devices ``gpu``: their one NUMA node, or none and the reason."""
    if shutil.which("numactl") is None:
        return CpuPinning((), "none: numactl not available")
    nodes: list[int] = []
    for device in parse_devices(gpu):
        resolved = _numa_node(device, sysfs_root)
        if isinstance(resolved, CpuPinning):
            # The first device that cannot be resolved decides the run.
            return resolved
        nodes.append(resolved)
    node = nodes[0]
    if any(other != node for other in nodes):
        return CpuPinning(
            (),
            "none: devices "
            + ",".join(parse_devices(gpu))
            + " span NUMA nodes "
            + ",".join(str(other) for other in nodes),
        )
    return CpuPinning(
        ("numactl", f"--cpunodebind={node}", f"--membind={node}"),
        f"numactl --cpunodebind={node} --membind={node}",
    )


def is_pinned(description: str) -> bool:
    """Whether a pinning record names a ``numactl`` prefix, and not a reason for none."""
    return description.startswith("numactl ")
