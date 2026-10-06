"""The CPU pinning of a run: a ``numactl`` prefix that binds the processes to the NUMA node of their GPUs."""

from __future__ import annotations

import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from benchmarks.execution.devices import parse_devices
from benchmarks.execution.provenance import run_text


PCI_BUS_ID = re.compile(
    r"^([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])$"
)
"""A PCI bus id as ``nvidia-smi`` prints it: the domain, then the bus, the device and the function."""


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


def _cpu_list(text: str) -> frozenset[int]:
    """The CPUs of a sysfs ``cpulist`` such as ``0-63,128-191``."""
    cpus: set[int] = set()
    for part in text.strip().split(","):
        first, _, last = part.partition("-")
        cpus.update(range(int(first), int(last or first) + 1))
    return frozenset(cpus)


def _cpu_ranges(cpus: frozenset[int]) -> str:
    """``cpus`` as the compact range list that ``numactl --physcpubind`` reads."""
    ordered = sorted(cpus)
    ranges: list[str] = []
    start = previous = ordered[0]
    for cpu in ordered[1:] + [None]:
        if cpu is not None and cpu == previous + 1:
            previous = cpu
            continue
        ranges.append(str(start) if start == previous else f"{start}-{previous}")
        if cpu is not None:
            start = previous = cpu
    return ",".join(ranges)


def resolve_cpu_pinning(
    gpu: str,
    *,
    sysfs_root: Path = Path("/sys"),
    allowed_cpus: frozenset[int] | None = None,
) -> CpuPinning:
    """The pinning of the devices ``gpu``: the CPUs of their one NUMA node that the process may use, or none and the reason."""
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
    node_path = sysfs_root / f"devices/system/node/node{node}/cpulist"
    try:
        node_cpus = _cpu_list(node_path.read_text())
    except (OSError, ValueError):
        return CpuPinning((), f"none: cannot read {node_path}")
    if allowed_cpus is None:
        allowed_cpus = frozenset(os.sched_getaffinity(0))
    usable = node_cpus & allowed_cpus
    if not usable:
        return CpuPinning((), f"none: the process may use no CPU of NUMA node {node}")
    if usable != node_cpus:
        cpus = _cpu_ranges(usable)
        return CpuPinning(
            ("numactl", f"--physcpubind={cpus}", f"--membind={node}"),
            f"numactl --physcpubind={cpus} --membind={node}",
        )
    return CpuPinning(
        ("numactl", f"--cpunodebind={node}", f"--membind={node}"),
        f"numactl --cpunodebind={node} --membind={node}",
    )


def is_pinned(description: str) -> bool:
    """Whether a pinning record names a ``numactl`` prefix, and not a reason for none."""
    return description.startswith("numactl ")
