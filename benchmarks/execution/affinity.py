"""Binding a benchmark process to its GPU's own NUMA node.

The training step is host-bound at benchmark sizes, so where the scheduler
places the process is what decides tokens/s: an unpinned run measures
placement, not kernels. ``resolve_cpu_pinning`` walks GPU index ->
PCI bus id (``nvidia-smi``) -> sysfs ``numa_node`` and returns a ``numactl``
command prefix that both runners prepend to the command they launch.

**Every failure path produces a description, never an exception.** No
``numactl`` on the box, a bus id the regex does not recognize, a PCI domain
wider than sysfs names its devices with, an unreadable ``numa_node``, or a
device that reports no affinity at all: each returns an empty prefix and a
reason. The run then proceeds unpinned and says so, because the reason is
recorded in the manifest as ``hardware_metadata.cpu_pinning`` and
``--resume`` compares it -- so pinned and unpinned runs cannot be silently
mixed, and a box that cannot pin is still measurable. Turning any of these
into a raise would trade a labelled result for no result.

Kept out of ``environment.py`` because it changes for hardware reasons --
sysfs layout, PCI domain widths, ``numactl`` availability -- rather than
benchmark ones, and because it is the one part of ``execution/`` the CPU
tests drive on its own: six test modules import ``CpuPinning`` to stub a
run's pinning, and ``tests/test_runner.py`` patches this module's
``run_text`` and ``shutil.which`` to walk each of the branches above.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from benchmarks.execution.provenance import run_text


PCI_BUS_ID = re.compile(
    r"^([0-9a-fA-F]{4,8}):([0-9a-fA-F]{2}:[0-9a-fA-F]{2}\.[0-9a-fA-F])$"
)


@dataclass(frozen=True)
class CpuPinning:
    """Command prefix binding training to the GPU's NUMA node.

    The training step is host-bound at benchmark sizes, so where the scheduler
    places the process decides tokens/s. Binding CPU and memory to the GPU's
    own NUMA node removes that draw. An empty prefix means unpinned; the
    description says why and is recorded in the manifest, where resume treats
    it as a comparability boundary like the GPU itself.
    """

    prefix: tuple[str, ...]
    description: str


def resolve_cpu_pinning(gpu: str, *, sysfs_root: Path = Path("/sys")) -> CpuPinning:
    if shutil.which("numactl") is None:
        return CpuPinning((), "none: numactl not available")
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
    return CpuPinning(
        ("numactl", f"--cpunodebind={node}", f"--membind={node}"),
        f"numactl --cpunodebind={node} --membind={node}",
    )
