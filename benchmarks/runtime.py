"""Runtime integration for launching TorchTitan benchmark arms."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from benchmarks.scenarios import Arm, Workload


BENCH_DIR = Path(__file__).resolve().parent.parent
TITAN_DIR = BENCH_DIR / "third_party" / "torchtitan"


@dataclass(frozen=True)
class RuntimePaths:
    """Filesystem locations needed to execute a benchmark."""

    bench_dir: Path
    titan_dir: Path
    cache_root: Path
    compiler_env: Path | None

    @classmethod
    def resolve(
        cls,
        *,
        cache_root: Path | None = None,
        compiler_env: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> RuntimePaths:
        environment = environment or os.environ
        cache = cache_root or _optional_path(environment.get("BENCHMARK_CACHE_ROOT"))
        if cache is None:
            cache = Path(tempfile.gettempdir()) / "torchtitan-benchmarks"

        compiler = compiler_env or _optional_path(environment.get("BENCH_COMPILER_ENV"))
        if compiler is None:
            legacy_toolset = Path("/opt/rh/gcc-toolset-13/enable")
            compiler = legacy_toolset if legacy_toolset.exists() else None

        return cls(
            bench_dir=BENCH_DIR,
            titan_dir=TITAN_DIR,
            cache_root=cache.expanduser().resolve(),
            compiler_env=compiler.expanduser().resolve() if compiler else None,
        )


def _optional_path(value: str | None) -> Path | None:
    return Path(value) if value else None


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


def command_for_arm(
    workload: Workload,
    arm: Arm,
    arm_dir: Path,
    extra_args: list[str] | tuple[str, ...],
) -> list[str]:
    """Build the TorchTitan command shared by all scenarios."""
    args = [
        "./run_train.sh",
        "--module",
        workload.module,
        "--config",
        arm.config or workload.config,
        "--training.seq-len",
        str(workload.seq_len),
        "--training.steps",
        str(workload.steps),
        "--training.local-batch-size",
        str(workload.local_batch_size),
        "--compile.enable",
        "--profiler.enable_profiling",
        "--profiler.profile_freq",
        str(workload.profile_freq),
        "--profiler.profiler_active",
        str(workload.profiler_active),
        "--profiler.profiler_warmup",
        str(workload.profiler_warmup),
    ]
    if workload.seed is not None:
        args.extend(("--debug.seed", str(workload.seed)))
    if arm.override_imports:
        args.extend(("--override.imports", ",".join(arm.override_imports)))
    return args + list(extra_args) + ["--dump-folder", str(arm_dir)]


def run_text(command: list[str], *, cwd: Path | None = None) -> str:
    """Run a metadata command, returning a diagnostic instead of failing."""
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, cwd=cwd
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def hardware_metadata(
    paths: RuntimePaths, gpu: str, hardware_label: str
) -> tuple[str, dict[str, str]]:
    """Collect the hardware and source provenance stored in the manifest."""
    query = run_text(
        [
            "nvidia-smi",
            "--id=" + gpu,
            "--query-gpu=index,name,uuid,driver_version",
            "--format=csv,noheader",
        ]
    ).strip()
    metadata = {
        "requested_gpu": gpu,
        "nvidia_smi": query,
        "torch_version": run_text(
            [sys.executable, "-c", "import torch; print(torch.__version__)"],
            cwd=paths.titan_dir,
        ).strip(),
        "torchtitan_git_rev": run_text(
            ["git", "rev-parse", "HEAD"], cwd=paths.titan_dir
        ).strip(),
        "benchmarks_git_rev": run_text(
            ["git", "rev-parse", "HEAD"], cwd=paths.bench_dir
        ).strip(),
    }
    if hardware_label != "auto":
        return hardware_label, metadata
    name = query.split(",")[1].strip() if "," in query else f"gpu{gpu}"
    label = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return label, metadata


def runtime_environment(
    paths: RuntimePaths,
    gpu: str,
    *,
    compile_mode: str = "default",
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Construct the isolated single-GPU TorchTitan environment.

    ``BENCH_COMPILE_MODE`` is set unconditionally, including for the default
    mode, so a value exported in the operator's shell cannot leak into a run.
    """
    result = dict(environment or os.environ)
    pythonpath = result.get("PYTHONPATH")
    result.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "BENCH_COMPILE_MODE": compile_mode,
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTHONPATH": f"{paths.bench_dir}{':' + pythonpath if pythonpath else ''}",
            "PATH": f"{Path(sys.executable).parent}:{result['PATH']}",
            "TORCH_EXTENSIONS_DIR": result.get(
                "TORCH_EXTENSIONS_DIR", str(paths.cache_root / "torch_extensions")
            ),
            "TORCHINDUCTOR_CACHE_DIR": result.get(
                "TORCHINDUCTOR_CACHE_DIR", str(paths.cache_root / "inductor_cache")
            ),
            "TRITON_CACHE_DIR": result.get(
                "TRITON_CACHE_DIR", str(paths.cache_root / "triton_cache")
            ),
            "NGPU": "1",
        }
    )
    return result


def add_compiler_environment(
    environment: dict[str, str], compiler_env: Path | None
) -> dict[str, str]:
    """Return an environment extended by an optional compiler setup script."""
    if compiler_env is None:
        return environment.copy()
    if not compiler_env.is_file():
        raise ValueError(f"compiler environment script does not exist: {compiler_env}")

    result = environment.copy()
    output = subprocess.check_output(
        ["bash", "-c", 'source "$1" && env -0', "bash", str(compiler_env)],
        env=result,
    )
    for entry in output.decode().split("\0"):
        if entry:
            key, value = entry.split("=", 1)
            result[key] = value
    return result
