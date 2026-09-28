"""The environment of a training process or a kernel worker."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping

from benchmarks.execution.paths import RuntimePaths


def device_environment(gpu: str, *, world_size: int) -> dict[str, str]:
    """The keys that make ``gpu`` a stable PCI index and state the rank count."""
    if world_size < 1:
        raise ValueError(f"world size {world_size} must be >= 1")
    return {
        "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
        "CUDA_VISIBLE_DEVICES": gpu,
        "NGPU": str(world_size),
    }


def runtime_environment(
    paths: RuntimePaths,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """The inherited environment, with the repository on ``PYTHONPATH`` and each build cache that the caller did not set."""
    result = dict(environment or os.environ)
    pythonpath = result.get("PYTHONPATH")
    result.update(
        {
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
        }
    )
    return result


def add_compiler_environment(
    environment: dict[str, str], compiler_env: Path | None
) -> dict[str, str]:
    """``environment`` with the variables that the ``compiler_env`` script exports; a ``None`` script gives a copy."""
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
