"""The environment a training or kernel-worker subprocess is launched with.

Two builders, in the order the runners call them.

``runtime_environment`` constructs the isolated single-GPU environment every
arm of a run shares. Three of its entries are load-bearing rather than
cosmetic: ``CUDA_DEVICE_ORDER=PCI_BUS_ID`` with ``CUDA_VISIBLE_DEVICES`` is
what makes the CLI's ``<gpu>`` argument a stable PCI index rather than a
driver-enumeration accident; ``NGPU=1`` pins the run single-GPU; and
``PYTHONPATH`` gets the repository root prepended, which is how the training
subprocess resolves ``--module benchmarks.models.piper_qwen3`` and every
``--override.imports`` path. (Prepended, not replaced -- an inherited
``PYTHONPATH`` survives behind it.) The three build-cache directories are
set from ``RuntimePaths.cache_root`` only if the caller has not already set
them, so ``--cache-root`` is a default and not an override.

That ``PYTHONPATH`` line is one half of the ``benchmarks`` name-shadowing
hazard: the training subprocess also runs with ``cwd`` inside the torchtitan
submodule, which ``python -m`` puts at ``sys.path[0]`` *ahead* of anything
here. ``tests/test_import_boundaries.py`` names this module for that reason;
see its ``BenchmarksNameShadowingTest``.

``add_compiler_environment`` then layers a sourced shell script on top for
the arms whose CUDA extension needs a C++20 host compiler. It sources the
script in a real ``bash`` and reads back ``env -0``, because the toolset
scripts are shell, not a key/value file. It is applied per arm (``e2e``) or
per scenario (``kernel-bench``) rather than to the base environment, so a
toolset that rewrites ``PATH`` or ``LD_LIBRARY_PATH`` cannot silently alter
an arm that never asked for it. A missing script raises: the alternative is
an arm that reports a JIT-build failure forty seconds into a run.

Neither function knows what a scenario is, and neither records anything --
the manifest's provenance block is ``provenance.py``, the ``numactl`` prefix
is ``affinity.py``, and the locations both of those and this one consume are
``paths.py``.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping

from benchmarks.execution.paths import RuntimePaths


def runtime_environment(
    paths: RuntimePaths,
    gpu: str,
    *,
    environment: Mapping[str, str] | None = None,
) -> dict[str, str]:
    """Construct the isolated single-GPU TorchTitan environment."""
    result = dict(environment or os.environ)
    pythonpath = result.get("PYTHONPATH")
    result.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
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
