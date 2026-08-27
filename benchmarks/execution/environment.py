"""The environment a training or kernel-worker subprocess is launched with.

Two builders, in the order the runners call them.

``runtime_environment`` constructs the isolated environment every arm of a
run shares. Four of its entries are load-bearing rather than cosmetic:
``CUDA_DEVICE_ORDER=PCI_BUS_ID`` with ``CUDA_VISIBLE_DEVICES`` is what makes
the CLI's ``<gpu>`` argument a stable PCI index rather than a
driver-enumeration accident; ``NGPU`` states how many ranks the run starts;
``LOG_RANK`` decides whose output reaches the log at all, and above one rank
it names every rank rather than TorchTitan's default of only the first;
and ``PYTHONPATH`` gets the repository root prepended, which is how the
training subprocess resolves ``--module benchmarks.models.piper_qwen3`` and
every ``--override.imports`` path. (Prepended, not replaced -- an inherited
``PYTHONPATH`` survives behind it.) The three build-cache directories are
set from ``RuntimePaths.cache_root`` only if the caller has not already set
them, so ``--cache-root`` is a default and not an override.

``world_size`` defaults to 1, which is the value every published number was
taken at and the only value ``kernel-bench`` ever asks for: a kernel worker
holds one arm in one process. The end-to-end runner passes
``ParallelismSpec.world_size`` instead, so ``NGPU`` follows the requested
mesh rather than the device count -- the two agree, because rule 1 of
``benchmarks/e2e/parallelism.py`` refuses a spec that does not fill the
device list. The parameter is an ``int`` rather than the spec, so this
module stays free of the end-to-end axis.

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


# What a multi-rank run adds, and nothing else does.
#
# ``LOG_RANK`` is TorchTitan's own variable: ``run_train.sh`` forwards it to
# ``torchrun --local-ranks-filter``, and it defaults to ``0`` there -- so
# without this every rank but the first writes to a console nobody reads,
# and a kernel that degraded on rank 1 alone is invisible.
#
# ``TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE`` names the **global** rank on
# every tee'd line. torchrun's own default is
# ``[${role_name}${local_rank}]:``, which renders the same shape on one node
# and a colliding one across nodes. ``benchmarks.artifacts.layout``'s
# ``logs_by_rank`` reads what this produces.
#
# **Neither is set at world size 1**, so a single-GPU log is byte for byte
# the log this repo has always written -- torchrun still prefixes it
# ``[rank0]:`` through its own default, and ``logs_by_rank`` returns such a
# log whole.
LOG_RANK_TEMPLATE = "[rank${rank}]:"


def _rank_logging(world_size: int) -> dict[str, str]:
    if world_size == 1:
        return {}
    return {
        "LOG_RANK": ",".join(str(rank) for rank in range(world_size)),
        "TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE": LOG_RANK_TEMPLATE,
    }


# The one host variable that stops a run before it trains a step.
#
# ``megatron/training/arguments.py`` asserts
# ``os.environ.get('CUDA_DEVICE_MAX_CONNECTIONS') != "1"`` under
# ``--use-megatron-fsdp``. ``runtime_environment`` copies the host
# environment into the child, so an ambient ``CUDA_DEVICE_MAX_CONNECTIONS=1``
# reaches Megatron and the sharded arm dies at argument parsing -- after the
# output directory exists and after every rank has started.
#
# **Nothing under ``benchmarks/`` sets the variable, and that is not the
# hazard.** The hazard is the operator's own shell: the variable is a common
# NCCL and tensor-parallel tuning knob, and a session that exported it once
# would fail every sharded cell of a matrix for a reason no log explains.
#
# **The check refuses; it does not repair.** Unsetting the variable would
# change how the driver schedules work on every rank, which moves the
# measurement rather than the record. The operator unsets it and says so.
CONNECTION_LIMIT_VARIABLE = "CUDA_DEVICE_MAX_CONNECTIONS"
CONNECTION_LIMIT_REFUSED_VALUE = "1"


def refuse_megatron_fsdp_connection_limit(
    environment: Mapping[str, str], *, megatron_fsdp: bool
) -> None:
    """Refuse a host that would make Megatron-FSDP fail at argument parsing.

    ``megatron_fsdp`` says whether any command line this run will start
    carries ``--use-megatron-fsdp``. The caller reads that off the built
    argv rather than re-deriving it from the parallelism spec, so this
    refusal cannot drift from the flag list that produces the flag.
    """
    if not megatron_fsdp:
        return
    value = environment.get(CONNECTION_LIMIT_VARIABLE)
    if value != CONNECTION_LIMIT_REFUSED_VALUE:
        return
    raise ValueError(
        f"{CONNECTION_LIMIT_VARIABLE}={value!r} in this shell, and this run "
        "sends --use-megatron-fsdp. Megatron asserts the variable is not "
        f"'1' under that flag, so every rank would die at argument parsing. "
        f"Unset {CONNECTION_LIMIT_VARIABLE} and start the run again"
    )


def runtime_environment(
    paths: RuntimePaths,
    gpu: str,
    *,
    environment: Mapping[str, str] | None = None,
    world_size: int = 1,
) -> dict[str, str]:
    """Construct the isolated TorchTitan environment for one run."""
    if world_size < 1:
        raise ValueError(f"world size {world_size} must be >= 1")
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
            "NGPU": str(world_size),
            **_rank_logging(world_size),
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
