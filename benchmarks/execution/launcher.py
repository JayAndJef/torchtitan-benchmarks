"""Turn an engine's launch into the command line and the child environment."""

from __future__ import annotations

import sys
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from benchmarks.e2e.engines.api import Launch
from benchmarks.execution.affinity import CpuPinning
from benchmarks.execution.environment import device_environment


TORCHRUN_MODULE = "torch.distributed.run"
"""The module name of torchrun, which the harness interpreter runs."""

LOG_RANK_TEMPLATE = "[rank${rank}]:"
"""The prefix that torchrun puts on every line a rank writes."""

ALLOCATOR_POLICY = "expandable_segments:True"
"""The CUDA allocator policy of every training process."""

LAUNCHER_KEYS = frozenset(
    {
        "CUDA_DEVICE_ORDER",
        "CUDA_VISIBLE_DEVICES",
        "NGPU",
        "LOG_RANK",
        "TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE",
        "PYTORCH_ALLOC_CONF",
    }
)
"""The environment keys that the launcher sets; a launch that sets one is refused."""

PINNING_DECLINED = "declined by engine"
"""The pinning record of a launch whose engine refuses the CPU pinning prefix."""


@dataclass(frozen=True)
class LaunchedCommand:
    """The processes of one arm, as the process runner starts them."""

    argv: tuple[str, ...]
    env: Mapping[str, str]
    """The whole child environment."""
    cpu_pinning: str
    """The pinning prefix, the host's reason for no prefix, or ``PINNING_DECLINED``."""


def torchrun_flags(world_size: int) -> tuple[str, ...]:
    """The torchrun arguments: one process per rank, and every rank's output teed."""
    return (
        "-m",
        TORCHRUN_MODULE,
        f"--nproc-per-node={world_size}",
        "--rdzv-backend",
        "c10d",
        "--rdzv-endpoint",
        "localhost:0",
        "--local-ranks-filter",
        ",".join(str(rank) for rank in range(world_size)),
        "--role",
        "rank",
        "--tee",
        "3",
    )


def command_line(
    launch: Launch, *, world_size: int, pinning: CpuPinning
) -> tuple[str, ...]:
    """The argv: the pinning prefix, the interpreter, torchrun, then the target."""
    if world_size < 1:
        raise ValueError(f"world size {world_size} must be >= 1")
    prefix = pinning.prefix if launch.pin else ()
    starter = torchrun_flags(world_size) if launch.processes == "per_rank" else ()
    return (*prefix, sys.executable, *starter, *launch.target)


def launcher_environment(
    launch: Launch, *, world_size: int, gpu: str
) -> dict[str, str]:
    """The values of ``LAUNCHER_KEYS`` for one launch."""
    result = {
        **device_environment(gpu, world_size=world_size),
        "PYTORCH_ALLOC_CONF": ALLOCATOR_POLICY,
    }
    if launch.processes == "per_rank":
        result["LOG_RANK"] = ",".join(str(rank) for rank in range(world_size))
        result["TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE"] = LOG_RANK_TEMPLATE
    return result


def build_command(
    launch: Launch,
    *,
    world_size: int,
    gpu: str,
    pinning: CpuPinning,
    base_env: Mapping[str, str],
) -> LaunchedCommand:
    """The command line and the child environment of one launch."""
    collisions = sorted(set(launch.env) & LAUNCHER_KEYS)
    if collisions:
        raise ValueError(
            f"the launch sets {', '.join(collisions)}, which the launcher owns; "
            "remove the keys from Launch.env"
        )
    return LaunchedCommand(
        argv=command_line(launch, world_size=world_size, pinning=pinning),
        env=MappingProxyType(
            {
                **base_env,
                **launcher_environment(launch, world_size=world_size, gpu=gpu),
                **launch.env,
            }
        ),
        cpu_pinning=pinning.description if launch.pin else PINNING_DECLINED,
    )
