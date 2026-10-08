"""Turn an engine's launch into the command line and the child environment."""

from __future__ import annotations

import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from benchmarks.e2e.engines.api import Launch
from benchmarks.execution.affinity import CpuPinning
from benchmarks.execution.environment import device_environment


TORCHRUN_MODULE = "benchmarks.execution.torchrun"
"""The module that runs torchrun with a tee that writes whole lines."""

LOG_RANK_TEMPLATE = "[rank${rank}]:"
"""The prefix that torchrun puts on every line a rank writes."""

RANK_PREFIX = r"\[rank\d+\]:"
"""The regular expression of that prefix; a torn write can put one inside the line of another rank."""

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


def holds_rank_prefix(text: str) -> bool:
    """Whether ``text`` holds a rank prefix."""
    return re.search(RANK_PREFIX, text) is not None


def own_line(rank: int, line: str) -> str:
    """``line`` without the prefix of ``rank``, which a log of one rank keeps at the start of each line."""
    own = LOG_RANK_TEMPLATE.replace("${rank}", str(rank))
    return line[len(own) :] if line.startswith(own) else line


@dataclass(frozen=True)
class LaunchedCommand:
    """The processes of one arm, as the process runner starts them."""

    argv: tuple[str, ...]
    env: Mapping[str, str]
    """The whole child environment."""
    cpu_pinning: str
    """The pinning prefix, the host's reason for no prefix, or ``PINNING_DECLINED``."""
    cwd: Path | None
    """The working directory of the processes; ``None`` keeps the working directory of the harness."""


def torchrun_flags(world_size: int) -> tuple[str, ...]:
    """The torchrun arguments; ``-u`` makes the tee threads write each line in one call."""
    return (
        "-u",
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
    """The argv: the pinning prefix, the interpreter, torchrun, then the target.

    Raises ``ValueError`` when the launch sets a key that the launcher owns.
    """
    collisions = sorted(set(launch.env) & LAUNCHER_KEYS)
    if collisions:
        raise ValueError(
            f"the launch sets {', '.join(collisions)}, which the launcher owns; "
            "remove the keys from Launch.env"
        )
    if world_size < 1:
        raise ValueError(f"world size {world_size} must be >= 1")
    prefix = pinning.prefix if launch.pin else ()
    starter = torchrun_flags(world_size) if launch.processes == "per_rank" else ()
    return (*prefix, sys.executable, *starter, *launch.target)


def launcher_environment(
    launch: Launch, *, world_size: int, gpu: str
) -> dict[str, str]:
    """The launcher's keys for one launch; a single process gets no rank logging."""
    result = {
        **device_environment(gpu, world_size=world_size),
        "PYTORCH_ALLOC_CONF": ALLOCATOR_POLICY,
    }
    if launch.processes == "per_rank":
        result["LOG_RANK"] = ",".join(str(rank) for rank in range(world_size))
        result["TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE"] = LOG_RANK_TEMPLATE
    return result


def environment_delta(
    launch: Launch, *, world_size: int, gpu: str
) -> dict[str, str]:
    """The keys that the launcher and the engine set on top of the inherited environment."""
    return {
        **launcher_environment(launch, world_size=world_size, gpu=gpu),
        **launch.env,
    }


def pinning_record(launch: Launch, pinning: CpuPinning) -> str:
    """The pinning of one launch: the host's description, or ``PINNING_DECLINED``."""
    return pinning.description if launch.pin else PINNING_DECLINED


def build_command(
    launch: Launch,
    *,
    world_size: int,
    gpu: str,
    pinning: CpuPinning,
    base_env: Mapping[str, str],
) -> LaunchedCommand:
    """The command line, the child environment and the working directory of one launch."""
    argv = command_line(launch, world_size=world_size, pinning=pinning)
    inherited = {
        key: value for key, value in base_env.items() if key not in LAUNCHER_KEYS
    }
    return LaunchedCommand(
        argv=argv,
        env=MappingProxyType(
            {**inherited, **environment_delta(launch, world_size=world_size, gpu=gpu)}
        ),
        cpu_pinning=pinning_record(launch, pinning),
        cwd=launch.cwd,
    )
