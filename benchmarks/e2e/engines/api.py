"""The records every engine reads, and the interface every engine implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import ClassVar, Literal

from benchmarks.e2e.parallelism import ParallelismSpec
from benchmarks.models.piper_qwen3.shape import PiperShape


class CompileMode(str, Enum):
    """The compile treatment of one arm."""

    TORCH = "torch"
    NONE = "none"
    __str__ = str.__str__


@dataclass(frozen=True, kw_only=True)
class EngineConfig:
    """The base of every engine's arm config; the config type selects the engine."""

    extra_flags: tuple[str, ...] = ()
    """Engine flags that the engine's check classifies before they pass through."""


@dataclass(frozen=True)
class Arm:
    """One measured implementation: a name, a description and an engine config."""

    name: str
    description: str
    config: EngineConfig


@dataclass(frozen=True)
class DataSpec:
    """The training data of one run."""

    dataset: str
    seq_len: int
    local_batch_size: int
    steps: int
    """The step count, which also sets the samples each rank materializes."""


@dataclass(frozen=True)
class ProfileWindow:
    """The profiler schedule of a profiled run."""

    freq: int = 20
    warmup: int = 5
    active: int = 5
    min_windows: int = 2


@dataclass(frozen=True)
class RunSpec:
    """The facts every arm of one run shares; no engine config overrides them."""

    shape: PiperShape
    data: DataSpec
    parallelism: ParallelismSpec
    ac_mode: Literal["sac", "none"]
    profile: bool
    window: ProfileWindow
    warmup_steps: int | None
    """The steps an unprofiled run discards; ``None`` exactly when ``profile`` is true."""
    seed: int | None = None

    def __post_init__(self) -> None:
        if (self.warmup_steps is None) != self.profile:
            raise ValueError(
                f"warmup_steps {self.warmup_steps!r} does not match profile "
                f"{self.profile}: a profiled run takes no warmup step count, "
                "and an unprofiled run needs one"
            )


class Engine(ABC):
    """One training engine, as the harness process sees it."""

    name: ClassVar[str]
    """The engine name that a manifest records."""
    config_type: ClassVar[type[EngineConfig]]

    @abstractmethod
    def check(self, run: RunSpec, arm: Arm) -> list[str]:
        """Every reason this arm cannot run; each one names its repair."""

    @abstractmethod
    def command(self, run: RunSpec, arm: Arm, arm_dir: Path) -> list[str]:
        """The command line that trains the arm, without the CPU pinning prefix."""

    @abstractmethod
    def validate(
        self, run: RunSpec, arm: Arm, arm_dir: Path, log_path: Path
    ) -> None:
        """Raise ``RuntimeError`` when the arm's log or traces refuse its numbers."""
