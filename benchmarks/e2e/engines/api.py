"""The records every engine reads, and the interface every engine implements."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType
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
    """Engine flags that pass through after the engine classifies them."""


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


@dataclass(frozen=True)
class Launch:
    """What one arm's training processes are."""

    target: tuple[str, ...]
    """The arguments after the interpreter; the first one is ``-m``."""
    processes: Literal["per_rank", "single"]
    """``per_rank`` starts one process per rank under torchrun; ``single`` starts one."""
    pin: bool
    """Whether the engine accepts the CPU pinning prefix."""
    env: Mapping[str, str] = field(default_factory=lambda: MappingProxyType({}))
    """The engine's own child environment keys; a launcher-owned key is refused."""
    host_compiler: bool = False
    """Whether the processes need the environment of the ``--compiler-env`` script."""
    cwd: Path | None = None
    """The working directory of the processes; ``None`` keeps the working directory of the harness."""

    def __post_init__(self) -> None:
        if self.target[:1] != ("-m",):
            raise ValueError(
                f"a launch target starts with '-m', got {self.target[:2]!r}"
            )
        if self.processes not in ("per_rank", "single"):
            raise ValueError(
                f"launch processes {self.processes!r} is not one of: per_rank, single"
            )


@dataclass(frozen=True)
class StepSample:
    """The figures that one rank logs for one training step."""

    rank: int
    step: int
    tokens_per_second: float
    """One rank's tokens divided by the step time and by ``cp * tp * pp``."""
    peak_memory_gib: float
    """The peak reserved memory since the previous step."""
    loss: float | None
    """``None`` on a rank that holds no loss."""
    grad_norm: float | None
    """``None`` when the engine cannot state the norm."""
    extras: Mapping[str, float] = field(default_factory=lambda: MappingProxyType({}))
    """Other figures of the step, under the names that the engine gives them."""


@dataclass(frozen=True)
class MeshObserved:
    """The mesh that an engine built, as its own log states it."""

    dp: int
    pp: int
    ep: int
    zero: int | None
    """The ZeRO level; ``None`` when the log cannot state it, which is the case at one data-parallel rank."""


@dataclass(frozen=True)
class RankEvidence:
    """The facts behind the published numbers of one rank, as the engine reads them from that rank's log."""

    rank: int
    completed: bool
    param_count: int | None
    """The whole-model parameter count; ``None`` when the rank does not state it."""
    mesh: MeshObserved | None
    """``None`` only when the rank printed nothing."""


class Engine(ABC):
    """One training engine, as the harness process sees it."""

    name: ClassVar[str]
    """The engine name that a manifest records."""
    config_type: ClassVar[type[EngineConfig]]

    @abstractmethod
    def check(self, run: RunSpec, arm: Arm) -> list[str]:
        """Every reason this arm cannot run; each one names its repair."""

    @abstractmethod
    def launch(self, run: RunSpec, arm: Arm, arm_dir: Path) -> Launch:
        """The training processes of the arm; the same inputs give the same launch."""

    @abstractmethod
    def execution_model(self, run: RunSpec, arm: Arm) -> str:
        """How the arm's processes hold the model state, as one manifest string."""

    def warnings(self, run: RunSpec, arm: Arm) -> list[str]:
        """What a reader must not conclude from this arm's numbers at this mesh."""
        return []

    @abstractmethod
    def read_steps(self, rank: int, text: str) -> list[StepSample]:
        """The step samples in one rank's log ``text``, in step order; a step line that does not parse raises ``ValueError``."""

    @abstractmethod
    def read_evidence(self, rank: int, text: str) -> RankEvidence:
        """The evidence in one rank's log ``text``; a log that states two values of one fact raises."""

    @abstractmethod
    def validate(
        self,
        run: RunSpec,
        arm: Arm,
        arm_dir: Path,
        rank_logs: Mapping[int, str],
    ) -> list[str]:
        """Every engine rule that the arm's logs or traces break; each string names the rank."""
