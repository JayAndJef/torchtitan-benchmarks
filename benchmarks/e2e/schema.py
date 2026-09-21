"""What a scenario declaration is made of: ``Workload``, ``Arm``, ``Scenario``.

These three are the declarative vocabulary, and ``benchmarks.e2e.registry``
builds the one scenario from them. They are declarations and nothing else:
no table, no instance, no rule.

The module imports the standard library alone, so a module that needs only
a type pays for no scenario table and no validator. The layering test in
``tests/test_schema.py`` holds that property.

Every other e2e record belongs to the module that builds it: the
parallelism spec and the pipeline schedule to
``benchmarks.e2e.parallelism``, the run axes to ``benchmarks.e2e.axes``,
the validation profile to ``benchmarks.e2e.validation``, the engine record
to ``benchmarks.e2e.engines`` and the resolved run to
``benchmarks.e2e.runner``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Workload:
    """Training settings shared by every arm in one scenario.

    ``replay_dataloader`` records that every TorchTitan arm of the scenario
    uses ``benchmarks.e2e.data.piper_qwen3``'s replay loader, whose
    materialized sample count must track ``steps``; the runner then delivers
    ``--dataloader.replay-steps`` alongside ``--training.steps``. Declared on
    the workload rather than sniffed from the config name because
    ``--model-size`` suffixes that name.
    """

    module: str
    config: str
    seq_len: int
    steps: int
    local_batch_size: int
    profile_freq: int = 20
    profiler_warmup: int = 5
    profiler_active: int = 5
    min_trace_windows: int = 2
    seed: int | None = None
    replay_dataloader: bool = False


@dataclass(frozen=True)
class Arm:
    """One implementation measured by a scenario.

    ``description`` is the one-line answer to "what is this arm?", shown by
    the ``scenarios`` command and recorded in the manifest. ``config``
    selects an arm-specific trainer config when the implementation
    difference must be expressed while building the model rather than as an
    override. When unset, the scenario workload config is used.

    ``compile`` is the compile treatment of this arm, and it has no
    default: every arm states it. ``"torch"`` asks for whole-block
    ``torch.compile``, ``"none"`` runs the blocks eager. It is an arm
    property and not a run axis, because two arms of one run may differ in
    it -- that difference is what the ``engines`` scenario measures. The
    manifest records it through ``asdict(arm)``, and validation rule 8
    reads it: the compile log line must be present under ``"torch"`` and
    absent under ``"none"``.

    ``engine`` names the record that runs the arm. One string carries both
    the command builder and the validation profile, so an arm cannot take
    one engine's argv and another engine's log rules.

    Attributes:
        overrides_per_block: The ``[Override]`` log lines expected per
            transformer block. ``validate_arm`` multiplies this by the
            shape's layer count, so one arm stays correct at every
            ``--model-size``.
        engine: The name of a record in ``benchmarks.e2e.engines``'
            ``ENGINES``. A plain string keeps ``asdict(arm)``
            JSON-serializable for the manifest.
    """

    name: str
    description: str
    compile: Literal["torch", "none"]
    config: str | None = None
    override_imports: tuple[str, ...] = ()
    overrides_per_block: int = 0
    trace_kernel_markers: tuple[str, ...] = ()
    requires_gcc_toolset: bool = False
    engine: str = "torchtitan"


@dataclass(frozen=True)
class Scenario:
    """A reproducible workload and its comparable implementation arms.

    ``supported_ac_modes`` restricts the global ``--ac`` axis: a scenario
    whose arms cannot honor a mode (e.g. an engine with no SAC-parity
    recompute) lists only the modes it supports. A ``run`` over every
    scenario skips an unsupported combination, and a ``--scenario`` that
    names the scenario errors.
    """

    name: str
    description: str
    workload: Workload
    arms: tuple[Arm, ...]
    supported_ac_modes: tuple[str, ...] = ("sac", "none")

    def arm(self, name: str) -> Arm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise ValueError(
            f"Unknown arm {name!r} for scenario {self.name!r}. "
            f"Available arms: {', '.join(arm.name for arm in self.arms)}"
        )
