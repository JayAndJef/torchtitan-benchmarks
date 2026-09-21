"""The e2e type declarations, in one module that imports nothing first-party.

Every other e2e module builds instances of these types, reads them or
records them. They are declarations and nothing else: no table, no
instance, no rule. The owning module keeps those.

The module imports the standard library alone. A module that only needs a
type therefore pays for no scenario table, no schedule table and no
validator, and the layering test in ``tests/test_schema.py`` holds that
property.

``Workload``, ``Arm`` and ``Scenario`` describe a scenario.
``ParallelismSpec`` and ``PipelineSchedule`` describe the parallelism run
axis. ``Engine`` joins an engine's command builder to its validation
profile, which ``benchmarks.e2e.validation`` owns. ``ResolvedRun`` is one
run with every open question answered. The run axes themselves live in
``benchmarks.e2e.axes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal


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


@dataclass(frozen=True)
class PipelineSchedule:
    """One pipeline schedule, and what a run may do with it.

    ``titan_name`` is the exact string TorchTitan's
    ``--parallelism.pipeline-parallel-schedule`` accepts. It is a separate
    field from ``name`` so that our own roster name can never be assumed to
    be the framework's: the two agree for all five registered schedules
    today, and a test pins each ``titan_name`` against
    ``torch.distributed.pipelining.schedules.get_schedule_class``'s own map.

    ``stages_per_rank`` is how many pipeline stages one rank holds under this
    schedule. It is 1 for the single-stage schedules and 2 for the
    multi-stage ones, which is TorchTitan's own default
    (``pipeline_parallel.py``: ``stages_per_rank = 1 if
    is_single_stage_schedule else 2``). Rules 7 and 12 both read it: the
    layer count has to divide by ``pp * stages_per_rank``, and a rank's
    warmup depth -- and therefore its peak activation memory -- grows with
    it.

    ``megatron_supported`` says whether **Megatron-LM** implements this
    schedule at all. Three of the five are PyTorch-only, so a run holding a
    megatron arm has no opponent for them and rule 5 refuses the
    combination.

    **It is not the same question as "can this repo's megatron driver run
    it", and rule 5 deliberately asks the library's question.**
    ``Interleaved1F1B`` is the case where the two answers differ: Megatron-LM
    implements it, so a cross-engine row is possible in principle, but
    our stock driver refuses a virtual pipeline degree.
    Rule 5 therefore lets that spec through and the driver fails it -- the
    declaration-without-a-builder pattern the kernel spans already use, where
    the failure lands at the place that owns the missing work rather than at
    a validator claiming the library cannot do it. A test pins that this
    combination passes, so nobody "fixes" it by writing a false ``False``
    here.

    ``requires_uncompiled`` records that the schedule raises on a compiled
    stage module. Only three of PyTorch's schedule classes call
    ``_check_torch_compile_compatibility``, and they are exactly the three
    marked here.
    """

    name: str
    titan_name: str
    megatron_supported: bool
    stages_per_rank: int
    requires_uncompiled: bool
    description: str


ZERO_MODES: tuple[int, ...] = (0, 1)
"""How the run holds the dense parameters.

A dense parameter is every parameter that is not a routed expert weight.
Each value names one ZeRO level, and the number is that level. ``0`` keeps
a whole copy of everything on each rank. ``1`` shards the optimizer states
alone.

Under ``1`` Megatron gets ``--use-distributed-optimizer`` alone, and
TorchTitan gets the whole data-parallel width as ``dp_shard`` plus
``--parallelism.fsdp-reshard-after-forward never``. The two engines then
move the same bytes per step: one parameter all-gather and one gradient
reduce-scatter.

Two statements this axis must not make. Communication volume does not
separate ZeRO-1 from ZeRO-2, because both move twice the parameters and
only the gradient lifetime differs. And no ZeRO level shards the activation
gradients, which belong to the ``--ac`` axis.

It is a comparability boundary at any expert degree, because it decides how
much optimizer state one rank holds and what the ranks exchange each step.
Every published number was measured under ``zero 0``, which is the default.

It is also what makes an expert degree legal, for a TorchTitan reason
rather than a preference: TorchTitan cannot split the experts while it
keeps the dense parameters replicated, because the expert mesh degree
``efsdp = dp_shard * cp * tp // ep`` needs ``dp_shard >= ep``. Megatron
holds every parity, so the two engines compare under an expert degree only
when both shard, and spec rule 14 refuses the replicated combination.
"""
DEFAULT_ZERO = 0


@dataclass(frozen=True)
class ParallelismSpec:
    """The parallelism degrees and pipeline settings for one run.

    Every field defaults to the single-GPU value, so ``ParallelismSpec()`` is
    the run this repo has always done and ``TRIVIAL_SPEC`` is that object.

    ``__post_init__`` enforces well-formedness only -- every degree is a
    positive count, and ``zero`` names a declared ZeRO level. That is
    not one of the sixteen validator rules; it is the precondition they
    assume. Without it a spec of ``dp=-1, pp=-1`` would have ``world_size``
    1 and walk past rule 1 on a one-GPU box, which is exactly the illegal
    mesh the rules exist to refuse. ``PiperShape`` guards its geometry the
    same way and for the same reason.

    **``zero`` takes the same treatment, and it must.**
    ``titan_mesh`` and ``execution_model`` are total functions over a spec
    and both branch on this value, so a spec carrying a level neither
    branch knows must not exist. A validator rule would be too late: both
    functions run on specs the validator never sees.

    ``zero`` sits here rather than beside ``--model-size`` as a
    run axis of its own, for the reason ``pp_schedule`` and
    ``pp_microbatch_size`` do: it is a treatment of one parallelism axis,
    and every function that needs it already takes the spec.
    """

    dp: int = 1
    pp: int = 1
    ep: int = 1
    pp_schedule: str | None = None
    pp_microbatch_size: int = 1
    zero: int = DEFAULT_ZERO

    def __post_init__(self) -> None:
        for field, value in (
            ("dp", self.dp),
            ("pp", self.pp),
            ("ep", self.ep),
            ("pp_microbatch_size", self.pp_microbatch_size),
        ):
            if value < 1:
                raise ValueError(
                    f"{field} must be >= 1, got {value}: a degree counts "
                    "ranks and a microbatch size counts rows"
                )
        if self.zero not in ZERO_MODES:
            raise ValueError(
                f"Unknown zero level {self.zero!r}. Available: "
                + ", ".join(str(mode) for mode in ZERO_MODES)
            )

    @property
    def world_size(self) -> int:
        """The number of ranks the run needs: ``dp * pp``.

        ``ep`` is absent on purpose. Both engines carve the expert ranks out
        of the data-parallel axis rather than adding a dimension, so ``ep``
        redistributes the ``dp`` ranks and never asks for more. Rule 9
        enforces the other half of that: ``ep`` has to divide ``dp``.
        """
        return self.dp * self.pp


@dataclass(frozen=True)
class Engine:
    """One training engine: how to launch an arm, and how to validate it.

    ``Arm.engine`` names a record of this type. The record is the one place
    that joins the two halves, so an arm cannot take one engine's command
    builder and another engine's validation profile. That pairing used to
    be two independent strings on the arm.

    ``command`` builds the argv for one arm. Every builder takes the same
    parameters, including the megatron run axes an engine may ignore, so
    the dispatcher passes one call through and no caller branches on the
    engine.

    ``is_megatron`` says whether this engine runs Megatron-LM. The
    parallelism rules and the three megatron run axes read it. It is a
    declared field rather than a name prefix: a prefix test fails open, and
    an engine that spelled the library another way would walk past a rule
    it needs.
    """

    name: str
    command: Callable[..., list[str]]
    validation: ValidationProfile
    is_megatron: bool


@dataclass(frozen=True)
class ResolvedRun:
    """One run, with every question the request left open answered.

    ``_resolve_run`` builds this record and does every refusal on the way:
    the scenario name, the arm subset, the mesh, the three megatron axes
    and the resume comparison. Whatever it returns is startable, so the
    caller reads fields instead of repeating checks.

    Attributes:
        paths: The resolved repository, cache and compiler-env locations.
        scenario: The scenario, with the size overrides already applied to
            its workload.
        arms: The arms this run starts, in the order the operator asked
            for.
        hardware: The provenance label of the output directory.
        metadata: The provenance block, including the CPU pinning.
        out_dir: Where the run writes.
        commands: One argv per arm name.
        axes: The eight global run axes, resolved.
        resumed: Whether the run continues a recorded directory.

    ``paths`` is typed ``Any`` because its type lives in
    ``benchmarks.execution.paths``, and this module imports nothing
    first-party. The alternative is to move a record of filesystem
    locations in beside the scenario declarations, where it does not
    belong: the training subprocess reads that module for one path
    constant and must not pay for the e2e types.
    """

    paths: Any
    scenario: Scenario
    arms: tuple[Arm, ...]
    hardware: str
    metadata: dict[str, str]
    out_dir: Path
    commands: dict[str, list[str]]
    axes: RunAxes
    resumed: bool
