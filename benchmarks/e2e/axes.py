"""The run axes, as the operator asked for them and as the runner resolved them.

``RequestedAxes`` is what the command line produced, with ``None`` for every
axis the operator left alone. ``RunAxes`` is the same eight axes after
``_resolve_run`` answered each one, and it is what the manifest records.
``RunRequest`` carries both, beside the paths and the overrides of one run.

The module sits above ``benchmarks.e2e.parallelism``, because a spec is one
of the axes, and below ``benchmarks.artifacts.manifests``, which reads the
resolved record.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from benchmarks.e2e.parallelism import ParallelismSpec


@dataclass(frozen=True)
class RequestedAxes:
    """The global run axes, as the operator asked for them.

    Every field is ``None``-able, and ``None`` means "not requested". A
    resume then inherits the value the manifest recorded, and a fresh run
    takes the axis default. ``_resolve_run`` reads this record and produces
    a ``RunAxes``, in which no field is ``None``-able any more.

    The two types are separate on purpose. One record can state "the
    operator said nothing" and the other cannot, so a resolved axis set
    cannot carry an unanswered question into the manifest.
    """

    ac_mode: str | None = None
    model_size: str | None = None
    parallelism: ParallelismSpec | None = None
    megatron_p2p_sync: str | None = None
    megatron_nan_guard: str | None = None
    megatron_precision: str | None = None
    profile: bool | None = None
    warmup_steps: int | None = None


@dataclass(frozen=True)
class RunAxes:
    """The global run axes of one run, every one of them resolved.

    These eight values are the comparability boundaries of a run. The
    manifest records each under its own key, and the resume check compares
    each one, so two directories that disagree about any of them are not
    one run.

    **No field has a default, and none may gain one.** A defaulted field
    would let a writer record ``dp 1 x pp 1`` for a run of any mesh, or
    ``stock`` for a run that held ten bytes per parameter. Grouping the
    axes in one record makes that rule one rule rather than eight.

    Attributes:
        ac_mode: The activation-checkpointing treatment of every arm.
        model_size: The canonical model shape name.
        parallelism: The mesh degrees and the pipeline settings.
        megatron_p2p_sync: Whether Megatron synchronizes after a pipeline
            message.
        megatron_nan_guard: Whether stock Megatron keeps its NaN and Inf
            check.
        megatron_precision: How stock Megatron holds the optimizer state.
        profile: Whether the run collects profiler traces.
        warmup_steps: Steps an unprofiled run discards before it measures.
            ``None`` under ``profile``, where the profiler schedule decides
            the sample set instead.
    """

    ac_mode: str
    model_size: str
    parallelism: ParallelismSpec
    megatron_p2p_sync: str
    megatron_nan_guard: str
    megatron_precision: str
    profile: bool
    warmup_steps: int | None


@dataclass(frozen=True)
class RunRequest:
    """User-selected inputs for one benchmark execution.

    ``axes`` carries the eight global run axes. A resume inherits each
    unanswered one from the manifest, except ``parallelism``.

    **A resume does not inherit the parallelism spec, and that is not an
    oversight.** The other axes are single values, so a resume can read one
    back and rebuild the run from it. A spec is six fields that together
    decide every arm's command line, and ``--resume`` compares no command
    line -- so a reconstruction that dropped one field would relaunch the
    arms differently and the gate would not see it. Omitting the flags on a
    resume therefore asks for the trivial spec, which matches a single-GPU
    directory and is refused against any other.

    Attributes:
        gpu: The ``<gpu>`` positional, kept exactly as the operator typed
            it. It names a device set, which ``parse_devices`` splits, but
            the string itself is never rewritten: manifests record it as
            ``hardware_metadata.requested_gpu`` and ``CUDA_VISIBLE_DEVICES``
            takes the same value.
        scenario_name: The requested scenario. There is no default.
            ``None`` means "not requested", which only a resume may leave
            unanswered, because the recorded manifest names the scenario
            there. ``_resolve_run`` refuses ``None`` in every other case.
        arm_names: An ordered subset of the scenario arms, matching the
            repeated ``run --arm NAME`` options exactly. Empty means every
            arm.
        axes: The eight global run axes, each one answered or left
            unrequested.
        occurrence: Which run of this scenario name this is, counting from
            1. A ``run`` may name one scenario twice, and the second and
            later occurrences need an output directory of their own.
    """

    gpu: str
    scenario_name: str | None = None
    arm_names: tuple[str, ...] = ()
    hardware: str = "auto"
    out_dir: Path | None = None
    resume_dir: Path | None = None
    seq_len: int | None = None
    steps: int | None = None
    batch: int | None = None
    extra_args: tuple[str, ...] | None = None
    torchtitan_args: tuple[str, ...] = ()
    megatron_args: tuple[str, ...] = ()
    timestamp: str | None = None
    occurrence: int = 1
    cache_root: Path | None = None
    compiler_env: Path | None = None
    axes: RequestedAxes = RequestedAxes()

    def __post_init__(self) -> None:
        if self.extra_args is not None and not self.torchtitan_args:
            object.__setattr__(self, "torchtitan_args", self.extra_args)
        elif self.torchtitan_args and self.extra_args is None:
            object.__setattr__(self, "extra_args", self.torchtitan_args)

