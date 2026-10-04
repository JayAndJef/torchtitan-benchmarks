"""The parallelism of one run: the degrees, the schedule names and the rules that refuse a mesh."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.models.piper_qwen3.shape import PiperShape


@dataclass(frozen=True)
class PipelineSchedule:
    """One pipeline schedule that a run may name."""

    name: str
    stages_per_rank: int
    """The pipeline stages that one rank holds; rules 7 and 12 read it."""
    description: str


ZERO_MODES: tuple[int, ...] = (0, 1)
"""The ZeRO levels of the dense parameters: 0 replicates them, and 1 shards the optimizer states."""

DEFAULT_ZERO = 0


@dataclass(frozen=True)
class ParallelismSpec:
    """The parallelism degrees and the pipeline settings of one run."""

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
        """The number of ranks: ``dp * pp``."""
        return self.dp * self.pp


MAX_WORLD_SIZE = 8
"""The GPU budget of one run."""

MAX_PP = 8
"""The deepest pipeline that a run may ask for."""

PP_SCHEDULES: dict[str, PipelineSchedule] = {
    "1F1B": PipelineSchedule(
        name="1F1B",
        stages_per_rank=1,
        description=(
            "One forward then one backward per rank, one stage per rank. The "
            "only schedule this pass targets, and the only one both engines "
            "and piper report."
        ),
    ),
    "Interleaved1F1B": PipelineSchedule(
        name="Interleaved1F1B",
        stages_per_rank=2,
        description=(
            "1F1B over two stages per rank. It needs a larger batch than the "
            "default workload gives: two chunks put rank 0's warmup at 4, so "
            "rule 12 refuses it at batch 4."
        ),
    ),
    "InterleavedZeroBubble": PipelineSchedule(
        name="InterleavedZeroBubble",
        stages_per_rank=2,
        description=(
            "Interleaved 1F1B with the weight gradient split out to fill the "
            "bubble."
        ),
    ),
    "ZBVZeroBubble": PipelineSchedule(
        name="ZBVZeroBubble",
        stages_per_rank=2,
        description=(
            "The V-shaped zero-bubble schedule: each rank holds one stage "
            "from each end of the model."
        ),
    ),
    "DualPipeV": PipelineSchedule(
        name="DualPipeV",
        stages_per_rank=2,
        description="The V-shaped DualPipe variant.",
    ),
}
"""The schedules that a run may name, by PyTorch's own schedule names."""

PP_SCHEDULE_CHOICES: tuple[str, ...] = tuple(PP_SCHEDULES)
"""Every ``--pp-schedule`` value."""

TRIVIAL_SPEC = ParallelismSpec()
"""The single-GPU run."""


def n_microbatches(spec: ParallelismSpec, *, local_batch_size: int) -> int:
    """How many microbatches one rank's batch splits into; rule 10 makes the division exact."""
    return local_batch_size // spec.pp_microbatch_size


def device_term(spec: ParallelismSpec) -> str:
    """The device count, as the first term of an execution-model string."""
    return "single-gpu" if spec.world_size == 1 else f"{spec.world_size}-gpu"


def data_parallel_term(spec: ParallelismSpec) -> str:
    """The data-parallel degree and a ZeRO level above 0, as one execution-model term."""
    if spec.zero == 0:
        return f"dp{spec.dp}"
    return f"dp{spec.dp}-zero{spec.zero}"


def degree_terms(spec: ParallelismSpec) -> tuple[str, ...]:
    """The pipeline and expert terms of an execution-model string; a degree of 1 has no term."""
    terms = []
    if spec.pp > 1:
        terms.append(f"pp{spec.pp}-{spec.pp_schedule}")
    if spec.ep > 1:
        terms.append(f"ep{spec.ep}")
    return tuple(terms)


def describe(
    spec: ParallelismSpec, *, local_batch_size: int
) -> dict[str, object]:
    """The JSON record of ``spec`` that a manifest holds: the six fields and two derived values."""
    return {
        "dp": spec.dp,
        "pp": spec.pp,
        "ep": spec.ep,
        "pp_schedule": spec.pp_schedule,
        "pp_microbatch_size": spec.pp_microbatch_size,
        "zero": spec.zero,
        "world_size": spec.world_size,
        "n_microbatches": n_microbatches(
            spec, local_batch_size=local_batch_size
        ),
    }


def parallelism_refusals(
    spec: ParallelismSpec,
    *,
    shape: PiperShape,
    local_batch_size: int,
    device_count: int,
) -> list[str]:
    """Every numbered rule that ``spec`` breaks; a rule that needs a failed rule's value is skipped."""
    if local_batch_size < 1:
        return [
            f"local batch size {local_batch_size} must be >= 1; it is the "
            "count the microbatch split divides"
        ]
    refusals = []
    # 1
    if spec.world_size != device_count:
        refusals.append(
            f"parallelism world size {spec.world_size} (dp {spec.dp} x pp "
            f"{spec.pp}) does not match the {device_count} device(s) "
            "requested; ep borrows ranks from the dp axis and never "
            "multiplies the world size"
        )
    # 2: the pipeline half first, so the message names the cap to lift.
    if spec.pp > MAX_PP:
        refusals.append(
            f"pipeline degree {spec.pp} exceeds the supported maximum "
            f"{MAX_PP}; no run plans a deeper pipeline, and nobody has "
            "checked one"
        )
    elif spec.world_size > MAX_WORLD_SIZE:
        refusals.append(
            f"parallelism world size {spec.world_size} exceeds the "
            f"{MAX_WORLD_SIZE}-GPU budget"
        )
    # 3
    if spec.pp == 1 and spec.pp_schedule is not None:
        refusals.append(
            f"pp_schedule {spec.pp_schedule!r} was requested at pp 1, where "
            "there is no pipeline to schedule"
        )
    if spec.pp > 1 and spec.pp_schedule is None:
        refusals.append(
            f"pp {spec.pp} needs a pipeline schedule; choose one of "
            + ", ".join(PP_SCHEDULE_CHOICES)
        )
    if spec.pp == 1 and spec.pp_microbatch_size != 1:
        refusals.append(
            f"pp_microbatch_size {spec.pp_microbatch_size} was requested at "
            "pp 1, where neither engine splits the batch into microbatches"
        )
    # 4
    schedule = None
    if spec.pp_schedule is not None:
        schedule = PP_SCHEDULES.get(spec.pp_schedule)
        if schedule is None:
            refusals.append(
                f"Unknown pipeline schedule {spec.pp_schedule!r}. Available: "
                + ", ".join(PP_SCHEDULE_CHOICES)
            )
    known_stages = spec.pp_schedule is None or schedule is not None
    stages_per_rank = schedule.stages_per_rank if schedule is not None else 1
    total_stages = spec.pp * stages_per_rank
    # 7
    if known_stages and shape.n_layers % total_stages:
        refusals.append(
            f"shape {shape.name!r} has {shape.n_layers} layers, which does "
            f"not divide evenly into {total_stages} pipeline stages "
            f"(pp {spec.pp} x {stages_per_rank} stage(s) per rank)"
        )
    # 8
    if spec.ep > shape.num_experts:
        refusals.append(
            f"expert degree {spec.ep} exceeds shape {shape.name!r}'s "
            f"{shape.num_experts} experts"
        )
    elif shape.num_experts % spec.ep:
        refusals.append(
            f"shape {shape.name!r}'s {shape.num_experts} experts do not "
            f"divide evenly across expert degree {spec.ep}"
        )
    # 9
    if spec.dp % spec.ep:
        refusals.append(
            f"expert degree {spec.ep} does not divide the data-parallel "
            f"degree {spec.dp}; ep takes its ranks out of the dp axis"
        )
    # 10
    if local_batch_size % spec.pp_microbatch_size:
        refusals.append(
            f"local batch size {local_batch_size} does not divide evenly "
            f"into microbatches of {spec.pp_microbatch_size}"
        )
    else:
        microbatches = n_microbatches(spec, local_batch_size=local_batch_size)
        # 11: both engines derive one microbatch group size only then.
        if microbatches % spec.pp:
            refusals.append(
                f"{microbatches} microbatches do not divide evenly across "
                f"pipeline degree {spec.pp}; the two engines would derive "
                "different microbatch group sizes and run different schedules"
            )
        # 12: below twice the stage count, 1F1B holds as much as GPipe.
        if known_stages and spec.pp > 1 and microbatches < 2 * total_stages:
            refusals.append(
                f"{microbatches} microbatches is below the {2 * total_stages} "
                f"that pp {spec.pp} x {stages_per_rank} stage(s) per rank needs "
                "for 1F1B to hold less than GPipe; raise --batch or lower "
                "--pp-microbatch-size"
            )
    # 14
    if spec.ep > 1 and spec.zero == 0:
        refusals.append(
            f"expert degree {spec.ep} needs --zero 1. TorchTitan cannot "
            "split the experts and keep the dense parameters replicated, "
            f"so under --zero {spec.zero} it would shard them while "
            "Megatron replicates them. The row would carry two changes "
            "rather than one"
        )
    return refusals


def zero_warnings(spec: ParallelismSpec) -> tuple[str, ...]:
    """What a reader must not conclude from this mesh, for every engine."""
    if spec.zero != 0 and spec.dp == 1:
        return (
            f"--zero {spec.zero} was requested at dp 1. "
            "The shard degree is 1 there, whatever the level says. Megatron "
            "shards the optimizer states over one rank and saves nothing, "
            "and TorchTitan skips its data-parallel path. So this run holds "
            "the dense parameters exactly as a replicated run holds them. "
            "Do not read this cell as a measurement of the sharded parity",
        )
    return ()
