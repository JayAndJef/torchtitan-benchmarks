"""The parallelism run axis: the degrees, the schedules and the validator.

One ``ParallelismSpec`` describes a whole run, exactly as ``--model-size``
and ``--ac`` each describe one. Every arm in one run
shares it, so the world size, the pipeline schedule and the microbatch count
are properties of the run rather than of an arm.

The CLI, runner, command builders, manifest writer, validation and
evaluation all import this module. The spec is resolved and validated in the parent
before any host probe, then the same object builds both engines' command lines
and the manifest record. The rules remain torch-free and directly testable on
a CPU.

This is e2e-only. ``benchmarks/models/piper_qwen3/shape.py`` sits under
``models/`` because two engines build from it; a parallelism degree has one
reader, the e2e run. ``kernel-bench`` runs one process and never needs it,
and the TorchTitan training subprocess never needs it either, because
``parallelize_piper1b`` reads the ``ParallelDims`` TorchTitan builds from the
command line.

``ParallelismSpec`` and ``PipelineSchedule`` are declared in
``benchmarks.e2e.schema``, with ``Workload``. This module holds the
schedule table, the derived values and the rules, and it imports no
scenario declaration: ``benchmarks/e2e/registry.py`` reads this module and
not the other way round.

Terms, used here with these meanings only:

rank
    One process. It owns one GPU.
world size
    The number of ranks in one job. Here it is ``dp * pp``.
degree
    The number of ranks one axis uses.
DP
    Data parallelism. Each rank reads different data and the ranks share the
    gradients.
PP
    Pipeline parallelism. Each rank holds different layers.
EP
    Expert parallelism. Each rank holds different MoE experts.
stage
    The layers one PP rank holds. A schedule may give a rank more than one.
microbatch
    One piece of a rank's batch. The pipeline moves one microbatch at a time.
bubble
    Idle time on a rank, caused by the start and the end of the pipeline.

**EP does not multiply into the world size.** Both engines take the expert
ranks out of the data-parallel axis rather than adding a fourth dimension.
Megatron subdivides its DP group (``parallel_state.py``). TorchTitan states
the constraint as ``dp_shard * cp * tp == efsdp * ep`` (``configs.py``), and
``titan_mesh`` satisfies it under either sharded value by giving the whole
data-parallel width to ``dp_shard``. So ``dp`` is the whole data-parallel
width and ``ep`` is a split of it, which is what ``titan_mesh`` and rule 9
encode.

**TP and CP are deliberately absent.** They are out of scope for this pass,
and a field nobody can set misleads a later reader into thinking the axis is
supported. Adding one means adding it to ``world_size``, to
``execution_model`` and to the throughput divisor at the same time.

**What this module refuses today.** Rule 14 refuses ``ep > 1`` at ZeRO
level 0, and rules 5 and 6 refuse three of the five registered schedules
for every cross-engine run. **The numbering keeps a gap at 13, at 15, at
16 and at 17.** Rule 15 refused a sharded level at ``dp`` 1. It now warns
instead, because it blocked ``dp 1 x pp 8``, which is the agreed 30B-A3B
matrix, and because no engine refuses that mesh. ``zero_warnings`` carries
the warning. The numbers of the deleted rules stay empty: messages, tests
and the agent guide all name the rules that remain.

Read a registered schedule as a declaration, never as a measurement: only
``1F1B`` is targeted, and the caps admit up to ``pp 8``. Real ``pp2``,
``dp2``, and ``dp2 x pp2`` correctness runs have passed on both engines, and
one ``dp 2 x pp 4`` cell at world size 8 has completed on the stock arm.
**No run has used ``pp 8``, and no run has used ``zero 1``.** The caps
admit both; that is a declaration and not evidence.
No parallel timing is citable, because the cells that ran were on a loaded
host and nobody repeated them on an idle one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from benchmarks.e2e.schema import Workload
from benchmarks.models.piper_qwen3.shape import PiperShape


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


MAX_WORLD_SIZE = 8
MAX_PP = 8
"""The GPU budget for this pass, and the pipeline depth it targets.

Named so that lifting either is one edit here rather than a hunt through
the rules. The budget is 8 because this host holds 8 devices, and the depth
is 8 because one pipeline of eight stages is the deepest split those
devices hold. Neither number is a property of an engine.

The two caps are equal, which makes rule 2's order load-bearing: the world
size is ``dp * pp``, so every spec above the pipeline cap is also above the
world-size cap. The pipeline half runs first, because its message names the
cap somebody has to lift. Do not delete that half to reach the same
verdict.

Before you run a V-shaped schedule, repair ``loss_visible_rank`` in the
evaluation rather than this cap. It returns ``(world_size // pp) * (pp -
1)``, which is right for the two schedules this repo runs and wrong for
``ZBVZeroBubble`` and ``DualPipeV``, where rank 0 holds the last stage and
the loss. No run has ever used one.

Sixteen stages are reachable at pp 8, because four of the five registered
schedules ask for two stages per rank. Rule 7 reads ``pp *
stages_per_rank`` for that reason.
"""



MEGATRON_ENGINES = frozenset({"megatron_stock"})
"""The ``Arm.engine`` names that drive Megatron-LM.

Rules 5 and 12 and the three megatron run axes read this set. It is
declared one by one, because a name prefix fails open and the complement of
``torchtitan`` fails the other way; a new engine is an edit here rather
than a silent classification. It is stated rather than derived from
``ENGINES``, because that module sits above this one;
``tests/test_engines.py`` pins the set equal to the engines whose
``is_megatron`` is true.
"""


PP_SCHEDULES: dict[str, PipelineSchedule] = {
    "1F1B": PipelineSchedule(
        name="1F1B",
        titan_name="1F1B",
        megatron_supported=True,
        stages_per_rank=1,
        requires_uncompiled=False,
        description=(
            "One forward then one backward per rank, one stage per rank. The "
            "only schedule this pass targets, and the only one both engines "
            "and piper report."
        ),
    ),
    "Interleaved1F1B": PipelineSchedule(
        name="Interleaved1F1B",
        titan_name="Interleaved1F1B",
        megatron_supported=True,
        stages_per_rank=2,
        requires_uncompiled=False,
        description=(
            "1F1B over two stages per rank. Both engines implement it, and "
            "their warmup formulas are the same expression at the default "
            "group sizes. It needs a larger batch than the default workload "
            "gives: two chunks put rank 0's warmup at 4, so rule 12 refuses "
            "it at batch 4. megatron_supported records what Megatron-LM "
            "implements, not what this repo's megatron driver handles -- "
            "that driver has no model-chunk list yet and is expected to "
            "raise."
        ),
    ),
    "InterleavedZeroBubble": PipelineSchedule(
        name="InterleavedZeroBubble",
        titan_name="InterleavedZeroBubble",
        megatron_supported=False,
        stages_per_rank=2,
        requires_uncompiled=True,
        description=(
            "Interleaved 1F1B with the weight gradient split out to fill the "
            "bubble. PyTorch-only, so no cross-engine row exists, and it "
            "raises on a compiled stage module."
        ),
    ),
    "ZBVZeroBubble": PipelineSchedule(
        name="ZBVZeroBubble",
        titan_name="ZBVZeroBubble",
        megatron_supported=False,
        stages_per_rank=2,
        requires_uncompiled=True,
        description=(
            "The V-shaped zero-bubble schedule: each rank holds one stage "
            "from each end of the model. PyTorch-only, and it raises on a "
            "compiled stage module."
        ),
    ),
    "DualPipeV": PipelineSchedule(
        name="DualPipeV",
        titan_name="DualPipeV",
        megatron_supported=False,
        stages_per_rank=2,
        requires_uncompiled=True,
        description=(
            "The V-shaped DualPipe variant. PyTorch-only, and it raises on a "
            "compiled stage module."
        ),
    ),
}
"""The five schedules this repo can name.

Two are targets and three are declarations that the validator refuses; see
each entry. The names are PyTorch's own, verified against its
``get_schedule_class`` map. PyTorch registers four more that are
deliberately absent: a schedule is registered here when somebody intends to
run it.
"""

PP_SCHEDULE_CHOICES: tuple[str, ...] = tuple(PP_SCHEDULES)
"""Every ``--pp-schedule`` value a command accepts.

It is derived from the registry, so registering a schedule is one entry
above and never a second edit here. ``MODEL_SIZE_CHOICES`` follows the same
rule.
"""


TRIVIAL_SPEC = ParallelismSpec()
"""The single-GPU run: no data parallelism, no pipeline, no experts split.

Every number this repo has published was measured under this spec.
"""


def titan_mesh(spec: ParallelismSpec) -> tuple[int, int]:
    """TorchTitan's ``(dp_replicate, dp_shard)`` for this spec.

    TorchTitan has no DDP class; ``fully_shard`` is its only data-parallel
    path. ``dp_shard=1`` therefore means HSDP over a shard group of one rank,
    which shards nothing and replicates across ``dp_replicate`` -- the
    closest thing TorchTitan has to Megatron's DDP, and the pairing a
    cross-engine DP row needs. That is ZeRO level 0. At level 1 the whole
    data-parallel width becomes the shard degree, which is pure FSDP.

    **``titan_reshard_after_forward`` is what keeps level 1 at ZeRO-1.**
    The mesh alone would gather and reshard every forward. That function
    pins the policy, so the level holds whole parameters through the step.

    **The test names level 0 rather than level 1.** A third level must not
    take the replicated mesh by omission: the replicated mesh is the one
    every published number was measured under, and a new level that
    silently inherited it would publish a parity the run did not have.

    **It reads the declared parity and does NOT infer one from ``ep``.** An
    earlier revision returned ``(dp // ep, ep)`` at ``ep > 1``, on the
    grounds that TorchTitan builds the expert mesh out of the shard axis.
    That is right about the expert mesh and wrong about the dense one.
    ``parallel_dims.py`` derives ``efsdp = dp_shard * cp * tp // ep`` and
    builds the sparse mesh as ``("pp", "dp_replicate", "efsdp", "ep")``, so
    ``dp_replicate`` replicates the experts too. At ``dp 4, ep 2`` the old
    branch gave dense sharded over 2 with a replica factor of 2, where
    Megatron shards the dense parameters over ``dp_cp`` -- 4 ranks -- and
    the experts over ``expt_dp`` -- 2. ``(1, 4)`` gives TorchTitan those
    same two numbers, term for term.

    The control cell says it a second way: at level 1 with
    ``dp 4, ep 1`` the mesh is ``(1, 4)``, so an ``ep``-inferred branch would
    move the dense treatment between the control cell and the expert cell,
    and the expert row would again carry two changes.

    Spec rule 9 keeps ``dp // ep`` whole, so ``efsdp`` is a whole degree of
    at least 1 at level 1. At level 0 with ``ep > 1``
    it would be ``1 // ep``, which is 0 and is not a degree -- and TorchTitan
    asserts no lower bound on it. Spec rule 14 is what keeps that mesh out of
    a run. ``replicate * shard == dp`` holds in both branches.

    **The caller must always deliver the shard degree explicitly.**
    ``data_parallel_shard_degree`` defaults to ``-1`` in TorchTitan, which
    means "take every remaining rank". A dp=2 run that omitted the flag would
    silently shard every parameter instead of replicating, and nothing in
    the log or the manifest would say so.
    """
    if spec.zero == 0:
        return (spec.dp, 1)
    return (1, spec.dp)


def titan_reshard_after_forward(spec: ParallelismSpec) -> str | None:
    """TorchTitan's ``--parallelism.fsdp-reshard-after-forward`` policy.

    ``"never"`` at ZeRO level 1, and ``None`` at every other level, which
    means the harness sends no flag and TorchTitan keeps its own default.

    **This one function is what makes level 1 ZeRO-1 on TorchTitan.** Under
    ``"never"`` FSDP2 gathers the parameters at the first microbatch
    forward and keeps them for the whole step, so the run shards the
    optimizer states and holds whole parameters. Under the default policy
    it reshards after every forward, which shards the parameters too.

    It is one function, and the launcher and the tests both read it. Two
    spellings of "which value forces the policy" would let the argv and the
    test disagree about what a recorded cell ran.
    """
    return "never" if spec.zero == 1 else None


def zero_warnings(
    spec: ParallelismSpec, *, engines: Iterable[str] = ()
) -> tuple[str, ...]:
    """What a reader must not conclude from this spec's own mesh.

    Both cases below are legal, and neither refuses anything. Each names a
    cell whose recorded ``zero`` level describes a mechanism the run does
    not really have, so a reader who takes the level at face value reads
    the cell wrongly.

    ``engines`` is the set of ``Arm.engine`` names the run holds. The
    second warning is about TorchTitan's FSDP2 alone, so a run of megatron
    arms alone does not get it: Megatron builds a ``DistributedOptimizer``
    and holds ZeRO-1 exactly, at every mesh. An empty set therefore emits
    the first warning only.

    **Nothing is emitted above ``dp`` 1 and above ``pp`` 1.** That is the
    configuration this axis exists to run. A warning on the intended cell is
    noise. An operator then ignores every warning.

    ``benchmarks/e2e/runner.py`` emits these when it resolves a run, and
    ``benchmarks/e2e/results.py`` appends them to ``results.json``, so the
    operator meets them while the run starts and a later reader meets them in
    the artifact.
    """
    warnings: list[str] = []
    if spec.zero != 0 and spec.dp == 1:
        warnings.append(
            f"--zero {spec.zero} was requested at dp 1. "
            "The shard degree is 1 there, whatever the level says. Megatron "
            "shards the optimizer states over one rank and saves nothing, "
            "and TorchTitan skips its data-parallel path. So this run holds "
            "the dense parameters exactly as a replicated run holds them. "
            "Do not read this cell as a measurement of the sharded parity"
        )
    titan = frozenset(engines) - MEGATRON_ENGINES
    if spec.zero == 1 and spec.pp == 1 and titan:
        warnings.append(
            "--zero 1 was requested at pp 1. One microbatch puts the "
            "gradient reduce-scatter inside the only backward pass, so the "
            f"TorchTitan arms ({', '.join(sorted(titan))}) hold ZeRO-2 "
            "rather than the ZeRO-1 shape the level names. Megatron holds "
            "ZeRO-1 at every mesh. Do not read the two engines of this cell "
            "as one ZeRO level"
        )
    return tuple(warnings)


def skip_dp(spec: ParallelismSpec) -> bool:
    """Whether ``parallelize_piper1b`` may skip TorchTitan's DP path.

    True exactly when no data-parallel machinery is needed, which keeps the
    plain-bf16 model this repo has always measured. A pipeline-only run
    qualifies: PP needs no gradient synchronization, so ``dp == 1`` there and
    FSDP never runs. That is what makes PP2 the cheapest honest cross-engine
    number.
    """
    return spec.dp == 1 and spec.ep == 1


def n_microbatches(spec: ParallelismSpec, *, local_batch_size: int) -> int:
    """How many microbatches one rank's batch splits into.

    Floor division, deliberately total: ``describe`` records this for any
    spec, including one a caller never validated. Rule 10 is what makes the
    division exact for a spec that ran.
    """
    return local_batch_size // spec.pp_microbatch_size


def execution_model(spec: ParallelismSpec) -> str:
    """How the training process executes the model, as one manifest string.

    **The trivial spec returns ``"single-gpu-plain-bf16-no-fsdp"``, exactly.**
    ``benchmarks/e2e/registry.py``'s ``EXECUTION_MODEL`` has held that string
    since schema 7 and every manifest since records it, so it is a fixed
    point rather than a format: composing it from parts here must reproduce
    it character for character, and a test pins it both as a literal and
    against that constant. (``_resume_mismatches`` does not read the field
    today, so the manifest is the only thing that would carry a drift -- and
    it would carry it into every directory silently.)

    Everything else composes from the same parts, in a fixed order: the
    device count, the parameter treatment, the data-parallel treatment, then
    the pipeline and expert axes when they are not trivial.

    **The parallel parts name degrees, not mechanisms.** One manifest carries
    one ``execution_model`` for a whole run, and a cross-engine run holds
    arms of both engines -- so a term only one engine's code produces would
    be false for the other's arm. ``dp2`` is true of both; ``fsdp2-replicate2
    -shard1`` would describe TorchTitan's DP path and misdescribe Megatron's
    DDP wrapper. ``no-fsdp`` survives in the trivial string because at world
    size 1 neither engine wraps the model at all, so it is true of both.
    ``titan_mesh``'s engine-specific resolution belongs in ``describe``,
    under names that say whose it is.

    **``plain-bf16`` is a statement about the STATE, and above ``dp`` 1 the
    two engines differ on one thing it does not cover.** Parameters,
    gradients and optimizer states stay bf16 on both engines at every
    degree, with no fp32 masters, which is what the term has always meant
    here. The gradient **collective** is not state and is not named:
    TorchTitan's FSDP2 reduces in fp32 and casts back, because the fork
    types ``training.mixed_precision_reduce`` as ``Literal["float32"]``,
    where megatron reduces in bf16. A term for that would have to name one
    engine's mechanism, which is what the paragraph above forbids, so the
    difference is documented rather than encoded. Cite it beside a
    cross-engine dp number.

    **``zero`` reaches the string, and it does not name an engine.**
    Both engines shard at level 1 and at level 3, so ``dp2-zero1`` is true
    of a TorchTitan arm and of a Megatron arm alike -- unlike
    ``fsdp2-replicate2-shard1``, which spells out one engine's mesh. The
    level number carries into the suffix, so a new level cannot take
    another level's suffix. Level 0 adds nothing, which is what keeps every
    replicated string this repo has already recorded exactly where it was.
    """
    devices = "single-gpu" if spec.world_size == 1 else f"{spec.world_size}-gpu"
    if skip_dp(spec):
        data_parallel = "no-fsdp"
    elif spec.zero == 0:
        data_parallel = f"dp{spec.dp}"
    else:
        data_parallel = f"dp{spec.dp}-zero{spec.zero}"
    parts = [devices, "plain-bf16", data_parallel]
    if spec.pp > 1:
        parts.append(f"pp{spec.pp}-{spec.pp_schedule}")
    if spec.ep > 1:
        parts.append(f"ep{spec.ep}")
    return "-".join(parts)


def describe(
    spec: ParallelismSpec, *, local_batch_size: int
) -> dict[str, object]:
    """Flat JSON-safe provenance record for the manifest.

    Mirrors ``PiperShape.describe``: the declared fields, then the values a
    reader would otherwise have to re-derive with this module in hand.

    ``zero`` is the declared parity, and ``dp_replicate`` and
    ``dp_shard`` are the **TorchTitan** mesh that parity resolves to. The
    names say whose the mesh is. Megatron is told neither; it gets a DP group
    size, and at ``ep > 1`` an ``expert_model_parallel_size`` that subdivides
    it. They are recorded anyway because the shard degree is the value a
    TorchTitan run must be given explicitly -- see ``titan_mesh`` -- so a
    manifest that omitted it could not distinguish replication from
    sharding after the fact. Both sides are recorded because neither derives the
    other for a reader without this module: the parity is what the operator
    asked for and the mesh is what one engine built from it.

    ``n_microbatches`` is arithmetic over two fields in the same record, and
    what it describes at ``pp == 1`` depends on the engine. Read it beside
    ``pp``, and beside the arm roster.

    **An earlier revision said it describes no split at ``pp`` 1. That is no
    longer true.** The stock Megatron-LM driver takes
    ``--micro-batch-size`` at every degree, so at ``pp`` 1 it runs
    ``local_batch_size // pp_microbatch_size`` passes of that size and this
    field counts them. TorchTitan still runs one pass over the whole local
    batch there, because it reads
    ``pipeline_parallel_microbatch_size`` only inside
    ``_build_pipeline_schedule``. So at ``pp`` 1 one number describes a real
    split for one engine and no split for the other. Spec rule 3 holds
    ``pp_microbatch_size`` at 1 there, which makes the two agree today; the
    field would part them the moment that rule changed.

    It also disagrees with the megatron log line arm rule 12 validates, by
    construction: the driver's ``pipeline_settings`` returns one microbatch
    at ``pp`` 1 and the rule demands ``microbatches=1`` there, while this
    record says ``local_batch_size // pp_microbatch_size``. **The log is the
    run and this is the arithmetic.** A ``dp`` degree above 1 is the first
    spec that makes the disagreement reachable with more than one rank, so a
    reader of such a manifest meets it for the first time there.
    """
    replicate, shard = titan_mesh(spec)
    return {
        "dp": spec.dp,
        "pp": spec.pp,
        "ep": spec.ep,
        "pp_schedule": spec.pp_schedule,
        "pp_microbatch_size": spec.pp_microbatch_size,
        "zero": spec.zero,
        "world_size": spec.world_size,
        "dp_replicate": replicate,
        "dp_shard": shard,
        "n_microbatches": n_microbatches(
            spec, local_batch_size=local_batch_size
        ),
    }


def validate_parallelism(
    spec: ParallelismSpec,
    *,
    shape: PiperShape,
    workload: Workload,
    engines: Iterable[str],
    device_count: int,
) -> None:
    """Refuse a spec this run cannot honor. Each message names one cause.

    ``engines`` is the set of ``Arm.engine`` names the run will start, so
    the megatron restriction follows the arm roster rather than a scenario
    name. ``device_count`` is how many devices the operator asked for.

    Rules 8 and 9 were dead behind rule 14 while it refused every
    ``ep > 1``. Rule 14 now refuses an expert degree only under the
    ``zero 0`` parity, so both rules are reachable: a sharded spec with
    an illegal expert count reaches rule 8, and one whose expert degree does
    not divide ``dp`` reaches rule 9. They were kept through the whole
    refusal for exactly this, and neither had to be invented here.

    The one thing this function does not check is well-formedness --
    ``ParallelismSpec.__post_init__`` has already refused a degree below 1,
    so every rule below may assume positive counts.
    """
    engines = frozenset(engines)
    local_batch_size = workload.local_batch_size

    # A precondition on the one argument this module does not own, checked
    # before the numbered rules so those rules may assume it.
    #
    # The batch is the one integer the microbatch arithmetic divides, and
    # neither Workload nor workload_with_overrides bounds it -- --batch takes
    # a bare int. Without this, batch 0 and batch -4 pass every rule at pp 1
    # and reach the manifest as n_microbatches 0 and -4.
    if local_batch_size < 1:
        raise ValueError(
            f"local batch size {local_batch_size} must be >= 1; it is the "
            "count the microbatch split divides"
        )

    # 1. The mesh has to be exactly the devices the operator asked for. Not
    #    "at most": a spec that under-fills the request would leave a GPU
    #    idle and publish the number under the full device list.
    if spec.world_size != device_count:
        raise ValueError(
            f"parallelism world size {spec.world_size} (dp {spec.dp} x pp "
            f"{spec.pp}) does not match the {device_count} device(s) "
            "requested; ep borrows ranks from the dp axis and never "
            "multiplies the world size"
        )

    # 2. The budget for this pass, in two halves.
    #
    #    **The pipeline half runs first, and the order is load-bearing.**
    #    The world size is ``dp * pp`` and is never below ``pp``, so every
    #    spec above MAX_PP is also above MAX_WORLD_SIZE now that the two
    #    caps are equal. Testing the world size first would make the
    #    pipeline half unreachable and every deep-pipeline refusal would
    #    name the GPU budget instead of the cap somebody has to lift.
    #    Both halves refuse every spec they refused before; this only
    #    chooses the more specific of two true messages.
    if spec.pp > MAX_PP:
        raise ValueError(
            f"pipeline degree {spec.pp} exceeds the supported maximum "
            f"{MAX_PP}; no run plans a deeper pipeline, and nobody has "
            "checked one"
        )
    if spec.world_size > MAX_WORLD_SIZE:
        raise ValueError(
            f"parallelism world size {spec.world_size} exceeds the "
            f"{MAX_WORLD_SIZE}-GPU budget"
        )

    # 3. A schedule names a pipeline. Without one it would be recorded in the
    #    manifest and delivered to nothing.
    if spec.pp == 1 and spec.pp_schedule is not None:
        raise ValueError(
            f"pp_schedule {spec.pp_schedule!r} was requested at pp 1, where "
            "there is no pipeline to schedule"
        )
    if spec.pp > 1 and spec.pp_schedule is None:
        raise ValueError(
            f"pp {spec.pp} needs a pipeline schedule; choose one of "
            + ", ".join(PP_SCHEDULE_CHOICES)
        )
    #    The microbatch size is the schedule's twin and takes the same rule.
    #    TorchTitan reads pipeline_parallel_microbatch_size only inside
    #    _build_pipeline_schedule, which runs only when pp > 1, so a value
    #    set at pp 1 is recorded in the manifest and delivered to nothing --
    #    and, because --resume gates on the parallelism record, it would make
    #    two otherwise identical single-GPU runs refuse to resume each other.
    if spec.pp == 1 and spec.pp_microbatch_size != 1:
        raise ValueError(
            f"pp_microbatch_size {spec.pp_microbatch_size} was requested at "
            "pp 1, where neither engine splits the batch into microbatches"
        )

    # 4. The name has to be one we declared, so that stages_per_rank and the
    #    two capability flags below exist to read.
    schedule: PipelineSchedule | None = None
    if spec.pp_schedule is not None:
        try:
            schedule = PP_SCHEDULES[spec.pp_schedule]
        except KeyError as error:
            raise ValueError(
                f"Unknown pipeline schedule {spec.pp_schedule!r}. Available: "
                + ", ".join(PP_SCHEDULE_CHOICES)
            ) from error

    # 5. A schedule Megatron does not implement has no cross-engine opponent,
    #    so a run holding a megatron arm cannot use it. The check reads the
    #    engines rather than the scenario name: a titan-only run may use a
    #    PyTorch-only schedule.
    #
    #    It reads MEGATRON_ENGINES rather than one engine name. A
    #    command builder is not a spec rule: it runs after the manifest
    #    records the mesh, and it cannot refuse a mesh nobody builds a
    #    command for.
    if (
        schedule is not None
        and engines & MEGATRON_ENGINES
        and not schedule.megatron_supported
    ):
        raise ValueError(
            f"pipeline schedule {schedule.name!r} is not implemented by "
            "Megatron-LM, and this run holds a megatron arm; there would be "
            "no cross-engine comparison"
        )

    # 6. DELETED. It refused a schedule that raises on a compiled stage
    #    module. Compile is an arm property, so a spec alone cannot answer
    #    it; ``_resolve_run`` (benchmarks.e2e.runner) reads the selected
    #    arms and names the one that compiles.

    # 7. Every stage holds the same number of transformer layers. An uneven
    #    split is a different model per rank, and TorchTitan produces one
    #    without a warning.
    #
    #    **This test is only the right test when the harness sends
    #    ``--parallelism.pipeline-parallel-first-stage-less-layers 0`` and its
    #    last-stage twin.** Both default to 1, which makes TorchTitan's
    #    divisor ``n_layers + 2`` rather than ``n_layers``: at 16 layers over
    #    4 stages weight 1 splits [4, 5, 4, 3] where Megatron splits
    #    [4, 4, 4, 4], and this rule would pass both. Four stages are
    #    reachable at pp 2, eight at pp 4 and sixteen at pp 8, because four
    #    of the five registered schedules ask for two stages per rank.
    #    ``launch.py`` sends both flags at every ``pp > 1``, and this rule
    #    assumes that.
    #
    #    Megatron needs no flag: it divides ``config.num_layers`` and asserts
    #    the remainder itself.
    stages_per_rank = schedule.stages_per_rank if schedule is not None else 1
    total_stages = spec.pp * stages_per_rank
    if shape.n_layers % total_stages:
        raise ValueError(
            f"shape {shape.name!r} has {shape.n_layers} layers, which does "
            f"not divide evenly into {total_stages} pipeline stages "
            f"(pp {spec.pp} x {stages_per_rank} stage(s) per rank)"
        )

    # 8 and 9. The expert split. Both rules are reachable since rule 14
    # stopped refusing every expert degree, and both were written and tested
    # through that whole refusal: an expert count that does not divide gives
    # the ranks different expert counts, and an ep that does not divide dp
    # cannot be carved out of the data-parallel axis at all. They run before
    # rule 14, so an illegal count is named by its own rule under either
    # parity.
    if spec.ep > shape.num_experts:
        raise ValueError(
            f"expert degree {spec.ep} exceeds shape {shape.name!r}'s "
            f"{shape.num_experts} experts"
        )
    if shape.num_experts % spec.ep:
        raise ValueError(
            f"shape {shape.name!r}'s {shape.num_experts} experts do not "
            f"divide evenly across expert degree {spec.ep}"
        )
    if spec.dp % spec.ep:
        raise ValueError(
            f"expert degree {spec.ep} does not divide the data-parallel "
            f"degree {spec.dp}; ep takes its ranks out of the dp axis"
        )

    # 10. A rank's batch has to split into whole microbatches.
    if local_batch_size % spec.pp_microbatch_size:
        raise ValueError(
            f"local batch size {local_batch_size} does not divide evenly "
            f"into microbatches of {spec.pp_microbatch_size}"
        )
    microbatches = n_microbatches(spec, local_batch_size=local_batch_size)

    # 11. Both engines derive the same microbatch group size only when the
    #     count divides by the PP degree. PyTorch relaxes that rule and
    #     recomputes the group size (``ScheduleInterleaved1F1B.__init__``:
    #     ``number_of_rounds = max(1, n_microbatches // pp_group_size)``);
    #     Megatron does not, so the two would run different schedules under
    #     one label. Vacuous at pp 1, which is why it is unconditional.
    #
    #     **It is conservative for plain 1F1B, deliberately.** That
    #     relaxation lives in the interleaved class, which ``Schedule1F1B``
    #     never touches: it has no group-size concept, and Megatron's
    #     ``forward_backward_pipelining_without_interleaving`` has none
    #     either, so both engines agree at any count. So this rule costs the
    #     odd-microbatch cells (pp 2 at batch 5, 7, ...) for no mechanism the
    #     targeted schedule has. It is kept whole because refusing a legal
    #     cell is the safe direction and admitting an illegal one is not;
    #     narrow it per schedule only with a measured interleaved cell in
    #     hand.
    if microbatches % spec.pp:
        raise ValueError(
            f"{microbatches} microbatches do not divide evenly across "
            f"pipeline degree {spec.pp}; the two engines would derive "
            "different microbatch group sizes and run different schedules"
        )

    # 12. Rank 0 holds `pp * stages_per_rank` microbatches at its warmup
    #     peak, so a count below twice that makes 1F1B hold as much as GPipe
    #     and save nothing.
    #
    #     Guarded on pp > 1, and that guard is load-bearing: there are no
    #     microbatches without a pipeline, and `--batch` takes any positive
    #     integer, so an unconditional rule would refuse the legal
    #     single-GPU `--batch 1` run this repo can do today.
    if spec.pp > 1 and microbatches < 2 * total_stages:
        raise ValueError(
            f"{microbatches} microbatches is below the {2 * total_stages} "
            f"that pp {spec.pp} x {stages_per_rank} stage(s) per rank needs "
            "for 1F1B to hold less than GPipe; raise --batch or lower "
            "--pp-microbatch-size"
        )

    # 13. DELETED. It refused a parallel run under graph capture, and
    #     graph capture no longer exists.

    # 14. Expert parallelism needs the sharded dense parity, on both
    #     engines. TorchTitan cannot split the experts while it keeps the
    #     dense parameters replicated: apply_fsdp_to_decoder sends every
    #     non-expert parameter to Shard(0) on the dense mesh, and the expert
    #     mesh degree efsdp = dp_shard * cp * tp // ep needs dp_shard >= ep.
    #     Megatron holds every parity. So an ep row under `--zero 0` would
    #     compare EP plus sharding against EP plus replication, which is two
    #     changes rather than one.
    #
    #     **The test is `== 0`, so level 1 passes.** titan_mesh gives it the
    #     whole data-parallel width as dp_shard, so dp_shard >= ep holds.
    #
    #     The refusal names the flags that repair it. The operator declares
    #     the parity rather than the rule deriving one from ep, so that the
    #     sharded ep 1 control cell can be expressed and the ep row carries
    #     one change against it.
    if spec.ep > 1 and spec.zero == 0:
        raise ValueError(
            f"expert degree {spec.ep} needs --zero 1. TorchTitan cannot "
            "split the experts and keep the dense parameters replicated, "
            f"so under --zero {spec.zero} it would shard them while "
            "Megatron replicates them. The row would carry two changes "
            "rather than one"
        )

    # 15. DELETED, and the number is kept empty on purpose. The rule refused
    #     a sharded value at dp 1, because the shard degree is 1 there
    #     whatever the value says.
    #
    #     **It refused the agreed 30B-A3B matrix.** That matrix is
    #     `dp 1 x pp 8`, which is the deepest split eight GPUs hold, and it
    #     asks for --zero 1 to cut the optimizer states. Neither engine refuses
    #     that mesh: Megatron builds a DistributedOptimizer over one rank and
    #     TorchTitan skips its data-parallel path. A rule that blocks a legal
    #     run to protect a reader is the wrong tool.
    #
    #     **A warning replaced it.** `zero_warnings` says the same
    #     thing to the same reader, at the same mesh, and takes no GPU away.
    #     The runner emits it and results.json records it.
    #
    #     **The numbers of the deleted rules stay empty.** The messages,
    #     the tests and the agent guide all name the rules that remain, so
    #     a renumber would break more than it tidies.

    # 16. DELETED, and the number is kept empty on purpose. The rule refused
    #     a sharded parity and an expert degree to an earlier megatron
    #     driver that implemented neither. The stock driver implements both.

    # 17. DELETED with ZeRO level 3. The rule refused that level under
    #     a pipeline, because Megatron's own sharded wrapper factors the
    #     world size into terms with no pipeline term. The level is gone,
    #     and the number is kept empty on purpose.
