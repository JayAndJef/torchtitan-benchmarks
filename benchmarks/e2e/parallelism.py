"""The parallelism run axis: the degrees, the schedules and the validator.

One ``ParallelismSpec`` describes a whole run, exactly as ``--model-size``,
``--compile-mode`` and ``--ac`` each describe one. Every arm in one run
shares it, so the world size, the pipeline schedule and the microbatch count
are properties of the run rather than of an arm.

The CLI, runner, launcher, manifest writer, validation, and evaluation paths
all import this module. The spec is resolved and validated in the parent
before any host probe, then the same object builds both engines' command lines
and the manifest record. The rules remain torch-free and directly testable on
a CPU.

This is e2e-only. ``benchmarks/models/piper_qwen3/shape.py`` sits under
``models/`` because two engines build from it; a parallelism degree has one
reader, the e2e run. ``kernel-bench`` runs one process and never needs it,
and the TorchTitan training subprocess never needs it either, because
``parallelize_piper1b`` reads the ``ParallelDims`` TorchTitan builds from the
command line.

It is a separate module from ``benchmarks/e2e/registry.py`` so the spec and
its rules are declared in one place rather than beside the scenarios. **It
does import that module**, for the three compile-mode constants, so an
importer pays for the scenario declarations as well. That cost is a few
torch-free dataclasses today. Should it ever matter -- the likely caller is
``benchmarks/e2e/megatron/train.py``, a worker that must stay cheap to
import -- move those constants into a module both can read, rather than
copying them here: two spellings of the compile-mode sets would let rules 6
and 13 disagree with the axis they gate.

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
``titan_mesh`` satisfies it under ``shard`` by giving the whole
data-parallel width to ``dp_shard``. So ``dp`` is the whole data-parallel
width and ``ep`` is a split of it, which is what ``titan_mesh`` and rule 9
encode.

**TP and CP are deliberately absent.** They are out of scope for this pass,
and a field nobody can set misleads a later reader into thinking the axis is
supported. Adding one means adding it to ``world_size``, to
``execution_model`` and to the throughput divisor at the same time.

**What this module refuses today.** Rule 14 refuses ``ep > 1`` under the
``replicate`` dense parity, rule 15 refuses ``shard`` at ``dp`` 1, rule 16
refuses both to the tuned megatron driver, and
rules 5 and 6 refuse three of the five registered schedules for every
cross-engine run. Read a registered schedule as a declaration, never as a
measurement: only ``1F1B`` is targeted, and the caps admit up to ``pp 8``.
Real ``pp2``, ``dp2``, and ``dp2 x pp2`` correctness runs have passed on both
engines, and one ``dp 2 x pp 4`` cell at world size 8 has completed on the
stock arm. **No run has used ``pp 8``, no run has used the sharded parity,
and no run has used an expert degree.** The caps admit all three; that is a
declaration and not evidence.
No parallel timing is citable, because the cells that ran were on a loaded
host and nobody repeated them on an idle one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from benchmarks.e2e.registry import (
    COMPILE_MODES,
    CUDAGRAPH_COMPILE_MODES,
    UNCOMPILED_COMPILE_MODES,
    Workload,
)
from benchmarks.models.piper_qwen3.shape import PiperShape


# The GPU budget for this pass, and the pipeline depth it targets. Named so
# that lifting either is one edit here rather than a hunt through the rules.
#
# The budget is 8 because this host holds 8 devices. The depth is 8 because
# one pipeline of eight stages is the deepest split those 8 devices hold, and
# the suite runs that cell. Neither number is a property of an engine. Both
# are a declaration of what somebody plans to run.
#
# **An earlier revision capped pp at 2 and gave a false reason.** It said the
# two engines count layers the same way only at pp <= 2. They agree at every
# degree, because ``benchmarks/e2e/launch.py`` always sends
# ``--parallelism.pipeline-parallel-first-stage-less-layers 0`` and its
# ``last`` twin. Those two flags make rule 7's arithmetic the right test. See
# rule 7 below.
#
# The two conventions, measured rather than assumed: TorchTitan counts the
# embedding and the output head as stages, through
# ``pipeline_parallel_first/last_stage_less_layers``, which both default to
# **1**. Megatron does not (``account_for_embedding/loss_in_pipeline_split``
# default False) and divides ``config.num_layers`` alone. TorchTitan's own
# splitter, run at weight 0 and at the default weight 1:
#
#     16 layers, 2 stages: [8, 8] either way.
#     16 layers, 4 stages: [4, 4, 4, 4] against **[4, 5, 4, 3]**.
#     16 layers, 8 stages: eight 2s against **[2, 3, 2, 2, 2, 2, 2, 1]**.
#     24 layers, 4 stages: [6, 6, 6, 6] against **[6, 7, 6, 5]**.
#     24 layers, 8 stages: eight 3s against **[3, 4, 3, 3, 3, 3, 3, 2]**.
#
# Megatron gives the weight 0 split in each row and does not raise. So the
# disagreement is TorchTitan's uneven split under its own default, and the
# harness removes it at every degree.
#
# Sixteen stages are reachable at pp 8, eight at pp 4 and four at pp 2,
# because **four of the five registered schedules ask for two stages per
# rank**: ``Interleaved1F1B``, ``InterleavedZeroBubble``, ``ZBVZeroBubble``
# and ``DualPipeV``. Rule 7 reads ``pp * stages_per_rank`` for that reason.
# An earlier revision of this comment said "either interleaved schedule",
# which names two of the four.
#
# **The two caps are now equal, and that makes rule 2's order load-bearing.**
# The world size is ``dp * pp``, which is never below ``pp``, so every spec
# above the pipeline cap is also above the world-size cap. The pipeline half
# therefore runs FIRST, and both halves still refuse every spec they refused
# before. The pipeline message names the cap somebody has to lift, which is
# what a reader of a ``pp 9`` refusal needs. Do not delete the pipeline half
# to reach the same verdict: that removes the specific message rather than
# choosing between two true ones.
#
# **The cap lift widens one recorded gap, and this comment records the
# widening.** ``benchmarks/e2e/results.py``'s ``loss_visible_rank`` returns
# ``(world_size // pp) * (pp - 1)``. That is right for the two schedules this
# repo runs and wrong for the two V-shaped ones. TorchTitan's
# ``_get_pp_rank_to_stage_indices_mapping`` gives rank 0 the pair
# ``(0, num_stages - 1)`` under ``ZBVZeroBubble`` and ``DualPipeV``, so
# **rank 0 holds the last stage and the loss**. TorchTitan's own
# ``_get_metrics_rank`` special-cases ``ZBVZeroBubble`` and returns 0; it
# does not special-case ``DualPipeV``, and this repo special-cases neither.
# Rule 5 refuses a V-shaped schedule only beside a megatron arm, so a
# titan-only run may still ask for one. Rule 6 narrows the exposure and does
# not close it: both V-shaped schedules set ``requires_uncompiled``, so such
# a run also has to ask for ``--compile-mode none``.
#
# The gap already existed at ``dp 1`` and ``dp 2`` with ``pp 2``. The lift to
# world size 8 and pp 4 added six ``(dp, pp)`` pairs: (1, 3), (1, 4), (2, 3),
# (2, 4), (3, 2) and (4, 2). The lift to pp 8 adds four more, all of them at
# ``dp`` 1: (1, 5), (1, 6), (1, 7) and (1, 8). No run has ever used a
# V-shaped schedule. Read this before you run one, and repair
# ``loss_visible_rank`` rather than the cap.
MAX_WORLD_SIZE = 8
MAX_PP = 8


# The ``Arm.launcher`` values that drive Megatron-LM. Rule 5 reads this set,
# because ``PipelineSchedule.megatron_supported`` is a fact about the library
# and more than one launcher runs it.
#
# **Declared one by one. Not a ``megatron`` prefix, and not "every launcher
# that is not torchtitan".** A prefix fails OPEN: a launcher that names the
# library another way -- ``mcore``, ``nemo`` -- would walk past rule 5, and
# the run would reach a schedule Megatron cannot run with no cross-engine
# opponent for it. The complement fails the other way: it would apply
# Megatron's restriction to an engine that is not Megatron, and refuse a
# legal titan-only cell. A declared set is wrong in neither direction, and a
# new launcher is then an edit here rather than a silent classification.
# ``tests/test_parallelism.py`` refuses a registry launcher this set and
# ``torchtitan`` do not name between them.
MEGATRON_LAUNCHERS = frozenset({"megatron", "megatron_stock"})


# The ``Arm.launcher`` values whose driver implements neither the sharded
# dense parity nor an expert degree. Rule 16 reads this set.
#
# ``benchmarks/e2e/megatron/train.py`` is the one such driver today. It calls
# ``initialize_model_parallel(pipeline_model_parallel_size=args.pp)`` with no
# expert size, it builds a plain replicated ``DistributedDataParallel``, and
# its command line carries neither an expert flag nor a sharding flag. So a
# spec that asks for either gets a run that ignores it, and the manifest
# records a parity and an expert degree the arm did not have.
#
# **Declared one by one, for the reason MEGATRON_LAUNCHERS is declared one
# by one.** This is a fact about one driver's source, not about Megatron-LM:
# ``megatron_stock`` hands the run to Megatron's own ``pretrain``, which
# implements both. A prefix or a complement would classify the two the same
# way and would be wrong about one of them. A new launcher is an edit here
# rather than a silent classification.
REPLICATE_ONLY_LAUNCHERS = frozenset({"megatron"})


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
    ``benchmarks/e2e/megatron/train.py`` has no model-chunk list and raises.
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


# The five schedules this repo can name. Two are targets and three are
# declarations that the validator refuses; see each entry.
#
# The names are PyTorch's own, verified against the ``schedule_map`` in
# ``torch/distributed/pipelining/schedules.py``'s ``get_schedule_class``.
# PyTorch registers four more (``GPipe``, ``LoopedBFS``, and the two abstract
# base classes) that are deliberately absent: a schedule is registered here
# when somebody intends to run it.
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

# Every ``--pp-schedule`` value a command accepts. Derived from the registry
# so that registering a schedule is one entry above and never a second edit
# here -- the same rule ``MODEL_SIZE_CHOICES`` follows in ``shape.py``.
PP_SCHEDULE_CHOICES: tuple[str, ...] = tuple(PP_SCHEDULES)


# How the run holds the DENSE parameters -- every parameter that is not a
# routed expert weight. ``replicate`` gives each rank a whole copy.
# ``shard`` splits one copy between the ranks of the data-parallel axis.
#
# **It is a comparability boundary at any expert degree**, because it decides
# how much optimizer state one rank holds and what the ranks exchange each
# step. Every number this repo has published was measured under
# ``replicate``, which is why that is the default.
#
# **It is also what makes an expert degree legal.** The reason is a
# TorchTitan constraint rather than a preference. TorchTitan cannot split
# the experts while it keeps the dense parameters replicated:
# ``apply_fsdp_to_decoder`` sends every non-expert parameter to ``Shard(0)``
# on the dense mesh, and the expert mesh degree
# ``efsdp = dp_shard * cp * tp // ep`` needs ``dp_shard >= ep``. Megatron
# holds either parity. So the two engines compare under an expert degree only
# when both shard, and spec rule 14 refuses the other combination.
DENSE_SHARDING_MODES = ("replicate", "shard")
DEFAULT_DENSE_SHARDING = "replicate"


@dataclass(frozen=True)
class ParallelismSpec:
    """The parallelism degrees and pipeline settings for one run.

    Every field defaults to the single-GPU value, so ``ParallelismSpec()`` is
    the run this repo has always done and ``TRIVIAL_SPEC`` is that object.

    ``__post_init__`` enforces well-formedness only -- every degree is a
    positive count, and ``dense_sharding`` names a declared mode. That is
    not one of the sixteen validator rules; it is the precondition they
    assume. Without it a spec of ``dp=-1, pp=-1`` would have ``world_size``
    1 and walk past rule 1 on a one-GPU box, which is exactly the illegal
    mesh the rules exist to refuse. ``PiperShape`` guards its geometry the
    same way and for the same reason.

    **``dense_sharding`` takes the same treatment, and it must.**
    ``titan_mesh`` and ``execution_model`` are total functions over a spec
    and both branch on this value, so a spec carrying a string neither
    branch knows must not exist. A validator rule would be too late: both
    functions run on specs the validator never sees.

    ``dense_sharding`` sits here rather than beside ``--model-size`` as a
    run axis of its own, for the reason ``pp_schedule`` and
    ``pp_microbatch_size`` do: it is a treatment of one parallelism axis,
    and every function that needs it already takes the spec.
    """

    dp: int = 1
    pp: int = 1
    ep: int = 1
    pp_schedule: str | None = None
    pp_microbatch_size: int = 1
    dense_sharding: str = DEFAULT_DENSE_SHARDING

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
        if self.dense_sharding not in DENSE_SHARDING_MODES:
            raise ValueError(
                f"Unknown dense sharding mode {self.dense_sharding!r}. "
                "Available: " + ", ".join(DENSE_SHARDING_MODES)
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


# The single-GPU run: no data parallelism, no pipeline, no experts split.
# Every number this repo has published was measured under this spec.
TRIVIAL_SPEC = ParallelismSpec()


def titan_mesh(spec: ParallelismSpec) -> tuple[int, int]:
    """TorchTitan's ``(dp_replicate, dp_shard)`` for this spec.

    TorchTitan has no DDP class; ``fully_shard`` is its only data-parallel
    path. ``dp_shard=1`` therefore means HSDP over a shard group of one rank,
    which shards nothing and replicates across ``dp_replicate`` -- the
    closest thing TorchTitan has to Megatron's DDP, and the pairing a
    cross-engine DP row needs. That is the ``replicate`` parity. Under
    ``shard`` the whole data-parallel width becomes the shard degree, which
    is pure FSDP.

    **It reads the declared parity and does NOT infer one from ``ep``.** An
    earlier revision returned ``(dp // ep, ep)`` at ``ep > 1``, on the
    grounds that TorchTitan builds the expert mesh out of the shard axis.
    That is right about the expert mesh and wrong about the dense one.
    ``parallel_dims.py`` derives ``efsdp = dp_shard * cp * tp // ep`` and
    builds the sparse mesh as ``("pp", "dp_replicate", "efsdp", "ep")``, so
    ``dp_replicate`` replicates the experts too. At ``dp 4, ep 2`` the old
    branch gave dense sharded over 2 with a replica factor of 2, where
    Megatron-FSDP shards the dense parameters over ``dp_cp`` -- 4 ranks --
    and the experts over ``expt_dp`` -- 2. ``(1, 4)`` gives TorchTitan those
    same two numbers, term for term.

    The control cell says it a second way: under ``shard`` at ``dp 4, ep 1``
    the mesh is ``(1, 4)``, so an ``ep``-inferred branch would move the dense
    treatment between the control cell and the expert cell, and the expert
    row would again carry two changes.

    Spec rule 9 keeps ``dp // ep`` whole, so ``efsdp`` is a whole degree of
    at least 1 under ``shard``. Under ``replicate`` at ``ep > 1`` it would be
    ``1 // ep``, which is 0 and is not a degree -- and TorchTitan asserts no
    lower bound on it. Spec rule 14 is what keeps that mesh out of a run.
    ``replicate * shard == dp`` holds in both branches.

    **The caller must always deliver the shard degree explicitly.**
    ``data_parallel_shard_degree`` defaults to ``-1`` in TorchTitan, which
    means "take every remaining rank". A dp=2 run that omitted the flag would
    silently run ZeRO-3 instead of the intended replication, and nothing in
    the log or the manifest would say so.
    """
    if spec.dense_sharding == "shard":
        return (1, spec.dp)
    return (spec.dp, 1)


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

    **``dense_sharding`` reaches the string, and it does not name an engine.**
    Both engines shard under ``shard``, so ``dp2-shard`` is true of a
    TorchTitan arm and of a Megatron arm alike -- unlike
    ``fsdp2-replicate2-shard1``, which spells out one engine's mesh. The
    default parity adds nothing, which is what keeps every string this repo
    has already recorded exactly where it was.
    """
    devices = "single-gpu" if spec.world_size == 1 else f"{spec.world_size}-gpu"
    if skip_dp(spec):
        data_parallel = "no-fsdp"
    elif spec.dense_sharding == "shard":
        data_parallel = f"dp{spec.dp}-shard"
    else:
        data_parallel = f"dp{spec.dp}"
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

    ``dense_sharding`` is the declared parity, and ``dp_replicate`` and
    ``dp_shard`` are the **TorchTitan** mesh that parity resolves to. The
    names say whose the mesh is. Megatron is told neither; it gets a DP group
    size, and at ``ep > 1`` an ``expert_model_parallel_size`` that subdivides
    it. They are recorded anyway because the shard degree is the value a
    TorchTitan run must be given explicitly -- see ``titan_mesh`` -- so a
    manifest that omitted it could not distinguish replication from ZeRO-3
    after the fact. Both sides are recorded because neither derives the
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
        "dense_sharding": spec.dense_sharding,
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
    compile_mode: str,
    engines: Iterable[str],
    device_count: int,
) -> None:
    """Refuse a spec this run cannot honor. Each message names one cause.

    ``engines`` is the set of ``Arm.launcher`` values the run will start, so
    the megatron restriction follows the arm roster rather than a scenario
    name. ``device_count`` is how many devices the operator asked for.

    Rules 8 and 9 were dead behind rule 14 while it refused every
    ``ep > 1``. Rule 14 now refuses an expert degree only under the
    ``replicate`` parity, so both rules are reachable: a ``shard`` spec with
    an illegal expert count reaches rule 8, and one whose expert degree does
    not divide ``dp`` reaches rule 9. They were kept through the whole
    refusal for exactly this, and neither had to be invented here.

    The one thing this function does not check is well-formedness --
    ``ParallelismSpec.__post_init__`` has already refused a degree below 1,
    so every rule below may assume positive counts.
    """
    engines = frozenset(engines)
    local_batch_size = workload.local_batch_size

    # Preconditions on the two arguments this module does not own, checked
    # before the numbered rules so those rules may assume them.
    #
    # The compile mode is checked because rules 6 and 13 read it in OPPOSITE
    # directions: rule 6 refuses anything outside UNCOMPILED_COMPILE_MODES
    # and so fails safe on an unknown name, while rule 13 refuses only names
    # inside CUDAGRAPH_COMPILE_MODES and so fails OPEN on one. An unknown
    # mode would therefore slip a graph-capturing run past rule 13. The CLI
    # spells the axis as a click.Choice today, so nothing reaches this from a
    # command line -- but a caller with a bare string does.
    if compile_mode not in COMPILE_MODES:
        raise ValueError(
            f"Unknown compile mode {compile_mode!r}. Available: "
            + ", ".join(COMPILE_MODES)
        )
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
    #    launchers rather than the scenario name: a titan-only run may use a
    #    PyTorch-only schedule.
    #
    #    It reads MEGATRON_LAUNCHERS rather than the one name "megatron".
    #    A second Megatron-LM launcher, "megatron_stock", walked past this
    #    rule while the test was an equality against one string, and a
    #    refusal inside a command builder covered it instead. A command
    #    builder is not a spec rule: it runs after the manifest records the
    #    mesh, and it cannot refuse a mesh nobody builds a command for.
    if (
        schedule is not None
        and engines & MEGATRON_LAUNCHERS
        and not schedule.megatron_supported
    ):
        raise ValueError(
            f"pipeline schedule {schedule.name!r} is not implemented by "
            "Megatron-LM, and this run holds a megatron arm; there would be "
            "no cross-engine comparison"
        )

    # 6. PyTorch's zero-bubble and DualPipeV classes call
    #    _check_torch_compile_compatibility, which raises on a compiled stage
    #    module. Refusing here beats failing inside the training subprocess.
    if (
        schedule is not None
        and schedule.requires_uncompiled
        and compile_mode not in UNCOMPILED_COMPILE_MODES
    ):
        raise ValueError(
            f"pipeline schedule {schedule.name!r} raises on a compiled stage "
            f"module, so it needs an uncompiled compile mode (one of "
            f"{', '.join(sorted(UNCOMPILED_COMPILE_MODES))}), not "
            f"{compile_mode!r}"
        )

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
    #    ``launch.py`` sends both flags at every ``pp > 1``; this rule
    #    assumes that, and ``tests/test_migration_contract.py`` freezes the
    #    argv at pp 2 and at pp 4.
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

    # 13. No parallel CUDA-graph run has ever been captured. The DP case has
    #     a recorded reason -- FSDP2 frees and reallocates the unsharded
    #     parameters each step, which moves their addresses and forces a
    #     re-capture every step -- and the failure mode is slow rather than
    #     broken, so
    #     validation rule 9 would still pass and the report would publish a
    #     slow number under a cuda-graph label. PP has no such record and is
    #     refused with it: one rule reads more clearly than two, and a
    #     single-GPU run keeps all three compile modes either way.
    if spec.world_size > 1 and compile_mode in CUDAGRAPH_COMPILE_MODES:
        raise ValueError(
            f"compile mode {compile_mode!r} is refused at world size "
            f"{spec.world_size}: no multi-rank run has been shown to capture "
            "its graphs once, and a repeated capture is slow rather than "
            "broken, so the run would publish a slow number under a "
            "cuda-graph label"
        )

    # 14. Expert parallelism needs the sharded dense parity, on both
    #     engines. TorchTitan cannot split the experts while it keeps the
    #     dense parameters replicated: apply_fsdp_to_decoder sends every
    #     non-expert parameter to Shard(0) on the dense mesh, and the expert
    #     mesh degree efsdp = dp_shard * cp * tp // ep needs dp_shard >= ep.
    #     Megatron holds either parity. So an ep row under `replicate` would
    #     compare EP plus sharding against EP plus replication, which is two
    #     changes rather than one.
    #
    #     The refusal names the flag that repairs it. The operator declares
    #     the parity rather than the rule deriving one from ep, so that the
    #     sharded ep 1 control cell can be expressed and the ep row carries
    #     one change against it.
    if spec.ep > 1 and spec.dense_sharding != "shard":
        raise ValueError(
            f"expert degree {spec.ep} needs --dense-sharding shard. "
            "TorchTitan cannot split the experts and keep the dense "
            f"parameters replicated, so under {spec.dense_sharding!r} it "
            "would shard them while Megatron replicates them. The row would "
            "carry two changes rather than one"
        )

    # 15. The sharded parity needs a data-parallel width to shard over. At
    #     dp 1 titan_mesh returns (1, 1) whatever the flag says, and
    #     Megatron-FSDP shards one copy over one rank, so the run holds the
    #     dense parameters exactly as a replicated run does. The manifest
    #     would then record a parity the run did not have, which is the one
    #     failure a recorded fact must not have.
    if spec.dense_sharding == "shard" and spec.dp == 1:
        raise ValueError(
            "--dense-sharding shard needs a data-parallel degree above 1. "
            f"At dp {spec.dp} the shard degree is 1 whatever the flag says. "
            "The manifest would record a parity the run did not have"
        )

    # 16. A driver that implements neither the sharded parity nor an expert
    #     degree may not be given one. Its command line carries no such
    #     flag, so the run would ignore the value and the manifest would
    #     record it anyway. That is the one failure a recorded fact must not
    #     have, and both halves became reachable in this pass: rule 14
    #     refused every expert degree before it, and the parity is new.
    #
    #     The expert half is checked first, and it is not redundant. Rule 14
    #     already ties an expert degree to the sharded parity, so the shard
    #     half alone would refuse every expert spec that reached here -- but
    #     under a message about sharding, which is not what the operator
    #     asked for. The explicit half also keeps the hole shut if rule 14
    #     ever narrows.
    #
    #     **Both messages name a repair, as rule 14's does.** ``engines`` is
    #     the launcher set of the arms the run will really start.
    #     ``run --arm NAME`` narrows it. So a subset that selects the
    #     TorchTitan arms alone takes this axis off the megatron driver and
    #     passes this rule.
    #
    #     **The messages name ``run --arm``, not a bare ``--arm``.**
    #     ``run-all`` carries PASSTHROUGH_CONTEXT, so it accepts the flag,
    #     forwards it to the training subprocess as a TorchTitan argument
    #     and records it in ``extra_torchtitan_args``. The operator would
    #     read the same refusal a second time with nothing to say the flag
    #     was ignored.
    #
    #     **Neither message promises the selected run then succeeds, and
    #     the wording stays at "passes this rule" for that reason.** The
    #     subprocess-side blocker is gone: ``parallelize_piper1b`` refused
    #     every shard degree above 1 while this rule was written, and it now
    #     refuses only a DROPPED flag -- it reads the raw configured
    #     ``data_parallel_shard_degree`` and admits any degree the run asked
    #     for. So a titan-only sharded roster is no longer refused inside
    #     the training subprocess.
    #
    #     **What is still missing is a run.** No sharded arm and no expert
    #     arm has executed on a GPU on either engine, so the path is
    #     declared and not measured. A message that said "measures this
    #     mesh" would promise the operator a measurement nobody has taken.
    #     Restore that wording after the first sharded cell passes
    #     ``validate_arm``, and not before.
    refused = engines & REPLICATE_ONLY_LAUNCHERS
    if refused and spec.ep > 1:
        raise ValueError(
            f"expert degree {spec.ep} is not implemented by the "
            f"{', '.join(sorted(refused))} driver, which this run holds. "
            "That driver passes no expert size to initialize_model_parallel. "
            "The run would train every expert on every rank. The manifest "
            "would record a split it did not have. Use run --arm to select "
            "the TorchTitan arms alone"
        )
    if refused and spec.dense_sharding == "shard":
        raise ValueError(
            "--dense-sharding shard is not implemented by the "
            f"{', '.join(sorted(refused))} driver, which this run holds. "
            "That driver builds a plain replicated DistributedDataParallel. "
            "The run would replicate the dense parameters. The manifest "
            "would record a sharded parity. Use run --arm to select the "
            "TorchTitan arms alone"
        )
