"""Benchmark scenario definitions and the global run axes.

Scenarios describe what differs between arms, and nothing else: command
construction (``benchmarks.e2e.launch``), provenance collection
(``benchmarks.execution.provenance``), and validation
(``benchmarks.e2e.validation``) each live in their own module and read these
declarations. That is what lets a new ablation be a registry entry rather
than a new copy of the training harness.

The activation-checkpointing and execution-model constants live here too:
they are per-run axes of a scenario execution, consumed by
command construction (``benchmarks.e2e.launch``), validation
(``benchmarks.e2e.validation``), and the manifest
(``benchmarks.artifacts.manifests``).
"""

from dataclasses import dataclass, replace
from typing import Literal

from benchmarks.models.piper_qwen3.shape import PIPER_1B


# How a SINGLE-GPU training process executes the model: plain bf16 params on
# one GPU, no FSDP wrapper, no fp32 masters
# (benchmarks.models.piper_qwen3.parallelize). Every manifest since schema 7
# records this string, so it is a fixed point rather than a format.
#
# The manifest no longer reads it. benchmarks/e2e/parallelism.py's
# execution_model composes the field from the run's own mesh, and this
# constant is what its trivial answer must reproduce character for
# character; tests/test_parallelism.py pins the two against each other. A
# manifest that self-describes has to describe the run it recorded, and a
# constant cannot describe two of them. Earlier schemas ran under FSDP2
# mixed precision.
EXECUTION_MODEL = "single-gpu-plain-bf16-no-fsdp"


# Activation checkpointing modes selectable per run. "sac" is TorchTitan's
# per-op SelectiveAC; "none" disables checkpointing entirely, delivered to
# TorchTitan as the tyro subcommand token "activation-checkpoint:none".
AC_MODES = ("sac", "none")

# Default values of the global run axes.
# ``_resolve_run`` (benchmarks.e2e.runner) and the ``run-all
# --all-scenarios`` pre-pass (benchmarks.cli.e2e) read these constants.
# Neither site repeats the literal value now.
DEFAULT_AC_MODE = "none"
DEFAULT_MODEL_SIZE = "1b"

# Whether a run collects profiler traces.
#
# Off is the default, and it is the treatment a published throughput number
# wants: the profiler costs GPU and host time on every window, and the whole
# 40-step floor exists to hold two of them. Under ``--profile`` the run
# collects the trace layout an external analysis tool reads, and the floor
# and every trace rule of ``benchmarks/e2e/validation.py`` apply again.
#
# It is a run axis and not an arm property: both engines either write the
# layout or write nothing, and a run that profiled one arm and not another
# would publish two treatments under one label.
DEFAULT_PROFILE = False

# How many steps an unprofiled run discards before it measures.
#
# The engines compile, autotune and fill their caches in the first steps of
# a run, so a throughput taken over them is not the throughput of the
# workload. Under ``--profile`` the profiler schedule decides the sample
# set instead (``benchmarks/e2e/results.py``'s ``stable_tps``), and this
# axis is refused; without it every step after the warmup is a sample
# (``measured_tps``).
#
# 10 is half of one profiler cycle, which is the span the profiled rule
# samples: ``stable_tps`` takes steps 2 to 10 of every 20. The two rules
# therefore discard a comparable prefix, and neither figure is the other's
# -- results are only comparable within one value of this axis.
DEFAULT_WARMUP_STEPS = 10

# The Megatron pipeline point-to-point sync treatment, selectable per run.
# Stock Megatron calls torch.cuda.synchronize() once per batched pipeline
# message (megatron/core/pipeline_parallel/p2p_communication.py, guarded by
# ``config.batch_p2p_comm and config.batch_p2p_sync``). "on" keeps that
# call, and it is what every number published before this default flipped
# was measured under. "off" sets the ``batch_p2p_sync`` config field False
# on the megatron driver, which removes the call, and it is the default
# here. TorchTitan arms receive nothing.
#
# It is a run axis and not a ParallelismSpec field. The value is a treatment
# of the pipeline messages, and execution_model names degrees rather than
# mechanisms. Megatron exposes no CLI flag for the field, so each driver
# takes the value from the harness and prints what its BUILT config carries.
# The measured effect and its caveats are in
# reports/20260901-p2p-sync-ab.md.
MEGATRON_P2P_SYNC_MODES = ("on", "off")
DEFAULT_MEGATRON_P2P_SYNC = "off"

# Stock Megatron's NaN/Inf guard, selectable per run. One Megatron argument,
# ``check_for_nan_in_loss_and_grad``, gates two host waits at Megatron-LM
# 59b72fa5: pretrain_gpt.py's loss_func evaluates the loss twice per
# microbatch through rerun_state_machine.validate_result, and training.py
# copies the same field into ddp_config.check_for_nan_in_grad, under which
# param_and_grad_buffer.py's check_grads evaluates every bucket's gradient
# norm twice per step. Each evaluation reads a device bool and synchronizes
# the stream. ``--rerun-mode disabled``, which the stock argv already sends,
# removes neither: validate_result still evaluates the rejection function
# under RerunMode.DISABLED and raises when it is set.
#
# "on" is stock Megatron, and it is what every number published before
# this default flipped was measured under. "off" sends Megatron's own
# --no-check-for-nan-in-loss-and-grad to the stock launcher, so a stock
# user can reproduce the argv, and it is the default here. TorchTitan arms
# receive nothing. The measured effect is in
# reports/20260905-host-sync-ab.md. Evaluation refuses
# a non-finite loss or grad norm on every arm under either value
# (benchmarks/e2e/results.py's refuse_non_finite_trajectories), which is the
# guard that has to exist before this one can be turned off.
MEGATRON_NAN_GUARD_MODES = ("on", "off")
DEFAULT_MEGATRON_NAN_GUARD = "off"

# Stock Megatron's optimizer precision, selectable per run.
#
# "stock" is --bf16 alone, and it is 18 bytes of optimizer state per
# parameter: the bf16 parameter 2, an fp32 master 4, fp32 gradients 4, and
# two fp32 Adam moments 8. That is what every published cell of this
# scenario ran, and it is the first of the four deliberate differences the
# stock arm carries against TorchTitan's 8.
#
# "lean" sends four flags and reaches 10 bytes:
#
#     --use-precision-aware-optimizer
#     --main-grads-dtype bf16
#     --exp-avg-dtype bf16
#     --exp-avg-sq-dtype bf16
#
#     tensor            stock   lean
#     parameter bf16        2      2
#     master                4      2
#     gradients             4      2
#     Adam moments          8      4
#     total                18     10
#
# **The master stays fp32, and it is never fp16.** store_param_remainders
# defaults True (optimizer_config.py) and needs TE >= 2.1.0; this repo runs
# 2.17.1. The bf16 parameter is the top half of the fp32 master and the
# remainder holds the low 16 bits, so the master costs 2 bytes and is still
# exactly fp32. --main-params-dtype accepts fp32 and fp16 only
# (arguments.py), so this axis never sends it.
#
# **--grad-reduce-in-bf16 is never sent either.** Under --bf16 Megatron
# turns fp32 accumulation on only when the main-grad dtype is fp32
# (arguments.py), so --main-grads-dtype bf16 leaves it off by itself. The
# second flag would state one fact twice.
#
# "lean" needs a sharded dense value. optimizer_config.py asserts
# use_distributed_optimizer under --use-precision-aware-optimizer, and the
# zero axis is the one owner of that flag. The value reaches the
# stock megatron launcher alone.
#
# **"lean" changes the numerics.** bf16 Adam moments and bf16 gradient
# accumulation are a real change, and at pp 8 the accumulation is 16-way in
# bf16. Read the loss trajectories beside any lean number. At dp 1 the
# distributed optimizer also runs its bucket bookkeeping for no saving.
# Both effects are unmeasured.
MEGATRON_PRECISION_MODES = ("stock", "lean")
DEFAULT_MEGATRON_PRECISION = "stock"


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
    """

    name: str
    description: str
    compile: Literal["torch", "none"]
    config: str | None = None
    override_imports: tuple[str, ...] = ()
    # [Override] log lines expected per transformer block. validate_arm
    # multiplies by the shape's layer count, so an arm stays correct at any
    # --model-size (16 lines at normal, 1 at huge).
    overrides_per_block: int = 0
    trace_kernel_markers: tuple[str, ...] = ()
    requires_gcc_toolset: bool = False
    # Which engine the runner launches and which validation profile applies.
    # Plain strings (registry keys in benchmarks.e2e.launch /
    # benchmarks.e2e.validation) so asdict(arm) stays JSON-serializable for
    # the manifest.
    launcher: str = "torchtitan"
    validation: str = "torchtitan"


@dataclass(frozen=True)
class Scenario:
    """A reproducible workload and its comparable implementation arms.

    ``supported_ac_modes`` restricts the global ``--ac`` axis: a scenario
    whose arms cannot honor a mode (e.g. an engine with no SAC-parity
    recompute) lists only the modes it supports; ``run-all --all-scenarios``
    skips unsupported combinations and a direct request errors.
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


PIPER_1B_WORKLOAD = Workload(
    module="benchmarks.models.piper_qwen3",
    config="qwen3_piper_1b",
    seq_len=1024,
    steps=40,
    local_batch_size=4,
)


PIPER_1B_MEGATRON_WORKLOAD = replace(
    PIPER_1B_WORKLOAD,
    config="qwen3_piper_1b_pretokenized",
    seed=42,
    replay_dataloader=True,
)

# The engine comparison: stock TorchTitan against stock Megatron-LM on one
# pre-tokenized c4_test stream. Three arms answer one question -- what does
# each engine cost per token at this mesh.
#
# The cross-engine metrics are tokens/s, total GPU kernel time, launch
# latency and peak memory.
#
# The ac axis is pinned to "none". Megatron's recompute options are not
# parity with TorchTitan's per-op SAC, and the Megatron arm does no
# recompute at all.
#
# **The Megatron arm is not plain bf16, and the manifest cannot say so.**
# With --bf16 and no --use-precision-aware-optimizer, Megatron keeps fp32
# master weights, fp32 optimizer moments, and forces
# accumulate_allreduce_grads_in_fp32, so the arm holds about 18 bytes per
# parameter against TorchTitan's 8 and reduces gradients in fp32. That is the
# stock treatment, and this scenario keeps it. ``execution_model`` is composed
# from the parallelism spec, so it reads "plain-bf16" for the whole run and
# describes the TorchTitan arms alone; the difference lives in the scenario
# description, in the arm description, and in the report.
#
# **At pp 1 the arms process the batch the same way.** The flag list sends
# --micro-batch-size 1 at every degree, and microbatch_geometry packs the
# whole local batch into one Megatron sample at pp 1
# (benchmarks/e2e/megatron_stock/flags.py). So Megatron runs one forward and
# backward pass over local_batch_size * seq_len tokens, and each titan arm
# runs one pass over (local_batch_size, seq_len): the same tokens, the same
# GEMM rows, the same block-diagonal mask, because cu_seqlens already marks
# every document. A pp 1 ratio is not biased by the batch mapping.
ENGINES = Scenario(
    name="engines",
    description=(
        "Stock TorchTitan, compiled and eager, against stock Megatron-LM on "
        "one pre-tokenized c4_test stream. This is a systems-throughput "
        "claim about configured engines, and four deliberate differences "
        "each move the number: the Megatron arm keeps fp32 master weights "
        "and reduces gradients in fp32, runs Megatron's unfused native cross "
        "entropy, keeps --init-method-std 0.01 with no weight transfer, and "
        "applies no permutation fusion. State all four beside every number. "
        "The manifest's execution_model reads plain-bf16 because it is "
        "composed from the parallelism spec; it describes the TorchTitan "
        "arms and not the Megatron one."
    ),
    workload=PIPER_1B_MEGATRON_WORKLOAD,
    supported_ac_modes=("none",),
    arms=(
        Arm(
            name="titan_compiled",
            description=(
                "TorchTitan qwen3_piper_1b on the pre-tokenized replay "
                "stream, with whole-block torch.compile"
            ),
            compile="torch",
        ),
        Arm(
            name="titan_eager",
            description=(
                "the same model and the same stream, and it runs the blocks "
                "eager"
            ),
            compile="none",
        ),
        Arm(
            name="megatron_stock",
            description=(
                "stock megatron.training.pretrain through pretrain_gpt's own "
                "providers: alltoall dispatcher, grouped GEMM, no aux router "
                "loss, unfused native cross entropy, no permute fusion, no "
                "distributed optimizer, --init-method-std 0.01. NOT PLAIN "
                "BF16: --bf16 alone keeps fp32 master weights, fp32 "
                "optimizer moments and an fp32 gradient reduction, which is "
                "about 18 bytes of state per parameter against TorchTitan's "
                "8. The manifest's execution_model says plain-bf16 and "
                "describes the other arms"
            ),
            # The stock driver compiles no whole transformer layer.
            compile="none",
            launcher="megatron_stock",
            validation="megatron_stock",
            # Both markers are MEASURED, on every rank. The dp 2 x pp 4
            # cell at out/20260826T172258Z carries them in all eight ranks
            # of both profiler windows, at 320 and 640 per window per rank.
            # _permute_kernel is 0 on every rank there, which is why this
            # arm does not declare it: --moe-permute-fusion is off here,
            # because stock Megatron defaults it off.
            #
            # That is one mesh at one shape. A deeper split gives each stage
            # fewer layers, so re-read every rank at pp 8 before citing a
            # marker there.
            trace_kernel_markers=(
                "cudnn_generated_fort_native_sdpa",
                "_mul_silu_split",
            ),
        ),
    ),
)


SCENARIOS = {"engines": ENGINES}


def scenario_by_name(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown scenario {name!r}. Available scenarios: {', '.join(SCENARIOS)}"
        ) from error
