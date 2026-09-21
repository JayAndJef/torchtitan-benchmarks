"""Benchmark scenario definitions and the global run axes.

The declarations themselves live in ``benchmarks.e2e.schema``. This module
holds the instances, the tables and the run-axis constants built from them.

Scenarios describe what differs between arms, and nothing else: command
construction (``benchmarks.e2e.engines``), provenance collection
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

from benchmarks.e2e.schema import Arm, Scenario, Workload


EXECUTION_MODEL = "single-gpu-plain-bf16-no-fsdp"
"""How a single-GPU training process executes the model.

Plain bf16 parameters on one GPU, no FSDP wrapper and no fp32 masters.
Every manifest since schema 7 records this string, so it is a fixed point
rather than a format. The manifest no longer reads it: ``execution_model``
composes the field from the run's own mesh, and this constant is what its
trivial answer must reproduce character for character.
"""


AC_MODES = ("sac", "none")
"""The activation-checkpointing modes selectable per run.

``sac`` is TorchTitan's per-op SelectiveAC. ``none`` disables checkpointing
entirely, delivered to TorchTitan as the tyro subcommand token
``activation-checkpoint:none``.
"""

DEFAULT_AC_MODE = "none"
"""The default values of the global run axes.

``_resolve_run`` and the sweep pre-pass read these constants. Neither site
repeats the literal value.
"""
DEFAULT_MODEL_SIZE = "30b-a3b"

DEFAULT_PROFILE = False
"""Whether a run collects profiler traces.

Off is the default, and it is the treatment a published throughput number
wants: the profiler costs GPU and host time on every window, and the whole
40-step floor exists to hold two of them. Under ``--profile`` the run
collects the trace layout an external analysis tool reads, and the floor
and every trace validation rule apply again.

It is a run axis and not an arm property. Both engines either write the
layout or write nothing, and a run that profiled one arm and not another
would publish two treatments under one label.
"""

DEFAULT_WARMUP_STEPS = 10
"""How many steps an unprofiled run discards before it measures.

The engines compile, autotune and fill their caches in the first steps of a
run, so a throughput taken over them is not the throughput of the workload.
Under ``--profile`` the profiler schedule decides the sample set instead
and this axis is refused; without it every step after the warmup is a
sample.

10 is half of one profiler cycle, which is the span the profiled rule
samples. The two rules therefore discard a comparable prefix, but neither
figure is the other's: results are only comparable within one value of this
axis.
"""

MEGATRON_P2P_SYNC_MODES = ("on", "off")
"""The Megatron pipeline point-to-point sync treatment, per run.

Stock Megatron calls ``torch.cuda.synchronize()`` once per batched pipeline
message, guarded by ``config.batch_p2p_comm and config.batch_p2p_sync``.
``on`` keeps that call, and it is what every number published before this
default flipped was measured under. ``off`` sets ``batch_p2p_sync`` False
on the megatron driver, which removes the call, and it is the default here.
TorchTitan arms receive nothing.

It is a run axis and not a ``ParallelismSpec`` field, because the value is
a treatment of the pipeline messages and ``execution_model`` names degrees
rather than mechanisms. Megatron exposes no CLI flag for the field, so the
driver takes the value from the harness and prints what its built config
carries.
"""
DEFAULT_MEGATRON_P2P_SYNC = "off"

MEGATRON_NAN_GUARD_MODES = ("on", "off")
"""Stock Megatron's NaN/Inf guard, selectable per run.

One Megatron argument, ``check_for_nan_in_loss_and_grad``, gates two host
waits at the pinned revision. The loss function evaluates the loss twice
per microbatch, and the same field reaches ``check_for_nan_in_grad``, under
which the gradient buffer evaluates every bucket's norm twice per step.
Each evaluation reads a device bool and synchronizes the stream.
``--rerun-mode disabled``, which the stock argv already sends, removes
neither.

``on`` is stock Megatron, and it is what every number published before this
default flipped was measured under. ``off`` sends Megatron's own
``--no-check-for-nan-in-loss-and-grad``, so a stock user can reproduce the
argv, and it is the default here. TorchTitan arms receive nothing.
Evaluation refuses a non-finite loss or grad norm on every arm under either
value, which is the guard that has to exist before this one can be turned
off.
"""
DEFAULT_MEGATRON_NAN_GUARD = "off"

MEGATRON_PRECISION_MODES = ("stock", "lean")
"""Stock Megatron's optimizer precision, selectable per run.

``stock`` is ``--bf16`` alone, and it holds 18 bytes of state per
parameter: the bf16 parameter 2, an fp32 master 4, fp32 gradients 4, and
two fp32 Adam moments 8. That is what every published cell ran, and it is
the first of the four deliberate differences the stock arm carries against
TorchTitan's 8.

``lean`` sends ``--use-precision-aware-optimizer`` and three bf16 dtype
flags, and reaches 10 bytes: master 2, gradients 2, Adam moments 4. The
master stays fp32 and is never fp16, because ``store_param_remainders``
holds the low 16 bits beside the bf16 parameter.
``--grad-reduce-in-bf16`` is never sent, because ``--main-grads-dtype
bf16`` already leaves fp32 accumulation off.

``lean`` needs a sharded dense value: Megatron asserts
``use_distributed_optimizer`` under the precision-aware optimizer, and the
zero axis is the one owner of that flag. The value reaches the stock
megatron command alone.

Warning: ``lean`` changes the numerics. bf16 Adam moments and bf16 gradient
accumulation are a real change, and at pp 8 the accumulation is 16-way in
bf16. Read the loss trajectories beside any lean number. At dp 1 the
distributed optimizer also runs its bucket bookkeeping for no saving. Both
effects are unmeasured.
"""
DEFAULT_MEGATRON_PRECISION = "stock"


C4_REPLAY_WORKLOAD = Workload(
    module="benchmarks.models.piper_qwen3",
    config="qwen3_piper_1b_pretokenized",
    seq_len=4096,
    steps=40,
    local_batch_size=4,
    seed=42,
    replay_dataloader=True,
)
"""The one workload every arm of every scenario runs.

The shape is not here. ``--model-size`` picks it at run time, and the
config name is a fixed token of the fork's config manager rather than a
shape. ``seed`` and ``replay_dataloader`` serve both engines: the launcher
reads them for the TorchTitan arms and the flag list reads the seed for the
Megatron arm.
"""

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
    workload=C4_REPLAY_WORKLOAD,
    supported_ac_modes=("none",),
    arms=(
        Arm(
            name="titan_compiled",
            description=(
                "TorchTitan on the pre-tokenized replay stream, with "
                "whole-block torch.compile"
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
            engine="megatron_stock",
            # Both markers are MEASURED, on every rank. The dp 2 x pp 4
            # cell at out/20260826T172258Z carries them in all eight ranks
            # of both profiler windows, at 320 and 640 per window per rank.
            # _permute_kernel is 0 on every rank there, which is why this
            # arm does not declare it: --moe-permute-fusion is off here,
            # because stock Megatron defaults it off.
            #
            # A deeper split gives each stage fewer layers, so re-read
            # every rank at pp 8 before citing a marker there.
            trace_kernel_markers=(
                "cudnn_generated_fort_native_sdpa",
                "_mul_silu_split",
            ),
        ),
    ),
)
"""The engine comparison, on one pre-tokenized c4_test stream.

Three arms answer one question: what does each engine cost per token at
this mesh. The cross-engine metrics are tokens/s, step time and peak
memory. The ac axis is pinned to ``none``, because Megatron's recompute
options are not parity with TorchTitan's per-op SAC and the Megatron arm
does no recompute at all.

The Megatron arm is not plain bf16, and the manifest cannot say so. Under
``--bf16`` alone Megatron keeps fp32 master weights and fp32 optimizer
moments, and it reduces gradients in fp32, so the arm holds about 18 bytes
per parameter against TorchTitan's 8. That is the stock treatment, and this
scenario keeps it. ``execution_model`` is composed from the parallelism
spec, so it describes the TorchTitan arms alone; the difference lives in
the scenario description, in the arm description, and in the report.

At pp 1 the arms process the batch the same way. The flag list sends
``--micro-batch-size 1`` at every degree, and ``microbatch_geometry`` packs
the whole local batch into one Megatron sample at pp 1. So both engines run
one forward and backward pass over the same tokens, the same GEMM rows and
the same block-diagonal mask, because ``cu_seqlens`` already marks every
document. A pp 1 ratio is not biased by the batch mapping.
"""


SCENARIOS = {"engines": ENGINES}


def scenario_by_name(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown scenario {name!r}. Available scenarios: {', '.join(SCENARIOS)}"
        ) from error
