"""Benchmark scenario definitions and the global run axes.

Scenarios describe what differs between arms, and nothing else: command
construction (``benchmarks.e2e.launch``), provenance collection
(``benchmarks.execution.provenance``), and validation
(``benchmarks.e2e.validation``) each live in their own module and read these
declarations. That is what lets a new ablation be a registry entry rather
than a new copy of the training harness.

The compile-mode, activation-checkpointing, and execution-model constants
live here too: they are per-run axes of a scenario execution, consumed by
command construction (``benchmarks.e2e.launch``), validation
(``benchmarks.e2e.validation``), and the manifest
(``benchmarks.artifacts.manifests``).
"""

from dataclasses import dataclass, replace

from benchmarks.models.piper_qwen3.shape import PIPER_1B
from benchmarks.traces.schema import Region


# How the training process executes the model. Constant since schema 7:
# plain bf16 params on one GPU, no FSDP wrapper, no fp32 masters
# (benchmarks.models.piper_qwen3.parallelize). Recorded so a manifest
# self-describes without a git-rev lookup; earlier schemas ran under FSDP2
# mixed precision.
EXECUTION_MODEL = "single-gpu-plain-bf16-no-fsdp"

# Engine-neutral compile modes selectable per run. "cuda-graph" replaced the
# torch-level name "reduce-overhead" in schema 8; TORCH_COMPILE_MODE maps it
# back to the --compile.mode value the TorchTitan fork applies per block.
# The two max-autotune modes were removed in schema 8 after the full matrix
# showed them to be GPU-time regressions at these shapes (see
# reports/20260807-mode-matrix-plain-bf16.md); schema <= 7 manifests may
# still record them and the old reduce-overhead name.
COMPILE_MODES = ("default", "cuda-graph")
TORCH_COMPILE_MODE = {"default": "default", "cuda-graph": "reduce-overhead"}
CUDAGRAPH_COMPILE_MODES = frozenset({"cuda-graph"})

# Activation checkpointing modes selectable per run (schema 8). "sac" is
# TorchTitan's per-op SelectiveAC (the historical treatment, implied by
# schema <= 7 manifests); "none" disables checkpointing entirely, delivered
# to TorchTitan as the tyro subcommand token "activation-checkpoint:none".
AC_MODES = ("sac", "none")


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
    """

    name: str
    description: str
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
    regions: tuple[Region, ...] = ()
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

PIPER_1B_UNFUSED_QKV_WORKLOAD = replace(
    PIPER_1B_WORKLOAD,
    config="qwen3_piper_1b_unfused_qkv",
    seed=42,
)

PIPER_1B_LM_HEAD_WORKLOAD = replace(
    PIPER_1B_WORKLOAD,
    config="qwen3_piper_1b_full_logits",
    seed=42,
)

def piper_block_regions(
    *, n_layers: int, profiler_active: int
) -> tuple[Region, ...]:
    """Per-block compiled regions for a model of ``n_layers`` layers.

    Each TransformerBlock emits one forward and one backward CompiledFxGraph
    annotation per step, so a profiler window holds
    ``n_layers * profiler_active`` invocations of each. That count is also
    the identity: measured on a real 16-layer trace, the forward graphs run
    {5, 80, 5} times per window and the backward graphs {5, 80}, so only a
    multi-layer model produces a count unique to the block graphs. At one
    layer the block graph would also run 5 times and
    ``pooled_window_metrics`` could not tell it from the loss- and
    embedding-side partitions -- which is why ``supports_block_regions`` is
    derived as ``n_layers > 1`` (False at huge) and such a run declares no
    regions rather than adding a tiebreak that would weaken validation
    rule 7.
    """
    invocations = n_layers * profiler_active
    return (
        Region(
            name="backward_block",
            phase="backward",
            invocations_per_window=invocations,
        ),
        Region(
            name="forward_block",
            phase="forward",
            invocations_per_window=invocations,
        ),
    )


# The normal-size instantiation: 16 layers x 5 active steps = 80.
PIPER_1B_REGIONS = piper_block_regions(
    n_layers=PIPER_1B.n_layers, profiler_active=5
)


PIPER_1B_ROPE = Scenario(
    name="piper1b_rope",
    description="TorchTitan RoPE, Helion RoPE, and TransformerEngine RoPE on piper-1B.",
    workload=PIPER_1B_WORKLOAD,
    regions=PIPER_1B_REGIONS,
    arms=(
        Arm(
            name="baseline",
            description="TorchTitan CosSinRoPE; rotate-half math fused by Inductor into block kernels",
        ),
        Arm(
            name="helion",
            description="TorchTitan HelionCosSinRoPE kernel, swapped in via config override",
            override_imports=(
                "torchtitan.overrides.helion_rope.helion_cos_sin_rope",
            ),
            overrides_per_block=1,
            trace_kernel_markers=("_helion__rope_cos_sin_fwd",),
        ),
        Arm(
            name="te",
            description="TransformerEngine CUDA RoPE (JIT-built, needs gcc-13), via config override",
            override_imports=(
                "benchmarks.models.piper_qwen3.components.rope.te_rope_override.te_rope",
            ),
            overrides_per_block=1,
            trace_kernel_markers=("fused_rope_forward_positions_kernel",),
            requires_gcc_toolset=True,
        ),
    ),
)


PIPER_1B_SWIGLU = Scenario(
    name="piper1b_swiglu",
    description="TorchTitan MoE SwiGLU versus the two Piper grouped-expert variants on piper-1B.",
    workload=PIPER_1B_WORKLOAD,
    regions=PIPER_1B_REGIONS,
    arms=(
        Arm(
            name="baseline",
            description="TorchTitan modern GroupedExperts: separate w1/w3 grouped GEMMs, plain-ops activation",
        ),
        Arm(
            name="piper_optimized_triton",
            description="fused w13 grouped GEMM + combined [R,2F] custom Triton activation op, via config override",
            override_imports=(
                "benchmarks.models.piper_qwen3.components.swiglu.combined_swiglu.piper_optimized_triton_fused_grouped_experts",
            ),
            overrides_per_block=1,
            trace_kernel_markers=(
                "_combined_silu_and_mul_forward_kernel",
                "_combined_silu_and_mul_backward_kernel",
            ),
        ),
        Arm(
            name="piper_optimized_inductor",
            description="fused w13 grouped GEMM, plain-ops SwiGLU left to Inductor, via config override",
            override_imports=(
                "benchmarks.models.piper_qwen3.components.swiglu.combined_swiglu.piper_optimized_inductor_fused_grouped_experts",
            ),
            overrides_per_block=1,
            # No trace_kernel_markers: the activation is deliberately plain
            # ops with no distinctive kernel name; Inductor fuses it into
            # neighboring generated kernels. The [Override] count is the
            # application check.
        ),
    ),
)


PIPER_1B_QKV = Scenario(
    name="piper1b_qkv",
    description="Separate Q/K/V projections versus fused QKV on piper-1B.",
    workload=PIPER_1B_UNFUSED_QKV_WORKLOAD,
    regions=PIPER_1B_REGIONS,
    arms=(
        Arm(
            name="baseline",
            description="TorchTitan QKVLinear: separate Q and KV GEMMs (qwen3_piper_1b_unfused_qkv config)",
        ),
        Arm(
            name="fused_qkv",
            description="TorchTitan FusedQKVLinear: one wqkv GEMM plus split (qwen3_piper_1b config)",
            config="qwen3_piper_1b",
        ),
    ),
)


PIPER_1B_LM_HEAD = Scenario(
    name="piper1b_lm_head",
    description=(
        "Piper full logits versus full-token PyTorch fused linear-CE and "
        "reference and Piper-optimized TransformerEngine fused CE."
    ),
    workload=PIPER_1B_LM_HEAD_WORKLOAD,
    regions=PIPER_1B_REGIONS,
    arms=(
        Arm(
            name="baseline",
            description="full-logits F.linear then TorchTitan CrossEntropyLoss, compiled",
        ),
        Arm(
            name="fused_linear_ce",
            description="torch.nn.functional.linear_cross_entropy: CE without materializing full logits",
            config="qwen3_piper_1b_fused_linear_ce",
        ),
        Arm(
            name="te_fused_ce",
            description="full logits then the vendored TransformerEngine Triton cross entropy",
            config="qwen3_piper_1b_te_fused_ce",
            trace_kernel_markers=("online_softmax_kernel", "cross_entropy_kernel"),
        ),
        Arm(
            name="piper_optimized_te_ce",
            description="TE CE reworked into one Triton kernel writing the pre-scaled bf16 grad in forward (TE: 2 fwd kernels + a bwd scaling pass)",
            config="qwen3_piper_1b_piper_optimized_te_ce",
            trace_kernel_markers=("piper_optimized_cross_entropy_kernel",),
        ),
    ),
)


PIPER_1B_MEGATRON_WORKLOAD = replace(
    PIPER_1B_WORKLOAD,
    config="qwen3_piper_1b_pretokenized",
    seed=42,
    replay_dataloader=True,
)

_PIPER_OPTIMIZED_SWIGLU_INDUCTOR = (
    "benchmarks.models.piper_qwen3.components.swiglu.combined_swiglu."
    "piper_optimized_inductor_fused_grouped_experts"
)

# The engine comparison: Megatron-LM is the baseline, the other arms are the
# best-improved TorchTitan configurations from the compile/ac matrix, all on
# a bit-identical pre-tokenized data stream. No per-block regions: region
# pooling rides on Inductor's compiled-graph annotations, which an eager
# Megatron arm honestly does not have — total GPU kernel time, tokens/s,
# launch latency, and peak memory are the cross-engine metrics (per-block
# detail for the titan arms lives in the four scenarios above). ac mode is
# pinned to "none": Megatron-at-its-best does no recompute and its recompute
# options are not parity with titan's per-op SAC.
PIPER_1B_MEGATRON = Scenario(
    name="piper1b_megatron",
    description=(
        "Megatron-LM (TransformerEngine) versus the best-improved TorchTitan "
        "configurations on identical data; single GPU, plain bf16, no AC."
    ),
    workload=PIPER_1B_MEGATRON_WORKLOAD,
    regions=(),
    supported_ac_modes=("none",),
    arms=(
        Arm(
            name="baseline",
            description=(
                "Megatron-LM + TE: bare GPTModel, THD packed attention, no "
                "recompute. --ac never affects this arm; under cuda-graph "
                "mode it uses Megatron's per-layer partial capture, which "
                "covers far less of the step than titan's whole-block graphs"
            ),
            launcher="megatron",
            validation="megatron",
            # cuDNN fused attention (a silent TE fallback to unfused
            # attention), megatron's fused SwiGLU+probs kernel, and TE's
            # fused MoE permute. The SwiGLU marker exists because the arm
            # once ran the unfused chunk/silu/mul path for a whole report:
            # it passed every other rule while costing 11.9 GPU ms/step.
            trace_kernel_markers=(
                "cudnn_generated_fort_native_sdpa",
                "_mul_silu_split",
                "_permute_kernel",
            ),
        ),
        Arm(
            name="titan_stock",
            description=(
                "TorchTitan qwen3_piper_1b (fused qkv, stock kernels) on the "
                "pre-tokenized replay stream — the engine-gap bridge arm"
            ),
        ),
        Arm(
            name="titan_swiglu",
            description=(
                "stock + piper_optimized_inductor fused-w13 grouped experts, "
                "via config override"
            ),
            override_imports=(_PIPER_OPTIMIZED_SWIGLU_INDUCTOR,),
            overrides_per_block=1,
        ),
        Arm(
            name="titan_lm_head",
            description="stock + the piper_optimized_te_ce loss",
            config="qwen3_piper_1b_piper_optimized_te_ce_pretokenized",
            trace_kernel_markers=("piper_optimized_cross_entropy_kernel",),
        ),
        Arm(
            name="titan_swiglu_lm_head",
            description="both improvements combined",
            config="qwen3_piper_1b_piper_optimized_te_ce_pretokenized",
            override_imports=(_PIPER_OPTIMIZED_SWIGLU_INDUCTOR,),
            overrides_per_block=1,
            trace_kernel_markers=("piper_optimized_cross_entropy_kernel",),
        ),
    ),
)


# Captured by profiling, never guessed: FA4's kernels are emitted by the CuTe
# DSL at compile time and their names appear nowhere in the torch source. The
# full symbols are long CUTLASS manglings; these two substrings are the stable
# parts, and Postprocess/Preprocess deliberately do not match the bwd marker.
_FA4_TRACE_MARKERS = ("FlashAttentionForwardSm90", "FlashAttentionBackwardSm90")


PIPER_1B_ATTENTION = Scenario(
    name="piper1b_attention",
    description=(
        "Inner-attention backends on piper-1B: FlexAttention versus "
        "FlashAttention-3 varlen versus FlexAttention lowered to "
        "FlashAttention-4. TE cannot be an arm here -- see CLAUDE.md."
    ),
    workload=PIPER_1B_WORKLOAD,
    regions=PIPER_1B_REGIONS,
    arms=(
        Arm(
            name="baseline",
            description="TorchTitan FlexAttention: an Inductor Triton template over a block-diagonal causal BlockMask (qwen3_piper_1b config)",
        ),
        Arm(
            name="flash_attention_3",
            description="FlashAttention-3 varlen over packed documents (qwen3_piper_1b_varlen config); needs the flash3 dependency group",
            config="qwen3_piper_1b_varlen",
            # FA3 degrades to FA2 rather than failing when it declines to
            # register, so pin its own kernel name: seeing pytorch_flash::
            # instead would mean the arm measured FA2 under an FA3 label.
            trace_kernel_markers=("FlashAttnFwdSm90", "FlashAttnBwdSm90"),
        ),
        Arm(
            name="flex_flash",
            description=(
                "FlexAttention lowered to FlashAttention-4 CuTe DSL kernels "
                "(qwen3_piper_1b_flex_flash config); same BlockMask as "
                "baseline, so this pair isolates the kernel family. Needs the "
                "fa4 group"
            ),
            config="qwen3_piper_1b_flex_flash",
            trace_kernel_markers=_FA4_TRACE_MARKERS,
        ),
    ),
)


SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        PIPER_1B_ROPE,
        PIPER_1B_SWIGLU,
        PIPER_1B_QKV,
        PIPER_1B_LM_HEAD,
        PIPER_1B_ATTENTION,
        PIPER_1B_MEGATRON,
    )
}


def scenario_by_name(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown scenario {name!r}. Available scenarios: {', '.join(SCENARIOS)}"
        ) from error
