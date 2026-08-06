"""Benchmark scenario definitions.

Scenarios describe what differs between arms. The runner owns command
construction, provenance collection, and validation, so new ablations do not
need to duplicate the training harness.
"""

from dataclasses import dataclass, replace


@dataclass(frozen=True)
class Workload:
    """Training settings shared by every arm in one scenario."""

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
    expected_override_count: int = 0
    trace_kernel_markers: tuple[str, ...] = ()
    requires_gcc_toolset: bool = False


@dataclass(frozen=True)
class Region:
    """One reported compiled region, identified by direction and call count.

    Graph hashes differ between arms, so a region is matched structurally:
    ``phase`` ("forward" or "backward") comes from whether the graph's CPU
    annotations nest inside ``CompiledFunctionBackward`` autograd frames, and
    ``invocations_per_window`` picks the graph among same-phase partitions
    while pinning the expected sample count. A count mismatch means the
    compiler partitioned the model differently and the graph-to-region
    mapping is no longer valid.
    """

    name: str
    phase: str
    invocations_per_window: int


@dataclass(frozen=True)
class Scenario:
    """A reproducible workload and its comparable implementation arms."""

    name: str
    description: str
    workload: Workload
    arms: tuple[Arm, ...]
    regions: tuple[Region, ...] = ()

    def arm(self, name: str) -> Arm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise ValueError(
            f"Unknown arm {name!r} for scenario {self.name!r}. "
            f"Available arms: {', '.join(arm.name for arm in self.arms)}"
        )


PIPER_1B_WORKLOAD = Workload(
    module="piper1b",
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

# Compiled-region layout of the piper-1B workload under --compile.enable:
# each of the 16 transformer blocks emits one forward and one backward
# CompiledFxGraph annotation per step, so a 5-step profiler window holds
# 16 x 5 = 80 invocations of each — a count unique to the block graphs
# (the embedding/loss-side partitions run 40 and 5 times and are not
# reported).
PIPER_1B_REGIONS = (
    Region(name="backward_block", phase="backward", invocations_per_window=80),
    Region(name="forward_block", phase="forward", invocations_per_window=80),
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
            expected_override_count=16,
            trace_kernel_markers=("_helion__rope_cos_sin_fwd",),
        ),
        Arm(
            name="te",
            description="TransformerEngine CUDA RoPE (JIT-built, needs gcc-13), via config override",
            override_imports=("piper1b.rope.te_rope_override.te_rope",),
            expected_override_count=16,
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
                "piper1b.swiglu.combined_swiglu.piper_optimized_triton_fused_grouped_experts",
            ),
            expected_override_count=16,
            trace_kernel_markers=(
                "_combined_silu_and_mul_forward_kernel",
                "_combined_silu_and_mul_backward_kernel",
            ),
        ),
        Arm(
            name="piper_optimized_inductor",
            description="fused w13 grouped GEMM, plain-ops SwiGLU left to Inductor, via config override",
            override_imports=(
                "piper1b.swiglu.combined_swiglu.piper_optimized_inductor_fused_grouped_experts",
            ),
            expected_override_count=16,
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


SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        PIPER_1B_ROPE,
        PIPER_1B_SWIGLU,
        PIPER_1B_QKV,
        PIPER_1B_LM_HEAD,
    )
}


def scenario_by_name(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown scenario {name!r}. Available scenarios: {', '.join(SCENARIOS)}"
        ) from error
