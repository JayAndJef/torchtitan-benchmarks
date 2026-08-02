"""Benchmark scenario definitions.

Scenarios describe what differs between arms. The runner owns command
construction, provenance collection, and validation, so new ablations do not
need to duplicate the training harness.
"""

from dataclasses import dataclass


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


@dataclass(frozen=True)
class Arm:
    """One implementation measured by a scenario."""

    name: str
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
    description="Stock RoPE, Helion RoPE, and TransformerEngine RoPE on piper-1B.",
    workload=PIPER_1B_WORKLOAD,
    regions=PIPER_1B_REGIONS,
    arms=(
        Arm(name="baseline"),
        Arm(
            name="helion",
            override_imports=(
                "torchtitan.overrides.helion_rope.helion_cos_sin_rope",
            ),
            expected_override_count=16,
            trace_kernel_markers=("_helion__rope_cos_sin_fwd",),
        ),
        Arm(
            name="te",
            override_imports=("te_rope_override.te_rope",),
            expected_override_count=16,
            trace_kernel_markers=("fused_rope_forward_kernel",),
            requires_gcc_toolset=True,
        ),
    ),
)


PIPER_1B_SWIGLU = Scenario(
    name="piper1b_swiglu",
    description="Stock MoE SwiGLU versus fused grouped-experts SwiGLU on piper-1B.",
    workload=PIPER_1B_WORKLOAD,
    regions=PIPER_1B_REGIONS,
    arms=(
        Arm(name="baseline"),
        Arm(
            name="fused_grouped_experts",
            override_imports=(
                "torchtitan.overrides.fused_swiglu.fused_grouped_experts",
            ),
            expected_override_count=16,
        ),
    ),
)


SCENARIOS = {scenario.name: scenario for scenario in (PIPER_1B_ROPE, PIPER_1B_SWIGLU)}


def scenario_by_name(name: str) -> Scenario:
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown scenario {name!r}. Available scenarios: {', '.join(SCENARIOS)}"
        ) from error
