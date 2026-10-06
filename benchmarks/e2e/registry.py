"""The scenarios, their arms and the defaults of the run-wide values."""

from benchmarks.e2e.engines.api import (
    REPLAY_DATASET,
    Arm,
    CompileMode,
    DataSpec,
    ProfileWindow,
)
from benchmarks.e2e.engines.megatron_stock.config import MegatronStockConfig
from benchmarks.e2e.engines.torchtitan.config import TorchTitanConfig
from benchmarks.e2e.schema import Scenario


AC_MODES = ("sac", "none")
"""The activation checkpointing modes: TorchTitan's per-op SelectiveAC, or none."""

DEFAULT_AC_MODE = "none"

DEFAULT_MODEL_SIZE = "30b-a3b"

DEFAULT_PROFILE = False
"""Whether a run collects profiler traces; a published throughput number wants no profiler."""

DEFAULT_WARMUP_STEPS = 10
"""How many steps an unprofiled run discards before it measures."""

SEED = 42
"""The seed of every run, which each engine passes to its own initialization."""

C4_REPLAY_DATA = DataSpec(
    dataset=REPLAY_DATASET,
    seq_len=4096,
    local_batch_size=4,
    steps=40,
)
"""The pre-tokenized c4_test stream that every arm trains on."""

ENGINES = Scenario(
    name="engines",
    description=(
        "Stock TorchTitan, compiled and eager, against stock Megatron-LM on "
        "one pre-tokenized c4_test stream. This is a systems-throughput "
        "claim about configured engines, and four deliberate differences "
        "each move the number: the Megatron arm keeps fp32 master weights "
        "and reduces gradients in fp32, runs Megatron's unfused native cross "
        "entropy, keeps --init-method-std 0.01 with no weight transfer, and "
        "applies no permutation fusion. State all four beside every number."
    ),
    data=C4_REPLAY_DATA,
    window=ProfileWindow(),
    supported_ac_modes=("none",),
    arms=(
        Arm(
            name="titan_compiled",
            description=(
                "TorchTitan on the pre-tokenized replay stream, with "
                "whole-block torch.compile"
            ),
            config=TorchTitanConfig(compile=CompileMode.TORCH),
        ),
        Arm(
            name="titan_eager",
            description=(
                "the same model and the same stream, and it runs the blocks "
                "eager"
            ),
            config=TorchTitanConfig(compile=CompileMode.NONE),
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
                "8"
            ),
            config=MegatronStockConfig(
                # Measured on all eight ranks of one dp 2 x pp 4 cell.
                trace_kernel_markers=(
                    "cudnn_generated_fort_native_sdpa",
                    "_mul_silu_split",
                ),
            ),
        ),
    ),
)
"""The engine comparison: what each engine costs per token at one mesh; it refuses ``--ac sac``, because Megatron has no recompute that matches TorchTitan's per-op SAC."""


ATTENTION_OVERRIDES = "benchmarks.models.piper_qwen3.components.attention"

FA3_ATTENTION = f"{ATTENTION_OVERRIDES}.fa3_override.packed_fa3_attention"
"""The override import of FA3 varlen attention on the loader's packed document offsets."""

FA3_MARKERS = ("FlashAttnFwdSm90",)
"""The trace marker of FA3 varlen attention: its Hopper forward kernel."""

ATTENTION = Scenario(
    name="attention",
    description=(
        "Compiled TorchTitan with two attention kernels, FlexAttention and FA3 "
        "varlen, against stock Megatron-LM, which runs TransformerEngine's "
        "cuDNN attention. The FA3 arm reads the exact document offsets of "
        "each microbatch, which the loader computes on the CPU. The "
        "Megatron arm carries the four "
        "differences of the engines scenario: fp32 master weights and an fp32 "
        "gradient reduction, unfused native cross entropy, "
        "--init-method-std 0.01 with no weight transfer, and no permutation "
        "fusion. State all four beside every cross-engine number."
    ),
    data=C4_REPLAY_DATA,
    window=ProfileWindow(),
    supported_ac_modes=("none",),
    arms=(
        ENGINES.arm("titan_compiled"),
        Arm(
            name="titan_compiled_fa3",
            description=(
                "titan_compiled with FA3 varlen attention on packed documents"
            ),
            config=TorchTitanConfig(
                compile=CompileMode.TORCH,
                overrides_per_block=1,
                override_imports=(FA3_ATTENTION,),
                trace_kernel_markers=FA3_MARKERS,
                packed_offsets=True,
            ),
        ),
        ENGINES.arm("megatron_stock"),
    ),
)
"""The attention comparison: whether a fused varlen kernel closes the attention gap between the engines; it refuses ``--ac sac``."""


MOE_OVERRIDES = "benchmarks.models.piper_qwen3.components.moe"

TE_GROUPED_EXPERTS = f"{MOE_OVERRIDES}.te_grouped_experts.te_grouped_experts"
"""The override import of the routed-expert GEMMs on TE's cuBLASLt grouped GEMM."""

TE_GROUPED_GEMM_MARKERS = ("setup_grouped_gemm_kernel", "_ptrGroup_")
"""The trace markers of TE's grouped GEMM: its setup kernel and the cuBLASLt grouped kernel."""

HOST_COUNT_DISPATCHER = f"{MOE_OVERRIDES}.host_count_dispatcher.host_count_dispatcher"
"""The override import of the all-to-all dispatcher that returns the rows of each local expert on the host."""

TE_PER_EXPERT_EXPERTS = f"{MOE_OVERRIDES}.te_per_expert_experts.te_per_expert_experts"
"""The override import of the routed-expert GEMMs as one TE cuBLAS GEMM per expert; it needs HOST_COUNT_DISPATCHER."""

TE_PER_EXPERT_MARKERS = ("torchtitan_benchmarks::te_per_expert_mm",)
"""The trace marker of the per-expert GEMMs: the profiler range of each op, because cuBLAS picks its kernel from the rows of each expert."""

EXPERTS = Scenario(
    name="experts",
    description=(
        "Compiled TorchTitan with three expert GEMM paths, against stock "
        "Megatron-LM: torch's _grouped_mm, TransformerEngine's cuBLASLt "
        "grouped GEMM with the split sizes on the device, and one "
        "TransformerEngine cuBLAS GEMM per expert with the split sizes on "
        "the host, as Megatron's GroupedLinear runs them. Each TE arm moves "
        "the expert GEMM path alone against titan_compiled. The TorchTitan "
        "arms share their init, seed and data, so their routing matches. "
        "The per-expert arm needs an expert-parallel degree above 1. The "
        "Megatron arm carries the four differences of the engines scenario: "
        "fp32 master weights and an fp32 gradient reduction, unfused native "
        "cross entropy, --init-method-std 0.01 with no weight transfer, and "
        "no permutation fusion. State all four beside every cross-engine "
        "number."
    ),
    data=C4_REPLAY_DATA,
    window=ProfileWindow(),
    supported_ac_modes=("none",),
    arms=(
        ENGINES.arm("titan_compiled"),
        Arm(
            name="titan_compiled_te_gemm",
            description=(
                "titan_compiled with the expert GEMMs on TE's cuBLASLt "
                "grouped GEMM"
            ),
            config=TorchTitanConfig(
                compile=CompileMode.TORCH,
                overrides_per_block=1,
                override_imports=(TE_GROUPED_EXPERTS,),
                trace_kernel_markers=TE_GROUPED_GEMM_MARKERS,
            ),
        ),
        Arm(
            name="titan_compiled_te_per_expert",
            description=(
                "titan_compiled with the expert GEMMs as one TE cuBLAS GEMM "
                "per expert, which reads the rows of each expert from the "
                "dispatcher's one host copy"
            ),
            config=TorchTitanConfig(
                compile=CompileMode.TORCH,
                overrides_per_block=2,
                override_imports=(HOST_COUNT_DISPATCHER, TE_PER_EXPERT_EXPERTS),
                trace_kernel_markers=TE_PER_EXPERT_MARKERS,
            ),
        ),
        ENGINES.arm("megatron_stock"),
    ),
)
"""The expert GEMM comparison: what each expert GEMM path costs, at the same routing; it refuses ``--ac sac``."""


STACKED = Scenario(
    name="stacked",
    description=(
        "Compiled TorchTitan with FA3 varlen attention and one "
        "TransformerEngine cuBLAS GEMM per expert together, against "
        "titan_compiled and stock Megatron-LM. The stacked arm moves two "
        "axes against titan_compiled, the attention kernel and the expert "
        "GEMM path, so it answers whether the two gains add. FA3 alone is "
        "the attention arm titan_compiled_fa3, and the per-expert GEMM alone "
        "is the experts arm titan_compiled_te_per_expert. The stacked arm "
        "needs an expert-parallel degree above 1. The Megatron arm carries "
        "the four differences of the engines scenario: fp32 master weights "
        "and an fp32 gradient reduction, unfused native cross entropy, "
        "--init-method-std 0.01 with no weight transfer, and no permutation "
        "fusion. State all four beside every cross-engine number."
    ),
    data=C4_REPLAY_DATA,
    window=ProfileWindow(),
    supported_ac_modes=("none",),
    arms=(
        ENGINES.arm("titan_compiled"),
        Arm(
            name="titan_compiled_fa3_te_per_expert",
            description=(
                "titan_compiled with FA3 varlen attention on packed documents "
                "and the expert GEMMs as one TE cuBLAS GEMM per expert"
            ),
            config=TorchTitanConfig(
                compile=CompileMode.TORCH,
                overrides_per_block=3,
                override_imports=(
                    FA3_ATTENTION,
                    HOST_COUNT_DISPATCHER,
                    TE_PER_EXPERT_EXPERTS,
                ),
                trace_kernel_markers=(*FA3_MARKERS, *TE_PER_EXPERT_MARKERS),
                packed_offsets=True,
            ),
        ),
        ENGINES.arm("megatron_stock"),
    ),
)
"""The stacked comparison: whether the FA3 and the per-expert GEMM gains add; it refuses ``--ac sac``."""


SCENARIOS = {
    "engines": ENGINES,
    "attention": ATTENTION,
    "experts": EXPERTS,
    "stacked": STACKED,
}
"""Every end-to-end scenario, by name."""


def scenario_by_name(name: str) -> Scenario:
    """The scenario ``name``; an unknown name raises and names the choices."""
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown scenario {name!r}. Available scenarios: {', '.join(SCENARIOS)}"
        ) from error
