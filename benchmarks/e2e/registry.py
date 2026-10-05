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

MOE_OVERRIDES = "benchmarks.models.piper_qwen3.components.moe"

FA3_ATTENTION = f"{ATTENTION_OVERRIDES}.fa3_override.packed_fa3_attention"
"""The override import of FA3 varlen attention, which reads the loader's offsets."""

TE_GROUPED_EXPERTS = f"{MOE_OVERRIDES}.te_grouped_experts.te_grouped_experts"
"""The override import of the routed-expert GEMMs on TE's cuBLASLt grouped GEMM."""

FA3_MARKERS = ("FlashAttnFwdSm90",)
"""The trace markers of FA3 varlen attention."""

TE_GROUPED_GEMM_MARKERS = ("setup_grouped_gemm_kernel", "_ptrGroup_")
"""The trace markers of TE's grouped GEMM: its setup kernel and the cuBLASLt grouped kernel."""

OVERRIDES = Scenario(
    name="overrides",
    description=(
        "Compiled TorchTitan in a 2 x 2 grid of overrides, against stock "
        "Megatron-LM. One axis is the attention kernel: FlexAttention or FA3 "
        "varlen, which reads the exact document offsets of each microbatch "
        "from the loader. The other axis is the expert GEMM library: torch's "
        "_grouped_mm or TransformerEngine's cuBLASLt grouped GEMM. Each "
        "single-axis arm moves one axis against titan_compiled, and the "
        "stacked arm moves both. The Megatron arm carries the four "
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
            name="titan_compiled_fa3_te_gemm",
            description=(
                "titan_compiled with FA3 varlen attention on packed documents "
                "and the expert GEMMs on TE's cuBLASLt grouped GEMM"
            ),
            config=TorchTitanConfig(
                compile=CompileMode.TORCH,
                overrides_per_block=2,
                override_imports=(FA3_ATTENTION, TE_GROUPED_EXPERTS),
                trace_kernel_markers=FA3_MARKERS + TE_GROUPED_GEMM_MARKERS,
                packed_offsets=True,
            ),
        ),
        ENGINES.arm("megatron_stock"),
    ),
)
"""The override comparison: what each attention kernel and each expert GEMM library costs, alone and stacked; it refuses ``--ac sac``."""


SCENARIOS = {"engines": ENGINES, "overrides": OVERRIDES}
"""Every end-to-end scenario, by name."""


def scenario_by_name(name: str) -> Scenario:
    """The scenario ``name``; an unknown name raises and names the choices."""
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown scenario {name!r}. Available scenarios: {', '.join(SCENARIOS)}"
        ) from error
