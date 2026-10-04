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
                override_imports=(
                    f"{ATTENTION_OVERRIDES}.fa3_override.packed_fa3_attention",
                ),
                trace_kernel_markers=("FlashAttnFwdSm90",),
                packed_offsets=True,
            ),
        ),
        ENGINES.arm("megatron_stock"),
    ),
)
"""The attention comparison: whether a fused varlen kernel closes the attention gap between the engines; it refuses ``--ac sac``."""


SCENARIOS = {"engines": ENGINES, "attention": ATTENTION}
"""Every end-to-end scenario, by name."""


def scenario_by_name(name: str) -> Scenario:
    """The scenario ``name``; an unknown name raises and names the choices."""
    try:
        return SCENARIOS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown scenario {name!r}. Available scenarios: {', '.join(SCENARIOS)}"
        ) from error
