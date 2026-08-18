"""Declarative kernel-isolation benchmark scenarios.

The kernel analog of ``scenarios.py``: each scenario names the arms that
compete head-to-head on one kernel family, how each arm is built, which arm
it is compared against, and which correctness gates it must pass. This
module is torch-free so the CLI can list scenarios without CUDA; arm
builders are referenced as dotted paths and resolved inside the GPU worker.

Never present these numbers as end-to-end results: they time kernels in
isolation on synthetic inputs at Piper-1B shapes.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from piper1b.model_shape import PiperShape, shape_by_name


MODES = ("forward", "backward", "forward_backward")


@dataclass(frozen=True)
class KernelWorkload:
    """What is run through the model, as opposed to what the model is.

    Kept separate from ``PiperShape`` so batch size is independent of
    geometry: the same shape is measurable at any batch, and a batch change
    is not a model change. Everything geometric -- dim, head counts, expert
    width, vocabulary, ``max_seq_len`` -- belongs to the shape, and no
    per-family shapes live in either: each scenario's inputs builder computes
    what it needs (swiglu rows = batch * seq_len * top_k, fused-qkv out
    features = (n_heads + 2 * n_kv_heads) * head_dim, lm_head tokens =
    batch * seq_len).
    """

    batch: int = 4
    seq_len: int = 1024


def validate_shape_and_workload(
    shape: PiperShape, workload: KernelWorkload
) -> None:
    """Raise ValueError on a shape/workload pair no scenario can run.

    Only constraints that genuinely cross the two live here. The purely
    geometric ones are structurally unrepresentable: ``PiperShape`` derives
    both head counts from ``dim`` and ``head_dim``, so
    ``n_heads * head_dim == dim`` and ``n_heads % n_kv_heads == 0`` cannot be
    violated and there is nothing to assert.
    """
    if workload.seq_len > shape.max_seq_len:
        raise ValueError(
            f"seq_len ({workload.seq_len}) exceeds max_seq_len "
            f"({shape.max_seq_len}); RoPE tables are sized by max_seq_len"
        )


def routing_divides_evenly(
    shape: PiperShape, workload: KernelWorkload
) -> bool:
    """Whether swiglu's synthetic workload routes rows evenly across experts.

    Scenario-scoped on purpose: only the swiglu inputs builder hands every
    expert an equal slice of ``batch * seq_len * top_k`` rows, so an uneven
    split is that scenario's problem and must not fail rope, qkv, lm_head or
    attention.
    """
    return (
        workload.batch * workload.seq_len * shape.top_k
    ) % shape.num_experts == 0


def resolve_shape_and_workload(
    *,
    model_size: str = "normal",
    batch: int | None = None,
    seq_len: int | None = None,
    max_seq_len: int | None = None,
) -> tuple[PiperShape, KernelWorkload]:
    """The geometry and the workload one kernel-bench run measures."""
    shape = shape_by_name(model_size)
    # max_seq_len first: validation rejects seq_len > max_seq_len, and the
    # attention sweep raises the ceiling precisely in order to run past it.
    if max_seq_len is not None:
        shape = replace(shape, max_seq_len=int(max_seq_len))
    workload = KernelWorkload()
    if batch is not None:
        workload = replace(workload, batch=int(batch))
    if seq_len is not None:
        workload = replace(workload, seq_len=int(seq_len))
    validate_shape_and_workload(shape, workload)
    return shape, workload


@dataclass(frozen=True)
class CorrectnessCheck:
    """One validity gate: compare named outputs against a reference.

    ``reference`` is another arm's name, or ``"fp64"`` for the scenario's
    ground-truth builder. ``kind`` selects the comparison: ``bitwise``
    (torch.equal), ``tolerance`` (max_abs / max_rel / max_rel_l2 bounds), or
    ``fp64_ulp`` (bf16 ULPs against the fp64 truth). Informational checks are
    recorded in results but never gate the run.

    Choosing a metric:

    - ``max_rel_l2`` (||a - b|| / ||b||) is the default. It is the only one
      that stays meaningful when the compared tensors differ in magnitude
      (weight gradients accumulate over thousands of rows and sit ~30x above
      activations, so one bf16 ULP there is a large absolute number) or when
      individual elements land near zero.
    - ``fp64_ulp`` reports the *mean* bf16 ULP error and suits elementwise
      kernels (RoPE), where it is the accuracy number this repo has always
      quoted (~0.24 ULP). It is a mean because the per-element maximum is
      meaningless wherever cancellation can drive a true value toward zero:
      dividing a negligible absolute error by that tiny magnitude reports
      thousands of ULPs for a numerically perfect kernel, and does so
      identically for the stock implementation.
    - ``bitwise`` where two implementations must agree exactly.
    """

    kind: str
    reference: str
    outputs: tuple[str, ...]
    max_mean_ulp: float | None = None
    max_abs: float | None = None
    max_rel: float | None = None
    max_rel_l2: float | None = None
    informational: bool = False


@dataclass(frozen=True)
class KernelArm:
    """One implementation in a head-to-head kernel comparison.

    ``builder`` is a dotted path ("module:function") resolved in the GPU
    worker; the function receives (shape, workload, inputs) and returns a
    BuiltArm whose per-mode closures are the timed operations.
    ``compare_to`` names the opponent arm for ratio and significance rows
    (None means the scenario baseline); comparisons only happen between arms
    sharing a mode.
    ``compiled`` records that the builder runs the arm under torch.compile,
    as production does; raw-kernel arms and floors stay eager on purpose.
    """

    name: str
    description: str
    builder: str
    modes: tuple[str, ...]
    compare_to: str | None = None
    correctness: tuple[CorrectnessCheck, ...] = ()
    requires_gcc_toolset: bool = False
    is_floor: bool = False
    compiled: bool = False


@dataclass(frozen=True)
class KernelScenario:
    """A kernel family and its comparable implementation arms.

    ``requires_balanced_routing`` marks a scenario whose inputs builder hands
    every expert an equal slice of the routed rows; the runner refuses to run
    it on a workload where they do not divide evenly, rather than silently
    rounding the split.
    """

    name: str
    description: str
    inputs_builder: str
    reference_builder: str | None
    arms: tuple[KernelArm, ...]
    baseline_arm: str
    requires_balanced_routing: bool = False

    def arm(self, name: str) -> KernelArm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise ValueError(
            f"Unknown arm {name!r} for kernel scenario {self.name!r}. "
            f"Available arms: {', '.join(arm.name for arm in self.arms)}"
        )

    @property
    def requires_gcc_toolset(self) -> bool:
        return any(arm.requires_gcc_toolset for arm in self.arms)


ROPE = KernelScenario(
    name="rope",
    description="TorchTitan CosSinRoPE vs Helion and TE RoPE kernels, BSHD bf16.",
    inputs_builder="benchmarks.kernel_arms:rope_inputs",
    reference_builder="benchmarks.kernel_arms:rope_reference",
    baseline_arm="baseline",
    arms=(
        KernelArm(
            name="copy_floor",
            description="q/k copy_ pair: the bandwidth floor for this shape",
            builder="benchmarks.kernel_arms:build_rope_copy_floor",
            modes=("forward",),
            is_floor=True,
        ),
        KernelArm(
            name="baseline",
            description="TorchTitan CosSinRoPE (backward via autograd)",
            builder="benchmarks.kernel_arms:build_rope_baseline",
            modes=("forward", "backward"),
            compiled=True,
            correctness=(
                CorrectnessCheck(
                    kind="fp64_ulp",
                    reference="fp64",
                    outputs=("q_out", "k_out", "dq", "dk"),
                    max_mean_ulp=1.0,
                ),
                CorrectnessCheck(
                    kind="tolerance",
                    reference="fp64",
                    outputs=("q_out", "k_out", "dq", "dk"),
                    max_rel_l2=2e-2,
                ),
            ),
        ),
        KernelArm(
            name="helion",
            description="TorchTitan HelionCosSinRoPE kernel",
            builder="benchmarks.kernel_arms:build_rope_helion",
            modes=("forward", "backward"),
            compiled=True,
            correctness=(
                CorrectnessCheck(
                    kind="fp64_ulp",
                    reference="fp64",
                    outputs=("q_out", "k_out", "dq", "dk"),
                    max_mean_ulp=1.0,
                ),
                CorrectnessCheck(
                    kind="tolerance",
                    reference="fp64",
                    outputs=("q_out", "k_out", "dq", "dk"),
                    max_rel_l2=2e-2,
                ),
            ),
        ),
        KernelArm(
            name="te",
            description="TransformerEngine RoPE CUDA kernel (positions path)",
            builder="benchmarks.kernel_arms:build_rope_te",
            modes=("forward", "backward"),
            requires_gcc_toolset=True,
            compiled=True,
            correctness=(
                CorrectnessCheck(
                    kind="fp64_ulp",
                    reference="fp64",
                    outputs=("q_out", "k_out", "dq", "dk"),
                    max_mean_ulp=1.0,
                ),
                CorrectnessCheck(
                    kind="tolerance",
                    reference="fp64",
                    outputs=("q_out", "k_out", "dq", "dk"),
                    max_rel_l2=2e-2,
                ),
                CorrectnessCheck(
                    kind="tolerance",
                    reference="helion",
                    outputs=("q_out", "k_out"),
                    max_rel_l2=2e-2,
                    informational=True,
                ),
            ),
        ),
    ),
)


SWIGLU = KernelScenario(
    name="swiglu",
    description="Grouped-expert SwiGLU layer: TorchTitan vs the two Piper variants.",
    inputs_builder="benchmarks.kernel_arms:swiglu_inputs",
    reference_builder=None,
    baseline_arm="baseline",
    requires_balanced_routing=True,
    arms=(
        KernelArm(
            name="baseline",
            description="TorchTitan modern GroupedExperts: separate w1/w3 GEMMs, plain-ops activation",
            builder="benchmarks.kernel_arms:build_swiglu_baseline",
            modes=MODES,
            compiled=True,
        ),
        KernelArm(
            name="piper_optimized_triton",
            description="Piper layer: fused w13 GEMM + combined [R,2F] custom Triton activation op",
            builder="benchmarks.kernel_arms:build_swiglu_piper_optimized_triton",
            modes=MODES,
            compiled=True,
            correctness=(
                CorrectnessCheck(
                    kind="tolerance",
                    reference="baseline",
                    outputs=("out", "x_grad", "w1_grad", "w2_grad", "w3_grad"),
                    max_rel_l2=2e-2,
                ),
            ),
        ),
        KernelArm(
            name="piper_optimized_inductor",
            description="Piper layer: fused w13 GEMM, plain-ops SwiGLU left to Inductor",
            builder="benchmarks.kernel_arms:build_swiglu_piper_optimized_inductor",
            modes=MODES,
            compiled=True,
            correctness=(
                CorrectnessCheck(
                    kind="tolerance",
                    reference="baseline",
                    outputs=("out", "x_grad", "w1_grad", "w2_grad", "w3_grad"),
                    max_rel_l2=2e-2,
                ),
            ),
        ),
    ),
)


QKV = KernelScenario(
    name="qkv",
    description="QKV projection: separate Q/KV GEMMs vs one fused GEMM.",
    inputs_builder="benchmarks.kernel_arms:qkv_inputs",
    reference_builder="benchmarks.kernel_arms:qkv_reference",
    baseline_arm="baseline",
    arms=(
        KernelArm(
            name="baseline",
            description="TorchTitan QKVLinear: separate Q and KV GEMMs",
            builder="benchmarks.kernel_arms:build_qkv_baseline",
            modes=MODES,
            compiled=True,
            correctness=(
                CorrectnessCheck(
                    kind="tolerance",
                    reference="fp64",
                    outputs=("q_out", "k_out", "v_out"),
                    max_rel_l2=2e-2,
                ),
            ),
        ),
        KernelArm(
            name="fused_qkv",
            description="TorchTitan FusedQKVLinear: one wqkv GEMM plus split",
            builder="benchmarks.kernel_arms:build_qkv_fused_qkv",
            modes=MODES,
            compiled=True,
            correctness=(
                CorrectnessCheck(
                    kind="tolerance",
                    reference="fp64",
                    outputs=("q_out", "k_out", "v_out"),
                    max_rel_l2=2e-2,
                ),
                CorrectnessCheck(
                    kind="tolerance",
                    reference="baseline",
                    outputs=("q_out", "k_out", "v_out", "x_grad"),
                    max_rel_l2=2e-2,
                ),
                CorrectnessCheck(
                    kind="bitwise",
                    reference="baseline",
                    outputs=("q_out", "k_out", "v_out"),
                    informational=True,
                ),
            ),
        ),
    ),
)


LM_HEAD = KernelScenario(
    name="lm_head",
    description="LM head + loss: full logits vs fused and TE-derived CE.",
    inputs_builder="benchmarks.kernel_arms:lm_head_inputs",
    reference_builder=None,
    baseline_arm="baseline",
    arms=(
        KernelArm(
            name="baseline",
            description="F.linear then TorchTitan CrossEntropyLoss (compiled)",
            builder="benchmarks.kernel_arms:build_lm_head_baseline",
            modes=("forward_backward",),
            compiled=True,
        ),
        KernelArm(
            name="fused_linear_ce",
            description="torch.nn.functional.linear_cross_entropy: CE without materializing full logits",
            builder="benchmarks.kernel_arms:build_lm_head_fused_linear_ce",
            modes=("forward_backward",),
            compiled=True,
            correctness=(
                CorrectnessCheck(
                    kind="tolerance",
                    reference="baseline",
                    outputs=("loss",),
                    max_rel=2e-3,
                ),
                CorrectnessCheck(
                    kind="tolerance",
                    reference="baseline",
                    outputs=("hidden_grad", "weight_grad"),
                    max_rel_l2=2e-2,
                ),
            ),
        ),
        KernelArm(
            name="te_fused_ce",
            description="Full logits then vendored TE Triton cross entropy",
            builder="benchmarks.kernel_arms:build_lm_head_te_fused_ce",
            modes=("forward_backward",),
            compiled=True,
            correctness=(
                CorrectnessCheck(
                    kind="tolerance",
                    reference="baseline",
                    outputs=("loss",),
                    max_rel=2e-3,
                ),
                CorrectnessCheck(
                    kind="tolerance",
                    reference="baseline",
                    outputs=("hidden_grad", "weight_grad"),
                    max_rel_l2=2e-2,
                ),
            ),
        ),
        KernelArm(
            name="piper_optimized_te_ce",
            description="TE CE reworked into one Triton kernel writing the pre-scaled bf16 grad in forward (TE: 2 fwd kernels + a bwd scaling pass)",
            builder="benchmarks.kernel_arms:build_lm_head_piper_optimized_te_ce",
            modes=("forward_backward",),
            compiled=True,
            correctness=(
                CorrectnessCheck(
                    kind="tolerance",
                    reference="baseline",
                    outputs=("loss",),
                    max_rel=2e-3,
                ),
                CorrectnessCheck(
                    kind="tolerance",
                    reference="baseline",
                    outputs=("hidden_grad", "weight_grad"),
                    max_rel_l2=5e-2,
                ),
            ),
        ),
    ),
)


ATTENTION_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("out", "dq", "dk", "dv"),
    # Attention is a reduction, so max/ULP metrics report garbage wherever
    # cancellation drives an output toward zero; rel_l2 is the only safe gate.
    # Measured headroom: both implementations land at ~2e-3 against fp64.
    max_rel_l2=2e-2,
)


ATTENTION = KernelScenario(
    name="attention",
    description=(
        "Inner attention at Piper-1B shapes with packed-document causal "
        "masking: FlexAttention vs FlashAttention-3 varlen vs FlexAttention "
        "lowered to FlashAttention-4."
    ),
    inputs_builder="benchmarks.kernel_arms:attention_inputs",
    reference_builder="benchmarks.kernel_arms:attention_reference",
    baseline_arm="baseline",
    arms=(
        KernelArm(
            name="baseline",
            description=(
                "TorchTitan FlexAttention: an Inductor-generated Triton "
                "template driven by a block-diagonal causal BlockMask"
            ),
            builder="benchmarks.kernel_arms:build_attention_baseline",
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(ATTENTION_GATE,),
        ),
        KernelArm(
            name="flex_flash",
            description=(
                "The same FlexAttention module and mask lowered to "
                "FlashAttention-4 CuTe DSL kernels instead of a Triton "
                "template (BACKEND=FLASH, 256x128 blocks); requires the fa4 "
                "dependency group"
            ),
            builder="benchmarks.kernel_arms:build_attention_flex_flash",
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(ATTENTION_GATE,),
        ),
        KernelArm(
            name="flash_attention_3",
            description=(
                "FlashAttention-3 varlen (CUTLASS sm90) over the same packed "
                "documents, via torch.nn.attention.varlen; requires the "
                "flash3 dependency group"
            ),
            builder="benchmarks.kernel_arms:build_attention_flash3",
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(
                ATTENTION_GATE,
                CorrectnessCheck(
                    kind="tolerance",
                    reference="baseline",
                    outputs=("out", "dq", "dk", "dv"),
                    max_rel_l2=2e-2,
                    informational=True,
                ),
            ),
        ),
    ),
)


KERNEL_SCENARIOS = {
    scenario.name: scenario
    for scenario in (ROPE, SWIGLU, QKV, LM_HEAD, ATTENTION)
}


def kernel_scenario_by_name(name: str) -> KernelScenario:
    try:
        return KERNEL_SCENARIOS[name]
    except KeyError:
        raise ValueError(
            f"Unknown kernel scenario {name!r}. "
            f"Available: {', '.join(KERNEL_SCENARIOS)}"
        ) from None


def shape_summary(
    scenario_name: str, shape: PiperShape, workload: KernelWorkload
) -> dict[str, object]:
    """Derived shapes recorded in the manifest for provenance."""
    batch, seq = workload.batch, workload.seq_len
    if scenario_name == "rope":
        return {
            "q": [batch, seq, shape.n_heads, shape.head_dim],
            "k": [batch, seq, shape.n_kv_heads, shape.head_dim],
            "positions": [batch, seq],
            "table_rows": shape.max_seq_len,
        }
    if scenario_name == "swiglu":
        rows = batch * seq * shape.top_k
        per_expert = rows // shape.num_experts
        return {
            "x": [rows, shape.dim],
            "tokens_per_expert": [per_expert] * shape.num_experts,
        }
    if scenario_name == "qkv":
        kv_out = shape.n_kv_heads * shape.head_dim
        return {
            "x": [batch, seq, shape.dim],
            "wq": [shape.n_heads * shape.head_dim, shape.dim],
            "wk": [kv_out, shape.dim],
            "wv": [kv_out, shape.dim],
            "wqkv": [shape.qkv_out_features, shape.dim],
        }
    if scenario_name == "lm_head":
        return {
            "hidden": [batch, seq, shape.dim],
            "weight": [shape.vocab_size, shape.dim],
            "tokens": batch * seq,
        }
    if scenario_name == "attention":
        return {
            "q": [batch, seq, shape.n_heads, shape.head_dim],
            "k": [batch, seq, shape.n_kv_heads, shape.head_dim],
            "v": [batch, seq, shape.n_kv_heads, shape.head_dim],
            "positions": [batch, seq],
            "packed_tokens": batch * seq,
            "max_seq_len": shape.max_seq_len,
        }
    raise ValueError(f"Unknown kernel scenario {scenario_name!r}")
