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


MODES = ("forward", "backward", "forward_backward")


@dataclass(frozen=True)
class Piper1BSpec:
    """Model constants every kernel family derives its shapes from.

    The spec deliberately holds no per-family shapes; each scenario's inputs
    builder computes what it needs (swiglu rows = batch * seq_len * top_k,
    fused-qkv out features = (n_heads + 2 * n_kv_heads) * head_dim, lm_head
    tokens = batch * seq_len).
    """

    dim: int = 1024
    n_heads: int = 16
    n_kv_heads: int = 8
    head_dim: int = 64
    num_experts: int = 4
    top_k: int = 2
    moe_hidden_dim: int = 3584
    vocab_size: int = 151936
    batch: int = 4
    seq_len: int = 1024
    max_seq_len: int = 2048
    theta: float = 1_000_000.0

    def validate(self) -> None:
        if self.seq_len > self.max_seq_len:
            raise ValueError(
                f"seq_len ({self.seq_len}) exceeds max_seq_len "
                f"({self.max_seq_len}); RoPE tables are sized by max_seq_len"
            )
        if self.n_heads * self.head_dim != self.dim:
            raise ValueError(
                f"n_heads * head_dim ({self.n_heads * self.head_dim}) must "
                f"equal dim ({self.dim})"
            )
        if self.n_heads % self.n_kv_heads:
            raise ValueError(
                f"n_heads ({self.n_heads}) must be divisible by n_kv_heads "
                f"({self.n_kv_heads})"
            )
        if (self.batch * self.seq_len * self.top_k) % self.num_experts:
            raise ValueError(
                "routed rows must divide evenly among experts for the "
                "balanced-routing workload"
            )


def spec_with_overrides(
    *, batch: int | None = None, seq_len: int | None = None
) -> Piper1BSpec:
    spec = Piper1BSpec()
    if batch is not None:
        spec = replace(spec, batch=int(batch))
    if seq_len is not None:
        spec = replace(spec, seq_len=int(seq_len))
    spec.validate()
    return spec


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
    worker; the function receives (spec, inputs) and returns a BuiltArm
    whose per-mode closures are the timed operations. ``compare_to`` names
    the opponent arm for ratio and significance rows (None means the
    scenario baseline); comparisons only happen between arms sharing a mode.
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
    """A kernel family and its comparable implementation arms."""

    name: str
    description: str
    inputs_builder: str
    reference_builder: str | None
    arms: tuple[KernelArm, ...]
    baseline_arm: str

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
    description="Stock CosSinRoPE vs Helion and TE RoPE kernels, BSHD bf16.",
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
    description="Grouped-expert SwiGLU, at layer and Triton-kernel scope.",
    inputs_builder="benchmarks.kernel_arms:swiglu_inputs",
    reference_builder=None,
    baseline_arm="baseline",
    arms=(
        KernelArm(
            name="baseline",
            description="stock GroupedExperts layer: separate w1/w3 GEMMs, unfused activation",
            builder="benchmarks.kernel_arms:build_swiglu_baseline",
            modes=MODES,
            compiled=True,
        ),
        KernelArm(
            name="titan_fused",
            description="TorchTitan FusedGroupedExperts layer: fused w13, stride-2 activation views",
            builder="benchmarks.kernel_arms:build_swiglu_titan_fused",
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
            name="piper_optimized",
            description="Piper layer: titan_fused with a combined [R,2F] activation layout",
            builder="benchmarks.kernel_arms:build_swiglu_piper_optimized",
            modes=MODES,
            compiled=True,
            correctness=(
                CorrectnessCheck(
                    kind="tolerance",
                    reference="baseline",
                    outputs=("out", "x_grad", "w1_grad", "w2_grad", "w3_grad"),
                    max_rel_l2=2e-2,
                ),
                CorrectnessCheck(
                    kind="bitwise",
                    reference="titan_fused",
                    outputs=("out", "x_grad"),
                ),
            ),
        ),
        KernelArm(
            name="titan_triton",
            description="TorchTitan SwiGLU Triton kernel alone (stride-2 gate/up views)",
            builder="benchmarks.kernel_arms:build_swiglu_titan_triton",
            modes=("forward", "backward"),
        ),
        KernelArm(
            name="piper_optimized_triton",
            description="Piper SwiGLU Triton kernel alone (combined [R,2F] layout)",
            builder="benchmarks.kernel_arms:build_swiglu_piper_optimized_triton",
            modes=("forward", "backward"),
            compare_to="titan_triton",
            correctness=(
                CorrectnessCheck(
                    kind="bitwise",
                    reference="titan_triton",
                    outputs=("fwd_out", "grad_gate_up"),
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
            description="F.linear_cross_entropy via FusedLinearCrossEntropyLoss",
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
            description="Full logits then Piper-optimized TE-derived CE",
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


KERNEL_SCENARIOS = {
    scenario.name: scenario for scenario in (ROPE, SWIGLU, QKV, LM_HEAD)
}


def kernel_scenario_by_name(name: str) -> KernelScenario:
    try:
        return KERNEL_SCENARIOS[name]
    except KeyError:
        raise ValueError(
            f"Unknown kernel scenario {name!r}. "
            f"Available: {', '.join(KERNEL_SCENARIOS)}"
        ) from None


def shape_summary(scenario_name: str, spec: Piper1BSpec) -> dict[str, object]:
    """Derived shapes recorded in the manifest for provenance."""
    batch, seq = spec.batch, spec.seq_len
    if scenario_name == "rope":
        return {
            "q": [batch, seq, spec.n_heads, spec.head_dim],
            "k": [batch, seq, spec.n_kv_heads, spec.head_dim],
            "positions": [batch, seq],
            "table_rows": spec.max_seq_len,
        }
    if scenario_name == "swiglu":
        rows = batch * seq * spec.top_k
        per_expert = rows // spec.num_experts
        return {
            "x": [rows, spec.dim],
            "gate_up": [rows, 2 * spec.moe_hidden_dim],
            "tokens_per_expert": [per_expert] * spec.num_experts,
        }
    if scenario_name == "qkv":
        kv_out = spec.n_kv_heads * spec.head_dim
        fused_out = (spec.n_heads + 2 * spec.n_kv_heads) * spec.head_dim
        return {
            "x": [batch, seq, spec.dim],
            "wq": [spec.n_heads * spec.head_dim, spec.dim],
            "wk": [kv_out, spec.dim],
            "wv": [kv_out, spec.dim],
            "wqkv": [fused_out, spec.dim],
        }
    if scenario_name == "lm_head":
        return {
            "hidden": [batch, seq, spec.dim],
            "weight": [spec.vocab_size, spec.dim],
            "tokens": batch * seq,
        }
    raise ValueError(f"Unknown kernel scenario {scenario_name!r}")
