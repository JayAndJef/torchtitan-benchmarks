"""Which kernel-isolation scenarios exist, and what competes inside each.

The kernel analog of ``e2e/registry.py``: each scenario names the arms that
compete head-to-head on one kernel family, how each arm is built, which arm
it is compared against, and which correctness gates it must pass. The
declaration types live next door in ``benchmarks.kernel.schema``; this module
is nothing but the five instances of them. Torch-free like the schema, so the
CLI can list scenarios without CUDA.

Builders are dotted ``module:function`` strings, one module per kernel
family (``benchmarks.kernel.operations.rope`` for the rope scenario, and so
on), resolved by ``resolve_symbol`` inside the GPU worker and never imported
here. That indirection is the reason this file can name every arm builder in
the repository while importing none of torch, torchtitan or
TransformerEngine.

**Do not move a scenario constant next to its family's builders.** Colocating
``ROPE`` with ``operations/rope.py`` looks like tidiness and is the one
change that breaks the arrangement: ``benchmarks.kernel.engine`` imports the
schema, and if it ever imported this module instead, ``engine -> registry ->
operations.<family> -> torchtitan`` would drag every operation module and its
model dependencies into the engine's import graph. Scenario declarations stay
here, on the parent side; only the strings point at the families.

Never present these numbers as end-to-end results: they time kernels in
isolation on synthetic inputs at Piper-1B shapes.
"""

from __future__ import annotations

from benchmarks.kernel.schema import (
    CorrectnessCheck,
    KernelArm,
    KernelScenario,
    MODES,
)


ROPE = KernelScenario(
    name="rope",
    description="TorchTitan CosSinRoPE vs Helion and TE RoPE kernels, BSHD bf16.",
    inputs_builder="benchmarks.kernel.operations.rope:rope_inputs",
    reference_builder="benchmarks.kernel.operations.rope:rope_reference",
    baseline_arm="baseline",
    arms=(
        KernelArm(
            name="copy_floor",
            description="q/k copy_ pair: the bandwidth floor for this shape",
            builder="benchmarks.kernel.operations.rope:build_rope_copy_floor",
            modes=("forward",),
            is_floor=True,
            eager_reason=(
                "a bandwidth floor, not an implementation: it measures what "
                "the memory traffic alone costs, and compiling a pair of "
                "copies would measure Inductor instead"
            ),
        ),
        KernelArm(
            name="baseline",
            description="TorchTitan CosSinRoPE (backward via autograd)",
            builder="benchmarks.kernel.operations.rope:build_rope_baseline",
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
            builder="benchmarks.kernel.operations.rope:build_rope_helion",
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
            builder="benchmarks.kernel.operations.rope:build_rope_te",
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
    inputs_builder="benchmarks.kernel.operations.swiglu:swiglu_inputs",
    reference_builder=None,
    baseline_arm="baseline",
    requires_balanced_routing=True,
    arms=(
        KernelArm(
            name="baseline",
            description="TorchTitan modern GroupedExperts: separate w1/w3 GEMMs, plain-ops activation",
            builder="benchmarks.kernel.operations.swiglu:build_swiglu_baseline",
            modes=MODES,
            compiled=True,
        ),
        KernelArm(
            name="piper_optimized_triton",
            description="Piper layer: fused w13 GEMM + combined [R,2F] custom Triton activation op",
            builder="benchmarks.kernel.operations.swiglu:build_swiglu_piper_optimized_triton",
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
            builder="benchmarks.kernel.operations.swiglu:build_swiglu_piper_optimized_inductor",
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
    inputs_builder="benchmarks.kernel.operations.qkv:qkv_inputs",
    reference_builder="benchmarks.kernel.operations.qkv:qkv_reference",
    baseline_arm="baseline",
    arms=(
        KernelArm(
            name="baseline",
            description="TorchTitan QKVLinear: separate Q and KV GEMMs",
            builder="benchmarks.kernel.operations.qkv:build_qkv_baseline",
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
            builder="benchmarks.kernel.operations.qkv:build_qkv_fused_qkv",
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
    inputs_builder="benchmarks.kernel.operations.lm_head:lm_head_inputs",
    reference_builder=None,
    baseline_arm="baseline",
    arms=(
        KernelArm(
            name="baseline",
            description="F.linear then TorchTitan CrossEntropyLoss (compiled)",
            builder="benchmarks.kernel.operations.lm_head:build_lm_head_baseline",
            modes=("forward_backward",),
            compiled=True,
        ),
        KernelArm(
            name="fused_linear_ce",
            description="torch.nn.functional.linear_cross_entropy: CE without materializing full logits",
            builder="benchmarks.kernel.operations.lm_head:build_lm_head_fused_linear_ce",
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
            builder="benchmarks.kernel.operations.lm_head:build_lm_head_te_fused_ce",
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
            builder="benchmarks.kernel.operations.lm_head:build_lm_head_piper_optimized_te_ce",
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
    inputs_builder="benchmarks.kernel.operations.attention:attention_inputs",
    reference_builder="benchmarks.kernel.operations.attention:attention_reference",
    baseline_arm="baseline",
    arms=(
        KernelArm(
            name="baseline",
            description=(
                "TorchTitan FlexAttention: an Inductor-generated Triton "
                "template driven by a block-diagonal causal BlockMask"
            ),
            builder="benchmarks.kernel.operations.attention:build_attention_baseline",
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
            builder="benchmarks.kernel.operations.attention:build_attention_flex_flash",
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
            builder="benchmarks.kernel.operations.attention:build_attention_flash3",
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
