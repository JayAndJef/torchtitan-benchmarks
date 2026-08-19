"""Which kernel-isolation scenarios exist, and what competes inside each.

The kernel analog of ``e2e/registry.py``: each scenario names the arms that
compete head-to-head on one kernel family, how each arm is built, which arm
it is compared against, and which correctness gates it must pass. The
declaration types live next door in ``benchmarks.kernel.schema``; this module
is nothing but instances of them. Torch-free like the schema, so the CLI can
list scenarios without CUDA.

Two kinds of scenario live here and they answer different questions. A
**single-engine** scenario ranks TorchTitan implementations of one kernel
against each other. A **cross-engine** scenario puts megatron-core's
implementation of one model component beside TorchTitan's, and its arms are
named ``engine/profile`` -- ``mcore/base`` against ``titan``. A cross-engine
ratio is a comparison of two *treatments* as much as two kernels, because
one side is compiled and the other is eager by design, so each arm states
its treatment and every eager arm states why.

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


QK_NORM = KernelScenario(
    name="qk_norm",
    description=(
        "Per-head QK RMSNorm on q and k before RoPE (titan "
        "GQAttention.q_norm/k_norm vs megatron self_attention."
        "q_layernorm/k_layernorm): torch.nn.RMSNorm under torch.compile "
        "against TransformerEngine RMSNorm run eager. TE norms run through "
        "the cuDNN backend on this host (NVTE_NORM_FWD_USE_CUDNN and "
        "NVTE_NORM_BWD_USE_CUDNN=1), so this is not megatron's native norm "
        "kernel. A norm may be at memory bandwidth, so copy_floor moves the "
        "same bytes and the x_floor column decides whether the ratio is a "
        "kernel claim at all; the floor declares forward only, so "
        "forward_backward carries no x_floor column. This number holds NO "
        "host serialization on either side (plan rule 5): neither path calls "
        ".cpu(), .item() or synchronize inside a timed closure. Measured on "
        "CUDA, Inductor writes one kernel per norm for the titan pair, so the "
        "row is two kernels against the mcore arm's two eager TE calls."
    ),
    inputs_builder="benchmarks.kernel.operations.qk_norm:qk_norm_inputs",
    reference_builder=(
        "benchmarks.kernel.operations.qk_norm:qk_norm_reference"
    ),
    baseline_arm="mcore/base",
    arms=(
        KernelArm(
            name="copy_floor",
            description=(
                "One read and one write of q and k: the bandwidth floor for "
                "this shape"
            ),
            builder=(
                "benchmarks.kernel.operations.qk_norm:"
                "build_qk_norm_copy_floor"
            ),
            modes=("forward",),
            is_floor=True,
            eager_reason=(
                "a bandwidth floor, not an implementation: compiling a pair "
                "of copies would measure Inductor rather than the bus"
            ),
        ),
        KernelArm(
            name="mcore/base",
            description=(
                "megatron self_attention.q_layernorm and k_layernorm: "
                "TransformerEngine RMSNorm (cuDNN norm backend on this host), "
                "eager as megatron runs it, on SBHD tensors"
            ),
            builder=(
                "benchmarks.kernel.operations.qk_norm:"
                "build_qk_norm_mcore_base"
            ),
            # No isolated backward. TE's operation fuser clears its saved
            # tensors while it runs backward (ops/fuser.py:225,258) and TE's
            # RMSNorm calls clear_tensor_data on both of them at the end of
            # op_backward, so the retained-graph re-run rope and qkv use
            # raises here. Both arms drop the mode and stay comparable;
            # backward cost is forward_backward minus forward.
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer, so every TE "
                "module it builds runs eager end to end; compiling this one "
                "would measure a treatment megatron never applies"
            ),
            correctness=(
                CorrectnessCheck(
                    kind="tolerance",
                    reference="fp64",
                    outputs=(
                        "q_out",
                        "k_out",
                        "dq",
                        "dk",
                        "q_weight_grad",
                        "k_weight_grad",
                    ),
                    # A norm is a reduction, so rel_l2 is the only safe metric
                    # (CLAUDE.md, "Choosing a correctness metric"). One value
                    # covers the weight gradients too: the reduction runs over
                    # 65,536 rows but the weight holds only head_dim = 64
                    # values, and torch accumulates that sum in fp32. Measured
                    # on CPU bf16 at the real normal shape: 1.66e-3 on the four
                    # activations, 1.36e-3 and 1.64e-3 on the two weight
                    # gradients.
                    max_rel_l2=2e-2,
                ),
            ),
        ),
        KernelArm(
            name="titan",
            description=(
                "TorchTitan GQAttention q_norm and k_norm: torch.nn.RMSNorm "
                "under torch.compile(fullgraph=True), as the per-block "
                "compile gives them end to end, on BSHD tensors"
            ),
            builder="benchmarks.kernel.operations.qk_norm:build_qk_norm_titan",
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(
                CorrectnessCheck(
                    kind="tolerance",
                    reference="fp64",
                    outputs=(
                        "q_out",
                        "k_out",
                        "dq",
                        "dk",
                        "q_weight_grad",
                        "k_weight_grad",
                    ),
                    max_rel_l2=2e-2,
                ),
                # Informational, and pointed this way round on purpose. The
                # two fp64 gates already enforce, and they are stronger: each
                # arm is right in absolute terms, which bounds the distance
                # between them. ``resolve_arm_skips`` closes the skip set over
                # correctness references, so a check pointing from the anchor
                # at ``titan`` would let a skipped titan arm take the anchor
                # down with it, and the anchor's loss costs the scenario.
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=(
                        "q_out",
                        "k_out",
                        "dq",
                        "dk",
                        "q_weight_grad",
                        "k_weight_grad",
                    ),
                    max_rel_l2=2e-2,
                    informational=True,
                ),
            ),
        ),
    ),
    # comparisons left at None: the derived set is exactly the one row this
    # scenario publishes, titan against mcore/base, with the floor excluded.
)


# One GEMM and two gradients, at the tolerance the qkv scenario already uses
# for the same class of operation.
ATTN_OUT_PROJ_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("out", "x_grad", "weight_grad"),
    max_rel_l2=2e-2,
)


ATTN_OUT_PROJ = KernelScenario(
    name="attn_out_proj",
    description=(
        "The attention output projection, cross-engine: megatron-core's "
        "TERowParallelLinear against TorchTitan's nn.Linear, over one shared "
        "weight. The titan arm is compiled (fullgraph=True) and the megatron "
        "arm is eager, which is how each engine runs it. The mcore arm also "
        "dispatches through TransformerEngine's own torch.autograd.Function "
        "with quantizer bookkeeping, where titan calls F.linear; at these "
        "shapes that host cost is a real part of the gap. The layout op "
        "before the projection is excluded on both sides, but the two "
        "excluded ops are NOT the same object: titan's is a materializing "
        "contiguous() copy of B*L*dim bf16 elements (8.4 MiB at the default "
        "workload, about a third of the GEMM's own device cost) because "
        "FlexAttention returns a transposed view, and megatron's is a free "
        "reshape. A cross-engine sum over the scenarios is short by that "
        "copy on the titan side until scenario 5 adopts it."
    ),
    inputs_builder=(
        "benchmarks.kernel.operations.attn_out_proj:attn_out_proj_inputs"
    ),
    reference_builder=(
        "benchmarks.kernel.operations.attn_out_proj:attn_out_proj_reference"
    ),
    baseline_arm="mcore/base",
    # Explicit, and it is the same pair the default derivation would produce.
    # Writing it down is what makes the direction of the published ratio a
    # declaration: this scenario reports titan against megatron, matching the
    # e2e piper1b_megatron scenario, where megatron is also the anchor.
    comparisons=(("titan", "mcore/base"),),
    arms=(
        KernelArm(
            name="mcore/base",
            description=(
                "Megatron-core self_attention.linear_proj: TE "
                "TERowParallelLinear, eager, tp_size 1 so no row-parallel "
                "reduce runs"
            ),
            builder=(
                "benchmarks.kernel.operations.attn_out_proj"
                ":build_attn_out_proj_mcore_base"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer, so every TE "
                "module it builds runs eager end to end; compiling this one "
                "would measure a treatment megatron never applies"
            ),
            correctness=(ATTN_OUT_PROJ_GATE,),
        ),
        KernelArm(
            name="titan",
            description=(
                "TorchTitan attention.wo: nn.Linear without a bias, under "
                "torch.compile(fullgraph=True)"
            ),
            builder=(
                "benchmarks.kernel.operations.attn_out_proj"
                ":build_attn_out_proj_titan"
            ),
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(
                ATTN_OUT_PROJ_GATE,
                # The cross-engine gate. Both arms already agree with fp64, so
                # this one is close to implied -- but it is the check that
                # states the scenario's claim directly: the two engines compute
                # the same function of the same weight, so a ratio between them
                # is a ratio of implementations and not of arithmetic.
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=("out", "x_grad", "weight_grad"),
                    max_rel_l2=2e-2,
                ),
            ),
        ),
    ),
)


# The gate both engines face, and the one the cross-engine row rests on.
# RMSNorm is a reduction over the last dimension, so max and ULP metrics report
# garbage wherever cancellation drives an output toward zero; rel_l2 is the only
# safe metric here (CLAUDE.md, "Choosing a correctness metric").
FFN_NORM_ACTIVATION_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("out", "x_grad"),
    max_rel_l2=2e-2,
)

# Separate, so a gain-gradient failure is legible on its own, but at the same
# tolerance. The gain gradient is not a different kind of number here: both
# engines accumulate it in fp32, so the row count does not widen the error.
# Measured at the default workload against the fp64 reference: out 1.661e-3,
# x_grad 1.663e-3, weight_grad 1.671e-3. All three sit at CLAUDE.md's ~2e-3 for
# a bf16 kernel, so all three take CLAUDE.md's 2e-2 gate.
FFN_NORM_GAIN_GRADIENT_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("weight_grad",),
    max_rel_l2=2e-2,
)


FFN_NORM = KernelScenario(
    name="ffn_norm",
    description=(
        "The norm in front of the MoE block: TorchTitan's ffn_norm against "
        "megatron-core's pre_mlp_layernorm. TE norms via the cuDNN backend -- "
        "NVTE_NORM_FWD_USE_CUDNN/NVTE_NORM_BWD_USE_CUDNN are set because TE's "
        "native RMSNorm kernels fail to launch on this box, so this is not "
        "TE's fastest norm and the number is not 'megatron's norm'. The titan "
        "arm is compiled and the mcore arm is eager, which is what each engine "
        "does end to end. copy_floor is the bandwidth reference: a norm at "
        "these shapes may be at memory bandwidth, and the x_floor column is "
        "what separates a slow kernel from a saturated bus. The mcore number "
        "also holds TE's per-call Python dispatch, which builds a fresh "
        "OperationFuser every call, so read the --burst residual before you "
        "rank the two kernels."
    ),
    inputs_builder="benchmarks.kernel.operations.ffn_norm:ffn_norm_inputs",
    reference_builder=(
        "benchmarks.kernel.operations.ffn_norm:ffn_norm_reference"
    ),
    baseline_arm="mcore/base",
    # Explicit, and exhaustive: this scenario publishes exactly one ratio. The
    # derived set would give the same pair today, but a cross-engine scenario
    # states which row it publishes rather than inheriting it.
    comparisons=(("titan", "mcore/base"),),
    arms=(
        KernelArm(
            name="copy_floor",
            description=(
                "One read of x and one write of y: the bandwidth floor for "
                "the forward traffic at this shape"
            ),
            builder=(
                "benchmarks.kernel.operations.ffn_norm:"
                "build_ffn_norm_copy_floor"
            ),
            modes=("forward",),
            is_floor=True,
            eager_reason=(
                "a bandwidth floor, not an implementation: compiling a copy "
                "would measure Inductor rather than the bus"
            ),
        ),
        KernelArm(
            name="mcore/base",
            description=(
                "megatron-core pre_mlp_layernorm off a real GPTModel: "
                "transformer_engine.pytorch.RMSNorm through the cuDNN norm "
                "backend, eager, as megatron runs it"
            ),
            builder=(
                "benchmarks.kernel.operations.ffn_norm:"
                "build_ffn_norm_mcore_base"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer, so every TE "
                "module it builds runs eager end to end; compiling this one "
                "would measure a treatment megatron never applies"
            ),
            correctness=(
                FFN_NORM_ACTIVATION_GATE,
                FFN_NORM_GAIN_GRADIENT_GATE,
            ),
        ),
        KernelArm(
            name="titan",
            description=(
                "TorchTitan ffn_norm: torch.nn.RMSNorm from the production "
                "_qwen3_norm config node, under torch.compile(fullgraph=True)"
            ),
            builder=(
                "benchmarks.kernel.operations.ffn_norm:build_ffn_norm_titan"
            ),
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(
                FFN_NORM_ACTIVATION_GATE,
                FFN_NORM_GAIN_GRADIENT_GATE,
                # The cross-engine gates. They are what make the ratio a
                # comparison of two implementations of one function: both arms
                # load the same gain and normalize with the same epsilon, so a
                # disagreement here means they no longer compute the same
                # thing. They sit on the non-anchor arm because
                # ``resolve_arm_skips`` closes the skip set over correctness
                # references, so a check pointing from the anchor at ``titan``
                # would let a skipped titan arm take the anchor down with it.
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=("out", "x_grad"),
                    max_rel_l2=2e-2,
                ),
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=("weight_grad",),
                    max_rel_l2=2e-2,
                ),
            ),
        ),
    ),
)


FINAL_NORM = KernelScenario(
    name="final_norm",
    description=(
        "The norm after the last transformer block (titan Decoder.norm vs "
        "megatron decoder.final_layernorm): torch.nn.RMSNorm against "
        "TransformerEngine RMSNorm. Both arms are EAGER, which is the "
        "production treatment on both engines -- apply_compile reaches only "
        "the children of model.layers and Decoder.norm is a sibling of them. "
        "TE norms run through the cuDNN backend on this host "
        "(NVTE_NORM_FWD_USE_CUDNN/NVTE_NORM_BWD_USE_CUDNN=1), so this is not "
        "megatron's native norm kernel. A norm may be at memory bandwidth, so "
        "copy_floor measures the same traffic and the x_floor column decides "
        "whether the ratio is a kernel claim at all. The floor declares "
        "forward only, so forward_backward carries no x_floor column."
    ),
    inputs_builder=(
        "benchmarks.kernel.operations.final_norm:final_norm_inputs"
    ),
    reference_builder=(
        "benchmarks.kernel.operations.final_norm:final_norm_reference"
    ),
    baseline_arm="mcore/base",
    arms=(
        KernelArm(
            name="copy_floor",
            description=(
                "One read of the hidden state and one write: the bandwidth "
                "floor for this shape"
            ),
            builder=(
                "benchmarks.kernel.operations.final_norm:"
                "build_final_norm_copy_floor"
            ),
            modes=("forward",),
            is_floor=True,
            eager_reason=(
                "a bandwidth floor, not an implementation: compiling a copy "
                "would measure Inductor rather than the bus"
            ),
        ),
        KernelArm(
            name="mcore/base",
            description=(
                "megatron decoder.final_layernorm: TransformerEngine RMSNorm "
                "(cuDNN norm backend on this host), eager as megatron runs it"
            ),
            builder=(
                "benchmarks.kernel.operations.final_norm:"
                "build_final_norm_mcore_base"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer, so every TE "
                "module it builds runs eager end to end; compiling this one "
                "would measure a treatment megatron never applies"
            ),
            correctness=(
                CorrectnessCheck(
                    kind="tolerance",
                    reference="fp64",
                    outputs=("out", "x_grad", "weight_grad"),
                    # A norm is a reduction, so rel_l2 is the only safe metric
                    # (CLAUDE.md, "Choosing a correctness metric"). Measured on
                    # CPU bf16 at the default workload: out 1.67e-3, x_grad
                    # 1.66e-3, weight_grad 1.70e-3. The gate holds ~12x.
                    max_rel_l2=2e-2,
                ),
            ),
        ),
        KernelArm(
            name="titan",
            description=(
                "TorchTitan Decoder.norm: torch.nn.RMSNorm, eager -- "
                "apply_compile reaches only the children of model.layers, and "
                "Decoder.norm is a sibling of them, so production runs this "
                "norm eager on both engines"
            ),
            builder=(
                "benchmarks.kernel.operations.final_norm:"
                "build_final_norm_titan"
            ),
            modes=("forward", "forward_backward"),
            # The one titan module arm in the registry that is not compiled,
            # and the reason is fidelity rather than convenience.
            # ``apply_compile`` walks ``model.layers.named_children()`` alone
            # (``distributed/compile.py:57-58``), and ``Decoder.__init__``
            # builds ``tok_embeddings``, ``norm`` and ``lm_head`` as siblings
            # of ``layers`` (``models/common/decoder.py:234,236,240-241``), so
            # this norm sits outside every compiled region in production.
            eager_reason=(
                "Decoder.norm sits outside every compiled region: "
                "apply_compile walks model.layers.named_children() alone, and "
                "the norm is a sibling of layers rather than a child of it, "
                "so compiling it here would measure a treatment production "
                "never applies to it"
            ),
            correctness=(
                CorrectnessCheck(
                    kind="tolerance",
                    reference="fp64",
                    outputs=("out", "x_grad", "weight_grad"),
                    max_rel_l2=2e-2,
                ),
                # The cross-engine agreement, recorded and not enforced. The
                # two fp64 gates above are the enforcement and they are
                # stronger: each arm is right in absolute terms, which bounds
                # the distance between them. An enforced arm-vs-arm gate could
                # only fail a run for a reason the fp64 gates already allow.
                # It sits on the non-anchor arm on purpose: resolve_arm_skips
                # closes the skip set over correctness references, so a check
                # pointing the other way would let a skipped titan arm take
                # the anchor with it.
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=("out", "x_grad", "weight_grad"),
                    max_rel_l2=2e-2,
                    informational=True,
                ),
            ),
        ),
    ),
    # comparisons left at None: the derived set is exactly the one row this
    # scenario publishes, titan against mcore/base, with the floor excluded.
)


KERNEL_SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        ROPE,
        SWIGLU,
        QKV,
        LM_HEAD,
        ATTENTION,
        QK_NORM,
        ATTN_OUT_PROJ,
        FFN_NORM,
        FINAL_NORM,
    )
}


def kernel_scenario_by_name(name: str) -> KernelScenario:
    try:
        return KERNEL_SCENARIOS[name]
    except KeyError:
        raise ValueError(
            f"Unknown kernel scenario {name!r}. "
            f"Available: {', '.join(KERNEL_SCENARIOS)}"
        ) from None
