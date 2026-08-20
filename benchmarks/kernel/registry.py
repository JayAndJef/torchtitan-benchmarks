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


# The gate every arm carries against the fp64 truth. RoPE is elementwise, so a
# max-based metric would be legitimate here -- but rel_l2 is the repository
# default and it is the only metric all five arms can share, because the two
# engines do not compute the rotation at the same width. Titan's CosSinRoPE
# upcasts q and k to fp32 and casts back
# (torchtitan/models/common/rope.py:335-343); megatron's UNFUSED path casts
# cos/sin down to the input dtype and multiplies in bf16
# (rope_utils.py:132-136). Both fused paths compute in float inside the
# kernel. 2e-2 is the value every other cross-engine scenario uses.
ROPE_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("q_out", "k_out", "dq", "dk"),
    max_rel_l2=2e-2,
)

# The historical accuracy gate of the three titan arms, carried across the
# re-homing unchanged. RoPE is elementwise, so the mean-ULP metric is valid
# here (CLAUDE.md forbids it on reductions, not on this), and the arms have
# reported ~0.24 mean bf16 ULP against a 1.0 bound for as long as the
# scenario has existed. Dropping it while re-homing the arms would weaken an
# existing gate for no reason.
#
# It is deliberately NOT declared on the two mcore arms. mcore/no_rope_fusion
# multiplies by a bf16 cos/sin, so its mean ULP is not the titan arms' number,
# and no measured value exists to bound it. A gate whose threshold was
# guessed is worse than the rel_l2 gate that already enforces.
ROPE_ULP_GATE = CorrectnessCheck(
    kind="fp64_ulp",
    reference="fp64",
    outputs=("q_out", "k_out", "dq", "dk"),
    max_mean_ulp=1.0,
)

# Every cross-engine check is declared ON the titan arms and REFERENCES the
# anchor, never the reverse. ``resolve_arm_skips`` closes the skip set over
# correctness references (runner.py:268-306), so a check pointing outward from
# the anchor would let a skipped titan arm take the anchor down with it -- and
# losing the anchor writes no results at all (runner.py:497-511), because
# every row is a ratio against it. Informational, because the two fp64 gates
# already enforce and are stronger: each arm is right in absolute terms, which
# bounds the distance between them.
#
# One frozen instance shared by three arms, as registry.py's sibling gates
# already are (QKV_PREP_GATE, ATTN_OUT_PROJ_GATE). CorrectnessCheck is a
# frozen dataclass, so sharing is safe and a factory function would be the odd
# one out in that file.
ROPE_CROSS_ENGINE_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="mcore/base",
    outputs=("q_out", "k_out", "dq", "dk"),
    max_rel_l2=2e-2,
    informational=True,
)


ROPE = KernelScenario(
    name="rope",
    description=(
        "Rotary position embedding on q and k (titan GQAttention.rope vs "
        "megatron apply_rotary_pos_emb): TorchTitan's CosSinRoPE, Helion and "
        "local TE-port kernels under torch.compile against megatron's THD "
        "path run eager. THE TWO ENGINES SHARE THEIR INNER ARITHMETIC AND "
        "NOTHING ABOVE IT: components/rope/te_rope_standalone.cu copies TE's "
        "fused_rope block functions but adds its own __global__ and its own "
        "BSHD launch configuration, while megatron at THD uses TE's THD "
        "launcher, whose grid is a function of the document count. So the "
        "cross-engine row is two implementations of one rotation, and it is "
        "neither a kernel-quality result nor a pure host-wrapper "
        "comparison. This number HOLDS host serialization on one arm (plan "
        "rule 5): mcore/no_rope_fusion runs _apply_rotary_pos_emb_thd, which "
        "is two device-to-host syncs and a Python loop over the packed "
        "documents, so its number scales with the document count and is not "
        "'unfused TE'. Megatron's per-step rotary_pos_emb build is hoisted to "
        "build time and charged to neither arm; RotaryEmbedding.forward is "
        "lru_cached, so what is hoisted is a cache lookup after the first "
        "step."
    ),
    inputs_builder="benchmarks.kernel.operations.rope:rope_inputs",
    reference_builder="benchmarks.kernel.operations.rope:rope_reference",
    baseline_arm="mcore/base",
    arms=(
        KernelArm(
            name="mcore/base",
            description=(
                "megatron apply_rotary_pos_emb with apply_rope_fusion=True: "
                "TE's fused_apply_rotary_pos_emb_thd on THD tensors, eager "
                "as megatron runs it"
            ),
            builder=(
                "benchmarks.kernel.operations.rope:build_rope_mcore_base"
            ),
            modes=("forward", "backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer, and "
                "rope_utils.py carries no jit_fuser decoration either, so "
                "this call is eager end to end; compiling it would measure a "
                "treatment megatron never applies"
            ),
            correctness=(ROPE_GATE,),
        ),
        KernelArm(
            name="mcore/no_rope_fusion",
            description=(
                "megatron with apply_rope_fusion=False: at THD that is "
                "_apply_rotary_pos_emb_thd, a .tolist() sync plus a Python "
                "loop over the packed documents -- NOT an unfused TE kernel, "
                "and the number scales with the document count. Refuses to "
                "build at batch 1, where megatron takes its other branch and "
                "rotates by global offsets instead"
            ),
            builder=(
                "benchmarks.kernel.operations.rope:"
                "build_rope_mcore_no_rope_fusion"
            ),
            modes=("forward", "backward"),
            eager_reason=(
                "the same reason as mcore/base, and more strongly: this path "
                "is a Python loop whose host cost is the thing under "
                "measurement, so compiling it would erase the effect"
            ),
            correctness=(ROPE_GATE,),
        ),
        KernelArm(
            name="titan",
            description=(
                "TorchTitan CosSinRoPE on BSHD tensors, under "
                "torch.compile(fullgraph=True) as the per-block compile "
                "gives it end to end"
            ),
            builder="benchmarks.kernel.operations.rope:build_rope_titan",
            modes=("forward", "backward"),
            compiled=True,
            correctness=(ROPE_GATE, ROPE_ULP_GATE, ROPE_CROSS_ENGINE_GATE),
        ),
        KernelArm(
            name="titan/helion",
            description=(
                "TorchTitan HelionCosSinRoPE: the cache gather and the "
                "rotation fused into one Helion kernel, marker-guarded "
                "because it degrades to the stock path rather than failing"
            ),
            builder=(
                "benchmarks.kernel.operations.rope:build_rope_titan_helion"
            ),
            modes=("forward", "backward"),
            compiled=True,
            correctness=(ROPE_GATE, ROPE_ULP_GATE, ROPE_CROSS_ENGINE_GATE),
        ),
        KernelArm(
            name="titan/te",
            description=(
                "our local CUDA port of TE's fused RoPE against titan's "
                "positions interface -- NOT the installed TransformerEngine. "
                "It copies TE's inner block functions and adds its own "
                "__global__ and its own BSHD launch, so the row against "
                "mcore/base shares the arithmetic and not the addressing or "
                "the grid; marker-guarded, and needs a C++20 host compiler"
            ),
            builder="benchmarks.kernel.operations.rope:build_rope_titan_te",
            modes=("forward", "backward"),
            # Per ARM, not per scenario: without gcc-13 this scenario still
            # measures the other four arms. resolve_arm_skips drops this one
            # alone, and no other arm names it as a correctness reference.
            requires_gcc_toolset=True,
            compiled=True,
            correctness=(ROPE_GATE, ROPE_ULP_GATE, ROPE_CROSS_ENGINE_GATE),
        ),
    ),
    # comparisons left at None: the derived set is exactly the four rows this
    # scenario publishes. Three are cross-engine (each titan arm against
    # mcore/base) and one is within-engine (mcore/no_rope_fusion against
    # mcore/base), which is the whole reason that arm exists. There is no
    # floor to exclude: the cross-engine roster retires rope/copy_floor,
    # because a bandwidth floor answers no cross-engine question. The suite's
    # only x_floor test moved to qk_norm with it, and this scenario no longer
    # reports an x_floor column at all.
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


# The gate both arms face. ``out`` is a gather, so it is not a reduction and an
# exact metric would be defensible there -- measured on CPU at a tiny shape the
# titan arm's rel_l2 against fp64 is exactly 0.0, because a gather copies bf16
# rows and promoting them afterwards loses nothing. The gradient is a reduction
# over batch*seq_len rows, so max and ULP metrics report garbage wherever
# cancellation drives an output toward zero, and rel_l2 is the only safe metric
# for it (CLAUDE.md, "Choosing a correctness metric"). One gate at one tolerance
# covers both rather than splitting a formality from a real check.
#
# ``weight_grad_rows`` is the gradient restricted to the rows the tokens
# touched. An fp64 reference for the whole [vocab_size, dim] gradient is
# 1.16 GiB at ``normal`` and 13.9 GiB at ``huge``, in a process that also holds a
# whole GPTModel.
#
# ``weight_grad_norm`` is a scalar over the *entire* table, and it is weak
# evidence rather than a proof. It bounds a gross write outside the touched rows
# and nothing finer: a Frobenius norm over U touched rows moves by
# sqrt(1 + k/U) - 1 when k further rows are contaminated, which at U ~ 4041 is
# 1.2e-4 for one stray row and needs about 163 of them to reach this gate.
# Nothing in this scenario bounds a small stray write.
EMBEDDING_STAGE_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("out", "weight_grad_rows", "weight_grad_norm"),
    max_rel_l2=2e-2,
)


EMBEDDING_STAGE = KernelScenario(
    name="embedding_stage",
    description=(
        "The token-embedding lookup at the top of the model, cross-engine: "
        "megatron-core's LanguageModelEmbedding against TorchTitan's "
        "tok_embeddings. NEITHER arm is charged a layout conversion, and that "
        "is a measurement, not an omission: megatron's "
        "transpose(0,1).contiguous() (language_model_embedding.py:124) is "
        "entered every run, but our driver packs THD as [1, batch*seq_len] "
        "(e2e/megatron/data.py:54), so the transposed view carries a size-1 "
        "dimension, is already contiguous, and .contiguous() returns self. "
        "Backward is free for the same reason. So the two arms differ in the "
        "lookup and in dispatch alone. The mcore arm does pay a wrapper titan "
        "has none of: an @nvtx_decorator that wraps unconditionally (the "
        "_nvtx_enabled check is inside the pushed range, utils.py:2712), a "
        "second nn.Module.__call__, and a Dropout that ATen short-circuits at "
        "p=0 -- sub-microsecond each, and this scenario is dispatch-bound. "
        "BOTH arms are EAGER, which is the production treatment on both "
        "engines: megatron compiles no whole layer, and apply_compile reaches "
        "only the children of model.layers while tok_embeddings is a sibling "
        "of them. The stage performs zero FLOPs, so every microsecond is bytes "
        "or dispatch and the x_floor column decides whether the ratio is a "
        "kernel claim at all. Token ids are drawn uniformly over the full "
        "151936-row vocabulary, which is the worst case for the gather: a real "
        "c4_test run touches ~2020 rows and keeps them in L2, so read the "
        "absolute number as an upper bound -- and note the uniform draw "
        "dilutes the ratio toward 1.0, because both arms gather the same rows "
        "through the same F.embedding call. The RoPE-state handoff is excluded "
        "on both sides; megatron's per-step rotary_pos_emb build belongs to the "
        "rope scenario's provenance."
    ),
    inputs_builder=(
        "benchmarks.kernel.operations.embedding_stage:embedding_stage_inputs"
    ),
    reference_builder=(
        "benchmarks.kernel.operations.embedding_stage"
        ":embedding_stage_reference"
    ),
    baseline_arm="mcore/base",
    # Explicit, and exhaustive: this scenario publishes exactly one ratio. The
    # derived set would give the same pair today, but a cross-engine scenario
    # states which row it publishes rather than inheriting it, and the direction
    # matches the e2e piper1b_megatron scenario, where megatron is also the
    # anchor.
    comparisons=(("titan", "mcore/base"),),
    arms=(
        KernelArm(
            name="copy_floor",
            description=(
                "One read and one write of a [batch, seq_len, dim] bf16 "
                "tensor: the bandwidth floor for the gather's traffic at this "
                "shape"
            ),
            builder=(
                "benchmarks.kernel.operations.embedding_stage:"
                "build_embedding_stage_copy_floor"
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
                "megatron GPTModel.embedding off a real model: "
                "VocabParallelEmbedding's F.embedding at tp_size 1, plus the "
                "module's own transpose(0,1).contiguous(), which is a free "
                "view at the THD packing this repo runs -- proved on a probe "
                "call, not assumed. Eager, as megatron runs it"
            ),
            builder=(
                "benchmarks.kernel.operations.embedding_stage:"
                "build_embedding_stage_mcore_base"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer, so every module "
                "GPTModel builds runs eager end to end; compiling this one "
                "would measure a treatment megatron never applies"
            ),
            correctness=(EMBEDDING_STAGE_GATE,),
        ),
        KernelArm(
            name="titan",
            description=(
                "TorchTitan Decoder.tok_embeddings: F.embedding from the "
                "production config node, BSD throughout with no layout "
                "conversion -- eager, because apply_compile reaches only the "
                "children of model.layers and tok_embeddings is a sibling of "
                "them"
            ),
            builder=(
                "benchmarks.kernel.operations.embedding_stage:"
                "build_embedding_stage_titan"
            ),
            modes=("forward", "forward_backward"),
            # The second titan module arm in the registry that is not compiled,
            # and the reason is fidelity rather than convenience.
            # ``apply_compile`` walks ``model.layers.named_children()`` alone
            # (``distributed/compile.py:57-58``), and ``Decoder.__init__``
            # builds ``tok_embeddings`` at ``models/common/decoder.py:234``
            # against ``self.layers`` at ``:236``, so the embedding sits outside
            # every compiled region in production. ``final_norm`` carries the
            # same correction for the same structural reason.
            eager_reason=(
                "Decoder.tok_embeddings sits outside every compiled region: "
                "apply_compile walks model.layers.named_children() alone and "
                "the embedding is built as a sibling of layers, so a "
                "torch.compile here would time a treatment no run of this model "
                "applies to the lookup"
            ),
            correctness=(
                EMBEDDING_STAGE_GATE,
                # The cross-engine gate, enforced. It is what makes the ratio a
                # comparison of two implementations of one function: both arms
                # gather the same rows of the same table, so a disagreement here
                # means they no longer compute the same thing. It sits on the
                # non-anchor arm on purpose: ``resolve_arm_skips`` closes the
                # skip set over correctness references, so a check pointing from
                # the anchor at ``titan`` would let a skipped titan arm take the
                # anchor -- and the whole scenario -- down with it.
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=("out", "weight_grad_rows", "weight_grad_norm"),
                    max_rel_l2=2e-2,
                ),
                # Recorded, not enforced, exactly as the qkv scenario records
                # its bitwise row. A gather is a copy: both engines call
                # F.embedding on the same bf16 table with the same ids, so the
                # forward outputs should be bit-identical, and this states that
                # claim in the results rather than leaving it implied by a
                # tolerance. Informational because the enforcement above is
                # already sufficient, and a future change that made the two
                # forwards differ in the last bit -- an fp8 cast, a fused
                # epilogue -- should show up as a recorded fact rather than
                # abort a measured run. The gradient is deliberately absent: it
                # is a scatter-add whose accumulation order neither engine
                # fixes, so bit-identity there is not even expected.
                CorrectnessCheck(
                    kind="bitwise",
                    reference="mcore/base",
                    outputs=("out",),
                    informational=True,
                ),
            ),
        ),
    ),
)


# The gate every arm faces. Six outputs, each producing its own row in
# results.json, at the tolerance CLAUDE.md sets for a bf16 kernel.
#
# ``max_rel_l2`` is the only safe metric here, and the reason is not stylistic:
# both halves of this cut are reductions. The RMSNorm reduces over ``dim`` and
# each projection is a dot product over ``dim``, so cancellation drives
# individual outputs toward zero, and a max or ULP metric divides a negligible
# absolute error by that tiny magnitude and reports thousands of ULPs for a
# numerically perfect kernel -- including the stock one (CLAUDE.md, "Choosing a
# correctness metric").
#
# ``qkv_weight_grad`` is reported in megatron's grouped interleave, which is
# also titan's fused layout. Two arms produce it by conversion and two hold it
# natively: ``titan/unfused_qkv`` and the fp64 reference call
# ``benchmarks.models.piper_qwen3.megatron_weights.grouped_qkv``, which proves
# its own inverse bitwise on every call; ``mcore/base`` and ``titan`` read the
# gradient of their own fused parameter and convert nothing.
#
# So **two** implementations of the interleave meet in this gate, not one. The
# first-party ``grouped_qkv`` puts the reference and the unfused arm in that
# layout, and torchtitan's own ``FusedQKVLinear._merge_qkv_on_load``
# (``models/common/attention.py:871-895``) is what puts ``titan``'s ``wqkv`` in
# it at load. They are the same cat and reshape written twice, once here and
# once upstream, and this gate is what holds them together at run time: a
# divergence moves ``qkv_weight_grad`` on the fused arms alone and fails them.
QKV_PREP_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=(
        "q_out",
        "k_out",
        "v_out",
        "x_grad",
        "qkv_weight_grad",
        "norm_weight_grad",
    ),
    max_rel_l2=2e-2,
)


QKV_PREP = KernelScenario(
    name="qkv_prep",
    description=(
        "The attention-input norm, the QKV projection and the split that "
        "follows it, cross-engine: megatron-core's fused "
        "TELayerNormColumnParallelLinear plus get_query_key_value_tensors "
        "against TorchTitan's attention_norm plus FusedQKVLinear. Both titan "
        "arms are compiled (fullgraph=True) and the megatron arm is eager, "
        "which is how each engine runs it. THE NORM IS INSIDE THE SCENARIO ON "
        "BOTH ENGINES, because megatron fuses it into linear_qkv and exposes "
        "no way to time either half alone -- so these numbers are NOT "
        "comparable to the qkv scenario's, whose arms are the same projections "
        "without a norm. The cut ends at three separate [B, L, N, H] tensors, "
        "so titan's split and megatron's view/SplitAlongDim/reshape are both "
        "timed. THE TWO ENGINES COMPUTE THE SAME FUNCTION BUT DO NOT "
        "MATERIALIZE THE SAME TENSORS: SplitAlongDim is torch.split off the "
        "FP8 path and returns views, and megatron reshapes only the query, so "
        "it hands k and v on as non-contiguous strided views while titan "
        "materializes all three. At batch 4 / seq 1024 / normal / bf16 that is "
        "8 MiB of forward copy for mcore/base, 16 MiB for titan and none for "
        "titan/unfused_qkv, whose three GEMMs write contiguous outputs "
        "already. This is real engine behaviour on both sides and is "
        "deliberately NOT equalized, but megatron does not avoid the cost -- it "
        "defers it to whoever consumes the strided views, which is the "
        "attention_core scenario for megatron and nowhere for titan. The "
        "scenarios therefore sum correctly, and this row read alone overstates "
        "titan's projection cost by roughly that traffic. The qk norms are "
        "excluded on both sides: scenario qk_norm owns them, and the megatron "
        "arm sets q_layernorm/k_layernorm to None, which is megatron's own "
        "representation of a model built without them."
    ),
    inputs_builder="benchmarks.kernel.operations.qkv_prep:qkv_prep_inputs",
    reference_builder=(
        "benchmarks.kernel.operations.qkv_prep:qkv_prep_reference"
    ),
    baseline_arm="mcore/base",
    # Explicit, and deliberately not the set the default derivation would
    # produce. The derivation would put titan/unfused_qkv against mcore/base,
    # and that row would change two things at once: the engine and the QKV
    # fusion. Fused against unfused is a titan-internal question, so its honest
    # opponent is titan. The first row is the cross-engine one and is anchored
    # on megatron, matching the e2e piper1b_megatron scenario and every other
    # cross-engine scenario in this partition.
    comparisons=(
        ("titan", "mcore/base"),
        ("titan/unfused_qkv", "titan"),
    ),
    arms=(
        KernelArm(
            name="mcore/base",
            description=(
                "Megatron-core self_attention: TE "
                "TELayerNormColumnParallelLinear with the RMSNorm fused into "
                "the GEMM prologue, then get_query_key_value_tensors; eager, "
                "tp_size 1, qk norms removed"
            ),
            builder=(
                "benchmarks.kernel.operations.qkv_prep"
                ":build_qkv_prep_mcore_base"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer, so every TE "
                "module it builds runs eager end to end; compiling this one "
                "would measure a treatment megatron never applies"
            ),
            correctness=(QKV_PREP_GATE,),
        ),
        KernelArm(
            name="titan",
            description=(
                "TorchTitan attention_norm plus FusedQKVLinear -- upstream "
                "qwen3's own default, since _build_qwen3_moe_layers takes "
                "fuse_qkv=True -- under torch.compile(fullgraph=True)"
            ),
            builder=(
                "benchmarks.kernel.operations.qkv_prep:build_qkv_prep_titan"
            ),
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(
                QKV_PREP_GATE,
                # The cross-engine gate, and the only enforcing check that
                # states the scenario's claim directly: the two engines compute
                # the same function of the same parameters, so a ratio between
                # them is a ratio of implementations and not of arithmetic.
                # Declared on ``titan`` and referencing ``mcore/base`` rather
                # than the reverse, because ``resolve_arm_skips`` closes the
                # skip set over correctness references: a check pointing the
                # other way would make the anchor's survival depend on a titan
                # arm.
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=(
                        "q_out",
                        "k_out",
                        "v_out",
                        "x_grad",
                        "qkv_weight_grad",
                        "norm_weight_grad",
                    ),
                    max_rel_l2=2e-2,
                ),
            ),
        ),
        KernelArm(
            name="titan/unfused_qkv",
            description=(
                "TorchTitan attention_norm plus QKVLinear: three separate "
                "wq/wk/wv GEMMs, under torch.compile(fullgraph=True). NOT an "
                "upstream configuration -- fuse_qkv defaults to True in both "
                "qwen3 layer builders, every registered flavor passes True "
                "explicitly, and only _debugmodel_non_fused_qkv turns it off -- "
                "so this arm answers a fusion question, not a question about "
                "how TorchTitan ships"
            ),
            builder=(
                "benchmarks.kernel.operations.qkv_prep"
                ":build_qkv_prep_titan_unfused_qkv"
            ),
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(
                QKV_PREP_GATE,
                # The enforcing arm-to-arm check, matching what the existing
                # qkv scenario gives its own fused arm. The fp64 gate alone does
                # not cover this: it bounds each arm against the truth at 2e-2,
                # which bounds the *pair* only transitively, at 4e-2 -- and the
                # pair is exactly what the published ("titan/unfused_qkv",
                # "titan") row is a ratio of. Direction follows the same rule as
                # the cross-engine check: the reference is the row's opponent,
                # so ``resolve_arm_skips`` never removes an arm that another
                # arm's row depends on. The references now chain --
                # titan/unfused_qkv -> titan -> mcore/base -- and the closure is
                # a fixed point, so losing the anchor skips all three rather
                # than leaving a dangling row. That is the anchor cost
                # KERNEL_BASELINE_ARMS already records for every cross-engine
                # scenario, not a new one.
                CorrectnessCheck(
                    kind="tolerance",
                    reference="titan",
                    outputs=(
                        "q_out",
                        "k_out",
                        "v_out",
                        "x_grad",
                        "qkv_weight_grad",
                        "norm_weight_grad",
                    ),
                    max_rel_l2=2e-2,
                ),
                # Informational, not enforcing, and carried for exactly the
                # reason the qkv scenario carries its own: with identical
                # weights the fused and unfused paths *should* agree bitwise,
                # and for a while they did, until compiled GEMM epilogues broke
                # bit-identity. Recording the difference without failing the run
                # keeps a change that restores or further degrades exact
                # agreement visible in results.json instead of invisible. The
                # enforcing check above is what bounds the pair; this one only
                # reports how much better than that bound the two actually
                # agree.
                CorrectnessCheck(
                    kind="bitwise",
                    reference="titan",
                    outputs=("q_out", "k_out", "v_out"),
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
# The gate all three arms face. An add is not a reduction, so an exact metric
# would be defensible on ``out`` -- and the informational bitwise gate below
# states that claim directly. rel_l2 is what enforces, because the two
# gradients are the output gradient itself and the whole set takes one
# tolerance rather than three metrics. 2e-2 is CLAUDE.md's gate for a bf16
# kernel, and a single bf16 add lands at ~0 against fp64: both addends are
# exact in fp64 and the sum rounds once.
ATTN_RESIDUAL_FP64_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("out", "attn_out_grad", "residual_grad"),
    max_rel_l2=2e-2,
)

# The cross-engine gate, enforced, and the within-engine one beside it. Both
# sit on a non-anchor arm and point at ``mcore/base``: ``resolve_arm_skips``
# closes the skip set over correctness references, so a check pointing from
# the anchor at another arm would let that arm's skip take the anchor -- and
# the whole scenario -- down with it. This holds even though the scenario
# publishes no cross-engine ratio. A gate is not a comparison; it is what
# proves the three arms compute one function.
ATTN_RESIDUAL_AGREEMENT_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="mcore/base",
    outputs=("out", "attn_out_grad", "residual_grad"),
    max_rel_l2=2e-2,
)

# Recorded, not enforced, as the qkv and embedding_stage scenarios record
# theirs. The three arms should agree bitwise: the exact sum of two bf16
# values fits in fp32, every torch backend accumulates a bf16 add in fp32,
# and so the correctly rounded bf16 result is the only result any of them can
# produce. This row is the published evidence for the declined cross-engine
# ratio -- if the two engines are bit-identical, there is no arithmetic left
# to compare and the only remaining difference is fusion scope.
#
# Informational rather than enforcing, because the claim is unmeasured: no
# arm of this scenario has run on a GPU, and CLAUDE.md records that compiled
# GEMM epilogues once broke a bit-identity the qkv scenario expected. An add
# has no epilogue, so the expectation is stronger here -- but the honest
# order is to record it, run it, and promote it only if the hardware agrees.
ATTN_RESIDUAL_BITWISE_GATE = CorrectnessCheck(
    kind="bitwise",
    reference="mcore/base",
    outputs=("out", "attn_out_grad", "residual_grad"),
    informational=True,
)


ATTN_RESIDUAL = KernelScenario(
    name="attn_residual",
    description=(
        "The residual add after attention: TorchTitan's x + attention(...) "
        "against megatron-core's self_attn_bda. BOTH ENGINES COMPUTE THE SAME "
        "FUNCTION, AND THEY COMPUTE IT IDENTICALLY. The base profile sets "
        "hidden_dropout 0.0 and add_bias_linear False, so "
        "_bias_dropout_add_func takes its no-bias branch, F.dropout(p=0.0) "
        "returns its own input, and what remains is out = residual + out. "
        "bias_dropout_add is the function's name, not this model's operation, "
        "and every arm proves the equality on the device before it is timed. "
        "THE DIFFERENCE IS FUSION SCOPE, NOT PRESENCE: titan's add is one node "
        "of a whole-block Inductor graph and folds into the next norm's "
        "prologue, while megatron's bias_dropout_add_fused_train is "
        "@jit_fuser-decorated, compiles as its own region and emits one "
        "standalone add that cannot fuse outward. Isolating the cut forces "
        "titan into megatron's fusion scope, so a titan-against-megatron ratio "
        "would report the isolation and not the engines. THIS SCENARIO "
        "THEREFORE PUBLISHES NO CROSS-ENGINE ROW, and the omission is a "
        "declaration -- see the comparisons tuple. The one row it does publish "
        "is within megatron, and it is a HOST-DISPATCH comparison rather than "
        "a kernel one: both mcore arms run the same single bf16 add on the "
        "same tensors, so their device work is identical by construction and "
        "the whole ratio is the compiled region's guard check against a fresh "
        "Python closure per call. Run --burst and read the residual before "
        "quoting it. The per-call resolution of "
        "self_attn_bda(training, bias_dropout_fusion) and the enclosing "
        "torch.enable_grad context are INSIDE the timed closure, because "
        "megatron enters both on every layer of every step and the resolution "
        "is where the unfused arm builds its closure. The "
        "attention_output_with_bias tuple is excluded: scenario 6 produces it. "
        "THE MOST LIKELY WAY THIS ROW PUBLISHES A WRONG NUMBER IS A NULL. One "
        "add moves 24 MiB, which is roughly 11-13 us of device work at "
        "normal/batch 4/seq 1024, and the two dispatch paths plausibly cost "
        "5 to 40 us each -- so this scenario sits AT the crossover rather "
        "than safely above it, unlike rope, where dispatch is 6-19x the "
        "floor. If the arms are device-bound, both report the same ~12 us, "
        "the ratio lands at 1.00 with a tight interval, and the whole "
        "declared delta is invisible. --burst cannot settle that, because "
        "CLAUDE.md records the residual test as one-sided. copy_floor is the "
        "instrument that settles it: AN x_floor NEAR 1.5 ON BOTH ARMS MEANS "
        "THE ROW MEASURED THE MEMORY BUS AND NOT THE FUSION. Read that "
        "column before quoting the ratio. The floor is a COPY, so it moves "
        "two thirds of the add's bytes -- multiply it by 1.5 before reading "
        "it as the add's device cost, because the raw x_floor column "
        "overstates the distance. No other arm declares bytes_moved, because "
        "one byte count cannot describe both forward and forward_backward. "
        "Finally, the flag NAMES THREE OPERATIONS AND THIS MODEL RUNS ONE: "
        "with hidden_dropout 0.0 and add_bias_linear False the bias add and "
        "the dropout do not exist, so bias_dropout_fusion selects a "
        "@jit_fuser region wrapped around a single add with nothing to fuse "
        "it to. The row is what wrapping one add in torch.compile costs, not "
        "what megatron's bias-dropout-add fusion costs."
    ),
    inputs_builder=(
        "benchmarks.kernel.operations.attn_residual:attn_residual_inputs"
    ),
    reference_builder=(
        "benchmarks.kernel.operations.attn_residual:attn_residual_reference"
    ),
    baseline_arm="mcore/base",
    # Explicit and exhaustive, and the derived set would be wrong here. It
    # would publish titan against mcore/base, which is the one row this
    # scenario exists to decline. Plan section C.0 rule 6 names scenario 7 as
    # one of the four within-engine-only scenarios.
    comparisons=(
        # Within megatron, and the scenario's reason to exist: what the
        # @jit_fuser region on bias_dropout_add costs against the eager
        # closure. Both sides emit one bf16 add, so the row is a dispatch
        # comparison and the caption must say so.
        ("mcore/no_bias_dropout_fusion", "mcore/base"),
        #
        # DECLINED, and recorded so nobody re-adds it: ("titan", "mcore/base").
        #
        # The two arms compute the same function -- the bitwise gate above
        # records it -- so the ratio would carry no arithmetic difference at
        # all. What it would carry is the isolation: titan's add emits no
        # kernel in production because it folds into the prologue of the next
        # norm, and an isolated arm has no neighbour to fold into. The number
        # would land near 1.0 and would be read as "the two engines add at the
        # same speed", which is a statement about this harness rather than
        # about either engine.
        #
        # The titan arm stays, and it is not decoration. It is the scenario-7
        # term of the attn_residual_norm span over 6+7+8, which is compared
        # against the sum of the scenarios it replaces: the span's claim is
        # that titan's add disappears into the norm, and that claim is
        # measured as span minus sum, so the sum needs this number. It carries
        # the cross-engine gate that proves the two engines compute one
        # function, which is the evidence this declined row rests on. It keeps
        # the partition's titan side complete, so qwen3/model.py:60 belongs to
        # a scenario. And it is the standing measurement of what that add
        # costs alone, if a later change stops Inductor from folding it.
        #
        # The counter-argument -- that a reader can divide the two absolute
        # numbers anyway -- is the same one the cross_entropy scenario records
        # as declined, and gets the same answer: results.json has nowhere to
        # put a caption, so a published row would carry the same visual status
        # as a genuine one. Promoting a ratio to a row is an editorial act.
    ),
    arms=(
        KernelArm(
            name="copy_floor",
            description=(
                "One read of attn_out and one write: the bandwidth floor for "
                "this shape. It is what separates a kernel result from a "
                "dispatch comparison, which this scenario needs because both "
                "sides of its published row run the same single bf16 add. A "
                "copy moves TWO THIRDS of the add's bytes -- the add reads two "
                "operands and writes one -- so this arm understates the device "
                "by a third and the x_floor column overstates the distance by "
                "the reciprocal. Multiply this arm's median by 1.5 before "
                "reading it as the add's device cost"
            ),
            builder=(
                "benchmarks.kernel.operations.attn_residual"
                ":build_attn_residual_copy_floor"
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
                "megatron's fused bias-dropout-add: the call site resolves to "
                "bias_dropout_add_fused_train, which IS a torch.compile "
                "wrapper -- @jit_fuser at fused_bias_dropout.py:69, and "
                "megatron/core/jit.py binds jit_fuser to torch.compile at "
                "line 21 and applies the binding at import on line 33. "
                "bias_dropout_fusion is True in the base profile because "
                "megatron's own argparse layer sets it, so this is megatron as "
                "a real run gets it. THE WHOLE TIMED PAYLOAD IS THE COMPILED "
                "CALL, which is why this arm declares compiled=True and the "
                "other mcore arm does not: torch.compile is the treatment "
                "under test, not a property of the surrounding harness. "
                "megatron compiles no whole transformer layer, so everything "
                "outside this one function stays eager -- the arm is compiled "
                "at the cut and eager around it, and no harness choice added "
                "either. The build refuses to continue unless the resolved "
                "callable really is a torch.compile wrapper, so the "
                "declaration is proved rather than asserted. NOTE THAT THIS "
                "IS NOT THE SAME TREATMENT AS A TITAN compiled=True ARM: "
                "megatron applied the compile at import, the harness applied "
                "nothing, and the region is one function rather than a whole "
                "block. This is the scenario anchor and the arm both other "
                "arms are gated against"
            ),
            builder=(
                "benchmarks.kernel.operations.attn_residual"
                ":build_attn_residual_mcore_base"
            ),
            modes=("forward", "forward_backward"),
            # compiled=True, and it is the one megatron arm in the whole
            # registry that takes it. The precedent it does not follow is
            # cross_entropy/mcore/ce_native, which declares compiled=False
            # around @jit_fuser helpers -- and the difference is where the
            # compile sits. There the arm's entry point is a plain method and
            # the compiled regions are helpers nested inside it. Here the
            # entry point IS the compiled function: the timed closure calls
            # the torch.compile wrapper directly. Declaring this arm eager
            # would put "eager vs eager" in the manifest for a row whose whole
            # delta is that compile, which is the mislabelling the
            # eager_reason contract exists to prevent.
            compiled=True,
            correctness=(ATTN_RESIDUAL_FP64_GATE,),
        ),
        KernelArm(
            name="mcore/no_bias_dropout_fusion",
            description=(
                "megatron with bias_dropout_fusion off: the call site "
                "resolves to bias_dropout_add_unfused, which builds a fresh "
                "Python closure on every call and then dispatches the same "
                "arithmetic eagerly. Turning the flag off is a DEVIATION from "
                "megatron, not a return to its default, because the "
                "TransformerConfig dataclass default is the opposite of what "
                "megatron's argparse layer gives a real run. The device work "
                "is the same single bf16 add the anchor runs, so the ratio "
                "against the anchor is host dispatch and nothing else"
            ),
            builder=(
                "benchmarks.kernel.operations.attn_residual"
                ":build_attn_residual_mcore_no_bias_dropout_fusion"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "this arm IS the eager side of the fusion delta: "
                "bias_dropout_fusion=False is what removes the torch.compile "
                "region, and compiling the arm would put it back under "
                "another name"
            ),
            correctness=(
                ATTN_RESIDUAL_FP64_GATE,
                ATTN_RESIDUAL_AGREEMENT_GATE,
                ATTN_RESIDUAL_BITWISE_GATE,
            ),
        ),
        KernelArm(
            name="titan",
            description=(
                "TorchTitan's x = x + self.attention(...) "
                "(qwen3/model.py:60): one binary operator, under "
                "torch.compile(fullgraph=True), applied to a function rather "
                "than to a module because the call site has no nn.Module "
                "wrapper and this scenario is dispatch-bound. THIS ARM IS IN "
                "NO PUBLISHED COMPARISON, by declaration: an isolated titan "
                "add emits a standalone kernel that production never emits, "
                "because in a whole-block graph it folds into the prologue of "
                "the next norm. It is here as the scenario-7 term of the "
                "attn_residual_norm span over 6+7+8, as the side of the "
                "cross-engine gate that proves both engines compute one "
                "function, and as the standing cost of the add alone"
            ),
            builder=(
                "benchmarks.kernel.operations.attn_residual"
                ":build_attn_residual_titan"
            ),
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(
                ATTN_RESIDUAL_FP64_GATE,
                ATTN_RESIDUAL_AGREEMENT_GATE,
                ATTN_RESIDUAL_BITWISE_GATE,
            ),
        ),
    ),
)


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


# One shared fp64 gate for all six arms, at the tolerance every bf16 reduction
# in this repo uses. ``max_rel_l2`` only: cross-entropy is a reduction over the
# vocabulary, and CLAUDE.md forbids a max/ULP metric on one -- cancellation
# drives individual gradient entries toward zero, so dividing a negligible
# absolute error by that magnitude reports thousands of ULPs for a perfect
# kernel.
#
# The six arms round the gradient to bf16 a different number of times, and the
# gate is deliberately not per-arm, because they all round exactly once at this
# workload: ``mcore/base`` stores in bf16 and rescales in bf16, but the scale is
# 1/4096 = 2**-12 exactly, which is a power of two and therefore exact;
# ``mcore/ce_native`` casts once (``fusions/fused_cross_entropy.py:82``, and
# unconditionally, regardless of the model dtype); ``mcore/no_ce_fusion``
# returns fp32 and autograd casts once; ``titan/full_logits`` and
# ``titan/te_fused_ce`` keep an fp32 buffer and round once; and
# ``titan/piper_optimized_te_ce`` applies its scale in fp32 inside the kernel
# before a single bf16 store. A workload whose ``batch * seq_len`` is not a
# power of two would make ``mcore/base`` round twice, which is a widened gate
# rather than a bug -- measure before widening.
CROSS_ENTROPY_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("loss", "logits_grad"),
    max_rel_l2=2e-2,
)


CROSS_ENTROPY = KernelScenario(
    name="cross_entropy",
    description=(
        "The loss and only the loss, cross-engine: megatron-core's "
        "compute_language_model_loss against TorchTitan's CrossEntropyLoss and "
        "the two fused losses this repo runs, over one shared set of "
        "materialized logits. The LM-head projection is scenario 15 and is "
        "excluded here, so no number is comparable to the lm_head scenario, "
        "which measures the projection and the loss together. Megatron's "
        "method materializes a transposed label copy before the kernel "
        "(language_module.py:172) and transposes the per-token loss back "
        "(:205); the titan arms are charged the same label preparation, so "
        "both engines pay two small layout kernels per call and the ratio is "
        "not a report of megatron's own layout. The three titan arms are "
        "compiled with the production CompileConfig(components=['loss']) and "
        "the three megatron arms are eager, which is how each engine runs this "
        "code. READ THE forward_backward ROW, not the forward row: the arms "
        "divide the work differently across that boundary, because mcore/base, "
        "titan/te_fused_ce and titan/piper_optimized_te_ce write the whole "
        "gradient inside forward, while titan/full_logits, mcore/ce_native and "
        "mcore/no_ce_fusion compute only a softmax or a log-softmax there and "
        "build the gradient in backward. A forward row therefore compares "
        "operations that are not the same operation. Note also that mcore/base "
        "and titan/te_fused_ce are the SAME TransformerEngine Triton kernel "
        "from two sources -- installed TE 2.17.1 for the megatron arm, our "
        "vendored snapshot under components/lm_head/ for the titan arm, and the "
        "snapshot writes its gradient into a separate fp32 buffer where "
        "installed TE overwrites and returns the caller's bf16 logits -- so "
        "this scenario publishes no ratio between them. Finally, megatron as "
        "NVIDIA ships it is mcore/no_ce_fusion, not mcore/ce_native: the tree's "
        "only default is cross_entropy_loss_fusion=False."
    ),
    inputs_builder=(
        "benchmarks.kernel.operations.cross_entropy:cross_entropy_inputs"
    ),
    reference_builder=(
        "benchmarks.kernel.operations.cross_entropy:cross_entropy_reference"
    ),
    baseline_arm="mcore/base",
    # Explicit and exhaustive, because the derived set is wrong here twice
    # over: it would compare the two megatron variants against megatron (right)
    # and all three titan arms against megatron (wrong for two of them). Each
    # row below states the one question it answers.
    comparisons=(
        # The scenario's reason to exist: TorchTitan's own loss against
        # megatron's, at the fastest CE megatron can reach. Two different
        # implementations of the same function, so the ratio is a real
        # cross-engine kernel comparison.
        ("titan/full_logits", "mcore/base"),
        # Within megatron: what megatron's own fused non-TE CE costs against
        # TE's. This is the fastest CE megatron's own training entrypoint
        # permits, because it refuses the TE fusion outright
        # (arguments.py:1631-1634) -- it is NOT "megatron as NVIDIA ships it",
        # which is the row below.
        ("mcore/ce_native", "mcore/base"),
        # Within megatron: megatron as NVIDIA ships it -- fusion off is the
        # tree's only default -- against megatron's fastest available CE. It is
        # also the floor for the row above, and the only row that isolates the
        # fusion itself rather than the choice of fused implementation.
        ("mcore/no_ce_fusion", "mcore/base"),
        # Within titan: what the vendored TE kernel buys over the bare torch
        # loss. The titan-side mirror of the ce_native row.
        ("titan/te_fused_ce", "titan/full_logits"),
        # Plan section C.5: the Piper arm against the snapshot it modifies, and
        # against nothing else. It is a rework of the vendored TE kernel, so
        # comparing it to titan/full_logits or to mcore/base would credit it
        # with the whole TE gain -- which is TE's. A reader who wants Piper
        # against the bare loss chains this row with the one above it, which is
        # the honest way to say it.
        ("titan/piper_optimized_te_ce", "titan/te_fused_ce"),
        #
        # DECLINED, and recorded so nobody re-adds it:
        # ("titan/te_fused_ce", "mcore/base").
        #
        # An earlier draft published it with the caption "version drift plus
        # our wrapper", arguing that a reader can compute the ratio from the two
        # absolute numbers anyway. That argument would justify every row this
        # mechanism exists to suppress, and CLAUDE.md describes the mechanism
        # for exactly this case: the empty tuple "declares a scenario that
        # publishes no ratio at all, which a scenario whose two sides are not a
        # like-for-like cut must be able to say."
        #
        # The two sides are one TransformerEngine kernel from two sources,
        # differing by the fp32 gradient buffer and our wrapper -- not a
        # like-for-like cut. And the caption cannot ship: results.json has
        # nowhere to put one, so the row would land with the same visual status
        # as the genuine cross-engine row above it. Promoting a ratio to a
        # published row is an editorial act, and this one would assert a cut
        # that does not exist.
    ),
    arms=(
        KernelArm(
            name="mcore/base",
            description=(
                "Megatron compute_language_model_loss with "
                "cross_entropy_fusion_impl='te': INSTALLED TransformerEngine "
                "2.17.1's Triton cross-entropy, megatron's fastest available "
                "loss path and not the one a stock pretrain_gpt.py user gets "
                "(that is mcore/no_ce_fusion). Eager. THIS ARM DESTROYS ITS "
                "INPUT: installed TE writes the bf16 gradient into the caller's "
                "logit buffer and returns it, so the 30 warmup calls overwrite "
                "the logits before the first timed sample and NO TIMED SAMPLE "
                "RUNS ON THE DECLARED INPUT. Every sample runs on the "
                "near-constant fixed point the kernel converges to (about 1/V "
                "per element, one entry near -1 per row), which no training "
                "step produces. The kernel work per call is unchanged by this "
                "-- fixed trip count, label-only branching, identical traffic, "
                "no denormals -- but a near-constant bf16 buffer is a "
                "memory-access pattern the original ~N(0,1) data is not, and "
                "whether that changes achieved bandwidth on an H200 is "
                "UNMEASURED"
            ),
            builder=(
                "benchmarks.kernel.operations.cross_entropy"
                ":build_cross_entropy_mcore_base"
            ),
            modes=("forward", "forward_backward"),
            compiled=False,
            eager_reason=(
                "megatron compiles no whole layer, and "
                "compute_language_model_loss is a plain method with no compile "
                "on it; the work is inside TE's hand-written Triton kernel, "
                "which is what NVIDIA ships instead of a compiled region"
            ),
            correctness=(CROSS_ENTROPY_GATE,),
        ),
        KernelArm(
            name="mcore/ce_native",
            description=(
                "The same method with cross_entropy_fusion_impl='native': "
                "megatron's own fused non-TE cross-entropy, which is what "
                "fusion selects once TE is declined and the only fused path "
                "megatron's own training entrypoint permits. NOT megatron as "
                "NVIDIA ships it -- that is mcore/no_ce_fusion, because "
                "cross_entropy_loss_fusion defaults to False. Upcasts the whole "
                "[tokens, vocab] tensor to fp32, makes ~6 full-tensor "
                "traversals, and keeps that fp32 softmax resident for backward, "
                "where it then builds the gradient"
            ),
            builder=(
                "benchmarks.kernel.operations.cross_entropy"
                ":build_cross_entropy_mcore_ce_native"
            ),
            modes=("forward", "forward_backward"),
            compiled=False,
            eager_reason=(
                "eager at the method level, as every megatron arm is -- but not "
                "eager inside it: fused_cross_entropy.py's four helpers (:12, "
                ":25, :47, :64) carry @jit_fuser, jit_fuser is rebound to "
                "torch.compile on torch >= 2.2 by enable_jit_fuser "
                "(megatron/core/jit.py:16-25), and jit.py:33 calls it at "
                "import, so the rebinding is unconditional here. This arm's "
                "kernels come from Inductor and its dispatch does not, which is "
                "precisely what 'eager megatron' means"
            ),
            correctness=(CROSS_ENTROPY_GATE,),
        ),
        KernelArm(
            name="mcore/no_ce_fusion",
            description=(
                "The same method with cross_entropy_loss_fusion=False: MEGATRON "
                "AS NVIDIA SHIPS IT, since that is the tree's only default "
                "(model_parallel_config.py:320) and this rev's arguments.py "
                "declares no flag to change it. The plain vocab-parallel "
                "cross-entropy, same arithmetic as ce_native with no @jit_fuser "
                "on any of it and one more all_reduce. Eager throughout, and "
                "the within-engine floor the fused paths are measured against"
            ),
            builder=(
                "benchmarks.kernel.operations.cross_entropy"
                ":build_cross_entropy_mcore_no_ce_fusion"
            ),
            modes=("forward", "forward_backward"),
            compiled=False,
            eager_reason=(
                "the arm IS megatron's unfused path: turning the fusion off is "
                "the treatment under test, so compiling it would erase the "
                "difference between this arm and ce_native"
            ),
            # is_floor stays False because this is a real megatron code path,
            # not a synthetic bandwidth bound like qk_norm/copy_floor. It is NOT
            # because a floor would lose its comparison row: with an explicit
            # ``comparisons`` tuple, ``comparison_pairs()`` returns it verbatim
            # and consults ``is_floor`` only in the derived branch. What
            # ``is_floor`` would actually do here is suppress this arm's
            # ``peak_memory_gib`` and add an x-floor column -- and peak memory
            # is one of the two things this scenario measures.
            correctness=(CROSS_ENTROPY_GATE,),
        ),
        KernelArm(
            name="titan/full_logits",
            description=(
                "TorchTitan's CrossEntropyLoss over materialized logits, under "
                "the production CompileConfig(components=['loss']). This is OUR "
                "benchmark baseline and not TorchTitan's default: all twelve "
                "upstream qwen3 configs wrap the same loss in "
                "ChunkedLossWrapper, which owns the LM head and so spans "
                "scenarios 15 and 16. Bare, it is the like-for-like cut against "
                "megatron's method. Upcasts the whole [tokens, vocab] tensor to "
                "fp32 before F.cross_entropy, and defers the gradient to "
                "backward rather than writing it in forward"
            ),
            builder=(
                "benchmarks.kernel.operations.cross_entropy"
                ":build_cross_entropy_titan_full_logits"
            ),
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(
                CROSS_ENTROPY_GATE,
                # The one enforcing cross-engine gate, and it is declared on the
                # titan side deliberately. ``resolve_arm_skips`` closes the skip
                # set over correctness references, so a gate pointing AT
                # mcore/base costs only this arm if megatron is unavailable,
                # while a gate declared ON mcore/base pointing at a titan arm
                # would take the anchor with it and cost the whole scenario.
                #
                # This arm is the right side to carry it: it shares no code with
                # the TE path, so the check is a genuine agreement between two
                # independent implementations rather than an implementation
                # against a near-copy of itself, which is what a
                # titan/te_fused_ce-vs-mcore/base gate would be.
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=("loss", "logits_grad"),
                    max_rel_l2=2e-2,
                ),
            ),
        ),
        KernelArm(
            name="titan/te_fused_ce",
            description=(
                "TECrossEntropyLoss over our VENDORED snapshot of TE's Triton "
                "cross-entropy (components/lm_head/te_cross_entropy.py), "
                "compiled with CompileConfig(components=['loss']). The name "
                "says TE and the code is ours: the snapshot writes its gradient "
                "into a separate fp32 buffer where installed TE 2.17.1 -- which "
                "mcore/base reaches -- overwrites and returns the caller's bf16 "
                "logits. Diffed against installed 2.17.1, that buffer is the "
                "ONLY substantive difference, so this arm is TE with one "
                "change; the scenario therefore publishes no ratio against "
                "mcore/base. tests/test_lm_head_losses.py pins the snapshot by "
                "SHA-256, which guards our drift and not a TE upgrade "
                "underneath mcore/base"
            ),
            builder=(
                "benchmarks.kernel.operations.cross_entropy"
                ":build_cross_entropy_titan_te_fused_ce"
            ),
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(CROSS_ENTROPY_GATE,),
        ),
        KernelArm(
            name="titan/piper_optimized_te_ce",
            description=(
                "Piper's rework of that same vendored snapshot, compiled with "
                "CompileConfig(components=['loss']). It takes the normalization "
                "scale in forward and applies it in fp32 before its single bf16 "
                "store, so backward returns the saved tensor untouched instead "
                "of rescaling the whole [tokens, vocab] buffer. Its declared "
                "opponent is titan/te_fused_ce, the snapshot it modifies -- not "
                "the scenario anchor"
            ),
            builder=(
                "benchmarks.kernel.operations.cross_entropy"
                ":build_cross_entropy_titan_piper_optimized_te_ce"
            ),
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(CROSS_ENTROPY_GATE,),
        ),
    ),
)


KERNEL_SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        ROPE,
        SWIGLU,
        QKV,
        LM_HEAD,
        ATTENTION,
        EMBEDDING_STAGE,
        QKV_PREP,
        QK_NORM,
        ATTN_OUT_PROJ,
        ATTN_RESIDUAL,
        FFN_NORM,
        FINAL_NORM,
        CROSS_ENTROPY,
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
