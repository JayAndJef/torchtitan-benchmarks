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
        "path run eager. THE TWO ENGINES MAY SHARE THEIR INNER ARITHMETIC "
        "AND SHARE NOTHING ABOVE IT: the header of "
        "components/rope/te_rope_standalone.cu states that it copies TE's "
        "fused_rope block functions, and this tree cannot check that -- the "
        "installed TE wheel ships no sources and the header names no TE "
        "version. What is certain is that the file adds its own __global__ "
        "and its own BSHD launch configuration, while megatron at THD uses "
        "TE's THD launcher, whose grid is a function of the document count. "
        "So the "
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
                # Recorded, not enforced, exactly as the qkv_prep scenario
                # records its bitwise row. A gather is a copy: both engines call
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
        "no way to time either half alone -- so a number here is the norm "
        "and the projection together, and it is NOT comparable to a "
        "projection timed without a norm. The cut ends at three separate "
        "[B, L, N, H] tensors, "
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
        "defers it to whoever consumes the strided views, and BOTH HALVES OF "
        "THAT DEFERRAL ARE COLLECTED. The value stays a strided view and its "
        "4 MiB is timed in attention_core (scenario 5); the key stops being a "
        "view at k_layernorm, and qk_norm (scenario 3) hands its megatron arm "
        "that same strided key, so its 4 MiB is timed there. The scenarios "
        "therefore account for both tensors, and this row read alone "
        "overstates titan's projection cost by roughly that traffic. THAT IS "
        "A PROSE LEDGER: attention_core declares no bytes_moved, so no "
        "assertion anywhere checks that the halves sum. The qk norms are "
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
                # The enforcing arm-to-arm check on the fused/unfused pair.
                # The fp64 gate alone does not cover this: it bounds each arm against the truth at 2e-2,
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
                # Informational, not enforcing, and carried for one reason:
                # with identical weights the fused and unfused paths
                # *should* agree bitwise,
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
        "kernel. BOTH ENGINES READ THE SAME q AND k VALUES, IN EACH ENGINE'S "
        "OWN MEMORY LAYOUT, AND THE LAYOUTS DIFFER ON PURPOSE. Titan gets two "
        "contiguous tensors, which is what its qkv_linear materializes. "
        "Megatron gets a contiguous query and ONE NON-CONTIGUOUS STRIDED "
        "VIEW -- the key. Its QKV GEMM writes one fused buffer and splits it "
        "into views; the query then leaves the buffer because megatron "
        "reshapes it, and the key does not, so k_layernorm norms the view "
        "itself. TE's RMSNorm calls input_.contiguous() inside op_forward, so "
        "the mcore arm pays a 4 MiB read plus a 4 MiB write of the key inside "
        "every timed call at batch 4 / seq 1024 / normal. THAT COPY IS THE "
        "HALF OF qkv_prep's DEFERRAL THAT USED TO BE TIMED IN NO SCENARIO; "
        "attention_core times the value's half. THAT LEDGER IS PROSE AND "
        "NOTHING CHECKS IT: attention_core declares no bytes_moved on any "
        "arm, so only this scenario publishes a number and the two cannot "
        "disagree. The charge is not mirrored "
        "onto titan, because titan already pays the same materialization "
        "inside qkv_prep, so each engine pays it exactly once across the "
        "partition. mcore/base therefore declares its OWN bytes_moved, 8 MiB "
        "above the shared count, so the GB/s column shows the asymmetry "
        "instead of absorbing it. A norm may be at memory bandwidth, so "
        "copy_floor moves the shared 24 MiB and the x_floor column decides "
        "whether the ratio is a kernel claim at all. READ mcore/base's "
        "x_floor AGAINST 1.33, NOT AGAINST 1.0. The column is a ratio of "
        "times and reads no bytes_moved, so its arithmetic is unchanged -- "
        "but mcore/base moves 32 MiB against the floor's 24, so an mcore/base "
        "running at exactly the floor's bandwidth reads about 32/24 = 1.33. "
        "Divide by 1.33 to recover the usual reading. titan and the floor "
        "declare the same count and still read against 1.0. The floor "
        "declares forward only, so "
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
                "eager as megatron runs it, on SBHD tensors -- a contiguous "
                "query and the strided key its QKV split really hands over"
            ),
            builder=(
                "benchmarks.kernel.operations.qk_norm:"
                "build_qk_norm_mcore_base"
            ),
            # No isolated backward. TE's operation fuser clears its saved
            # tensors while it runs backward (ops/fuser.py:225,258) and TE's
            # RMSNorm calls clear_tensor_data on both of them at the end of
            # op_backward, so the retained-graph re-run rope and expert_mlp
            # use raises here. Both arms drop the mode and stay comparable;
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


# One GEMM and two gradients, at the tolerance qkv_prep already uses for the
# same class of operation.
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

# Recorded, not enforced, as the qkv_prep and embedding_stage scenarios
# record theirs. The three arms should agree bitwise: the exact sum of two bf16
# values fits in fp32, every torch backend accumulates a bf16 add in fp32,
# and so the correctly rounded bf16 result is the only result any of them can
# produce. This row is the published evidence for the declined cross-engine
# ratio -- if the two engines are bit-identical, there is no arithmetic left
# to compare and the only remaining difference is fusion scope.
#
# Informational rather than enforcing, because the claim is unmeasured: no
# arm of this scenario has run on a GPU, and CLAUDE.md records that compiled
# GEMM epilogues once broke a bit-identity the fused/unfused QKV pair
# expected. An add
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
        "what megatron's bias-dropout-add fusion costs. WARNING: THE x_floor "
        "COLUMN STILL LETS A READER RECOVER THE CROSS-ENGINE RATIO THIS "
        "SCENARIO REFUSES TO PUBLISH. The merge divides every non-floor arm's "
        "median by the floor's, so x_floor(titan) divided by "
        "x_floor(mcore/base) is exactly the suppressed median ratio: the "
        "floor median cancels, and no shared bytes_moved is needed for it. It "
        "exists in forward alone, because the floor declares forward alone, "
        "and no other arm declares bytes_moved, so neither side of that "
        "quotient carries a GB/s figure that could reproduce it a second way "
        "-- the GB/s heading is printed unconditionally, and both rows under "
        "it read n/a. "
        "Suppressing comparisons removes the ROW and not the "
        "NUMBER. The column cannot be removed without deleting the floor, and "
        "the floor is what decides whether the published row is a kernel "
        "result at all, so the hazard is stated instead: that quotient is "
        "meaningless for the reason above, and it carries the isolation "
        "rather than either engine."
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


# The scenario's outputs fall into two classes, and the gate roster below is
# that split written down. A router is a GEMM, a top-k and a softmax: the GEMM
# is continuous, and the top-k is a DISCRETE DECISION. Two implementations that
# agree to the last bit on the logits may still order two near-equal logits
# differently, and then they route a token to a different expert. That is not a
# wrong kernel. It is the one place a norm-based tolerance cannot tell a wrong
# kernel from a legitimate tie.
#
# So the enforced gates are the TIE-IMMUNE outputs, and they are enforced on
# every arm across every precision boundary:
#
#   logits          the gate GEMM, which holds no discrete decision at all. A
#                   wrong gate matrix, a wrong dtype or a wrong GEMM lands
#                   here and nowhere else.
#   prob_row_sums   1.0 for every token whichever experts were selected. It
#                   catches a missing renormalization (the top-2 of a 4-way
#                   softmax sums to about 0.7, rel_l2 about 0.3), a wrong k,
#                   and a stray scaling factor (route_scale=2 doubles it).
#   selected_count  exactly top_k for every token, and an integer fp32
#                   represents exactly -- so it takes a BITWISE gate. It
#                   catches token dropping and a degenerate routing map.
#
# and the TIE-SENSITIVE outputs -- probs, routing_map, x_grad and
# gate_weight_grad -- are enforced only between sides that select on the same
# precision, and recorded rather than enforced across a precision boundary.
#
# **What one flipped token costs, so a first GPU failure is legible.** At the
# default workload (8192 tokens, 4 experts, top_k 2, dim 1024) a single token
# routed differently moves rel_l2 by roughly 1e-2 on ``probs`` and roughly
# 2e-2 on ``x_grad``, and breaks ``routing_map`` outright. rel_l2 grows as the
# square root of the flip count, so the 2e-2 gate tolerates a handful of flips
# on ``probs`` and about ONE on ``x_grad``. Both figures are
# order-of-magnitude estimates from the tensor norms, not measurements: read
# them as "one flip reaches the gate limit", not a threshold. Nobody ran it.
#
# **The informational routing_map row is the instrument that tells a reader
# which failure they are looking at**, and it is why it is declared rather
# than dropped. If ``x_grad`` fails and ``routing_map`` is bitwise equal, the
# two sides selected identically and the disagreement is arithmetic -- a real
# fault. If ``routing_map`` differs too, the disagreement is a tie. Read that
# row before touching a threshold. Widening a gate to absorb a tie would also
# absorb the fault the gate exists for.
#
# rel_l2 throughout, and no max or ULP metric anywhere. A router is a reduction
# over dim and a softmax over the experts, and both drive individual outputs
# toward zero, which is exactly the cancellation case CLAUDE.md's rule names.
MOE_ROUTER_TIE_IMMUNE_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("logits", "prob_row_sums"),
    max_rel_l2=2e-2,
)

# Bitwise, and safe to enforce across every precision boundary in the
# scenario: torch.topk returns distinct indices, so this row is the integer
# top_k for every token however the selection came out, and fp32 holds a small
# integer exactly. ``moe_router_reference`` returns it as fp32 rather than
# fp64 for that reason -- torch.equal compares dtypes, and an fp64 reference
# row would fail against every arm's fp32 row for a reason no reader could
# recover.
MOE_ROUTER_SELECTED_COUNT_GATE = CorrectnessCheck(
    kind="bitwise",
    reference="fp64",
    outputs=("selected_count",),
)

# The tie-sensitive outputs, enforced. The fp64 reference selects on fp64
# logits and these three arms select on fp32 ones, so they are the same
# selection except on a tie fp32 cannot resolve. The gradients are named here
# and not folded into the gate above because they cover the BACKWARD, which is
# a separate implementation on each engine: a forward that agrees is no
# evidence about it, and a gradient that arrives scaled or masked is visible
# nowhere else.
MOE_ROUTER_FP32_SELECTION_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("probs", "x_grad", "gate_weight_grad"),
    max_rel_l2=2e-2,
)

# The same three outputs for the one arm that selects on bf16. Recorded, never
# enforced: bf16 top-k may legitimately pick a different expert than an fp64
# reference, and that difference IS this arm's measurement. Enforcing it would
# fail the arm for doing the thing its name says it does.
MOE_ROUTER_BF16_SELECTION_RECORD = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("probs", "x_grad", "gate_weight_grad"),
    max_rel_l2=2e-2,
    informational=True,
)

# Never enforced anywhere, on any arm. See the routing_map paragraph above:
# this is the diagnostic that separates a tie from a fault, and a diagnostic
# that can fail the run is not one.
MOE_ROUTER_ROUTING_MAP_RECORD = CorrectnessCheck(
    kind="bitwise",
    reference="fp64",
    outputs=("routing_map",),
    informational=True,
)


MOE_ROUTER = KernelScenario(
    name="moe_router",
    description=(
        "The MoE router, cross-engine: TorchTitan's TokenChoiceTopKRouter "
        "against megatron-core's MoELayer.route/TopKRouter, over one shared "
        "gate matrix. The cut starts at the hidden state the MoE block "
        "receives and ends at the routing decision. It stops there: the "
        "one-hot routing map and the per-expert counts titan builds after the "
        "router returns belong to scenario 10, and MoELayer.preprocess is "
        "scenario 10 on the megatron side. "
        "BOTH ENGINES ROUTE AT THE SAME PRECISION -- titan wraps its gate in "
        "torch.autocast(float32) and mcore/base sets moe_router_dtype='fp32' "
        "-- so the plan's caption that this row is a precision difference is "
        "WRONG in that direction. It is wrong in the other direction too: the "
        "two GEMMs differ in their OPERAND precision. Titan's autocast casts "
        "both operands UP, so it materializes a full fp32 copy of the hidden "
        "state and runs an fp32 GEMM; megatron hands TE the bf16 operands and "
        "asks only for an fp32 output. Titan therefore moves about 5x the "
        "bytes (40 MiB against 8 at the default workload) in a scenario whose "
        "device work is one bandwidth-bound read. THE CROSS-ENGINE ROW IS "
        "THEREFORE NOT A VERDICT ON TITAN'S ROUTER KERNEL, and the arm that "
        "would separate the two -- a titan arm with the autocast removed -- is "
        "NOT DECLARED. Each arm carries its own bytes_moved so the GB/s and "
        "x_floor columns show the asymmetry instead of absorbing it. "
        "mcore/router_bf16 publishes against mcore/base, never against titan, "
        "and its row is a PRECISION result and not a speed one: the delta is "
        "the dtype of the [T, E] outputs, about 0.4% of the arm's traffic, "
        "which is an order of magnitude below this repo's measured noise "
        "floor. The titan arm is compiled (fullgraph=True) and the three "
        "megatron arms are eager, which is what each engine does end to end, "
        "so every cross-engine row is also a comparison of two compile "
        "treatments. The scenario is expected to be DISPATCH-BOUND: a "
        "megatron forward reads 8 MiB and computes 33.5 MFLOP at the default "
        "workload, so copy_floor says how far above the bus each arm sits, "
        "and the --burst residual must be read before anything here is ranked "
        "as a kernel."
    ),
    inputs_builder=(
        "benchmarks.kernel.operations.moe_router:moe_router_inputs"
    ),
    reference_builder=(
        "benchmarks.kernel.operations.moe_router:moe_router_reference"
    ),
    baseline_arm="mcore/base",
    # Explicit, and exhaustive. It is the same set the default derivation
    # would produce, and writing it down is what makes the direction of every
    # published row a declaration rather than an accident of arm order. Note
    # what is deliberately ABSENT: there is no ("titan", "mcore/router_bf16")
    # row. Titan routes in fp32, so mcore/base is already its like-for-like
    # opponent, and a titan-against-bf16 row would move the engine and the
    # precision together.
    comparisons=(
        ("titan", "mcore/base"),
        ("mcore/router_fusion", "mcore/base"),
        ("mcore/router_bf16", "mcore/base"),
    ),
    # NOT requires_balanced_routing, and that is a decision. The flag exists
    # so swiglu_inputs can hand every expert an equal slice of synthetic rows.
    # This scenario materializes no per-expert tensor and hands no expert a
    # slice: the router computes the split itself, and its output is [T, E]
    # whatever the split turns out to be. Declaring the flag would refuse
    # workloads this scenario measures correctly.
    arms=(
        KernelArm(
            name="copy_floor",
            description=(
                "One read of the [B, L, D] hidden state and one write: the "
                "bandwidth reference the four router arms are read against"
            ),
            builder=(
                "benchmarks.kernel.operations.moe_router:"
                "build_moe_router_copy_floor"
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
                "megatron-core MoELayer.route off a real GPTModel: TopKRouter "
                "with moe_router_dtype='fp32', unfused, eager, as our "
                "megatron arm runs it"
            ),
            builder=(
                "benchmarks.kernel.operations.moe_router:"
                "build_moe_router_mcore_base"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer, so the router "
                "runs eager as megatron runs it. One exception is stated "
                "rather than hidden: TopKRouter._apply_expert_bias is "
                "@jit_fuser decorated and megatron binds jit_fuser to "
                "torch.compile at import, so one compiled region runs per "
                "call. At moe_router_enable_expert_bias=False its body does "
                "nothing, so what remains is Dynamo's per-call guard "
                "evaluation -- host cost, in a host-bound scenario"
            ),
            correctness=(
                MOE_ROUTER_TIE_IMMUNE_GATE,
                MOE_ROUTER_SELECTED_COUNT_GATE,
                MOE_ROUTER_FP32_SELECTION_GATE,
                MOE_ROUTER_ROUTING_MAP_RECORD,
            ),
        ),
        KernelArm(
            name="mcore/router_fusion",
            description=(
                "The same router with moe_router_fusion=True: "
                "TransformerEngine's fused top-k score-function kernel in "
                "place of megatron's torch sequence. One substitution -- the "
                "flag's second read site is inside an is_aux_loss_enabled() "
                "branch the base profile leaves False"
            ),
            builder=(
                "benchmarks.kernel.operations.moe_router:"
                "build_moe_router_mcore_router_fusion"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "the same treatment as mcore/base, and it must be: this row "
                "is a within-engine fusion delta, so a different compile "
                "treatment on one side would be measured as the fusion"
            ),
            correctness=(
                MOE_ROUTER_TIE_IMMUNE_GATE,
                MOE_ROUTER_SELECTED_COUNT_GATE,
                MOE_ROUTER_FP32_SELECTION_GATE,
                MOE_ROUTER_ROUTING_MAP_RECORD,
                # Against the anchor, and enforced on the tie-sensitive
                # outputs too: both sides select on fp32 logits produced by
                # the same gate GEMM, so this is the check that says the fused
                # kernel computes megatron's own routing and not another one.
                # It is the only gate that can see a fused kernel that is
                # numerically valid and wrong -- the fp64 gates above would
                # pass a fusion that quietly changed the score function, as
                # long as it stayed a valid routing.
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=(
                        "logits",
                        "probs",
                        "prob_row_sums",
                        "x_grad",
                        "gate_weight_grad",
                    ),
                    max_rel_l2=2e-2,
                ),
                CorrectnessCheck(
                    kind="bitwise",
                    reference="mcore/base",
                    outputs=("selected_count",),
                ),
                CorrectnessCheck(
                    kind="bitwise",
                    reference="mcore/base",
                    outputs=("routing_map",),
                    informational=True,
                ),
            ),
        ),
        KernelArm(
            name="mcore/router_bf16",
            description=(
                "The same router with moe_router_dtype=None. THE DELTA IS "
                "BF16, NOT FP32: the base profile already sets fp32, so this "
                "arm removes the request rather than adding it. The gate GEMM "
                "writes bf16 logits and the top-k selects on them, so this "
                "arm may legitimately route a token to a different expert "
                "than any fp32 arm. Read its row as a PRECISION result only: "
                "the delta is the dtype of the [T, E] outputs, about 32 KiB "
                "of the arm's ~8.4 MiB, so no speed difference it reports can "
                "be separated from noise"
            ),
            builder=(
                "benchmarks.kernel.operations.moe_router:"
                "build_moe_router_mcore_router_bf16"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "the same treatment as mcore/base, and it must be: this row "
                "is a within-engine precision delta, so a different compile "
                "treatment on one side would be measured as the precision"
            ),
            correctness=(
                MOE_ROUTER_TIE_IMMUNE_GATE,
                MOE_ROUTER_SELECTED_COUNT_GATE,
                # Recorded, not enforced. This is the one arm whose selection
                # precision differs from the reference's.
                MOE_ROUTER_BF16_SELECTION_RECORD,
                MOE_ROUTER_ROUTING_MAP_RECORD,
                # Against the anchor, tie-immune outputs only. logits belongs
                # here and is the row that says the delta is precision and
                # nothing else: bf16 logits sit about 4e-3 from fp32 ones,
                # comfortably inside 2e-2, so a failure means the arm changed
                # more than the dtype.
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=("logits", "prob_row_sums"),
                    max_rel_l2=2e-2,
                ),
                CorrectnessCheck(
                    kind="bitwise",
                    reference="mcore/base",
                    outputs=("selected_count",),
                ),
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=("probs", "x_grad", "gate_weight_grad"),
                    max_rel_l2=2e-2,
                    informational=True,
                ),
                CorrectnessCheck(
                    kind="bitwise",
                    reference="mcore/base",
                    outputs=("routing_map",),
                    informational=True,
                ),
            ),
        ),
        KernelArm(
            name="titan",
            description=(
                "TorchTitan TokenChoiceTopKRouter from the production MoE "
                "config node, under torch.compile(fullgraph=True). It takes "
                "the softmax over all E experts, selects the top k and "
                "renormalizes; megatron selects first and takes the softmax "
                "over k. Softmax is monotone and the renormalization cancels "
                "the shared denominator, so the two compute one function. The "
                "gate GEMM runs under torch.autocast(float32), which is why "
                "this arm is like for like with mcore/base on precision"
            ),
            builder=(
                "benchmarks.kernel.operations.moe_router:"
                "build_moe_router_titan"
            ),
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(
                MOE_ROUTER_TIE_IMMUNE_GATE,
                MOE_ROUTER_SELECTED_COUNT_GATE,
                MOE_ROUTER_FP32_SELECTION_GATE,
                MOE_ROUTER_ROUTING_MAP_RECORD,
                # The cross-engine gates, and they are what make the published
                # row a comparison of two implementations of one function
                # rather than of two functions. Both sides select on fp32
                # logits from the same bf16 gate, so the tie-sensitive outputs
                # are enforced here too.
                #
                # Every one of them sits on the NON-ANCHOR arm, and none of
                # them points outward from mcore/base. resolve_arm_skips
                # closes the skip set over correctness references, so a check
                # declared on the anchor and pointing at titan would add the
                # anchor to the skip set whenever the titan arm is skipped.
                # The scenario would then lose every row.
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=(
                        "logits",
                        "probs",
                        "prob_row_sums",
                        "x_grad",
                        "gate_weight_grad",
                    ),
                    max_rel_l2=2e-2,
                ),
                CorrectnessCheck(
                    kind="bitwise",
                    reference="mcore/base",
                    outputs=("selected_count",),
                ),
                CorrectnessCheck(
                    kind="bitwise",
                    reference="mcore/base",
                    outputs=("routing_map",),
                    informational=True,
                ),
            ),
        ),
    ),
)


# The permutation gates are BITWISE, and that is a decision about what a
# permutation is rather than a claim about how good these kernels are.
#
# A permute moves bits and computes nothing, so both engines produce the
# permuted buffers by a pure copy -- an index_select on the torch path, a TE
# kernel on the fused one -- and bitwise equality is achievable rather than
# strict. It is also the only metric that can police the failure. A norm-based
# gate sees a permutation only at a small enough workload: with N output rows,
# swapping one pair gives ||a - b|| / ||b|| ~ 2 / sqrt(N). That is 2.2e-2 at
# the default workload, just ABOVE the 2e-2 gate every neighbouring scenario
# uses -- and it FALLS as the workload grows, reaching 1.6e-2 at batch 8. A
# tolerance gate would therefore catch a misplaced row at one batch size and
# miss it at the next, and the row it misplaces feeds the expert GEMM in
# scenario 11. A bitwise gate does not depend on the workload.
DISPATCH_PERMUTE_PERMUTATION_GATE = CorrectnessCheck(
    kind="bitwise",
    reference="fp64",
    outputs=("permuted_tokens", "permuted_probs", "tokens_per_expert"),
)

# The gradient gates are NOT the permutation gate, and the split is
# deliberate. The gradient of a gather is a scatter-add: x_grad accumulates
# top_k output-gradient rows per token in bf16 and rounds, so it cannot be
# bitwise against an fp64 truth. probs_grad is a pure scatter of distinct
# values and rounds nothing, but it is gated the same way, because a gate that
# enforced bitwise equality on one gradient and not the other would assert
# something about the implementation that this scenario has not checked.
DISPATCH_PERMUTE_GRADIENT_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("x_grad", "probs_grad"),
    max_rel_l2=2e-2,
)

# The cross-engine permutation gate. It is what makes the published ratio a
# comparison of two implementations of one function rather than of two
# different functions, and it is bitwise for the reason above.
#
# The two engines reach the same permutation by constructions that look
# nothing alike. TorchTitan sorts the flattened [T, K] expert-id tensor with a
# stable ASCENDING argsort (torchtitan/models/common/token_dispatcher.py:
# 93-95). Megatron transposes the [T, E] routing map, flattens it, and sorts
# with a stable DESCENDING one (megatron/core/transformer/moe/moe_utils.py:
# 468-475). Both come out expert-major and, within one expert, in ascending
# token order -- a token reaches a given expert at most once, so the flat-slot
# order is the token order. tests/test_kernel_dispatch_permute.py transcribes
# both from the pinned sources and checks them against a third derivation that
# is neither engine's.
#
# It sits on the TITAN arm and references mcore/base, never the reverse.
# resolve_arm_skips closes the skip set over correctness references, so a
# check pointing from the anchor at titan would let a skipped titan arm take
# the anchor down with it -- and losing the anchor costs the whole scenario.
DISPATCH_PERMUTE_CROSS_ENGINE_PERMUTATION_GATE = CorrectnessCheck(
    kind="bitwise",
    reference="mcore/base",
    outputs=("permuted_tokens", "permuted_probs", "tokens_per_expert"),
)

# The cross-engine gradient gate. Tolerance rather than bitwise, because the
# two engines may accumulate the scatter-add in a different order and bf16
# addition is not associative.
DISPATCH_PERMUTE_CROSS_ENGINE_GRADIENT_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="mcore/base",
    outputs=("x_grad", "probs_grad"),
    max_rel_l2=2e-2,
)


DISPATCH_PERMUTE = KernelScenario(
    name="dispatch_permute",
    description=(
        "The step that turns a routing decision into an expert-major token "
        "buffer: TorchTitan's routing-map and counts construction plus "
        "token_dispatcher.dispatch, against megatron-core's MoELayer."
        "preprocess plus ALL THREE dispatch phases (dispatch_preprocess, "
        "token_dispatch, dispatch_postprocess). The triple is the cut because "
        "MoEAllGatherTokenDispatcher.token_dispatch is guarded by 'tp_size > "
        "1 or ep_size > 1' (token_dispatcher.py:282) and is therefore the "
        "IDENTITY at world_size=1: a cut on MoELayer.dispatch alone would "
        "measure a zero on the megatron side and would read as a spectacular "
        "win. permute() runs in dispatch_postprocess. Every arm carries a "
        "guard that RAISES unless the measured call really permuted. "
        "THIS SCENARIO'S NUMBER IS DEVICE TIME PLUS HOST SERIALIZATION, and "
        "only on the megatron side: dispatch_postprocess runs a blocking "
        ".cpu() on the per-expert counts every call (token_dispatcher.py:317) "
        "and TorchTitan has no counterpart, so a burst cannot pipeline "
        "through it and per-call time stops falling with --burst-k on one "
        "engine alone. The cost is real and megatron pays it at every layer "
        "of every step, but a reader who takes the ratio for a kernel-speed "
        "comparison has read it wrong. The titan arm is compiled "
        "(fullgraph=True) and every mcore arm is eager, which is what each "
        "engine does end to end, so the cross-engine row compares two compile "
        "treatments as well as two implementations. One boundary is "
        "asymmetric and is charged to TorchTitan: megatron's router returns "
        "the one-hot map with the probabilities, so the map is built inside "
        "scenario 9, while titan builds it here -- two to three extra kernel "
        "launches on the titan side in a scenario that is dispatch-bound. "
        "Read scenario 9's row beside this one. copy_floor is the bandwidth "
        "reference: a permute is a gather, so if the arms sit near the floor "
        "the ratios compare how close four implementations get to the memory "
        "bus and no ratio here is a kernel-quality claim. Two further columns "
        "are not like-for-like, and no arm can be charged for either. "
        "peak_memory_gib: megatron's dispatchers keep a call's intermediates "
        "on the instance (self.local_map, self.local_probs, "
        "self.reversed_local_input_permutation_mapping) where TorchTitan's "
        "returns a frozen LocalDispatchMetadata and stores nothing, so in "
        "forward mode the megatron arms hold state the titan arm does not. "
        "x_floor: the floor copies a contiguous buffer and does no gather, so "
        "it understates the device work every arm does and overstates every "
        "arm's distance from the bus -- and on the megatron arms that "
        "distance is dominated by the host sync above rather than by device "
        "inefficiency."
    ),
    inputs_builder=(
        "benchmarks.kernel.operations.dispatch_permute:dispatch_permute_inputs"
    ),
    reference_builder=(
        "benchmarks.kernel.operations.dispatch_permute:"
        "dispatch_permute_reference"
    ),
    baseline_arm="mcore/base",
    # The synthetic routing decision hands every expert an equal slice of
    # batch * seq_len * top_k. An uneven split must skip this scenario
    # loudly -- named numbers, a recorded error, a nonzero exit -- rather than
    # be capped or rounded, exactly as swiglu already does. See merge note 2:
    # this flag is what breaks test_only_swiglu_needs_balanced_routing, and
    # widening that test is the fix.
    requires_balanced_routing=True,
    # Explicit and exhaustive. It is the same set the schema would derive, and
    # it is written out anyway: a cross-engine scenario states which rows it
    # publishes rather than inheriting them, and each row below answers a
    # different question.
    comparisons=(
        # The scenario's reason to exist, and the row Rule 5 governs. Both
        # sides do the same work -- build the expert-major buffer -- and the
        # megatron side additionally blocks on a device-to-host copy that the
        # titan side does not have.
        ("titan", "mcore/base"),
        # Within megatron: what TransformerEngine's permutation fusion is
        # worth. It is a PART of this scenario and not the whole of it. Under
        # the allgather dispatcher the cut holds exactly one read site of
        # moe_permute_fusion -- permute() in dispatch_postprocess
        # (token_dispatcher.py:324) -- beside MoELayer.preprocess, the
        # local_map and local_probs slices, the .cpu() at :317 and the by-hand
        # probability permutation at :328-330. So the row is diluted by
        # everything the flag does not touch and UNDERSTATES the fusion.
        ("mcore/no_permute_fusion", "mcore/base"),
        # Within megatron: the dispatcher's local permute and sync strategy.
        # NOT communication -- see the arm's own description.
        ("mcore/dispatcher_alltoall", "mcore/base"),
    ),
    arms=(
        KernelArm(
            name="copy_floor",
            description=(
                "One read and one write of the permuted [N, D] buffer: the "
                "bandwidth floor for the forward traffic at this shape. It is "
                "a LOWER bound and the direction of the bias is known -- the "
                "floor reads a contiguous buffer where the arms gather N rows "
                "scattered through a [T, D] one, so the floor understates the "
                "device work and the x_floor column overstates the distance "
                "between an arm and the device"
            ),
            builder=(
                "benchmarks.kernel.operations.dispatch_permute:"
                "build_dispatch_permute_copy_floor"
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
                "megatron-core MoELayer.preprocess plus all three dispatch "
                "phases, off a real GPTModel: the allgather token dispatcher "
                "with TransformerEngine's permutation fusion on, which is "
                "what the e2e megatron arm runs. Eager. Its number includes a "
                "blocking device-to-host copy of the per-expert counts "
                "(token_dispatcher.py:317), which runs on every call"
            ),
            builder=(
                "benchmarks.kernel.operations.dispatch_permute:"
                "build_dispatch_permute_mcore_base"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer, and no method "
                "this closure calls carries @jit_fuser: token_dispatcher.py "
                "holds exactly one such decorator (:1860), on "
                "MoEFlexTokenDispatcher.dispatch_preprocess, and that class "
                "cannot be built at world_size=1 (:1775,1793 assert "
                "tp_size * ep_size > 1). So this arm is eager all the way "
                "down, and compiling it would measure a treatment megatron "
                "never applies"
            ),
            correctness=(
                DISPATCH_PERMUTE_PERMUTATION_GATE,
                DISPATCH_PERMUTE_GRADIENT_GATE,
            ),
        ),
        KernelArm(
            name="mcore/no_permute_fusion",
            description=(
                "megatron with moe_permute_fusion=False: permute() falls from "
                "TransformerEngine's fused kernel (moe_utils.py:404-410) to "
                "the torch path (:461-486) -- a transpose plus contiguous, a "
                "stable descending argsort, a slice, a modulo and an "
                "index_select. Same permutation, different implementation. "
                "The flag has ONE read site inside this cut, so the row "
                "against the anchor is diluted by preprocess, the map slices "
                "and the device-to-host copy, and it understates what the "
                "fusion is worth to the permute itself"
            ),
            builder=(
                "benchmarks.kernel.operations.dispatch_permute:"
                "build_dispatch_permute_mcore_no_permute_fusion"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "the same reason as mcore/base: megatron compiles no whole "
                "transformer layer, and the torch permute path this arm takes "
                "carries no @jit_fuser either"
            ),
            correctness=(
                DISPATCH_PERMUTE_PERMUTATION_GATE,
                DISPATCH_PERMUTE_GRADIENT_GATE,
            ),
        ),
        KernelArm(
            name="mcore/dispatcher_alltoall",
            description=(
                "megatron with the alltoall token dispatcher instead of the "
                "allgather one. THE COLLECTIVE IS INERT AT world_size=1 AND "
                "THE CLASS IS NOT: _AllToAll.forward returns its input "
                "unchanged at one rank (tensor_parallel/mappings.py:433-435), "
                "so this arm measures the dispatcher's LOCAL permute and sync "
                "strategy and NEVER communication. Three differences, all "
                "local: it permutes in dispatch_preprocess "
                "(token_dispatcher.py:655-677) rather than in "
                "dispatch_postprocess; it passes probs=probs into permute "
                "(:671) so ONE fused kernel permutes tokens and probabilities "
                "together, where the allgather dispatcher permutes the "
                "probabilities by hand afterwards with "
                ".T.contiguous().masked_select(...) (:328-330); and it issues "
                "its device-to-host copies on a side stream at "
                "cuda_dtoh_point and waits at cuda_sync_point (:918-955), "
                "where the allgather dispatcher's .cpu() at :317 is "
                "unconditional and blocking. One extra piece of local work "
                "has no counterpart on the anchor and may DOMINATE this row: "
                "with num_local_experts > 1, dispatch_postprocess calls "
                "sort_chunks_by_idxs (:778-785), and at one rank the index "
                "vector is the identity, so the rows come out in the anchor's "
                "order while a full [T*K, D] read and write -- about 16 MiB "
                "each way at the normal shape -- is still done and still "
                "costs. Read this row as a property of running the alltoall "
                "dispatcher at one rank, not as a dispatcher design verdict"
            ),
            builder=(
                "benchmarks.kernel.operations.dispatch_permute:"
                "build_dispatch_permute_mcore_dispatcher_alltoall"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "the same reason as mcore/base: megatron compiles no whole "
                "transformer layer, and neither dispatcher class carries a "
                "@jit_fuser method this closure reaches"
            ),
            correctness=(
                DISPATCH_PERMUTE_PERMUTATION_GATE,
                DISPATCH_PERMUTE_GRADIENT_GATE,
            ),
        ),
        KernelArm(
            name="titan",
            description=(
                "TorchTitan's routing-map and per-expert-counts construction "
                "(moe.py:465-470) plus token_dispatcher.dispatch "
                "(moe.py:144-157), under torch.compile(fullgraph=True), which "
                "is the production treatment -- the whole of MoE.forward sits "
                "inside the per-block compile apply_compile wraps around each "
                "Qwen3TransformerBlock. The dispatcher is the production "
                "AllToAllTokenDispatcher with ep_mesh=None, which takes the "
                "LocalTokenDispatcher branch by an explicit test "
                "(token_dispatcher.py:415-421) rather than by accident. This "
                "arm is charged the routing-map construction that megatron "
                "does inside scenario 9"
            ),
            builder=(
                "benchmarks.kernel.operations.dispatch_permute:"
                "build_dispatch_permute_titan"
            ),
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(
                DISPATCH_PERMUTE_PERMUTATION_GATE,
                DISPATCH_PERMUTE_GRADIENT_GATE,
                DISPATCH_PERMUTE_CROSS_ENGINE_PERMUTATION_GATE,
                DISPATCH_PERMUTE_CROSS_ENGINE_GRADIENT_GATE,
            ),
        ),
    ),
)


# The fp64 gate every TorchTitan arm carries. The four titan arms compute the
# expert MLP WITHOUT the routing probabilities, because titan's dispatcher
# applies them in combine, so they are gated on the reference's unweighted
# names.
#
# Measured margin, on CPU, over a batched stand-in at the real ``normal``
# geometry (dim 1024, expert width 3584, 4 experts, 128 routed rows), against
# this module's own fp64 truth: out 3.90e-3, x_grad 4.22e-3, w1_grad 3.76e-3,
# w2_grad 3.88e-3, w3_grad 3.90e-3. The gate sits at 2e-2. The worst output is
# x_grad, so the real headroom is **4.7x**, not the 5x a single 3.9e-3 figure
# suggests; quote the range rather than one number. **The device kernels have
# never been run**, so that margin is evidence about the arithmetic of a
# ``bmm`` stand-in, and not about ``torch._grouped_mm``, TE's grouped GEMM or
# TE's SwiGLU.
EXPERT_MLP_TITAN_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("out", "x_grad", "w1_grad", "w2_grad", "w3_grad"),
    max_rel_l2=2e-2,
)

# The fp64 gate every megatron-core arm carries, over the reference's OTHER
# branch. The distinct names are a guard and not a convention:
# ``benchmarks/kernel/engine/correctness.py:43-49`` raises when an arm does not
# produce an output a check names, so a gate that pointed one engine at the
# other's truth would fail loudly rather than compare two functions.
#
# ``probs_grad_weighted`` is the sharpest of the six. It is identically zero
# for any implementation that dropped the routing probabilities, which is the
# one thing that distinguishes megatron's semantics at this cut, and it is the
# quantity two of the four megatron arms compute with an extra reduction that
# the fused arm folds away.
EXPERT_MLP_MCORE_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=(
        "out_weighted",
        "x_grad_weighted",
        "probs_grad_weighted",
        "w1_grad_weighted",
        "w2_grad_weighted",
        "w3_grad_weighted",
    ),
    max_rel_l2=2e-2,
)

# The two output rosters, named once so the arm-to-arm checks below cannot
# drift from the fp64 gates above.
#
# Those checks each reference that row's OPPONENT. Direction is not a style
# choice: ``resolve_arm_skips`` (``benchmarks/kernel/runner.py:281-319``)
# closes the skip set over correctness references, so a check pointing outward
# from an arm that another row depends on would take that row down with it.
# Referencing the opponent makes the closure follow the same edges the
# comparisons do.
#
# They are tighter than the fp64 hub, and that is why they exist beside it.
# The hub bounds each arm at 2e-2 and therefore bounds a *pair* only
# transitively, at 4e-2 -- and the pair is exactly what each published row is a
# ratio of. ``qkv_prep`` makes the same argument for its own arm-to-arm check.
EXPERT_MLP_TITAN_OUTPUTS = ("out", "x_grad", "w1_grad", "w2_grad", "w3_grad")
EXPERT_MLP_MCORE_OUTPUTS = (
    "out_weighted",
    "x_grad_weighted",
    "probs_grad_weighted",
    "w1_grad_weighted",
    "w2_grad_weighted",
    "w3_grad_weighted",
)


EXPERT_MLP = KernelScenario(
    name="expert_mlp",
    description=(
        "The routed-expert MLP itself: TorchTitan's inner_experts "
        "(GroupedExperts) against megatron-core's experts call "
        "(TEGroupedMLP), plus three megatron variants and three TorchTitan "
        "ones. THIS SCENARIO PUBLISHES NO CROSS-ENGINE RATIO, AND NO READER "
        "MAY FORM ONE BY DIVIDING TWO MEDIANS. The routing probabilities are "
        "applied on opposite sides of this boundary -- megatron folds them "
        "into the fused activation kernel inside the experts "
        "(weighted_bias_swiglu_impl), while TorchTitan applies them in "
        "combine, one scenario later -- so megatron's side of the cut does "
        "strictly more work and the two engines compute two functions here. "
        "Every published row therefore has both arms on one engine, the "
        "correctness output names differ per engine so no gate can cross the "
        "boundary either, and the cross-engine row belongs to the "
        "expert_combine span over scenarios 11 and 12, the smallest "
        "enclosure in which both engines have applied the probabilities "
        "exactly once. THE PIPER ARMS ARE MEASURED AGAINST TORCHTITAN'S OWN "
        "W13 FUSION, not against unfused experts: FusedGroupedExperts is "
        "upstream's fusion of the gate and up projections, with the same "
        "(E, F, 2, D) parameter, the same single grouped GEMM and the same "
        "save/load split hooks, so what the two Piper arms buy is the "
        "combined activation layout and nothing more. THE "
        "titan/fused_grouped_experts vs titan ROW MOVES TWO AXES AT ONCE AND "
        "IS NOT A MEASUREMENT OF THE W13 FUSION ALONE: FusedGroupedExperts "
        "both fuses the two grouped GEMMs into one AND replaces the anchor's "
        "plain-ops SwiGLU with the silu_and_mul custom op, which is opaque to "
        "Inductor under fullgraph=True where the anchor's plain ops fuse into "
        "their neighbours. The two changes push in OPPOSITE directions, so a "
        "ratio near 1.0 on that row may be a real fusion gain cancelled by a "
        "lost activation fusion rather than a fusion that bought nothing. "
        "Read it as upstream's fused expert layer against upstream's unfused "
        "one, and attribute nothing in it to the GEMM count. The other two "
        "titan rows move one axis each. THE NUMBER IS DEVICE "
        "TIME ON BOTH SIDES: megatron's tokens_per_expert.tolist() would be a "
        "blocking sync on a device tensor, but its allgather dispatcher has "
        "already moved that tensor to the host, so the inputs builder hands "
        "megatron a CPU count tensor and titan a device one, as each engine "
        "receives in production. Neither engine pays a layout conversion "
        "here; both consume the same (rows, dim) permuted batch. All four "
        "megatron arms run EAGER, because megatron compiles no whole "
        "transformer layer, and all four TorchTitan arms run under "
        "torch.compile(fullgraph=True), because that is what they face end "
        "to end -- but no published row crosses that difference. This "
        "scenario SUPERSEDES the swiglu scenario, whose three arms are titan, "
        "titan/piper_optimized_triton and titan/piper_optimized_inductor "
        "here, built the same way from the same shared weights. THE PRINTED "
        "MEDIANS ARE THE ONLY CHANNEL BY WHICH THE FORBIDDEN RATIO CAN BE "
        "FORMED, AND THAT IS WHY THE SENTENCE ABOVE IS SHOUTED: this scenario "
        "declares no floor and no arm declares bytes_moved, so no arm here "
        "carries an x_floor or a GB/s figure at all -- both headings are "
        "printed unconditionally and every row under them reads n/a. Neither "
        "of the two "
        "derived columns its sibling within-engine-only scenarios must "
        "caption holds a number here to reproduce the ratio a second way. "
        "Suppressing "
        "comparisons removes the ROW and not the NUMBER, in every one of them."
    ),
    inputs_builder=(
        "benchmarks.kernel.operations.expert_mlp:expert_mlp_inputs"
    ),
    reference_builder=(
        "benchmarks.kernel.operations.expert_mlp:expert_mlp_reference"
    ),
    # The anchor. See merge note 1: this is the one cross-engine scenario that
    # anchors on the titan side, because it is the one that publishes no
    # cross-engine row, so the anchor's only remaining job is to decide whose
    # loss costs the scenario -- and GroupedExperts has neither a megatron nor
    # a TransformerEngine dependency to lose.
    baseline_arm="titan",
    # The synthetic rows are split evenly across the experts, exactly as
    # swiglu_inputs splits them, so a workload where batch * seq_len * top_k
    # does not divide by num_experts must fail loudly rather than be capped or
    # rounded. The inputs builder re-asserts it too, with both numbers named,
    # for callers that reach it without passing through the runner.
    requires_balanced_routing=True,
    # Explicit, exhaustive, and deliberately NOT the derived set. The
    # derivation would pair every non-anchor arm with ``titan``. All FOUR
    # mcore arms are non-anchor and none is a floor, so it would publish four
    # cross-engine rows this scenario has evidence against -- mcore/base,
    # mcore/no_bias_activation_fusion, mcore/te_activation_func and
    # mcore/no_grouped_gemm, each against titan. It would also credit both
    # Piper arms with a w13 fusion TorchTitan already ships. The count is four
    # and not three: it was checked by constructing the scenario with
    # comparisons=None and reading comparison_pairs() back.
    #
    # Six rows, and every one of them within one engine:
    #
    #   * three megatron deltas against the megatron base, which is the
    #     within-megatron fusion and implementation question; and
    #   * TorchTitan's own w13 fusion against unfused experts, then each Piper
    #     layout against that fusion, which is what each of them modified.
    #
    # The row swiglu published -- a Piper arm against unfused experts -- is
    # deliberately absent. It is recoverable from the per-replicate samples
    # results.json keeps, and the fused-against-unfused row above supplies the
    # factor the two differ by. It is not declared because the printed table
    # keys its comparison rows by (arm, mode)
    # (``benchmarks/kernel/results/reporting.py:113-114``), so a second row for
    # one arm in one mode would be written to results.json and then silently
    # dropped from the table.
    comparisons=(
        ("mcore/no_bias_activation_fusion", "mcore/base"),
        ("mcore/te_activation_func", "mcore/base"),
        ("mcore/no_grouped_gemm", "mcore/base"),
        ("titan/fused_grouped_experts", "titan"),
        ("titan/piper_optimized_triton", "titan/fused_grouped_experts"),
        ("titan/piper_optimized_inductor", "titan/fused_grouped_experts"),
    ),
    arms=(
        # ---- megatron-core -------------------------------------------------
        #
        # Every megatron arm declares ("forward", "forward_backward") and no
        # isolated backward. The retained-graph trick re-runs one backward
        # graph many times, and TransformerEngine's GroupedLinear backward
        # calls clear_tensor_data on its saved inputs
        # (transformer_engine/pytorch/module/grouped_linear.py:1129), so a
        # second pass would read cleared storage; TE's SwiGLU operation clears
        # its saved tensors too. Backward cost stays recoverable as
        # forward_backward minus forward.
        KernelArm(
            name="mcore/base",
            description=(
                "megatron-core TEGroupedMLP off a real GPTModel: one TE "
                "grouped GEMM for the doubled fc1, weighted_bias_swiglu_impl "
                "for the activation -- which folds the routing probabilities "
                "into the same kernel -- and one TE grouped GEMM for fc2. "
                "Eager, four local experts, no expert parallelism"
            ),
            builder=(
                "benchmarks.kernel.operations.expert_mlp"
                ":build_expert_mlp_mcore_base"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer, so every TE "
                "module it builds runs eager end to end; compiling this one "
                "would measure a treatment megatron never applies"
            ),
            correctness=(EXPERT_MLP_MCORE_GATE,),
        ),
        KernelArm(
            name="mcore/no_bias_activation_fusion",
            description=(
                "the same layer with bias_activation_fusion=False: the "
                "activation falls to megatron's chunk/silu/mul path plus a "
                "SEPARATE probability multiply with a dtype round trip, so "
                "four kernels and an extra full-width intermediate where the "
                "base has one kernel. This is the dataclass-default handicap "
                "megatron's own argparse layer would have turned on, and it "
                "once cost 11.9 GPU ms/step end to end"
            ),
            builder=(
                "benchmarks.kernel.operations.expert_mlp"
                ":build_expert_mlp_mcore_no_bias_activation_fusion"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer; and this arm "
                "IS the removal of a fusion, so compiling it would let "
                "Inductor put back what the flag took away"
            ),
            correctness=(
                EXPERT_MLP_MCORE_GATE,
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=EXPERT_MLP_MCORE_OUTPUTS,
                    max_rel_l2=2e-2,
                ),
            ),
        ),
        KernelArm(
            name="mcore/te_activation_func",
            description=(
                "the same layer running TransformerEngine's own SwiGLU "
                "operation. A TWO-FLAG delta -- use_te_activation_func=True "
                "AND bias_activation_fusion=False -- because "
                "TransformerConfig refuses the pair and the base profile sets "
                "the second flag on. REPORT THE MECHANISM, NOT ONLY THE "
                "RATIO: the TE path applies the routing probabilities as a "
                "SEPARATE multiply where megatron's kernel folds them into "
                "the activation, so this is a kernel-COUNT difference as much "
                "as a kernel-speed one. The builder proves the TE module "
                "actually runs with a forward hook, not merely that "
                "TEGroupedMLP picked it"
            ),
            builder=(
                "benchmarks.kernel.operations.expert_mlp"
                ":build_expert_mlp_mcore_te_activation_func"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer, and this arm "
                "is TransformerEngine's own operation run the way megatron "
                "runs it"
            ),
            correctness=(
                EXPERT_MLP_MCORE_GATE,
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=EXPERT_MLP_MCORE_OUTPUTS,
                    max_rel_l2=2e-2,
                ),
            ),
        ),
        KernelArm(
            name="mcore/no_grouped_gemm",
            description=(
                "the same layer as SequentialMLP: four separate TE linear "
                "MLPs called in turn instead of one grouped launch. Each of "
                "them still takes the fused weighted-SwiGLU path, so the arm "
                "isolates the grouping and nothing else. Delivered by "
                "config.moe_grouped_gemm, which reaches the layer spec "
                "through get_gpt_decoder_block_spec on this rev -- and the "
                "builder proves the EXPERT CLASS changed, because megatron "
                "polices that agreement nowhere and both disagreement "
                "directions are numerically correct"
            ),
            builder=(
                "benchmarks.kernel.operations.expert_mlp"
                ":build_expert_mlp_mcore_no_grouped_gemm"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer; compiling the "
                "per-expert loop would also let Inductor recover the grouping "
                "this arm exists to remove"
            ),
            correctness=(
                EXPERT_MLP_MCORE_GATE,
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=EXPERT_MLP_MCORE_OUTPUTS,
                    max_rel_l2=2e-2,
                ),
            ),
        ),
        # ---- TorchTitan ----------------------------------------------------
        #
        # Every titan arm declares all three modes. The isolated backward is
        # kept here and dropped on the megatron side, which is an asymmetry
        # this scenario can afford precisely because it publishes no
        # cross-engine row: no table compares a titan backward against a
        # megatron one, and dropping it from both -- which qkv_prep, ffn_norm
        # and attn_out_proj do, and must -- would delete the swiglu scenario's
        # backward numbers and buy nothing.
        KernelArm(
            name="titan",
            description=(
                "TorchTitan GroupedExperts: separate w1 and w3 grouped GEMMs, "
                "plain-ops SwiGLU, one grouped GEMM for w2, under "
                "torch.compile(fullgraph=True). Upstream's own expert layer, "
                "and this scenario's anchor"
            ),
            builder=(
                "benchmarks.kernel.operations.expert_mlp"
                ":build_expert_mlp_titan"
            ),
            modes=MODES,
            compiled=True,
            correctness=(EXPERT_MLP_TITAN_GATE,),
        ),
        KernelArm(
            name="titan/fused_grouped_experts",
            description=(
                "TorchTitan's OWN w13 fusion, torchtitan.overrides."
                "fused_swiglu.FusedGroupedExperts: one (E, F, 2, D) "
                "parameter, one grouped GEMM for both projections, then an "
                "unbind into gate and up and torchtitan's own silu_and_mul "
                "custom op. UPSTREAM CODE, not ours, and the honest opponent "
                "of both Piper arms -- without it they would be credited with "
                "a fusion TorchTitan already ships"
            ),
            builder=(
                "benchmarks.kernel.operations.expert_mlp"
                ":build_expert_mlp_titan_fused_grouped_experts"
            ),
            modes=MODES,
            compiled=True,
            correctness=(
                EXPERT_MLP_TITAN_GATE,
                CorrectnessCheck(
                    kind="tolerance",
                    reference="titan",
                    outputs=EXPERT_MLP_TITAN_OUTPUTS,
                    max_rel_l2=2e-2,
                ),
            ),
        ),
        KernelArm(
            name="titan/piper_optimized_triton",
            description=(
                "the Piper layer that keeps the combined [R, 2F] activation "
                "tensor: same fused w13 GEMM as titan/fused_grouped_experts, "
                "and a custom Triton op that consumes the combined tensor "
                "directly instead of unbinding it, returning one interleaved "
                "gradient instead of two. Its opponent is that fusion, so its "
                "ratio is what the combined LAYOUT buys"
            ),
            builder=(
                "benchmarks.kernel.operations.expert_mlp"
                ":build_expert_mlp_titan_piper_optimized_triton"
            ),
            modes=MODES,
            compiled=True,
            correctness=(
                EXPERT_MLP_TITAN_GATE,
                CorrectnessCheck(
                    kind="tolerance",
                    reference="titan/fused_grouped_experts",
                    outputs=EXPERT_MLP_TITAN_OUTPUTS,
                    max_rel_l2=2e-2,
                ),
            ),
        ),
        KernelArm(
            name="titan/piper_optimized_inductor",
            description=(
                "the Piper layer with the activation left to Inductor: same "
                "fused w13 GEMM again, and silu(gate) * up written as plain "
                "ops so Inductor can fuse it into its neighbours. It repeats "
                "FusedGroupedExperts' own unbind, which is why that fusion is "
                "its opponent -- the two differ in the activation "
                "implementation and in nothing else"
            ),
            builder=(
                "benchmarks.kernel.operations.expert_mlp"
                ":build_expert_mlp_titan_piper_optimized_inductor"
            ),
            modes=MODES,
            compiled=True,
            correctness=(
                EXPERT_MLP_TITAN_GATE,
                CorrectnessCheck(
                    kind="tolerance",
                    reference="titan/fused_grouped_experts",
                    outputs=EXPERT_MLP_TITAN_OUTPUTS,
                    max_rel_l2=2e-2,
                ),
            ),
        ),
    ),
)


# Each engine is gated against its OWN fp64 truth, under its own output names.
# That is not a stylistic choice: at this cut the two engines compute
# different functions, because megatron applies the routing probabilities
# inside TEGroupedMLP (scenario 11) and titan applies them here. A shared
# ``out`` name would force one side to be gated against the other's function,
# and the difference is a row-by-row scaling rather than a tolerance.
#
# rel_l2 rather than a ULP metric, because a combine is a scatter-add and a
# scatter-add is a reduction. CLAUDE.md's rule applies directly: cancellation
# drives individual outputs toward zero, so a per-element relative error
# divides a negligible absolute error by a negligible magnitude and reports
# thousands of ULPs for a numerically perfect kernel.
#
# The gradient is named next to the output, and it is not a restatement of it.
# The forward is a scatter-add and the backward is a gather, and they are
# separate implementations on both engines -- TE's fused_unpermute has its own
# backward, the torch path's is autograd's, and titan's custom op registers
# one by hand. A forward that agrees is no evidence about any of them.
MOE_COMBINE_MCORE_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("mcore_out", "mcore_expert_out_grad"),
    max_rel_l2=2e-2,
)

# The titan gate carries a third output the megatron side does not have.
# ``titan_scores_grad`` is the gradient of the probability multiply, and it is
# the one number that proves the multiply happened inside the timed region.
# Without it a titan combine that silently stopped scoring would still pass
# every remaining check that the scenario runs.
MOE_COMBINE_TITAN_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("titan_out", "titan_expert_out_grad", "titan_scores_grad"),
    max_rel_l2=2e-2,
)

# The within-engine check, and the premise of both published rows: each delta
# changes the implementation and not the arithmetic. Enforced rather than
# informational, because unlike the qkv_prep fused/unfused pair these arms
# expected to be bit-identical -- TE's fused unpermute and torch's scatter_add
# may accumulate in a different order -- but they must agree to bf16
# tolerance, and a delta that changed the result is a delta that changed the
# operation.
MOE_COMBINE_VARIANT_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="mcore/base",
    outputs=("mcore_out", "mcore_expert_out_grad"),
    max_rel_l2=2e-2,
)


MOE_COMBINE = KernelScenario(
    name="moe_combine",
    description=(
        "The step that puts routed expert outputs back into token order: "
        "TorchTitan's token_dispatcher.combine(...) against megatron-core's "
        "THREE combine phases -- combine_preprocess, token_combine and "
        "combine_postprocess. All three, because "
        "MoEAllGatherTokenDispatcher.token_combine is guarded by tp_size > 1 "
        "or ep_size > 1 and is therefore THE IDENTITY at world_size=1, and "
        "MoELayer.combine wraps that method and nothing else: a cut there "
        "would have measured a zero on the megatron side and read as a "
        "spectacular win. The work is in the neighbouring phases, and the two "
        "dispatcher classes place it differently -- allgather unpermutes in "
        "combine_preprocess, alltoall in combine_postprocess -- so the cut "
        "names the triple. THIS SCENARIO PUBLISHES NO CROSS-ENGINE RATIO. The "
        "routing probabilities are applied on OPPOSITE sides of this "
        "boundary: megatron multiplies them inside TEGroupedMLP "
        "(weighted_bias_swiglu_impl, scenario 11), and titan multiplies them "
        "here -- LocalTokenDispatcher.combine is documented 'Score and "
        "scatter_add routed expert outputs'. So the two sides compute "
        "different functions of the same rows and a ratio would compare two "
        "different amounts of work; the cross-engine row belongs to the "
        "expert_combine span (11+12), which is declared elsewhere. For the "
        "same reason there is no cross-engine correctness check either: each "
        "engine is gated against its own fp64 truth, under its own output "
        "names. The two rows published are megatron's own. no_permute_fusion "
        "swaps TE's fused_unpermute for the torch path (a zeroed tensor and a "
        "scatter_add with an expanded index) at ONE read site inside a "
        "scenario that also holds token_combine and combine_postprocess. "
        "dispatcher_alltoall MEASURES THE DISPATCHER'S LOCAL UNPERMUTE AND "
        "SYNC STRATEGY, NOT COMMUNICATION: _AllToAll.forward returns its "
        "input unchanged at world_size=1, so no bytes move, but the class "
        "still unsorts chunks in combine_preprocess and unpermutes in "
        "combine_postprocess where allgather does neither. Every number here "
        "is device time plus host dispatch and carries NO blocking "
        "device-to-host synchronization inside the timed region -- the "
        "allgather .cpu() and every _maybe_dtoh_and_synchronize sit in the "
        "dispatch phases, which run once at build time. Both engines receive "
        "one expert-output tensor in one (expert, token) row order, checked "
        "at build time against titan's own argsort, and a build-time guard "
        "RAISES unless the megatron combine reproduces the unpermute of those "
        "rows. copy_floor performs the same traffic and the same additions "
        "with no indirection at all: a scatter-add is bandwidth-bound at "
        "these shapes, so the x_floor column is what separates an unpermute "
        "result from a bandwidth result. WARNING: THE gbps AND x_floor COLUMNS "
        "STILL LET A READER RECOVER THE CROSS-ENGINE RATIO THIS SCENARIO "
        "REFUSES TO PUBLISH. Suppressing comparisons removes the ROW and not "
        "the NUMBER. Every arm shares one bytes_moved, so gbps(titan) divided by "
        "gbps(mcore/base) is exactly the suppressed median ratio, and the two "
        "x_floor values divide to the same number. Neither column can be "
        "removed without deleting the floor, so the hazard is stated instead: "
        "that quotient is meaningless for the reason above, and doubly so "
        "because titan is the only compiled arm and titan alone forces a "
        "deterministic scatter_add. gbps counts forward traffic in every mode. "
        "SEPARATELY: THE GB/s FIGURE DESCRIBES MEGATRON'S TRAFFIC AND "
        "UNDERSTATES TITAN'S. The one declared bytes_moved is what an "
        "unpermute moves -- read the routed rows, write one row per token -- "
        "and titan's combine additionally scales every routed row by its "
        "probability, which megatron applied in scenario 11 and does not "
        "repeat here. That is at least one more pass over the [rows, dim] "
        "tensor. By how much the titan figure is low is NOT measured here: it "
        "depends on whether Inductor fuses the cast, the multiply and the "
        "cast back into one pass or materializes an fp32 copy. Read no "
        "bandwidth achievement off the titan row. The two published rows are "
        "both within megatron, so their ratios are unaffected. "
        "Two asymmetries are declared rather than hidden: megatron's "
        "combine_postprocess holds the shape restore where titan's equivalent "
        "view sits after the combine call, so the mcore side is charged one "
        "extra Python call and no kernel; and titan's deterministic_scatter_add "
        "enables deterministic algorithms around its own scatter_add where "
        "megatron's unpermute does not. Neither reaches a published row."
    ),
    inputs_builder=(
        "benchmarks.kernel.operations.moe_combine:moe_combine_inputs"
    ),
    reference_builder=(
        "benchmarks.kernel.operations.moe_combine:moe_combine_reference"
    ),
    baseline_arm="mcore/base",
    # The synthetic routing is a round-robin, exactly balanced only when
    # batch * seq_len * top_k divides num_experts. At any other workload one
    # expert's group is short and the tokens_per_expert the manifest records
    # stops describing the tensors the arms hold. The runner skips the
    # scenario loudly instead, and moe_combine_inputs re-asserts it with named
    # numbers for any direct caller.
    requires_balanced_routing=True,
    # Explicit, exhaustive, and NOT the derived set. Left None the schema
    # would pair every non-floor arm with the anchor, which would publish
    # titan against mcore/base -- the one row this scenario has evidence
    # against. The two rows below are real questions: each is one megatron
    # field, with everything else held fixed.
    comparisons=(
        ("mcore/no_permute_fusion", "mcore/base"),
        ("mcore/dispatcher_alltoall", "mcore/base"),
    ),
    arms=(
        # The floor decides whether the two published rows are kernel results
        # at all. A combine must read every routed row, add them top_k at a
        # time and write one row per token; that is bandwidth-bound by
        # construction, which is why the plan names this scenario and
        # scenario 10 as the strongest floor candidates after the norms.
        # Declared exactly as rope, ffn_norm and moe_residual declare theirs:
        # forward only, is_floor, and an eager_reason rather than
        # compiled=False.
        KernelArm(
            name="copy_floor",
            description=(
                "torch.sum over a contiguous [tokens, top_k, dim] view: the "
                "same traffic and the same additions a combine performs, with "
                "no index read and no scattered write. The combine, if the "
                "routing had already put every token's rows side by side"
            ),
            builder=(
                "benchmarks.kernel.operations.moe_combine:"
                "build_moe_combine_copy_floor"
            ),
            modes=("forward",),
            is_floor=True,
            eager_reason=(
                "a bandwidth floor, not an implementation: compiling a sum "
                "would measure Inductor rather than the bus"
            ),
        ),
        KernelArm(
            name="mcore/base",
            description=(
                "megatron-core's allgather dispatcher off a real GPTModel, "
                "all three combine phases: TransformerEngine's "
                "fused_unpermute in combine_preprocess, an inert token_combine "
                "and a view in combine_postprocess"
            ),
            builder=(
                "benchmarks.kernel.operations.moe_combine:"
                "build_moe_combine_mcore_base"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer, and none of "
                "the three combine phases carries @jit_fuser -- the single "
                "decorator in token_dispatcher.py sits on "
                "MoEFlexTokenDispatcher, a class that asserts "
                "tp_size * ep_size > 1 and is never built here. TE's "
                "fused_unpermute is a hand-written kernel, not a compile "
                "treatment"
            ),
            correctness=(MOE_COMBINE_MCORE_GATE,),
        ),
        KernelArm(
            name="mcore/no_permute_fusion",
            description=(
                "the same allgather dispatcher with moe_permute_fusion=False: "
                "unpermute falls to the torch path, a zeroed [tokens, dim] "
                "tensor and a scatter_add with an index expanded to the "
                "hidden width. ONE read site inside a scenario that is larger "
                "than the unpermute"
            ),
            builder=(
                "benchmarks.kernel.operations.moe_combine:"
                "build_moe_combine_mcore_no_permute_fusion"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "the arm IS the removal of a TE kernel: it replaces "
                "fused_unpermute with torch operations, and compiling those "
                "would erase the difference the published row measures"
            ),
            correctness=(MOE_COMBINE_MCORE_GATE, MOE_COMBINE_VARIANT_GATE),
        ),
        KernelArm(
            name="mcore/dispatcher_alltoall",
            description=(
                "megatron's alltoall token dispatcher, fusion left on. THIS "
                "ARM MEASURES THE DISPATCHER'S LOCAL UNPERMUTE AND SYNC "
                "STRATEGY, NOT COMMUNICATION: _AllToAll.forward returns its "
                "input unchanged at world_size=1. What differs is local -- it "
                "unsorts chunks in combine_preprocess (a real row copy, even "
                "though the chunk order is the identity at one rank) and "
                "unpermutes in combine_postprocess, where the allgather class "
                "unpermutes in combine_preprocess and does nothing else"
            ),
            builder=(
                "benchmarks.kernel.operations.moe_combine:"
                "build_moe_combine_mcore_dispatcher_alltoall"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "the same reason as mcore/base: megatron compiles no whole "
                "layer and no combine phase of this class carries @jit_fuser "
                "either"
            ),
            correctness=(MOE_COMBINE_MCORE_GATE, MOE_COMBINE_VARIANT_GATE),
        ),
        KernelArm(
            name="titan",
            description=(
                "TorchTitan's token_dispatcher.combine(...) under "
                "torch.compile(fullgraph=True): score the routed rows, then "
                "scatter_add them into a zeroed [tokens, dim] tensor. Built "
                "as AllToAllTokenDispatcher, the class production builds, "
                "which delegates to LocalTokenDispatcher.combine at EP=1. "
                "Measured and gated, and deliberately in no comparison -- "
                "read its number as titan's cost for this cut, never as a "
                "ratio against megatron, whose combine does not apply the "
                "probabilities"
            ),
            builder=(
                "benchmarks.kernel.operations.moe_combine:"
                "build_moe_combine_titan"
            ),
            modes=("forward", "forward_backward"),
            compiled=True,
            # No cross-engine check, and that is the same fact the missing
            # ratio states. The two engines compute different functions here,
            # so a gate between them would have to be given either an extra
            # multiply charged to one engine inside its timed region, or two
            # different synthetic expert-output tensors. Both are worse than
            # declaring none. Note also the direction rule this scenario would
            # have had to obey if it declared one: every cross-engine check
            # goes ON the titan arm and REFERENCES mcore/base, because
            # resolve_arm_skips closes the skip set over correctness
            # references and a check pointing out from the anchor would let a
            # skipped titan arm take the anchor -- and every row -- with it.
            correctness=(MOE_COMBINE_TITAN_GATE,),
        ),
    ),
)


# The gate every arm faces, and rel_l2 rather than a ULP metric. An add is not
# a reduction, but the hazard CLAUDE.md's ULP rule names is CANCELLATION, and
# the reduction is only where that rule met it first. ``residual + x`` on two
# independent operands cancels wherever their signs oppose, so individual
# outputs land near zero and a per-element relative error divides a negligible
# absolute error by a negligible magnitude -- the same failure the rule warns
# about, reached without a reduction. rel_l2 takes one norm over the whole
# tensor and is immune to it, and it is the metric every neighbouring
# cross-engine scenario uses on this class of tensor.
#
# ``x_grad`` and ``residual_grad`` are named next to ``out``, and they are not
# a restatement of it. An add that consumed one operand and dropped the other
# moves ``out`` by about 100%, so the forward output already catches that case
# on its own. What the two gradients cover is the BACKWARD, which is a
# separate implementation on both engines: megatron's @jit_fuser region
# compiles its own and titan's torch.compile region compiles its own, so a
# forward that agrees is no evidence about either. A gradient that arrives
# scaled or masked is visible here and nowhere else.
# ``_require_both_gradients`` covers only the coarser case of no gradient at
# all.
MOE_RESIDUAL_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("out", "x_grad", "residual_grad"),
    max_rel_l2=2e-2,
)


MOE_RESIDUAL = KernelScenario(
    name="moe_residual",
    description=(
        "The residual add after the MoE block: TorchTitan's x + moe(...) "
        "against megatron-core's mlp_bda(...). The two engines compute the "
        "SAME add -- the base profile sets hidden_dropout=0.0 and "
        "add_bias_linear=False, so megatron takes the no-bias branch of "
        "_bias_dropout_add_func and F.dropout(p=0.0) returns its own input, "
        "leaving residual + x. They differ in fusion SCOPE: end to end "
        "titan's add is one node inside a whole-block torch.compile region, "
        "where Inductor may fuse it into a neighbour, while megatron's "
        "bias_dropout_add_fused_train is @jit_fuser-decorated and compiles as "
        "a region of its own. Isolating the scenario gives titan megatron's "
        "scope and takes that freedom away, so THIS SCENARIO "
        "PUBLISHES NO CROSS-ENGINE RATIO: the one comparison declared is "
        "megatron's own bias_dropout_fusion on/off delta, compiled "
        "bias_dropout_add_fused_train against eager "
        "bias_dropout_add_unfused. READ THAT ROW AS A DISPATCH COMPARISON "
        "UNLESS THE FLOOR SAYS OTHERWISE: the measurand is one bf16 "
        "elementwise add, which carries only tens of microseconds of device "
        "work at these shapes, and the two arms differ precisely in their "
        "dispatch paths -- a torch.compile wrapper's per-call guard "
        "evaluation against a per-call python closure allocation plus eager "
        "op dispatch. Neither difference is device work. copy_floor moves "
        "the same bytes and the x_floor column is what can say the arms are "
        "device-bound; the --burst residual cannot, because a ladder may "
        "plateau at a dispatch cost bursting never amortizes. The titan arm "
        "is measured and gated but "
        "ranked against nothing; read its number as an upper bound on what "
        "this node can cost titan end to end, never as titan's share of "
        "a step. There is no span here either: titan's downstream partner is "
        "the next block's attention-input norm, which megatron has already "
        "fused into linear_qkv (scenario 2), so no cut leaves both engines "
        "having done the same work. This scenario is the far end of the "
        "within-engine ffn_norm_to_moe_residual span instead, because "
        "fused_residual_rmsnorm is backward-only and joins pre_mlp_layernorm "
        "to mlp_bda. WARNING: THE gbps AND x_floor COLUMNS STILL LET A READER "
        "RECOVER THE CROSS-ENGINE RATIO THIS SCENARIO REFUSES TO PUBLISH. "
        "Every arm here hands the merge the same bytes_moved, so gbps(titan) "
        "divided by gbps(mcore/base) is exactly the suppressed median ratio "
        "on a denominator that is identical by construction, and the two "
        "x_floor values divide to the same number because the floor median "
        "cancels. gbps is the stronger channel, because it reads as a "
        "bandwidth achievement rather than as a ratio, and it counts forward "
        "traffic in every mode. Suppressing comparisons removes the ROW and "
        "not the NUMBER. Neither column can be removed without deleting the "
        "floor or the byte count, and the floor is what decides whether the "
        "published row is a kernel result at all, so the hazard is stated "
        "instead: that quotient carries the isolation this scenario takes "
        "away from titan, and not either engine's add."
    ),
    inputs_builder=(
        "benchmarks.kernel.operations.moe_residual:moe_residual_inputs"
    ),
    reference_builder=(
        "benchmarks.kernel.operations.moe_residual:moe_residual_reference"
    ),
    baseline_arm="mcore/base",
    # Explicit, exhaustive, and NOT the derived set. Left None the schema would
    # pair every non-floor arm with the anchor, which would publish
    # titan against mcore/base -- the one row this scenario has evidence
    # against. See the description and the module docstring: at this cut the
    # engines compute the same add and differ only in the scope isolation
    # removes, so that ratio would land near 1.0 and would mean nothing. The
    # row below is a real question: it is megatron's own fusion flag, with
    # everything else held fixed.
    comparisons=(("mcore/no_bias_dropout_fusion", "mcore/base"),),
    arms=(
        # The floor decides whether the one published row is a kernel result
        # at all. Both mcore arms sit above an operation with tens of
        # microseconds of device work, and they differ in dispatch cost, so
        # without a bandwidth reference a reader cannot tell a fusion result
        # from a dispatch result. Declared exactly as ``rope`` and
        # ``ffn_norm`` declare theirs: forward only, is_floor, and an
        # eager_reason rather than compiled=False.
        KernelArm(
            name="copy_floor",
            description=(
                "torch.add(x, residual, out=out): two reads and one write of "
                "a [batch, seq_len, dim] bf16 tensor, and the bandwidth floor "
                "for the forward traffic at this shape"
            ),
            builder=(
                "benchmarks.kernel.operations.moe_residual:"
                "build_moe_residual_copy_floor"
            ),
            modes=("forward",),
            is_floor=True,
            eager_reason=(
                "a bandwidth floor, not an implementation: compiling an add "
                "would measure Inductor rather than the bus"
            ),
        ),
        KernelArm(
            name="mcore/base",
            description=(
                "megatron-core mlp_bda off a real GPTModel: "
                "bias_dropout_add_fused_train, which megatron itself "
                "decorates with @jit_fuser (torch.compile on torch >= 2.2), "
                "resolved per call exactly as transformer_layer.py:980 "
                "resolves it"
            ),
            builder=(
                "benchmarks.kernel.operations.moe_residual:"
                "build_moe_residual_mcore_base"
            ),
            modes=("forward", "forward_backward"),
            # The one arm in the package whose compile treatment is the
            # engine's own choice rather than the harness's: the builder wraps
            # nothing, and _assert_mcore_bda refuses to continue unless
            # megatron.core.jit.jit_fuser really is torch.compile.
            compiled=True,
            correctness=(MOE_RESIDUAL_GATE,),
        ),
        KernelArm(
            name="mcore/no_bias_dropout_fusion",
            description=(
                "the same call site with bias_dropout_fusion=False: "
                "bias_dropout_add_unfused, plain eager python, with a fresh "
                "closure allocated per call as megatron allocates it"
            ),
            builder=(
                "benchmarks.kernel.operations.moe_residual:"
                "build_moe_residual_mcore_no_bias_dropout_fusion"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "the arm IS the removal of megatron's compiled region: "
                "compiling it would erase the difference the published row "
                "measures"
            ),
            correctness=(
                MOE_RESIDUAL_GATE,
                # The within-engine check, and the one that states the
                # published row's premise: the fusion flag changes the
                # implementation and not the arithmetic. Informational,
                # following the qkv_prep precedent -- a compiled region may
                # legitimately round differently from the eager one, so
                # equality is recorded rather than enforced while
                # MOE_RESIDUAL_GATE still enforces closeness on both arms.
                CorrectnessCheck(
                    kind="bitwise",
                    reference="mcore/base",
                    outputs=("out", "x_grad", "residual_grad"),
                    informational=True,
                ),
            ),
        ),
        KernelArm(
            name="titan",
            description=(
                "TorchTitan's residual add, the x + of x = x + "
                "self.moe(self.ffn_norm(x)), under "
                "torch.compile(fullgraph=True). Measured and gated, and "
                "deliberately in no comparison"
            ),
            builder=(
                "benchmarks.kernel.operations.moe_residual:"
                "build_moe_residual_titan"
            ),
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(
                MOE_RESIDUAL_GATE,
                # The cross-engine gates, and the reason the titan arm exists
                # at all. No ratio is published between the engines, so this
                # is where the scenario's central claim gets tested rather
                # than asserted: the two engines compute one function of one
                # pair of operands. Without the arm the claim would rest on
                # this file.
                #
                # Both gates sit on the NON-ANCHOR arm. ``resolve_arm_skips``
                # closes the skip set over correctness references, so a check
                # pointing from ``mcore/base`` at ``titan`` would let a
                # skipped titan arm take the anchor -- and with it every row
                # in the scenario -- down with it.
                CorrectnessCheck(
                    kind="tolerance",
                    reference="mcore/base",
                    outputs=("out", "x_grad", "residual_grad"),
                    max_rel_l2=2e-2,
                ),
                # Stronger than the gate above, and informational because a
                # compiled region is entitled to reassociate. Both engines
                # evaluate residual + x on identical bf16 operands in the same
                # operand order, so equality is what a reader should expect;
                # recording it is what would show a future lowering silently
                # changing the operation.
                CorrectnessCheck(
                    kind="bitwise",
                    reference="mcore/base",
                    outputs=("out", "x_grad", "residual_grad"),
                    informational=True,
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
# One GEMM and two gradients, at the tolerance every cross-engine projection
# in this registry already uses. A projection is a reduction over ``dim``, so
# the gate is rel_l2 and never a max or a ULP count: cancellation drives
# individual outputs toward zero and a per-element metric then reports a huge
# number for arithmetic that is exactly right.
LM_HEAD_PROJECTION_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("out", "x_grad", "weight_grad"),
    max_rel_l2=2e-2,
)


LM_HEAD_PROJECTION = KernelScenario(
    name="lm_head_projection",
    description=(
        "The language-model head, cross-engine: megatron-core's "
        "ColumnParallelLinear against TorchTitan's nn.Linear, over one shared "
        "[vocab_size, dim] weight. This is the projection alone. The loss is "
        "scenario 16's cut, and no arm here computes one. "
        "The megatron arm is NOT TransformerEngine, and the row must not be "
        "read as TE against torch: GPTModel selects "
        "tensor_parallel.ColumnParallelLinear for the output layer, and its "
        "TE variant is reachable only under an mxfp8 recipe this build never "
        "sets. The megatron arm is megatron-native, and it dispatches through "
        "linear_with_grad_accumulation_and_async_allreduce, so it pays two "
        "torch.autograd.Function.apply calls that titan's F.linear does not. "
        "At tensor-parallel size 1 that path runs no collective, but the "
        "dispatch is not free and it is a real part of the gap. "
        "Both arms are eager, and the titan arm's treatment is the "
        "correction: torchtitan's apply_compile walks model.layers only, and "
        "lm_head is a sibling of layers rather than a child, so no compiled "
        "region reaches it end-to-end. One titan configuration does compile a "
        "projection of this shape -- the fused_linear_ce loss owns the head "
        "under the LossWithLMHead protocol -- but that is a different cut, it "
        "fuses the loss in, and it belongs to a span this scenario does not "
        "declare."
    ),
    inputs_builder=(
        "benchmarks.kernel.operations.lm_head_projection"
        ":lm_head_projection_inputs"
    ),
    reference_builder=(
        "benchmarks.kernel.operations.lm_head_projection"
        ":lm_head_projection_reference"
    ),
    baseline_arm="mcore/base",
    # Explicit, and the same single pair the default derivation would produce.
    # Writing it down makes the direction of the published ratio a
    # declaration: this scenario reports titan against megatron, which matches
    # the e2e piper1b_megatron scenario, where megatron is also the anchor.
    comparisons=(("titan", "mcore/base"),),
    arms=(
        KernelArm(
            name="mcore/base",
            description=(
                "Megatron-core GPTModel.output_layer: "
                "tensor_parallel.ColumnParallelLinear, NOT TransformerEngine, "
                "eager, tp_size 1 so no all-gather and no dgrad all-reduce "
                "run"
            ),
            builder=(
                "benchmarks.kernel.operations.lm_head_projection"
                ":build_lm_head_projection_mcore_base"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer and no output "
                "layer, so this module runs eager end to end; compiling it "
                "would measure a treatment megatron never applies"
            ),
            correctness=(LM_HEAD_PROJECTION_GATE,),
        ),
        KernelArm(
            name="titan",
            description=(
                "TorchTitan decoder.lm_head: nn.Linear without a bias, eager, "
                "because apply_compile reaches model.layers only and lm_head "
                "is a sibling of layers"
            ),
            builder=(
                "benchmarks.kernel.operations.lm_head_projection"
                ":build_lm_head_projection_titan"
            ),
            modes=("forward", "forward_backward"),
            # No compiled=True, and the eager_reason is the whole finding of
            # this scenario. The plan asserts every titan module arm runs
            # under torch.compile(fullgraph=True). That is true of every arm
            # built from a transformer block and false here, for the same
            # reason it is false in scenario 1 (tok_embeddings) and scenario
            # 14 (norm): all three are siblings of ``layers`` in the decoder,
            # and apply_compile walks the children of ``layers`` alone.
            eager_reason=(
                "torchtitan's apply_compile compiles each TransformerBlock in "
                "model.layers; lm_head is a sibling of layers in the decoder, "
                "so it sits outside every compiled region end-to-end and a "
                "compiled arm here would measure a treatment no run applies"
            ),
            correctness=(
                LM_HEAD_PROJECTION_GATE,
                # The cross-engine gate, declared on the titan arm and
                # pointing at the anchor. The direction is required, not
                # stylistic: resolve_arm_skips closes the skip set over
                # correctness references, so a gate declared on the anchor and
                # pointing at titan would make the anchor depend on titan and
                # cost the whole scenario if titan were ever skipped.
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


# The gate every arm faces. Attention is a reduction over the sequence, so
# rel_l2 is the only safe metric and a max or a ULP count is forbidden
# (CLAUDE.md, "Choosing a correctness metric"): cancellation drives
# individual outputs toward zero, and a per-element metric then reports a
# huge number for arithmetic that is exactly right.
#
# Measured on an H200 at dim 256, 4 q heads over 2 kv groups, head_dim 64,
# batch 2, seq 128, over 8 seeded packed documents: megatron lands at 1.76e-3
# (out), 2.75e-3 (dq), 3.31e-3 (dk), 2.81e-3 (dv), and TorchTitan's
# FlexAttention at 1.76e-3 / 2.75e-3 / 2.72e-3 / 2.22e-3. The gate keeps
# roughly six times that headroom, which the normal shape needs: the
# reduction there runs over eight times as many keys.
ATTENTION_CORE_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=("out", "dq", "dk", "dv"),
    max_rel_l2=2e-2,
)


# Recorded, not enforced, and pointed at the anchor from every other arm.
#
# Informational because the fp64 gates above are stronger: each arm is right
# in absolute terms, which bounds the distance between any two of them. And
# the cross-engine agreement has been checked at exactly one small shape, so
# enforcing it would risk the whole scenario on a number nobody has measured
# where it will run. Measured there, the two engines agree to 4.3e-4 on
# ``out`` and to 2.9e-3 on the widest gradient.
#
# The direction is required, not stylistic. ``resolve_arm_skips`` closes the
# skip set over correctness references, so a check declared on the anchor and
# pointing outward would let a skipped arm take the anchor down -- and the
# anchor's loss costs the scenario.
ATTENTION_CORE_CROSS_ARM = CorrectnessCheck(
    kind="tolerance",
    reference="mcore/base",
    outputs=("out", "dq", "dk", "dv"),
    max_rel_l2=2e-2,
    informational=True,
)


ATTENTION_CORE = KernelScenario(
    name="attention_core",
    description=(
        "Inner attention at Piper-1B shapes with packed-document causal "
        "masking, cross-engine: megatron-core's TEDotProductAttention "
        "against TorchTitan's FlexAttention, the same FlexAttention lowered "
        "to FlashAttention-4, and VarlenAttention over FlashAttention-3. "
        "The cut runs from q/k/v after RoPE to the attention output; the "
        "projections belong to qkv_prep and attn_out_proj. Both engines read "
        "the same q/k/v VALUES, in each engine's own MEMORY LAYOUT, and the "
        "layouts differ on purpose. Titan gets three contiguous tensors, "
        "which is what its projection materializes. Megatron gets a "
        "contiguous query, a contiguous key and ONE NON-CONTIGUOUS STRIDED "
        "VIEW -- the value. Its QKV GEMM writes one fused buffer and splits "
        "it into three views; the query and the key then leave the buffer "
        "because the norm and the rotation write fresh tensors, and the "
        "value is neither normed nor rotated. TransformerEngine does not "
        "recognize that layout and copies the value inside every timed "
        "megatron call -- 4 MiB per forward at batch 4 / seq 1024 / normal, "
        "paid by every backend because get_qkv_layout runs before the "
        "backend is chosen. THAT COPY IS THE COST qkv_prep SAYS MEGATRON "
        "DEFERS TO THIS SCENARIO, so a megatron number here is attention "
        "plus megatron's own layout adaptation for the value. The key's "
        "equal half is absorbed by k_layernorm in the engine, which belongs "
        "to scenario qk_norm, and that scenario hands its megatron arm the "
        "same kind of strided key -- so the key's 4 MiB is timed there and "
        "this one declines to double-book it. Megatron's "
        "[T, N*H] output is canonicalized back outside every timed closure. "
        "COMPILE TREATMENT DIFFERS BY ARM AND THE RATIOS ARE COMPARISONS OF "
        "TREATMENTS. Every megatron arm is eager, because megatron compiles "
        "no whole transformer layer. 'titan' and 'titan/flex_flash' are "
        "compiled by FlexAttention's own class-level torch.compile, which "
        "carries max_autotune AND coordinate_descent_tuning; "
        "'titan/flash_attention_3' gets a plain torch.compile(fullgraph=True) "
        "with no autotune. So the Triton template is the only autotuned arm "
        "in the scenario, and a row against it is not a kernel-quality claim "
        "on its own. "
        "WHICH KERNEL EACH MEGATRON ARM RUNS IS PINNED BY THE PROFILE AND "
        "ENFORCED BY A GUARD, because every backend computes the same "
        "function and no correctness gate can tell them apart: mcore/base "
        "runs cuDNN FusedAttention, mcore/attn_flash3 runs FlashAttention 3, "
        "mcore/attn_unfused runs TE's torch implementation. There is no "
        "mcore FlashAttention-4 arm: TE prefers FA3 on sm90 whenever both "
        "are installed and no megatron setting reaches past that, so "
        "'titan/flex_flash' has no megatron opponent and is published "
        "against 'titan' instead, "
        "which isolates the lowering. Expect it to lose on Hopper for a "
        "reason that is not about FA4: FlexAttention's packed-interval mask "
        "optimization is gated on compute capability 10/11, so partial "
        "blocks evaluate the mask per lane here. "
        "No bandwidth floor is declared, deliberately: attention's "
        "arithmetic grows with the square of the sequence length while its "
        "traffic grows linearly, so a copy floor would bound nothing. That "
        "does NOT make these numbers device time -- run --burst before "
        "ranking anything. "
        "The number includes host serialization on neither side: no timed "
        "closure calls .cpu(), .item() or synchronize, and every mask form "
        "is built once in the inputs builder."
    ),
    inputs_builder=(
        "benchmarks.kernel.operations.attention_core:attention_core_inputs"
    ),
    reference_builder=(
        "benchmarks.kernel.operations.attention_core:attention_core_reference"
    ),
    baseline_arm="mcore/base",
    # Explicit and exhaustive, and every arm names EXACTLY ONE opponent.
    # reporting.py keys its printed table on (arm, mode), so a second
    # opponent for one arm would be dropped from the terminal while surviving
    # in results.json. Merge note 12 names the two rows a fixed renderer
    # would unlock.
    #
    # The derived set would have compared all five non-anchor arms against
    # mcore/base, which loses the row this scenario exists for:
    # titan/flash_attention_3 against mcore/attn_flash3 is the same kernel
    # family and the same THD masking through two host stacks, and it is the
    # only pair here that isolates the engine.
    comparisons=(
        # Within megatron: FlashAttention-3 against the cuDNN kernel.
        ("mcore/attn_flash3", "mcore/base"),
        # Within megatron: the unfused path against the fused one.
        ("mcore/attn_unfused", "mcore/base"),
        # The headline cross-engine row. An autotuned Triton template against
        # an eager cuDNN kernel -- read the description before ranking it.
        ("titan", "mcore/base"),
        # Within titan: the same module, the same BlockMask and the same
        # mask_mod, lowered two ways. The one pair here that isolates the
        # kernel family alone.
        ("titan/flex_flash", "titan"),
        # Cross-engine, one kernel family: both sides run FlashAttention-3
        # varlen over the same cu_seqlens, and only the host stack differs.
        ("titan/flash_attention_3", "mcore/attn_flash3"),
    ),
    arms=(
        KernelArm(
            name="mcore/base",
            description=(
                "megatron self_attention.core_attention with "
                "attention_backend PINNED to fused: TransformerEngine's cuDNN "
                "FusedAttention over THD cu_seqlens, eager as megatron runs "
                "it. Pinned rather than left at AttnBackend.auto. THE BASE "
                "PROFILE STILL CARRIES NO attention_backend, so the e2e "
                "megatron arm and every other cross-engine scenario's "
                "mcore/base still run at auto, and this arm's profile is the "
                "only pinned one. Measured on this host the two make the "
                "same selection, so this arm runs the kernel the e2e arm "
                "runs; pinning makes that a property of the profile instead "
                "of the device"
            ),
            builder=(
                "benchmarks.kernel.operations.attention_core"
                ":build_attention_core_mcore_base"
            ),
            # No isolated backward on either engine. TE's fused-attention
            # autograd function consumes its saved-tensor context on the
            # first backward and then raises, so the retained-graph re-run
            # other scenarios use is unavailable; the titan arms record the
            # same. Backward cost is forward_backward minus forward.
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer, and "
                "TEDotProductAttention carries no jit_fuser, so this module "
                "runs eager end to end; compiling it would measure a "
                "treatment megatron never applies"
            ),
            correctness=(ATTENTION_CORE_GATE,),
        ),
        KernelArm(
            name="mcore/attn_flash3",
            description=(
                "megatron with attention_backend pinned to flash, which "
                "TransformerEngine resolves to FlashAttention 3.0.0 on sm90. "
                "The opponent of titan/flash_attention_3: the same kernel "
                "family and the same THD cu_seqlens through a different host "
                "stack. The GENERATION is enforced by the arm's guard and not "
                "by megatron -- flash_attention_version writes "
                "NVTE_FLASH_ATTN_V2/V3/V4, which TransformerEngine 2.17.1 "
                "reads nowhere, and FA3 degrades to FA2 rather than failing"
            ),
            builder=(
                "benchmarks.kernel.operations.attention_core"
                ":build_attention_core_mcore_flash3"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer, so its "
                "attention runs eager whichever backend it selects"
            ),
            correctness=(ATTENTION_CORE_GATE, ATTENTION_CORE_CROSS_ARM),
        ),
        KernelArm(
            name="mcore/attn_unfused",
            description=(
                "megatron with attention_backend pinned to unfused: "
                "TransformerEngine's own torch implementation, which "
                "materializes the score matrix. NOT a bandwidth floor -- it "
                "is a real implementation and it publishes a ratio. IT WILL "
                "DOMINATE THE PEAK-MEMORY COLUMN AND IT CAN EXHAUST THE "
                "DEVICE. Its scores are [SEGMENTS, n_heads, max_seqlen, "
                "max_seqlen], where SEGMENTS is cu_seqlens.numel() - 1 "
                "(ConvertTHDtoBSHD reads exactly that, "
                "dot_product_attention/utils.py:2153) and NOT the real "
                "document count. cu_seqlens is padded to 128 entries, so "
                "segments is 127 at every workload: one bf16 score tensor is "
                "4.0 GiB at seq 1024, 15.9 GiB at 2048 and 63.5 GiB at 4096 "
                "on the normal shape, and 47.6 GiB at seq 1024 on huge. The "
                "arm needs the scores, the saved probabilities and the "
                "backward gradient, so budget three of those. A sweep past "
                "seq 2048 will OOM, and because the correctness pass has no "
                "per-arm exception handling that OOM takes the whole "
                "scenario with it"
            ),
            builder=(
                "benchmarks.kernel.operations.attention_core"
                ":build_attention_core_mcore_unfused"
            ),
            modes=("forward", "forward_backward"),
            eager_reason=(
                "megatron compiles no whole transformer layer, so its "
                "attention runs eager whichever backend it selects"
            ),
            correctness=(ATTENTION_CORE_GATE, ATTENTION_CORE_CROSS_ARM),
        ),
        KernelArm(
            name="titan",
            description=(
                "TorchTitan FlexAttention: an Inductor-generated Triton "
                "template driven by a block-diagonal causal BlockMask. "
                "Compiled by the class's own torch.compile of flex_attention, "
                "which carries max_autotune and coordinate_descent_tuning, so "
                "this is the ONLY autotuned arm in the scenario"
            ),
            builder=(
                "benchmarks.kernel.operations.attention_core"
                ":build_attention_core_titan"
            ),
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(ATTENTION_CORE_GATE, ATTENTION_CORE_CROSS_ARM),
        ),
        KernelArm(
            name="titan/flex_flash",
            description=(
                "The same FlexAttention module, mask_mod and BlockMask "
                "lowered to FlashAttention-4 CuTe DSL kernels instead of a "
                "Triton template (BACKEND=FLASH, 256x128 blocks); requires "
                "the fa4 dependency group. Published against titan, because "
                "only the lowering differs and because TransformerEngine "
                "cannot select FA4 on sm90, so there is no megatron opponent"
            ),
            builder=(
                "benchmarks.kernel.operations.attention_core"
                ":build_attention_core_titan_flex_flash"
            ),
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(ATTENTION_CORE_GATE, ATTENTION_CORE_CROSS_ARM),
        ),
        KernelArm(
            name="titan/flash_attention_3",
            description=(
                "FlashAttention-3 varlen (CUTLASS sm90) over the same packed "
                "documents, via torch.nn.attention.varlen; requires the "
                "flash3 dependency group. Wrapped in a plain "
                "torch.compile(fullgraph=True) with NO autotune, unlike the "
                "two FlexAttention arms"
            ),
            builder=(
                "benchmarks.kernel.operations.attention_core"
                ":build_attention_core_titan_flash3"
            ),
            modes=("forward", "forward_backward"),
            compiled=True,
            correctness=(ATTENTION_CORE_GATE, ATTENTION_CORE_CROSS_ARM),
        ),
    ),
)


KERNEL_SCENARIOS = {
    scenario.name: scenario
    for scenario in (
        ROPE,
        SWIGLU,
        LM_HEAD,
        EMBEDDING_STAGE,
        QKV_PREP,
        QK_NORM,
        ATTN_OUT_PROJ,
        ATTN_RESIDUAL,
        FFN_NORM,
        MOE_ROUTER,
        DISPATCH_PERMUTE,
        EXPERT_MLP,
        MOE_COMBINE,
        MOE_RESIDUAL,
        FINAL_NORM,
        LM_HEAD_PROJECTION,
        CROSS_ENTROPY,
        ATTENTION_CORE,
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
