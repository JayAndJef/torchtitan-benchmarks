"""Which kernel spans exist, and what each one replaces.

A **span** is an implementation that fuses across a scenario cut. It belongs
to no single scenario, so it is declared over an ordered scenario range and
its claim is the span against the **sum of the scenarios it replaces**. The
declaration types live in ``benchmarks.kernel.schema``; this module is
nothing but instances of them, exactly as ``benchmarks.kernel.registry`` is
for scenarios.

**Why spans live here and not in ``registry.py``.** A span names arms inside
other scenarios, so this module has to import the scenario registry to check
that each named arm exists. The dependency runs one way -- a span knows its
scenarios, a scenario knows nothing of spans -- and a separate module is what
makes that visible. Putting the cross-check inline in the scenario registry
would bury module-level validation in the middle of a data file and would
invite a scenario to reference a span back.

**A span is not a scenario, and the two rosters are disjoint.**
``KernelSpan`` composes a ``KernelScenario`` (its own head-to-head) rather
than subclassing one, so a span cannot be handed to anything that expects a
scenario without someone writing ``.measurement``. A span that reached
``KERNEL_SCENARIOS`` would run as a bare scenario and publish one of its two
totals under a name that promises both.

**Every span row carries one systematic bias, and it favours the span.** The
parts total pays one host dispatch chain per enclosed scenario; the span pays
one. CLAUDE.md records that roughly 85% of a kernel number in this repository
is host dispatch rather than device time, so a parts total over N scenarios
holds N-1 extra chains that no fusion removed -- the harness stopped paying
them because it timed one closure instead of N. The published ratio is
therefore smaller than fusion alone would make it, and the effect grows with
the length of the range.

The bias is a property of the **range length**, not of what a span fuses, so
the engine states it and no declaration has to remember to. It is printed
under every span table (``benchmarks.kernel.results.reporting``) and recorded
in every span results file (``KERNEL_SPAN_METHODOLOGY``). Nothing corrects
for it: separating the two would need profiler-summed device time, and
nothing in this repository measures that. A span's own ``description`` should
still say what the fusion is, so a reader knows what the remainder of the
ratio is supposed to be.

Torch-free and parent-side, like the schema and the scenario registry: arm
builders are dotted strings resolved inside the GPU worker, so declaring a
span costs no import.

**NO BUILDER EXISTS FOR ANY DECLARED SPAN YET.** A span arm's ``builder`` is
a dotted string, and every one of them names a module under
``benchmarks.kernel.operations`` that nobody has written. A declaration is
therefore complete and a measurement is not: ``--span <name>`` measures every
enclosed scenario, then dies in the span's correctness worker at
``resolve_symbol``, because the parent plans the enclosed scenarios first.
Read the roster below as a specification of the arms a later commit must
build, and read no number off it.

Never present these numbers as end-to-end results, and never present a span
total as a scenario total: a span answers "what does fusing across these cuts
buy", which no single scenario asks.
"""

from __future__ import annotations

from benchmarks.kernel.registry import KERNEL_SCENARIOS
from benchmarks.kernel.schema import (
    CorrectnessCheck,
    KernelArm,
    KernelScenario,
    KernelSpan,
    SpanParts,
    validate_span_parts,
)


# Both engines produce the same tensors at this cut, under the same names,
# which is the property that makes the span honest: inside ``expert_mlp`` and
# ``moe_combine`` the two engines carry DIFFERENT output names on purpose, so
# that no gate can cross a boundary at which they compute different functions.
EXPERT_COMBINE_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=(
        "out",
        "x_grad",
        "probs_grad",
        "w1_grad",
        "w2_grad",
        "w3_grad",
    ),
    max_rel_l2=2e-2,
)

# The gate that earns the cross-engine row. If megatron and TorchTitan agree
# here to 2e-2 then both have applied the routing probabilities exactly once,
# which is the claim the span is built on; if they do not, the enclosure is
# wrong and the ratio means nothing.
EXPERT_COMBINE_CROSS_ARM = CorrectnessCheck(
    kind="tolerance",
    reference="mcore/base",
    outputs=EXPERT_COMBINE_GATE.outputs,
    max_rel_l2=2e-2,
)


EXPERT_COMBINE = KernelSpan(
    measurement=KernelScenario(
        name="expert_combine",
        description=(
            "The routed-expert MLP and the combine as ONE cut, cross-engine: "
            "megatron-core's experts call followed by its three combine "
            "phases, against TorchTitan's inner_experts followed by "
            "token_dispatcher.combine. THIS IS THE ONLY HONEST CROSS-ENGINE "
            "NUMBER FOR THIS REGION, and that is why the span exists. The "
            "routing probabilities land on OPPOSITE SIDES of the "
            "expert_mlp/moe_combine boundary: megatron multiplies them "
            "inside TEGroupedMLP, in weighted_bias_swiglu_impl, and "
            "TorchTitan multiplies them in combine. So each of the two "
            "enclosed scenarios declares no cross-engine row -- at either "
            "cut alone the two engines compute different functions of the "
            "same rows -- and this span is the SMALLEST ENCLOSURE in which "
            "both engines have applied the probabilities exactly once. The "
            "cross-engine correctness gate is what proves that: the two arms "
            "produce the same tensors here, under the same names, where the "
            "enclosed scenarios deliberately name theirs apart. "
            "COMPILE TREATMENT: the megatron arm is eager and the TorchTitan "
            "arm runs under torch.compile(fullgraph=True), as each engine "
            "runs this code end to end, so the row compares two treatments "
            "and not two kernels alone. "
            "THE SPAN-VERSUS-PARTS ROW CARRIES A BIAS AND IT FAVOURS THE "
            "SPAN: the parts total pays one host dispatch chain per enclosed "
            "scenario -- two here -- and the span pays one, and roughly 85% "
            "of a kernel number in this repository is host dispatch rather "
            "than device time. Two scenarios is the shortest range any span "
            "declares, so the bias is smallest here; it is not zero. "
            "READ THE TWO ROWS AS TWO DIFFERENT STATISTICS. The titan "
            "against mcore/base row is a within-span comparison, measured in "
            "one block-major sweep and PAIRED, exactly as a scenario's row "
            "is. The span against the parts sum is UNPAIRED -- every worker "
            "of both enclosed scenarios separates a span replicate from the "
            "part replicate that shares its index -- so its interval is "
            "published as unpaired_ratio_ci_* and may never be set beside a "
            "scenario interval as the same quantity."
        ),
        inputs_builder=(
            "benchmarks.kernel.operations.expert_combine"
            ":expert_combine_inputs"
        ),
        reference_builder=(
            "benchmarks.kernel.operations.expert_combine"
            ":expert_combine_reference"
        ),
        arms=(
            KernelArm(
                name="mcore/base",
                description=(
                    "megatron-core off a real GPTModel: TEGroupedMLP, whose "
                    "fused activation kernel already folds the routing "
                    "probabilities in, then all three combine phases of the "
                    "allgather dispatcher. The probabilities are applied "
                    "once, on the expert side of the cut"
                ),
                builder=(
                    "benchmarks.kernel.operations.expert_combine"
                    ":build_expert_combine_mcore_base"
                ),
                modes=("forward", "forward_backward"),
                compiled=False,
                eager_reason=(
                    "megatron compiles no whole transformer layer, so this "
                    "is how megatron runs the region end to end"
                ),
                correctness=(EXPERT_COMBINE_GATE,),
            ),
            KernelArm(
                name="titan",
                description=(
                    "TorchTitan under torch.compile(fullgraph=True): "
                    "GroupedExperts, then token_dispatcher.combine, which "
                    "scores the routed rows and scatter-adds them back into "
                    "token order. The probabilities are applied once, on the "
                    "combine side of the cut"
                ),
                builder=(
                    "benchmarks.kernel.operations.expert_combine"
                    ":build_expert_combine_titan"
                ),
                modes=("forward", "forward_backward"),
                compiled=True,
                correctness=(
                    EXPERT_COMBINE_GATE,
                    EXPERT_COMBINE_CROSS_ARM,
                ),
            ),
        ),
        baseline_arm="mcore/base",
        # The span builds its own routed rows, so it needs the same even
        # split its two enclosed scenarios need.
        requires_balanced_routing=True,
        # Declared rather than derived, because this one row is the reason
        # the span exists.
        comparisons=(("titan", "mcore/base"),),
    ),
    scenarios=("expert_mlp", "moe_combine"),
    parts=(
        SpanParts(arm="mcore/base", parts=("mcore/base", "mcore/base")),
        # expert_mlp anchors on ``titan`` and moe_combine on ``mcore/base``,
        # and neither anchor decides a part: what an arm replaces is named
        # here, per arm, and read from nowhere else.
        SpanParts(arm="titan", parts=("titan", "titan")),
    ),
)


ATTN_RESIDUAL_NORM_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="fp64",
    outputs=(
        "out",
        "attn_out_grad",
        "residual_grad",
        "proj_weight_grad",
        "norm_weight_grad",
    ),
    max_rel_l2=2e-2,
)

# Two weights cross this cut, so neither may be called ``weight_grad``:
# ``attn_out_proj`` and ``ffn_norm`` each name one that way, and a span that
# reused the name would publish a gate over whichever of the two the builder
# happened to return.
ATTN_RESIDUAL_NORM_CROSS_ARM = CorrectnessCheck(
    kind="tolerance",
    reference="mcore/base",
    outputs=ATTN_RESIDUAL_NORM_GATE.outputs,
    max_rel_l2=2e-2,
)


ATTN_RESIDUAL_NORM = KernelSpan(
    measurement=KernelScenario(
        name="attn_residual_norm",
        description=(
            "The attention output projection, the residual add after it and "
            "the norm in front of the MoE block, as ONE cut, cross-engine: "
            "megatron-core's linear_proj, self_attn_bda and "
            "pre_mlp_layernorm against TorchTitan's self.wo(...), "
            "x + attention(...) and self.ffn_norm(x). Megatron needs no "
            "coarsening to meet it -- those are three separate submodules. "
            "THE RANGE IS 6+7+8 AND NOT 6+7, AND THAT IS THE WHOLE POINT. "
            "TorchTitan's residual add fuses FORWARD, into the prologue of "
            "the next norm, and not backward into the epilogue of the "
            "projection GEMM: qwen3/model.py:60 writes "
            "x = x + self.attention(self.attention_norm(x), ...) and :63 "
            "immediately consumes it as self.ffn_norm(x), and Inductor "
            "cannot fold an elementwise add of a DIFFERENT tensor into a "
            "GEMM epilogue in the general case. A 6+7 span therefore cuts "
            "on the wrong side: both engines emit a GEMM and then a "
            "separate add, the ratio lands near 1.0, and it lands there for "
            "the same reason attn_residual alone declares no cross-engine "
            "row. "
            "THE LAYOUT OP BEFORE THE PROJECTION IS EXCLUDED ON BOTH SIDES, "
            "and the two excluded operations are not the same object: "
            "titan's is a materializing contiguous() copy, because "
            "FlexAttention returns a transposed view, and megatron's is a "
            "free reshape. This span inherits attn_out_proj's exclusion, so "
            "the titan side is short that copy exactly as a cross-engine "
            "sum over the three scenarios is. "
            "TE NORMS RUN THROUGH THE cuDNN BACKEND here: "
            "NVTE_NORM_FWD_USE_CUDNN and NVTE_NORM_BWD_USE_CUDNN are set "
            "because TE's native RMSNorm kernels fail to launch on this "
            "box, so the megatron number is not TE's fastest norm and must "
            "not be published as 'megatron's norm'. "
            "COMPILE TREATMENT: the megatron arm is eager at the point it "
            "is timed from and the TorchTitan arm runs under "
            "torch.compile(fullgraph=True). Note that attn_residual's own "
            "mcore/base arm declares compiled=True, because ITS timed "
            "closure calls bias_dropout_add_fused_train directly and that "
            "function is the torch.compile wrapper; here the same function "
            "is one call inside a plain Python closure. The treatment of "
            "the region is identical -- only the field differs, because the "
            "field describes the entry point. "
            "THE SPAN-VERSUS-PARTS ROW CARRIES A BIAS AND IT FAVOURS THE "
            "SPAN: the parts total pays one host dispatch chain per "
            "enclosed scenario -- three here -- and the span pays one, and "
            "roughly 85% of a kernel number in this repository is host "
            "dispatch rather than device time. This range is longer than "
            "expert_combine's, so more of any gain below 1.0 is the two "
            "chains the harness stopped paying. "
            "READ THE TWO ROWS AS TWO DIFFERENT STATISTICS. The titan "
            "against mcore/base row is a within-span comparison, measured "
            "in one block-major sweep and PAIRED. The span against the "
            "parts sum is UNPAIRED -- every worker of all three enclosed "
            "scenarios separates a span replicate from the part replicate "
            "that shares its index -- so its interval is published as "
            "unpaired_ratio_ci_* and may never be set beside a scenario "
            "interval as the same quantity."
        ),
        inputs_builder=(
            "benchmarks.kernel.operations.attn_residual_norm"
            ":attn_residual_norm_inputs"
        ),
        reference_builder=(
            "benchmarks.kernel.operations.attn_residual_norm"
            ":attn_residual_norm_reference"
        ),
        arms=(
            KernelArm(
                name="mcore/base",
                description=(
                    "megatron-core off a real GPTModel: TERowParallelLinear, "
                    "then self_attn_bda, then pre_mlp_layernorm as a "
                    "transformer_engine.pytorch.RMSNorm through the cuDNN "
                    "norm backend. Three submodules called in turn, as "
                    "TransformerLayer.forward calls them"
                ),
                builder=(
                    "benchmarks.kernel.operations.attn_residual_norm"
                    ":build_attn_residual_norm_mcore_base"
                ),
                modes=("forward", "forward_backward"),
                compiled=False,
                eager_reason=(
                    "megatron compiles no whole transformer layer. The "
                    "@jit_fuser bias_dropout_add_fused_train inside the "
                    "closure still compiles as its own region, which is "
                    "megatron's choice and not the harness's"
                ),
                correctness=(ATTN_RESIDUAL_NORM_GATE,),
            ),
            KernelArm(
                name="titan",
                description=(
                    "TorchTitan under torch.compile(fullgraph=True): wo, "
                    "the residual add and ffn_norm in one graph, which is "
                    "the scope apply_compile gives them end to end and the "
                    "scope in which the add folds into the norm's prologue"
                ),
                builder=(
                    "benchmarks.kernel.operations.attn_residual_norm"
                    ":build_attn_residual_norm_titan"
                ),
                modes=("forward", "forward_backward"),
                compiled=True,
                correctness=(
                    ATTN_RESIDUAL_NORM_GATE,
                    ATTN_RESIDUAL_NORM_CROSS_ARM,
                ),
            ),
        ),
        baseline_arm="mcore/base",
        comparisons=(("titan", "mcore/base"),),
    ),
    scenarios=("attn_out_proj", "attn_residual", "ffn_norm"),
    parts=(
        SpanParts(
            arm="mcore/base",
            parts=("mcore/base", "mcore/base", "mcore/base"),
        ),
        SpanParts(arm="titan", parts=("titan", "titan", "titan")),
    ),
)


# The base arm IS the reference here, and there is no fp64 truth. See the
# span's description: an fp64 reference over this range would have to
# reproduce the top-k routing decision from an fp64 norm output, and a near
# tie would select a different expert. What the fusion claims is that it
# changes the backward implementation and not the arithmetic, so the arm it
# has to match is the unfused one.
FUSED_RESIDUAL_RMSNORM_GATE = CorrectnessCheck(
    kind="tolerance",
    reference="mcore/base",
    outputs=("out", "x_grad", "norm_weight_grad"),
    max_rel_l2=2e-2,
)

# Informational, and it is evidence either way. The fusion is documented as
# backward-only, so the forward output should be the unfused one bit for
# bit; a difference here says the forward changed too, which no part of this
# declaration expects.
FUSED_RESIDUAL_RMSNORM_FORWARD_IS_UNCHANGED = CorrectnessCheck(
    kind="bitwise",
    reference="mcore/base",
    outputs=("out",),
    informational=True,
)


FFN_NORM_TO_MOE_RESIDUAL = KernelSpan(
    measurement=KernelScenario(
        name="ffn_norm_to_moe_residual",
        description=(
            "megatron's own residual-plus-RMSNorm backward fusion, over the "
            "six cuts its residual crosses: pre_mlp_layernorm, the router, "
            "the dispatch, the experts, the combine and mlp_bda. "
            "WITHIN MEGATRON ONLY, AND BACKWARD ONLY. "
            "config.fused_residual_rmsnorm (transformer_config.py:508, "
            "default False) is documented as fusing the residual connection "
            "and the RMSNorm BACKWARD pass when TE is used, so a "
            "forward-mode arm would be the base arm under another name. "
            "Both arms therefore declare forward_backward and nothing else. "
            "ITS PARTNER IS mlp_bda AND NOT self_attn_bda, which is what "
            "sets the range. In this build's layer spec "
            "pre_mlp_layernorm=backend.layer_norm(has_residual=True) "
            "(gpt_layer_specs.py:336) is the ONLY has_residual norm -- the "
            "attention-input norm is fused inside "
            "TELayerNormColumnParallelLinear and has no separate module -- "
            "and transformer_layer.py unpacks that norm's (output, residual) "
            "tuple and hands the residual to mlp_bda. The gate is two-level: "
            "use_fused_residual = config.fused_residual_rmsnorm and "
            "has_residual (extensions/transformer_engine.py:1046), so with "
            "the flag off the same site builds a plain te.pytorch.RMSNorm. "
            "EXPECT DILUTION, AND SAY SO BESIDE ANY NUMBER. The range "
            "encloses the whole MoE block, so the expert GEMMs may swamp a "
            "norm-plus-add backward fusion and the ratio may land inside the "
            "noise. Report the ffn_norm and moe_residual forward_backward "
            "numbers from the same run next to the span total -- and record "
            "that those two per-scenario numbers no longer line up between "
            "the arms, because the fusion is what moves work across the cut "
            "between them. "
            "NO TITAN ARM, AND THE OMISSION IS A DECLARATION. Every cut in "
            "this range that can publish a cross-engine row already does, "
            "and the two that cannot -- expert_mlp and moe_combine -- are "
            "covered by the expert_combine span. A titan arm here would add "
            "one coarser cross-engine ratio over the whole MoE block, one "
            "compiled graph against six eager modules, which is the "
            "engine-design difference this partition exists to decompose "
            "rather than to restate. "
            "THERE IS NO fp64 REFERENCE, deliberately: this range encloses a "
            "top-k router, so an fp64 truth would have to reproduce the "
            "routing decision from an fp64 norm output, and a near tie would "
            "select a different expert. The fusion's claim is that it changes "
            "the backward implementation and not the arithmetic, so the arm "
            "it must match is the unfused one, and mcore/base is the gate. "
            "THE SPAN-VERSUS-PARTS ROW CARRIES A BIAS AND IT FAVOURS THE "
            "SPAN, AND THIS IS THE LONGEST RANGE DECLARED, SO IT CARRIES THE "
            "MOST OF IT: the parts total pays one host dispatch chain per "
            "enclosed scenario -- six here -- and the span pays one, and "
            "roughly 85% of a kernel number in this repository is host "
            "dispatch rather than device time. Five chains the harness "
            "stopped paying sit inside any ratio below 1.0, and no fusion "
            "removed them. "
            "READ THE TWO ROWS AS TWO DIFFERENT STATISTICS. The "
            "fused-against-base row is a within-span comparison, measured in "
            "one block-major sweep and PAIRED. The span against the parts "
            "sum is UNPAIRED -- roughly 150 workers separate a span "
            "replicate from the part replicate that shares its index -- so "
            "its interval is published as unpaired_ratio_ci_* and may never "
            "be set beside a scenario interval as the same quantity."
        ),
        inputs_builder=(
            "benchmarks.kernel.operations.ffn_norm_to_moe_residual"
            ":ffn_norm_to_moe_residual_inputs"
        ),
        reference_builder=None,
        arms=(
            KernelArm(
                name="mcore/base",
                description=(
                    "megatron-core off a real GPTModel with "
                    "fused_residual_rmsnorm off: pre_mlp_layernorm is a "
                    "plain transformer_engine.pytorch.RMSNorm, and the "
                    "residual reaches mlp_bda as the layer's own "
                    "hidden_states. The unfused side of the flag"
                ),
                builder=(
                    "benchmarks.kernel.operations.ffn_norm_to_moe_residual"
                    ":build_ffn_norm_to_moe_residual_mcore_base"
                ),
                modes=("forward_backward",),
                compiled=False,
                eager_reason=(
                    "megatron compiles no whole transformer layer. The "
                    "@jit_fuser regions inside the range -- the router and "
                    "bias_dropout_add_fused_train -- still compile as their "
                    "own regions, which is megatron's choice and not the "
                    "harness's"
                ),
            ),
            KernelArm(
                name="mcore/fused_residual_rmsnorm",
                description=(
                    "the same six cuts with fused_residual_rmsnorm=True: "
                    "pre_mlp_layernorm becomes TEFusedResidualRMSNorm, "
                    "returns (output, residual), and fuses the residual add "
                    "into the norm's BACKWARD pass. The fusion under test, "
                    "and it exists in backward alone"
                ),
                builder=(
                    "benchmarks.kernel.operations.ffn_norm_to_moe_residual"
                    ":build_ffn_norm_to_moe_residual_mcore_fused"
                ),
                modes=("forward_backward",),
                compiled=False,
                eager_reason=(
                    "the same treatment as the arm it is measured against; "
                    "the flag changes a norm module, not a compile scope"
                ),
                correctness=(
                    FUSED_RESIDUAL_RMSNORM_GATE,
                    FUSED_RESIDUAL_RMSNORM_FORWARD_IS_UNCHANGED,
                ),
            ),
        ),
        baseline_arm="mcore/base",
        # The range holds the dispatch, the experts and the combine, so the
        # span builds routed rows and needs the same even split they need.
        requires_balanced_routing=True,
        comparisons=(("mcore/fused_residual_rmsnorm", "mcore/base"),),
    ),
    scenarios=(
        "ffn_norm",
        "moe_router",
        "dispatch_permute",
        "expert_mlp",
        "moe_combine",
        "moe_residual",
    ),
    parts=(
        SpanParts(arm="mcore/base", parts=("mcore/base",) * 6),
        # The fusion has no arm in any enclosed scenario, because it exists
        # only across the cut between the first and the last. What it
        # replaces is the unfused megatron at every cut it crosses, and a
        # name-based correspondence could not say that.
        SpanParts(
            arm="mcore/fused_residual_rmsnorm", parts=("mcore/base",) * 6
        ),
    ),
)


KERNEL_SPANS: dict[str, KernelSpan] = {
    span.name: span for span in (
        EXPERT_COMBINE,
        ATTN_RESIDUAL_NORM,
        FFN_NORM_TO_MOE_RESIDUAL,
    )
}


def _validate_roster(
    spans: dict[str, KernelSpan], scenarios: dict[str, KernelScenario]
) -> None:
    """Refuse a roster this repository cannot measure, at import.

    Both halves fail here rather than after a GPU has measured every arm of
    a span and of every scenario it encloses.

    The disjointness half is the one a test cannot own. A name in both
    rosters would be measured **twice** in one run -- once as a scenario,
    without its parts, and once as the span, with them -- and ``--scenario``
    and ``--span`` would each accept it. The rule belongs to the registry
    that could break it.
    """
    for name, span in spans.items():
        if name in scenarios:
            raise ValueError(
                f"{name!r} names both a kernel span and a kernel scenario. "
                "The two rosters are disjoint: one run would measure that "
                "name twice, once with its parts total and once without"
            )
        validate_span_parts(span, scenarios)


_validate_roster(KERNEL_SPANS, KERNEL_SCENARIOS)


def kernel_span_by_name(name: str) -> KernelSpan:
    try:
        return KERNEL_SPANS[name]
    except KeyError:
        raise ValueError(
            f"Unknown kernel span {name!r}. "
            f"Available: {', '.join(KERNEL_SPANS) or '(none declared)'}"
        ) from None
