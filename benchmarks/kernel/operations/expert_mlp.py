"""Arm builders for the ``expert_mlp`` kernel scenario.

The routed-expert MLP itself, on both engines, at the level where the two are
substitutable: permuted rows in, expert outputs out.

* TorchTitan: ``self.inner_experts(routed_input_RD,
  num_global_tokens_per_local_expert_e)`` in ``RoutedExperts.forward``
  (``third_party/torchtitan/torchtitan/models/common/moe.py:159``), which is a
  ``GroupedExperts`` (``:35``).
* Megatron-core: ``self.experts(dispatched_input, tokens_per_expert,
  permuted_probs)`` in ``MoELayer.routed_experts_compute``
  (``third_party/Megatron-LM/megatron/core/transformer/moe/moe_layer.py:558``),
  which is a ``TEGroupedMLP``
  (``megatron/core/transformer/moe/experts.py:182``).

**THIS SCENARIO IS WITHIN-ENGINE ONLY AND PUBLISHES NO CROSS-ENGINE RATIO.**
The routing probabilities are applied on opposite sides of this boundary.
Megatron multiplies them **inside** the experts -- ``weighted_bias_swiglu_impl``
folds them into the fused activation kernel (``experts.py:830``) -- while
TorchTitan applies them in combine, which is one scenario later
(``models/common/token_dispatcher.py:168-171``, whose own docstring reads
"Score and scatter_add routed expert outputs"). So megatron's side of this cut
does strictly more work than titan's, and no arrangement of the inputs removes
that: it is where the two engines draw the boundary, not a difference in how
fast they compute one thing. The cross-engine row belongs to the
``expert_combine`` span over scenarios 11 and 12, the smallest enclosure in
which both engines have applied the probabilities exactly once. That span is
separate work and is not declared here.

Three consequences follow, and each is enforced rather than described:

* ``comparisons`` is declared explicitly, and both arms of every published row
  are on one engine.
* There is **no cross-engine ``CorrectnessCheck``** and no correctness
  reference that crosses the engine boundary. The two clusters are
  independent, which is what "within-engine only" means at this cut. Each arm
  is gated against the fp64 truth for **its own** engine's semantics, and the
  output names differ per engine (``out`` against ``out_weighted``) so a check
  cannot silently compare the two. ``run_correctness`` raises on an output
  name an arm does not produce
  (``benchmarks/kernel/engine/correctness.py:43-49``), so the naming is a
  guard and not a convention.
* ``probs`` is an input **only the megatron arms read**. Its presence in the
  shared inputs and its absence from every titan closure is the scenario's
  asymmetry, written down.

**This scenario supersedes the ``swiglu`` scenario once it is registered.**
``swiglu``'s three arms are ``titan``, ``titan/piper_optimized_triton`` and
``titan/piper_optimized_inductor`` here, built the same way from the same
shared fp32 weights, so their numbers carry across. The future tense is
deliberate: these builders are not reachable from
``benchmarks/kernel/registry.py`` yet, because the scenario declaration lands
in its own commit, and until it does ``swiglu`` is still the only scenario
measuring this layer.

**``titan/fused_grouped_experts`` exists because its absence misattributes
credit.** ``FusedGroupedExperts``
(``third_party/torchtitan/torchtitan/overrides/fused_swiglu.py:508``) is
TorchTitan's **own** w13 fusion: the same ``(E, F, 2, D)`` ``w13`` parameter
(``:534``), the same ``reshape(E, F*2, D).transpose(-2, -1)`` and single
``torch._grouped_mm`` (``:557-558``), and the same save/load split hooks
(``:565-579``) as the two Piper arms. So the honest claim for the Piper arms is
"what the combined activation layout buys over titan's own fusion", not "what
fusing w13 buys over unfused experts", and their published opponent is
``titan/fused_grouped_experts`` rather than ``titan``. The ``titan`` row that
the ``swiglu`` scenario published is not declared here; it is recoverable from
the per-replicate samples ``results.json`` keeps, and the
``titan/fused_grouped_experts`` against ``titan`` row is what supplies the
factor it differs by.

**That last row moves TWO axes, and is not a measurement of the w13 fusion
alone.** ``GroupedExperts`` runs **two** ``torch._grouped_mm`` calls and writes
its activation as plain ops -- ``F.silu(gemm(w1)) * gemm(w3)``
(``models/common/moe.py:93-105``). ``FusedGroupedExperts`` runs **one** grouped
GEMM and calls ``silu_and_mul_op`` (``fused_swiglu.py:560``), which is a
``@torch.library.custom_op`` (``:286``). Both arms run under
``torch.compile(fullgraph=True)``, and a custom op is opaque to Inductor where
the anchor's plain ops fuse into their neighbours. So the row is the w13 fusion
**plus a lost activation fusion**, and the two push in opposite directions --
the same mechanism CLAUDE.md records for ``swiglu`` as "the swiglu combined
layout wins eager, loses compiled". A ratio near 1.0 there is therefore
ambiguous between "the fusion bought nothing" and "the fusion gain was
cancelled", and the first reading is the likeliest wrong number this scenario
can publish. Read it as upstream's fused expert layer against upstream's
unfused one, and attribute nothing in it to the GEMM count.

Changing the roster would not fix this, because both arms are real upstream
implementations and the question between them is real. The GEMM count is
separately recoverable, because ``titan/piper_optimized_inductor`` differs from
``FusedGroupedExperts`` **only** in the activation and writes that activation
as the same plain ops the anchor uses: ``titan/piper_optimized_inductor``
against ``titan`` isolates it. That row is not declared, because the printed
table keys its rows by ``(arm, mode)`` and would drop a second row for one arm
in one mode; it is computable from the per-replicate samples.

**The anchor is ``titan``, and that departs from the other cross-engine
scenarios.** ``baseline_arm`` does exactly two things in this tree: it supplies
the default comparison derivation, which this scenario does not use because it
declares its rows, and it decides whose loss costs the scenario -- the merge
writes no results at all when the anchor produced no samples
(``benchmarks/kernel/results/merge.py:311,359``). Here the published rows are
two independent within-engine families, so the anchor is the opponent of
nothing across engines and the only live question is which arm is least likely
to be lost. ``GroupedExperts`` is plain torch plus ``torch._grouped_mm``;
``mcore/base`` needs the megatron submodule on ``sys.path``, a process group,
TransformerEngine, and TE's grouped linear in particular. Losing ``mcore/base``
as an opponent costs its three rows and warns, because the merge tolerates an
absent opponent (``merge.py:460-471``), where losing the anchor writes nothing
at all.

**That protection is narrower than it sounds, and the limit is worth stating
rather than discovering.** It covers a **timing** worker lost after the gates
passed, and nothing earlier. It does **not** cover a TransformerEngine or
megatron failure at *build* time, because ``gate_outputs``
(``benchmarks/kernel/engine/run.py:289-298``) builds every arm of the scenario
in one interpreter with no ``try``/``except``, and ``resolve_arm_skips``
(``benchmarks/kernel/runner.py:281-319``) only pre-skips on
``requires_gcc_toolset``, which none of these eight arms declares. So a TE that
fails to import takes down the correctness pass, no timing worker launches, and
**every** arm lands at ``status: skipped`` -- ``titan`` included. Anchoring on
the titan side does not save the TorchTitan ranking from that; nothing in the
current harness does. Splitting the correctness pass per arm is the work that
would, and CLAUDE.md already records it as the live ``run_correctness_pass``
constraint.

**Neither engine pays a layout conversion, and none is hoisted out of a timed
closure to achieve that.** Megatron flattens ``[s, b, h]`` to ``[s*b, h]`` in
``dispatch_preprocess`` (``token_dispatcher.py:274``) and then permutes, so
``TEGroupedMLP`` receives a 2-D ``(rows, dim)`` tensor; titan's
``RoutedExperts.forward`` views ``[b, l, d]`` as ``[t, d]`` before dispatch
(``moe.py:144``), so ``GroupedExperts`` receives the same 2-D shape. Both arms
take the one canonical ``(rows, dim)`` tensor unchanged.

**The number is device time on both sides, and no device-to-host
synchronization sits inside any timed closure.** ``TEGroupedMLP.forward`` calls
``tokens_per_expert.tolist()`` (``experts.py:769``) and ``SequentialMLP.forward``
does the same (``:1355``), either of which would block on a CUDA tensor. Neither
does here, because megatron's allgather dispatcher has already moved that
tensor to the host: ``dispatch_postprocess`` computes
``self.local_map.sum(dim=0).long().cpu()`` (``token_dispatcher.py:317``) and
hands the CPU result to the experts. So the inputs builder produces **two**
count tensors -- a device ``int32`` one for titan, whose ``torch.cumsum`` feeds
``torch._grouped_mm``'s offsets (``moe.py:81``), and a CPU ``int64`` one for
megatron. Handing megatron the device tensor instead would charge it a sync it
does not pay in production. A small amount of real host work does remain inside
megatron's interval -- a ``tolist`` over ``num_experts`` elements and, on the
ungrouped arm, a ``torch.split`` into that many pieces -- and it is megatron's,
not the harness's.

**There is no ``copy_floor``, and the reason is arithmetic rather than
omission.** At the default workload and the ``normal`` shape the timed region is
three grouped GEMMs: ``2*R*D*2F`` for fc1 plus ``2*R*F*D`` for fc2, about
180 GFLOP against about 300 MB of traffic, an arithmetic intensity near
600 FLOP/byte. An H200's bf16 machine balance is near 200 FLOP/byte, so these
arms are compute-bound by roughly a factor of three and a bandwidth floor would
sit far below all of them and bound nothing. The two sibling GEMM scenarios,
``qkv_prep`` and ``attn_out_proj``, declare no floor for the same reason; the
three that do declare one -- ``qk_norm``, ``ffn_norm`` and ``final_norm`` -- are
norms, where the floor is the only thing that separates "this kernel is slow"
from "this scenario is at bandwidth". What a compute-bound scenario wants
instead is a FLOP roofline, which is a different diagnostic and is not built
here. No arm sets ``bytes_moved``, so the table prints no GB/s column.

**Modes: the megatron arms cannot expose an isolated ``backward``, and the titan
arms still do.** The retained-graph trick re-runs one backward graph many times,
and TransformerEngine's ``GroupedLinear`` backward calls
``clear_tensor_data(*inputmats)``
(``transformer_engine/pytorch/module/grouped_linear.py:1129``), which replaces
each saved input with an empty tensor, so a second pass would read cleared
storage; TE's ``SwiGLU`` operation clears its saved tensors too. The four titan
arms keep the mode, because this scenario publishes no cross-engine row: no
table compares a titan ``backward`` against a megatron one, so dropping it from
both -- which ``qkv_prep``, ``ffn_norm`` and ``attn_out_proj`` do, and must,
because they publish such a row -- would delete the ``swiglu`` scenario's
backward numbers and buy nothing. Megatron's backward cost stays recoverable as
``forward_backward`` minus ``forward``.

**The routing probabilities require grad on the megatron side, deliberately.**
In production they carry a gradient back to the router, and two of the four
megatron arms pay for it: on the ``use_te_activation_func`` and the unfused
paths the probability multiply is an ordinary ``*``, whose backward adds a
reduction over the expert width that a non-requiring input would skip
(``experts.py:825-826`` and ``:878-879``). The fused arm is unaffected either
way, because ``WeightedSwiGLUFunction.backward`` computes both gradients
unconditionally
(``megatron/core/fusions/fused_bias_swiglu.py:202-206``). Requiring the
gradient is therefore the choice that charges every arm what megatron charges
it.

**Memory, and why the huge shape is untested.** The shared fp32 weight state is
three ``(E, F, D)``-sized tensors, and every arm then materializes its own bf16
copy. At ``normal`` that is 168 MiB of fp32 state and about 84 MiB per arm. At
``huge`` (dim 12288, expert width 43008) the same three tensors are 23.6 GiB and
one arm's bf16 weights are 11.8 GiB, on top of a ``build_model`` that allocates
the whole 10.5 B-parameter megatron model before one layer is read out of it,
and on top of an fp64 reference whose weight gradients alone are 47.3 GiB.
Nothing in this scenario has ever run on a GPU at either shape; the ``huge``
figures are arithmetic, not a measurement, and the warning CLAUDE.md already
carries for ``swiglu`` applies here unchanged.

**Every torchtitan, megatron and TransformerEngine import is deferred into the
builder that needs it**, which is the rule across ``operations/``.
``benchmarks.models.piper_qwen3.mcore_profiles`` is the one module-scope
exception: it is torch-free parent-side data, and the three profile deltas
below are derived from it.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    WEIGHT_STD,
    _compile_module,
    _navigate,
    _randn,
    _require_grads,
    _reset_grads,
    initialize_megatron_single_rank,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import (
    BASE,
    McoreProfile,
    derive,
)
from benchmarks.models.piper_qwen3.shape import PiperShape


# The eight arm names, spelled as the partition spells them. The slash says
# which engine an arm is and which profile of that engine, and it reaches a
# filename through ``benchmarks/kernel/runner.py``, which routes it through
# ``schema.fragment_stem``.
MCORE_BASE_ARM = "mcore/base"
MCORE_NO_BIAS_ACTIVATION_FUSION_ARM = "mcore/no_bias_activation_fusion"
MCORE_TE_ACTIVATION_FUNC_ARM = "mcore/te_activation_func"
MCORE_NO_GROUPED_GEMM_ARM = "mcore/no_grouped_gemm"
TITAN_ARM = "titan"
TITAN_FUSED_ARM = "titan/fused_grouped_experts"
TITAN_PIPER_TRITON_ARM = "titan/piper_optimized_triton"
TITAN_PIPER_INDUCTOR_ARM = "titan/piper_optimized_inductor"

# The layer the megatron arms read their expert layer from. Every layer holds
# the same module class at ``moe_layer_freq=1`` and the weights are overwritten
# from the shared seeded tensors, so the index is arbitrary; it is fixed here
# so the provenance line can name it.
MCORE_LAYER = 0

# The two parameter names the cross-engine weight map yields for one expert of
# the grouped layer, and the component tag it files them under.
# ``benchmarks/models/piper_qwen3/megatron_weights.py:179-190`` is the
# authority on the correspondence -- ``linear_fc1.weight{e}`` is
# ``cat([w1[e], w3[e]])`` and ``linear_fc2.weight{e}`` is ``w2[e]`` -- and
# ``tests/test_kernel_expert_mlp.py`` pins that it still yields exactly these
# names and exactly these tensors. This module restates the concatenation
# rather than calling ``transfer_weights``, which needs a whole titan state
# dict for a whole model; a concatenation is not an interleave, and the test
# is what ties the two together.
MCORE_FC1_WEIGHT_NAME = (
    "decoder.layers.{layer}.mlp.experts.linear_fc1.weight{expert}"
)
MCORE_FC2_WEIGHT_NAME = (
    "decoder.layers.{layer}.mlp.experts.linear_fc2.weight{expert}"
)
MCORE_WEIGHT_COMPONENT = "experts"

# The three keys of the shared fp32 weight state. They are the names
# ``GroupedExperts`` exposes, and the names all three fused titan modules
# accept through their ``load_state_dict`` merge hooks, which is what lets one
# state dict load into all four titan arms.
TITAN_STATE_KEYS = ("w1_EFD", "w2_EDF", "w3_EFD")

# The expert-module class each megatron arm must build. ``moe_grouped_gemm``
# decides it, through ``TESpecProvider.grouped_mlp_modules``
# (``megatron/core/extensions/transformer_engine_spec_provider.py:76-108``).
GROUPED_EXPERT_CLASS = "TEGroupedMLP"
UNGROUPED_EXPERT_CLASS = "SequentialMLP"


def mcore_experts_path(layer: int = MCORE_LAYER) -> str:
    """The attribute path from a ``GPTModel`` down to the expert layer.

    Derived from ``MCORE_FC1_WEIGHT_NAME`` so the navigation this module walks
    and the weight map the gates rely on cannot drift apart.
    """
    name = MCORE_FC1_WEIGHT_NAME.format(layer=layer, expert=0)
    suffix = ".linear_fc1.weight0"
    assert name.endswith(suffix)
    return name[: -len(suffix)]


# ---------------------------------------------------------------------------
# The three megatron variants, as deltas
# ---------------------------------------------------------------------------
#
# Declared here rather than added to ``MCORE_PROFILES``, for the reason
# ``benchmarks/kernel/operations/cross_entropy.py`` gives for its own two: they
# are this scenario's roster, and the e2e megatron arm and every other
# cross-engine scenario run ``base``, so a registry entry would suggest
# otherwise.

NO_BIAS_ACTIVATION_FUSION_PROFILE: McoreProfile = derive(
    BASE,
    name="no_bias_activation_fusion",
    description=(
        "megatron with the fused SwiGLU turned off: the expert activation "
        "falls to the chunk/silu/mul path plus a separate probability "
        "multiply, which is the accidental handicap the base profile's own "
        "comments record as having cost 11.9 GPU ms/step end to end"
    ),
    config_overrides={"bias_activation_fusion": False},
)

# A TWO-flag delta, and it has to be. ``transformer_config.py:2166-2171``
# raises when ``bias_activation_fusion`` and ``use_te_activation_func`` are
# both true, and the base profile sets the first one on. So this profile
# cannot be written as the single flag its name suggests.
#
# It is delivered by **config alone**. ``get_moe_module_spec_for_backend``
# accepts ``use_te_activation_func`` and never reads it
# (``megatron/core/models/gpt/moe_module_specs.py:65-96``): it calls
# ``backend.grouped_mlp_modules(...)``, and ``TESpecProvider`` always wires
# ``activation_func=self.activation_func()`` into the grouped submodules
# (``transformer_engine_spec_provider.py:78-85``). The TE activation module is
# therefore present in the spec at every profile, and
# ``TEGroupedMLP.__init__`` is what picks it, on the config flag
# (``experts.py:230-233``). ``_assert_te_activation_ran`` below proves it was
# picked *and* that it runs.
TE_ACTIVATION_FUNC_PROFILE: McoreProfile = derive(
    BASE,
    name="te_activation_func",
    description=(
        "megatron running TransformerEngine's own SwiGLU operation instead "
        "of megatron's fused weighted-SwiGLU kernel. A two-flag delta, "
        "because TransformerConfig refuses use_te_activation_func beside "
        "bias_activation_fusion. The mechanism is a kernel COUNT difference "
        "as much as a kernel speed one: the TE path applies the routing "
        "probabilities as a separate multiply, where the megatron kernel "
        "folds them into the activation"
    ),
    config_overrides={
        "use_te_activation_func": True,
        "bias_activation_fusion": False,
    },
)

# Delivered by **config alone on this rev**, and the claim that it needs a spec
# kwarg is stale. That claim was true while
# ``benchmarks/models/piper_qwen3/megatron_model.py`` called the inner factory
# ``get_gpt_layer_with_transformer_engine_spec`` directly, which receives no
# config. The builder now derives the spec with ``get_gpt_decoder_block_spec``,
# and ``get_gpt_decoder_layer_specs`` reads ``config.moe_grouped_gemm`` off the
# built config and passes it down (``gpt_layer_specs.py:593``). The field is
# live here, not inert -- but it is read twice, once to choose the class and
# once at run time, and megatron checks the agreement nowhere, so
# ``_assert_mcore_experts`` proves the class actually changed rather than
# trusting the flag.
NO_GROUPED_GEMM_PROFILE: McoreProfile = derive(
    BASE,
    name="no_grouped_gemm",
    description=(
        "megatron with one GEMM pair per expert instead of the grouped "
        "kernel: SequentialMLP over four TE linear MLPs, against "
        "TEGroupedMLP's single grouped launch"
    ),
    config_overrides={"moe_grouped_gemm": False},
)


# ---------------------------------------------------------------------------
# Shared inputs
# ---------------------------------------------------------------------------


@dataclass
class ExpertMlpInputs:
    """One permuted row batch, one weight triple, and the two count forms.

    ``probs`` is read by the megatron arms alone. Every titan closure ignores
    it, which is the boundary asymmetry this scenario exists to keep honest
    rather than to hide.

    ``counts_device`` and ``counts_host`` hold the same ``num_experts``
    numbers and differ
    only in device and dtype. Both are built here, never inside a timed
    closure, and the module docstring records why megatron gets the host one.
    """

    x: torch.Tensor  # (R, D) bf16, the permuted rows
    grad_out: torch.Tensor  # (R, D) bf16
    probs: torch.Tensor  # (R,) fp32 routing probabilities, megatron only
    counts_device: torch.Tensor  # (E,) int32, on the run device -- titan
    counts_host: torch.Tensor  # (E,) int64, on the CPU -- megatron
    stock_state: dict[str, torch.Tensor]  # w1_EFD / w2_EDF / w3_EFD, fp32


def _require_even_routing(shape: PiperShape, workload: KernelWorkload) -> int:
    """The rows per expert, or a named failure.

    ``KernelScenario.requires_balanced_routing`` makes both the runner and
    ``engine.run._prepare`` refuse an uneven pair before this builder is
    reached. This raise is for every other caller -- a direct
    ``python -m benchmarks.kernel.worker`` invocation, a test, a future
    span builder -- because the failure it prevents is one that *measures*
    rather than one that stops: an even split that does not cover the rows it
    was built beside hands some expert rows nobody's GEMM reads, and every
    number from that run is a fraction of the work its label claims. Capping
    or rounding the split would be the same wrongness spelled more politely.
    """
    rows = workload.batch * workload.seq_len * shape.top_k
    if rows % shape.num_experts:
        raise ValueError(
            f"expert_mlp: {rows} routed rows (batch {workload.batch} x seq "
            f"{workload.seq_len} x top_k {shape.top_k}) do not divide evenly "
            f"among {shape.num_experts} experts; {rows // shape.num_experts} "
            f"rows per expert would cover "
            f"{shape.num_experts * (rows // shape.num_experts)} "
            f"of them and leave {rows % shape.num_experts} unread"
        )
    return rows // shape.num_experts


def expert_mlp_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> ExpertMlpInputs:
    rows = workload.batch * workload.seq_len * shape.top_k
    per_expert = _require_even_routing(shape, workload)
    hidden = shape.moe_hidden_dim
    stock_state = {
        "w1_EFD": _randn(
            (shape.num_experts, hidden, shape.dim),
            device,
            generator,
            torch.float32,
            WEIGHT_STD,
        ),
        "w2_EDF": _randn(
            (shape.num_experts, shape.dim, hidden),
            device,
            generator,
            torch.float32,
            WEIGHT_STD,
        ),
        "w3_EFD": _randn(
            (shape.num_experts, hidden, shape.dim),
            device,
            generator,
            torch.float32,
            WEIGHT_STD,
        ),
    }
    # One routing probability per routed row, drawn as a real top-k softmax
    # would produce them: ``top_k`` logits per token, softmaxed over the
    # selected k, so each token's scores sum to one. That is megatron's
    # post-softmax top-k (``moe_router_pre_softmax=False`` in the base
    # profile) and Qwen3's renormalized top-k, which are algebraically the
    # same thing. A constant would let an arm that dropped the probabilities
    # pass every gate, and a uniform draw would not have the spread a softmax
    # pair has. The row-to-expert assignment is still the synthetic even split
    # this scenario hands both engines, not a routing map: values reach timing
    # only through this multiply, and CLAUDE.md's rule is that the two engines
    # get the same synthetic counts rather than matched router weights.
    logits = torch.randn(
        (rows // shape.top_k, shape.top_k),
        device=device,
        generator=generator,
        dtype=torch.float32,
    )
    probs = torch.softmax(logits, dim=-1).reshape(rows).contiguous()
    return ExpertMlpInputs(
        x=_randn((rows, shape.dim), device, generator),
        grad_out=_randn((rows, shape.dim), device, generator),
        # fp32, and the route to that dtype is indirect enough to spell out.
        # ``moe_router_dtype="fp32"`` in the base profile sets the **logit**
        # dtype, not the probability dtype:
        # ``megatron/core/transformer/moe/router.py:106-111`` reads the field
        # and hands ``router_dtype`` to ``router_gating_linear``, so ``logits``
        # come back fp32. The probabilities inherit it at ``:295``, which is
        # ``torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(
        # logits)`` -- the softmax is fp32 unconditionally and ``type_as``
        # casts back to the logit dtype, which is fp32 here. There is no
        # downcast on the path to the dispatcher.
        probs=probs,
        counts_device=torch.full(
            (shape.num_experts,), per_expert, device=device, dtype=torch.int32
        ),
        counts_host=torch.full(
            (shape.num_experts,), per_expert, device="cpu", dtype=torch.int64
        ),
        stock_state=stock_state,
    )


# ---------------------------------------------------------------------------
# The fp64 reference, in both engines' semantics
# ---------------------------------------------------------------------------


def _fp64_expert_mlp(
    shape: PiperShape, inputs: ExpertMlpInputs, weighted: bool
) -> dict[str, torch.Tensor]:
    """One expert MLP in fp64, per expert slice, with its gradients.

    ``weighted`` selects megatron's semantics: the routing probability scales
    the **activation**, between fc1 and fc2, which is where
    ``weighted_bias_swiglu_impl`` applies it (``experts.py:830``). Applying it
    to the output instead would be algebraically equal -- fc2 is a bias-free
    linear and the probability is a per-row scalar -- but it would be the
    reference rewriting the implementation, and the point of an fp64 truth is
    that it does not.

    Every parameter is quantized to bf16 before it is promoted to fp64, which
    is ``qkv_reference``'s and ``ffn_norm_reference``'s rule: a truth built
    from the unrounded fp32 values would charge every arm for an input cast
    none of them performs. ``x`` is already bf16, so its promotion is exact,
    and ``probs`` is fp32 on both the arms and here, so it is not quantized.
    """
    per_expert = inputs.x.shape[0] // shape.num_experts

    def leaf(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(torch.bfloat16).double().detach().requires_grad_()

    x = inputs.x.double().detach().requires_grad_()
    w1 = leaf(inputs.stock_state["w1_EFD"])
    w2 = leaf(inputs.stock_state["w2_EDF"])
    w3 = leaf(inputs.stock_state["w3_EFD"])
    probs = inputs.probs.double().detach().requires_grad_()

    pieces = []
    for expert in range(shape.num_experts):
        rows = slice(expert * per_expert, (expert + 1) * per_expert)
        x_e = x[rows]
        gate = x_e @ w1[expert].transpose(-2, -1)
        up = x_e @ w3[expert].transpose(-2, -1)
        activation = torch.nn.functional.silu(gate) * up
        if weighted:
            activation = activation * probs[rows].unsqueeze(-1)
        pieces.append(activation @ w2[expert].transpose(-2, -1))
    out = torch.cat(pieces, dim=0)
    torch.autograd.backward(out, inputs.grad_out.double())

    suffix = "_weighted" if weighted else ""
    result = {
        f"out{suffix}": out.detach(),
        f"x_grad{suffix}": x.grad,
        f"w1_grad{suffix}": w1.grad,
        f"w2_grad{suffix}": w2.grad,
        f"w3_grad{suffix}": w3.grad,
    }
    if weighted:
        result["probs_grad_weighted"] = probs.grad
    return result


def expert_mlp_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: ExpertMlpInputs
) -> dict[str, torch.Tensor]:
    """fp64 truth for both engines' semantics, under two sets of names.

    Two branches rather than one, because the two engines really do compute
    two functions here. The titan arms are gated on the unweighted names and
    the megatron arms on the ``_weighted`` ones, so a check can never quietly
    compare one engine's output against the other's truth.

    The branches run one at a time and the first is released before the second
    builds, because an fp64 expert MLP's saved activations are the largest
    thing this scenario allocates: a ``(rows, 2 * moe_hidden_dim)`` fp64
    intermediate is 470 MiB at the default workload and the ``normal`` shape.

    ``probs_grad_weighted`` is the one output that has no unweighted twin, and
    it is the sharpest check the megatron arms get: it is exactly zero-valued
    for any implementation that dropped the routing probabilities, and it is
    the quantity two of the four megatron arms compute with an extra reduction
    the fused arm folds away.
    """
    weighted = _fp64_expert_mlp(shape, inputs, weighted=True)
    gc.collect()
    plain = _fp64_expert_mlp(shape, inputs, weighted=False)
    return {**weighted, **plain}


# ---------------------------------------------------------------------------
# TorchTitan arms
# ---------------------------------------------------------------------------


def _titan_arm(
    name: str,
    module: nn.Module,
    compiled: Callable[..., torch.Tensor],
    inputs: ExpertMlpInputs,
    weight_grads: Callable[[nn.Module], dict[str, torch.Tensor | None]],
) -> BuiltArm:
    """The three timed closures every titan arm shares.

    All four arms run this same code over their own module, so the comparison
    measures the four modules and nothing about how each arm was written.

    ``module`` is the module *inside* the compile wrapper, because that is
    where the parameters whose gradients are cleared and read actually live.

    **No closure here reads ``inputs.probs``.** TorchTitan's ``inner_experts``
    never sees the routing probabilities; its dispatcher applies them in
    combine. That absence is the scenario's asymmetry and
    ``tests/test_kernel_expert_mlp.py`` pins it by rebuilding an arm over
    different probabilities and asserting the outputs do not move.
    """
    forward_leaf = inputs.x.clone().requires_grad_()
    backward_leaf = inputs.x.clone().requires_grad_()
    round_trip_leaf = inputs.x.clone().requires_grad_()
    check_leaf = inputs.x.clone().requires_grad_()
    retained = compiled(backward_leaf, inputs.counts_device)

    def forward():
        return compiled(forward_leaf, inputs.counts_device)

    def backward() -> None:
        _reset_grads(backward_leaf, module)
        torch.autograd.backward(retained, inputs.grad_out, retain_graph=True)

    def forward_backward() -> None:
        _reset_grads(round_trip_leaf, module)
        out = compiled(round_trip_leaf, inputs.counts_device)
        torch.autograd.backward(out, inputs.grad_out)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(check_leaf, module)
        out = compiled(check_leaf, inputs.counts_device)
        torch.autograd.backward(out, inputs.grad_out)
        return _require_grads(
            name,
            {
                "out": out.detach(),
                "x_grad": check_leaf.grad,
                **weight_grads(module),
            },
        )

    return BuiltArm(
        name=name,
        calls={
            "forward": forward,
            "backward": backward,
            "forward_backward": forward_backward,
        },
        correctness_outputs=correctness_outputs,
        notes={"engine": "torchtitan", "module": type(module).__name__},
    )


def _stock_weight_grads(module: nn.Module) -> dict[str, torch.Tensor | None]:
    return {
        "w1_grad": module.w1_EFD.grad,
        "w2_grad": module.w2_EDF.grad,
        "w3_grad": module.w3_EFD.grad,
    }


def _fused_weight_grads(module: nn.Module) -> dict[str, torch.Tensor | None]:
    """Unpack a ``(E, F, 2, D)`` w13 gradient into the stock layout's names.

    All three fused titan modules build ``w13`` with
    ``torch.stack([w1_EFD, w3_EFD], dim=2)`` in their load hooks -- upstream's
    at ``overrides/fused_swiglu.py:577`` and the two Piper ones at
    ``components/swiglu/combined_swiglu.py:397`` -- so slice 0 is w1 and slice
    1 is w3 for all of them. Without this unpacking the gates would compare
    nothing at all: the fused arms hold no parameter named ``w1_EFD``.
    """
    grad = module.w13.grad
    if grad is None:
        return {"w1_grad": None, "w2_grad": module.w2_EDF.grad, "w3_grad": None}
    return {
        "w1_grad": grad[:, :, 0, :],
        "w2_grad": module.w2_EDF.grad,
        "w3_grad": grad[:, :, 1, :],
    }


def _build_titan_module(
    config_cls, shape: PiperShape, inputs: ExpertMlpInputs
) -> nn.Module:
    """One expert layer, loaded from the shared fp32 state, then cast.

    The load order matters and is the same one ``swiglu`` and ``qkv_prep``
    use: build in fp32, load the fp32 shared values, then cast to bf16 in one
    step, so every arm rounds the same values once. The three fused modules
    see unquantized values in their ``load_state_dict`` merge hooks, which is
    what makes their ``w13`` the bf16 rounding of the same numbers the stock
    arm holds in two parameters.
    """
    module = config_cls.Config(
        dim=shape.dim,
        hidden_dim=shape.moe_hidden_dim,
        num_experts=shape.num_experts,
    ).build()
    module.to(inputs.x.device)
    module.load_state_dict(inputs.stock_state)
    module.to(torch.bfloat16)
    return module


def build_expert_mlp_titan(
    shape: PiperShape, workload: KernelWorkload, inputs: ExpertMlpInputs
) -> BuiltArm:
    """TorchTitan's ``GroupedExperts``: separate w1 and w3 grouped GEMMs.

    The module ``_build_qwen3_moe_layers`` puts behind ``inner_experts`` for
    every block, built here from its own config node rather than extracted
    from a ``Trainer.Config``: the node takes nothing but the three geometry
    numbers, and this arm overwrites the parameters it initializes anyway.

    Compiled with ``fullgraph=True``, because that is what a titan module
    faces end to end. Every megatron arm runs eager, because megatron compiles
    no whole layer; this scenario publishes no row between the two, so no
    published ratio here compares two compile treatments.
    """
    from torchtitan.models.common.moe import GroupedExperts

    module = _build_titan_module(GroupedExperts, shape, inputs)
    return _titan_arm(
        TITAN_ARM, module, _compile_module(module), inputs, _stock_weight_grads
    )


def build_expert_mlp_titan_fused_grouped_experts(
    shape: PiperShape, workload: KernelWorkload, inputs: ExpertMlpInputs
) -> BuiltArm:
    """TorchTitan's **own** w13 fusion, and the anchor the Piper arms modified.

    ``FusedGroupedExperts`` (``overrides/fused_swiglu.py:508``) holds the same
    ``(E, F, 2, D)`` ``w13`` parameter, runs the same single
    ``torch._grouped_mm`` over it, and carries the same save/load split hooks
    as both Piper arms. What it does differently is the activation: it unbinds
    the combined result into two ``(R, F)`` tensors and calls torchtitan's own
    ``silu_and_mul`` custom op (``:559-560``).

    One residual difference, recorded because "only the activation" rests on
    it: upstream's ``silu_and_mul_op`` takes ``offsets_E`` and skips the
    inactive capacity-padding rows (``fused_swiglu.py:514-515``), and the Piper
    Inductor variant passes none, because plain ops have nowhere to put them.
    At this scenario's even split there are no padding rows, so the difference
    is numerically inert here. It is part of the activation's signature rather
    than something outside it, and an uneven split would make it live.

    Its existence is the correction the roster needs. Without it, both Piper
    arms would be published against ``titan`` and would be credited with a w13
    fusion TorchTitan already ships. Note that this contradicts CLAUDE.md,
    which currently records ``torchtitan.overrides.fused_swiglu`` as "no longer
    benchmarked"; that sentence stops being true with this arm.
    """
    from torchtitan.overrides.fused_swiglu import FusedGroupedExperts

    module = _build_titan_module(FusedGroupedExperts, shape, inputs)
    return _titan_arm(
        TITAN_FUSED_ARM,
        module,
        _compile_module(module),
        inputs,
        _fused_weight_grads,
    )


def build_expert_mlp_titan_piper_optimized_triton(
    shape: PiperShape, workload: KernelWorkload, inputs: ExpertMlpInputs
) -> BuiltArm:
    """The Piper layer that keeps the combined ``[R, 2F]`` activation tensor.

    Same fused w13 GEMM as ``titan/fused_grouped_experts``, and one difference
    from it: the custom Triton op consumes the combined tensor directly
    instead of unbinding it into gate and up
    (``components/swiglu/combined_swiglu.py:377``), and its backward returns
    one interleaved gradient instead of two. Its published opponent is
    ``titan/fused_grouped_experts`` for exactly that reason.
    """
    from benchmarks.models.piper_qwen3.components.swiglu.combined_swiglu import (
        CombinedSwiGLUFusedGroupedExperts,
    )

    module = _build_titan_module(
        CombinedSwiGLUFusedGroupedExperts, shape, inputs
    )
    return _titan_arm(
        TITAN_PIPER_TRITON_ARM,
        module,
        _compile_module(module),
        inputs,
        _fused_weight_grads,
    )


def build_expert_mlp_titan_piper_optimized_inductor(
    shape: PiperShape, workload: KernelWorkload, inputs: ExpertMlpInputs
) -> BuiltArm:
    """The Piper layer with the activation left to Inductor.

    Same fused w13 GEMM again, and the activation written as plain ops so
    Inductor can fuse it into its neighbours
    (``components/swiglu/combined_swiglu.py:450-451``). It repeats
    ``FusedGroupedExperts``'s own ``reshape(-1, F, 2).unbind(-1)``, which is
    why ``titan/fused_grouped_experts`` is its opponent: the two differ in the
    activation implementation and in nothing else.
    """
    from benchmarks.models.piper_qwen3.components.swiglu.combined_swiglu import (
        InductorSwiGLUFusedGroupedExperts,
    )

    module = _build_titan_module(
        InductorSwiGLUFusedGroupedExperts, shape, inputs
    )
    return _titan_arm(
        TITAN_PIPER_INDUCTOR_ARM,
        module,
        _compile_module(module),
        inputs,
        _fused_weight_grads,
    )


# ---------------------------------------------------------------------------
# Megatron-core arms
# ---------------------------------------------------------------------------


def mcore_expert_parameters(
    experts: Any, shape: PiperShape
) -> list[tuple[torch.nn.Parameter, torch.nn.Parameter]]:
    """The ``(fc1, fc2)`` weight pair of each local expert, either class.

    ``TEGroupedMLP`` holds one grouped linear whose per-expert parameters are
    named ``weight0`` .. ``weight{E-1}``
    (``transformer_engine/pytorch/module/grouped_linear.py:1348-1349``), which
    is the naming the cross-engine weight map already uses.
    ``SequentialMLP`` holds ``E`` separate ``MLP`` modules under
    ``local_experts`` (``megatron/core/transformer/moe/experts.py:1294-1305``),
    each with one ``linear_fc1.weight``.

    Both spellings are read here so the ungrouped arm can be loaded from the
    same shared state as the grouped ones. A ``None`` parameter raises rather
    than returning: TE's ``single_grouped_weight`` option replaces
    ``weight{i}`` with one packed ``weight`` and sets each of them to ``None``
    (``grouped_linear.py:1474-1482``), and an arm loaded through a ``None``
    would raise inside a timed closure instead of at build.
    """
    pairs: list[tuple[torch.nn.Parameter, torch.nn.Parameter]] = []
    grouped = hasattr(experts, "linear_fc1")
    for expert in range(shape.num_experts):
        if grouped:
            fc1 = getattr(experts.linear_fc1, f"weight{expert}", None)
            fc2 = getattr(experts.linear_fc2, f"weight{expert}", None)
        else:
            local = experts.local_experts[expert]
            fc1 = getattr(local.linear_fc1, "weight", None)
            fc2 = getattr(local.linear_fc2, "weight", None)
        if fc1 is None or fc2 is None:
            raise RuntimeError(
                f"expert_mlp: expert {expert} exposes no fc1/fc2 weight pair "
                f"on {type(experts).__name__}. TransformerEngine's "
                "single_grouped_weight option packs the per-expert weights "
                "into one parameter and leaves weight{i} as None; this arm "
                "cannot be loaded from the shared state through it"
            )
        pairs.append((fc1, fc2))
    return pairs


def _assert_mcore_experts(
    arm: str, experts: Any, profile: McoreProfile, shape: PiperShape
) -> dict[str, Any]:
    """Refuse to time an expert layer that is not the one the arm claims.

    Every check **raises**. ``BuiltArm.notes`` reaches no artifact -- neither
    ``results.json`` nor ``manifest.json`` carries it -- so a fact recorded
    there is a fact no reader sees, and a guard that cannot raise is not a
    guard.

    Each entry names a way this arm could measure something other than what
    its label says, and none of them can be caught by a correctness gate,
    because every one of them is numerically valid:

    * The **class**, which is what ``no_grouped_gemm`` changes and what
      ``mcore/base`` depends on. It is also the one guard against a silent
      TransformerEngine fallback: ``TESpecProvider.grouped_mlp_modules``
      builds ``SequentialMLP`` when ``TEColumnParallelGroupedLinear`` is
      ``None`` (``transformer_engine_spec_provider.py:78-108``), so a TE
      without the grouped linear would make ``mcore/base`` and
      ``mcore/no_grouped_gemm`` the *same module* and publish a ratio of one
      as "the grouped kernel buys nothing".
    * ``moe_grouped_gemm`` and ``use_te_activation_func`` on the built config,
      because ``mcore_profiles.FUSION_FIELDS`` covers neither, so
      ``declared_mismatches`` cannot see a delta that failed to take.
    * ``bias_activation_fusion``, in both directions, through
      ``declared_mismatches``. On the TE-activation arm it is the second half
      of a two-flag delta, and an arm that set only the first flag would build
      a config ``transformer_config.py:2166-2171`` refuses.
    * ``moe_apply_probs_on_input``, because it moves the probability multiply
      to the front of fc1 (``experts.py:783-791``) and resets the
      probabilities to ones, which is a different function and a different
      kernel count.
    * ``moe_mlp_glu_interleave_size``, because it makes the activation read an
      interleaved gate/up layout (``experts.py:809-813``) that the shared
      weights are not in.
    * ``use_transformer_engine_op_fuser``, because ``TEGroupedMLP.forward``
      then takes ``_fused_forward`` (``experts.py:753-760``), which is a
      different implementation and not this arm.
    * ``fp8`` and ``fp4``, because either adds quantization padding inside the
      timed call (``experts.py:772-780``).
    * ``moe_latent_size``, because it puts a down/up projection pair around
      the expert MLP, so the timed call is no longer the layer this arm names.
    * The **local-expert roster**, in both directions: a grouped layer that
      exposes a ``local_experts`` list is not the class it claims, and an
      ungrouped one holding a different count would run a different number of
      GEMM pairs than its label.
    * The activation recompute, because it wraps the activation in a
      checkpoint that re-runs inside backward (``experts.py:882-893``).
    * ``num_local_experts``, because an expert-parallel split would make this
      arm measure a fraction of the layer under the whole layer's label.
    * The weight shapes, because a wrong navigation would hand back another
      linear of the same model.
    """
    config = experts.config

    from benchmarks.models.piper_qwen3.mcore_profiles import (
        FUSION_FIELDS,
        declared_mismatches,
    )

    built = {name: getattr(config, name, None) for name in FUSION_FIELDS}
    wrong = declared_mismatches(profile, built)
    if wrong:
        raise RuntimeError(
            f"{arm}: megatron profile {profile.name!r} did not take: "
            + "; ".join(wrong)
            + " -- see benchmarks/models/piper_qwen3/mcore_profiles.py"
        )

    grouped = bool(profile.config_overrides["moe_grouped_gemm"])
    te_activation = bool(
        profile.config_overrides.get("use_te_activation_func", False)
    )
    for name, declared in (
        ("moe_grouped_gemm", grouped),
        ("use_te_activation_func", te_activation),
    ):
        actual = bool(getattr(config, name, False))
        if actual != declared:
            raise RuntimeError(
                f"{arm}: profile {profile.name!r} declares {name}="
                f"{declared!r} and the built config has {actual!r}. "
                "mcore_profiles.FUSION_FIELDS does not cover this flag, so "
                "declared_mismatches cannot see it"
            )

    expected_class = GROUPED_EXPERT_CLASS if grouped else UNGROUPED_EXPERT_CLASS
    actual_class = type(experts).__name__
    if actual_class != expected_class:
        raise RuntimeError(
            f"{arm}: the expert layer is {actual_class}, not {expected_class}. "
            "The class comes from the layer spec, which reads "
            "config.moe_grouped_gemm, and TESpecProvider.grouped_mlp_modules "
            "falls back to SequentialMLP when TEColumnParallelGroupedLinear "
            "is absent -- so a grouped arm that built the ungrouped class "
            "would publish an identical implementation under two labels"
        )
    if grouped:
        if hasattr(experts, "local_experts"):
            raise RuntimeError(
                f"{arm}: a grouped expert layer holds no local_experts list"
            )
    else:
        local = getattr(experts, "local_experts", None)
        if local is None or len(local) != shape.num_experts:
            raise RuntimeError(
                f"{arm}: SequentialMLP holds "
                f"{0 if local is None else len(local)} local experts, "
                f"expected {shape.num_experts}; the arm would measure a "
                "different number of GEMM pairs than its label claims"
            )

    activation = getattr(experts, "activation_func", None)
    if te_activation:
        if not isinstance(activation, nn.Module):
            raise RuntimeError(
                f"{arm}: activation_func is {type(activation).__name__}, not "
                "a module. TEGroupedMLP picks the TE activation only when "
                "use_te_activation_func is set AND the spec carries one "
                "(experts.py:230-233); this arm measures the plain path "
                "under the TE label"
            )
        origin = type(activation).__module__
        if not origin.startswith("transformer_engine"):
            raise RuntimeError(
                f"{arm}: activation_func comes from {origin!r}, not from "
                "transformer_engine; the arm's whole claim is that TE's own "
                "SwiGLU operation runs"
            )
        if bool(getattr(config, "bias_activation_fusion", False)):
            raise RuntimeError(
                f"{arm}: bias_activation_fusion is still on beside "
                "use_te_activation_func. This is a two-flag delta and "
                "transformer_config.py:2166-2171 refuses the pair"
            )
    elif isinstance(activation, nn.Module):
        raise RuntimeError(
            f"{arm}: activation_func is the module "
            f"{type(activation).__name__}, but this arm declares the plain "
            "path. megatron's fused branch tests activation_func == F.silu "
            "(experts.py:828) and raises when it is not"
        )

    for name in (
        "moe_apply_probs_on_input",
        "moe_mlp_glu_interleave_size",
        "use_transformer_engine_op_fuser",
        "fp8",
        "fp4",
        "moe_latent_size",
    ):
        value = getattr(config, name, None)
        if value:
            raise RuntimeError(
                f"{arm}: config.{name} is {value!r}; see this guard's "
                "docstring for what each of these moves inside the timed call"
            )
    modules = getattr(config, "recompute_modules", None) or ()
    if getattr(config, "recompute_granularity", None) == "selective" and (
        "moe_act" in modules
    ):
        raise RuntimeError(
            f"{arm}: the activation recompute is on, so the timed call holds "
            "a checkpoint that re-runs the activation inside backward"
        )
    if int(getattr(experts, "num_local_experts", 0)) != shape.num_experts:
        raise RuntimeError(
            f"{arm}: the layer holds "
            f"{getattr(experts, 'num_local_experts', None)} local experts and "
            f"the shape declares {shape.num_experts}; an expert-parallel "
            "split would measure a fraction of the layer"
        )

    hidden = shape.moe_hidden_dim
    for expert, (fc1, fc2) in enumerate(
        mcore_expert_parameters(experts, shape)
    ):
        if tuple(fc1.shape) != (2 * hidden, shape.dim):
            raise RuntimeError(
                f"{arm}: expert {expert} fc1 weight is {tuple(fc1.shape)}, "
                f"expected {(2 * hidden, shape.dim)}. The gated width is "
                "doubled once, and the shared state supplies both halves"
            )
        if tuple(fc2.shape) != (shape.dim, hidden):
            raise RuntimeError(
                f"{arm}: expert {expert} fc2 weight is {tuple(fc2.shape)}, "
                f"expected {(shape.dim, hidden)}"
            )
    return {
        "engine": "megatron",
        "profile": profile.name,
        "module": actual_class,
        "moe_grouped_gemm": grouped,
        "use_te_activation_func": te_activation,
    }


def _assert_te_activation_ran(
    arm: str, experts: Any, inputs: ExpertMlpInputs
) -> None:
    """Refuse the TE-activation arm unless the TE module actually runs.

    ``_assert_mcore_experts`` proves the module was *picked*. This proves it
    is *reached*, which is a different claim: the branch that calls it is
    chosen inside ``bias_act_func`` on ``config.use_te_activation_func``
    (``experts.py:815``), and a future rev could route past it while leaving
    the attribute in place.

    A forward hook is what proves it, rather than a marker kernel name. The
    two paths are numerically equal, so no correctness gate can separate them,
    and TE generates its activation kernel names, so
    ``_assert_kernel_marker``'s greppable-name trick has nothing to grep for
    that this module can name from source. A hook needs no name and no
    profiler.

    The probe runs once, at build time, outside every timed region --
    ``ffn_norm``'s ``_require_a_real_norm`` probes the same way and for the
    same reason.
    """
    fired: list[bool] = []
    handle = experts.activation_func.register_forward_hook(
        lambda *_: fired.append(True)
    )
    try:
        with torch.no_grad():
            experts(inputs.x, inputs.counts_host, inputs.probs)
    finally:
        handle.remove()
    if not fired:
        raise RuntimeError(
            f"{arm}: the TransformerEngine activation module never ran "
            "during a probe forward. TEGroupedMLP holds it, but "
            "bias_act_func reached another branch, so this arm measures "
            "megatron's own activation under a TransformerEngine label"
        )


def _mcore_weight_grads(
    experts: Any, shape: PiperShape
) -> dict[str, torch.Tensor | None]:
    """Megatron's expert weight gradients, back in TorchTitan's layout.

    The inverse of the cross-engine map's ``cat([w1[e], w3[e]], dim=0)``
    (``benchmarks/models/piper_qwen3/megatron_weights.py:184``): the first
    ``moe_hidden_dim`` rows of each expert's fc1 gradient are w1's and the rest
    are w3's, because megatron's gated activation splits the fc1 output into
    halves (``experts.py:866`` chunks it, and TE's own SwiGLU documents the
    same convention). Canonicalizing here is what lets one fp64 truth serve
    both engines' weight-gradient gates, and it is what would catch a row order
    that silently swapped the gate and the up projection.
    """
    hidden = shape.moe_hidden_dim
    pairs = mcore_expert_parameters(experts, shape)
    if any(fc1.grad is None or fc2.grad is None for fc1, fc2 in pairs):
        return {
            "w1_grad_weighted": None,
            "w2_grad_weighted": None,
            "w3_grad_weighted": None,
        }
    return {
        "w1_grad_weighted": torch.stack(
            [fc1.grad[:hidden] for fc1, _ in pairs], dim=0
        ),
        "w2_grad_weighted": torch.stack([fc2.grad for _, fc2 in pairs], dim=0),
        "w3_grad_weighted": torch.stack(
            [fc1.grad[hidden:] for fc1, _ in pairs], dim=0
        ),
    }


def _mcore_arm(
    arm: str,
    profile: McoreProfile,
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: ExpertMlpInputs,
) -> BuiltArm:
    """One megatron expert layer, taken off a real ``GPTModel``.

    The layer is navigated out of the built model rather than constructed
    here. That matters for this scenario above all others: the expert class is
    a *spec* decision made from ``config.moe_grouped_gemm``, and the
    constructor arguments ``moe_layer.py:326`` passes -- ``num_local_experts``,
    the ``pg_collection``, the submodule triple -- would all have to be written
    a second time and kept in agreement by hand. ``build_model`` and
    ``get_gpt_decoder_block_spec`` decide instead, which is what makes
    ``no_grouped_gemm`` a profile delta rather than a hand-built spec.

    The rest of the model is released as soon as the layer is extracted. That
    release is what keeps ``memory_pass`` honest: resetting the peak counter
    before a call does not exclude resident memory, so a retained model would
    be charged to this arm.

    Eager on purpose: megatron compiles no whole transformer layer, so this is
    the treatment megatron gives these modules. ``KernelArm.eager_reason``
    records it.
    """
    initialize_megatron_single_rank()

    from benchmarks.models.piper_qwen3.megatron_model import build_model

    model = build_model(seq_len=workload.seq_len, shape=shape, profile=profile)
    experts = _navigate(model, mcore_experts_path())
    notes = _assert_mcore_experts(arm, experts, profile, shape)
    with torch.no_grad():
        for expert, (fc1, fc2) in enumerate(
            mcore_expert_parameters(experts, shape)
        ):
            fc1.copy_(
                torch.cat(
                    [
                        inputs.stock_state["w1_EFD"][expert],
                        inputs.stock_state["w3_EFD"][expert],
                    ],
                    dim=0,
                )
            )
            fc2.copy_(inputs.stock_state["w2_EDF"][expert])
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if notes["use_te_activation_func"]:
        _assert_te_activation_ran(arm, experts, inputs)
    print(
        f"expert_mlp/{arm}: {mcore_experts_path()} is "
        f"{type(experts).__module__}.{type(experts).__qualname__} "
        f"(profile {profile.name}, moe_grouped_gemm="
        f"{notes['moe_grouped_gemm']}, use_te_activation_func="
        f"{notes['use_te_activation_func']}, eager)",
        flush=True,
    )

    forward_leaf = inputs.x.clone().requires_grad_()
    round_trip_leaf = inputs.x.clone().requires_grad_()
    check_leaf = inputs.x.clone().requires_grad_()
    # One probability leaf per closure, each requiring grad. See the module
    # docstring: in production these carry a gradient back to the router, and
    # two of the four megatron arms pay a reduction for it that a non-requiring
    # input would skip.
    forward_probs = inputs.probs.clone().requires_grad_()
    round_trip_probs = inputs.probs.clone().requires_grad_()
    check_probs = inputs.probs.clone().requires_grad_()

    def forward():
        return experts(forward_leaf, inputs.counts_host, forward_probs)

    def forward_backward() -> None:
        _reset_grads(round_trip_leaf, round_trip_probs, experts)
        out, _ = experts(round_trip_leaf, inputs.counts_host, round_trip_probs)
        torch.autograd.backward(out, inputs.grad_out)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(check_leaf, check_probs, experts)
        out, _ = experts(check_leaf, inputs.counts_host, check_probs)
        torch.autograd.backward(out, inputs.grad_out)
        return _require_grads(
            arm,
            {
                "out_weighted": out.detach(),
                "x_grad_weighted": check_leaf.grad,
                "probs_grad_weighted": check_probs.grad,
                **_mcore_weight_grads(experts, shape),
            },
        )

    return BuiltArm(
        name=arm,
        calls={"forward": forward, "forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
        notes=notes,
    )


def build_expert_mlp_mcore_base(
    shape: PiperShape, workload: KernelWorkload, inputs: ExpertMlpInputs
) -> BuiltArm:
    """Megatron-core's ``TEGroupedMLP`` with every fusion the base profile sets.

    One TE grouped GEMM for the doubled fc1, ``weighted_bias_swiglu_impl`` for
    the activation -- which folds the routing probabilities into the same
    kernel (``experts.py:830``) -- and one TE grouped GEMM for fc2.
    """
    return _mcore_arm(MCORE_BASE_ARM, BASE, shape, workload, inputs)


def build_expert_mlp_mcore_no_bias_activation_fusion(
    shape: PiperShape, workload: KernelWorkload, inputs: ExpertMlpInputs
) -> BuiltArm:
    """The same layer with the fused SwiGLU declined.

    ``bias_act_func`` falls through to its last branch (``experts.py:858-879``):
    ``torch.chunk`` into gate and up, ``F.silu``, a multiply, and then a second
    multiply by the probabilities with a dtype round trip. Four kernels and an
    extra full-width intermediate where the fused arm has one kernel.
    """
    return _mcore_arm(
        MCORE_NO_BIAS_ACTIVATION_FUSION_ARM,
        NO_BIAS_ACTIVATION_FUSION_PROFILE,
        shape,
        workload,
        inputs,
    )


def build_expert_mlp_mcore_te_activation_func(
    shape: PiperShape, workload: KernelWorkload, inputs: ExpertMlpInputs
) -> BuiltArm:
    """The same layer running TransformerEngine's own SwiGLU operation.

    ``bias_act_func`` takes its first branch (``experts.py:815-826``): TE's
    ``SwiGLU`` operation over the combined tensor, and then a **separate**
    multiply by the routing probabilities. So against ``mcore/base`` this is a
    kernel-count difference as much as a kernel-speed one, and a table that
    reports only the ratio hides which of the two it measured.

    Two flags, not one: ``transformer_config.py:2166-2171`` refuses
    ``use_te_activation_func`` beside ``bias_activation_fusion``, and the base
    profile sets the latter on.
    """
    return _mcore_arm(
        MCORE_TE_ACTIVATION_FUNC_ARM,
        TE_ACTIVATION_FUNC_PROFILE,
        shape,
        workload,
        inputs,
    )


def build_expert_mlp_mcore_no_grouped_gemm(
    shape: PiperShape, workload: KernelWorkload, inputs: ExpertMlpInputs
) -> BuiltArm:
    """The same layer as ``SequentialMLP``: one GEMM pair per expert.

    ``config.moe_grouped_gemm=False`` reaches the layer spec through
    ``get_gpt_decoder_layer_specs`` (``gpt_layer_specs.py:593``), and
    ``TESpecProvider.grouped_mlp_modules`` then builds ``SequentialMLP`` over
    ``TEColumnParallelLinear``/``TERowParallelLinear``
    (``transformer_engine_spec_provider.py:101-108``). The forward splits the
    rows four ways and calls four ``MLP`` modules in turn
    (``experts.py:1353-1369``); each of those still takes the fused
    weighted-SwiGLU path, so this arm isolates the grouping and nothing else.

    The config field alone would be inert under a hand-built layer spec, which
    is what the builder used to use. ``_assert_mcore_experts`` proves the class
    actually changed, because megatron polices that agreement nowhere and both
    disagreement directions are numerically correct.
    """
    return _mcore_arm(
        MCORE_NO_GROUPED_GEMM_ARM,
        NO_GROUPED_GEMM_PROFILE,
        shape,
        workload,
        inputs,
    )
