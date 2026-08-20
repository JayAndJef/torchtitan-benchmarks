"""Arm builders for the ``moe_router`` cross-engine kernel scenario.

The MoE router, head to head across the two engines. The cut starts at the
hidden state the MoE block receives and ends at the routing decision: which
experts each token goes to, and with what weight.

* TorchTitan: ``self.router(x_BLD, self.expert_bias_E)`` in ``MoE.forward``
  (``third_party/torchtitan/torchtitan/models/common/moe.py:459``; the module
  is built at ``:377`` and its class is ``TokenChoiceTopKRouter``, ``:186``).
  ``expert_bias_E`` is ``None`` here, because
  ``benchmarks/models/piper_qwen3/config_registry.py:111`` sets
  ``load_balance_coeff = None`` and ``moe.py:388-397`` builds the buffer only
  when it is not None.
* Megatron-core: ``MoELayer.route``
  (``third_party/Megatron-LM/megatron/core/transformer/moe/moe_layer.py:438``),
  which calls ``TopKRouter.forward``
  (``third_party/Megatron-LM/megatron/core/transformer/moe/router.py:843``).

**Where the cut stops, and what belongs to scenario 10.** ``MoE.forward``
builds the one-hot ``routing_map_BLE`` and the per-expert counts *after* the
router returns (``moe.py:465-470``). Plan section C.1 gives those to scenario
10, ``dispatch_permute``, so they are outside this scenario on the titan side.
Megatron's router returns its routing map itself, so the map is inside the
scenario on the mcore side. That asymmetry is real, it is small, and it is
declared rather than equalized -- see "What each engine materializes" below.
``MoELayer.preprocess`` belongs to scenario 10 on the mcore side and is not
called here.

**The activation unflatten in the plan's cut is an early return in this
build, so it is charged to neither engine.**
``TransformerLayer._maybe_unflatten_for_moe``
(``transformer_layer.py:763``) returns its input unchanged at ``:780``
whenever ``packed_seq_params.tokens_per_sample`` is ``None``.
``PackedSeqParams.tokens_per_sample`` defaults to ``None``
(``megatron/core/packed_seq_params.py:27``) and
``benchmarks/e2e/megatron/train.py:273-279`` never sets it, so the branch this
build takes writes nothing and launches nothing.
``_assert_unflatten_is_an_early_return`` proves that at build time against a
``PackedSeqParams`` built exactly as the driver builds it, rather than leaving
the claim in this docstring. A driver change that starts setting the field
fails the build instead of silently moving a transpose into one engine's
number.

**The headline caption, and it corrects the plan.** Plan section C.3 says
titan's router runs bf16 and calls the cross-engine row a precision
difference. **Source says both sides compute the router in fp32.**

* ``TokenChoiceTopKRouter.forward`` wraps the gate GEMM in
  ``torch.autocast(device_type=..., dtype=torch.float32)``
  (``moe.py:292-293``) and takes the softmax on the fp32 result (``:300``).
  ``torch.autocast`` does not refuse fp32 on CUDA: the ``cuda`` branch of
  ``torch/amp/autocast_mode.py`` is ``:283-296`` and returns without checking
  the dtype, and the branch that disables autocast for a dtype outside
  ``[bfloat16, float16]`` is the ``elif`` at ``:297``, which ``cuda`` never
  reaches. ``ATen/autocast_mode.h:216-225`` then returns the requested dtype
  with no check. On **CPU** the ``elif`` does fire and autocast is disabled,
  so no CPU test can observe this and ``_assert_titan_router_is_fp32`` is the
  only check that can. So ``F.linear`` casts both bf16 operands up
  and runs a true fp32 GEMM.
* ``mcore/base`` sets ``moe_router_dtype="fp32"``
  (``benchmarks/models/piper_qwen3/mcore_profiles.py``), read at
  ``router.py:107,109``, so ``router_gating_linear`` asks for an fp32 output
  (``moe_utils.py:1380-1381``) and the score function runs in fp32
  (``moe_utils.py:889-907``).

The two therefore **route at the same precision**, and ``titan`` against
``mcore/base`` is this scenario's cross-engine row.

**But they are not like for like on cost, and that is the caption a reader
must not lose.** The two GEMMs differ in their *operand* precision. Titan's
autocast casts BOTH operands up, so ``F.linear`` materializes a full fp32
copy of the ``[B, L, D]`` hidden state and runs an fp32 GEMM; megatron hands
TE the **bf16** operands and asks only for an fp32 output
(``moe_utils.py:1380-1381``; only the ``te_general_gemm is None`` fallback at
``:1383-1388`` upcasts, and ``_assert_mcore_gating_gemm_takes_the_te_path``
refuses to build a megatron arm on that branch, because
``mcore_bytes_moved`` describes the TE branch and nothing else).
Numerically that changes nothing -- a bf16 value
upcasts to fp32 exactly, so the two GEMMs sum the same exact products and
differ only in accumulation order. **In cost it changes everything this
scenario measures**: the upcast is 4N bytes written and 4N read where the
whole megatron forward moves 2N, so titan carries about five times the
traffic (see ``titan_bytes_moved``) and dispatches an extra kernel, in a
scenario whose device work is one bandwidth-bound read.

So the cross-engine row is **not** a verdict on TorchTitan's router kernel.
It is a comparison of two routers of which one is asked for fp32 operands and
the other is not. The arm that would separate the two -- a titan arm with the
autocast removed -- **is not declared**, and until it is, no reading of this
row may attribute the gap to the kernel. Every table must say so, and
``titan_bytes_moved`` is what makes the asymmetry visible in the GB/s and
``x_floor`` columns rather than hidden inside them.

``mcore/router_bf16`` is the arm whose precision is *not* like for like, and
it publishes against ``mcore/base`` for that reason. Comparing it against
``titan`` would change the engine and the precision in one row.
**Nothing measured here has ever run on a GPU.**

**The two engines compute one function, by two orderings.** Titan takes the
softmax over all ``E`` experts, selects the top ``k``, and renormalizes over
the selected ``k`` (``moe.py:292-293,300,312-314,318,327-329`` for the pieces,
``route_norm=True`` at ``models/qwen3/__init__.py:173``). Megatron selects the
top ``k`` on the logits and takes the softmax over those ``k``
(``moe_router_pre_softmax=False``, ``moe_utils.py:900-907``). Softmax is
monotone and the renormalization cancels the shared denominator, so the two
produce the same probabilities from the same logits. ``mcore_profiles.BASE``
already records the identity beside the flag. ``build_moe_router_titan``
refuses to build when ``route_norm`` is off, because without it titan
publishes an unnormalized weight against megatron's normalized one.

**Ties are where the two orderings may legitimately disagree, so no gate
enforces the selection.** Two experts whose logits are equal to the working
precision may be ordered either way, and the two arms then route a token
differently. The dense probabilities of such a token differ by two whole
entries, which no norm-based tolerance can separate from a wrong kernel. The
enforced cross-arm gates are therefore the tie-immune ones -- the logits, the
per-token probability sum and the per-token expert count -- and the dense
``probs``, the ``routing_map`` and the gradients are enforced only between
arms of the same precision. Between precisions they are recorded and never
enforced. See ``moe_router_reference`` for what each output is.

**What each engine materializes, and why the memory column is not a kernel
statement.** Megatron writes a dense ``[T, E]`` probability tensor and a dense
``[T, E]`` boolean routing map. Titan writes a ``[B, L, E]`` score tensor, a
``[B, L, K]`` probability tensor and a ``[B, L, K]`` **int64** index tensor.
At ``E = 4`` and ``K = 2`` those are all small next to the ``[B, L, D]`` input,
which is what both arms read and what dominates the traffic.

**This scenario is expected to be dispatch-bound, and ``copy_floor`` is how a
reader sees it.** A megatron router forward reads ``batch * seq_len * dim``
bf16 elements once -- 8 MiB at the default workload -- and writes a few
hundred KiB. The GEMM is ``2 * B * L * D * E`` flops, 33.5 MFLOP at the
default workload, which is arithmetically negligible. So the device work
there is one bandwidth-bound read, and CLAUDE.md's measured per-call host
dispatch for module arms is tens to hundreds of microseconds.

The floor is the only column that separates the two, and it sits differently
against each engine. **It moves about twice a megatron arm's traffic** -- it
reads and writes ``[B, L, D]``, where that router reads it and writes
``[T, E]`` -- so ``x_floor`` is a *lower* bound on how far a megatron arm
sits above the device, not an estimate of it. **Against ``titan`` the
inequality reverses**: titan moves about 40 MiB to the floor's 16, because of
the fp32 upcast above, so a titan ``x_floor`` below 1.0 would be a real
bandwidth statement rather than an impossible one.

**The number this scenario publishes is device time plus host serialization.**
Neither engine's router contains a device-to-host copy on this path -- that
belongs to scenario 10, where ``MoEAllGatherTokenDispatcher.
dispatch_postprocess`` calls ``.cpu()`` unconditionally
(``token_dispatcher.py:317``) -- so nothing here forces a sync inside a burst.
But the arms sit far above the bandwidth floor, so the measured interval holds
host stalls the device cannot hide. Read ``--burst`` before ranking anything.

**The mcore arms are eager, with one exception that must be stated.**
Megatron compiles no whole transformer layer, so the router runs eager as
megatron runs it. But ``TopKRouter._apply_expert_bias`` is ``@jit_fuser``
decorated (``router.py:730``) and ``megatron/core/jit.py:21`` binds
``jit_fuser`` to ``torch.compile`` on torch >= 2.2 (inside
``enable_jit_fuser``, which runs at module scope at ``:33``; ``:24`` is the
``noop_decorator`` fallback, not the bind), so one ``torch.compile``
region runs on every router call (``router.py:833``). At
``moe_router_enable_expert_bias=False`` its body does nothing, so it compiles
to an empty graph and what remains is Dynamo's per-call guard evaluation --
host cost, in a scenario that is host-bound. Every arm's ``eager_reason``
says so.

**The balanced-routing invariant does not apply to this scenario, and that is
a decision rather than an omission.** ``routing_divides_evenly``
(``benchmarks/kernel/schema.py``) exists so ``expert_mlp_inputs`` can hand every
expert an equal slice of ``batch * seq_len * top_k`` synthetic rows. This
scenario materializes no per-expert tensor and hands no expert a slice: the
router computes the split itself from the data, and its output shape is
``[T, E]`` whatever the split turns out to be. Declaring
``requires_balanced_routing`` here would refuse workloads this scenario
measures correctly. At the default workload the quantity divides anyway
(``4 * 1024 * 2 = 8192`` routed rows across ``4`` experts, ``8192 % 4 == 0``).
``moe_router_inputs`` does assert the invariant this scenario really has --
``top_k <= num_experts``, without which ``torch.topk`` raises inside a timed
closure -- and it names both numbers when it fails.

Every torchtitan, megatron and TransformerEngine import is deferred into the
builder that needs it, which is the rule across ``operations/``.
``benchmarks.models.piper_qwen3.mcore_profiles`` is the one exception at
module scope: it is torch-free parent-side data, and this module derives its
two variant profiles from it.
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
from benchmarks.models.piper_qwen3.mcore_profiles import BASE, derive, McoreProfile
from benchmarks.models.piper_qwen3.shape import PiperShape


# The five arm names, spelled as plan section C.2 spells them. The slash says
# which engine an arm is and which profile of that engine; ``fragment_stem``
# in ``benchmarks/kernel/schema.py`` is what keeps it out of a file path.
FLOOR_ARM_NAME = "copy_floor"
MCORE_BASE_ARM_NAME = "mcore/base"
MCORE_FUSION_ARM_NAME = "mcore/router_fusion"
MCORE_BF16_ARM_NAME = "mcore/router_bf16"
TITAN_ARM_NAME = "titan"

# The layer each engine reads its router from. Every layer holds the same
# module class at ``moe_layer_freq=1``, and the gate weight is overwritten
# from the shared seeded tensor, so the index is arbitrary and is fixed here
# so the provenance line can name it.
LAYER = 0

# The two profile deltas this scenario measures, derived from the base rather
# than retyped. ``derive`` merges over ``BASE.config_overrides`` and
# ``McoreProfile.__post_init__`` re-validates the merged result.
#
# **Both are real deltas against the base, and each was checked against the
# pinned rev rather than assumed.** ``moe_router_fusion`` is absent from
# ``BASE.config_overrides`` and defaults to ``False``
# (``transformer_config.py:915``). ``moe_router_dtype`` is present in
# ``BASE.config_overrides`` as ``"fp32"``, so the delta that earns an arm is
# ``None`` -- bf16 routing -- and *not* fp32, which the base already sets. An
# earlier draft of the plan declared an ``mcore/router_fp32`` arm that changed
# nothing at all.
ROUTER_FUSION = derive(
    BASE,
    name="router_fusion",
    description=(
        "base plus moe_router_fusion=True: TransformerEngine's fused top-k "
        "score-function kernel in place of megatron's torch op sequence"
    ),
    config_overrides={"moe_router_fusion": True},
)

ROUTER_BF16 = derive(
    BASE,
    name="router_bf16",
    description=(
        "base with moe_router_dtype=None: the gate GEMM writes bf16 logits "
        "and the top-k selects on them. The base is already fp32, so the "
        "delta is bf16"
    ),
    config_overrides={"moe_router_dtype": None},
)

# What each mcore arm's ``moe_router_dtype`` and ``moe_router_fusion`` must be
# once the config is built. Read off the profiles rather than written twice:
# the guard's job is to prove the delta reached the built config, and a
# separately typed expectation could agree with neither.
MCORE_ARM_PROFILES: dict[str, McoreProfile] = {
    MCORE_BASE_ARM_NAME: BASE,
    MCORE_FUSION_ARM_NAME: ROUTER_FUSION,
    MCORE_BF16_ARM_NAME: ROUTER_BF16,
}

# The correspondence the cross-engine weight map already owns, quoted rather
# than reimplemented. ``benchmarks/models/piper_qwen3/megatron_weights.py``
# pairs these two names under the ``router`` component tag, and for an
# ``[E, D]`` gate matrix that pairing is the identity -- there is no reshape
# here to get wrong. ``tests/test_kernel_moe_router.py`` asserts the map still
# yields exactly this pair from exactly this source.
MCORE_WEIGHT_NAME = "decoder.layers.{layer}.mlp.router.weight"
TITAN_WEIGHT_NAME = "layers.{layer}.moe.router.gate.weight"
WEIGHT_COMPONENT = "router"

# The megatron class each side must be, checked at build time. Several names
# in this system overclaim; these two are what the scenario says it measures.
MCORE_LAYER_CLASS = "MoELayer"
MCORE_ROUTER_CLASS = "TopKRouter"
TITAN_ROUTER_CLASS = "TokenChoiceTopKRouter"


def mcore_moe_layer_path(layer: int = LAYER) -> str:
    """The attribute path from a ``GPTModel`` down to the MoE layer.

    Derived from ``MCORE_WEIGHT_NAME`` rather than written beside it, so the
    navigation and the weight map can only agree.
    """
    name = MCORE_WEIGHT_NAME.format(layer=layer)
    suffix = ".router.weight"
    assert name.endswith(suffix)
    return name[: -len(suffix)]


def mcore_transformer_layer_path(layer: int = LAYER) -> str:
    """The attribute path down to the transformer layer holding that MoE layer.

    Derived from ``mcore_moe_layer_path`` for the same reason: the activation
    unflatten this scenario must prove is an early return is a method of the
    *transformer* layer, not of the MoE layer, so both paths have to name one
    place in the model.
    """
    path = mcore_moe_layer_path(layer)
    suffix = ".mlp"
    assert path.endswith(suffix)
    return path[: -len(suffix)]


@dataclass
class MoeRouterInputs:
    """One hidden-state batch, one gate matrix, and one canonical gradient.

    ``gate_weight`` is fp32 and every arm loads it, so all four arms hold the
    same gate by construction. Both engines store the gate in bf16 -- megatron
    through ``Router.reset_parameters``'s cast to ``params_dtype``
    (``router.py:84``) plus ``build_model``'s blanket ``.bfloat16()``, titan
    through this module's own cast -- so each arm rounds the same fp32 values
    once.

    ``grad_probs`` is the **canonical** output gradient: one fp32 value per
    (token, expert). Each arm converts it into its own native form at build
    time, never inside a timed closure. Megatron takes it whole, because its
    router returns dense ``[T, E]`` probabilities. Titan takes it gathered at
    its own selected experts, because its router returns ``[B, L, K]``. The
    two are the same canonical gradient restricted to each arm's own
    selection, which is the only definition that survives a tie broken
    differently.
    """

    x: torch.Tensor  # (B, L, D) bf16, the MoE block's input
    grad_probs: torch.Tensor  # (B, L, E) fp32, the canonical output gradient
    gate_weight: torch.Tensor  # (E, D) fp32, shared by every arm
    mcore_bytes_moved: int  # the three megatron arms' forward traffic
    titan_bytes_moved: int  # titan's, which is about 5x larger; see below
    floor_bytes_moved: int  # the copy floor's forward traffic


def moe_router_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> MoeRouterInputs:
    """The tensors every arm shares, drawn once from the run's generator.

    The one invariant this scenario really has is asserted here, with both
    numbers named: a top-k wider than the expert count is not a routing this
    scenario can cap or round, it is a ``torch.topk`` that raises inside a
    timed closure. The *balanced*-routing invariant is a different statement
    and does not apply -- see this module's docstring.
    """
    if shape.top_k > shape.num_experts:
        raise ValueError(
            f"moe_router: top_k ({shape.top_k}) exceeds num_experts "
            f"({shape.num_experts}); no router can select "
            f"{shape.top_k} of {shape.num_experts} experts"
        )
    batch, seq = workload.batch, workload.seq_len
    x = _randn((batch, seq, shape.dim), device, generator)
    grad_probs = _randn(
        (batch, seq, shape.num_experts), device, generator, torch.float32
    )
    gate_weight = _randn(
        (shape.num_experts, shape.dim), device, generator, torch.float32, WEIGHT_STD
    )
    tokens = batch * seq
    x_bytes = x.numel() * x.element_size()
    return MoeRouterInputs(
        x=x,
        grad_probs=grad_probs,
        gate_weight=gate_weight,
        # Megatron's forward traffic: one read of x, one write of the dense
        # fp32 probabilities and one of the boolean routing map. All three
        # megatron arms carry it, so the GB/s column compares them; the bf16
        # arm writes half as many probability bytes, which is below the
        # resolution of that column. The merge divides this by every mode's
        # median, so the GB/s figure is a bandwidth statement in ``forward``
        # alone -- ``rope`` and ``ffn_norm`` record the same property.
        mcore_bytes_moved=x_bytes + tokens * shape.num_experts * (4 + 1),
        # **Titan moves about five times that, and the two arms must not share
        # one figure.** Titan's router wraps its gate in
        # torch.autocast(float32) (moe.py:292-293), and F.linear is an
        # autocast lower_precision_fp op, so BOTH operands are cast UP: the
        # bf16 [B, L, D] hidden state is materialized as a full fp32 copy
        # before the GEMM runs. Megatron never does this -- router_gating_
        # linear hands TE the bf16 operands and asks only for an fp32 output
        # (moe_utils.py:1380-1381). That branch is not assumed: the torch.mm
        # fallback at :1383-1388 makes the same fp32 copy, so
        # _assert_mcore_gating_gemm_takes_the_te_path raises rather than let
        # mcore_bytes_moved describe a path the arm did not run.
        #
        # So titan reads x (2N bytes), writes the fp32 copy (4N) and reads it
        # back in the mm (4N), which is 5 * x_bytes, plus its own three
        # outputs: [B, L, E] fp32 scores, [B, L, K] fp32 top-k scores and
        # [B, L, K] int64 indices. At the default workload that is about
        # 40 MiB against megatron's 8, and one shared figure would have made
        # the GB/s and x_floor columns hide the largest asymmetry in the
        # scenario.
        #
        # **This is derived, not measured, and it assumes the default
        # lowering.** Inductor emits a cast kernel plus ``extern_kernels.mm``,
        # which needs a materialized fp32 operand. A Triton mm template with a
        # fused prologue would not materialize it, and kernel-bench sets no
        # max_autotune, so that is not today's path -- but it is what would
        # make this figure wrong.
        titan_bytes_moved=(
            5 * x_bytes
            + tokens * (shape.num_experts * 4 + shape.top_k * (4 + 8))
        ),
        # The floor reads and writes x, so it moves about twice what MEGATRON
        # does and about 40% of what titan does. Recorded separately rather
        # than shared, so the GB/s column stays true for every arm; the
        # consequence for the x_floor column is in this module's docstring.
        floor_bytes_moved=2 * x_bytes,
    )


def moe_router_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeRouterInputs
) -> dict[str, torch.Tensor]:
    """fp64 truth for the routing both engines compute, and its gradients.

    The gate matrix is quantized to bf16 before it is promoted to fp64, which
    is the ``qkv_reference`` rule: an fp64 truth built from the unrounded
    matrix would charge every arm for an input cast none of them performs.

    Written in megatron's ordering -- top-k on the logits, then softmax over
    the selected k -- because that ordering has one fewer operation. Titan's
    softmax-then-top-k-then-renormalize computes the same function; see this
    module's docstring for why.

    Seven outputs, and they fall into two classes a reader must keep apart:

    * **Tie-immune.** ``logits`` is the gate GEMM and holds no discrete
      decision. ``prob_row_sums`` is 1.0 for every token whichever experts
      were selected, so it catches a missing renormalization, a wrong k or a
      stray scaling factor. ``selected_count`` is exactly ``top_k`` for every
      token, so it catches token dropping and a degenerate routing map. These
      three are what every arm is gated on, across every precision boundary.
    * **Tie-sensitive.** ``probs`` and ``routing_map`` encode *which* experts
      were selected, and ``x_grad`` and ``gate_weight_grad`` are gradients
      that flow only through the selected ones. One token whose two closest
      logits are ordered differently moves two whole entries of ``probs``, so
      these are gated only between arms that select on the same precision.

    ``routing_map`` and ``selected_count`` are returned as fp32 rather than
    fp64, because they are the two outputs gated **bitwise**:
    ``torch.equal`` compares dtypes as well as values, so an fp64 row would
    fail against every arm's fp32 row for a reason no reader could recover.
    Both hold small integers that fp32 represents exactly, so nothing is lost.
    The five tolerance-gated outputs stay fp64, because ``_check_rows``
    promotes both sides to fp32 itself and the extra precision is the point of
    a reference.
    """
    batch, seq = workload.batch, workload.seq_len
    tokens = batch * seq
    x = inputs.x.reshape(tokens, shape.dim).double().detach().requires_grad_()
    weight = (
        inputs.gate_weight.to(torch.bfloat16).double().detach().requires_grad_()
    )

    logits = x @ weight.t()
    top_logits, top_indices = torch.topk(logits, shape.top_k, dim=-1)
    top_probs = torch.softmax(top_logits, dim=-1)
    probs = torch.zeros_like(logits).scatter(1, top_indices, top_probs)
    routing_map = torch.zeros_like(logits).scatter(
        1, top_indices, torch.ones_like(top_probs)
    )
    torch.autograd.backward(
        probs, inputs.grad_probs.reshape(tokens, shape.num_experts).double()
    )
    return {
        "logits": logits.detach(),
        "probs": probs.detach(),
        "routing_map": routing_map.detach().float(),
        "prob_row_sums": probs.detach().sum(dim=-1),
        "selected_count": routing_map.detach().sum(dim=-1).float(),
        "x_grad": x.grad.reshape(batch, seq, shape.dim),
        "gate_weight_grad": weight.grad,
    }


def _canonical_outputs(
    *,
    arm: str,
    shape: PiperShape,
    workload: KernelWorkload,
    probs: torch.Tensor,
    routing_map: torch.Tensor,
    logits: torch.Tensor,
    x_grad: torch.Tensor | None,
    gate_weight_grad: torch.Tensor | None,
) -> dict[str, torch.Tensor]:
    """One arm's outputs in the shapes and dtypes every gate compares.

    Both engines are put into ``[T, E]`` for the routing tensors, because that
    is the shape megatron's router returns natively and the shape titan's
    ``[B, L, *]`` tensors reshape into for free. The dense form is also the
    only one in which a differently-broken tie is *visible* rather than
    silently reindexed.

    Every conversion here runs in ``correctness_outputs``, outside every timed
    region. Doing it inside one would charge titan for the scatter that plan
    section C.1 gives to scenario 10.
    """
    tokens = workload.batch * workload.seq_len
    expected = (tokens, shape.num_experts)
    for name, tensor in (
        ("logits", logits),
        ("probs", probs),
        ("routing_map", routing_map),
    ):
        if tuple(tensor.shape) != expected:
            raise RuntimeError(
                f"{arm}: canonical {name} is {tuple(tensor.shape)}, expected "
                f"{expected}; the arm returned a routing this scenario cannot "
                "compare"
            )
    _require_grads(
        arm, {"x_grad": x_grad, "gate_weight_grad": gate_weight_grad}
    )
    return {
        "logits": logits.float(),
        "probs": probs.float(),
        "routing_map": routing_map.float(),
        "prob_row_sums": probs.float().sum(dim=-1),
        "selected_count": routing_map.float().sum(dim=-1),
        "x_grad": x_grad.float().reshape(
            workload.batch, workload.seq_len, shape.dim
        ),
        "gate_weight_grad": gate_weight_grad.float(),
    }


def _router_arm(
    *,
    name: str,
    owner: nn.Module,
    call: Callable[[torch.Tensor], Any],
    backward_target: Callable[[Any], torch.Tensor],
    canonical: Callable[[torch.Tensor, Any], dict[str, torch.Tensor]],
    x_native: torch.Tensor,
    grad_native: torch.Tensor,
    bytes_moved: int,
    notes: dict[str, Any] | None = None,
) -> BuiltArm:
    """Forward and forward+backward over one engine's native tensor shape.

    Every arm runs this same code over its own callable, so the comparison
    measures the two routers and nothing about how each arm was written.

    ``x_native`` and ``grad_native`` already carry that engine's shape,
    because a reshape inside a timed closure would be timed as part of the
    router. ``backward_target`` picks the one differentiable output
    production actually consumes: megatron's dense probabilities, and titan's
    top-k scores. Titan's ``scores_BLE`` is deliberately left ungraded --
    ``MoE.forward`` uses it only to build the boolean routing map
    (``moe.py:465-469``), which carries no gradient.
    """
    forward_leaf = x_native.clone().requires_grad_()
    round_trip_leaf = x_native.clone().requires_grad_()
    check_leaf = x_native.clone().requires_grad_()

    def forward():
        return call(forward_leaf)

    def forward_backward() -> None:
        _reset_grads(round_trip_leaf, owner)
        torch.autograd.backward(
            backward_target(call(round_trip_leaf)), grad_native
        )

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(check_leaf, owner)
        result = call(check_leaf)
        torch.autograd.backward(backward_target(result), grad_native)
        _require_grads(name, {"x_grad": check_leaf.grad})
        return canonical(check_leaf.grad, result)

    return BuiltArm(
        name=name,
        calls={"forward": forward, "forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
        bytes_moved=bytes_moved,
        notes=dict(notes or {}),
    )


def build_moe_router_copy_floor(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeRouterInputs
) -> BuiltArm:
    """The bandwidth floor: one read of x and one write, and nothing else.

    The router's device work is one bandwidth-bound pass over ``x`` plus a
    handful of kilobytes of output, so the interesting question about every
    other arm in this scenario is how far above the bus it sits. This arm is
    the only column that answers it. Forward only: the forward traffic is
    exactly a copy, and a floor for the forward+backward traffic would be an
    invention rather than a measurement.

    It reads **and** writes ``[B, L, D]``, where the router reads that and
    writes ``[T, E]``, so it moves about twice the router's bytes. The
    ``x_floor`` column is therefore a lower bound on the gap, not an estimate
    of it, and this module's docstring says so where a reader will find it.
    """
    out = torch.empty_like(inputs.x)

    def forward() -> None:
        out.copy_(inputs.x)

    return BuiltArm(
        name=FLOOR_ARM_NAME,
        calls={"forward": forward},
        correctness_outputs=dict,
        bytes_moved=inputs.floor_bytes_moved,
    )


# --------------------------------------------------------------------------
# TorchTitan
# --------------------------------------------------------------------------


def _assert_titan_router_config(config: Any, shape: PiperShape) -> None:
    """Refuse a router node that is not the one the production block builds.

    Every check **raises**. ``BuiltArm.notes`` reaches no artifact, so a fact
    recorded there is a fact no reader sees, and a guard that cannot raise is
    not a guard.

    Each entry names a way this arm could measure something other than what
    its label says, and none of them is visible to a correctness gate that
    compares titan only against itself:

    * ``route_norm``, because it is what makes titan's softmax-then-top-k
      equal megatron's top-k-then-softmax. With it off, titan publishes an
      unnormalized weight against megatron's normalized one, and the
      cross-engine row stops comparing two implementations of one function.
    * ``score_func``, because sigmoid and softmax are different functions.
    * ``route_scale``, because any value but 1.0 rescales every probability
      and megatron's equivalent (``moe_router_topk_scaling_factor``) is None.
    * ``top_k`` and ``num_experts``, because they are the routing itself.
    * the group-limited fields, because they add a second selection stage
      megatron's base profile does not run.
    * ``_debug_force_load_balance``, because it replaces the routing with a
      round-robin assignment (``moe.py:219-233``) that is numerically valid
      and completely unlike a router.
    * the gate's shape and bias, because a bias would add a term megatron has
      no counterpart for (``add_bias_linear=False`` makes ``Router.bias``
      None, ``router.py:66-72``).
    """
    checks: tuple[tuple[str, Any, Any], ...] = (
        ("num_experts", config.num_experts, shape.num_experts),
        ("top_k", config.top_k, shape.top_k),
        ("score_func", config.score_func, "softmax"),
        ("route_norm", config.route_norm, True),
        ("route_scale", config.route_scale, 1.0),
        ("num_expert_groups", config.num_expert_groups, None),
        ("num_limited_groups", config.num_limited_groups, None),
        (
            "_debug_force_load_balance",
            config._debug_force_load_balance,
            False,
        ),
        ("gate.in_features", config.gate.in_features, shape.dim),
        ("gate.out_features", config.gate.out_features, shape.num_experts),
        ("gate.bias", config.gate.bias, False),
    )
    for field, actual, expected in checks:
        if actual != expected:
            raise RuntimeError(
                f"{TITAN_ARM_NAME}: the production router node declares "
                f"{field}={actual!r}, and this scenario measures "
                f"{field}={expected!r}. The two engines would not compute the "
                "same function, so the cross-engine ratio would compare two "
                "different routings."
            )


def _assert_titan_router_is_fp32(
    scores: torch.Tensor, top_scores: torch.Tensor
) -> None:
    """Refuse to publish the cross-engine row if titan stopped computing fp32.

    This is the run-time half of this module's headline caption. Titan's
    router asks for fp32 by wrapping its gate in
    ``torch.autocast(dtype=torch.float32)``, and today's torch honours that on
    CUDA. If a future torch, or a future torchtitan, drops it, the gate GEMM
    silently becomes bf16 and ``titan`` stops being like for like with
    ``mcore/base`` on precision -- which is exactly the mistake plan section
    C.3 already made in the other direction. Nothing else in the harness can
    see it: a bf16 router is numerically valid and passes every tie-immune
    gate.
    """
    for name, tensor in (("scores_BLE", scores), ("topk_scores_BLK", top_scores)):
        if tensor.dtype is not torch.float32:
            raise RuntimeError(
                f"{TITAN_ARM_NAME}: {name} came back as {tensor.dtype}, not "
                "float32. TokenChoiceTopKRouter.forward computes its gate "
                "under torch.autocast(dtype=torch.float32), so a lower "
                "precision here means the autocast no longer applies. This "
                "scenario's cross-engine row claims both engines route in "
                "fp32 and must not be published without it."
            )


def build_moe_router_titan(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeRouterInputs
) -> BuiltArm:
    """TorchTitan's ``TokenChoiceTopKRouter``, compiled.

    The config node is taken from the production model config rather than
    reconstructed, so ``score_func``, ``route_norm``, ``route_scale``,
    ``top_k`` and the gate's shape come from
    ``_build_qwen3_moe_layers``/``make_router_config`` and not from values
    retyped here. That matters more for this scenario than for the norm
    scenarios: the norms are a pure function of ``dim``, while a router node
    carries five behavioural fields that each change what the arm computes.

    ``load_balance_coeff`` is checked because it decides the router's second
    argument. ``MoE.__init__`` builds ``expert_bias_E`` only when the
    coefficient is not None (``moe.py:388-397``) and ``MoE.forward`` passes
    whatever it built (``:459``). Our config sets it to None
    (``config_registry.py:108-111``), so production calls the router with
    ``None`` and so does this arm. A build where it is not None would have
    production adding a bias to the selection scores that megatron's base
    profile does not add.

    Compiled, because that is what a titan module faces end to end. The three
    mcore arms are eager, because megatron compiles no whole layer; the ratio
    between them is a comparison of two treatments, and every table says so.
    """
    from benchmarks.models.piper_qwen3.config_registry import _piper_1b_model

    moe_config = _piper_1b_model(fuse_qkv=True, shape=shape).layers[LAYER].moe
    if moe_config.load_balance_coeff is not None:
        raise RuntimeError(
            f"{TITAN_ARM_NAME}: the production MoE node sets "
            f"load_balance_coeff={moe_config.load_balance_coeff!r}, so "
            "MoE.forward would hand the router an expert_bias_E buffer. This "
            "arm calls the router with None, which would then be a different "
            "function from the one production runs."
        )
    config = moe_config.router
    _assert_titan_router_config(config, shape)

    module = config.build().to(inputs.x.device)
    module.load_state_dict({"gate.weight": inputs.gate_weight})
    module.to(torch.bfloat16)
    if type(module).__name__ != TITAN_ROUTER_CLASS:
        raise RuntimeError(
            f"{TITAN_ARM_NAME}: the production node built "
            f"{type(module).__name__}, not {TITAN_ROUTER_CLASS}"
        )
    compiled = _compile_module(module)

    def call(leaf: torch.Tensor):
        # The engine's own entry point, called the way MoE.forward calls it:
        # the router, the hidden state, and the expert bias -- None here,
        # because load_balance_coeff is None and the check above proves it.
        return compiled(leaf, None)

    # One untimed forward, for two things at once: the fp32 guard, and the
    # selected experts the native gradient is gathered at. The selection is a
    # deterministic function of the fixed inputs, so the indices this call
    # produces are the indices every later call produces.
    #
    # Grad-enabled on purpose, because the timed closures are. It does NOT
    # save a compile, and an earlier version of this comment claimed it did.
    # Dynamo's tensor guards include requires_grad and the probe input is not
    # a leaf that requires it, so the probe compiles its own graph and the
    # first timed call compiles a second -- measured on CPU at 1 graph after
    # the probe and 2 after the first timed leaf, identically for a no_grad
    # probe and a grad-enabled one. What grad-enabled buys is fidelity: the
    # probe runs the branches the timed closures run, and torch.is_grad_enabled
    # gates real work on this path. The build pays one extra compile, outside
    # every timed region. The outputs are detached immediately, so the probe's
    # autograd graph is released here.
    top_scores, top_indices, scores = call(inputs.x)
    _assert_titan_router_is_fp32(scores, top_scores)
    top_scores, top_indices = top_scores.detach(), top_indices.detach()
    del scores
    grad_native = inputs.grad_probs.gather(-1, top_indices).to(top_scores.dtype)

    tokens = workload.batch * workload.seq_len

    def canonical(
        x_grad: torch.Tensor, result: Any
    ) -> dict[str, torch.Tensor]:
        top_scores_BLK, top_indices_BLK, _ = result
        flat_indices = top_indices_BLK.reshape(tokens, shape.top_k)
        flat_scores = top_scores_BLK.detach().reshape(tokens, shape.top_k).float()
        zeros = torch.zeros(
            (tokens, shape.num_experts),
            device=flat_scores.device,
            dtype=torch.float32,
        )
        # The gate logits, which titan's router does not return. Recomputed
        # out of band, eagerly, through the module's own submodule and its own
        # autocast -- the two lines moe.py:292-293 run. It therefore tests
        # that both engines hold the same gate and compute the same GEMM; it
        # does not test the compiled lowering, and no gate in this scenario
        # claims to.
        with torch.no_grad(), torch.autocast(
            device_type=inputs.x.device.type, dtype=torch.float32
        ):
            logits = module.gate(inputs.x)
        return _canonical_outputs(
            arm=TITAN_ARM_NAME,
            shape=shape,
            workload=workload,
            probs=zeros.scatter(1, flat_indices, flat_scores),
            routing_map=zeros.scatter(
                1, flat_indices, torch.ones_like(flat_scores)
            ),
            logits=logits.reshape(tokens, shape.num_experts),
            x_grad=x_grad,
            gate_weight_grad=module.gate.weight.grad,
        )

    print(
        f"moe_router/{TITAN_ARM_NAME}: {type(module).__name__} "
        f"(top_k={config.top_k}, score_func={config.score_func!r}, "
        f"route_norm={config.route_norm}, expert_bias=None), compiled, "
        f"gate GEMM under torch.autocast(float32)",
        flush=True,
    )
    return _router_arm(
        name=TITAN_ARM_NAME,
        owner=module,
        call=call,
        backward_target=lambda result: result[0],
        canonical=canonical,
        x_native=inputs.x,
        grad_native=grad_native,
        bytes_moved=inputs.titan_bytes_moved,
        notes={
            "router": TITAN_ROUTER_CLASS,
            "compiled": True,
            "router_precision": "fp32 (torch.autocast)",
        },
    )


# --------------------------------------------------------------------------
# Megatron-core
# --------------------------------------------------------------------------


def _assert_unflatten_is_an_early_return(layer: Any, arm: str) -> None:
    """Prove the activation unflatten writes nothing under our driver.

    Plan section C.1 puts "activation unflatten" inside this scenario's mcore
    cut. In this build it is an early return, so it is charged to neither
    engine -- and that claim is proved here rather than asserted in a
    docstring, against a ``PackedSeqParams`` built exactly as
    ``benchmarks/e2e/megatron/train.py:273-279`` builds it.

    Two conditions, and both must be checked. ``_maybe_unflatten_for_moe``
    returns early when the layer is not an MoE layer *or* when
    ``tokens_per_sample`` is None (``transformer_layer.py:775-780``), so the
    early return alone would also be satisfied by a dense layer -- which has
    no router at all and would make every other number here meaningless. The
    MoE check therefore comes first.

    The method is private. Calling it here is evidence, not measurement: it
    runs once at build time, outside every timed region, and its result is
    used only to decide whether this scenario may be measured at all.
    """
    if not getattr(layer, "is_moe_layer", False):
        raise RuntimeError(
            f"{arm}: transformer layer {LAYER} is not an MoE layer, so it has "
            "no router to measure"
        )
    unflatten = getattr(layer, "_maybe_unflatten_for_moe", None)
    if unflatten is None:
        raise RuntimeError(
            f"{arm}: TransformerLayer has no _maybe_unflatten_for_moe; this "
            "megatron revision moved the activation unflatten, so the cut "
            "this scenario declares must be re-derived before it measures "
            "anything"
        )
    from megatron.core.packed_seq_params import PackedSeqParams

    device = next(layer.parameters()).device
    cu_seqlens = torch.tensor([0, 2, 4], dtype=torch.int32, device=device)
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=cu_seqlens,
        cu_seqlens_kv=cu_seqlens,
        max_seqlen_q=4,
        max_seqlen_kv=4,
    )
    probe = torch.zeros(4, 1, layer.config.hidden_size, device=device)
    hidden, _, mbs = unflatten(probe, None, packed)
    if mbs is not None or hidden is not probe:
        raise RuntimeError(
            f"{arm}: _maybe_unflatten_for_moe transposed its input under the "
            "driver's PackedSeqParams. It is inside this scenario's declared "
            "cut, so its copy must be charged to both engines rather than to "
            "megatron alone; re-cut the scenario before measuring it."
        )


def _assert_mcore_router(
    arm: str, moe_layer: Any, router: Any, shape: PiperShape
) -> dict[str, Any]:
    """Refuse to time a router that is not the one this arm's label claims.

    Every check raises, for the reason ``_assert_titan_router_config`` gives.
    The list has three parts.

    **The delta.** ``moe_router_dtype`` and ``moe_router_fusion`` must be what
    the arm's profile declares. This is the check plan section C.3 says a
    verifier hunts for: a variant arm whose flag never reached the built
    config measures the base and publishes it under the variant's name, and
    no correctness gate can see it because both are numerically valid.

    **The classes.** ``MoELayer`` and ``TopKRouter``, because
    ``InferenceTopKRouter`` returns a compact ``[tokens, topk]`` index routing
    instead of a dense map (``moe_module_specs.py:86-89``), which is a
    different operation returning different shapes.

    **The dead branches.** ``TopKRouter.forward`` and ``.routing`` hold seven
    optional stages, and each one that is live adds work titan has no
    counterpart for: the input jitter (``router.py:709-729``), the z-loss
    (``:642-707``), the forced-load-balancing and forced-bias overrides
    (``:856-864``), the aux losses (``:802-829``), token dropping
    (``:789-798``), the expert bias (``:731-742``) and the routing replay
    (``:258-260``). Every one is off under the base profile. Asserting that
    is what lets this scenario say it measures a gate GEMM, a top-k and a
    softmax -- and ``is_aux_loss_enabled`` in particular is what makes the
    second ``moe_router_fusion`` read site (``:808``) dead, so
    ``mcore/router_fusion``'s delta is exactly one kernel substitution.
    """
    profile = MCORE_ARM_PROFILES[arm]
    config = router.config
    layer_class = type(moe_layer).__name__
    router_class = type(router).__name__
    if layer_class != MCORE_LAYER_CLASS:
        raise RuntimeError(
            f"{arm}: {mcore_moe_layer_path()} is {layer_class}, not "
            f"{MCORE_LAYER_CLASS}; the spec derivation took another branch"
        )
    if router_class != MCORE_ROUTER_CLASS:
        raise RuntimeError(
            f"{arm}: the MoE layer built {router_class}, not "
            f"{MCORE_ROUTER_CLASS}; an inference router returns a compact "
            "index routing, not the dense map this scenario compares"
        )

    # ``moe_router_dtype`` must be PRESENT, and the two lookups below are
    # deliberately asymmetric. For ``moe_router_fusion`` an absent key and
    # ``False`` are the same state: the dataclass default is False
    # (``transformer_config.py:915``) and ``BASE`` omits the key on purpose,
    # so ``.get(..., False)`` reads the truth. For ``moe_router_dtype`` an
    # absent key and ``None`` are the same VALUE and opposite INTENTIONS --
    # the dataclass default is also None (``transformer_config.py:790``), so
    # a key deleted from ``BASE`` would give expected None against a built
    # config of None, this check would pass, and ``mcore/base`` would route
    # in bf16 while its label and this scenario's whole headline say fp32.
    # That is exactly the silent precision change every guard here exists to
    # prevent, so the key is required rather than defaulted.
    if "moe_router_dtype" not in profile.config_overrides:
        raise RuntimeError(
            f"{arm}: profile {profile.name!r} does not state "
            "moe_router_dtype. Absent and None are the same value here and "
            "the opposite intention, so this scenario cannot tell a profile "
            "that chose bf16 from one that forgot to choose. Every profile "
            "this scenario builds must say which precision it routes in."
        )
    expected_dtype = profile.config_overrides["moe_router_dtype"]
    expected_fusion = profile.config_overrides.get("moe_router_fusion", False)
    deltas: tuple[tuple[str, Any, Any], ...] = (
        ("moe_router_dtype", config.moe_router_dtype, expected_dtype),
        ("moe_router_fusion", config.moe_router_fusion, expected_fusion),
    )
    for field, actual, expected in deltas:
        if actual != expected:
            raise RuntimeError(
                f"{arm}: profile {profile.name!r} declares {field}="
                f"{expected!r} and the built config has {actual!r}. The arm "
                "would measure another profile's implementation under this "
                "arm's label."
            )

    # Split from the "must be absent" list below, because several of those
    # hold tensors when they are live and ``tensor != None`` is not a
    # comparison this code may perform.
    equal: tuple[tuple[str, Any, Any], ...] = (
        ("config.num_moe_experts", config.num_moe_experts, shape.num_experts),
        ("config.moe_router_topk", config.moe_router_topk, shape.top_k),
        (
            "config.moe_router_score_function",
            config.moe_router_score_function,
            "softmax",
        ),
        ("config.moe_router_pre_softmax", config.moe_router_pre_softmax, False),
        ("router.routing_type", router.routing_type, "none"),
        ("router.enable_expert_bias", router.enable_expert_bias, False),
        (
            "config.moe_router_force_load_balancing",
            config.moe_router_force_load_balancing,
            False,
        ),
        ("router.is_aux_loss_enabled()", router.is_aux_loss_enabled(), False),
        ("config.cuda_graph_impl", config.cuda_graph_impl, "none"),
    )
    for field, actual, expected in equal:
        if actual != expected:
            raise RuntimeError(
                f"{arm}: {field} is {actual!r}, expected {expected!r}. This "
                "scenario measures a gate GEMM, a top-k and a softmax; that "
                "setting adds a stage titan's router has no counterpart for."
            )

    absent: tuple[tuple[str, Any], ...] = (
        (
            "config.moe_router_topk_scaling_factor",
            config.moe_router_topk_scaling_factor,
        ),
        ("config.moe_router_num_groups", config.moe_router_num_groups),
        ("config.moe_router_group_topk", config.moe_router_group_topk),
        ("config.moe_z_loss_coeff", config.moe_z_loss_coeff),
        ("config.moe_input_jitter_eps", config.moe_input_jitter_eps),
        ("config.moe_expert_capacity_factor", config.moe_expert_capacity_factor),
        ("config.moe_router_force_biased", config.moe_router_force_biased),
        ("router.expert_bias", router.expert_bias),
        ("router.bias", router.bias),
        ("router.router_replay", router.router_replay),
    )
    for field, actual in absent:
        if actual is not None:
            raise RuntimeError(
                f"{arm}: {field} is set, so routing() runs a stage this "
                "scenario does not measure and titan's router has no "
                "counterpart for."
            )

    # A process global, not a config field, and it belongs in this guard for
    # the same reason the config fields do: it changes which implementation
    # runs. With deterministic algorithms on, the UNFUSED score function
    # builds its dense routing through index_put_ instead of scatter
    # (``moe_utils.py:934``), and the FUSED arm never reaches that code --
    # ``topk_routing_with_score_function`` returns into TE at ``:831``. So the
    # flag moves ``mcore/base`` and ``mcore/router_bf16`` and leaves
    # ``mcore/router_fusion`` where it was, and the difference lands inside
    # the fusion row as if it were the fusion.
    if torch.are_deterministic_algorithms_enabled():
        raise RuntimeError(
            f"{arm}: torch.are_deterministic_algorithms_enabled() is True. "
            "It selects a different dense-routing construction in megatron's "
            "unfused score function and none at all in the fused one, so the "
            "mcore/router_fusion row would measure this flag as well as the "
            "fusion."
        )

    if not router.training:
        raise RuntimeError(
            f"{arm}: the router is in eval mode, so routing() skips the "
            "branches production runs; both engines are measured training"
        )
    expected_shape = (shape.num_experts, shape.dim)
    if tuple(router.weight.shape) != expected_shape:
        raise RuntimeError(
            f"{arm}: the gate matrix is {tuple(router.weight.shape)}, "
            f"expected {expected_shape}; the attribute path "
            f"{mcore_moe_layer_path()!r} reached the wrong module"
        )
    return {
        "router": router_class,
        "moe_layer": layer_class,
        "profile": profile.name,
        "moe_router_dtype": config.moe_router_dtype,
        "moe_router_fusion": config.moe_router_fusion,
        "compiled": False,
    }


def _assert_te_router_fusion_is_reachable(arm: str) -> None:
    """Refuse ``mcore/router_fusion`` when TE ships no fused router kernel.

    ``topk_routing_with_score_function`` raises rather than degrading when the
    symbol is missing (``moe_utils.py:822-825``), so there is no silent
    fallback to guard against and no trace marker is needed -- the same
    argument the ``flex_flash`` arm's provenance makes about ``BACKEND=
    "FLASH"``. What this adds is the failure *at build time*, with the reason,
    instead of a ValueError from inside the first timed closure.

    The symbol is bound at import from
    ``transformer_engine.pytorch.router`` when TE is at least 2.7.0.dev
    (``megatron/core/extensions/transformer_engine.py:3597-3607``); this box
    ships TE 2.17.1, but that is a property of the environment and not of the
    declaration.
    """
    from megatron.core.transformer.moe import moe_utils

    if getattr(moe_utils, "fused_topk_with_score_function", None) is None:
        raise RuntimeError(
            f"{arm}: megatron bound fused_topk_with_score_function to None, "
            "so TransformerEngine ships no fused router kernel here (it needs "
            "TE >= 2.7.0.dev). The arm has nothing to measure."
        )


def _assert_mcore_gating_gemm_takes_the_te_path(arm: str, router: Any) -> None:
    """Refuse a megatron arm whose gate GEMM would upcast BOTH operands.

    ``mcore_bytes_moved`` is a DECLARED constant, and what it declares is the
    TransformerEngine branch of ``RouterGatingLinearFunction.forward``
    (``moe_utils.py:1380-1382``): TE receives the bf16 operands and returns an
    fp32 output, so the arm reads the hidden state once.

    The ``elif`` below it (``moe_utils.py:1383-1388``) is a different cost. It
    runs ``inp.to(router_dtype)``, which materializes a full fp32 copy of the
    ``[T, D]`` hidden state -- the same copy titan's autocast makes, and the
    copy this scenario's headline finding says megatron does NOT make. On that
    branch ``mcore_bytes_moved`` is about five times too small, so the GB/s
    column is about five times too high, and the finding inverts from "titan
    moves five times more" into "the two engines move the same". Nothing in a
    run would say so, because the byte count is declared rather than measured.

    **This is a cost guard and not a correctness guard.** A bf16 value upcasts
    to fp32 exactly, so both branches sum the same products and differ only in
    accumulation order. No gate can see the difference, which is why a guard
    has to.

    Both halves of megatron's own condition are checked, because both select
    the same ``elif``:

    * ``te_general_gemm is None``. Every fused symbol in ``moe_utils`` is
      bound to ``None`` together when TransformerEngine does not import
      (``moe_utils.py:35-61``), so a worker that cannot load TE takes the
      fallback silently.
    * ``router_dtype == torch.float64``. No profile sets it today -- ``BASE``
      routes fp32 and ``ROUTER_BF16`` routes the input dtype -- and a future
      fp64 profile would take the fallback with TE present.

    The dtype is read off the BUILT config, the way ``Router.gating`` reads it
    (``router.py:105-111``), rather than off the profile, so a delta that did
    not take is caught here too.
    """
    from megatron.core.transformer.moe import moe_utils

    if getattr(moe_utils, "te_general_gemm", None) is None:
        raise RuntimeError(
            f"{arm}: megatron bound te_general_gemm to None, so "
            "RouterGatingLinearFunction takes the torch.mm fallback "
            "(moe_utils.py:1383-1388). That branch upcasts BOTH operands and "
            "materializes an fp32 copy of the hidden state, which is the copy "
            "mcore_bytes_moved declares megatron does not make -- the GB/s and "
            "x_floor columns would be about 5x wrong and this scenario's "
            "cross-engine finding would invert. TransformerEngine must import "
            "in this worker."
        )
    if router.config.moe_router_dtype == "fp64":
        raise RuntimeError(
            f"{arm}: moe_router_dtype is 'fp64', so "
            "RouterGatingLinearFunction skips TE and takes the torch.mm "
            "fallback (moe_utils.py:1380,1383-1388) even with "
            "TransformerEngine present. That branch upcasts both operands, "
            "which mcore_bytes_moved does not describe. Declare a byte count "
            "for that path before measuring an fp64 router."
        )


def _release_untimed_moe_submodules(moe_layer: Any, arm: str) -> int:
    """Drop the expert weights, which this scenario never calls.

    ``memory_pass`` reads ``torch.cuda.max_memory_allocated``, which is a
    total and not a delta, so anything still resident is charged to the arm.
    One MoE layer's experts are 44 M parameters at the normal shape and 6.3 G
    at the huge one, against activations measured in megabytes -- so a
    retained expert stack would make this scenario's memory column a statement
    about what each arm happened to keep, and titan's router holds nothing but
    an ``[E, D]`` gate.

    ``route`` provably does not reach them: it is
    ``apply_module(self.router)(hidden_states, padding_mask)``
    (``moe_layer.py:444``) under a decorator that reads
    ``self.config.cuda_graph_impl`` and returns (``moe_utils.py:1680-1681``).
    ``del`` rather than ``= None``, so a revision that starts touching
    ``self.experts`` raises ``AttributeError`` instead of reading a None that
    some other branch could interpret as "no experts". The caller proves the
    method still runs immediately afterwards.
    """
    freed = 0
    for name in ("experts", "shared_experts"):
        child = getattr(moe_layer, name, None)
        if child is None:
            continue
        freed += sum(p.numel() for p in child.parameters())
        delattr(moe_layer, name)
    if freed == 0:
        raise RuntimeError(
            f"{arm}: the MoE layer holds no expert parameters, so it is not "
            "the layer this scenario navigated to"
        )
    return freed


def _build_mcore_arm(
    arm: str,
    profile: McoreProfile,
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: MoeRouterInputs,
) -> BuiltArm:
    """The body all three megatron arms share; only the profile differs.

    The MoE layer is taken out of a real ``GPTModel`` built by
    ``benchmarks.models.piper_qwen3.megatron_model.build_model`` from the same
    ``PiperShape`` and the same profile family the e2e megatron arm uses. A
    hand-constructed ``TopKRouter`` would take a ``TransformerConfig`` and a
    ``ProcessGroupCollection`` written a second time, and it would not give
    this arm ``MoELayer.route`` -- the entry point plan section C.1 names.

    Eager on purpose, with the one ``@jit_fuser`` exception this module's
    docstring states. ``KernelArm.eager_reason`` records both halves.
    """
    initialize_megatron_single_rank()

    from benchmarks.models.piper_qwen3.megatron_model import build_model

    model = build_model(seq_len=workload.seq_len, shape=shape, profile=profile)
    layer = _navigate(model, mcore_transformer_layer_path())
    _assert_unflatten_is_an_early_return(layer, arm)
    moe_layer = _navigate(model, mcore_moe_layer_path())
    router = moe_layer.router
    notes = _assert_mcore_router(arm, moe_layer, router, shape)
    _assert_mcore_gating_gemm_takes_the_te_path(arm, router)
    if profile.config_overrides.get("moe_router_fusion", False):
        _assert_te_router_fusion_is_reachable(arm)

    if router.weight.shape != inputs.gate_weight.shape:
        raise RuntimeError(
            f"{arm}: the router gate is {tuple(router.weight.shape)} and the "
            f"shared inputs hold {tuple(inputs.gate_weight.shape)}"
        )
    with torch.no_grad():
        router.weight.copy_(inputs.gate_weight)

    freed = _release_untimed_moe_submodules(moe_layer, arm)
    del model, layer
    gc.collect()
    torch.cuda.empty_cache()

    tokens = workload.batch * workload.seq_len

    def call(leaf: torch.Tensor):
        # Megatron's own entry point, called the way MoELayer.forward calls it
        # (moe_layer.py:655, with padding_mask None -- our driver passes none
        # and GPTModel defaults it to None).
        return moe_layer.route(leaf)

    # One untimed forward, which proves route() still runs after the expert
    # stack was released and gives the probs dtype the native gradient needs.
    # Grad-enabled, matching the timed closures, and detached immediately so
    # the probe's autograd graph is released here.
    probe_probs, probe_map = call(inputs.x.reshape(tokens, 1, shape.dim))
    probe_probs, probe_map = probe_probs.detach(), probe_map.detach()
    for name, tensor, expected in (
        ("probs", probe_probs, (tokens, shape.num_experts)),
        ("routing_map", probe_map, (tokens, shape.num_experts)),
    ):
        if tuple(tensor.shape) != expected:
            raise RuntimeError(
                f"{arm}: route() returned {name} of {tuple(tensor.shape)}, "
                f"expected {expected}"
            )
    grad_native = inputs.grad_probs.reshape(
        tokens, shape.num_experts
    ).to(probe_probs.dtype)

    def canonical(x_grad: torch.Tensor, result: Any) -> dict[str, torch.Tensor]:
        probs, routing_map = result
        # The gate logits, through megatron's own method, out of band. The
        # titan arm recomputes its logits the same way and for the same
        # reason: neither router returns them.
        with torch.no_grad():
            logits = router.gating(
                inputs.x.reshape(tokens, 1, shape.dim)
            ).reshape(tokens, shape.num_experts)
        return _canonical_outputs(
            arm=arm,
            shape=shape,
            workload=workload,
            probs=probs.detach(),
            routing_map=routing_map.detach(),
            logits=logits,
            x_grad=x_grad,
            gate_weight_grad=router.weight.grad,
        )

    print(
        f"moe_router/{arm}: {mcore_moe_layer_path()}.router is "
        f"{type(router).__module__}.{type(router).__qualname__} "
        f"(profile {profile.name}, moe_router_dtype="
        f"{router.config.moe_router_dtype!r}, moe_router_fusion="
        f"{router.config.moe_router_fusion}), eager except the @jit_fuser "
        f"_apply_expert_bias; released {freed} expert parameters",
        flush=True,
    )
    return _router_arm(
        name=arm,
        owner=router,
        call=call,
        backward_target=lambda result: result[0],
        canonical=canonical,
        x_native=inputs.x.reshape(tokens, 1, shape.dim),
        grad_native=grad_native,
        bytes_moved=inputs.mcore_bytes_moved,
        notes=notes,
    )


def build_moe_router_mcore_base(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeRouterInputs
) -> BuiltArm:
    """Megatron-core's router as our megatron arm runs it: fp32, unfused."""
    return _build_mcore_arm(MCORE_BASE_ARM_NAME, BASE, shape, workload, inputs)


def build_moe_router_mcore_router_fusion(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeRouterInputs
) -> BuiltArm:
    """The same router with ``moe_router_fusion=True``.

    One substitution, and its scope is exactly one call site under this
    profile. ``moe_router_fusion`` reaches ``routing()`` twice -- at
    ``router.py:786`` for the top-k score function and at ``:808`` for the
    aux-loss scores -- and the second is inside
    ``if ... self.is_aux_loss_enabled()`` (``:802``), which
    ``_assert_mcore_router`` proves is False. So the delta replaces megatron's
    torch sequence (a ``topk``, a ``softmax``, two ``zeros_like`` and two
    ``scatter``s, ``moe_utils.py:900-949``) with one TransformerEngine
    autograd function (``transformer_engine/pytorch/router.py:68-125``).
    """
    return _build_mcore_arm(
        MCORE_FUSION_ARM_NAME, ROUTER_FUSION, shape, workload, inputs
    )


def build_moe_router_mcore_router_bf16(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeRouterInputs
) -> BuiltArm:
    """The same router with ``moe_router_dtype=None``: bf16 routing.

    **The delta is bf16, not fp32.** ``mcore_profiles.BASE`` already sets
    ``moe_router_dtype="fp32"``, so this arm removes the fp32 request rather
    than adding it: ``Router.gating`` falls through to ``router_dtype =
    input.dtype`` (``router.py:105-110``) and the gate GEMM writes bf16
    logits. The top-k then selects on bf16 values, so this arm may legitimately
    route a token to a different expert than any fp32 arm. That is the
    measurement, not a fault, and it is why no gate enforces this arm's
    selection.
    """
    return _build_mcore_arm(
        MCORE_BF16_ARM_NAME, ROUTER_BF16, shape, workload, inputs
    )
