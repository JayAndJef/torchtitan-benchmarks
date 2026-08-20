"""Arm builders for the ``moe_combine`` kernel scenario.

The step that puts routed expert outputs back into token order, on both
engines:

* TorchTitan: ``token_dispatcher.combine(...)``
  (``third_party/torchtitan/torchtitan/models/common/moe.py:162``, inside
  ``RoutedExperts.forward``).
* Megatron-core: **all three combine phases** of the token dispatcher --
  ``combine_preprocess``, ``token_combine`` and ``combine_postprocess``
  (``third_party/Megatron-LM/megatron/core/transformer/moe/
  token_dispatcher.py:334``, ``:352`` and ``:367`` for the allgather class).

**Why all three phases, and why cutting anywhere else measures a zero.**
``MoEAllGatherTokenDispatcher.token_combine`` is guarded by ``if self.tp_size
> 1 or self.ep_size > 1`` (``token_dispatcher.py:361``). At ``world_size=1``
it is **the identity**. ``MoELayer.combine`` (``moe_layer.py:566-572``) is a
thin wrapper around exactly that method and nothing else, so a cut there
would have measured zero on the megatron side and published a spectacular
win. The work lives in the neighbouring phases: ``unpermute(...)`` runs in
``combine_preprocess`` (``:348``).

The cut names the whole triple rather than the phase that holds the unpermute
today, because the two dispatcher classes place it differently. The allgather
class unpermutes in ``combine_preprocess``; the alltoall class unpermutes in
``combine_postprocess`` (``:873-893``) and does its chunk unsort in
``combine_preprocess`` (``:793-832``). Naming the triple keeps the cut correct
for either, and for a dispatcher that has not been written yet.

**The cut is on the dispatcher's phase API, not on
``MoELayer.routed_experts_compute``.** That method is decorated
``@internal_api`` (``moe_layer.py:528``), which exempts it from megatron's
backward-compatibility guarantee. The three phase methods are declared on the
public ``MoETokenDispatcher`` base (``token_dispatcher.py:159``, ``:179``,
``:194``).

**THIS SCENARIO PUBLISHES NO CROSS-ENGINE RATIO.** The registry says so in an
explicit ``comparisons`` tuple rather than by leaving it ``None``, because
left ``None`` the schema derives every non-floor arm against the anchor and
would publish exactly the row that must not be published.

**The reason: the routing probabilities are applied on opposite sides of this
boundary.** Megatron multiplies them inside ``TEGroupedMLP``
(``weighted_bias_swiglu_impl``), which is scenario 11 ``expert_mlp``; its
combine never sees them -- the allgather ``combine_preprocess`` calls
``unpermute`` with no ``probs`` argument, and so does the alltoall
``combine_postprocess``. TorchTitan applies them here:
``LocalTokenDispatcher.combine`` (``torchtitan/models/common/
token_dispatcher.py:142``) is documented "Score and scatter_add routed expert
outputs" (``:151``) and multiplies by
``metadata.topk_scores_experts_sorted_N`` at ``:169-171``, before the
scatter-add. Note the engine qualifier on that path: both engines ship a file
named ``token_dispatcher.py``, and every unqualified ``token_dispatcher.py``
citation in this module is megatron's. So the two sides compute **different
functions** of the same routed rows:

* megatron: ``out[t] = sum_k y[row(t, k)]``
* titan:    ``out[t] = sum_k p[t, k] * y[row(t, k)]``

A ratio between them would compare two different amounts of work. The
cross-engine row belongs to the ``expert_combine`` span (11+12), the smallest
enclosure in which both engines have applied the probabilities exactly once.
That span is deferred and is declared elsewhere.

**There is no cross-engine ``CorrectnessCheck`` either, and that is the same
fact stated once more.** Two engines that compute different functions have
nothing a gate can honestly assert. Making them agree would need either an
extra multiply charged to one engine inside its timed region -- which changes
the measurand -- or two different synthetic expert-output tensors, one
pre-weighted, which breaks the one-experiment property that both engines see
the same rows. Both are worse than declaring no gate. Each engine is gated
against its own fp64 truth instead, under its own output names
(``mcore_out`` / ``titan_out``), so a reader of the results file meets the
asymmetry rather than a missing row.

**The published rows are megatron's own, and there are two.**
``mcore/no_permute_fusion`` against ``mcore/base``, and
``mcore/dispatcher_alltoall`` against ``mcore/base``.

**``no_permute_fusion``: one read site inside this scenario, not the whole
scenario.** ``moe_permute_fusion`` gates TE's ``fused_unpermute``
(``moe_utils.py:545``) against the torch path (``:601-621``: a zeroed output
tensor and a ``scatter_add_`` with an expanded index, or ``index_add_`` when
deterministic algorithms are on). Under the **allgather** dispatcher this
scenario holds exactly **one** such read site -- ``unpermute`` in
``combine_preprocess`` (``token_dispatcher.py:348``) -- and it also holds
``token_combine`` and ``combine_postprocess``, which the flag does not touch.
Report the row as a comparison of the two unpermute implementations inside a
scenario that is larger than the unpermute.

``mcore/base`` really does take the fused path.
``TransformerConfig.__post_init__`` raises when ``moe_permute_fusion`` is on
and TE lacks any of the five fused symbols
(``transformer_config.py:2677-2694``), and this box runs TE 2.17.1 against a
2.1.0 floor. ``_assert_mcore_dispatcher`` re-checks ``fused_unpermute is not
None`` directly, because a raise inside a config constructor is evidence
about the config and not about the symbol this arm calls.

**``dispatcher_alltoall``: the collective is inert at ``world_size=1``, the
class is not.** ``_AllToAll.forward`` returns its input unchanged when
``group.size() == 1`` (``tensor_parallel/mappings.py:434-437``), so no bytes
move. What differs is the **local** work and the phase the unpermute sits in:

* allgather combine: ``unpermute`` in ``combine_preprocess``, an inert
  ``token_combine``, then a ``view`` in ``combine_postprocess``.
* alltoall combine: a chunk unsort in ``combine_preprocess``
  (``sort_chunks_by_idxs``, a real row copy even though the chunk order is
  the identity at one rank -- ``restore_output_by_local_experts`` is
  ``[0, 1, ..., E-1]``), an inert ``token_combine`` that is still an
  ``autograd.Function.apply``, then ``unpermute`` plus a ``view`` in
  ``combine_postprocess``.

**Caption: this arm measures the dispatcher's local unpermute and sync
strategy, NOT communication.** Say it beside any number, or a reader will
take it for a communication result.

**Every number here is device time plus host dispatch, and no arm carries a
blocking device-to-host synchronization inside its timed region.** This is the
half of plan A.2 that scenario 10 ``dispatch_permute`` cannot say: the
allgather ``.cpu()`` at ``token_dispatcher.py:317`` and every
``_maybe_dtoh_and_synchronize`` call site sit in the **dispatch** phases,
which this scenario runs once at build time and never times. The one path
that would reintroduce a sync is ``sort_chunks_by_idxs`` unfused, which calls
``split_sizes.tolist()`` (``moe_utils.py:656``) -- reachable only by an
alltoall arm with ``moe_permute_fusion=False``, a combination this roster
deliberately does not declare.

**The floor is declared, and it is what decides whether the two published
rows are kernel results at all.** A combine is a scatter-add: it must read
``N x dim`` bf16 elements, add them ``top_k`` at a time, and write ``T x dim``
of them. That is bandwidth-bound by construction, which is why the plan names
this scenario and scenario 10 as the next best case for a floor after the
norms. ``copy_floor`` performs the same traffic and the same additions with
no indirection at all, so the ``x_floor`` column separates "this unpermute is
slow" from "this scenario is at bandwidth".

**TWO DERIVED COLUMNS STILL REPRODUCE THE SUPPRESSED CROSS-ENGINE RATIO, AND
NEITHER CAN BE TURNED OFF.** ``comparisons`` removes the *row*, not the
numbers a reader can divide.

* ``gbps`` is ``bytes_moved / median`` (``benchmarks/kernel/results/
  merge.py:426-427``), and every arm here hands the merge the SAME
  ``inputs.bytes_moved``. So ``gbps(titan) / gbps(mcore/base)`` is exactly the
  inverse of the median ratio this scenario declines to publish. This is the
  stronger of the two channels, because the denominator is identical by
  construction and the column looks like a bandwidth achievement rather than
  like a ratio.
* ``x_floor`` is ``median / floor_median`` for every non-floor arm that shares
  a mode with a floor (``merge.py:428-429``), so the two arms' ``x_floor``
  values divide to the same forbidden number. It exists only in ``forward``,
  because the floor declares ``forward`` only.

The declaration cannot suppress either without deleting the floor or the
byte count, and the floor is the diagnostic that decides whether either
published row is a kernel result at all. So the scenario ``description``
states the hazard rather than removing it. The quotient is not merely
unpublished, it is wrong twice over: ``titan`` is the only compiled arm, and
it applies routing probabilities that megatron already applied in scenario 11.

**``gbps`` is a forward statement even in the ``forward_backward`` row**, as
in ``ffn_norm`` and ``moe_residual``: ``bytes_moved`` counts the forward
traffic and the merge divides it by whichever median it holds.

**Two asymmetries are declared rather than hidden. Neither reaches a ratio,
because no published row crosses the two engines.**

*The trailing reshape.* Megatron's ``combine_postprocess`` ends with
``hidden_states.view(self.hidden_shape)``, so the mcore triple contains its
shape restore. Titan's equivalent ``out_TD.view(B, -1, D)`` sits *after* the
combine call, on the next statement (``moe.py:171``, against the call at
``:162``), and is therefore outside the cut the plan names. Both are
metadata-only views on contiguous tensors and neither emits a kernel, so the
mcore side is charged one extra Python call and nothing else.

*The determinism mode of the scatter-add.* Titan's combine calls
``deterministic_scatter_add`` (``torchtitan/ops/scatter_add.py:11-20``), which
sets ``torch.use_deterministic_algorithms(True, warn_only=False)`` around its
own ``scatter_add`` and restores the previous value in a ``finally``. So the
titan arm runs a deterministic kernel on every call, inside the timed region.
Megatron's unfused ``unpermute`` takes ``scatter_add_``
(``moe_utils.py:619``) and reaches ``index_add_`` (``:612``) only when
deterministic algorithms are already enabled globally, which this harness
never does. A deterministic scatter-add is normally the slower kernel. This
is a second reason a cross-engine quotient here would be meaningless, and it
is recorded so that nobody computes one and reads it as a kernel difference.

**Both engines receive the same synthetic expert outputs, in one row order,
and that order is a checked fact rather than an assumption.** Every arm's own
dispatch runs at build time on the shared routing decision, outside every
timed closure, and only the combine is timed. The canonical row order is
``(expert ascending, token ascending)``:

* megatron's unfused ``permute`` argsorts ``routing_map.bool().T`` descending
  and stable (``moe_utils.py:467-475``), whose flat index is ``expert *
  num_tokens + token``;
* titan's ``_local_reorder`` argsorts ``topk_expert_ids.view(-1)`` stable
  (``torchtitan/models/common/token_dispatcher.py:93-95``), whose flat index
  is ``token * top_k +
  slot``, and a token reaches an expert at most once, so within one expert
  the flat order is the token order.

``moe_combine_inputs`` derives the order from the routing map, refuses to
continue unless titan's own argsort reproduces it, and
``_assert_mcore_combine`` refuses any megatron arm whose combine does not
reproduce the matching sum. That last guard is the one the plan requires: a
scenario whose mcore side is the identity at ``world_size=1`` reads as a
spectacular win rather than as a bug, so the guard **raises**. It is not
recorded in ``BuiltArm.notes``, because notes reach no artifact and a fact no
reader ever sees is not a guard.

Every torchtitan, megatron and TransformerEngine import is deferred into the
builder that needs it, which is the rule across ``operations/``.
``benchmarks.models.piper_qwen3.mcore_profiles`` is the one module-scope
exception, as in ``ffn_norm`` and ``moe_residual``: it is torch-free
parent-side data, and the two derived profiles below are declared from it.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Callable

import torch

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    _randn,
    _require_grads,
    _reset_grads,
    initialize_megatron_single_rank,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import (
    BASE,
    DISPATCHER_ALLTOALL,
    McoreProfile,
    NO_PERMUTE_FUSION,
)
from benchmarks.models.piper_qwen3.shape import PiperShape


# The five arm names, spelled as plan section C.2 spells them. The slash says
# which engine an arm is and which profile of that engine; ``schema.
# fragment_stem`` is what keeps it out of a fragment filename. ``copy_floor``
# carries no engine because it is not an implementation of anything, which is
# the spelling ``rope``, ``ffn_norm`` and ``moe_residual`` use for their
# floors.
COPY_FLOOR_ARM = "copy_floor"
MCORE_BASE_ARM = "mcore/base"
MCORE_NO_PERMUTE_FUSION_ARM = "mcore/no_permute_fusion"
MCORE_ALLTOALL_ARM = "mcore/dispatcher_alltoall"
TITAN_ARM = "titan"

# The layer the mcore arms read the dispatcher from. Every layer holds the
# same MoE layer at ``moe_layer_freq=1``, so the index is arbitrary and is
# fixed here so the provenance line can name it.
MCORE_LAYER = 0

# The comm backend our production config asks for
# (``benchmarks/models/piper_qwen3/config_registry.py:101``). It resolves to
# ``AllToAllTokenDispatcher`` (``models/common/config_utils.py:364-368``),
# whose ``combine`` delegates to ``LocalTokenDispatcher.combine`` when
# ``ep_mesh is None`` (``token_dispatcher.py:602-611``). The titan arm builds
# that class rather than ``LocalTokenDispatcher`` directly, so it pays the
# same delegation branch production pays.
TITAN_COMM_BACKEND = "standard"

# How far a built arm's combine may sit from the canonical unpermute before
# the build refuses to continue. Same value as the correctness gates: the
# quantity being compared is a bf16 accumulation of ``top_k`` terms against an
# fp32 truth, and a permutation that is wrong at all is wrong by ~100%, not by
# 3%.
COMBINE_GUARD_REL_L2 = 2e-2




@dataclass
class MoeCombineInputs:
    """One routing decision, one set of routed expert outputs, both engines.

    ``expert_out_ND`` is the tensor every arm combines. Its rows are in the
    canonical ``(expert ascending, token ascending)`` order that both engines'
    permutations produce, so no arm gathers it and no arm is charged for a
    layout adapter. ``row_token_N`` and ``row_expert_N`` are that order,
    derived from ``routing_map_TE`` alone. ``row_token_N`` is the working
    half: the reference, the floor's guard and both engines' checks index
    with it. ``row_expert_N`` is recorded rather than used, and that is
    deliberate -- it is the half that states the order is expert-MAJOR, and
    the tests assert the pair. A builder that ever consumes it is reading
    expert identity out of a synthetic tensor that carries none.

    The routing decision reaches both engines in the two forms they take:
    ``routing_map_TE`` plus dense ``probs_TE`` for megatron
    (``moe_utils.py:947-948`` builds exactly that pair), and
    ``topk_expert_ids_TK`` plus ``topk_scores_TK`` for titan
    (``models/common/moe.py:312-318``). They encode one decision, and
    ``moe_combine_inputs`` refuses to return a pair that does not.

    ``x_BLD`` is the MoE block input. Both engines read it for its shape
    alone inside the combine -- titan allocates ``torch.zeros_like(x_TD)`` and
    megatron restores ``self.hidden_shape`` -- so its values reach no number.
    It is a real draw anyway, because a zero tensor could not show an arm that
    returned its own input.
    """

    x_BLD: torch.Tensor  # (B, L, D) bf16, the MoE block input
    expert_out_ND: torch.Tensor  # (N = T*K, D) bf16, canonical row order
    grad_out_TD: torch.Tensor  # (T, D) bf16, incoming gradient of the combine
    routing_map_TE: torch.Tensor  # (T, E) bool, megatron form
    probs_TE: torch.Tensor  # (T, E) fp32 dense, megatron form
    topk_expert_ids_TK: torch.Tensor  # (T, K) int64, titan form
    topk_scores_TK: torch.Tensor  # (T, K) fp32, titan form
    tokens_per_expert_E: torch.Tensor  # (E,) int64, both engines' counts
    row_token_N: torch.Tensor  # (N,) int64, canonical row -> token
    row_expert_N: torch.Tensor  # (N,) int64, canonical row -> expert
    scores_sorted_N: torch.Tensor  # (N,) fp32, probs in canonical row order
    bytes_moved: int


def _canonical_row_order(
    routing_map_TE: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The ``(expert ascending, token ascending)`` row order, from the map.

    ``nonzero`` on the flattened transpose enumerates the True entries in
    row-major order over ``[num_experts, num_tokens]``, so the flat index is
    ``expert * num_tokens + token`` and the enumeration is already the order
    both engines' permutations produce. It is written with ``nonzero`` rather
    than with the ``argsort`` megatron uses because the two agree here and
    ``nonzero`` says what the result means.
    """
    num_tokens = routing_map_TE.shape[0]
    flat = routing_map_TE.t().reshape(-1)
    order = torch.nonzero(flat, as_tuple=False).squeeze(-1)
    return order % num_tokens, order // num_tokens


def moe_combine_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> MoeCombineInputs:
    batch, seq = workload.batch, workload.seq_len
    tokens = batch * seq
    experts, top_k = shape.num_experts, shape.top_k
    rows = tokens * top_k

    # Loud, with named numbers, rather than capped or rounded. The synthetic
    # routing below is the round-robin titan itself uses for a forced balance
    # (``models/common/moe.py:226-233``), and it is exactly balanced only when
    # the rows divide. At any other workload one expert's group is short, the
    # ``tokens_per_expert`` the manifest records stops describing the tensors
    # the arms hold, and nothing downstream would say so.
    if rows % experts:
        raise ValueError(
            f"moe_combine: {rows} routed rows (batch {batch} x seq {seq} x "
            f"top_k {top_k}) do not divide evenly among {experts} experts; "
            f"{rows % experts} rows would land in a short group and the "
            "recorded tokens_per_expert would describe a split the inputs "
            "do not have"
        )
    if top_k > experts:
        raise ValueError(
            f"moe_combine: top_k {top_k} exceeds num_experts {experts}, so a "
            "token would have to reach one expert twice and the canonical row "
            "order would not be a bijection"
        )

    x_BLD = _randn((batch, seq, shape.dim), device, generator)
    expert_out_ND = _randn((rows, shape.dim), device, generator)
    grad_out_TD = _randn((tokens, shape.dim), device, generator)

    # Round-robin top-k, which is titan's own balanced-routing path. The K
    # experts of one token are ``top_k`` consecutive residues, so they are
    # distinct whenever ``top_k <= num_experts``, and every expert receives
    # exactly ``rows // num_experts`` of them.
    topk_expert_ids_TK = (
        torch.arange(rows, device=device, dtype=torch.int64) % experts
    ).reshape(tokens, top_k)
    # Routing scores live in (0, 1) because titan's router is a sigmoid and
    # megatron's is a softmax; both are fp32 here, matching titan's fp32 gate
    # autocast and megatron's ``moe_router_dtype="fp32"``.
    topk_scores_TK = torch.sigmoid(
        torch.randn(
            (tokens, top_k), device=device, generator=generator, dtype=torch.float32
        )
    )

    routing_map_TE = torch.zeros(
        (tokens, experts), device=device, dtype=torch.bool
    ).scatter(1, topk_expert_ids_TK, True)
    probs_TE = torch.zeros(
        (tokens, experts), device=device, dtype=torch.float32
    ).scatter(1, topk_expert_ids_TK, topk_scores_TK)

    row_token_N, row_expert_N = _canonical_row_order(routing_map_TE)
    if row_token_N.numel() != rows:
        raise RuntimeError(
            f"moe_combine: the routing map holds {row_token_N.numel()} routed "
            f"pairs and top-k routing declares {rows}; a token reached one "
            "expert twice, so the two engine forms are not one decision"
        )
    # The linchpin, checked rather than asserted in prose: titan's own
    # ``_local_reorder`` argsort must reproduce the order derived from the
    # map. If it ever stops doing so, every arm still combines correctly but
    # they stop combining the SAME rows, and only this raises.
    titan_order = torch.argsort(topk_expert_ids_TK.reshape(-1), stable=True)
    if not torch.equal(titan_order // top_k, row_token_N):
        raise RuntimeError(
            "moe_combine: titan's expert-sorted token order does not match "
            "the order derived from the routing map, so the two engines would "
            "combine different rows of one expert-output tensor"
        )
    scores_sorted_N = topk_scores_TK.reshape(-1)[titan_order].contiguous()

    return MoeCombineInputs(
        x_BLD=x_BLD,
        expert_out_ND=expert_out_ND,
        grad_out_TD=grad_out_TD,
        routing_map_TE=routing_map_TE,
        probs_TE=probs_TE,
        topk_expert_ids_TK=topk_expert_ids_TK,
        topk_scores_TK=topk_scores_TK,
        tokens_per_expert_E=routing_map_TE.sum(dim=0).long(),
        row_token_N=row_token_N,
        row_expert_N=row_expert_N,
        scores_sorted_N=scores_sorted_N,
        bytes_moved=(rows + tokens) * shape.dim * expert_out_ND.element_size(),
    )


def moe_combine_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeCombineInputs
) -> dict[str, torch.Tensor]:
    """Two fp64 truths, because the two engines compute two functions.

    Named per engine on purpose. A single ``out`` would force one of the two
    sides to be gated against the other's function, and the difference is not
    a tolerance question -- titan's output is megatron's scaled row by row.
    The names are what carry that fact into ``results.json``.

    Nothing is quantized before promotion. Both operands are already the bf16
    and fp32 tensors the arms hold bit for bit, so there is no parameter whose
    rounding a truth could charge to an arm by accident.

    The two combined outputs are returned as ``[B, L, D]``, which is the
    canonical shape ``_combine_arm`` reshapes every arm's output to. The two
    gradients keep the ``[N, D]`` and ``[N]`` shapes of the tensors they
    belong to, because those are already engine-neutral.
    """
    canonical = (workload.batch, workload.seq_len, shape.dim)
    tokens = workload.batch * workload.seq_len
    grad = inputs.grad_out_TD.double()
    index = inputs.row_token_N.reshape(-1, 1).expand(-1, shape.dim)

    mcore_y = inputs.expert_out_ND.double().detach().requires_grad_()
    mcore_out = torch.zeros(
        (tokens, shape.dim), device=mcore_y.device, dtype=torch.float64
    ).scatter_add(0, index, mcore_y)
    torch.autograd.backward(mcore_out, grad)

    titan_y = inputs.expert_out_ND.double().detach().requires_grad_()
    titan_s = inputs.scores_sorted_N.double().detach().requires_grad_()
    titan_out = torch.zeros(
        (tokens, shape.dim), device=titan_y.device, dtype=torch.float64
    ).scatter_add(0, index, titan_y * titan_s.unsqueeze(-1))
    torch.autograd.backward(titan_out, grad)

    return {
        "mcore_out": mcore_out.detach().reshape(canonical),
        "mcore_expert_out_grad": mcore_y.grad,
        "titan_out": titan_out.detach().reshape(canonical),
        "titan_expert_out_grad": titan_y.grad,
        "titan_scores_grad": titan_s.grad,
    }


def _canonical_combine(
    inputs: MoeCombineInputs, dim: int, tokens: int
) -> torch.Tensor:
    """The unweighted combine of the shared rows, in fp32, on the device.

    The build-time guard compares every megatron arm against this. It is the
    same arithmetic ``moe_combine_reference`` performs for ``mcore_out``,
    computed in fp32 rather than fp64 because a guard runs on every build and
    an fp64 pass over ``N x dim`` is a cost the gates already pay once.
    """
    index = inputs.row_token_N.reshape(-1, 1).expand(-1, dim)
    return torch.zeros(
        (tokens, dim), device=inputs.expert_out_ND.device, dtype=torch.float32
    ).scatter_add(0, index, inputs.expert_out_ND.float())


def _rel_l2(value: torch.Tensor, truth: torch.Tensor) -> float:
    delta = (value.float() - truth.float()).norm()
    norm = truth.float().norm()
    return (delta / norm).item() if norm else delta.item()


# What a missing gradient means at this cut, handed to the shared guard as its
# ``detail``. A combine that dropped an operand is the failure it catches.
# Megatron has a spelling of it -- an unpermute that ignored its input would
# still return a correctly shaped zero tensor -- and so does titan, whose
# ``deterministic_scatter_add`` returns ``grad_output`` for its first argument
# and a gather for its third, so a wiring error can silence one without
# silencing the other.
MISSING_COMBINE_GRADIENT = (
    "the combine did not consume that operand, so the correctness "
    "gate has nothing to compare"
)


def _combine_arm(
    *,
    name: str,
    call: Callable[[torch.Tensor], torch.Tensor],
    inputs: MoeCombineInputs,
    grad_native: torch.Tensor,
    canonical: tuple[int, ...],
    extra_leaves: tuple[torch.Tensor, ...] = (),
    extra_outputs: Callable[[], dict[str, torch.Tensor | None]] = dict,
    output_prefix: str,
) -> BuiltArm:
    """The timed closures every implementation arm in this scenario shares.

    All four run this same code over their own ``call``, so a published row
    measures the two combines and nothing about how either arm was written.

    ``call`` takes the routed expert outputs and returns the combined tensor
    in that engine's native layout; ``canonical`` puts it back to ``[B, L, D]``
    for the gates, which run outside every timed region. ``grad_native`` is
    already in the shape the engine's output carries, because a reshape inside
    a timed closure would be timed as part of the combine.

    ``extra_leaves`` are the arm's other differentiable inputs -- titan's
    routing scores, which megatron's combine does not have -- and are reset
    beside the expert outputs so a repeated ``forward_backward`` measures one
    backward rather than a growing sum.

    The expert outputs get three separate leaves rather than one, because a
    timed backward that shared a ``.grad`` with the correctness hook would
    perturb the tensors the gates read. ``extra_leaves`` are NOT split that
    way: titan's ``scores_leaf`` is bound into the metadata object once, at
    build time, so all three closures share it. That is safe only because
    ``_reset_grads`` clears it at the top of both closures that read a
    gradient, and because correctness runs before any timing worker starts.
    Splitting it would mean rebuilding the metadata per closure, which is
    work production does inside ``dispatch`` and not inside ``combine``.
    """
    forward_leaf = inputs.expert_out_ND.clone().requires_grad_()
    round_trip_leaf = inputs.expert_out_ND.clone().requires_grad_()
    check_leaf = inputs.expert_out_ND.clone().requires_grad_()

    def forward():
        return call(forward_leaf)

    def forward_backward() -> None:
        _reset_grads(round_trip_leaf, *extra_leaves)
        torch.autograd.backward(call(round_trip_leaf), grad_native)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(check_leaf, *extra_leaves)
        out = call(check_leaf)
        torch.autograd.backward(out, grad_native)
        named: dict[str, torch.Tensor | None] = {
            f"{output_prefix}_out": out.detach().reshape(canonical),
            f"{output_prefix}_expert_out_grad": check_leaf.grad,
        }
        named.update(extra_outputs())
        return _require_grads(name, named, MISSING_COMBINE_GRADIENT)

    return BuiltArm(
        name=name,
        calls={"forward": forward, "forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
        bytes_moved=inputs.bytes_moved,
    )


def build_moe_combine_copy_floor(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeCombineInputs
) -> BuiltArm:
    """The floor: the same traffic and the same additions, with no indirection.

    A combine must read every one of the ``N x dim`` routed elements, add them
    ``top_k`` at a time, and write ``T x dim`` results. This arm does exactly
    that over a contiguous ``[T, top_k, dim]`` view and nothing else -- it is
    "the combine, if the routing had already put every token's rows side by
    side". So the gap between the floor and an arm is the cost of the
    permutation: the index read, the scattered write, and the kernel's own
    dispatch.

    **This arm decides whether the two published rows are kernel results at
    all.** A scatter-add at these shapes is bandwidth-bound, which is why the
    plan names this scenario and scenario 10 as the strongest floor candidates
    after the norms. Without it a reader cannot tell "TE's fused unpermute is
    faster than the torch path" from "both arms are at the bus", and the
    ``--burst`` residual cannot tell them apart either: CLAUDE.md records that
    a ladder can plateau at a dispatch cost bursting never amortizes, so an
    unflagged row is not evidence of device-boundedness.

    The view is over the canonical row order, where ``top_k`` consecutive rows
    belong to ONE expert and to ``top_k`` different tokens. That makes the
    floor's sum arithmetically unrelated to any arm's output, which is the
    point: a floor is a bandwidth statement, not an implementation, so it
    carries no correctness outputs and no gate.

    Forward only, for ``ffn_norm``'s reason: the forward traffic is exactly
    this, and a floor for the forward+backward traffic would be an invention
    rather than a measurement.
    """
    tokens = workload.batch * workload.seq_len
    source = inputs.expert_out_ND.view(tokens, shape.top_k, shape.dim)
    out = torch.empty(
        (tokens, shape.dim),
        device=inputs.expert_out_ND.device,
        dtype=inputs.expert_out_ND.dtype,
    )

    def forward() -> None:
        torch.sum(source, dim=1, out=out)

    return BuiltArm(
        name=COPY_FLOOR_ARM,
        calls={"forward": forward},
        correctness_outputs=dict,
        bytes_moved=inputs.bytes_moved,
    )


def build_moe_combine_titan(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeCombineInputs
) -> BuiltArm:
    """TorchTitan's ``token_dispatcher.combine(...)``, compiled.

    The dispatcher is built through ``make_token_dispatcher_config`` with the
    ``comm_backend`` our production config asks for, so the arm holds the
    class production holds -- ``AllToAllTokenDispatcher`` -- and pays its
    ``ep_mesh is None`` delegation branch. Building ``LocalTokenDispatcher``
    directly would measure a class no run of this model ever constructs.

    **The dispatch runs at build time and only the combine is timed.**
    ``dispatch`` is scenario 10's cut, and its metadata is combine's input:
    ``token_indices_experts_sorted_N`` and ``topk_scores_experts_sorted_N``
    are what ``_local_reorder`` produced, taken from the real call rather than
    reimplemented here.

    The scores are re-bound to a leaf that requires grad. End to end they come
    from the router and carry a gradient, and the multiply that consumes them
    is the whole reason this scenario publishes no cross-engine ratio -- so
    gating ``titan_scores_grad`` is what proves the multiply happened inside
    the timed region. The metadata object is rebuilt once, at build time, with
    that leaf in it: production constructs the metadata inside ``dispatch``,
    so constructing it inside the timed closure would charge combine for work
    combine does not do.

    Compiled, because that is what a titan module faces end to end. Every
    mcore arm runs eager, because megatron compiles no whole layer; the ratio
    between them is a comparison of two treatments, which is one more reason
    this scenario publishes none.
    """
    from dataclasses import replace

    from torchtitan.models.common.config_utils import make_token_dispatcher_config
    from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher

    batch, seq = workload.batch, workload.seq_len
    tokens = batch * seq
    dispatcher = make_token_dispatcher_config(
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        comm_backend=TITAN_COMM_BACKEND,
    ).build()
    if type(dispatcher) is not AllToAllTokenDispatcher:
        raise RuntimeError(
            f"{TITAN_ARM}: comm_backend {TITAN_COMM_BACKEND!r} built "
            f"{type(dispatcher).__qualname__}, not AllToAllTokenDispatcher; "
            "the arm would measure a class production does not build"
        )
    if dispatcher.ep_mesh is not None:
        raise RuntimeError(
            f"{TITAN_ARM}: the dispatcher holds an EP mesh, so combine would "
            "take its all-to-all branch. This harness is single-GPU and the "
            "arm measures the local combine"
        )

    x_TD = inputs.x_BLD.reshape(tokens, shape.dim)
    with torch.no_grad():
        _, _, metadata = dispatcher.dispatch(
            x_TD,
            inputs.topk_scores_TK,
            inputs.topk_expert_ids_TK,
            inputs.tokens_per_expert_E,
        )
    if not torch.equal(metadata.token_indices_experts_sorted_N, inputs.row_token_N):
        raise RuntimeError(
            f"{TITAN_ARM}: dispatch produced a token order the shared inputs "
            "do not describe, so the arm would combine rows of expert_out_ND "
            "that belong to other tokens"
        )
    if not torch.equal(
        metadata.topk_scores_experts_sorted_N, inputs.scores_sorted_N
    ):
        raise RuntimeError(
            f"{TITAN_ARM}: dispatch produced routing scores the fp64 reference "
            "does not hold, so titan_scores_grad would be gated against a "
            "truth built from other numbers"
        )
    scores_leaf = inputs.scores_sorted_N.clone().requires_grad_()
    metadata = replace(metadata, topk_scores_experts_sorted_N=scores_leaf)

    def combine(expert_out_ND: torch.Tensor) -> torch.Tensor:
        # models/common/moe.py:162-168 verbatim. The two padding arguments are
        # the values MoE.forward computes at sp_size 1 (``:427-440``);
        # LocalTokenDispatcher.combine deletes both, and passing them is what
        # keeps the call the one production makes.
        return dispatcher.combine(
            expert_out_ND,
            metadata,
            x_TD,
            num_local_tokens_after_padding=tokens,
            local_seq_len_after_padding=seq,
        )

    print(
        f"moe_combine/{TITAN_ARM}: "
        f"{type(dispatcher).__qualname__}.combine over {tokens} tokens x "
        f"{shape.top_k} routed rows (ep_mesh=None, so it delegates to "
        "LocalTokenDispatcher.combine: score then scatter_add)",
        flush=True,
    )
    return _combine_arm(
        name=TITAN_ARM,
        call=torch.compile(combine, fullgraph=True),
        inputs=inputs,
        grad_native=inputs.grad_out_TD,
        canonical=tuple(inputs.x_BLD.shape),
        extra_leaves=(scores_leaf,),
        extra_outputs=lambda: {"titan_scores_grad": scores_leaf.grad},
        output_prefix="titan",
    )


def _assert_mcore_dispatcher(
    dispatcher: Any, arm: str, expected_class: type, permute_fusion: bool
) -> None:
    """Refuse to time anything but the dispatcher this arm claims.

    Four checks, every one of which raises. Each names a way the arm could
    measure something other than megatron's combine while every correctness
    gate still passed, because the wrong operation here is usually still
    numerically right.

    1. The class is exactly the expected one. This is the whole of the
       ``dispatcher_alltoall`` delta: ``moe_token_dispatcher_type`` is read
       once, in ``MoELayer.__init__`` (``moe_layer.py:299-313``), and the two
       classes place the unpermute in different phases. ``type(...) is`` and
       not ``isinstance``, because a subclass is a different implementation.
    2. Both parallel sizes are 1. Everything this scenario says about
       ``token_combine`` being inert is a statement about that, and at any
       other size the cut is not the one the module documents.
    3. ``config.moe_permute_fusion`` is what the arm's profile declares, in
       both directions. This is the ``no_permute_fusion`` delta, delivered
       by the config field alone: ``unpermute`` reads it per call at
       ``token_dispatcher.py:348`` (and ``:893`` under alltoall), and no
       layer spec is involved. A delta that did not take would publish the
       base implementation under the variant's name, and a delta that took
       when it should not would do the reverse. The value is read from the
       BUILT config rather than from the profile: the profile is what was
       asked for, and the config is what was built.
    4. When the fusion is on, TE's ``fused_unpermute`` symbol really exists.
       ``TransformerConfig.__post_init__`` raises without it
       (``transformer_config.py:2677-2694``), so this is a second net -- but
       it is the net that checks the symbol this arm's timed region calls,
       rather than the five the config constructor checked.
    There is deliberately no fifth check for "the fusion is off, so
    ``fused_unpermute`` did not run". No object records which branch a past
    call took, so the only evidence available is check 3 restated, and a
    check that repeats another one is a count rather than a guard.
    """
    from megatron.core.transformer.moe.moe_utils import fused_unpermute

    if type(dispatcher) is not expected_class:
        raise RuntimeError(
            f"{arm}: MoELayer built {type(dispatcher).__qualname__}, not "
            f"{expected_class.__qualname__}. moe_token_dispatcher_type did "
            "not reach the layer, so this arm would publish the other "
            "dispatcher's local work under its own name"
        )
    if dispatcher.tp_size != 1 or dispatcher.ep_size != 1:
        raise RuntimeError(
            f"{arm}: the dispatcher reports tp_size {dispatcher.tp_size} and "
            f"ep_size {dispatcher.ep_size}; this scenario is single-rank and "
            "its account of token_combine holds only at 1 and 1"
        )
    built = dispatcher.config.moe_permute_fusion
    if built is not permute_fusion:
        raise RuntimeError(
            f"{arm}: the built config has moe_permute_fusion={built!r}, "
            f"expected {permute_fusion!r}; the profile delta did not reach "
            "the config, so this arm would measure the other arm's unpermute"
        )
    if permute_fusion and fused_unpermute is None:
        raise RuntimeError(
            f"{arm}: moe_permute_fusion is on but TransformerEngine exposes "
            "no fused_unpermute, so unpermute would raise inside the timed "
            "region rather than measure the fused kernel"
        )


def _assert_mcore_combine(
    combine: Callable[[torch.Tensor], torch.Tensor],
    arm: str,
    inputs: MoeCombineInputs,
    dispatcher: Any,
    hidden_shape: tuple[int, ...],
    tokens: int,
    dim: int,
) -> None:
    """Refuse to time a megatron combine that does not unpermute. It raises.

    **This is the guard the plan requires before any MoE scenario reports
    anything**, and the reason it raises rather than warns: an mcore side that
    is the identity at ``world_size=1`` reads as a spectacular win, not as a
    bug. It is also not recorded in ``BuiltArm.notes``, because notes reach no
    artifact and a fact no reader ever sees is not a guard.

    Three checks.

    1. ``token_combine`` returns its input unchanged. That is the inert
       collective, and stating it as a checked fact is what separates it from
       the inert CLASS the ``dispatcher_alltoall`` arm exists to disprove.
       Value equality rather than object identity, because the alltoall class
       routes through ``_AllToAll.apply`` and an autograd function is free to
       hand back a different Python object holding the same storage.
    2. The triple's output has the block's hidden shape, which holds ``T``
       rows where the input holds ``T * top_k``. An identity cannot pass this,
       whatever else it does.
    3. The output equals the canonical unpermute of the shared rows. This is
       what proves the arm restored the ORIGINAL token order rather than some
       order, and with it that the arm combined the same rows every other arm
       combines. A permutation that is wrong is wrong by about 100%, so the
       threshold is the gates' own ``2e-2`` and not a tuned number.
    """
    with torch.no_grad():
        preprocessed = dispatcher.combine_preprocess(inputs.expert_out_ND)
        combined = dispatcher.token_combine(preprocessed)
        if tuple(combined.shape) != tuple(preprocessed.shape) or not torch.equal(
            combined, preprocessed
        ):
            raise RuntimeError(
                f"{arm}: token_combine changed its input at tp_size 1 and "
                "ep_size 1, where the collective is documented inert. Either "
                "the process group is larger than this harness supports or "
                "the phase now does local work this scenario has not accounted "
                "for"
            )
        out = combine(inputs.expert_out_ND)

    if tuple(out.shape) != tuple(hidden_shape):
        raise RuntimeError(
            f"{arm}: the combine triple returned {tuple(out.shape)}, not the "
            f"block hidden shape {tuple(hidden_shape)}. The unpermute did not "
            "run, so this arm would time a pass-through and publish it as a "
            "win over titan"
        )
    error = _rel_l2(out.reshape(tokens, dim), _canonical_combine(inputs, dim, tokens))
    if not error <= COMBINE_GUARD_REL_L2:
        raise RuntimeError(
            f"{arm}: the combine triple sits {error:.3e} (rel_l2) from the "
            f"unpermute of the shared rows, above {COMBINE_GUARD_REL_L2:.0e}. "
            "The arm restored some order but not the original token order, so "
            "it is not combining the rows the other arms combine"
        )


def _build_mcore_arm(
    *,
    arm: str,
    profile: McoreProfile,
    expected_class_name: str,
    permute_fusion: bool,
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: MoeCombineInputs,
) -> BuiltArm:
    """One megatron arm: build the model, take the dispatcher, drop the model.

    The dispatcher is navigated off a real ``GPTModel`` rather than
    constructed here, which is the rule for every mcore arm in this package
    and earns its keep more here than anywhere: the dispatcher CLASS is the
    ``dispatcher_alltoall`` delta, and ``MoELayer.__init__`` is the only place
    ``moe_token_dispatcher_type`` is read. A hand construction would name the
    class this module already believes in and could not disagree with it.

    **The dispatch phases run here, once, and are never timed.** Every combine
    phase reads state only a dispatch writes -- ``reversed_local_input_
    permutation_mapping``, ``hidden_shape_before_permute``, ``local_map`` and
    ``hidden_shape`` for allgather, plus ``num_global_tokens_per_local_expert``
    and ``routing_map`` for alltoall. Running the real phases is what makes
    the timed closure the engine's own combine rather than a transcription of
    it, and it puts every device-to-host copy in this scenario outside the
    measured interval: the allgather ``.cpu()`` at ``token_dispatcher.py:317``
    and every ``_maybe_dtoh_and_synchronize`` call are dispatch-side.

    They run under ``no_grad`` so the timed backward walks the combine alone.

    **The whole model is released.** A dispatcher holds ``self.config``, a few
    index tensors and no parameters, so the arm keeps kilobytes of the
    gigabytes ``build_model`` allocated. Dropping it matters anyway:
    ``max_memory_allocated`` is a maximum over time of the bytes currently
    allocated, so a retained model would be charged to the arm's
    ``peak_memory_gib`` by the first allocation inside the timed call.

    Eager on purpose. Megatron compiles no whole layer, and none of the three
    combine phases carries ``@jit_fuser``: the single ``@jit_fuser`` in
    ``token_dispatcher.py`` sits at ``:1860``, on
    ``MoEFlexTokenDispatcher.dispatch_preprocess``, and that class asserts
    ``tp_size * ep_size > 1`` and is never built here. TE's
    ``fused_unpermute`` is a hand-written kernel, not a compile treatment.
    ``KernelArm.compiled`` records it.
    """
    initialize_megatron_single_rank()

    from megatron.core.transformer.moe.token_dispatcher import (
        MoEAllGatherTokenDispatcher,
        MoEAlltoAllTokenDispatcher,
    )

    from benchmarks.models.piper_qwen3.megatron_model import build_model

    classes = {
        "MoEAllGatherTokenDispatcher": MoEAllGatherTokenDispatcher,
        "MoEAlltoAllTokenDispatcher": MoEAlltoAllTokenDispatcher,
    }
    expected_class = classes[expected_class_name]

    batch, seq = workload.batch, workload.seq_len
    tokens = batch * seq
    # THD, the form our megatron driver runs: it packs each batch's rows into
    # one sequence, so every hidden state reaching a layer is (t, 1, h).
    # ``dispatch_preprocess`` records this as ``hidden_shape`` and
    # ``combine_postprocess`` restores it, so it is the shape the timed
    # triple returns.
    hidden_shape = (tokens, 1, shape.dim)

    model = build_model(seq_len=seq, shape=shape, profile=profile)
    dispatcher = model.decoder.layers[MCORE_LAYER].mlp.token_dispatcher
    _assert_mcore_dispatcher(dispatcher, arm, expected_class, permute_fusion)

    with torch.no_grad():
        hidden, probs = dispatcher.dispatch_preprocess(
            inputs.x_BLD.reshape(hidden_shape),
            inputs.routing_map_TE,
            inputs.probs_TE,
        )
        hidden, probs = dispatcher.token_dispatch(hidden, probs)
        permuted, tokens_per_expert, permuted_probs = dispatcher.dispatch_postprocess(
            hidden, probs
        )
    if tuple(permuted.shape) != tuple(inputs.expert_out_ND.shape):
        raise RuntimeError(
            f"{arm}: dispatch produced {tuple(permuted.shape)} routed rows and "
            f"the shared expert outputs hold {tuple(inputs.expert_out_ND.shape)}; "
            "the arm would combine a tensor of the wrong length"
        )

    print(
        f"moe_combine/{arm}: decoder.layers[{MCORE_LAYER}].mlp.token_dispatcher "
        f"is {type(dispatcher).__qualname__} "
        f"(profile {profile.name}, moe_permute_fusion={permute_fusion}, "
        f"tp_size={dispatcher.tp_size}, ep_size={dispatcher.ep_size}, "
        f"tokens_per_expert={tokens_per_expert.tolist()})",
        flush=True,
    )
    del model, hidden, probs, permuted, permuted_probs, tokens_per_expert
    gc.collect()
    torch.cuda.empty_cache()

    def combine(expert_out_ND: torch.Tensor) -> torch.Tensor:
        # The three public phase methods, in the order MoELayer calls them:
        # combine_preprocess ends routed_experts_compute (moe_layer.py:562),
        # token_combine is the whole of MoELayer.combine (:566-572), and
        # combine_postprocess opens MoELayer.postprocess (:583).
        hidden_states = dispatcher.combine_preprocess(expert_out_ND)
        hidden_states = dispatcher.token_combine(hidden_states)
        return dispatcher.combine_postprocess(hidden_states)

    _assert_mcore_combine(
        combine, arm, inputs, dispatcher, hidden_shape, tokens, shape.dim
    )
    return _combine_arm(
        name=arm,
        call=combine,
        inputs=inputs,
        grad_native=inputs.grad_out_TD.reshape(hidden_shape),
        canonical=tuple(inputs.x_BLD.shape),
        output_prefix="mcore",
    )


def build_moe_combine_mcore_base(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeCombineInputs
) -> BuiltArm:
    """Megatron's allgather dispatcher at the base profile: fused unpermute.

    ``combine_preprocess`` calls ``unpermute`` with
    ``fused=config.moe_permute_fusion``, which the base profile sets True, so
    the timed region runs TransformerEngine's ``fused_unpermute``.
    ``token_combine`` is the identity here and ``combine_postprocess`` is a
    view, so this arm is the closest the roster comes to timing the unpermute
    alone -- which is why it is the anchor both published rows compare against.
    """
    return _build_mcore_arm(
        arm=MCORE_BASE_ARM,
        profile=BASE,
        expected_class_name="MoEAllGatherTokenDispatcher",
        permute_fusion=True,
        shape=shape,
        workload=workload,
        inputs=inputs,
    )


def build_moe_combine_mcore_no_permute_fusion(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeCombineInputs
) -> BuiltArm:
    """The same dispatcher with ``moe_permute_fusion=False``: the torch path.

    One field, one read site inside this scenario. ``unpermute`` falls to
    ``moe_utils.py:601-621``: allocate a zeroed ``[T, dim]`` tensor and
    ``scatter_add_`` the routed rows into it with an index expanded to the
    hidden width, or ``index_add_`` when deterministic algorithms are enabled.
    ``probs`` is None on this path, so none of the prob-permuting branches
    above it runs.

    Eager, and the arm IS that: replacing a TE kernel with torch operations is
    the difference the published row measures.
    """
    return _build_mcore_arm(
        arm=MCORE_NO_PERMUTE_FUSION_ARM,
        profile=NO_PERMUTE_FUSION,
        expected_class_name="MoEAllGatherTokenDispatcher",
        permute_fusion=False,
        shape=shape,
        workload=workload,
        inputs=inputs,
    )


def build_moe_combine_mcore_dispatcher_alltoall(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeCombineInputs
) -> BuiltArm:
    """The alltoall dispatcher class, with the fusion left on.

    **This arm measures the dispatcher's local unpermute and sync strategy,
    not communication.** At ``world_size=1`` ``_AllToAll.forward`` returns its
    input (``tensor_parallel/mappings.py:434-437``), so ``token_combine``
    moves no bytes -- but it is still an ``autograd.Function.apply`` and the
    class still does different local work in the phases around it. It unsorts
    chunks in ``combine_preprocess`` (``sort_chunks_by_idxs``, a real row copy
    even though ``restore_output_by_local_experts`` is the identity order at
    one rank) and unpermutes in ``combine_postprocess``, where the allgather
    class unpermutes in ``combine_preprocess`` and does nothing else. So the
    row against ``mcore/base`` is "one extra chunk copy and one extra autograd
    node", and it is a real megatron choice rather than a harness artifact.

    ``moe_permute_fusion`` stays on, deliberately. The two deltas are declared
    as separate arms and combining them would confound the two rows -- and
    under alltoall the unfused ``sort_chunks_by_idxs`` also calls
    ``split_sizes.tolist()`` (``moe_utils.py:656``), which would put a
    device-to-host read inside the timed region and break this scenario's
    device-time declaration.
    """
    return _build_mcore_arm(
        arm=MCORE_ALLTOALL_ARM,
        profile=DISPATCHER_ALLTOALL,
        expected_class_name="MoEAlltoAllTokenDispatcher",
        permute_fusion=True,
        shape=shape,
        workload=workload,
        inputs=inputs,
    )
