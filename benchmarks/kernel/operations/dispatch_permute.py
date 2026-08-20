"""Arm builders for the ``dispatch_permute`` cross-engine kernel scenario.

The step that turns a routing decision into an expert-major token buffer, on
both engines. TorchTitan builds the routing map and the per-expert counts in
``MoE.forward``
(``third_party/torchtitan/torchtitan/models/common/moe.py:465-470``) and then
calls ``token_dispatcher.dispatch(...)``
(``models/common/moe.py:152-157``, inside ``RoutedExperts.forward``).
Megatron-core runs ``MoELayer.preprocess``
(``third_party/Megatron-LM/megatron/core/transformer/moe/moe_layer.py:447-487``)
and then **all three dispatch phases** -- ``dispatch_preprocess``,
``token_dispatch`` and ``dispatch_postprocess``.

**Why the cut names all three phases, and why cutting anywhere else would
have published a spectacular win.** ``MoEAllGatherTokenDispatcher.
token_dispatch`` is guarded by ``if self.tp_size > 1 or self.ep_size > 1``
(``token_dispatcher.py:282``). At ``world_size=1`` that guard is False and the
method **returns its two arguments unchanged**: it is the identity.
``MoELayer.dispatch`` (``moe_layer.py:489-498``) is a thin wrapper around
exactly that method, so a scenario cut there would have measured a zero on the
megatron side and would have read as a win rather than as a bug. The work
lives in the neighbouring phases -- ``permute(...)`` runs in
``dispatch_postprocess`` (``token_dispatcher.py:319-326``).

The cut names the whole triple rather than the phase that holds the permute
today, because the two dispatchers place it differently. The allgather
dispatcher permutes in ``dispatch_postprocess``; the alltoall dispatcher
permutes in ``dispatch_preprocess`` (``token_dispatcher.py:662-676``). Naming
the triple keeps the cut correct for either, and for a future dispatcher.

**The cut is on the dispatcher's public phase API, not on
``routed_experts_compute``.** ``MoELayer.routed_experts_compute``
(``moe_layer.py:528-564``) is the method that actually calls
``dispatch_postprocess`` in production, and it is decorated ``@internal_api``
(``:528``), which exempts it from megatron's backward-compatibility guarantee.
``dispatch_preprocess`` / ``token_dispatch`` / ``dispatch_postprocess`` are
abstract methods on ``MoETokenDispatcher`` (``token_dispatcher.py:96-211``) and
are public. ``MoELayer.preprocess`` and ``MoELayer.dispatch`` are public too,
so the first two phases are reached through the layer and the third through
the dispatcher.

**THIS SCENARIO'S NUMBER IS DEVICE TIME PLUS HOST SERIALIZATION, on every
megatron arm.** Plan rule 5 requires the declaration, and this is it.
``MoEAllGatherTokenDispatcher.dispatch_postprocess`` runs
``tokens_per_expert = self.local_map.sum(dim=0).long().cpu()``
(``token_dispatcher.py:317``) on **every** call. That ``.cpu()`` is a blocking
device-to-host copy, and a burst cannot pipeline through it, so per-call time
stops falling with ``--burst-k`` on the megatron side alone. The alltoall arm
pays a deferred form of the same cost: its copy is issued on a side stream at
``cuda_dtoh_point`` (``:918-951``) and waited for at ``cuda_sync_point``
(``:953-955``), which ``preprocess`` sets to ``"before_finish"`` at ``:602``.
TorchTitan's dispatcher has no counterpart -- it hands
``num_local_tokens_per_expert_E`` through as a device tensor. The cost is real
and megatron pays it on every layer of every step; it is not a harness
artifact. But a reader who takes this scenario's ratio for a kernel-speed
comparison has read it wrong.

**The peak-memory column is not like-for-like either, and this is the one
asymmetry no arm can be charged for.** Megatron's dispatchers keep a call's
intermediates on the instance: ``dispatch_postprocess`` assigns
``self.local_map`` and ``self.local_probs`` (``token_dispatcher.py:309-315``,
``:328-330``) and ``self.reversed_local_input_permutation_mapping``
(``:319``), and the alltoall dispatcher does the same in
``dispatch_preprocess``. TorchTitan's ``LocalTokenDispatcher.dispatch``
returns a frozen ``LocalDispatchMetadata`` and stores nothing
(``models/common/token_dispatcher.py:136-140``). In ``forward`` mode, where
nothing releases the graph, the megatron arms therefore hold state the titan
arm does not. Read ``peak_memory_gib`` as a property of each engine's
dispatcher design, never as a ranking.

**One boundary is asymmetric between the two engines, and it is charged to
TorchTitan.** Megatron's router returns ``(probs, routing_map)`` together
(``moe/router.py:835``), so the one-hot map is built inside scenario 9,
``moe_router``. TorchTitan's router returns ``(topk_scores, topk_expert_ids,
scores)`` and the map is built afterwards, in ``MoE.forward``
(``moe.py:465-470``) -- which the partition places here, in scenario 10. So the
titan arm pays a ``zeros_like`` plus a ``scatter_`` plus a ``sum`` that the
megatron arms do not, and the megatron arms paid the equivalent in scenario 9.
The tensors are small (``batch * seq_len * num_experts`` booleans, 16 KiB at
``normal``), but this scenario is dispatch-bound, and the charge is **two to
three extra kernel launches** on one side. Read the cross-engine row with that
in mind, and read scenario 9's row beside it. Moving the map construction out
of this scenario is not the fix: it would leave ``moe.py:465-470`` in no
scenario at all, and the partition's rule is that every call belongs to exactly
one cut.

**Glue this module reimplements rather than calls, and why that is free.**
``TransformerLayer._maybe_unflatten_for_moe`` and ``_maybe_reflatten_from_moe``
(``transformer_layer.py:763-800``) bracket the MoE call in production. Both are
**inert here**, and that is verified rather than assumed: the first returns
``(hidden_states, padding_mask, None)`` at ``:775-780`` unless
``packed_seq_params.tokens_per_sample`` is set, the second returns its input
when ``mbs is None`` (``:798-799``), and ``PackedSeqParams.tokens_per_sample``
defaults to ``None`` (``megatron/core/packed_seq_params.py:27``) and is set
**nowhere** in this repository. TorchTitan's own glue is inert for the mirror
reason: ``MoE.forward``'s sequence-dim padding block (``moe.py:426-450``)
reads ``sp_size = 1`` at ``world_size=1``, so ``seq_pad`` and
``seq_dim_pad_tokens`` are both 0 and the block is host arithmetic that
launches no kernel.

**A permute is a gather, so the permutation gate is bitwise and the
gradient gate is not.** A norm-based metric cannot see a permutation that
moves the right values to the wrong places. With ``N`` output rows, swapping
one pair of them gives ``||a - b|| / ||b|| ~ sqrt(2 / N)``, which is 1.6e-2 at
the default workload -- **below** the 2e-2 gate every neighbouring scenario
uses. A single misplaced row would therefore pass a tolerance gate. Both
engines produce the permuted tokens by a pure copy (``index_select`` on the
torch path, a TE kernel on the fused one), so bitwise equality is both
achievable and the only metric that sees the failure. The gradients are
accumulations of ``top_k`` bf16 values and round, so they are gated with
``max_rel_l2`` -- and that gate is **not** the permutation gate.

**The scenario itself is not in the registry yet.** This module ships ahead
of its declaration: the ``KernelScenario``, the ``KernelArm`` roster and the
``CorrectnessCheck`` literals described above live in
``reports/20260819-partc/decl/dispatch_permute.decl.py`` until a merge agent
pastes them into ``benchmarks/kernel/registry.py``. So every statement here
about what "the scenario declares" describes that fragment, not a name any
import can resolve today, and none of the engine's declaration gates --
the mode check at ``engine/run.py``, the ``eager_reason`` rule in
``schema.py`` -- runs against this module until the paste lands.

**Both engines permute into the same order, and that is what makes the
cross-engine bitwise gate legitimate.** TorchTitan sorts the flattened
``[T, K]`` expert-id tensor with a **stable ascending** argsort
(``token_dispatcher.py:93-95``), so within one expert the tokens come out in
ascending flat-slot order, which is ascending token order because a token
reaches a given expert at most once. Megatron transposes the ``[T, E]`` routing
map, flattens it and sorts with a **stable descending** argsort
(``moe_utils.py:468-475``), so the True entries come out expert-major and, within
one expert, in ascending token order. The two constructions look nothing alike
and produce the same permutation. ``tests/test_kernel_dispatch_permute.py``
transcribes both from the pinned sources and checks the agreement on CPU.

**No layout conversion sits inside a timed closure, on either engine.** The
megatron arms consume a ``[T, 1, D]`` THD view of the same storage -- the form
our driver produces (``benchmarks/e2e/megatron/data.py``), and the form
``attn_residual`` and ``moe_residual`` already use -- and
``dispatch_preprocess`` immediately does ``hidden_states.view(-1, D)``
(``token_dispatcher.py:274``). The titan arm consumes ``[B, L, D]`` and
``RoutedExperts.forward`` does ``x_BLD.view(T, D)`` (``moe.py:144``). Both
views are free, both are of one contiguous buffer, and both are taken from the
canonical tensor the inputs builder draws once.

Every torchtitan, megatron and TransformerEngine import is deferred into the
builder that needs it, which is the rule across ``operations/``.
``benchmarks.models.piper_qwen3.mcore_profiles`` is the one exception at module
scope: it is torch-free parent-side data, and this module derives two profiles
from it.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Callable

import torch

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    _randn,
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
# which engine an arm is and which profile of that engine; it reaches a
# filename through ``benchmarks.kernel.schema.fragment_stem``, which replaces
# it, so a timing worker does not try to write into a directory nobody made.
COPY_FLOOR_ARM = "copy_floor"
MCORE_BASE_ARM = "mcore/base"
MCORE_NO_PERMUTE_FUSION_ARM = "mcore/no_permute_fusion"
MCORE_DISPATCHER_ALLTOALL_ARM = "mcore/dispatcher_alltoall"
TITAN_ARM = "titan"

# The layer each engine reads its call site from. Every layer holds the same
# classes at ``moe_layer_freq=1`` and at torchtitan's every-layer MoE, so the
# index is arbitrary and is fixed here so the provenance line names one layer.
MCORE_LAYER = 0
TITAN_LAYER = 0

# The two dispatcher classes this scenario builds, named as strings so the
# guard can report what it found without importing megatron to describe a
# failure.
ALLGATHER_DISPATCHER = "MoEAllGatherTokenDispatcher"
ALLTOALL_DISPATCHER = "MoEAlltoAllTokenDispatcher"




@dataclass
class DispatchPermuteInputs:
    """One hidden-state batch, one routing decision, and both engines' views.

    The routing decision is drawn **once** and delivered to both engines in the
    form each one consumes, which is what plan section B.4 requires: the one
    place routing values reach timing is here, so both engines must receive the
    same ``tokens_per_expert`` and the same routing map, exactly as
    ``swiglu_inputs`` does today.

    * TorchTitan's ``MoE.forward`` consumes ``topk_expert_ids_BLK`` and
      ``topk_scores_BLK`` from its router and builds the one-hot map itself.
    * Megatron's ``MoELayer.preprocess`` consumes a dense ``[T, E]`` ``probs``
      and a dense ``[T, E]`` bool ``routing_map``, which is what its router
      returns (``moe/router.py:835``).

    ``probs_TE`` and ``topk_scores_TK`` hold the **same values**:
    ``probs_TE[t, topk_expert_ids_TK[t, k]] == topk_scores_TK[t, k]`` exactly,
    and ``probs_TE`` is zero at every unrouted entry, which is the form
    megatron's ``topk_routing_with_score_function`` produces. Both are fp32 on
    both engines: megatron's base profile sets ``moe_router_dtype="fp32"``
    (``router.py:107-111``) and torchtitan computes its gate under an fp32
    autocast and softmaxes in fp32 (``moe.py:292-300``), so this scenario
    carries no precision difference. Scenario 9's cross-engine row does; this
    one does not.

    ``expected_slot_token_N`` and ``expected_slot_expert_N`` are the
    permutation this scenario claims both engines produce, derived here by a
    third construction that is neither engine's -- a per-expert ``nonzero``.
    They are what ``_require_a_real_permutation`` and
    ``dispatch_permute_reference`` both read, so the claim is checked against
    an independent derivation rather than against one engine's answer.
    """

    x_BLD: torch.Tensor  # (B, L, D) bf16, contiguous
    topk_expert_ids_TK: torch.Tensor  # (T, K) int64
    topk_scores_TK: torch.Tensor  # (T, K) fp32
    probs_TE: torch.Tensor  # (T, E) fp32, zero off-route
    routing_map_TE: torch.Tensor  # (T, E) bool
    tokens_per_expert_E: torch.Tensor  # (E,) int64
    expected_slot_token_N: torch.Tensor  # (N,) int64
    expected_slot_expert_N: torch.Tensor  # (N,) int64
    grad_permuted_ND: torch.Tensor  # (N, D) bf16
    grad_probs_N: torch.Tensor  # (N,) fp32
    # The floor's own traffic: one read and one write of the permuted buffer.
    # It is the floor's number and no other arm's -- see
    # ``build_dispatch_permute_copy_floor``.
    copy_bytes: int


def dispatch_permute_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> DispatchPermuteInputs:
    """Draw the hidden states and one exactly-balanced routing decision.

    **The split is exact, and an unbalanced one fails loudly with named
    numbers rather than being capped or rounded.** Two conditions have to
    hold. ``batch * seq_len * top_k`` must divide by ``num_experts``, which is
    the invariant ``KernelScenario.requires_balanced_routing`` expresses and
    ``benchmarks/kernel/runner.py:552`` checks -- **once the declaration
    lands**; until then the raises below are the only thing enforcing it,
    which is why they carry named numbers. And
    ``batch * seq_len`` must divide by ``num_experts`` too, which is stronger
    and belongs to this construction rather than to the scenario: the token
    offsets come from a random permutation reduced modulo ``num_experts``, and
    each residue appears equally often only when the tokens divide.

    The construction: permute the tokens randomly, reduce the permutation
    modulo ``num_experts`` to get one offset per token, and give token ``t``
    the ``top_k`` consecutive experts starting at its offset. The offsets are
    exactly balanced, so every expert receives exactly
    ``batch * seq_len * top_k / num_experts`` slots; the ``top_k`` experts of
    one token are distinct because they are consecutive residues and
    ``top_k <= num_experts``; and which token draws which offset is random.

    **The structure that survives, stated rather than hidden:** the roster of
    per-token expert windows is the ``num_experts`` cyclic windows of width
    ``top_k``, so two experts inside one window read the same token subset and
    the second one's gather may find the first one's rows in L2. Both engines
    receive the identical map, so the effect is shared and cancels in every
    ratio this scenario publishes. It would not cancel against a number
    measured on real router output, which no arm here is.
    """
    batch, seq = workload.batch, workload.seq_len
    tokens = batch * seq
    experts, top_k = shape.num_experts, shape.top_k
    slots = tokens * top_k
    if top_k > experts:
        raise ValueError(
            f"dispatch_permute: top_k {top_k} exceeds num_experts {experts}, "
            "so no token can be routed to top_k distinct experts"
        )
    if slots % experts:
        raise ValueError(
            f"dispatch_permute: {slots} routed slots (batch {batch} x seq "
            f"{seq} x top_k {top_k}) do not divide evenly among {experts} "
            f"experts; the remainder is {slots % experts}"
        )
    if tokens % experts:
        raise ValueError(
            f"dispatch_permute: {tokens} tokens (batch {batch} x seq {seq}) "
            f"do not divide evenly among {experts} experts; the remainder is "
            f"{tokens % experts}. This scenario draws one expert offset per "
            "token from a random permutation reduced modulo num_experts, and "
            "that is exactly balanced only when the tokens divide."
        )

    x_BLD = _randn((batch, seq, shape.dim), device, generator)

    # A random permutation of the tokens, built from torch.rand rather than
    # torch.randperm: rand takes the run's generator on every device and every
    # torch build, and a stable argsort of it is a permutation. The stability
    # matters only if two draws tie, which fp32 makes vanishingly rare and
    # which would otherwise make the inputs depend on the sort implementation.
    token_order = torch.rand(
        (tokens,), device=device, generator=generator, dtype=torch.float32
    ).argsort(stable=True)
    offsets_T = token_order % experts
    topk_expert_ids_TK = (
        offsets_T.unsqueeze(1) + torch.arange(top_k, device=device)
    ) % experts
    topk_scores_TK = torch.rand(
        (tokens, top_k), device=device, generator=generator, dtype=torch.float32
    )
    probs_TE = torch.zeros(
        (tokens, experts), device=device, dtype=torch.float32
    ).scatter_(1, topk_expert_ids_TK, topk_scores_TK)
    routing_map_TE = torch.zeros(
        (tokens, experts), device=device, dtype=torch.bool
    ).scatter_(1, topk_expert_ids_TK, True)
    tokens_per_expert_E = routing_map_TE.sum(dim=0).long()

    per_expert = slots // experts
    if not bool((tokens_per_expert_E == per_expert).all()):
        raise ValueError(
            f"dispatch_permute: the drawn routing gives per-expert counts "
            f"{tokens_per_expert_E.tolist()}, and an exact split of {slots} "
            f"slots among {experts} experts is {per_expert} each. Both engines "
            "must receive one balanced tokens_per_expert, so this is a failure "
            "rather than something to round."
        )

    # The expected permutation, derived a third way: for each expert in
    # ascending order, the tokens routed to it in ascending order. Neither
    # engine's algorithm, and the reference both the gate and the build-time
    # guard read.
    slot_tokens: list[torch.Tensor] = []
    slot_experts: list[torch.Tensor] = []
    for expert in range(experts):
        rows = torch.nonzero(routing_map_TE[:, expert], as_tuple=False).flatten()
        slot_tokens.append(rows)
        slot_experts.append(torch.full_like(rows, expert))
    expected_slot_token_N = torch.cat(slot_tokens)
    expected_slot_expert_N = torch.cat(slot_experts)

    # The property that makes an identity arm detectable at all. If the
    # expected order happened to be the identity on its first ``tokens``
    # entries, an arm that returned its own input would satisfy the bitwise
    # gate on those rows, and the scenario would have no instrument for the
    # failure it exists to catch.
    identity = torch.arange(tokens, device=device)
    if bool(torch.equal(expected_slot_token_N[:tokens], identity)):
        raise ValueError(
            "dispatch_permute: the drawn routing permutes the first "
            f"{tokens} slots into token order, so a dispatcher that returned "
            "its own input would satisfy the permutation gate on them. "
            "Change --seed."
        )

    return DispatchPermuteInputs(
        x_BLD=x_BLD,
        topk_expert_ids_TK=topk_expert_ids_TK,
        topk_scores_TK=topk_scores_TK,
        probs_TE=probs_TE,
        routing_map_TE=routing_map_TE,
        tokens_per_expert_E=tokens_per_expert_E,
        expected_slot_token_N=expected_slot_token_N,
        expected_slot_expert_N=expected_slot_expert_N,
        grad_permuted_ND=_randn((slots, shape.dim), device, generator),
        grad_probs_N=torch.randn(
            (slots,), device=device, generator=generator, dtype=torch.float32
        ),
        copy_bytes=2 * slots * shape.dim * x_BLD.element_size(),
    )


def dispatch_permute_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: DispatchPermuteInputs
) -> dict[str, torch.Tensor]:
    """The exact permutation, and the fp64 truth for both gradients.

    **The forward half is exact, not approximate.** A permutation moves bits
    and computes nothing, so the reference gathers the same bf16 and fp32
    values the arms gather and the gate that reads it is ``bitwise``. The
    scenario's fragment names this builder as its ``reference_builder``, and
    the
    engine spells that reference ``"fp64"`` whatever it computes
    (``benchmarks/kernel/engine/correctness.py``); the name is the engine's,
    the arithmetic is this function's.

    **The gradient half is genuinely fp64.** The gradient of a gather is a
    scatter-add, so ``x_grad`` accumulates ``top_k`` output-gradient rows per
    token and rounds; the arms accumulate in bf16 and land about one bf16 ULP
    away. ``probs_grad`` is a pure scatter of distinct values and rounds
    nothing, but it is reported in the same dense ``[T, E]`` canonical form and
    gated with the same tolerance, because a gate that enforced bitwise
    equality on one gradient and not the other would say something about the
    implementation that this scenario has not checked.

    ``probs_grad`` is dense ``[T, E]`` on both sides. Megatron's leaf is
    already that shape; the titan arm scatters its ``[T, K]`` gradient into it
    at the routed positions, which is the same canonicalization the inputs
    builder used to produce ``probs_TE`` from ``topk_scores_TK``.
    """
    tokens = workload.batch * workload.seq_len
    order = inputs.expected_slot_token_N
    expert = inputs.expected_slot_expert_N
    x_TD = inputs.x_BLD.reshape(tokens, shape.dim)

    x_grad = torch.zeros(
        (tokens, shape.dim), device=x_TD.device, dtype=torch.float64
    ).index_add_(0, order, inputs.grad_permuted_ND.double())
    probs_grad = torch.zeros(
        (tokens, shape.num_experts), device=x_TD.device, dtype=torch.float64
    )
    probs_grad[order, expert] = inputs.grad_probs_N.double()

    return {
        "permuted_tokens": x_TD[order],
        "permuted_probs": inputs.probs_TE[order, expert],
        "tokens_per_expert": inputs.tokens_per_expert_E,
        "x_grad": x_grad.reshape(inputs.x_BLD.shape),
        "probs_grad": probs_grad,
    }


def _require_a_real_permutation(
    call: Callable[[], tuple[Any, Any, Any]],
    inputs: DispatchPermuteInputs,
    shape: PiperShape,
    workload: KernelWorkload,
    arm: str,
) -> None:
    """Refuse to time an arm that did not actually permute.

    **This is the guard plan section "Guards that must exist before the MoE
    scenarios report anything" requires, and it raises rather than warns.**
    ``BuiltArm.notes`` reaches no artifact, so a fact recorded there is a fact
    no reader ever sees. It runs in every worker that builds this arm --
    correctness and timing alike -- because a timing worker is exactly where a
    silent identity would turn into a published number.

    The failure it exists for: a scenario whose megatron side is the identity
    at ``world_size=1`` reads as a spectacular win, not as a bug.
    ``MoEAllGatherTokenDispatcher.token_dispatch`` really is the identity here
    (``token_dispatcher.py:282``), and so is ``_AllToAll.forward``
    (``tensor_parallel/mappings.py:433-436``), so an arm cut one phase away
    from the permute would run, would gate clean against nothing, and would
    report a number an order of magnitude too small.

    Four checks, in increasing strength.

    The output must have ``batch * seq_len * top_k`` rows, not
    ``batch * seq_len``. An identity arm returns the input, whose row count is
    the second, so this alone catches the failure at every ``top_k > 1``.

    The output must not be the input tensor. That covers ``top_k == 1``, which
    no registered shape uses today but which the check must not depend on.

    The permuted tokens and the permuted probabilities must equal the expected
    permutation **bitwise**. This is the decisive check: it sees a permutation
    that moved the right values to the wrong places, which no norm-based
    metric at this row count can. Both engines produce these buffers by a pure
    copy, so bitwise equality is the right relation and not a strict one.

    The per-expert counts must equal the ones the inputs builder drew. They are
    what the expert GEMM in scenario 11 would slice by, and megatron returns
    them on the host while titan returns them on the device, so they are
    compared after a canonicalization rather than in the arms' native forms.
    """
    tokens = workload.batch * workload.seq_len
    slots = tokens * shape.top_k
    x_TD = inputs.x_BLD.reshape(tokens, shape.dim)
    with torch.no_grad():
        permuted, tokens_per_expert, permuted_probs = call()
    if tuple(permuted.shape) != (slots, shape.dim):
        raise RuntimeError(
            f"{arm}: the dispatch returned a {tuple(permuted.shape)} token "
            f"buffer and this cut produces ({slots}, {shape.dim}). A "
            f"({tokens}, {shape.dim}) buffer is the input handed back: at "
            "world_size=1 token_dispatch is the identity on both megatron "
            "dispatchers, so a cut one phase away from permute() measures "
            "nothing and reads as a win."
        )
    if permuted is x_TD or permuted.data_ptr() == x_TD.data_ptr():
        raise RuntimeError(
            f"{arm}: the dispatch returned its own input buffer, so it "
            "permuted nothing."
        )
    expected_tokens = x_TD[inputs.expected_slot_token_N]
    if not torch.equal(permuted, expected_tokens):
        raise RuntimeError(
            f"{arm}: the permuted tokens are not the expert-major permutation "
            "of the shared routing map. The arm moved the right values to the "
            "wrong places, which a rel_l2 gate at this row count cannot see."
        )
    if permuted_probs.dim() != 1:
        raise RuntimeError(
            f"{arm}: the dispatch returned a {tuple(permuted_probs.shape)} "
            "probability tensor and every dispatcher in this cut returns a "
            "flat one slot per row. The timed closures hand it to "
            "torch.autograd.backward without reshaping it, so a second "
            "dimension here would put a view op inside one engine's timed "
            "region and not the other's."
        )
    expected_probs = inputs.probs_TE[
        inputs.expected_slot_token_N, inputs.expected_slot_expert_N
    ]
    if not torch.equal(permuted_probs, expected_probs):
        raise RuntimeError(
            f"{arm}: the permuted probabilities do not follow the same "
            "permutation as the tokens; the expert GEMM downstream would "
            "weight each row with another row's probability."
        )
    counts = _canonical_counts(tokens_per_expert, inputs.tokens_per_expert_E)
    if not torch.equal(counts, inputs.tokens_per_expert_E):
        raise RuntimeError(
            f"{arm}: the dispatch reports per-expert counts "
            f"{counts.tolist()} and the shared routing map gives "
            f"{inputs.tokens_per_expert_E.tolist()}."
        )


def _canonical_counts(
    counts: torch.Tensor, reference: torch.Tensor
) -> torch.Tensor:
    """One per-expert count vector, on the reference's device and dtype.

    Megatron's allgather dispatcher returns ``tokens_per_expert`` on the
    **host** -- ``self.local_map.sum(dim=0).long().cpu()``
    (``token_dispatcher.py:317``) -- and torchtitan returns it on the device.
    ``torch.equal`` and the correctness engine's subtraction both need one
    device, and the move happens here, outside every timed closure, so no
    published sample pays for it. That the megatron arm pays a blocking
    device-to-host copy *inside* its timed closure is the scenario's declared
    property, not something this function creates or hides.
    """
    return counts.reshape(-1).to(device=reference.device, dtype=reference.dtype)


def _require_gradients(arm: str, leaves: dict[str, torch.Tensor]) -> None:
    """Turn a missing gradient into a named failure, not an AttributeError."""
    missing = sorted(name for name, leaf in leaves.items() if leaf.grad is None)
    if missing:
        raise RuntimeError(
            f"{arm}: backward produced no gradient for {', '.join(missing)}; "
            "the dispatch detached an operand, so it does not carry the "
            "gradient path this scenario measures"
        )


def _dispatch_permute_arm(
    *,
    name: str,
    bind: Callable[[torch.Tensor, torch.Tensor], Callable[[], tuple]],
    x_native: torch.Tensor,
    probs_native: torch.Tensor,
    inputs: DispatchPermuteInputs,
    dense_probs_grad: Callable[[torch.Tensor], torch.Tensor],
    notes: dict[str, Any] | None = None,
) -> BuiltArm:
    """Forward and forward+backward over one engine's native tensor shapes.

    ``bind`` receives the two differentiable leaves of one closure and returns
    the timed call over them. It is a factory rather than a two-argument
    function so each engine can hoist whatever is genuinely build-time work,
    which is the shape ``attn_residual`` established.

    Three independent leaf pairs, matching every other module scenario here:
    the ``forward`` closure never runs a backward, so a shared leaf would let
    one mode's gradient state reach another mode's timing.

    **There is no isolated ``backward`` mode, on any arm.** The retained-graph
    trick ``rope`` and ``qkv`` use re-runs one backward graph several times,
    and this scenario cannot: the megatron arms reach TransformerEngine's own
    ``permute`` autograd functions, whose saved-tensor lifetime under a
    repeated backward is unverified here and unverifiable without a GPU.
    ``ffn_norm`` and ``attention`` drop the mode for the same class of reason.
    Dropping it from **every** arm keeps them comparable, and the backward cost
    stays recoverable as ``forward_backward`` minus ``forward``.

    ``forward`` runs over leaves that require a gradient, matching ``qkv``,
    ``ffn_norm`` and ``attention``: that is the condition production runs in,
    and an inference-mode call would let either engine skip the bookkeeping a
    training step pays for.
    """
    forward_x = x_native.clone().requires_grad_()
    forward_probs = probs_native.clone().requires_grad_()
    round_trip_x = x_native.clone().requires_grad_()
    round_trip_probs = probs_native.clone().requires_grad_()
    check_x = x_native.clone().requires_grad_()
    check_probs = probs_native.clone().requires_grad_()

    forward_call = bind(forward_x, forward_probs)
    round_trip_call = bind(round_trip_x, round_trip_probs)
    check_call = bind(check_x, check_probs)
    canonical = tuple(inputs.x_BLD.shape)

    def forward() -> tuple:
        return forward_call()

    def forward_backward() -> None:
        _reset_grads(round_trip_x, round_trip_probs)
        permuted, _, permuted_probs = round_trip_call()
        # No reshape on either operand. Both engines return a flat probability
        # vector -- ``_require_a_real_permutation`` refuses an arm that does
        # not -- so a defensive reshape here would add a view op to one timed
        # region and could add it to only one engine's.
        torch.autograd.backward(
            (permuted, permuted_probs),
            (inputs.grad_permuted_ND, inputs.grad_probs_N),
        )

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(check_x, check_probs)
        permuted, tokens_per_expert, permuted_probs = check_call()
        torch.autograd.backward(
            (permuted, permuted_probs),
            (inputs.grad_permuted_ND, inputs.grad_probs_N),
        )
        _require_gradients(name, {"x_grad": check_x, "probs_grad": check_probs})
        return {
            "permuted_tokens": permuted.detach(),
            "permuted_probs": permuted_probs.detach(),
            "tokens_per_expert": _canonical_counts(
                tokens_per_expert, inputs.tokens_per_expert_E
            ),
            "x_grad": check_x.grad.reshape(canonical),
            "probs_grad": dense_probs_grad(check_probs.grad),
        }

    return BuiltArm(
        name=name,
        calls={"forward": forward, "forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
        notes=dict(notes or {}),
    )


def build_dispatch_permute_copy_floor(
    shape: PiperShape, workload: KernelWorkload, inputs: DispatchPermuteInputs
) -> BuiltArm:
    """The bandwidth floor: one read and one write of the permuted buffer.

    **A gather is bandwidth-bound, which is why the plan names this scenario
    the next best floor candidate after the norms.** The whole forward is a
    copy of ``batch * seq_len * top_k`` rows into a new buffer, plus an argsort
    and some index arithmetic over one-row-per-slot integers. If the arms sit
    near this floor, the published ratios compare how close four
    implementations get to the memory bus, and no ratio here is a kernel-
    quality claim. If they sit far above it -- which is what CLAUDE.md's
    measured rope numbers would predict for a scenario with this many small
    launches -- the ratios are a comparison of host dispatch. The floor is the
    only instrument in the scenario that tells those two readings apart, and
    CLAUDE.md records that the ``--burst`` residual cannot: a flat ladder means
    ``k`` stopped buying amortization, not that the number became device time.

    **It is a lower bound, and the direction of the bias is stated.** The floor
    reads a contiguous ``[N, D]`` buffer; the gather reads ``N`` rows scattered
    through a ``[T, D]`` one. At the normal shape a row is 2 KiB, far above any
    sector granularity, so the scatter costs little -- but it costs something,
    and the floor therefore **understates** the device work the arms do. The
    ``x_floor`` column consequently **overstates** the distance between an arm
    and the device.

    **This arm alone carries ``bytes_moved``.** The merge divides one byte
    count by every mode's median (``results/merge.py:417-427``), so a count on
    a two-mode arm would attach the forward's GB/s figure to the
    ``forward_backward`` row as well. ``attn_residual`` and ``embedding_stage``
    publish theirs on the floor for the same reason. The floor has one mode, so
    its GB/s figure is a bandwidth statement and nothing else.

    Eager on purpose, and forward only: a floor measures the device rather than
    an implementation, and a floor for the ``forward_backward`` traffic would be
    an invention -- the backward of a gather is a scatter-add whose traffic
    depends on how the accumulation is ordered.
    """
    tokens = workload.batch * workload.seq_len
    slots = tokens * shape.top_k
    source = inputs.x_BLD.reshape(tokens, shape.dim)[
        inputs.expected_slot_token_N
    ].contiguous()
    if tuple(source.shape) != (slots, shape.dim):
        raise RuntimeError(
            f"{COPY_FLOOR_ARM}: the floor buffer is {tuple(source.shape)} and "
            f"the permuted buffer is ({slots}, {shape.dim}); the floor would "
            "not bound the traffic it claims to bound"
        )
    out = torch.empty_like(source)

    def forward() -> None:
        out.copy_(source)

    return BuiltArm(
        name=COPY_FLOOR_ARM,
        calls={"forward": forward},
        correctness_outputs=dict,
        bytes_moved=inputs.copy_bytes,
    )


def _titan_dispatch(
    dispatcher: Any,
    x_BLD: torch.Tensor,
    topk_scores_BLK: torch.Tensor,
    topk_expert_ids_BLK: torch.Tensor,
    scores_BLE: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """TorchTitan's routing-map construction and its dispatch, in one call.

    Transcribed from two adjacent call sites, both inside the scenario's cut.
    ``MoE.forward`` builds the one-hot map and the counts
    (``moe.py:465-470``); ``RoutedExperts.forward`` flattens to ``[T, .]`` and
    calls the dispatcher (``moe.py:144-157``).

    Three pieces of the production path are deliberately absent, and each is a
    no-op at ``world_size=1`` rather than an omission this scenario benefits
    from. ``MoE.forward``'s sequence-dim padding block (``moe.py:426-450``)
    reads ``sp_size = 1``, so both pad counts are zero and it launches no
    kernel. The ``tokens_per_expert_E.add_`` accumulation (``:481-483``) is
    behind ``if self.load_balance_coeff is not None``, and our config sets that
    to ``None`` (``benchmarks/models/piper_qwen3/config_registry.py:111``); the
    fork carries that gate specifically so the in-place input mutation does not
    block cudagraph capture. And the shared-expert branch (``:496-498``) is
    ``None`` because the Piper config declares no shared expert.

    Returns the three tensors the megatron triple returns, in the same order,
    so both engines' timed closures have one shape and the gate compares like
    with like. The ``LocalDispatchMetadata`` dataclass is still constructed
    inside ``dispatch`` -- this function unpacks it rather than avoiding it --
    because returning a frozen dataclass from a ``fullgraph=True`` region is a
    property of the harness, not of the engine.
    """
    batch, seq, dim = x_BLD.shape
    top_k = topk_scores_BLK.shape[-1]
    tokens = batch * seq
    routing_map_BLE = torch.zeros_like(scores_BLE, dtype=torch.bool).scatter_(
        -1, topk_expert_ids_BLK, True
    )
    num_local_tokens_per_expert_E = routing_map_BLE.sum(dim=(0, 1))
    routed_input_RD, tokens_per_expert_E, metadata = dispatcher.dispatch(
        x_BLD.view(tokens, dim),
        topk_scores_BLK.view(tokens, top_k),
        topk_expert_ids_BLK.view(tokens, top_k),
        num_local_tokens_per_expert_E,
    )
    return (
        routed_input_RD,
        tokens_per_expert_E,
        metadata.topk_scores_experts_sorted_N,
    )


def titan_dispatcher(shape: PiperShape):
    """The production token dispatcher, built from the production config node.

    ``_piper_1b_model`` sets ``moe_comm_backend="standard"``
    (``benchmarks/models/piper_qwen3/config_registry.py:101``), which
    ``make_token_dispatcher_config`` turns into an
    ``AllToAllTokenDispatcher.Config``
    (``third_party/torchtitan/torchtitan/models/common/config_utils.py:
    364-368``). So the production object at every world size is
    ``AllToAllTokenDispatcher``, and constructing ``LocalTokenDispatcher``
    directly here would build a class production never builds.

    **At ``world_size=1`` that object dispatches through the local path, and
    it does so by an explicit branch rather than by accident.**
    ``AllToAllTokenDispatcher.dispatch`` reads ``if self.ep_mesh is None:
    return LocalTokenDispatcher.dispatch(self, ...)``
    (``models/common/token_dispatcher.py:416-423``). ``ep_mesh`` is set only by
    ``wire_meshes``, which ``RoutedExperts.parallelize`` calls with
    ``parallel_dims.get_optional_mesh("ep")`` -- ``None`` at one rank -- and
    which this build never calls at all. ``_require_titan_dispatch_contract``
    asserts the state that branch reads.
    """
    from benchmarks.models.piper_qwen3.config_registry import _piper_1b_model

    model_config = _piper_1b_model(fuse_qkv=True, shape=shape)
    block = model_config.layers[TITAN_LAYER]
    return block.moe.routed_experts.token_dispatcher.build()


def _require_titan_dispatch_contract(dispatcher: Any, shape: PiperShape) -> None:
    """Refuse a titan dispatcher whose state changes which branch runs.

    ``ep_mesh`` decides between the local reorder and an all-to-all
    (``token_dispatcher.py:416-423``); a wired mesh at one rank would assert
    inside the collective rather than return a wrong number, but the arm would
    no longer be the cut this scenario declares. ``sp_size`` above 1 turns on
    the sequence-parallel index remap (``:216-230``) and changes what
    ``combine`` indexes, which is scenario 12's cut and not this one's. The
    two geometry fields are read straight off the shape both engines build
    from, so a dispatcher configured for another expert count could not reach a
    table under this shape's label.
    """
    problems: list[str] = []
    if dispatcher.ep_mesh is not None:
        problems.append(
            f"ep_mesh is {dispatcher.ep_mesh!r}, expected None; a wired expert "
            "mesh takes dispatch away from the local reorder this scenario cuts"
        )
    if dispatcher.sp_size != 1:
        problems.append(
            f"sp_size is {dispatcher.sp_size!r}, expected 1; sequence-parallel "
            "coordinates change the token indices combine reads"
        )
    if dispatcher.num_experts != shape.num_experts:
        problems.append(
            f"num_experts is {dispatcher.num_experts!r} and the shape declares "
            f"{shape.num_experts}"
        )
    if dispatcher.top_k != shape.top_k:
        problems.append(
            f"top_k is {dispatcher.top_k!r} and the shape declares "
            f"{shape.top_k}"
        )
    if problems:
        raise RuntimeError(f"{TITAN_ARM}: " + "; ".join(problems))


def build_dispatch_permute_titan(
    shape: PiperShape, workload: KernelWorkload, inputs: DispatchPermuteInputs
) -> BuiltArm:
    """TorchTitan's map construction plus ``token_dispatcher.dispatch``.

    Compiled with ``fullgraph=True``, which is the production treatment: the
    whole of ``MoE.forward`` sits inside the per-block ``torch.compile`` that
    ``apply_compile`` wraps around each ``Qwen3TransformerBlock``. The megatron
    arms run eager, because megatron compiles no whole layer, so **the
    cross-engine ratio this scenario publishes is a comparison of two compile
    treatments as well as of two implementations**. ``KernelArm.compiled``
    records both sides and plan rule 1 requires every table to print them.

    ``fullgraph=True`` turns a graph break into a build failure rather than
    into a silently half-eager arm. That is the direction this scenario needs:
    the titan side is one argsort and two gathers, so a graph break would move
    a large fraction of the measurand into eager dispatch without changing any
    number's label.

    The compile is applied to a function rather than to a module because
    ``AllToAllTokenDispatcher`` is a ``Configurable`` and not an
    ``nn.Module``, so ``operations/common.py``'s ``_compile_module`` does not
    apply. ``attn_residual`` compiles a function for the same reason.
    """
    dispatcher = titan_dispatcher(shape)
    _require_titan_dispatch_contract(dispatcher, shape)
    print(
        f"dispatch_permute/{TITAN_ARM}: "
        f"layers[{TITAN_LAYER}].moe.routed_experts.token_dispatcher is "
        f"{type(dispatcher).__name__} with ep_mesh=None, so dispatch takes the "
        "LocalTokenDispatcher branch",
        flush=True,
    )
    compiled = torch.compile(_titan_dispatch, fullgraph=True)

    batch, seq = workload.batch, workload.seq_len
    tokens = batch * seq
    experts, top_k = shape.num_experts, shape.top_k
    topk_expert_ids_BLK = inputs.topk_expert_ids_TK.reshape(batch, seq, top_k)
    # ``scores_BLE`` is the router's third output, and this scenario reads only
    # its shape and dtype: ``MoE.forward`` uses it as the template for
    # ``torch.zeros_like(scores_BLE, dtype=torch.bool)`` (``moe.py:465``) and
    # for nothing else, so no gradient flows through it and its values do not
    # reach any measured quantity. Passing the shared dense probabilities keeps
    # the scenario to one probability tensor instead of two.
    scores_BLE = inputs.probs_TE.reshape(batch, seq, experts)

    def bind(
        x_leaf: torch.Tensor, probs_leaf: torch.Tensor
    ) -> Callable[[], tuple]:
        topk_scores_BLK = probs_leaf.view(batch, seq, top_k)

        def call() -> tuple:
            return compiled(
                dispatcher,
                x_leaf,
                topk_scores_BLK,
                topk_expert_ids_BLK,
                scores_BLE,
            )

        return call

    def dense_probs_grad(grad_TK: torch.Tensor) -> torch.Tensor:
        # Titan's leaf is [T, K] and megatron's is the dense [T, E]. The
        # canonical form is the dense one, so the titan gradient is scattered
        # back to the routed positions -- the same correspondence the inputs
        # builder used to make ``probs_TE`` out of ``topk_scores_TK``. It runs
        # in ``correctness_outputs`` alone and never inside a timed closure.
        return torch.zeros(
            (tokens, experts), device=grad_TK.device, dtype=grad_TK.dtype
        ).scatter_(1, inputs.topk_expert_ids_TK, grad_TK.reshape(tokens, top_k))

    _require_a_real_permutation(
        bind(inputs.x_BLD, inputs.topk_scores_TK),
        inputs,
        shape,
        workload,
        TITAN_ARM,
    )
    return _dispatch_permute_arm(
        name=TITAN_ARM,
        bind=bind,
        x_native=inputs.x_BLD,
        probs_native=inputs.topk_scores_TK,
        inputs=inputs,
        dense_probs_grad=dense_probs_grad,
        notes={"dispatcher": type(dispatcher).__name__},
    )


def _require_mcore_dispatch_contract(
    moe_layer: Any,
    arm: str,
    shape: PiperShape,
    dispatcher_class: str,
    permute_fusion: bool,
) -> dict[str, Any]:
    """Refuse an mcore arm whose configuration makes this a different cut.

    Every value read here reaches the timed closure, and each of them can
    change what the closure computes or which branch it takes without changing
    whether it runs.

    **The dispatcher class is the delta ``mcore/dispatcher_alltoall``
    measures**, and it is read off the built object rather than off the profile
    that asked for it. ``MoELayer.__init__`` maps
    ``config.moe_token_dispatcher_type`` onto the class
    (``moe_layer.py:299-322``), and this is the direct evidence that the map
    took effect.

    **``moe_permute_fusion`` is the delta ``mcore/no_permute_fusion``
    measures, and the config field is decisive here.** ``permute`` reads it as
    its ``fused=`` argument at the one call site inside this cut
    (``token_dispatcher.py:324`` for the allgather dispatcher, ``:673`` for the
    alltoall one) and branches on that argument alone
    (``moe_utils.py:404-433`` fused, ``:461-486`` torch). There is no second
    delivery mechanism that could disagree, which is what separates this flag
    from ``moe_grouped_gemm`` -- where the expert class comes from the layer
    spec and the config field alone is inert. When the flag is on, the fused
    symbols must also exist: ``TransformerConfig`` already raises without them
    (``transformer_config.py:2677-2693``, TE >= 2.1.0 against 2.17.1 here), and
    this asserts the same fact at the arm so the claim is checked where it is
    made.

    **Six further fields decide which branch ``permute`` takes at all.**
    ``moe_expert_capacity_factor`` and ``moe_router_padding_for_quantization``
    make the output size dynamic and add a synchronize
    (``token_dispatcher.py:542-549``); ``moe_pad_expert_input_to_capacity``
    turns on the drop-and-pad path (``moe_utils.py:439-460``), which is a
    different permutation; ``batch_invariant_mode`` forbids the fused path,
    though **only on the alltoall arm and only indirectly** -- the assert at
    ``moe_utils.py:400-401`` keys on ``permute``'s
    ``return_batch_invariant_inverse_map`` argument, and the alltoall
    dispatcher is the one caller in this cut that forwards the config field
    into it (``token_dispatcher.py:676``), so on the two allgather arms the
    field reaches nothing here at all; ``moe_latent_size`` inserts a
    projection into ``MoELayer.preprocess`` (``moe_layer.py:482-483``) that
    belongs to no scenario in this partition; and ``cuda_graph_impl`` other
    than ``"none"`` lets the ``maybe_skip_or_early_return_by_cudagraph``
    decorator take ``preprocess`` down a skip or an early-return branch
    (``moe_utils.py:1680-1685``).

    ``num_local_experts`` and ``local_expert_indices`` prove the arm measures
    every expert: ``dispatch_postprocess`` slices the routing map to the local
    experts (``token_dispatcher.py:309-315``), and a partial slice would permute
    a subset of the rows under the scenario's label.
    """
    config = moe_layer.config
    dispatcher = moe_layer.token_dispatcher
    found = type(dispatcher).__name__
    problems: list[str] = []
    if found != dispatcher_class:
        problems.append(
            f"token_dispatcher is {found}, and this arm declares "
            f"{dispatcher_class}"
        )
    if config.moe_permute_fusion is not permute_fusion:
        problems.append(
            f"config.moe_permute_fusion is {config.moe_permute_fusion!r}, and "
            f"this arm declares {permute_fusion!r}"
        )
    if config.moe_latent_size is not None:
        problems.append(
            f"config.moe_latent_size is {config.moe_latent_size!r}, expected "
            "None; a latent projection inside preprocess belongs to no "
            "scenario in this partition"
        )
    if config.cuda_graph_impl != "none":
        problems.append(
            f"config.cuda_graph_impl is {config.cuda_graph_impl!r}, expected "
            "'none'; the cudagraph decorator can skip preprocess entirely"
        )
    if config.moe_expert_capacity_factor is not None:
        problems.append(
            f"config.moe_expert_capacity_factor is "
            f"{config.moe_expert_capacity_factor!r}, expected None; token "
            "dropping makes the permuted size dynamic and adds a synchronize"
        )
    if config.moe_pad_expert_input_to_capacity:
        problems.append(
            "config.moe_pad_expert_input_to_capacity is True; the "
            "drop-and-pad path is a different permutation"
        )
    if config.moe_router_padding_for_quantization:
        problems.append(
            "config.moe_router_padding_for_quantization is True; the routing "
            "map is padded before permute and the permutation changes"
        )
    if config.batch_invariant_mode:
        problems.append(
            "config.batch_invariant_mode is True, which forbids the fused "
            "permute on the alltoall arm: dispatch_preprocess forwards it as "
            "permute's return_batch_invariant_inverse_map "
            "(token_dispatcher.py:676), and permute then asserts not fused "
            "(moe_utils.py:400-401). It is refused on every arm because an "
            "arm that took the unfused path under a fused label would be a "
            "wrong number rather than a missing one"
        )
    if config.overlap_dispatch_backward_with_experts_wgrad:
        problems.append(
            "config.overlap_dispatch_backward_with_experts_wgrad is True; "
            "MoELayer.dispatch then wraps the hidden states in "
            "_RegisterDelayedWgradForExperts.apply (moe_layer.py:496-497), "
            "which puts an autograd Function inside the timed closure and "
            "defers part of the expert weight gradient out of it"
        )
    if dispatcher.tp_size != 1 or dispatcher.ep_size != 1:
        problems.append(
            f"tp_size is {dispatcher.tp_size} and ep_size is "
            f"{dispatcher.ep_size}, expected 1 and 1"
        )
    if dispatcher.num_local_experts != shape.num_experts:
        problems.append(
            f"num_local_experts is {dispatcher.num_local_experts} and the "
            f"shape declares {shape.num_experts} experts; the arm would permute "
            "a subset of the rows"
        )
    if list(dispatcher.local_expert_indices) != list(range(shape.num_experts)):
        problems.append(
            f"local_expert_indices is {list(dispatcher.local_expert_indices)}, "
            f"expected {list(range(shape.num_experts))}"
        )
    if permute_fusion:
        from megatron.core.transformer.moe import moe_utils

        missing = sorted(
            name
            for name in ("fused_permute", "fused_permute_with_probs")
            if getattr(moe_utils, name, None) is None
        )
        if missing:
            problems.append(
                f"moe_permute_fusion is on and {', '.join(missing)} is None, "
                "so permute() would raise rather than take the fused path"
            )
    if problems:
        raise RuntimeError(f"{arm}: " + "; ".join(problems))
    return {
        "dispatcher": found,
        "moe_permute_fusion": bool(config.moe_permute_fusion),
        "moe_token_dispatcher_type": str(config.moe_token_dispatcher_type),
    }


def _mcore_dispatch(
    moe_layer: Any,
    hidden_TBD: torch.Tensor,
    probs_TE: torch.Tensor,
    routing_map_TE: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Megatron's ``preprocess`` plus all three dispatch phases.

    The first two phases are reached through the public ``MoELayer`` methods
    (``moe_layer.py:447-487`` and ``:489-498``) and the third through the
    public dispatcher method ``dispatch_postprocess``, because ``MoELayer``
    offers no public wrapper for it: the method that calls it in production is
    ``routed_experts_compute``, which is ``@internal_api`` (``moe_layer.py:
    528``) and therefore outside megatron's compatibility guarantee.

    **``moe_layer.dispatch`` is the identity here and is called anyway.** At
    ``world_size=1`` ``token_dispatch`` returns its arguments unchanged
    (``token_dispatcher.py:282`` for the allgather dispatcher; for the alltoall
    one the two ``all_to_all`` calls at ``:702`` and ``:715`` reach
    ``_AllToAll.forward``, which returns its input at ``mappings.py:433-436``).
    It is inside the cut because the partition names the whole triple, and
    because the alltoall dispatcher's inert collective still costs two
    ``autograd.Function.apply`` dispatches per call -- real host work that
    megatron pays at every layer.
    """
    hidden, probs = moe_layer.preprocess(hidden_TBD, probs_TE, routing_map_TE)
    hidden, probs = moe_layer.dispatch(hidden, probs)
    return moe_layer.token_dispatcher.dispatch_postprocess(hidden, probs)


def _build_dispatch_permute_mcore(
    *,
    arm: str,
    profile: McoreProfile,
    dispatcher_class: str,
    permute_fusion: bool,
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: DispatchPermuteInputs,
) -> BuiltArm:
    """One megatron arm: ``MoELayer.preprocess`` plus all three phases, eager.

    The three megatron arms differ in one profile field each and in nothing
    else, so they share this body. Splitting them into three hand-written
    builders would let the difference drift into something the profile does not
    describe.

    The layer comes off a real ``GPTModel`` rather than from a hand
    construction. That matters here specifically: the dispatcher class is
    chosen by ``MoELayer.__init__`` from ``config.moe_token_dispatcher_type``
    (``moe_layer.py:299-322``), and ``num_local_experts`` and
    ``local_expert_indices`` are derived from the expert-parallel group size
    (``:181-194``). Both are exactly what ``mcore_profiles.py`` records about
    hand-written layer specs: a value retyped by hand can disagree silently and
    publish one implementation under another's label.

    **Reading it off a real model costs a real model, and the experts are then
    released.** ``build_model`` allocates the whole network to hand back one
    ``MoELayer``, and the expert weights inside that layer are its largest part
    -- about 88 MiB of bf16 at ``normal`` and about 12.7 GiB at ``huge``. No
    phase in this cut touches ``self.experts``: ``preprocess`` and ``dispatch``
    read ``self.config`` and ``self.token_dispatcher``, and
    ``dispatch_postprocess`` is a method on the dispatcher. So the attribute is
    cleared before the model is dropped, which keeps ``memory_pass`` a statement
    about the dispatch rather than about a resident expert layer.
    ``_require_a_real_permutation`` then runs the closure end to end, so a
    release that broke the cut fails the build rather than a table.

    Eager on purpose, and the reason is specific to this cut rather than
    generic. Megatron compiles no whole layer, and **no method this closure
    calls carries ``@jit_fuser``**: ``token_dispatcher.py`` holds exactly one
    such decorator (``:1860``), on ``MoEFlexTokenDispatcher.
    dispatch_preprocess``, and this scenario never builds that class: it is
    reached only through ``moe_token_dispatcher_type="flex"``
    (``moe_layer.py:313``), which neither profile here sets, and which two of
    its three backends refuse at one rank outright (``:1775`` deepep and
    ``:1793`` ncclep assert ``tp_size * ep_size > 1``; the ``hybridep``
    branch at ``:1784-1790`` carries no such assert, so "flex cannot be built
    at ``world_size=1``" would be too strong a claim to make). So this arm is
    eager all
    the way down, unlike ``attn_residual``'s ``mcore/base``, whose entry point
    is itself a ``torch.compile`` wrapper.
    """
    initialize_megatron_single_rank()

    from megatron.core.transformer.moe.moe_layer import MoELayer

    from benchmarks.models.piper_qwen3.megatron_model import build_model

    model = build_model(seq_len=workload.seq_len, shape=shape, profile=profile)
    moe_layer = model.decoder.layers[MCORE_LAYER].mlp
    if not isinstance(moe_layer, MoELayer):
        raise RuntimeError(
            f"{arm}: decoder.layers[{MCORE_LAYER}].mlp is "
            f"{type(moe_layer).__name__}, not a MoELayer; this scenario cuts "
            "the MoE dispatch phases and a dense MLP has none"
        )
    notes = _require_mcore_dispatch_contract(
        moe_layer, arm, shape, dispatcher_class, permute_fusion
    )
    notes["profile"] = profile.name
    print(
        f"dispatch_permute/{arm}: "
        f"decoder.layers[{MCORE_LAYER}].mlp.token_dispatcher is "
        f"{notes['dispatcher']} (profile {profile.name}, "
        f"moe_permute_fusion={notes['moe_permute_fusion']})",
        flush=True,
    )
    # See the docstring: no phase in this cut reads self.experts, and the
    # weights are the largest thing the layer holds.
    moe_layer.experts = None
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    batch, seq = workload.batch, workload.seq_len
    tokens = batch * seq

    def bind(
        x_leaf: torch.Tensor, probs_leaf: torch.Tensor
    ) -> Callable[[], tuple]:
        # THD, the form our megatron driver runs: the driver packs each batch
        # into one sequence, so a hidden state reaches the MoE layer as
        # (t, 1, h). ``dispatch_preprocess`` immediately views it to (t, h)
        # (``token_dispatcher.py:274``), so the leading layout is a label. A
        # free view of a contiguous tensor, taken here and never inside a timed
        # closure.
        hidden_TBD = x_leaf.view(tokens, 1, x_leaf.shape[-1])

        def call() -> tuple:
            return _mcore_dispatch(
                moe_layer, hidden_TBD, probs_leaf, inputs.routing_map_TE
            )

        return call

    _require_a_real_permutation(
        bind(inputs.x_BLD, inputs.probs_TE), inputs, shape, workload, arm
    )
    return _dispatch_permute_arm(
        name=arm,
        bind=bind,
        x_native=inputs.x_BLD,
        probs_native=inputs.probs_TE,
        inputs=inputs,
        # Megatron's leaf is already the canonical dense [T, E] form, so this
        # is the identity and the titan arm carries the conversion.
        dense_probs_grad=lambda grad_TE: grad_TE,
        notes=notes,
    )


def build_dispatch_permute_mcore_base(
    shape: PiperShape, workload: KernelWorkload, inputs: DispatchPermuteInputs
) -> BuiltArm:
    """megatron at its own best: allgather dispatcher, TE permutation fusion.

    The base profile sets ``moe_token_dispatcher_type="allgather"`` and
    ``moe_permute_fusion=True``, which is what the e2e megatron arm runs, so
    this is the anchor every other row in the scenario is measured against.

    **Its number includes a blocking device-to-host copy.**
    ``dispatch_postprocess`` runs ``.cpu()`` on the per-expert counts at
    ``token_dispatcher.py:317``, unconditionally, on every call. See the module
    docstring: plan rule 5 makes that a declared property of the scenario
    rather than a footnote.
    """
    return _build_dispatch_permute_mcore(
        arm=MCORE_BASE_ARM,
        profile=BASE,
        dispatcher_class=ALLGATHER_DISPATCHER,
        permute_fusion=True,
        shape=shape,
        workload=workload,
        inputs=inputs,
    )


def build_dispatch_permute_mcore_no_permute_fusion(
    shape: PiperShape, workload: KernelWorkload, inputs: DispatchPermuteInputs
) -> BuiltArm:
    """megatron with the TE permutation fusion off.

    ``permute`` falls from ``fused_permute`` (``moe_utils.py:404-410``, TE's
    ``moe_permute``) to the torch path (``:461-486``): a ``.bool().T.
    contiguous()``, a stable descending argsort of the flattened map, a slice
    to ``num_out_tokens``, a modulo, and an ``index_select``.

    **This flag is a part of the scenario, not the whole of it.** Under the
    allgather dispatcher the cut holds exactly one read site --
    ``permute`` in ``dispatch_postprocess`` (``token_dispatcher.py:324``) --
    beside ``MoELayer.preprocess``, the ``local_map`` and ``local_probs``
    slices, the ``.cpu()`` at ``:317`` and the by-hand probability permutation
    at ``:328-330``. So the row against the anchor is diluted by everything
    the flag does not touch, and it understates what the fusion is worth to
    the permute itself.
    """
    return _build_dispatch_permute_mcore(
        arm=MCORE_NO_PERMUTE_FUSION_ARM,
        profile=NO_PERMUTE_FUSION,
        dispatcher_class=ALLGATHER_DISPATCHER,
        permute_fusion=False,
        shape=shape,
        workload=workload,
        inputs=inputs,
    )


def build_dispatch_permute_mcore_dispatcher_alltoall(
    shape: PiperShape, workload: KernelWorkload, inputs: DispatchPermuteInputs
) -> BuiltArm:
    """megatron with the alltoall dispatcher instead of the allgather one.

    **THE COLLECTIVE IS INERT AT ``world_size=1``; THE CLASS IS NOT. This arm
    measures the dispatcher's local permute and sync strategy, never
    communication.** ``_AllToAll.forward`` returns its input unchanged when the
    group has one rank (``tensor_parallel/mappings.py:433-436``), so nothing
    is transferred. What differs is the local work, in three places.

    Where the permute runs. The alltoall dispatcher permutes in
    ``dispatch_preprocess`` (``token_dispatcher.py:662-676``), not in
    ``dispatch_postprocess``.

    How the probabilities are permuted. It passes ``probs=probs`` into
    ``permute`` (``:671``), so **one** fused kernel permutes tokens and
    probabilities together (``moe_utils.py:412-433``,
    ``fused_permute_with_probs``). The allgather dispatcher permutes the
    probabilities by hand afterwards, with
    ``.T.contiguous().masked_select(...)`` (``token_dispatcher.py:328-330``).

    How the counts reach the host. The alltoall dispatcher issues its
    device-to-host copies on a side stream at ``cuda_dtoh_point`` and waits at
    ``cuda_sync_point`` (``:918-955``), which ``preprocess`` sets to
    ``"before_finish"`` at ``:602``. The allgather dispatcher's ``.cpu()`` at
    ``:317`` is unconditional and blocking.

    One extra piece of local work has no counterpart on the anchor and is
    **not** a difference in the permutation: with ``num_local_experts > 1``,
    ``dispatch_postprocess`` calls ``sort_chunks_by_idxs`` (``:779-785``) to
    order the chunks by local expert. At one rank the index vector
    ``sort_input_by_local_experts`` is ``arange(E).reshape(1, E).T.ravel()``
    (``:429-435``), which is ``[0..E-1]`` -- the identity -- so the rows come
    out in the anchor's order and the cross-arm gates hold, while the work is
    still done and still costs.
    """
    return _build_dispatch_permute_mcore(
        arm=MCORE_DISPATCHER_ALLTOALL_ARM,
        profile=DISPATCHER_ALLTOALL,
        dispatcher_class=ALLTOALL_DISPATCHER,
        permute_fusion=True,
        shape=shape,
        workload=workload,
        inputs=inputs,
    )
