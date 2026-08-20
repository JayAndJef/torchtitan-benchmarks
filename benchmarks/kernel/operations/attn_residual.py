"""Arm builders for the ``attn_residual`` kernel scenario.

The residual add after attention, on both engines. TorchTitan writes it as
``x = x + self.attention(self.attention_norm(x), attention_masks, positions)``
(``third_party/torchtitan/torchtitan/models/qwen3/model.py:60``). Megatron-core
writes it as ``self_attn_bda(...)``
(``third_party/Megatron-LM/megatron/core/transformer/transformer_layer.py:
683-686``, inside ``TransformerLayer._forward_attention``).

**Both engines compute ``residual + attn_out``, and they compute it
identically.** The base profile sets ``hidden_dropout`` to 0.0 and
``add_bias_linear`` to False (``mcore_profiles.py:158,153``), so
``_bias_dropout_add_func`` takes the ``bias is None`` branch
(``fusions/fused_bias_dropout.py:53-59``) and ``F.dropout(p=0.0)`` returns its
own input tensor. What remains on that branch is the single statement
``out = residual + out``. The name ``bias_dropout_add`` describes the
function's whole domain, not the case this model runs.
``_require_exact_residual_add`` proves the equality on the device before any
arm is timed, so the claim is a guard rather than a comment.

**The difference between the two engines is fusion scope, not presence.**
TorchTitan's add is one node of a whole-block Inductor graph. It folds into
the prologue of the next norm and emits no kernel of its own. Megatron's
``bias_dropout_add_fused_train`` carries ``@jit_fuser``
(``fused_bias_dropout.py:69``), which ``megatron/core/jit.py:20-21`` rebinds to
``torch.compile`` on torch >= 2.2 and ``:33`` applies at import. It therefore
compiles as its own region and emits one standalone add that cannot fuse
outward. An isolated titan arm has no neighbour to fold into, so isolation
forces titan into megatron's fusion scope.

**So this scenario publishes no cross-engine ratio, and the omission is
deliberate.** A titan-against-mcore row would land near 1.0 and would report
the isolation, not the engines. ``KernelScenario.comparisons`` is an explicit
tuple that names the within-engine row alone; plan section C.0 rule 6 lists
this scenario as one of the four in that class. Do not "restore" the missing
row.

**What the published row measures is host dispatch.** The two megatron arms
run one bf16 add each, over the same tensors, on the same device. In
``forward`` their device work is identical, and that is traced rather than
assumed: at ``prob=0.0`` the compiled region's AOTAutograd forward graph is
one ``aten.add.Tensor`` and its backward returns the tangent twice, because
decomposition deletes the zero-probability dropout. So the difference between
the arms is dispatch: ``bias_dropout_add_fused_train`` enters a compiled
region and checks its Dynamo guards, while ``bias_dropout_add_unfused``
allocates a fresh Python closure on every call
(``fused_bias_dropout.py:62-66``) and dispatches ``F.dropout`` and ``add``
eagerly. Read the ``--burst`` residual before quoting the number, and never
present this row as a kernel-speed claim.

**The identical-device-work claim is stated for ``forward`` alone.** The
eager backward is ``AddBackward0`` plus two ``AccumulateGrad``; the compiled
backward is a separately compiled region returning the same tangent twice.
Whether Inductor inserts a copy eager does not is **unmeasured**, and no
guard here checks it, so read the ``forward_backward`` row as a comparison of
two backward paths rather than as dispatch alone.

**THE MOST LIKELY WAY THIS ROW PUBLISHES A WRONG NUMBER IS A NULL.** The
measurand is a CUDA-event interval around a burst, so it collapses onto
device time wherever the host keeps the stream fed. One add here moves
24 MiB, which is roughly 11-13 us of device work at ``normal``, batch 4,
sequence 1024. The two dispatch paths plausibly cost anything from 5 to 40 us
each, so this scenario sits **at the crossover** rather than safely above it,
unlike ``rope``, where dispatch is 6-19x the floor. If the arms are
device-bound, both report the same ~12 us, the ratio lands at 1.00 with a
tight interval, and the whole declared delta is invisible -- which is the most
publishable-looking wrong answer a benchmark can give. ``--burst`` cannot
settle it, because CLAUDE.md records the residual test as one-sided.
**``copy_floor`` is the instrument that settles it: an ``x_floor`` near 1.5 on
BOTH arms means the row measured the memory bus and not the fusion.** Read
that column before quoting the ratio.

**The flag names three operations and this model runs one.** With
``hidden_dropout`` at 0.0 and ``add_bias_linear`` False, the bias add and the
dropout do not exist, so ``bias_dropout_fusion`` selects a ``@jit_fuser``
region wrapped around a single add with nothing to fuse it to. The row is
therefore "what wrapping one add in ``torch.compile`` costs", not "what
megatron's bias-dropout-add fusion costs". The arm keeps the flag's name
because the flag is what the profile sets; the caption must not.

**One host cost of the real call site is excluded.** The real site is
bracketed by ``nvtx_range_push``/``nvtx_range_pop``
(``transformer_layer.py:676`` and ``:687``), and the timed closure reproduces
lines 683-686 only. Both calls early-return when NVTX is disabled
(``megatron/core/utils.py:2712-2713``), which is the state of a run without
profiling, so the omission is small -- but it is an omission and not a
neutrality.

**The dispatch that selects the implementation is inside the timed closure,
on purpose.** The call site re-evaluates
``self.self_attn_bda(self.training, self.config.bias_dropout_fusion)`` on
every forward (``transformer_layer.py:684``), and that call is where the
unfused arm builds its closure. Hoisting it would move half of the measured
delta out of the measurement. The enclosing
``with self.bias_dropout_add_exec_handler():`` (``:683``, bound to
``torch.enable_grad`` at ``:525``) is inside the closure for the same reason:
megatron enters that context on every layer of every step.

**One piece is excluded, and it belongs upstream.** Megatron's
``attention_output_with_bias`` tuple is the return value of
``self.self_attention(...)`` (``transformer_layer.py:653``), so scenario 6
produces it and this scenario receives it. Each arm builds its tuple once, at
build time, over the leaf it will use.

**There is no isolated ``backward`` mode.** The gradient of an add is its own
input gradient, so a separate backward row would measure the dispatch of
``torch.autograd.backward`` and little else. Both engines declare ``forward``
and ``forward_backward``, and the backward cost stays recoverable as the
difference.

**A floor arm carries the byte count, and no other arm does.** The forward
moves two reads and one write; the ``forward_backward`` row moves something
else, so one count on a two-mode arm would attach the forward's GB/s figure
to both rows. ``embedding_stage`` publishes its count on the floor alone for
the same reason, and ``copy_floor`` here does the same.

**The floor is what tells a kernel result from a dispatch comparison.** Both
sides of the published row run one bf16 add on the same tensors, so their
device work is equal and the ratio is entirely host dispatch. The floor gives
that claim a second, independent reading. One bf16 add at ``normal``, batch
4, sequence 1024 moves 24 MiB -- two reads and one write of an 8 MiB tensor --
which is exactly the traffic ``rope/copy_floor`` moves, and CLAUDE.md records
that floor's measured device cost as 11-13 us. It is a copy, so it moves two thirds of the
add's bytes and understates the device by the same third --
``build_attn_residual_copy_floor`` states the 1.5x correction and the
direction of the bias.

Scenario 13, ``moe_residual``, is this scenario's twin. It cuts at
``mlp_bda`` (``transformer_layer.py:979-982``) against titan's
``x = x + self.moe(self.ffn_norm(x))`` (``qwen3/model.py:63``), it declares
the same three arms, and it declines the cross-engine row for the same
reason. The two modules stay separate because the partition gives each cut
one scenario.

Every torchtitan, megatron and TransformerEngine import is deferred into the
builder that needs it, which is the rule across ``operations/``.
``benchmarks.models.piper_qwen3.mcore_profiles`` is the one exception at
module scope: it is torch-free parent-side data, and this module derives one
profile from it.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Callable

import torch

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    _randn,
    _randn_like,
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


# The three arm names, spelled as plan section C.2 spells them. The slash says
# which engine an arm is and which profile of that engine. It reaches a
# filename through ``benchmarks/kernel/runner.py``'s ``fragment_path``, which
# routes it through ``benchmarks.kernel.schema.fragment_stem``. **A tree
# without ``fragment_stem`` cannot run this scenario**: a timing worker would
# write into a ``timing__mcore`` directory nobody creates, and the arm would
# land as ``failed``.
COPY_FLOOR_ARM = "copy_floor"
MCORE_BASE_ARM = "mcore/base"
MCORE_NO_FUSION_ARM = "mcore/no_bias_dropout_fusion"
TITAN_ARM = "titan"

# The layer the two mcore arms read their call site from. Every layer holds
# the same function at ``moe_layer_freq=1``, so the index is arbitrary. It is
# fixed here so the provenance line names one layer.
MCORE_LAYER = 0
MCORE_LAYER_PATH = f"decoder.layers.{MCORE_LAYER}"

# The name megatron gives the function this scenario times. Both arms resolve
# to a callable through it, and the guard below reads the resolved object
# rather than this string.
MCORE_BDA_ATTRIBUTE = "self_attn_bda"


# The one megatron variant this scenario adds, as a delta rather than a second
# copy of 28 flags. Declared here rather than added to ``MCORE_PROFILES``,
# which is the rule ``cross_entropy.py`` records for its two variants: the
# registry is the roster of profiles other systems run, and the e2e megatron
# arm plus every other cross-engine scenario run ``base``.
#
# ``bias_dropout_fusion`` is one of ``mcore_profiles.FUSION_FIELDS``, so
# ``benchmarks/e2e/megatron/train.py`` would assert the built config against
# this declaration in both directions if a run ever took this profile. The
# base profile sets the flag True because megatron's own argparse layer does:
# ``--no-bias-dropout-fusion`` is ``action="store_false"``, and the
# ``TransformerConfig`` dataclass default is the opposite of what a real
# megatron run gets. Turning it off is therefore a deviation from megatron,
# not a return to its default.
NO_BIAS_DROPOUT_FUSION_PROFILE: McoreProfile = derive(
    BASE,
    name="no_bias_dropout_fusion",
    description=(
        "megatron with the bias-dropout-add fusion off: the call site "
        "resolves to bias_dropout_add_unfused, which builds a Python closure "
        "per call and dispatches the same arithmetic eagerly. The device work "
        "is one bf16 add either way, so this profile isolates the cost of the "
        "@jit_fuser region and nothing else"
    ),
    config_overrides={"bias_dropout_fusion": False},
)


@dataclass
class AttnResidualInputs:
    """The attention output, the block input, and one output gradient.

    ``attn_out`` is what titan's ``self.attention(...)`` returns and what
    megatron carries as ``attention_output_with_bias[0]``. ``residual`` is the
    block input that both engines add to it: titan's ``x`` at
    ``qwen3/model.py:60`` and megatron's ``residual`` at
    ``transformer_layer.py:637``.

    Both are ``[B, L, D]`` bf16 and contiguous. The mcore arms take a free
    ``[B*L, 1, D]`` view of the same storage, because our megatron driver runs
    THD and ``Attention.forward`` hands the layer a ``(t, b, h)`` tensor. An
    add treats every leading dimension as a row index, so the view is a label
    and not a measured difference. It is taken at build time either way.
    """

    attn_out: torch.Tensor  # (B, L, D) bf16, contiguous
    residual: torch.Tensor  # (B, L, D) bf16, contiguous
    grad_out: torch.Tensor  # (B, L, D) bf16
    # The floor's own traffic: one read and one write of one tensor. It is
    # the floor's number, not the add's. The add reads two operands and
    # writes one, so its forward traffic is 1.5x this. See
    # ``build_attn_residual_copy_floor`` for what that costs a reader.
    copy_bytes: int


def attn_residual_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> AttnResidualInputs:
    """Three independent bf16 tensors at the canonical activation shape.

    The two addends are drawn separately rather than derived from each other.
    An arm that dropped one operand would otherwise still pass a gate built
    from a shared draw.
    """
    attn_out = _randn(
        (workload.batch, workload.seq_len, shape.dim), device, generator
    )
    return AttnResidualInputs(
        attn_out=attn_out,
        residual=_randn_like(attn_out, generator),
        grad_out=_randn_like(attn_out, generator),
        copy_bytes=2 * attn_out.numel() * attn_out.element_size(),
    )


def attn_residual_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: AttnResidualInputs
) -> dict[str, torch.Tensor]:
    """fp64 truth for the sum and for both input gradients.

    The addends are promoted from bf16 and are not rounded again, because
    neither engine casts them: both add two bf16 tensors and round once. The
    two gradients are the output gradient itself, which is a weak statement
    about the arithmetic and a strong one about the graph. An arm that
    detached an operand, scaled a gradient, or applied a real dropout mask
    fails here.
    """
    attn_out = inputs.attn_out.double().detach().requires_grad_()
    residual = inputs.residual.double().detach().requires_grad_()
    out = residual + attn_out
    torch.autograd.backward(out, inputs.grad_out.double())
    return {
        "out": out.detach(),
        "attn_out_grad": attn_out.grad,
        "residual_grad": residual.grad,
    }


def _is_torch_compiled(function: object) -> bool:
    """Whether ``torch.compile`` wraps this callable.

    ``torch.compile`` returns a plain function that carries
    ``_torchdynamo_orig_callable``. The attribute is private, and the torch
    pin in this repository is exact, so a torch bump that renames it makes
    ``_require_bda_dispatch`` raise rather than pass silently. That is the
    correct direction: the whole published row rests on one arm being
    compiled and the other not.
    """
    return getattr(function, "_torchdynamo_orig_callable", None) is not None


def _require_exact_residual_add(
    call: Callable[[], torch.Tensor],
    attn_out: torch.Tensor,
    residual: torch.Tensor,
    arm: str,
) -> None:
    """Refuse an arm that does not compute exactly ``residual + attn_out``.

    This is the scenario's central claim, checked on the device in every
    worker that builds an arm. Four ways to break it are silent without the
    check, and each of them publishes a wrong number rather than a missing
    one.

    A profile with ``hidden_dropout > 0`` makes ``F.dropout`` a real mask, so
    the mcore arms would compute a different function and would also stop
    being deterministic. A profile with ``add_bias_linear`` True moves
    ``_bias_dropout_add_func`` onto its bias branch
    (``fused_bias_dropout.py:42-52``), which is one extra add. A profile with
    ``fp32_residual_connection`` True upcasts the residual at
    ``transformer_layer.py:639-640``, which changes the output dtype. And a
    titan arm whose compile treatment silently changed its arithmetic would
    otherwise reach a table.

    The comparison is bitwise, and it can be. The exact sum of two bf16
    values fits in fp32, and every torch backend accumulates a bf16 add in
    fp32, so the correctly rounded bf16 result is the only result either side
    can produce. A tolerance here would accept the bias branch instead of
    refusing it.

    **What it does not check is the operand order.** IEEE-754 addition
    commutes bit for bit, so an arm that swapped the two addends passes this
    guard. The order is read off the two call sites instead
    (``qwen3/model.py:60`` and ``fused_bias_dropout.py:58``), and for an add
    the swap changes no result -- but a reader must not take this guard as
    evidence about it.

    The probe tensors require no gradient, so the ``torch.enable_grad``
    context the mcore closure enters records nothing. ``_bias_dropout_add_
    func`` still takes its out-of-place path, because ``inplace`` needs
    ``not training`` and both mcore arms run a module in train mode.
    """
    with torch.no_grad():
        expected = residual + attn_out
    observed = call()
    if observed.shape != expected.shape or observed.dtype != expected.dtype:
        raise RuntimeError(
            f"{arm}: the call returned a {tuple(observed.shape)} "
            f"{observed.dtype} tensor, and residual + attn_out is "
            f"{tuple(expected.shape)} {expected.dtype}. The arm does not "
            "compute the residual add this scenario declares."
        )
    if not torch.equal(observed.detach(), expected):
        raise RuntimeError(
            f"{arm}: the call does not return residual + attn_out bitwise. "
            "Check hidden_dropout, add_bias_linear and "
            "fp32_residual_connection on the profile: each of them turns this "
            "call site into a different function, and this scenario declines "
            "its cross-engine row precisely because the function is one "
            "function today."
        )


def _require_bda_dispatch(
    *,
    arm: str,
    fused: bool,
    resolved: object,
    fused_train: object,
) -> dict[str, Any]:
    """Refuse to time an arm whose call site resolved the wrong way.

    ``resolved`` is what ``get_bias_dropout_add(training, fused)`` returned
    (``fused_bias_dropout.py:83-94``). ``fused_train`` is megatron's
    ``bias_dropout_add_fused_train`` itself. Three facts are checked, and each
    of them **raises**, because ``BuiltArm.notes`` reaches no artifact and a
    fact recorded there is a fact no reader sees.

    First, ``fused_train`` must be a ``torch.compile`` wrapper. That is the
    delta the published row measures. ``megatron/core/jit.py:7`` binds
    ``jit_fuser`` to ``torch.jit.script`` and ``:16-24`` rebinds it to
    ``torch.compile`` only on torch >= 2.2. ``megatron/training/global_vars.
    py:156-157`` can call ``disable_jit_fuser()``, but only a call that ran
    **before** ``fused_bias_dropout`` was imported can reach the decorator:
    that module imports ``jit_fuser`` by value (``fused_bias_dropout.py:6``)
    and applies it at ``:69``. Under either of those the two arms would
    run the same eager arithmetic through two Python indirections, and the row
    would report a non-delta as a fusion result.

    Second, the fused arm must resolve to that exact object. Third, the
    unfused arm must resolve to something else, that something else must not
    be compiled, and it must come from ``bias_dropout_add_unfused``. A
    resolution that returned the fused function under the unfused label is the
    failure that makes the two arms byte-identical, and no correctness gate
    can see it: both are numerically right.
    """
    if not _is_torch_compiled(fused_train):
        raise RuntimeError(
            f"{arm}: bias_dropout_add_fused_train is not a torch.compile "
            "wrapper, so bias_dropout_fusion selects between two eager "
            "callables and the published ratio measures no fusion. Check "
            "megatron/core/jit.py:16-24 and the torch version. Note that a "
            "later disable_jit_fuser() cannot cause this: fused_bias_dropout "
            "imports jit_fuser by value and applies it at import, so only a "
            "call that ran BEFORE that import can reach the decorator."
        )
    if fused:
        if resolved is not fused_train:
            raise RuntimeError(
                f"{arm}: the call site resolved to "
                f"{getattr(resolved, '__qualname__', resolved)!r}, not to "
                "bias_dropout_add_fused_train. This arm claims megatron's "
                "fused path and would publish another one."
            )
    else:
        if resolved is fused_train:
            raise RuntimeError(
                f"{arm}: the call site resolved to "
                "bias_dropout_add_fused_train, which is the base arm's "
                "callable. bias_dropout_fusion=False did not reach the call "
                "site, so the two arms would measure one implementation."
            )
        if _is_torch_compiled(resolved):
            raise RuntimeError(
                f"{arm}: the unfused call site resolved to a torch.compile "
                "wrapper, so this arm is not the eager side of the fusion "
                "delta."
            )
        qualname = getattr(resolved, "__qualname__", "")
        if "bias_dropout_add_unfused" not in qualname:
            raise RuntimeError(
                f"{arm}: the call site resolved to {qualname!r}, which does "
                "not come from bias_dropout_add_unfused "
                "(fused_bias_dropout.py:62-66). The arm would publish an "
                "implementation this scenario never named."
            )
    # No ``compiled`` key. ``KernelArm.compiled`` is the authority, it is
    # what the manifest records, and it is what a reader checks without a
    # GPU -- so a copy of it here could only drift. ``resolved_is_compiled``
    # is a different fact: it is what this build observed, and it is the
    # evidence the declaration rests on.
    return {
        "bias_dropout_fusion": fused,
        "resolved": getattr(resolved, "__qualname__", str(resolved)),
        "resolved_is_compiled": _is_torch_compiled(resolved),
    }


def _require_mcore_layer_contract(layer: Any, arm: str, fused: bool) -> None:
    """Refuse a layer whose configuration makes this cut a different cut.

    Every value read here reaches the timed closure, and every one of them can
    change what the closure computes without changing whether it runs.

    ``training`` decides which of two functions ``get_bias_dropout_add``
    returns (``fused_bias_dropout.py:89-92``). An evaluation-mode layer
    returns ``bias_dropout_add_fused_inference``, whose ``_bias_dropout_add_
    func`` call passes ``training=False`` and therefore takes the **in-place**
    path when no input requires a gradient. A timed loop over an in-place add
    accumulates into its own input.

    ``config.bias_dropout_fusion`` is the flag the arm's profile sets, read
    back off the built config rather than trusted from the profile.

    ``hidden_dropout`` is the third argument of the call and must be 0.0, or
    ``F.dropout`` stops being an identity.

    ``config.add_bias_linear`` decides whether ``linear_proj`` returns a real
    bias, and therefore which branch of ``_bias_dropout_add_func`` runs.

    ``config.fp32_residual_connection`` upcasts the residual before the call
    (``transformer_layer.py:639-640``), which makes the two engines compute in
    two dtypes.
    """
    problems: list[str] = []
    if layer.training is not True:
        problems.append(
            f"layer.training is {layer.training!r}, expected True; an "
            "evaluation-mode layer resolves to the inference function, which "
            "adds in place"
        )
    if layer.config.bias_dropout_fusion is not fused:
        problems.append(
            f"config.bias_dropout_fusion is "
            f"{layer.config.bias_dropout_fusion!r}, and this arm declares "
            f"{fused!r}"
        )
    if float(layer.hidden_dropout) != 0.0:
        problems.append(
            f"layer.hidden_dropout is {layer.hidden_dropout!r}, expected 0.0; "
            "a real dropout mask makes this call a different function and a "
            "nondeterministic one"
        )
    if layer.config.add_bias_linear is not False:
        problems.append(
            f"config.add_bias_linear is {layer.config.add_bias_linear!r}, "
            "expected False; a real bias moves the call onto the bias branch "
            "of _bias_dropout_add_func"
        )
    if layer.config.fp32_residual_connection is not False:
        problems.append(
            f"config.fp32_residual_connection is "
            f"{layer.config.fp32_residual_connection!r}, expected False; the "
            "call site upcasts the residual to fp32 and the two engines stop "
            "computing in one dtype"
        )
    if problems:
        raise RuntimeError(f"{arm}: " + "; ".join(problems))


def _residual_arm(
    *,
    name: str,
    bind: Callable[
        [torch.Tensor, torch.Tensor], Callable[[], torch.Tensor]
    ],
    attn_native: torch.Tensor,
    residual_native: torch.Tensor,
    grad_native: torch.Tensor,
    canonical: tuple[int, ...],
    notes: dict[str, Any] | None = None,
) -> BuiltArm:
    """Forward and forward+backward over one engine's native tensor shape.

    ``bind`` receives the two leaves of one closure and returns the timed call
    over them. It is a factory rather than a two-argument function so each
    engine can hoist whatever is genuinely build-time work: the mcore arms
    build their ``(attn_out, bias)`` tuple once per closure, because scenario
    6 produces that tuple and this scenario receives it.

    Three independent leaf pairs, matching every other module scenario here:
    the ``forward`` closure never runs a backward, so a shared leaf would let
    one mode's gradient state reach another mode's timing.

    ``canonical`` puts every gate output back into ``[B, L, D]``. The mcore
    arms run over a ``[B*L, 1, D]`` view, and a gate compares two arms
    element by element.
    """
    forward_attn = attn_native.clone().requires_grad_()
    forward_residual = residual_native.clone().requires_grad_()
    round_trip_attn = attn_native.clone().requires_grad_()
    round_trip_residual = residual_native.clone().requires_grad_()
    check_attn = attn_native.clone().requires_grad_()
    check_residual = residual_native.clone().requires_grad_()

    forward_call = bind(forward_attn, forward_residual)
    round_trip_call = bind(round_trip_attn, round_trip_residual)
    check_call = bind(check_attn, check_residual)

    def forward() -> torch.Tensor:
        return forward_call()

    def forward_backward() -> None:
        _reset_grads(round_trip_attn, round_trip_residual)
        torch.autograd.backward(round_trip_call(), grad_native)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(check_attn, check_residual)
        out = check_call()
        torch.autograd.backward(out, grad_native)
        missing = sorted(
            label
            for label, leaf in (
                ("attn_out_grad", check_attn),
                ("residual_grad", check_residual),
            )
            if leaf.grad is None
        )
        if missing:
            raise RuntimeError(
                f"{name}: backward produced no gradient for "
                f"{', '.join(missing)}; the call detached an operand, so the "
                "arm does not carry the residual stream this scenario "
                "measures"
            )
        return {
            "out": out.detach().reshape(canonical),
            "attn_out_grad": check_attn.grad.reshape(canonical),
            "residual_grad": check_residual.grad.reshape(canonical),
        }

    return BuiltArm(
        name=name,
        calls={"forward": forward, "forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
        notes=dict(notes or {}),
    )


def build_attn_residual_copy_floor(
    shape: PiperShape, workload: KernelWorkload, inputs: AttnResidualInputs
) -> BuiltArm:
    """The bandwidth floor: one read, one write, and one eager launch.

    **This scenario needs a floor more than a bandwidth scenario does.** The
    published row puts two megatron arms that run the same single bf16 add
    against each other. Their device work is equal by construction, so the
    whole ratio is host dispatch -- a Dynamo guard check against a Python
    closure allocation, neither of which is device work. Without a floor
    nothing in the scenario separates a kernel result from a dispatch
    comparison, and CLAUDE.md records that the ``--burst`` residual test is
    one-sided: a flat ladder means ``k`` stopped buying amortization, not
    that the number became device time. The floor is the second, independent
    reading.

    **It is a lower bound, and the reader must correct for it.** A copy reads
    one tensor and writes one; the residual add reads two and writes one, so
    the add's forward traffic is **1.5x** this arm's. Multiply this arm's
    median by 1.5 before reading it as the add's device cost. The raw
    ``x_floor`` column therefore **overstates** the distance between an arm
    and the device, which is the same direction as this scenario's own
    conclusion -- so the correction is stated here rather than left implicit.

    No arrangement of ``copy_`` moves exactly two reads and one write, and an
    eager ``torch.add(..., out=...)`` would be the measured operation rather
    than a floor for it. A copy keeps the arm's name true and keeps the bias
    nameable in one number.

    Eager on purpose, and forward only: a floor measures the device, not an
    implementation, and a floor for the ``forward_backward`` traffic would be
    an invention rather than a measurement -- the backward of an add moves
    the output gradient to two places and runs no kernel of the forward's
    shape.
    """
    out = torch.empty_like(inputs.attn_out)

    def forward() -> None:
        out.copy_(inputs.attn_out)

    return BuiltArm(
        name=COPY_FLOOR_ARM,
        calls={"forward": forward},
        correctness_outputs=dict,
        bytes_moved=inputs.copy_bytes,
    )


def build_attn_residual_titan(
    shape: PiperShape, workload: KernelWorkload, inputs: AttnResidualInputs
) -> BuiltArm:
    """TorchTitan's ``x + self.attention(...)``, compiled.

    There is no config node to build and no module to extract: the call site
    is one binary operator inside ``Qwen3TransformerBlock.forward``
    (``qwen3/model.py:60``). The operand order is megatron's order as well:
    the residual is on the left in both trees (``qwen3/model.py:60`` and
    ``fused_bias_dropout.py:58``, ``out = residual + out``).

    **That order is read off the two sources, and no guard checks it.** An
    earlier draft claimed ``_require_exact_residual_add`` checked it bitwise.
    It cannot: IEEE-754 addition commutes bit for bit, so ``a + b`` and
    ``b + a`` are the same value and ``torch.equal`` accepts either. The claim
    is withdrawn rather than weakened, and ``tests/test_kernel_attn_
    residual.py`` pins the limit so nobody re-adds it.

    ``torch.compile(fullgraph=True)`` is the production treatment for a titan
    arm and is the treatment ``operations/common.py`` gives every other one.
    It is applied to a function here, not to a module, because an
    ``nn.Module`` wrapper would add a ``__call__`` this call site does not
    have, and this scenario is dispatch-bound.

    **This arm is in no published comparison, and it still earns its place.**
    It is the scenario-7 term of the ``attn_residual_norm`` span over 6+7+8,
    which is compared against the sum of the scenarios it replaces: without a
    titan number here the span has nothing to be measured against on the titan
    side, and the span's whole claim is that titan's add disappears into the
    next norm. It carries the cross-engine gate that proves the two engines
    compute one function, which is the evidence the declined ratio rests on.
    It keeps the partition's titan side complete, so ``qwen3/model.py:60``
    belongs to a scenario. And it stands as the measured cost of that add
    alone, if a later change ever stops Inductor from folding it.
    """
    def residual_add(
        residual: torch.Tensor, attn_out: torch.Tensor
    ) -> torch.Tensor:
        return residual + attn_out

    # fullgraph=True, the same treatment ``_compile_module`` applies. A graph
    # break becomes a build failure here instead of a partially eager arm.
    compiled = torch.compile(residual_add, fullgraph=True)

    def bind(
        attn_leaf: torch.Tensor, residual_leaf: torch.Tensor
    ) -> Callable[[], torch.Tensor]:
        def call() -> torch.Tensor:
            return compiled(residual_leaf, attn_leaf)

        return call

    _require_exact_residual_add(
        bind(inputs.attn_out, inputs.residual),
        inputs.attn_out,
        inputs.residual,
        TITAN_ARM,
    )
    return _residual_arm(
        name=TITAN_ARM,
        bind=bind,
        attn_native=inputs.attn_out,
        residual_native=inputs.residual,
        grad_native=inputs.grad_out,
        canonical=tuple(inputs.attn_out.shape),
    )


def _build_attn_residual_mcore(
    *,
    arm: str,
    profile: McoreProfile,
    fused: bool,
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: AttnResidualInputs,
) -> BuiltArm:
    """Megatron-core's ``self_attn_bda`` call site, eager, at one profile.

    The two mcore arms differ in one profile field and in nothing else, so
    they share this body. Splitting them into two hand-written builders would
    let the difference drift into something the profile does not describe.

    The callable comes off a real ``GPTModel`` rather than from a direct
    import of ``get_bias_dropout_add``. The spec decides it:
    ``gpt_layer_specs.py:335`` writes ``self_attn_bda=get_bias_dropout_add``
    in the non-MLA branch this build takes, and ``build_module`` returns a
    plain function unchanged (``spec_utils.py:86-87``). A direct import would
    keep timing that function after a rev changed the spec entry, which is the
    hazard ``mcore_profiles.py`` records for hand-written layer specs.

    **Reading it off a real model costs a real model.** ``build_model``
    allocates the whole network to hand back one function reference, two
    bools and a float: about 2.1 GiB of bf16 parameters at ``normal`` and
    about 21 GiB at ``huge``, and the correctness pass builds it twice in one
    interpreter, once per mcore arm. The builds are sequential and each is
    dropped before the next, so the peak is one model rather than two -- but
    at ``huge`` this is by far the most expensive thing in the scenario, and
    it buys spec fidelity rather than any measured quantity.

    The model is dropped as soon as the five values are captured, and the drop
    is what keeps ``memory_pass`` honest: ``max_memory_allocated`` is a
    maximum over time of the bytes currently allocated, so a resident model
    would be charged to the arm. What survives it is ``get_bias_dropout_add``
    (a module-level function), two bools, one float, and
    ``torch.enable_grad`` (a class, not an instance). None of them references
    the model, so this arm holds no parameters at all.
    """
    initialize_megatron_single_rank()

    from megatron.core.fusions.fused_bias_dropout import (
        bias_dropout_add_fused_train,
    )
    from megatron.core.transformer.transformer_layer import TransformerLayer

    from benchmarks.models.piper_qwen3.megatron_model import build_model

    model = build_model(
        seq_len=workload.seq_len,
        shape=shape,
        profile=profile,
    )
    layer = model.decoder.layers[MCORE_LAYER]
    # ``isinstance`` and not an exact type check, deliberately.
    # ``MoETransformerLayer`` (``transformer_layer.py:1509``) subclasses
    # ``TransformerLayer`` and overrides ``_forward_mlp*`` only, so it
    # inherits the very ``_forward_attention`` this scenario cuts and is a
    # correct layer to read the callable off. What the check refuses is a
    # class that does not carry that method at all.
    if not isinstance(layer, TransformerLayer):
        raise RuntimeError(
            f"{arm}: {MCORE_LAYER_PATH} is {type(layer).__name__}, which is "
            "not a TransformerLayer or a subclass of one; the call site this "
            "scenario times lives in TransformerLayer._forward_attention"
        )
    _require_mcore_layer_contract(layer, arm, fused)

    bda = getattr(layer, MCORE_BDA_ATTRIBUTE)
    training = layer.training
    fusion = layer.config.bias_dropout_fusion
    hidden_dropout = layer.hidden_dropout
    exec_handler = layer.bias_dropout_add_exec_handler
    notes = _require_bda_dispatch(
        arm=arm,
        fused=fused,
        resolved=bda(training, fusion),
        fused_train=bias_dropout_add_fused_train,
    )
    notes["profile"] = profile.name
    del layer, model
    gc.collect()
    torch.cuda.empty_cache()

    def bind(
        attn_leaf: torch.Tensor, residual_leaf: torch.Tensor
    ) -> Callable[[], torch.Tensor]:
        # The tuple is scenario 6's output, so it is built once here rather
        # than on every call. The bias slot is None because add_bias_linear is
        # False, which _require_mcore_layer_contract has already asserted.
        x_with_bias = (attn_leaf, None)

        def call() -> torch.Tensor:
            # Both the exec handler and the per-call resolution are part of
            # the call site (transformer_layer.py:683-686). The resolution is
            # where the unfused arm builds its closure, so hoisting it would
            # move half of the measured delta out of the measurement.
            with exec_handler():
                return bda(training, fusion)(
                    x_with_bias, residual_leaf, hidden_dropout
                )

        return call

    tokens = workload.batch * workload.seq_len
    # THD, the form our megatron driver runs: the driver packs each batch into
    # one sequence, and the layer receives a (t, b, h) tensor. A free view of a
    # contiguous tensor, taken here and never inside a timed closure.
    attn_native = inputs.attn_out.reshape(tokens, 1, shape.dim)
    residual_native = inputs.residual.reshape(tokens, 1, shape.dim)
    _require_exact_residual_add(
        bind(attn_native, residual_native), attn_native, residual_native, arm
    )
    return _residual_arm(
        name=arm,
        bind=bind,
        attn_native=attn_native,
        residual_native=residual_native,
        grad_native=inputs.grad_out.reshape(tokens, 1, shape.dim),
        canonical=tuple(inputs.attn_out.shape),
        notes=notes,
    )


def build_attn_residual_mcore_base(
    shape: PiperShape, workload: KernelWorkload, inputs: AttnResidualInputs
) -> BuiltArm:
    """megatron at its own best: ``bias_dropout_add_fused_train``.

    The base profile leaves ``bias_dropout_fusion`` True, which is what
    megatron's argparse layer gives a real run. It is the scenario anchor,
    which is why every cross-engine gate points at it rather than away from
    it.

    **This arm declares ``compiled=True``, and it is the only megatron arm in
    the registry that does.** ``bias_dropout_add_fused_train`` carries
    ``@jit_fuser`` (``fused_bias_dropout.py:69``), ``megatron/core/jit.py:21``
    binds ``jit_fuser`` to ``torch.compile`` on torch >= 2.2, and ``:33``
    applies that binding at import -- so the decorator is already
    ``torch.compile`` when ``fused_bias_dropout`` imports the name. The timed
    closure therefore calls a ``torch.compile`` wrapper directly, and that
    call is the arm's whole payload.

    The precedent it does not follow is ``cross_entropy``'s
    ``mcore/ce_native``, which declares ``compiled=False`` around
    ``@jit_fuser`` helpers. The difference is where the compile sits: there
    the arm's entry point is a plain method and the compiled regions are
    helpers nested inside it, so the arm is eager and its kernels are not.
    Here the entry point **is** the compiled function. Declaring this arm
    eager would put "eager against eager" in the manifest for a row whose
    entire delta is that compile.

    ``_require_bda_dispatch`` proves the declaration rather than trusting it:
    the build refuses to continue unless the resolved callable really is a
    ``torch.compile`` wrapper.
    """
    return _build_attn_residual_mcore(
        arm=MCORE_BASE_ARM,
        profile=BASE,
        fused=True,
        shape=shape,
        workload=workload,
        inputs=inputs,
    )


def build_attn_residual_mcore_no_bias_dropout_fusion(
    shape: PiperShape, workload: KernelWorkload, inputs: AttnResidualInputs
) -> BuiltArm:
    """megatron with the fusion off: ``bias_dropout_add_unfused``.

    ``get_bias_dropout_add`` builds a fresh closure on every call in this
    branch (``fused_bias_dropout.py:62-66``), and the arithmetic then
    dispatches eagerly. The device work is the same single bf16 add the base
    arm runs, so the row against the anchor is a host-dispatch comparison and
    must be captioned as one.
    """
    return _build_attn_residual_mcore(
        arm=MCORE_NO_FUSION_ARM,
        profile=NO_BIAS_DROPOUT_FUSION_PROFILE,
        fused=False,
        shape=shape,
        workload=workload,
        inputs=inputs,
    )
