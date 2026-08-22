"""Arm builders for the ``cross_entropy`` kernel scenario.

The loss, and only the loss. Both engines receive logits that already exist
and turn them into one scalar plus one logit gradient:

* TorchTitan: ``CrossEntropyLoss`` (``third_party/torchtitan/torchtitan/
  components/loss.py:344``), whose ``fn`` is ``cross_entropy_loss``
  (``:40-72``), ending in ``F.cross_entropy(pred.flatten(0, 1).float(),
  labels.flatten(0, 1), reduction="sum", ignore_index=-100)``.
* Megatron-core: ``compute_language_model_loss`` (``third_party/Megatron-LM/
  megatron/core/models/common/language_module/language_module.py:161``),
  called from ``GPTModel.forward`` at ``gpt_model.py:794``.

**The LM-head projection is scenario 15 and is not measured here.** Neither
arm multiplies anything by a weight: the shared inputs hold the logits
themselves. The existing ``lm_head`` scenario measures the projection and the
loss together, so **no number here is comparable to a number there**, and the
re-homed titan arms carry new names to say so -- ``lm_head/baseline`` is
``titan/full_logits``.

Layout charge, which is the thing this scenario had to decide
-----------------------------------------------------------

Megatron's method does three layout operations the titan loss does not:
``labels = labels.transpose(0, 1).contiguous()`` at ``language_module.py:172``
(a materializing ``[b, s] -> [s, b]`` int64 copy), ``torch.as_strided(labels,
labels.size(), (labels.size()[1], 1))`` at ``:176`` (a free re-view at this
configuration -- the tensor is already contiguous with exactly those strides,
so it emits no kernel), and ``loss = loss.transpose(0, 1).contiguous()`` at
``:205`` (a materializing ``[s, b] -> [b, s]`` fp32 copy of the per-token
loss). TorchTitan's loss flattens instead, and a flatten of a contiguous
tensor is a view.

**Both sides are charged the label preparation, and the scenario description
says so.** Each titan arm runs ``labels.transpose(0, 1).contiguous()`` inside
its timed closure and hands the loss a ``[b, s]`` view of the result, so the
loss's own ``flatten`` / ``.contiguous()`` then materializes a second copy.
The two engines therefore pay two small layout kernels each: megatron 32 KiB
of labels plus 16 KiB of loss, titan 32 KiB of labels twice, against 1.16 GiB
of logits at the default workload. **So this is a launch-count charge, not a
bandwidth charge** -- roughly two extra kernel dispatches per call on each
side -- and in a benchmark whose numbers are substantially host dispatch that
is the part worth equalizing. Charging megatron alone would have put a
megatron-only dispatch cost into the published ratio.

Two rules this scenario has to state, and neither is optional
-------------------------------------------------------------

**1. The cross-engine row measures version drift, not kernel quality.**
``mcore/base`` and ``titan/te_fused_ce`` run the *same implementation* from
two sources. ``cross_entropy_fusion_impl="te"`` routes megatron through
``te_parallel_cross_entropy`` (``megatron/core/extensions/
transformer_engine.py:3529-3542``), which calls the **installed**
``transformer_engine.pytorch.cross_entropy.parallel_cross_entropy`` -- TE
2.17.1 in this environment. ``titan/te_fused_ce`` calls our **vendored
snapshot** of that same TE Triton kernel, at
``benchmarks/models/piper_qwen3/components/lm_head/te_cross_entropy.py`` and
its two siblings.

The snapshot is not a copy. Installed TE 2.17.1 writes the gradient **into
the caller's logit buffer** and returns it (``triton/cross_entropy.py``:
``cross_entropy_kernel`` takes ``X_ptr`` and no gradient pointer, and
``cross_entropy_forward`` ends ``return loss, _input``), so the gradient is
stored in the input dtype -- bf16 here -- and the logits are destroyed. The
vendored file allocates ``grad_input = torch.empty_like(_input,
dtype=torch.float32)`` (``te_triton_cross_entropy.py:58``), passes
``grad_input_ptr`` / ``grad_input_stride`` to the kernel, and returns that
instead (``:116``). So the two differ in gradient precision *and* in peak
memory, and any ratio between them is a statement about one kernel plus our
wrapper. **The scenario therefore declines to publish that ratio**: the two
sides are not a like-for-like cut, and ``results.json`` has nowhere to put the
caption that would say so. The declined row is recorded beside
``comparisons`` in the registry declaration. Both arms' absolute numbers are
still published, so never let a reader take them for independent
implementations, and never report the quotient as "our TE beats megatron's
TE".

**Blocking prerequisite: no number from this scenario may be published
today, and the fix is not this module's to make.** ``ArmResult``
(``benchmarks/kernel/results/schema.py``) carries ``name``, ``modes``,
``peak_memory_gib``, ``burst_us_per_call``, ``status`` and
``status_reason`` -- no ``compiled`` and no ``eager_reason``.
``KernelScenarioResult`` carries no ``description``, and a comparison row is
``{"arm", "opponent", "mode"}`` plus statistics, with nowhere for a caption.
So plan rule 1 -- every cross-engine row prints the compile treatment of
both sides -- cannot be satisfied from ``results.json``, and it binds the
genuine cross-engine row ``titan/full_logits`` against ``mcore/base`` as
hard as it binds the row this scenario declines. Everything declared here
reaches the **manifest** only. The two existing homes worth considering
before new fields are added are ``KernelScenarioResult.methodology``, a
free-form dict the parent already assembles, and ``warnings``. That is
parent-side schema work, scheduled after a separate consolidation; this
module records the requirement and does not act on it.

``tests/test_lm_head_losses.py:25-65`` pins the three vendored files by
SHA-256 after normalizing their local import lines back to the upstream
spelling. Verified present.

**The snapshot's ancestry is verified, not assumed.** Diffing each vendored
file against the installed TransformerEngine 2.17.1 -- ``common/triton/
cross_entropy.py``, ``pytorch/triton/cross_entropy.py`` and ``pytorch/
cross_entropy.py``, using the same import rewrites the SHA-256 test applies --
returns the fp32 gradient buffer and nothing else of substance: the extra
``grad_input_ptr`` / ``grad_input_stride`` kernel arguments and the stores that
target them, ``torch.empty_like(_input, dtype=torch.float32)``, the renamed
``_input`` -> ``grad_input`` through forward and backward, and the docstring
sentences that described the old upcast. So ``titan/te_fused_ce`` is TE 2.17.1
with one change, and the row against ``mcore/base`` is that one change plus our
wrapper. Note what the SHA-256 test still does not do: it freezes our snapshot
against *our* drift, and it would not notice a TE upgrade underneath
``mcore/base``. Re-run the diff after any TE bump.

**2. Two arm names overclaim, and provenance corrects them here.**
``titan/te_fused_ce`` is the vendored snapshot, not installed TE 2.17.1 --
the name says "TE" and the arm is ours. And ``titan/full_logits`` is *our
benchmark baseline*, not TorchTitan's default: **all twelve** upstream qwen3
configs wrap the loss as ``ChunkedLossWrapper.Config(loss_fn=CrossEntropyLoss.
Config(global_vocab_size=...))``. Ten declare it literally
(``torchtitan/models/qwen3/config_registry.py:31`` and nine identical sites)
and the remaining two inherit it -- ``qwen3_debugmodel_moe_param_groups``
(``:60``) and ``qwen3_debugmodel_non_fused_qkv`` (``:290``) each derive from a
wrapped config and never reassign ``loss``. The class is
``components/loss.py:570``. That wrapper owns the LM head, which makes it a
scenario-15-plus-16 arm rather than a scenario-16 arm. Bare ``CrossEntropyLoss`` is the like-for-like cut against
``compute_language_model_loss``, and it is the cut we measure, but a reader
must not read it as "what TorchTitan does".

Why ``titan/piper_optimized_te_ce`` faces ``titan/te_fused_ce``
--------------------------------------------------------------

``piper_optimized_cross_entropy`` is a rework of the vendored snapshot, not
of anything titan ships: it keeps the online-softmax structure and changes
the gradient boundary. Comparing it against ``titan/full_logits`` or against
``mcore/base`` would credit it with the whole vendored-TE gain, which belongs
to TE. Its honest opponent is the snapshot it modifies, and the scenario
declares exactly that row.

The optimization, stated so the row can be read: the TE-family backward
rescales the whole saved gradient buffer by ``grad_output``
(``element_mul_kernel``), because the buffer was written before the caller's
normalization was known. Piper's kernel takes ``gradient_scale`` in forward
and applies it in fp32 *before* the single bf16 store
(``piper_optimized_cross_entropy.py:91-92``), so backward returns the saved
tensor untouched (``:171-175``). That removes a full-tensor read-modify-write
from every backward and costs one bf16 rounding rather than two -- which is
why this arm gates at the same tolerance as the fp32-buffer arms rather than
a looser one.

Modes: forward and forward_backward, no isolated backward
---------------------------------------------------------

Every arm declares ``("forward", "forward_backward")``. **Four of the six
cannot expose an isolated ``backward``**, because their backward mutates the
tensor their forward saved, so a retained graph re-run would fold the same
buffer again on every call:

* ``mcore/base`` and ``titan/te_fused_ce`` multiply the saved buffer in place
  (``element_mul_kernel``, reached from ``cross_entropy_backward``);
* ``mcore/ce_native`` and ``mcore/no_ce_fusion`` mutate the saved fp32 softmax
  in place. ``prepare_gradient_calculation_operands`` aliases it --
  ``grad_input = softmax`` (``tensor_parallel/cross_entropy.py:88-91``) -- and
  ``calculate_gradients`` then writes through the alias at ``:111``
  (``grad_2d[...] -= softmax_update``) and ``:114``
  (``grad_input.mul_(grad_output...)``). The fused path reaches the same two
  lines through its own ``@jit_fuser`` wrapper
  (``fusions/fused_cross_entropy.py:73-79``).

The other two **could** expose it and do not:
``titan/piper_optimized_te_ce`` returns the saved tensor untouched
(``piper_optimized_cross_entropy.py:166-175``) and ``titan/full_logits`` runs
``F.cross_entropy``, whose backward mutates nothing. **The mode is dropped
everywhere for comparability**: a scenario where two arms carry a third row
and four do not publishes a column most of the roster cannot fill, and the
two that could would be the only arms whose backward is measured without its
forward. ``attention_core``, ``attn_out_proj`` and ``final_norm`` drop the mode for
their own reasons; backward cost stays recoverable as the difference of the
two declared modes.

Which is also why ``forward`` alone is the weaker of the two rows here, and
the scenario description says so. The arms split the work differently across
the boundary: the three TE-family arms compute **and store the whole
gradient inside forward** (that is what the fp32 buffer is), while
``titan/full_logits``, ``mcore/ce_native`` and ``mcore/no_ce_fusion`` compute
a softmax or a log-softmax in forward and build the gradient in backward
(``fusions/fused_cross_entropy.py:119`` and
``tensor_parallel/cross_entropy.py:181`` save it; both ``backward`` methods
derive the gradient from it). So a ``forward`` row compares operations that
are not the same operation, and ``forward_backward`` is the sound one.

Host serialization each arm pays, declared rather than discovered
-----------------------------------------------------------------

These are per-call host costs that differ across the arms and that a
dispatch-bound number will contain:

* ``mcore/base`` and ``titan/te_fused_ce`` allocate ``torch.tensor(1.0,
  device=...)`` on **every** backward, for the "is grad_output one?" check.
  At this scenario's shapes ``grad_output`` is a ``[b, s]`` tensor, so
  ``torch.equal`` returns False on the shape alone without reading the
  device -- but the scalar is still built and copied host-to-device first.
* ``mcore/ce_native`` enqueues two ``all_reduce`` calls
  (``fusions/fused_cross_entropy.py:94,112``) and ``mcore/no_ce_fusion``
  three (``tensor_parallel/cross_entropy.py:130,146,150``). At world size 1
  they reduce nothing, but they are dispatched.
* ``titan/full_logits`` and ``titan/piper_optimized_te_ce`` pay none of these.

Why every arm clones its own logits
-----------------------------------

``benchmarks/kernel/engine/run.py:_prepare`` builds the shared inputs once
per pass and ``gate_outputs`` hands the same object to every arm in turn. One
arm -- ``mcore/base`` -- **destroys its input**, as described above, so
handing it the shared tensor would corrupt every arm built after it. Each arm
therefore clones a private leaf per mode, and ``correctness_outputs`` clones
a fresh one so the gate always sees pristine logits.

That leaves ``mcore/base`` timing repeated destructive forwards. **State the
consequence plainly rather than only the defence: the 30 warmup calls already
overwrite the buffer, so not one timed sample of this arm runs on the logits
the scenario declares.** Every sample runs on the fixed point the kernel
converges to -- a near-constant buffer around ``1/V`` with one entry near
``-1`` per row -- which no training step produces. The arm ``description``
says this, so a reader of the manifest meets it without reading this file.

The defence is that the kernel work per call is identical from the first call
to the last: both Triton kernels loop over the vocabulary with a fixed trip
count, they branch only on the label (``y == ignore_idx``, ``y < n_cols``),
which does not change, and the values converge toward ``1/V`` rather than
toward zero, so nothing becomes denormal -- under ``forward_backward`` the
extra ``2**-12`` scaling bottoms out around 1.6e-9, far above the bf16 minimum
normal.

**The one residual risk, which nothing here rules out:** a near-constant bf16
buffer is a *memory-traffic* pattern that the first call's ~N(0, 1) data is
not. Whether that changes achieved bandwidth on an H200 is unmeasured, and no
CPU-side argument can settle it. If this arm ever looks implausibly fast,
measure it against a freshly filled buffer before believing the number.

Every torchtitan, megatron and TransformerEngine import is deferred into a
builder body: one arm per process is the point, and a process that measures
one arm must not import the other's stack.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any, Callable

import torch

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    MCORE_BLANK_MLP,
    _randn,
    initialize_megatron_single_rank,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE, McoreProfile, derive
from benchmarks.models.piper_qwen3.shape import PiperShape


# The six arm names, spelled as plan section C.2 spells them. The slash says
# which engine an arm is and which profile of that engine; it reaches a
# filename through ``benchmarks/kernel/runner.py``'s ``fragment_path``, which
# routes it through ``benchmarks.kernel.schema.fragment_stem``. **A tree
# without ``fragment_stem`` cannot run this scenario**: every timing worker
# would try to write into a ``timing__mcore`` or ``timing__titan`` directory
# nobody creates, and the arm would land as ``failed``.
MCORE_BASE_ARM = "mcore/base"
MCORE_CE_NATIVE_ARM = "mcore/ce_native"
MCORE_NO_CE_FUSION_ARM = "mcore/no_ce_fusion"
TITAN_FULL_LOGITS_ARM = "titan/full_logits"
TITAN_TE_FUSED_CE_ARM = "titan/te_fused_ce"
TITAN_PIPER_ARM = "titan/piper_optimized_te_ce"


# Rows of the fp64 reference computed at once. 256 rows of 151936 fp64
# elements is 296.75 MiB per temporary (311.16 MB), and the loop holds at most
# two of them plus the fp32 result. See ``cross_entropy_reference``.
REFERENCE_ROW_CHUNK = 256

# The attributes ``compute_language_model_loss`` reads off ``self``, taken
# from ``language_module.py:172-205``: ``self.config`` for both branch
# conditions, ``self.pg_collection.tp`` for the fused paths and
# ``self.tp_group`` for the unfused one. ``_release_model_parameters`` drops
# every parameter-holding submodule and then asserts these survive, so the
# release cannot silently break the arm.
MCORE_LOSS_ATTRIBUTES = ("config", "pg_collection", "tp_group")

# The submodules the loss method never touches. Dropping them is what keeps
# ``peak_memory_gib`` a statement about the loss rather than about a resident
# model -- ``memory_pass`` reads ``torch.cuda.max_memory_allocated``,
# which counts everything resident, and memory is a headline metric here
# because the arms differ in exactly that (an fp32 gradient buffer, an fp32
# softmax kept for backward, or neither).
MCORE_RELEASED_SUBMODULES = (
    "embedding",
    "decoder",
    "output_layer",
    "rotary_pos_emb",
    # Absent at every profile this scenario declares -- ``GPTModel`` builds it
    # only under ``mtp_process`` (``gpt_model.py:236-242``) -- and listed
    # anyway. A future MTP profile would otherwise leave a second set of
    # parameters resident, and ``peak_memory_gib`` is this scenario's
    # secondary metric. Naming a submodule that is not there costs nothing:
    # the loop checks ``model._modules`` first.
    "mtp",
)


# The two megatron variants, as deltas rather than as second copies of 28
# flags. Declared here rather than added to ``MCORE_PROFILES`` because they
# are this scenario's roster: the e2e megatron arm and the other cross-engine
# scenarios all run ``base``, and a registry entry would suggest otherwise.
#
# **Which of the three is "megatron as NVIDIA ships it" is no_ce_fusion, not
# ce_native.** Megatron's only default in this tree is
# ``cross_entropy_loss_fusion: bool = False``
# (``model_parallel_config.py:320``), so the shipped path is the *unfused*
# one; ``cross_entropy_fusion_impl`` defaults to ``'native'`` (``:325``) but
# is dead while fusion is off. And this is not the argparse-versus-dataclass
# trap the base profile's comments warn about: on this rev
# ``megatron/training/arguments.py`` declares **no** ``add_argument`` for
# either flag -- the only mentions are the assert at ``:1631-1634`` -- so
# there is no argparse default overriding the dataclass one.
#
# CLAUDE.md calls ``"native"`` "megatron as NVIDIA ships it". That reading is
# defensible in its original context, which is the narrower choice between
# ``te`` and ``native`` **given that fusion is on** -- the e2e megatron arm
# has fusion on and picks between the two. It is wrong here, because this
# scenario puts all three paths in front of the same reader, and the label
# would point at the middle one.
CE_NATIVE_PROFILE: McoreProfile = derive(
    BASE,
    name="ce_native",
    description=(
        "megatron's fused non-TE cross-entropy: its own @jit_fuser "
        "implementation, which is what fusion selects once TE is declined. "
        "Not megatron's shipped default -- that is no_ce_fusion -- but it is "
        "the only fused path megatron's own training entrypoint permits, "
        "because arguments.py refuses fusion plus 'te'"
    ),
    config_overrides={"cross_entropy_fusion_impl": "native"},
)

NO_CE_FUSION_PROFILE: McoreProfile = derive(
    BASE,
    name="no_ce_fusion",
    description=(
        "megatron as NVIDIA ships it: no cross-entropy fusion at all, which "
        "is the tree's only default (cross_entropy_loss_fusion=False), so "
        "this is the plain eager vocab-parallel cross-entropy a stock user "
        "runs -- and the floor the two fused paths are measured against"
    ),
    config_overrides={"cross_entropy_loss_fusion": False},
)


@dataclass
class CrossEntropyInputs:
    """One batch of logits and its labels, in the canonical titan layout.

    ``logits`` is ``[B, L, V]`` bf16 and contiguous. The mcore arms
    materialize their own ``[L, B, V]`` copy at build time, because megatron
    runs the model in SBHD and a transpose inside a timed closure would
    charge one engine for the harness's choice of storage order.

    ``labels`` stays ``[B, L]`` on both sides, which is exactly the shape
    ``compute_language_model_loss`` documents and then transposes
    (``language_module.py:164,172``). Every label is valid: there is no
    ``-100`` here, so ``valid_tokens`` is ``B * L`` and both engines
    normalize by the same number.
    """

    logits: torch.Tensor  # (B, L, V) bf16, contiguous
    labels: torch.Tensor  # (B, L) int64
    valid_tokens: float


def cross_entropy_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> CrossEntropyInputs:
    """The shared logits and labels every arm consumes.

    The logits are drawn at unit scale, which is where the projection this
    scenario excludes actually puts them: ``lm_head_inputs`` draws a hidden
    state at unit scale and a weight at ``1/sqrt(dim)``, so each logit is a
    sum of ``dim`` terms with variance ``1/dim`` and lands at unit variance
    too. A much wider draw would make the softmax one-hot and a much narrower
    one would make it uniform; neither is what the loss sees in a run.

    Labels are uniform over the vocabulary, so the expected loss is near
    ``log(V)`` and no row is degenerate.
    """
    batch, seq = workload.batch, workload.seq_len
    return CrossEntropyInputs(
        logits=_randn((batch, seq, shape.vocab_size), device, generator),
        labels=torch.randint(
            shape.vocab_size,
            (batch, seq),
            device=device,
            generator=generator,
            dtype=torch.int64,
        ),
        valid_tokens=float(batch * seq),
    )


def cross_entropy_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: CrossEntropyInputs
) -> dict[str, torch.Tensor]:
    """fp64 loss and logit gradient, computed a block of rows at a time.

    **What this costs, and why it is chunked.** The obvious implementation --
    ``F.cross_entropy(inputs.logits.double(), ...)`` -- allocates an fp64 copy
    of the whole logit tensor (4.64 GiB at the default workload), an fp64
    log-softmax of the same size, and an fp64 gradient, so it asks for about
    14 GiB to check a tensor the arms hold in 1.16. The ``attention_core``
    scenario's per-(row, kv group) reference exists for the same reason.

    Here the loop holds ``REFERENCE_ROW_CHUNK`` rows at a time: 256 rows of
    151936 fp64 elements is 296.75 MiB (311.16 MB -- the earlier text mixed
    the two units), and at most two such temporaries are live, so the
    transient cost is about 0.6 GiB. The returned gradient is
    stored in **fp32** -- 2.32 GiB, held for the whole correctness pass --
    which loses nothing, because ``benchmarks/kernel/engine/correctness.py``
    casts both sides to fp32 before subtracting them anyway, and every arm's
    gradient is bf16.

    The truth is computed from the bf16 values the arms actually hold, not
    from an unrounded draw: otherwise every arm would be charged for the
    input cast rather than for its own arithmetic.

    Gated on ``max_rel_l2`` only. Cross-entropy is a reduction over the
    vocabulary, and CLAUDE.md's rule against max/ULP metrics on reductions
    applies: cancellation drives individual gradient entries toward zero, and
    dividing a negligible absolute error by that magnitude reports a huge
    relative error for a numerically perfect kernel.
    """
    logits = inputs.logits
    batch, seq, vocab = logits.shape
    rows = batch * seq
    flat = logits.reshape(rows, vocab)
    targets = inputs.labels.reshape(rows)
    grad = torch.empty(
        (rows, vocab), device=logits.device, dtype=torch.float32
    )
    loss_sum = torch.zeros((), device=logits.device, dtype=torch.float64)
    scale = 1.0 / inputs.valid_tokens
    for start in range(0, rows, REFERENCE_ROW_CHUNK):
        stop = min(start + REFERENCE_ROW_CHUNK, rows)
        # ``.double()`` already made a private copy, so every step below may
        # work in place on it.
        block = flat[start:stop].double()
        index = targets[start:stop].unsqueeze(1)
        block -= block.amax(dim=-1, keepdim=True)
        log_sum_exp = block.exp().sum(dim=-1, keepdim=True).log()
        picked = block.gather(1, index)
        loss_sum += (log_sum_exp - picked).sum()
        block -= log_sum_exp
        block.exp_()
        block.scatter_add_(
            1, index, torch.full_like(index, -1, dtype=block.dtype)
        )
        block *= scale
        grad[start:stop] = block
    return {
        "loss": (loss_sum * scale).float(),
        "logits_grad": grad.view(batch, seq, vocab),
    }


def _cross_entropy_arm(
    name: str,
    make_leaf: Callable[[], torch.Tensor],
    loss_call: Callable[[torch.Tensor], torch.Tensor],
    to_canonical: Callable[[torch.Tensor], torch.Tensor],
) -> BuiltArm:
    """Forward and forward+backward over one loss, on private leaves.

    Each mode owns a leaf, so one mode's autograd state never reaches
    another's, and ``correctness_outputs`` builds a third: ``mcore/base``
    overwrites its input, so the gate must start from pristine logits
    whatever the timed closures did to theirs.

    Backward runs ``torch.autograd.grad`` rather than ``.backward()``, and
    that is a fidelity choice rather than a convenience. In a real step the
    logits are the *output of the LM head*, not a leaf, so the gradient is
    produced and consumed by the projection's backward and ``AccumulateGrad``
    never runs on them. ``.backward()`` here would add an accumulation whose
    steal-or-clone decision depends on refcounts, which could put a 1.16 GiB
    copy on some arms and not others -- a harness artifact running in the
    same direction as the effect under test.

    ``to_canonical`` maps the arm's gradient back to ``[B, L, V]`` so the two
    engines' outputs are comparable; it is the identity on the titan side.
    """
    forward_leaf = make_leaf()
    round_trip_leaf = make_leaf()

    def forward():
        return loss_call(forward_leaf)

    def forward_backward() -> None:
        torch.autograd.grad(loss_call(round_trip_leaf), round_trip_leaf)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        check_leaf = make_leaf()
        loss = loss_call(check_leaf)
        (gradient,) = torch.autograd.grad(loss, check_leaf)
        return {
            "loss": loss.detach().float(),
            "logits_grad": to_canonical(gradient),
        }

    return BuiltArm(
        name=name,
        calls={"forward": forward, "forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
    )


# ---------------------------------------------------------------------------
# TorchTitan arms
# ---------------------------------------------------------------------------


def _loss_compile():
    """The production ``CompileConfig(components=["loss"])``, built on demand.

    ``CompileConfig.components`` defaults to ``["model", "loss"]``
    (``third_party/torchtitan/torchtitan/config/configs.py:263-272``), so a
    real run compiles the loss; ``BaseLoss._maybe_compile``
    (``components/loss.py:318``) is what applies it. All three titan arms get
    it, and ``KernelArm.compiled`` records ``True`` for them. A module-scope
    constant would need ``torchtitan.config`` at module scope, which the
    deferred-import rule forbids; the existing ``lm_head`` module carries the
    identical helper for the identical reason.
    """
    from torchtitan.config import CompileConfig

    return CompileConfig(enable=True, components=["loss"])


def _prepared_labels(labels: torch.Tensor) -> torch.Tensor:
    """Megatron's label preparation, charged to the titan arms as well.

    ``language_module.py:172`` materializes ``[b, s] -> [s, b]`` before the
    CE kernel and ``:205`` transposes the per-token loss back. TorchTitan's
    loss does neither. Rather than charge one engine for its own layout, this
    runs the same materializing transpose on the titan side and hands the
    loss a ``[b, s]`` **view** of the result -- whose last stride is ``b``,
    not 1, so the loss's own ``flatten`` (or, for the TE-family arms, the
    ``target.stride(-1) != 1`` guard) materializes a second small copy. Two
    layout kernels per call on each engine, 32 KiB each, against 1.16 GiB of
    logits. See the module docstring: this is a launch-count charge.

    Runs eagerly, outside the compiled ``fn``, exactly as megatron's does --
    ``compute_language_model_loss`` is a plain method with no compile on it.
    """
    return labels.transpose(0, 1).contiguous().transpose(0, 1)


def _titan_arm(name: str, inputs: CrossEntropyInputs, loss_object) -> BuiltArm:
    """One titan loss object, wired into the shared arm shape."""

    def loss_call(logits: torch.Tensor) -> torch.Tensor:
        loss, _ = loss_object(
            logits, _prepared_labels(inputs.labels), inputs.valid_tokens
        )
        return loss

    return _cross_entropy_arm(
        name,
        lambda: inputs.logits.clone().requires_grad_(),
        loss_call,
        lambda gradient: gradient,
    )


def build_cross_entropy_titan_full_logits(
    shape: PiperShape, workload: KernelWorkload, inputs: CrossEntropyInputs
) -> BuiltArm:
    """TorchTitan's ``CrossEntropyLoss`` over materialized logits, compiled.

    This is the like-for-like cut against ``compute_language_model_loss``:
    logits in, scalar out, no LM head. It is **not** what an upstream qwen3
    config builds -- those wrap it in ``ChunkedLossWrapper``, which owns the
    head and therefore spans scenarios 15 and 16. See the module docstring.

    ``cross_entropy_loss`` upcasts the whole ``[tokens, V]`` tensor to fp32
    before ``F.cross_entropy`` (``components/loss.py:69-72``), which is why
    this arm's peak memory sits well above the fused ones.
    """
    from torchtitan.components.loss import CrossEntropyLoss

    loss_object = CrossEntropyLoss.Config(
        global_vocab_size=shape.vocab_size
    ).build(compile_config=_loss_compile())
    return _titan_arm(TITAN_FULL_LOGITS_ARM, inputs, loss_object)


def build_cross_entropy_titan_te_fused_ce(
    shape: PiperShape, workload: KernelWorkload, inputs: CrossEntropyInputs
) -> BuiltArm:
    """The **vendored** TE Triton cross-entropy, compiled.

    ``TECrossEntropyLoss._loss_sum`` calls
    ``benchmarks/models/piper_qwen3/components/lm_head/te_cross_entropy.py``'s
    ``parallel_cross_entropy``, our frozen snapshot -- not the installed
    TransformerEngine 2.17.1 that ``mcore/base`` reaches. The arm name says
    "TE" and the code is ours; the module docstring gives the divergence.
    """
    from benchmarks.models.piper_qwen3.components.lm_head.losses import (
        TECrossEntropyLoss,
    )

    loss_object = TECrossEntropyLoss.Config().build(
        compile_config=_loss_compile()
    )
    return _titan_arm(TITAN_TE_FUSED_CE_ARM, inputs, loss_object)


def build_cross_entropy_titan_piper_optimized_te_ce(
    shape: PiperShape, workload: KernelWorkload, inputs: CrossEntropyInputs
) -> BuiltArm:
    """Piper's rework of the vendored snapshot, compiled.

    Its opponent is ``titan/te_fused_ce`` and not the scenario anchor, because
    it modifies that snapshot and nothing else. ``PiperOptimizedCrossEntropy
    Loss.__call__`` normalizes inside the kernel, so the loss it returns is
    already the mean and its backward returns the saved gradient untouched.
    """
    from benchmarks.models.piper_qwen3.components.lm_head.losses import (
        PiperOptimizedCrossEntropyLoss,
    )

    loss_object = PiperOptimizedCrossEntropyLoss.Config().build(
        compile_config=_loss_compile()
    )
    return _titan_arm(TITAN_PIPER_ARM, inputs, loss_object)


# ---------------------------------------------------------------------------
# Megatron-core arms
# ---------------------------------------------------------------------------


def _assert_profile_took(profile: McoreProfile, config: Any) -> None:
    """Refuse to time a variant whose flags did not reach the built config.

    Both directions are failures, which is what makes this scenario's three
    megatron arms distinguishable at all: a flag declared on that came out
    off is the dataclass-default handicap
    (``benchmarks/models/piper_qwen3/mcore_profiles.py`` records what that
    once cost), and a flag declared off that came out on publishes the base
    implementation under the variant's name. No correctness gate can see
    either, because all three CE paths are numerically right.

    Same shape as the guard ``benchmarks/e2e/megatron/train.py:239-248`` runs
    for the e2e arm, reading the same ``declared_mismatches``.
    """
    from benchmarks.models.piper_qwen3.mcore_profiles import (
        FUSION_FIELDS,
        declared_mismatches,
    )

    built = {name: getattr(config, name, None) for name in FUSION_FIELDS}
    wrong = declared_mismatches(profile, built)
    if wrong:
        raise RuntimeError(
            f"megatron profile {profile.name!r} did not take: "
            + "; ".join(wrong)
            + " -- see benchmarks/models/piper_qwen3/mcore_profiles.py"
        )


def _assert_te_cross_entropy_available(profile: McoreProfile) -> None:
    """Refuse the TE profile when megatron could not import TE's CE.

    ``language_module.py:196`` raises "Trying to use a TE block when it's not
    present" *inside* the loss, which would surface from within a timed
    closure after the model was built. Megatron sets the symbol to ``None``
    in a bare ``except`` at ``language_module.py:13-16``, so any TE import
    failure lands here silently.
    """
    from megatron.core.models.common.language_module import language_module

    if language_module.te_parallel_cross_entropy is None:
        raise RuntimeError(
            f"profile {profile.name!r} asks for cross_entropy_fusion_impl="
            "'te', but megatron imported no te_parallel_cross_entropy; this "
            "arm would raise inside its own timed closure"
        )


def _release_model_parameters(model) -> None:
    """Drop every submodule the loss method never reads.

    ``compute_language_model_loss`` reads ``self.config``,
    ``self.pg_collection`` and ``self.tp_group`` and nothing else
    (``language_module.py:172-205``), so the embedding, the decoder and the
    output layer are dead weight once the bound method is in hand -- but they
    are 0.67 GiB of *resident* weight at ``1b`` and 7.80 GiB at ``huge``
    (``MCORE_BLANK_MLP`` leaves the mlp part out), and ``memory_pass``
    reports ``torch.cuda.max_memory_allocated``, which counts them. Peak
    memory is a headline metric for this scenario precisely because the arms
    differ in what the loss keeps alive, so a resident model would drown the
    thing being measured.

    The bound method holds ``self``, so the model object itself cannot be
    dropped the way ``final_norm`` drops its ``GPTModel``. Assigning ``None``
    over a registered submodule is the supported ``nn.Module`` spelling, and
    the guard below re-proves the three attributes the method needs.
    """
    for name in MCORE_RELEASED_SUBMODULES:
        if name in model._modules:
            setattr(model, name, None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    for attribute in MCORE_LOSS_ATTRIBUTES:
        if getattr(model, attribute, None) is None:
            raise RuntimeError(
                f"releasing the model parameters removed {attribute!r}, which "
                "compute_language_model_loss reads; the arm would raise "
                "inside its own timed closure"
            )


def _mcore_arm(
    name: str,
    profile: McoreProfile,
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: CrossEntropyInputs,
) -> BuiltArm:
    """One megatron profile's ``compute_language_model_loss``, eager.

    The method comes off a real ``GPTModel`` built by
    ``benchmarks/models/piper_qwen3/megatron_model.py``, which is the builder
    every other cross-engine scenario uses and the one whose layer spec
    megatron derives. Constructing a bare ``LanguageModule`` from the same
    ``TransformerConfig`` would reach an identical bound method far more
    cheaply -- the method reads only the three attributes above, and
    ``GPTModel.__init__`` sets them through the same
    ``LanguageModule.__init__`` path -- and it was rejected anyway: it would
    make this the one cross-engine scenario whose megatron side is assembled
    here rather than by ``build_model``, and a megatron bump that moved the
    method or gave it a fourth attribute would then be a silent difference
    rather than a loud one. The cost is wall clock in fifteen timing workers,
    not fidelity.

    The reduction is ``.sum() / valid_tokens``, which is byte for byte what
    the megatron driver's ``loss_func`` does
    (``benchmarks/e2e/megatron/train.py:288-292``: ``output_tensor.sum() /
    output_tensor.numel()``, and every token here is valid). Without it the
    arm would return a ``[b, s]`` tensor where the titan arms return a
    scalar, and the two sides would not be gated on the same quantity.

    Eager, with no ``torch.compile`` anywhere: megatron compiles no whole
    layer, and ``compute_language_model_loss`` is a plain method. The
    ``ce_native`` profile is the partial exception, and its ``eager_reason``
    in the registry says so.
    """
    from benchmarks.models.piper_qwen3.megatron_model import build_model

    initialize_megatron_single_rank()
    if profile.config_overrides.get("cross_entropy_loss_fusion") and (
        profile.config_overrides.get("cross_entropy_fusion_impl") == "te"
    ):
        _assert_te_cross_entropy_available(profile)
    model = build_model(
        seq_len=workload.seq_len,
        shape=shape,
        profile=profile,
        blank_parts=MCORE_BLANK_MLP,
    )
    _assert_profile_took(profile, model.config)
    compute_loss = model.compute_language_model_loss
    _release_model_parameters(model)

    labels = inputs.labels
    scale = 1.0 / inputs.valid_tokens

    def loss_call(logits: torch.Tensor) -> torch.Tensor:
        return compute_loss(labels, logits).sum() * scale

    return _cross_entropy_arm(
        name,
        # Megatron runs SBHD, so its logits are [L, B, V]. Materialized once
        # at build time: a transpose inside the timed closure would charge
        # megatron for the harness's storage order, which plan section B.3
        # forbids.
        lambda: inputs.logits.transpose(0, 1).contiguous().requires_grad_(),
        loss_call,
        lambda gradient: gradient.transpose(0, 1),
    )


def build_cross_entropy_mcore_base(
    shape: PiperShape, workload: KernelWorkload, inputs: CrossEntropyInputs
) -> BuiltArm:
    """Megatron's fastest available loss path: installed TE's Triton CE.

    ``cross_entropy_fusion_impl="te"`` routes through
    ``te_parallel_cross_entropy`` into **installed TransformerEngine**. Report
    every number from this arm as "megatron with its fastest available CE",
    never as megatron as shipped: megatron's own training entrypoint refuses
    this combination (``megatron/training/arguments.py:1631-1634``, "known
    stability issues") while the core config only warns and we build the
    config directly. The shipped path is ``mcore/no_ce_fusion``.

    **This arm destroys its input logits**, and the 30 warmup calls mean no
    timed sample runs on the declared logits at all. The arm ``description``
    in the registry says so, because that is the text the manifest records;
    the module docstring holds why it is still sound to time and what the one
    residual risk is.
    """
    return _mcore_arm(MCORE_BASE_ARM, BASE, shape, workload, inputs)


def build_cross_entropy_mcore_ce_native(
    shape: PiperShape, workload: KernelWorkload, inputs: CrossEntropyInputs
) -> BuiltArm:
    """Megatron's fused non-TE cross-entropy.

    This is what fusion selects once TE is declined, and it is **not** the CE
    a stock user gets -- ``mcore/no_ce_fusion`` is, because
    ``cross_entropy_loss_fusion`` defaults to ``False``. It is, however, the
    only fused path megatron's own entrypoint permits.

    ``fused_vocab_parallel_cross_entropy`` upcasts the whole ``[tokens, V]``
    tensor to fp32, makes several full-tensor traversals, and keeps that fp32
    softmax resident for backward
    (``fusions/fused_cross_entropy.py:93-119``). It also casts the gradient
    to bf16 unconditionally (``:82``), regardless of the model dtype.
    """
    return _mcore_arm(
        MCORE_CE_NATIVE_ARM, CE_NATIVE_PROFILE, shape, workload, inputs
    )


def build_cross_entropy_mcore_no_ce_fusion(
    shape: PiperShape, workload: KernelWorkload, inputs: CrossEntropyInputs
) -> BuiltArm:
    """Megatron as NVIDIA ships it: the cross-entropy fusion off.

    ``cross_entropy_loss_fusion=False`` is megatron's only default in this
    tree (``model_parallel_config.py:320``, and no ``add_argument`` overrides
    it), so this arm is the path a stock ``pretrain_gpt.py`` user runs. It
    reaches ``tensor_parallel.vocab_parallel_cross_entropy``, the same
    arithmetic as ``ce_native`` with no ``@jit_fuser`` on any of it and one
    more ``all_reduce``, and it is the within-engine floor the two fused paths
    are measured against.
    """
    return _mcore_arm(
        MCORE_NO_CE_FUSION_ARM, NO_CE_FUSION_PROFILE, shape, workload, inputs
    )
