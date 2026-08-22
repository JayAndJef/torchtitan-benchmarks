"""Arm builders for the ``moe_residual`` kernel scenario.

The residual add after the MoE block, on both engines:

* TorchTitan: the ``x + ...`` of ``x = x + self.moe(self.ffn_norm(x))`` in
  ``Qwen3TransformerBlock.forward``
  (``third_party/torchtitan/torchtitan/models/qwen3/model.py:63``).
* Megatron-core: ``mlp_bda(...)``
  (``third_party/Megatron-LM/megatron/core/transformer/transformer_layer.py:
  980``, inside ``_forward_post_mlp``, which ``_forward_mlp`` calls at
  ``:943``).

**This scenario publishes no cross-engine ratio, and no span.** Both
decisions are recorded in the registry's ``comparisons`` tuple and in the
scenario description; the evidence for each is below, because a reader who
meets the arms first has to know why the obvious row is missing.

**Why there is no cross-engine row: both engines compute the same add, and
they differ only in fusion scope.** The profile sets ``hidden_dropout=0.0``,
``attention_dropout=0.0`` and ``add_bias_linear=False``
(``benchmarks/models/piper_qwen3/mcore_profiles.py``), so megatron's
``_bias_dropout_add_func`` takes the ``bias is None`` branch
(``fusions/fused_bias_dropout.py:53-59``) and ``F.dropout(x, p=0.0,
training=True)`` returns its own input **object**, which
``tests/test_kernel_moe_residual.py`` pins as an identity rather than as a
closeness. What is left is ``out = residual + out``, which is titan's ``x +
self.moe(...)`` written in the same operand order. ``bias_dropout_add`` is
only the name of the kernel.

The real difference is **scope**. End to end, titan's add is one node inside
a whole-block ``torch.compile`` region, so Inductor is free to fuse it into a
neighbouring pointwise kernel; megatron's
``bias_dropout_add_fused_train`` is ``@jit_fuser`` decorated
(``fused_bias_dropout.py:69-73``) and ``megatron/core/jit.py`` rebinds
``jit_fuser`` from ``torch.jit.script`` (``:7``) to ``torch.compile``
(``:16-24``) on torch >= 2.2, so it is a compiled region of
its own with the surrounding graph outside it. Isolating the scenario gives
titan megatron's scope and takes that freedom away. A ratio measured under
that treatment describes the harness rather than the engines, so the scenario
declines to publish it.

**Nothing here has measured whether Inductor really fuses titan's node**, and
the decision does not need it. The scope difference is structural: the fork
compiles whole blocks (``distributed/compile.py``) and megatron compiles this
function alone. Whether the fusion then happens changes how large the
distortion is, not whether it exists.

**Why there is no span either, and this reason is specific to scenario 13.**
Titan's downstream fusion partner for this residual is the *next block's*
attention-input norm. On megatron that norm is already fused inside
``linear_qkv`` -- the forced coarsening scenario 2 (``qkv_prep``) records --
so there is no cut at which both engines have done the same work. The
statement is about the cross-engine case only: megatron does offer a
within-engine fusion across scenarios 8..13, which
``mcore/fused_residual_rmsnorm`` measures on the ``ffn_norm_to_moe_residual``
span. This scenario is that span's far end, because
``fused_residual_rmsnorm`` is backward-only and joins ``pre_mlp_layernorm``
to ``mlp_bda``, not to ``self_attn_bda``: ``pre_mlp_layernorm`` carries
``has_residual=True`` (``gpt_layer_specs.py:336``),
``transformer_layer.py:829-836`` unpacks its ``(output, residual)`` tuple,
and ``:943`` hands that residual to ``_forward_post_mlp``. Spans are declared
elsewhere; this module declares none.

**What the scenario does publish: megatron's ``bias_dropout_fusion`` on/off
delta.** That is a genuine within-engine question -- compiled
``bias_dropout_add_fused_train`` against eager ``bias_dropout_add_unfused``
-- and it is the one row the registry declares.

**What the titan arm is for, given it is in no comparison.** Three things,
and none of them is a ratio.

1. **It keeps the partition a partition.** Scenarios 1..16 are declared to
   cover every call in titan's block forward and backward. Titan's ``x +
   self.moe(...)`` belongs to this scenario or to none, and "to none" would
   make the coverage claim false.
2. **It is the evidence for the no-row decision, not a casualty of it.** The
   cross-engine ``CorrectnessCheck`` on this arm compares titan's output and
   both input gradients against ``mcore/base``. That check is what turns
   "both engines compute ``residual + x`` identically" from an assertion in
   this docstring into a measured fact on the box that runs it. Deleting the
   arm would delete the only test of the premise the decision rests on.
3. **Its own number bounds what titan's fusion scope is worth here.**
   Isolated, titan's add is a standalone Inductor pointwise kernel. Inside
   the block graph it is a node Inductor may fuse into a neighbour. So the
   arm's per-call time is an upper bound on what this one node can cost
   titan end to end -- read it that way, and never as titan's share of a
   step.

Read no ranking into the titan row, and never sum this scenario across
engines.

**The published row is expected to be a dispatch comparison, and
``copy_floor`` is what can say so.** The operation moves three traversals of
``batch * seq_len * dim`` bf16 elements and performs one add per element, so
it is memory-bound by construction and carries only tens of microseconds of
device work. The two mcore arms differ in their *dispatch* paths -- guard
evaluation against closure allocation plus eager op dispatch -- and neither
difference is device work. So the fourth arm is a bandwidth floor, exactly as
in ``rope`` and ``ffn_norm``: without it a reader cannot tell a fusion result
from a dispatch result. ``bytes_moved`` is recorded alongside, so the merge
prints both the GB/s column and the ``x_floor`` column.
The GB/s column is a bandwidth statement in ``forward`` alone -- the merge
divides ``bytes_moved`` by every mode's median, and ``rope`` and ``ffn_norm``
record the same property.

**There is no isolated ``backward`` mode, and it would measure nothing.** The
backward of ``residual + x`` routes the incoming gradient to both operands
and launches no kernel on either engine, so an isolated backward would time
the autograd engine rather than an implementation. ``forward_backward`` is
still declared, because the fused arm's compiled region carries autograd
bookkeeping the eager arm does not -- but read ``forward_backward`` minus
``forward`` here as host cost, not as device work.

**No layout conversion is charged to either engine, and none is needed.** An
elementwise add treats every leading dimension as a row index. Megatron's
tensors reach ``mlp_bda`` in the THD form our driver produces, ``[B*L, 1,
D]`` (``benchmarks/e2e/megatron/train.py`` builds ``PackedSeqParams`` with
``qkv_format="thd"``), and titan's are ``[B, L, D]``; both hold the same
``B * L * D`` contiguous elements. The mcore arms take their view at build
time and canonicalize back to ``[B, L, D]`` for the gates only.

The MoE flatten/unflatten glue is outside this cut on both engines and is
reimplemented nowhere here. **In ``TransformerLayer``, which is the class
this build instantiates**, ``_maybe_reflatten_from_moe`` runs at
``transformer_layer.py:920-922``, *before* ``_forward_post_mlp`` at ``:943``,
so ``mlp_bda`` already sees the reflattened tensor. The order is the opposite
in ``MoETransformerLayer``, which reflattens at ``:1780``, *after* the bda:
that class is not built here (``megatron_model.py`` swaps it in only under
``cuda_graph_impl == "local"``, which no kernel builder requests), so the
statement above is about this build and not about megatron in general.

The glue is inert in this harness in either class, because
``_maybe_unflatten_for_moe`` returns early unless
``packed_seq_params.tokens_per_sample`` is set (``:775-780``) and no code in
this repository sets that field.

Every megatron import is deferred into the builder that needs it, which is
the rule across ``operations/``. ``benchmarks.models.piper_qwen3.
mcore_profiles`` is the one module-scope exception, as in ``ffn_norm`` and
``cross_entropy``: it is torch-free parent-side data, and both profiles the
two mcore arms take are read from it.
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
    _randn_like,
    _require_grads,
    _reset_grads,
    initialize_megatron_single_rank,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import (
    BASE,
    NO_BIAS_DROPOUT_FUSION,
    McoreProfile,
)
from benchmarks.models.piper_qwen3.shape import PiperShape


# The four arm names, spelled as plan section C.2 spells the three
# implementations. The slash says which engine an arm is and which profile of
# that engine; ``schema.fragment_stem`` is what keeps it out of a fragment
# filename. ``copy_floor`` carries no engine because it is not an
# implementation of anything, which is the same spelling ``rope`` and
# ``ffn_norm`` use for their floors.
COPY_FLOOR_ARM = "copy_floor"
MCORE_BASE_ARM = "mcore/base"
MCORE_NO_FUSION_ARM = "mcore/no_bias_dropout_fusion"
TITAN_ARM = "titan"

# The layer the mcore arms read the operation from. Every layer holds the same
# ``mlp_bda`` at ``moe_layer_freq=1``, so the index is arbitrary and is fixed
# here so the provenance line can name it.
MCORE_LAYER = 0


# The one megatron variant this scenario measures is a delta on one field,
# and it is declared once for the whole repository in
# ``benchmarks/models/piper_qwen3/mcore_profiles.py``. Scenario 7
# (``attn_residual``) cuts the other call site the same field gates --
# ``self_attn_bda`` at ``transformer_layer.py:684`` against ``mlp_bda`` at
# ``:980`` -- so the two scenarios share one profile rather than each holding
# a copy. ``NO_BIAS_DROPOUT_FUSION`` is imported above and stays out of
# ``MCORE_PROFILES``: that dict is the roster of profiles other systems run,
# and the e2e megatron arm runs ``base``.


@dataclass
class MoeResidualInputs:
    """The two operands both engines add, and the gradient they add under.

    ``x`` stands in for the MoE block's output and ``residual`` for the block
    input. Both are canonical ``[B, L, D]`` bf16. The values reach no timing
    -- an add costs what it costs -- and exist only so the gates compare two
    implementations on one pair of tensors.

    This scenario holds no weight at all, which is why there is no shared
    parameter to load and no weight gradient to gate. It is the only cut in
    the partition with that property.
    """

    x: torch.Tensor  # (B, L, D) bf16, the MoE output
    residual: torch.Tensor  # (B, L, D) bf16, the block input
    grad_out: torch.Tensor  # (B, L, D) bf16
    bytes_moved: int


def moe_residual_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> MoeResidualInputs:
    x = _randn((workload.batch, workload.seq_len, shape.dim), device, generator)
    residual = _randn_like(x, generator)
    grad_out = _randn_like(x, generator)
    return MoeResidualInputs(
        x=x,
        residual=residual,
        grad_out=grad_out,
        # Two reads and one write. The merge divides this by every mode's
        # median, so the GB/s column is a bandwidth statement in ``forward``
        # alone.
        bytes_moved=3 * x.numel() * x.element_size(),
    )


def moe_residual_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeResidualInputs
) -> dict[str, torch.Tensor]:
    """fp64 truth for the add and both input gradients.

    No input is quantized before promotion, unlike ``ffn_norm`` and
    ``attn_out_proj``: both operands are already bf16 tensors the arms hold
    bit-for-bit, so there is no fp32 parameter whose rounding a truth could
    charge to the arms by accident.

    The gradient half of this reference is not a formality. ``grad_out``
    reaches both operands unchanged only while the dropout is an identity and
    the bias branch is unused; a profile that broke either would scale or mask
    one of them, and this is the check that would see it.
    """
    x = inputs.x.double().detach().requires_grad_()
    residual = inputs.residual.double().detach().requires_grad_()
    out = residual + x
    torch.autograd.backward(out, inputs.grad_out.double())
    return {
        "out": out.detach(),
        "x_grad": x.grad,
        "residual_grad": residual.grad,
    }


def _residual_arm(
    *,
    name: str,
    call: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    x_native: torch.Tensor,
    residual_native: torch.Tensor,
    grad_native: torch.Tensor,
    canonical: tuple[int, ...],
    bytes_moved: int,
) -> BuiltArm:
    """The timed closures every arm in this scenario shares.

    All three arms run this same code over their own ``call``, so the
    published row measures the two operations and nothing about how either
    arm was written.

    ``x_native``, ``residual_native`` and ``grad_native`` are already in the
    shape that engine's call expects, because a reshape inside a timed closure
    would be timed as part of the add. ``canonical`` puts the outputs back to
    ``[B, L, D]`` for the gates, which run outside the timed region.

    Both operands are leaves that require grad. That is the condition
    production runs in, and it is what makes ``forward_backward`` and the two
    gradient gates possible at all.

    It is **not** what closes megatron's in-place branch, and the earlier
    revision of this docstring said it did. ``_bias_dropout_add_func``
    computes ``inplace`` as a conjunction of four terms, and the first of
    them is ``not training`` (``fused_bias_dropout.py:22-27``). ``training``
    is ``True`` on both mcore arms -- ``_assert_mcore_bda`` check 2 enforces
    it, and ``bias_dropout_add_fused_train`` hardcodes it at ``:73``. So the
    first term alone closes the branch, whatever the three ``requires_grad``
    terms hold.
    """
    forward_x = x_native.clone().requires_grad_()
    forward_residual = residual_native.clone().requires_grad_()
    round_trip_x = x_native.clone().requires_grad_()
    round_trip_residual = residual_native.clone().requires_grad_()
    check_x = x_native.clone().requires_grad_()
    check_residual = residual_native.clone().requires_grad_()

    def forward():
        return call(forward_x, forward_residual)

    def forward_backward() -> None:
        _reset_grads(round_trip_x, round_trip_residual)
        torch.autograd.backward(
            call(round_trip_x, round_trip_residual), grad_native
        )

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(check_x, check_residual)
        out = call(check_x, check_residual)
        torch.autograd.backward(out, grad_native)
        # ``IdentityFuncOp`` is the failure this names: megatron's
        # ``TransformerLayerSubmodules.mlp_bda`` defaults to it
        # (``transformer_layer.py:280``), and its call returns a function that
        # hands back its first argument and discards the rest
        # (``identity_op.py:35-40``). An arm wired that way drops the residual
        # entirely and would read as a spectacular win rather than as a bug.
        # ``_assert_mcore_bda`` refuses it at build time; this is the second
        # net, and it covers the titan arm too.
        return _require_grads(
            name,
            {
                "out": out.detach().reshape(canonical),
                "x_grad": None
                if check_x.grad is None
                else check_x.grad.reshape(canonical),
                "residual_grad": None
                if check_residual.grad is None
                else check_residual.grad.reshape(canonical),
            },
            detail=(
                "the operation did not consume both operands, so it is not a "
                "residual add"
            ),
        )

    return BuiltArm(
        name=name,
        calls={"forward": forward, "forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
        bytes_moved=bytes_moved,
    )


def build_moe_residual_copy_floor(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeResidualInputs
) -> BuiltArm:
    """The bandwidth floor: two reads, one add per element, one write.

    **This arm decides whether the published row is a kernel result at all.**
    The measurand is one bf16 elementwise add over ``batch * seq_len * dim``
    elements. That is a few tens of microseconds of device work, and the two
    mcore arms differ precisely in their *dispatch* paths -- ``mcore/base``
    evaluates a ``torch.compile`` wrapper's guards per call, and
    ``mcore/no_bias_dropout_fusion`` allocates a python closure and dispatches
    eager ops per call. Neither difference is device work. Without a floor a
    reader cannot tell a fusion result from a dispatch result, and the
    ``--burst`` residual cannot tell them apart either: CLAUDE.md records that
    a ladder can plateau at a dispatch cost bursting never amortizes, so an
    unflagged row is not evidence of device-boundedness.

    The floor is deliberately the *minimum*: ``torch.add(..., out=...)``
    allocates nothing and builds no autograd graph. So the gap between the
    floor and an arm holds three things at once -- python and dispatch cost,
    autograd bookkeeping, and the output allocation. Read the ``x_floor``
    column as "how far above pure traffic this arm sits", never as an
    attribution of that distance.

    Forward only, for ``ffn_norm``'s reason: the forward traffic is exactly
    this, and a floor for the forward+backward traffic would be an invention
    rather than a measurement.
    """
    out = torch.empty_like(inputs.x)

    def forward() -> None:
        torch.add(inputs.x, inputs.residual, out=out)

    return BuiltArm(
        name=COPY_FLOOR_ARM,
        calls={"forward": forward},
        correctness_outputs=dict,
        bytes_moved=inputs.bytes_moved,
    )


def _residual_add(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
    """TorchTitan's residual add, as the block writes it.

    ``qwen3/model.py:63`` is ``x = x + self.moe(self.ffn_norm(x))``, so the
    block input is the left operand and the MoE output the right one. Megatron
    writes the same order (``out = residual + out``,
    ``fused_bias_dropout.py:58``), which is why the cross-engine gate can ask
    for bit-identity rather than for closeness.
    """
    return residual + x


def build_moe_residual_titan(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeResidualInputs
) -> BuiltArm:
    """TorchTitan's ``x + moe(...)``, compiled.

    **The only titan arm in this package that imports no torchtitan symbol,
    and there is nothing to import.** The residual add is an expression inside
    ``Qwen3TransformerBlock.forward``, not a module and not a config node, so
    there is no object to build and no ``[Override]`` to count.
    ``tests/test_kernel_moe_residual.py`` pins the expression against the
    submodule source instead, which is the only evidence a CPU can offer that
    this arm cuts where the scenario says it cuts.

    ``torch.compile(fullgraph=True)`` is ``common._compile_module``'s
    treatment, applied to a function rather than to a module because titan has
    no module here. Wrapping the add in an ``nn.Module`` to reach the shared
    helper would invent a module titan does not have and would change nothing
    Inductor generates.

    **This compile is the scenario's central caveat.** End to end the add is
    one node inside the block's graph, where Inductor may fuse it into a
    neighbour; here it is a graph of its own and emits its own kernel.
    Isolation therefore hands titan megatron's fusion scope, which is exactly
    why the registry publishes no cross-engine ratio. See this module's
    docstring.
    """
    return _residual_arm(
        name=TITAN_ARM,
        call=torch.compile(_residual_add, fullgraph=True),
        x_native=inputs.x,
        residual_native=inputs.residual,
        grad_native=inputs.grad_out,
        canonical=tuple(inputs.x.shape),
        bytes_moved=inputs.bytes_moved,
    )


def _assert_mcore_bda(layer: Any, arm: str, fused: bool) -> Any:
    """Refuse to time anything but the operation this arm claims, and return it.

    Seven checks, every one of which raises. Each names a way the arm could
    measure something other than megatron's post-MoE residual add while every
    correctness gate still passed, because the wrong operation here is usually
    still numerically right.

    1. ``mlp_bda`` is megatron's ``get_bias_dropout_add``. The submodule field
       defaults to ``IdentityFuncOp`` (``transformer_layer.py:280``), which
       returns a function that hands back its first argument. This check
       proves the field was not left at that default. It proves nothing about
       *which* spec branch ran, because every branch writes
       ``mlp_bda=get_bias_dropout_add`` -- the TE branch this build takes at
       ``gpt_layer_specs.py:338``, and six others identically.
    2. The layer is in training mode. ``get_bias_dropout_add`` selects
       ``bias_dropout_add_fused_inference`` otherwise
       (``fused_bias_dropout.py:89-92``), which is a different function and a
       different arm.
    3. ``hidden_dropout`` is exactly 0.0. It is the ``prob`` the call site
       passes (``transformer_layer.py:981``); at any other value the operation
       stops being ``residual + x``, stops being deterministic, and stops
       being comparable to titan, which has no dropout at all.
    4. ``add_bias_linear`` is False, so the MoE layer's output tuple carries
       ``None`` for the bias (``moe_layer.py:561`` asserts it) and
       ``_bias_dropout_add_func`` takes the no-bias branch. With a bias the
       operation grows an addition titan has no counterpart for.
    5. ``config.bias_dropout_fusion`` is what this arm's profile declares.
       This is the delta, and it is delivered by the config field alone -- the
       layer spec names ``get_bias_dropout_add`` for both arms and the flag is
       read per call at ``transformer_layer.py:980``. A delta that did not
       take would publish the base implementation under the variant's name.
    6. The resolved operation is the expected one of the two: the module-level
       ``bias_dropout_add_fused_train`` when fused, and neither fused function
       when not.
    7. ``megatron.core.jit.jit_fuser`` is ``torch.compile``. That is what makes
       ``KernelArm.compiled`` true for ``mcore/base``: the compile comes from
       megatron's own decorator, applied at import of ``fused_bias_dropout``,
       not from this harness. Below torch 2.2 ``enable_jit_fuser``
       (``jit.py:16-24``) leaves the module-level ``torch.jit.script``
       binding (``:7``) in place, and on ImportError it installs a no-op
       decorator; either would leave the arm eager under a compiled label.
    """
    from megatron.core.fusions.fused_bias_dropout import (
        bias_dropout_add_fused_inference,
        bias_dropout_add_fused_train,
        get_bias_dropout_add,
    )
    from megatron.core.jit import jit_fuser

    if layer.mlp_bda is not get_bias_dropout_add:
        raise RuntimeError(
            f"{arm}: layer.mlp_bda is {layer.mlp_bda!r}, not megatron's "
            "get_bias_dropout_add. TransformerLayerSubmodules defaults the "
            "field to IdentityFuncOp, which returns its first argument and "
            "drops the residual, so this arm would time an identity"
        )
    if not layer.training:
        raise RuntimeError(
            f"{arm}: the layer is in eval mode, so get_bias_dropout_add "
            "returns bias_dropout_add_fused_inference; that is a different "
            "function and a different arm"
        )
    dropout = float(layer.hidden_dropout)
    if dropout != 0.0:
        raise RuntimeError(
            f"{arm}: hidden_dropout is {dropout!r}, not 0.0. The scenario "
            "measures residual + x; a live dropout makes the operation "
            "nondeterministic and gives titan nothing to be compared with"
        )
    if layer.config.add_bias_linear is not False:
        raise RuntimeError(
            f"{arm}: add_bias_linear is {layer.config.add_bias_linear!r}, not "
            "False. The MoE output would carry a bias, _bias_dropout_add_func "
            "would take its bias branch, and the operation would no longer be "
            "the add titan writes"
        )
    if layer.config.bias_dropout_fusion is not fused:
        raise RuntimeError(
            f"{arm}: config.bias_dropout_fusion is "
            f"{layer.config.bias_dropout_fusion!r}, expected {fused!r}; the "
            "profile delta did not reach the built config, so this arm would "
            "publish the other arm's implementation under its own name"
        )

    resolved = layer.mlp_bda(layer.training, layer.config.bias_dropout_fusion)
    if fused:
        if resolved is not bias_dropout_add_fused_train:
            raise RuntimeError(
                f"{arm}: the resolved operation is {resolved!r}, not "
                "bias_dropout_add_fused_train"
            )
        if jit_fuser is not torch.compile:
            raise RuntimeError(
                f"{arm}: megatron.core.jit.jit_fuser is {jit_fuser!r}, not "
                "torch.compile, so bias_dropout_add_fused_train was never "
                "compiled and this arm is eager under a compiled label"
            )
    elif resolved in (bias_dropout_add_fused_train, bias_dropout_add_fused_inference):
        raise RuntimeError(
            f"{arm}: bias_dropout_fusion is off but the resolved operation is "
            f"{resolved!r}, one of the fused pair"
        )
    return resolved


def _build_mcore_arm(
    *,
    arm: str,
    profile: McoreProfile,
    fused: bool,
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: MoeResidualInputs,
) -> BuiltArm:
    """One megatron arm: build the model, take the operation, drop the model.

    The operation is navigated off a real ``GPTModel`` rather than imported
    directly, which is the rule for every mcore arm in this package and earns
    its keep here: ``mlp_bda`` is a *submodule field*, so importing
    ``get_bias_dropout_add`` and calling it would prove nothing about what
    megatron's own spec derivation wired into the layer. ``_assert_mcore_bda``
    checks the wiring instead.

    **The whole model is released, and this arm keeps nothing of it.**
    ``mlp_bda`` is a free function and the two flags are plain values, so
    unlike every other mcore arm in the package this one holds no parameter at
    all. Dropping the model matters anyway: ``max_memory_allocated`` is a
    maximum over time of the bytes currently allocated, so a retained model
    would be charged to the arm's ``peak_memory_gib`` by the first allocation
    inside the timed call.

    **The operation is resolved inside the timed closure, exactly as
    ``transformer_layer.py:980`` resolves it.** That per-call resolution is
    part of the delta: the fused branch returns a module-level function, while
    the unfused branch allocates a fresh closure on every call
    (``bias_dropout_add_unfused``, ``fused_bias_dropout.py:62-66``). Hoisting
    it would hand the unfused arm a cost megatron does not save it.

    The ``with self.bias_dropout_add_exec_handler()`` that wraps the call site
    (``transformer_layer.py:979``, ``= torch.enable_grad``,
    ``transformer_layer.py:525``) is deliberately **not** reproduced. Grad is
    already enabled in every timed closure here, so it is a no-op that both
    mcore arms would pay identically -- it cannot move the one row this
    scenario publishes, and leaving it out keeps the per-call number a
    statement about the add.
    """
    initialize_megatron_single_rank()

    from benchmarks.models.piper_qwen3.megatron_model import build_model

    model = build_model(
        seq_len=workload.seq_len,
        shape=shape,
        profile=profile,
        blank_parts=MCORE_BLANK_MLP,
    )
    layer = model.decoder.layers[MCORE_LAYER]
    resolved = _assert_mcore_bda(layer, arm, fused)
    mlp_bda = layer.mlp_bda
    training = layer.training
    fusion = layer.config.bias_dropout_fusion
    dropout = float(layer.hidden_dropout)
    print(
        f"moe_residual/{arm}: decoder.layers[{MCORE_LAYER}].mlp_bda resolves "
        f"to {getattr(resolved, '__qualname__', resolved)!r} "
        f"(profile {profile.name}, bias_dropout_fusion={fusion}, "
        f"hidden_dropout={dropout})",
        flush=True,
    )
    del model, layer, resolved
    gc.collect()
    torch.cuda.empty_cache()

    def call(x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        # transformer_layer.py:980-982 verbatim, minus the enable_grad handler
        # this builder's docstring accounts for. The bias is None because
        # add_bias_linear is False, which moe_layer.py:561 also asserts.
        return mlp_bda(training, fusion)((x, None), residual, dropout)

    tokens = workload.batch * workload.seq_len
    # THD, the form our megatron driver runs: the driver packs each batch's
    # rows into one sequence, so every hidden state reaching a layer is
    # (t, 1, h). A view, not a copy, and taken here rather than in a closure.
    return _residual_arm(
        name=arm,
        call=call,
        x_native=inputs.x.reshape(tokens, 1, shape.dim),
        residual_native=inputs.residual.reshape(tokens, 1, shape.dim),
        grad_native=inputs.grad_out.reshape(tokens, 1, shape.dim),
        canonical=tuple(inputs.x.shape),
        bytes_moved=inputs.bytes_moved,
    )


def build_moe_residual_mcore_base(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeResidualInputs
) -> BuiltArm:
    """Megatron-core's ``mlp_bda`` at the base profile: the fused add.

    ``bias_dropout_fusion`` is True, so the call site resolves to
    ``bias_dropout_add_fused_train``, which megatron decorates with
    ``@jit_fuser`` -- ``torch.compile`` on torch >= 2.2. **The compile comes
    from megatron, not from this harness**, which is why the arm declares
    ``compiled=True`` while the builder wraps nothing. It is the one arm in
    the package whose compile treatment is the engine's own choice, and
    ``_assert_mcore_bda``'s ``jit_fuser`` check is what makes the label
    checkable.
    """
    return _build_mcore_arm(
        arm=MCORE_BASE_ARM,
        profile=BASE,
        fused=True,
        shape=shape,
        workload=workload,
        inputs=inputs,
    )


def build_moe_residual_mcore_no_bias_dropout_fusion(
    shape: PiperShape, workload: KernelWorkload, inputs: MoeResidualInputs
) -> BuiltArm:
    """The same call site with ``bias_dropout_fusion=False``: the eager add.

    The delta is delivered by the config field alone. The layer spec names
    ``get_bias_dropout_add`` under both profiles, and the flag is read per
    call at ``transformer_layer.py:980``, so no spec keyword is involved and
    ``_assert_mcore_bda`` checks the built config rather than the built class.

    Eager on purpose, and the arm *is* that eager-ness:
    ``bias_dropout_add_unfused`` is plain python around
    ``_bias_dropout_add_func``. Compiling it here would erase the difference
    the arm exists to measure.
    """
    return _build_mcore_arm(
        arm=MCORE_NO_FUSION_ARM,
        profile=NO_BIAS_DROPOUT_FUSION,
        fused=False,
        shape=shape,
        workload=workload,
        inputs=inputs,
    )
