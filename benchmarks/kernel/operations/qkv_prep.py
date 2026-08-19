"""Arm builders for the ``qkv_prep`` cross-engine kernel scenario.

The attention-input norm, the QKV projection and the split that follows it,
head to head across the two engines. The cut starts at the block's residual
stream and ends at three separate ``[B, L, N, H]`` tensors, which is the one
place both engines hold the same three objects.

* TorchTitan: ``self.attention_norm(x)`` in ``Qwen3TransformerBlock.forward``
  (``third_party/torchtitan/torchtitan/models/qwen3/model.py:60``, the module
  built at ``:51``), then ``self.qkv_linear(x_BLD)`` in
  ``GQAttention.forward``
  (``third_party/torchtitan/torchtitan/models/common/attention.py:954``, the
  module built at ``:933``).
* Megatron-core: ``self_attention.linear_qkv``, which holds the norm inside
  it, then ``get_query_key_value_tensors``
  (``third_party/Megatron-LM/megatron/core/transformer/attention.py:1814``;
  ``linear_qkv`` is built at ``:1682``).

**The norm is inside the scenario on both engines, and that is a cost.**
Megatron builds ``linear_qkv`` as one ``TELayerNormColumnParallelLinear``
(``gpt_layer_specs.py:324`` writes ``linear_qkv=backend.
column_parallel_layer_norm_linear()``, and ``TESpecProvider`` resolves that to
the class at ``extensions/transformer_engine_spec_provider.py:55-57``). TE
fuses the RMSNorm into the GEMM's prologue, and no public entry point runs the
GEMM alone or the norm alone. So a partition that put the norm outside would
have to split a module megatron does not split, and the megatron arm would
measure something megatron never runs. The norm therefore sits inside, on both
sides. The cost is that the re-homed ``qkv`` arms gain a norm: **a number from
this scenario is not comparable to a number from the ``qkv`` scenario**, and
the scenario ``description`` says so.

**The scenario ends at separate q/k/v, so both engines' layout work is
timed.** TorchTitan's fused path splits the packed buffer and then
materializes all three outputs (``models/common/attention.py:809`` defines
``_split``, the returns are at ``:819-821``). Megatron reshapes the packed
buffer, splits it with TE's ``SplitAlongDim`` and reshapes the query
(``transformer/attention.py:1869`` views, ``:1889-1905`` is the ungated split
-- ``SplitAlongDim`` at ``:1903`` -- and ``:1908`` reshapes). Neither piece is
free and neither is the same op, so charging each engine its own is the only
symmetric choice available: the alternative -- stopping at the packed buffer
-- would leave titan's three copies unassigned, and no later scenario in the
partition can adopt them.

**Which op does the copying, on the titan side.** ``qkv`` is viewed as
``[b, s, n_kv, heads_per_group + 2, head_dim]`` (``:801``) and split on
``dim=-2``, so at ``heads_per_group=2`` the three results differ: ``xq`` is a
``[b, s, n_kv, 2, hd]`` strided slice whose ``reshape`` to ``[b, s, -1, hd]``
merges two dimensions that are not adjacent in memory, so the **reshape**
copies and the trailing ``.contiguous()`` is a no-op; ``xk`` and ``xv`` are
``[b, s, n_kv, 1, hd]`` slices whose ``reshape`` only drops a size-1
dimension, so it returns a view and the ``.contiguous()`` copies. Three
materializations, two of them by ``.contiguous()``. The code comment at
``:815-817`` describes only the second case.

**The cut is symmetric in what it computes and asymmetric in what it
materializes. That asymmetry is real, it is timed, and it is declared rather
than equalized.** For a non-FP8 tensor ``SplitAlongDim.forward`` is
``torch.split`` and returns views (TE
``pytorch/utils.py:437-440``), and megatron reshapes only the query. So
megatron hands ``key`` and ``value`` on as non-contiguous strided views and
never materializes them, while titan materializes all three. At the default
workload (batch 4, seq 1024, ``normal``, bf16) the forward copies are
``B*L*n_heads*head_dim*2 = 8 MiB`` for q and ``B*L*n_kv_heads*head_dim*2 =
4 MiB`` each for k and v, so ``mcore/base`` pays 8 MiB, ``titan`` pays 16 MiB
and ``titan/unfused_qkv`` pays none -- its three GEMMs write contiguous
outputs already. The backward path carries the mirror image of the same
asymmetry.

Both engines really do this, so measuring it is correct and equalizing it
would be the distortion. What a reader needs is the consequence: megatron
does not avoid the cost, it **defers** it to whoever consumes the strided
views, which is ``attention_core`` (scenario 5) for megatron and nowhere for
titan. The sixteen scenarios therefore sum correctly, but this row read alone
overstates titan's projection cost by roughly 8 MiB per direction of traffic
that megatron will pay later. The scenario ``description`` says so.

**The qk norms are excluded, and excluding them takes an explicit step.**
``get_query_key_value_tensors`` also applies ``q_layernorm`` and
``k_layernorm`` before it returns (``transformer/attention.py:1922-1926``).
Those belong to scenario 3, ``qk_norm``, whose ``mcore/base`` arm is exactly
those two modules, and the partition allows no overlap. TorchTitan already
draws the line where this scenario needs it: ``qkv_linear`` returns pre-norm
q/k/v and ``GQAttention.forward`` applies ``q_norm``/``k_norm`` afterwards
(``models/common/attention.py:959-960``). The mcore arm therefore sets
``q_layernorm`` and ``k_layernorm`` to ``None`` before it times anything.
That is not a rewrite of megatron's code: it is the state megatron's own
``SelfAttention.__init__`` produces when ``qk_layernorm`` is off
(``transformer/attention.py:1722`` sets ``q_norm_cls = k_norm_cls = None``,
and ``:1724-1742`` builds ``None`` from it), and
``get_query_key_value_tensors`` already branches on it. The arm still runs
megatron's own view, ``SplitAlongDim`` and reshape.

**There is no isolated ``backward`` mode, and there cannot be one.** The
retained-graph trick the ``rope`` and ``qkv`` scenarios use re-runs one
backward graph many times. TE's ``LayerNormLinear`` backward frees what it
saved on the way out -- ``clear_tensor_data(mu)`` and
``clear_tensor_data(rsigma)`` at
``transformer_engine/pytorch/module/layernorm_linear.py:1113-1114``, plus
``ln_out`` at ``:1047`` and ``:1050`` -- so a second pass reads cleared
tensors.
Both engines drop the mode, which keeps the arms comparable, and backward cost
stays recoverable as ``forward_backward`` minus ``forward``.
``attn_out_proj`` and ``ffn_norm`` drop it for the same class of reason.

**All three arms hold the same weights, through the map that already owns the
correspondence.** ``benchmarks/models/piper_qwen3/megatron_weights.py`` pairs
titan's ``attention.qkv_linear.wq/wk/wv.weight`` with megatron's
``self_attention.linear_qkv.weight`` under the ``qkv`` component tag
(``:133-142``), through ``grouped_qkv``, and pairs titan's
``attention_norm.weight`` with ``linear_qkv.layer_norm_weight`` under the same
tag (``:146-150``). ``grouped_qkv`` proves its own interleave bitwise with
``assert_qkv_roundtrip`` on every call. This module calls that function and
reimplements nothing -- but it is not the only implementation of the
interleave the scenario touches. Loading a fused titan module runs
torchtitan's own ``FusedQKVLinear._merge_qkv_on_load``
(``models/common/attention.py:871-895``), which is the same cat and reshape
written a second time upstream. The two agree, and the fp64 gate is what says
so at run time.

Every torchtitan, megatron and TransformerEngine import is deferred into the
builder that needs it. ``benchmarks.models.piper_qwen3.mcore_profiles`` and
``benchmarks.models.piper_qwen3.megatron_weights`` are the two module-scope
exceptions: the first is torch-free parent-side data this module reads the
declared epsilon from, and the second is the weight map, which is torch and
nothing heavier.
"""

from __future__ import annotations

import gc
import os
import socket
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    WEIGHT_STD,
    _compile_module,
    _randn,
    _reset_grads,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE
from benchmarks.models.piper_qwen3.megatron_weights import grouped_qkv
from benchmarks.models.piper_qwen3.shape import PiperShape


# The three arm names, spelled as plan section C.2 spells them. The slash says
# which engine an arm is and which profile of that engine, and it reaches a
# filename through ``benchmarks/kernel/runner.py``'s ``fragment_path``, which
# routes it through ``schema.fragment_stem``. **A tree without
# ``fragment_stem`` cannot run this scenario**: two of these three names carry
# a slash, and a timing worker would write into a directory nobody creates.
MCORE_ARM_NAME = "mcore/base"
TITAN_ARM_NAME = "titan"
TITAN_UNFUSED_ARM_NAME = "titan/unfused_qkv"

# The layer the mcore arm reads its attention block from. Every layer holds
# the same modules at the same shape, and the weights are overwritten from the
# shared seeded tensors, so the index is arbitrary and is fixed here so the
# provenance line can name it.
MCORE_LAYER = 0

# The seed megatron's CUDA RNG tracker takes before the model builds. Every
# parameter this arm measures is overwritten afterwards, so the value reaches
# no measured tensor; megatron cannot initialize a TE module without a seeded
# tracker at all.
MCORE_INIT_SEED = 42

# The epsilon both engines use, read from the mcore profile rather than
# retyped. ``build_qkv_prep_titan`` and its unfused twin refuse to build when
# torchtitan's ``_qwen3_norm`` disagrees with it: two sides that normalize
# with different epsilons compute two functions, not two implementations of
# one.
NORM_EPS = float(BASE.config_overrides["layernorm_epsilon"])

# The class the mcore arm must build. Several names in this system overclaim
# "TE"; this one is TE, and the guard is what lets a published table say so.
MCORE_MODULE_CLASS = "TELayerNormColumnParallelLinear"

# The two parameter names ``benchmarks.models.piper_qwen3.megatron_weights``
# yields under its ``qkv`` component, and the four titan names it reads them
# from. That module is the authority on the correspondence and this scenario
# does not reimplement it: ``tests/test_kernel_qkv_prep.py`` asserts the map
# still yields exactly this pair from exactly these sources, and still yields
# the QKV matrix through ``grouped_qkv``. A transfer for two parameters cannot
# go through ``transfer_weights``, which needs a whole titan state dict, so
# the test is what ties the two together.
MCORE_WEIGHT_NAME = "decoder.layers.{layer}.self_attention.linear_qkv.weight"
MCORE_NORM_WEIGHT_NAME = (
    "decoder.layers.{layer}.self_attention.linear_qkv.layer_norm_weight"
)
MCORE_WEIGHT_COMPONENT = "qkv"
TITAN_WEIGHT_NAMES = (
    "layers.{layer}.attention.qkv_linear.wq.weight",
    "layers.{layer}.attention.qkv_linear.wk.weight",
    "layers.{layer}.attention.qkv_linear.wv.weight",
)
TITAN_NORM_WEIGHT_NAME = "layers.{layer}.attention_norm.weight"

# The three keys of the shared weight state dict. They are the names
# ``QKVLinear`` exposes and the names ``FusedQKVLinear``'s merge hook accepts,
# which is what lets both titan arms load one dict.
QKV_STATE_KEYS = ("wq.weight", "wk.weight", "wv.weight")


def mcore_attention_path(layer: int = MCORE_LAYER) -> str:
    """The attribute path from a ``GPTModel`` down to the attention block.

    Derived from ``MCORE_WEIGHT_NAME`` rather than written beside it, so the
    navigation and the weight map can only agree.
    """
    name = MCORE_WEIGHT_NAME.format(layer=layer)
    suffix = ".linear_qkv.weight"
    assert name.endswith(suffix)
    return name[: -len(suffix)]


def _navigate(root: object, path: str) -> Any:
    """Walk a dotted attribute path, indexing on a numeric segment."""
    node = root
    for segment in path.split("."):
        node = node[int(segment)] if segment.isdigit() else getattr(node, segment)
    return node


@dataclass
class QkvPrepInputs:
    """One hidden-state batch, one norm gain, and one QKV weight triple.

    Everything is drawn once and every arm loads all of it, so the three arms
    hold the same values by construction rather than by a transposition
    written here. ``weight_state`` is in titan's unfused spelling because that
    is the spelling both the fused module's merge hook and
    ``megatron_weights.grouped_qkv`` accept as input.
    """

    x: torch.Tensor  # (B, L, dim) bf16, the block's residual stream
    grad_q: torch.Tensor  # (B, L, n_heads, head_dim) bf16
    grad_k: torch.Tensor  # (B, L, n_kv_heads, head_dim) bf16
    grad_v: torch.Tensor  # (B, L, n_kv_heads, head_dim) bf16
    weight_state: dict[str, torch.Tensor]  # wq/wk/wv .weight, fp32
    norm_weight: torch.Tensor  # (dim,) fp32 RMSNorm gain
    eps: float


def qkv_prep_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> QkvPrepInputs:
    batch, seq = workload.batch, workload.seq_len
    q_out = shape.n_heads * shape.head_dim
    kv_out = shape.n_kv_heads * shape.head_dim
    weight_state = {
        "wq.weight": _randn(
            (q_out, shape.dim), device, generator, torch.float32, WEIGHT_STD
        ),
        "wk.weight": _randn(
            (kv_out, shape.dim), device, generator, torch.float32, WEIGHT_STD
        ),
        "wv.weight": _randn(
            (kv_out, shape.dim), device, generator, torch.float32, WEIGHT_STD
        ),
    }
    # A trained RMSNorm gain sits near 1.0, and both engines initialize it to
    # exactly ones. A gain of exactly ones cannot show a gain an arm never
    # loaded, so the shared gain is 1 + N(0, WEIGHT_STD). ``ffn_norm_inputs``
    # draws its gain the same way and for the same reason.
    noise = torch.randn(
        (shape.dim,), device=device, generator=generator, dtype=torch.float32
    )
    return QkvPrepInputs(
        x=_randn((batch, seq, shape.dim), device, generator),
        grad_q=_randn(
            (batch, seq, shape.n_heads, shape.head_dim), device, generator
        ),
        grad_k=_randn(
            (batch, seq, shape.n_kv_heads, shape.head_dim), device, generator
        ),
        grad_v=_randn(
            (batch, seq, shape.n_kv_heads, shape.head_dim), device, generator
        ),
        weight_state=weight_state,
        norm_weight=1.0 + WEIGHT_STD * noise,
        eps=NORM_EPS,
    )


def qkv_prep_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: QkvPrepInputs
) -> dict[str, torch.Tensor]:
    """fp64 truth for the norm, the three projections and every gradient.

    Each parameter is quantized to bf16 before it is promoted to fp64, so the
    truth is the truth for the parameters the arms actually hold. An fp64
    truth built from the unrounded values would charge every arm for an input
    cast none of them performs; ``qkv_reference`` and ``ffn_norm_reference``
    make the same correction.

    ``qkv_weight_grad`` is reported in megatron's grouped interleave, which is
    also titan's fused layout. The reference produces it with ``grouped_qkv``
    -- the same function the arms' weights are transferred through, and it
    runs ``assert_qkv_roundtrip`` here too. ``grouped_qkv`` names weights, but
    it is a pure concatenate-and-reshape, so it applies unchanged to the
    gradients of those weights.

    **Two implementations of the interleave are in play, not one.** The
    reference and ``titan/unfused_qkv`` call ``grouped_qkv``; ``mcore/base``
    and ``titan`` hold the layout in their own fused parameter and return its
    gradient directly, and titan's copy of the layout comes from torchtitan's
    ``FusedQKVLinear._merge_qkv_on_load``
    (``models/common/attention.py:871-895``), which this arm reaches through
    ``load_state_dict``. The two agree arithmetically, and this gate is the
    run-time proof: a divergence would move ``qkv_weight_grad`` on the fused
    arms alone.
    """
    batch, seq = workload.batch, workload.seq_len

    def leaf(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.to(torch.bfloat16).double().detach().requires_grad_()

    x = inputs.x.double().detach().requires_grad_()
    gain = leaf(inputs.norm_weight)
    wq = leaf(inputs.weight_state["wq.weight"])
    wk = leaf(inputs.weight_state["wk.weight"])
    wv = leaf(inputs.weight_state["wv.weight"])

    hidden = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + inputs.eps) * gain
    q_out = F.linear(hidden, wq).view(batch, seq, shape.n_heads, shape.head_dim)
    k_out = F.linear(hidden, wk).view(
        batch, seq, shape.n_kv_heads, shape.head_dim
    )
    v_out = F.linear(hidden, wv).view(
        batch, seq, shape.n_kv_heads, shape.head_dim
    )
    torch.autograd.backward(
        (q_out, k_out, v_out),
        (
            inputs.grad_q.double(),
            inputs.grad_k.double(),
            inputs.grad_v.double(),
        ),
    )
    return {
        "q_out": q_out.detach(),
        "k_out": k_out.detach(),
        "v_out": v_out.detach(),
        "x_grad": x.grad,
        "qkv_weight_grad": grouped_qkv(wq.grad, wk.grad, wv.grad, shape),
        "norm_weight_grad": gain.grad,
    }


def _require_grads(arm: str, outputs: dict[str, torch.Tensor | None]) -> None:
    """Turn a missing gradient into a named failure, not an AttributeError."""
    missing = sorted(name for name, value in outputs.items() if value is None)
    if missing:
        raise RuntimeError(
            f"{arm}: backward produced no gradient for {', '.join(missing)}; "
            "the correctness gate cannot compare a tensor that does not exist"
        )


def _weight_grad_outputs(
    arm: str, qkv_grad: torch.Tensor | None, norm_grad: torch.Tensor | None
) -> dict[str, torch.Tensor]:
    """The two canonical weight gradients, checked for existence first."""
    outputs = {"qkv_weight_grad": qkv_grad, "norm_weight_grad": norm_grad}
    _require_grads(arm, outputs)
    return outputs


def _qkv_prep_arm(
    *,
    name: str,
    owner: nn.Module,
    call: Callable[
        [torch.Tensor], tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    ],
    x_native: torch.Tensor,
    grads_native: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    canonical_in: tuple[int, ...],
    canonical_out: tuple[tuple[int, ...], ...],
    weight_grads: Callable[[], dict[str, torch.Tensor]],
    notes: dict[str, Any] | None = None,
) -> BuiltArm:
    """Forward and forward+backward over one engine's native tensor shape.

    Every arm runs this same code over its own callable, so the comparison
    measures the two engines and nothing about how each arm was written.

    ``x_native`` and ``grads_native`` already carry the shape that engine's
    modules expect, because a reshape inside a timed closure would be timed as
    part of the projection. ``canonical_in`` and ``canonical_out`` put the
    tensors back into ``[B, L, *]`` for the gates, which run outside every
    timed region.

    ``owner`` is the module whose parameter gradients are cleared between
    calls. For a compiled titan arm that is the module *inside* the compile
    wrapper, because that is where the parameters live.
    """
    forward_leaf = x_native.clone().requires_grad_()
    round_trip_leaf = x_native.clone().requires_grad_()
    check_leaf = x_native.clone().requires_grad_()

    def forward():
        return call(forward_leaf)

    def forward_backward() -> None:
        _reset_grads(round_trip_leaf, owner)
        torch.autograd.backward(call(round_trip_leaf), grads_native)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(check_leaf, owner)
        q_out, k_out, v_out = call(check_leaf)
        torch.autograd.backward((q_out, k_out, v_out), grads_native)
        _require_grads(name, {"x_grad": check_leaf.grad})
        return {
            "q_out": q_out.detach().reshape(canonical_out[0]),
            "k_out": k_out.detach().reshape(canonical_out[1]),
            "v_out": v_out.detach().reshape(canonical_out[2]),
            "x_grad": check_leaf.grad.reshape(canonical_in),
            **weight_grads(),
        }

    return BuiltArm(
        name=name,
        calls={"forward": forward, "forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
        notes=dict(notes or {}),
    )


class _TitanQkvPrep(nn.Module):
    """The norm and the projection, in the order the block runs them.

    Not a reimplementation of anything: it holds the two production modules
    and calls them exactly as ``Qwen3TransformerBlock.forward`` and
    ``GQAttention.forward`` do -- ``self.qkv_linear(self.attention_norm(x))``,
    which is ``models/qwen3/model.py:60`` composed with
    ``models/common/attention.py:954``.

    It exists so the whole cut compiles as **one** graph. That is the
    treatment the block gives it: ``apply_compile`` wraps each entry of
    ``model.layers`` with ``fullgraph=True``, and ``attention_norm`` and
    ``attention`` are both attributes of that block
    (``models/qwen3/model.py:41,51``), so no compile boundary sits between
    them in production either.
    """

    def __init__(self, attention_norm: nn.Module, qkv_linear: nn.Module) -> None:
        super().__init__()
        self.attention_norm = attention_norm
        self.qkv_linear = qkv_linear

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.qkv_linear(self.attention_norm(x))


def _titan_norm(shape: PiperShape, inputs: QkvPrepInputs) -> nn.Module:
    """TorchTitan's ``attention_norm``, built from the production config node.

    ``_qwen3_norm`` is the helper ``_build_qwen3_moe_layers`` passes as
    ``attention_norm=`` for every block
    (``third_party/torchtitan/torchtitan/models/qwen3/__init__.py:151``), so
    the epsilon comes from upstream rather than from a value retyped here.
    ``build_ffn_norm_titan`` builds its own norm the same way.
    """
    from torchtitan.models.qwen3 import _qwen3_norm

    config = _qwen3_norm(shape.dim)
    if float(config.eps) != inputs.eps:
        raise RuntimeError(
            f"titan builds attention_norm with eps {config.eps!r} and the "
            f"mcore profile declares {inputs.eps!r}. The two engines would "
            "normalize with different epsilons, so the ratio would not "
            "compare two implementations of one function."
        )
    return config.build()


def _build_titan_arm(
    *,
    name: str,
    shape: PiperShape,
    inputs: QkvPrepInputs,
    qkv_linear: nn.Module,
    weight_grads: Callable[[nn.Module], Callable[[], dict[str, torch.Tensor]]],
    notes: dict[str, Any],
) -> BuiltArm:
    """Finish either titan arm: load the shared values, cast, compile, time.

    The load order matters. Both modules are built in fp32 and loaded from the
    fp32 shared tensors, then cast to bf16 in one step, so the two titan arms
    and the megatron arm all round the same fp32 values once. ``_finalize_qkv``
    in ``qkv.py`` does the same, and it is what lets the fused arm's
    state-dict merge hook see unquantized values.
    """
    attention_norm = _titan_norm(shape, inputs)
    module = _TitanQkvPrep(attention_norm, qkv_linear)
    module.to(inputs.x.device)
    # Loaded per child rather than through one prefixed dict, because
    # ``FusedQKVLinear`` merges wq/wk/wv into wqkv in a load_state_dict pre
    # hook and a child call is the shape that hook was written against.
    attention_norm.load_state_dict({"weight": inputs.norm_weight})
    qkv_linear.load_state_dict(inputs.weight_state)
    module.to(torch.bfloat16)

    grads = weight_grads(module)
    return _qkv_prep_arm(
        name=name,
        owner=module,
        call=_compile_module(module),
        x_native=inputs.x,
        grads_native=(inputs.grad_q, inputs.grad_k, inputs.grad_v),
        canonical_in=tuple(inputs.x.shape),
        canonical_out=(
            tuple(inputs.grad_q.shape),
            tuple(inputs.grad_k.shape),
            tuple(inputs.grad_v.shape),
        ),
        weight_grads=grads,
        notes=notes,
    )


def build_qkv_prep_titan(
    shape: PiperShape, workload: KernelWorkload, inputs: QkvPrepInputs
) -> BuiltArm:
    """TorchTitan as upstream configures it: ``FusedQKVLinear``, compiled.

    This is the *default* qwen3 build, not a variant.
    ``_build_qwen3_moe_layers`` takes ``fuse_qkv: bool = True``
    (``third_party/torchtitan/torchtitan/models/qwen3/__init__.py:139``) and
    passes it to ``make_gqa_config`` at ``:161``. Every registered qwen3 MoE
    flavor passes ``fuse_qkv=True`` **explicitly** (``:211``, ``:267``,
    ``:305``, ``:343``, ``:378``, ``:413``, ``:448``, ``:489``, ``:530``,
    ``:571``); none of them relies on the default. The default and the flavors
    agree, so the conclusion is the same -- fused is what upstream runs -- but
    the reason is a roster, not an omission.

    ``wqkv.weight`` is already in megatron's grouped interleave. The module's
    own load hook builds it that way -- ``_merge_qkv_on_load``
    (``models/common/attention.py:871-895``) reshapes wq/wk/wv to
    ``(n_kv, *, head_dim, dim)`` and concatenates on dim 1 at ``:893`` -- and
    its save hook reads it back the same way (``:841-867``). That is the
    layout ``megatron_weights.grouped_qkv`` builds. The weight gradient
    therefore needs no conversion here, and the two engines'
    ``qkv_weight_grad`` tensors are comparable element for element.

    **Two implementations of that interleave meet in this scenario**: the
    first-party ``grouped_qkv``, which ``assert_qkv_roundtrip`` proves against
    its own inverse, and torchtitan's upstream hook above, which this arm
    reaches through ``load_state_dict``. They are arithmetically the same cat
    and reshape, and the fp64 gate is what holds them together at run time --
    a divergence would move ``qkv_weight_grad`` on this arm alone and fail it.

    Built from ``FusedQKVLinear.Config`` directly rather than extracted from a
    ``Trainer.Config``. ``make_gqa_config`` adds only ``param_init`` to the
    node (``models/common/config_utils.py:201-206``), which chooses values
    this arm then overwrites from the shared weight state; the ``qkv``
    scenario builds the same module the same way.
    """
    from torchtitan.models.common import FusedQKVLinear, Linear

    qkv_linear = FusedQKVLinear.Config(
        head_dim=shape.head_dim,
        n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads,
        wqkv=Linear.Config(
            in_features=shape.dim, out_features=shape.qkv_out_features
        ),
    ).build()

    def weight_grads(module: nn.Module) -> Callable[[], dict[str, torch.Tensor]]:
        def outputs() -> dict[str, torch.Tensor]:
            return _weight_grad_outputs(
                TITAN_ARM_NAME,
                module.qkv_linear.wqkv.weight.grad,
                module.attention_norm.weight.grad,
            )

        return outputs

    return _build_titan_arm(
        name=TITAN_ARM_NAME,
        shape=shape,
        inputs=inputs,
        qkv_linear=qkv_linear,
        weight_grads=weight_grads,
        notes={
            "qkv_module": "FusedQKVLinear",
            "compiled": True,
            "upstream_default": True,
        },
    )


def build_qkv_prep_titan_unfused_qkv(
    shape: PiperShape, workload: KernelWorkload, inputs: QkvPrepInputs
) -> BuiltArm:
    """TorchTitan with three separate projections, compiled.

    Three GEMMs, not two: ``QKVLinear.__init__`` builds ``wq``, ``wk`` and
    ``wv`` as three separate ``Linear`` modules
    (``models/common/attention.py:725-727``) and calls all three
    (``:733``). ``wk`` and ``wv`` share a config node, which is what makes the
    "Q and KV" phrasing tempting, but they are distinct modules with distinct
    weights and distinct launches.

    **This is not upstream's default, and nothing published from this arm may
    imply that it is.** ``fuse_qkv`` defaults to ``True`` in both qwen3 layer
    builders (``models/qwen3/__init__.py:94`` and ``:139``) and every
    registered qwen3 flavor passes ``fuse_qkv=True`` explicitly. The only
    construction in the
    fork that passes ``fuse_qkv=False`` is ``_debugmodel_non_fused_qkv``
    (``:228``, the flag at ``:233``), whose own comment calls it a way to keep
    exercising the separate wq/wk/wv path. The generic helper
    ``make_gqa_config`` does default the flag to ``False``
    (``models/common/config_utils.py:183``), but no qwen3 caller reaches that
    default.

    The arm exists because the fusion question is worth a number and this
    scenario is the only place the norm sits in front of it. Its opponent is
    ``titan``, not the megatron anchor: fused against unfused is a
    titan-internal question, and running it against megatron would publish a
    ratio in which two things changed at once.

    This arm is the one that has to **convert**: it holds three separate
    weight gradients, and ``grouped_qkv`` -- the same function the weight
    transfer uses -- puts them in megatron's grouped interleave so all three
    arms' ``qkv_weight_grad`` tensors are one comparable object. The other two
    arms hold that layout natively and return it unconverted.
    """
    from torchtitan.models.common import Linear, QKVLinear

    qkv_linear = QKVLinear.Config(
        head_dim=shape.head_dim,
        wq=Linear.Config(
            in_features=shape.dim,
            out_features=shape.n_heads * shape.head_dim,
        ),
        wkv=Linear.Config(
            in_features=shape.dim,
            out_features=shape.n_kv_heads * shape.head_dim,
        ),
    ).build()

    def weight_grads(module: nn.Module) -> Callable[[], dict[str, torch.Tensor]]:
        def outputs() -> dict[str, torch.Tensor]:
            separate = {
                key: getattr(module.qkv_linear, key).weight.grad
                for key in ("wq", "wk", "wv")
            }
            _require_grads(TITAN_UNFUSED_ARM_NAME, separate)
            return _weight_grad_outputs(
                TITAN_UNFUSED_ARM_NAME,
                grouped_qkv(
                    separate["wq"], separate["wk"], separate["wv"], shape
                ),
                module.attention_norm.weight.grad,
            )

        return outputs

    return _build_titan_arm(
        name=TITAN_UNFUSED_ARM_NAME,
        shape=shape,
        inputs=inputs,
        qkv_linear=qkv_linear,
        weight_grads=weight_grads,
        notes={
            "qkv_module": "QKVLinear",
            "compiled": True,
            "upstream_default": False,
        },
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def initialize_megatron_single_rank(seed: int = MCORE_INIT_SEED) -> None:
    """Put megatron's process-global state in place, once per process.

    Megatron needs its checkout on ``sys.path``, a process group, a
    model-parallel state and a seeded CUDA RNG tracker before a ``GPTModel``
    builds; ``benchmarks/e2e/megatron/train.py:137-163`` does the same steps
    for the e2e arm. Each step is guarded, so a second mcore arm in the same
    interpreter -- which the correctness pass builds -- does not repeat it.
    ``model_parallel_cuda_manual_seed`` resets the tracker itself
    (``tensor_parallel/random.py:494``), so it is safe to repeat.

    ``configure_te_environment`` runs before any TransformerEngine import,
    which is what routes the norms through cuDNN on this box.

    NOTE for the registry merge: several Part C scenarios need this function,
    and each currently carries its own copy. This one is verbatim from
    ``ffn_norm.py`` -- but ``final_norm.py`` and ``qk_norm.py`` carry a
    differently named ``_bootstrap_megatron`` that seeds only when it
    initializes model parallel. The hoist must therefore **choose** a seeding
    behaviour and recheck every arm against it, not merge identical text.
    """
    from benchmarks.models.piper_qwen3.megatron_bootstrap import (
        add_megatron_to_path,
        configure_te_environment,
    )

    add_megatron_to_path()
    configure_te_environment()

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)
    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(seed)


def _assert_mcore_qkv_prep(
    attention: Any, shape: PiperShape, eps: float
) -> dict[str, Any]:
    """Refuse to time an attention block that is not the one claimed.

    Every check **raises**. ``BuiltArm.notes`` reaches no artifact -- neither
    ``results.json`` nor ``manifest.json`` carries it -- so a fact recorded
    there is a fact no reader sees, and a guard that cannot raise is not a
    guard.

    The list is not decoration. Each entry names a way this arm could measure
    something other than what its label says, and none of them can be caught
    by a correctness gate, because every one of them is numerically valid:

    * The class, because ``TELayerNormColumnParallelLinear`` is what makes the
      norm fused into the GEMM. A local ``ColumnParallelLinear`` would have a
      separate norm in front of it, and this scenario's whole partition
      argument rests on the fusion.
    * ``normalization``, because RMSNorm and LayerNorm are different
      functions, and TE builds whichever ``config.normalization`` names.
    * ``zero_centered_gamma``, because TE then computes with ``1 + gamma``.
      Copying titan's gain into that parameterization would silently shift
      every gain by one.
    * ``eps``, for the reason ``_titan_norm`` refuses a mismatch.
    * The weight shape, because a wrong navigation would hand back another
      linear of the same model, and because the ``attention_output_gate``
      option widens ``linear_qkv`` (``transformer/attention.py:1679-1681``).
    * ``parallel_mode`` and ``tp_size``, because at ``tp_size > 1`` the
      column-parallel path adds collectives and the ``num_query_groups <
      world_size`` branch at ``transformer/attention.py:1841-1861`` runs an
      all-gather inside the timed call, which the titan arms have no
      counterpart for.
    * ``test_mode``, because it runs ``run_realtime_tests`` -- a pair of
      distributed all-gathers -- inside ``get_query_key_value_tensors``
      (``transformer/attention.py:1928-1929``).
    """
    linear_qkv = attention.linear_qkv
    class_name = type(linear_qkv).__name__
    if class_name != MCORE_MODULE_CLASS:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: linear_qkv is {class_name}, not "
            f"{MCORE_MODULE_CLASS}; the spec derivation took another branch "
            "and the input norm may not be fused into the GEMM at all"
        )
    normalization = getattr(linear_qkv, "normalization", None)
    if normalization != "RMSNorm":
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: linear_qkv normalizes with {normalization!r}, "
            "not 'RMSNorm'; titan's attention_norm is an RMSNorm, so the two "
            "arms would compute different functions"
        )
    if bool(getattr(linear_qkv, "zero_centered_gamma", False)):
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: linear_qkv sets zero_centered_gamma, so it "
            "computes with (1 + gamma); the shared gain would be shifted by "
            "one relative to the titan arms"
        )
    module_eps = getattr(linear_qkv, "eps", None)
    if module_eps is None or float(module_eps) != eps:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: linear_qkv normalizes with eps {module_eps!r} "
            f"and the shared inputs declare {eps!r}"
        )
    expected = (shape.qkv_out_features, shape.dim)
    actual = tuple(linear_qkv.weight.shape)
    if actual != expected:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: linear_qkv.weight is {actual}, expected "
            f"{expected}; the attribute path "
            f"{mcore_attention_path()!r} reached the wrong module, or "
            "attention_output_gate widened the projection"
        )
    norm_weight = getattr(linear_qkv, "layer_norm_weight", None)
    if norm_weight is None or tuple(norm_weight.shape) != (shape.dim,):
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: linear_qkv has no [{shape.dim}] "
            "layer_norm_weight, so the norm this scenario claims to measure "
            "is not inside it"
        )
    parallel_mode = getattr(linear_qkv, "parallel_mode", None)
    tp_size = int(getattr(linear_qkv, "tp_size", 1))
    if parallel_mode != "column":
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: linear_qkv.parallel_mode is "
            f"{parallel_mode!r}, expected 'column'"
        )
    if tp_size != 1 or int(getattr(attention, "world_size", 1)) != 1:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: tp_size {tp_size} / world_size "
            f"{getattr(attention, 'world_size', None)}; above 1 the timed "
            "call holds tensor-parallel collectives the titan arms have no "
            "counterpart for, so the ratio would stop comparing two "
            "projections"
        )
    if bool(getattr(attention.config, "test_mode", False)):
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: config.test_mode is set, so "
            "get_query_key_value_tensors runs run_realtime_tests -- two "
            "distributed all-gathers -- inside the timed call"
        )
    return {
        "module": class_name,
        "normalization": normalization,
        "parallel_mode": parallel_mode,
        "tp_size": tp_size,
        "compiled": False,
    }


def _drop_qk_layernorms(attention: Any) -> None:
    """Remove the qk norms, which scenario 3 owns.

    ``get_query_key_value_tensors`` applies them at
    ``transformer/attention.py:1922-1926``, and the ``qk_norm`` scenario
    already measures exactly those two modules as its own ``mcore/base`` arm.
    The partition allows no overlap, and titan's ``qkv_linear`` returns
    pre-norm q/k/v anyway (``models/common/attention.py:959-960`` applies
    ``q_norm``/``k_norm`` after it), so leaving them in would also make the
    two sides compute different functions.

    ``None`` is megatron's own representation of "no qk norm"
    (``transformer/attention.py:1722,1724-1742``), and the reader branches on
    it, so this removes work without editing megatron's code path.

    Both must be present first. If they are already ``None``, the profile
    turned ``qk_layernorm`` off, which means the built model is not the piper
    model and every other number from it is suspect too.
    """
    for name in ("q_layernorm", "k_layernorm"):
        if getattr(attention, name, None) is None:
            raise RuntimeError(
                f"{MCORE_ARM_NAME}: self_attention.{name} is already None, so "
                "the profile built the model without qk_layernorm; the piper "
                "model has it, and scenario 3 measures it"
            )
        setattr(attention, name, None)


def build_qkv_prep_mcore_base(
    shape: PiperShape, workload: KernelWorkload, inputs: QkvPrepInputs
) -> BuiltArm:
    """Megatron-core's fused norm+QKV linear and its split, eager.

    The attention block is taken out of a real ``GPTModel`` built by
    ``benchmarks.models.piper_qwen3.megatron_model.build_model`` from the same
    ``PiperShape`` and the ``base`` profile the e2e megatron arm uses. A
    hand-constructed ``TELayerNormColumnParallelLinear`` would need every
    argument ``transformer/attention.py:1682-1695`` passes -- ``init_method``,
    ``gather_output``, ``skip_bias_add``, ``tp_comm_buffer_name='qkv'``,
    ``tp_group``, ``pg_collection`` -- written a second time and kept in
    agreement by hand. It would also not give this arm
    ``get_query_key_value_tensors``, which is half of what the scenario
    measures.

    The rest of the model is dropped as soon as the block is extracted, and
    that drop is what keeps ``memory_pass`` honest: resetting the peak counter
    before a call does not exclude resident memory, so a retained model would
    be charged to the arm.

    The weights are copied in before the first forward. TE hands an
    ``is_first_microbatch`` flag down into its own forward and flips it to
    False afterwards, so whatever the first call does extra, it does once,
    from the shared values, and the warmup absorbs it before any burst is
    timed.

    Eager on purpose: megatron compiles no whole transformer layer, so this is
    the treatment megatron gives these modules. ``KernelArm.eager_reason``
    records it, and every table that prints the ratio prints both treatments.
    """
    initialize_megatron_single_rank()

    from benchmarks.models.piper_qwen3.megatron_model import build_model

    model = build_model(seq_len=workload.seq_len, shape=shape, profile=BASE)
    attention = _navigate(model, mcore_attention_path())
    notes = _assert_mcore_qkv_prep(attention, shape, inputs.eps)
    _drop_qk_layernorms(attention)
    linear_qkv = attention.linear_qkv
    print(
        f"qkv_prep/{MCORE_ARM_NAME}: {mcore_attention_path()}.linear_qkv is "
        f"{type(linear_qkv).__module__}.{type(linear_qkv).__qualname__} "
        f"(profile {BASE.name}, cuDNN norm backend, qk norms dropped to "
        f"scenario 3)",
        flush=True,
    )
    with torch.no_grad():
        linear_qkv.weight.copy_(
            grouped_qkv(
                inputs.weight_state["wq.weight"],
                inputs.weight_state["wk.weight"],
                inputs.weight_state["wv.weight"],
                shape,
            )
        )
        linear_qkv.layer_norm_weight.copy_(inputs.norm_weight)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    def call(
        leaf: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # The engine's own entry point, called the way SelfAttention.forward
        # calls it (transformer/attention.py:1814 with output_gate False and
        # split_qkv True, both defaults). It returns query [s, b, n_heads, hd]
        # and key/value [s, b, n_kv_heads, hd].
        query, key, value = attention.get_query_key_value_tensors(leaf)
        return query, key, value

    def weight_grads() -> dict[str, torch.Tensor]:
        return _weight_grad_outputs(
            MCORE_ARM_NAME,
            linear_qkv.weight.grad,
            linear_qkv.layer_norm_weight.grad,
        )

    tokens = workload.batch * workload.seq_len
    # THD, the form our megatron driver runs: the driver packs each batch's
    # rows into one sequence, so hidden_states arrives as (t, 1, h). Built
    # here, never inside a timed closure.
    return _qkv_prep_arm(
        name=MCORE_ARM_NAME,
        owner=linear_qkv,
        call=call,
        x_native=inputs.x.reshape(tokens, 1, shape.dim),
        grads_native=(
            inputs.grad_q.reshape(tokens, 1, shape.n_heads, shape.head_dim),
            inputs.grad_k.reshape(tokens, 1, shape.n_kv_heads, shape.head_dim),
            inputs.grad_v.reshape(tokens, 1, shape.n_kv_heads, shape.head_dim),
        ),
        canonical_in=tuple(inputs.x.shape),
        canonical_out=(
            tuple(inputs.grad_q.shape),
            tuple(inputs.grad_k.shape),
            tuple(inputs.grad_v.shape),
        ),
        weight_grads=weight_grads,
        notes=notes,
    )
