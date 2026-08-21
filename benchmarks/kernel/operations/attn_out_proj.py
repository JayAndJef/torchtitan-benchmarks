"""Arm builders for the ``attn_out_proj`` kernel scenario.

The attention output projection, head to head across the two engines. One
GEMM per side, over the same weight and the same activations:

* TorchTitan: ``self.wo(out_BLD)`` inside ``GQAttention.forward``
  (``third_party/torchtitan/torchtitan/models/common/attention.py:975``).
  ``wo`` is ``torchtitan.models.common.Linear``, which is ``nn.Linear`` with
  ``bias=False`` (``models/common/linear.py:26-40``).
* Megatron-core: ``self_attention.linear_proj``
  (``third_party/Megatron-LM/megatron/core/transformer/attention.py:1618``,
  built at ``:413-426``).

**What ``linear_proj`` actually is.** Under this build it is
``TERowParallelLinear``, a TransformerEngine module and not a megatron-native
``tensor_parallel.RowParallelLinear``. The chain is fixed and has no branch we
do not take: ``megatron_model.build_model`` calls
``get_gpt_decoder_block_spec(config, use_transformer_engine=True)``, which
reaches ``get_gpt_layer_with_transformer_engine_spec`` at
``gpt_layer_specs.py:591`` (the MoE layer spec, which every layer of this
model uses); with ``multi_latent_attention=False`` the submodule factory takes
the ``else`` branch at ``:317`` and writes
``linear_proj=backend.row_parallel_linear()`` at ``:326``; and ``backend`` is
``TESpecProvider`` (``:233``, taken because ``config.use_kitchen`` defaults to
False at ``transformer_config.py:639`` and no profile sets it), whose
``row_parallel_linear`` returns ``TERowParallelLinear``
(``extensions/transformer_engine_spec_provider.py:47-49``).
``TERowParallelLinear`` subclasses ``TELinear``
(``extensions/transformer_engine.py:1867``), which subclasses
``te.pytorch.Linear`` (``:1077``).

**So the two sides differ in host stack as well as in kernel.** TE dispatches
through its own ``torch.autograd.Function``
(``transformer_engine/pytorch/module/linear.py:1290``, applied at ``:1853``)
with quantizer bookkeeping and a per-call ``is_first_microbatch`` flag
(``extensions/transformer_engine.py:1303-1319``), where titan's ``nn.Linear``
calls ``F.linear``. TE also returns a ``(tensor, bias)`` pair, which is the
interface ``SelfAttention.forward`` unpacks, so this arm unpacks it too. None
of that is charged to the other engine, and none of it is hidden: it is what
megatron pays on every layer.

**The row-parallel reduce is absent, not inert.** TE runs its all-reduce or
reduce-scatter only under ``parallel_mode == "row" and args.tp_size > 1``
(``linear.py:512``). At ``world_size=1`` ``tp_size`` is 1, so control falls to
the plain ``else`` and no collective is enqueued. Both arms are therefore one
GEMM and nothing else. Neither side is the identity.

**The layout op between attention and this projection is excluded, on both
engines.** Titan runs ``out_BLNH.contiguous()`` then ``.view(B, L, -1)``
before ``wo`` (``attention.py:972-974``), and the ``contiguous`` is a real
copy because ``FlexAttention.forward`` returns a transposed view (``:353``).
Megatron runs ``core_attn_out.reshape(t, 1, -1)`` before ``linear_proj``
(``attention.py:1598-1603``), which is free. Three reasons put both outside
this scenario. First, neither op is inside the module under test. Second, the
titan cost belongs to the *attention arm* that produced the tensor -- the
attention_core scenario declares three titan arms with three output layouts,
and
this scenario declares one -- so it cannot be attributed here. Third, it
cannot be made symmetric by construction: adding a copy to megatron would
invent work megatron does not do, and dropping titan's would hide work titan
does do. The piece is therefore unassigned, and the report records it as such
rather than charging it to one engine.

**Both arms see the same GEMM, in two shapes.** The canonical activation is
``[B, L, n_heads*head_dim]``. The mcore arm takes the THD form our megatron
driver produces, ``[B*L, 1, n_heads*head_dim]``; both flatten to the same
``[B*L, in_features]`` matrix inside the linear -- TE flattens and then
restores the logical rank at ``linear.py:530``, and ``F.linear`` folds the
leading dims -- so the shape is a label and not a measured difference. Every
reshape happens at build time. Outputs are canonicalized back to
``[B, L, dim]`` in ``correctness_outputs`` only.

**There is no isolated ``backward`` mode, and there cannot be one.** The
retained-graph trick the other module scenarios use -- run
``torch.autograd.backward(..., retain_graph=True)`` repeatedly over one graph
-- fails on TE: ``_Linear.backward`` clears ``ctx.backward_objects`` on its
way out (``linear.py:1365``, so the saved tensors do not outlive backward
under ``retain_graph``), and a second pass reads ``None``. Both arms drop the
mode, which keeps them comparable; backward cost is still recoverable as
``forward_backward`` minus ``forward``. ``attention.py`` in this package drops
it for the same class of reason.

Every torchtitan and megatron import is deferred into a builder body, so a
process that measures one arm never imports the other arm's stack.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    WEIGHT_STD,
    _compile_module,
    _navigate,
    _projection_arm,
    _randn,
    _reset_grads,
    initialize_megatron_single_rank,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.shape import PiperShape


# The two arm names, spelled as plan section C.2 spells them. The slash is
# load-bearing: it says which engine an arm is and which profile of that
# engine, which a cross-engine roster needs and a single-engine one does not.
# It reaches a filename through ``benchmarks/kernel/runner.py``'s
# ``fragment_path``, which routes it through ``schema.fragment_stem`` for
# exactly this reason. **A tree without ``fragment_stem`` cannot run this
# scenario**: the timing worker would write into a ``timing__mcore`` directory
# nobody creates, and the arm would land as ``failed``.
MCORE_ARM_NAME = "mcore/base"
TITAN_ARM_NAME = "titan"

# The layer both engines take the module from. Any layer would do -- every
# block holds the same projection at the same shape -- and the weight is
# overwritten from the shared seeded tensor, so the depth-scaled parameter
# init both engines apply never reaches a measurement.
MCORE_LAYER = 0

# The two parameter names ``benchmarks.models.piper_qwen3.megatron_weights``
# pairs under its ``attn_out`` component. That module is the authority on the
# titan-to-megatron correspondence, and this scenario does not reimplement it:
# ``tests/test_kernel_attn_out_proj.py`` asserts the map still yields exactly
# this pair, and still yields it as a straight copy with no reshape. A
# transfer for one parameter cannot go through ``transfer_weights`` itself,
# which needs a whole titan state dict, so the test is what ties the two.
TITAN_WEIGHT_NAME = "layers.{layer}.attention.wo.weight"
MCORE_WEIGHT_NAME = "decoder.layers.{layer}.self_attention.linear_proj.weight"
MCORE_WEIGHT_COMPONENT = "attn_out"

# The class the mcore arm must build, checked rather than assumed. Four names
# in this system overclaim "TE"; this one is TE, and the guard is what lets a
# published table say so.
MCORE_MODULE_CLASS = "TERowParallelLinear"


def mcore_module_path(layer: int = MCORE_LAYER) -> str:
    """The attribute path from a ``GPTModel`` down to the module under test.

    Derived from ``MCORE_WEIGHT_NAME`` rather than written beside it, so the
    navigation and the weight map can only agree.
    """
    name = MCORE_WEIGHT_NAME.format(layer=layer)
    assert name.endswith(".weight")
    return name[: -len(".weight")]


@dataclass
class AttnOutProjInputs:
    x: torch.Tensor  # (B, L, n_heads*head_dim) bf16, the canonical activation
    grad_out: torch.Tensor  # (B, L, dim) bf16
    weight: torch.Tensor  # (dim, n_heads*head_dim) fp32, shared by both arms


def attn_out_proj_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> AttnOutProjInputs:
    """The one activation, gradient and weight both engines are handed.

    ``in_features`` is ``n_heads * head_dim``, which is what megatron calls
    ``query_projection_size`` (``transformer/attention.py:333``) and what
    ``make_gqa_config`` gives ``wo`` (``models/common/config_utils.py:231``).
    It equals ``dim`` at every registered shape, because ``PiperShape``
    derives ``n_heads`` from ``dim // head_dim``; it is spelled out anyway, so
    a future shape that breaks the equality breaks nothing here.
    """
    in_features = shape.n_heads * shape.head_dim
    return AttnOutProjInputs(
        x=_randn(
            (workload.batch, workload.seq_len, in_features), device, generator
        ),
        grad_out=_randn(
            (workload.batch, workload.seq_len, shape.dim), device, generator
        ),
        weight=_randn(
            (shape.dim, in_features),
            device,
            generator,
            torch.float32,
            WEIGHT_STD,
        ),
    )


def attn_out_proj_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: AttnOutProjInputs
) -> dict[str, torch.Tensor]:
    """fp64 truth for the projection and both gradients.

    The weight is quantized to bf16 before it is promoted to fp64, so the
    truth is the truth for the weight the arms actually hold. Comparing
    against an fp64 projection of the unrounded weight would charge both arms
    for the input cast, which is the same correction ``qkv_reference`` makes.
    """
    in_features = shape.n_heads * shape.head_dim
    x64 = inputs.x.double()
    grad64 = inputs.grad_out.double()
    weight64 = inputs.weight.to(torch.bfloat16).double()
    return {
        "out": F.linear(x64, weight64),
        "x_grad": grad64 @ weight64,
        "weight_grad": grad64.reshape(-1, shape.dim).transpose(0, 1)
        @ x64.reshape(-1, in_features),
    }


def build_attn_out_proj_titan(
    shape: PiperShape, workload: KernelWorkload, inputs: AttnOutProjInputs
) -> BuiltArm:
    """TorchTitan's ``wo``: ``nn.Linear(bias=False)``, compiled.

    Built directly from ``Linear.Config`` rather than extracted from a
    ``Trainer.Config``. The registry gives ``wo`` exactly three things --
    ``in_features``, ``out_features`` and a ``param_init``
    (``third_party/torchtitan/torchtitan/models/common/config_utils.py:230-234``)
    -- and ``param_init`` only chooses values, which a GEMM's cost does not
    depend on and which this arm overwrites from the shared seeded weight. A
    trainer-config extraction would add a whole model build and change
    nothing that is measured.

    ``torch.compile(fullgraph=True)`` is the production treatment for a titan
    module arm, and the mcore arm opposite it runs eager because megatron
    compiles no whole layer. Every table that prints this ratio must print
    both treatments.
    """
    from torchtitan.models.common import Linear

    module = Linear.Config(
        in_features=shape.n_heads * shape.head_dim,
        out_features=shape.dim,
    ).build()
    module.to(inputs.x.device)
    module.load_state_dict({"weight": inputs.weight})
    module.to(torch.bfloat16)

    compiled = _compile_module(module)

    def call(leaf: torch.Tensor) -> torch.Tensor:
        return compiled(leaf)

    return _projection_arm(
        name=TITAN_ARM_NAME,
        weight_owner=module,
        call=call,
        x_native=inputs.x,
        grad_native=inputs.grad_out,
        canonical_in=tuple(inputs.x.shape),
        canonical_out=tuple(inputs.grad_out.shape),
        notes={"module": type(module).__name__, "compiled": True},
    )


def _assert_mcore_linear_proj(module: object, shape: PiperShape) -> dict[str, Any]:
    """Refuse to time a module that is not the one the scenario claims.

    Four things are checked, and every one of them **raises**. Nothing here is
    merely noted: ``BuiltArm.notes`` reaches no artifact -- neither
    ``results.json`` nor ``manifest.json`` carries it -- so a fact recorded
    there is a fact no reader ever sees. A guard that cannot raise is not a
    guard.

    The class, because "TE" is overclaimed elsewhere in this system and a
    reader must be able to trust the label here. The weight shape, because a
    wrong navigation would otherwise hand back some other linear of the same
    model. ``parallel_mode`` and ``tp_size``, because together they decide
    whether TE enqueues a row-parallel collective (``linear.py:512``). At
    ``tp_size == 1`` it does not, which is the case this scenario measures and
    the case a kernel worker can produce. At ``tp_size > 1`` the timed call
    would hold a collective the titan arm has no counterpart for, so the ratio
    would stop being a comparison of two GEMMs -- that raises rather than
    running.
    """
    class_name = type(module).__name__
    if class_name != MCORE_MODULE_CLASS:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: linear_proj is {class_name}, not "
            f"{MCORE_MODULE_CLASS}; the spec derivation took another branch "
            "and this arm would publish a different implementation"
        )
    expected = (shape.dim, shape.n_heads * shape.head_dim)
    actual = tuple(module.weight.shape)
    if actual != expected:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: linear_proj.weight is {actual}, expected "
            f"{expected}; "
            f"the attribute path {mcore_module_path()!r} reached the wrong "
            "module"
        )
    parallel_mode = getattr(module, "parallel_mode", None)
    tp_size = int(getattr(module, "tp_size", 1))
    if parallel_mode != "row":
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: linear_proj.parallel_mode is "
            f"{parallel_mode!r}, "
            "expected 'row'"
        )
    if tp_size != 1:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: linear_proj.tp_size is {tp_size}, expected 1; "
            "a row-parallel all-reduce would run inside the timed call "
            "(linear.py:512) and the titan arm has no counterpart for it, so "
            "the ratio would no longer compare two GEMMs"
        )
    return {
        "module": class_name,
        "parallel_mode": parallel_mode,
        "tp_size": tp_size,
        "row_parallel_reduce_runs": tp_size > 1,
        "compiled": False,
    }


def build_attn_out_proj_mcore_base(
    shape: PiperShape, workload: KernelWorkload, inputs: AttnOutProjInputs
) -> BuiltArm:
    """Megatron-core's ``self_attention.linear_proj``, eager.

    The module is taken out of a real ``GPTModel`` built by
    ``benchmarks.models.piper_qwen3.megatron_model.build_model`` from the same
    ``PiperShape`` and the ``base`` profile the e2e megatron arm uses. A
    hand-constructed ``TERowParallelLinear`` would need every argument
    ``transformer/attention.py:413-426`` passes -- ``init_method``,
    ``input_is_parallel``, ``skip_bias_add``, ``tp_comm_buffer_name='proj'``,
    ``tp_group``, ``pg_collection`` -- written a second time and kept in
    agreement by hand, which is the duplication ``mcore_profiles.py`` records
    as the reason the layer spec is derived rather than written.

    The rest of the model is dropped as soon as the projection is extracted,
    and that drop is what keeps ``memory_pass`` honest. Resetting the peak
    counter before a call (``engine/measurement.py:147``) does **not** exclude
    resident memory: ``max_memory_allocated`` is a maximum over time of the
    bytes currently allocated, so the first allocation inside the timed call
    lifts the peak to resident-plus-new. A retained model would therefore be
    charged to the arm. It holds ~2.1 GiB of bf16 parameters at the normal
    shape and ~21 GiB at ``huge``, in a process that the correctness pass also
    asks to build every other arm of the scenario, so dropping it buys
    headroom as well.

    The weight is copied in before the first forward, which matters on this
    module and on no other in the package. TE passes an ``is_first_microbatch``
    flag down into its own forward and flips it to False afterwards
    (``extensions/transformer_engine.py:1305-1312``); the flag is True on the
    first call because ``config.disable_parameter_transpose_cache`` defaults to
    False (``transformer_config.py:1103``) and no profile sets it, and nothing
    here calls ``set_is_first_microbatch``, so it stays False for the rest of
    the process. Whatever that first call does extra, it does once, from the
    seeded weight, and the warmup absorbs it before any burst is timed.
    """
    initialize_megatron_single_rank()

    from benchmarks.models.piper_qwen3.mcore_profiles import BASE
    from benchmarks.models.piper_qwen3.megatron_model import build_model

    model = build_model(
        seq_len=workload.seq_len,
        shape=shape,
        profile=BASE,
    )
    linear_proj = _navigate(model, mcore_module_path())
    notes = _assert_mcore_linear_proj(linear_proj, shape)
    with torch.no_grad():
        linear_proj.weight.copy_(inputs.weight)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    def call(leaf: torch.Tensor) -> torch.Tensor:
        # (output, bias). SelfAttention.forward unpacks the pair exactly here
        # (transformer/attention.py:1618). bias is None: the profile sets
        # add_bias_linear False, so te_return_bias is False
        # (extensions/transformer_engine.py:1127) and the forward returns
        # (out, None) at :1319.
        output, _bias = linear_proj(leaf)
        return output

    tokens = workload.batch * workload.seq_len
    in_features = shape.n_heads * shape.head_dim
    # THD, the form our megatron driver runs: the driver packs each batch's
    # rows into one sequence, and Attention.forward reshapes the core-attention
    # output to (t, b=1, h) before linear_proj (transformer/attention.py:
    # 1598-1603). Built here, never inside a timed closure.
    return _projection_arm(
        name=MCORE_ARM_NAME,
        weight_owner=linear_proj,
        call=call,
        x_native=inputs.x.reshape(tokens, 1, in_features),
        grad_native=inputs.grad_out.reshape(tokens, 1, shape.dim),
        canonical_in=tuple(inputs.x.shape),
        canonical_out=tuple(inputs.grad_out.shape),
        notes=notes,
    )
