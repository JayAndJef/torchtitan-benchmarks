"""Arm builders for the ``lm_head_projection`` kernel scenario.

The output projection at the bottom of the model, head to head across the two
engines. One GEMM per side, over one shared ``[vocab_size, dim]`` weight:

* TorchTitan: ``self.lm_head(h)`` inside ``Decoder.forward``
  (``third_party/torchtitan/torchtitan/models/common/decoder.py:284``).
  ``lm_head`` is ``torchtitan.models.common.Linear``, which is ``nn.Linear``
  with ``bias=False`` (``models/common/linear.py:26-40``). Our registry sets
  that config node at
  ``benchmarks/models/piper_qwen3/config_registry.py:123-127``.
* Megatron-core: ``GPTModel.output_layer``, built at
  ``third_party/Megatron-LM/megatron/core/models/gpt/gpt_model.py:264-286``
  and called at ``:762``.

The scenario publishes one ratio, titan against megatron.

**The megatron arm is NOT TransformerEngine, and the name ``mcore/base`` must
not be read as "TE".** ``gpt_model.py:264-267`` chooses
``TELMHeadColumnParallelLinear`` only when ``is_mxfp8_output_proj_active
(config)`` is true, and that predicate
(``megatron/core/fp8_utils.py:732-747``) needs TransformerEngine present
**and** ``config.fp8_output_proj`` **and** ``config.fp8`` **and** an mxfp8
recipe. The ``base`` profile
(``benchmarks/models/piper_qwen3/mcore_profiles.py``) sets none of the three
config fields, so the ``else`` branch runs and the module is megatron-native
``tensor_parallel.ColumnParallelLinear``. ``_assert_mcore_output_layer``
checks the class by name and **raises**, which is what lets a published table
say so. Note the process still imports TransformerEngine, because
``build_model`` derives the layer spec with ``use_transformer_engine=True``;
the import is not the claim. The module under test is the claim, and it is
torch.

**So the two sides differ in host stack, and here they agree on the kernel.**
Both engines end at one ``torch.matmul`` over the same two matrices. Megatron
reaches it through two ``torch.autograd.Function.apply`` calls that titan has
no counterpart for: ``copy_to_tensor_model_parallel_region``
(``tensor_parallel/mappings.py:492-495``, which is
``_CopyToModelParallelRegion.apply``, the class at ``:201-218``) and
``linear_with_grad_accumulation_and_async_allreduce``
(``tensor_parallel/layers.py:754``, ``.apply`` at ``:862``, the Function at
``:524``, whose forward runs ``torch.matmul(total_input, weight.t())`` at
``:575``). The second is additionally wrapped in ``torch.amp.custom_fwd`` and
``custom_bwd`` (aliased at ``layers.py:74-75``, applied at ``:528`` and
``:581``). Titan calls ``F.linear``. That cost is
real, megatron pays it on every step, and it is named here rather than left
to a reader.

**No collective runs, and the extra Function is still not free.** At
``world_size == 1`` every parallel branch is inert:
``_CopyToModelParallelRegion.forward`` is the identity (``mappings.py:
210-213``) and its backward's ``_reduce`` returns its input unchanged at group
size 1 (``mappings.py:27-28``); ``allreduce_dgrad`` is False because it needs
``world_size > 1`` (``layers.py:1064-1066``); ``sequence_parallel`` is False;
``explicit_expert_comm`` is False because ``is_expert`` is False; and
``gather_output`` is False because ``GPTModel`` passes ``gather_output=not
self.parallel_output`` (``gpt_model.py:280``) and ``parallel_output`` defaults
True (``:103``). The guard raises on each of those rather than noting it.
Inert is not absent: the two ``apply`` calls happen, and they are the
asymmetry above.

**Both arms are eager, and on the titan side that corrects the usual rule.**
Megatron compiles no whole transformer layer, so everything ``GPTModel``
builds runs eager end to end. TorchTitan's ``lm_head`` is a *sibling* of
``layers`` -- ``Decoder.__init__`` builds ``self.layers`` at ``decoder.py:
236`` and ``self.lm_head`` at ``:241`` -- and ``apply_compile`` walks
``model.layers.named_children()`` alone
(``third_party/torchtitan/torchtitan/distributed/compile.py:57-58``), so the
projection sits outside every compiled region in production. ``"model" in
compile_config.components`` gates ``apply_compile`` and nothing else
(``models/qwen3/parallelize.py:41-43,67-68``). Compiling this arm would
measure a treatment production never applies to it. ``final_norm`` and
``embedding_stage`` carry the same correction, for the same structural
reason, and it does not extend to the scenarios whose modules live inside a
block.

**One titan configuration does compile a projection, and it is not this
one.** ``CompileConfig.components`` defaults to ``["model", "loss"]``
(``torchtitan/config/configs.py:267``), and under the
``qwen3_piper_1b_fused_linear_ce`` config the loss owns the LM head through
the ``LossWithLMHead`` protocol (``torchtitan/trainer.py:444-466``), so the
projection runs inside the compiled loss. That arrangement is a **different
cut**: it fuses the projection into the cross entropy and never materializes
the logits. It belongs to a span that covers the projection and the loss
together, not to this scenario. Every arm here materializes ``[batch,
seq_len, vocab_size]`` logits, which is what makes the two engines
substitutable at this boundary.

**Both arms see the same GEMM, in two shapes.** The canonical activation is
``[batch, seq_len, dim]``. The mcore arm takes the THD form our megatron
driver produces, ``[batch*seq_len, 1, dim]``: the driver packs a batch as one
row (``benchmarks/e2e/megatron/data.py:54``), and megatron runs
``[sequence, batch, hidden]``. Both flatten to the same ``[batch*seq_len,
dim]`` matrix inside the linear -- ``torch.matmul`` folds the leading
dimensions, and ``F.linear`` does the same -- so the shape is a label and not
a measured difference, exactly as ``attn_out_proj`` and ``qkv_prep`` record
of the identical reshape. Every reshape happens at build time. Outputs are
canonicalized back to ``[batch, seq_len, vocab_size]`` in
``correctness_outputs`` only, which runs outside every timed region.

Two facts about that layout are worth stating rather than leaving implicit.
The reshape of a contiguous tensor is a contiguous view, so megatron's
``prepare_input_tensors_for_wgrad_compute`` (``megatron/core/utils.py:
1205-1222``) finds both of its tensors contiguous and its two
``.contiguous()`` calls return ``self``.

**And the ``cross_entropy`` scenario orders its rows differently, so do not
chain the two tensors.** Scenario 16 records its mcore logits as ``[seq_len,
batch, vocab_size]``; this one produces ``[batch*seq_len, 1, vocab_size]``.
Those are two different permutations of the same rows. Scenario 16
transposes the canonical ``[batch, seq_len, vocab_size]``, which puts the
rows in sequence-major order. This scenario reshapes it, which keeps them in
batch-major order. The two agree at ``batch == 1`` and disagree above it.
Neither is wrong: scenario 16 builds megatron's own SBHD convention, and
this scenario builds what our THD driver delivers
(``benchmarks/e2e/megatron/data.py:54``). A reader may chain the two
scenarios' *costs*. A reader may not treat one scenario's output tensor as
the other's input.

**This scenario is compute-bound, and ``attn_out_proj`` is not.** The
forward is ``2 * batch * seq_len * dim * vocab_size`` = 1.27 TFLOP at the
default workload and the ``normal`` shape, against 8.6 GFLOP for the
attention output projection at the same workload -- a factor of 148 for the
same tensor rank. At ``huge`` it is 15.3 TFLOP. The scenario therefore
declares **no bandwidth floor**: a copy floor answers "is this arm moving
bytes at device speed", which is the wrong question for a GEMM. The right
reference is a FLOP roofline, which nothing in this repository computes.

**The fp64 gate is the largest allocation in this package.** The logits are
``[batch, seq_len, vocab_size]``: 1.16 GiB of bf16 at the default workload,
and 4.64 GiB once the reference promotes them to fp64. The reference also
holds an fp64 weight gradient, 1.16 GiB at ``normal`` and 13.9 GiB at
``huge``, and the correctness pass holds the whole reference while it builds
each arm. Adding the two arms' outputs and the gates' fp32 temporaries gives
roughly 16 GiB at ``normal`` and 52 GiB at ``huge`` for that pass. **Those
two totals are arithmetic, not a measurement**; replace them with a real run
before reporting anything about them. A timing worker is far smaller: it
holds one arm and no reference.

**There is no isolated ``backward`` mode.** ``forward`` and
``forward_backward`` are what every cross-engine scenario in this package
declares, and ``attn_out_proj`` cannot declare a third because
TransformerEngine clears its saved context. Keeping the same two modes here
keeps the roster uniform. Backward cost stays recoverable as
``forward_backward`` minus ``forward``.

Every torchtitan and megatron import is deferred into a builder body, so a
process that measures one arm never imports the other arm's stack.

``_projection_arm`` and ``_navigate`` come from
``benchmarks/kernel/operations/common.py``, which ``attn_out_proj.py`` and
``qkv_prep.py`` read them from too. This module held a copy of each while the
four Part C branches ran in parallel; the merge removed both copies.
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
    MCORE_BLANK_MLP,
    _navigate,
    _projection_arm,
    _randn,
    _reset_grads,
    initialize_megatron_single_rank,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.shape import PiperShape


# The two arm names, spelled as plan section C.2 spells them. The slash says
# which engine an arm is and which profile of that engine. It reaches a
# filename through ``benchmarks/kernel/runner.py``'s ``fragment_path``, which
# routes it through ``schema.fragment_stem``.
MCORE_ARM_NAME = "mcore/base"
TITAN_ARM_NAME = "titan"

# The two parameter names ``benchmarks.models.piper_qwen3.megatron_weights``
# pairs under its ``lm_head`` component. That module is the authority on the
# titan-to-megatron correspondence and this scenario does not reimplement it:
# ``tests/test_kernel_lm_head_projection.py`` asserts the map still yields
# exactly this pair, and still yields it as a straight copy with no reshape.
# A transfer for one parameter cannot go through ``transfer_weights`` itself,
# which needs a whole titan state dict, so the test is what ties the two.
TITAN_WEIGHT_NAME = "lm_head.weight"
MCORE_WEIGHT_NAME = "output_layer.weight"
MCORE_WEIGHT_COMPONENT = "lm_head"

# The classes each arm must build, checked rather than assumed. The megatron
# one is the load-bearing check of this scenario: an arm named ``mcore/base``
# reads as TransformerEngine, and this one is not.
MCORE_MODULE_CLASS = "ColumnParallelLinear"
TITAN_MODULE_CLASS = "Linear"


def mcore_module_path() -> str:
    """The attribute path from a ``GPTModel`` down to the module under test.

    Derived from ``MCORE_WEIGHT_NAME`` rather than written beside it, so the
    navigation and the weight map can only agree. There is one output layer
    whatever ``n_layers`` is, so -- unlike ``attn_out_proj`` -- the path holds
    no layer index and this scenario does no arithmetic over the layer count.
    """
    assert MCORE_WEIGHT_NAME.endswith(".weight")
    return MCORE_WEIGHT_NAME[: -len(".weight")]


@dataclass
class LmHeadProjectionInputs:
    x: torch.Tensor  # (B, L, dim) bf16, the canonical hidden state
    grad_out: torch.Tensor  # (B, L, vocab_size) bf16
    weight: torch.Tensor  # (vocab_size, dim) fp32, shared by both arms


def lm_head_projection_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> LmHeadProjectionInputs:
    """The one hidden state, gradient and weight both engines are handed.

    The weight is drawn at ``1 / sqrt(dim)`` rather than at this package's
    ``WEIGHT_STD``, and stays fp32 so each arm casts it once. Both choices are
    deliberate. The scale is the one the existing ``lm_head`` scenario already
    uses for this same weight (``lm_head_inputs``), and it is what puts the
    logits at unit scale at every shape -- which is the scale the
    ``cross_entropy`` scenario says its own logits are drawn at, so the
    producer and the consumer of a logit tensor agree. A fixed standard
    deviation would put the logits at 0.64 at ``normal`` and 2.2 at ``huge``.

    Values reach no published number here: a GEMM costs what it costs, and
    every gate is ``max_rel_l2``, which is scale-invariant.
    """
    return LmHeadProjectionInputs(
        x=_randn(
            (workload.batch, workload.seq_len, shape.dim), device, generator
        ),
        grad_out=_randn(
            (workload.batch, workload.seq_len, shape.vocab_size),
            device,
            generator,
        ),
        weight=_randn(
            (shape.vocab_size, shape.dim),
            device,
            generator,
            torch.float32,
            1.0 / shape.dim**0.5,
        ),
    )


def lm_head_projection_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: LmHeadProjectionInputs
) -> dict[str, torch.Tensor]:
    """fp64 truth for the projection and both gradients.

    The weight is quantized to bf16 before it is promoted to fp64, so the
    truth is the truth for the weight the arms actually hold. An fp64
    projection of the unrounded weight would charge both arms for the input
    cast, which is the same correction ``attn_out_proj_reference`` and
    ``qkv_reference`` make.

    This function is the memory ceiling of the scenario. See the module
    docstring for the arithmetic.
    """
    x64 = inputs.x.double()
    grad64 = inputs.grad_out.double()
    weight64 = inputs.weight.to(torch.bfloat16).double()
    return {
        "out": F.linear(x64, weight64),
        "x_grad": grad64 @ weight64,
        "weight_grad": grad64.reshape(-1, shape.vocab_size).transpose(0, 1)
        @ x64.reshape(-1, shape.dim),
    }


def titan_lm_head_module(shape: PiperShape, device: torch.device):
    """The ``Decoder.lm_head`` config node, built as a module.

    ``Decoder.__init__`` builds it with ``self.lm_head = config.lm_head.build
    ()`` (``third_party/torchtitan/torchtitan/models/common/decoder.py:241``),
    and ``_piper_1b_model`` is where our registry sets that node
    (``benchmarks/models/piper_qwen3/config_registry.py:123-127``). So this is
    the production module built by the production config, and the extraction
    is one attribute read.

    Built inside ``with device`` and at a bf16 default dtype rather than built
    and then moved, which is what ``titan_embedding_module`` does and for the
    same reason: ``nn.Linear.__init__`` calls ``reset_parameters`` over
    ``[vocab_size, dim]``, which on the host is 594 MiB and several seconds at
    ``normal`` and 7.0 GiB at ``huge``. Those values are overwritten with the
    shared weight immediately afterwards.

    ``param_init`` on the config node is never applied, because nothing here
    calls ``init_weights``. It only chooses values, which a GEMM's cost does
    not depend on and which this arm overwrites.
    """
    from benchmarks.models.piper_qwen3.config_registry import _piper_1b_model

    node = _piper_1b_model(fuse_qkv=True, shape=shape).lm_head
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with device:
            return node.build()
    finally:
        torch.set_default_dtype(previous_dtype)


def _assert_titan_lm_head(module: object, shape: PiperShape) -> dict[str, Any]:
    """Refuse to time a titan head that is not the plain projection.

    The class, because ``Linear`` has a subclass in the same module
    (``ScaledBiasRowwiseLinear``, ``models/common/linear.py:43``) whose forward
    divides the bias by a TP degree. The weight shape, because a config change
    would otherwise hand this arm some other linear.

    The weight dtype, because no correctness gate can see it. This arm takes
    its dtype from ``titan_lm_head_module``'s own ``set_default_dtype`` call
    rather than from a config, so it is the likelier of the two arms to lose
    the cast. An uncast weight times an fp32 GEMM against a bf16 one and
    still passes the 2e-2 gate, because the bf16 quantization of ``x``
    dominates rel_l2.

    The bias, because megatron builds its output layer with ``bias=False``
    (``gpt_model.py:278``) and a bias add here would be work with no
    opponent.
    """
    class_name = type(module).__name__
    if class_name != TITAN_MODULE_CLASS:
        raise RuntimeError(
            f"{TITAN_ARM_NAME}: lm_head is {class_name}, not "
            f"{TITAN_MODULE_CLASS}; the config node no longer builds the "
            "module this arm claims to time"
        )
    expected = (shape.vocab_size, shape.dim)
    actual = tuple(module.weight.shape)
    if actual != expected:
        raise RuntimeError(
            f"{TITAN_ARM_NAME}: lm_head.weight is {actual}, expected "
            f"{expected}"
        )
    if module.weight.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{TITAN_ARM_NAME}: lm_head.weight is {module.weight.dtype}, "
            "expected torch.bfloat16; the arm would time a GEMM at another "
            "precision and no correctness gate can see it -- the bf16 "
            "quantization of the activation dominates rel_l2"
        )
    if module.bias is not None:
        raise RuntimeError(
            f"{TITAN_ARM_NAME}: lm_head carries a bias; megatron builds its "
            "output layer with bias=False, so the arms would no longer "
            "compute the same function"
        )
    return {
        "module": class_name,
        "transformer_engine": False,
        "compiled": False,
    }


def build_lm_head_projection_titan(
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: LmHeadProjectionInputs,
) -> BuiltArm:
    """TorchTitan's ``lm_head``: ``nn.Linear(bias=False)``, eager.

    Eager is the production treatment for this module and not a relaxation of
    the package's compile rule. See the module docstring: ``apply_compile``
    cannot reach a sibling of ``model.layers``.
    """
    module = titan_lm_head_module(shape, inputs.x.device)
    notes = _assert_titan_lm_head(module, shape)
    with torch.no_grad():
        module.weight.copy_(inputs.weight)

    def call(leaf: torch.Tensor) -> torch.Tensor:
        return module(leaf)

    return _projection_arm(
        name=TITAN_ARM_NAME,
        weight_owner=module,
        call=call,
        x_native=inputs.x,
        grad_native=inputs.grad_out,
        canonical_in=tuple(inputs.x.shape),
        canonical_out=tuple(inputs.grad_out.shape),
        notes=notes,
    )


def _assert_mcore_output_layer(
    module: object, shape: PiperShape
) -> dict[str, Any]:
    """Refuse to time a module that is not the one the scenario claims.

    Ten things are checked, and every one of them **raises**. Nothing here
    is merely noted: ``BuiltArm.notes`` reaches no artifact -- neither
    ``results.json`` nor ``manifest.json`` carries it -- so a fact recorded
    there is a fact no reader ever sees. A guard that cannot raise is not a
    guard.

    The class first, because it is this scenario's headline claim. An arm
    named ``mcore/base`` reads as TransformerEngine. This module is
    megatron-native, and the mxfp8 branch at ``gpt_model.py:264-267`` is the
    only thing that could make it TE.

    Then the weight shape, because a wrong navigation would hand back some
    other linear of the same model.

    Then the weight dtype. The gate cannot see this one. Both arms cast the
    shared fp32 weight, and an arm that failed to cast would time an fp32
    GEMM against a bf16 one. Measured on CPU at dim 1024, an uncast weight
    gives ``out`` a rel_l2 of 1.65e-3 and ``weight_grad`` 1.3e-8, well inside
    the 2e-2 gate: the bf16 quantization of ``x`` dominates the metric, so a
    reduction gate is blind to the weight's dtype. Only a guard catches it.

    Then four parallel-mode facts, because each of them would put work inside
    the timed call that the titan arm has no counterpart for.
    ``output_size_per_partition`` equals ``output_size`` only at tensor
    parallel size 1 (``layers.py:966``), so it is the shard check.
    ``gather_output`` would add an all-gather after the GEMM,
    ``sequence_parallel`` an all-gather before it and a reduce-scatter in
    backward, and ``allreduce_dgrad`` an all-reduce of the input gradient.

    Then two facts of a different kind, each of which breaks the wgrad gate
    rather than adding work. With ``gradient_accumulation_fusion`` on,
    backward writes the weight gradient into a ``main_grad`` buffer
    (``layers.py:655``) instead of running the plain
    ``grad_output.t().matmul(total_input)`` at ``:724``, so
    ``correctness_outputs`` would read ``None``. Megatron's own ``__init__``
    also raises on it without apex (``layers.py:1068-1077``), and apex is
    absent here. With ``defer_embedding_wgrad_compute`` on, ``forward``
    appends the input to ``embedding_activation_buffer`` on every call
    (``layers.py:1147-1152``), which grows without bound across a burst of
    thousands, and backward sets ``wgrad_compute=False``
    (``layers.py:603-606``), so ``.grad`` stays ``None`` for the same reason.
    ``ModelParallelConfig`` defaults it False (``model_parallel_config.py:
    410``) and the ``base`` profile never sets it, so this check is
    unreachable today. It is checked anyway, because the guard reads module
    attributes and never the config, and because it is the last remaining
    route to the failure the fusion check exists to prevent.

    Last the bias, because megatron builds this layer with ``bias=False``
    (``gpt_model.py:278``) and titan's head has none either, so a bias add
    would be work with no opponent.
    """
    class_name = type(module).__name__
    if class_name != MCORE_MODULE_CLASS:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: output_layer is {class_name}, not "
            f"{MCORE_MODULE_CLASS}; the mxfp8 branch at gpt_model.py:264-267 "
            "was taken and this arm would publish a TransformerEngine module "
            "under a megatron-native label"
        )
    expected = (shape.vocab_size, shape.dim)
    actual = tuple(module.weight.shape)
    if actual != expected:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: output_layer.weight is {actual}, expected "
            f"{expected}; the attribute path {mcore_module_path()!r} reached "
            "the wrong module"
        )
    if module.weight.dtype != torch.bfloat16:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: output_layer.weight is "
            f"{module.weight.dtype}, expected torch.bfloat16; the arm would "
            "time a GEMM at another precision and no correctness gate can "
            "see it -- the bf16 quantization of the activation dominates "
            "rel_l2"
        )
    partition = int(module.output_size_per_partition)
    if partition != shape.vocab_size:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: output_layer holds {partition} of "
            f"{shape.vocab_size} rows, so tensor parallel size is above 1 and "
            "this arm would measure one shard of a projection the titan arm "
            "computes whole"
        )
    for attribute, wanted, why in (
        (
            "gather_output",
            False,
            "an all-gather would run after the GEMM",
        ),
        (
            "sequence_parallel",
            False,
            "an all-gather would run before the GEMM and a reduce-scatter "
            "inside backward",
        ),
        (
            "allreduce_dgrad",
            False,
            "an all-reduce of the input gradient would run inside backward",
        ),
        (
            "gradient_accumulation_fusion",
            False,
            "backward would write the weight gradient into a main_grad "
            "buffer instead of .grad, so nothing would gate the wgrad GEMM",
        ),
        (
            "defer_embedding_wgrad_compute",
            False,
            "forward would append its input to embedding_activation_buffer "
            "on every timed call and backward would skip the wgrad GEMM, so "
            ".grad would stay None and the buffer would grow without bound",
        ),
    ):
        found = getattr(module, attribute)
        if bool(found) != wanted:
            raise RuntimeError(
                f"{MCORE_ARM_NAME}: output_layer.{attribute} is {found!r}, "
                f"expected {wanted!r}; {why} and the titan arm has no "
                "counterpart for it"
            )
    if module.bias is not None:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: output_layer carries a bias; GPTModel builds "
            "it with bias=False (gpt_model.py:278) and titan's lm_head has "
            "none, so the arms would no longer compute the same function"
        )
    return {
        "module": class_name,
        # The claim this scenario exists to keep honest.
        "transformer_engine": False,
        "tensor_parallel_size": 1,
        "collectives_run": False,
        "compiled": False,
    }


def build_lm_head_projection_mcore_base(
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: LmHeadProjectionInputs,
) -> BuiltArm:
    """Megatron-core's ``GPTModel.output_layer``, eager.

    The module is taken out of a real ``GPTModel`` built by
    ``benchmarks.models.piper_qwen3.megatron_model.build_model`` from the same
    ``PiperShape`` and the ``base`` profile the e2e megatron arm uses. A
    hand-constructed ``ColumnParallelLinear`` would need every argument
    ``gpt_model.py:269-286`` passes -- ``init_method``, ``bias``,
    ``skip_bias_add``, ``gather_output``, ``skip_weight_param_allocation``,
    the two wgrad-deferral buffers and ``tp_group`` -- written a second time
    and kept in agreement by hand, which is the duplication
    ``mcore_profiles.py`` records as the reason the layer spec is derived
    rather than written.

    The rest of the model is dropped as soon as the projection is extracted,
    and that drop is what keeps ``memory_pass`` honest. Resetting the peak
    counter before a call (``engine/measurement.py:152``) does **not** exclude
    resident memory: ``max_memory_allocated`` is a maximum over time of the
    bytes currently allocated, so the first allocation inside the timed call
    lifts the peak to resident-plus-new. A retained model would therefore be
    charged to the arm. It holds 0.67 GiB of bf16 parameters at ``1b`` and
    7.80 GiB at ``huge`` -- ``MCORE_BLANK_MLP`` leaves the mlp part out -- in
    a process that the
    correctness pass also asks to build the titan arm and to hold the fp64
    reference.

    The call unpacks ``(output, bias)``, which is the interface
    ``GPTModel._postprocess`` unpacks at ``gpt_model.py:762``. The bias is
    None: the layer is built with ``bias=False`` and ``skip_bias_add=False``,
    so ``forward`` returns ``(output, None)`` at ``layers.py:1205-1206``.
    ``skip_bias_add=False`` also makes the in-forward ``bias`` local
    ``self.bias``, which is None, so the matmul adds nothing (``:1135``).
    ``_postprocess`` then calls ``_scale_logits`` on the result, which returns
    its argument unchanged unless ``use_mup`` is set
    (``language_module.py:339-343``); no profile sets it, so nothing is
    excluded by leaving that call out.
    """
    initialize_megatron_single_rank()

    from benchmarks.models.piper_qwen3.mcore_profiles import BASE
    from benchmarks.models.piper_qwen3.megatron_model import build_model

    model = build_model(
        seq_len=workload.seq_len,
        shape=shape,
        profile=BASE,
        blank_parts=MCORE_BLANK_MLP,
    )
    output_layer = _navigate(model, mcore_module_path())
    notes = _assert_mcore_output_layer(output_layer, shape)
    with torch.no_grad():
        output_layer.weight.copy_(inputs.weight)
    del model
    gc.collect()
    torch.cuda.empty_cache()

    def call(leaf: torch.Tensor) -> torch.Tensor:
        output, _bias = output_layer(leaf)
        return output

    tokens = workload.batch * workload.seq_len
    # THD, the form our megatron driver runs: the driver packs each batch's
    # rows into one sequence (benchmarks/e2e/megatron/data.py:54), so the
    # decoder hands the output layer (t, b=1, h). Reshaped here, never inside
    # a timed closure. Both tensors are contiguous, so the reshape is a view
    # and megatron's own contiguity calls in backward are no-ops.
    return _projection_arm(
        name=MCORE_ARM_NAME,
        weight_owner=output_layer,
        call=call,
        x_native=inputs.x.reshape(tokens, 1, shape.dim),
        grad_native=inputs.grad_out.reshape(tokens, 1, shape.vocab_size),
        canonical_in=tuple(inputs.x.shape),
        canonical_out=tuple(inputs.grad_out.shape),
        notes=notes,
    )
