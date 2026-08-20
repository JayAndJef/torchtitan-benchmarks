"""Arm builders for the ``ffn_norm`` cross-engine kernel scenario.

The norm in front of the MoE block, on both engines. TorchTitan calls it
``self.ffn_norm(x)`` in ``Qwen3TransformerBlock.forward``
(``third_party/torchtitan/torchtitan/models/qwen3/model.py:63``). Megatron-core
calls it ``pre_mlp_layernorm``
(``third_party/Megatron-LM/megatron/core/transformer/transformer_layer.py:759``,
through ``_forward_mlp`` at ``:827``).

**Caption every number from this scenario as "TE norms via the cuDNN
backend".** ``megatron_bootstrap.configure_te_environment`` sets
``NVTE_NORM_FWD_USE_CUDNN=1`` and ``NVTE_NORM_BWD_USE_CUDNN=1``, because TE's
native tuned RMSNorm kernels fail to launch on this box's cuda-compat stack.
The mcore arm therefore measures TE's cuDNN norm path, which is not TE's
fastest norm. A number from this scenario is not "megatron's norm". Revisit
the caption on a host with a native CUDA 13 driver.

**Which class the mcore side really builds, and what its forward returns.**
The spec entry is ``backend.layer_norm(has_residual=True)``
(``gpt_layer_specs.py:336``, the non-MLA branch this build takes), which
``TESpecProvider.layer_norm`` resolves to the class adapter
``_TENormWithResidual``
(``extensions/transformer_engine_spec_provider.py:29-33,69``). That adapter
calls ``TENorm(..., has_residual=True)``, and ``TENorm.__new__`` computes
``use_fused_residual = config.fused_residual_rmsnorm and has_residual``
(``extensions/transformer_engine.py:1046``). The base profile leaves
``fused_residual_rmsnorm`` at its ``False`` default
(``transformer_config.py:508``), so the product is False and ``:1062`` builds a
plain ``transformer_engine.pytorch.RMSNorm``. **Its forward returns one
tensor, not a ``(output, residual)`` tuple.** ``transformer_layer.py:829-837``
reads the tuple form only when it finds one, and takes the ``residual =
hidden_states`` branch here. ``_require_a_real_norm`` below refuses to time an
arm whose module returns a tuple, so a profile that turns the fusion on cannot
reach a published number under this arm's label.

**No layout conversion sits inside the timed region, on either engine, and
none is needed.** Megatron works in SBHD and titan in BSD, but an RMSNorm
reduces over the last dimension alone and treats every leading dimension as a
row index. TE flattens the input to ``(-1, hidden_size)`` and reshapes the
output back (``transformer_engine/pytorch/ops/basic/rmsnorm.py:182,206``), and
``torch.nn.RMSNorm`` does the same in effect. Both engines therefore see
``batch * seq_len`` contiguous rows of ``dim`` elements, whichever order the
two leading dimensions carry. Both arms take the same canonical ``[B, L, D]``
tensor, and neither pays a transpose.

**There is no isolated ``backward`` mode, on either arm.** TE's RMSNorm
backward calls ``clear_tensor_data(x)`` and ``clear_tensor_data(rstdevs)``
(``rmsnorm.py:238-239``), which replaces each saved tensor with an empty one
(``transformer_engine/pytorch/utils.py:54-75``). The retained-graph trick the
rope and qkv scenarios use re-runs backward over one graph, so the second call
would read cleared tensors. ``attention`` drops the mode for the same class of
reason. Dropping it from **both** arms keeps them comparable, and backward
cost stays recoverable as ``forward_backward`` minus ``forward``.

Every torchtitan, megatron and TransformerEngine import is deferred into the
builder that needs it, which is the rule across ``operations/``.
``benchmarks.models.piper_qwen3.mcore_profiles`` is the one exception at module
scope: it is torch-free parent-side data, and this module reads the declared
norm epsilon from it.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    WEIGHT_STD,
    _compile_module,
    _randn,
    _randn_like,
    _require_grads,
    _reset_grads,
    initialize_megatron_single_rank,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE
from benchmarks.models.piper_qwen3.shape import PiperShape


# The epsilon both engines use, taken from the mcore profile rather than
# retyped. Megatron reads it as ``config.layernorm_epsilon`` and hands it to
# ``pre_mlp_layernorm`` (``transformer_layer.py:404-407``). TorchTitan's
# ``_qwen3_norm`` carries its own copy, and ``build_ffn_norm_titan`` refuses to
# build when the two disagree: a scenario whose two sides normalize with
# different epsilons compares two functions, not two implementations.
NORM_EPS = float(BASE.config_overrides["layernorm_epsilon"])

# The layer the mcore arm reads its module from. Every layer holds the same
# module class at ``moe_layer_freq=1``, so the index is arbitrary and is fixed
# here so the provenance line names it.
MCORE_LAYER = 0


@dataclass
class FfnNormInputs:
    """One canonical hidden-state batch and one shared norm gain.

    ``weight`` is fp32 and both engines load it, so the two arms hold the same
    gain by construction. The cross-engine map states the same correspondence
    -- ``benchmarks/models/piper_qwen3/megatron_weights.py:166-170`` pairs
    titan's ``layers.{L}.ffn_norm.weight`` with megatron's
    ``decoder.layers.{L}.pre_mlp_layernorm.weight`` under the ``ffn_norm``
    component tag -- and for a norm that map is the identity on a ``[dim]``
    vector. There is no reshape to reimplement here, and no second
    implementation of one.
    """

    x: torch.Tensor  # (B, L, D) bf16, contiguous
    grad_out: torch.Tensor  # (B, L, D) bf16
    weight: torch.Tensor  # (D,) fp32 gain, shared by both engines
    eps: float
    bytes_moved: int


def ffn_norm_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> FfnNormInputs:
    batch, seq = workload.batch, workload.seq_len
    x = _randn((batch, seq, shape.dim), device, generator)
    grad_out = _randn_like(x, generator)
    # A trained RMSNorm gain sits near 1.0, and both engines initialize it to
    # exactly ones. A gain of exactly ones cannot show a gain the arm never
    # loaded, so the shared gain is 1 + N(0, WEIGHT_STD).
    noise = torch.randn(
        (shape.dim,), device=device, generator=generator, dtype=torch.float32
    )
    weight = 1.0 + WEIGHT_STD * noise
    return FfnNormInputs(
        x=x,
        grad_out=grad_out,
        weight=weight,
        eps=NORM_EPS,
        # Forward traffic: one read of x and one write of y. The merge divides
        # this by every mode's median, so the GB/s column is a bandwidth
        # statement in ``forward`` alone; ``rope`` records the same property.
        bytes_moved=2 * x.numel() * x.element_size(),
    )


def ffn_norm_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: FfnNormInputs
) -> dict[str, torch.Tensor]:
    """fp64 RMSNorm truth for the tensors the arms actually hold.

    The gain is quantized to bf16 before it is promoted to fp64, which is the
    ``qkv_reference`` rule: an fp64 truth built from the unrounded gain would
    charge both arms for an input cast neither of them performs.
    """
    x = inputs.x.double().detach().requires_grad_()
    weight = inputs.weight.to(torch.bfloat16).double().detach().requires_grad_()
    scale = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + inputs.eps)
    out = x * scale * weight
    torch.autograd.backward(out, inputs.grad_out.double())
    return {
        "out": out.detach(),
        "x_grad": x.grad,
        "weight_grad": weight.grad,
    }


def _require_a_real_norm(module, probe: torch.Tensor, arm: str) -> None:
    """Refuse an arm whose norm is a tuple form, or is not a norm at all.

    Two failures, both silent without this check, and both of the class the
    plan calls a spectacular win rather than a bug.

    ``pre_mlp_layernorm`` is declared ``has_residual=True``, so the class it
    becomes depends on ``config.fused_residual_rmsnorm``
    (``extensions/transformer_engine.py:1046``). Under the base profile the
    flag is off and the module is a plain ``te.pytorch.RMSNorm`` returning one
    tensor. A profile that turns the flag on returns a tuple
    (``TEFusedResidualRMSNorm.forward``, ``:993-1010``), which is a different
    operation and belongs to a different arm.

    ``gpt_layer_specs.py:336`` also spells the entry ``backend.layer_norm(
    has_residual=True) if num_experts else IdentityOp``, and ``IdentityOp``
    returns its input unchanged. A shape with no experts would therefore time
    the identity and report an enormous win over titan.
    """
    with torch.no_grad():
        result = module(probe)
    if not isinstance(result, torch.Tensor):
        raise RuntimeError(
            f"{arm}: the norm returned {type(result).__name__}, not a Tensor. "
            "A tuple means the residual fusion is active "
            "(config.fused_residual_rmsnorm), which is a different operation "
            "and needs its own arm."
        )
    if result is probe or torch.equal(result, probe):
        raise RuntimeError(
            f"{arm}: the norm returned its own input, so it computes nothing. "
            "Megatron substitutes IdentityOp for pre_mlp_layernorm when the "
            "layer has no experts; an identity arm reads as a win, not as a "
            "bug."
        )


def _norm_arm(
    name: str,
    module,
    weight: torch.Tensor,
    inputs: FfnNormInputs,
) -> BuiltArm:
    """The timed closures both engines share.

    Both arms run this same code over their own module, so the comparison
    measures the two modules and nothing about how each arm was written.

    ``forward`` runs over a leaf that requires grad, matching ``qkv`` and
    ``attention``: that is the condition production runs in, and TE selects a
    different SM margin for an inference call (``rmsnorm.py:186``).
    """
    forward_leaf = inputs.x.clone().requires_grad_()
    round_trip_leaf = inputs.x.clone().requires_grad_()
    check_leaf = inputs.x.clone().requires_grad_()

    def forward():
        return module(forward_leaf)

    def forward_backward() -> None:
        _reset_grads(round_trip_leaf, module)
        torch.autograd.backward(module(round_trip_leaf), inputs.grad_out)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(check_leaf, module)
        out = module(check_leaf)
        torch.autograd.backward(out, inputs.grad_out)
        return _require_grads(
            name,
            {
                "out": out.detach(),
                "x_grad": check_leaf.grad,
                "weight_grad": weight.grad,
            },
        )

    return BuiltArm(
        name=name,
        calls={"forward": forward, "forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
        bytes_moved=inputs.bytes_moved,
    )


def build_ffn_norm_copy_floor(
    shape: PiperShape, workload: KernelWorkload, inputs: FfnNormInputs
) -> BuiltArm:
    """The bandwidth floor: one read of x and one write of y, and nothing else.

    A norm at these shapes may be at memory bandwidth, and then the cross-engine
    ratio measures how close each implementation sits to a copy rather than how
    good its kernel is. This arm is the only thing that separates "the cuDNN
    norm is slow" from "this scenario is at bandwidth", which matters here more
    than anywhere because the mcore arm runs TE's cuDNN path and not TE's own
    tuned kernels. Forward only: the forward traffic is exactly a copy, and a
    floor for the forward+backward traffic would be an invention rather than a
    measurement.
    """
    out = torch.empty_like(inputs.x)

    def forward() -> None:
        out.copy_(inputs.x)

    return BuiltArm(
        name="copy_floor",
        calls={"forward": forward},
        correctness_outputs=dict,
        bytes_moved=inputs.bytes_moved,
    )


def build_ffn_norm_titan(
    shape: PiperShape, workload: KernelWorkload, inputs: FfnNormInputs
) -> BuiltArm:
    """TorchTitan's ``ffn_norm``, built from the production config node.

    ``_qwen3_norm`` is the helper ``_build_qwen3_moe_layers`` passes as
    ``ffn_norm=`` for every block
    (``third_party/torchtitan/torchtitan/models/qwen3/__init__.py:152``), so
    the epsilon and the parameter initialization come from upstream rather than
    from a value retyped here. No ``Trainer.Config`` extraction is needed: the
    module under test is one config node with no surrounding trainer state, and
    this is the same direct build ``qkv`` uses for ``QKVLinear``.
    """
    from torchtitan.models.qwen3 import _qwen3_norm

    config = _qwen3_norm(shape.dim)
    if float(config.eps) != inputs.eps:
        raise RuntimeError(
            f"titan builds ffn_norm with eps {config.eps!r} and the mcore "
            f"profile declares {inputs.eps!r}. The two arms would normalize "
            "with different epsilons, so the ratio would not compare two "
            "implementations of one function."
        )
    module = config.build().to(inputs.x.device)
    module.load_state_dict({"weight": inputs.weight})
    module.to(torch.bfloat16)
    weight = module.weight
    _require_a_real_norm(module, inputs.x, "titan")
    # Compiled, because that is what a titan module faces end to end. The mcore
    # arm runs eager, because megatron compiles no whole layer; the ratio
    # between them is a comparison of two treatments, and every table says so.
    return _norm_arm("titan", _compile_module(module), weight, inputs)


def build_ffn_norm_mcore_base(
    shape: PiperShape, workload: KernelWorkload, inputs: FfnNormInputs
) -> BuiltArm:
    """Megatron-core's ``pre_mlp_layernorm``, taken off a real ``GPTModel``.

    The module is navigated out of the built model rather than constructed
    here, at ``model.decoder.layers[MCORE_LAYER].pre_mlp_layernorm``. That
    matters for this scenario specifically: the class depends on the spec entry
    ``backend.layer_norm(has_residual=True)`` **and** on
    ``config.fused_residual_rmsnorm``, so a hand construction would have to
    retype ``has_residual`` and could get it wrong in a way no correctness gate
    can see. ``build_model`` and ``get_gpt_decoder_block_spec`` decide instead.

    The rest of the model is released as soon as the norm is extracted. A norm
    module holds no reference to its parent or to the ``TransformerConfig``
    (``TENorm.__new__`` reads the config and stores none of it), so the arm
    keeps one ``[dim]`` parameter and the peak-memory column stays a statement
    about the norm.

    Eager on purpose: megatron compiles no whole layer, so this is the
    treatment megatron gives the module. ``KernelArm.compiled`` records it.
    """
    initialize_megatron_single_rank()

    from benchmarks.models.piper_qwen3.megatron_model import build_model

    model = build_model(seq_len=workload.seq_len, shape=shape, profile=BASE)
    module = model.decoder.layers[MCORE_LAYER].pre_mlp_layernorm
    kind = f"{type(module).__module__}.{type(module).__qualname__}"
    print(
        f"ffn_norm/mcore/base: "
        f"decoder.layers[{MCORE_LAYER}].pre_mlp_layernorm is {kind} "
        f"(profile {BASE.name}, cuDNN norm backend)",
        flush=True,
    )
    if module.weight.shape != inputs.weight.shape:
        raise RuntimeError(
            f"mcore/base: pre_mlp_layernorm holds a {tuple(module.weight.shape)} "
            f"gain and the shared inputs hold {tuple(inputs.weight.shape)}"
        )
    with torch.no_grad():
        module.weight.copy_(inputs.weight)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    _require_a_real_norm(module, inputs.x, "mcore/base")
    return _norm_arm("mcore/base", module, module.weight, inputs)
