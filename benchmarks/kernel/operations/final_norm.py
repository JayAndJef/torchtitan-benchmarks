"""Arm builders for the ``final_norm`` kernel scenario.

The norm after the last transformer block, before the LM head. TorchTitan
calls it at ``Decoder.norm(h)``
(``third_party/torchtitan/torchtitan/models/common/decoder.py:277``);
megatron-core calls it at ``decoder.final_layernorm``
(``third_party/Megatron-LM/megatron/core/transformer/transformer_block.py:693``).
The scenario is cross-engine and it publishes one ratio.

**The two sides are two implementations, not one implementation twice.**
TorchTitan's ``RMSNorm`` is ``torch.nn.RMSNorm``
(``torchtitan/models/common/nn_modules.py:134``), so Inductor generates its
kernel. Megatron's is TransformerEngine's ``RMSNorm``: our build calls
``get_gpt_decoder_block_spec(config, use_transformer_engine=True)``
(``benchmarks/models/piper_qwen3/megatron_model.py:130``), which sets
``layer_norm=TENorm`` (``gpt_layer_specs.py:708-709``), and ``TENorm.__new__``
returns ``transformer_engine.pytorch.RMSNorm`` for
``normalization == "RMSNorm"`` (``extensions/transformer_engine.py:1052-1062``).
``_assert_te_rmsnorm`` re-proves that at build time, so a spec change cannot
publish a torch fallback under the TE label.

**Caption every number from this scenario "TE norms via cuDNN backend".**
``configure_te_environment`` sets ``NVTE_NORM_FWD_USE_CUDNN=1`` and
``NVTE_NORM_BWD_USE_CUDNN=1``
(``benchmarks/models/piper_qwen3/megatron_bootstrap.py:55-56``), because TE's
native tuned RMSNorm kernels fail to launch on this box's cuda-compat stack.
The mcore arm therefore measures TE's cuDNN norm path, which is not TE's
fastest norm. Revisit on a host with a native CUDA 13 driver.

**Both arms run eager, and that is the production treatment on both engines.**
This scenario is the exception to the rule that a titan module arm compiles.
``apply_compile`` compiles only the children of ``model.layers``
(``third_party/torchtitan/torchtitan/distributed/compile.py:57-58``), and
``Decoder.__init__`` puts the final norm at ``self.norm``, a *sibling* of
``self.layers`` (``models/common/decoder.py:236,240``), consumed at
``:277`` outside every compiled region. The only other compile in a run is
the loss (``CompileConfig.components`` defaults to ``["model", "loss"]``),
which this norm is not. So production runs it eager, exactly as megatron does,
and an eager-against-eager row is both the like-for-like comparison and the
faithful one. ``KernelArm.compiled`` records ``False`` on both arms.

The same correction applies to the embedding and the lm-head projection
(plan scenarios 1 and 15): those modules are siblings of ``self.layers`` too.
It does **not** apply to scenarios 2 through 13, whose modules sit inside a
transformer block and really are compiled end to end.

**Both arms may sit at memory bandwidth, so the scenario declares a floor.**
A norm reads the input once and writes the output once. ``copy_floor`` does
exactly that traffic and nothing else, so the ``x_floor`` column separates
"this kernel is slow" from "this scenario is at bandwidth". Read a ratio
between two arms that both sit near the floor as a dispatch comparison, not as
a kernel-quality claim.

**No layout conversion sits inside the timed region, on either engine.** A
norm reduces over the last dimension and treats every leading dimension as a
row index. Megatron runs the model in SBHD and TorchTitan in BSD, but TE's op
flattens with ``input_.contiguous().view((-1, inner_dim))``
(``transformer_engine/pytorch/ops/basic/rmsnorm.py:182``), and
``torch.nn.RMSNorm`` does the same reduction. Both arms therefore receive the
same contiguous ``[B, L, D]`` tensor: the row count, the row length and the
kernel work are identical, and only the row order would differ. Hand either
arm a transposed *view* instead and the ``.contiguous()`` above would charge
that engine an 8 MiB copy inside the timed call, so the inputs builder
materializes one contiguous tensor and both arms share it.

**There is no isolated ``backward`` mode, and there cannot be one.** Every
other module scenario times backward by re-running a retained graph. TE
forbids that twice over: ``restore_from_func_ctx`` clears
``ctx.tensor_objects`` and raises on the second call
(``transformer_engine/pytorch/quantized_tensor.py:195-201``, reached from
``ops/fuser.py:220``), and ``op_backward`` frees the saved activations with
``clear_tensor_data`` (``ops/basic/rmsnorm.py:238-239``). Both arms therefore
declare ``forward`` and ``forward_backward`` only, exactly as the
``attention_core`` scenario does. Backward cost stays recoverable as the
difference.

Every implementation import is deferred into the builder that needs it. One
arm per process is the point: the mcore arm pays for megatron and TE, and the
titan arm pays for torchtitan, and neither pays for the other.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    WEIGHT_STD,
    _randn,
    _randn_like,
    _reset_grads,
    initialize_megatron_single_rank,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.shape import PiperShape


@dataclass
class FinalNormInputs:
    x: torch.Tensor  # (B, L, D) bf16, contiguous
    grad_out: torch.Tensor  # (B, L, D) bf16
    weight: torch.Tensor  # (D,) bf16 gain, loaded into both engines
    eps: float
    copy_bytes: int  # one read plus one write of x: the floor's traffic


def final_norm_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> FinalNormInputs:
    """The hidden state, the upstream gradient, and the shared norm gain.

    ``eps`` comes from the mcore profile rather than from a literal here,
    because that registry is torch-free and every worker can read it cheaply.
    The titan builder refuses to build when its own config disagrees, so the
    two engines are checked against one value and the fp64 reference uses it.

    The gain is centered on 1.0, which is where a trained RMSNorm gain sits
    (``_NORM_INIT`` initializes it to ones,
    ``third_party/torchtitan/torchtitan/models/qwen3/__init__.py:52``). An
    all-ones gain would hide an implementation that ignores the parameter, so
    the builder draws a seeded perturbation around it.
    """
    from benchmarks.models.piper_qwen3.mcore_profiles import BASE

    batch, seq = workload.batch, workload.seq_len
    x = _randn((batch, seq, shape.dim), device, generator)
    gain = torch.ones(
        (shape.dim,), device=device, dtype=torch.float32
    ) + _randn((shape.dim,), device, generator, torch.float32, WEIGHT_STD)
    return FinalNormInputs(
        x=x,
        grad_out=_randn_like(x, generator),
        weight=gain.to(torch.bfloat16),
        eps=float(BASE.config_overrides["layernorm_epsilon"]),
        copy_bytes=2 * x.numel() * x.element_size(),
    )


def final_norm_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: FinalNormInputs
) -> dict[str, torch.Tensor]:
    """fp64 RMSNorm forward and backward over the bf16 values the arms hold.

    The gain is promoted from its bf16 storage rather than from an unrounded
    draw, for the reason ``qkv_reference`` states: the truth must be the truth
    for the weights the arms actually hold, or both arms are charged for the
    input cast.
    """
    x = inputs.x.double().detach().requires_grad_()
    weight = inputs.weight.double().detach().requires_grad_()
    rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + inputs.eps)
    out = (x / rms) * weight
    torch.autograd.backward(out, inputs.grad_out.double())
    return {
        "out": out.detach(),
        "x_grad": x.grad,
        "weight_grad": weight.grad,
    }


def _final_norm_arm(
    name: str,
    module,
    weight: torch.Tensor,
    inputs: FinalNormInputs,
) -> BuiltArm:
    """Forward and forward+backward over one norm module.

    Each mode owns its own leaf, so one mode's autograd state never reaches
    another's. ``weight`` is the parameter itself, captured before any
    compile wrapper, so ``weight_grad`` names the same tensor on both engines.
    """
    if not weight.requires_grad:
        raise RuntimeError(
            f"{name}: the norm gain does not require grad, so backward would "
            "skip the gain gradient and the arm would do less work than its "
            "opponent"
        )

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
        return {
            "out": out.detach(),
            "x_grad": check_leaf.grad,
            "weight_grad": weight.grad,
        }

    return BuiltArm(
        name=name,
        calls={"forward": forward, "forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
    )


def build_final_norm_copy_floor(
    shape: PiperShape, workload: KernelWorkload, inputs: FinalNormInputs
) -> BuiltArm:
    """The bandwidth floor: one read of x and one write, and nothing else.

    A norm moves the same bytes, so an arm near this floor is bandwidth-bound
    and the ratio above it is not a kernel-quality claim. Eager on purpose: a
    floor is a measurement of the device, not an implementation.
    """
    out = torch.empty_like(inputs.x)

    def forward() -> None:
        out.copy_(inputs.x)

    return BuiltArm(
        name="copy_floor",
        calls={"forward": forward},
        correctness_outputs=dict,
        bytes_moved=inputs.copy_bytes,
    )


def titan_final_norm_module(shape: PiperShape):
    """The ``Decoder.norm`` config node, built as a module.

    This is the whole titan extraction path for this scenario, and it needs no
    ``Trainer.Config``. ``Decoder.__init__`` builds the model-level norm with
    ``self.norm = config.norm.build()``
    (``third_party/torchtitan/torchtitan/models/common/decoder.py:240``), and
    ``_piper_1b_model`` is where our registry sets that node
    (``benchmarks/models/piper_qwen3/config_registry.py:115``). So the module
    below is the production module, configured by the production config, and
    the extraction is one attribute read.

    The norm sits at the model level rather than inside a block, so exactly
    one of it exists whatever ``n_layers`` is. That is why a config read is
    enough here: there is no per-layer node to select, and no arithmetic over
    ``n_layers`` anywhere in this scenario.
    """
    from benchmarks.models.piper_qwen3.config_registry import _piper_1b_model

    return _piper_1b_model(fuse_qkv=True, shape=shape).norm.build()


def build_final_norm_titan(
    shape: PiperShape, workload: KernelWorkload, inputs: FinalNormInputs
) -> BuiltArm:
    """TorchTitan's ``Decoder.norm``: ``torch.nn.RMSNorm``, eager.

    Eager, not compiled, because ``Decoder.norm`` is a sibling of
    ``Decoder.layers`` and ``apply_compile`` reaches only the layers. See the
    module docstring: this is the production treatment, not a relaxation.
    """
    from torchtitan.models.qwen3 import _qwen3_norm

    declared_eps = float(_qwen3_norm(shape.dim).eps)
    if declared_eps != inputs.eps:
        raise RuntimeError(
            f"titan eps {declared_eps} != mcore eps {inputs.eps}; the two "
            "engines would compute different functions and the ratio between "
            "them would not be a kernel comparison"
        )
    module = titan_final_norm_module(shape).to(
        device=inputs.x.device, dtype=torch.bfloat16
    )
    with torch.no_grad():
        module.weight.copy_(inputs.weight)
    weight = module.weight
    return _final_norm_arm("titan", module, weight, inputs)


def _assert_te_rmsnorm(module, eps: float) -> None:
    """Refuse to time a final norm that is not TE's RMSNorm.

    The spec derivation chooses the class, so a megatron bump, a profile
    change, or a missing TE could substitute a torch norm here. The arm would
    still be numerically right, and no correctness gate could see it, but the
    published row would compare titan against torch under a TE label.
    """
    import transformer_engine.pytorch as te

    if not isinstance(module, te.RMSNorm):
        raise RuntimeError(
            f"mcore_base: decoder.final_layernorm is {type(module).__name__}, "
            "not transformer_engine.pytorch.RMSNorm; this arm would publish a "
            "torch norm under the TE label"
        )
    if float(module.eps) != eps:
        raise RuntimeError(
            f"mcore_base: TE eps {module.eps} != {eps}, so the two engines "
            "compute different functions"
        )


def build_final_norm_mcore_base(
    shape: PiperShape, workload: KernelWorkload, inputs: FinalNormInputs
) -> BuiltArm:
    """Megatron's ``decoder.final_layernorm``: TE RMSNorm, eager.

    The module comes from a whole ``GPTModel`` and is reached at
    ``model.decoder.final_layernorm``. Building the whole model costs the
    shape's full parameter memory and a minute of wall clock per worker, and
    it buys the one property a hand-built lookalike cannot have: the module is
    the one megatron's own spec derivation constructs, with every kwarg
    megatron gives it.
    """
    from benchmarks.models.piper_qwen3.mcore_profiles import BASE
    from benchmarks.models.piper_qwen3.megatron_model import build_model

    initialize_megatron_single_rank(torch.initial_seed())
    model = build_model(seq_len=workload.seq_len, shape=shape, profile=BASE)
    module = model.decoder.final_layernorm
    if module is None:
        raise RuntimeError(
            "mcore_base: the built GPTModel carries no decoder.final_layernorm"
        )
    _assert_te_rmsnorm(module, inputs.eps)
    with torch.no_grad():
        module.weight.copy_(inputs.weight)
    weight = module.weight
    # Best effort: nothing else here references the rest of the model, so the
    # other parameters are free to be collected while the norm is timed.
    del model
    return _final_norm_arm("mcore_base", module, weight, inputs)
