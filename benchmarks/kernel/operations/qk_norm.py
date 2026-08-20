"""Arm builders for the ``qk_norm`` cross-engine kernel scenario.

Per-head RMSNorm on q and k, before RoPE. TorchTitan runs it as
``q_norm(xq_BLNH)`` and ``k_norm(xk_BLNH)``
(``third_party/torchtitan/torchtitan/models/common/attention.py:959-960``).
Megatron-core runs it as ``self_attention.q_layernorm`` and
``self_attention.k_layernorm``
(``third_party/Megatron-LM/megatron/core/transformer/attention.py:1922-1926``).

**Caption every number from this scenario "TE norms via cuDNN backend".**
``megatron_bootstrap.configure_te_environment`` sets
``NVTE_NORM_FWD_USE_CUDNN=1`` and ``NVTE_NORM_BWD_USE_CUDNN=1``, because TE's
native tuned RMSNorm kernels fail to launch on this box's cuda-compat stack.
So the ``mcore/base`` arm measures TE's cuDNN norm path, not TE's fastest
norm. A host with a native CUDA 13 driver must repeat the measurement.

**The scenario times the pair, not one module twice.** Both engines hold two
separate norm modules with two separate weights, and q carries ``n_heads``
rows against k's ``n_kv_heads`` -- 1.5x the q row count in total at the piper
geometry, not 2x. One module timed twice measures the wrong row count and
folds two weights into one. Each timed closure therefore runs ``q_norm`` on q
and ``k_norm`` on k, in that order, on both engines.

**No layout conversion sits inside a timed closure.** Megatron works in SBHD
and TorchTitan in BSD, but the norm reduces over the last dimension alone and
treats every leading dimension as a row. ``qk_norm_inputs`` materializes both
layouts up front, each contiguous, so each engine reads its native form and
neither pays a transpose. The two forms hold the same rows of the same length,
so the work is equal.

**One consequence is a gap in the partition, and it is recorded here rather
than hidden.** In the engine megatron's ``key`` reaches ``k_layernorm`` as a
non-contiguous strided view of the fused QKV buffer -- ``qkv_prep`` declares
that asymmetry, and ``get_query_key_value_tensors`` reassigns ``key`` to this
norm's contiguous output (``transformer/attention.py:1929``). So the read of
the strided key, about 4 MiB per forward at the default workload, is really
this norm's cost. Because the contiguous key above is what the megatron arm
receives, **no scenario times it**: ``attention_core`` measures only the
value's half and says so, and declines to double-book the key's.

**Owed work**: hand this scenario's megatron arm a strided key, the way
``attention_core`` hands its arms a strided value. That is a change to the
inputs builder here, and it needs the same treatment ``attention_core`` gave
its leaves -- ``Tensor.clone()`` silently returns a contiguous tensor for a
non-dense view, so a cloned input would delete the effect. ``correctness_outputs`` maps the megatron outputs back
to BLNH, outside the timed region.

**Compile treatment differs by engine, and the declaration records it.** The
titan arm runs under ``torch.compile(fullgraph=True)``, because that is what
it faces end to end. The mcore arm runs eager, because megatron compiles no
whole layer. Report the ratio as a comparison of two treatments.

**The mcore arm builds the real GPTModel and navigates to the submodule.**
That is what makes the arm unrefutable: the class of ``q_layernorm`` comes
from megatron's own spec derivation, and no code here rebuilds it. The arm
keeps the two norm modules, drops the model, and collects, so the rest of the
1.07 B parameters do not sit inside ``memory_pass``.

Every megatron, TransformerEngine and torchtitan import is deferred into the
builder that needs it. Per-arm process isolation only pays off when a process
imports the one arm it builds, and the mcore arm alone needs megatron on
``sys.path`` and the TE environment variables set before TE loads.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
import torch.nn as nn

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    WEIGHT_STD,
    _compile_module,
    _randn,
    _randn_like,
    _reset_grads,
    initialize_megatron_single_rank,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE
from benchmarks.models.piper_qwen3.shape import PiperShape


# The attribute names each engine gives the two modules under test. They are
# the names the shared cross-engine weight map already declares for the
# ``qk_norm`` component (``benchmarks/models/piper_qwen3/megatron_weights.py``
# lines 156-165), and ``tests/test_kernel_qk_norm.py`` reads them back out of
# that map rather than trusting the copies here.
MCORE_Q_NORM_ATTR = "q_layernorm"
MCORE_K_NORM_ATTR = "k_layernorm"
TITAN_Q_NORM_ATTR = "q_norm"
TITAN_K_NORM_ATTR = "k_norm"
WEIGHT_COMPONENT = "qk_norm"

# The names every arm and the fp64 reference return. The scenario declaration
# in ``benchmarks/kernel/registry.py`` repeats them as literals, because a
# declaration may not import a builder module; the test pins the two copies
# against each other.
ACTIVATION_OUTPUTS = ("q_out", "k_out", "dq", "dk")
WEIGHT_GRAD_OUTPUTS = ("q_weight_grad", "k_weight_grad")

# Both engines read this one number: torchtitan through ``_qwen3_norm`` (which
# passes ``torchtitan.models.qwen3._EPS``) and megatron through
# ``layernorm_epsilon`` on the base profile. The fp64 reference needs a single
# value, and it is valid for both arms only because the two agree. A test
# asserts the agreement, so this module reads the profile and nothing else.
NORM_EPS = float(BASE.config_overrides["layernorm_epsilon"])


@dataclass
class QkNormInputs:
    """One q/k pair in both engine-native layouts, plus the two weights.

    ``*_BLNH`` is TorchTitan's layout and ``*_SBNH`` is megatron's. Each is
    contiguous and each is built here, never inside a timed closure.
    """

    q_BLNH: torch.Tensor
    k_BLNH: torch.Tensor
    gq_BLNH: torch.Tensor
    gk_BLNH: torch.Tensor
    q_SBNH: torch.Tensor
    k_SBNH: torch.Tensor
    gq_SBNH: torch.Tensor
    gk_SBNH: torch.Tensor
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    eps: float
    qk_bytes: int


def _to_sbnh(tensor: torch.Tensor) -> torch.Tensor:
    """Megatron's [s, b, n, h] view of a titan [b, l, n, h] tensor."""
    return tensor.transpose(0, 1).contiguous()


def _to_blnh(tensor: torch.Tensor) -> torch.Tensor:
    """The inverse of ``_to_sbnh``, for comparison outside the timed region."""
    return tensor.transpose(0, 1)


def _identity(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def _norm_weight(
    shape: PiperShape, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    """A seeded weight near 1.0, in bf16.

    Both engines initialize a qk norm weight to all ones. All ones hides two
    real faults: it makes q and k interchangeable, and it makes a swapped or
    dropped weight invisible to every gate. A trained norm holds values near
    1.0, so the arms get that instead. The value changes no timing.
    """
    offsets = _randn(
        (shape.head_dim,), device, generator, torch.float32, WEIGHT_STD
    )
    return (1.0 + offsets).to(torch.bfloat16)


def qk_norm_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> QkNormInputs:
    batch, seq = workload.batch, workload.seq_len
    q = _randn((batch, seq, shape.n_heads, shape.head_dim), device, generator)
    k = _randn(
        (batch, seq, shape.n_kv_heads, shape.head_dim), device, generator
    )
    gq = _randn_like(q, generator)
    gk = _randn_like(k, generator)
    return QkNormInputs(
        q_BLNH=q,
        k_BLNH=k,
        gq_BLNH=gq,
        gk_BLNH=gk,
        q_SBNH=_to_sbnh(q),
        k_SBNH=_to_sbnh(k),
        gq_SBNH=_to_sbnh(gq),
        gk_SBNH=_to_sbnh(gk),
        q_weight=_norm_weight(shape, device, generator),
        k_weight=_norm_weight(shape, device, generator),
        eps=NORM_EPS,
        # Read q and k, write q_out and k_out: the forward traffic, and what
        # copy_floor moves. The backward moves more, so read the GB/s column
        # in forward mode only.
        qk_bytes=2 * (q.numel() + k.numel()) * q.element_size(),
    )


def _rms_norm_fp64(
    x64: torch.Tensor, w64: torch.Tensor, eps: float
) -> torch.Tensor:
    scale = torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + eps)
    return x64 * scale * w64


def qk_norm_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: QkNormInputs
) -> dict[str, torch.Tensor]:
    """The fp64 truth for the weights the arms actually hold.

    The weights are already bf16, so promoting them to fp64 gives the exact
    values both engines read. An fp64 truth built from unrounded weights would
    charge both arms for the input cast.
    """
    outputs: dict[str, torch.Tensor] = {}
    for tag, x, grad, weight in (
        ("q", inputs.q_BLNH, inputs.gq_BLNH, inputs.q_weight),
        ("k", inputs.k_BLNH, inputs.gk_BLNH, inputs.k_weight),
    ):
        leaf = x.double().detach().requires_grad_()
        gamma = weight.double().detach().requires_grad_()
        out = _rms_norm_fp64(leaf, gamma, inputs.eps)
        torch.autograd.backward(out, grad.double())
        outputs[f"{tag}_out"] = out.detach()
        outputs[f"d{tag}"] = leaf.grad
        outputs[f"{tag}_weight_grad"] = gamma.grad
    return outputs


class _NormPair(nn.Module):
    """The two norms of this cut, as one module.

    Production compiles a whole transformer block, so both norms sit in one
    Inductor graph. Compiling the pair keeps that scope. It emits no fusion
    across the two: q and k differ in shape, so Inductor still writes one
    kernel each.

    **That is backend-dependent, and it is measured on the backend that
    matters.** On CUDA, Inductor emits ``triton_per_fused__fused_rms_norm_0``
    and ``..._1`` -- two forward kernels, one per norm -- against the mcore
    arm's two eager TE module calls, so the row compares two kernels to two
    kernels. The CPU backend does fuse the pair into one kernel, so a reader
    who checks this claim on a CPU box will see it fail. Re-measure on CUDA
    after any torch bump: if the two ever fuse there, the row becomes one
    kernel against two and the difference belongs to the compile treatment,
    not to either norm.
    """

    def __init__(self, q_norm: nn.Module, k_norm: nn.Module) -> None:
        super().__init__()
        self.q_norm = q_norm
        self.k_norm = k_norm

    def forward(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.q_norm(q), self.k_norm(k)


def _load_norm_weight(module: nn.Module, weight: torch.Tensor) -> None:
    if tuple(module.weight.shape) != tuple(weight.shape):
        raise ValueError(
            f"norm weight shape {tuple(module.weight.shape)} does not match "
            f"the shared weight {tuple(weight.shape)}; the two engines would "
            "normalize over different widths"
        )
    with torch.no_grad():
        module.weight.copy_(weight)


def _qk_norm_arm(
    name: str,
    inputs: QkNormInputs,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    gq: torch.Tensor,
    gk: torch.Tensor,
    call,
    canonical,
    q_module: nn.Module,
    k_module: nn.Module,
) -> BuiltArm:
    """Forward and forward+backward, over independent leaf sets.

    There is no isolated ``backward`` mode, and the megatron side is why.
    TE's operation fuser sets ``ctx._saved_tensors_range = None`` and
    ``ctx.saved_tensors = None`` while it runs backward
    (``transformer_engine/pytorch/ops/fuser.py:225,258``), and TE's RMSNorm
    calls ``clear_tensor_data`` on both saved tensors at the end of
    ``op_backward`` (``ops/basic/rmsnorm.py``). A second backward over the
    same graph therefore raises. The retained-graph trick rope and qkv use is
    not available here, so both arms drop the mode and stay comparable;
    backward cost is still forward_backward minus forward.
    """

    def leaves() -> tuple[torch.Tensor, torch.Tensor]:
        return q.clone().requires_grad_(), k.clone().requires_grad_()

    forward_leaves = leaves()
    round_trip_leaves = leaves()
    check_leaves = leaves()

    def forward():
        return call(*forward_leaves)

    def forward_backward() -> None:
        _reset_grads(*round_trip_leaves, q_module, k_module)
        torch.autograd.backward(call(*round_trip_leaves), (gq, gk))

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(*check_leaves, q_module, k_module)
        q_out, k_out = call(*check_leaves)
        torch.autograd.backward((q_out, k_out), (gq, gk))
        return {
            "q_out": canonical(q_out.detach()),
            "k_out": canonical(k_out.detach()),
            "dq": canonical(check_leaves[0].grad),
            "dk": canonical(check_leaves[1].grad),
            "q_weight_grad": q_module.weight.grad,
            "k_weight_grad": k_module.weight.grad,
        }

    return BuiltArm(
        name=name,
        calls={"forward": forward, "forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
        bytes_moved=inputs.qk_bytes,
    )


def build_qk_norm_copy_floor(
    shape: PiperShape, workload: KernelWorkload, inputs: QkNormInputs
) -> BuiltArm:
    """The bandwidth floor for this shape: copy q and k, and nothing else.

    A norm reads a row, reduces 64 values and writes the row back, so it may
    be at memory bandwidth. Where it is, the cross-engine ratio measures the
    device and not the two kernels. This arm is what separates "the cuDNN norm
    is slow" from "this scenario is at bandwidth", and Rule 3 makes that
    distinction necessary rather than merely useful. It is a floor, not an
    implementation: it has no correctness gate and it enters no ratio row.
    """
    q_out = torch.empty_like(inputs.q_BLNH)
    k_out = torch.empty_like(inputs.k_BLNH)

    def forward() -> None:
        q_out.copy_(inputs.q_BLNH)
        k_out.copy_(inputs.k_BLNH)

    return BuiltArm(
        name="copy_floor",
        calls={"forward": forward},
        correctness_outputs=dict,
        bytes_moved=inputs.qk_bytes,
    )


def _build_titan_norms(
    shape: PiperShape, inputs: QkNormInputs
) -> tuple[nn.Module, nn.Module]:
    """The two torchtitan qk norms, configured as the registry configures them.

    ``_qwen3_norm`` is the one expression ``_build_qwen3_moe_layers`` passes as
    ``qk_norm``, so this calls the production constructor rather than a
    lookalike. The scenario needs no ``Trainer.Config`` extraction for that
    reason: the config node is a pure function of the shape, and
    ``tests/test_kernel_qk_norm.py`` asserts it equals the node
    ``_piper_1b_model`` puts on layer 0.
    """
    from torchtitan.models.qwen3 import _qwen3_norm

    device = inputs.q_BLNH.device
    modules = []
    for weight in (inputs.q_weight, inputs.k_weight):
        module = _qwen3_norm(shape.head_dim).build()
        module.to(device=device, dtype=torch.bfloat16)
        _load_norm_weight(module, weight)
        modules.append(module)
    return modules[0], modules[1]


def build_qk_norm_titan(
    shape: PiperShape, workload: KernelWorkload, inputs: QkNormInputs
) -> BuiltArm:
    q_module, k_module = _build_titan_norms(shape, inputs)
    pair = _compile_module(_NormPair(q_module, k_module))
    return _qk_norm_arm(
        "titan",
        inputs,
        q=inputs.q_BLNH,
        k=inputs.k_BLNH,
        gq=inputs.gq_BLNH,
        gk=inputs.gk_BLNH,
        call=pair,
        canonical=_identity,
        q_module=q_module,
        k_module=k_module,
    )


def _assert_te_rmsnorm(module, attribute: str, shape: PiperShape) -> None:
    """Refuse to time a module that is not TE's RMSNorm.

    A scenario whose mcore side is the identity reads as a spectacular win,
    not as a bug. With ``qk_layernorm`` off, the *spec* holds ``IdentityOp``
    (``gpt_layer_specs.py:327-332``) but the *built attribute* is ``None``:
    ``attention.py:1711-1722`` accepts ``IdentityOp`` as a permitted spec
    value and sets the norm class to ``None``, and ``:1724-1732`` assigns
    that. So the ``None`` check below is the one that fires, and the
    exact-type check never sees ``IdentityOp``.

    A qk norm is never ``TEFusedResidualRMSNorm`` at any setting of
    ``fused_residual_rmsnorm``: ``transformer_engine_spec_provider.py:59-69``
    builds it with ``has_residual=False``, and
    ``extensions/transformer_engine.py:1046`` gates the fused class on
    ``config.fused_residual_rmsnorm and has_residual``. The exact-type check
    still costs nothing, and the width check catches a geometry that no
    longer normalizes per head.
    """
    import transformer_engine.pytorch as te

    if module is None:
        raise RuntimeError(
            f"megatron built no {attribute}; the arm would measure nothing "
            "and publish it as a win"
        )
    if type(module) is not te.RMSNorm:
        raise RuntimeError(
            f"megatron built {type(module).__name__} for {attribute}, not "
            "transformer_engine.pytorch.RMSNorm; this arm claims to measure "
            "TE's norm"
        )
    if tuple(module.weight.shape) != (shape.head_dim,):
        raise RuntimeError(
            f"{attribute} normalizes over {tuple(module.weight.shape)}, not "
            f"the head dimension ({shape.head_dim},)"
        )


# The two norm modules together hold 128 bf16 values. Anything above this is
# the model, not the norms, and the budget is wide enough that allocator
# rounding cannot reach it.
_RESIDUAL_BUDGET_BYTES = 64 * 2**20


def _report_build_residual(before: int) -> None:
    """Say so if the dropped GPTModel did not free.

    ``memory_pass`` reports ``max_memory_allocated``, which counts every live
    allocation. A surviving reference to the 1.07 B-parameter model adds about
    2 GiB to this arm's peak memory and nothing to the titan arm's, so the
    memory column would then compare two engines and one model.

    This reports and does not raise. The timing columns are unaffected, and
    peak memory is the secondary metric of this scenario, so a hard failure
    would cost the whole ratio to protect a column the reader can discount.
    The worker's stdout lands in ``kernel_bench.log``, which the run keeps.
    """
    residual = torch.cuda.memory_allocated() - before
    if residual > _RESIDUAL_BUDGET_BYTES:
        print(
            "WARNING qk_norm/mcore/base: the megatron model did not free "
            f"({residual / 2**20:.0f} MiB still allocated). Read this arm's "
            "peak_memory_gib as the model plus the norms, not as the norms."
        )


def build_qk_norm_mcore_base(
    shape: PiperShape, workload: KernelWorkload, inputs: QkNormInputs
) -> BuiltArm:
    """Megatron-core's q/k norms, taken from the model megatron builds.

    The two modules come off a real ``GPTModel``: the class is whatever
    megatron's own spec derivation chooses, so nothing here can build a
    lookalike by mistake. The model itself is then dropped, and only the two
    norms stay alive. Peak memory is a published column, so a resident 1.07 B
    parameters would make the mcore arm look expensive for a reason that has
    nothing to do with the norm.
    """
    initialize_megatron_single_rank(torch.initial_seed() % (2**31))
    from benchmarks.models.piper_qwen3.megatron_model import build_model

    before = torch.cuda.memory_allocated()
    model = build_model(seq_len=workload.seq_len, shape=shape, profile=BASE)
    attention = model.decoder.layers[0].self_attention
    q_module = getattr(attention, MCORE_Q_NORM_ATTR)
    k_module = getattr(attention, MCORE_K_NORM_ATTR)
    _assert_te_rmsnorm(q_module, MCORE_Q_NORM_ATTR, shape)
    _assert_te_rmsnorm(k_module, MCORE_K_NORM_ATTR, shape)
    if q_module is k_module:
        raise RuntimeError(
            "megatron shares one module between q_layernorm and k_layernorm; "
            "the scenario times a pair of norms with two weights"
        )
    del attention, model
    gc.collect()
    torch.cuda.empty_cache()
    _report_build_residual(before)

    _load_norm_weight(q_module, inputs.q_weight)
    _load_norm_weight(k_module, inputs.k_weight)

    def call(q: torch.Tensor, k: torch.Tensor):
        # Eager, as megatron runs it: megatron compiles no whole layer, and
        # neither norm carries a jit_fuser decoration.
        return q_module(q), k_module(k)

    return _qk_norm_arm(
        "mcore/base",
        inputs,
        q=inputs.q_SBNH,
        k=inputs.k_SBNH,
        gq=inputs.gq_SBNH,
        gk=inputs.gk_SBNH,
        call=call,
        canonical=_to_blnh,
        q_module=q_module,
        k_module=k_module,
    )
