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

**No transpose sits inside a timed closure.** Megatron works in SBHD and
TorchTitan in BSD, but the norm reduces over the last dimension alone and
treats every leading dimension as a row. ``qk_norm_inputs`` materializes both
layouts up front, so each engine reads its native form and neither pays a
transpose inside a timed call. The two forms hold the same rows of the same
length, so the arithmetic is equal.

**The two MEMORY layouts are not equal, and the difference is part of the
measurand.** Megatron's QKV GEMM writes one fused ``[s, b, ng, (Q + 2) * H]``
tensor, where ``ng`` counts the key/value groups and ``Q`` the query heads per
group, and ``get_query_key_value_tensors`` splits it along the last dimension
(``transformer/attention.py:1869`` views, ``:1902-1905`` splits). All three
results start as strided views of that one buffer. Two things then happen, and
only one of them touches the key:

* The query leaves the buffer at ``:1908``. ``query.reshape(s, b, -1, H)``
  merges the group dimension with the query heads inside a group, and the two
  are not adjacent in memory because the row is ``(Q + 2) * H`` wide and not
  ``Q * H``, so the reshape allocates and copies. ``q_layernorm`` therefore
  reads a **contiguous** tensor (``:1923``).
* The key takes no such step. It reaches ``k_layernorm`` at ``:1926`` as the
  original strided view: storage offset ``Q * H``, and the fused buffer's own
  strides.

``qk_norm_inputs`` reproduces exactly that. It builds the fused buffer, splits
it megatron's way, and hands the ``mcore/base`` arm a contiguous query and a
**strided key**. Handing it two tidy contiguous tensors would measure a layout
megatron never produces, and it would delete a copy megatron really pays --
which flatters megatron, the anchor of every ratio in this scenario.

**The copy is real, it is inside the timed call, and TE is where it happens.**
``transformer_engine.pytorch.ops.basic.rmsnorm.RMSNorm.op_forward`` begins
``x = maybe_dequantize(input_.contiguous(), dtype).view((-1, inner_dim))``
(``ops/basic/rmsnorm.py:182``), so a non-contiguous input is materialized
before the norm kernel launches. At the default workload the key holds 4 MiB,
so the mcore arm moves 4 MiB more of reads and 4 MiB more of writes than the
titan arm. ``mcore_qk_bytes`` declares that, and the ``mcore/base`` arm reports
it as its own ``bytes_moved``, following ``moe_router``: a per-arm count makes
the GB/s column show the asymmetry instead of absorbing it.

**Read this arm's ``x_floor`` against 1.33, not against 1.0.** The column is
``mode_median / floor_median`` (``results/merge.py:430``), so the arithmetic
reads no ``bytes_moved`` at all and nothing about the formula changed. The
**interpretation** did. ``copy_floor`` copies 24 MiB and ``mcore/base`` moves
32 MiB, so an ``mcore/base`` running at exactly the floor's bandwidth reads
``x_floor`` about 32/24 = **1.33**. The usual test -- x_floor near 1.0 means
the arm is at the bus -- is therefore wrong for this one arm, and only for
it: ``titan`` and the floor declare the same count and still read against
1.0. Divide an ``mcore/base`` x_floor by 1.33 to recover the usual reading.
The scenario ``description`` says the same thing, because a reader of
``results.json`` holds no module docstring.

**The charge is asymmetric per arm and symmetric per partition, and that is
the point.** The partition plan asks, in its layout table, for a layout
conversion to be declared on both engines when it sits inside the module
under test and cannot be hoisted. That rule is the plan's and is written
nowhere in CLAUDE.md; the reasoning below stands on its own either way.

The rule governs a conversion the harness introduces by choosing a cut. This
one is not that. It sits inside the module under test on one engine only, and
the reason is the engine: titan's ``qkv_linear`` materializes a contiguous
``xk`` before ``k_norm`` ever runs, and ``qkv_prep`` already charges titan for
that materialization. Giving the titan norm a strided key too would invent
work titan does not do, and it would book the same key against titan twice. So
each engine pays this materialization exactly once across the partition --
titan in ``qkv_prep``, megatron here.

**What this closes.** ``qkv_prep`` declares that megatron defers its split
copy to whoever consumes the strided views. The value's half is timed by
``attention_core``; the key's half was timed nowhere, because this scenario
fed its megatron arm a contiguous key. It is timed here now, so both halves of
the deferral are collected.

**That ledger is prose, and no test can check it.** ``attention_core``
declares no ``bytes_moved`` on any arm and holds no ``copy_floor``, so only
this scenario publishes a number for its half. The two cannot disagree
numerically, because there is nothing to disagree with. Read "the partition
sums" as a statement about which module measures which tensor, never as an
arithmetic identity that some assertion enforces.

**Two conventions for the same tensor, named every time.** The key holds
4 MiB. Moving it costs 8 MiB of traffic, because ``qk_bytes`` counts one read
plus one write. ``qkv_prep`` and ``attention_core`` speak in tensor size and
say 4 MiB; ``mcore_qk_bytes`` speaks in traffic and adds 8 MiB. Both are
right. Never put the two numbers side by side without naming which convention
each one uses.

**A cloned input would delete the effect, silently.** ``Tensor.clone()``
preserves a layout only for a tensor that is non-overlapping **and** dense. A
strided view of the fused buffer is neither, so ``clone`` returns a contiguous
tensor. The megatron leaf sets therefore clone the *buffer* and re-split it,
and make each leaf with ``detach()``; ``attention_core`` does the same for its
value. ``correctness_outputs`` maps the megatron outputs back to BLNH, outside
the timed region.

**The layout moves no value, so it moves no gate.** A copy is exact, both
engines read the same numbers in the same order, and TE flattens the leading
dimensions after the copy, so the reduction is identical. The correctness
gates are unchanged.

**Two memory costs are harness artifacts, and both are named rather than
hidden.** At the default workload the inputs grow by 12 MiB for **every** arm
in the scenario: the fused buffer is 16 MiB and the contiguous ``k_SBNH`` it
replaces was 4 MiB. On top of that, ``mcore/base`` holds three independent
leaf sets, and a megatron set now carries a whole fused buffer (16 MiB) plus
the query copy (8 MiB) where it used to carry a q/k pair (12 MiB) -- 36 MiB
more. So ``peak_memory_gib`` rises about 12 MiB on ``titan`` and
``copy_floor`` and about 48 MiB on ``mcore/base``, and the difference between
the two belongs to the three leaf sets rather than to megatron. Sharing one
buffer across the three sets would remove it and was declined: three
independent sets is what keeps one mode's gradient out of another mode's
leaf, and ``attention_core`` clones per set for the same reason.

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

    ``*_BLNH`` is TorchTitan's layout and ``*_SBNH`` is megatron's. Every
    tensor here is built at input time, never inside a timed closure.

    **Titan's three tensors are contiguous. Megatron's key is not.**
    ``qkv_fused_SBGR`` is the fused ``[s, b, ng, (Q + 2) * H]`` buffer
    megatron's QKV GEMM writes. ``q_SBNH`` is the contiguous tensor
    ``query.reshape`` allocates before ``q_layernorm``; ``k_SBNH`` is the
    strided view ``k_layernorm`` really receives, at storage offset
    ``Q * H`` inside that buffer. The module docstring gives the citations.

    The two gradient seeds stay contiguous on both engines, because both
    norms write a contiguous output: TE restores the logical shape of an
    already contiguous result (``ops/basic/rmsnorm.py:206``).
    """

    q_BLNH: torch.Tensor
    k_BLNH: torch.Tensor
    gq_BLNH: torch.Tensor
    gk_BLNH: torch.Tensor
    qkv_fused_SBGR: torch.Tensor
    q_SBNH: torch.Tensor
    k_SBNH: torch.Tensor
    gq_SBNH: torch.Tensor
    gk_SBNH: torch.Tensor
    q_weight: torch.Tensor
    k_weight: torch.Tensor
    eps: float
    qk_bytes: int
    mcore_qk_bytes: int


def _to_sbnh(tensor: torch.Tensor) -> torch.Tensor:
    """Megatron's contiguous [s, b, n, h] form of a titan [b, l, n, h] tensor.

    This is the transpose, and nothing else. The megatron arm's own q and k
    take a further step through the fused QKV buffer, which is where their
    layout is decided; see ``_megatron_qkv_buffer``.
    """
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


def _megatron_qkv_buffer(
    shape: PiperShape, q_SBNH: torch.Tensor, k_SBNH: torch.Tensor
) -> torch.Tensor:
    """The fused ``[s, b, ng, (Q + 2) * H]`` tensor megatron's QKV GEMM writes.

    The query and the key slots hold the canonical values, so both engines
    read the same numbers and only the memory layout differs. The head order
    agrees with titan's by construction: titan's query head ``n`` belongs to
    key/value group ``n // Q``, which is the group megatron's interleave puts
    it in, and ``megatron_weights.assert_qkv_roundtrip`` proves that mapping
    bitwise for the weights.

    The value slot stays zero. This cut holds no value, and nothing here ever
    reads that region -- it exists because it is what puts the key at offset
    ``Q * H`` with a stride of ``(Q + 2) * H`` between groups.
    """
    seq, batch = q_SBNH.shape[0], q_SBNH.shape[1]
    groups, per_group = shape.n_kv_heads, shape.heads_per_group
    head_dim = shape.head_dim
    query_width = per_group * head_dim
    fused = torch.zeros(
        (seq, batch, groups, (per_group + 2) * head_dim),
        dtype=q_SBNH.dtype,
        device=q_SBNH.device,
    )
    fused[..., :query_width] = q_SBNH.reshape(seq, batch, groups, query_width)
    fused[..., query_width : query_width + head_dim] = k_SBNH
    return fused


def _megatron_qk_norm_inputs(
    shape: PiperShape, fused: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The q and k tensors megatron's two norms really receive.

    One ``torch.split`` along the last dimension, exactly as
    ``get_query_key_value_tensors`` does it (``transformer/attention.py:1905``
    on the pinned rev; the ``SplitAlongDim`` branch above it is dead here,
    because TransformerEngine 2.17.1 exports no ``_SplitAlongDim`` from
    ``transformer_engine.pytorch.attention`` and megatron falls back). Both
    branches return views, so the layout is the same either way.

    The query then leaves the buffer, because ``reshape`` cannot merge the
    group dimension with the query heads inside a group when the row is
    ``(Q + 2) * H`` wide. The key does not. That reshape is megatron's own
    work and it belongs to ``qkv_prep``, so it runs here, at build time, and
    never inside a timed closure.
    """
    per_group, head_dim = shape.heads_per_group, shape.head_dim
    query, key, _value = torch.split(
        fused, [per_group * head_dim, head_dim, head_dim], dim=3
    )
    return (
        query.reshape(fused.shape[0], fused.shape[1], -1, head_dim),
        key,
    )


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
    fused = _megatron_qkv_buffer(shape, _to_sbnh(q), _to_sbnh(k))
    q_SBNH, k_SBNH = _megatron_qk_norm_inputs(shape, fused)
    # Read q and k, write q_out and k_out: the forward traffic, and what
    # copy_floor moves. The backward moves more, so read the GB/s column
    # in forward mode only.
    qk_bytes = 2 * (q.numel() + k.numel()) * q.element_size()
    return QkNormInputs(
        q_BLNH=q,
        k_BLNH=k,
        gq_BLNH=gq,
        gk_BLNH=gk,
        qkv_fused_SBGR=fused,
        q_SBNH=q_SBNH,
        k_SBNH=k_SBNH,
        gq_SBNH=_to_sbnh(gq),
        gk_SBNH=_to_sbnh(gk),
        q_weight=_norm_weight(shape, device, generator),
        k_weight=_norm_weight(shape, device, generator),
        eps=NORM_EPS,
        qk_bytes=qk_bytes,
        # The megatron arm moves one more read and one more write of the key:
        # TE's RMSNorm calls input_.contiguous() on the strided view inside
        # op_forward (ops/basic/rmsnorm.py:182), which is inside the timed
        # closure. 8 MiB above qk_bytes at the default workload.
        mcore_qk_bytes=qk_bytes + 2 * k.numel() * k.element_size(),
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


def _titan_leaves(inputs: QkNormInputs):
    """One differentiable leaf set in titan's layout.

    Both tensors are contiguous, which is what ``qkv_linear`` materializes,
    so ``clone`` carries the layout unchanged.
    """

    def make_leaves() -> tuple[torch.Tensor, torch.Tensor]:
        return (
            inputs.q_BLNH.clone().requires_grad_(),
            inputs.k_BLNH.clone().requires_grad_(),
        )

    return make_leaves


def _mcore_leaves(shape: PiperShape, inputs: QkNormInputs):
    """One differentiable leaf set in megatron's layout, strides included.

    One fused buffer per leaf set, split megatron's own way. The query leaf
    is the contiguous tensor ``reshape`` allocates; the key leaf is a view
    into that buffer and keeps its strides. ``detach`` is what makes a
    strided view a leaf -- ``clone`` would return a contiguous tensor and
    would delete the copy this arm exists to measure.
    """

    def make_leaves() -> tuple[torch.Tensor, torch.Tensor]:
        fused = inputs.qkv_fused_SBGR.clone()
        return tuple(
            tensor.detach().requires_grad_()
            for tensor in _megatron_qk_norm_inputs(shape, fused)
        )

    return make_leaves


def _qk_norm_arm(
    name: str,
    *,
    make_leaves,
    bytes_moved: int,
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

    ``make_leaves`` is the engine's own, because the layout is part of the
    measurand: see ``_titan_leaves`` and ``_mcore_leaves``.
    """
    forward_leaves = make_leaves()
    round_trip_leaves = make_leaves()
    check_leaves = make_leaves()

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
        bytes_moved=bytes_moved,
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
        make_leaves=_titan_leaves(inputs),
        bytes_moved=inputs.qk_bytes,
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

    The arm reads a contiguous query and a **strided key**, because that is
    what ``get_query_key_value_tensors`` hands its two norms. TE materializes
    the key inside ``op_forward``, so this arm's ``bytes_moved`` is
    ``mcore_qk_bytes`` and not the shared ``qk_bytes``. The module docstring
    carries the citations and the arithmetic.
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
        make_leaves=_mcore_leaves(shape, inputs),
        bytes_moved=inputs.mcore_qk_bytes,
        gq=inputs.gq_SBNH,
        gk=inputs.gk_SBNH,
        call=call,
        canonical=_to_blnh,
        q_module=q_module,
        k_module=k_module,
    )
