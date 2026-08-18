"""GPU arm builders for kernel-isolation benchmarks.

Each ``*_inputs`` function materializes one scenario's shared tensors from
the model geometry (``PiperShape``) and the workload run through it
(``KernelWorkload``); each ``build_*`` function constructs one arm around
those tensors and returns a ``BuiltArm`` whose closures are the timed
operations. Both take the pair even where an individual builder reads only
one of them, because ``kernel_bench`` resolves every builder by dotted path
and calls them identically. Private helpers below take only what they use.
Timing closures and correctness runs use separate leaf tensors so retained
backward graphs are never disturbed.

``piper1b.rope.te_rope_override`` is imported only inside ``build_rope_te``:
importing it JIT-builds the TE CUDA extension, which needs the gcc-13
environment the runner injects for the rope scenario.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.config import CompileConfig
from torchtitan.models.common import FusedQKVLinear, Linear, QKVLinear
from torchtitan.models.common.moe import GroupedExperts
from torchtitan.models.common.rope import CosSinRoPE
from torchtitan.overrides.helion_rope import HelionCosSinRoPE

from benchmarks.kernel_bench import BuiltArm
from benchmarks.kernels import KernelWorkload
from piper1b.lm_head.losses import (
    FusedLinearCrossEntropyLoss,
    PiperOptimizedCrossEntropyLoss,
    TECrossEntropyLoss,
)
from piper1b.model_shape import PiperShape
from piper1b.swiglu.combined_swiglu import (
    CombinedSwiGLUFusedGroupedExperts,
    InductorSwiGLUFusedGroupedExperts,
)


WEIGHT_STD = 0.02
LOSS_COMPILE = CompileConfig(enable=True, components=["loss"])
HELION_MARKER = "_helion__rope_cos_sin_fwd"
TE_MARKER = "fused_rope_forward_positions_kernel"


def _randn(
    shape: tuple[int, ...],
    device: torch.device,
    generator: torch.Generator,
    dtype: torch.dtype = torch.bfloat16,
    scale: float = 1.0,
) -> torch.Tensor:
    values = torch.randn(
        shape, device=device, generator=generator, dtype=torch.float32
    )
    return (values * scale).to(dtype)


def _reset_grads(*tensors: torch.Tensor | nn.Module) -> None:
    for item in tensors:
        if isinstance(item, nn.Module):
            for parameter in item.parameters():
                parameter.grad = None
        else:
            item.grad = None


def _compile_module(module: nn.Module) -> nn.Module:
    """Compile a module-scope arm the way production runs it.

    Eager isolation races custom ops against materialization costs Inductor
    deletes, which inverts verdicts (the swiglu combined layout wins eager
    and loses compiled). fullgraph turns a graph break into a build failure
    instead of silently timing partially-eager code.
    """
    return torch.compile(module, fullgraph=True)


def _assert_kernel_marker(closure, marker: str, arm: str) -> None:
    """Refuse to time an arm whose fast path silently fell back.

    Helion and TE RoPE modules degrade to the numerically correct stock path
    on ineligible inputs, so correctness gates cannot catch a mis-timed arm;
    only the presence of the arm's marker kernel in a profile can.
    """
    with profile(activities=[ProfilerActivity.CUDA]) as captured:
        closure()
        torch.cuda.synchronize()
    if not any(marker in event.name for event in captured.events()):
        raise RuntimeError(
            f"{arm}: marker kernel {marker!r} absent from a profiled call; "
            f"the override fell back to the stock path"
        )


# --- rope ---------------------------------------------------------------


@dataclass
class RopeInputs:
    q: torch.Tensor
    k: torch.Tensor
    gq: torch.Tensor
    gk: torch.Tensor
    positions: torch.Tensor
    qk_bytes: int


def rope_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> RopeInputs:
    batch, seq = workload.batch, workload.seq_len
    q = _randn((batch, seq, shape.n_heads, shape.head_dim), device, generator)
    k = _randn((batch, seq, shape.n_kv_heads, shape.head_dim), device, generator)
    gq = _randn_like(q, generator)
    gk = _randn_like(k, generator)
    positions = (
        torch.arange(seq, device=device, dtype=torch.int64)
        .unsqueeze(0)
        .expand(batch, -1)
        .contiguous()
    )
    qk_bytes = 2 * (q.numel() + k.numel()) * q.element_size()
    return RopeInputs(
        q=q, k=k, gq=gq, gk=gk, positions=positions, qk_bytes=qk_bytes
    )


def _randn_like(reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    values = torch.randn(
        reference.shape,
        device=reference.device,
        generator=generator,
        dtype=torch.float32,
    )
    return values.to(reference.dtype)


def _rope_tables_fp64(
    shape: PiperShape, workload: KernelWorkload, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    half = shape.head_dim // 2
    inv_freq = 1.0 / (
        shape.rope_theta
        ** (
            torch.arange(0, shape.head_dim, 2, dtype=torch.float64, device=device)[
                :half
            ]
            / shape.head_dim
        )
    )
    t = torch.arange(workload.seq_len, dtype=torch.float64, device=device)
    angles = torch.cat([torch.outer(t, inv_freq)] * 2, dim=-1)
    return angles.cos(), angles.sin()


def _rope_truth_forward(
    x: torch.Tensor, cos64: torch.Tensor, sin64: torch.Tensor, half: int
) -> torch.Tensor:
    xf = x.double()
    c = cos64.unsqueeze(0).unsqueeze(2)
    s = sin64.unsqueeze(0).unsqueeze(2)
    x1, x2 = xf[..., :half], xf[..., half:]
    rotated = torch.cat((-x2, x1), dim=-1)
    return xf * c + rotated * s


def _rope_truth_backward(
    grad: torch.Tensor, cos64: torch.Tensor, sin64: torch.Tensor, half: int
) -> torch.Tensor:
    gf = grad.double()
    c = cos64.unsqueeze(0).unsqueeze(2)
    s = sin64.unsqueeze(0).unsqueeze(2)
    g1, g2 = gf[..., :half], gf[..., half:]
    s1, s2 = s[..., :half], s[..., half:]
    low = g1 * c[..., :half] + g2 * s2
    high = g2 * c[..., half:] - g1 * s1
    return torch.cat((low, high), dim=-1)


def rope_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: RopeInputs
) -> dict[str, torch.Tensor]:
    cos64, sin64 = _rope_tables_fp64(shape, workload, inputs.q.device)
    half = shape.head_dim // 2
    return {
        "q_out": _rope_truth_forward(inputs.q, cos64, sin64, half),
        "k_out": _rope_truth_forward(inputs.k, cos64, sin64, half),
        "dq": _rope_truth_backward(inputs.gq, cos64, sin64, half),
        "dk": _rope_truth_backward(inputs.gk, cos64, sin64, half),
    }


def build_rope_copy_floor(
    shape: PiperShape, workload: KernelWorkload, inputs: RopeInputs
) -> BuiltArm:
    q_out = torch.empty_like(inputs.q)
    k_out = torch.empty_like(inputs.k)

    def forward() -> None:
        q_out.copy_(inputs.q)
        k_out.copy_(inputs.k)

    return BuiltArm(
        name="copy_floor",
        calls={"forward": forward},
        correctness_outputs=dict,
        bytes_moved=inputs.qk_bytes,
        floor=True,
    )


def build_rope_baseline(
    shape: PiperShape, workload: KernelWorkload, inputs: RopeInputs
) -> BuiltArm:
    module = CosSinRoPE.Config(
        dim=shape.head_dim,
        max_seq_len=shape.max_seq_len,
        theta=shape.rope_theta,
    ).build()
    module.init_states(buffer_device=inputs.q.device)
    module = _compile_module(module)
    q_leaf = inputs.q.clone().requires_grad_()
    k_leaf = inputs.k.clone().requires_grad_()
    retained = module(q_leaf, k_leaf, inputs.positions)

    def forward():
        return module(inputs.q, inputs.k, inputs.positions)

    def backward() -> None:
        q_leaf.grad = None
        k_leaf.grad = None
        torch.autograd.backward(
            retained, (inputs.gq, inputs.gk), retain_graph=True
        )

    def correctness_outputs() -> dict[str, torch.Tensor]:
        q_out, k_out = module(inputs.q, inputs.k, inputs.positions)
        backward()
        return {
            "q_out": q_out,
            "k_out": k_out,
            "dq": q_leaf.grad,
            "dk": k_leaf.grad,
        }

    return BuiltArm(
        name="baseline",
        calls={"forward": forward, "backward": backward},
        correctness_outputs=correctness_outputs,
        bytes_moved=inputs.qk_bytes,
    )


def build_rope_helion(
    shape: PiperShape, workload: KernelWorkload, inputs: RopeInputs
) -> BuiltArm:
    module = HelionCosSinRoPE.Config(
        dim=shape.head_dim,
        max_seq_len=shape.max_seq_len,
        theta=shape.rope_theta,
    ).build()
    module.init_states(buffer_device=inputs.q.device)
    module = _compile_module(module)
    q_leaf = inputs.q.clone().requires_grad_()
    k_leaf = inputs.k.clone().requires_grad_()
    retained = module(q_leaf, k_leaf, inputs.positions)

    def forward():
        return module(inputs.q, inputs.k, inputs.positions)

    def backward() -> None:
        q_leaf.grad = None
        k_leaf.grad = None
        torch.autograd.backward(
            retained, (inputs.gq, inputs.gk), retain_graph=True
        )

    forward()  # compile the no-grad variant before profiling it
    _assert_kernel_marker(forward, HELION_MARKER, "helion")

    def correctness_outputs() -> dict[str, torch.Tensor]:
        q_out, k_out = module(inputs.q, inputs.k, inputs.positions)
        backward()
        return {
            "q_out": q_out,
            "k_out": k_out,
            "dq": q_leaf.grad,
            "dk": k_leaf.grad,
        }

    return BuiltArm(
        name="helion",
        calls={"forward": forward, "backward": backward},
        correctness_outputs=correctness_outputs,
        bytes_moved=inputs.qk_bytes,
    )


def build_rope_te(
    shape: PiperShape, workload: KernelWorkload, inputs: RopeInputs
) -> BuiltArm:
    from piper1b.rope.te_rope_override import TECosSinRoPE

    module = TECosSinRoPE.Config(
        dim=shape.head_dim,
        max_seq_len=shape.max_seq_len,
        theta=shape.rope_theta,
    ).build()
    module.init_states(buffer_device=inputs.q.device)
    module = _compile_module(module)
    q_leaf = inputs.q.clone().requires_grad_()
    k_leaf = inputs.k.clone().requires_grad_()
    retained = module(q_leaf, k_leaf, inputs.positions)

    def forward():
        return module(inputs.q, inputs.k, inputs.positions)

    def backward() -> None:
        q_leaf.grad = None
        k_leaf.grad = None
        torch.autograd.backward(
            retained, (inputs.gq, inputs.gk), retain_graph=True
        )

    forward()  # compile the no-grad variant before profiling it
    _assert_kernel_marker(forward, TE_MARKER, "te")

    def correctness_outputs() -> dict[str, torch.Tensor]:
        q_out, k_out = module(inputs.q, inputs.k, inputs.positions)
        backward()
        return {
            "q_out": q_out,
            "k_out": k_out,
            "dq": q_leaf.grad,
            "dk": k_leaf.grad,
        }

    return BuiltArm(
        name="te",
        calls={"forward": forward, "backward": backward},
        correctness_outputs=correctness_outputs,
        bytes_moved=inputs.qk_bytes,
    )


# --- swiglu ---------------------------------------------------------------


@dataclass
class SwigluInputs:
    x: torch.Tensor
    grad_out: torch.Tensor
    counts: torch.Tensor
    stock_state: dict[str, torch.Tensor]


def swiglu_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> SwigluInputs:
    rows = workload.batch * workload.seq_len * shape.top_k
    hidden = shape.moe_hidden_dim
    per_expert = rows // shape.num_experts
    counts = torch.full(
        (shape.num_experts,), per_expert, device=device, dtype=torch.int32
    )
    stock_state = {
        "w1_EFD": _randn(
            (shape.num_experts, hidden, shape.dim),
            device,
            generator,
            torch.float32,
            WEIGHT_STD,
        ),
        "w2_EDF": _randn(
            (shape.num_experts, shape.dim, hidden),
            device,
            generator,
            torch.float32,
            WEIGHT_STD,
        ),
        "w3_EFD": _randn(
            (shape.num_experts, hidden, shape.dim),
            device,
            generator,
            torch.float32,
            WEIGHT_STD,
        ),
    }
    return SwigluInputs(
        x=_randn((rows, shape.dim), device, generator),
        grad_out=_randn((rows, shape.dim), device, generator),
        counts=counts,
        stock_state=stock_state,
    )


def _swiglu_module_arm(
    name: str,
    module: nn.Module,
    inputs: SwigluInputs,
    weight_grads,
) -> BuiltArm:
    forward_leaf = inputs.x.clone().requires_grad_()
    backward_leaf = inputs.x.clone().requires_grad_()
    round_trip_leaf = inputs.x.clone().requires_grad_()
    check_leaf = inputs.x.clone().requires_grad_()
    retained = module(backward_leaf, inputs.counts)

    def forward():
        return module(forward_leaf, inputs.counts)

    def backward() -> None:
        _reset_grads(backward_leaf, module)
        torch.autograd.backward(retained, inputs.grad_out, retain_graph=True)

    def forward_backward() -> None:
        _reset_grads(round_trip_leaf, module)
        out = module(round_trip_leaf, inputs.counts)
        torch.autograd.backward(out, inputs.grad_out)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(check_leaf, module)
        out = module(check_leaf, inputs.counts)
        torch.autograd.backward(out, inputs.grad_out)
        grads = weight_grads(module)
        return {"out": out.detach(), "x_grad": check_leaf.grad, **grads}

    return BuiltArm(
        name=name,
        calls={
            "forward": forward,
            "backward": backward,
            "forward_backward": forward_backward,
        },
        correctness_outputs=correctness_outputs,
    )


def _stock_weight_grads(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        "w1_grad": module.w1_EFD.grad,
        "w2_grad": module.w2_EDF.grad,
        "w3_grad": module.w3_EFD.grad,
    }


def _fused_weight_grads(module: nn.Module) -> dict[str, torch.Tensor]:
    return {
        "w1_grad": module.w13.grad[:, :, 0, :],
        "w2_grad": module.w2_EDF.grad,
        "w3_grad": module.w13.grad[:, :, 1, :],
    }


def _build_swiglu_module(config_cls, shape: PiperShape, inputs: SwigluInputs):
    module = config_cls.Config(
        dim=shape.dim,
        hidden_dim=shape.moe_hidden_dim,
        num_experts=shape.num_experts,
    ).build()
    module.to(inputs.x.device)
    module.load_state_dict(inputs.stock_state)
    module.to(torch.bfloat16)
    return _compile_module(module)


def build_swiglu_baseline(
    shape: PiperShape, workload: KernelWorkload, inputs: SwigluInputs
) -> BuiltArm:
    module = _build_swiglu_module(GroupedExperts, shape, inputs)
    return _swiglu_module_arm(
        "baseline", module, inputs, _stock_weight_grads
    )


def build_swiglu_piper_optimized_triton(
    shape: PiperShape, workload: KernelWorkload, inputs: SwigluInputs
) -> BuiltArm:
    module = _build_swiglu_module(
        CombinedSwiGLUFusedGroupedExperts, shape, inputs
    )
    return _swiglu_module_arm(
        "piper_optimized_triton", module, inputs, _fused_weight_grads
    )


def build_swiglu_piper_optimized_inductor(
    shape: PiperShape, workload: KernelWorkload, inputs: SwigluInputs
) -> BuiltArm:
    module = _build_swiglu_module(
        InductorSwiGLUFusedGroupedExperts, shape, inputs
    )
    return _swiglu_module_arm(
        "piper_optimized_inductor", module, inputs, _fused_weight_grads
    )


# --- qkv ------------------------------------------------------------------


@dataclass
class QkvInputs:
    x: torch.Tensor
    gq: torch.Tensor
    gk: torch.Tensor
    gv: torch.Tensor
    weight_state: dict[str, torch.Tensor]


def qkv_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> QkvInputs:
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
    return QkvInputs(
        x=_randn((batch, seq, shape.dim), device, generator),
        gq=_randn(
            (batch, seq, shape.n_heads, shape.head_dim), device, generator
        ),
        gk=_randn(
            (batch, seq, shape.n_kv_heads, shape.head_dim), device, generator
        ),
        gv=_randn(
            (batch, seq, shape.n_kv_heads, shape.head_dim), device, generator
        ),
        weight_state=weight_state,
    )


def qkv_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: QkvInputs
) -> dict[str, torch.Tensor]:
    batch, seq = workload.batch, workload.seq_len
    x64 = inputs.x.double()

    def project(weight: torch.Tensor, heads: int) -> torch.Tensor:
        quantized = weight.to(torch.bfloat16).double()
        return F.linear(x64, quantized).view(
            batch, seq, heads, shape.head_dim
        )

    return {
        "q_out": project(inputs.weight_state["wq.weight"], shape.n_heads),
        "k_out": project(inputs.weight_state["wk.weight"], shape.n_kv_heads),
        "v_out": project(inputs.weight_state["wv.weight"], shape.n_kv_heads),
    }


def _qkv_arm(name: str, module: nn.Module, inputs: QkvInputs) -> BuiltArm:
    forward_leaf = inputs.x.clone().requires_grad_()
    backward_leaf = inputs.x.clone().requires_grad_()
    round_trip_leaf = inputs.x.clone().requires_grad_()
    check_leaf = inputs.x.clone().requires_grad_()
    grads = (inputs.gq, inputs.gk, inputs.gv)
    retained = module(backward_leaf)

    def forward():
        return module(forward_leaf)

    def backward() -> None:
        _reset_grads(backward_leaf, module)
        torch.autograd.backward(retained, grads, retain_graph=True)

    def forward_backward() -> None:
        _reset_grads(round_trip_leaf, module)
        torch.autograd.backward(module(round_trip_leaf), grads)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(check_leaf, module)
        q_out, k_out, v_out = module(check_leaf)
        torch.autograd.backward((q_out, k_out, v_out), grads)
        return {
            "q_out": q_out.detach(),
            "k_out": k_out.detach(),
            "v_out": v_out.detach(),
            "x_grad": check_leaf.grad,
        }

    return BuiltArm(
        name=name,
        calls={
            "forward": forward,
            "backward": backward,
            "forward_backward": forward_backward,
        },
        correctness_outputs=correctness_outputs,
    )


def _finalize_qkv(module: nn.Module, inputs: QkvInputs) -> nn.Module:
    module.to(inputs.x.device)
    module.load_state_dict(inputs.weight_state)
    module.to(torch.bfloat16)
    return _compile_module(module)


def build_qkv_baseline(
    shape: PiperShape, workload: KernelWorkload, inputs: QkvInputs
) -> BuiltArm:
    module = QKVLinear.Config(
        head_dim=shape.head_dim,
        wq=Linear.Config(
            in_features=shape.dim, out_features=shape.n_heads * shape.head_dim
        ),
        wkv=Linear.Config(
            in_features=shape.dim,
            out_features=shape.n_kv_heads * shape.head_dim,
        ),
    ).build()
    return _qkv_arm("baseline", _finalize_qkv(module, inputs), inputs)


def build_qkv_fused_qkv(
    shape: PiperShape, workload: KernelWorkload, inputs: QkvInputs
) -> BuiltArm:
    module = FusedQKVLinear.Config(
        head_dim=shape.head_dim,
        n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads,
        wqkv=Linear.Config(
            in_features=shape.dim, out_features=shape.qkv_out_features
        ),
    ).build()
    return _qkv_arm("fused_qkv", _finalize_qkv(module, inputs), inputs)


# --- attention --------------------------------------------------------------

# Captured by profiling, never guessed. FA3 degrades to FA2 rather than
# failing when it declines to register, so the marker is what separates a real
# FA3 measurement from an FA2 one wearing its label; the FA2 kernel names are
# recorded as the failure signature.
FLEX_ATTENTION_MARKER = "flex_attention"
FA3_MARKER = "FlashAttnFwdSm90"
FA2_MARKERS = ("pytorch_flash::flash_fwd", "flash_bwd_dq_dk_dv_loop")
# FA4 needs no failure signature: BACKEND="FLASH" raises when flash_attn.cute
# is missing rather than degrading to Triton, so an arm that runs at all ran FA4.
FA4_MARKER = "FlashAttentionForwardSm90"

# The FLASH backend wants a coarser q block than flex's default 128. Torch does
# not validate it -- the value is forwarded verbatim into FA4's block-sparse
# tensors -- so a mismatch surfaces inside FA4, not as a torch-level error.
FLEX_FLASH_BLOCK_SIZE = (256, 128)


@dataclass
class AttentionInputs:
    q: torch.Tensor  # (B, L, n_heads, head_dim)
    k: torch.Tensor  # (B, L, n_kv_heads, head_dim)
    v: torch.Tensor
    grad_out: torch.Tensor
    positions: torch.Tensor  # (B, L) int32, resets to 0 at document starts
    block_mask: object  # BlockMask at flex's default 128 block size
    block_mask_flash: object  # the same mask at FLEX_FLASH_BLOCK_SIZE
    cu_seqlens: torch.Tensor  # int32, packed document boundaries for THD
    scale: float
    num_documents: int


def _packed_positions(
    workload: KernelWorkload, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    """Synthetic packed-document positions: reset to 0 at each document start.

    A seeded mix of document lengths rather than one document per row, because
    a single full-length document would make the block-diagonal mask dense and
    hide exactly the sparsity these kernels exist to exploit. Every row starts
    at 0, which both the flex document mask and cu_seqlens construction
    require.
    """
    rows = []
    low = max(1, workload.seq_len // 16)
    high = max(low + 1, workload.seq_len // 2)
    for _ in range(workload.batch):
        positions, remaining = [], workload.seq_len
        while remaining > 0:
            length = int(
                torch.randint(low, high, (1,), generator=generator).item()
            )
            length = min(length, remaining)
            positions.extend(range(length))
            remaining -= length
        rows.append(positions)
    return torch.tensor(rows, device=device, dtype=torch.int32)


def attention_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> AttentionInputs:
    from torch.nn.attention.flex_attention import and_masks
    from torchtitan.models.common.attention import (
        create_attention_mask,
        create_varlen_metadata_for_document,
        get_causal_mask_mod,
        get_efficient_causal_mask_mod_for_packed_document,
    )

    batch, seq = workload.batch, workload.seq_len
    q = _randn((batch, seq, shape.n_heads, shape.head_dim), device, generator)
    k = _randn((batch, seq, shape.n_kv_heads, shape.head_dim), device, generator)
    v = _randn((batch, seq, shape.n_kv_heads, shape.head_dim), device, generator)
    grad_out = _randn_like(q, generator)

    cpu_generator = torch.Generator(device="cpu").manual_seed(
        int(generator.initial_seed()) & 0x7FFFFFFF
    )
    positions = _packed_positions(workload, device, cpu_generator)

    # Both mask forms are built ONCE here, never inside a timed closure:
    # create_varlen_metadata_for_document contains a .item() device-to-host
    # sync, and create_block_mask is itself a compiled call.
    def mask_at(block_size):
        return create_attention_mask(
            and_masks(
                get_causal_mask_mod(),
                get_efficient_causal_mask_mod_for_packed_document(positions),
            ),
            batch,
            None,
            seq,
            seq,
            device=device,
            BLOCK_SIZE=block_size,
            separate_full_blocks=True,
        )

    varlen = create_varlen_metadata_for_document(positions)

    return AttentionInputs(
        q=q,
        k=k,
        v=v,
        grad_out=grad_out,
        positions=positions,
        block_mask=mask_at(128),
        block_mask_flash=mask_at(FLEX_FLASH_BLOCK_SIZE),
        cu_seqlens=varlen.cu_seq_q,
        scale=shape.head_dim**-0.5,
        num_documents=int((positions == 0).sum()),
    )


def attention_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: AttentionInputs
) -> dict[str, torch.Tensor]:
    """fp64 masked-softmax attention, computed per (row, kv group).

    Chunked on purpose: the full [B, n_heads, L, L] fp64 score tensor is
    8.6 GiB at batch 4 / seq 4096 and 69 GiB at batch 32, so a one-shot
    reference would OOM exactly at the shapes worth measuring. Per chunk it
    is [heads_per_kv, L, L], which stays a few hundred MiB.
    """
    batch, seq = workload.batch, workload.seq_len
    heads_per_kv = shape.heads_per_group
    device = inputs.q.device

    document = torch.cumsum((inputs.positions == 0).int(), dim=1) - 1
    causal = torch.tril(torch.ones(seq, seq, device=device, dtype=torch.bool))

    out = torch.empty_like(inputs.q, dtype=torch.float64)
    dq = torch.empty_like(out)
    dk = torch.zeros(
        (batch, seq, shape.n_kv_heads, shape.head_dim),
        device=device,
        dtype=torch.float64,
    )
    dv = torch.zeros_like(dk)

    for b in range(batch):
        same_document = document[b][:, None] == document[b][None, :]
        mask = same_document & causal
        for group in range(shape.n_kv_heads):
            lo, hi = group * heads_per_kv, (group + 1) * heads_per_kv
            q_chunk = (
                inputs.q[b, :, lo:hi].double().detach().transpose(0, 1).requires_grad_()
            )
            k_chunk = inputs.k[b, :, group].double().detach().requires_grad_()
            v_chunk = inputs.v[b, :, group].double().detach().requires_grad_()

            scores = (q_chunk @ k_chunk.transpose(-1, -2)) * inputs.scale
            scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
            chunk = torch.softmax(scores, dim=-1) @ v_chunk

            grad = inputs.grad_out[b, :, lo:hi].double().transpose(0, 1)
            torch.autograd.backward(chunk, grad)

            out[b, :, lo:hi] = chunk.detach().transpose(0, 1)
            dq[b, :, lo:hi] = q_chunk.grad.transpose(0, 1)
            dk[b, :, group] = k_chunk.grad
            dv[b, :, group] = v_chunk.grad

    return {"out": out, "dq": dq, "dk": dk, "dv": dv}


def _attention_arm(name: str, inputs: AttentionInputs, call) -> BuiltArm:
    """Forward and forward+backward only, with independent leaf sets.

    There is no isolated ``backward`` mode here, for the same reason
    ``lm_head`` has none: the retained-graph trick every other scenario uses
    (run backward repeatedly with retain_graph=True) is incompatible with TE's
    fused-attention autograd function, which consumes its saved-tensor context
    on the first backward and then raises "ctx must have .tensor_objects".
    Dropping the mode from BOTH arms keeps them comparable -- backward cost is
    still recoverable as forward_backward minus forward.
    """

    def leaves():
        return (
            inputs.q.clone().requires_grad_(),
            inputs.k.clone().requires_grad_(),
            inputs.v.clone().requires_grad_(),
        )

    forward_leaves = leaves()
    round_trip_leaves = leaves()
    check_leaves = leaves()

    def forward():
        return call(*forward_leaves)

    def forward_backward() -> None:
        _reset_grads(*round_trip_leaves)
        torch.autograd.backward(call(*round_trip_leaves), inputs.grad_out)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(*check_leaves)
        out = call(*check_leaves)
        torch.autograd.backward(out, inputs.grad_out)
        return {
            "out": out.detach(),
            "dq": check_leaves[0].grad,
            "dk": check_leaves[1].grad,
            "dv": check_leaves[2].grad,
        }

    return BuiltArm(
        name=name,
        calls={"forward": forward, "forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
    )


def build_attention_baseline(
    shape: PiperShape, workload: KernelWorkload, inputs: AttentionInputs
) -> BuiltArm:
    from torchtitan.models.common.attention import FlexAttention

    module = FlexAttention.Config().build()
    enable_gqa = shape.n_heads > shape.n_kv_heads

    def call(q, k, v):
        # FlexAttention already holds a class-level torch.compile of
        # flex_attention, so the module is NOT wrapped again here; wrapping it
        # risks a double compile or a graph break around its spmd context.
        return module(
            q, k, v,
            attention_masks=inputs.block_mask,
            scale=inputs.scale,
            enable_gqa=enable_gqa,
        )

    call(*[t.clone().requires_grad_() for t in (inputs.q, inputs.k, inputs.v)])
    _assert_kernel_marker(
        lambda: call(inputs.q, inputs.k, inputs.v),
        FLEX_ATTENTION_MARKER,
        "baseline",
    )
    return _attention_arm("baseline", inputs, call)


def build_attention_flex_flash(
    shape: PiperShape, workload: KernelWorkload, inputs: AttentionInputs
) -> BuiltArm:
    """FlexAttention lowered to FlashAttention-4 instead of a Triton template.

    Same module, same BlockMask semantics and same mask_mod as baseline -- only
    the lowering differs -- so this pair isolates the kernel family, whereas
    baseline vs flash_attention_3 also changes the masking mechanism.
    """
    from torchtitan.models.common.attention import FlexAttention

    module = FlexAttention.Config(
        block_size=FLEX_FLASH_BLOCK_SIZE,
        kernel_options={"BACKEND": "FLASH"},
    ).build()
    enable_gqa = shape.n_heads > shape.n_kv_heads

    def call(q, k, v):
        # Not wrapped in _compile_module, matching baseline: the class holds
        # its own compile of flex_attention.
        return module(
            q, k, v,
            attention_masks=inputs.block_mask_flash,
            scale=inputs.scale,
            enable_gqa=enable_gqa,
        )

    call(*[t.clone().requires_grad_() for t in (inputs.q, inputs.k, inputs.v)])
    _assert_kernel_marker(
        lambda: call(inputs.q, inputs.k, inputs.v),
        FA4_MARKER,
        "flex_flash",
    )
    return _attention_arm("flex_flash", inputs, call)


def build_attention_flash3(
    shape: PiperShape, workload: KernelWorkload, inputs: AttentionInputs
) -> BuiltArm:
    """FlashAttention-3 varlen over the packed (THD) sequences.

    VarlenAttention's constructor activates FA3, so building this arm at all
    requires the flash3 dependency group; without it torch's registry raises
    ModuleNotFoundError rather than silently falling back.
    """
    from torchtitan.models.common.attention import (
        VarlenAttention,
        VarlenMetadata,
    )

    # Compiled: the baseline is too, via FlexAttention's class-level compile.
    module = _compile_module(VarlenAttention.Config().build())
    enable_gqa = shape.n_heads > shape.n_kv_heads
    metadata = VarlenMetadata(
        cu_seq_q=inputs.cu_seqlens,
        cu_seq_k=inputs.cu_seqlens,
        max_q=workload.seq_len,
        max_k=workload.seq_len,
    )

    def call(q, k, v):
        return module(
            q, k, v,
            attention_masks=metadata,
            scale=inputs.scale,
            enable_gqa=enable_gqa,
        )

    call(*[t_.clone().requires_grad_() for t_ in (inputs.q, inputs.k, inputs.v)])
    _assert_kernel_marker(
        lambda: call(inputs.q, inputs.k, inputs.v),
        FA3_MARKER,
        "flash_attention_3",
    )
    return _attention_arm("flash_attention_3", inputs, call)


# --- lm_head ----------------------------------------------------------------


@dataclass
class LmHeadInputs:
    hidden: torch.Tensor
    weight: torch.Tensor
    labels: torch.Tensor
    valid_tokens: float


def lm_head_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> LmHeadInputs:
    batch, seq = workload.batch, workload.seq_len
    weight = _randn(
        (shape.vocab_size, shape.dim),
        device,
        generator,
        torch.bfloat16,
        1.0 / shape.dim**0.5,
    )
    labels = torch.randint(
        shape.vocab_size,
        (batch, seq),
        device=device,
        generator=generator,
        dtype=torch.int64,
    )
    return LmHeadInputs(
        hidden=_randn((batch, seq, shape.dim), device, generator),
        weight=weight,
        labels=labels,
        valid_tokens=float(batch * seq),
    )


def _lm_head_arm(name: str, inputs: LmHeadInputs, loss_call) -> BuiltArm:
    hidden = inputs.hidden.clone().requires_grad_()
    weight = inputs.weight.clone().requires_grad_()

    def forward_backward():
        hidden.grad = None
        weight.grad = None
        loss = loss_call(hidden, weight)
        loss.backward()
        return loss

    def correctness_outputs() -> dict[str, torch.Tensor]:
        loss = forward_backward()
        return {
            "loss": loss.detach().float(),
            "hidden_grad": hidden.grad,
            "weight_grad": weight.grad,
        }

    return BuiltArm(
        name=name,
        calls={"forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
    )


def build_lm_head_baseline(
    shape: PiperShape, workload: KernelWorkload, inputs: LmHeadInputs
) -> BuiltArm:
    loss_obj = CrossEntropyLoss.Config(
        global_vocab_size=shape.vocab_size
    ).build(compile_config=LOSS_COMPILE)

    def loss_call(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, weight)
        loss, _ = loss_obj(logits, inputs.labels, inputs.valid_tokens)
        return loss

    return _lm_head_arm("baseline", inputs, loss_call)


def build_lm_head_fused_linear_ce(
    shape: PiperShape, workload: KernelWorkload, inputs: LmHeadInputs
) -> BuiltArm:
    loss_obj = FusedLinearCrossEntropyLoss.Config(
        batch_chunk_size=None, chunking_method=None
    ).build(compile_config=LOSS_COMPILE)
    hidden = inputs.hidden.clone().requires_grad_()
    lm_head = nn.Linear(
        shape.dim, shape.vocab_size, bias=False, device=hidden.device
    ).to(torch.bfloat16)
    with torch.no_grad():
        lm_head.weight.copy_(inputs.weight)
    loss_obj.set_lm_head(lm_head)

    def forward_backward():
        hidden.grad = None
        lm_head.weight.grad = None
        loss, _ = loss_obj(hidden, inputs.labels, inputs.valid_tokens)
        loss.backward()
        return loss

    def correctness_outputs() -> dict[str, torch.Tensor]:
        loss = forward_backward()
        return {
            "loss": loss.detach().float(),
            "hidden_grad": hidden.grad,
            "weight_grad": lm_head.weight.grad,
        }

    return BuiltArm(
        name="fused_linear_ce",
        calls={"forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
    )


def build_lm_head_te_fused_ce(
    shape: PiperShape, workload: KernelWorkload, inputs: LmHeadInputs
) -> BuiltArm:
    loss_obj = TECrossEntropyLoss.Config().build(compile_config=LOSS_COMPILE)

    def loss_call(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, weight)
        loss, _ = loss_obj(logits, inputs.labels, inputs.valid_tokens)
        return loss

    return _lm_head_arm("te_fused_ce", inputs, loss_call)


def build_lm_head_piper_optimized_te_ce(
    shape: PiperShape, workload: KernelWorkload, inputs: LmHeadInputs
) -> BuiltArm:
    loss_obj = PiperOptimizedCrossEntropyLoss.Config().build(
        compile_config=LOSS_COMPILE
    )

    def loss_call(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, weight)
        loss, _ = loss_obj(logits, inputs.labels, inputs.valid_tokens)
        return loss

    return _lm_head_arm("piper_optimized_te_ce", inputs, loss_call)
