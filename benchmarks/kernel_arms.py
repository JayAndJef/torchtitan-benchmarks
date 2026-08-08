"""GPU arm builders for kernel-isolation benchmarks.

Each ``*_inputs`` function materializes one scenario's shared tensors from
the Piper-1B spec; each ``build_*`` function constructs one arm around those
tensors and returns a ``BuiltArm`` whose closures are the timed operations.
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
from benchmarks.kernels import Piper1BSpec
from piper1b.lm_head.losses import (
    FusedLinearCrossEntropyLoss,
    PiperOptimizedCrossEntropyLoss,
    TECrossEntropyLoss,
)
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
    spec: Piper1BSpec, device: torch.device, generator: torch.Generator
) -> RopeInputs:
    batch, seq = spec.batch, spec.seq_len
    q = _randn((batch, seq, spec.n_heads, spec.head_dim), device, generator)
    k = _randn((batch, seq, spec.n_kv_heads, spec.head_dim), device, generator)
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
    spec: Piper1BSpec, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    half = spec.head_dim // 2
    inv_freq = 1.0 / (
        spec.theta
        ** (
            torch.arange(0, spec.head_dim, 2, dtype=torch.float64, device=device)[
                :half
            ]
            / spec.head_dim
        )
    )
    t = torch.arange(spec.seq_len, dtype=torch.float64, device=device)
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
    spec: Piper1BSpec, inputs: RopeInputs
) -> dict[str, torch.Tensor]:
    cos64, sin64 = _rope_tables_fp64(spec, inputs.q.device)
    half = spec.head_dim // 2
    return {
        "q_out": _rope_truth_forward(inputs.q, cos64, sin64, half),
        "k_out": _rope_truth_forward(inputs.k, cos64, sin64, half),
        "dq": _rope_truth_backward(inputs.gq, cos64, sin64, half),
        "dk": _rope_truth_backward(inputs.gk, cos64, sin64, half),
    }


def build_rope_copy_floor(
    spec: Piper1BSpec, inputs: RopeInputs
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


def build_rope_baseline(spec: Piper1BSpec, inputs: RopeInputs) -> BuiltArm:
    module = CosSinRoPE.Config(
        dim=spec.head_dim, max_seq_len=spec.max_seq_len, theta=spec.theta
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


def build_rope_helion(spec: Piper1BSpec, inputs: RopeInputs) -> BuiltArm:
    module = HelionCosSinRoPE.Config(
        dim=spec.head_dim, max_seq_len=spec.max_seq_len, theta=spec.theta
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


def build_rope_te(spec: Piper1BSpec, inputs: RopeInputs) -> BuiltArm:
    from piper1b.rope.te_rope_override import TECosSinRoPE

    module = TECosSinRoPE.Config(
        dim=spec.head_dim, max_seq_len=spec.max_seq_len, theta=spec.theta
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
    spec: Piper1BSpec, device: torch.device, generator: torch.Generator
) -> SwigluInputs:
    rows = spec.batch * spec.seq_len * spec.top_k
    hidden = spec.moe_hidden_dim
    per_expert = rows // spec.num_experts
    counts = torch.full(
        (spec.num_experts,), per_expert, device=device, dtype=torch.int32
    )
    stock_state = {
        "w1_EFD": _randn(
            (spec.num_experts, hidden, spec.dim),
            device,
            generator,
            torch.float32,
            WEIGHT_STD,
        ),
        "w2_EDF": _randn(
            (spec.num_experts, spec.dim, hidden),
            device,
            generator,
            torch.float32,
            WEIGHT_STD,
        ),
        "w3_EFD": _randn(
            (spec.num_experts, hidden, spec.dim),
            device,
            generator,
            torch.float32,
            WEIGHT_STD,
        ),
    }
    return SwigluInputs(
        x=_randn((rows, spec.dim), device, generator),
        grad_out=_randn((rows, spec.dim), device, generator),
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


def _build_swiglu_module(config_cls, spec: Piper1BSpec, inputs: SwigluInputs):
    module = config_cls.Config(
        dim=spec.dim,
        hidden_dim=spec.moe_hidden_dim,
        num_experts=spec.num_experts,
    ).build()
    module.to(inputs.x.device)
    module.load_state_dict(inputs.stock_state)
    module.to(torch.bfloat16)
    return _compile_module(module)


def build_swiglu_baseline(
    spec: Piper1BSpec, inputs: SwigluInputs
) -> BuiltArm:
    module = _build_swiglu_module(GroupedExperts, spec, inputs)
    return _swiglu_module_arm(
        "baseline", module, inputs, _stock_weight_grads
    )


def build_swiglu_piper_optimized_triton(
    spec: Piper1BSpec, inputs: SwigluInputs
) -> BuiltArm:
    module = _build_swiglu_module(
        CombinedSwiGLUFusedGroupedExperts, spec, inputs
    )
    return _swiglu_module_arm(
        "piper_optimized_triton", module, inputs, _fused_weight_grads
    )


def build_swiglu_piper_optimized_inductor(
    spec: Piper1BSpec, inputs: SwigluInputs
) -> BuiltArm:
    module = _build_swiglu_module(
        InductorSwiGLUFusedGroupedExperts, spec, inputs
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
    spec: Piper1BSpec, device: torch.device, generator: torch.Generator
) -> QkvInputs:
    batch, seq = spec.batch, spec.seq_len
    q_out = spec.n_heads * spec.head_dim
    kv_out = spec.n_kv_heads * spec.head_dim
    weight_state = {
        "wq.weight": _randn(
            (q_out, spec.dim), device, generator, torch.float32, WEIGHT_STD
        ),
        "wk.weight": _randn(
            (kv_out, spec.dim), device, generator, torch.float32, WEIGHT_STD
        ),
        "wv.weight": _randn(
            (kv_out, spec.dim), device, generator, torch.float32, WEIGHT_STD
        ),
    }
    return QkvInputs(
        x=_randn((batch, seq, spec.dim), device, generator),
        gq=_randn((batch, seq, spec.n_heads, spec.head_dim), device, generator),
        gk=_randn(
            (batch, seq, spec.n_kv_heads, spec.head_dim), device, generator
        ),
        gv=_randn(
            (batch, seq, spec.n_kv_heads, spec.head_dim), device, generator
        ),
        weight_state=weight_state,
    )


def qkv_reference(
    spec: Piper1BSpec, inputs: QkvInputs
) -> dict[str, torch.Tensor]:
    batch, seq = spec.batch, spec.seq_len
    x64 = inputs.x.double()

    def project(weight: torch.Tensor, heads: int) -> torch.Tensor:
        quantized = weight.to(torch.bfloat16).double()
        return F.linear(x64, quantized).view(
            batch, seq, heads, spec.head_dim
        )

    return {
        "q_out": project(inputs.weight_state["wq.weight"], spec.n_heads),
        "k_out": project(inputs.weight_state["wk.weight"], spec.n_kv_heads),
        "v_out": project(inputs.weight_state["wv.weight"], spec.n_kv_heads),
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


def build_qkv_baseline(spec: Piper1BSpec, inputs: QkvInputs) -> BuiltArm:
    module = QKVLinear.Config(
        head_dim=spec.head_dim,
        wq=Linear.Config(
            in_features=spec.dim, out_features=spec.n_heads * spec.head_dim
        ),
        wkv=Linear.Config(
            in_features=spec.dim, out_features=spec.n_kv_heads * spec.head_dim
        ),
    ).build()
    return _qkv_arm("baseline", _finalize_qkv(module, inputs), inputs)


def build_qkv_fused_qkv(spec: Piper1BSpec, inputs: QkvInputs) -> BuiltArm:
    fused_out = (spec.n_heads + 2 * spec.n_kv_heads) * spec.head_dim
    module = FusedQKVLinear.Config(
        head_dim=spec.head_dim,
        n_heads=spec.n_heads,
        n_kv_heads=spec.n_kv_heads,
        wqkv=Linear.Config(in_features=spec.dim, out_features=fused_out),
    ).build()
    return _qkv_arm("fused_qkv", _finalize_qkv(module, inputs), inputs)


# --- attention --------------------------------------------------------------

# Captured by profiling, never guessed: TE picks its backend at runtime and
# can silently fall back to an unfused path.
TE_ATTENTION_MARKER = "cudnn_generated_fort_native_sdpa"
FLEX_ATTENTION_MARKER = "flex_attention"
# The FA2 kernels torch's varlen path uses. If a FlashAttention-3 arm is added,
# seeing these means FA3 declined to register -- a failure, not a success.
FA2_MARKERS = ("pytorch_flash::flash_fwd", "flash_bwd_dq_dk_dv_loop")


@dataclass
class AttentionInputs:
    q: torch.Tensor  # (B, L, n_heads, head_dim)
    k: torch.Tensor  # (B, L, n_kv_heads, head_dim)
    v: torch.Tensor
    grad_out: torch.Tensor
    positions: torch.Tensor  # (B, L) int32, resets to 0 at document starts
    block_mask: object  # BlockMask for FlexAttention
    cu_seqlens: torch.Tensor  # int32, packed document boundaries for THD
    scale: float
    num_documents: int


def _packed_positions(
    spec: Piper1BSpec, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    """Synthetic packed-document positions: reset to 0 at each document start.

    A seeded mix of document lengths rather than one document per row, because
    a single full-length document would make the block-diagonal mask dense and
    hide exactly the sparsity these kernels exist to exploit. Every row starts
    at 0, which both the flex document mask and cu_seqlens construction
    require.
    """
    rows = []
    low = max(1, spec.seq_len // 16)
    high = max(low + 1, spec.seq_len // 2)
    for _ in range(spec.batch):
        positions, remaining = [], spec.seq_len
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
    spec: Piper1BSpec, device: torch.device, generator: torch.Generator
) -> AttentionInputs:
    from torch.nn.attention.flex_attention import and_masks
    from torchtitan.models.common.attention import (
        create_attention_mask,
        create_varlen_metadata_for_document,
        get_causal_mask_mod,
        get_efficient_causal_mask_mod_for_packed_document,
    )

    batch, seq = spec.batch, spec.seq_len
    q = _randn((batch, seq, spec.n_heads, spec.head_dim), device, generator)
    k = _randn((batch, seq, spec.n_kv_heads, spec.head_dim), device, generator)
    v = _randn((batch, seq, spec.n_kv_heads, spec.head_dim), device, generator)
    grad_out = _randn_like(q, generator)

    cpu_generator = torch.Generator(device="cpu").manual_seed(
        int(generator.initial_seed()) & 0x7FFFFFFF
    )
    positions = _packed_positions(spec, device, cpu_generator)

    # Both mask forms are built ONCE here, never inside a timed closure:
    # create_varlen_metadata_for_document contains a .item() device-to-host
    # sync, and create_block_mask is itself a compiled call.
    block_mask = create_attention_mask(
        and_masks(
            get_causal_mask_mod(),
            get_efficient_causal_mask_mod_for_packed_document(positions),
        ),
        batch,
        None,
        seq,
        seq,
        device=device,
        BLOCK_SIZE=128,
        separate_full_blocks=True,
    )
    varlen = create_varlen_metadata_for_document(positions)

    return AttentionInputs(
        q=q,
        k=k,
        v=v,
        grad_out=grad_out,
        positions=positions,
        block_mask=block_mask,
        cu_seqlens=varlen.cu_seq_q,
        scale=spec.head_dim**-0.5,
        num_documents=int((positions == 0).sum()),
    )


def attention_reference(
    spec: Piper1BSpec, inputs: AttentionInputs
) -> dict[str, torch.Tensor]:
    """fp64 masked-softmax attention, computed per (row, kv group).

    Chunked on purpose: the full [B, n_heads, L, L] fp64 score tensor is
    8.6 GiB at batch 4 / seq 4096 and 69 GiB at batch 32, so a one-shot
    reference would OOM exactly at the shapes worth measuring. Per chunk it
    is [heads_per_kv, L, L], which stays a few hundred MiB.
    """
    batch, seq = spec.batch, spec.seq_len
    heads_per_kv = spec.n_heads // spec.n_kv_heads
    device = inputs.q.device

    document = torch.cumsum((inputs.positions == 0).int(), dim=1) - 1
    causal = torch.tril(torch.ones(seq, seq, device=device, dtype=torch.bool))

    out = torch.empty_like(inputs.q, dtype=torch.float64)
    dq = torch.empty_like(out)
    dk = torch.zeros(
        (batch, seq, spec.n_kv_heads, spec.head_dim),
        device=device,
        dtype=torch.float64,
    )
    dv = torch.zeros_like(dk)

    for b in range(batch):
        same_document = document[b][:, None] == document[b][None, :]
        mask = same_document & causal
        for group in range(spec.n_kv_heads):
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
    spec: Piper1BSpec, inputs: AttentionInputs
) -> BuiltArm:
    from torchtitan.models.common.attention import FlexAttention

    module = FlexAttention.Config().build()
    enable_gqa = spec.n_heads > spec.n_kv_heads

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


def build_attention_te(spec: Piper1BSpec, inputs: AttentionInputs) -> BuiltArm:
    # Imported inside the builder: TE needs its environment configured before
    # import, and the rope arms' precedent is to keep TE out of module scope.
    from megatron_baseline.location import configure_te_environment

    configure_te_environment()
    from transformer_engine.pytorch import DotProductAttention

    batch, seq = spec.batch, spec.seq_len
    tokens = batch * seq
    module = DotProductAttention(
        num_attention_heads=spec.n_heads,
        kv_channels=spec.head_dim,
        num_gqa_groups=spec.n_kv_heads,
        attention_dropout=0.0,
        qkv_format="thd",
        # THD packing requires a padding mask type; "causal" alone is rejected.
        attn_mask_type="padding_causal",
        softmax_scale=inputs.scale,
    ).cuda()

    def call(q, k, v):
        out = module(
            q.reshape(tokens, spec.n_heads, spec.head_dim),
            k.reshape(tokens, spec.n_kv_heads, spec.head_dim),
            v.reshape(tokens, spec.n_kv_heads, spec.head_dim),
            cu_seqlens_q=inputs.cu_seqlens,
            cu_seqlens_kv=inputs.cu_seqlens,
            max_seqlen_q=seq,
            max_seqlen_kv=seq,
        )
        return out.view(batch, seq, spec.n_heads, spec.head_dim)

    _assert_kernel_marker(
        lambda: call(inputs.q, inputs.k, inputs.v),
        TE_ATTENTION_MARKER,
        "te_attention",
    )
    return _attention_arm("te_attention", inputs, call)


# --- lm_head ----------------------------------------------------------------


@dataclass
class LmHeadInputs:
    hidden: torch.Tensor
    weight: torch.Tensor
    labels: torch.Tensor
    valid_tokens: float


def lm_head_inputs(
    spec: Piper1BSpec, device: torch.device, generator: torch.Generator
) -> LmHeadInputs:
    batch, seq = spec.batch, spec.seq_len
    weight = _randn(
        (spec.vocab_size, spec.dim),
        device,
        generator,
        torch.bfloat16,
        1.0 / spec.dim**0.5,
    )
    labels = torch.randint(
        spec.vocab_size,
        (batch, seq),
        device=device,
        generator=generator,
        dtype=torch.int64,
    )
    return LmHeadInputs(
        hidden=_randn((batch, seq, spec.dim), device, generator),
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
    spec: Piper1BSpec, inputs: LmHeadInputs
) -> BuiltArm:
    loss_obj = CrossEntropyLoss.Config(
        global_vocab_size=spec.vocab_size
    ).build(compile_config=LOSS_COMPILE)

    def loss_call(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, weight)
        loss, _ = loss_obj(logits, inputs.labels, inputs.valid_tokens)
        return loss

    return _lm_head_arm("baseline", inputs, loss_call)


def build_lm_head_fused_linear_ce(
    spec: Piper1BSpec, inputs: LmHeadInputs
) -> BuiltArm:
    loss_obj = FusedLinearCrossEntropyLoss.Config(
        batch_chunk_size=None, chunking_method=None
    ).build(compile_config=LOSS_COMPILE)
    hidden = inputs.hidden.clone().requires_grad_()
    lm_head = nn.Linear(
        spec.dim, spec.vocab_size, bias=False, device=hidden.device
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
    spec: Piper1BSpec, inputs: LmHeadInputs
) -> BuiltArm:
    loss_obj = TECrossEntropyLoss.Config().build(compile_config=LOSS_COMPILE)

    def loss_call(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, weight)
        loss, _ = loss_obj(logits, inputs.labels, inputs.valid_tokens)
        return loss

    return _lm_head_arm("te_fused_ce", inputs, loss_call)


def build_lm_head_piper_optimized_te_ce(
    spec: Piper1BSpec, inputs: LmHeadInputs
) -> BuiltArm:
    loss_obj = PiperOptimizedCrossEntropyLoss.Config().build(
        compile_config=LOSS_COMPILE
    )

    def loss_call(hidden: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        logits = F.linear(hidden, weight)
        loss, _ = loss_obj(logits, inputs.labels, inputs.valid_tokens)
        return loss

    return _lm_head_arm("piper_optimized_te_ce", inputs, loss_call)
