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
from torchtitan.overrides.fused_swiglu import (
    FusedGroupedExperts,
    silu_and_mul_backward_kernel,
    silu_and_mul_forward_kernel,
)
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
    combined_silu_and_mul_backward_kernel,
    combined_silu_and_mul_forward_kernel,
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
    # The overrides expose fused backward kernels; stock has none, so its
    # backward is what autograd generates from the same forward.
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
    backward_op = torch.ops.torchtitan.helion_cossin_rope_bwd

    def forward():
        return module(inputs.q, inputs.k, inputs.positions)

    def backward():
        return backward_op(inputs.gq, inputs.gk, module.cache, inputs.positions)

    _assert_kernel_marker(forward, HELION_MARKER, "helion")

    def correctness_outputs() -> dict[str, torch.Tensor]:
        q_out, k_out = module(inputs.q, inputs.k, inputs.positions)
        dq, dk = backward_op(
            inputs.gq, inputs.gk, module.cache, inputs.positions
        )
        return {"q_out": q_out, "k_out": k_out, "dq": dq, "dk": dk}

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
    backward_op = torch.ops.torchtitan_benchmarks.te_rope_bwd

    def forward():
        return module(inputs.q, inputs.k, inputs.positions)

    def backward():
        return backward_op(
            inputs.gq, inputs.gk, module.te_angles, inputs.positions
        )

    _assert_kernel_marker(forward, TE_MARKER, "te")

    def correctness_outputs() -> dict[str, torch.Tensor]:
        q_out, k_out = module(inputs.q, inputs.k, inputs.positions)
        dq, dk = backward_op(
            inputs.gq, inputs.gk, module.te_angles, inputs.positions
        )
        return {"q_out": q_out, "k_out": k_out, "dq": dq, "dk": dk}

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
    gate_up: torch.Tensor
    kernel_grad_out: torch.Tensor
    counts: torch.Tensor
    offsets: torch.Tensor
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
    offsets = torch.cumsum(counts, dim=0, dtype=torch.int32)
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
        gate_up=_randn((rows, 2 * hidden), device, generator),
        kernel_grad_out=_randn((rows, hidden), device, generator),
        counts=counts,
        offsets=offsets,
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
    return module


def build_swiglu_baseline(
    spec: Piper1BSpec, inputs: SwigluInputs
) -> BuiltArm:
    module = _build_swiglu_module(GroupedExperts, spec, inputs)
    return _swiglu_module_arm(
        "baseline", module, inputs, _stock_weight_grads
    )


def build_swiglu_titan_fused(
    spec: Piper1BSpec, inputs: SwigluInputs
) -> BuiltArm:
    module = _build_swiglu_module(FusedGroupedExperts, spec, inputs)
    return _swiglu_module_arm(
        "titan_fused", module, inputs, _fused_weight_grads
    )


def build_swiglu_piper_optimized(
    spec: Piper1BSpec, inputs: SwigluInputs
) -> BuiltArm:
    module = _build_swiglu_module(
        CombinedSwiGLUFusedGroupedExperts, spec, inputs
    )
    return _swiglu_module_arm(
        "piper_optimized", module, inputs, _fused_weight_grads
    )


def build_swiglu_titan_triton(
    spec: Piper1BSpec, inputs: SwigluInputs
) -> BuiltArm:
    rows = inputs.gate_up.shape[0]
    hidden = spec.moe_hidden_dim
    gate, up = inputs.gate_up.reshape(rows, hidden, 2).unbind(-1)

    def forward():
        return silu_and_mul_forward_kernel(gate, up, inputs.offsets)

    def backward():
        return silu_and_mul_backward_kernel(
            inputs.kernel_grad_out, gate, up, inputs.offsets
        )

    def correctness_outputs() -> dict[str, torch.Tensor]:
        grad_gate, grad_up = backward()
        grad_gate_up = torch.stack((grad_gate, grad_up), dim=-1).reshape_as(
            inputs.gate_up
        )
        return {"fwd_out": forward(), "grad_gate_up": grad_gate_up}

    return BuiltArm(
        name="titan_triton",
        calls={"forward": forward, "backward": backward},
        correctness_outputs=correctness_outputs,
    )


def build_swiglu_piper_optimized_triton(
    spec: Piper1BSpec, inputs: SwigluInputs
) -> BuiltArm:
    def forward():
        return combined_silu_and_mul_forward_kernel(
            inputs.gate_up, inputs.offsets
        )

    def backward():
        return combined_silu_and_mul_backward_kernel(
            inputs.kernel_grad_out, inputs.gate_up, inputs.offsets
        )

    def correctness_outputs() -> dict[str, torch.Tensor]:
        return {"fwd_out": forward(), "grad_gate_up": backward()}

    return BuiltArm(
        name="piper_optimized_triton",
        calls={"forward": forward, "backward": backward},
        correctness_outputs=correctness_outputs,
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
    return module


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
