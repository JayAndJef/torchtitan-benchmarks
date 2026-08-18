"""Arm builders for the ``rope`` kernel scenario.

Four arms over one pair of q/k tensors: the ``copy_floor`` bandwidth floor,
TorchTitan's ``CosSinRoPE``, the Helion kernel, and TransformerEngine's CUDA
kernel. ``rope_inputs`` materializes the shared tensors and
``rope_reference`` the fp64 ground truth; the three real arms are
structurally identical and differ only in which module they build, which is
deliberate -- the timed closures, the retained backward graph and the
correctness hook must be the same code for the comparison to mean anything.

Two facts specific to this family:

* ``HELION_MARKER`` and ``TE_MARKER`` are the kernel names the fallback guard
  greps for. Both modules degrade to the *numerically correct* stock path
  when their eligibility checks fail, so a correctness gate cannot tell a
  fallback from a success and only ``_assert_kernel_marker`` can. Each arm
  calls ``forward()`` once before profiling it, to compile the no-grad
  variant so the profile captures kernels rather than compilation.
* **Every implementation import is deferred into the builder that needs it**,
  which is the rule across ``operations/`` rather than a habit of this
  module. ``te_rope_override`` is the sharpest case -- importing it
  JIT-builds a CUDA extension needing the gcc-13 environment the runner
  injects for this scenario only, so a module-scope import would make the
  whole operations package unimportable without a C++20 compiler. The other
  two follow the same rule for the same reason at a lower cost: one arm per
  process means a process pays only for the arm it builds.
  ``tests/test_import_boundaries.py`` pins this, and deliberately never
  imports ``te_rope_override`` itself.

Both ``rope_inputs`` and every ``build_*`` take (shape, workload, ...) even
where they read only one, because ``resolve_symbol`` calls them all
identically.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    _assert_kernel_marker,
    _compile_module,
    _randn,
    _randn_like,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.shape import PiperShape


HELION_MARKER = "_helion__rope_cos_sin_fwd"
TE_MARKER = "fused_rope_forward_positions_kernel"


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
    from torchtitan.models.common.rope import CosSinRoPE

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
    from torchtitan.overrides.helion_rope import HelionCosSinRoPE

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
    from benchmarks.models.piper_qwen3.components.rope.te_rope_override import (
        TECosSinRoPE,
    )

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
