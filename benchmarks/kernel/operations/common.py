"""The pieces every kernel family's builders need and none of them owns.

Four unrelated-looking helpers held together by one fact: each has three or
more of the five family modules as consumers, so leaving any of them in a
family module would make ``rope`` a dependency of ``attention``, or ``swiglu``
of ``qkv``.

* ``_randn`` / ``_randn_like`` / ``WEIGHT_STD`` -- seeded input material. Every
  tensor either builder kind materializes goes through them, drawn in fp32
  from the run's generator and cast down, so the bf16 inputs an arm sees do
  not depend on the dtype the caller happened to ask for.
* ``_reset_grads`` -- clears leaf and parameter grads between timed calls
  (swiglu, qkv, attention).
* ``_compile_module`` -- the production compile treatment, ``fullgraph=True``.
  Applying it is a measurement decision, not a convenience: eager isolation
  races custom ops against materialization costs Inductor deletes, which
  inverted the swiglu verdict outright.
* ``_assert_kernel_marker`` -- the silent-fallback guard the rope overrides
  and all three attention arms depend on.

``_randn_like`` is the reason this module is not simply "whatever rope did
not need": it sat inside the rope section of the single arm-builder module
this package replaced, while ``attention_inputs`` called it from three
hundred lines away. That is exactly the coupling a per-family split has to
resolve rather than inherit.

The leading underscores are kept from that file and are still accurate:
nothing outside this package calls these. The engine resolves builder paths
by ``module:function`` and every such path names a public ``*_inputs``,
``*_reference`` or ``build_*`` symbol in a family module, so no dotted string
anywhere in the repository refers to anything here.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile


WEIGHT_STD = 0.02


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


def _randn_like(reference: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    values = torch.randn(
        reference.shape,
        device=reference.device,
        generator=generator,
        dtype=torch.float32,
    )
    return values.to(reference.dtype)


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
