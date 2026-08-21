"""The pieces every kernel family's builders need and none of them owns.

Eight unrelated-looking helpers held together by one fact: each has three or
more of the family modules as consumers, so leaving any of them in a family
module would make ``rope`` a dependency of ``attention_core``, or
``expert_mlp`` of ``qkv_prep``.

* ``_randn`` / ``_randn_like`` / ``WEIGHT_STD`` -- seeded input material. Every
  tensor either builder kind materializes goes through them, drawn in fp32
  from the run's generator and cast down, so the bf16 inputs an arm sees do
  not depend on the dtype the caller happened to ask for.
* ``_reset_grads`` -- clears leaf and parameter grads between timed calls
  (expert_mlp, qkv_prep, attention_core).
* ``_require_grads`` -- turns a gradient the backward never produced into a
  named failure (ffn_norm, qkv_prep, embedding_stage, attn_residual,
  moe_residual).
* ``_compile_module`` -- the production compile treatment, ``fullgraph=True``.
  Applying it is a measurement decision, not a convenience: eager isolation
  races custom ops against materialization costs Inductor deletes, which
  inverted the combined-SwiGLU verdict outright.
* ``_assert_kernel_marker`` -- the silent-fallback guard the rope overrides
  and all three titan attention_core arms depend on.
* ``_navigate`` -- walks a dotted attribute path down a built model
  (attn_out_proj, qkv_prep, lm_head_projection).
* ``_projection_arm`` -- the timed closures every cross-engine linear shares
  (attn_out_proj, lm_head_projection).
* ``initialize_megatron_single_rank`` -- the process-global state every
  megatron-core arm needs before a ``GPTModel`` builds. Seven family modules
  call it.

``_randn_like`` is the reason this module is not simply "whatever rope did
not need": it sat inside the rope section of the single arm-builder module
this package replaced, while ``attention_inputs`` called it from three
hundred lines away. That is exactly the coupling a per-family split has to
resolve rather than inherit.

The leading underscores are kept from that file and are still accurate:
nothing outside this package calls the underscored names. The engine resolves
builder paths by ``module:function`` and every such path names a public
``*_inputs``, ``*_reference`` or ``build_*`` symbol in a family module, so no
dotted string anywhere in the repository refers to anything here.

**Module scope stays free of megatron, TransformerEngine and torchtitan.**
This module is worker-side (``tests/test_import_boundaries.py``
``WORKER_SIDE_MODULES``), and the parent never imports it, but every family
module does. A module-scope megatron import here would therefore land in
every kernel worker of every scenario, including the ones that build no
megatron arm at all, and per-arm process isolation depends on the opposite.
``tests/test_import_boundaries.py``'s
``test_operations_modules_defer_every_implementation_import`` sweeps this
file with the rest of the package and enforces it.
"""

from __future__ import annotations

import os
import socket
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile

from benchmarks.kernel.engine.arm import BuiltArm


WEIGHT_STD = 0.02

# The seed megatron's CUDA RNG tracker takes when the caller names none. The
# tracker decides megatron's own weight initialization; every mcore arm in
# this package overwrites the parameters it measures from the scenario's
# shared inputs afterwards, so this value fixes the rest of the built model
# rather than anything a table reports.
MCORE_INIT_SEED = 42


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


# What a missing gradient means when no call site says otherwise. Every caller
# that has a sharper sentence passes it as ``detail``; the sentence is the
# only part of the message that ever differed between the five copies this
# helper replaced.
MISSING_GRADIENT_DETAIL = (
    "the correctness gate cannot compare a tensor that does not exist"
)


def _require_grads(
    arm: str,
    outputs: dict[str, torch.Tensor | None],
    detail: str = MISSING_GRADIENT_DETAIL,
) -> dict[str, torch.Tensor]:
    """Turn a missing gradient into a named failure, not an AttributeError.

    ``outputs`` is the correctness dictionary the arm is about to return, so
    the check runs over exactly the tensors the gates compare. The dictionary
    comes back unchanged, which lets a caller write ``return
    _require_grads(name, {...})`` instead of naming it twice. A caller that
    wants only the guard may discard the result.

    ``detail`` is the second half of the message, and it is what each cut
    knows and this module does not. A residual add whose backward reached one
    operand is not the same failure as a norm whose gain took no gradient,
    and the sentence a reader needs names the operation rather than the
    helper. The default says what is true of every caller.
    """
    missing = sorted(name for name, value in outputs.items() if value is None)
    if missing:
        raise RuntimeError(
            f"{arm}: backward produced no gradient for {', '.join(missing)}; "
            f"{detail}"
        )
    return outputs


def _compile_module(module: nn.Module) -> nn.Module:
    """Compile a module-scope arm the way production runs it.

    Eager isolation races custom ops against materialization costs Inductor
    deletes, which inverts verdicts (the combined SwiGLU layout wins eager
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


def _navigate(root: object, path: str) -> Any:
    """Walk a dotted attribute path, indexing on a numeric segment.

    Every cross-engine arm reaches its megatron module this way, from a path
    the scenario derives from the weight name it already owns. The numeric
    segment is what ``decoder.layers.0.self_attention.linear_proj`` needs;
    a path that names no layer -- ``output_layer`` -- takes the same call.
    """
    node = root
    for segment in path.split("."):
        node = node[int(segment)] if segment.isdigit() else getattr(node, segment)
    return node


def _projection_arm(
    *,
    name: str,
    weight_owner: nn.Module,
    call: Callable[[torch.Tensor], torch.Tensor],
    x_native: torch.Tensor,
    grad_native: torch.Tensor,
    canonical_in: tuple[int, ...],
    canonical_out: tuple[int, ...],
    notes: dict[str, Any] | None = None,
) -> BuiltArm:
    """Forward and forward+backward over one engine's native tensor shape.

    ``x_native`` and ``grad_native`` are already in the shape that engine's
    module expects, because a reshape inside a timed closure would be timed as
    the projection. ``canonical_in`` and ``canonical_out`` put the outputs
    back into ``[batch, seq_len, *]`` for the gates, which run outside the
    timed region.

    ``weight_owner`` is the uncompiled module. A compiled arm times a
    ``torch.compile`` wrapper, and the parameters the gradients land on belong
    to the module inside it.

    The gradients are cleared before every ``forward_backward``. Without that
    the weight gradient would accumulate across a burst of thousands of calls,
    and the arm would time an addition over the whole weight on top of the
    GEMM.
    """
    forward_leaf = x_native.clone().requires_grad_()
    round_trip_leaf = x_native.clone().requires_grad_()
    check_leaf = x_native.clone().requires_grad_()

    def forward():
        return call(forward_leaf)

    def forward_backward() -> None:
        _reset_grads(round_trip_leaf, weight_owner)
        torch.autograd.backward(call(round_trip_leaf), grad_native)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(check_leaf, weight_owner)
        out = call(check_leaf)
        torch.autograd.backward(out, grad_native)
        return {
            "out": out.detach().reshape(canonical_out),
            "x_grad": check_leaf.grad.reshape(canonical_in),
            "weight_grad": weight_owner.weight.grad,
        }

    return BuiltArm(
        name=name,
        calls={"forward": forward, "forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
        notes=dict(notes or {}),
    )


def _free_port() -> int:
    """A port nobody holds, for the single-rank rendezvous store.

    Copied from ``benchmarks/e2e/megatron/train.py`` rather than imported: an
    ``operations`` module must not pull the e2e driver into a kernel worker.
    Timing workers run one after another, so a fixed port would eventually
    meet its own predecessor in TIME_WAIT.
    """
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def initialize_megatron_single_rank(seed: int = MCORE_INIT_SEED) -> None:
    """Put megatron's process-global state in place. Safe to repeat.

    Megatron needs four things before a ``GPTModel`` builds, and all four are
    process-global: its checkout on ``sys.path``, a torch.distributed process
    group, an initialized ``parallel_state``, and a seeded CUDA RNG tracker.
    ``benchmarks/e2e/megatron/train.py`` does the same steps in the same
    order for the e2e arm.

    Two orderings are load-bearing. ``add_megatron_to_path`` must run before
    any megatron import, because the submodule is not pip-installed. And
    ``configure_te_environment`` must run before anything imports
    TransformerEngine, because it sets the ``NVTE_NORM_*`` and cuDNN-frontend
    variables TE reads at import; that is what routes the norms through
    cuDNN on this host.

    **On a second call in one process this seeds the RNG tracker again, and
    that is the point.** Every arm is timed alone in its own process, so a
    timing worker always finds model parallel uninitialized and seeds. The
    correctness pass builds every arm of a scenario in one interpreter, so
    the second mcore arm finds it initialized. Were the seed conditional on
    that branch, the second arm would draw its weight initialization from a
    tracker the first arm's build had already advanced, and would differ from
    the arm the timing worker measures -- which is exactly what
    ``_seeded_build``'s re-seed exists to prevent.
    ``model_parallel_cuda_manual_seed`` calls ``_CUDA_RNG_STATE_TRACKER.
    reset()`` before it adds any state
    (``megatron/core/tensor_parallel/random.py``), and ``reset()`` clears both
    the state dict and the seed set, so repeating it raises nothing.

    The process group and ``parallel_state`` are each guarded, because
    repeating either does raise.

    ``seed`` is the value handed to ``model_parallel_cuda_manual_seed``.
    Callers that want the run's own seed instead of the default pass
    ``torch.initial_seed()``, which ``_seeded_build`` sets immediately before
    every builder runs.

    Device 0 is the requested GPU: the worker runs under
    ``CUDA_VISIBLE_DEVICES=<gpu>``
    (``benchmarks/execution/environment.py``), which is also what
    ``engine/run.py``'s ``torch.device("cuda")`` resolves to.

    Every heavy import is deferred into this body. See this module's
    docstring for why that is a hard requirement rather than a style
    preference.
    """
    from benchmarks.models.piper_qwen3.megatron_bootstrap import (
        add_megatron_to_path,
        configure_te_environment,
    )

    add_megatron_to_path()
    configure_te_environment()

    if not torch.distributed.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
        torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if not parallel_state.model_parallel_is_initialized():
        parallel_state.initialize_model_parallel()
    model_parallel_cuda_manual_seed(seed)
