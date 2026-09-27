"""Build the TorchTitan Qwen3-1B model in this process, from the registry.

The e2e arms never need this: they hand ``--config`` and ``--override.imports``
to a training subprocess, and the fork's ``ConfigManager`` does the work. But a
kernel arm has no subprocess and no ConfigManager, so anything that wants "the
module production builds, configured the way production configures it" has to
build it here.

Cross-engine kernel arms are the callers in prospect. They need this build so
that what they time is the production module and not a hand-constructed
lookalike -- the difference being every default the registry sets and a hand
construction forgets.

**Overrides are counted, not trusted.** ``--override.imports`` is applied in
the training subprocess and ``validate_arm`` proves it landed by counting
``[Override]`` log lines. In-process there are no log lines to count, so
``apply_config_overrides`` counts the replacements ``apply_overrides`` returns
and raises when the count is wrong. That check is the only guard some arms can
have: ``piper_optimized_inductor`` deliberately emits no distinctive kernel
name, so ``_assert_kernel_marker`` cannot see it, and an override that silently
matched nothing would leave the arm measuring the baseline under its own name.

The expected count is ``overrides_per_block * n_layers``, which is the same
arithmetic ``validate_arm`` rule 2 uses. Verified against the fork: the swiglu
override on a 2-layer model returns exactly 2 replacement lines, one per block.

**The override registry is process-global and that is safe here.** The fork
keeps registered overrides in a module-level registry, so the correctness pass
-- which builds every arm in one interpreter -- accumulates them. Accumulation
is inert: ``apply_overrides`` activates only the targets it is *named*, so an
override another arm registered is never applied and never conflicts.
"""

from __future__ import annotations

from typing import Sequence

import torch
from torchtitan.models.common import CosSinRoPE, Embedding, Linear
from torchtitan.models.qwen3 import (
    _build_qwen3_moe_layers,
    _EMBEDDING_INIT,
    _output_linear_init,
    _qwen3_norm,
    Qwen3Model,
)

from benchmarks.models.piper_qwen3.shape import PiperShape


def _piper_1b_model(
    *, fuse_qkv: bool, shape: PiperShape, attn_backend: str = "flex"
) -> Qwen3Model.Config:
    dim = shape.dim
    head_dim = shape.head_dim
    n_layers = shape.n_layers
    vocab_size = shape.vocab_size
    layers = _build_qwen3_moe_layers(
        fuse_qkv=fuse_qkv,
        n_layers=n_layers,
        dim=dim,
        n_heads=shape.n_heads,
        n_kv_heads=shape.n_kv_heads,
        head_dim=head_dim,
        moe_hidden_dim=shape.moe_hidden_dim,
        num_experts=shape.num_experts,
        top_k=shape.top_k,
        attn_backend=attn_backend,
        moe_comm_backend="standard",
        rope=CosSinRoPE.Config(
            dim=head_dim,
            max_seq_len=shape.max_seq_len,
            theta=shape.rope_theta,
        ),
    )
    # piper sets load_balance_coeff=None (no aux-free load balancing); the
    # torchtitan builder defaults to 1e-3, so match piper explicitly.
    for layer in layers:
        layer.moe.load_balance_coeff = None
    return Qwen3Model.Config(
        vocab_size=vocab_size,
        dim=dim,
        norm=_qwen3_norm(dim),
        tok_embeddings=Embedding.Config(
            num_embeddings=vocab_size,
            embedding_dim=dim,
            # Real init: weight tying is OFF (as in piper), so the embedding is
            # a free parameter. (_EMBEDDING_SKIP_INIT is only valid with tying.)
            param_init=_EMBEDDING_INIT,
        ),
        lm_head=Linear.Config(
            in_features=dim,
            out_features=vocab_size,
            param_init=_output_linear_init(dim),
        ),
        layers=layers,
    )


def apply_config_overrides(
    config,
    targets: Sequence[str],
    *,
    expected: int,
) -> list[str]:
    """Apply ``targets`` to ``config`` in place and prove they landed.

    ``expected`` is how many config nodes must be replaced. Pass 0 with no
    targets. A mismatch raises: too few means an override matched nothing (a
    wrong target class, or a glob that selects no node), too many means it
    claimed nodes the arm did not intend to change.

    Returns the replacement lines, which are the same strings the training
    subprocess logs as ``[Override] ...``.
    """
    from torchtitan.config.override import OverrideConfig, apply_overrides

    if not targets and expected == 0:
        return []
    lines = apply_overrides(OverrideConfig(imports=list(targets)), config)
    if len(lines) != expected:
        raise RuntimeError(
            f"expected {expected} override replacement(s) from "
            f"{list(targets)}, got {len(lines)}: "
            + ("; ".join(lines) if lines else "nothing matched")
            + ". An override that matches nothing leaves the arm measuring "
            "the baseline under its own name, and the arms whose kernels have "
            "no distinctive name have no other guard."
        )
    return lines


def build_titan_model(
    *,
    shape: PiperShape,
    fuse_qkv: bool = True,
    attn_backend: str = "flex",
    overrides: Sequence[str] = (),
    overrides_per_block: int = 0,
    device: str | torch.device = "cuda",
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 42,
    eval_mode: bool = False,
):
    """The registry's Qwen3 model, built on ``device`` in ``dtype``.

    ``dtype`` is delivered as the default dtype during construction rather
    than by casting afterwards, because that is what the trainer does:
    ``trainer.py:299-303`` opens ``utils.set_default_dtype`` around
    ``model_config.build()``. Constructing in bf16 also never allocates the
    fp32 parameters a later cast would have to make first.

    The default dtype is process-global, so the ``finally`` restores it. The
    correctness pass builds every arm in one interpreter, and a leaked bf16
    default would change every tensor a later arm creates -- including the
    fp64 references the attention gates compare against.

    **This does not make ``training.dtype`` the whole dtype story.** There is
    no mixed-precision wrapper and no autocast in ``parallelize_piper1b``, but
    upstream wraps the MoE router gate in ``torch.autocast(dtype=float32)``
    (``models/common/moe.py:292``), and this model has MoE on every layer. An
    arm that times the router times an fp32 GEMM, and the fp32 copy of the
    hidden state that autocast materializes to feed it.

    ``seed`` is set before ``init_states`` so two processes that build the
    same arm get the same parameters. Kernel-bench relies on that: the
    correctness pass and the timing pass are different processes, and a gate
    that checked different weights than the timing measured would prove
    nothing.
    """
    config = _piper_1b_model(
        fuse_qkv=fuse_qkv, shape=shape, attn_backend=attn_backend
    )
    apply_config_overrides(
        config, overrides, expected=overrides_per_block * shape.n_layers
    )
    device = torch.device(device)
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    try:
        with device:
            model = config.build()
    finally:
        torch.set_default_dtype(previous_dtype)
    torch.manual_seed(seed)
    model.init_states(buffer_device=device)
    if eval_mode:
        model.eval()
    return model
