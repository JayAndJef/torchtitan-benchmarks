"""Arm builders for the ``attention`` kernel scenario.

Inner attention only -- the level at which the three implementations are
substitutable, and the level that keeps this from re-measuring the projection
work ``qkv`` already covers. All three arms consume the same q/k/v and the
same synthetic packed-document boundaries, delivered in the three mask forms
the backends need: a flex ``BlockMask`` at the default 128 block size, the
same mask at ``FLEX_FLASH_BLOCK_SIZE``, and THD ``cu_seqlens``.

**Every mask is built once, in ``attention_inputs``, and never inside a timed
closure.** ``create_varlen_metadata_for_document`` contains a device-to-host
sync and ``create_block_mask`` is itself a compiled call; either one inside a
closure would be timed as attention. Nothing on the torch side validates the
FLASH block size -- it is forwarded verbatim into FA4's block-sparse tensors,
so a mismatch surfaces inside FA4 rather than as a torch-level error.

The marker constants are captured by profiling, never guessed, because two of
the three backends can be wrong without failing. FA3 degrades to FA2 rather
than raising when it declines to register, so ``FA3_MARKER`` is what separates
a real FA3 measurement from an FA2 one wearing its label, and ``FA2_MARKERS``
records the failure signature. FA4 is the opposite case and still guarded:
``BACKEND="FLASH"`` hard-raises when ``flash_attn.cute`` is missing, so an arm
that runs at all ran FA4, and ``FA4_MARKER`` instead protects against a future
refactor quietly dropping ``kernel_options`` and leaving the arm measuring the
baseline under an FA4 label.

Every torchtitan import here is deferred into a function body -- ``FlexAttention``
and the mask helpers into the builders that need them, ``VarlenAttention``
into the FA3 arm alone. That last one is load-bearing rather than stylistic:
``VarlenAttention``'s constructor activates FA3, so a module-scope import
would make this whole module unimportable without the ``flash3`` dependency
group, taking the baseline and ``flex_flash`` arms down with it.
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
    _reset_grads,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.shape import PiperShape


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
