"""Arm builders for the ``embedding_stage`` kernel scenario.

The token-embedding lookup at the top of the model. TorchTitan runs it at
``Decoder.forward``'s ``h = self.tok_embeddings(tokens)``
(``third_party/torchtitan/torchtitan/models/common/decoder.py:272``);
megatron-core runs it inside ``GPTModel._preprocess`` as
``self.embedding(input_ids=..., position_ids=...)``
(``third_party/Megatron-LM/megatron/core/models/gpt/gpt_model.py:345``), which
reaches ``LanguageModelEmbedding.forward``
(``megatron/core/models/common/embeddings/language_model_embedding.py:103``).
The scenario is cross-engine and it publishes one ratio.

**Neither engine pays a layout copy, and the first version of this scenario
charged titan for one. That was wrong, and here is why.** Megatron's embedding
ends with::

    if not self.reduce_scatter_embeddings:
        # Data format change to avoid explicit tranposes : [b s h] --> [s b h].
        embeddings = embeddings.transpose(0, 1).contiguous()

at ``language_model_embedding.py:122-124``, and that branch *is* entered here
(``reduce_scatter_embeddings`` at ``:51-56`` needs ``config.sequence_parallel``,
which stays at megatron's ``False`` default,
``megatron/core/model_parallel_config.py:94``). It is nonetheless **free**. Our
megatron driver packs a batch as THD, one row of ``[1, batch * seq_len]``
(``benchmarks/e2e/megatron/data.py:54``), so ``embeddings`` is ``[1, T, D]``
and the transpose yields ``[T, 1, D]``. PyTorch's contiguity test skips
size-1 dimensions, so that view is already contiguous and ``.contiguous()``
returns ``self``. Measured on the pinned torch (2.14.0.dev20260729+cu130) at
the default workload::

    [1,T,D].transpose(0,1) -> (4096, 1, 1024) strides (1024, 4194304, 1)
        is_contiguous=True,  contiguous() returned the same object
    [B,L,D].transpose(0,1) -> (1024, 4, 1024) strides (1024, 1048576, 1)
        is_contiguous=False, contiguous() allocated

The backward is free for the same reason: the incoming ``[T, 1, D]`` gradient
transposes back to a contiguous ``[1, T, D]``.

So megatron performs this transpose in no run this repository makes, and the
symmetric-looking charge would have been a penalty against an opponent's cost
of zero. It was measured, at the ``normal`` shape and per forward call, as
16 MiB of traffic on the anchor against 32 MiB on the titan arm -- on a
scenario with no arithmetic in it at all. ``benchmarks/kernel/schema.py:418-423``
already records the identical ``[T, 1, ·]`` reshape at ``attn_out_proj`` as "a
label and not a measured difference"; this scenario now says the same thing
about the same packing, and the cut is the lookup on both engines.

**The megatron arm still pays a wrapper the titan arm does not.**
``LanguageModelEmbedding.forward`` carries ``@nvtx_decorator()``
(``:102``), and the decorator wraps unconditionally: ``_nvtx_enabled``
defaults to ``False`` and is checked *inside* ``nvtx_range_push`` /
``nvtx_range_pop`` (``megatron/core/utils.py:2673``, ``:2712``, ``:2739``), so
a disabled profile still costs a Python closure and two function calls. Add to
that a second ``nn.Module.__call__`` (the outer ``LanguageModelEmbedding``
around the inner ``VocabParallelEmbedding``) and a ``Dropout`` call that ATen
short-circuits at ``p == 0``. Sub-microsecond each, and this scenario is
dispatch-bound, so they are named here rather than left to a reader.

**We reimplement ``GPTModel._preprocess``'s training branch rather than call
it.** The arm calls ``model.embedding`` directly and supplies ``position_ids
=None``. Reimplemented from ``gpt_model.py:339-357``, whose training branch is
four things: a ``padding_mask`` shape assertion, the embedding call, a
sequence-parallel scatter, and a sequence-parallel padding-mask scatter. Our
driver passes no ``padding_mask`` and ``sequence_parallel`` is ``False``, so
three of the four are inert and the fourth is the call this arm makes. Calling
``_preprocess`` itself would additionally build ``rotary_pos_emb``
(``gpt_model.py:372-411``), which belongs to another scenario -- see below.

**The RoPE-state handoff is excluded, on both engines, and it is not free on
megatron's side.** The partition names ``tok_embeddings(tokens)`` plus the
position / RoPE-state handoff as this cut. On the titan side that handoff is
empty: ``Decoder.forward`` takes ``positions`` as an argument and passes it
straight to each block (``decoder.py:261-275``), so there is no work to time.
On the megatron side it is not empty -- ``_preprocess`` builds
``rotary_pos_emb`` every step where titan's ``CosSinRoPE`` holds precomputed
buffers -- and that asymmetry is assigned to the RoPE scenario's provenance
rather than absorbed here, where it would appear as an unexplained megatron
cost inside a lookup.

**Both arms are eager, for two different reasons, and neither is a
relaxation.** Megatron compiles no whole transformer layer, so everything
``GPTModel`` builds runs eager end to end. TorchTitan's ``tok_embeddings`` is
a *sibling* of ``layers`` -- ``Decoder.__init__`` builds it at
``decoder.py:234`` and ``self.layers`` at ``:236`` -- and ``apply_compile``
walks ``model.layers.named_children()`` alone
(``third_party/torchtitan/torchtitan/distributed/compile.py:57-58``), so the
embedding sits outside every compiled region in production. The registry
records a distinct ``eager_reason`` on each arm. This is the same correction
``final_norm`` carries, for the same structural reason, and it does not extend
to the scenarios whose modules live inside a block.

**The scenario declares a floor, because it has no arithmetic at all.** A
gather performs zero FLOPs: every microsecond either moves bytes or
dispatches a kernel. Ranking two arms without a bandwidth reference would be
ranking two numbers with no scale, so ``copy_floor`` runs the forward's
traffic -- one read and one write over ``[B, L, D]``, 16 MiB at the ``normal``
shape and the default workload -- and the ``x_floor`` column says whether the
ratio is a kernel claim at all. Two known biases, both stated rather than
corrected: the floor reads a contiguous ``[B, L, D]`` tensor where the real
gather reads scattered rows of a ``[V, D]`` table, so it understates the
gather's DRAM cost; and the floor's two buffers fit in an H200's L2 at the
``normal`` shape, where the embedding table does not. Only the floor declares
``bytes_moved``: its byte count is the forward traffic, and publishing it on
an arm would attach a forward GB/s figure to the ``forward_backward`` row as
well.

**Tokens are drawn uniformly over the whole vocabulary, which is the worst
case for the gather.** This repo trains on ``c4_test``, whose tokenizer emits
about 2020 distinct ids against a 151936-row table, so a real run touches
~4 MiB of embedding rows and keeps them in L2. A uniform draw of
``batch * seq_len`` ids over 151936 rows touches ~4096 scattered rows of a
297 MiB table and pays DRAM for every one. Read the absolute number as an
upper bound on the gather's memory cost. It also dilutes the ratio: both arms
gather the same rows through the same ``F.embedding`` call, so a large shared
term pushes the published number toward 1.0 and shrinks whatever difference
the two engines' wrappers create.

**The weight gradient is gated on its touched rows, and the whole-table norm
beside it is weak evidence rather than a proof.** An fp64 reference for a
``[V, D]`` gradient is 1.16 GiB at ``normal`` and 13.9 GiB at ``huge``, in a
process that also holds a whole ``GPTModel``, so the reference accumulates
only the rows the tokens touch. ``weight_grad_norm`` -- a scalar over the
*entire* table -- is reported next to the compacted rows to catch a write
outside them, and it catches only a gross one. A Frobenius norm over ``U``
touched rows moves by ``sqrt(1 + k/U) - 1`` when ``k`` further rows of similar
magnitude are contaminated; at ``U ~ 4041`` that is 1.2e-4 for one stray row
and needs roughly 163 of them to reach the 2e-2 gate. **A handful of wrongly
written rows is invisible to every gate this scenario has.** Comparing an
arm's whole-table norm against its own compacted-row norm was considered and
rejected: it removes the cross-precision confound but not the sensitivity,
because the untouched rows are exact zeros and contribute no error to either
reduction, so it would be the same 1.2e-4 with a tighter threshold. Say
"nothing here bounds a small stray write" rather than implying the norm does.

Every implementation import is deferred into the builder that needs it. One
arm per process is the point: the mcore arm pays for megatron and TE, the
titan arm pays for torchtitan, and neither pays for the other.
"""

from __future__ import annotations

import gc
import weakref
from dataclasses import dataclass

import torch

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    MCORE_BLANK_MLP,
    WEIGHT_STD,
    _randn,
    _require_grads,
    _reset_grads,
    initialize_megatron_single_rank,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.shape import PiperShape


MCORE_ARM_NAME = "mcore/base"
TITAN_ARM_NAME = "titan"

# Checked by name rather than by ``isinstance``, so the guard is testable on a
# host with no megatron: the class lives behind a deferred import and a CPU
# test builds a stand-in with the same attributes. ``attn_out_proj`` reads its
# mcore module class the same way and for the same reason.
MCORE_MODULE_CLASS = "LanguageModelEmbedding"
TITAN_MODULE_CLASS = "Embedding"


@dataclass
class EmbeddingStageInputs:
    tokens: torch.Tensor  # (B, L) int64, uniform over [0, vocab_size)
    grad_out: torch.Tensor  # (B, L, D) bf16, the canonical upstream gradient
    weight: torch.Tensor  # (V, D) bf16, loaded into both engines
    touched_ids: torch.Tensor  # (U,) int64, sorted unique ids in ``tokens``
    copy_bytes: int  # the floor's traffic: one read and one write of (B, L, D)


def embedding_stage_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> EmbeddingStageInputs:
    """The token ids, the upstream gradient, and the shared embedding table.

    The table is drawn at ``WEIGHT_STD``, which is where both engines
    initialize it: TorchTitan's ``_EMBEDDING_INIT`` and megatron's
    ``embedding_init_method`` are both a normal draw at ``init_method_std``.
    Values do not reach the timing -- a gather costs the same whatever it
    moves -- but they reach every correctness gate, and an all-ones table
    would hide an arm that returned the wrong rows.

    Token ids are uniform over the full vocabulary. The module docstring
    states what that does and does not model.

    ``touched_ids`` is computed once here rather than in each arm, so the
    correctness pass compares the same rows on both sides. ``torch.unique``
    returns them sorted, which is what lets the reference index into a compact
    gradient with ``searchsorted``.
    """
    batch, seq = workload.batch, workload.seq_len
    tokens = torch.randint(
        0,
        shape.vocab_size,
        (batch, seq),
        device=device,
        generator=generator,
        dtype=torch.int64,
    )
    grad_out = _randn((batch, seq, shape.dim), device, generator)
    # Drawn in fp32 and cast, like every other input here. The transient fp32
    # table is 594 MiB at ``normal`` and 7.0 GiB at ``huge``; it is freed
    # before any arm builds.
    weight = _randn(
        (shape.vocab_size, shape.dim),
        device,
        generator,
        torch.bfloat16,
        WEIGHT_STD,
    )
    return EmbeddingStageInputs(
        tokens=tokens,
        grad_out=grad_out,
        weight=weight,
        touched_ids=torch.unique(tokens),
        copy_bytes=2 * grad_out.numel() * grad_out.element_size(),
    )


def embedding_stage_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: EmbeddingStageInputs
) -> dict[str, torch.Tensor]:
    """fp64 truth for the gather and for its gradient.

    The forward is *exact*: a gather copies bf16 rows, so promoting them to
    fp64 afterwards loses nothing and both arms should sit at rel_l2 0.0
    against this. The gradient is not exact -- it sums ``batch * seq_len``
    bf16 rows into one accumulator, and each engine's kernel chooses its own
    order -- so it carries the usual bf16 reduction error.

    The gradient is accumulated over the touched rows only. See the module
    docstring for why, and for what the whole-table norm beside it does and
    does not detect.
    """
    dim = shape.dim
    out = inputs.weight[inputs.tokens].double()
    flat_ids = inputs.tokens.reshape(-1)
    # ``touched_ids`` is sorted, so a search gives each token its row in the
    # compact gradient. Exact by construction: every id in ``tokens`` is in
    # ``touched_ids``.
    compact_rows = torch.searchsorted(inputs.touched_ids, flat_ids)
    weight_grad_rows = torch.zeros(
        (inputs.touched_ids.numel(), dim),
        device=inputs.weight.device,
        dtype=torch.float64,
    )
    weight_grad_rows.index_add_(
        0, compact_rows, inputs.grad_out.reshape(-1, dim).double()
    )
    return {
        "out": out,
        "weight_grad_rows": weight_grad_rows,
        "weight_grad_norm": torch.linalg.vector_norm(weight_grad_rows).reshape(
            1
        ),
    }


def _assert_output_layout(
    name: str,
    out: torch.Tensor,
    native_shape: tuple[int, ...],
    canonical_shape: tuple[int, ...],
    to_canonical,
) -> None:
    """Refuse to time an arm whose output is not the buffer both arms declare.

    Three properties, and none of them proves the absence of a copy -- that is
    what ``_assert_layout_conversion_is_free`` is for, and the first version of
    this module wrongly used a contiguity test for it. A ``[T, 1, D]`` view of
    a ``[1, T, D]`` tensor *is* contiguous, so ``is_contiguous()`` returns
    green on exactly the free-view case the check was meant to detect.

    What the contiguity assertion is still worth: an arm that returned a
    strided view would hand ``embedding_dense_backward`` a strided gradient,
    which ATen materializes, so one arm's backward would pay a copy the other
    does not. The dtype assertion covers ``fp32_residual_connection``, which
    would make megatron return fp32 here
    (``language_model_embedding.py:139-140``).
    """
    if tuple(out.shape) != tuple(native_shape):
        raise RuntimeError(
            f"{name}: the embedding returned shape {tuple(out.shape)}, "
            f"expected {tuple(native_shape)}"
        )
    if out.dtype is not torch.bfloat16:
        raise RuntimeError(
            f"{name}: the embedding returned {out.dtype}, expected "
            "torch.bfloat16; an upcast here would move twice the bytes of the "
            "opposing arm"
        )
    if not out.is_contiguous():
        raise RuntimeError(
            f"{name}: the embedding returned a strided view, so its backward "
            "would materialize a gradient the opposing arm's does not"
        )
    canonical = to_canonical(out)
    if tuple(canonical.shape) != tuple(canonical_shape):
        raise RuntimeError(
            f"{name}: the canonical output is {tuple(canonical.shape)}, "
            f"expected {tuple(canonical_shape)}; the two arms would be "
            "compared across different token orders"
        )


def gathered_and_output(module, call, ids: torch.Tensor):
    """One call, returning both the raw gather and the module's output.

    A forward hook on ``word_embeddings`` is the only way to see the tensor
    ``LanguageModelEmbedding.forward`` transposes, because the module returns
    the transposed one and keeps no reference to the other.
    """
    captured: list[torch.Tensor] = []
    handle = module.word_embeddings.register_forward_hook(
        lambda _module, _args, output: captured.append(output)
    )
    try:
        out = call(ids)
    finally:
        handle.remove()
    if not captured:
        raise RuntimeError(
            "the embedding did not call word_embeddings, so the arm is not "
            "timing the gather it claims to"
        )
    return captured[-1], out


def _assert_layout_conversion_is_free(
    name: str, gathered: torch.Tensor, out: torch.Tensor
) -> None:
    """Refuse to time a megatron embedding that materializes its transpose.

    The scenario charges no layout conversion to either engine, and that is
    only honest while megatron's ``transpose(0, 1).contiguous()`` costs
    nothing. It costs nothing at the THD packing our driver uses -- a leading
    dimension of 1 makes the transposed view contiguous, so ``.contiguous()``
    returns ``self``. Change the packing to ``[batch, seq_len]`` and the same
    line allocates and copies ``batch * seq_len * dim`` bf16 elements, the
    titan arm gains nothing to match it, and the published ratio silently
    absorbs the difference.

    Storage identity is the test, not ``is_contiguous()``: the free view is
    contiguous, so a contiguity test passes on precisely the case this exists
    to distinguish.
    """
    if out.untyped_storage().data_ptr() != gathered.untyped_storage().data_ptr():
        raise RuntimeError(
            f"{name}: the embedding's transpose(0, 1).contiguous() allocated a "
            f"new buffer for a gather of shape {tuple(gathered.shape)}, so "
            "this arm pays a layout copy the opposing arm has no counterpart "
            "for; the scenario declares that neither engine pays one"
        )


def _embedding_stage_arm(
    name: str,
    *,
    module,
    weight: torch.Tensor,
    call,
    tokens_native: torch.Tensor,
    grad_native: torch.Tensor,
    native_shape: tuple[int, ...],
    canonical_shape: tuple[int, ...],
    to_canonical,
    touched_ids: torch.Tensor,
) -> BuiltArm:
    """Forward and forward+backward over one engine's embedding lookup.

    ``weight`` is the parameter itself, captured before anything wraps the
    module, so ``weight_grad_rows`` names the same tensor on both engines.
    There is no input leaf: token ids are integers and carry no gradient, so
    the whole gradient of this stage is the table's.

    ``grad_native`` must be contiguous in the arm's own layout. A transposed
    view would make ``embedding_dense_backward`` materialize it inside the
    timed call, which is a copy the opposing arm does not pay.

    **There is no isolated ``backward`` mode, deliberately.** The other module
    scenarios time backward by re-running a retained graph. Here the first
    backward allocates and zero-fills a dense ``[V, D]`` gradient and every
    later one accumulates into the existing buffer, so a retained-graph
    backward would alternate between two different amounts of traffic under
    one label. ``forward_backward`` clears the gradient before each call
    instead, which is what both engines do per step (both drivers zero
    gradients to ``None``), and backward cost stays recoverable as the
    difference between the two modes.
    """
    if not weight.requires_grad:
        raise RuntimeError(
            f"{name}: the embedding table does not require grad, so backward "
            "would skip the scatter-add that dominates this stage and the arm "
            "would do far less work than its opponent"
        )
    if not grad_native.is_contiguous():
        raise RuntimeError(
            f"{name}: the upstream gradient is a strided view, so the timed "
            "backward would materialize it and charge this arm a copy its "
            "opponent does not pay"
        )

    def forward():
        return call(tokens_native)

    def forward_backward() -> None:
        _reset_grads(module)
        torch.autograd.backward(call(tokens_native), grad_native)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(module)
        out = call(tokens_native)
        torch.autograd.backward(out, grad_native)
        grad = _require_grads(
            name,
            {"weight_grad": weight.grad},
            detail="nothing gates the scatter-add this stage is measured for",
        )["weight_grad"]
        return {
            "out": to_canonical(out.detach()),
            "weight_grad_rows": grad.index_select(0, touched_ids),
            # Over the whole table. It bounds a gross write outside the
            # touched rows and nothing finer; see the module docstring for the
            # arithmetic.
            "weight_grad_norm": torch.linalg.vector_norm(
                grad, dtype=torch.float32
            ).reshape(1),
        }

    probe = call(tokens_native)
    _assert_output_layout(
        name, probe, native_shape, canonical_shape, to_canonical
    )
    del probe

    return BuiltArm(
        name=name,
        calls={"forward": forward, "forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
    )


def build_embedding_stage_copy_floor(
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: EmbeddingStageInputs,
) -> BuiltArm:
    """The bandwidth floor: the gather's read and write, and nothing else.

    Eager on purpose: a floor measures the device, not an implementation.

    The source is ``grad_out`` because it is already allocated and already the
    right shape and dtype. Only its bytes matter here; its values never do.
    """
    source = inputs.grad_out
    destination = torch.empty_like(source)

    def forward() -> None:
        destination.copy_(source)

    return BuiltArm(
        name="copy_floor",
        calls={"forward": forward},
        correctness_outputs=dict,
        bytes_moved=inputs.copy_bytes,
    )


def titan_embedding_module(shape: PiperShape, device: torch.device):
    """The ``Decoder.tok_embeddings`` config node, built as a module.

    ``Decoder.__init__`` builds it with ``self.tok_embeddings =
    config.tok_embeddings.build()``
    (``third_party/torchtitan/torchtitan/models/common/decoder.py:234``), and
    ``_piper_1b_model`` is where our registry sets that node
    (``benchmarks/models/piper_qwen3/config_registry.py:116-122``). So this is
    the production module built by the production config, and the extraction
    is one attribute read -- there is one embedding whatever ``n_layers`` is,
    so no per-layer selection and no arithmetic over the layer count appears
    anywhere in this scenario.

    Built inside ``with device`` and at a bf16 default dtype rather than built
    and then moved, which is what ``build_titan_model`` does and for a reason
    that bites harder here: ``nn.Embedding.__init__`` allocates ``[V, D]`` and
    runs ``normal_`` over it, which on the host is 594 MiB and several seconds
    at ``normal`` and 7.0 GiB at ``huge``. The values are overwritten with the
    shared table immediately afterwards.
    """
    from benchmarks.models.piper_qwen3.config_registry import _piper_1b_model

    node = _piper_1b_model(fuse_qkv=True, shape=shape).tok_embeddings
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with device:
            return node.build()
    finally:
        torch.set_default_dtype(previous_dtype)


def _assert_titan_embedding(module, shape: PiperShape) -> None:
    """Refuse to time a titan embedding that is not the plain gather.

    ``Embedding.forward`` takes a vocab-parallel branch when ``tp_group`` is
    set -- a mask, a clamp, a gather and a masked multiply
    (``models/common/embedding.py:63-79``) -- and ``F.embedding``'s own
    optional behaviours (``padding_idx``, ``max_norm``, ``scale_grad_by_freq``,
    ``sparse``) each change the kernel or the gradient. None of them has a
    megatron counterpart in this configuration, so any of them would make the
    ratio compare two different functions rather than two implementations of
    one.
    """
    class_name = type(module).__name__
    if class_name != TITAN_MODULE_CLASS:
        raise RuntimeError(
            f"{TITAN_ARM_NAME}: tok_embeddings is {class_name}, not "
            f"{TITAN_MODULE_CLASS}; the config node no longer builds the "
            "module this arm claims to time"
        )
    expected = (shape.vocab_size, shape.dim)
    actual = tuple(module.weight.shape)
    if actual != expected:
        raise RuntimeError(
            f"{TITAN_ARM_NAME}: tok_embeddings.weight is {actual}, expected "
            f"{expected}"
        )
    if getattr(module, "tp_group", None) is not None:
        raise RuntimeError(
            f"{TITAN_ARM_NAME}: tok_embeddings carries a tp_group, so forward "
            "takes the vocab-parallel branch; megatron runs at tp_size 1 here "
            "and the two arms would no longer compute the same function"
        )
    for attribute, wanted in (
        ("padding_idx", None),
        ("max_norm", None),
        ("scale_grad_by_freq", False),
        ("sparse", False),
    ):
        found = getattr(module, attribute)
        if found != wanted:
            raise RuntimeError(
                f"{TITAN_ARM_NAME}: tok_embeddings.{attribute} is {found!r}, "
                f"expected {wanted!r}; it changes the gather or its gradient "
                "and megatron applies no counterpart"
            )


def build_embedding_stage_titan(
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: EmbeddingStageInputs,
) -> BuiltArm:
    """TorchTitan's ``tok_embeddings``: ``F.embedding``, eager, BSD throughout.

    Eager because ``apply_compile`` never reaches ``Decoder.tok_embeddings``;
    see the module docstring. No layout conversion is applied here, because
    megatron performs none either at the packing our driver runs -- an earlier
    version of this arm appended ``.transpose(0, 1).contiguous()`` to match a
    megatron copy that does not exist, and the module docstring records why
    that was withdrawn.
    """
    batch, seq, dim = workload.batch, workload.seq_len, shape.dim
    device = inputs.weight.device
    module = titan_embedding_module(shape, device)
    _assert_titan_embedding(module, shape)
    with torch.no_grad():
        module.weight.copy_(inputs.weight)

    return _embedding_stage_arm(
        TITAN_ARM_NAME,
        module=module,
        weight=module.weight,
        call=module,
        tokens_native=inputs.tokens,
        grad_native=inputs.grad_out,
        native_shape=(batch, seq, dim),
        canonical_shape=(batch, seq, dim),
        # Titan runs BSD end to end, so its native layout is the canonical one
        # and there is nothing to convert.
        to_canonical=lambda out: out,
        touched_ids=inputs.touched_ids,
    )


def _assert_mcore_embedding(module, shape: PiperShape) -> None:
    """Refuse to time a megatron embedding that is not the plain gather.

    Seven ways ``LanguageModelEmbedding`` could stop being the module this arm
    claims, each of which leaves the gates green because each is numerically
    valid on its own terms:

    * ``add_position_embedding`` adds a second lookup and an add
      (``language_model_embedding.py:116-118``) that titan has no counterpart
      for.
    * ``num_tokentypes`` adds a third lookup, a permute and an add
      (``:126-130``).
    * ``reduce_scatter_embeddings`` moves the transpose into
      ``VocabParallelEmbedding`` and follows it with a collective
      (``tensor_parallel/layers.py:360-372``), so the arm would time a
      collective the scenario does not declare.
    * ``use_mup`` with ``mup_embedding_mult != 1.0`` adds an elementwise
      multiply (``:135-136``). Inert at every profile in this repo, so this is
      defence in depth.
    * ``fp32_residual_connection`` upcasts the output (``:139-140``), doubling
      the bytes the arm moves.
    * a nonzero ``hidden_dropout`` adds a mask kernel and makes the arm
      nondeterministic (``:90``, ``:156``). At ``p == 0`` ATen's ``dropout``
      returns its input untouched, so the module is exactly the gather.
    * a ``[V, D]`` weight that is not the full table means the vocabulary is
      sharded and the arm is timing a fraction of the work.

    ``deterministic_mode`` is refused as well: it swaps ``F.embedding`` for
    ``weight[masked_input]`` (``tensor_parallel/layers.py:351-355``), which is
    a different backward kernel, and titan exposes no matching switch.
    """
    class_name = type(module).__name__
    if class_name != MCORE_MODULE_CLASS:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: GPTModel.embedding is {class_name}, not "
            f"{MCORE_MODULE_CLASS}; the model no longer builds the module "
            "this arm claims to time"
        )
    if module.add_position_embedding:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: the embedding adds a learned position "
            "embedding, which titan has no counterpart for; this model is "
            "built with position_embedding_type='rope'"
        )
    if module.num_tokentypes > 0 or module.tokentype_embeddings is not None:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: the embedding carries {module.num_tokentypes} "
            "token types, adding a lookup and a permute titan has no "
            "counterpart for"
        )
    if module.reduce_scatter_embeddings:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: reduce_scatter_embeddings is set, so the "
            "transpose moves inside VocabParallelEmbedding and a collective "
            "follows it; this scenario declares no collective"
        )
    config = getattr(module, "config", None)
    if config is None:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: the embedding carries no config, so the "
            "settings that add work to its forward cannot be read"
        )
    if getattr(config, "use_mup", False) and (
        float(getattr(config, "mup_embedding_mult", 1.0)) != 1.0
    ):
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: use_mup scales the embeddings by "
            f"{config.mup_embedding_mult}, adding an elementwise multiply "
            "titan has no counterpart for"
        )
    if getattr(config, "fp32_residual_connection", False):
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: fp32_residual_connection upcasts the output, "
            "so this arm would move twice the bytes of the opposing arm"
        )
    dropout = float(module.embedding_dropout.p)
    if dropout != 0.0:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: hidden_dropout is {dropout}, so the timed "
            "call holds a mask kernel titan has no counterpart for and the "
            "arm is nondeterministic"
        )
    expected = (shape.vocab_size, shape.dim)
    actual = tuple(module.word_embeddings.weight.shape)
    if actual != expected:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: word_embeddings.weight is {actual}, expected "
            f"{expected}; a shorter table means the vocabulary is sharded and "
            "this arm is timing a fraction of the gather"
        )
    if getattr(module.word_embeddings, "deterministic_mode", False):
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: deterministic_mode selects weight[ids] over "
            "F.embedding, which is a different backward kernel; titan exposes "
            "no matching switch"
        )


def _watch_released_parameters(
    model: object, module: torch.nn.Module, kept: torch.Tensor
) -> tuple[weakref.ref, ...]:
    """Take a weak reference to every parameter this arm does not keep.

    The arm keeps one tensor, the embedding table.
    ``_assert_mcore_embedding`` has already refused a learned position
    embedding and a token-type embedding, so the module owns nothing else.
    This re-reads the module and proves that rather than assuming it: a
    second parameter here would stay alive for a good reason, and
    ``_assert_parameters_released`` could then no longer separate it from a
    leak.
    """
    also_kept = [
        parameter for parameter in module.parameters() if parameter is not kept
    ]
    if also_kept:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: the embedding holds {len(also_kept)} "
            "parameters besides the table, so the release check cannot "
            "separate a kept tensor from a leaked one"
        )
    return tuple(
        weakref.ref(parameter)
        for parameter in model.parameters()
        if parameter is not kept
    )


def _assert_parameters_released(watched: tuple[weakref.ref, ...]) -> None:
    """Refuse to time an arm that still holds the model it dropped.

    ``memory_pass`` reads ``torch.cuda.max_memory_allocated``, which is a
    total and not a delta. It therefore charges every live allocation to the
    arm. A surviving ``GPTModel`` adds about 2 GiB to this arm's peak memory
    at the 1b shape and nothing to the titan arm's. The transient window
    between the build and the first sample loop costs more: the model is
    88.8 GiB at the 48b shape, and a device that carries it into that window
    can run out of memory.

    **This reads object identity, and a byte budget could not do the job
    here.** ``rope`` and ``qk_norm`` compare ``torch.cuda.memory_allocated``
    against a 64 MiB budget, which works because each of them keeps a few
    hundred bytes. This arm keeps the whole table: 297 MiB at the 1b shape
    and 1.16 GiB at 48b, both far above any budget that could still find a
    leaked layer. A weak reference has no such problem, and it needs no CUDA
    device to answer.

    **This also raises where those two print a warning.** Allocator rounding
    moves a byte count, so a failure there can come from the host. A weak
    reference that survives ``gc.collect()`` means this file kept a
    reference, which is a defect here and nowhere else. This arm is the
    scenario anchor, and a lost anchor writes no ``results.json`` at all;
    that price is accepted, because the check runs in the correctness pass,
    ahead of every timing worker, so a false alarm costs a run rather than a
    published number. A peak memory column that quietly grew is what this
    check exists to prevent, and a printed warning does not prevent it.
    """
    alive = [
        parameter
        for parameter in (reference() for reference in watched)
        if parameter is not None
    ]
    if not alive:
        return
    resident = sum(
        parameter.numel() * parameter.element_size() for parameter in alive
    )
    raise RuntimeError(
        f"{MCORE_ARM_NAME}: {len(alive)} of {len(watched)} megatron "
        f"parameters survived the release ({resident / 2**20:.0f} MiB still "
        "resident). The builder kept a reference to the GPTModel, so this "
        "arm's peak_memory_gib would report the model and the table "
        "together."
    )


def build_embedding_stage_mcore_base(
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: EmbeddingStageInputs,
) -> BuiltArm:
    """Megatron's ``GPTModel.embedding``: the gather, and a free transpose.

    The module comes from a whole ``GPTModel`` and is reached at
    ``model.embedding``. Building the whole model costs the shape's full
    parameter memory and a minute of wall clock per worker, and it buys the
    one property a hand-built ``LanguageModelEmbedding`` cannot have: the
    module is the one megatron's own constructor builds
    (``gpt_model.py:168-175``), with every kwarg megatron gives it.

    Token ids arrive as ``[1, batch * seq_len]``, which is the shape our
    megatron driver packs (``benchmarks/e2e/megatron/data.py:54``, THD with
    ``cu_seqlens`` at the document boundaries). The output is ``[batch *
    seq_len, 1, dim]`` after the transpose, holding the same rows in the same
    order as titan's ``[batch, seq_len, dim]`` once both are put back into the
    canonical shape -- and that packing is exactly why the transpose is free.
    ``_assert_layout_conversion_is_free`` proves it on a probe call rather
    than assuming it.
    """
    from benchmarks.models.piper_qwen3.mcore_profiles import BASE
    from benchmarks.models.piper_qwen3.megatron_model import build_model

    batch, seq, dim = workload.batch, workload.seq_len, shape.dim
    tokens = batch * seq

    initialize_megatron_single_rank(torch.initial_seed())
    model = build_model(
        seq_len=seq,
        shape=shape,
        profile=BASE,
        blank_parts=MCORE_BLANK_MLP,
    )
    module = model.embedding
    if module is None:
        raise RuntimeError(
            f"{MCORE_ARM_NAME}: the built GPTModel carries no embedding"
        )
    _assert_mcore_embedding(module, shape)
    with torch.no_grad():
        module.word_embeddings.weight.copy_(inputs.weight)
    weight = module.word_embeddings.weight
    # Nothing else here references the rest of the model, so the other
    # parameters are free while the embedding is timed. The collect is what
    # makes them go: a GPTModel holds reference cycles, so dropping the name
    # alone leaves the parameters resident until the next collection.
    watched = _watch_released_parameters(model, module, weight)
    del model
    gc.collect()
    torch.cuda.empty_cache()
    _assert_parameters_released(watched)

    def call(ids: torch.Tensor) -> torch.Tensor:
        # position_ids is unused at position_embedding_type='rope', which
        # ``_assert_mcore_embedding`` has just proved. This is the call
        # ``GPTModel._preprocess`` makes at gpt_model.py:345.
        return module(input_ids=ids, position_ids=None)

    tokens_native = inputs.tokens.reshape(1, tokens)
    gathered, probe = gathered_and_output(module, call, tokens_native)
    _assert_layout_conversion_is_free(MCORE_ARM_NAME, gathered, probe)
    del gathered, probe

    return _embedding_stage_arm(
        MCORE_ARM_NAME,
        module=module,
        weight=weight,
        call=call,
        tokens_native=tokens_native,
        # Contiguous: a reshape of a contiguous tensor. Its transpose back to
        # [1, T, D] inside backward is contiguous for the same size-1 reason
        # the forward's is, so no copy is paid in either direction.
        grad_native=inputs.grad_out.reshape(tokens, 1, dim),
        native_shape=(tokens, 1, dim),
        canonical_shape=(batch, seq, dim),
        to_canonical=lambda out: out.reshape(batch, seq, dim),
        touched_ids=inputs.touched_ids,
    )
