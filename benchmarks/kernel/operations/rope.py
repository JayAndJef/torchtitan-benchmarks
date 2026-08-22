"""Arm builders for the ``rope`` cross-engine kernel scenario.

Rotary position embedding on q and k, after the qk norms and before the
attention core. TorchTitan runs it as one call,
``self.rope(xq_BLNH, xk_BLNH, positions)``
(``third_party/torchtitan/torchtitan/models/common/attention.py:963``).
Megatron-core runs it as two calls of ``apply_rotary_pos_emb``, one for the
query and one for the key
(``third_party/Megatron-LM/megatron/core/transformer/attention.py:1500-1506``
and ``:1518-1524``). Both closures here do q and then k, in that order, so
each arm carries the same ``n_heads + n_kv_heads`` rows -- 1.5x the query row
count at the piper geometry, not 2x.

**Rule 7: the two sides share their innermost arithmetic and nothing above
it.** ``benchmarks/models/piper_qwen3/components/rope/te_rope_standalone.cu``
states in its own header (lines 1-11) that ``fused_rope_block_forward``,
``fused_rope_block_backward``, ``fused_rope_forward_kernel`` and
``fused_rope_backward_kernel`` are copied from TransformerEngine's
``common/fused_rope/fused_rope.cu``, and that the local plumbing reproduces
TE's launcher configuration **for the BSHD tensor format**. Read carefully,
that header supports a narrower claim than "both sides run the same kernel",
and three things in the file itself bound it:

* **Only ``fused_rope_block_forward`` is common.** The two ``__global__``
  entry points differ and take different arguments: TE's
  ``fused_rope_forward_kernel`` (``:143``) takes ``const int *cu_seqlens``,
  while ``fused_rope_forward_positions_kernel`` (``:236``) is local, is not
  TE's, and takes ``const int64_t *positions``, which it reads at
  ``positions[b_id * s + s_id]`` (``:243``).
* **The launch geometry is not shared, and megatron's is not in this tree.**
  Both local launchers are ``dim3 blocks(s, b)`` over the dense tensor
  (``:361``, ``:423``). TE's THD kernel body computes
  ``t_id = s_id + start; if (t_id >= end) return;`` (``:157-158``), which only
  makes sense under a grid whose s-extent spans the longest document and whose
  b-extent is the document count. So ``mcore/base`` carries a device-side term
  that varies with the packing draw, and ``titan/te`` does not.
* **The copy itself cannot be checked from here.** The installed
  TransformerEngine 2.17.1 wheel ships no ``fused_rope.cu``, and this header
  names no TE version, so nothing in this tree proves the copied code still
  matches the binary megatron calls.

Report ``titan/te`` against ``mcore/base`` as two implementations that share
their inner arithmetic. Do **not** report it as one kernel beating another,
and do **not** report it as measuring the host wrapper alone.

**Rule 5: ``mcore/no_rope_fusion`` is device time PLUS host serialization,
and the number scales with the document count.** With ``cu_seqlens`` set, the
unfused path is ``_apply_rotary_pos_emb_thd``
(``megatron/core/models/common/embeddings/rope_utils.py:189``), which calls
``.tolist()`` on the segment lengths (``:222``) and ``.item()`` on the last
offset (``:224``) -- two device-to-host syncs -- and then walks the documents
in **Python**, one ``_apply_rotary_pos_emb_bshd`` call and one
``output.narrow(...).copy_(...)`` each (``:252-266``). So the row is a fused
CUDA kernel against a Python loop, not "fused TE against unfused TE".

**That branch is chosen on the data, and the other branch is refused.**
``has_packed_freqs`` (``:225``) compares the table's ``seq_len`` rows against
the ``batch * seq_len`` packed tokens, so it is True exactly at
``batch == 1``, and CASE 1 then rotates by each document's *global* offset
where every other arm here restarts at each document start -- a different
rotation, not a different implementation of one.
``_check_thd_reference_path`` raises on it at build time and captions CASE 2
with the document count the run actually measured.

**Rule 1: the compile treatments differ, by design.** The three titan arms run
under ``torch.compile(fullgraph=True)``, because that is what a transformer
block gives them end to end. Both mcore arms run eager: megatron compiles no
whole layer, and ``rope_utils.py`` carries no ``jit_fuser`` decoration at all,
so eager is what megatron itself runs.

**Marker guards are mandatory on ``titan/helion`` and ``titan/te``.** Both
modules fall back to the *numerically correct* stock path when their
eligibility checks fail, so no correctness gate can tell a fallback from a
success and only ``_assert_kernel_marker`` can. This roster moves both arms
from ``arange`` positions onto packed-document positions, which is exactly the
kind of change that trips an eligibility check, so ``_titan_arm`` refuses to
build either arm without its marker. ``mcore/base`` needs no such guard: with
``cu_seqlens`` set, ``apply_rotary_pos_emb`` takes the fused branch
unconditionally -- every ``use_unfused`` escape lives inside the
``cu_seqlens is None`` arm (``rope_utils.py:293-317``) -- and *asserts* when
the TE symbol is missing (``:319``) rather than degrading. Note that
``TransformerConfig.__post_init__`` is **not** the proof here: it raises only
when *both* fused symbols are ``None`` (``transformer_config.py:2210-2213``),
so it cannot show that the THD one resolved. ``_mcore_arm`` checks that one by
name instead.

**The provenance this scenario owes a reader**, recorded here because plan
section C.1 assigns it nowhere else: megatron builds ``rotary_pos_emb`` inside
``GPTModel.forward`` on every step (``gpt_model.py:403-410``), where titan's
``CosSinRoPE`` holds a precomputed buffer. This scenario hoists that build to
arm-construction time and charges it to neither arm. The cost it hides is
smaller than it looks: ``RotaryEmbedding.forward`` is ``@lru_cache(maxsize=32)``
(``rotary_pos_embedding.py:177``) and the key is constant across steps, so
after the first step megatron pays a cache lookup and not a table build.

**No layout conversion is charged to either engine, and none is hidden.**
Megatron's THD form of a titan ``[B, L, N, H]`` tensor is
``[B*L, N, H]`` -- a reshape of the two leading dimensions of an already
contiguous tensor, so ``rope_inputs`` builds it as a **view** of the same
storage. Both engines therefore read the same bytes in the same order, and
this scenario has no counterpart to the transpose ``qk_norm`` has to
materialize.

Every megatron, TransformerEngine and torchtitan import is deferred into the
builder that needs it. ``te_rope_override`` is the sharpest case -- importing
it JIT-builds a CUDA extension that needs the gcc-13 environment -- but the
rule is the package's, not this module's: one arm per process only pays off
when a process imports the one arm it builds.
``tests/test_import_boundaries.py`` enforces it.
"""

from __future__ import annotations

import gc
from dataclasses import dataclass

import torch
import torch.nn as nn

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    MCORE_BLANK_MLP,
    _assert_kernel_marker,
    _compile_module,
    _randn,
    _randn_like,
    initialize_megatron_single_rank,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE, McoreProfile, derive
from benchmarks.models.piper_qwen3.shape import PiperShape


# The kernel names the silent-fallback guard greps for. ``HELION_MARKER`` is
# emitted by ``torchtitan.overrides.helion_rope``; ``TE_MARKER`` is the one
# ``__global__`` our local port adds on top of TE's verbatim device code, so
# its presence proves the positions path ran and not the stock module.
HELION_MARKER = "_helion__rope_cos_sin_fwd"
TE_MARKER = "fused_rope_forward_positions_kernel"

# The five arm names, spelled as plan section C.2 spells them. The slash says
# which engine an arm is and which profile of that engine; it reaches a
# filename through ``benchmarks.kernel.schema.fragment_stem``.
MCORE_BASE_ARM = "mcore/base"
MCORE_NO_ROPE_FUSION_ARM = "mcore/no_rope_fusion"
TITAN_ARM = "titan"
TITAN_HELION_ARM = "titan/helion"
TITAN_TE_ARM = "titan/te"

# The arms whose module degrades to the correct stock path instead of
# failing, and the kernel each must show. ``_titan_arm`` refuses to build one
# of these without its marker, so a future arm cannot inherit the timed
# closures and lose the only guard that can catch a fallback.
FALLBACK_MARKERS = {
    TITAN_HELION_ARM: HELION_MARKER,
    TITAN_TE_ARM: TE_MARKER,
}

# The names every arm and the fp64 reference return, in titan's BLNH layout.
# The scenario declaration repeats them as literals, because a declaration may
# not import a builder module; the test pins the two copies against each other.
OUTPUTS = ("q_out", "k_out", "dq", "dk")

# The one megatron variant this scenario adds, as a delta rather than a second
# copy of 28 flags. Declared here rather than added to ``MCORE_PROFILES``
# because it is this scenario's roster: the e2e megatron arm and every other
# cross-engine scenario run ``base``, and a registry entry would suggest
# otherwise. ``apply_rope_fusion`` is already in ``FUSION_FIELDS``, so the
# driver's declared-state check polices the flip in both directions.
NO_ROPE_FUSION_PROFILE: McoreProfile = derive(
    BASE,
    name="no_rope_fusion",
    description=(
        "megatron with RoPE fusion declined: apply_rope_fusion=False, which "
        "at THD selects _apply_rotary_pos_emb_thd -- two device-to-host "
        "syncs and a Python loop over the packed documents, not an unfused "
        "TE kernel"
    ),
    config_overrides={"apply_rope_fusion": False},
)


@dataclass
class RopeInputs:
    """One q/k pair in both engine-native forms, plus the packed metadata.

    ``*_BLNH`` is TorchTitan's layout. ``*_THD`` is megatron's, and each is a
    **view** of the matching BLNH tensor rather than a copy: ``[B, L, N, H]``
    is contiguous, so ``[B*L, N, H]`` is the same storage read in the same
    order. Nothing here is built inside a timed closure.

    ``positions`` and ``cu_seqlens`` are two spellings of one packing.
    ``positions`` restarts at 0 at every document start, which is the form
    titan's RoPE modules index with; ``cu_seqlens`` holds the offsets of those
    same starts over the flattened batch, which is the form megatron's THD
    path derives its positions from.
    """

    q_BLNH: torch.Tensor
    k_BLNH: torch.Tensor
    gq_BLNH: torch.Tensor
    gk_BLNH: torch.Tensor
    q_THD: torch.Tensor
    k_THD: torch.Tensor
    gq_THD: torch.Tensor
    gk_THD: torch.Tensor
    positions: torch.Tensor
    cu_seqlens: torch.Tensor
    num_documents: int
    rotary_seq_len: int
    qk_bytes: int


def _packed_positions(
    workload: KernelWorkload, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    """Synthetic packed-document positions, int64, resetting at each start.

    A seeded mix of document lengths rather than one document per row, for the
    reason ``attention.py``'s twin gives: a single full-length document per row
    makes the packing trivial and hides the addressing both THD paths exist to
    do. The dtype differs from that twin's int32 and has to: the TE positions
    kernel declares ``const int64_t *positions``
    (``components/rope/te_rope_standalone.cu:236``), and ``TECosSinRoPE``
    falls back to the stock path on any other integer width.
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
    return torch.tensor(rows, device=device, dtype=torch.int64)


def _cu_seqlens(positions: torch.Tensor) -> torch.Tensor:
    """The document offsets of ``positions``, flattened over the batch.

    The same construction ``benchmarks/e2e/megatron/data.py:82-88`` performs
    for the e2e megatron arm, reimplemented rather than imported: an
    ``operations`` module must not pull the e2e driver into a kernel worker.
    Every row starts a document, so the row boundaries are among the offsets
    and the two engines cut the batch identically.

    **Unlike the driver, this pads nothing.** The driver pads ``cu_seqlens`` to
    one length across steps so graph capture sees static shapes
    (``benchmarks/e2e/megatron/train.py:170-186``). A kernel run times one
    fixed batch, so there is no second shape to agree with, and every padded
    entry would add a zero-length iteration that only ``mcore/no_rope_fusion``
    pays for.
    """
    flat = positions.reshape(-1)
    starts = (flat == 0).nonzero(as_tuple=True)[0].to(torch.int32)
    if int(starts[0]) != 0:
        raise ValueError("the first packed position is not a document start")
    total = torch.tensor(
        [flat.numel()], dtype=torch.int32, device=positions.device
    )
    return torch.cat([starts, total])


def _thd(tensor: torch.Tensor) -> torch.Tensor:
    """Megatron's [t, n, h] view of a titan [b, l, n, h] tensor."""
    batch, seq = tensor.shape[0], tensor.shape[1]
    return tensor.view(batch * seq, *tensor.shape[2:])


def _blnh(tensor: torch.Tensor, batch: int, seq: int) -> torch.Tensor:
    """The inverse of ``_thd``, for comparison outside the timed region."""
    return tensor.view(batch, seq, *tensor.shape[1:])


def rope_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> RopeInputs:
    batch, seq = workload.batch, workload.seq_len
    q = _randn((batch, seq, shape.n_heads, shape.head_dim), device, generator)
    k = _randn(
        (batch, seq, shape.n_kv_heads, shape.head_dim), device, generator
    )
    gq = _randn_like(q, generator)
    gk = _randn_like(k, generator)
    # torch.randint needs a CPU generator, and the document lengths are drawn
    # on the host in any case. Seeded from the run's generator so the packing
    # rebuilds bit-identically in every worker process.
    cpu_generator = torch.Generator(device="cpu").manual_seed(
        int(generator.initial_seed()) & 0x7FFFFFFF
    )
    positions = _packed_positions(workload, device, cpu_generator)
    cu_seqlens = _cu_seqlens(positions)
    return RopeInputs(
        q_BLNH=q,
        k_BLNH=k,
        gq_BLNH=gq,
        gk_BLNH=gk,
        q_THD=_thd(q),
        k_THD=_thd(k),
        gq_THD=_thd(gq),
        gk_THD=_thd(gk),
        positions=positions,
        cu_seqlens=cu_seqlens,
        num_documents=int(cu_seqlens.numel()) - 1,
        # What megatron's own driver pins max_seqlen to, and therefore what
        # get_rotary_seq_len returns for it
        # (rotary_pos_embedding.py:238-241, e2e/megatron/train.py:276-277).
        # No document is longer than a row, so the table covers every
        # position either engine looks up.
        rotary_seq_len=seq,
        # Read q and k, write q_out and k_out: the forward traffic. The
        # backward moves more, so read the GB/s column in forward mode only.
        qk_bytes=2 * (q.numel() + k.numel()) * q.element_size(),
    )


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


def _rope_angles_fp64(
    shape: PiperShape, workload: KernelWorkload, device: torch.device
) -> torch.Tensor:
    """The [seq_len, head_dim] angle table both engines build in fp32.

    One expression, two readers: torchtitan's ``CosSinRoPE._precompute_cache``
    and megatron's ``RotaryEmbedding.__init__`` write the same ``inv_freq``
    and the same doubled concatenation
    (``rotary_pos_embedding.py:79-81`` and ``:165-170``). The reference
    computes it in fp64 so a gate measures the arms and not the table.
    """
    half = shape.head_dim // 2
    exponents = torch.arange(
        0, shape.head_dim, 2, dtype=torch.float64, device=device
    )[:half]
    inv_freq = 1.0 / (shape.rope_theta ** (exponents / shape.head_dim))
    steps = torch.arange(workload.seq_len, dtype=torch.float64, device=device)
    return torch.cat([torch.outer(steps, inv_freq)] * 2, dim=-1)


def rope_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: RopeInputs
) -> dict[str, torch.Tensor]:
    """The fp64 truth, indexed by the packed positions.

    Backward is the analytic adjoint rather than an autograd replay. The
    rotation is ``y = x*cos + R(x)*sin`` with ``R`` the rotate-half matrix, so
    ``dx = g*cos + R^T(g*sin)`` and ``R^T = -R``; the two halves of ``sin``
    are equal, which is what lets ``R(g*sin)`` be written ``R(g)*sin``.
    ``tests/test_kernel_rope.py`` checks the closed form against torch
    autograd rather than taking this paragraph on trust.
    """
    table = _rope_angles_fp64(shape, workload, inputs.q_BLNH.device)
    angles = table[inputs.positions].unsqueeze(2)
    cos, sin = angles.cos(), angles.sin()
    outputs: dict[str, torch.Tensor] = {}
    for tag, x, grad in (
        ("q", inputs.q_BLNH, inputs.gq_BLNH),
        ("k", inputs.k_BLNH, inputs.gk_BLNH),
    ):
        xf, gf = x.double(), grad.double()
        outputs[f"{tag}_out"] = xf * cos + _rotate_half(xf) * sin
        outputs[f"d{tag}"] = gf * cos - _rotate_half(gf) * sin
    return outputs


def _rope_arm(
    name: str,
    inputs: RopeInputs,
    *,
    q: torch.Tensor,
    k: torch.Tensor,
    gq: torch.Tensor,
    gk: torch.Tensor,
    call,
    canonical,
) -> BuiltArm:
    """Forward and an isolated backward, over one retained graph.

    The shape both engines get, so the two are comparable mode by mode.

    **Every mode runs over ``requires_grad`` leaves, including forward.**
    ``requires_grad`` is a Dynamo guard, so a compiled module called on
    non-grad tensors runs an *inference* graph that saves nothing for
    backward, and that is not the graph a transformer block runs. The
    asymmetry would not cancel across engines either -- megatron's side is
    eager and would be unaffected -- so it would bias the cross-engine row
    toward the three titan arms. ``final_norm`` and ``qk_norm`` time their
    forwards on leaves for the same reason.

    ``backward`` re-runs ``torch.autograd.backward`` over a graph built once
    at construction with ``retain_graph=True``, so only backward kernels are
    timed. That trick is available here and not in ``qk_norm`` because
    neither side clears its saved tensors: TE's fused RoPE is a plain
    ``torch.autograd.Function`` holding ``freqs`` and ``cu_seqlens``
    (``attention/rope.py:152``), and the unfused path is ordinary torch ops.

    ``correctness_outputs`` reads the retained graph for **both** the outputs
    and the gradients. Taking the outputs from a second, inference-mode call
    would leave a hole no gate could see: RoPE's adjoint depends on the angles
    alone and not on the forward outputs, so a wrong forward inside the timed
    graph would still produce a correct ``dq``.
    """

    def leaves() -> tuple[torch.Tensor, torch.Tensor]:
        return q.clone().requires_grad_(), k.clone().requires_grad_()

    forward_leaves = leaves()
    graph_leaves = leaves()
    retained = call(*graph_leaves)

    def forward():
        return call(*forward_leaves)

    def backward() -> None:
        graph_leaves[0].grad = None
        graph_leaves[1].grad = None
        torch.autograd.backward(retained, (gq, gk), retain_graph=True)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        backward()
        q_out, k_out = retained
        return {
            "q_out": canonical(q_out.detach()),
            "k_out": canonical(k_out.detach()),
            "dq": canonical(graph_leaves[0].grad),
            "dk": canonical(graph_leaves[1].grad),
        }

    return BuiltArm(
        name=name,
        calls={"forward": forward, "backward": backward},
        correctness_outputs=correctness_outputs,
        bytes_moved=inputs.qk_bytes,
    )


def _identity(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def _titan_module(shape: PiperShape, inputs: RopeInputs, cls) -> nn.Module:
    """One torchtitan RoPE module, built from the shared geometry.

    ``cls`` is a ``CosSinRoPE`` subclass, and each one carries the identical
    ``Config``, so the three titan arms differ in the class alone. That is
    deliberate: the timed closures, the retained graph and the correctness
    hook must be the same code for the comparison to mean anything.
    """
    module = cls.Config(
        dim=shape.head_dim,
        max_seq_len=shape.max_seq_len,
        theta=shape.rope_theta,
    ).build()
    module.init_states(buffer_device=inputs.q_BLNH.device)
    return module


def _titan_arm(
    name: str, module: nn.Module, inputs: RopeInputs, marker: str | None = None
) -> BuiltArm:
    """A titan arm over ``module``, with the fallback guard where one is owed.

    ``module`` arrives compiled from the builder. The tests pass an eager one,
    which exercises the same closures without an Inductor compile on a CPU.
    """
    expected = FALLBACK_MARKERS.get(name)
    if expected is not None and marker != expected:
        raise ValueError(
            f"{name} falls back to the numerically correct stock path when "
            f"its eligibility checks fail, so it must be built with marker "
            f"{expected!r}, not {marker!r}; no correctness gate can catch a "
            "fallback"
        )

    def call(q: torch.Tensor, k: torch.Tensor):
        return module(q, k, inputs.positions)

    arm = _rope_arm(
        name,
        inputs,
        q=inputs.q_BLNH,
        k=inputs.k_BLNH,
        gq=inputs.gq_BLNH,
        gk=inputs.gk_BLNH,
        call=call,
        canonical=_identity,
    )
    if marker is not None:
        # Profile the closure the timing pass will time, not a lookalike: a
        # module called on non-grad tensors compiles a separate inference
        # graph, and the guard has to watch the graph that gets published.
        # The untimed call ahead of it is what makes the profile capture
        # kernels rather than compilation.
        forward = arm.calls["forward"]
        forward()
        _assert_kernel_marker(forward, marker, name)
    return arm


def build_rope_titan(
    shape: PiperShape, workload: KernelWorkload, inputs: RopeInputs
) -> BuiltArm:
    """TorchTitan's stock ``CosSinRoPE``, compiled as a block gives it."""
    from torchtitan.models.common.rope import CosSinRoPE

    module = _compile_module(_titan_module(shape, inputs, CosSinRoPE))
    return _titan_arm(TITAN_ARM, module, inputs)


def build_rope_titan_helion(
    shape: PiperShape, workload: KernelWorkload, inputs: RopeInputs
) -> BuiltArm:
    """TorchTitan's Helion RoPE kernel, with its mandatory marker guard."""
    from torchtitan.overrides.helion_rope import HelionCosSinRoPE

    module = _compile_module(_titan_module(shape, inputs, HelionCosSinRoPE))
    return _titan_arm(
        TITAN_HELION_ARM, module, inputs, marker=HELION_MARKER
    )


def build_rope_titan_te(
    shape: PiperShape, workload: KernelWorkload, inputs: RopeInputs
) -> BuiltArm:
    """Our local port of TE's fused RoPE, against titan's positions interface.

    Rule 4 in one line: this is **not** the installed TransformerEngine. It is
    a standalone CUDA extension that copies TE's inner block functions and
    adds its own ``__global__`` entry point and its own BSHD launch
    configuration. See this module's docstring for what the row against
    ``mcore/base`` therefore does and does not measure.
    """
    from benchmarks.models.piper_qwen3.components.rope.te_rope_override import (
        TECosSinRoPE,
    )

    module = _compile_module(_titan_module(shape, inputs, TECosSinRoPE))
    return _titan_arm(TITAN_TE_ARM, module, inputs, marker=TE_MARKER)


def _assert_rope_config(config, profile: McoreProfile, shape: PiperShape):
    """Refuse to time a megatron config that computes a different rotation.

    Four of the five checks are about the *mathematics*: the fp64 reference is
    one rotate-half NeoX rotation at ``mscale`` 1.0, and each of these fields
    would silently make megatron compute something else while every gate
    still ran against the old truth. The fifth is the arm's own identity --
    ``apply_rope_fusion`` selects the branch the arm is named for, and a
    profile delta that did not take would publish ``base`` under the other
    label.
    """
    declared = bool(profile.config_overrides["apply_rope_fusion"])
    if bool(config.apply_rope_fusion) is not declared:
        raise RuntimeError(
            f"profile {profile.name!r} declares apply_rope_fusion="
            f"{declared} but the built config reports "
            f"{bool(config.apply_rope_fusion)}; the arm would measure the "
            "other branch under this label"
        )
    if config.rotary_interleaved:
        raise RuntimeError(
            "rotary_interleaved rotates adjacent pairs instead of halves; "
            "titan has no such mode and the fp64 reference is rotate-half"
        )
    if config.multi_latent_attention:
        raise RuntimeError(
            "multi_latent_attention selects MLA-style interleaving inside "
            "apply_rotary_pos_emb; this scenario compares plain RoPE"
        )
    if config.mrope_section is not None:
        raise RuntimeError(
            "mrope_section selects multimodal RoPE, which titan's "
            "CosSinRoPE does not implement"
        )
    if config.kv_channels != shape.head_dim:
        raise RuntimeError(
            f"megatron rotates {config.kv_channels} channels against the "
            f"shape's head_dim {shape.head_dim}; the two engines would "
            "rotate different widths"
        )


# The freqs table and the two process-group handles are all this arm keeps.
# Anything above this is the model, and the budget is wide enough that
# allocator rounding cannot reach it.
_RESIDUAL_BUDGET_BYTES = 64 * 2**20


def _report_build_residual(name: str, before: int) -> None:
    """Say so if the dropped GPTModel did not free.

    ``memory_pass`` reports ``max_memory_allocated``, which counts every live
    allocation. A surviving reference to the model would add its whole build
    to this arm's peak memory and nothing to a titan arm's. That build is
    0.67 GiB at ``1b`` and 7.80 GiB at ``huge``, because ``MCORE_BLANK_MLP``
    leaves the mlp part out.
    This reports and does not raise: the timing columns are unaffected, peak
    memory is a secondary metric here, and a hard failure would cost the
    anchor and with it the whole scenario. The worker's stdout lands in
    ``kernel_bench.log``, which the run keeps.
    """
    residual = torch.cuda.memory_allocated() - before
    if residual > _RESIDUAL_BUDGET_BYTES:
        print(
            f"WARNING rope/{name}: the megatron model did not free "
            f"({residual / 2**20:.0f} MiB still allocated). Read this arm's "
            "peak_memory_gib as the model plus the rotation, not as the "
            "rotation."
        )


def _check_thd_reference_path(inputs: RopeInputs, freqs: torch.Tensor) -> None:
    """Refuse the CASE 1 branch, and caption the CASE 2 one.

    ``_apply_rotary_pos_emb_thd`` has two branches and picks between them on
    the data, not on the configuration. ``has_packed_freqs = freqs.size(0) ==
    total_seqlen`` (``rope_utils.py:225``) compares the table's ``seq_len``
    rows against the ``batch * seq_len`` packed tokens, so it is True exactly
    at ``batch == 1``.

    **CASE 1 is a different rotation, not a different implementation of one.**
    It indexes ``freqs`` by each document's *global* offset --
    ``seq_start_offset = cu_seqlens[i].item()`` (``:239``) into
    ``freqs[offset : offset + x.size(0)]`` (``:186``) -- where this scenario's
    fp64 reference, ``mcore/base`` and all three titan arms restart the angle
    at every document start. So the arm would compute something its own anchor
    does not, and the fp64 gate would fire for a reason no caption names. This
    raises instead: ``--batch 1`` is a documented flag, and an arm that cannot
    answer the scenario's question at a supported setting must say so at build
    time rather than at gate time.

    CASE 2 is the branch the scenario is declared against. Rule 5 asks every
    scenario to say whether its number holds host serialization; this one
    does, and by how much depends on the packing, so the caption carries the
    count the run actually measured rather than a range.
    """
    total_tokens = int(inputs.cu_seqlens[-1])
    if freqs.size(0) == total_tokens:
        raise RuntimeError(
            f"rope/{MCORE_NO_ROPE_FUSION_ARM}: the rotary table has "
            f"{freqs.size(0)} rows and the packed batch has {total_tokens} "
            "tokens, so _apply_rotary_pos_emb_thd takes CASE 1 "
            "(rope_utils.py:234-250), which rotates by each document's global "
            "offset instead of restarting at every document start. That is a "
            "different rotation from this scenario's anchor and from its fp64 "
            "reference. It happens exactly at batch == 1; run this scenario "
            "at batch > 1."
        )
    print(
        f"rope/{MCORE_NO_ROPE_FUSION_ARM}: _apply_rotary_pos_emb_thd takes "
        f"CASE 2 (the per-document loop) over {inputs.num_documents} "
        "documents, per q and per k, per call. This arm's number is device "
        "time PLUS two device-to-host syncs and a Python loop, and it scales "
        "with the document count."
    )


def _mcore_arm(
    name: str,
    profile: McoreProfile,
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: RopeInputs,
) -> BuiltArm:
    """Megatron's ``apply_rotary_pos_emb``, over the model megatron builds.

    The config and the freqs table both come off a real ``GPTModel``: the
    dispatch inside ``apply_rotary_pos_emb`` reads fields this module never
    writes, and the table comes from megatron's own ``RotaryEmbedding``, so
    nothing here can build a lookalike by mistake. The closure then repeats
    ``attention.py:1495-1524`` -- query first, key second, one ``cu_seqlens``
    and one ``mscale`` for both -- and the model is dropped, because peak
    memory is a published column and a resident model would make every mcore
    arm look expensive for a reason unrelated to the rotation.
    """
    initialize_megatron_single_rank()
    from megatron.core.models.common.embeddings.rope_utils import (
        apply_rotary_pos_emb,
        fused_apply_rotary_pos_emb_thd,
    )

    from benchmarks.models.piper_qwen3.megatron_model import build_model

    before = torch.cuda.memory_allocated()
    model = build_model(
        seq_len=workload.seq_len,
        shape=shape,
        profile=profile,
        blank_parts=MCORE_BLANK_MLP,
    )
    config = model.config
    _assert_rope_config(config, profile, shape)
    if config.apply_rope_fusion and fused_apply_rotary_pos_emb_thd is None:
        raise RuntimeError(
            "apply_rope_fusion is on but megatron resolved no TE fused THD "
            "symbol; the arm would assert inside the first timed call"
        )

    attention = model.decoder.layers[0].self_attention
    cp_group = attention.pg_collection.cp
    mscale = float(attention._yarn_concentration_factor)
    if mscale != 1.0:
        raise RuntimeError(
            f"megatron scales the rotation by mscale={mscale}; the fp64 "
            "reference and titan's CosSinRoPE apply no such factor"
        )
    # What GPTModel.forward computes every step, hoisted here. See this
    # module's docstring for why that is recorded rather than charged.
    freqs = model.rotary_pos_emb(
        inputs.rotary_seq_len, packed_seq=True, cp_group=None
    )
    expected_freqs = (inputs.rotary_seq_len, 1, 1, shape.head_dim)
    if tuple(freqs.shape) != expected_freqs:
        raise RuntimeError(
            f"megatron's rotary table is {tuple(freqs.shape)}, not "
            f"{expected_freqs}; the two engines would index different rows"
        )
    del attention, model
    gc.collect()
    torch.cuda.empty_cache()
    _report_build_residual(name, before)
    if not config.apply_rope_fusion:
        _check_thd_reference_path(inputs, freqs)

    def call(q: torch.Tensor, k: torch.Tensor):
        # Eager, as megatron runs it: it compiles no whole layer, and
        # rope_utils.py carries no jit_fuser decoration.
        return (
            apply_rotary_pos_emb(
                q,
                freqs,
                config=config,
                cu_seqlens=inputs.cu_seqlens,
                mscale=mscale,
                cp_group=cp_group,
            ),
            apply_rotary_pos_emb(
                k,
                freqs,
                config=config,
                cu_seqlens=inputs.cu_seqlens,
                mscale=mscale,
                cp_group=cp_group,
            ),
        )

    batch, seq = workload.batch, workload.seq_len
    return _rope_arm(
        name,
        inputs,
        q=inputs.q_THD,
        k=inputs.k_THD,
        gq=inputs.gq_THD,
        gk=inputs.gk_THD,
        call=call,
        canonical=lambda tensor: _blnh(tensor, batch, seq),
    )


def build_rope_mcore_base(
    shape: PiperShape, workload: KernelWorkload, inputs: RopeInputs
) -> BuiltArm:
    """TE's fused THD RoPE, which is the anchor of this scenario."""
    return _mcore_arm(MCORE_BASE_ARM, BASE, shape, workload, inputs)


def build_rope_mcore_no_rope_fusion(
    shape: PiperShape, workload: KernelWorkload, inputs: RopeInputs
) -> BuiltArm:
    """Megatron's THD Python reference. See Rule 5 in the module docstring."""
    return _mcore_arm(
        MCORE_NO_ROPE_FUSION_ARM,
        NO_ROPE_FUSION_PROFILE,
        shape,
        workload,
        inputs,
    )
