"""Arm builders for the ``attention_core`` cross-engine kernel scenario.

Inner attention, head to head across the two engines. The cut starts at
q/k/v after RoPE and ends at the attention output, before the output
projection. It is the level at which the implementations are substitutable,
and the level that keeps this from re-measuring the projection work
``qkv_prep`` and ``attn_out_proj`` already cover.

* TorchTitan: ``inner_attention(...)`` inside ``GQAttention.forward``
  (``third_party/torchtitan/torchtitan/models/common/attention.py``), which
  is ``FlexAttention`` or ``VarlenAttention`` depending on the arm.
* Megatron-core: ``self_attention.core_attention``, reached at
  ``third_party/Megatron-LM/megatron/core/transformer/attention.py:1559-1566``
  -- the branch this build takes, because ``checkpoint_core_attention`` is
  False (no recompute) and ``inference_context`` is None. The module is
  ``TEDotProductAttention``
  (``megatron/core/extensions/transformer_engine.py:1995``), constructed at
  ``attention.py:382-390``.

**No timing number exists.** No arm has been timed, and no ratio has been
computed. Two things HAVE been measured on an H200, on 2026-08-20, and both
are correctness rather than speed:

* which TransformerEngine backend each megatron arm resolves to -- see
  "Which backend each mcore arm runs" below;
* that the two engines' outputs agree with the fp64 reference and with each
  other. At ``dim`` 256, 4 q heads over 2 kv groups, head_dim 64, batch 2,
  seq 128, over 8 seeded packed documents, ``rel_l2`` against the fp64
  reference is 1.76e-3 (``out``), 2.75e-3 (``dq``), 3.31e-3 (``dk``) and
  2.81e-3 (``dv``) for megatron, and 1.76e-3 / 2.75e-3 / 2.72e-3 / 2.22e-3
  for TorchTitan's FlexAttention. The two engines agree with each other to
  4.3e-4 on ``out`` and to 2.9e-3 on the widest gradient. That is what
  validates the head ordering behind the canonicalization below, the claim
  that the THD ``cu_seqlens`` and the flex ``BlockMask`` are one predicate,
  and the shared scale -- none of which a CPU test can reach.

Both of those were probes beside the scenario. **The scenario's own
correctness pass has now run**, on an H200 on 2026-08-20, at the default
workload (``normal``, batch 4, seq 1024) and through
``benchmarks.kernel.worker``. All six arms built, all 24 fp64 gates passed,
and the widest of them was ``mcore/attn_unfused``'s ``dk`` at 4.43e-3
against the 2e-2 threshold. The 20 informational cross-arm rows passed too,
the widest being 5.26e-3.

Two results of that pass are worth keeping. TE resolved the three megatron
arms to three different kernels, which is the whole premise of the roster:
``FusedAttention NVTE_F16_arbitrary_seqlen``, ``FlashAttention 3.0.0`` and
``UnfusedDotProductAttention``. And ``mcore/attn_flash3`` and
``titan/flash_attention_3`` land on **identical** distances from
``mcore/base`` -- 7.493e-4 / 6.980e-4 / 2.895e-3 / 2.867e-3 on out/dq/dk/dv
-- which is the evidence that the pair really is one kernel family and one
masking through two host stacks. **That pass needed the cuDNN workaround
below.** It still produced no timing number.

Which layout each engine reads
------------------------------

TorchTitan's inner attention takes and returns ``[B, L, N, H]``.
``FlexAttention.forward`` transposes to ``[B, N, L, H]`` and back
(``attention.py:327-353``) and ``VarlenAttention.forward`` reshapes to
``[T, N, H]`` (``:139-141``); both are views of contiguous storage and
neither materializes a copy.

Megatron's THD path takes ``[T, N, H]`` and returns ``[T, N*H]``. The
reshape megatron then applies (``attention.py:1598-1603``) turns that into
``[T, 1, N*H]`` for the output projection; it is a view, it belongs to
``attn_out_proj``, and this scenario does not run it.

``attention_core_inputs`` materializes q, k and v once, in ``[B, L, N, H]``,
and the titan arms read exactly that. **The megatron arms do not read a
reshape of it**, and that difference is deliberate.

Megatron's QKV GEMM writes one fused ``[T, G, (Q + 2) * H]`` tensor, and
``get_query_key_value_tensors`` splits it into three strided views. Two of
them stop being views before this cut and one does not:

* the query is reshaped, normed and rotated;
* the key is normed and rotated (``BASE`` sets ``qk_layernorm: True``, so
  ``k_layernorm`` is a real norm);
* **the value gets neither**, and megatron's own comment records that
  choice (``attention.py:1534``).

So ``core_attention`` receives a contiguous query, a contiguous key, and
**one non-contiguous strided view** -- the value. TE then classifies the
three tensors in ``get_qkv_layout``, does not recognize the layout, and
forces ``.contiguous()`` (``dot_product_attention/utils.py:2428-2431``).
Only the value moves: **4 MiB per forward** at batch 4 / seq 1024 /
``normal``. It runs inside every timed megatron call here, because
``get_qkv_layout`` is called unconditionally at
``dot_product_attention.py:1453``, before the backend is chosen, so cuDNN,
FA3 and the unfused path all pay it.

This scenario therefore builds the fused buffer and hands each megatron arm
those three tensors. The sibling scenario ``qkv_prep`` states in its own
description that megatron **defers** this copy to the arm that consumes the
views, and names this scenario as that arm. Handing megatron three
contiguous tensors instead would measure a layout megatron never produces.

**The other half of the deferred copy belongs to another scenario, and that
scenario now collects it.** In the engine the key's 4 MiB is absorbed by
``k_layernorm``, which belongs to ``qk_norm``. That scenario builds the same
fused buffer and hands its megatron arm the same kind of strided view
(``operations/qk_norm.py``), so the key's half is timed there and this one
declines to double-book it. **That ledger is prose only**: this scenario
declares no ``bytes_moved`` on any arm and holds no ``copy_floor``, so only
``qk_norm`` publishes a number and nothing asserts that the halves sum.

Verified on CPU, because strides and TE's classifier need no device:
``torch.split`` of the fused buffer gives the value the strides
``(G*R, R, 1)`` at offset ``(Q + 1) * H``; ``get_qkv_layout`` returns
``thd_thd_thd`` only **after** copying it; and ``Tensor.clone()`` of such a
view returns a contiguous tensor. The last fact is why
``_Layout.make_leaves`` exists and why no leaf set here is built with
``clone``.

The output canonicalization runs OUTSIDE the timed closures, in
``correctness_outputs``: megatron's ``[T, N*H]`` becomes ``[B, L, N, H]``
by a view, which is what the fp64 reference and every cross-arm gate read.

Every mask form is built once, never inside a timed closure
-----------------------------------------------------------

``create_varlen_metadata_for_document`` contains a device-to-host sync and
``create_block_mask`` is itself a compiled call, so either one inside a
closure would be timed as attention. ``attention_core_inputs`` builds all
three forms up front: a flex ``BlockMask`` at the default 128 block size,
the same mask at ``FLEX_FLASH_BLOCK_SIZE``, and THD ``cu_seqlens``.

The three forms describe **one** masking function. Every row of
``positions`` starts at 0, so the row boundaries of the ``[B, L]`` tensor
are also document boundaries, and the block-diagonal causal mask flex builds
per row is the same predicate the flat ``cu_seqlens`` describes over
``B * L``. ``cu_seqlens`` is padded to a fixed multiple with trailing
full-offset entries, which is a packed form TransformerEngine already
consumes in this repo. It is **not** the e2e megatron arm's exact tensor;
see the padding note below.

Nothing on the torch side validates the FLASH block size: it is forwarded
verbatim into FA4's block-sparse tensors, so a mismatch surfaces inside FA4
rather than as a torch-level error.

``cu_seqlens`` arrives padded, and **both engines receive the identical
tensor**. ``create_varlen_metadata_for_document`` pads to a fixed multiple
with trailing full-offset entries, so at the default workload 18 real
documents become 128 entries, 109 of them zero-length. TransformerEngine
accepts that -- measured, on all three backends, at the default workload --
and the titan FA3 arm already consumes the same tensor in the existing
``attention`` scenario.

**The two padding rules differ, and this scenario uses torchtitan's.**
``create_varlen_metadata_for_document`` pads to a multiple of
``_CU_SEQLENS_MULTIPLE`` = 128 (``models/common/attention.py:600``,
``:654-661``), so the segment count here is a fixed 128 whatever the draw
contains. The e2e megatron driver pads to the run's own maximum document
count instead (``benchmarks/e2e/megatron/train.py:177-186``). So this
scenario's segment count is **not** the e2e arm's, and it is not "exactly
what the driver feeds it". Both engines here receive the identical tensor,
which is what the cross-engine comparison needs; the e2e difference is a
property of the isolated workload, not an asymmetry between these arms.

**The segment count, not the document count, is what TE allocates on, and
``mcore/attn_unfused`` can therefore exhaust the device.**
``ConvertTHDtoBSHD.forward`` reads ``batch_size = cu_seqlens.shape[0] - 1``
(``dot_product_attention/utils.py:2153``), so the unfused arm builds its
scores over **127** segments and not over the 18 real documents. One bf16
score tensor is then 4.0 GiB at seq 1024, 15.9 GiB at 2048 and 63.5 GiB at
4096 on the normal shape, and 47.6 GiB at seq 1024 on huge; the arm holds
the scores, the saved probabilities and the backward gradient. A sweep past
seq 2048 will OOM. Because the correctness pass has no per-arm exception
handling (``benchmarks/kernel/engine/run.py:289-298``), that OOM takes the
whole scenario with it. An adversarial reviewer measured the segment counts
on the real inputs builder; the arm description carries the table.

Three ways to break this scenario that nothing guards
-----------------------------------------------------

**This scenario does not run as shipped on a host that has cuDNN in
``/usr/lib64``, and the failure is an environment split, not a bug here.**
Measured on this H200 on 2026-08-20, on the first run this scenario ever
had. The five other arms build, and ``titan/flash_attention_3`` then dies
inside ``torch.nn.attention.varlen._varlen_attn`` with "cuDNN version
incompatibility: PyTorch was compiled against (9, 24, 0) but found runtime
version (9, 23, 2)".

The mechanism, and every step of it was observed:

* ``libtransformer_engine.so`` carries ``DT_NEEDED`` on ``libcudnn.so.9``
  and its seven engine libraries, and it carries no ``RUNPATH``.
* torch loads its own cuDNN **lazily**. After ``import torch`` only the
  dispatcher ``nvidia/cudnn/lib/libcudnn.so.9`` is mapped; the engine
  libraries are not.
* So at TE import the loader resolves TE's seven entries down the default
  search path and binds ``/usr/lib64`` cuDNN 9.23.2, which no part of this
  project pins.
* ``torch.backends.cudnn.version()`` then reports 9.23.2 and
  ``backends/cudnn/__init__.py`` raises, because torch wants
  ``runtime_minor >= compile_minor``.
* ``torch.nn.attention.varlen`` asks for that version in
  ``_should_use_cudnn``, which is ``@lru_cache``d (``varlen.py:29``) and so
  asks once per device index. One raise is enough: torchtitan's
  ``VarlenAttention`` cannot run in a process that has imported TE.

The system library is ``libcudnn9-cuda-12`` (``rpm -qf``), so this is a
**CUDA 12 build inside a cu13 process**. ``megatron_bootstrap.py:57``
already sets ``CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13`` for that
same mismatch, arrived at independently and never connected to this.

**cuDNN is the only library that splits.** ``libcublas``, ``libcublasLt``,
``libcudart`` and ``libnccl`` all resolve to the venv wheels in the same
process, because torch loads those eagerly and TE's ``DT_NEEDED`` entries
find them already mapped. The cuDNN split is an accident of lazy loading,
and its consequence is that **which cuDNN every megatron arm in this repo
runs is decided by the host, not by the pin**.

**Measured on 2026-08-21, and it settles two questions this file used to
leave open** (``reports/20260821-cudnn-version-comparison.md``).

*The version changes no value.* Every correctness gate row matches to the
float64 bit pattern under 9.23.2 and 9.24.0, and the raw output bytes of all
eight tensors hash identically. TE selects the same backend either way
(``NVTE_F16_arbitrary_seqlen``). So the cuDNN version is **not a
comparability boundary for numbers**, and no published megatron figure is
numerically wrong because of it.

*``LD_LIBRARY_PATH`` alone does not move TE, and is worse than doing
nothing.* TE binds cuDNN in Python before any ``DT_NEEDED`` resolution:
``transformer_engine/common/__init__.py:345`` tries the system copy first,
and its last resort at ``:330`` is ``ctypes.CDLL("libcudnn.so",
RTLD_GLOBAL)`` -- the **unversioned** name. The wheel directory ships only
``libcudnn.so.9``, so the loader skips it and takes ``/usr/lib64``.
``RTLD_GLOBAL`` then captures TE's own entries. Setting only
``LD_LIBRARY_PATH`` leaves torch reporting 9.24.0 while TE still runs
9.23.2, which is a silent split in the opposite direction. Moving the whole
process needs ``CUDNN_PATH`` **and** ``LD_LIBRARY_PATH`` together; that
combination was measured to leave zero ``/usr/lib64/libcudnn`` mappings.

``PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK=1``
leaves TE on 9.23.2 and only stops torch refusing to answer: the version
read returns 92302 rather than raising
(``torch/backends/cudnn/__init__.py:58``). It selects no kernel here, but
**not** for the reason this file gave until 2026-08-20. The version read is
the *first* test in ``_can_use_cudnn`` (``varlen.py:54``), not a later one,
and the predicate that actually rejects this arm is the fourth,
``window_size != [-1, -1]`` (``varlen.py:60``), because torchtitan passes
``(-1, 0)``. ``enable_gqa`` is the fifth test and never decides it. The
outcome holds and the gate values stand; the stated mechanism was wrong.

**The flag is not confined to the process you set it in.**
``benchmarks/execution/environment.py:56`` builds the child environment as
``dict(environment or os.environ)``, so an exported flag reaches *every*
worker. A timing run started from a shell that exports it publishes every
number under it. Do **not** initialize torch's cuDNN before TE
imports as a third option: that loads the wheel's ``libcudnn_graph.so.9``
and leaves TE running a 9.24.0 graph engine against 9.23.2 ops, which is a
combination nobody tests.

**Do not set ``PackedSeqParams.cu_seqlens_q_padded``.** TE then computes
``pad_between_seqs = True`` (``dot_product_attention.py:1561-1570``), and
``utils.py:976-990`` disables FA2, FA4 **and** the unfused path. Probed on
this host with ``unfused`` requested and ``pad_between_seqs=True``, TE
reports no backend available and raises. The builders here leave the field
unset, and nothing checks that they do.

**Do not declare the ``local`` backend.** ``ATTENTION_BACKENDS`` offers the
name because it is a megatron enum member, but ``local`` zeroes all three
NVTE variables while the layer spec still builds ``TEDotProductAttention``,
so TE finds no backend and raises. No profile here uses it.

Which backend each mcore arm runs, and how the guard knows
----------------------------------------------------------

``LanguageModule.__init__`` calls ``_set_attention_backend``
(``language_module.py:49``), which turns ``config.attention_backend`` into
``NVTE_FLASH_ATTN`` / ``NVTE_FUSED_ATTN`` / ``NVTE_UNFUSED_ATTN``
(``:129-148``). TransformerEngine reads those three variables inside
``get_attention_backend``
(``transformer_engine/pytorch/attention/dot_product_attention/utils.py:457-463``)
and records what it chose in a module global, ``_attention_backends``
(``dot_product_attention.py:68``, written at ``:1662-1678``).

``_assert_te_selected_backend`` forces a fresh resolution, runs one real
forward through the arm's own closure, and reads that global back. It is
TransformerEngine's own decision about the arm's own call, not a
re-derivation of it. **It raises. It never warns**, because
``check_and_set_env_variable`` only *disables* the other backends: it does
not prove the requested one is reachable, and an arm that fell through to
another backend would publish a cuDNN number under a FlashAttention label.

Measured on an H200 (sm90, TE 2.17.1, cuDNN 9.23.2, FA3 3.0.0, FA4
4.0.0b25) with megatron's exact THD parameters -- ``qkv_layout='thd_thd_thd'``,
``attn_mask_type='padding_causal'``, 16 heads over 8 groups, head_dim 64:

===============  ===========================================================
backend asked    what TransformerEngine selected
===============  ===========================================================
``auto``         ``FusedAttention (sub-backend 1)`` -- the cuDNN kernel
``fused``        the same, and this is why ``mcore/base`` pins ``fused``
``flash``        ``FlashAttention (3.0.0)``
``unfused``      ``UnfusedDotProductAttention``
===============  ===========================================================

Two consequences a reader must not lose:

**No megatron SETTING selects FA4 on this device, and FA4 itself is not
unreachable.** TE prefers FA3 on sm90 whenever both are installed --
"Disabling FlashAttention 4 to prefer FlashAttention 3 on SM90"
(``utils.py:504-515``) -- and no ``attention_backend`` value, and no other
``TransformerConfig`` field, reaches past that preference.

The preference is a policy, and the policy is guarded on a **mutable class
attribute**: ``FlashAttentionUtils.v3_is_installed`` (``utils.py:136``).
Setting it False makes ``get_attention_backend`` return
``flash_attention_backend = 4.0.0b25``, and ``backends.py:1013-1016``
dispatches on that ``major``, so the FA4 kernel really would run. An
adversarial reviewer measured both on this H200.

So the honest sentence is: **this module declines to monkeypatch TE's
version bookkeeping to build a benchmark arm.** An arm built that way would
measure TE with a field of its own state falsified, and would publish it
beside arms that ran TE as shipped. That is a choice, not a capability
limit, and ``titan/flex_flash`` therefore runs against ``titan``, which
isolates the lowering instead.

**Megatron's ``flash_attention_version`` field does not reach this TE.**
``language_module.py:154-159`` pins the generation by writing
``NVTE_FLASH_ATTN_V2/V3/V4``. Those names appear nowhere in TE 2.17.1 --
neither in its Python (``os.getenv("NVTE_...")`` over the whole attention
package yields ``NVTE_FLASH_ATTN``, ``NVTE_FUSED_ATTN`` and
``NVTE_UNFUSED_ATTN`` and nothing else) nor in either shipped shared object.
So the field writes variables nothing reads, and three arms differing only
in it would be one configuration under three labels. This module therefore
declares one ``flash`` arm and pins the generation with the guard instead of
with the field.

Why the environment is cleared before every megatron build
-----------------------------------------------------------

``check_and_set_env_variable`` **asserts** that any value already present
equals the one it is about to write (``language_module.py:124-126``). The
correctness pass builds every arm of a scenario in one interpreter
(``benchmarks/kernel/engine/run.py:289-298``), so the second megatron arm
would meet the first arm's variables and die on that assertion. Its own
message says the fix: unset them. ``clear_te_attention_environment`` does,
before each build.

The same one-process pass is why ``_assert_te_selected_backend`` sets
``backend_selection_requires_update``. TE re-runs the selection only when
``attention_params`` changes (``dot_product_attention.py:1655-1662``), and
two arms of this scenario differ in nothing but the environment, so the
second arm would otherwise reuse the first arm's cached choice -- and both
the measurement and a guard that merely read the global would be wrong
together.

Compile treatment, and it differs by engine
--------------------------------------------

Every megatron arm runs eager, because megatron compiles no whole
transformer layer and ``TEDotProductAttention`` carries no ``jit_fuser``.
The titan arms are all compiled, by two different mechanisms:
``FlexAttention`` holds a class-level ``torch.compile`` of
``flex_attention`` (``attention.py:251-255``), so ``titan`` and
``titan/flex_flash`` are NOT wrapped again here -- wrapping risks a double
compile or a graph break around the spmd context -- and only
``titan/flash_attention_3`` takes ``_compile_module``.

**``titan`` is autotuned and the FlashAttention arms are not.**
``FlexAttention.inductor_configs`` sets ``max_autotune=True`` and
``coordinate_descent_tuning=True`` (``attention.py:235-249``), which the
class-level compile applies. ``titan/flex_flash`` shares that class and
therefore that treatment; ``titan/flash_attention_3`` gets a plain
``torch.compile(fullgraph=True)``, and every megatron arm gets nothing. Read
any row against ``titan`` as a comparison of tuned Triton against an
untuned or eager opponent.

There is no isolated ``backward`` mode
---------------------------------------

The retained-graph trick most scenarios use is unavailable on both sides
here. TE's fused-attention autograd function consumes its saved-tensor
context on the first backward and then raises "ctx must have
.tensor_objects", and the existing ``attention`` scenario records the same
for its own arms. Both engines therefore declare ``forward`` and
``forward_backward`` only, which keeps them comparable; backward cost is
still forward_backward minus forward.

No bandwidth floor
------------------

Attention is a reduction over ``L`` per query, so its arithmetic grows with
the square of the sequence length while its traffic grows linearly. A copy
floor would bound nothing here, which is the reason ``qkv_prep`` and
``attn_out_proj`` declare none either. That does **not** make these numbers
device time: CLAUDE.md's "Method" section records that a burst-amortized
per-call cost holds host dispatch too, and only ``--burst`` separates the
two. Run it before ranking anything.

Deferred imports
----------------

Every torchtitan, TransformerEngine and megatron import lives inside the
function that needs it. ``VarlenAttention``'s constructor activates FA3, so
a module-scope import would make this whole module unimportable without the
``flash3`` dependency group, taking every other arm down with it. The
megatron arms alone need the submodule on ``sys.path`` and the TE
environment set before TE loads.

Relationship to the existing ``attention`` scenario
----------------------------------------------------

The three titan arms here are the re-homed ``baseline``, ``flex_flash`` and
``flash_attention_3``, renamed ``titan``, ``titan/flex_flash`` and
``titan/flash_attention_3``. Their treatment is unchanged, so their numbers
stay comparable to the older scenario's. This module deliberately shares no
code with ``benchmarks/kernel/operations/attention.py``: that module is a
holdover scheduled for removal with its scenario, and an import would make
the deletion a two-module change. The duplication ends when it does.
"""

from __future__ import annotations

import gc
import os
from dataclasses import dataclass
from typing import Any, Callable

import torch

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.operations.common import (
    _assert_kernel_marker,
    _compile_module,
    _navigate,
    _randn,
    _randn_like,
    _reset_grads,
    initialize_megatron_single_rank,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE, derive
from benchmarks.models.piper_qwen3.shape import PiperShape


# Captured by profiling, never guessed. FA3 degrades to FA2 rather than
# failing when it declines to register, so the marker is what separates a real
# FA3 measurement from an FA2 one wearing its label; the FA2 kernel names are
# recorded as the failure signature. FA4 needs no failure signature --
# BACKEND="FLASH" raises when flash_attn.cute is missing rather than degrading
# to Triton -- so its marker guards the reverse mistake: a refactor that drops
# kernel_options and leaves the arm measuring the baseline under an FA4 label.
FLEX_ATTENTION_MARKER = "flex_attention"
FA3_MARKER = "FlashAttnFwdSm90"
FA2_MARKERS = ("pytorch_flash::flash_fwd", "flash_bwd_dq_dk_dv_loop")
FA4_MARKER = "FlashAttentionForwardSm90"

# The FLASH backend wants a coarser q block than flex's default 128. Torch does
# not validate it -- the value is forwarded verbatim into FA4's block-sparse
# tensors -- so a mismatch surfaces inside FA4, not as a torch-level error.
FLEX_FLASH_BLOCK_SIZE = (256, 128)

# Where the megatron module lives on a built GPTModel, as data rather than as
# a chain of getattr calls. ``_navigate`` walks it and indexes the numeric
# segment.
MCORE_SELF_ATTENTION_PATH = "decoder.layers.0.self_attention"
MCORE_CORE_ATTENTION_ATTR = "core_attention"

# The names every arm and the fp64 reference return. The scenario declaration
# in ``benchmarks/kernel/registry.py`` repeats them as literals, because a
# declaration may not import a builder module; the test pins the two copies
# against each other.
GATED_OUTPUTS = ("out", "dq", "dk", "dv")

# The three TransformerEngine variables megatron writes for a backend choice,
# plus the three generation pins it also writes. TE 2.17.1 reads only the
# first three; the last three are cleared anyway, so a future TE that does
# read them cannot inherit a stale value from an earlier arm in this process.
TE_ATTENTION_ENVIRONMENT = (
    "NVTE_FLASH_ATTN",
    "NVTE_FUSED_ATTN",
    "NVTE_UNFUSED_ATTN",
    "NVTE_FLASH_ATTN_V2",
    "NVTE_FLASH_ATTN_V3",
    "NVTE_FLASH_ATTN_V4",
)

# The three megatron profiles this scenario declares. Each is BASE plus one
# field, and the field is the one TransformerConfig leaves at
# ``AttnBackend.auto`` (``transformer_config.py:144``).
#
# BASE itself is NOT changed. Pinning the backend there would move every
# megatron arm in every scenario, and the e2e megatron arm with them. The two
# settings agree on this host -- ``auto`` and ``fused`` both resolve to the
# cuDNN kernel for these parameters, measured -- but that agreement is a
# property of sm90 with FA3 installed, not of the configuration.
ATTN_FUSED = derive(
    BASE,
    name="attn_fused",
    description=(
        "megatron with attention_backend pinned to fused: NVTE_FUSED_ATTN=1 "
        "and the other two 0, so TransformerEngine selects its cuDNN "
        "FusedAttention kernel by construction rather than by its own "
        "Hopper preference. Measured to be the same selection auto makes "
        "here, which is what makes this arm the base"
    ),
    config_overrides={"attention_backend": "fused"},
)
ATTN_FLASH = derive(
    BASE,
    name="attn_flash",
    description=(
        "megatron with attention_backend pinned to flash: NVTE_FLASH_ATTN=1 "
        "and the other two 0. On sm90 with FA3 installed TransformerEngine "
        "resolves that to FlashAttention 3, because it disables FA4 to "
        "prefer FA3 there; the arm's guard enforces the generation, since "
        "megatron's flash_attention_version field reaches nothing in TE "
        "2.17.1"
    ),
    config_overrides={"attention_backend": "flash"},
)
ATTN_UNFUSED = derive(
    BASE,
    name="attn_unfused",
    description=(
        "megatron with attention_backend pinned to unfused: TE's own torch "
        "implementation, which materializes the score matrix. It is a real "
        "implementation and not a bandwidth floor, so it publishes a ratio "
        "against the base"
    ),
    config_overrides={"attention_backend": "unfused"},
)

# What ``_backend_verdict`` may be asked to enforce. ``flash3`` is a
# generation and the other two are families, and that asymmetry is the
# scenario's central finding: megatron names a family and cannot name a
# generation, so the guard names the one megatron cannot.
EXPECTED_BACKENDS = ("fused", "flash3", "unfused")

# The megatron roster as data: arm name -> (profile, the AttnBackend member
# the profile names, the selection TransformerEngine must then make). The
# three builders read this rather than repeating the strings, so a test can
# check the roster against the registry declaration without a GPU.
#
# The second and third entries differ for the flash arm on purpose. Megatron
# can ask for the FlashAttention *family*; only the guard can hold it to
# generation 3.
MCORE_ARMS: dict[str, tuple[Any, str, str]] = {
    "mcore/base": (ATTN_FUSED, "fused", "fused"),
    "mcore/attn_flash3": (ATTN_FLASH, "flash", "flash3"),
    "mcore/attn_unfused": (ATTN_UNFUSED, "unfused", "unfused"),
}

# Every arm of the scenario, in declaration order. The registry repeats these
# names as literals, because a declaration may not import a builder module.
ARM_NAMES = (
    "mcore/base",
    "mcore/attn_flash3",
    "mcore/attn_unfused",
    "titan",
    "titan/flex_flash",
    "titan/flash_attention_3",
)


@dataclass
class AttentionCoreInputs:
    """One q/k/v triple, and every mask form the six arms need.

    ``*_BLNH`` is what titan reads. Titan materializes all three tensors
    contiguously.

    **Megatron does not, and this scenario reproduces that.** Megatron's QKV
    GEMM writes one fused ``[T, G, (Q + 2) * H]`` tensor, where ``G`` counts
    the key/value groups and ``Q`` the query heads per group.
    ``qkv_fused_TGR`` is that buffer.
    ``get_query_key_value_tensors`` splits it into three strided views, and
    the query and the key are then normed and rotated, which writes a fresh
    contiguous tensor for each. **The value is neither normed nor rotated**,
    so ``v_TNH`` alone stays a view of the buffer.

    **That stride is part of the measurand.** TE's ``get_qkv_layout``
    classifies the three tensors, does not recognize this layout, and then
    forces ``.contiguous()``
    (``dot_product_attention/utils.py:2428-2431``). Only the value moves:
    4 MiB per forward at the default workload, inside every timed megatron
    call. The sibling scenario ``qkv_prep`` states that megatron **defers**
    this copy to the arm that consumes the views; this scenario is that arm
    for the value. See the module docstring for the key's half, which
    ``qk_norm`` now measures.
    """

    q_BLNH: torch.Tensor
    k_BLNH: torch.Tensor
    v_BLNH: torch.Tensor
    grad_BLNH: torch.Tensor
    qkv_fused_TGR: torch.Tensor
    q_TNH: torch.Tensor
    k_TNH: torch.Tensor
    v_TNH: torch.Tensor
    grad_TD: torch.Tensor
    positions: torch.Tensor  # (B, L) int32, resets to 0 at document starts
    block_mask: object  # BlockMask at flex's default 128 block size
    block_mask_flash: object  # the same mask at FLEX_FLASH_BLOCK_SIZE
    cu_seqlens: torch.Tensor  # int32, packed document boundaries for THD
    max_seqlen: int
    scale: float
    num_documents: int


def _packed_positions(
    workload: KernelWorkload, device: torch.device, generator: torch.Generator
) -> torch.Tensor:
    """Synthetic packed-document positions: reset to 0 at each document start.

    A seeded mix of document lengths rather than one document per row, because
    a single full-length document would make the block-diagonal mask dense and
    hide exactly the sparsity these kernels exist to exploit. Every row starts
    at 0, which the flex document mask, the cu_seqlens construction and the
    fp64 reference all require -- and which is what makes the per-row flex
    mask and the flat THD packing the same predicate.
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


def _megatron_core_attention_inputs(
    shape: PiperShape, fused: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The three tensors megatron's ``core_attention`` really receives.

    ``get_query_key_value_tensors`` splits the fused buffer with one
    ``torch.split`` along the last dimension, so all three start as strided
    views. **Two of them stop being views before the cut, and the third does
    not**, and this scenario begins after RoPE:

    * The query is reshaped (``attention.py:1908``), then normed
      (``:1923``), then rotated (``:1500``). The reshape alone writes a
      fresh tensor, because it merges the group dimension with the query
      heads inside a group and the fused row is wider than ``Q * H``.
    * The key is normed (``:1926``) and rotated (``:1518``). ``BASE`` sets
      ``qk_layernorm: True``, so ``k_layernorm`` is a real norm and not an
      ``IdentityOp`` (``gpt_layer_specs.py:148-152``). A norm reading a
      non-dense view writes a contiguous output.
    * The value gets neither. Megatron's own comment records the choice
      (``attention.py:1534``: RoPE is deliberately not applied to value), so
      the value reaches ``core_attention`` as the original strided view.

    So the layout at this cut is contiguous, contiguous, strided -- and the
    copy TE then forces is the value alone.
    """
    per_group, head_dim = shape.heads_per_group, shape.head_dim
    query, key, value = torch.split(
        fused, [per_group * head_dim, head_dim, head_dim], dim=2
    )
    return (
        # Both reshape and contiguous copy here, which is what the norm and
        # the rotation do in the engine. Those costs belong to the qk_norm
        # and rope scenarios, so they must happen outside every timed
        # closure -- and they do, because this runs at build time.
        query.reshape(fused.shape[0], shape.n_heads, head_dim),
        key.contiguous(),
        value,
    )


def _megatron_qkv_buffer(
    shape: PiperShape,
    q_BLNH: torch.Tensor,
    k_BLNH: torch.Tensor,
    v_BLNH: torch.Tensor,
) -> torch.Tensor:
    """The fused ``[T, G, (Q + 2) * H]`` tensor megatron's QKV GEMM writes.

    The values are the canonical ones, so every arm reads the same numbers
    and only the memory layout differs. The head order agrees with titan's
    by construction: titan's query head ``n`` belongs to key/value group
    ``n // Q``, which is the group megatron's interleave puts it in, and
    ``megatron_weights.assert_qkv_roundtrip`` proves that mapping bitwise
    for the weights.
    """
    tokens = q_BLNH.shape[0] * q_BLNH.shape[1]
    groups, per_group = shape.n_kv_heads, shape.heads_per_group
    head_dim = shape.head_dim
    fused = torch.empty(
        (tokens, groups, (per_group + 2) * head_dim),
        dtype=q_BLNH.dtype,
        device=q_BLNH.device,
    )
    query_width = per_group * head_dim
    fused[..., :query_width] = q_BLNH.reshape(tokens, groups, query_width)
    fused[..., query_width : query_width + head_dim] = k_BLNH.reshape(
        tokens, groups, head_dim
    )
    fused[..., query_width + head_dim :] = v_BLNH.reshape(
        tokens, groups, head_dim
    )
    return fused


def attention_core_inputs(
    shape: PiperShape,
    workload: KernelWorkload,
    device: torch.device,
    generator: torch.Generator,
) -> AttentionCoreInputs:
    from torch.nn.attention.flex_attention import and_masks
    from torchtitan.models.common.attention import (
        create_attention_mask,
        create_varlen_metadata_for_document,
        get_causal_mask_mod,
        get_efficient_causal_mask_mod_for_packed_document,
    )

    batch, seq = workload.batch, workload.seq_len
    tokens = batch * seq
    q = _randn((batch, seq, shape.n_heads, shape.head_dim), device, generator)
    k = _randn(
        (batch, seq, shape.n_kv_heads, shape.head_dim), device, generator
    )
    v = _randn(
        (batch, seq, shape.n_kv_heads, shape.head_dim), device, generator
    )
    grad = _randn_like(q, generator)

    cpu_generator = torch.Generator(device="cpu").manual_seed(
        int(generator.initial_seed()) & 0x7FFFFFFF
    )
    positions = _packed_positions(workload, device, cpu_generator)

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

    fused = _megatron_qkv_buffer(shape, q, k, v)
    q_mcore, k_mcore, v_mcore = _megatron_core_attention_inputs(shape, fused)

    return AttentionCoreInputs(
        q_BLNH=q,
        k_BLNH=k,
        v_BLNH=v,
        grad_BLNH=grad,
        qkv_fused_TGR=fused,
        # Megatron's three tensors with megatron's strides: a contiguous
        # query, a contiguous key, and the value as a strided view of the
        # fused buffer. The values equal q_BLNH, k_BLNH and v_BLNH element
        # for element.
        q_TNH=q_mcore,
        k_TNH=k_mcore,
        v_TNH=v_mcore,
        # Megatron's core attention returns [T, N*H], so its backward seed has
        # that shape. It is the same view of the same gradient the titan arms
        # receive as [B, L, N, H], element for element, which is what makes
        # the two backward measurements comparable at all.
        grad_TD=grad.reshape(tokens, shape.n_heads * shape.head_dim),
        positions=positions,
        block_mask=mask_at(128),
        block_mask_flash=mask_at(FLEX_FLASH_BLOCK_SIZE),
        cu_seqlens=varlen.cu_seq_q,
        # Pinned to seq_len on both sides, matching what
        # benchmarks/e2e/megatron/train.py hands the e2e arm and what
        # create_varlen_metadata_for_document reports.
        max_seqlen=seq,
        scale=shape.head_dim**-0.5,
        num_documents=int((positions == 0).sum()),
    )


def attention_core_reference(
    shape: PiperShape, workload: KernelWorkload, inputs: AttentionCoreInputs
) -> dict[str, torch.Tensor]:
    """fp64 masked-softmax attention, computed per (row, kv group).

    Chunked on purpose: the full [B, n_heads, L, L] fp64 score tensor is
    8.6 GiB at batch 4 / seq 4096 and 69 GiB at batch 32, so a one-shot
    reference would OOM exactly at the shapes worth measuring. Per chunk it
    is [heads_per_kv, L, L], which stays a few hundred MiB.

    Written in TorchTitan's [B, L, N, H] layout, which is the canonical form
    every arm's ``correctness_outputs`` returns.
    """
    batch, seq = workload.batch, workload.seq_len
    heads_per_kv = shape.heads_per_group
    device = inputs.q_BLNH.device

    document = torch.cumsum((inputs.positions == 0).int(), dim=1) - 1
    causal = torch.tril(torch.ones(seq, seq, device=device, dtype=torch.bool))

    out = torch.empty_like(inputs.q_BLNH, dtype=torch.float64)
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
                inputs.q_BLNH[b, :, lo:hi]
                .double()
                .detach()
                .transpose(0, 1)
                .requires_grad_()
            )
            k_chunk = (
                inputs.k_BLNH[b, :, group].double().detach().requires_grad_()
            )
            v_chunk = (
                inputs.v_BLNH[b, :, group].double().detach().requires_grad_()
            )

            scores = (q_chunk @ k_chunk.transpose(-1, -2)) * inputs.scale
            scores = scores.masked_fill(~mask[None, :, :], float("-inf"))
            chunk = torch.softmax(scores, dim=-1) @ v_chunk

            grad = inputs.grad_BLNH[b, :, lo:hi].double().transpose(0, 1)
            torch.autograd.backward(chunk, grad)

            out[b, :, lo:hi] = chunk.detach().transpose(0, 1)
            dq[b, :, lo:hi] = q_chunk.grad.transpose(0, 1)
            dk[b, :, group] = k_chunk.grad
            dv[b, :, group] = v_chunk.grad

    return {"out": out, "dq": dq, "dk": dk, "dv": dv}


@dataclass(frozen=True)
class _Layout:
    """One engine's native tensors, and the way back to the canonical form.

    ``q``/``k``/``v``/``grad`` are already in the shape **and the strides**
    that engine's module expects, because a reshape inside a timed closure
    would be timed as attention. ``to_canonical`` puts an output or a
    gradient back into ``[B, L, N, H]`` for the gates, which run outside the
    timed region.

    ``make_leaves`` builds one differentiable leaf set. It exists because
    ``Tensor.clone()`` **cannot** carry the megatron layout: the value is a
    strided view of the fused QKV buffer, which is non-overlapping but not
    dense, so ``preserve_format`` gives up and returns a contiguous tensor.
    Cloning the arm's inputs would therefore hand TE a layout it
    recognizes, and would delete the copy this scenario exists to measure,
    silently.
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    grad: torch.Tensor
    to_canonical_out: Callable[[torch.Tensor], torch.Tensor]
    to_canonical_qkv: Callable[[torch.Tensor], torch.Tensor]
    make_leaves: Callable[[], tuple[torch.Tensor, torch.Tensor, torch.Tensor]]


def _identity(tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def _titan_layout(inputs: AttentionCoreInputs) -> _Layout:
    def make_leaves() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # All three are contiguous, which is what titan's projection
        # materializes, so a clone carries the layout unchanged.
        return (
            inputs.q_BLNH.clone().requires_grad_(),
            inputs.k_BLNH.clone().requires_grad_(),
            inputs.v_BLNH.clone().requires_grad_(),
        )

    return _Layout(
        q=inputs.q_BLNH,
        k=inputs.k_BLNH,
        v=inputs.v_BLNH,
        grad=inputs.grad_BLNH,
        to_canonical_out=_identity,
        to_canonical_qkv=_identity,
        make_leaves=make_leaves,
    )


def _mcore_layout(
    shape: PiperShape, workload: KernelWorkload, inputs: AttentionCoreInputs
) -> _Layout:
    batch, seq = workload.batch, workload.seq_len

    def out_to_canonical(tensor: torch.Tensor) -> torch.Tensor:
        # [T, N*H] -> [B, L, N, H]. A view: megatron's THD output is
        # contiguous and its trailing dimension is head-major, the same
        # order titan's [B, L, N, H] holds.
        return tensor.reshape(batch, seq, shape.n_heads, shape.head_dim)

    def qkv_to_canonical(tensor: torch.Tensor) -> torch.Tensor:
        # [T, N, H] -> [B, L, N, H], the inverse of the reshape the inputs
        # builder used to make the THD views.
        return tensor.reshape(batch, seq, *tensor.shape[1:])

    def make_leaves() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # One buffer per leaf set, split megatron's own way. The value leaf
        # is a view into that buffer and keeps its strides. The query and
        # the key leaves are contiguous, because the norm and the rotation
        # write a fresh tensor for each before this cut. ``detach`` is what
        # makes a strided view a leaf; ``clone`` would make it contiguous
        # instead.
        fused = inputs.qkv_fused_TGR.clone()
        return tuple(
            tensor.detach().requires_grad_()
            for tensor in _megatron_core_attention_inputs(shape, fused)
        )

    return _Layout(
        q=inputs.q_TNH,
        k=inputs.k_TNH,
        v=inputs.v_TNH,
        grad=inputs.grad_TD,
        to_canonical_out=out_to_canonical,
        to_canonical_qkv=qkv_to_canonical,
        make_leaves=make_leaves,
    )


def _attention_core_arm(
    name: str, layout: _Layout, call: Callable[..., torch.Tensor]
) -> BuiltArm:
    """Forward and forward+backward only, with independent leaf sets.

    There is no isolated ``backward`` mode, and both engines are why. The
    retained-graph trick other scenarios use re-runs backward over one graph;
    TE's fused-attention autograd function consumes its saved-tensor context
    on the first backward and then raises "ctx must have .tensor_objects",
    and the existing ``attention`` scenario records the same for the titan
    arms. Dropping the mode from every arm keeps them comparable -- backward
    cost is still forward_backward minus forward.
    """

    forward_leaves = layout.make_leaves()
    round_trip_leaves = layout.make_leaves()
    check_leaves = layout.make_leaves()

    def forward():
        return call(*forward_leaves)

    def forward_backward() -> None:
        _reset_grads(*round_trip_leaves)
        torch.autograd.backward(call(*round_trip_leaves), layout.grad)

    def correctness_outputs() -> dict[str, torch.Tensor]:
        _reset_grads(*check_leaves)
        out = call(*check_leaves)
        torch.autograd.backward(out, layout.grad)
        return {
            "out": layout.to_canonical_out(out.detach()),
            "dq": layout.to_canonical_qkv(check_leaves[0].grad),
            "dk": layout.to_canonical_qkv(check_leaves[1].grad),
            "dv": layout.to_canonical_qkv(check_leaves[2].grad),
        }

    return BuiltArm(
        name=name,
        calls={"forward": forward, "forward_backward": forward_backward},
        correctness_outputs=correctness_outputs,
    )


def _probe_leaves(layout: _Layout) -> tuple[torch.Tensor, ...]:
    """A throwaway leaf set for a guard's one call.

    A guard must not consume the arm's own leaves: ``_assert_kernel_marker``
    and ``_assert_te_selected_backend`` each run a forward, and running it on
    ``forward_leaves`` would leave an autograd graph attached to the tensors
    the timing pass then measures.
    """
    return layout.make_leaves()


# ---------------------------------------------------------------------------
# TorchTitan arms
# ---------------------------------------------------------------------------


def build_attention_core_titan(
    shape: PiperShape, workload: KernelWorkload, inputs: AttentionCoreInputs
) -> BuiltArm:
    """TorchTitan FlexAttention: an Inductor Triton template over a BlockMask.

    Not wrapped in ``_compile_module``: ``FlexAttention`` holds a class-level
    ``torch.compile`` of ``flex_attention`` with ``max_autotune`` and
    ``coordinate_descent_tuning`` on, so wrapping it again risks a double
    compile or a graph break around its spmd context. The arm is compiled and
    autotuned; every row against it must say so.
    """
    from torchtitan.models.common.attention import FlexAttention

    module = FlexAttention.Config().build()
    enable_gqa = shape.n_heads > shape.n_kv_heads
    layout = _titan_layout(inputs)

    def call(q, k, v):
        return module(
            q,
            k,
            v,
            attention_masks=inputs.block_mask,
            scale=inputs.scale,
            enable_gqa=enable_gqa,
        )

    call(*_probe_leaves(layout))
    _assert_kernel_marker(
        lambda: call(layout.q, layout.k, layout.v),
        FLEX_ATTENTION_MARKER,
        "titan",
    )
    return _attention_core_arm("titan", layout, call)


def build_attention_core_titan_flex_flash(
    shape: PiperShape, workload: KernelWorkload, inputs: AttentionCoreInputs
) -> BuiltArm:
    """FlexAttention lowered to FlashAttention-4 instead of a Triton template.

    Same module, same BlockMask semantics and same mask_mod as ``titan`` --
    only the lowering differs -- so that pair isolates the kernel family.

    **Expected to lose on sm90, and that is not a verdict on FA4.**
    FlexAttention's packed-interval mask optimization is gated on compute
    capability 10/11, so on Hopper partial blocks evaluate the mask per lane.
    Report this arm as "FA4 running the generic per-lane mask path on sm90".
    """
    from torchtitan.models.common.attention import FlexAttention

    module = FlexAttention.Config(
        block_size=FLEX_FLASH_BLOCK_SIZE,
        kernel_options={"BACKEND": "FLASH"},
    ).build()
    enable_gqa = shape.n_heads > shape.n_kv_heads
    layout = _titan_layout(inputs)

    def call(q, k, v):
        # Not wrapped, matching titan: the class holds its own compile.
        return module(
            q,
            k,
            v,
            attention_masks=inputs.block_mask_flash,
            scale=inputs.scale,
            enable_gqa=enable_gqa,
        )

    call(*_probe_leaves(layout))
    _assert_kernel_marker(
        lambda: call(layout.q, layout.k, layout.v),
        FA4_MARKER,
        "titan/flex_flash",
    )
    return _attention_core_arm("titan/flex_flash", layout, call)


def build_attention_core_titan_flash3(
    shape: PiperShape, workload: KernelWorkload, inputs: AttentionCoreInputs
) -> BuiltArm:
    """FlashAttention-3 varlen over the same packed documents.

    ``VarlenAttention``'s constructor activates FA3, so building this arm at
    all requires the ``flash3`` dependency group; without it torch's registry
    raises rather than silently falling back. ``FA3_MARKER`` guards the other
    direction: FA3 degrades to FA2 rather than raising when it declines to
    register, and an FA2 kernel under an FA3 label is the failure this
    scenario exists to prevent.
    """
    from torchtitan.models.common.attention import (
        VarlenAttention,
        VarlenMetadata,
    )

    # Compiled explicitly: titan and titan/flex_flash are compiled too, by
    # FlexAttention's class-level compile. This one carries no autotune.
    module = _compile_module(VarlenAttention.Config().build())
    enable_gqa = shape.n_heads > shape.n_kv_heads
    layout = _titan_layout(inputs)
    metadata = VarlenMetadata(
        cu_seq_q=inputs.cu_seqlens,
        cu_seq_k=inputs.cu_seqlens,
        max_q=inputs.max_seqlen,
        max_k=inputs.max_seqlen,
    )

    def call(q, k, v):
        return module(
            q,
            k,
            v,
            attention_masks=metadata,
            scale=inputs.scale,
            enable_gqa=enable_gqa,
        )

    call(*_probe_leaves(layout))
    _assert_kernel_marker(
        lambda: call(layout.q, layout.k, layout.v),
        FA3_MARKER,
        "titan/flash_attention_3",
    )
    return _attention_core_arm("titan/flash_attention_3", layout, call)


# ---------------------------------------------------------------------------
# Megatron-core arms
# ---------------------------------------------------------------------------


def clear_te_attention_environment() -> None:
    """Drop every backend variable a previous megatron build wrote.

    ``check_and_set_env_variable`` asserts that a variable already present
    holds the value it is about to write (``language_module.py:124-126``), so
    a second arm asking for a different backend in the same interpreter dies
    on that assertion rather than reconfiguring. Megatron's own message names
    this as the fix. The correctness pass builds every arm of a scenario in
    one process, so this runs before every megatron build here.

    One TransformerEngine read is NOT undone by this, and it is harmless:
    ``utils.py:66`` caches ``NVTE_FLASH_ATTN`` in a module global at import
    time. Its only reader is ``:1472``, which gates a block of
    ``logger.warning`` calls about uninstalled FlashAttention versions. The
    selection itself reads the environment afresh on every call
    (``utils.py:457-463``), so a value frozen by whichever arm imported TE
    first changes no measurement.
    """
    for name in TE_ATTENTION_ENVIRONMENT:
        os.environ.pop(name, None)


def _assert_mcore_core_attention_is_te(arm: str, module: Any) -> None:
    """Refuse a core attention that is not megatron's TE module.

    ``get_gpt_decoder_block_spec`` picks the class, and this arm's whole
    claim is that the class is ``TEDotProductAttention`` -- the module that
    reads the NVTE variables the profile sets. A spec that resolved to
    megatron's local ``DotProductAttention`` would ignore
    ``attention_backend`` entirely, so all three megatron arms would be one
    implementation under three labels and no correctness gate could see it:
    every backend computes the same function.
    """
    from megatron.core.extensions.transformer_engine import (
        TEDotProductAttention,
    )

    if module is None:
        raise RuntimeError(
            f"{arm}: megatron built no {MCORE_CORE_ATTENTION_ATTR}; the arm "
            "would measure nothing and publish it as a win"
        )
    if not isinstance(module, TEDotProductAttention):
        raise RuntimeError(
            f"{arm}: megatron built {type(module).__name__} for "
            f"{MCORE_CORE_ATTENTION_ATTR}, not TEDotProductAttention. Only "
            "the TE module reads NVTE_FLASH_ATTN/NVTE_FUSED_ATTN/"
            "NVTE_UNFUSED_ATTN, so every attention_backend profile would "
            "build the same implementation under a different label"
        )


def _assert_mcore_cut_matches_the_reference(
    arm: str, attention: Any, module: Any, shape: PiperShape
) -> None:
    """Refuse a megatron attention that computes something else.

    Each of these would make the megatron arm compute a different function
    from the fp64 reference and from the titan arms. The gates would catch
    all three, at a tolerance failure that names no cause; this names one.

    * ``attn_mask_type`` must be ``causal``. ``TEDotProductAttention.forward``
      upgrades it to ``padding_causal`` under ``qkv_format="thd"``
      (``extensions/transformer_engine.py:2262-2268``), which is the same
      predicate with the document boundaries applied. ``no_mask`` would not
      be.
    * ``softmax_scale`` must be unset, so TE falls back to
      ``1/sqrt(kv_channels)`` (``dot_product_attention.py:446-449``), and
      ``kv_channels`` must equal the shape's ``head_dim``. Both halves are
      checked, because only the pair gives the titan arms' ``scale``.
      ``mcore_profiles.py`` derives ``kv_channels`` from the same shape
      today, so this asserts a link rather than a coincidence.
    * ``attention_dropout`` must be 0.0. Dropout would make the arm
      nondeterministic and would add device work no other arm carries.
    """
    from megatron.core.transformer.enums import AttnMaskType

    if attention.attn_mask_type is not AttnMaskType.causal:
        raise RuntimeError(
            f"{arm}: megatron's self attention declares attn_mask_type "
            f"{attention.attn_mask_type!r}, not AttnMaskType.causal; this "
            "scenario's fp64 reference is causal within each packed document"
        )
    if module.config.softmax_scale is not None:
        raise RuntimeError(
            f"{arm}: softmax_scale is {module.config.softmax_scale!r}, so "
            "TransformerEngine will not use 1/sqrt(head_dim); the titan arms "
            f"and the fp64 reference use {shape.head_dim ** -0.5!r}"
        )
    if module.config.kv_channels != shape.head_dim:
        raise RuntimeError(
            f"{arm}: kv_channels is {module.config.kv_channels!r}, not the "
            f"shape's head_dim {shape.head_dim!r}. TransformerEngine derives "
            "the softmax scale from kv_channels, so this arm would use a "
            "different scale than the titan arms and the fp64 reference"
        )
    if module.config.attention_dropout != 0.0:
        raise RuntimeError(
            f"{arm}: attention_dropout is {module.config.attention_dropout!r}"
            "; a dropout mask makes this arm nondeterministic and adds device "
            "work no other arm in the scenario carries"
        )


def _assert_config_pins_the_backend(
    arm: str, module: Any, expected: str
) -> None:
    """Refuse a build whose profile delta did not reach the config.

    ``attention.py:374-381`` hands ``core_attention`` the model's own config
    at ``world_size == 1``, because ``num_query_groups < world_size`` is
    False, so this reads the object the module runs with rather than a copy
    of it. A delta that failed to arrive would leave ``AttnBackend.auto``
    here, and ``auto`` enables all three NVTE variables -- which on this host
    resolves to the same cuDNN kernel ``fused`` does, so the ``flash`` and
    ``unfused`` arms would silently measure the base.
    """
    from megatron.core.transformer.enums import AttnBackend

    wanted = AttnBackend[expected]
    if module.config.attention_backend is not wanted:
        raise RuntimeError(
            f"{arm}: the built config carries attention_backend "
            f"{module.config.attention_backend!r}, not {wanted!r}. The "
            "profile delta did not reach TransformerConfig, so this arm "
            "would measure whatever backend TransformerEngine prefers and "
            "publish it under this label"
        )


def _flash_major_version(version: Any) -> int | None:
    """The FlashAttention major generation TE recorded, as an integer.

    TE stores a ``packaging`` version object. Read ``major`` where it exists
    and parse the string otherwise, so a future TE that records a plain
    string does not make this guard silently return None.
    """
    if version is None:
        return None
    major = getattr(version, "major", None)
    if isinstance(major, int):
        return major
    head = str(version).split(".", 1)[0]
    return int(head) if head.isdigit() else None


def _assert_te_selected_backend(
    arm: str,
    expected: str,
    probe: Callable[[], Any],
    shape: PiperShape,
) -> dict[str, Any]:
    """Refuse an arm whose kernel is not the one its name claims.

    This is the mandatory guard every megatron arm of this scenario carries,
    and it reads TransformerEngine's own decision rather than re-deriving it.
    ``dot_product_attention._attention_backends`` is the module global TE
    writes at ``:1662-1678`` from ``get_attention_backend``'s return value.

    Two of the three things that make it non-vacuous are here, and the third
    is in ``_backend_verdict``:

    * It sets ``backend_selection_requires_update`` first. TE re-runs the
      selection only when ``attention_params`` changes (``:1655-1662``), and
      the arms of this scenario differ in nothing but the environment, so a
      later arm in the correctness pass would otherwise inherit an earlier
      arm's answer -- and the guard would confirm it.
    * It runs ``probe``, which is the arm's own closure over its own tensors,
      so the recorded decision is about the call the timing pass will make.

    The reading of the record is split out because it needs neither TE nor a
    device, and a guard nothing exercises is a guard nobody has checked.
    """
    import transformer_engine.pytorch.attention.dot_product_attention.dot_product_attention as dpa  # noqa: E501

    dpa._attention_backends["backend_selection_requires_update"] = True
    probe()
    torch.cuda.synchronize()
    return _backend_verdict(arm, expected, dict(dpa._attention_backends), shape)


def _backend_verdict(
    arm: str, expected: str, record: dict[str, Any], shape: PiperShape
) -> dict[str, Any]:
    """Read one recorded backend selection, or raise saying what it was.

    ``record`` is a copy of TransformerEngine's ``_attention_backends``.

    The record's ``attention_params`` is checked before its verdict, which is
    the third thing that makes this guard non-vacuous: a record left by some
    other module's attention -- a different layout, a different mask type, a
    different head count -- cannot satisfy this arm.

    **That check is four fields, and it separates modules, not arms.** It
    reads ``qkv_layout``, ``attn_mask_type``, ``num_heads`` and
    ``num_gqa_groups`` only, and it does not read ``head_dim_qk``,
    ``max_seqlen_q``, ``qkv_dtype``, ``is_training``, ``attention_dropout``
    or ``deterministic``. It also cannot separate the three mcore arms of
    this scenario from each other: their ``AttentionParams`` are identical,
    because the arms differ only in ``os.environ`` and the environment is
    not a field of ``AttentionParams`` (``utils.py:272-306``). Forcing
    ``backend_selection_requires_update`` is what separates them, and
    ``_assert_te_selected_backend`` owns that half.

    ``expected`` is one of ``fused``, ``flash3`` or ``unfused``. ``flash3``
    checks the generation as well as the family, because megatron cannot pin
    it: ``flash_attention_version`` writes ``NVTE_FLASH_ATTN_V2/V3/V4`` and
    TE 2.17.1 reads no such variable. FA3 also degrades to FA2 rather than
    failing, which is the failure this catches on that side.

    Returns the named selection, so a caller may put it in the arm's notes.
    """
    if expected not in EXPECTED_BACKENDS:
        raise ValueError(
            f"{arm}: unknown expected backend {expected!r}; this guard "
            f"enforces one of {', '.join(EXPECTED_BACKENDS)}"
        )
    params = record.get("attention_params")
    if params is None:
        raise RuntimeError(
            f"{arm}: TransformerEngine recorded no attention_params, so it "
            "never ran its backend selection for this arm's call. The guard "
            "cannot say which kernel would run"
        )
    observed = (
        getattr(params, "qkv_layout", None),
        getattr(params, "attn_mask_type", None),
        getattr(params, "num_heads", None),
        getattr(params, "num_gqa_groups", None),
    )
    wanted = ("thd_thd_thd", "padding_causal", shape.n_heads, shape.n_kv_heads)
    if observed != wanted:
        raise RuntimeError(
            f"{arm}: the recorded backend selection describes {observed}, not "
            f"this arm's call {wanted}. Reading it would report some other "
            "attention module's kernel under this arm's name"
        )

    flash = bool(record.get("use_flash_attention"))
    fused = bool(record.get("use_fused_attention"))
    unfused = bool(record.get("use_unfused_attention"))
    generation = _flash_major_version(record.get("flash_attention_backend"))
    selected = (
        f"FlashAttention {record.get('flash_attention_backend')}"
        if flash
        else (
            f"FusedAttention {record.get('fused_attention_backend')}"
            if fused
            else "UnfusedDotProductAttention" if unfused else "NoBackend"
        )
    )

    if expected == "fused":
        satisfied = fused and not flash and not unfused
    elif expected == "unfused":
        satisfied = unfused and not flash and not fused
    else:  # flash3
        satisfied = flash and not fused and not unfused and generation == 3

    if not satisfied:
        raise RuntimeError(
            f"{arm}: TransformerEngine selected {selected}, but this arm is "
            f"published as {expected!r}. check_and_set_env_variable only "
            "disables the other backends; it does not make the requested one "
            "reachable, so an arm that falls through publishes one kernel "
            "under another's label"
        )
    # The generation is reported only when FlashAttention was actually
    # selected. TE leaves ``flash_attention_backend`` set to the version it
    # WOULD have used even on a call it gave to cuDNN -- observed on this
    # host under ``attention_backend=auto``, where the record reads
    # ``use_flash_attention=False`` beside ``flash_attention_backend=3.0.0``
    # -- so copying it unconditionally would put "FlashAttention 3" in a
    # cuDNN arm's provenance.
    return {
        "te_selected_backend": selected,
        "te_flash_generation": generation if flash else None,
    }


# The two norm modules of a qk_norm arm hold 128 values; a core attention
# module holds no parameters at all. Anything above this budget is the
# dropped GPTModel, not the module under test, and the budget is wide enough
# that allocator rounding cannot reach it.
_RESIDUAL_BUDGET_BYTES = 64 * 2**20


def _report_build_residual(arm: str, before: int) -> None:
    """Say so if the dropped GPTModel did not free.

    ``memory_pass`` reports ``max_memory_allocated``, which counts every live
    allocation. A surviving reference to the 1.07 B-parameter model adds
    about 2 GiB to this arm's peak memory and nothing to a titan arm's, so
    the memory column would then compare two engines and one model.

    This reports and does not raise. The timing columns are unaffected, and
    peak memory is a secondary metric here, so a hard failure would cost the
    whole ratio to protect a column the reader can discount. The worker's
    stdout lands in ``kernel_bench.log``, which the run keeps.
    """
    residual = torch.cuda.memory_allocated() - before
    if residual > _RESIDUAL_BUDGET_BYTES:
        print(
            f"WARNING attention_core/{arm}: the megatron model did not free "
            f"({residual / 2**20:.0f} MiB still allocated). Read this arm's "
            "peak_memory_gib as the model plus the attention, not as the "
            "attention."
        )


def _build_mcore_arm(
    arm: str,
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: AttentionCoreInputs,
) -> BuiltArm:
    """One megatron core-attention arm, taken from the model megatron builds.

    The module comes off a real ``GPTModel``, so its class is whatever
    megatron's own spec derivation chooses and nothing here can build a
    lookalike by mistake. The model is then dropped and only the module
    stays alive: peak memory is a published column, and a resident 1.07 B
    parameters would make every megatron arm look expensive for a reason
    that has nothing to do with attention.

    The profile, the ``AttnBackend`` member it names and the selection
    TransformerEngine must then make all come from ``MCORE_ARMS``. The last
    two are different strings on purpose -- ``flash`` is a family and
    ``flash3`` is a generation, and the gap between them is exactly what
    megatron cannot express.
    """
    profile, backend, expected = MCORE_ARMS[arm]
    clear_te_attention_environment()
    initialize_megatron_single_rank(torch.initial_seed() % (2**31))
    from megatron.core.packed_seq_params import PackedSeqParams

    from benchmarks.models.piper_qwen3.megatron_model import build_model

    before = torch.cuda.memory_allocated()
    model = build_model(
        seq_len=workload.seq_len, shape=shape, profile=profile
    )
    attention = _navigate(model, MCORE_SELF_ATTENTION_PATH)
    module = getattr(attention, MCORE_CORE_ATTENTION_ATTR, None)
    _assert_mcore_core_attention_is_te(arm, module)
    _assert_mcore_cut_matches_the_reference(arm, attention, module, shape)
    _assert_config_pins_the_backend(arm, module, backend)
    attn_mask_type = attention.attn_mask_type
    del attention, model
    gc.collect()
    torch.cuda.empty_cache()
    _report_build_residual(arm, before)

    # Built once, here. PackedSeqParams is a plain dataclass, but building it
    # inside the closure would put its construction inside the timed region,
    # and the cu_seqlens it holds came from a builder that syncs.
    packed = PackedSeqParams(
        qkv_format="thd",
        cu_seqlens_q=inputs.cu_seqlens,
        cu_seqlens_kv=inputs.cu_seqlens,
        max_seqlen_q=inputs.max_seqlen,
        max_seqlen_kv=inputs.max_seqlen,
    )
    layout = _mcore_layout(shape, workload, inputs)

    def call(q, k, v):
        # Eager, as megatron runs it: megatron compiles no whole transformer
        # layer and TEDotProductAttention carries no jit_fuser. The argument
        # list is the one Attention.forward uses at attention.py:1559-1566,
        # with attention_mask=None because the THD path derives every
        # boundary from cu_seqlens -- which is what
        # benchmarks/e2e/megatron/train.py:283 passes too.
        return module(
            q,
            k,
            v,
            None,
            attn_mask_type=attn_mask_type,
            attention_bias=None,
            packed_seq_params=packed,
        )

    notes = _assert_te_selected_backend(
        arm, expected, lambda: call(*_probe_leaves(layout)), shape
    )
    built = _attention_core_arm(arm, layout, call)
    built.notes.update(notes)
    return built


def build_attention_core_mcore_base(
    shape: PiperShape, workload: KernelWorkload, inputs: AttentionCoreInputs
) -> BuiltArm:
    """megatron's cuDNN FusedAttention, pinned rather than preferred.

    ``attention_backend`` is pinned to ``fused`` instead of left at
    ``AttnBackend.auto``. Measured on this host, the two make the same
    selection for these parameters -- TE disables FlashAttention "to give
    FusedAttention preference on Hopper+" (``utils.py:1541-1547``) -- so this
    arm runs the kernel the e2e megatron arm runs. Pinning is what makes that
    a property of the profile rather than of the device.
    """
    return _build_mcore_arm("mcore/base", shape, workload, inputs)


def build_attention_core_mcore_flash3(
    shape: PiperShape, workload: KernelWorkload, inputs: AttentionCoreInputs
) -> BuiltArm:
    """megatron routed to FlashAttention, which resolves to FA3 on sm90.

    The opponent of ``titan/flash_attention_3``: the same kernel family and
    the same THD ``cu_seqlens`` masking, reached through two different host
    stacks. The guard enforces generation 3, because megatron cannot -- its
    ``flash_attention_version`` writes variables TE 2.17.1 does not read.
    """
    return _build_mcore_arm("mcore/attn_flash3", shape, workload, inputs)


def build_attention_core_mcore_unfused(
    shape: PiperShape, workload: KernelWorkload, inputs: AttentionCoreInputs
) -> BuiltArm:
    """TransformerEngine's unfused torch attention.

    It materializes the score matrix, so it is the slow path rather than a
    bandwidth floor, and it publishes a ratio against the base like any other
    implementation. Expect its memory to dominate the peak-memory column:
    the scores are ``[documents, n_heads, max_seqlen, max_seqlen]`` and
    ``max_seqlen`` is pinned to the full sequence length on both engines.
    """
    return _build_mcore_arm("mcore/attn_unfused", shape, workload, inputs)
