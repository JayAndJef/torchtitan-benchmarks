"""CPU tests for the ``attention_core`` cross-engine kernel scenario.

The scenario needs a GPU, megatron-core and TransformerEngine to measure
anything. It does not need any of the three to be *wrong*, and the failures
that would ship a wrong number are reachable here:

* the fp64 reference is wrong, so every gate passes against a wrong truth;
* the two engines read different tensors, or the megatron output is
  canonicalized into the wrong head order, so the ratio compares two
  different computations;
* a profile delta is not a delta, or names a backend megatron does not have;
* a guard is vacuous -- it accepts a record that describes another module's
  attention, or a selection that is not the one the arm's name claims.

The guards carry most of the weight, and the reason is stated once here:
**every backend computes the same function**, so no correctness gate can
tell a cuDNN arm from a FlashAttention arm. Only the guards can. Their
verdict is split out of the TE call as ``_backend_verdict`` precisely so it
can be exercised without a device, and every branch of it is exercised
below.

Two claims in the module are not testable here and were measured on hardware
instead: which backend each profile resolves to, and that the two engines'
outputs agree with the fp64 reference. Both are recorded in the module
docstring with their numbers.
"""

import ast
import os
import re
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.kernel.operations import attention_core
from benchmarks.kernel.operations.attention_core import (
    ARM_NAMES,
    ATTN_FLASH,
    ATTN_FUSED,
    ATTN_UNFUSED,
    EXPECTED_BACKENDS,
    FA2_MARKERS,
    FA3_MARKER,
    FA4_MARKER,
    FLEX_ATTENTION_MARKER,
    FLEX_FLASH_BLOCK_SIZE,
    GATED_OUTPUTS,
    MCORE_ARMS,
    MCORE_CORE_ATTENTION_ATTR,
    MCORE_SELF_ATTENTION_PATH,
    TE_ATTENTION_ENVIRONMENT,
    _assert_config_pins_the_backend,
    _assert_mcore_core_attention_is_te,
    _assert_mcore_cut_matches_the_reference,
    _attention_core_arm,
    _backend_verdict,
    _flash_major_version,
    _mcore_layout,
    _packed_positions,
    _probe_leaves,
    _titan_layout,
    attention_core_inputs,
    attention_core_reference,
    clear_te_attention_environment,
)
from benchmarks.kernel.schema import KernelWorkload, shape_summary
from benchmarks.models.piper_qwen3.mcore_profiles import BASE
from benchmarks.models.piper_qwen3.shape import PIPER_1B, PiperShape

# head_dim stays 64, the real value and the only geometry the kernels read.
# 4 query heads over 2 kv groups keeps grouped-query attention exercised while
# the fp64 reference stays cheap.
TINY = PiperShape.derived(
    name="tiny", dim=256, n_layers=1, vocab_size=64, max_seq_len=64
)
TINY_WORKLOAD = KernelWorkload(batch=2, seq_len=32)


def cpu_inputs(shape=TINY, workload=TINY_WORKLOAD, seed=0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return attention_core_inputs(
        shape, workload, torch.device("cpu"), generator
    )


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual.double() - expected.double()).norm()
    return float(difference / expected.double().norm())


def _megatron_available() -> bool:
    """Whether the pinned megatron submodule can be reached on this host.

    Several guards read ``megatron.core.transformer.enums``, which imports
    nothing but ``enum`` -- no torch, no TransformerEngine, no device. It
    still needs the submodule checked out, so the tests skip rather than fail
    on a tree that has not initialized it.

    **Called from inside a test, never at module scope**, and that is not a
    style choice. ``add_megatron_to_path`` inserts the checkout at
    ``sys.path[0]``, and Megatron-LM ships a ``tests/`` directory of its own,
    so from then on ``import tests.<anything>`` resolves to megatron's tests
    rather than to this suite's. ``unittest discover`` imports every module
    before it runs any test, so a module-scope call breaks the later modules
    that do ``from tests.test_runner import ...``. Calling it at run time
    leaves every import already done.
    """
    from benchmarks.models.piper_qwen3.megatron_bootstrap import (
        add_megatron_to_path,
    )

    try:
        add_megatron_to_path()
        import megatron.core.transformer.enums  # noqa: F401
    except Exception:
        return False
    return True


def require_megatron(case: unittest.TestCase) -> None:
    if not _megatron_available():
        case.skipTest("megatron submodule not checked out")


def te_record(
    *,
    flash=False,
    fused=False,
    unfused=False,
    flash_version=None,
    fused_backend=None,
    qkv_layout="thd_thd_thd",
    attn_mask_type="padding_causal",
    num_heads=TINY.n_heads,
    num_gqa_groups=TINY.n_kv_heads,
) -> dict:
    """One entry of TransformerEngine's ``_attention_backends`` global.

    A plain dict of the same keys TE writes at
    ``dot_product_attention.py:1662-1678``, with an ``AttentionParams``
    stand-in whose attributes are the four the verdict reads. Not a mock of
    TE: the verdict only ever reads attributes, so this runs the same code
    path a real record does.
    """
    return {
        "attention_params": SimpleNamespace(
            qkv_layout=qkv_layout,
            attn_mask_type=attn_mask_type,
            num_heads=num_heads,
            num_gqa_groups=num_gqa_groups,
        ),
        "use_flash_attention": flash,
        "flash_attention_backend": flash_version,
        "use_fused_attention": fused,
        "fused_attention_backend": fused_backend,
        "use_unfused_attention": unfused,
        "backend_selection_requires_update": False,
    }


class ProfileDeltaTests(unittest.TestCase):
    """Each megatron profile must be a real delta, and only one field of it."""

    def test_each_profile_changes_exactly_the_backend_field(self) -> None:
        for profile, expected in (
            (ATTN_FUSED, "fused"),
            (ATTN_FLASH, "flash"),
            (ATTN_UNFUSED, "unfused"),
        ):
            with self.subTest(profile=profile.name):
                moved = {
                    key
                    for key in set(profile.config_overrides)
                    | set(BASE.config_overrides)
                    if profile.config_overrides.get(key)
                    != BASE.config_overrides.get(key)
                }
                self.assertEqual(moved, {"attention_backend"})
                self.assertEqual(
                    profile.config_overrides["attention_backend"], expected
                )

    def test_the_base_profile_still_leaves_the_backend_unset(self) -> None:
        """The delta is only a delta while BASE declares nothing.

        If BASE ever pins a backend, every arm here becomes a no-op against
        it and three labels describe one configuration -- the failure the
        plan's own verification section calls out by name.
        """
        self.assertNotIn("attention_backend", BASE.config_overrides)

    def test_the_three_profiles_have_distinct_names(self) -> None:
        names = {ATTN_FUSED.name, ATTN_FLASH.name, ATTN_UNFUSED.name}
        self.assertEqual(len(names), 3)

    def test_every_named_backend_is_a_megatron_enum_member(self) -> None:
        require_megatron(self)
        from megatron.core.transformer.enums import AttnBackend

        for profile in (ATTN_FUSED, ATTN_FLASH, ATTN_UNFUSED):
            with self.subTest(profile=profile.name):
                name = profile.config_overrides["attention_backend"]
                self.assertIn(name, AttnBackend.__members__)


class ArmRosterTests(unittest.TestCase):
    def test_the_mcore_table_covers_every_mcore_arm_name(self) -> None:
        self.assertEqual(
            sorted(MCORE_ARMS),
            sorted(name for name in ARM_NAMES if name.startswith("mcore/")),
        )

    def test_the_roster_is_exactly_these_six_names_in_this_order(
        self,
    ) -> None:
        """The names AND the order, because both are published.

        A name is what a `results.json` row is keyed by and what a merge note
        or a report cites. The order is the sweep order and puts the anchor
        first. A mutation reviewer showed that the previous roster test read
        only the three ``mcore/`` names, so any titan name could be renamed
        or reordered and no test in the suite noticed.
        """
        self.assertEqual(
            ARM_NAMES,
            (
                "mcore/base",
                "mcore/attn_flash3",
                "mcore/attn_unfused",
                "titan",
                "titan/flex_flash",
                "titan/flash_attention_3",
            ),
        )

    def test_every_expected_backend_is_one_the_verdict_enforces(self) -> None:
        for arm, (_, _, expected) in MCORE_ARMS.items():
            with self.subTest(arm=arm):
                self.assertIn(expected, EXPECTED_BACKENDS)

    def test_the_flash_arm_asks_a_family_and_is_held_to_a_generation(
        self,
    ) -> None:
        """The asymmetry that is this scenario's central finding.

        Megatron can name the FlashAttention family and nothing more:
        ``flash_attention_version`` writes ``NVTE_FLASH_ATTN_V2/V3/V4``, and
        TE 2.17.1 reads no such variable. So the profile says ``flash`` and
        only the guard says ``flash3``. If these two ever became equal, the
        arm would stop enforcing its own name.
        """
        _, backend, expected = MCORE_ARMS["mcore/attn_flash3"]
        self.assertEqual(backend, "flash")
        self.assertEqual(expected, "flash3")
        self.assertNotEqual(backend, expected)

    def test_the_anchor_names_the_backend_the_guard_enforces(self) -> None:
        """base is the one arm whose family and generation cannot differ."""
        _, backend, expected = MCORE_ARMS["mcore/base"]
        self.assertEqual((backend, expected), ("fused", "fused"))

    def test_the_environment_roster_covers_what_megatron_writes(self) -> None:
        """Every variable ``_set_attention_backend`` can write is cleared.

        ``language_module.py`` writes three backend variables and three
        generation pins. A variable left behind by an earlier arm makes
        ``check_and_set_env_variable`` assert, which kills the correctness
        pass and with it every arm of the scenario.
        """
        self.assertEqual(
            set(TE_ATTENTION_ENVIRONMENT),
            {
                "NVTE_FLASH_ATTN",
                "NVTE_FUSED_ATTN",
                "NVTE_UNFUSED_ATTN",
                "NVTE_FLASH_ATTN_V2",
                "NVTE_FLASH_ATTN_V3",
                "NVTE_FLASH_ATTN_V4",
            },
        )

    def test_the_roster_matches_what_megatrons_source_writes(self) -> None:
        """Read the variable names out of the pinned submodule, not memory.

        A submodule bump that renames one of these leaves this module
        clearing a variable nothing sets and failing to clear the one that
        matters, and the failure is an assertion inside megatron rather than
        anything this repo can attribute.
        """
        require_megatron(self)
        from benchmarks.models.piper_qwen3.megatron_bootstrap import (
            megatron_dir,
        )

        source = (
            megatron_dir()
            / "megatron"
            / "core"
            / "models"
            / "common"
            / "language_module"
            / "language_module.py"
        ).read_text()
        body = source.split("def _set_attention_backend", 1)[1].split(
            "\n    def ", 1
        )[0]
        written = set(re.findall(r'"(NVTE_[A-Z0-9_]+)"', body))
        self.assertEqual(
            written,
            {"NVTE_FLASH_ATTN", "NVTE_FUSED_ATTN", "NVTE_UNFUSED_ATTN"},
        )
        # The generation pin is written through an f-string, so the literal
        # names never appear in the source. Its presence is also the evidence
        # for this scenario's central correction, so assert it rather than
        # inferring it.
        self.assertRegex(body, r'f"NVTE_FLASH_ATTN_V\{version\}"')
        written |= {f"NVTE_FLASH_ATTN_V{version}" for version in (2, 3, 4)}
        self.assertEqual(written, set(TE_ATTENTION_ENVIRONMENT))

    def test_transformer_engine_still_ignores_the_generation_pin(self) -> None:
        """The refutation that shaped this roster, as a self-invalidating test.

        Megatron pins the FlashAttention generation by writing
        ``NVTE_FLASH_ATTN_V2/V3/V4`` (``language_module.py:154-159``). The
        installed TransformerEngine reads no such variable, so the field is
        inert, so ``mcore/attn_flash2`` and ``mcore/attn_flash4`` cannot be
        declared: three arms differing only in it would be one configuration
        under three labels.

        If a TE upgrade starts reading them, this fails -- and the failure is
        the signal to split the flash arm by generation and to declare the
        two arms the plan asked for.

        **The sweep is the whole Python package, and that is still less than
        the claim.** The claim is that TE reads the variable nowhere, and TE
        ships two compiled objects as well. Those were checked by hand with
        ``strings`` and hold no ``NVTE_FLASH`` string at all, but no test
        rereads them, so a future TE that read the pin from C++ would leave
        this test green. Repeat the ``strings`` check before you trust a
        green run against a new TE.
        """
        try:
            import transformer_engine.pytorch  # noqa: F401
        except Exception as error:  # pragma: no cover - host dependent
            self.skipTest(f"TransformerEngine unavailable: {error}")
        import transformer_engine

        root = Path(transformer_engine.__file__).resolve().parent
        sources = sorted(root.rglob("*.py"))
        # Non-vacuity: an empty sweep would pass the assertion below without
        # having read anything, and a package move is exactly how that
        # happens.
        self.assertGreater(len(sources), 3)
        self.assertTrue(
            any("NVTE_FLASH_ATTN" in path.read_text() for path in sources),
            "the sweep found no NVTE_FLASH_ATTN at all, so it is reading the "
            "wrong directory",
        )
        reads = [
            str(path)
            for path in sources
            if "NVTE_FLASH_ATTN_V" in path.read_text()
        ]
        self.assertEqual(
            reads,
            [],
            "TransformerEngine now reads the FlashAttention generation pin, so "
            "megatron's flash_attention_version is live: split the flash arm "
            "by generation and declare the arms this roster had to drop",
        )

    def test_clearing_the_environment_removes_every_variable(self) -> None:
        previous = {
            name: os.environ.get(name) for name in TE_ATTENTION_ENVIRONMENT
        }
        try:
            for name in TE_ATTENTION_ENVIRONMENT:
                os.environ[name] = "1"
            clear_te_attention_environment()
            for name in TE_ATTENTION_ENVIRONMENT:
                self.assertNotIn(name, os.environ)
            # And it is safe to repeat on a clean environment.
            clear_te_attention_environment()
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value


class InputsTests(unittest.TestCase):
    def test_the_gradient_seed_is_a_view_of_the_canonical_gradient(
        self,
    ) -> None:
        """Both engines are seeded with the same bytes in a different shape.

        Megatron's core attention returns ``[T, N*H]``, so its backward seed
        has that shape. If that were a copy of a different tensor, the two
        engines' backward measurements would not be comparable at all.
        """
        inputs = cpu_inputs()
        self.assertEqual(
            inputs.grad_BLNH.untyped_storage().data_ptr(),
            inputs.grad_TD.untyped_storage().data_ptr(),
        )
        self.assertTrue(
            torch.equal(
                inputs.grad_BLNH.reshape(-1), inputs.grad_TD.reshape(-1)
            )
        )

    def test_megatron_reads_the_canonical_values(self) -> None:
        """Only the layout differs between the engines, never the numbers."""
        inputs = cpu_inputs()
        batch, seq = TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len
        for native, mcore in (
            (inputs.q_BLNH, inputs.q_TNH),
            (inputs.k_BLNH, inputs.k_TNH),
            (inputs.v_BLNH, inputs.v_TNH),
        ):
            with self.subTest(shape=tuple(mcore.shape)):
                self.assertTrue(
                    torch.equal(
                        native,
                        mcore.reshape(batch, seq, *mcore.shape[1:]),
                    )
                )

    def test_the_value_alone_reaches_megatron_as_a_strided_view(
        self,
    ) -> None:
        """The measurand, not a detail, and it is the value alone.

        Megatron's QKV GEMM writes one fused buffer and
        ``get_query_key_value_tensors`` splits it into three views. The
        query and the key then leave the buffer, because the norm and the
        rotation write fresh tensors. The value gets neither, so
        ``core_attention`` receives one non-contiguous view and TE copies it
        inside the timed call.
        """
        inputs = cpu_inputs()
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        groups, head_dim = TINY.n_kv_heads, TINY.head_dim
        row = (TINY.heads_per_group + 2) * head_dim
        self.assertEqual(
            tuple(inputs.qkv_fused_TGR.shape), (tokens, groups, row)
        )
        self.assertFalse(inputs.v_TNH.is_contiguous())
        self.assertEqual(inputs.v_TNH.stride(), (groups * row, row, 1))
        self.assertEqual(
            inputs.v_TNH.untyped_storage().data_ptr(),
            inputs.qkv_fused_TGR.untyped_storage().data_ptr(),
        )
        # The value sits after the query block and the key block.
        self.assertEqual(
            inputs.v_TNH.storage_offset(),
            (TINY.heads_per_group + 1) * head_dim,
        )

    def test_the_packed_boundaries_carry_their_padding(self) -> None:
        """The LENGTH of cu_seqlens is load-bearing, not only its content.

        TE reads ``batch_size = cu_seqlens.shape[0] - 1``
        (``dot_product_attention/utils.py:2153``), so the padded length is
        what the unfused arm allocates its scores over. A mutation reviewer
        showed that trimming a trailing padding entry passed every test,
        because the other assertions compare two counts that both shrink.
        """
        inputs = cpu_inputs()
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        # torchtitan pads to a multiple of its own constant, so the count is
        # a property of the padding rule and not of the seeded draw.
        self.assertEqual(inputs.cu_seqlens.numel() % 128, 0)
        self.assertGreater(inputs.cu_seqlens.numel() - 1, inputs.num_documents)
        self.assertEqual(int(inputs.cu_seqlens[0]), 0)
        self.assertEqual(int(inputs.cu_seqlens[-1]), tokens)
        # The real boundaries come first: one entry per document, plus the
        # opening zero.
        self.assertEqual(
            int(inputs.cu_seqlens[inputs.num_documents]), tokens
        )
        # Every padding entry is the full offset, so it describes a
        # zero-length segment rather than a second copy of the last document.
        tail = inputs.cu_seqlens[inputs.num_documents :]
        self.assertTrue(bool((tail == tokens).all()))

    def test_the_query_and_the_key_reach_megatron_contiguous(self) -> None:
        """Both leave the fused buffer before this cut, and that is engine.

        The query is reshaped, normed and rotated; the key is normed and
        rotated. Each step writes a fresh tensor, and those costs belong to
        ``qkv_prep``, ``qk_norm`` and ``rope``. They must not happen again
        inside a timed closure here.
        """
        inputs = cpu_inputs()
        for tensor in (inputs.q_TNH, inputs.k_TNH):
            with self.subTest(shape=tuple(tensor.shape)):
                self.assertTrue(tensor.is_contiguous())
                self.assertNotEqual(
                    tensor.untyped_storage().data_ptr(),
                    inputs.qkv_fused_TGR.untyped_storage().data_ptr(),
                )

    def test_megatron_still_norms_the_key_and_still_skips_the_value(
        self,
    ) -> None:
        """Why the key is contiguous and the value is not. Self-invalidating.

        Two facts on the pinned megatron rev decide this layout, and a
        submodule bump can change either. If ``k_layernorm`` stopped
        reassigning the key, or the value started taking a rotation, the
        strides at this cut would move and the inputs builder would be
        modelling an engine that no longer exists.
        """
        require_megatron(self)
        from benchmarks.models.piper_qwen3.megatron_bootstrap import (
            megatron_dir,
        )

        source = (
            Path(megatron_dir()) / "megatron/core/transformer/attention.py"
        ).read_text()
        self.assertIn("key = apply_module(self.k_layernorm)(key)", source)
        self.assertTrue(
            BASE.config_overrides["qk_layernorm"],
            "with qk_layernorm off, k_layernorm is an IdentityOp and the key "
            "would reach attention as a strided view too",
        )
        # Megatron leaves the value alone, and says so in a comment.
        self.assertNotIn("value = apply_rotary_pos_emb(", source)
        self.assertIn(
            "# value_layer = apply_rotary_pos_emb(value_layer, k_pos_emb)",
            source,
        )

    def test_transformer_engine_does_not_recognize_the_megatron_layout(
        self,
    ) -> None:
        """The reason the strided views cost anything. Self-invalidating.

        TE's ``get_qkv_layout`` classifies q, k and v by their strides. It
        does not recognize megatron's interleaved GQA split, so it forces
        ``.contiguous()`` on all three and returns ``thd_thd_thd``. This test
        FAILS if a TE upgrade starts to recognize the layout -- at which
        point the copy disappears and the module docstring's claim about it
        must be rewritten rather than carried.

        Strides need no device, so this runs on CPU.
        """
        try:
            from transformer_engine.pytorch.attention.dot_product_attention.utils import (  # noqa: E501
                get_qkv_layout,
            )
        except Exception as error:  # pragma: no cover - host dependent
            raise unittest.SkipTest(
                f"TransformerEngine is not importable: {error}"
            )
        inputs = cpu_inputs()
        layout, q, k, v, _, _ = get_qkv_layout(
            inputs.q_TNH, inputs.k_TNH, inputs.v_TNH, qkv_format="thd"
        )
        self.assertEqual(layout, "thd_thd_thd")
        # The value is the only tensor TE has to move.
        self.assertEqual(q.data_ptr(), inputs.q_TNH.data_ptr())
        self.assertEqual(k.data_ptr(), inputs.k_TNH.data_ptr())
        self.assertNotEqual(v.data_ptr(), inputs.v_TNH.data_ptr())

    def test_the_shapes_follow_the_geometry(self) -> None:
        inputs = cpu_inputs()
        batch, seq = TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len
        tokens = batch * seq
        self.assertEqual(
            tuple(inputs.q_BLNH.shape),
            (batch, seq, TINY.n_heads, TINY.head_dim),
        )
        self.assertEqual(
            tuple(inputs.k_BLNH.shape),
            (batch, seq, TINY.n_kv_heads, TINY.head_dim),
        )
        self.assertEqual(tuple(inputs.v_BLNH.shape), tuple(inputs.k_BLNH.shape))
        self.assertEqual(
            tuple(inputs.q_TNH.shape), (tokens, TINY.n_heads, TINY.head_dim)
        )
        self.assertEqual(
            tuple(inputs.grad_TD.shape),
            (tokens, TINY.n_heads * TINY.head_dim),
        )
        self.assertEqual(inputs.q_BLNH.dtype, torch.bfloat16)
        self.assertEqual(inputs.max_seqlen, seq)
        self.assertAlmostEqual(inputs.scale, TINY.head_dim**-0.5)

    def test_every_row_starts_a_document(self) -> None:
        """What makes the flex mask and the THD packing one predicate.

        The flex ``BlockMask`` is per row and ``cu_seqlens`` is flat over
        ``B * L``. They agree only while every row boundary is also a
        document boundary, which holds because every row starts at position
        0.
        """
        inputs = cpu_inputs()
        self.assertTrue(bool((inputs.positions[:, 0] == 0).all()))

    def test_the_documents_are_not_all_one_full_row(self) -> None:
        """A single full-length document would hide the sparsity under test."""
        inputs = cpu_inputs()
        self.assertGreater(inputs.num_documents, TINY_WORKLOAD.batch)

    def test_cu_seqlens_are_nondecreasing_and_end_at_the_token_total(
        self,
    ) -> None:
        inputs = cpu_inputs()
        cu = inputs.cu_seqlens
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        self.assertEqual(cu.dtype, torch.int32)
        self.assertEqual(int(cu[0]), 0)
        self.assertEqual(int(cu[-1]), tokens)
        self.assertTrue(bool((torch.diff(cu) >= 0).all()))
        # The real boundaries come first and the padding is trailing
        # zero-length segments at the token total, which is what
        # benchmarks/e2e/megatron/train.py sends the e2e arm too.
        self.assertEqual(
            int((cu == tokens).sum()), cu.numel() - inputs.num_documents
        )

    def test_the_two_block_masks_use_the_declared_block_sizes(self) -> None:
        inputs = cpu_inputs()
        self.assertEqual(inputs.block_mask.BLOCK_SIZE, (128, 128))
        self.assertEqual(
            inputs.block_mask_flash.BLOCK_SIZE, FLEX_FLASH_BLOCK_SIZE
        )

    def test_inputs_rebuild_bit_identically_from_the_same_seed(self) -> None:
        """Every worker rebuilds these; a drift would desynchronize the arms."""
        first, second = cpu_inputs(), cpu_inputs()
        self.assertTrue(torch.equal(first.q_BLNH, second.q_BLNH))
        self.assertTrue(torch.equal(first.grad_BLNH, second.grad_BLNH))
        self.assertTrue(torch.equal(first.positions, second.positions))
        self.assertTrue(torch.equal(first.cu_seqlens, second.cu_seqlens))
        self.assertNotEqual(
            cpu_inputs(seed=1).q_BLNH.reshape(-1)[0],
            first.q_BLNH.reshape(-1)[0],
        )

    def test_the_positions_helper_packs_whole_rows(self) -> None:
        generator = torch.Generator(device="cpu").manual_seed(3)
        positions = _packed_positions(
            TINY_WORKLOAD, torch.device("cpu"), generator
        )
        self.assertEqual(
            tuple(positions.shape),
            (TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len),
        )
        for row in positions:
            # Each document runs 0, 1, 2, ... so a step is either +1 or a
            # reset to 0. Anything else is a packing bug the mask would hide.
            steps = torch.diff(row)
            self.assertTrue(bool(((steps == 1) | (row[1:] == 0)).all()))

    def test_no_arm_declares_bytes_moved(self) -> None:
        """Attention is compute-bound, so a bandwidth number bounds nothing.

        The scenario declares no floor for the same reason. An arm that
        started declaring ``bytes_moved`` would publish a GB/s column the
        cut cannot support.
        """
        inputs = cpu_inputs()
        layout = _titan_layout(inputs)
        arm = _attention_core_arm("probe", layout, lambda q, k, v: q)
        self.assertIsNone(arm.bytes_moved)


class ReferenceTests(unittest.TestCase):
    def test_the_reference_matches_a_dense_fp64_computation(self) -> None:
        """The chunked reference against the obvious one, at a shape both fit.

        The chunking exists because a one-shot ``[B, n_heads, L, L]`` fp64
        score tensor is 8.6 GiB at batch 4 / seq 4096. Small enough, the two
        must agree exactly in structure and to fp64 round-off in value.
        """
        inputs = cpu_inputs()
        got = attention_core_reference(TINY, TINY_WORKLOAD, inputs)

        batch, seq = TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len
        document = torch.cumsum((inputs.positions == 0).int(), dim=1) - 1
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool))
        mask = (document[:, :, None] == document[:, None, :]) & causal[None]

        q = inputs.q_BLNH.double().detach().requires_grad_()
        k = inputs.k_BLNH.double().detach().requires_grad_()
        v = inputs.v_BLNH.double().detach().requires_grad_()
        # [B, N, L, H] with the kv heads repeated across their query group.
        repeats = TINY.heads_per_group
        scores = (
            q.transpose(1, 2)
            @ k.transpose(1, 2).repeat_interleave(repeats, dim=1).transpose(
                -1, -2
            )
        ) * inputs.scale
        scores = scores.masked_fill(~mask[:, None, :, :], float("-inf"))
        out = torch.softmax(scores, dim=-1) @ v.transpose(
            1, 2
        ).repeat_interleave(repeats, dim=1)
        out = out.transpose(1, 2)
        torch.autograd.backward(out, inputs.grad_BLNH.double())

        self.assertLess(rel_l2(got["out"], out.detach()), 1e-12)
        self.assertLess(rel_l2(got["dq"], q.grad), 1e-12)
        self.assertLess(rel_l2(got["dk"], k.grad), 1e-12)
        self.assertLess(rel_l2(got["dv"], v.grad), 1e-12)

    def test_the_reference_respects_the_document_boundaries(self) -> None:
        """A reference that leaked across documents would pass every gate.

        Zeroing one document's values must leave every other document's
        output untouched. A causal-only reference -- the easy mistake --
        fails this, because a later document attends back into the zeroed one.
        """
        inputs = cpu_inputs()
        baseline = attention_core_reference(TINY, TINY_WORKLOAD, inputs)

        starts = (inputs.positions[0] == 0).nonzero().flatten().tolist()
        self.assertGreaterEqual(len(starts), 2)
        first, second = starts[0], starts[1]
        inputs.v_BLNH[0, first:second] = 0.0
        perturbed = attention_core_reference(TINY, TINY_WORKLOAD, inputs)

        changed = (
            perturbed["out"][0, first:second] - baseline["out"][0, first:second]
        ).abs()
        untouched = (
            perturbed["out"][0, second:] - baseline["out"][0, second:]
        ).abs()
        self.assertGreater(float(changed.max()), 0.0)
        self.assertEqual(float(untouched.max()), 0.0)
        # And the other row of the batch is a separate packing entirely.
        self.assertEqual(
            float((perturbed["out"][1] - baseline["out"][1]).abs().max()), 0.0
        )

    def test_the_reference_returns_every_gated_output(self) -> None:
        got = attention_core_reference(TINY, TINY_WORKLOAD, cpu_inputs())
        self.assertEqual(sorted(got), sorted(GATED_OUTPUTS))
        for name, tensor in got.items():
            with self.subTest(output=name):
                self.assertEqual(tensor.dtype, torch.float64)


class LayoutTests(unittest.TestCase):
    def test_the_titan_layout_needs_no_conversion(self) -> None:
        inputs = cpu_inputs()
        layout = _titan_layout(inputs)
        self.assertIs(layout.q, inputs.q_BLNH)
        self.assertIs(layout.grad, inputs.grad_BLNH)
        sample = torch.randn_like(inputs.q_BLNH)
        self.assertIs(layout.to_canonical_out(sample), sample)
        self.assertIs(layout.to_canonical_qkv(sample), sample)

    def test_the_mcore_layout_round_trips_to_the_canonical_form(self) -> None:
        """The claim the whole cross-engine comparison rests on.

        Megatron returns ``[T, N*H]``. If that folded the head dimension in a
        different order than titan's ``[B, L, N, H]``, every gate would
        compare permuted tensors and the scenario would publish a ratio
        between two different computations. Measured against real
        TransformerEngine output on an H200 too; this is the CPU half.
        """
        inputs = cpu_inputs()
        layout = _mcore_layout(TINY, TINY_WORKLOAD, inputs)
        self.assertIs(layout.q, inputs.q_TNH)
        self.assertIs(layout.grad, inputs.grad_TD)

        canonical = layout.to_canonical_out(inputs.grad_TD)
        self.assertEqual(tuple(canonical.shape), tuple(inputs.grad_BLNH.shape))
        self.assertTrue(torch.equal(canonical, inputs.grad_BLNH))

        for thd, native in (
            (inputs.q_TNH, inputs.q_BLNH),
            (inputs.k_TNH, inputs.k_BLNH),
        ):
            with self.subTest(shape=tuple(thd.shape)):
                back = layout.to_canonical_qkv(thd)
                self.assertEqual(tuple(back.shape), tuple(native.shape))
                self.assertTrue(torch.equal(back, native))

    def test_a_megatron_leaf_set_keeps_megatron_strides(self) -> None:
        """A leaf set must carry the layout, or the copy is never timed.

        The timed closures call the leaves, not the inputs. A leaf set built
        with ``clone`` would be contiguous and TE would recognize it, so the
        arm would measure a layout megatron never produces.
        """
        inputs = cpu_inputs()
        layout = _mcore_layout(TINY, TINY_WORKLOAD, inputs)
        query, key, value = layout.make_leaves()
        self.assertTrue(query.is_contiguous())
        self.assertTrue(key.is_contiguous())
        self.assertFalse(value.is_contiguous())
        self.assertEqual(value.stride(), inputs.v_TNH.stride())
        self.assertEqual(
            value.storage_offset(), inputs.v_TNH.storage_offset()
        )
        for leaf, reference in (
            (query, inputs.q_TNH),
            (key, inputs.k_TNH),
            (value, inputs.v_TNH),
        ):
            with self.subTest(shape=tuple(leaf.shape)):
                self.assertTrue(leaf.is_leaf)
                self.assertTrue(leaf.requires_grad)
                self.assertTrue(torch.equal(leaf, reference))
                self.assertNotEqual(
                    leaf.untyped_storage().data_ptr(),
                    reference.untyped_storage().data_ptr(),
                )

    def test_a_titan_leaf_set_is_contiguous_and_independent(self) -> None:
        inputs = cpu_inputs()
        layout = _titan_layout(inputs)
        first = layout.make_leaves()
        second = layout.make_leaves()
        for leaf, other, reference in zip(
            first, second, (inputs.q_BLNH, inputs.k_BLNH, inputs.v_BLNH)
        ):
            with self.subTest(shape=tuple(leaf.shape)):
                self.assertTrue(leaf.is_contiguous())
                self.assertTrue(leaf.is_leaf and leaf.requires_grad)
                self.assertTrue(torch.equal(leaf, reference))
                self.assertNotEqual(leaf.data_ptr(), other.data_ptr())
                self.assertNotEqual(leaf.data_ptr(), reference.data_ptr())

    def test_cloning_a_strided_view_returns_a_contiguous_tensor(self) -> None:
        """The torch behaviour ``make_leaves`` exists for.

        ``Tensor.clone()`` preserves the format only for a tensor that is
        non-overlapping AND dense. A strided view of the fused QKV buffer is
        not dense, so ``clone`` silently gives back a contiguous tensor. If
        torch ever changes that, this test fails and ``make_leaves`` can be
        simplified -- but nothing may rely on ``clone`` until it does.
        """
        inputs = cpu_inputs()
        self.assertFalse(inputs.v_TNH.is_contiguous())
        self.assertTrue(inputs.v_TNH.clone().is_contiguous())



class _Recorder:
    """A stand-in attention that remembers what it was handed.

    The timed closures own their leaves, so a test cannot reach them from
    outside. This is the one thing that can: the closure passes its leaves
    to the call, so the call records them.
    """

    def __init__(self) -> None:
        self.seen: list[tuple] = []
        self.grad_at_entry: list = []

    def __call__(self, q, k, v):
        self.seen.append((q, k, v))
        self.grad_at_entry.append(
            None if q.grad is None else float(q.grad.abs().sum())
        )
        repeats = q.shape[-2] // k.shape[-2]
        return (
            q
            + k.repeat_interleave(repeats, dim=-2)
            + v.repeat_interleave(repeats, dim=-2)
        )


class SharedClosureTests(unittest.TestCase):
    """The closures every arm shares, over a stand-in attention.

    The stand-in is one line of arithmetic, not a mock of either engine: what
    is under test is the leaf handling, the gradient seeding and the
    canonicalization, none of which cares which kernel ran.
    """

    @staticmethod
    def _call(q, k, v):
        heads = q.shape[-2]
        groups = k.shape[-2]
        return q + k.repeat_interleave(heads // groups, dim=-2) + v.repeat_interleave(
            heads // groups, dim=-2
        )

    def test_the_builder_exposes_exactly_the_declared_modes(self) -> None:
        arm = _attention_core_arm(
            "probe", _titan_layout(cpu_inputs()), self._call
        )
        self.assertEqual(sorted(arm.calls), ["forward", "forward_backward"])

    def test_the_three_leaf_sets_are_independent(self) -> None:
        """A guard's call, a timed call and a gate must not share a graph.

        Sharing one leaf set would let a forward_backward burst accumulate
        gradients into the tensors the gate then reads, and would leave an
        autograd graph attached to the tensors the timing pass measures.

        This reads the tensors each closure was handed, not only the output
        keys. A mutation reviewer collapsed the three sets into one and this
        test still passed, because ``correctness_outputs`` resets the
        gradients of whatever it was given before it reads them.
        """
        recorder = _Recorder()
        arm = _attention_core_arm(
            "probe", _titan_layout(cpu_inputs()), recorder
        )
        arm.calls["forward"]()
        arm.calls["forward_backward"]()
        outputs = arm.correctness_outputs()
        self.assertEqual(sorted(outputs), sorted(GATED_OUTPUTS))
        for name in GATED_OUTPUTS:
            self.assertIsNotNone(outputs[name])

        pointers = [q.data_ptr() for q, _, _ in recorder.seen]
        self.assertEqual(len(pointers), 3)
        self.assertEqual(
            len(set(pointers)), 3, "the three closures share a leaf set"
        )

    def test_a_guard_probe_never_touches_the_arm_tensors(self) -> None:
        """``_probe_leaves`` must hand a guard something it may consume.

        A guard runs a forward. On the arm's own tensors that would leave an
        autograd graph attached to what the timing pass measures, which is
        the function's stated reason to exist.
        """
        inputs = cpu_inputs()
        layout = _titan_layout(inputs)
        probes = _probe_leaves(layout)
        self.assertEqual(len(probes), 3)
        for probe, native in zip(probes, (layout.q, layout.k, layout.v)):
            with self.subTest(shape=tuple(probe.shape)):
                self.assertNotEqual(probe.data_ptr(), native.data_ptr())
                self.assertTrue(probe.is_leaf and probe.requires_grad)
                self.assertTrue(torch.equal(probe, native))

    def test_the_gate_keeps_the_gradients_apart(self) -> None:
        """dk and dv must not be swapped, and shapes cannot detect that.

        The key and the value gradients have the same shape, so the earlier
        shape checks pass either way. This closure weights them differently,
        which makes a swap arithmetic rather than cosmetic.
        """

        def weighted(q, k, v):
            repeats = q.shape[-2] // k.shape[-2]
            return (
                q
                + 2.0 * k.repeat_interleave(repeats, dim=-2)
                + 3.0 * v.repeat_interleave(repeats, dim=-2)
            )

        inputs = cpu_inputs()
        outputs = _attention_core_arm(
            "probe", _titan_layout(inputs), weighted
        ).correctness_outputs()
        batch, seq, heads, head_dim = inputs.grad_BLNH.shape
        groups = inputs.k_BLNH.shape[2]
        # Each key/value group receives the gradient of the query heads that
        # read it, summed.
        pooled = (
            inputs.grad_BLNH.double()
            .reshape(batch, seq, groups, heads // groups, head_dim)
            .sum(dim=3)
        )
        self.assertLess(rel_l2(outputs["dk"], 2.0 * pooled), 2e-2)
        self.assertLess(rel_l2(outputs["dv"], 3.0 * pooled), 2e-2)
        # And the swap this test exists for is far outside that gate.
        self.assertGreater(rel_l2(outputs["dk"], 3.0 * pooled), 0.2)

    def test_forward_backward_clears_the_gradients_between_calls(self) -> None:
        """A burst is thousands of calls; accumulation would be timed work.

        The gradient is read where it accumulates -- on the tensors
        ``forward_backward`` itself is handed. Reading it through
        ``correctness_outputs`` cannot see this: that path owns a different
        leaf set and resets it first, so a mutation reviewer deleted the
        reset and the old version of this test still passed.
        """
        recorder = _Recorder()
        arm = _attention_core_arm(
            "probe", _titan_layout(cpu_inputs()), recorder
        )
        for _ in range(4):
            arm.calls["forward_backward"]()
        # Every call must start from a cleared gradient. The first sees None
        # because nothing ran; the rest see None because the closure reset
        # them.
        self.assertEqual(recorder.grad_at_entry, [None, None, None, None])

    def test_the_mcore_closures_return_canonical_tensors(self) -> None:
        """The gates read one shape whichever engine produced it."""
        inputs = cpu_inputs()
        layout = _mcore_layout(TINY, TINY_WORKLOAD, inputs)
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len

        def call(q, k, v):
            heads, groups = q.shape[-2], k.shape[-2]
            repeats = heads // groups
            summed = (
                q
                + k.repeat_interleave(repeats, dim=-2)
                + v.repeat_interleave(repeats, dim=-2)
            )
            # Megatron's THD output shape.
            return summed.reshape(tokens, heads * q.shape[-1])

        outputs = _attention_core_arm(
            "probe", layout, call
        ).correctness_outputs()
        canonical = (
            TINY_WORKLOAD.batch,
            TINY_WORKLOAD.seq_len,
            TINY.n_heads,
            TINY.head_dim,
        )
        self.assertEqual(tuple(outputs["out"].shape), canonical)
        self.assertEqual(tuple(outputs["dq"].shape), canonical)
        self.assertEqual(
            tuple(outputs["dk"].shape),
            (
                TINY_WORKLOAD.batch,
                TINY_WORKLOAD.seq_len,
                TINY.n_kv_heads,
                TINY.head_dim,
            ),
        )


class BackendVerdictTests(unittest.TestCase):
    """The mandatory guard, branch by branch.

    Every branch matters, because no correctness gate can see any of them:
    cuDNN, FlashAttention and the unfused path compute the same function, so
    an arm that fell through to the wrong one passes every gate the scenario
    declares.
    """

    def test_each_arm_accepts_the_selection_its_name_claims(self) -> None:
        for expected, record in (
            ("fused", te_record(fused=True, fused_backend="NVTE_F16")),
            ("flash3", te_record(flash=True, flash_version="3.0.0")),
            ("unfused", te_record(unfused=True)),
        ):
            with self.subTest(expected=expected):
                notes = _backend_verdict("arm", expected, record, TINY)
                self.assertIn("te_selected_backend", notes)

    def test_a_fused_arm_refuses_a_flash_selection(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "FlashAttention"):
            _backend_verdict(
                "mcore/base",
                "fused",
                te_record(flash=True, flash_version="3.0.0"),
                TINY,
            )

    def test_a_flash_arm_refuses_a_fused_selection(self) -> None:
        """The failure this scenario exists to prevent, in one direction.

        TE prefers FusedAttention on Hopper, so an arm whose ``flash``
        request did not reach it lands here and would otherwise publish the
        anchor's own kernel under a FlashAttention label.
        """
        with self.assertRaisesRegex(RuntimeError, "FusedAttention"):
            _backend_verdict(
                "mcore/attn_flash3",
                "flash3",
                te_record(fused=True, fused_backend="NVTE_F16"),
                TINY,
            )

    def test_a_flash3_arm_refuses_flash2_and_flash4(self) -> None:
        """The generation check megatron cannot deliver.

        FA3 degrades to FA2 rather than failing, and a future TE could
        prefer FA4 on this device. Both would publish under an FA3 label.
        """
        for version in ("2.7.3", "4.0.0b25"):
            with self.subTest(version=version):
                with self.assertRaisesRegex(RuntimeError, "FlashAttention"):
                    _backend_verdict(
                        "mcore/attn_flash3",
                        "flash3",
                        te_record(flash=True, flash_version=version),
                        TINY,
                    )

    def test_an_unfused_arm_refuses_every_fast_path(self) -> None:
        for record in (
            te_record(fused=True, fused_backend="NVTE_F16"),
            te_record(flash=True, flash_version="3.0.0"),
        ):
            with self.subTest(record=record["use_fused_attention"]):
                with self.assertRaises(RuntimeError):
                    _backend_verdict(
                        "mcore/attn_unfused", "unfused", record, TINY
                    )

    def test_no_backend_at_all_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "NoBackend"):
            _backend_verdict("mcore/base", "fused", te_record(), TINY)

    def test_a_record_that_names_two_backends_is_refused(self) -> None:
        """TE returns exactly one, so two means the record is not TE's.

        ``utils.py:1550-1554`` forces exclusivity before ``get_attention_
        backend`` returns, so a record with two flags set describes
        something this guard must not read. A mutation reviewer removed
        every ``not flash and not unfused`` conjunct and no test noticed,
        because every stand-in record here set exactly one flag.
        """
        for expected, record in (
            ("fused", te_record(fused=True, unfused=True)),
            ("fused", te_record(fused=True, flash=True, flash_version="3.0")),
            ("unfused", te_record(unfused=True, fused=True)),
            (
                "flash3",
                te_record(flash=True, flash_version="3.0", fused=True),
            ),
        ):
            with self.subTest(expected=expected):
                with self.assertRaises(RuntimeError):
                    _backend_verdict("mcore/base", expected, record, TINY)

    def test_a_record_from_another_module_cannot_satisfy_the_guard(
        self,
    ) -> None:
        """The non-vacuity check: the verdict must match THIS cut.

        What the parameter check catches is a record left by a **different
        module** -- another scenario's arm, or a build that ran at another
        shape or mask -- because ``_attention_backends`` is a module global
        of TE, shared by every module in the process.

        It does **not** separate the three mcore arms of this scenario from
        each other. Their ``AttentionParams`` are identical: they differ
        only in ``os.environ``, which is not a field of ``AttentionParams``
        (``dot_product_attention/utils.py:272-306``). The only thing that
        forces a fresh selection for the second arm is
        ``backend_selection_requires_update = True``, which
        ``_assert_te_selected_backend`` sets. Do not delete that line
        believing this check covers it.
        """
        for wrong in (
            {"qkv_layout": "sbhd_sbhd_sbhd"},
            {"attn_mask_type": "causal"},
            {"num_heads": TINY.n_heads + 1},
            {"num_gqa_groups": TINY.n_kv_heads + 1},
        ):
            with self.subTest(**wrong):
                with self.assertRaisesRegex(RuntimeError, "describes"):
                    _backend_verdict(
                        "mcore/base",
                        "fused",
                        te_record(
                            fused=True, fused_backend="NVTE_F16", **wrong
                        ),
                        TINY,
                    )

    def test_an_empty_record_is_refused(self) -> None:
        """TE writes nothing when its selection never ran for this call."""
        record = te_record(fused=True)
        record["attention_params"] = None
        with self.assertRaisesRegex(RuntimeError, "no attention_params"):
            _backend_verdict("mcore/base", "fused", record, TINY)

    def test_an_unknown_expectation_is_refused_before_anything_else(
        self,
    ) -> None:
        with self.assertRaisesRegex(ValueError, "unknown expected backend"):
            _backend_verdict("mcore/base", "cudnn", te_record(fused=True), TINY)

    def test_the_guard_reads_the_shape_it_was_given(self) -> None:
        """A record for the normal shape must not satisfy a tiny-shape arm."""
        with self.assertRaises(RuntimeError):
            _backend_verdict(
                "mcore/base",
                "fused",
                te_record(
                    fused=True,
                    fused_backend="NVTE_F16",
                    num_heads=PIPER_1B.n_heads,
                    num_gqa_groups=PIPER_1B.n_kv_heads,
                ),
                TINY,
            )

    def test_a_fused_arm_records_no_flash_generation(self) -> None:
        """TE leaves a FlashAttention version beside a cuDNN verdict.

        Observed on this host under ``attention_backend=auto``: the record
        reads ``use_flash_attention=False`` and
        ``flash_attention_backend=3.0.0`` together, because TE resolved the
        version it would have used before its Hopper policy declined it.
        Copying the field unconditionally would put "FlashAttention 3" in a
        cuDNN arm's recorded provenance.
        """
        notes = _backend_verdict(
            "mcore/base",
            "fused",
            te_record(
                fused=True, fused_backend="NVTE_F16", flash_version="3.0.0"
            ),
            TINY,
        )
        self.assertEqual(notes["te_selected_backend"], "FusedAttention NVTE_F16")
        self.assertIsNone(notes["te_flash_generation"])
        flash = _backend_verdict(
            "mcore/attn_flash3",
            "flash3",
            te_record(flash=True, flash_version="3.0.0"),
            TINY,
        )
        self.assertEqual(flash["te_flash_generation"], 3)

    def test_the_flash_generation_is_read_from_a_version_object(self) -> None:
        """TE stores a packaging version, not a string, and may change that."""
        from packaging.version import Version

        self.assertEqual(_flash_major_version(Version("3.0.0")), 3)
        self.assertEqual(_flash_major_version(Version("4.0.0b25")), 4)
        self.assertEqual(_flash_major_version("2.7.3"), 2)
        self.assertIsNone(_flash_major_version(None))
        self.assertIsNone(_flash_major_version("unknown"))


class McoreBuildGuardTests(unittest.TestCase):
    """The build-time refusals, over stand-ins for the megatron objects.

    Each catches a configuration that computes the wrong thing or measures
    the wrong module. None of them needs a device: they read attributes.
    """

    def test_a_missing_core_attention_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "built no"):
            _assert_mcore_core_attention_is_te("mcore/base", None)

    def test_a_module_that_is_not_the_te_class_is_refused(self) -> None:
        """The failure no gate can see, and it would silence every arm.

        Only ``TEDotProductAttention`` reads the NVTE variables the profiles
        set. A spec that resolved to megatron's local attention would build
        one implementation for all three arms, and all three would pass every
        correctness gate.
        """
        require_megatron(self)
        try:
            import megatron.core.extensions.transformer_engine  # noqa: F401
        except Exception as error:  # pragma: no cover - host dependent
            self.skipTest(f"TransformerEngine unavailable: {error}")
        lookalike = type("DotProductAttention", (), {})()
        with self.assertRaisesRegex(RuntimeError, "TEDotProductAttention"):
            _assert_mcore_core_attention_is_te("mcore/base", lookalike)

    def test_a_non_causal_mask_type_is_refused(self) -> None:
        require_megatron(self)
        from megatron.core.transformer.enums import AttnMaskType

        attention = SimpleNamespace(attn_mask_type=AttnMaskType.no_mask)
        module = SimpleNamespace(
            config=SimpleNamespace(softmax_scale=None, attention_dropout=0.0)
        )
        with self.assertRaisesRegex(RuntimeError, "attn_mask_type"):
            _assert_mcore_cut_matches_the_reference(
                "mcore/base", attention, module, TINY
            )

    def test_a_foreign_softmax_scale_or_dropout_is_refused(self) -> None:
        require_megatron(self)
        from megatron.core.transformer.enums import AttnMaskType

        attention = SimpleNamespace(attn_mask_type=AttnMaskType.causal)
        with self.assertRaisesRegex(RuntimeError, "softmax_scale"):
            _assert_mcore_cut_matches_the_reference(
                "mcore/base",
                attention,
                SimpleNamespace(
                    config=SimpleNamespace(
                        softmax_scale=0.5,
                        kv_channels=TINY.head_dim,
                        attention_dropout=0.0,
                    )
                ),
                TINY,
            )
        with self.assertRaisesRegex(RuntimeError, "attention_dropout"):
            _assert_mcore_cut_matches_the_reference(
                "mcore/base",
                attention,
                SimpleNamespace(
                    config=SimpleNamespace(
                        softmax_scale=None,
                        kv_channels=TINY.head_dim,
                        attention_dropout=0.1,
                    )
                ),
                TINY,
            )

    def test_the_configured_cut_passes(self) -> None:
        """The guard must not refuse the configuration the profiles build."""
        require_megatron(self)
        from megatron.core.transformer.enums import AttnMaskType

        _assert_mcore_cut_matches_the_reference(
            "mcore/base",
            SimpleNamespace(attn_mask_type=AttnMaskType.causal),
            SimpleNamespace(
                config=SimpleNamespace(
                    softmax_scale=None,
                    kv_channels=TINY.head_dim,
                    attention_dropout=BASE.config_overrides[
                        "attention_dropout"
                    ],
                )
            ),
            TINY,
        )

    def test_a_kv_channels_that_is_not_the_head_dim_is_refused(self) -> None:
        """The other half of the scale, and it is not the obvious half.

        TE does not read ``head_dim``. With ``softmax_scale`` unset it uses
        ``1/sqrt(kv_channels)``, so a profile that set ``kv_channels`` to
        anything else would run a different scale than the titan arms and
        the fp64 reference, and every gate would fail at a tolerance that
        names no cause.
        """
        require_megatron(self)
        from megatron.core.transformer.enums import AttnMaskType

        with self.assertRaisesRegex(RuntimeError, "kv_channels"):
            _assert_mcore_cut_matches_the_reference(
                "mcore/base",
                SimpleNamespace(attn_mask_type=AttnMaskType.causal),
                SimpleNamespace(
                    config=SimpleNamespace(
                        softmax_scale=None,
                        kv_channels=TINY.head_dim * 2,
                        attention_dropout=0.0,
                    )
                ),
                TINY,
            )

    def test_a_backend_that_did_not_reach_the_config_is_refused(self) -> None:
        """auto is the value a lost delta leaves behind, and it is silent.

        On this host ``auto`` resolves to the same cuDNN kernel ``fused``
        does, so a flash or unfused arm whose delta was dropped would measure
        the anchor and pass every gate.
        """
        require_megatron(self)
        from megatron.core.transformer.enums import AttnBackend

        module = SimpleNamespace(
            config=SimpleNamespace(attention_backend=AttnBackend.auto)
        )
        with self.assertRaisesRegex(RuntimeError, "attention_backend"):
            _assert_config_pins_the_backend(
                "mcore/attn_flash3", module, "flash"
            )
        for arm, (_, backend, _) in MCORE_ARMS.items():
            with self.subTest(arm=arm):
                pinned = SimpleNamespace(
                    config=SimpleNamespace(
                        attention_backend=AttnBackend[backend]
                    )
                )
                _assert_config_pins_the_backend(arm, pinned, backend)


class MarkerTests(unittest.TestCase):
    def test_the_two_flash_markers_cannot_satisfy_each_other(self) -> None:
        """FA4 spells FlashAttention out where FA3 abbreviates it.

        If either marker were a substring of the other's kernel names, an arm
        could run the wrong generation and still pass its own guard.
        """
        self.assertNotIn(FA3_MARKER, FA4_MARKER)
        self.assertNotIn(FA4_MARKER, FA3_MARKER)
        for marker in (FA3_MARKER, FA4_MARKER):
            for fa2 in FA2_MARKERS:
                self.assertNotIn(marker, fa2)

    def test_the_flex_marker_is_not_a_flash_marker(self) -> None:
        for marker in (FA3_MARKER, FA4_MARKER):
            self.assertNotIn(FLEX_ATTENTION_MARKER, marker)


class ShapeSummaryTests(unittest.TestCase):
    def test_the_manifest_records_both_engine_layouts(self) -> None:
        summary = shape_summary("attention_core", PIPER_1B, KernelWorkload())
        batch, seq = KernelWorkload().batch, KernelWorkload().seq_len
        self.assertEqual(
            summary["q_titan_BLNH"],
            [batch, seq, PIPER_1B.n_heads, PIPER_1B.head_dim],
        )
        self.assertEqual(
            summary["q_mcore_THD"],
            [batch * seq, PIPER_1B.n_heads, PIPER_1B.head_dim],
        )
        self.assertEqual(
            summary["out_mcore_TD"],
            [batch * seq, PIPER_1B.n_heads * PIPER_1B.head_dim],
        )
        self.assertEqual(
            summary["flex_flash_block_size"], list(FLEX_FLASH_BLOCK_SIZE)
        )
        self.assertEqual(summary["max_seqlen"], seq)

    def test_the_manifest_branch_is_exactly_this_mapping(self) -> None:
        """Every key, not a sample of them.

        The manifest is how a reader without this repo learns what the arms
        consumed. A mutation reviewer showed that reading five of the keys
        left the other ten free to drift, including the fused-buffer width
        and the flex block size, which are the two that describe the layout
        asymmetry this scenario measures.
        """
        workload = KernelWorkload()
        batch, seq = workload.batch, workload.seq_len
        tokens = batch * seq
        heads, groups = PIPER_1B.n_heads, PIPER_1B.n_kv_heads
        head_dim = PIPER_1B.head_dim
        self.assertEqual(
            shape_summary("attention_core", PIPER_1B, workload),
            {
                "q_titan_BLNH": [batch, seq, heads, head_dim],
                "k_titan_BLNH": [batch, seq, groups, head_dim],
                "v_titan_BLNH": [batch, seq, groups, head_dim],
                "qkv_mcore_fused_TGR": [
                    tokens,
                    groups,
                    (PIPER_1B.heads_per_group + 2) * head_dim,
                ],
                "q_mcore_THD": [tokens, heads, head_dim],
                "k_mcore_THD": [tokens, groups, head_dim],
                "v_mcore_THD": [tokens, groups, head_dim],
                "v_mcore_is_a_strided_view": True,
                "out_titan_BLNH": [batch, seq, heads, head_dim],
                "out_mcore_TD": [tokens, heads * head_dim],
                "positions": [batch, seq],
                "packed_tokens": tokens,
                "flex_block_size": 128,
                "flex_flash_block_size": list(FLEX_FLASH_BLOCK_SIZE),
                "max_seqlen": seq,
                "max_seq_len": PIPER_1B.max_seq_len,
            },
        )

    def test_the_navigation_path_names_the_first_layer(self) -> None:
        """The path is data, so it can drift away from the model silently."""
        self.assertEqual(
            MCORE_SELF_ATTENTION_PATH, "decoder.layers.0.self_attention"
        )
        self.assertEqual(MCORE_CORE_ATTENTION_ATTR, "core_attention")


class BuilderWiringTests(unittest.TestCase):
    """Which arm each builder builds, and which guard it arms.

    **No test can call these builders without a GPU**: five of the six reach
    TransformerEngine, megatron or torchtitan, and the sixth compiles. So
    they are read instead. A mutation reviewer showed what that gap costs --
    pointing the FA3 builder at the cuDNN anchor, or dropping the FA4
    ``kernel_options``, passed all 61 tests and the whole suite. Each of
    those publishes one arm's number under another arm's name, which is the
    failure this scenario exists to prevent.

    Reading the source is weaker than running it. It is what is available
    here, and it is stronger than nothing.
    """

    #: builder -> (arm name, marker constant or None)
    WIRING = {
        "build_attention_core_mcore_base": ("mcore/base", None),
        "build_attention_core_mcore_flash3": ("mcore/attn_flash3", None),
        "build_attention_core_mcore_unfused": ("mcore/attn_unfused", None),
        "build_attention_core_titan": ("titan", "FLEX_ATTENTION_MARKER"),
        "build_attention_core_titan_flex_flash": (
            "titan/flex_flash",
            "FA4_MARKER",
        ),
        "build_attention_core_titan_flash3": (
            "titan/flash_attention_3",
            "FA3_MARKER",
        ),
    }

    MARKERS = ("FLEX_ATTENTION_MARKER", "FA4_MARKER", "FA3_MARKER")

    @staticmethod
    def _body(name: str) -> ast.FunctionDef:
        """The builder's body, with its docstring removed.

        The docstrings name other arms on purpose -- they explain the
        comparisons -- so a search over them would report every arm in every
        builder.
        """
        tree = ast.parse(Path(attention_core.__file__).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == name:
                body = list(node.body)
                if (
                    body
                    and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)
                ):
                    body = body[1:]
                return ast.Module(body=body, type_ignores=[])
        raise AssertionError(f"no builder named {name}")

    @classmethod
    def _strings(cls, name: str) -> set:
        return {
            node.value
            for node in ast.walk(cls._body(name))
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }

    @classmethod
    def _names(cls, name: str) -> set:
        return {
            node.id
            for node in ast.walk(cls._body(name))
            if isinstance(node, ast.Name)
        }

    def test_every_declared_arm_has_exactly_one_builder(self) -> None:
        self.assertEqual(
            sorted(name for name, _ in self.WIRING.values()),
            sorted(ARM_NAMES),
        )
        for builder in self.WIRING:
            with self.subTest(builder=builder):
                self.assertTrue(hasattr(attention_core, builder))

    def test_each_builder_names_its_own_arm_and_no_other(self) -> None:
        """A builder wired to another arm's name is a wrong number."""
        for builder, (arm, _) in self.WIRING.items():
            with self.subTest(builder=builder):
                strings = self._strings(builder)
                self.assertIn(arm, strings)
                for other in ARM_NAMES:
                    if other != arm:
                        self.assertNotIn(other, strings)

    def test_each_titan_builder_arms_its_own_marker(self) -> None:
        """The silent-fallback guards, one per titan arm.

        FA3 degrades to FA2 rather than failing, and a future refactor could
        drop the FA4 lowering, so each arm is held to the kernel name only
        it can produce. A builder holding another arm's marker guards
        nothing.
        """
        for builder, (_, marker) in self.WIRING.items():
            with self.subTest(builder=builder):
                names = self._names(builder)
                if marker is None:
                    self.assertIn("_build_mcore_arm", names)
                    continue
                self.assertIn("_assert_kernel_marker", names)
                self.assertIn(marker, names)
                for other in self.MARKERS:
                    if other != marker:
                        self.assertNotIn(other, names)

    def test_calling_a_megatron_builder_asks_for_its_own_arm(self) -> None:
        """The runtime half, for the three builders that can have one.

        ``_build_mcore_arm`` is where the arm name chooses the profile, the
        backend and the expected kernel, all three from one table. Replacing
        it records the name the builder really passes, which is stronger
        than reading the source: it survives a refactor that moves the
        literal.

        The titan builders cannot be called here -- each constructs a
        torchtitan attention module and runs a guard forward -- so they keep
        the source-level checks above.
        """
        recorded = []

        def spy(arm, shape, workload, inputs):
            recorded.append((arm, shape, workload, inputs))
            return "built"

        original = attention_core._build_mcore_arm
        attention_core._build_mcore_arm = spy
        try:
            for builder, (arm, _) in self.WIRING.items():
                if not arm.startswith("mcore/"):
                    continue
                with self.subTest(builder=builder):
                    recorded.clear()
                    result = getattr(attention_core, builder)(
                        TINY, TINY_WORKLOAD, "inputs-sentinel"
                    )
                    self.assertEqual(result, "built")
                    self.assertEqual(len(recorded), 1)
                    self.assertEqual(recorded[0][0], arm)
                    # The builder forwards what it was given, unchanged.
                    self.assertEqual(
                        recorded[0][1:], (TINY, TINY_WORKLOAD, "inputs-sentinel")
                    )
        finally:
            attention_core._build_mcore_arm = original

    def test_the_flex_flash_builder_still_asks_for_the_flash_backend(
        self,
    ) -> None:
        """Without the kernel option the arm IS the baseline.

        ``titan`` and ``titan/flex_flash`` are the same module with the same
        mask. The lowering is the only difference, and one dict carries it.
        """
        strings = self._strings("build_attention_core_titan_flex_flash")
        self.assertIn("BACKEND", strings)
        self.assertIn("FLASH", strings)
        self.assertNotIn(
            "BACKEND", self._strings("build_attention_core_titan")
        )


class ProfileDeliveryTests(unittest.TestCase):
    def test_build_model_resolves_the_attention_backend_field(self) -> None:
        """The profile's name must become megatron's enum, or nothing changes.

        ``mcore_profiles`` keeps the field as a name so the registry stays
        torch-free, and ``build_model`` is the only place that turns it into
        an ``AttnBackend`` member. Drop that step and all three megatron arms
        build at ``AttnBackend.auto``, which on this host resolves to the
        same cuDNN kernel the anchor runs -- three labels, one kernel, every
        gate green. A mutation reviewer showed no test covered it.

        ``build_model`` needs a GPU, so this reads the resolution rather than
        running it.
        """
        from benchmarks.models.piper_qwen3 import megatron_model

        tree = ast.parse(Path(megatron_model.__file__).read_text())
        loops = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.For)
            and isinstance(node.iter, ast.Name)
            and node.iter.id == "ATTENTION_BACKEND_FIELDS"
        ]
        self.assertEqual(len(loops), 1, "the resolution loop is gone")
        assigned = [
            node
            for node in ast.walk(loops[0])
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Name)
                and target.value.id == "kwargs"
                for target in node.targets
            )
        ]
        self.assertEqual(
            len(assigned), 1, "the loop no longer writes back into kwargs"
        )
        self.assertIn(
            "attention_backends",
            {
                node.id
                for node in ast.walk(assigned[0])
                if isinstance(node, ast.Name)
            },
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
