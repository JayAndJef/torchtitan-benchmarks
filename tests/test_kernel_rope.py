"""CPU tests for the ``rope`` cross-engine kernel scenario.

The scenario needs a GPU, megatron-core and TransformerEngine to measure
anything. It does not need any of the three to be *wrong*, and the five
failures that would ship a wrong number are all reachable on a CPU:

* the packing drifts between its two spellings, so titan rotates by one set of
  positions and megatron by another;
* the megatron form stops being a view of the titan form, so the two arms read
  different bytes and the ratio compares two inputs;
* the fp64 reference is wrong, so every gate passes against a wrong truth;
* an arm that silently degrades to the stock path is built without the marker
  guard that is the only thing able to catch it;
* the ``no_rope_fusion`` delta stops being a delta, so the arm measures
  ``base`` under another label.

The tests below cover those five. They do not cover the megatron arms' own
build, which needs a device, and they compile nothing: ``_titan_arm`` takes
the module already wrapped, so a test can hand it an eager one and exercise
the same closures without an Inductor compile.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.kernel.operations.rope import (
    FALLBACK_MARKERS,
    HELION_MARKER,
    MCORE_BASE_ARM,
    MCORE_NO_ROPE_FUSION_ARM,
    NO_ROPE_FUSION_PROFILE,
    OUTPUTS,
    TE_MARKER,
    TITAN_ARM,
    TITAN_HELION_ARM,
    TITAN_TE_ARM,
    _assert_rope_config,
    _blnh,
    _check_thd_reference_path,
    _rope_angles_fp64,
    _rope_arm,
    _rotate_half,
    _thd,
    _titan_arm,
    _titan_module,
    rope_inputs,
    rope_reference,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE, FUSION_FIELDS
from benchmarks.models.piper_qwen3.shape import PiperShape

# head_dim stays 64, the real value, because it is the width the rotation
# turns and the only geometry either engine reads here. Everything else
# shrinks: 4 query heads over 2 kv groups keeps q and k different sizes, which
# is why both closures rotate a pair rather than one tensor twice.
TINY = PiperShape(name="tiny", dim=256, n_layers=1, vocab_size=64)
# seq_len 32 gives _packed_positions a low of 2 and a high of 16, so a row
# holds several documents and the packing is exercised rather than degenerate.
TINY_WORKLOAD = KernelWorkload(batch=2, seq_len=32)

# The gate the scenario declaration carries, applied to all four outputs.
GATE = 2e-2


def cpu_inputs(seed: int = 0):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return rope_inputs(TINY, TINY_WORKLOAD, torch.device("cpu"), generator)


def rel_l2(actual: torch.Tensor, expected: torch.Tensor) -> float:
    difference = (actual.double() - expected.double()).norm()
    return float(difference / expected.double().norm())


class InputsTest(unittest.TestCase):
    def test_the_megatron_form_is_a_view_of_the_titan_form(self):
        """Both engines must read one q and one k, from one allocation.

        This is the claim that lets the scenario say it charges no layout
        conversion to either side. A copy would be a second allocation with a
        different cache history, and a separate draw would make the ratio a
        comparison of two inputs.
        """
        inputs = cpu_inputs()
        for titan, mcore in (
            (inputs.q_BLNH, inputs.q_THD),
            (inputs.k_BLNH, inputs.k_THD),
            (inputs.gq_BLNH, inputs.gq_THD),
            (inputs.gk_BLNH, inputs.gk_THD),
        ):
            with self.subTest(tensor=tuple(titan.shape)):
                self.assertEqual(titan.data_ptr(), mcore.data_ptr())
                self.assertTrue(
                    torch.equal(
                        _blnh(
                            mcore, TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len
                        ),
                        titan,
                    )
                )

    def test_every_timed_tensor_is_contiguous(self):
        """Neither engine may pay for a copy inside a timed closure."""
        inputs = cpu_inputs()
        for name in (
            "q_BLNH",
            "k_BLNH",
            "gq_BLNH",
            "gk_BLNH",
            "q_THD",
            "k_THD",
            "gq_THD",
            "gk_THD",
            "positions",
            "cu_seqlens",
        ):
            with self.subTest(tensor=name):
                self.assertTrue(getattr(inputs, name).is_contiguous())

    def test_shapes_follow_the_geometry(self):
        inputs = cpu_inputs()
        batch, seq = TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len
        self.assertEqual(
            tuple(inputs.q_BLNH.shape),
            (batch, seq, TINY.n_heads, TINY.head_dim),
        )
        self.assertEqual(
            tuple(inputs.k_BLNH.shape),
            (batch, seq, TINY.n_kv_heads, TINY.head_dim),
        )
        self.assertEqual(
            tuple(inputs.q_THD.shape),
            (batch * seq, TINY.n_heads, TINY.head_dim),
        )
        self.assertEqual(inputs.q_BLNH.dtype, torch.bfloat16)
        self.assertNotEqual(TINY.n_heads, TINY.n_kv_heads)

    def test_positions_are_int64_as_the_te_kernel_requires(self):
        """``te_rope_standalone.cu:236`` declares ``const int64_t *positions``.

        ``TECosSinRoPE`` falls back to the numerically correct stock path on
        any other width, which no correctness gate can see. The attention
        scenario's twin builder uses int32 and is right to: it feeds flex and
        THD metadata, not this kernel.
        """
        self.assertEqual(cpu_inputs().positions.dtype, torch.int64)

    def test_positions_restart_at_every_document_start(self):
        """A position is either 0 or one past its predecessor."""
        positions = cpu_inputs().positions
        self.assertEqual(
            tuple(positions.shape),
            (TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len),
        )
        for row in positions.tolist():
            self.assertEqual(row[0], 0)
            for previous, current in zip(row, row[1:]):
                self.assertIn(current, (0, previous + 1))

    def test_the_packing_holds_more_than_one_document_per_row(self):
        """Negative control: one document per row would test nothing.

        Both THD paths exist to address a packed batch. A degenerate packing
        would let a wrong ``cu_seqlens`` pass every assertion here.
        """
        inputs = cpu_inputs()
        self.assertGreater(inputs.num_documents, TINY_WORKLOAD.batch)

    def test_cu_seqlens_is_the_other_spelling_of_the_same_packing(self):
        """Derived independently, by walking the positions in Python.

        ``_cu_seqlens`` uses a tensor expression, so a reimplementation here
        is a second opinion rather than a restatement. Megatron never sees
        ``positions``: it derives every token's rotation angle from these
        offsets, so a disagreement rotates the two engines differently while
        both stay internally consistent.
        """
        inputs = cpu_inputs()
        lengths, current = [], 0
        for position in inputs.positions.reshape(-1).tolist():
            if position == 0 and current:
                lengths.append(current)
                current = 0
            current += 1
        lengths.append(current)
        expected, running = [0], 0
        for length in lengths:
            running += length
            expected.append(running)
        self.assertEqual(inputs.cu_seqlens.tolist(), expected)
        self.assertEqual(inputs.cu_seqlens.dtype, torch.int32)
        self.assertEqual(
            int(inputs.cu_seqlens[-1]),
            TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len,
        )
        self.assertEqual(inputs.num_documents, len(lengths))

    def test_cu_seqlens_carries_no_padding(self):
        """Every segment is non-empty, unlike the e2e driver's padded form.

        ``benchmarks/e2e/megatron/train.py:170-186`` pads with trailing
        ``total_tokens`` entries so graph capture sees one shape across steps.
        A kernel run times one fixed batch, and each padded entry would add a
        zero-length iteration that only ``mcore/no_rope_fusion`` pays for.
        """
        cu_seqlens = cpu_inputs().cu_seqlens
        segments = cu_seqlens[1:] - cu_seqlens[:-1]
        self.assertTrue(bool((segments > 0).all()))

    def test_the_rotary_length_matches_what_megatron_would_ask_for(self):
        """``get_rotary_seq_len`` returns ``max_seqlen_q`` under packing.

        ``rotary_pos_embedding.py:238-241`` returns
        ``max(max_seqlen_q, max_seqlen_kv)``, and
        ``benchmarks/e2e/megatron/train.py:276-277`` pins both to ``seq_len``.
        No document is longer than a row, so the table covers every position
        either engine looks up.
        """
        inputs = cpu_inputs()
        self.assertEqual(inputs.rotary_seq_len, TINY_WORKLOAD.seq_len)
        self.assertLessEqual(
            int(inputs.positions.max()) + 1, inputs.rotary_seq_len
        )

    def test_the_seed_decides_the_inputs_and_the_packing(self):
        one, two = cpu_inputs(seed=7), cpu_inputs(seed=7)
        self.assertTrue(torch.equal(one.q_BLNH, two.q_BLNH))
        self.assertTrue(torch.equal(one.positions, two.positions))
        self.assertTrue(torch.equal(one.cu_seqlens, two.cu_seqlens))
        other = cpu_inputs(seed=8)
        self.assertFalse(torch.equal(one.q_BLNH, other.q_BLNH))
        self.assertFalse(torch.equal(one.positions, other.positions))

    def test_bytes_moved_counts_the_forward_traffic(self):
        inputs = cpu_inputs()
        expected = 2 * (inputs.q_BLNH.numel() + inputs.k_BLNH.numel()) * 2
        self.assertEqual(inputs.qk_bytes, expected)


class ReferenceTest(unittest.TestCase):
    def test_the_angle_table_matches_megatrons_own_construction(self):
        """One expression, two readers, and the fp64 truth is the third.

        Reimplemented from ``rotary_pos_embedding.py:79-81`` and
        ``get_emb`` at ``:165-170``. If either engine's table moved, this
        scenario would be rotating by an angle no arm computes.
        """
        table = _rope_angles_fp64(TINY, TINY_WORKLOAD, torch.device("cpu"))
        inv_freq = 1.0 / (
            TINY.rope_theta
            ** (
                torch.arange(0, TINY.head_dim, 2, dtype=torch.float64)
                / TINY.head_dim
            )
        )
        steps = torch.arange(TINY_WORKLOAD.seq_len, dtype=torch.float64)
        freqs = torch.outer(steps, inv_freq)
        expected = torch.cat((freqs, freqs), dim=-1)
        self.assertEqual(
            tuple(table.shape), (TINY_WORKLOAD.seq_len, TINY.head_dim)
        )
        self.assertTrue(torch.allclose(table, expected, rtol=0, atol=0))

    def test_forward_matches_a_per_token_rotation(self):
        """An independent implementation, so the truth is not self-checked."""
        inputs = cpu_inputs()
        reference = rope_reference(TINY, TINY_WORKLOAD, inputs)
        table = _rope_angles_fp64(TINY, TINY_WORKLOAD, torch.device("cpu"))
        half = TINY.head_dim // 2
        for tag, x in (("q", inputs.q_BLNH), ("k", inputs.k_BLNH)):
            with self.subTest(tensor=tag):
                expected = torch.empty(x.shape, dtype=torch.float64)
                for b in range(TINY_WORKLOAD.batch):
                    for s in range(TINY_WORKLOAD.seq_len):
                        angles = table[int(inputs.positions[b, s])]
                        cos, sin = angles.cos(), angles.sin()
                        row = x[b, s].double()
                        rotated = torch.cat(
                            (-row[..., half:], row[..., :half]), dim=-1
                        )
                        expected[b, s] = row * cos + rotated * sin
                self.assertLess(rel_l2(reference[f"{tag}_out"], expected), 1e-12)

    def test_backward_matches_torch_autograd(self):
        """The closed form is the adjoint, not an approximation of it."""
        inputs = cpu_inputs()
        reference = rope_reference(TINY, TINY_WORKLOAD, inputs)
        table = _rope_angles_fp64(TINY, TINY_WORKLOAD, torch.device("cpu"))
        angles = table[inputs.positions].unsqueeze(2)
        cos, sin = angles.cos(), angles.sin()
        for tag, x, grad in (
            ("q", inputs.q_BLNH, inputs.gq_BLNH),
            ("k", inputs.k_BLNH, inputs.gk_BLNH),
        ):
            with self.subTest(tensor=tag):
                leaf = x.double().detach().requires_grad_()
                out = leaf * cos + _rotate_half(leaf) * sin
                torch.autograd.backward(out, grad.double())
                self.assertLess(
                    rel_l2(reference[f"d{tag}"], leaf.grad), 1e-12
                )

    def test_the_reference_returns_every_declared_output(self):
        """The scenario declaration repeats these names as literals."""
        reference = rope_reference(TINY, TINY_WORKLOAD, cpu_inputs())
        self.assertEqual(set(reference), set(OUTPUTS))
        self.assertEqual(
            tuple(reference["q_out"].shape), tuple(cpu_inputs().q_BLNH.shape)
        )

    def test_the_reference_reads_the_packed_positions(self):
        """Negative control: an arange truth would gate the wrong rotation.

        Both THD paths address a packed batch, and the whole scenario turns on
        that. A reference that ignored ``positions`` would agree with an arm
        that ignored them too, and the pair would pass every gate.
        """
        inputs = cpu_inputs()
        reference = rope_reference(TINY, TINY_WORKLOAD, inputs)
        table = _rope_angles_fp64(TINY, TINY_WORKLOAD, torch.device("cpu"))
        arange = (
            torch.arange(TINY_WORKLOAD.seq_len)
            .unsqueeze(0)
            .expand(TINY_WORKLOAD.batch, -1)
        )
        angles = table[arange].unsqueeze(2)
        unpacked = inputs.q_BLNH.double() * angles.cos() + _rotate_half(
            inputs.q_BLNH.double()
        ) * angles.sin()
        self.assertGreater(rel_l2(reference["q_out"], unpacked), 0.1)


class TitanArmTest(unittest.TestCase):
    def test_the_builder_uses_the_production_rope_config_node(self):
        """The arm must build the RoPE the piper config puts on a block.

        The node is a pure function of the shape, so the builder constructs
        the same one the model registry constructs. This test is what keeps
        that true: a theta, a max_seq_len or a scaling mode that moves in
        torchtitan fails here rather than in a published ratio.
        """
        from torchtitan.models.common.rope import CosSinRoPE

        from benchmarks.models.piper_qwen3 import config_registry
        from benchmarks.models.piper_qwen3.shape import shape_by_name

        shape = shape_by_name("normal")
        trainer = config_registry.qwen3_piper_1b(size="normal")
        production = trainer.model_spec.model.layers[0].attention.rope
        self.assertEqual(
            production,
            CosSinRoPE.Config(
                dim=shape.head_dim,
                max_seq_len=shape.max_seq_len,
                theta=shape.rope_theta,
            ),
        )

    def test_the_arm_matches_the_reference_in_bf16(self):
        """The real arithmetic, uncompiled, against the fp64 truth.

        ``build_rope_titan`` wraps the module in ``torch.compile``. Inductor
        changes the kernels and not the mathematics, so running it eager
        checks the positions, the cache and the gate value on a CPU.
        """
        from torchtitan.models.common.rope import CosSinRoPE

        inputs = cpu_inputs()
        module = _titan_module(TINY, inputs, CosSinRoPE)
        arm = _titan_arm(TITAN_ARM, module, inputs)
        measured = arm.correctness_outputs()
        reference = rope_reference(TINY, TINY_WORKLOAD, inputs)
        for name in OUTPUTS:
            with self.subTest(output=name):
                self.assertLess(rel_l2(measured[name], reference[name]), GATE)

    def test_the_arm_exposes_the_declared_modes(self):
        """``_seeded_build`` compares this set against the declaration."""
        from torchtitan.models.common.rope import CosSinRoPE

        inputs = cpu_inputs()
        arm = _titan_arm(
            TITAN_ARM, _titan_module(TINY, inputs, CosSinRoPE), inputs
        )
        self.assertEqual(arm.name, TITAN_ARM)
        self.assertEqual(set(arm.calls), {"forward", "backward"})
        self.assertEqual(arm.bytes_moved, inputs.qk_bytes)
        self.assertEqual(set(arm.correctness_outputs()), set(OUTPUTS))

    def test_the_backward_mode_reruns_the_retained_graph(self):
        """Backward is repeatable, which is what the timing pass needs.

        ``retain_graph=True`` is what lets a burst re-run the same backward.
        Neither engine clears its saved tensors here, so a second call must
        produce the same gradients rather than raise.
        """
        from torchtitan.models.common.rope import CosSinRoPE

        inputs = cpu_inputs()
        arm = _titan_arm(
            TITAN_ARM, _titan_module(TINY, inputs, CosSinRoPE), inputs
        )
        first = arm.correctness_outputs()["dq"].clone()
        arm.calls["backward"]()
        second = arm.correctness_outputs()["dq"]
        self.assertTrue(torch.equal(first, second))


class MarkerGuardTest(unittest.TestCase):
    """The only guard able to catch a silent fallback, and its enforcement.

    ``HelionCosSinRoPE`` and ``TECosSinRoPE`` degrade to the *numerically
    correct* stock path when their eligibility checks fail, so a correctness
    gate cannot tell a fallback from a success. This roster moves both arms
    onto packed-document positions, which is exactly the kind of change that
    trips an eligibility check.
    """

    def test_both_fallback_arms_declare_the_kernel_they_must_show(self):
        self.assertEqual(FALLBACK_MARKERS[TITAN_HELION_ARM], HELION_MARKER)
        self.assertEqual(FALLBACK_MARKERS[TITAN_TE_ARM], TE_MARKER)
        self.assertEqual(HELION_MARKER, "_helion__rope_cos_sin_fwd")
        self.assertEqual(TE_MARKER, "fused_rope_forward_positions_kernel")

    def test_a_fallback_arm_without_its_marker_is_refused(self):
        from torchtitan.models.common.rope import CosSinRoPE

        inputs = cpu_inputs()
        module = _titan_module(TINY, inputs, CosSinRoPE)
        with self.assertRaises(ValueError):
            _titan_arm(TITAN_HELION_ARM, module, inputs)

    def test_a_fallback_arm_with_the_wrong_marker_is_refused(self):
        from torchtitan.models.common.rope import CosSinRoPE

        inputs = cpu_inputs()
        module = _titan_module(TINY, inputs, CosSinRoPE)
        with self.assertRaises(ValueError):
            _titan_arm(TITAN_TE_ARM, module, inputs, marker=HELION_MARKER)

    def test_the_stock_arm_needs_no_marker(self):
        """``CosSinRoPE`` has no fast path and therefore no fallback."""
        self.assertNotIn(TITAN_ARM, FALLBACK_MARKERS)


class McoreProfileTest(unittest.TestCase):
    def test_no_rope_fusion_is_a_real_delta(self):
        """The base really sets the flag, so the arm really flips one.

        ``TransformerConfig``'s dataclass default is ``False``
        (``transformer_config.py:499``), so an arm that merely failed to set
        the flag would look identical to this one and measure nothing new.
        """
        self.assertIs(BASE.config_overrides["apply_rope_fusion"], True)
        self.assertIs(
            NO_ROPE_FUSION_PROFILE.config_overrides["apply_rope_fusion"], False
        )
        differences = {
            key
            for key in BASE.config_overrides
            if BASE.config_overrides[key]
            != NO_ROPE_FUSION_PROFILE.config_overrides[key]
        }
        self.assertEqual(differences, {"apply_rope_fusion"})
        self.assertEqual(
            set(NO_ROPE_FUSION_PROFILE.config_overrides),
            set(BASE.config_overrides),
        )

    def test_the_flag_is_a_policed_fusion_field(self):
        """So the driver's declared-versus-built check covers the flip."""
        self.assertIn("apply_rope_fusion", FUSION_FIELDS)

    def test_the_config_guard_refuses_a_delta_that_did_not_take(self):
        """A profile flag that failed to reach the config is a wrong label."""
        config = _StubConfig(apply_rope_fusion=True)
        with self.assertRaises(RuntimeError):
            _assert_rope_config(config, NO_ROPE_FUSION_PROFILE, TINY)
        _assert_rope_config(config, BASE, TINY)

    def test_the_config_guard_refuses_a_rotation_titan_cannot_match(self):
        """Four fields that silently change what megatron computes."""
        for field, value in (
            ("rotary_interleaved", True),
            ("multi_latent_attention", True),
            ("mrope_section", [16, 24, 24]),
            ("kv_channels", TINY.head_dim * 2),
        ):
            with self.subTest(field=field):
                config = _StubConfig(**{field: value})
                with self.assertRaises(RuntimeError):
                    _assert_rope_config(config, BASE, TINY)

    def test_the_arm_names_are_the_ones_the_plan_spells(self):
        self.assertEqual(MCORE_BASE_ARM, "mcore/base")
        self.assertEqual(MCORE_NO_ROPE_FUSION_ARM, "mcore/no_rope_fusion")
        self.assertEqual(TITAN_ARM, "titan")
        self.assertEqual(TITAN_HELION_ARM, "titan/helion")
        self.assertEqual(TITAN_TE_ARM, "titan/te")


class _StubConfig:
    """The five fields ``_assert_rope_config`` reads, at their good values.

    A real ``TransformerConfig`` needs megatron on ``sys.path``, which a CPU
    test must not require. The guard reads attributes and nothing else, so a
    stub exercises every branch of it.
    """

    def __init__(self, **overrides):
        self.apply_rope_fusion = True
        self.rotary_interleaved = False
        self.multi_latent_attention = False
        self.mrope_section = None
        self.kv_channels = TINY.head_dim
        for name, value in overrides.items():
            setattr(self, name, value)


class ThdReferenceBranchTest(unittest.TestCase):
    """``_apply_rotary_pos_emb_thd`` picks its branch on the data.

    ``has_packed_freqs = freqs.size(0) == total_seqlen``
    (``rope_utils.py:225``) compares the table's ``seq_len`` rows against the
    ``batch * seq_len`` packed tokens, so it is True exactly at ``batch == 1``
    -- and CASE 1 then rotates by each document's global offset, which is a
    different rotation from the one every other arm here computes.
    """

    def _inputs_at(self, batch: int):
        workload = KernelWorkload(batch=batch, seq_len=TINY_WORKLOAD.seq_len)
        generator = torch.Generator(device="cpu").manual_seed(0)
        return workload, rope_inputs(
            TINY, workload, torch.device("cpu"), generator
        )

    def _freqs_for(self, inputs):
        """A stand-in for megatron's table, at the row count the arm asks for."""
        return torch.zeros(inputs.rotary_seq_len, 1, 1, TINY.head_dim)

    def test_only_batch_one_collides_the_table_with_the_token_count(self):
        for batch, collides in ((1, True), (2, False), (4, False)):
            with self.subTest(batch=batch):
                _, inputs = self._inputs_at(batch)
                self.assertIs(
                    inputs.rotary_seq_len == int(inputs.cu_seqlens[-1]),
                    collides,
                )

    def test_the_guard_refuses_the_case_one_branch(self):
        _, inputs = self._inputs_at(1)
        with self.assertRaises(RuntimeError) as caught:
            _check_thd_reference_path(inputs, self._freqs_for(inputs))
        self.assertIn("CASE 1", str(caught.exception))
        self.assertIn("batch > 1", str(caught.exception))

    def test_the_guard_passes_the_case_two_branch(self):
        _, inputs = self._inputs_at(2)
        _check_thd_reference_path(inputs, self._freqs_for(inputs))


class TimedGraphTest(unittest.TestCase):
    """What the timed closures run, and what the gate reads.

    Both are checked through ``_rope_arm`` with a stub ``call``, because the
    properties are the arm wrapper's and not any engine's. A real module
    would hide them: its output is the same tensor whichever graph produced
    it.
    """

    def _probe_arm(self, seen: list, scales: list):
        inputs = cpu_inputs()

        def call(q: torch.Tensor, k: torch.Tensor):
            seen.append((q.requires_grad, k.requires_grad))
            scales.append(len(scales) + 1)
            return q * scales[-1], k * scales[-1]

        return inputs, _rope_arm(
            "probe",
            inputs,
            q=inputs.q_BLNH,
            k=inputs.k_BLNH,
            gq=inputs.gq_BLNH,
            gk=inputs.gk_BLNH,
            call=call,
            canonical=lambda tensor: tensor,
        )

    def test_the_timed_forward_runs_on_grad_leaves(self):
        """``requires_grad`` is a Dynamo guard, so a non-grad forward would
        time an inference graph production never runs."""
        seen, scales = [], []
        _, arm = self._probe_arm(seen, scales)
        seen.clear()
        arm.calls["forward"]()
        self.assertEqual(seen, [(True, True)])

    def test_the_gate_reads_the_graph_the_backward_ran_on(self):
        """A second forward call would open a hole no gate could see.

        RoPE's adjoint depends on the angles alone, not on the forward
        outputs, so outputs taken from one graph and gradients from another
        would let a wrong forward pass every check. The stub returns a
        different scale on every call, so a fresh call is visible here.
        """
        seen, scales = [], []
        inputs, arm = self._probe_arm(seen, scales)
        self.assertEqual(scales, [1])
        measured = arm.correctness_outputs()
        self.assertEqual(scales, [1])
        self.assertTrue(torch.equal(measured["q_out"], inputs.q_BLNH))
        self.assertTrue(torch.equal(measured["dq"], inputs.gq_BLNH))


class ThdViewTest(unittest.TestCase):
    def test_the_two_views_are_inverses(self):
        original = torch.arange(2 * 3 * 4 * 5).reshape(2, 3, 4, 5)
        self.assertTrue(
            torch.equal(_blnh(_thd(original), 2, 3), original)
        )
        self.assertEqual(tuple(_thd(original).shape), (6, 4, 5))


if __name__ == "__main__":
    unittest.main()
