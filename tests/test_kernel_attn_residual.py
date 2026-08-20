"""CPU tests for the ``attn_residual`` kernel scenario.

Everything here runs without a GPU, without megatron and without
TransformerEngine. Four things are testable that way, and they are the four a
wrong scenario gets wrong quietly:

* the profile delta is **one** field, so the two mcore arms differ in the
  thing the published row names and in nothing else;
* the fp64 reference is the plain sum both engines compute, and its two
  gradients are the output gradient;
* the shared timed closures behave, over a stand-in ``bind`` in place of the
  two real call sites; and
* every build guard raises, exercised with stand-in callables and a stand-in
  layer.

The mcore side itself needs a CUDA device, a process group and TE, so no test
here touches it. What a CPU can still check about that side is the guards:
``_require_bda_dispatch`` is what stops an unfused arm that silently resolved
to the fused callable, and ``_require_mcore_layer_contract`` is what stops a
layer whose configuration makes this cut a different cut.
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.kernel.operations.attn_residual import (
    _is_torch_compiled,
    _require_bda_dispatch,
    _require_exact_residual_add,
    _require_mcore_layer_contract,
    _residual_arm,
    attn_residual_inputs,
    attn_residual_reference,
    AttnResidualInputs,
    build_attn_residual_copy_floor,
    COPY_FLOOR_ARM,
    MCORE_BASE_ARM,
    MCORE_NO_FUSION_ARM,
    NO_BIAS_DROPOUT_FUSION_PROFILE,
    TITAN_ARM,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE, FUSION_FIELDS
from benchmarks.models.piper_qwen3.shape import PiperShape, shape_by_name

# Small enough to run in milliseconds, wide enough that a row is a real row.
TINY = PiperShape(name="tiny", dim=256, n_layers=2, vocab_size=64)
TINY_WORKLOAD = KernelWorkload(batch=2, seq_len=16)


def _inputs(shape=TINY, workload=TINY_WORKLOAD, seed=0) -> AttnResidualInputs:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return attn_residual_inputs(shape, workload, torch.device("cpu"), generator)


def _bind_add(attn_leaf: torch.Tensor, residual_leaf: torch.Tensor):
    """The stand-in for both engines' call sites: residual on the left."""

    def call() -> torch.Tensor:
        return residual_leaf + attn_leaf

    return call


class ProfileDeltaTests(unittest.TestCase):
    """One field, or the published row does not name what it measures."""

    def test_the_variant_changes_exactly_one_field(self) -> None:
        changed = {
            key: value
            for key, value in NO_BIAS_DROPOUT_FUSION_PROFILE.config_overrides.items()
            if BASE.config_overrides.get(key) != value
        }
        self.assertEqual(changed, {"bias_dropout_fusion": False})

    def test_the_variant_drops_no_field_of_the_base(self) -> None:
        """A delta adds and replaces; it must never remove a fusion flag."""
        self.assertEqual(
            set(NO_BIAS_DROPOUT_FUSION_PROFILE.config_overrides),
            set(BASE.config_overrides),
        )

    def test_the_base_really_turns_the_fusion_on(self) -> None:
        """Off is the deviation, because megatron's argparse layer sets it."""
        self.assertIs(BASE.config_overrides["bias_dropout_fusion"], True)

    def test_the_flag_is_a_declared_fusion_field(self) -> None:
        """So the megatron driver would assert it in both directions."""
        self.assertIn("bias_dropout_fusion", FUSION_FIELDS)

    def test_the_base_makes_the_call_site_a_plain_add(self) -> None:
        """The three profile values the scenario's whole claim rests on.

        With these, ``_bias_dropout_add_func`` takes its no-bias branch and
        ``F.dropout(p=0.0)`` returns its input, so megatron computes
        ``residual + x`` and nothing else.
        """
        self.assertEqual(BASE.config_overrides["hidden_dropout"], 0.0)
        self.assertIs(BASE.config_overrides["add_bias_linear"], False)
        self.assertNotIn("fp32_residual_connection", BASE.config_overrides)

    def test_the_variant_keeps_those_three_values(self) -> None:
        overrides = NO_BIAS_DROPOUT_FUSION_PROFILE.config_overrides
        self.assertEqual(overrides["hidden_dropout"], 0.0)
        self.assertIs(overrides["add_bias_linear"], False)
        self.assertNotIn("fp32_residual_connection", overrides)


class CopyFloorTests(unittest.TestCase):
    """The bandwidth floor, which is what makes the published row readable.

    Both mcore arms run one bf16 add on the same tensors, so the ratio
    between them is host dispatch and nothing else. The floor is the second
    reading that says so, independently of the one-sided ``--burst``
    residual test.
    """

    def _floor(self, inputs: AttnResidualInputs):
        return build_attn_residual_copy_floor(TINY, TINY_WORKLOAD, inputs)

    def test_the_floor_declares_forward_only(self) -> None:
        """A floor for the backward traffic would be an invention."""
        self.assertEqual(sorted(self._floor(_inputs()).calls), ["forward"])

    def test_the_floor_is_named_for_the_registry(self) -> None:
        self.assertEqual(self._floor(_inputs()).name, COPY_FLOOR_ARM)
        self.assertEqual(COPY_FLOOR_ARM, "copy_floor")

    def test_the_floor_declares_the_copy_traffic(self) -> None:
        """Its own bytes, not the add's: the GB/s column is the floor's."""
        inputs = _inputs()
        self.assertEqual(self._floor(inputs).bytes_moved, inputs.copy_bytes)

    def test_the_copy_traffic_is_one_read_and_one_write(self) -> None:
        inputs = _inputs()
        expected = 2 * inputs.attn_out.numel() * inputs.attn_out.element_size()
        self.assertEqual(inputs.copy_bytes, expected)

    def test_the_add_moves_one_and_a_half_times_the_floor(self) -> None:
        """The stated correction, pinned so the docstring cannot drift.

        A copy reads one tensor and writes one. The add reads two and writes
        one, so the floor understates the device by a third and the x_floor
        column overstates the distance by the reciprocal.
        """
        inputs = _inputs()
        element = inputs.attn_out.numel() * inputs.attn_out.element_size()
        self.assertEqual(3 * element, 1.5 * inputs.copy_bytes)

    def test_the_floor_moves_the_bytes_it_declares(self) -> None:
        """It runs one copy of one tensor, and the copy really happens."""
        inputs = _inputs()
        floor = self._floor(inputs)
        floor.calls["forward"]()
        self.assertEqual(floor.correctness_outputs(), {})


class ArmNameTests(unittest.TestCase):
    def test_the_names_match_the_plan_roster(self) -> None:
        self.assertEqual(MCORE_BASE_ARM, "mcore/base")
        self.assertEqual(MCORE_NO_FUSION_ARM, "mcore/no_bias_dropout_fusion")
        self.assertEqual(TITAN_ARM, "titan")


class InputsTests(unittest.TestCase):
    def test_the_three_tensors_share_one_canonical_shape(self) -> None:
        inputs = _inputs()
        expected = (TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len, TINY.dim)
        for name in ("attn_out", "residual", "grad_out"):
            with self.subTest(tensor=name):
                tensor = getattr(inputs, name)
                self.assertEqual(tuple(tensor.shape), expected)
                self.assertEqual(tensor.dtype, torch.bfloat16)
                self.assertTrue(tensor.is_contiguous())

    def test_the_two_addends_are_drawn_independently(self) -> None:
        """A shared draw would let an arm that dropped one operand pass."""
        inputs = _inputs()
        self.assertFalse(torch.equal(inputs.attn_out, inputs.residual))
        self.assertFalse(torch.equal(inputs.attn_out, inputs.grad_out))
        self.assertFalse(torch.equal(inputs.residual, inputs.grad_out))

    def test_one_seed_rebuilds_the_inputs_bit_identically(self) -> None:
        """Every worker rebuilds these, and the gates compare across workers."""
        first, second = _inputs(seed=7), _inputs(seed=7)
        self.assertTrue(torch.equal(first.attn_out, second.attn_out))
        self.assertTrue(torch.equal(first.residual, second.residual))
        self.assertTrue(torch.equal(first.grad_out, second.grad_out))

    def test_the_canonical_tensor_reshapes_to_the_mcore_thd_view(self) -> None:
        """The mcore arms take a free view, never a copy."""
        inputs = _inputs()
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        view = inputs.attn_out.reshape(tokens, 1, TINY.dim)
        self.assertEqual(view.data_ptr(), inputs.attn_out.data_ptr())


class ReferenceTests(unittest.TestCase):
    def test_the_reference_is_the_fp64_sum(self) -> None:
        inputs = _inputs()
        reference = attn_residual_reference(TINY, TINY_WORKLOAD, inputs)
        expected = inputs.residual.double() + inputs.attn_out.double()
        self.assertTrue(torch.equal(reference["out"], expected))

    def test_both_gradients_are_the_output_gradient(self) -> None:
        """An add routes its output gradient to both inputs unchanged."""
        inputs = _inputs()
        reference = attn_residual_reference(TINY, TINY_WORKLOAD, inputs)
        grad = inputs.grad_out.double()
        self.assertTrue(torch.equal(reference["attn_out_grad"], grad))
        self.assertTrue(torch.equal(reference["residual_grad"], grad))

    def test_the_reference_names_exactly_the_gated_outputs(self) -> None:
        reference = attn_residual_reference(TINY, TINY_WORKLOAD, _inputs())
        self.assertEqual(
            sorted(reference), ["attn_out_grad", "out", "residual_grad"]
        )


class ArmClosureTests(unittest.TestCase):
    """The closures both engines share, over a stand-in call site."""

    def _arm(self, inputs: AttnResidualInputs):
        return _residual_arm(
            name="stand_in",
            bind=_bind_add,
            attn_native=inputs.attn_out,
            residual_native=inputs.residual,
            grad_native=inputs.grad_out,
            canonical=tuple(inputs.attn_out.shape),
        )

    def test_the_arm_declares_forward_and_forward_backward_only(self) -> None:
        """An isolated backward would time autograd's dispatch and no kernel."""
        arm = self._arm(_inputs())
        self.assertEqual(sorted(arm.calls), ["forward", "forward_backward"])

    def test_the_arm_declares_no_bytes_moved(self) -> None:
        """One count cannot describe both of the declared modes."""
        self.assertIsNone(self._arm(_inputs()).bytes_moved)

    def test_the_named_outputs_match_the_fp64_reference(self) -> None:
        inputs = _inputs()
        outputs = self._arm(inputs).correctness_outputs()
        reference = attn_residual_reference(TINY, TINY_WORKLOAD, inputs)

        self.assertEqual(
            sorted(outputs), ["attn_out_grad", "out", "residual_grad"]
        )
        for name in outputs:
            with self.subTest(output=name):
                delta = (outputs[name].double() - reference[name]).norm()
                self.assertLess(
                    (delta / reference[name].double().norm()).item(), 2e-2
                )

    def test_a_repeated_round_trip_does_not_accumulate_a_gradient(self) -> None:
        """Every timed call must measure one backward, not a growing sum."""
        inputs = _inputs()
        arm = self._arm(inputs)
        round_trip = arm.calls["forward_backward"]
        round_trip()
        first = arm.correctness_outputs()["attn_out_grad"].clone()
        for _ in range(3):
            round_trip()
        second = arm.correctness_outputs()["attn_out_grad"]
        self.assertTrue(torch.equal(first, second))

    def test_the_forward_closure_leaves_the_check_leaves_alone(self) -> None:
        """Three leaf pairs, so one mode's graph never reaches another's."""
        inputs = _inputs()
        arm = self._arm(inputs)
        arm.calls["forward"]()
        outputs = arm.correctness_outputs()
        self.assertTrue(
            torch.equal(
                outputs["out"], inputs.residual + inputs.attn_out
            )
        )

    def test_the_outputs_are_canonicalized_from_the_mcore_view(self) -> None:
        """A gate compares two arms element by element, in one shape."""
        inputs = _inputs()
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        canonical = tuple(inputs.attn_out.shape)
        arm = _residual_arm(
            name="stand_in_thd",
            bind=_bind_add,
            attn_native=inputs.attn_out.reshape(tokens, 1, TINY.dim),
            residual_native=inputs.residual.reshape(tokens, 1, TINY.dim),
            grad_native=inputs.grad_out.reshape(tokens, 1, TINY.dim),
            canonical=canonical,
        )
        for name, value in arm.correctness_outputs().items():
            with self.subTest(output=name):
                self.assertEqual(tuple(value.shape), canonical)

    def test_a_detached_operand_is_named_rather_than_dereferenced(self) -> None:
        def bind_detached(attn_leaf, residual_leaf):
            return lambda: residual_leaf + attn_leaf.detach()

        inputs = _inputs()
        arm = _residual_arm(
            name="stand_in_detached",
            bind=bind_detached,
            attn_native=inputs.attn_out,
            residual_native=inputs.residual,
            grad_native=inputs.grad_out,
            canonical=tuple(inputs.attn_out.shape),
        )
        with self.assertRaisesRegex(RuntimeError, "attn_out_grad"):
            arm.correctness_outputs()


class ExactAddGuardTests(unittest.TestCase):
    """The guard that turns the scenario's central claim into a check."""

    def setUp(self) -> None:
        self.inputs = _inputs()

    def _check(self, call) -> None:
        _require_exact_residual_add(
            call, self.inputs.attn_out, self.inputs.residual, "stand_in"
        )

    def test_a_plain_residual_add_passes(self) -> None:
        self._check(_bind_add(self.inputs.attn_out, self.inputs.residual))

    def test_a_second_addend_is_refused(self) -> None:
        """What the bias branch of _bias_dropout_add_func would compute."""
        bias = torch.ones_like(self.inputs.attn_out)
        with self.assertRaisesRegex(RuntimeError, "bitwise"):
            self._check(
                lambda: self.inputs.residual + self.inputs.attn_out + bias
            )

    def test_a_dropped_operand_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "bitwise"):
            self._check(lambda: self.inputs.residual.clone())

    def test_an_upcast_output_is_refused(self) -> None:
        """What fp32_residual_connection would produce."""
        with self.assertRaisesRegex(RuntimeError, "does not compute"):
            self._check(
                lambda: self.inputs.residual.float() + self.inputs.attn_out
            )

    def test_a_swapped_operand_order_is_NOT_refused(self) -> None:
        """The limit of this guard, pinned so nobody claims it checks order.

        IEEE-754 addition commutes bit for bit, so ``attn_out + residual``
        and ``residual + attn_out`` are the same value and ``torch.equal``
        accepts either. The operand order is read off the two call sites
        (``qwen3/model.py:60`` and ``fused_bias_dropout.py:58``) and nothing
        here is evidence about it.
        """
        self._check(lambda: self.inputs.attn_out + self.inputs.residual)

    def test_a_reshaped_output_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "does not compute"):
            self._check(
                lambda: (self.inputs.residual + self.inputs.attn_out).reshape(
                    -1, TINY.dim
                )
            )


def _compiled_stand_in():
    """A real ``torch.compile`` wrapper, for the mechanism guard."""

    def bias_dropout_add_fused_train(x_with_bias, residual, prob):
        return residual + x_with_bias[0]

    return torch.compile(bias_dropout_add_fused_train)


def _unfused_stand_in():
    """The shape ``bias_dropout_add_unfused`` returns: a fresh closure."""

    def bias_dropout_add_unfused(training):
        def _bias_dropout_add(x_with_bias, residual, prob):
            return residual + x_with_bias[0]

        return _bias_dropout_add

    return bias_dropout_add_unfused(True)


class DispatchGuardTests(unittest.TestCase):
    def test_torch_compile_is_recognized(self) -> None:
        self.assertTrue(_is_torch_compiled(_compiled_stand_in()))

    def test_a_plain_closure_is_not_recognized(self) -> None:
        self.assertFalse(_is_torch_compiled(_unfused_stand_in()))

    def test_the_fused_arm_accepts_the_compiled_callable(self) -> None:
        fused_train = _compiled_stand_in()
        notes = _require_bda_dispatch(
            arm=MCORE_BASE_ARM,
            fused=True,
            resolved=fused_train,
            fused_train=fused_train,
        )
        self.assertIs(notes["bias_dropout_fusion"], True)
        self.assertIs(notes["resolved_is_compiled"], True)
        # And no copy of KernelArm.compiled: the registry is the authority.
        self.assertNotIn("compiled", notes)

    def test_the_unfused_arm_accepts_the_plain_closure(self) -> None:
        notes = _require_bda_dispatch(
            arm=MCORE_NO_FUSION_ARM,
            fused=False,
            resolved=_unfused_stand_in(),
            fused_train=_compiled_stand_in(),
        )
        self.assertIs(notes["bias_dropout_fusion"], False)
        self.assertIs(notes["resolved_is_compiled"], False)
        self.assertNotIn("compiled", notes)

    def test_an_uncompiled_fusion_mechanism_is_refused(self) -> None:
        """jit_fuser is torch.jit.script below torch 2.2, and can be disabled."""
        with self.assertRaisesRegex(RuntimeError, "not a torch.compile"):
            _require_bda_dispatch(
                arm=MCORE_BASE_ARM,
                fused=True,
                resolved=_unfused_stand_in(),
                fused_train=_unfused_stand_in(),
            )

    def test_a_fused_arm_that_resolved_elsewhere_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "not to"):
            _require_bda_dispatch(
                arm=MCORE_BASE_ARM,
                fused=True,
                resolved=_unfused_stand_in(),
                fused_train=_compiled_stand_in(),
            )

    def test_an_unfused_arm_that_resolved_to_the_fused_one_is_refused(
        self,
    ) -> None:
        """The failure that makes the two arms one implementation."""
        fused_train = _compiled_stand_in()
        with self.assertRaisesRegex(RuntimeError, "did not reach the call"):
            _require_bda_dispatch(
                arm=MCORE_NO_FUSION_ARM,
                fused=False,
                resolved=fused_train,
                fused_train=fused_train,
            )

    def test_an_unfused_arm_that_resolved_to_another_compile_is_refused(
        self,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "not the eager side"):
            _require_bda_dispatch(
                arm=MCORE_NO_FUSION_ARM,
                fused=False,
                resolved=_compiled_stand_in(),
                fused_train=_compiled_stand_in(),
            )

    def test_an_unfused_arm_from_another_factory_is_refused(self) -> None:
        def somewhere_else(x_with_bias, residual, prob):
            return residual + x_with_bias[0]

        with self.assertRaisesRegex(RuntimeError, "bias_dropout_add_unfused"):
            _require_bda_dispatch(
                arm=MCORE_NO_FUSION_ARM,
                fused=False,
                resolved=somewhere_else,
                fused_train=_compiled_stand_in(),
            )


def _layer(**overrides):
    """A stand-in for one built ``TransformerLayer``, at the base contract."""
    config = {
        "bias_dropout_fusion": True,
        "add_bias_linear": False,
        "fp32_residual_connection": False,
    }
    config.update(overrides.pop("config", {}))
    return SimpleNamespace(
        training=overrides.pop("training", True),
        hidden_dropout=overrides.pop("hidden_dropout", 0.0),
        config=SimpleNamespace(**config),
    )


class LayerContractTests(unittest.TestCase):
    def test_the_base_contract_passes(self) -> None:
        _require_mcore_layer_contract(_layer(), MCORE_BASE_ARM, True)

    def test_the_unfused_contract_passes(self) -> None:
        _require_mcore_layer_contract(
            _layer(config={"bias_dropout_fusion": False}),
            MCORE_NO_FUSION_ARM,
            False,
        )

    def test_an_evaluation_mode_layer_is_refused(self) -> None:
        """It resolves to the inference function, which adds in place."""
        with self.assertRaisesRegex(RuntimeError, "evaluation-mode"):
            _require_mcore_layer_contract(
                _layer(training=False), MCORE_BASE_ARM, True
            )

    def test_a_flag_that_did_not_take_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "bias_dropout_fusion"):
            _require_mcore_layer_contract(
                _layer(), MCORE_NO_FUSION_ARM, False
            )

    def test_a_real_dropout_probability_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "hidden_dropout"):
            _require_mcore_layer_contract(
                _layer(hidden_dropout=0.1), MCORE_BASE_ARM, True
            )

    def test_a_linear_bias_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "add_bias_linear"):
            _require_mcore_layer_contract(
                _layer(config={"add_bias_linear": True}), MCORE_BASE_ARM, True
            )

    def test_an_fp32_residual_connection_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "fp32_residual_connection"):
            _require_mcore_layer_contract(
                _layer(config={"fp32_residual_connection": True}),
                MCORE_BASE_ARM,
                True,
            )

    def test_every_problem_is_reported_at_once(self) -> None:
        """A guard that stops at the first fault costs a second build."""
        with self.assertRaises(RuntimeError) as raised:
            _require_mcore_layer_contract(
                _layer(training=False, hidden_dropout=0.1),
                MCORE_BASE_ARM,
                True,
            )
        message = str(raised.exception)
        self.assertIn("evaluation-mode", message)
        self.assertIn("hidden_dropout", message)


class RegisteredShapeTests(unittest.TestCase):
    def test_the_inputs_build_at_every_registered_model_size(self) -> None:
        """``--model-size`` is single-valued but not fixed, so both must work."""
        for name in ("normal", "huge"):
            with self.subTest(size=name):
                shape = shape_by_name(name)
                workload = KernelWorkload(batch=1, seq_len=2)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(0)
                inputs = attn_residual_inputs(
                    shape, workload, torch.device("cpu"), generator
                )
                self.assertEqual(inputs.attn_out.shape[-1], shape.dim)
                self.assertEqual(inputs.residual.shape[-1], shape.dim)


if __name__ == "__main__":
    unittest.main()
