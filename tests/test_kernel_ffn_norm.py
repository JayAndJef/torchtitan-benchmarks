"""CPU tests for the ``ffn_norm`` cross-engine kernel scenario.

Everything here runs without a GPU, without megatron and without
TransformerEngine. Three things are testable that way, and they are the three
that a wrong scenario gets wrong quietly:

* the two engines declare **one** epsilon, and the titan arm builds the same
  config node the production block builds;
* the fp64 reference is the RMSNorm both engines implement, proved against
  ``torch.nn.RMSNorm`` in fp64; and
* the shared timed closures and the build guards behave, exercised over a
  plain CPU ``torch.nn.RMSNorm`` in place of the two real modules.

The mcore side itself needs a CUDA device, a process group and TE, so no test
here touches it. What a CPU can still check about that side is the guard:
``_require_a_real_norm`` is what stops an ``IdentityOp`` or a residual-fused
tuple from reaching a published number, and both cases are exercised with
stand-in modules.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from benchmarks.kernel.operations.common import WEIGHT_STD
from benchmarks.kernel.operations.ffn_norm import (
    _norm_arm,
    _require_a_real_norm,
    _require_grads,
    build_ffn_norm_copy_floor,
    ffn_norm_inputs,
    ffn_norm_reference,
    FfnNormInputs,
    NORM_EPS,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE
from benchmarks.models.piper_qwen3.shape import PiperShape, shape_by_name

# Small enough to run in milliseconds, wide enough that the norm reduces over
# a real row rather than a handful of elements.
TINY = PiperShape(name="tiny", dim=256, n_layers=2, vocab_size=64)
TINY_WORKLOAD = KernelWorkload(batch=2, seq_len=16)


def _inputs(shape=TINY, workload=TINY_WORKLOAD, seed=0) -> FfnNormInputs:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return ffn_norm_inputs(shape, workload, torch.device("cpu"), generator)


def _rel_l2(value: torch.Tensor, truth: torch.Tensor) -> float:
    delta = (value.double() - truth.double()).norm()
    return (delta / truth.double().norm()).item()


class SharedEpsilonTests(unittest.TestCase):
    """One epsilon, or the two arms compute two different functions."""

    def test_the_module_reads_its_epsilon_from_the_mcore_profile(self) -> None:
        self.assertEqual(
            NORM_EPS, float(BASE.config_overrides["layernorm_epsilon"])
        )

    def test_torchtitan_declares_the_same_epsilon(self) -> None:
        """The value the titan builder refuses to disagree with."""
        from torchtitan.models.qwen3 import _qwen3_norm

        self.assertEqual(float(_qwen3_norm(TINY.dim).eps), NORM_EPS)

    def test_the_production_block_uses_that_norm_config_for_ffn_norm(self) -> None:
        """The titan arm builds ``_qwen3_norm(dim)``, so it must be the node.

        Without this, the arm could time a norm the registry never puts in
        front of the MoE block, and every other test would still pass.
        """
        from torchtitan.models.qwen3 import _qwen3_norm

        from benchmarks.models.piper_qwen3.config_registry import _piper_1b_model

        expected = _qwen3_norm(TINY.dim)
        config = _piper_1b_model(fuse_qkv=True, shape=TINY)
        self.assertEqual(len(config.layers), TINY.n_layers)
        for layer in config.layers:
            self.assertEqual(layer.ffn_norm, expected)


class InputsTests(unittest.TestCase):
    def test_the_inputs_are_one_canonical_bsd_batch(self) -> None:
        inputs = _inputs()
        self.assertEqual(
            tuple(inputs.x.shape),
            (TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len, TINY.dim),
        )
        self.assertEqual(inputs.x.dtype, torch.bfloat16)
        self.assertTrue(inputs.x.is_contiguous())
        self.assertEqual(tuple(inputs.grad_out.shape), tuple(inputs.x.shape))
        self.assertEqual(inputs.grad_out.dtype, torch.bfloat16)
        self.assertEqual(inputs.eps, NORM_EPS)

    def test_the_shared_gain_is_near_one_and_is_not_exactly_one(self) -> None:
        """Both engines initialize the gain to ones, which hides a lost load."""
        inputs = _inputs()
        self.assertEqual(tuple(inputs.weight.shape), (TINY.dim,))
        self.assertEqual(inputs.weight.dtype, torch.float32)
        self.assertAlmostEqual(float(inputs.weight.mean()), 1.0, places=1)
        self.assertAlmostEqual(
            float(inputs.weight.std()), WEIGHT_STD, places=2
        )
        self.assertFalse(torch.equal(inputs.weight, torch.ones(TINY.dim)))

    def test_bytes_moved_is_one_read_and_one_write_of_x(self) -> None:
        inputs = _inputs()
        self.assertEqual(inputs.bytes_moved, 2 * inputs.x.numel() * 2)

    def test_one_seed_rebuilds_the_inputs_bit_identically(self) -> None:
        """Every worker rebuilds these, and the gates compare across workers."""
        first, second = _inputs(seed=7), _inputs(seed=7)
        self.assertTrue(torch.equal(first.x, second.x))
        self.assertTrue(torch.equal(first.grad_out, second.grad_out))
        self.assertTrue(torch.equal(first.weight, second.weight))


class ReferenceTests(unittest.TestCase):
    def test_the_reference_is_torch_rmsnorm_in_fp64(self) -> None:
        """Exact agreement, so the truth is the operation and not a rewrite."""
        inputs = _inputs()
        reference = ffn_norm_reference(TINY, TINY_WORKLOAD, inputs)

        module = nn.RMSNorm(TINY.dim, eps=NORM_EPS, dtype=torch.float64)
        with torch.no_grad():
            module.weight.copy_(inputs.weight.to(torch.bfloat16).double())
        x = inputs.x.double().detach().requires_grad_()
        out = module(x)
        torch.autograd.backward(out, inputs.grad_out.double())

        self.assertTrue(torch.equal(reference["out"], out.detach()))
        self.assertTrue(torch.equal(reference["x_grad"], x.grad))
        self.assertTrue(torch.equal(reference["weight_grad"], module.weight.grad))

    def test_the_reference_holds_the_gain_the_arms_hold(self) -> None:
        """The fp64 truth quantizes the gain to bf16 first, as ``qkv_prep`` does.

        An fp64 truth built from the unrounded fp32 gain would charge both arms
        for an input cast neither performs, and the difference is measurable.
        """
        inputs = _inputs()
        quantized = ffn_norm_reference(TINY, TINY_WORKLOAD, inputs)["out"]

        exact = FfnNormInputs(
            x=inputs.x,
            grad_out=inputs.grad_out,
            weight=inputs.weight.to(torch.bfloat16).float(),
            eps=inputs.eps,
            bytes_moved=inputs.bytes_moved,
        )
        self.assertTrue(
            torch.equal(
                quantized, ffn_norm_reference(TINY, TINY_WORKLOAD, exact)["out"]
            )
        )
        self.assertNotEqual(
            float((inputs.weight - inputs.weight.to(torch.bfloat16).float()).abs().max()),
            0.0,
        )


class FloorTests(unittest.TestCase):
    def test_the_floor_copies_and_declares_forward_alone(self) -> None:
        inputs = _inputs()
        floor = build_ffn_norm_copy_floor(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(sorted(floor.calls), ["forward"])
        self.assertEqual(floor.bytes_moved, inputs.bytes_moved)
        self.assertEqual(floor.correctness_outputs(), {})
        floor.calls["forward"]()


class ArmClosureTests(unittest.TestCase):
    """The closures both engines share, over a stand-in CPU norm."""

    def _arm(self, inputs: FfnNormInputs):
        module = nn.RMSNorm(TINY.dim, eps=NORM_EPS)
        with torch.no_grad():
            module.weight.copy_(inputs.weight)
        module.to(torch.bfloat16)
        return _norm_arm("stand_in", module, module.weight, inputs)

    def test_the_arm_declares_forward_and_forward_backward_only(self) -> None:
        """No isolated backward: TE clears its saved tensors after the first."""
        arm = self._arm(_inputs())
        self.assertEqual(
            sorted(arm.calls), ["forward", "forward_backward"]
        )

    def test_the_named_outputs_match_the_fp64_reference(self) -> None:
        """The gates the scenario declares are reachable in bf16."""
        inputs = _inputs()
        outputs = self._arm(inputs).correctness_outputs()
        reference = ffn_norm_reference(TINY, TINY_WORKLOAD, inputs)

        self.assertEqual(sorted(outputs), ["out", "weight_grad", "x_grad"])
        self.assertLess(_rel_l2(outputs["out"], reference["out"]), 2e-2)
        self.assertLess(_rel_l2(outputs["x_grad"], reference["x_grad"]), 2e-2)
        self.assertLess(
            _rel_l2(outputs["weight_grad"], reference["weight_grad"]), 5e-2
        )

    def test_a_repeated_round_trip_does_not_accumulate_a_gradient(self) -> None:
        """Every timed call must measure one backward, not a growing sum."""
        inputs = _inputs()
        arm = self._arm(inputs)
        round_trip = arm.calls["forward_backward"]
        round_trip()
        first = arm.correctness_outputs()["x_grad"].clone()
        for _ in range(3):
            round_trip()
        second = arm.correctness_outputs()["x_grad"]
        self.assertTrue(torch.equal(first, second))


class _TupleNorm(nn.Module):
    """A stand-in for ``TEFusedResidualRMSNorm``: (output, residual)."""

    def forward(self, x):
        return x * 2, x


class GuardTests(unittest.TestCase):
    def test_a_real_norm_passes_the_guard(self) -> None:
        inputs = _inputs()
        module = nn.RMSNorm(TINY.dim, eps=NORM_EPS)
        with torch.no_grad():
            module.weight.copy_(inputs.weight)
        module.to(torch.bfloat16)
        _require_a_real_norm(module, inputs.x, "stand_in")

    def test_a_tuple_return_is_refused(self) -> None:
        """The residual fusion is a different operation, not this arm."""
        with self.assertRaisesRegex(RuntimeError, "fused_residual_rmsnorm"):
            _require_a_real_norm(_TupleNorm(), _inputs().x, "stand_in")

    def test_an_identity_module_is_refused(self) -> None:
        """megatron substitutes IdentityOp when a layer has no experts."""
        with self.assertRaisesRegex(RuntimeError, "returned its own input"):
            _require_a_real_norm(nn.Identity(), _inputs().x, "stand_in")

    def test_a_missing_gradient_is_named_rather_than_dereferenced(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "weight_grad, x_grad"):
            _require_grads(
                "stand_in",
                {"out": torch.zeros(1), "x_grad": None, "weight_grad": None},
            )


class RegisteredShapeTests(unittest.TestCase):
    def test_the_inputs_build_at_every_registered_model_size(self) -> None:
        """``--model-size`` is single-valued but not fixed, so both must work."""
        for name in ("normal", "huge"):
            with self.subTest(size=name):
                shape = shape_by_name(name)
                workload = KernelWorkload(batch=1, seq_len=2)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(0)
                inputs = ffn_norm_inputs(
                    shape, workload, torch.device("cpu"), generator
                )
                self.assertEqual(tuple(inputs.weight.shape), (shape.dim,))
                self.assertEqual(inputs.x.shape[-1], shape.dim)


if __name__ == "__main__":
    unittest.main()
