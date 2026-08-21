"""CPU tests for the ``final_norm`` kernel scenario.

What is testable without a GPU is the half of the scenario that decides
whether the numbers mean anything: the shared inputs, the fp64 reference both
arms are gated against, the titan extraction path, and the weight-map fact the
cross-engine comparison rests on.

The mcore arm is **not** covered here and cannot be. It needs CUDA,
TransformerEngine and a megatron process group, and TE does not even import on
a host without the cuda-compat stack. Its guards therefore live in the builder
itself (``_assert_te_rmsnorm``), where they run on the box that measures.

The gate constants below mirror the ones the scenario declares. A test that
invented its own tolerance would pass while the run failed.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.kernel.engine.run import resolve_symbol
from benchmarks.kernel.operations.final_norm import (
    FinalNormInputs,
    build_final_norm_copy_floor,
    build_final_norm_titan,
    final_norm_inputs,
    final_norm_reference,
    titan_final_norm_module,
    _final_norm_arm,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE
from benchmarks.models.piper_qwen3.shape import PiperShape

# Small enough to run in a second, and every mechanism under test is the one a
# GPU arm uses. dim 256 keeps the RMS reduction long enough to be meaningful.
TINY = PiperShape.derived(name="tiny", dim=256, n_layers=2, vocab_size=64)
WORKLOAD = KernelWorkload(batch=2, seq_len=16)

# The rel_l2 the declaration gates on. Measured on CPU at this shape: 1.6e-3.
DECLARED_GATE = 2e-2

# The dotted paths the registry declaration names. The engine resolves these
# strings inside the worker, so a renamed builder is a run-time failure on the
# GPU rather than an import error here.
DECLARED_BUILDER_PATHS = (
    "benchmarks.kernel.operations.final_norm:final_norm_inputs",
    "benchmarks.kernel.operations.final_norm:final_norm_reference",
    "benchmarks.kernel.operations.final_norm:build_final_norm_copy_floor",
    "benchmarks.kernel.operations.final_norm:build_final_norm_mcore_base",
    "benchmarks.kernel.operations.final_norm:build_final_norm_titan",
)

# The modes both arms declare. There is deliberately no isolated "backward":
# TE clears its saved-tensor context on the first backward, so a retained
# graph cannot be re-run.
DECLARED_MODES = {"forward", "forward_backward"}


def make_inputs(seed: int = 0) -> FinalNormInputs:
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return final_norm_inputs(TINY, WORKLOAD, torch.device("cpu"), generator)


def rel_l2(value: torch.Tensor, truth: torch.Tensor) -> float:
    delta = value.double() - truth.double()
    return (delta.norm() / truth.double().norm()).item()


class InputsTests(unittest.TestCase):
    def test_the_shared_tensors_have_the_shapes_both_engines_need(self) -> None:
        inputs = make_inputs()
        rows = (WORKLOAD.batch, WORKLOAD.seq_len, TINY.dim)
        self.assertEqual(tuple(inputs.x.shape), rows)
        self.assertEqual(tuple(inputs.grad_out.shape), rows)
        self.assertEqual(tuple(inputs.weight.shape), (TINY.dim,))
        for tensor in (inputs.x, inputs.grad_out, inputs.weight):
            self.assertEqual(tensor.dtype, torch.bfloat16)

    def test_the_input_is_contiguous(self) -> None:
        """The layout charge the scenario refuses to hand either engine.

        TE flattens with ``input_.contiguous().view(-1, D)``. A transposed
        view would put an 8 MiB copy inside the mcore arm's timed call and
        nowhere in titan's, which would read as a kernel difference.
        """
        self.assertTrue(make_inputs().x.is_contiguous())

    def test_the_upstream_gradient_is_not_a_copy_of_the_input(self) -> None:
        inputs = make_inputs()
        self.assertFalse(torch.equal(inputs.x, inputs.grad_out))

    def test_the_gain_is_centered_on_one_and_is_not_all_ones(self) -> None:
        """An all-ones gain would hide an arm that ignores the parameter."""
        weight = make_inputs().weight.float()
        self.assertAlmostEqual(weight.mean().item(), 1.0, places=1)
        self.assertGreater(weight.std().item(), 0.0)

    def test_the_eps_comes_from_the_mcore_profile(self) -> None:
        self.assertEqual(
            make_inputs().eps,
            float(BASE.config_overrides["layernorm_epsilon"]),
        )

    def test_the_floor_byte_count_is_one_read_plus_one_write(self) -> None:
        inputs = make_inputs()
        self.assertEqual(
            inputs.copy_bytes, 2 * inputs.x.numel() * inputs.x.element_size()
        )

    def test_one_seed_gives_one_set_of_tensors(self) -> None:
        """Every worker rebuilds the inputs, so they must not drift."""
        first, second = make_inputs(3), make_inputs(3)
        self.assertTrue(torch.equal(first.x, second.x))
        self.assertTrue(torch.equal(first.grad_out, second.grad_out))
        self.assertTrue(torch.equal(first.weight, second.weight))


class ReferenceTests(unittest.TestCase):
    """The fp64 truth both arms are gated against.

    An error here fails both arms identically and invisibly, so the reference
    is checked against torch's own ``rms_norm`` rather than against itself.
    """

    def test_the_forward_matches_torch_rms_norm_in_fp64(self) -> None:
        inputs = make_inputs()
        reference = final_norm_reference(TINY, WORKLOAD, inputs)
        expected = F.rms_norm(
            inputs.x.double(),
            [TINY.dim],
            inputs.weight.double(),
            eps=inputs.eps,
        )
        self.assertLess(rel_l2(reference["out"], expected), 1e-12)

    def test_the_gradients_match_torch_rms_norm_in_fp64(self) -> None:
        inputs = make_inputs()
        reference = final_norm_reference(TINY, WORKLOAD, inputs)
        x = inputs.x.double().detach().requires_grad_()
        weight = inputs.weight.double().detach().requires_grad_()
        out = F.rms_norm(x, [TINY.dim], weight, eps=inputs.eps)
        x_grad, weight_grad = torch.autograd.grad(
            out, (x, weight), inputs.grad_out.double()
        )
        self.assertLess(rel_l2(reference["x_grad"], x_grad), 1e-12)
        self.assertLess(rel_l2(reference["weight_grad"], weight_grad), 1e-12)

    def test_the_reference_names_every_gated_output(self) -> None:
        reference = final_norm_reference(TINY, WORKLOAD, make_inputs())
        self.assertEqual(
            set(reference), {"out", "x_grad", "weight_grad"}
        )
        for tensor in reference.values():
            self.assertEqual(tensor.dtype, torch.float64)


class TitanExtractionTests(unittest.TestCase):
    """The titan side needs no ``Trainer.Config``, and this pins that."""

    def test_the_module_is_the_node_the_production_config_carries(self) -> None:
        from benchmarks.models.piper_qwen3.config_registry import (
            _piper_1b_model,
        )

        declared = _piper_1b_model(fuse_qkv=True, shape=TINY).norm
        module = titan_final_norm_module(TINY)
        self.assertIsInstance(module, nn.RMSNorm)
        self.assertEqual(module.normalized_shape, (declared.normalized_shape,))
        self.assertEqual(module.eps, declared.eps)

    def test_the_extraction_does_not_depend_on_the_layer_count(self) -> None:
        """The norm sits at the model level, so one exists at any depth."""
        one_layer = PiperShape.derived(name="one", dim=256, n_layers=1, vocab_size=64)
        self.assertEqual(
            titan_final_norm_module(one_layer).weight.shape,
            titan_final_norm_module(TINY).weight.shape,
        )

    def test_both_engines_declare_the_same_epsilon(self) -> None:
        """A silent divergence would make the ratio compare two functions."""
        from torchtitan.models.qwen3 import _qwen3_norm

        self.assertEqual(
            float(_qwen3_norm(TINY.dim).eps),
            float(BASE.config_overrides["layernorm_epsilon"]),
        )

    def test_the_titan_arm_agrees_with_the_fp64_reference(self) -> None:
        """The whole titan half, minus the compile wrapper.

        ``build_final_norm_titan`` adds ``torch.compile`` on top of this, and
        an isolated CPU test must not depend on a working Inductor toolchain.
        Everything else -- the config extraction, the dtype, the shared gain,
        the closures -- is the code the GPU arm runs.
        """
        inputs = make_inputs()
        module = titan_final_norm_module(TINY).to(dtype=torch.bfloat16)
        with torch.no_grad():
            module.weight.copy_(inputs.weight)
        arm = _final_norm_arm("titan", module, module.weight, inputs)
        reference = final_norm_reference(TINY, WORKLOAD, inputs)
        outputs = arm.correctness_outputs()
        for name in ("out", "x_grad", "weight_grad"):
            with self.subTest(output=name):
                self.assertLess(
                    rel_l2(outputs[name], reference[name]), DECLARED_GATE
                )

    def test_the_arm_refuses_a_gain_that_takes_no_gradient(self) -> None:
        """Backward would skip the gain, so the arm would do less work."""
        inputs = make_inputs()
        module = titan_final_norm_module(TINY).to(dtype=torch.bfloat16)
        module.weight.requires_grad_(False)
        with self.assertRaises(RuntimeError):
            _final_norm_arm("titan", module, module.weight, inputs)


class ArmShapeTests(unittest.TestCase):
    def test_the_arms_expose_the_declared_modes(self) -> None:
        """``_seeded_build`` raises when the builder disagrees, so pin both."""
        inputs = make_inputs()
        module = titan_final_norm_module(TINY).to(dtype=torch.bfloat16)
        arm = _final_norm_arm("titan", module, module.weight, inputs)
        self.assertEqual(set(arm.calls), DECLARED_MODES)

    def test_the_floor_copies_the_input_and_declares_its_bytes(self) -> None:
        inputs = make_inputs()
        floor = build_final_norm_copy_floor(TINY, WORKLOAD, inputs)
        self.assertEqual(set(floor.calls), {"forward"})
        self.assertEqual(floor.bytes_moved, inputs.copy_bytes)
        floor.calls["forward"]()
        self.assertEqual(floor.correctness_outputs(), {})

    def test_every_declared_builder_path_resolves(self) -> None:
        for path in DECLARED_BUILDER_PATHS:
            with self.subTest(path=path):
                self.assertTrue(callable(resolve_symbol(path)))

    def test_the_titan_builder_refuses_a_mismatched_epsilon(self) -> None:
        """The two engines must compute one function, or the ratio is empty."""
        from dataclasses import replace

        inputs = replace(make_inputs(), eps=1e-3)
        with self.assertRaises(RuntimeError):
            build_final_norm_titan(TINY, WORKLOAD, inputs)


class WeightMapTests(unittest.TestCase):
    """The cross-engine transfer this scenario needs, taken from the map.

    ``benchmarks/models/piper_qwen3/megatron_weights.py`` owns the
    titan-to-megatron parameter map. For this scenario the map says the
    transfer is the identity, which is why both builders load one shared gain
    instead of converting anything. These tests pin that reading, so a map
    change cannot leave the builders silently wrong.
    """

    def setUp(self) -> None:
        from benchmarks.models.piper_qwen3.titan_model import build_titan_model

        self.model = build_titan_model(
            shape=TINY, device="cpu", dtype=torch.float32
        )
        self.state = dict(self.model.state_dict())

    def final_norm_transfers(self):
        from benchmarks.models.piper_qwen3.megatron_weights import (
            weight_transfers,
        )

        return [
            transfer
            for transfer in weight_transfers(self.state, TINY)
            if transfer[0] == "final_norm"
        ]

    def test_the_map_holds_exactly_one_final_norm_transfer(self) -> None:
        self.assertEqual(len(self.final_norm_transfers()), 1)

    def test_the_transfer_names_the_attribute_path_the_builder_walks(
        self,
    ) -> None:
        """``build_final_norm_mcore_base`` reads
        ``model.decoder.final_layernorm``. The map names that parameter."""
        _, name, _ = self.final_norm_transfers()[0]
        self.assertEqual(name, "decoder.final_layernorm.weight")

    def test_the_transfer_is_the_identity_on_titan_norm_weight(self) -> None:
        _, _, tensor = self.final_norm_transfers()[0]
        self.assertTrue(torch.equal(tensor, self.state["norm.weight"]))
        self.assertEqual(tuple(tensor.shape), (TINY.dim,))


if __name__ == "__main__":
    unittest.main()
