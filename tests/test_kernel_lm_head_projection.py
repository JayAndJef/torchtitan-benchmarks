"""CPU tests for the ``lm_head_projection`` kernel scenario.

The two arm builders need a GPU, megatron and torchtitan, so nothing here
builds one. What is checkable on a CPU is everything that decides whether a
GPU run means what the table says it means:

* the seeded inputs, including the weight scale, which this scenario draws
  differently from the rest of the cross-engine roster and on purpose;
* the fp64 truth the gates measure against;
* the agreement between this module's megatron attribute path and
  ``benchmarks.models.piper_qwen3.megatron_weights``, which is the
  repository's authority on which titan weight equals which megatron weight;
* the two guards that refuse to time the wrong module; and
* the shape canonicalization, which is what lets the correctness engine
  subtract a megatron THD output from a titan BSD output at all.

**The megatron guard carries this scenario's headline claim.** The plan's
roster names the anchor ``mcore/base``, and at every other cross-engine
scenario that arm is a TransformerEngine module. Here it is not: megatron's
output layer is ``tensor_parallel.ColumnParallelLinear`` unless an mxfp8
recipe is active, and no profile in this repository sets one. A reader must
not take the published row as "TE's GEMM against torch's GEMM", so the guard
compares the class by name and raises, and the test below feeds it a
stand-in named after the TE class to prove the refusal is real.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from benchmarks.kernel.operations.lm_head_projection import (
    MCORE_MODULE_CLASS,
    MCORE_WEIGHT_COMPONENT,
    MCORE_WEIGHT_NAME,
    TITAN_MODULE_CLASS,
    TITAN_WEIGHT_NAME,
    _assert_mcore_output_layer,
    _assert_titan_lm_head,
    _navigate,
    _projection_arm,
    lm_head_projection_inputs,
    lm_head_projection_reference,
    mcore_module_path,
)
from benchmarks.kernel.operations.common import WEIGHT_STD
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.megatron_weights import (
    COMPONENTS,
    weight_transfers,
)
from benchmarks.models.piper_qwen3.shape import PiperShape

# Small enough to run in milliseconds, and still a real geometry. The
# vocabulary is deliberately larger than ``dim``, as it is at both registered
# shapes, so a test that confused the two axes would fail.
TINY = PiperShape(name="tiny", dim=128, n_layers=2, vocab_size=192)
WORKLOAD = KernelWorkload(batch=2, seq_len=4)


def _inputs(shape: PiperShape = TINY, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return lm_head_projection_inputs(
        shape, WORKLOAD, torch.device("cpu"), generator
    )


class InputsTests(unittest.TestCase):
    def test_shapes_and_dtypes_are_what_both_engines_consume(self) -> None:
        inputs = _inputs()
        self.assertEqual(
            tuple(inputs.x.shape),
            (WORKLOAD.batch, WORKLOAD.seq_len, TINY.dim),
        )
        self.assertEqual(
            tuple(inputs.grad_out.shape),
            (WORKLOAD.batch, WORKLOAD.seq_len, TINY.vocab_size),
        )
        self.assertEqual(
            tuple(inputs.weight.shape), (TINY.vocab_size, TINY.dim)
        )
        # Activations are bf16 because both engines run plain bf16. The weight
        # stays fp32 so each arm casts it once, the way qkv and attn_out_proj
        # do.
        self.assertEqual(inputs.x.dtype, torch.bfloat16)
        self.assertEqual(inputs.grad_out.dtype, torch.bfloat16)
        self.assertEqual(inputs.weight.dtype, torch.float32)

    def test_the_weight_is_drawn_at_one_over_sqrt_dim(self) -> None:
        """Not at this package's ``WEIGHT_STD``, and the difference matters.

        ``1 / sqrt(dim)`` puts the logits at unit scale at every shape, which
        is the scale the ``cross_entropy`` scenario states its own logits are
        drawn at, so the producer and the consumer of a logit tensor agree.
        A fixed standard deviation would not. The two constants are far
        enough apart at this geometry that the sample deviation separates
        them.
        """
        expected = 1.0 / TINY.dim**0.5
        self.assertNotAlmostEqual(expected, WEIGHT_STD, places=3)
        # 192*128 samples: the sample sd sits well inside 10% of the truth.
        found = _inputs().weight.std().item()
        self.assertAlmostEqual(found / expected, 1.0, delta=0.1)
        self.assertGreater(abs(found / WEIGHT_STD - 1.0), 0.1)

    def test_the_generator_alone_decides_the_values(self) -> None:
        """Two arms in two processes must see the same tensors.

        Every timing worker rebuilds the inputs from a seeded generator. If
        anything here read the global RNG, one arm would face different data
        than the other and the published ratio would be a ratio of two
        problems.
        """
        torch.manual_seed(1)
        first = _inputs(seed=7)
        torch.manual_seed(2)
        second = _inputs(seed=7)
        for name in ("x", "grad_out", "weight"):
            self.assertTrue(
                torch.equal(getattr(first, name), getattr(second, name)), name
            )
        different = _inputs(seed=8)
        self.assertFalse(torch.equal(first.x, different.x))


class ReferenceTests(unittest.TestCase):
    def test_the_reference_is_the_fp64_projection_and_both_gradients(
        self,
    ) -> None:
        inputs = _inputs()
        reference = lm_head_projection_reference(TINY, WORKLOAD, inputs)
        self.assertEqual(sorted(reference), ["out", "weight_grad", "x_grad"])
        weight64 = inputs.weight.to(torch.bfloat16).double()
        x64 = inputs.x.double()
        grad64 = inputs.grad_out.double()
        torch.testing.assert_close(
            reference["out"], x64 @ weight64.T, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            reference["x_grad"], grad64 @ weight64, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            reference["weight_grad"],
            grad64.reshape(-1, TINY.vocab_size).T @ x64.reshape(-1, TINY.dim),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(reference["out"].dtype, torch.float64)

    def test_the_reference_rounds_the_weight_to_bfloat16_first(self) -> None:
        """Otherwise both arms pay for the input cast in the gate.

        The truth has to be the truth for the weight the arms hold. A value
        below is chosen so bf16 rounds it: the two projections then differ,
        and only one of them is the one to gate against.
        """
        inputs = _inputs()
        # 1 + 2**-10 needs 11 mantissa bits; bf16 has 8, so it rounds to 1.
        exact = torch.full_like(inputs.weight, 1.0 + 2.0**-10)
        inputs.weight = exact
        reference = lm_head_projection_reference(TINY, WORKLOAD, inputs)
        rounded = inputs.x.double() @ exact.to(torch.bfloat16).double().T
        unrounded = inputs.x.double() @ exact.double().T
        torch.testing.assert_close(
            reference["out"], rounded, rtol=1e-12, atol=1e-12
        )
        self.assertFalse(torch.allclose(rounded, unrounded))


class WeightMapAgreementTests(unittest.TestCase):
    """This scenario must not reimplement the titan-to-megatron map."""

    def test_the_module_path_drops_only_the_weight_suffix(self) -> None:
        self.assertEqual(mcore_module_path(), "output_layer")
        self.assertEqual(mcore_module_path() + ".weight", MCORE_WEIGHT_NAME)

    def test_the_module_path_holds_no_layer_index(self) -> None:
        """One output layer exists whatever ``n_layers`` is.

        ``attn_out_proj`` reaches a per-layer module and formats a layer
        number into its path. This scenario must not, because a numeric
        segment here would be a layer count this model does not have at the
        head, and would silently pick the wrong module if it ever resolved.
        """
        self.assertNotIn(".", mcore_module_path())
        self.assertFalse(any(part.isdigit() for part in MCORE_WEIGHT_NAME))

    def test_the_component_this_scenario_measures_is_a_declared_one(
        self,
    ) -> None:
        self.assertIn(MCORE_WEIGHT_COMPONENT, COMPONENTS)

    def test_the_map_pairs_titan_lm_head_with_megatron_output_layer(
        self,
    ) -> None:
        """The names this module navigates are the names the map declares.

        ``weight_transfers`` is a generator over the whole model, so the test
        stops at the first ``lm_head`` entry rather than build every tensor.
        """
        state = _partial_titan_state(TINY)
        component, megatron_name, tensor = _first_lm_head(state, TINY)
        self.assertEqual(component, MCORE_WEIGHT_COMPONENT)
        self.assertEqual(megatron_name, MCORE_WEIGHT_NAME)
        self.assertIn(TITAN_WEIGHT_NAME, state)
        # The same object, not an equal one: the map applies no reshape and no
        # transpose to this weight, which is why one seeded tensor can be
        # copied into both engines without a layout step.
        self.assertIs(tensor, state[TITAN_WEIGHT_NAME])

    def test_the_shared_weight_has_the_shape_both_engines_expect(self) -> None:
        inputs = _inputs()
        state = _partial_titan_state(TINY)
        self.assertEqual(
            tuple(inputs.weight.shape),
            tuple(state[TITAN_WEIGHT_NAME].shape),
        )


def _partial_titan_state(shape: PiperShape) -> dict[str, torch.Tensor]:
    """Just the titan weights ``weight_transfers`` reads before ``lm_head``.

    The generator yields the embedding first and the lm head second, so
    nothing after those two is ever reached and nothing after them is built.
    """
    return {
        "tok_embeddings.weight": torch.randn(shape.vocab_size, shape.dim),
        "lm_head.weight": torch.randn(shape.vocab_size, shape.dim),
    }


def _first_lm_head(state: dict[str, torch.Tensor], shape: PiperShape):
    for entry in weight_transfers(state, shape):
        if entry[0] == MCORE_WEIGHT_COMPONENT:
            return entry
    raise AssertionError("weight_transfers yields no lm_head entry")


class _FakeGptModel:
    """A ``GPTModel`` reduced to the one attribute this scenario walks."""

    def __init__(self, module: object) -> None:
        self.output_layer = module


class ColumnParallelLinear(nn.Module):
    """A stand-in with the attributes the megatron guard reads.

    Named for the class the guard demands, because the guard compares class
    names. A real ``ColumnParallelLinear`` needs megatron, a process group
    and a CUDA device, none of which this test has.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        output_size_per_partition: int | None = None,
        gather_output: bool = False,
        sequence_parallel: bool = False,
        allreduce_dgrad: bool = False,
        gradient_accumulation_fusion: bool = False,
        defer_embedding_wgrad_compute: bool = False,
        bias: bool = False,
        dtype: torch.dtype = torch.bfloat16,
    ) -> None:
        super().__init__()
        # bf16 by default, because that is what both guards demand. The
        # parameter exists so a test can build the one wrong module no
        # correctness gate can see.
        self.weight = nn.Parameter(
            torch.zeros(out_features, in_features, dtype=dtype)
        )
        self.bias = (
            nn.Parameter(torch.zeros(out_features, dtype=dtype))
            if bias
            else None
        )
        self.output_size_per_partition = (
            out_features
            if output_size_per_partition is None
            else output_size_per_partition
        )
        self.gather_output = gather_output
        self.sequence_parallel = sequence_parallel
        self.allreduce_dgrad = allreduce_dgrad
        self.gradient_accumulation_fusion = gradient_accumulation_fusion
        self.defer_embedding_wgrad_compute = defer_embedding_wgrad_compute


class TELMHeadColumnParallelLinear(ColumnParallelLinear):
    """The class megatron picks under an mxfp8 recipe, and only then."""


class Linear(ColumnParallelLinear):
    """A stand-in for TorchTitan's ``Linear``, named for the guard."""


class ScaledBiasRowwiseLinear(Linear):
    """TorchTitan's own subclass of ``Linear``, which the guard must refuse."""


class NavigationTests(unittest.TestCase):
    def test_the_path_reaches_the_module(self) -> None:
        target = ColumnParallelLinear(TINY.vocab_size, TINY.dim)
        model = _FakeGptModel(target)
        self.assertIs(_navigate(model, mcore_module_path()), target)

    def test_a_path_that_does_not_exist_raises(self) -> None:
        model = _FakeGptModel(ColumnParallelLinear(4, 4))
        with self.assertRaises(AttributeError):
            _navigate(model, "output_projection")


class McoreGuardTests(unittest.TestCase):
    def _module(self, **kwargs) -> ColumnParallelLinear:
        return ColumnParallelLinear(TINY.vocab_size, TINY.dim, **kwargs)

    def test_a_correct_module_passes_and_records_that_it_is_not_te(
        self,
    ) -> None:
        """The claim the scenario description rests on.

        ``mcore/base`` is TransformerEngine at every other cross-engine
        scenario. Here it is megatron-native, so the guard records the fact
        and the row must never be read as a TE comparison.
        """
        notes = _assert_mcore_output_layer(self._module(), TINY)
        self.assertEqual(notes["module"], MCORE_MODULE_CLASS)
        self.assertFalse(notes["transformer_engine"])
        self.assertEqual(notes["tensor_parallel_size"], 1)
        self.assertFalse(notes["collectives_run"])
        self.assertFalse(notes["compiled"])

    def test_the_transformer_engine_output_layer_is_refused(self) -> None:
        """The mxfp8 branch at ``gpt_model.py:264-267``, caught.

        Its class is a subclass of the one the scenario claims, so an
        ``isinstance`` check would pass it and publish a TransformerEngine
        module under a megatron-native label. The guard compares the class
        name, and this test is what proves the difference.
        """
        wrong = TELMHeadColumnParallelLinear(TINY.vocab_size, TINY.dim)
        self.assertIsInstance(wrong, ColumnParallelLinear)
        with self.assertRaisesRegex(RuntimeError, MCORE_MODULE_CLASS):
            _assert_mcore_output_layer(wrong, TINY)

    def test_a_module_of_the_wrong_shape_is_refused(self) -> None:
        """A wrong attribute path lands on some other linear of the model."""
        with self.assertRaisesRegex(RuntimeError, "output_layer.weight"):
            _assert_mcore_output_layer(
                ColumnParallelLinear(TINY.dim, TINY.dim), TINY
            )

    def test_a_sharded_output_layer_is_refused(self) -> None:
        """At tensor parallel size above 1 the arm holds one shard.

        The titan arm computes the whole projection, so the ratio would
        compare two different amounts of arithmetic.
        """
        with self.assertRaisesRegex(RuntimeError, "tensor parallel"):
            _assert_mcore_output_layer(
                self._module(output_size_per_partition=TINY.vocab_size // 2),
                TINY,
            )

    def test_gather_output_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "gather_output"):
            _assert_mcore_output_layer(self._module(gather_output=True), TINY)

    def test_sequence_parallel_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "sequence_parallel"):
            _assert_mcore_output_layer(
                self._module(sequence_parallel=True), TINY
            )

    def test_allreduce_dgrad_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "allreduce_dgrad"):
            _assert_mcore_output_layer(
                self._module(allreduce_dgrad=True), TINY
            )

    def test_gradient_accumulation_fusion_is_refused(self) -> None:
        """With it on, the weight gradient never reaches ``.grad``.

        Backward writes into a ``main_grad`` buffer instead, so
        ``correctness_outputs`` would read ``None`` and nothing would gate
        the wgrad GEMM.
        """
        with self.assertRaisesRegex(
            RuntimeError, "gradient_accumulation_fusion"
        ):
            _assert_mcore_output_layer(
                self._module(gradient_accumulation_fusion=True), TINY
            )

    def test_deferred_embedding_wgrad_is_refused(self) -> None:
        """With it on, the weight gradient never reaches ``.grad`` either.

        Forward also appends its input to ``embedding_activation_buffer`` on
        every call, so a burst of thousands grows the buffer without bound.
        The ``base`` profile never sets it and megatron defaults it False, so
        this check is unreachable today. The guard reads module attributes
        and never the config, so it is checked anyway.
        """
        with self.assertRaisesRegex(
            RuntimeError, "defer_embedding_wgrad_compute"
        ):
            _assert_mcore_output_layer(
                self._module(defer_embedding_wgrad_compute=True), TINY
            )

    def test_an_uncast_weight_is_refused(self) -> None:
        """The one wrong module no correctness gate can see.

        Both arms cast the shared fp32 weight to bf16. An arm that failed to
        cast would time an fp32 GEMM against a bf16 one, and it would still
        pass the 2e-2 rel_l2 gate: the bf16 quantization of the activation
        dominates the metric. Only this guard catches it.
        """
        with self.assertRaisesRegex(RuntimeError, "bfloat16"):
            _assert_mcore_output_layer(
                self._module(dtype=torch.float32), TINY
            )

    def test_a_bias_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "bias"):
            _assert_mcore_output_layer(self._module(bias=True), TINY)


class TitanGuardTests(unittest.TestCase):
    def _module(self, **kwargs) -> Linear:
        return Linear(TINY.vocab_size, TINY.dim, **kwargs)

    def test_a_correct_module_passes(self) -> None:
        notes = _assert_titan_lm_head(self._module(), TINY)
        self.assertEqual(notes["module"], TITAN_MODULE_CLASS)
        self.assertFalse(notes["transformer_engine"])
        # Eager, and that is the production treatment. ``apply_compile``
        # walks ``model.layers.named_children()`` alone, and ``lm_head`` is a
        # sibling of ``layers`` rather than a child of it.
        self.assertFalse(notes["compiled"])

    def test_the_scaled_bias_subclass_is_refused(self) -> None:
        """TorchTitan's own subclass divides the bias by a TP degree."""
        wrong = ScaledBiasRowwiseLinear(TINY.vocab_size, TINY.dim)
        with self.assertRaisesRegex(RuntimeError, TITAN_MODULE_CLASS):
            _assert_titan_lm_head(wrong, TINY)

    def test_a_module_of_the_wrong_shape_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "lm_head.weight"):
            _assert_titan_lm_head(Linear(TINY.dim, TINY.dim), TINY)

    def test_an_uncast_weight_is_refused(self) -> None:
        """The likelier of the two arms to lose the cast.

        This arm takes its dtype from ``titan_lm_head_module``'s own
        ``set_default_dtype`` call rather than from a config.
        """
        with self.assertRaisesRegex(RuntimeError, "bfloat16"):
            _assert_titan_lm_head(self._module(dtype=torch.float32), TINY)

    def test_a_bias_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "bias"):
            _assert_titan_lm_head(self._module(bias=True), TINY)


class ProjectionArmTests(unittest.TestCase):
    """The helper both arms are built from, run on a plain ``nn.Linear``.

    The layout test is the one worth stating plainly.
    ``engine/correctness.py`` pairs the two sides by output *name* and
    subtracts. Torch aligns shapes from the right, so a canonicalization that
    returned a ``[t, 1, V]`` tensor against a ``[B, L, V]`` one raises at
    every batch above 1, and at ``batch == 1`` the two broadcast silently.
    The test runs both native layouts through the same helper and requires
    identical canonical outputs.
    """

    def _arm(self, native: str, return_owner: bool = False):
        torch.manual_seed(0)
        in_features, out_features = 8, 6
        batch, seq = 2, 3
        module = nn.Linear(in_features, out_features, bias=False)
        x = torch.arange(
            batch * seq * in_features, dtype=torch.float32
        ).reshape(batch, seq, in_features) / 100.0
        grad = torch.ones(batch, seq, out_features)
        if native == "mcore":
            x_native = x.reshape(batch * seq, 1, in_features)
            grad_native = grad.reshape(batch * seq, 1, out_features)
        else:
            x_native, grad_native = x, grad
        arm = _projection_arm(
            name=native,
            weight_owner=module,
            call=lambda leaf: module(leaf),
            x_native=x_native,
            grad_native=grad_native,
            canonical_in=(batch, seq, in_features),
            canonical_out=(batch, seq, out_features),
        )
        return (arm, module) if return_owner else arm

    def test_the_arm_exposes_the_two_declared_modes_and_no_third(self) -> None:
        """The two modes every cross-engine scenario in the package declares.

        ``attn_out_proj`` cannot declare an isolated ``backward``, because
        TransformerEngine clears its saved context on the way out of
        backward. Keeping the same two modes here keeps the roster uniform,
        and backward cost stays recoverable as ``forward_backward`` minus
        ``forward``.
        """
        arm = self._arm("titan")
        self.assertEqual(sorted(arm.calls), ["forward", "forward_backward"])

    def test_both_native_layouts_produce_identical_canonical_outputs(
        self,
    ) -> None:
        titan = self._arm("titan").correctness_outputs()
        mcore = self._arm("mcore").correctness_outputs()
        self.assertEqual(sorted(titan), ["out", "weight_grad", "x_grad"])
        self.assertEqual(sorted(titan), sorted(mcore))
        for name in titan:
            self.assertEqual(
                tuple(titan[name].shape), tuple(mcore[name].shape), name
            )
            torch.testing.assert_close(titan[name], mcore[name])

    def test_the_canonical_shapes_are_the_ones_the_reference_reports(
        self,
    ) -> None:
        outputs = self._arm("mcore").correctness_outputs()
        self.assertEqual(tuple(outputs["out"].shape), (2, 3, 6))
        self.assertEqual(tuple(outputs["x_grad"].shape), (2, 3, 8))
        self.assertEqual(tuple(outputs["weight_grad"].shape), (6, 8))

    def test_forward_backward_clears_gradients_between_calls(self) -> None:
        """A timed closure runs thousands of times in one burst.

        Without the reset the weight gradient would grow without bound, and
        the arm would time an addition over ``[vocab_size, dim]`` on top of
        the projection.

        Read the gradient off the module directly. ``correctness_outputs``
        calls ``_reset_grads`` itself and then runs its own forward and
        backward, so a value read through it depends only on that last call.
        """
        arm, owner = self._arm("titan", return_owner=True)

        arm.calls["forward_backward"]()
        after_one = owner.weight.grad.clone()
        for _ in range(3):
            arm.calls["forward_backward"]()
        after_four = owner.weight.grad.clone()

        torch.testing.assert_close(after_one, after_four)
        # And prove the test can fail: four unreset backwards would be 4x.
        self.assertFalse(
            torch.allclose(after_one * 4, after_four),
            "the gradient did not accumulate, so this test proves nothing",
        )

    def test_forward_returns_the_projection(self) -> None:
        arm = self._arm("titan")
        self.assertEqual(tuple(arm.calls["forward"]().shape), (2, 3, 6))


if __name__ == "__main__":
    unittest.main()
