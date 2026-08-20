"""CPU tests for the ``moe_residual`` cross-engine kernel scenario.

Everything here runs without a GPU. The scenario's central claim -- that
megatron's ``mlp_bda`` and titan's ``x + moe(...)`` are the same add, and that
they differ only in fusion scope -- rests on four premises, and a CPU can check
all four:

* the ``base`` profile zeroes the dropout and turns the linear bias off, so
  megatron's ``_bias_dropout_add_func`` takes its no-bias branch;
* ``F.dropout(x, p=0.0, training=True)`` returns its own input **object**, so
  the branch reduces to ``residual + x`` rather than merely to something close
  to it;
* megatron's source still spells that branch, still decorates the fused
  function with ``@jit_fuser``, and still reads ``bias_dropout_fusion`` per
  call at the ``mlp_bda`` call site; and
* titan's block still writes ``x = x + self.moe(...)``.

**No test here imports megatron**, and that is a cost decision rather than an
oversight. ``megatron/core/__init__.py`` imports TransformerEngine, so the
cheapest possible ``from megatron.core...`` costs about 9 s and pulls a GPU
library into a suite that otherwise runs in about two seconds. The pinned
sources are read as **text** instead, which is what a submodule bump has to
invalidate anyway. ``tests/test_lm_head_losses.py`` hashes vendored sources for
the same reason.

What that leaves untested on CPU is ``_assert_mcore_bda`` itself, whose first
statement is the megatron import. It runs on every mcore build, so a wiring
error raises there rather than reaching a number.
"""

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from benchmarks.execution.paths import TITAN_DIR
from benchmarks.kernel.operations.moe_residual import (
    _require_grads,
    _residual_add,
    _residual_arm,
    build_moe_residual_copy_floor,
    build_moe_residual_titan,
    COPY_FLOOR_ARM,
    MCORE_BASE_ARM,
    MCORE_NO_FUSION_ARM,
    MoeResidualInputs,
    moe_residual_inputs,
    moe_residual_reference,
    NO_BIAS_DROPOUT_FUSION,
    TITAN_ARM,
)
from benchmarks.kernel.schema import fragment_stem, KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE
from benchmarks.models.piper_qwen3.megatron_bootstrap import megatron_dir
from benchmarks.models.piper_qwen3.shape import PiperShape, shape_by_name

# Small enough to run in milliseconds, and wide enough that a lost row or a
# transposed view shows up rather than cancelling.
TINY = PiperShape(name="tiny", dim=256, n_layers=2, vocab_size=64)
TINY_WORKLOAD = KernelWorkload(batch=2, seq_len=16)

MEGATRON = megatron_dir()
FUSED_BIAS_DROPOUT = MEGATRON / "megatron/core/fusions/fused_bias_dropout.py"
TRANSFORMER_LAYER = MEGATRON / "megatron/core/transformer/transformer_layer.py"
JIT = MEGATRON / "megatron/core/jit.py"
GPT_LAYER_SPECS = MEGATRON / "megatron/core/models/gpt/gpt_layer_specs.py"
QWEN3_MODEL = TITAN_DIR / "torchtitan/models/qwen3/model.py"


def _inputs(shape=TINY, workload=TINY_WORKLOAD, seed=0) -> MoeResidualInputs:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return moe_residual_inputs(shape, workload, torch.device("cpu"), generator)


def _rel_l2(value: torch.Tensor, truth: torch.Tensor) -> float:
    delta = (value.double() - truth.double()).norm()
    return (delta / truth.double().norm()).item()


def _source(path: Path) -> str:
    if not path.exists():
        raise unittest.SkipTest(f"{path} is absent; the submodule is not checked out")
    return path.read_text()


class ProfileTests(unittest.TestCase):
    """The one delta this scenario declares, and the premises it leaves alone."""

    def test_the_base_profile_declares_the_fusion_on(self) -> None:
        self.assertIs(BASE.config_overrides["bias_dropout_fusion"], True)

    def test_the_variant_turns_exactly_one_field_off(self) -> None:
        """A delta that moved a second field would confound the row.

        The two arms are one flag apart by construction. Anything else in the
        diff makes the published ratio a comparison of two configurations
        rather than of two implementations of one add.
        """
        differing = {
            key
            for key in set(BASE.config_overrides) | set(
                NO_BIAS_DROPOUT_FUSION.config_overrides
            )
            if BASE.config_overrides.get(key)
            != NO_BIAS_DROPOUT_FUSION.config_overrides.get(key)
        }
        self.assertEqual(differing, {"bias_dropout_fusion"})
        self.assertIs(
            NO_BIAS_DROPOUT_FUSION.config_overrides["bias_dropout_fusion"], False
        )

    def test_the_variant_is_recordable_provenance(self) -> None:
        """A profile must reach a manifest, so it stays JSON-safe data."""
        described = NO_BIAS_DROPOUT_FUSION.describe()
        self.assertEqual(described["name"], "no_bias_dropout_fusion")
        self.assertIn("unfused", described["description"])
        self.assertIs(described["config_overrides"]["bias_dropout_fusion"], False)

    def test_the_base_profile_reduces_the_operation_to_a_plain_add(self) -> None:
        """The three fields the "same add" argument rests on.

        A profile that lost any of them would leave megatron computing a
        dropout or a bias add that titan has no counterpart for, and the
        cross-engine correctness gate would be comparing two operations.
        """
        self.assertEqual(BASE.config_overrides["hidden_dropout"], 0.0)
        self.assertEqual(BASE.config_overrides["attention_dropout"], 0.0)
        self.assertIs(BASE.config_overrides["add_bias_linear"], False)


class DropoutIdentityTests(unittest.TestCase):
    """``p=0.0`` is an identity by construction, not by luck."""

    def test_a_zero_probability_dropout_returns_its_own_input(self) -> None:
        """Object identity, so no rounding and no RNG draw is involved.

        Statistical equality would still leave the operation nondeterministic
        across runs. This is stronger, and it is what lets the cross-engine
        gate ask for bit-identity.
        """
        x = torch.randn(4, 8)
        self.assertIs(F.dropout(x, p=0.0, training=True), x)

    def test_the_no_bias_branch_is_bitwise_the_residual_add(self) -> None:
        """megatron's branch, transcribed, against titan's expression.

        The transcription is deliberate: importing megatron to run four lines
        costs 9 s. The source pin below is what keeps the transcription
        honest.
        """
        x = torch.randn(4, 1, 8)
        residual = torch.randn(4, 1, 8)
        out = F.dropout(x, p=0.0, training=True, inplace=False)
        out = residual + out
        self.assertTrue(torch.equal(out, _residual_add(x, residual)))


class InputsTests(unittest.TestCase):
    def test_the_inputs_are_three_canonical_bsd_tensors(self) -> None:
        inputs = _inputs()
        expected = (TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len, TINY.dim)
        for name in ("x", "residual", "grad_out"):
            with self.subTest(tensor=name):
                tensor = getattr(inputs, name)
                self.assertEqual(tuple(tensor.shape), expected)
                self.assertEqual(tensor.dtype, torch.bfloat16)
                self.assertTrue(tensor.is_contiguous())

    def test_the_two_operands_are_separate_draws(self) -> None:
        """An aliased pair would hide an arm that added a tensor to itself."""
        inputs = _inputs()
        self.assertFalse(torch.equal(inputs.x, inputs.residual))
        self.assertFalse(torch.equal(inputs.x, inputs.grad_out))

    def test_bytes_moved_is_two_reads_and_one_write(self) -> None:
        inputs = _inputs()
        self.assertEqual(inputs.bytes_moved, 3 * inputs.x.numel() * 2)

    def test_one_seed_rebuilds_the_inputs_bit_identically(self) -> None:
        """Every worker rebuilds these, and the gates compare across workers."""
        first, second = _inputs(seed=7), _inputs(seed=7)
        self.assertTrue(torch.equal(first.x, second.x))
        self.assertTrue(torch.equal(first.residual, second.residual))
        self.assertTrue(torch.equal(first.grad_out, second.grad_out))


class ReferenceTests(unittest.TestCase):
    def test_the_reference_is_the_fp64_add(self) -> None:
        inputs = _inputs()
        reference = moe_residual_reference(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(sorted(reference), ["out", "residual_grad", "x_grad"])
        self.assertTrue(
            torch.equal(
                reference["out"], inputs.residual.double() + inputs.x.double()
            )
        )

    def test_the_incoming_gradient_reaches_both_operands_unchanged(self) -> None:
        """An add routes its gradient, and this is what proves it still does.

        A live dropout would scale one branch and a bias would not; either
        would break this equality before it broke the forward one.
        """
        inputs = _inputs()
        reference = moe_residual_reference(TINY, TINY_WORKLOAD, inputs)
        truth = inputs.grad_out.double()
        self.assertTrue(torch.equal(reference["x_grad"], truth))
        self.assertTrue(torch.equal(reference["residual_grad"], truth))


class ArmClosureTests(unittest.TestCase):
    """The closures all three arms share, over a stand-in call."""

    def _arm(self, inputs: MoeResidualInputs, native=None):
        def call(x, residual):
            return residual + x

        shape = tuple(inputs.x.shape) if native is None else native
        return _residual_arm(
            name="stand_in",
            call=call,
            x_native=inputs.x.reshape(shape),
            residual_native=inputs.residual.reshape(shape),
            grad_native=inputs.grad_out.reshape(shape),
            canonical=tuple(inputs.x.shape),
            bytes_moved=inputs.bytes_moved,
        )

    def test_the_arm_declares_forward_and_forward_backward_only(self) -> None:
        """No isolated backward: an add's backward launches no kernel."""
        self.assertEqual(
            sorted(self._arm(_inputs()).calls), ["forward", "forward_backward"]
        )

    def test_the_forward_closure_builds_a_graph(self) -> None:
        """The forward mode carries autograd, as production does.

        A detached forward would measure a cheaper operation than the one the
        engines run, and it would leave ``forward_backward`` and both gradient
        gates with no graph to walk.
        """
        out = self._arm(_inputs()).calls["forward"]()
        self.assertIsNotNone(out.grad_fn)

    def test_the_named_outputs_match_the_fp64_reference(self) -> None:
        inputs = _inputs()
        outputs = self._arm(inputs).correctness_outputs()
        reference = moe_residual_reference(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(sorted(outputs), ["out", "residual_grad", "x_grad"])
        for name in outputs:
            with self.subTest(output=name):
                self.assertLess(_rel_l2(outputs[name], reference[name]), 2e-2)

    def test_a_thd_arm_returns_canonical_outputs(self) -> None:
        """The mcore arms run ``[t, 1, D]`` and the gates compare ``[B, L, D]``.

        Without this the cross-engine check would fail on shape rather than on
        numbers, and a reshape that reordered elements would pass it.
        """
        inputs = _inputs()
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        outputs = self._arm(inputs, native=(tokens, 1, TINY.dim)).correctness_outputs()
        reference = moe_residual_reference(TINY, TINY_WORKLOAD, inputs)
        for name in outputs:
            with self.subTest(output=name):
                self.assertEqual(tuple(outputs[name].shape), tuple(inputs.x.shape))
                self.assertLess(_rel_l2(outputs[name], reference[name]), 2e-2)

    def test_a_repeated_round_trip_does_not_accumulate_a_gradient(self) -> None:
        """Every timed call must measure one backward, not a growing sum."""
        arm = self._arm(_inputs())
        round_trip = arm.calls["forward_backward"]
        round_trip()
        first = arm.correctness_outputs()["x_grad"].clone()
        for _ in range(3):
            round_trip()
        self.assertTrue(torch.equal(first, arm.correctness_outputs()["x_grad"]))

    def test_the_titan_arm_declares_the_shared_modes_and_bytes(self) -> None:
        """The compiled titan arm, built end to end on CPU."""
        inputs = _inputs()
        arm = build_moe_residual_titan(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(arm.name, TITAN_ARM)
        self.assertEqual(sorted(arm.calls), ["forward", "forward_backward"])
        self.assertEqual(arm.bytes_moved, inputs.bytes_moved)

    def test_the_titan_arm_runs_and_matches_the_reference(self) -> None:
        """The one arm a CPU can build is also run here.

        It is the only executable evidence outside a GPU that the compiled
        closure computes the scenario's operation. Both mcore arms need a
        device, a process group and TransformerEngine.
        """
        inputs = _inputs()
        arm = build_moe_residual_titan(TINY, TINY_WORKLOAD, inputs)
        arm.calls["forward_backward"]()
        outputs = arm.correctness_outputs()
        reference = moe_residual_reference(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(sorted(outputs), ["out", "residual_grad", "x_grad"])
        for name in outputs:
            with self.subTest(output=name):
                self.assertLess(_rel_l2(outputs[name], reference[name]), 2e-2)

    def test_the_floor_declares_forward_alone_and_the_shared_traffic(
        self,
    ) -> None:
        """A floor for the backward traffic would be an invention."""
        inputs = _inputs()
        arm = build_moe_residual_copy_floor(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(arm.name, COPY_FLOOR_ARM)
        self.assertEqual(sorted(arm.calls), ["forward"])
        self.assertEqual(arm.bytes_moved, inputs.bytes_moved)
        self.assertEqual(arm.correctness_outputs(), {})

    def test_no_arm_makes_the_shared_operands_require_grad(self) -> None:
        """The floor's operands are the scenario's shared tensors.

        ``_residual_arm`` clones before it calls ``requires_grad_``. Were it to
        mark the shared tensors in place, a floor built after it in the same
        interpreter would start building a graph per call and would stop being
        a floor -- silently, and only in the correctness pass, which is the one
        pass that builds every arm together.
        """
        inputs = _inputs()
        self._arm(inputs)
        build_moe_residual_titan(TINY, TINY_WORKLOAD, inputs)
        self.assertFalse(inputs.x.requires_grad)
        self.assertFalse(inputs.residual.requires_grad)
        self.assertIsNone(
            build_moe_residual_copy_floor(TINY, TINY_WORKLOAD, inputs).calls[
                "forward"
            ]()
        )


class GuardTests(unittest.TestCase):
    def test_a_missing_gradient_is_named_rather_than_dereferenced(self) -> None:
        """An operation that ignored an operand is the failure this catches."""
        with self.assertRaisesRegex(RuntimeError, "residual_grad, x_grad"):
            _require_grads(
                "stand_in",
                {"out": torch.zeros(1), "x_grad": None, "residual_grad": None},
            )

    def test_a_complete_set_passes_through_unchanged(self) -> None:
        outputs = {
            "out": torch.zeros(1),
            "x_grad": torch.ones(1),
            "residual_grad": torch.ones(1),
        }
        self.assertIs(_require_grads("stand_in", outputs), outputs)


class PinnedSourceTests(unittest.TestCase):
    """The lines on the pinned revisions that the two arms depend on.

    Each of these is a fact the module's docstring states. A submodule bump
    that moves one is exactly the event that has to re-open this scenario, so
    the failure is the signal rather than the nuisance.
    """

    def test_megatron_still_spells_the_no_bias_branch_as_a_residual_add(self) -> None:
        source = _source(FUSED_BIAS_DROPOUT)
        self.assertIn(
            "out = torch.nn.functional.dropout(x, p=prob, training=training, "
            "inplace=inplace)",
            source,
        )
        self.assertIn("out = residual + out", source)

    def test_the_fused_train_function_still_carries_the_jit_fuser(self) -> None:
        """``mcore/base`` declares ``compiled=True`` on the strength of this."""
        self.assertIn(
            "@jit_fuser\ndef bias_dropout_add_fused_train(", _source(FUSED_BIAS_DROPOUT)
        )

    def test_the_unfused_path_is_a_closure_rather_than_a_module_function(
        self,
    ) -> None:
        """``mcore/no_bias_dropout_fusion`` allocates one per call, on purpose.

        The builder resolves the operation inside the timed closure because
        megatron does. Hoisting it would hand the eager arm a saving the
        engine does not give it.
        """
        self.assertIn(
            "def bias_dropout_add_unfused(training):\n"
            "    def _bias_dropout_add(x_with_bias, residual, prob):",
            _source(FUSED_BIAS_DROPOUT),
        )

    def test_megatron_still_maps_its_jit_fuser_onto_torch_compile(self) -> None:
        self.assertIn("jit_fuser = torch.compile", _source(JIT))

    def test_the_call_site_still_reads_the_flag_per_call(self) -> None:
        """The delta is delivered by the config field and by nothing else."""
        self.assertIn(
            "hidden_states = self.mlp_bda(self.training, "
            "self.config.bias_dropout_fusion)(\n"
            "                    mlp_output_with_bias, residual, self.hidden_dropout\n"
            "                )",
            _source(TRANSFORMER_LAYER),
        )

    def test_the_submodule_field_still_defaults_to_the_identity(self) -> None:
        """The hazard ``_assert_mcore_bda``'s first check exists for."""
        self.assertIn(
            "mlp_bda: Union[ModuleSpec, type] = IdentityFuncOp",
            _source(TRANSFORMER_LAYER),
        )

    def test_every_spec_branch_still_wires_the_real_operation(self) -> None:
        """Not just the branch this build takes -- every one of them.

        The assertion is deliberately branch-blind, because
        ``_assert_mcore_bda`` check 1 is too: ``mlp_bda is
        get_bias_dropout_add`` holds in all seven branches, so neither the
        guard nor this test identifies which branch ran. What both establish
        is that no branch leaves the field at its ``IdentityFuncOp`` default.
        """
        source = _source(GPT_LAYER_SPECS)
        self.assertEqual(source.count("mlp_bda="), 7)
        self.assertEqual(source.count("mlp_bda=get_bias_dropout_add"), 7)

    def test_the_moe_reflatten_runs_before_the_residual_add_in_the_base_layer(
        self,
    ) -> None:
        """So the flatten glue is outside the cut, on megatron's side too.

        ``_forward_post_mlp`` holds the ``mlp_bda`` call. If the reflatten ever
        moved after it, the arms would have to reproduce the glue to stay
        faithful, and the cut would no longer be the add alone.

        The slice below starts at the **first** ``_forward_mlp``, so it reads
        ``TransformerLayer`` alone. The order is reversed in
        ``MoETransformerLayer``, which reflattens after the bda; the next test
        is what establishes that this build never instantiates that class.
        """
        source = _source(TRANSFORMER_LAYER)
        body = source[source.index("    def _forward_mlp(") :]
        body = body[: body.index("    def _forward_post_mlp(")]
        self.assertIn("self._maybe_reflatten_from_moe(", body)
        self.assertIn("return self._forward_post_mlp(", body)
        self.assertLess(
            body.index("self._maybe_reflatten_from_moe("),
            body.index("return self._forward_post_mlp("),
        )

    def test_this_build_instantiates_the_base_transformer_layer(self) -> None:
        """Which decides which of the two orderings above applies.

        ``MoETransformerLayer`` reaches a model only through one assignment,
        and only under one argument this scenario never passes.
        """
        from benchmarks.models.piper_qwen3 import megatron_model

        signature = inspect.signature(megatron_model.build_model)
        self.assertIsNone(signature.parameters["cuda_graph_impl"].default)

        builder = _source(Path(megatron_model.__file__))
        guard = builder.index('if cuda_graph_impl == "local":')
        end = builder.index("    model = GPTModel(")
        swap = builder[guard:end]
        self.assertIn("import MoETransformerLayer", swap)
        self.assertIn("layer_spec.module = MoETransformerLayer", swap)
        self.assertNotIn("MoETransformerLayer", builder[:guard])
        self.assertNotIn("MoETransformerLayer", builder[end:])

        operations = _source(
            Path(__file__).resolve().parent.parent
            / "benchmarks/kernel/operations/moe_residual.py"
        )
        self.assertEqual(operations.count("build_model("), 1)
        self.assertIn(
            "build_model(seq_len=workload.seq_len, shape=shape, profile=profile)",
            operations,
        )

    def test_titan_still_writes_the_residual_add_the_titan_arm_reproduces(
        self,
    ) -> None:
        """The only evidence a CPU can offer that the titan arm cuts here.

        The arm imports no torchtitan symbol, because there is none to import.
        """
        self.assertIn("x = x + self.moe(", _source(QWEN3_MODEL))


class RegisteredShapeTests(unittest.TestCase):
    def test_the_inputs_build_at_every_registered_model_size(self) -> None:
        """``--model-size`` is single-valued but not fixed, so both must work."""
        for name in ("normal", "huge"):
            with self.subTest(size=name):
                shape = shape_by_name(name)
                workload = KernelWorkload(batch=1, seq_len=2)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(0)
                inputs = moe_residual_inputs(
                    shape, workload, torch.device("cpu"), generator
                )
                self.assertEqual(inputs.x.shape[-1], shape.dim)
                self.assertEqual(tuple(inputs.residual.shape), tuple(inputs.x.shape))


class ArmNameTests(unittest.TestCase):
    """**Not a registry test, and it cannot be one yet.**

    The ``moe_residual`` scenario is not in ``benchmarks/kernel/registry.py``:
    it lands as a declaration fragment that a merge agent pastes, because
    several scenarios are written in parallel and one file cannot take four
    concurrent edits. So these assertions compare the module's own constants
    against the names the fragment spells. **After the merge, replace them
    with assertions against** ``KERNEL_SCENARIOS["moe_residual"]``, which is
    the only version of this test that can catch a divergence.
    """

    def test_each_builder_returns_the_arm_name_the_fragment_declares(
        self,
    ) -> None:
        inputs = _inputs()
        self.assertEqual(
            build_moe_residual_copy_floor(TINY, TINY_WORKLOAD, inputs).name,
            COPY_FLOOR_ARM,
        )
        self.assertEqual(
            build_moe_residual_titan(TINY, TINY_WORKLOAD, inputs).name, TITAN_ARM
        )

    def test_the_four_names_are_unique_and_survive_the_fragment_stem(
        self,
    ) -> None:
        """A stem collision would make two arms overwrite one fragment file.

        ``schema.fragment_stem`` replaces the slash, so two arms whose names
        differ only there would collide. ``KernelScenario.__post_init__``
        raises on that, and this is the CPU-side statement of the same
        invariant while the scenario is still a fragment.
        """
        names = [COPY_FLOOR_ARM, MCORE_BASE_ARM, MCORE_NO_FUSION_ARM, TITAN_ARM]
        self.assertEqual(len(set(names)), 4)
        self.assertEqual(len({fragment_stem(name) for name in names}), 4)


if __name__ == "__main__":
    unittest.main()
