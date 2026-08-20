"""The span engine: the declaration type, the results slot, the merge.

A **span** is an implementation that fuses across a scenario cut. It belongs
to no single scenario, so it is declared over an ordered scenario range and
its claim is the span against the **sum of the scenarios it replaces**. Two
totals, and a reader must be able to see both.

**No span is declared in ``benchmarks/kernel/spans.py`` at this rev**, and
declaring the five that the cross-engine partition wants is a separate piece
of work. So the span under test here is synthetic: it is declared in this
file, over two scenarios the registry really holds, naming arms those
scenarios really declare. That is deliberate. A test that built its own
scenarios too would prove the span type agrees with itself and nothing about
whether a span can be declared over the real partition.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kernel.registry import KERNEL_SCENARIOS
from benchmarks.kernel.schema import (
    KernelArm,
    KernelScenario,
    KernelSpan,
    SpanParts,
    validate_span_parts,
)
from benchmarks.kernel.spans import KERNEL_SPANS, kernel_span_by_name


# The span's own head-to-head. Its arms name builders that do not exist:
# nothing here builds an arm, and a builder path is a string resolved inside
# the GPU worker, which is the property that lets a declaration be tested
# without torch.
def span_arm(name: str, modes=("forward", "forward_backward")) -> KernelArm:
    return KernelArm(
        name=name,
        description=f"synthetic {name}",
        builder=f"tests.does_not_exist:{name.replace('/', '_')}",
        modes=modes,
        compiled=True,
    )


def span_measurement(
    arms=("mcore/base", "titan"), baseline: str = "mcore/base"
) -> KernelScenario:
    return KernelScenario(
        name="test_expert_combine",
        description="synthetic span over the expert and combine cuts",
        inputs_builder="tests.does_not_exist:inputs",
        reference_builder=None,
        arms=tuple(span_arm(name) for name in arms),
        baseline_arm=baseline,
    )


def make_span(**overrides) -> KernelSpan:
    """A span over ``expert_mlp`` + ``moe_combine``, both real scenarios."""
    fields = {
        "measurement": span_measurement(),
        "scenarios": ("expert_mlp", "moe_combine"),
        "parts": (
            SpanParts(arm="mcore/base", parts=("mcore/base", "mcore/base")),
            SpanParts(arm="titan", parts=("titan", "titan")),
        ),
    }
    fields.update(overrides)
    return KernelSpan(**fields)


class SpanDeclarationTests(unittest.TestCase):
    """What a span declares, and what it refuses to declare."""

    def test_a_span_names_an_ordered_scenario_range(self) -> None:
        span = make_span()
        self.assertEqual(span.scenarios, ("expert_mlp", "moe_combine"))
        self.assertEqual(span.name, "test_expert_combine")

    def test_the_parts_correspondence_is_positional_and_readable(self) -> None:
        """``parts_for`` pairs each part with the cut it covers, in order."""
        span = make_span()
        self.assertEqual(
            span.parts_for("titan"),
            (("expert_mlp", "titan"), ("moe_combine", "titan")),
        )

    def test_a_part_arm_need_not_share_the_span_arms_name(self) -> None:
        """The ``fused_linear_ce`` shape: two parts, neither named like the arm.

        A name-based correspondence could not express this at all, which is
        why the correspondence is declared.
        """
        span = KernelSpan(
            measurement=KernelScenario(
                name="test_fused_linear_ce",
                description="synthetic titan-only span",
                inputs_builder="tests.does_not_exist:inputs",
                reference_builder=None,
                arms=(span_arm("titan/fused"),),
                baseline_arm="titan/fused",
            ),
            scenarios=("lm_head_projection", "cross_entropy"),
            parts=(
                SpanParts(
                    arm="titan/fused", parts=("titan", "titan/full_logits")
                ),
            ),
        )
        validate_span_parts(span, KERNEL_SCENARIOS)
        self.assertEqual(
            span.parts_for("titan/fused"),
            (
                ("lm_head_projection", "titan"),
                ("cross_entropy", "titan/full_logits"),
            ),
        )

    def test_a_single_engine_span_over_cross_engine_scenarios(self) -> None:
        """The ``ffn_norm_to_moe_residual`` shape.

        Both arms are mcore, both enclosed scenarios are cross-engine, and
        the enclosed ``titan`` arms are named by no ``SpanParts``. Nothing
        special happens, because the parts are declared rather than derived
        from the engine.
        """
        span = KernelSpan(
            measurement=span_measurement(
                arms=("mcore/base", "mcore/fused"), baseline="mcore/base"
            ),
            scenarios=("expert_mlp", "moe_combine"),
            parts=(
                SpanParts(
                    arm="mcore/base", parts=("mcore/base", "mcore/base")
                ),
                SpanParts(
                    arm="mcore/fused", parts=("mcore/base", "mcore/base")
                ),
            ),
        )
        validate_span_parts(span, KERNEL_SCENARIOS)
        summed = {
            arm for _, arm in span.parts_for("mcore/fused")
        }
        self.assertEqual(summed, {"mcore/base"})

    def test_a_span_over_one_scenario_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            make_span(
                scenarios=("expert_mlp",),
                parts=(
                    SpanParts(arm="mcore/base", parts=("mcore/base",)),
                    SpanParts(arm="titan", parts=("titan",)),
                ),
            )
        self.assertIn("at least two scenarios", str(caught.exception))

    def test_a_repeated_scenario_in_the_range_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            make_span(
                scenarios=("expert_mlp", "expert_mlp"),
                parts=(
                    SpanParts(
                        arm="mcore/base", parts=("mcore/base", "mcore/base")
                    ),
                    SpanParts(arm="titan", parts=("titan", "titan")),
                ),
            )
        self.assertIn("appears twice", str(caught.exception))

    def test_a_parts_tuple_of_the_wrong_length_is_refused(self) -> None:
        """A short tuple would drop a cut from the sum without saying so."""
        with self.assertRaises(ValueError) as caught:
            make_span(
                parts=(
                    SpanParts(arm="mcore/base", parts=("mcore/base",)),
                    SpanParts(arm="titan", parts=("titan", "titan")),
                )
            )
        self.assertIn("positional", str(caught.exception))

    def test_a_span_arm_with_no_parts_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            make_span(
                parts=(
                    SpanParts(
                        arm="mcore/base", parts=("mcore/base", "mcore/base")
                    ),
                )
            )
        self.assertIn("titan", str(caught.exception))
        self.assertIn("publishes no claim", str(caught.exception))

    def test_parts_declared_twice_for_one_arm_are_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            make_span(
                parts=(
                    SpanParts(
                        arm="mcore/base", parts=("mcore/base", "mcore/base")
                    ),
                    SpanParts(
                        arm="mcore/base",
                        parts=("mcore/no_grouped_gemm", "mcore/base"),
                    ),
                    SpanParts(arm="titan", parts=("titan", "titan")),
                )
            )
        self.assertIn("declares its parts twice", str(caught.exception))

    def test_parts_for_an_undeclared_arm_are_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            make_span(
                parts=(
                    SpanParts(
                        arm="mcore/base", parts=("mcore/base", "mcore/base")
                    ),
                    SpanParts(arm="titan", parts=("titan", "titan")),
                    SpanParts(arm="ghost", parts=("titan", "titan")),
                )
            )
        self.assertIn("Unknown arm 'ghost'", str(caught.exception))

    def test_an_unnamed_part_is_refused(self) -> None:
        with self.assertRaises(ValueError) as caught:
            make_span(
                parts=(
                    SpanParts(arm="mcore/base", parts=("mcore/base", "  ")),
                    SpanParts(arm="titan", parts=("titan", "titan")),
                )
            )
        self.assertIn("has no name", str(caught.exception))


class SpanRegistryCrossCheckTests(unittest.TestCase):
    """``validate_span_parts``: the half that needs the scenario registry."""

    def test_a_declared_span_passes_against_the_real_registry(self) -> None:
        validate_span_parts(make_span(), KERNEL_SCENARIOS)

    def test_an_unknown_scenario_in_the_range_is_refused(self) -> None:
        span = make_span(
            scenarios=("expert_mlp", "no_such_scenario"),
        )
        with self.assertRaises(ValueError) as caught:
            validate_span_parts(span, KERNEL_SCENARIOS)
        self.assertIn("unknown scenario 'no_such_scenario'", str(caught.exception))

    def test_a_part_arm_the_scenario_does_not_declare_is_refused(self) -> None:
        span = make_span(
            parts=(
                SpanParts(arm="mcore/base", parts=("mcore/base", "mcore/base")),
                SpanParts(arm="titan", parts=("titan", "titan/ghost")),
            )
        )
        with self.assertRaises(ValueError) as caught:
            validate_span_parts(span, KERNEL_SCENARIOS)
        self.assertIn("Unknown arm 'titan/ghost'", str(caught.exception))

    def test_a_part_missing_a_mode_the_span_declares_is_refused(self) -> None:
        """Otherwise that mode's total sums fewer terms than the range holds.

        ``expert_mlp/titan`` declares ``backward``; ``moe_combine/titan``
        does not. A span arm declaring ``backward`` would therefore have a
        one-term sum on a two-scenario range, and the row would read as a
        span-versus-parts comparison.
        """
        span = make_span(
            measurement=KernelScenario(
                name="test_expert_combine",
                description="synthetic",
                inputs_builder="tests.does_not_exist:inputs",
                reference_builder=None,
                arms=(
                    span_arm("mcore/base"),
                    span_arm("titan", modes=("forward", "backward")),
                ),
                baseline_arm="mcore/base",
            ),
        )
        with self.assertRaises(ValueError) as caught:
            validate_span_parts(span, KERNEL_SCENARIOS)
        message = str(caught.exception)
        self.assertIn("backward", message)
        self.assertIn("moe_combine/titan", message)


class SpanTypeIsNotAScenarioTests(unittest.TestCase):
    """A span may not be handed to anything that expects a scenario."""

    def test_a_span_is_not_a_kernel_scenario(self) -> None:
        """Composition, not inheritance.

        A subclass would make a span pass every ``isinstance`` check a
        scenario passes, and a span added to ``KERNEL_SCENARIOS`` by mistake
        would run as a bare scenario -- publishing one of its two totals
        under a name that promises both.
        """
        self.assertNotIsInstance(make_span(), KernelScenario)
        self.assertIsInstance(make_span().measurement, KernelScenario)

    def test_the_span_registry_is_empty_and_disjoint_from_the_scenarios(
        self,
    ) -> None:
        """The mechanism landed before any declaration, deliberately.

        The disjointness half is the one that keeps mattering after the five
        spans are declared: a name in both rosters would be measured twice,
        once with its parts and once without.
        """
        self.assertEqual(KERNEL_SPANS, {})
        self.assertEqual(set(KERNEL_SPANS) & set(KERNEL_SCENARIOS), set())

    def test_an_unknown_span_name_names_what_is_available(self) -> None:
        with self.assertRaises(ValueError) as caught:
            kernel_span_by_name("expert_combine")
        self.assertIn("(none declared)", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
