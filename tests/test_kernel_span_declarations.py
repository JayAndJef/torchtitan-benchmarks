"""The spans this repository declares, and what each one claims.

``tests/test_kernel_spans.py`` proves the span **mechanism** against a
synthetic span. This file is the other half: it reads the real roster in
``benchmarks/kernel/spans.py`` and asks whether each declaration says a true
thing about the scenarios it encloses.

The split matters because the two fail for different reasons. A mechanism
test fails when the type or the merge is wrong. A declaration test fails when
a scenario is renamed, an arm is re-homed, or a span's stated reason stops
matching the registry -- which is what happened to three kernel scenarios in
one day, and is why every name here is checked against the live registry
rather than against the plan that proposed it.

**No builder exists for any span yet**, so nothing here builds an arm and
nothing here needs a GPU. A builder path is a dotted string resolved inside
the worker; these tests read the declaration, which is the whole of what a
span is until somebody writes the arms.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kernel.registry import KERNEL_SCENARIOS
from benchmarks.kernel.schema import KernelSpan
from benchmarks.kernel.spans import KERNEL_SPANS, _validate_roster


# Every span declared at this rev, in declaration order. The pin is the
# point: a span added without a line here is a span nobody described.
DECLARED_SPANS = ("expert_combine", "attn_residual_norm")

# How many enclosed scenarios each range holds, spelled as the description
# spells it. The dispatch-chain bias grows with this number, so a reader must
# meet the number in the caption and not only in the range.
CHAIN_WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six"}


def engine_of(arm_name: str) -> str:
    """``mcore``, ``titan`` or ``other`` for one arm name.

    Arms are named ``engine`` or ``engine/profile`` across both registries,
    so the first component is the engine. ``copy_floor`` is neither.
    """
    head = arm_name.split("/", 1)[0]
    return head if head in ("mcore", "titan") else "other"


def crosses_engines(pair: tuple[str, str]) -> bool:
    arm, opponent = pair
    return (
        engine_of(arm) != engine_of(opponent)
        and "other" not in (engine_of(arm), engine_of(opponent))
    )


class SpanRosterTests(unittest.TestCase):
    """What is declared, and that every name in it is live."""

    def test_the_roster_is_exactly_the_declared_spans(self) -> None:
        self.assertEqual(tuple(KERNEL_SPANS), DECLARED_SPANS)

    def test_the_declared_roster_passes_its_own_validation(self) -> None:
        """Already run at import; asserted here so the failure is named."""
        _validate_roster(KERNEL_SPANS, KERNEL_SCENARIOS)

    def test_a_span_named_after_a_scenario_is_refused(self) -> None:
        """Otherwise one run measures that name twice.

        Once as a scenario, without its parts, and once as the span, with
        them. ``--scenario`` and ``--span`` would each accept it, and the two
        results files would disagree about what the name means.
        """
        span = KERNEL_SPANS["expert_combine"]
        with self.assertRaises(ValueError) as caught:
            _validate_roster({"expert_mlp": span}, KERNEL_SCENARIOS)
        self.assertIn("expert_mlp", str(caught.exception))
        self.assertIn("disjoint", str(caught.exception))

    def test_every_span_names_only_live_scenarios(self) -> None:
        for span in KERNEL_SPANS.values():
            for name in span.scenarios:
                with self.subTest(span=span.name, scenario=name):
                    self.assertIn(name, KERNEL_SCENARIOS)

    def test_every_part_arm_is_live_in_the_scenario_it_names(self) -> None:
        """A re-homed arm is the way a span goes stale without a rename."""
        for span in KERNEL_SPANS.values():
            for arm in span.arms:
                for scenario_name, part in span.parts_for(arm.name):
                    with self.subTest(span=span.name, part=part):
                        # Raises with the roster in the message if absent.
                        KERNEL_SCENARIOS[scenario_name].arm(part)

    def test_every_part_declares_every_mode_its_span_arm_declares(
        self,
    ) -> None:
        """Or that mode's parts total sums fewer terms than the range holds.

        A short sum is not a missing row: it is a smaller number in the
        denominator of a published ratio.
        """
        for span in KERNEL_SPANS.values():
            for arm in span.arms:
                for scenario_name, part in span.parts_for(arm.name):
                    part_arm = KERNEL_SCENARIOS[scenario_name].arm(part)
                    with self.subTest(span=span.name, part=part):
                        self.assertEqual(
                            [
                                mode
                                for mode in arm.modes
                                if mode not in part_arm.modes
                            ],
                            [],
                        )

    def test_every_span_arm_names_what_it_replaces(self) -> None:
        """One part per enclosed scenario, positionally, for every arm."""
        for span in KERNEL_SPANS.values():
            for arm in span.arms:
                with self.subTest(span=span.name, arm=arm.name):
                    self.assertEqual(
                        [name for name, _ in span.parts_for(arm.name)],
                        list(span.scenarios),
                    )

    def test_every_span_arm_carries_a_description(self) -> None:
        for span in KERNEL_SPANS.values():
            self.assertTrue(span.description.strip(), span.name)
            for arm in span.arms:
                with self.subTest(span=span.name, arm=arm.name):
                    self.assertTrue(arm.description.strip())

    def test_every_builder_names_its_own_spans_operations_module(
        self,
    ) -> None:
        """The rule the scenarios already keep, applied to the spans.

        A builder left in another unit's module still resolves, so nothing
        would fail -- but ``resolve_symbol`` imports the whole module to
        reach one function, and a span's arms would then drag another unit's
        imports into the worker that measures them.
        """
        for span in KERNEL_SPANS.values():
            expected = f"benchmarks.kernel.operations.{span.name}"
            paths = [span.measurement.inputs_builder]
            if span.measurement.reference_builder is not None:
                paths.append(span.measurement.reference_builder)
            paths.extend(arm.builder for arm in span.arms)
            for path in paths:
                with self.subTest(span=span.name, path=path):
                    self.assertIn(":", path)
                    self.assertEqual(path.partition(":")[0], expected)

    def test_every_description_states_the_dispatch_chain_bias(self) -> None:
        """The bias favours the span, and it grows with the range.

        The parts total pays one host dispatch chain per enclosed scenario
        and the span pays one, and roughly 85% of a kernel number here is
        host dispatch. The engine states the bias generically under every
        table; a declaration that omitted it would let a reader meet the
        range without meeting its cost.
        """
        for span in KERNEL_SPANS.values():
            with self.subTest(span=span.name):
                self.assertIn(
                    "host dispatch chain per enclosed scenario",
                    span.description,
                )
                self.assertIn(
                    CHAIN_WORDS[len(span.scenarios)], span.description
                )

    def test_every_description_says_the_claim_is_unpaired(self) -> None:
        """Every scenario ratio is paired. This one is not.

        Roughly 150 workers separate a six-scenario span's replicate 0 from
        its first part's, so the estimator resamples each side on its own.
        A reader who sets that interval beside a scenario's has been told
        the wrong thing about both.
        """
        for span in KERNEL_SPANS.values():
            with self.subTest(span=span.name):
                self.assertIn("UNPAIRED", span.description)
                self.assertIn("unpaired_ratio_ci_", span.description)


class ExpertCombineTests(unittest.TestCase):
    """11+12: the only cut at which the two engines are comparable here."""

    def span(self) -> KernelSpan:
        return KERNEL_SPANS["expert_combine"]

    def test_it_replaces_the_expert_cut_and_the_combine_cut(self) -> None:
        self.assertEqual(
            self.span().scenarios, ("expert_mlp", "moe_combine")
        )

    def test_each_arm_replaces_its_own_engine_at_both_cuts(self) -> None:
        self.assertEqual(
            self.span().parts_for("mcore/base"),
            (("expert_mlp", "mcore/base"), ("moe_combine", "mcore/base")),
        )
        self.assertEqual(
            self.span().parts_for("titan"),
            (("expert_mlp", "titan"), ("moe_combine", "titan")),
        )

    def test_it_publishes_the_cross_engine_row_both_scenarios_decline(
        self,
    ) -> None:
        """The reason the span exists, stated as an assertion.

        The routing probabilities land on opposite sides of the 11/12
        boundary: megatron folds them into the fused activation kernel
        inside ``TEGroupedMLP`` and TorchTitan applies them in combine. So
        neither enclosed scenario may publish a cross-engine row, and this
        span is the smallest enclosure in which both engines have applied
        them exactly once.
        """
        self.assertEqual(
            self.span().comparison_pairs(), (("titan", "mcore/base"),)
        )
        for name in self.span().scenarios:
            with self.subTest(scenario=name):
                self.assertEqual(
                    [
                        pair
                        for pair in KERNEL_SCENARIOS[name].comparison_pairs()
                        if crosses_engines(pair)
                    ],
                    [],
                )

    def test_the_gate_crosses_the_boundary_the_two_scenarios_refuse(
        self,
    ) -> None:
        """A cross-engine ratio needs a cross-engine gate under it.

        Inside ``expert_mlp`` and ``moe_combine`` the two engines carry
        different correctness output names on purpose, so no gate can cross
        either cut. Here they carry the same names and the titan arm is
        gated against ``mcore/base``: if the two engines do not agree at
        this enclosure, the enclosure is wrong and the ratio means nothing.
        """
        titan = self.span().arm("titan")
        references = [check.reference for check in titan.correctness]
        self.assertIn("mcore/base", references)
        self.assertIn("fp64", references)
        cross = next(
            check
            for check in titan.correctness
            if check.reference == "mcore/base"
        )
        mcore = self.span().arm("mcore/base")
        self.assertEqual(
            set(cross.outputs), set(mcore.correctness[0].outputs)
        )
        self.assertFalse(cross.informational)

    def test_it_declares_the_balanced_routing_its_scenarios_declare(
        self,
    ) -> None:
        """The span builds its own routed rows, so it needs the same split."""
        self.assertTrue(self.span().measurement.requires_balanced_routing)
        for name in self.span().scenarios:
            with self.subTest(scenario=name):
                self.assertTrue(
                    KERNEL_SCENARIOS[name].requires_balanced_routing
                )

    def test_both_arms_declare_the_treatment_their_engine_runs(self) -> None:
        """Megatron compiles no whole layer; every titan module arm does.

        The published row therefore compares two compile treatments as well
        as two engines, which is why each arm has to state its own.
        """
        self.assertFalse(self.span().arm("mcore/base").compiled)
        self.assertTrue(
            self.span().arm("mcore/base").eager_reason.strip()
        )
        self.assertTrue(self.span().arm("titan").compiled)
        self.assertIsNone(self.span().arm("titan").eager_reason)



class AttnResidualNormTests(unittest.TestCase):
    """6+7+8: the cut that lands on the side the residual add fuses to."""

    def span(self) -> KernelSpan:
        return KERNEL_SPANS["attn_residual_norm"]

    def test_it_replaces_the_projection_the_add_and_the_next_norm(
        self,
    ) -> None:
        self.assertEqual(
            self.span().scenarios,
            ("attn_out_proj", "attn_residual", "ffn_norm"),
        )

    def test_each_arm_replaces_its_own_engine_at_all_three_cuts(
        self,
    ) -> None:
        for arm in ("mcore/base", "titan"):
            with self.subTest(arm=arm):
                self.assertEqual(
                    [part for _, part in self.span().parts_for(arm)],
                    [arm] * 3,
                )

    def test_the_range_ends_at_the_norm_the_add_fuses_into(self) -> None:
        """A 6+7 span would cut on the wrong side of the residual add.

        TorchTitan's add fuses FORWARD, into the prologue of the next norm,
        and not backward into the epilogue of the projection GEMM. Ending
        the range at ``attn_residual`` leaves both engines emitting a GEMM
        and then a separate add, which is the shape that puts a ratio near
        1.0 -- and it is exactly why ``attn_residual`` alone publishes no
        cross-engine row.
        """
        self.assertEqual(self.span().scenarios[-1], "ffn_norm")
        self.assertEqual(
            [
                pair
                for pair in KERNEL_SCENARIOS["attn_residual"].comparison_pairs()
                if crosses_engines(pair)
            ],
            [],
        )

    def test_it_publishes_the_cross_engine_row_the_middle_cut_declines(
        self,
    ) -> None:
        self.assertEqual(
            self.span().comparison_pairs(), (("titan", "mcore/base"),)
        )

    def test_the_gate_crosses_the_engines(self) -> None:
        titan = self.span().arm("titan")
        references = [check.reference for check in titan.correctness]
        self.assertIn("mcore/base", references)
        self.assertIn("fp64", references)

    def test_the_two_weights_carry_distinct_gradient_names(self) -> None:
        """``weight_grad`` names one tensor, and this cut crosses two.

        ``attn_out_proj`` and ``ffn_norm`` each gate a ``weight_grad``. A
        span that reused the name would gate whichever of the two the
        builder returned, and the other would go unchecked.
        """
        outputs = self.span().arm("titan").correctness[0].outputs
        self.assertIn("proj_weight_grad", outputs)
        self.assertIn("norm_weight_grad", outputs)
        self.assertNotIn("weight_grad", outputs)

    def test_the_megatron_arm_is_eager_where_it_is_timed_from(self) -> None:
        """And its middle part is not, which is not a contradiction.

        ``attn_residual/mcore/base`` declares ``compiled=True`` because its
        timed closure calls ``bias_dropout_add_fused_train`` directly, and
        that function IS megatron's ``torch.compile`` wrapper. Here the same
        function is one call inside a plain Python closure, so the arm is
        eager at its entry point while the region inside it still compiles.
        The field describes the entry point, and the treatment of the region
        is the same on both sides of the comparison.
        """
        arm = self.span().arm("mcore/base")
        self.assertFalse(arm.compiled)
        self.assertTrue(
            KERNEL_SCENARIOS["attn_residual"].arm("mcore/base").compiled
        )


if __name__ == "__main__":
    unittest.main()
