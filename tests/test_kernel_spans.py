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

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.summaries import summarize
from benchmarks.kernel.registry import KERNEL_SCENARIOS
from benchmarks.kernel.results.schema import (
    ArmResult,
    CorrectnessResult,
    KERNEL_RESULTS_SCHEMA_VERSION,
    KernelScenarioResult,
    KernelSpanResult,
    ModeResult,
    SpanPartResult,
    load_kernel_results,
    write_kernel_results,
)
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



def sample_span_result() -> KernelSpanResult:
    """A two-arm span over ``expert_mlp`` + ``moe_combine``.

    The numbers are chosen so the two totals cannot be confused by accident:
    the span's own median is 30.0 and the parts sum to 20.0 + 10.0 = 30.0
    only for ``mcore/base``, while ``titan`` spans 40.0 against a parts sum
    of 25.0 + 25.0.
    """
    span_replicates = ((30.0, 30.0), (30.0, 30.0))
    return KernelSpanResult(
        span="test_expert_combine",
        scenarios=("expert_mlp", "moe_combine"),
        hardware="test-gpu",
        model_size="normal",
        model_shape={"name": "normal", "dim": 1024, "n_layers": 16},
        workload={"batch": 4, "seq_len": 1024},
        shapes={
            "expert_mlp": {"x": [8192, 1024]},
            "moe_combine": {"x": [4, 1024, 1024]},
        },
        replicates=2,
        samples_per_replicate=2,
        burst_k=16,
        warmup_calls=1,
        seed=0,
        arms={
            "mcore/base": ArmResult(
                name="mcore/base",
                modes={
                    "forward": ModeResult(
                        summary=summarize([30.0, 30.0, 30.0, 30.0]),
                        replicates_us=span_replicates,
                        derived={"gbps": float("inf")},
                    )
                },
                peak_memory_gib=1.5,
                compiled=False,
                eager_reason="megatron compiles no whole layer",
            )
        },
        parts={
            "mcore/base": (
                SpanPartResult(
                    scenario="expert_mlp",
                    arm="mcore/base",
                    replicate_medians_us={"forward": (20.0, 20.0)},
                    results_path="out/x/kernels/expert_mlp/hw/results.json",
                ),
                SpanPartResult(
                    scenario="moe_combine",
                    arm="mcore/base",
                    replicate_medians_us={"forward": (10.0, 10.0)},
                    results_path="out/x/kernels/moe_combine/hw/results.json",
                ),
            )
        },
        comparisons=[
            {
                "arm": "mcore/base",
                "opponent": "titan",
                "mode": "forward",
                "median_ratio": 1.2,
                "ratio_ci_low": 1.1,
                "ratio_ci_high": 1.3,
            }
        ],
        parts_comparisons=[
            {
                "arm": "mcore/base",
                "mode": "forward",
                "span_median_us": 30.0,
                "parts_median_us": 30.0,
                "median_ratio": 1.0,
                "cross_sweep_ratio_ci_low": 0.95,
                "cross_sweep_ratio_ci_high": 1.05,
            }
        ],
        correctness=[
            CorrectnessResult(
                arm="mcore/base",
                reference="fp64",
                kind="tolerance",
                output="out",
                metric="max_rel_l2",
                value=1e-3,
                threshold=2e-2,
                passed=True,
                informational=False,
            )
        ],
        all_correctness_passed=True,
        methodology={"interpretation": "replicated_sweeps"},
        environment={"torch_version": "test"},
        warnings=("example",),
    )


class SpanResultsSchemaTests(unittest.TestCase):
    """Schema 7: two shapes of kernel results file, and two totals in one."""

    def test_a_span_file_names_a_span_and_not_a_scenario(self) -> None:
        """``scenario`` is absent, deliberately.

        A span name is not a scenario name. Putting one in a field called
        ``scenario`` would send a reader to ``KERNEL_SCENARIOS`` to look it
        up, which is the repurposing every bump in this file refuses.
        """
        payload = sample_span_result().to_dict()
        self.assertEqual(payload["kind"], "kernel_span")
        self.assertEqual(payload["span"], "test_expert_combine")
        self.assertNotIn("scenario", payload)
        self.assertEqual(payload["scenarios"], ["expert_mlp", "moe_combine"])

    def test_the_two_totals_live_in_different_fields(self) -> None:
        """The span's own number, and the sum of what it replaces.

        ``arms`` is the field every kernel results file uses for a measured
        arm, so the span's own number is where a reader already looks.
        ``parts`` is the other side, and it names every term of the sum so
        the total is auditable rather than asserted.
        """
        payload = sample_span_result().to_dict()
        own = payload["arms"]["mcore/base"]["modes"]["forward"]
        self.assertEqual(own["replicates_us"], [[30.0, 30.0], [30.0, 30.0]])

        parts = payload["parts"]["mcore/base"]
        self.assertEqual(
            [(part["scenario"], part["arm"]) for part in parts],
            [("expert_mlp", "mcore/base"), ("moe_combine", "mcore/base")],
        )
        per_replicate = [
            part["replicate_medians_us"]["forward"] for part in parts
        ]
        self.assertEqual(per_replicate, [[20.0, 20.0], [10.0, 10.0]])
        # The parts total of replicate 0, derived from the file alone.
        self.assertEqual(sum(values[0] for values in per_replicate), 30.0)

    def test_each_part_records_where_it_was_measured(self) -> None:
        """Provenance a reader needs and cannot derive.

        The parts total is a number this file did not take.
        """
        parts = sample_span_result().to_dict()["parts"]["mcore/base"]
        self.assertTrue(
            all(part["results_path"].endswith("results.json") for part in parts)
        )

    def test_the_claim_is_not_in_the_within_span_comparisons(self) -> None:
        """Two lists, two names, neither mistakable for the other.

        ``comparisons`` keeps the meaning it has in a scenario file: arm
        against arm, both measured inside the span, so it carries the honest
        ``ratio_ci_*`` names. ``parts_comparisons`` is span against parts,
        and its interval is renamed because the two sides were measured in
        separate sweeps: replicate r of each is the same index but not
        adjacent in time, so drift between the sweeps lands in the ratio
        instead of cancelling.
        """
        payload = sample_span_result().to_dict()
        within = payload["comparisons"][0]
        self.assertEqual(within["opponent"], "titan")
        self.assertIn("ratio_ci_low", within)

        claim = payload["parts_comparisons"][0]
        self.assertNotIn("opponent", claim)
        self.assertNotIn("ratio_ci_low", claim)
        self.assertIn("cross_sweep_ratio_ci_low", claim)
        self.assertEqual(claim["span_median_us"], 30.0)
        self.assertEqual(claim["parts_median_us"], 30.0)

    def test_a_span_result_round_trips_through_json(self) -> None:
        result = sample_span_result()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.json"
            write_kernel_results(result, path)
            raw = json.loads(path.read_text())
            loaded = load_kernel_results(path)

        self.assertEqual(raw["schema_version"], KERNEL_RESULTS_SCHEMA_VERSION)
        self.assertIsInstance(loaded, KernelSpanResult)
        self.assertEqual(loaded.span, result.span)
        self.assertEqual(loaded.scenarios, result.scenarios)
        self.assertEqual(loaded.shapes, result.shapes)
        self.assertEqual(loaded.parts, result.parts)
        self.assertEqual(loaded.parts_comparisons, result.parts_comparisons)
        self.assertEqual(loaded.comparisons, result.comparisons)
        self.assertEqual(loaded.correctness, result.correctness)
        self.assertEqual(loaded.warnings, result.warnings)
        self.assertEqual(
            loaded.arms["mcore/base"].modes["forward"].replicates_us,
            result.arms["mcore/base"].modes["forward"].replicates_us,
        )
        self.assertEqual(loaded.arms["mcore/base"].eager_reason,
                         result.arms["mcore/base"].eager_reason)
        # Non-finite floats are nulled for strict JSON, as in a scenario file.
        self.assertIsNone(
            raw["arms"]["mcore/base"]["modes"]["forward"]["derived"]["gbps"]
        )

    def test_a_span_file_is_refused_by_the_scenario_reader(self) -> None:
        """And the reverse.

        The version check has always been exact so an older file is rejected
        rather than half-read. The kind check is the same discipline applied
        to the second shape: a span file read as a scenario file would
        publish half of a two-total claim as a whole one.
        """
        with self.assertRaisesRegex(ValueError, "kernel_scenario"):
            KernelScenarioResult.from_dict(sample_span_result().to_dict())

        with self.assertRaisesRegex(ValueError, "kernel_span"):
            KernelSpanResult.from_dict(
                {**sample_span_result().to_dict(), "kind": "kernel"}
            )

    def test_an_older_span_schema_is_rejected_outright(self) -> None:
        for version in (6, 99):
            with self.subTest(version=version):
                with tempfile.TemporaryDirectory() as temporary:
                    path = Path(temporary) / "results.json"
                    payload = sample_span_result().to_dict()
                    payload["schema_version"] = version
                    path.write_text(json.dumps(payload))
                    with self.assertRaisesRegex(ValueError, "unsupported"):
                        load_kernel_results(path)


if __name__ == "__main__":
    unittest.main()
