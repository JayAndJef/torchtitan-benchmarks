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
from statistics import median
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.summaries import summarize
from click.testing import CliRunner

from benchmarks.cli.kernel import kernel_bench_command
from benchmarks.execution.affinity import CpuPinning
from benchmarks.kernel.registry import KERNEL_SCENARIOS
from benchmarks.kernel.results.merge import (
    MeasuredScenario,
    merge_kernel_span_fragments,
)
from benchmarks.kernel.results.reporting import render_kernel_span_results
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
    CORRECTNESS_FRAGMENT_KIND,
    KernelArm,
    KernelScenario,
    KernelSpan,
    SpanParts,
    TIMING_FRAGMENT_KIND,
    resolve_shape_and_workload,
    validate_span_parts,
)
from benchmarks.kernel.runner import (
    KernelRunRequest,
    MeasurementUnit,
    execute_kernel_run,
    measurement_plan,
    planned_commands,
)
from benchmarks.kernel.schema import timing_fragment_path
from benchmarks.kernel.spans import KERNEL_SPANS, kernel_span_by_name


METADATA = {
    "requested_gpu": "7",
    "nvidia_smi": "Test GPU",
    "torch_version": "test",
}
PINNING = CpuPinning(("numactl", "--cpunodebind=1"), "numactl test")


def patched_environment():
    return (
        mock.patch(
            "benchmarks.kernel.runner.hardware_metadata",
            return_value=("test-gpu", dict(METADATA)),
        ),
        mock.patch(
            "benchmarks.kernel.runner.resolve_cpu_pinning", return_value=PINNING
        ),
    )


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
        # Every key ``kernel_comparison`` returns, because the renderer reads
        # them by subscript: a row missing one is a wiring bug and must
        # raise rather than print a blank.
        comparisons=[
            {
                "arm": "mcore/base",
                "opponent": "titan",
                "mode": "forward",
                "median_ratio": 1.2,
                "ratio_ci_low": 1.1,
                "ratio_ci_high": 1.3,
                "welch_p": 0.5,
                "mwu_p": 0.4,
                "cohens_d": 0.1,
            }
        ],
        parts_comparisons=[
            {
                "arm": "mcore/base",
                "mode": "forward",
                "parts": [
                    "expert_mlp/mcore/base",
                    "moe_combine/mcore/base",
                ],
                "span_median_us": 30.0,
                "parts_median_us": 30.0,
                "median_ratio": 1.0,
                "part_medians_us": [20.0, 10.0],
                "unpaired_ratio_ci_low": 0.95,
                "unpaired_ratio_ci_high": 1.05,
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
        and its interval is renamed because nothing pairs the two sides at
        all: the runner runs each unit to completion, so a span's replicate
        r and a part's replicate r share nothing but the number.
        """
        payload = sample_span_result().to_dict()
        within = payload["comparisons"][0]
        self.assertEqual(within["opponent"], "titan")
        self.assertIn("ratio_ci_low", within)

        claim = payload["parts_comparisons"][0]
        self.assertNotIn("opponent", claim)
        self.assertNotIn("ratio_ci_low", claim)
        self.assertIn("unpaired_ratio_ci_low", claim)
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



SPAN_REPLICATES = 3

# Replicate-major samples per (unit, arm), and every property of this table
# is load-bearing. The central claim of a span is that the parts total is the
# **sum, per replicate, of each part arm's median in that replicate**. Three
# wrong ways to compute it agree with the right one on a degenerate fixture,
# so this one is built so that all four answers differ:
#
# * **Per-replicate medians differ across replicates.** Otherwise reversing
#   the pairing -- span replicate r against parts replicate R-1-r -- gives
#   the same answer, and nothing tests that the indices line up.
# * **Samples inside a replicate are asymmetric**, so the median is not the
#   mean and a pooled median is not the median of the per-replicate medians.
# * **The two part arms carry their median at a different index of the raw
#   list.** This is the subtle one. Element-wise summing two *sorted* lists
#   of equal length puts the sum of the middles in the middle, so a
#   sample-level sum reproduces the sum of medians exactly and the mutation
#   is invisible. Real burst samples arrive unsorted; these are unsorted, and
#   the two parts are unsorted differently.
#
# ``tests/test_kernel_spans.py`` states the four answers this produces.
SPAN_SAMPLES = {
    # medians 6, 12, 24. The median sits at index 0 of each raw list.
    ("expert_mlp", "mcore/base"): (
        (6.0, 5.0, 30.0),
        (12.0, 11.0, 36.0),
        (24.0, 23.0, 48.0),
    ),
    # medians 4, 8, 16. The median sits at index 2.
    ("moe_combine", "mcore/base"): (
        (1.0, 20.0, 4.0),
        (2.0, 40.0, 8.0),
        (3.0, 60.0, 16.0),
    ),
    # medians 10, 40, 40, and a pooled median of 11.0 -- deliberately not 40.
    ("test_expert_combine", "mcore/base"): (
        (10.0, 5.0, 11.0),
        (40.0, 6.0, 200.0),
        (40.0, 7.0, 201.0),
    ),
    # The second arm, so a span publishes more than one claim. medians 14.
    ("expert_mlp", "titan"): ((14.0, 13.0, 40.0),) * SPAN_REPLICATES,
    ("moe_combine", "titan"): ((2.0, 30.0, 7.0),) * SPAN_REPLICATES,
    ("test_expert_combine", "titan"): ((14.0, 1.0, 100.0),) * SPAN_REPLICATES,
}
# Every other arm of the enclosed scenarios. They are measured because the
# run measures whole scenarios, and no span names them.
SPAN_SAMPLES_DEFAULT = ((5.0, 4.0, 9.0),) * SPAN_REPLICATES


def unit_samples(unit: str, arm: str) -> tuple[tuple[float, ...], ...]:
    return SPAN_SAMPLES.get((unit, arm), SPAN_SAMPLES_DEFAULT)


def span_correctness_fragment(unit: str = "test_expert_combine") -> dict:
    return {
        "kind": CORRECTNESS_FRAGMENT_KIND,
        "scenario": unit,
        "rows": [],
        "all_passed": True,
        "environment": {"device": "Test GPU", "torch_version": "test"},
    }


def span_timing_fragment(arm: str, replicate: int) -> dict:
    return {
        "kind": TIMING_FRAGMENT_KIND,
        "scenario": "test_expert_combine",
        "arm": arm,
        "replicate": replicate,
        "modes": {
            mode: list(unit_samples("test_expert_combine", arm)[replicate])
            for mode in ("forward", "forward_backward")
        },
        "bytes_moved": None,
        "peak_memory_gib": 1.5 if replicate == 0 else None,
        "burst_us_per_call": None,
    }


def measured_part(
    scenario: str,
    arm: str,
    replicates: tuple[tuple[float, ...], ...] | None = None,
) -> KernelScenarioResult:
    """A part scenario's result, carrying one arm's replicate-major samples."""
    if replicates is None:
        replicates = unit_samples(scenario, arm)
    pooled = [value for replicate in replicates for value in replicate]
    modes = {
        mode: ModeResult(
            summary=summarize(pooled),
            replicates_us=tuple(tuple(r) for r in replicates),
        )
        for mode in ("forward", "forward_backward")
    }
    return KernelScenarioResult(
        scenario=scenario,
        hardware="test-gpu",
        model_size="normal",
        model_shape={"name": "normal"},
        workload={"batch": 4, "seq_len": 1024},
        shapes={},
        replicates=len(replicates),
        samples_per_replicate=len(replicates[0]),
        burst_k=16,
        warmup_calls=1,
        seed=0,
        arms={arm: ArmResult(name=arm, modes=modes)},
        comparisons=[],
        correctness=[],
        all_correctness_passed=True,
        methodology={},
        environment={},
    )


def merge_span(
    *,
    span: KernelSpan | None = None,
    timings: list[dict] | None = None,
    parts: dict[str, MeasuredScenario] | None = None,
    replicates: int = SPAN_REPLICATES,
):
    shape, workload = resolve_shape_and_workload()
    if timings is None:
        timings = [
            span_timing_fragment(arm, replicate)
            for replicate in range(replicates)
            for arm in ("mcore/base", "titan")
        ]
    if parts is None:
        parts = {
            "expert_mlp": MeasuredScenario(
                result=measured_part("expert_mlp", "mcore/base"),
                results_path="out/x/kernels/expert_mlp/hw/results.json",
            ),
            "moe_combine": MeasuredScenario(
                result=measured_part("moe_combine", "mcore/base"),
                results_path="out/x/kernels/moe_combine/hw/results.json",
            ),
        }
    return merge_kernel_span_fragments(
        span=span or make_span(),
        shape=shape,
        workload=workload,
        hardware="test-gpu",
        replicates=replicates,
        samples_per_replicate=SPAN_REPLICATES,
        burst_k=16,
        warmup_calls=1,
        seed=0,
        correctness=span_correctness_fragment(),
        timings=timings,
        parts=parts,
    )


class SpanMergeTests(unittest.TestCase):
    """The parent assembles the second total, from the same run's scenarios."""

    def test_the_parts_total_is_the_sum_of_per_replicate_medians(self) -> None:
        """The central claim, against the ways of getting it wrong.

        The fixture's parts are ``expert_mlp/mcore/base`` at per-replicate
        medians 6, 12, 24 and ``moe_combine/mcore/base`` at 4, 8, 16, so
        their own medians are 12 and 8 and the parts total is 20.0. The span
        itself reads 10, 40, 40, so its median is 40.0.

        ==============================  ============  ============
        estimator                       parts median  span median
        ==============================  ============  ============
        median per side, then summed            20.0          40.0
        pooled medians                          31.0          11.0
        element-wise sum of samples             44.0          40.0
        ==============================  ============  ============

        No two rows agree in either cell, so each mutation is visible in at
        least one assertion below.
        """
        result = merge_span()
        row = next(
            row
            for row in result.parts_comparisons
            if row["arm"] == "mcore/base" and row["mode"] == "forward"
        )
        self.assertAlmostEqual(row["parts_median_us"], 20.0)
        self.assertEqual(row["part_medians_us"], [12.0, 8.0])
        # The median of the span's per-replicate medians. Its pooled median
        # is 11.0.
        self.assertAlmostEqual(row["span_median_us"], 40.0)
        self.assertAlmostEqual(row["median_ratio"], 2.0)
        self.assertEqual(row["n_replicates"], SPAN_REPLICATES)
        self.assertEqual(
            row["parts"], ["expert_mlp/mcore/base", "moe_combine/mcore/base"]
        )
        self.assertLessEqual(row["unpaired_ratio_ci_low"], row["median_ratio"])
        self.assertGreaterEqual(
            row["unpaired_ratio_ci_high"], row["median_ratio"]
        )

    def test_the_estimate_does_not_move_when_a_part_is_reordered(
        self,
    ) -> None:
        """Because nothing pairs replicate r of one side with r of another.

        The runner runs each unit to completion, so a span's replicate 0 and
        an enclosed scenario's replicate 0 are separated by every worker in
        between. Any permutation of a part's replicate order is as justified
        as the identity, so the published number must not depend on which
        one it was handed. An estimator that moved here would report a
        choice among equals as a measurement.
        """
        shuffled = {
            "expert_mlp": MeasuredScenario(
                result=measured_part(
                    "expert_mlp",
                    "mcore/base",
                    tuple(reversed(unit_samples("expert_mlp", "mcore/base"))),
                ),
                results_path="a",
            ),
            "moe_combine": MeasuredScenario(
                result=measured_part("moe_combine", "mcore/base"),
                results_path="b",
            ),
        }
        straight = merge_span().parts_comparisons[0]
        reordered = merge_span(parts=shuffled).parts_comparisons[0]
        for field in (
            "parts_median_us",
            "span_median_us",
            "median_ratio",
            "unpaired_ratio_ci_low",
            "unpaired_ratio_ci_high",
        ):
            with self.subTest(field=field):
                self.assertAlmostEqual(straight[field], reordered[field])

    def test_the_published_total_is_auditable_from_the_same_file(self) -> None:
        """"Auditable" is a property here, not a word.

        A reader recomputes the parts total from the ``parts`` block the
        same file records, and must land on the ``parts_median_us`` the
        claim row publishes. Nothing else in the file states the sum, so
        without this the breakdown and the headline could disagree and no
        test would notice.
        """
        result = merge_span()
        for row in result.parts_comparisons:
            with self.subTest(arm=row["arm"], mode=row["mode"]):
                parts = result.parts[row["arm"]]
                per_part = [
                    median(part.replicate_medians_us[row["mode"]])
                    for part in parts
                ]
                self.assertEqual(row["part_medians_us"], per_part)
                self.assertAlmostEqual(row["parts_median_us"], sum(per_part))
                # And the names in the row are the names in the breakdown,
                # in the same order, so a reader knows which term is which.
                self.assertEqual(
                    row["parts"],
                    [f"{part.scenario}/{part.arm}" for part in parts],
                )

    def test_the_span_total_is_auditable_from_its_own_samples(self) -> None:
        """The other half. The span's raw samples are in ``arms``."""
        result = merge_span()
        for row in result.parts_comparisons:
            with self.subTest(arm=row["arm"], mode=row["mode"]):
                replicates = result.arms[row["arm"]].modes[
                    row["mode"]
                ].replicates_us
                self.assertAlmostEqual(
                    row["span_median_us"],
                    median([median(replicate) for replicate in replicates]),
                )

    def test_the_parts_breakdown_reaches_the_file(self) -> None:
        """So the sum is auditable and not merely asserted."""
        parts = merge_span().parts["mcore/base"]
        self.assertEqual(
            [(part.scenario, part.arm) for part in parts],
            [("expert_mlp", "mcore/base"), ("moe_combine", "mcore/base")],
        )
        self.assertEqual(
            parts[0].replicate_medians_us["forward"], (6.0, 12.0, 24.0)
        )
        self.assertEqual(
            parts[1].replicate_medians_us["forward"], (4.0, 8.0, 16.0)
        )
        self.assertTrue(parts[1].results_path.endswith("results.json"))

    def test_the_claim_carries_an_unpaired_interval_and_no_honest_one(
        self,
    ) -> None:
        """Nothing pairs a span with its parts, so nothing here is paired.

        ``ratio_ci_low`` names the per-replicate interval a scenario
        publishes, which cancels drift because the sweep puts the two arms
        within seconds of each other. This one cancels none, so a reader of
        that name must find nothing. ``ratio`` and
        ``replicate_ratio_spread`` are gone with the pairing that produced
        them, and so is the ``cross_sweep_`` name, which said the pairing
        was loose when it does not exist.
        """
        row = merge_span().parts_comparisons[0]
        for absent in (
            "ratio",
            "ratio_ci_low",
            "ratio_ci_high",
            "replicate_ratio_spread",
            "cross_sweep_ratio_ci_low",
            "cross_sweep_replicate_ratio_spread",
        ):
            with self.subTest(field=absent):
                self.assertNotIn(absent, row)
        self.assertIn("unpaired_ratio_ci_low", row)
        self.assertIn("unpaired_ratio_ci_high", row)

    def test_the_claim_carries_no_two_sample_test(self) -> None:
        """The parts side is a handful of per-replicate medians.

        A Welch or Mann-Whitney between three of them and the span's pooled
        bursts is a diagnostic of nothing, so the row does not carry one at
        all.
        """
        row = merge_span().parts_comparisons[0]
        for absent in ("welch_p", "mwu_p", "cohens_d", "n_base", "n_arm"):
            self.assertNotIn(absent, row)

    def test_the_within_span_rows_keep_the_honest_names(self) -> None:
        """They are measured entirely inside the span's own sweep."""
        row = merge_span().comparisons[0]
        self.assertEqual(row["opponent"], "mcore/base")
        self.assertIn("ratio_ci_low", row)
        self.assertIn("welch_p", row)

    def test_the_span_records_the_range_and_every_cut_it_crosses(self) -> None:
        result = merge_span()
        self.assertEqual(result.span, "test_expert_combine")
        self.assertEqual(result.scenarios, ("expert_mlp", "moe_combine"))
        self.assertEqual(
            sorted(result.shapes), ["expert_mlp", "moe_combine"]
        )
        self.assertIn("span_claim", result.methodology)

    def test_a_missing_part_costs_that_arm_its_claim_and_nothing_more(
        self,
    ) -> None:
        """The mirror of a missing opponent in a scenario.

        ``titan``'s parts are absent from both scenario results, so it
        carries no claim. ``mcore/base`` still does, and the within-span
        comparison still stands: it was measured inside the span and a
        scenario that failed elsewhere in the run cannot reach it.
        """
        result = merge_span()
        self.assertIn("mcore/base", result.parts)
        self.assertNotIn("titan", result.parts)
        self.assertTrue(
            any(
                "titan" in warning and "no parts total" in warning
                for warning in result.warnings
            )
        )
        self.assertTrue(result.comparisons)

    def test_a_span_with_no_claim_at_all_is_refused(self) -> None:
        """A scenario wearing a span's name.

        The same failure anchor loss raises for: a file that states the
        span's own number and nothing about it reads like a span that
        declared no claim.
        """
        with self.assertRaises(ValueError) as caught:
            merge_span(parts={})
        self.assertIn("no arm carries a parts total", str(caught.exception))

    def test_a_part_measured_at_another_replicate_count_is_refused(
        self,
    ) -> None:
        """One request produces both sides, so the counts cannot differ.

        Index r would otherwise pair two different replicates, which is the
        one thing the pairing exists to prevent. A wiring error, not a data
        condition, so it raises.
        """
        parts = {
            "expert_mlp": MeasuredScenario(
                # Two replicates against the span's three.
                result=measured_part(
                    "expert_mlp", "mcore/base", ((20.0, 21.0), (22.0, 23.0))
                ),
                results_path="a",
            ),
            "moe_combine": MeasuredScenario(
                result=measured_part("moe_combine", "mcore/base"),
                results_path="b",
            ),
        }
        with self.assertRaises(ValueError) as caught:
            merge_span(parts=parts)
        self.assertIn("paired by replicate index", str(caught.exception))

    def test_a_failed_part_arm_is_not_summed(self) -> None:
        """A part with a status other than ``ok`` measured nothing usable."""
        broken = measured_part("moe_combine", "mcore/base")
        broken = KernelScenarioResult(
            **{
                **{
                    field: getattr(broken, field)
                    for field in broken.__dataclass_fields__
                },
                "arms": {
                    "mcore/base": ArmResult(
                        name="mcore/base",
                        modes={},
                        status="failed",
                        status_reason="lost every replicate",
                    )
                },
            }
        )
        parts = {
            "expert_mlp": MeasuredScenario(
                result=measured_part("expert_mlp", "mcore/base"),
                results_path="a",
            ),
            "moe_combine": MeasuredScenario(result=broken, results_path="b"),
        }
        with self.assertRaises(ValueError) as caught:
            merge_span(parts=parts)
        self.assertIn("no arm carries a parts total", str(caught.exception))



# One value per (unit, arm), so every median is exact and the two totals can
# be checked by arithmetic. The samples carry a spread so the within-span
# Welch has a variance to work with.
def span_run_process_runner():
    """A ``process_runner`` that plays the worker protocol for both kinds.

    It reads ``--span`` or ``--scenario``, exactly as the real worker does,
    which is what makes the argv the runner emits part of what is under test.
    """

    def fake_process(command, **kwargs):
        if "--span" in command:
            unit = command[command.index("--span") + 1]
            declaration = kernel_span_by_name(unit)
        else:
            unit = command[command.index("--scenario") + 1]
            declaration = KERNEL_SCENARIOS[unit]
        mode = command[command.index("--mode") + 1]
        if mode == "correctness":
            Path(command[command.index("--fragment") + 1]).write_text(
                json.dumps(
                    {
                        "kind": CORRECTNESS_FRAGMENT_KIND,
                        "scenario": unit,
                        "rows": [],
                        "all_passed": True,
                        "environment": {
                            "device": "Test GPU",
                            "torch_version": "test",
                        },
                    }
                )
            )
            return SimpleNamespace(returncode=0)
        fragments_dir = Path(command[command.index("--fragments-dir") + 1])
        arm = command[command.index("--arm") + 1]
        replicate = int(command[command.index("--replicate") + 1])
        # The same table the direct merge tests use, so the run path and
        # the merge path are checked against one set of expected numbers.
        samples = list(unit_samples(unit, arm)[replicate])
        timing_fragment_path(fragments_dir, arm, replicate).write_text(
            json.dumps(
                {
                    "kind": TIMING_FRAGMENT_KIND,
                    "scenario": unit,
                    "arm": arm,
                    "replicate": replicate,
                    "modes": {
                        mode_name: samples
                        for mode_name in declaration.arm(arm).modes
                    },
                    "bytes_moved": None,
                    "peak_memory_gib": 1.5 if replicate == 0 else None,
                    "burst_us_per_call": None,
                }
            )
        )
        return SimpleNamespace(returncode=0)

    return fake_process


def declared(span: KernelSpan):
    """Put one span in the registry for the duration of a test."""
    return mock.patch.dict(
        "benchmarks.kernel.spans.KERNEL_SPANS", {span.name: span}
    )


class SpanPlanTests(unittest.TestCase):
    """Which units a run measures, and in which order."""

    def test_the_enclosed_scenarios_run_before_the_span(self) -> None:
        """The span's second total is taken from their results.

        Reading it from an older run's results.json would have made the
        hardware, shape, workload, seed, burst_k and replicate count things
        to verify instead of things one request fixed.
        """
        span = make_span()
        with declared(span):
            plan = measurement_plan(
                KernelRunRequest(gpu="0", scenario_names=(), span_names=(span.name,))
            )
        self.assertEqual(
            [(unit.name, unit.kind) for unit in plan],
            [
                ("expert_mlp", "scenario"),
                ("moe_combine", "scenario"),
                ("test_expert_combine", "span"),
            ],
        )

    def test_a_scenario_asked_for_twice_is_measured_once(self) -> None:
        """Once by name and once as part of the span's range.

        Measuring it twice would spend the GPU on it twice and leave two
        different numbers for one thing, with nothing to say which of them
        the span summed.
        """
        span = make_span()
        with declared(span):
            plan = measurement_plan(
                KernelRunRequest(
                    gpu="0",
                    scenario_names=("moe_combine", "rope"),
                    span_names=(span.name,),
                )
            )
        self.assertEqual(
            [unit.name for unit in plan],
            ["moe_combine", "rope", "expert_mlp", "test_expert_combine"],
        )

    def test_a_span_is_named_with_span_in_the_worker_argv(self) -> None:
        """So a reader of the manifest never has to guess which roster."""
        span = make_span()
        with declared(span):
            commands = planned_commands(
                MeasurementUnit(measurement=span.measurement, span=span),
                KernelRunRequest(gpu="0", scenario_names=()),
                Path("/tmp/fragments"),
                (),
            )
        for command in commands:
            self.assertIn("--span", command)
            self.assertNotIn("--scenario", command)
            self.assertEqual(
                command[command.index("--span") + 1], "test_expert_combine"
            )


class SpanRunTests(unittest.TestCase):
    """One run measures a span and the scenarios it replaces."""

    def _run(self, span: KernelSpan, temporary: str):
        metadata_patch, pinning_patch = patched_environment()
        with metadata_patch, pinning_patch, declared(span), mock.patch(
            "benchmarks.kernel.runner.BENCH_DIR", Path(temporary)
        ):
            return execute_kernel_run(
                KernelRunRequest(
                    gpu="7",
                    scenario_names=(),
                    span_names=(span.name,),
                    replicates=SPAN_REPLICATES,
                    timestamp="stamp",
                ),
                process_runner=span_run_process_runner(),
                environment={"PATH": "/usr/bin"},
            )

    def test_a_span_run_publishes_both_totals(self) -> None:
        """The whole path: two scenario merges, then the span merge.

        The numbers are the ones the direct merge tests pin, because the
        run and the merge read one sample table. What this adds is that the
        parts really came out of ``merge_kernel_fragments`` on the enclosed
        scenarios rather than out of a hand-built result.
        """
        span = make_span()
        with tempfile.TemporaryDirectory() as temporary:
            outcomes = self._run(span, temporary)
            by_name = {outcome.scenario: outcome for outcome in outcomes}
            self.assertEqual(
                sorted(by_name),
                ["expert_mlp", "moe_combine", "test_expert_combine"],
            )
            self.assertFalse(by_name["test_expert_combine"].failed)
            result = by_name["test_expert_combine"].result
            self.assertIsInstance(result, KernelSpanResult)

            rows = {
                (row["arm"], row["mode"]): row
                for row in result.parts_comparisons
            }
            # Part medians 12 and 8, summed: 20.0.
            anchor = rows[("mcore/base", "forward")]
            self.assertAlmostEqual(anchor["parts_median_us"], 20.0)
            self.assertEqual(anchor["part_medians_us"], [12.0, 8.0])
            self.assertAlmostEqual(anchor["span_median_us"], 40.0)
            self.assertAlmostEqual(anchor["median_ratio"], 2.0)

            # 14 + 7 against a span of 14.
            win = rows[("titan", "forward")]
            self.assertAlmostEqual(win["parts_median_us"], 21.0)
            self.assertAlmostEqual(win["span_median_us"], 14.0)
            self.assertAlmostEqual(win["median_ratio"], 14.0 / 21.0)

    def test_the_span_file_sits_apart_from_the_scenario_files(self) -> None:
        """A glob over the scenarios must not sweep up a span beside them.

        A span total and a scenario total answer different questions, so a
        path pattern must not be able to pool them.
        """
        span = make_span()
        with tempfile.TemporaryDirectory() as temporary:
            outcomes = self._run(span, temporary)
            by_name = {outcome.scenario: outcome for outcome in outcomes}
            span_dir = by_name["test_expert_combine"].out_dir
            self.assertEqual(span_dir.parent.parent.name, "spans")
            scenario_dirs = {
                by_name[name].out_dir for name in ("expert_mlp", "moe_combine")
            }
            self.assertTrue(
                all("spans" not in path.parts for path in scenario_dirs)
            )
            raw = json.loads((span_dir / "results.json").read_text())
            self.assertEqual(raw["kind"], "kernel_span")
            manifest = json.loads((span_dir / "manifest.json").read_text())
            self.assertEqual(manifest["kind"], "kernel_span")
            self.assertEqual(manifest["unit_kind"], "span")
            # The name goes under the field that says what kind of name it
            # is, exactly as the results file does it. A span name in
            # "scenario" sends a reader to KERNEL_SCENARIOS for a name that
            # is not there.
            self.assertEqual(manifest["span"], "test_expert_combine")
            self.assertIsNone(manifest["scenario"])
            scenario_manifest = json.loads(
                (by_name["expert_mlp"].out_dir / "manifest.json").read_text()
            )
            self.assertEqual(scenario_manifest["scenario"], "expert_mlp")
            self.assertIsNone(scenario_manifest["span"])
            self.assertEqual(
                manifest["span_scenarios"], ["expert_mlp", "moe_combine"]
            )
            self.assertEqual(
                manifest["parts"],
                {
                    "mcore/base": ["mcore/base", "mcore/base"],
                    "titan": ["titan", "titan"],
                },
            )

    def test_the_printed_table_names_both_totals_and_marks_the_interval(
        self,
    ) -> None:
        """A reader of the terminal is the reader most likely to quote."""
        span = make_span()
        with tempfile.TemporaryDirectory() as temporary:
            outcomes = self._run(span, temporary)
            result = next(
                outcome.result
                for outcome in outcomes
                if outcome.scenario == "test_expert_combine"
            )
        rendered = render_kernel_span_results(result)
        self.assertIn("kernel span: test_expert_combine", rendered)
        self.assertIn("replaces: expert_mlp + moe_combine", rendered)
        self.assertIn("against the sum of the scenarios it replaces:", rendered)
        self.assertIn("span us", rendered)
        self.assertIn("parts us", rendered)
        # The terms and their values, so the total adds up on the terminal.
        self.assertIn(
            "expert_mlp/mcore/base 12.00 + moe_combine/mcore/base 8.00",
            rendered,
        )
        row = next(
            line
            for line in rendered.splitlines()
            if "expert_mlp/titan 14.00 + moe_combine/titan 7.00" in line
        )
        self.assertIn("14.00", row)
        self.assertIn("21.00", row)
        self.assertIn("0.6667", row)
        self.assertIn("The 'parts us' column is a", rendered)
        self.assertIn("SUM", rendered)
        # The heading carries the word, so the column needs no mark. "~"
        # already means "within-process" in the scenario table, and one mark
        # with two meanings teaches a reader the wrong thing.
        self.assertIn("unpaired 95% CI", rendered)
        self.assertNotIn("~[", rendered)
        self.assertIn("NOTHING PAIRS the two sides", rendered)
        self.assertIn("unpaired_ratio_ci_*", rendered)


class SpanCliTests(unittest.TestCase):
    def test_a_span_is_opt_in_and_never_a_default(self) -> None:
        """One span can drag six scenarios into a run that asked for none.

        ``--scenario`` still defaults to every scenario, so making
        ``--span`` default to every span would silently change what a bare
        ``kernel-bench <gpu>`` costs.
        """
        with mock.patch(
            "benchmarks.cli.kernel.execute_kernel_run", return_value=()
        ) as execute:
            result = CliRunner().invoke(kernel_bench_command, ["3"])
        self.assertEqual(result.exit_code, 0, result.output)
        request = execute.call_args.args[0]
        # --scenario still defaults to the whole roster; --span does not.
        self.assertEqual(request.scenario_names, tuple(KERNEL_SCENARIOS))
        self.assertEqual(request.span_names, ())
        option = next(
            param
            for param in kernel_bench_command.params
            if param.name == "span_names"
        )
        self.assertTrue(option.multiple)
        self.assertEqual(list(option.type.choices), list(KERNEL_SPANS))

    def test_out_refuses_a_span_outright(self) -> None:
        """A span run is never one unit.

        It measures every scenario the span replaces as well, so they would
        all resolve to the one directory --out names. The choice list is
        widened here because no span is declared at this rev; the guard
        under test is the one after the parse.
        """
        span = make_span()
        option = next(
            param
            for param in kernel_bench_command.params
            if param.name == "span_names"
        )
        with declared(span), mock.patch.object(
            option.type, "choices", [span.name]
        ), mock.patch(
            "benchmarks.cli.kernel.execute_kernel_run", return_value=()
        ):
            refused = CliRunner().invoke(
                kernel_bench_command,
                ["3", "--span", span.name, "--out", "/tmp/x"],
            )
            allowed = CliRunner().invoke(
                kernel_bench_command, ["3", "--span", span.name]
            )
        self.assertNotEqual(refused.exit_code, 0)
        self.assertIn("no --span", refused.output)
        # Without --out the same request is fine, and it asks for no
        # scenario of its own: the plan derives them from the range.
        self.assertEqual(allowed.exit_code, 0, allowed.output)

    def test_arm_does_not_combine_with_a_span(self) -> None:
        """An arm name belongs to one scenario's roster.

        A span run measures the span and every scenario the span replaces,
        so it holds several rosters. One arm selection cannot say which of
        them it names, and a per-scenario selection has no meaning across a
        range. The choice list is widened here because no span is declared
        at this rev; the guard under test is the one after the parse.
        """
        span = make_span()
        option = next(
            param
            for param in kernel_bench_command.params
            if param.name == "span_names"
        )
        with declared(span), mock.patch.object(
            option.type, "choices", [span.name]
        ), mock.patch(
            "benchmarks.cli.kernel.execute_kernel_run", return_value=()
        ):
            refused = CliRunner().invoke(
                kernel_bench_command,
                [
                    "3",
                    "--span",
                    span.name,
                    "--scenario",
                    "expert_mlp",
                    "--arm",
                    "titan",
                ],
            )
        self.assertNotEqual(refused.exit_code, 0)
        self.assertIn("--arm does not combine with --span", refused.output)

    def test_scenarios_lists_the_span_roster_even_when_it_is_empty(
        self,
    ) -> None:
        """A span is the one unit that is not a default.

        A reader who never sees the heading has no way to learn that --span
        exists, so the heading prints whether or not a span is declared.
        """
        from benchmarks.cli.main import cli

        empty = CliRunner().invoke(cli, ["scenarios"])
        self.assertEqual(empty.exit_code, 0, empty.output)
        self.assertIn("kernel spans (kernel-bench --span)", empty.output)
        self.assertIn("(none declared)", empty.output)

        span = make_span()
        with declared(span):
            listed = CliRunner().invoke(cli, ["scenarios"])
        self.assertEqual(listed.exit_code, 0, listed.output)
        self.assertIn(
            "test_expert_combine [replaces expert_mlp + moe_combine]",
            listed.output,
        )
        # What each arm is compared against, which is the half a reader
        # cannot infer from the range.
        self.assertIn(
            "against expert_mlp/titan + moe_combine/titan", listed.output
        )


class SpanDispatchBiasTests(unittest.TestCase):
    """The one bias every span row carries, stated where the number is.

    The parts total pays one host dispatch chain per enclosed scenario and
    the span pays one. Roughly 85% of a kernel number in this repository is
    host dispatch, so the parts total holds N-1 chains no fusion removed and
    the ratio is smaller than fusion alone would make it. The direction of
    the bias is the direction of the conclusion a span is written to
    support, which is why it is stated rather than left to the reader.

    It is a property of the range LENGTH, not of what a span fuses, so the
    engine states it and no declaration has to remember to. These tests are
    what stop it being deleted quietly.
    """

    def test_the_results_file_records_the_bias(self) -> None:
        methodology = merge_span().methodology
        self.assertIn("span_dispatch_bias", methodology)
        note = methodology["span_dispatch_bias_note"]
        self.assertIn("ONE HOST DISPATCH CHAIN PER ENCLOSED SCENARIO", note)
        self.assertIn("SMALLER", note)
        self.assertIn("never as fusion alone", note)

    def test_the_printed_table_states_the_bias_and_counts_the_chains(
        self,
    ) -> None:
        """With the range's own length in it, not a general remark."""
        rendered = render_kernel_span_results(merge_span())
        self.assertIn("BIAS, and it favours the span", rendered)
        self.assertIn("2 of them here", rendered)
        self.assertIn("1 extra chain(s)", rendered)
        self.assertIn("never as fusion alone", rendered)

    def test_the_bias_is_printed_on_every_span_not_only_a_long_one(
        self,
    ) -> None:
        """A two-scenario span already pays it once."""
        rendered = render_kernel_span_results(sample_span_result())
        self.assertIn("BIAS, and it favours the span", rendered)


if __name__ == "__main__":
    unittest.main()
