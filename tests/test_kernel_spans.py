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



SPAN_REPLICATES = 3


def span_correctness_fragment() -> dict:
    return {
        "kind": CORRECTNESS_FRAGMENT_KIND,
        "scenario": "test_expert_combine",
        "rows": [],
        "all_passed": True,
        "environment": {"device": "Test GPU", "torch_version": "test"},
    }


def span_timing_fragment(arm: str, replicate: int, value: float) -> dict:
    return {
        "kind": TIMING_FRAGMENT_KIND,
        "scenario": "test_expert_combine",
        "arm": arm,
        "replicate": replicate,
        "modes": {
            mode: [value, value]
            for mode in ("forward", "forward_backward")
        },
        "bytes_moved": None,
        "peak_memory_gib": 1.5 if replicate == 0 else None,
        "burst_us_per_call": None,
    }


def measured_part(
    scenario: str, arm: str, per_replicate: tuple[float, ...]
) -> KernelScenarioResult:
    """A part scenario's result, carrying one arm at known per-replicate values."""
    modes = {
        mode: ModeResult(
            summary=summarize([value for value in per_replicate]),
            replicates_us=tuple((value, value) for value in per_replicate),
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
        replicates=len(per_replicate),
        samples_per_replicate=2,
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
        # Values that drift with the replicate, so the within-span
        # comparison is not a test between two identical samples, and so a
        # mis-paired replicate would change the answer rather than
        # reproduce it.
        timings = [
            span_timing_fragment(
                arm,
                replicate,
                (30.0 if arm == "mcore/base" else 36.0) + replicate,
            )
            for replicate in range(replicates)
            for arm in ("mcore/base", "titan")
        ]
    if parts is None:
        parts = {
            "expert_mlp": MeasuredScenario(
                result=measured_part(
                    "expert_mlp",
                    "mcore/base",
                    tuple(20.0 + 0.6 * r for r in range(replicates)),
                ),
                results_path="out/x/kernels/expert_mlp/hw/results.json",
            ),
            "moe_combine": MeasuredScenario(
                result=measured_part(
                    "moe_combine",
                    "mcore/base",
                    tuple(10.0 + 0.4 * r for r in range(replicates)),
                ),
                results_path="out/x/kernels/moe_combine/hw/results.json",
            ),
        }
    return merge_kernel_span_fragments(
        span=span or make_span(),
        shape=shape,
        workload=workload,
        hardware="test-gpu",
        replicates=replicates,
        samples_per_replicate=2,
        burst_k=16,
        warmup_calls=1,
        seed=0,
        correctness=span_correctness_fragment(),
        timings=timings,
        parts=parts,
    )


class SpanMergeTests(unittest.TestCase):
    """The parent assembles the second total, from the same run's scenarios."""

    def test_the_parts_total_is_the_sum_of_the_scenarios(self) -> None:
        """(20.0 + 0.6r) + (10.0 + 0.4r) against a span of 30.0 + r.

        The two sides agree replicate by replicate, so the ratio is 1.0 --
        the honest verdict on a span that costs exactly what its cuts cost
        separately. It reads 1.0 only if the sum is taken per replicate; a
        mis-paired index would not.
        """
        result = merge_span()
        row = next(
            row
            for row in result.parts_comparisons
            if row["arm"] == "mcore/base" and row["mode"] == "forward"
        )
        self.assertAlmostEqual(row["parts_median_us"], 31.0)
        self.assertAlmostEqual(row["span_median_us"], 31.0)
        self.assertAlmostEqual(row["median_ratio"], 1.0)
        self.assertAlmostEqual(row["ratio"], 1.0)
        self.assertEqual(row["n_replicates"], SPAN_REPLICATES)
        self.assertEqual(
            row["parts"], ["expert_mlp/mcore/base", "moe_combine/mcore/base"]
        )

    def test_the_parts_breakdown_reaches_the_file(self) -> None:
        """So the sum is auditable and not merely asserted."""
        parts = merge_span().parts["mcore/base"]
        self.assertEqual(
            [(part.scenario, part.arm) for part in parts],
            [("expert_mlp", "mcore/base"), ("moe_combine", "mcore/base")],
        )
        self.assertEqual(
            parts[0].replicate_medians_us["forward"], (20.0, 20.6, 21.2)
        )
        self.assertTrue(parts[1].results_path.endswith("results.json"))

    def test_the_claim_carries_a_cross_sweep_interval_and_no_honest_one(
        self,
    ) -> None:
        """The span and its parts were measured in separate sweeps.

        Replicate r of each shares an index but not a moment, so drift
        between the sweeps lands in the ratio instead of cancelling. A reader
        of ``ratio_ci_low`` must find nothing.
        """
        row = merge_span().parts_comparisons[0]
        self.assertNotIn("ratio_ci_low", row)
        self.assertNotIn("ratio_ci_high", row)
        self.assertNotIn("replicate_ratio_spread", row)
        self.assertIn("cross_sweep_ratio_ci_low", row)
        self.assertIn("cross_sweep_ratio_ci_high", row)
        self.assertIn("cross_sweep_replicate_ratio_spread", row)

    def test_the_claim_carries_no_two_sample_test(self) -> None:
        """The parts side is one summed value per replicate.

        A Welch or Mann-Whitney between three synthetic sums and the span's
        pooled bursts is a diagnostic of nothing, so the row does not carry
        one at all.
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
                result=measured_part("expert_mlp", "mcore/base", (20.0, 21.0)),
                results_path="a",
            ),
            "moe_combine": MeasuredScenario(
                result=measured_part(
                    "moe_combine",
                    "mcore/base",
                    tuple(10.0 + r for r in range(SPAN_REPLICATES)),
                ),
                results_path="b",
            ),
        }
        with self.assertRaises(ValueError) as caught:
            merge_span(parts=parts)
        self.assertIn("paired by replicate index", str(caught.exception))

    def test_a_failed_part_arm_is_not_summed(self) -> None:
        """A part with a status other than ``ok`` measured nothing usable."""
        broken = measured_part(
            "moe_combine",
            "mcore/base",
            tuple(10.0 + r for r in range(SPAN_REPLICATES)),
        )
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
                result=measured_part(
                    "expert_mlp",
                    "mcore/base",
                    tuple(20.0 + r for r in range(SPAN_REPLICATES)),
                ),
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
UNIT_ARM_VALUE = {
    ("expert_mlp", "mcore/base"): 20.0,
    ("expert_mlp", "titan"): 22.0,
    ("moe_combine", "mcore/base"): 10.0,
    ("moe_combine", "titan"): 11.0,
    ("test_expert_combine", "mcore/base"): 30.0,
    ("test_expert_combine", "titan"): 30.0,
}


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
        value = UNIT_ARM_VALUE.get((unit, arm), 5.0)
        timing_fragment_path(fragments_dir, arm, replicate).write_text(
            json.dumps(
                {
                    "kind": TIMING_FRAGMENT_KIND,
                    "scenario": unit,
                    "arm": arm,
                    "replicate": replicate,
                    "modes": {
                        mode_name: [value - 0.1, value, value + 0.1]
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
                    scenario_names=("moe_combine", "qkv"),
                    span_names=(span.name,),
                )
            )
        self.assertEqual(
            [unit.name for unit in plan],
            ["moe_combine", "qkv", "expert_mlp", "test_expert_combine"],
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
                    replicates=2,
                    timestamp="stamp",
                ),
                process_runner=span_run_process_runner(),
                environment={"PATH": "/usr/bin"},
            )

    def test_a_span_run_publishes_both_totals(self) -> None:
        """30.0 measured against 20.0 + 10.0 summed, and 30.0 against 33.0."""
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
            even = rows[("mcore/base", "forward")]
            self.assertAlmostEqual(even["span_median_us"], 30.0)
            self.assertAlmostEqual(even["parts_median_us"], 30.0)
            self.assertAlmostEqual(even["median_ratio"], 1.0)

            win = rows[("titan", "forward")]
            self.assertAlmostEqual(win["span_median_us"], 30.0)
            self.assertAlmostEqual(win["parts_median_us"], 33.0)
            self.assertAlmostEqual(win["median_ratio"], 30.0 / 33.0)

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
        self.assertIn("expert_mlp/mcore/base + moe_combine/mcore/base", rendered)
        # Every interval on this table is marked: there is no condition under
        # which a span and its parts are measured adjacently.
        row = next(
            line
            for line in rendered.splitlines()
            if "expert_mlp/titan + moe_combine/titan" in line
        )
        self.assertIn("~[", row)
        self.assertIn("30.00", row)
        self.assertIn("33.00", row)
        self.assertIn("0.9091", row)
        self.assertIn("The 'parts us' column is a", rendered)
        self.assertIn("SUM", rendered)
        self.assertIn("cross_sweep_ratio_ci_*", rendered)


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


if __name__ == "__main__":
    unittest.main()
