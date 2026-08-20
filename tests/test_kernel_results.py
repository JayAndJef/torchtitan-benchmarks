"""Round-trip tests for the kernel results schema."""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.summaries import summarize
from benchmarks.kernel.results.merge import (
    BURST_RESIDUAL_FLAG,
    _burst_residual,
)
from benchmarks.kernel.registry import KERNEL_SCENARIOS
from benchmarks.kernel.results.reporting import (
    ARM_FIELD,
    render_kernel_results,
)
from benchmarks.kernel.results.schema import (
    ArmResult,
    CorrectnessResult,
    KERNEL_RESULTS_SCHEMA_VERSION,
    KernelScenarioResult,
    ModeResult,
    load_kernel_results,
    write_kernel_results,
)


def sample_result() -> KernelScenarioResult:
    replicates = ((10.0, 11.0), (12.0, 11.0))
    samples = [value for replicate in replicates for value in replicate]
    return KernelScenarioResult(
        scenario="swiglu",
        hardware="test-gpu",
        model_size="normal",
        model_shape={"name": "normal", "dim": 1024, "n_layers": 16},
        workload={"batch": 4, "seq_len": 1024},
        shapes={"x": [8192, 1024]},
        replicates=2,
        samples_per_replicate=2,
        burst_k=16,
        warmup_calls=1,
        seed=0,
        arms={
            "baseline": ArmResult(
                name="baseline",
                modes={
                    "forward": ModeResult(
                        summary=summarize(samples),
                        replicates_us=replicates,
                        derived={"gbps": float("inf")},
                    )
                },
                peak_memory_gib=1.5,
            )
        },
        comparisons=[
            {
                "arm": "piper_optimized_triton",
                "opponent": "baseline",
                "mode": "forward",
                "median_ratio": 1.05,
                "welch_p": 0.5,
                "mwu_p": 0.4,
                "ratio_ci_low": 1.01,
                "ratio_ci_high": 1.09,
                "cohens_d": 0.1,
            }
        ],
        correctness=[
            CorrectnessResult(
                arm="piper_optimized_triton",
                reference="baseline",
                kind="bitwise",
                output="fwd_out",
                metric="equal",
                value=1.0,
                threshold=None,
                passed=True,
                informational=False,
            )
        ],
        all_correctness_passed=True,
        methodology={"interpretation": "replicated_sweeps"},
        environment={"torch_version": "test"},
        warnings=("example",),
    )


class KernelResultsTests(unittest.TestCase):
    def test_schema_six_records_the_burst_parameters_and_statuses(self) -> None:
        """Schema 3 replaced n/warmup; reusing either name is the bug.
        Schema 4 added the per-arm status. Schema 5 renamed the interval a
        batched run publishes. Schema 6 carries the declared compile
        treatment and the scenario's own description."""
        self.assertEqual(KERNEL_RESULTS_SCHEMA_VERSION, 6)
        payload = sample_result().to_dict()
        # Schema 2's model/workload split, still asserted.
        self.assertNotIn("spec", payload)
        self.assertEqual(payload["model_size"], "normal")
        self.assertEqual(payload["model_shape"]["dim"], 1024)
        self.assertEqual(payload["workload"], {"batch": 4, "seq_len": 1024})
        # The round-robin cycle counts must not survive under their old
        # names: they counted a different thing and were printed as such.
        self.assertNotIn("n", payload)
        self.assertNotIn("warmup", payload)
        self.assertEqual(payload["replicates"], 2)
        self.assertEqual(payload["samples_per_replicate"], 2)
        self.assertEqual(payload["burst_k"], 16)
        self.assertEqual(payload["warmup_calls"], 1)
        self.assertEqual(payload["arms"]["baseline"]["status"], "ok")
        self.assertIsNone(payload["arms"]["baseline"]["status_reason"])
        # Schema 6. Every arm names its compile treatment, measured or not.
        self.assertIn("compiled", payload["arms"]["baseline"])
        self.assertIn("eager_reason", payload["arms"]["baseline"])
        self.assertIn("description", payload)

    def test_an_unmeasured_arm_carries_a_status_and_a_reason(self) -> None:
        """An arm this host could not run and an arm nobody declared must not
        read the same way."""
        result = replace(
            sample_result(),
            arms={
                "te": ArmResult(
                    name="te",
                    modes={},
                    status="skipped",
                    status_reason="no C++20 host compiler",
                )
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.json"
            write_kernel_results(result, path)
            loaded = load_kernel_results(path)
        self.assertEqual(loaded.arms["te"].status, "skipped")
        self.assertEqual(loaded.arms["te"].status_reason, "no C++20 host compiler")
        with self.assertRaisesRegex(ValueError, "unknown arm status"):
            ArmResult(name="te", modes={}, status="nope")

    def test_the_compile_treatment_survives_the_round_trip(self) -> None:
        """A field the dataclass holds but ``to_dict`` drops is worse than a
        missing field: the schema says the file carries the treatment and the
        file does not. Round-trip both directions and both values."""
        result = replace(
            sample_result(),
            description="titan compiled against megatron eager",
            arms={
                "mcore/base": ArmResult(
                    name="mcore/base",
                    modes={},
                    status="skipped",
                    status_reason="no GPU",
                    compiled=False,
                    eager_reason="megatron compiles no whole layer",
                ),
                "titan": ArmResult(name="titan", modes={}, compiled=True),
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.json"
            write_kernel_results(result, path)
            raw = json.loads(path.read_text())
            loaded = load_kernel_results(path)
        # On disk, so a reader with no registry can still read the row.
        self.assertFalse(raw["arms"]["mcore/base"]["compiled"])
        self.assertEqual(
            raw["arms"]["mcore/base"]["eager_reason"],
            "megatron compiles no whole layer",
        )
        self.assertTrue(raw["arms"]["titan"]["compiled"])
        self.assertIsNone(raw["arms"]["titan"]["eager_reason"])
        self.assertEqual(
            raw["description"], "titan compiled against megatron eager"
        )
        # And back, so the evaluation path sees what the run recorded.
        self.assertFalse(loaded.arms["mcore/base"].compiled)
        self.assertEqual(
            loaded.arms["mcore/base"].eager_reason,
            "megatron compiles no whole layer",
        )
        self.assertTrue(loaded.arms["titan"].compiled)
        self.assertEqual(
            loaded.description, "titan compiled against megatron eager"
        )

    def test_the_compile_treatment_survives_the_round_trip(self) -> None:
        """A field the dataclass holds but ``to_dict`` drops is worse than a
        missing field: the schema says the file carries the treatment and the
        file does not. Round-trip both directions and both values."""
        result = replace(
            sample_result(),
            description="titan compiled against megatron eager",
            arms={
                "mcore/base": ArmResult(
                    name="mcore/base",
                    modes={},
                    status="skipped",
                    status_reason="no GPU",
                    compiled=False,
                    eager_reason="megatron compiles no whole layer",
                ),
                "titan": ArmResult(name="titan", modes={}, compiled=True),
            },
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.json"
            write_kernel_results(result, path)
            raw = json.loads(path.read_text())
            loaded = load_kernel_results(path)
        # On disk, so a reader with no registry can still read the row.
        self.assertFalse(raw["arms"]["mcore/base"]["compiled"])
        self.assertEqual(
            raw["arms"]["mcore/base"]["eager_reason"],
            "megatron compiles no whole layer",
        )
        self.assertTrue(raw["arms"]["titan"]["compiled"])
        self.assertIsNone(raw["arms"]["titan"]["eager_reason"])
        self.assertEqual(
            raw["description"], "titan compiled against megatron eager"
        )
        # And back, so the evaluation path sees what the run recorded.
        self.assertFalse(loaded.arms["mcore/base"].compiled)
        self.assertEqual(
            loaded.arms["mcore/base"].eager_reason,
            "megatron compiles no whole layer",
        )
        self.assertTrue(loaded.arms["titan"].compiled)
        self.assertEqual(
            loaded.description, "titan compiled against megatron eager"
        )

    def test_replicate_boundaries_survive_the_round_trip(self) -> None:
        """The boundaries are the repetition unit; a flat list loses them."""
        result = sample_result()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.json"
            write_kernel_results(result, path)
            raw = json.loads(path.read_text())
            loaded = load_kernel_results(path)
        self.assertEqual(
            raw["arms"]["baseline"]["modes"]["forward"]["replicates_us"],
            [[10.0, 11.0], [12.0, 11.0]],
        )
        self.assertEqual(
            loaded.arms["baseline"].modes["forward"].replicates_us,
            ((10.0, 11.0), (12.0, 11.0)),
        )
        # The pooled view is derived, never stored twice.
        self.assertEqual(
            loaded.arms["baseline"].modes["forward"].samples_us,
            (10.0, 11.0, 12.0, 11.0),
        )

    def test_round_trip_through_json(self) -> None:
        result = sample_result()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.json"
            write_kernel_results(result, path)
            raw = json.loads(path.read_text())
            loaded = load_kernel_results(path)

        self.assertEqual(raw["schema_version"], KERNEL_RESULTS_SCHEMA_VERSION)
        self.assertEqual(raw["kind"], "kernel")
        self.assertEqual(loaded.model_size, result.model_size)
        self.assertEqual(loaded.workload, result.workload)
        # Non-finite floats must be nulled for strict JSON.
        self.assertIsNone(
            raw["arms"]["baseline"]["modes"]["forward"]["derived"]["gbps"]
        )
        self.assertEqual(loaded.scenario, result.scenario)
        self.assertEqual(
            loaded.arms["baseline"].modes["forward"].samples_us,
            result.arms["baseline"].modes["forward"].samples_us,
        )
        self.assertEqual(loaded.correctness, result.correctness)
        self.assertEqual(loaded.warnings, result.warnings)

    def test_a_flagged_arm_says_so_beside_its_ratio(self) -> None:
        """The mark belongs in the timing table, not only under the ladder.

        A reader who quotes a ratio reads that row and nothing else, so the
        qualification has to sit on it.
        """
        result = replace(
            sample_result(),
            arms={
                "copy_floor": ArmResult(
                    name="copy_floor",
                    modes={
                        "forward": ModeResult(
                            summary=summarize([13.18]),
                            replicates_us=((13.18,),),
                            derived={"burst_residual": 0.019},
                        )
                    },
                ),
                "helion": ArmResult(
                    name="helion",
                    modes={
                        "forward": ModeResult(
                            summary=summarize([226.01]),
                            replicates_us=((226.01,),),
                            derived={"burst_residual": 0.039},
                        )
                    },
                ),
            },
        )
        rendered = render_kernel_results(result)
        floor, module = [
            line for line in rendered.splitlines()
            if line.startswith(("  copy_floor", "  helion"))
        ]
        self.assertIn("1.9 ", floor)
        self.assertNotIn("!", floor)
        self.assertIn("3.9!", module)
        self.assertIn("resid %", rendered)

    def test_no_ladder_reads_as_unknown_rather_than_converged(self) -> None:
        """--burst is off by default, and absent evidence is not evidence.

        A residual of 0.0 would claim every default run measured device
        time, which is the exact claim the column exists to stop.
        """
        self.assertIsNone(_burst_residual(None, "forward"))
        self.assertIsNone(_burst_residual({}, "forward"))
        # One rung cannot show a fall, so it is unknown too.
        self.assertIsNone(_burst_residual({"forward": {"1": 5.0}}, "forward"))
        # A ladder on another mode says nothing about this one.
        self.assertIsNone(
            _burst_residual({"backward": {"16": 9.0, "64": 8.0}}, "forward")
        )
        rendered = render_kernel_results(sample_result())
        row = next(
            line for line in rendered.splitlines()
            if line.startswith("  baseline")
        )
        self.assertNotIn("!", row)

    def test_the_flag_separates_the_measured_rope_arms_from_the_floor(
        self,
    ) -> None:
        """The threshold is calibrated on real data, so pin it to that data.

        Measured on an H200, rope forward. ``copy_floor`` is the one arm
        there that is genuinely device-bound, and it is the one arm the
        threshold clears -- which is what makes the other three readable as
        dispatch-bound rather than as slow kernels.
        """
        ladders = {
            "copy_floor": {"16": 13.43, "64": 13.18},
            "baseline": {"16": 76.91, "64": 73.10},
            "helion": {"16": 234.87, "64": 226.01},
            "te": {"16": 122.60, "64": 114.40},
        }
        flagged = {
            name: _burst_residual({"forward": rungs}, "forward")
            > BURST_RESIDUAL_FLAG
            for name, rungs in ladders.items()
        }
        self.assertEqual(
            flagged,
            {"copy_floor": False, "baseline": True, "helion": True, "te": True},
        )

    def test_a_flat_ladder_is_not_a_device_bound_verdict(self) -> None:
        """The flag is one-sided, and rope backward is why.

        Measured on an H200: ``baseline`` backward reads 187/156/157/159 us
        across k=1/4/16/64, flat from k=4 on, against ~12 us of device work.
        The residual is therefore tiny and the row carries no mark -- but
        the number is still ~92% dispatch. A plateau says ``k`` stopped
        buying amortization, never that the measurement became device time.
        """
        residual = _burst_residual(
            {"backward": {"1": 187.14, "4": 156.17, "16": 156.64,
                          "64": 158.68}},
            "backward",
        )
        self.assertLess(residual, BURST_RESIDUAL_FLAG)
        # The report must not let that silence read as a device-time claim.
        # The caveat rides with the ladder, because an unmarked row is only
        # readable at all once a ladder has run.
        result = replace(
            sample_result(),
            arms={
                "baseline": ArmResult(
                    name="baseline",
                    modes={
                        "backward": ModeResult(
                            summary=summarize([164.93]),
                            replicates_us=((164.93,),),
                            derived={"burst_residual": residual},
                        )
                    },
                    burst_us_per_call={
                        "backward": {"1": 187.14, "4": 156.17,
                                     "16": 156.64, "64": 158.68}
                    },
                )
            },
        )
        rendered = render_kernel_results(result)
        row = next(
            line for line in rendered.splitlines()
            if line.startswith("  baseline")
        )
        self.assertNotIn("!", row)
        self.assertIn("one-sided", rendered)
        self.assertIn("NOT thereby device-bound", rendered)

    def test_unsupported_schema_rejected(self) -> None:
        # Every retired version, plus one that never existed. The loader
        # enforces exact equality, so an older file is refused rather than
        # half-read, and each bump owes this list its predecessor: 1 is the
        # pre-split schema, 2 the pre-burst one, 3 the one without a per-arm
        # status, and 4 the one before the isolation fields.
        for version in (1, 2, 3, 4, 99):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "results.json"
                payload = sample_result().to_dict()
                payload["schema_version"] = version
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    load_kernel_results(path)


class IsolationReportingTests(unittest.TestCase):
    """The printed table confesses as much as the file does.

    ``results.json`` records how replicates were mapped onto processes, and
    for a while the printed report did not: the ``95% CI`` heading was the
    same whether the interval was taken across processes or inside one. A
    reader of the terminal is the reader most likely to quote a number.
    """

    def _rendered(self, replicates_per_process: int) -> str:
        result = sample_result()
        methodology = dict(result.methodology)
        methodology["replicates_per_process"] = replicates_per_process
        methodology["arm_isolation"] = (
            "one_process_per_arm_replicate"
            if replicates_per_process == 1
            else "one_process_per_arm_replicate_block"
        )
        comparisons = [dict(row) for row in result.comparisons]
        if replicates_per_process > 1:
            for row in comparisons:
                row["within_process_ratio_ci_low"] = row.pop("ratio_ci_low")
                row["within_process_ratio_ci_high"] = row.pop("ratio_ci_high")
        # The sample result compares an arm it does not carry, and the table
        # prints a row per arm rather than per comparison. Give the compared
        # arm a body so its interval reaches the table at all.
        arms = dict(result.arms)
        arms["piper_optimized_triton"] = replace(
            arms["baseline"], name="piper_optimized_triton"
        )
        return render_kernel_results(
            replace(
                result,
                arms=arms,
                methodology=methodology,
                comparisons=comparisons,
            )
        )

    def test_the_honest_state_is_printed_too(self) -> None:
        """Printed always, so the honest case is stated and not inferred."""
        rendered = self._rendered(1)
        self.assertIn(
            "arm isolation: one_process_per_arm_replicate (1 replicate per "
            "process)",
            rendered,
        )
        self.assertNotIn("WITHIN-PROCESS", rendered)

    def test_a_batched_run_is_marked_above_the_table(self) -> None:
        rendered = self._rendered(3)
        self.assertIn(
            "arm isolation: one_process_per_arm_replicate_block "
            "(3 replicates per process)",
            rendered,
        )
        self.assertIn("WITHIN-PROCESS", rendered)
        self.assertIn("lower bound", rendered)

    def test_a_within_process_interval_is_marked_in_the_row(self) -> None:
        """It is printed, not suppressed, and never printed unmarked.

        Suppressing it would leave the column empty, which reads as an
        interval that could not be computed rather than one that must not be
        quoted.
        """
        honest = self._rendered(1)
        batched = self._rendered(3)
        self.assertIn("[1.0100,1.0900]", honest)
        self.assertNotIn("~[", honest)
        self.assertIn("~[1.0100,1.0900]", batched)


class ScenarioDescriptionReportingTests(unittest.TestCase):
    """The description must reach the reader of the table, not only the file.

    A scenario whose two engines compute different functions at the cut
    publishes no cross-engine ratio, and says so in its description. Its table
    still prints every arm's median in one column, so the prohibition has to
    print beside them. Before this it reached results.json alone.
    """

    def test_the_description_prints_above_the_tables(self) -> None:
        result = replace(
            sample_result(),
            description=(
                "THIS SCENARIO PUBLISHES NO CROSS-ENGINE RATIO, AND NO READER "
                "MAY FORM ONE BY DIVIDING TWO MEDIANS."
            ),
        )
        rendered = render_kernel_results(result)
        self.assertIn("NO CROSS-ENGINE RATIO", rendered)
        # Above the tables: a warning printed under the numbers it forbids has
        # already been disobeyed.
        self.assertLess(
            rendered.index("NO CROSS-ENGINE RATIO"),
            rendered.index("median"),
        )

    def test_a_scenario_with_no_description_prints_no_blank_block(self) -> None:
        """Every existing scenario predates the field; none may grow a gap."""
        rendered = render_kernel_results(replace(sample_result(), description=None))
        self.assertNotIn("\n\n\n", rendered)

    def test_a_long_description_is_wrapped_rather_than_printed_raw(self) -> None:
        """It states a prohibition, so it is a paragraph and not a label."""
        rendered = render_kernel_results(
            replace(sample_result(), description="word " * 200)
        )
        body = [line for line in rendered.splitlines() if line.startswith("word")]
        self.assertGreater(len(body), 1)
        self.assertTrue(all(len(line) <= 78 for line in body), body)


class ArmColumnWidthTests(unittest.TestCase):
    def test_the_arm_column_fits_every_declared_arm_name(self) -> None:
        """A name wider than the field pushes the rest of its row right.

        The table is read by eye, so a misaligned row is read as a different
        column. The field was 22 while two declared names were already 28,
        and the MoE scenarios brought a 31.
        """
        longest = max(
            (arm.name for scenario in KERNEL_SCENARIOS.values()
             for arm in scenario.arms),
            key=len,
        )
        self.assertGreaterEqual(ARM_FIELD, len(longest), longest)


if __name__ == "__main__":
    unittest.main()
