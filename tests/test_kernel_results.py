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
from benchmarks.kernel.results.reporting import render_kernel_results
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
    def test_schema_four_records_the_burst_parameters_and_statuses(self) -> None:
        """Schema 3 replaced n/warmup; reusing either name is the bug.
        Schema 4 added the per-arm status."""
        self.assertEqual(KERNEL_RESULTS_SCHEMA_VERSION, 4)
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

    def test_unsupported_schema_rejected(self) -> None:
        # 1 is the pre-split schema and 2 the pre-burst one; rejecting
        # both is the point of each bump.
        for version in (1, 2, 99):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "results.json"
                payload = sample_result().to_dict()
                payload["schema_version"] = version
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    load_kernel_results(path)


if __name__ == "__main__":
    unittest.main()
