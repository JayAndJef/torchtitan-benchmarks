"""Round-trip tests for the kernel results schema."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kernel_results import (
    ArmResult,
    CorrectnessResult,
    KERNEL_RESULTS_SCHEMA_VERSION,
    KernelScenarioResult,
    ModeResult,
    load_kernel_results,
    write_kernel_results,
)
from benchmarks.metrics import summarize


def sample_result() -> KernelScenarioResult:
    samples = [10.0, 11.0, 12.0, 11.0]
    return KernelScenarioResult(
        scenario="swiglu",
        hardware="test-gpu",
        spec={"dim": 1024, "batch": 4},
        shapes={"x": [8192, 1024]},
        n=4,
        warmup=1,
        seed=0,
        arms={
            "stock_kernel": ArmResult(
                name="stock_kernel",
                modes={
                    "forward": ModeResult(
                        summary=summarize(samples),
                        samples_us=tuple(samples),
                        derived={"gbps": float("inf")},
                    )
                },
                peak_memory_gib=1.5,
            )
        },
        comparisons=[
            {
                "arm": "stock_kernel",
                "opponent": "combined_kernel",
                "mode": "forward",
                "median_ratio": 1.05,
                "welch_p": 0.5,
                "mwu_p": 0.4,
                "wilcoxon_p": None,
                "cohens_d": 0.1,
            }
        ],
        correctness=[
            CorrectnessResult(
                arm="combined_kernel",
                reference="stock_kernel",
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
        methodology={"interpretation": "paired_interleaved_repeats"},
        environment={"torch_version": "test"},
        warnings=("example",),
    )


class KernelResultsTests(unittest.TestCase):
    def test_round_trip_through_json(self) -> None:
        result = sample_result()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.json"
            write_kernel_results(result, path)
            raw = json.loads(path.read_text())
            loaded = load_kernel_results(path)

        self.assertEqual(raw["schema_version"], KERNEL_RESULTS_SCHEMA_VERSION)
        self.assertEqual(raw["kind"], "kernel")
        # Non-finite floats must be nulled for strict JSON.
        self.assertIsNone(
            raw["arms"]["stock_kernel"]["modes"]["forward"]["derived"]["gbps"]
        )
        self.assertEqual(loaded.scenario, result.scenario)
        self.assertEqual(
            loaded.arms["stock_kernel"].modes["forward"].samples_us,
            result.arms["stock_kernel"].modes["forward"].samples_us,
        )
        self.assertEqual(loaded.correctness, result.correctness)
        self.assertEqual(loaded.warnings, result.warnings)

    def test_unsupported_schema_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "results.json"
            payload = sample_result().to_dict()
            payload["schema_version"] = 99
            path.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError, "unsupported"):
                load_kernel_results(path)


if __name__ == "__main__":
    unittest.main()
