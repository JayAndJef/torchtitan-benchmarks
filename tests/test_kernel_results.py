"""Round-trip tests for the kernel results schema."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kernel.results.schema import (
    ArmResult,
    CorrectnessResult,
    KERNEL_RESULTS_SCHEMA_VERSION,
    KernelScenarioResult,
    ModeResult,
    load_kernel_results,
    write_kernel_results,
)
from benchmarks.artifacts.summaries import summarize


def sample_result() -> KernelScenarioResult:
    samples = [10.0, 11.0, 12.0, 11.0]
    return KernelScenarioResult(
        scenario="swiglu",
        hardware="test-gpu",
        model_size="normal",
        model_shape={"name": "normal", "dim": 1024, "n_layers": 16},
        workload={"batch": 4, "seq_len": 1024},
        shapes={"x": [8192, 1024]},
        n=4,
        warmup=1,
        seed=0,
        arms={
            "baseline": ArmResult(
                name="baseline",
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
                "arm": "piper_optimized_triton",
                "opponent": "baseline",
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
        methodology={"interpretation": "paired_interleaved_repeats"},
        environment={"torch_version": "test"},
        warnings=("example",),
    )


class KernelResultsTests(unittest.TestCase):
    def test_schema_two_records_model_identity(self) -> None:
        """Schema 2 replaced the flat spec with the model/workload split."""
        self.assertEqual(KERNEL_RESULTS_SCHEMA_VERSION, 2)
        payload = sample_result().to_dict()
        self.assertNotIn("spec", payload)
        self.assertEqual(payload["model_size"], "normal")
        self.assertEqual(payload["model_shape"]["dim"], 1024)
        self.assertEqual(payload["workload"], {"batch": 4, "seq_len": 1024})

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

    def test_unsupported_schema_rejected(self) -> None:
        # 1 is the pre-split schema; rejecting it is the point of the bump.
        for version in (1, 99):
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "results.json"
                payload = sample_result().to_dict()
                payload["schema_version"] = version
                path.write_text(json.dumps(payload))
                with self.assertRaisesRegex(ValueError, "unsupported"):
                    load_kernel_results(path)


if __name__ == "__main__":
    unittest.main()
