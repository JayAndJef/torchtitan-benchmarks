"""CPU-only tests for the kernel-benchmark registry and statistics."""

import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kernels import (
    KERNEL_SCENARIOS,
    MODES,
    Piper1BSpec,
    kernel_scenario_by_name,
    shape_summary,
    spec_with_overrides,
)
from benchmarks.kernel_stats import kernel_comparison


class RegistryTests(unittest.TestCase):
    def test_expected_scenarios_and_baselines(self) -> None:
        self.assertEqual(
            list(KERNEL_SCENARIOS), ["rope", "swiglu", "qkv", "lm_head"]
        )
        for scenario in KERNEL_SCENARIOS.values():
            self.assertIn(
                scenario.baseline_arm, [arm.name for arm in scenario.arms]
            )
            for arm in scenario.arms:
                self.assertTrue(set(arm.modes) <= set(MODES), arm.name)
                if arm.compare_to is not None:
                    opponent = scenario.arm(arm.compare_to)
                    self.assertTrue(set(arm.modes) & set(opponent.modes))
                for check in arm.correctness:
                    self.assertIn(
                        check.kind, ("bitwise", "tolerance", "fp64_ulp")
                    )
                    if check.reference == "fp64":
                        self.assertIsNotNone(scenario.reference_builder)
                    else:
                        scenario.arm(check.reference)

    def test_builder_paths_resolve_without_importing_torch(self) -> None:
        # Registry import must stay torch-free; the dotted paths just need
        # to be well-formed module:function references.
        registry = importlib.import_module("benchmarks.kernels")
        self.assertNotIn("torch", vars(registry))
        for scenario in KERNEL_SCENARIOS.values():
            references = [scenario.inputs_builder] + [
                arm.builder for arm in scenario.arms
            ]
            if scenario.reference_builder:
                references.append(scenario.reference_builder)
            for reference in references:
                module, _, function = reference.partition(":")
                self.assertEqual(module, "benchmarks.kernel_arms")
                self.assertTrue(function.isidentifier(), reference)

    def test_only_te_requires_gcc_toolset(self) -> None:
        self.assertTrue(kernel_scenario_by_name("rope").requires_gcc_toolset)
        for name in ("swiglu", "qkv", "lm_head"):
            self.assertFalse(
                kernel_scenario_by_name(name).requires_gcc_toolset
            )

    def test_unknown_names_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown kernel scenario"):
            kernel_scenario_by_name("nope")
        with self.assertRaisesRegex(ValueError, "Unknown arm"):
            kernel_scenario_by_name("rope").arm("nope")

    def test_spec_shape_arithmetic(self) -> None:
        spec = Piper1BSpec()
        spec.validate()
        swiglu = shape_summary("swiglu", spec)
        self.assertEqual(swiglu["x"], [8192, 1024])
        self.assertEqual(swiglu["gate_up"], [8192, 7168])
        self.assertEqual(swiglu["tokens_per_expert"], [2048] * 4)
        qkv = shape_summary("qkv", spec)
        self.assertEqual(qkv["wqkv"], [2048, 1024])
        self.assertEqual(qkv["wk"], [512, 1024])
        lm_head = shape_summary("lm_head", spec)
        self.assertEqual(lm_head["tokens"], 4096)
        self.assertEqual(lm_head["weight"], [151936, 1024])
        rope = shape_summary("rope", spec)
        self.assertEqual(rope["q"], [4, 1024, 16, 64])
        self.assertEqual(rope["k"], [4, 1024, 8, 64])

    def test_spec_overrides_validate(self) -> None:
        spec = spec_with_overrides(batch=1, seq_len=2048)
        self.assertEqual(shape_summary("swiglu", spec)["x"], [4096, 1024])
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            spec_with_overrides(seq_len=4096)


class KernelComparisonTests(unittest.TestCase):
    def test_shifted_distributions_are_detected(self) -> None:
        base = [100.0 + 0.1 * (i % 7) for i in range(50)]
        arm = [110.0 + 0.1 * (i % 5) for i in range(50)]
        row = kernel_comparison(base, arm)
        self.assertAlmostEqual(row["median_ratio"], 1.1, places=1)
        self.assertLess(row["welch_p"], 1e-6)
        self.assertLess(row["mwu_p"], 1e-6)
        self.assertLess(row["wilcoxon_p"], 1e-6)
        self.assertGreater(row["cohens_d"], 2.0)
        self.assertEqual(row["arm_faster_fraction"], 0.0)

    def test_identical_samples_guard_wilcoxon(self) -> None:
        values = [100.0] * 20
        row = kernel_comparison(values, list(values))
        self.assertIsNone(row["wilcoxon_p"])
        self.assertAlmostEqual(row["median_ratio"], 1.0)
        self.assertEqual(row["cohens_d"], 0.0)

    def test_pooled_sd_is_df_weighted(self) -> None:
        base = [10.0, 12.0, 14.0, 16.0]
        arm = [11.0, 13.0]
        row = kernel_comparison(base, arm)
        import statistics

        sd_base = statistics.stdev(base)
        sd_arm = statistics.stdev(arm)
        pooled = (
            ((len(base) - 1) * sd_base**2 + (len(arm) - 1) * sd_arm**2)
            / (len(base) + len(arm) - 2)
        ) ** 0.5
        expected = (statistics.mean(arm) - statistics.mean(base)) / pooled
        self.assertAlmostEqual(row["cohens_d"], expected)


if __name__ == "__main__":
    unittest.main()
