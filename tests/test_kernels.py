"""CPU-only tests for the kernel-benchmark registry and statistics."""

import importlib
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kernel.registry import (
    KERNEL_SCENARIOS,
    KernelWorkload,
    MODES,
    kernel_scenario_by_name,
    resolve_shape_and_workload,
    routing_divides_evenly,
    shape_summary,
)
from benchmarks.kernel.engine.statistics import kernel_comparison


class RegistryTests(unittest.TestCase):
    def test_expected_scenarios_and_baselines(self) -> None:
        self.assertEqual(
            list(KERNEL_SCENARIOS),
            ["rope", "swiglu", "qkv", "lm_head", "attention"],
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

    def test_attention_arms(self) -> None:
        scenario = KERNEL_SCENARIOS["attention"]
        self.assertEqual(
            [arm.name for arm in scenario.arms],
            ["baseline", "flex_flash", "flash_attention_3"],
        )
        # No isolated backward: see _attention_arm's docstring.
        for arm in scenario.arms:
            self.assertEqual(arm.modes, ("forward", "forward_backward"))
            self.assertTrue(arm.compiled, arm.name)

    def test_builder_paths_resolve_without_importing_torch(self) -> None:
        # Registry import must stay torch-free; the dotted paths just need
        # to be well-formed module:function references.
        registry = importlib.import_module("benchmarks.kernel.registry")
        self.assertNotIn("torch", vars(registry))
        for scenario in KERNEL_SCENARIOS.values():
            references = [scenario.inputs_builder] + [
                arm.builder for arm in scenario.arms
            ]
            if scenario.reference_builder:
                references.append(scenario.reference_builder)
            for reference in references:
                module, _, function = reference.partition(":")
                self.assertEqual(module, "benchmarks.kernel.operations.arms")
                self.assertTrue(function.isidentifier(), reference)

    def test_only_raw_kernels_and_floors_stay_eager(self) -> None:
        """Eager arms are deliberate: ``rope/copy_floor`` is a bandwidth
        floor, not an implementation."""
        eager = {
            (scenario.name, arm.name)
            for scenario in KERNEL_SCENARIOS.values()
            for arm in scenario.arms
            if not arm.compiled
        }
        self.assertEqual(eager, {("rope", "copy_floor")})

    def test_every_arm_has_a_description(self) -> None:
        from benchmarks.e2e.registry import SCENARIOS

        for registry in (KERNEL_SCENARIOS, SCENARIOS):
            for scenario in registry.values():
                for arm in scenario.arms:
                    self.assertTrue(
                        arm.description.strip(),
                        f"{scenario.name}/{arm.name} needs a description",
                    )

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

    def test_only_swiglu_needs_balanced_routing(self) -> None:
        balanced = {
            scenario.name
            for scenario in KERNEL_SCENARIOS.values()
            if scenario.requires_balanced_routing
        }
        self.assertEqual(balanced, {"swiglu"})


class ShapeAndWorkloadTests(unittest.TestCase):
    def test_shape_arithmetic(self) -> None:
        shape, workload = resolve_shape_and_workload()
        self.assertEqual((workload.batch, workload.seq_len), (4, 1024))
        swiglu = shape_summary("swiglu", shape, workload)
        self.assertEqual(swiglu["x"], [8192, 1024])
        self.assertEqual(swiglu["tokens_per_expert"], [2048] * 4)
        qkv = shape_summary("qkv", shape, workload)
        self.assertEqual(qkv["wqkv"], [2048, 1024])
        self.assertEqual(qkv["wk"], [512, 1024])
        lm_head = shape_summary("lm_head", shape, workload)
        self.assertEqual(lm_head["tokens"], 4096)
        self.assertEqual(lm_head["weight"], [151936, 1024])
        rope = shape_summary("rope", shape, workload)
        self.assertEqual(rope["q"], [4, 1024, 16, 64])
        self.assertEqual(rope["k"], [4, 1024, 8, 64])

    def test_model_size_selects_the_geometry_not_the_workload(self) -> None:
        shape, workload = resolve_shape_and_workload(model_size="huge")
        self.assertEqual(shape.name, "huge")
        self.assertEqual(shape.dim, 12288)
        # The workload is untouched by the size: batch is not a model property.
        self.assertEqual(workload, KernelWorkload())
        self.assertEqual(
            shape_summary("lm_head", shape, workload)["hidden"],
            [4, 1024, 12288],
        )
        with self.assertRaisesRegex(ValueError, "Unknown model size"):
            resolve_shape_and_workload(model_size="enormous")

    def test_workload_overrides_apply(self) -> None:
        shape, workload = resolve_shape_and_workload(batch=1, seq_len=2048)
        self.assertEqual(
            shape_summary("swiglu", shape, workload)["x"], [4096, 1024]
        )

    def test_seq_len_is_bounded_by_the_shapes_ceiling(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            resolve_shape_and_workload(seq_len=4096)

    def test_max_seq_len_override_replaces_the_shape_before_the_check(
        self,
    ) -> None:
        """Ordering is load-bearing: the attention sweep raises the ceiling
        precisely so it can then set seq_len above the old one."""
        shape, workload = resolve_shape_and_workload(
            seq_len=4096, max_seq_len=4096
        )
        self.assertEqual(shape.max_seq_len, 4096)
        self.assertEqual(workload.seq_len, 4096)
        # replace() on the registered shape, not a mutation of it.
        self.assertEqual(shape.name, "normal")
        self.assertEqual(resolve_shape_and_workload()[0].max_seq_len, 2048)
        self.assertEqual(
            shape_summary("attention", shape, workload)["max_seq_len"], 4096
        )

    def test_routing_needs_an_even_row_count(self) -> None:
        """rows = batch * seq_len * top_k, and both shapes carry top_k 2 with
        4 experts, so the split fails exactly when batch * seq_len is odd."""
        shape, workload = resolve_shape_and_workload(batch=3, seq_len=1025)
        self.assertFalse(routing_divides_evenly(shape, workload))  # 6150 rows
        # An odd batch alone is fine at the default even seq_len: 3 * 1024 * 2
        # = 6144 rows, which 4 experts do split.
        self.assertTrue(
            routing_divides_evenly(*resolve_shape_and_workload(batch=3))
        )
        self.assertTrue(
            routing_divides_evenly(*resolve_shape_and_workload(batch=4))
        )

    def test_split_covers_every_former_spec_field(self) -> None:
        """The twelve fields the single flat spec used to carry are all still
        reachable, each from exactly one of the two objects."""
        shape, workload = resolve_shape_and_workload()
        geometry = (
            "dim",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "num_experts",
            "top_k",
            "moe_hidden_dim",
            "vocab_size",
            "max_seq_len",
        )
        for name in geometry:
            self.assertTrue(hasattr(shape, name), name)
            # No facade: the workload must not answer geometry questions.
            self.assertFalse(hasattr(workload, name), name)
        for name in ("batch", "seq_len"):
            self.assertTrue(hasattr(workload, name), name)
            self.assertFalse(hasattr(shape, name), name)
        # The twelfth: the old spec's "theta" is PiperShape's "rope_theta".
        self.assertEqual(shape.rope_theta, 1_000_000.0)
        self.assertFalse(hasattr(shape, "theta"))


class BalancedRoutingInvariantTests(unittest.TestCase):
    """The invariant must hold at every entry point, not just the runner's.

    ``execute_kernel_run`` skips an unbalanced swiglu scenario loudly before
    it spawns a worker, but that is the friendly path, not the guard:
    ``python -m benchmarks.kernel.worker`` and direct ``run_kernel_scenario``
    callers (``tests/test_kernel_gpu_smoke.py``) never pass through it. Left
    unchecked there, ``swiglu_inputs`` builds ``batch * seq_len * top_k`` rows
    and then splits them into ``num_experts`` equal blocks that do not cover
    them -- a different workload measured under the scenario's name.
    """

    def test_run_kernel_scenario_rejects_an_uneven_split(self) -> None:
        from benchmarks.kernel.engine.run import RunOptions, run_kernel_scenario

        shape, workload = resolve_shape_and_workload(batch=3, seq_len=1025)
        with self.assertRaises(ValueError) as caught:
            run_kernel_scenario(
                kernel_scenario_by_name("swiglu"),
                shape,
                workload,
                RunOptions(),
                "test",
            )
        message = str(caught.exception)
        # The numbers, not just a complaint: 3 x 1025 x 2 rows over 4 experts.
        self.assertIn("6150 routed rows", message)
        self.assertIn("batch 3 x seq 1025 x top_k 2", message)
        self.assertIn("4 experts", message)
        # Raised before the CUDA check, so a CPU-only host -- where this test
        # runs -- diagnoses the workload rather than the missing device.


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
