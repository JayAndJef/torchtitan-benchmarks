"""Component-attribution tests over synthetic traces (no GPU needed).

The last test is the load-bearing one: it re-derives a real run's
``gpu_time.kernel_ms_per_step`` from its traces and requires an exact match,
so the classifier can never drift away from the harness's own total. It skips
when the run is absent, because ``out/`` is gitignored.
"""

import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.components import (
    CLASS_ORDER,
    ModelShapes,
    analyze_arm,
    classify_megatron,
    classify_titan,
)


SHAPES = ModelShapes()


class TraceBuilder:
    """Assemble a chrome trace with a real device -> launch -> cpu_op chain."""

    def __init__(self):
        self.events = []
        self._next = 0
        self.clock = 0.0

    def step(self, name="ProfilerStep#20"):
        self.events.append(
            {"ph": "X", "cat": "user_annotation", "name": name, "tid": 1,
             "ts": 0.0, "dur": 1_000_000.0}
        )

    def launch(self, frames, kernel, duration, *, dims=None, stream=7, at=None):
        """One kernel, launched under a nest of cpu_op frames."""
        self._next += 1
        token = self._next
        start = self.clock if at is None else at
        span = max(duration * 4, 40.0)
        for depth, frame in enumerate(frames):
            self.events.append(
                {"ph": "X", "cat": "cpu_op", "name": frame, "tid": 1,
                 "ts": start + depth, "dur": span - 2 * depth,
                 "args": {"External id": token,
                          **({"Input Dims": dims} if dims is not None else {})}}
            )
        self.events.append(
            {"ph": "X", "cat": "cuda_runtime", "name": "cudaLaunchKernel", "tid": 1,
             "ts": start + len(frames) + 1, "dur": 5.0,
             "args": {"correlation": token, "External id": token}}
        )
        self.events.append(
            {"ph": "X", "cat": "kernel", "name": kernel, "pid": 0, "tid": stream,
             "ts": start + 100.0, "dur": duration,
             "args": {"correlation": token}}
        )
        if at is None:
            self.clock += span + 200.0
        return self

    def write(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(path, "wt") as handle:
            json.dump({"traceEvents": self.events}, handle)
        return path


class ClassificationTest(unittest.TestCase):
    def test_frame_first_beats_shape_first_on_grouped_linear_backward(self):
        """The trap that misfiled 122.8 ms/step of MoE backward.

        Megatron reports _GroupedLinearBackward's Input Dims with the expert
        dimensions stripped, so the shape looks like an ordinary projection
        GEMM. Only the frame identifies it.
        """
        dims = [[98304, 1024], []]
        self.assertEqual(
            classify_megatron("nvjet_tst_128", ("_GroupedLinearBackward",), dims, SHAPES),
            "moe_expert_gemm",
        )
        # Same kernel and same shape, different frame -> a different class.
        self.assertEqual(
            classify_megatron("nvjet_tst_128", ("_LayerNormLinear",), dims, SHAPES),
            "attn_projection_gemm",
        )

    def test_non_gemm_kernels_inside_the_grouped_frame_still_count(self):
        self.assertEqual(
            classify_megatron("elementwise_kernel", ("_GroupedLinear",), None, SHAPES),
            "moe_expert_gemm",
        )

    def test_layer_norm_linear_frame_splits_by_kernel_name(self):
        """One frame holds both the norm and the projection GEMM."""
        frames = ("_LayerNormLinear",)
        self.assertEqual(classify_megatron("ln_fwd_kernel", frames, None, SHAPES), "norm")
        self.assertEqual(
            classify_megatron("nvjet_hst_64", frames, None, SHAPES),
            "attn_projection_gemm",
        )

    def test_router_gemm_matches_expert_axis_on_either_side(self):
        """The weight-grad GEMM carries num_experts as its LEADING axis."""
        forward = [[49152, 1024], [1024, 4], [49152, 4]]
        weight_grad = [[4, 49152], [49152, 1024], [4, 1024]]
        for dims in (forward, weight_grad):
            self.assertEqual(
                classify_titan("triton_mm", ("aten::mm",), dims, SHAPES),
                "moe_routing_permute",
            )

    def test_batch_four_activation_is_not_mistaken_for_the_router(self):
        """num_experts == 4 collides with a batch of 4; 2-D-only saves it."""
        dims = [[4, 1024, 64], [4, 64, 1024]]
        self.assertEqual(
            classify_titan("triton_bmm", ("aten::bmm",), dims, SHAPES),
            "attn_projection_gemm",
        )

    def test_every_attention_backend_lands_in_attention_core(self):
        """A backend the classifier does not know reads as 0.0 attention.

        This actually happened: FlashAttention-3's CUTLASS kernels fell into
        other_elementwise, so the FA3 arm reported attention_core = 0.000 and
        looked like it did no attention at all.
        """
        cases = [
            "triton_tem_fused_flex_attention_0",
            "void cutlass::device_kernel<flash::enable_sm90<flash::FlashAttnFwdSm90<...>>>",
            "void cutlass::device_kernel<flash::FlashAttnBwdPreprocess<...>>",
            "void flash::prepare_varlen_num_blocks_kernel<1, true>(...)",
            "void pytorch_flash::flash_fwd_kernel<...>",
            "cudnn_generated_fort_native_sdpa_sm90_flash_kernel",
        ]
        for kernel in cases:
            with self.subTest(kernel=kernel[:40]):
                self.assertEqual(
                    classify_titan(kernel, ("aten::foo",), None, SHAPES),
                    "attention_core",
                )

    def test_lm_head_wins_over_the_router_rule(self):
        dims = [[49152, 1024], [1024, 151936]]
        self.assertEqual(
            classify_titan("triton_mm", ("aten::mm",), dims, SHAPES), "lm_head_gemm"
        )


class AggregationTest(unittest.TestCase):
    def _arm(self, builder: TraceBuilder, label="arm"):
        with tempfile.TemporaryDirectory() as tmp:
            path = builder.write(Path(tmp) / "rank0_trace.json.gz")
            return analyze_arm([path], label, SHAPES)

    def test_classes_partition_the_total(self):
        builder = TraceBuilder()
        builder.step()
        builder.launch(("_GroupedLinear",), "nvjet_a", 100.0)
        builder.launch(("_LayerNormLinear",), "ln_fwd", 50.0)
        builder.launch(("FusedAttnFunc",), "cudnn_sdpa", 25.0)
        builder.launch(("something::unknown",), "mystery_kernel", 7.0)
        arm = self._arm(builder)

        self.assertEqual(sum(arm.summed_us.values()), 182.0)
        self.assertEqual(arm.summed_total_us, 182.0)
        self.assertEqual(sum(arm.counts.values()), arm.total_events)
        self.assertTrue(set(arm.summed_us) <= set(CLASS_ORDER))

    def test_union_is_less_than_summed_when_streams_overlap(self):
        builder = TraceBuilder()
        builder.step()
        # Same wall-clock window, two different streams: real concurrency.
        builder.launch(("_GroupedLinear",), "nvjet_a", 100.0, stream=7, at=0.0)
        builder.launch(("_GroupedLinear",), "nvjet_b", 100.0, stream=8, at=0.0)
        arm = self._arm(builder)

        self.assertEqual(arm.summed_total_us, 200.0)
        self.assertEqual(arm.busy_total_us, 100.0)
        self.assertEqual(len(arm.streams), 2)

    def test_union_equals_summed_on_a_single_stream(self):
        builder = TraceBuilder()
        builder.step()
        builder.launch(("_GroupedLinear",), "nvjet_a", 100.0, stream=7, at=0.0)
        builder.launch(("_GroupedLinear",), "nvjet_b", 100.0, stream=7, at=5000.0)
        arm = self._arm(builder)

        self.assertEqual(arm.summed_total_us, 200.0)
        self.assertEqual(arm.busy_total_us, 200.0)

    def test_fused_rope_surfaces_as_a_bound_not_a_zero(self):
        builder = TraceBuilder()
        builder.step()
        frames = ("Torch-Compiled Region: 1/0", "aten::_fused_rms_norm")
        builder.launch(frames, "triton_poi_fused_rms_norm_neg_0", 40.0)
        builder.launch(frames, "triton_poi_fused_rms_norm_1", 10.0)
        arm = self._arm(builder)

        self.assertEqual(arm.engine, "titan")
        self.assertEqual(arm.summed_us.get("norm_rope_fused"), 40.0)
        self.assertEqual(arm.summed_us.get("norm"), 10.0)
        # RoPE is never credited as free: it has no row of its own, and the
        # bound is what gets rendered.
        self.assertEqual(arm.summed_us.get("rope", 0.0), 0.0)
        self.assertEqual(arm.fused_bound_us, 40.0)

    def test_loss_region_epilogue_joins_cross_entropy(self):
        """The reduction epilogue carries no CE-specific name."""
        builder = TraceBuilder()
        builder.step()
        region = "Region: 2/0"
        builder.launch((region, "aten::foo"), "piper_optimized_cross_entropy_kernel", 90.0)
        builder.launch((region, "aten::sum"), "triton_red_fused_sum_0", 4.0)
        builder.launch(("aten::elsewhere",), "triton_red_fused_sum_0", 3.0)
        arm = self._arm(builder)

        self.assertEqual(arm.summed_us.get("cross_entropy"), 94.0)
        self.assertEqual(arm.summed_us.get("other_elementwise"), 3.0)

    def test_attribution_health_counts_unlaunched_events(self):
        builder = TraceBuilder()
        builder.step()
        builder.launch(("_GroupedLinear",), "nvjet_a", 100.0)
        builder.events.append(
            {"ph": "X", "cat": "kernel", "name": "multi_tensor_apply_kernel",
             "pid": 0, "tid": 7, "ts": 900_000.0, "dur": 10.0, "args": {}}
        )
        arm = self._arm(builder)

        self.assertEqual(arm.no_launch_events, 1)
        self.assertEqual(arm.rule_compared, 1)
        self.assertEqual(arm.rule_agree, 1)
        # An orphan kernel still lands in a class: the partition is total.
        self.assertEqual(arm.summed_total_us, 110.0)
        self.assertEqual(arm.summed_us.get("optimizer"), 10.0)


class ReproductionGateTest(unittest.TestCase):
    """The column total must equal the harness's own number, exactly."""

    RUN = (
        Path(__file__).resolve().parent.parent
        / "out/20260808-b48-default/piper1b_megatron/nvidia-h200"
    )

    def test_reproduces_results_json_kernel_ms_per_step(self):
        if not (self.RUN / "results.json").exists():
            self.skipTest(f"{self.RUN} not present (out/ is gitignored)")
        expected = json.loads((self.RUN / "results.json").read_text())["gpu_time"]
        from benchmarks.artifacts import trace_files

        for arm_name, metrics in expected.items():
            target = metrics.get("kernel_ms_per_step")
            if target is None:
                continue
            with self.subTest(arm=arm_name):
                arm = analyze_arm(
                    trace_files(self.RUN / arm_name), arm_name, SHAPES
                )
                self.assertAlmostEqual(
                    arm.ms_per_step(arm.summed_total_us), target, places=4
                )
                # Classes partition it, so the rows must sum to the same total.
                self.assertAlmostEqual(
                    sum(
                        arm.ms_per_step(arm.summed_us.get(component, 0.0))
                        for component in CLASS_ORDER
                    ),
                    target,
                    places=4,
                )


if __name__ == "__main__":
    unittest.main()
