"""Component-attribution tests over synthetic traces (no GPU needed).

The last test is the load-bearing one: it re-derives a real run's
``gpu_time.kernel_ms_per_step`` from its traces and requires an exact match,
so the classifier can never drift away from the harness's own total. It skips
when the run is absent, because ``out/`` is gitignored.
"""

import gzip
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.components import (
    CLASS_ORDER,
    ModelShapes,
    analyze_arm,
    classify_megatron,
    classify_titan,
    render,
    to_json,
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
    def test_megatron_te_cross_entropy_uses_the_autograd_frame(self):
        cases = (
            ("online_softmax_kernel", ("CrossEntropyFunction",)),
            ("cross_entropy_kernel", ("CrossEntropyFunction",)),
            (
                "element_mul_kernel",
                (
                    "autograd::engine::evaluate_function: CrossEntropyFunctionBackward",
                    "CrossEntropyFunctionBackward",
                ),
            ),
        )
        for kernel, frames in cases:
            with self.subTest(kernel=kernel):
                self.assertEqual(
                    classify_megatron(kernel, frames, None, SHAPES),
                    "cross_entropy",
                )

        # The kernel name is generic; only its semantic loss frame makes it CE.
        self.assertEqual(
            classify_megatron("element_mul_kernel", ("aten::mul",), None, SHAPES),
            "other_elementwise",
        )
        self.assertEqual(
            classify_megatron(
                "native_ce_kernel", ("_VocabParallelCrossEntropy",), None, SHAPES
            ),
            "cross_entropy",
        )

    def test_titan_stock_and_te_cross_entropy_paths(self):
        vocab_dims = [[4, 1024, 151936]]
        stock = (
            "triton_red_fused__log_softmax__to_copy_prepare_softmax_online_view_0",
            "triton_red_fused__log_softmax__log_softmax_backward_data__to_copy_"
            "arange_eq_expand_nll_loss_backward_scalar_tensor_view_0",
        )
        for kernel in stock:
            with self.subTest(kernel=kernel[:50]):
                self.assertEqual(
                    classify_titan(
                        kernel, ("CompiledFunction",), vocab_dims, SHAPES
                    ),
                    "cross_entropy",
                )

        for kernel in (
            "element_mul_kernel",
            "void at::native::vectorized_elementwise_kernel<8, copy_functor>",
        ):
            with self.subTest(kernel=kernel[:50]):
                self.assertEqual(
                    classify_titan(
                        kernel, ("CrossEntropyFunctionBackward",), None, SHAPES
                    ),
                    "cross_entropy",
                )

        self.assertEqual(
            classify_titan(
                "void at::native::reduce_kernel<512, 1>",
                ("torch_nn::_linear_cross_entropy_batch_chunked",),
                None,
                SHAPES,
            ),
            "cross_entropy",
        )
        self.assertEqual(
            classify_titan(
                "generic_loss_kernel", ("CrossEntropyLoss",), None, SHAPES
            ),
            "cross_entropy",
        )

        # Neither a generic mul nor the MoE router's plain softmax is CE.
        self.assertEqual(
            classify_titan("element_mul_kernel", ("aten::mul",), None, SHAPES),
            "other_elementwise",
        )
        self.assertEqual(
            classify_titan(
                "triton_poi_fused__softmax__unsafe_view_prepare_softmax_online_9",
                ("CompiledFunction",),
                [[4, 1024, 4]],
                SHAPES,
            ),
            "other_elementwise",
        )
        self.assertEqual(
            classify_titan(
                "triton_red_fused__log_softmax__to_copy_0",
                ("CompiledFunction",),
                [[4, 1024, 4]],
                SHAPES,
            ),
            "other_elementwise",
        )

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
            # FlashAttention-4, as emitted by flex's FLASH backend. Real
            # symbols, truncated. Note these spell "FlashAttention" out, so
            # the abbreviated FA3 marker above does not match them.
            "kernel_cutlass_kernel_flash_attncuteflash_fwd_sm90FlashAttentionForwardSm90_object_at__tensor0000o2",
            "kernel_cutlass_kernel_flash_attncuteflash_bwd_sm90FlashAttentionBackwardSm90_object_at__tensor0000o1",
            "kernel_cutlass_kernel_flash_attncuteflash_bwd_preprocessFlashAttentionBackwardPreprocess_object_at_",
            "kernel_cutlass_kernel_flash_attncuteflash_bwd_postprocessFlashAttentionBackwardPostprocess_object_a",
        ]
        for kernel in cases:
            with self.subTest(kernel=kernel[:40]):
                self.assertEqual(
                    classify_titan(kernel, ("aten::foo",), None, SHAPES),
                    "attention_core",
                )

    def test_override_rope_kernels_are_not_filed_as_elementwise(self):
        """The rope arms replace the fused stock path with their own kernel."""
        for kernel in (
            "_helion__rope_cos_sin_fwd",
            "fused_rope_forward_positions_kernel",
            "fused_rope_backward_positions_kernel",
        ):
            with self.subTest(kernel=kernel):
                self.assertEqual(
                    classify_titan(kernel, ("aten::foo",), None, SHAPES), "rope"
                )

    def test_fused_stock_rope_still_reports_as_a_bound(self):
        self.assertEqual(
            classify_titan(
                "triton_poi_fused_rms_norm_neg_0", ("aten::foo",), None, SHAPES
            ),
            "norm_rope_fused",
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

    def test_classification_audit_warns_and_surfaces_the_fallback(self):
        builder = TraceBuilder()
        builder.step()
        builder.launch(("something::unknown",), "mystery_kernel", 7.0)
        arm = self._arm(builder)

        output = io.StringIO()
        with redirect_stdout(output):
            render([arm], {"compile_mode": "default"}, False, False)
        rendered = output.getvalue()
        self.assertIn("WARNING: expected non-zero component(s) missing", rendered)
        self.assertIn("cross_entropy", rendered)
        self.assertIn("mystery_kernel", rendered)

        payload = to_json([arm], {"compile_mode": "default"})
        self.assertEqual(payload["schema_version"], 2)
        audit = payload["arms"][0]["classification_audit"]
        self.assertTrue(audit["expected_component_check_applied"])
        self.assertIn("cross_entropy", audit["missing_expected_components"])
        self.assertEqual(
            audit["largest_other_elementwise"][0]["kernel"], "mystery_kernel"
        )

        skipped_output = io.StringIO()
        with redirect_stdout(skipped_output):
            render([arm], {"compile_mode": "cuda-graph"}, False, False)
        self.assertNotIn("WARNING:", skipped_output.getvalue())
        self.assertIn("expected-zero check skipped", skipped_output.getvalue())


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
