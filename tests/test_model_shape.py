"""CPU-only tests for the --model-size axis.

Covers the shape arithmetic (pinned against the numbers a real run logs),
the closure of every scenario arm's config over every registered size, the
derivation of regions and override counts from the shape, and the
manifest/resume plumbing.
"""

import gzip
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.manifests import write_manifest
from benchmarks.e2e.launch import command_for_arm
from benchmarks.e2e.registry import (
    SCENARIOS,
    piper_block_regions,
    scenario_by_name,
)
from benchmarks.e2e.runner import RunRequest, execute_run
from benchmarks.e2e.validation import validate_arm
from benchmarks.execution.environment import CpuPinning
from benchmarks.models.piper_qwen3.shape import (
    HUGE,
    NORMAL,
    PIPER_SHAPES,
    PiperShape,
)
from tests.test_runner import _SAC_LINE, _compiled_line


class ShapeArithmeticTests(unittest.TestCase):
    def test_normal_matches_the_numbers_a_real_run_logs(self) -> None:
        # These four are the exact values torchtitan prints
        # ("Total parameter count: dense D, sparse S, vision 0, active A")
        # and the constant tools/megatron_parity_check.py asserts. They were
        # duplicated by hand in the megatron baseline before schema 9.
        self.assertEqual(NORMAL.param_count, 1_066_241_024)
        self.assertEqual(NORMAL.nparams_dense, 361_532_416)
        self.assertEqual(NORMAL.nparams_sparse, 704_708_608)
        self.assertEqual(NORMAL.nparams_active, 713_919_488)
        self.assertEqual(NORMAL.num_flops_per_token(1024), 3_551_348_736)

    def test_normal_geometry(self) -> None:
        self.assertEqual(
            (NORMAL.dim, NORMAL.n_layers, NORMAL.n_heads, NORMAL.n_kv_heads),
            (1024, 16, 16, 8),
        )
        self.assertEqual(NORMAL.moe_hidden_dim, 3584)
        self.assertEqual(NORMAL.qkv_out_features, 2048)
        self.assertEqual(NORMAL.heads_per_group, 2)
        self.assertTrue(NORMAL.supports_block_regions)

    def test_huge_geometry_is_derived_not_hardcoded(self) -> None:
        self.assertEqual(HUGE.n_layers, 1)
        self.assertEqual(HUGE.n_heads, HUGE.dim // 64)
        self.assertEqual(HUGE.n_kv_heads, HUGE.n_heads // 2)
        self.assertEqual(HUGE.moe_hidden_dim, HUGE.dim * 7 // 2)
        self.assertEqual(HUGE.qkv_out_features, 2 * HUGE.dim)
        self.assertEqual(HUGE.head_dim, NORMAL.head_dim)
        self.assertEqual(HUGE.vocab_size, NORMAL.vocab_size)
        # The reason the huge shape exists: at one layer the embedding tables
        # must not dominate. embedding+lm_head / one layer = 6753/dim.
        self.assertLess(2 * HUGE.vocab_size / (45 * HUGE.dim), 1.0)
        self.assertFalse(HUGE.supports_block_regions)

    def test_dim_must_admit_integral_kv_heads(self) -> None:
        with self.assertRaisesRegex(ValueError, "2\\*head_dim"):
            PiperShape(name="bad", dim=1000, n_layers=1)
        with self.assertRaisesRegex(ValueError, "n_layers"):
            PiperShape(name="bad", dim=1024, n_layers=0)

    def test_describe_is_json_safe(self) -> None:
        for shape in PIPER_SHAPES.values():
            described = shape.describe(seq_len=1024)
            self.assertEqual(json.loads(json.dumps(described)), described)
            self.assertEqual(described["name"], shape.name)

    def test_block_region_support_is_derived_from_the_layer_count(self) -> None:
        # Not a per-shape flag anyone can set wrong: a 1-layer block graph is
        # not structurally identifiable, at any dim.
        self.assertTrue(NORMAL.supports_block_regions)
        self.assertFalse(HUGE.supports_block_regions)
        self.assertFalse(
            PiperShape(name="probe", dim=1024, n_layers=1).supports_block_regions
        )
        self.assertTrue(
            PiperShape(name="probe", dim=1024, n_layers=2).supports_block_regions
        )

    def test_parity_gate_is_shape_data(self) -> None:
        # tools/megatron_parity_check.py reads these; the huge gate is wider
        # only because bf16 accumulation scales with the reduction length.
        self.assertEqual(NORMAL.parity_gate, 2e-2)
        self.assertEqual(HUGE.parity_gate, 5e-2)
        self.assertEqual(
            PiperShape(name="probe", dim=1024, n_layers=2).parity_gate, 2e-2
        )
        for shape in PIPER_SHAPES.values():
            self.assertEqual(
                shape.describe(seq_len=1024)["parity_gate"], shape.parity_gate
            )


class ConfigSizeClosureTests(unittest.TestCase):
    def test_every_scenario_arm_builds_at_every_size(self) -> None:
        """Every arm's config resolves and accepts every registered size.

        This is the closure the runner depends on: it emits ``--config <name>
        --config-arg size=<size>`` for whatever the arm names, so a config
        that did not accept the keyword -- or accepted it and ignored it --
        would publish a run under the wrong shape.
        """
        import inspect

        import benchmarks.models.piper_qwen3.config_registry as registry

        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                if arm.launcher != "torchtitan":
                    continue
                name = arm.config or scenario.workload.config
                with self.subTest(scenario=scenario.name, arm=arm.name):
                    factory = getattr(registry, name, None)
                    self.assertTrue(
                        callable(factory), f"{name} is not a config factory"
                    )
                    parameter = inspect.signature(factory).parameters.get("size")
                    self.assertIsNotNone(parameter, f"{name} takes no size")
                    self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                    self.assertEqual(parameter.default, "normal")
                    self._assert_every_size_lands(factory, name)

    def _assert_every_size_lands(self, factory, name: str) -> None:
        for size in tuple(PIPER_SHAPES):
            shape = PIPER_SHAPES[size]
            # One config gates its attention backend on the host GPU
            # (qwen3_piper_1b_flex_flash wants sm90+), which this CPU-only
            # suite does not have. Skipping it would retire the only check on
            # a config that accepts ``size`` and ignores it, so pretend the
            # capability instead: the fork imports has_cuda_capability inside
            # get_attention_config, so patching its source module reaches it,
            # and nothing here runs a kernel.
            with mock.patch(
                "torchtitan.tools.utils.has_cuda_capability", return_value=True
            ):
                model = factory(size=size).model_spec.model
            self.assertEqual(model.dim, shape.dim, f"{name} at {size}")
            self.assertEqual(len(model.layers), shape.n_layers, f"{name} at {size}")

    def test_the_private_builders_require_an_explicit_shape(self) -> None:
        """No default shape on the builders every public config calls.

        All eleven call sites pass ``shape=`` today, so a default could only
        ever be reached by a future omission -- and would then build the
        normal geometry silently, under whatever size the run asked for. That
        is the exact silent fallback the ``size`` keyword replaced.
        """
        import inspect

        from benchmarks.models.piper_qwen3.config_registry import (
            _piper_1b_model,
            _piper_1b_trainer,
        )

        for builder in (_piper_1b_model, _piper_1b_trainer):
            with self.subTest(builder=builder.__name__):
                parameter = inspect.signature(builder).parameters["shape"]
                self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(parameter.default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            _piper_1b_model(fuse_qkv=True)
        with self.assertRaises(TypeError):
            _piper_1b_trainer(fuse_qkv=True, loss_kind="full_logits")

    def test_the_megatron_builder_requires_an_explicit_shape(self) -> None:
        """The same rule on the other engine's builder, for the same reason.

        ``build_model`` is the megatron twin of ``_piper_1b_model``: both
        construct the same geometry from the same ``PiperShape``, and the
        whole point of that sharing is that a size cannot drift between the
        engines. A default here would reintroduce the drift on one side --
        and it would land in the arm with the least protection, since
        ``tools/megatron_parity_check.py`` builds the model outside the
        harness and so never reaches validation rule 11's parameter-count
        check. Signature inspection only; this imports no megatron.
        """
        import inspect

        from benchmarks.models.piper_qwen3.megatron_model import build_model

        parameter = inspect.signature(build_model).parameters["shape"]
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_size_round_trips_through_the_config_argument(self) -> None:
        from benchmarks.models.piper_qwen3.config_registry import qwen3_piper_1b

        self.assertEqual(qwen3_piper_1b(size="huge").model_spec.model.dim, 12288)
        self.assertEqual(qwen3_piper_1b(size="normal").model_spec.model.dim, 1024)
        # The default is the normal shape, so an unparameterized call is the
        # historical config.
        self.assertEqual(qwen3_piper_1b().model_spec.model.dim, NORMAL.dim)
        with self.assertRaisesRegex(ValueError, "Unknown model size"):
            qwen3_piper_1b(size="enormous")

    def test_built_models_carry_the_requested_shape(self) -> None:
        from benchmarks.models.piper_qwen3.config_registry import qwen3_piper_1b

        normal = qwen3_piper_1b().model_spec.model
        self.assertEqual(normal.dim, NORMAL.dim)
        self.assertEqual(len(normal.layers), NORMAL.n_layers)

        huge = qwen3_piper_1b(size="huge").model_spec.model
        self.assertEqual(huge.dim, HUGE.dim)
        self.assertEqual(len(huge.layers), HUGE.n_layers)
        self.assertEqual(huge.vocab_size, HUGE.vocab_size)
        # The load_balance_coeff=None fixup must survive the parameterization.
        self.assertIsNone(huge.layers[0].moe.load_balance_coeff)

    def test_pretokenized_configs_pass_the_size_down_to_their_delegate(self) -> None:
        from benchmarks.models.piper_qwen3.config_registry import (
            qwen3_piper_1b_piper_optimized_te_ce_pretokenized,
            qwen3_piper_1b_pretokenized,
        )

        for factory in (
            qwen3_piper_1b_pretokenized,
            qwen3_piper_1b_piper_optimized_te_ce_pretokenized,
        ):
            for size, shape in PIPER_SHAPES.items():
                with self.subTest(config=factory.__name__, size=size):
                    config = factory(size=size)
                    self.assertEqual(config.model_spec.model.dim, shape.dim)
                    # replay_steps tracks the config's own step count.
                    self.assertEqual(
                        config.dataloader.replay_steps, config.training.steps
                    )


class CommandTests(unittest.TestCase):
    def test_titan_command_delivers_the_size_as_a_config_argument(self) -> None:
        scenario = scenario_by_name("piper1b_megatron")
        arm = scenario.arm("titan_stock")
        command = command_for_arm(
            scenario.workload, arm, Path("/tmp/arm"), (), model_size="huge"
        )
        # The config name is the arm's, unmangled: the shape rides alongside.
        self.assertEqual(
            command[command.index("--config") + 1],
            "qwen3_piper_1b_pretokenized",
        )
        self.assertEqual(
            command[command.index("--config-arg") + 1], "size=huge"
        )
        self.assertFalse([token for token in command if token.endswith("_huge")])
        # replay_steps must track --training.steps or the loader hard-fails.
        self.assertEqual(
            command[command.index("--dataloader.replay-steps") + 1],
            command[command.index("--training.steps") + 1],
        )

    def test_the_default_size_is_delivered_explicitly_too(self) -> None:
        scenario = scenario_by_name("piper1b_rope")
        command = command_for_arm(
            scenario.workload, scenario.arm("baseline"), Path("/tmp/arm"), ()
        )
        self.assertEqual(
            command[command.index("--config") + 1], "qwen3_piper_1b"
        )
        self.assertEqual(
            command[command.index("--config-arg") + 1], "size=normal"
        )

    def test_non_replay_scenarios_do_not_get_the_replay_flag(self) -> None:
        scenario = scenario_by_name("piper1b_rope")
        command = command_for_arm(
            scenario.workload, scenario.arm("baseline"), Path("/tmp/arm"), ()
        )
        self.assertNotIn("--dataloader.replay-steps", command)

    def test_megatron_command_carries_the_model_size(self) -> None:
        scenario = scenario_by_name("piper1b_megatron")
        command = command_for_arm(
            scenario.workload,
            scenario.arm("baseline"),
            Path("/tmp/arm"),
            (),
            "default",
            "none",
            model_size="huge",
        )
        self.assertEqual(command[command.index("--model-size") + 1], "huge")

    def test_megatron_driver_accepts_the_flag(self) -> None:
        from benchmarks.e2e.megatron.train import parse_args

        parsed = parse_args(
            [
                "--seq-len", "1024", "--steps", "80", "--batch", "4",
                "--seed", "42", "--profile-freq", "20",
                "--profiler-warmup", "5", "--profiler-active", "5",
                "--mode", "default", "--model-size", "huge", "/tmp/arm",
            ]
        )
        self.assertEqual(parsed.model_size, "huge")
        self.assertEqual(parse_args(
            [
                "--seq-len", "1024", "--steps", "40", "--batch", "4",
                "--seed", "42", "--profile-freq", "20",
                "--profiler-warmup", "5", "--profiler-active", "5",
                "--mode", "default", "/tmp/arm",
            ]
        ).model_size, "normal")


class RegionDerivationTests(unittest.TestCase):
    def test_the_factory_reproduces_the_historical_counts(self) -> None:
        self.assertEqual(
            [
                (r.name, r.phase, r.invocations_per_window)
                for r in piper_block_regions(n_layers=16, profiler_active=5)
            ],
            [("backward_block", "backward", 80), ("forward_block", "forward", 80)],
        )

    def test_one_layer_regions_would_collide_so_huge_declares_none(self) -> None:
        # At one layer the block graph runs profiler_active times per window,
        # which the loss-side partitions also do -- the reason
        # supports_block_regions is False rather than rescaled to 5.
        self.assertEqual(
            piper_block_regions(n_layers=1, profiler_active=5)[0]
            .invocations_per_window,
            5,
        )
        self.assertFalse(HUGE.supports_block_regions)


def _size_line(shape) -> str:
    return (
        "[titan] - root - INFO - Model qwen3 piper_1B "
        f"size: {shape.param_count:,} total parameters\n"
    )


class ValidationRuleElevenTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        for iteration in ("iteration_20", "iteration_40"):
            trace = root / "profiling" / "traces" / iteration / "rank0_trace.json.gz"
            trace.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(trace, "wt") as trace_file:
                trace_file.write("cudaLaunchKernel\n")
        return root / "baseline.log"

    def test_a_log_without_the_size_marker_fails(self) -> None:
        scenario = scenario_by_name("piper1b_rope")
        arm = scenario.arm("baseline")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._fixture(root)
            head = _compiled_line("default") + _SAC_LINE

            log.write_text(head + "Training completed\n")
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate_arm(arm, root, log, scenario.workload)

            log.write_text(head + _size_line(NORMAL) + "Training completed\n")
            validate_arm(arm, root, log, scenario.workload)

            # The normal-size marker must not satisfy a huge-size run.
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate_arm(
                    arm, root, log, scenario.workload, model_size="huge"
                )

            log.write_text(head + _size_line(HUGE) + "Training completed\n")
            validate_arm(arm, root, log, scenario.workload, model_size="huge")

    def test_override_count_scales_with_the_layer_count(self) -> None:
        scenario = scenario_by_name("piper1b_swiglu")
        arm = scenario.arm("piper_optimized_triton")
        applied = (
            f"[Override] {arm.override_imports[0]}: "
            "model_spec.model.layers.0.moe ...\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for iteration in ("iteration_20", "iteration_40"):
                trace = (
                    root / "profiling" / "traces" / iteration / "rank0_trace.json.gz"
                )
                trace.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(trace, "wt") as trace_file:
                    trace_file.write("_combined_silu_and_mul_forward_kernel\n")
                    trace_file.write("_combined_silu_and_mul_backward_kernel\n")
            log = root / "arm.log"
            head = _compiled_line("default") + _SAC_LINE

            log.write_text(head + _size_line(HUGE) + "Training completed\n" + applied)
            validate_arm(
                arm, root, log, scenario.workload, model_size="huge"
            )

            log.write_text(
                head + _size_line(HUGE) + "Training completed\n" + applied * 16
            )
            with self.assertRaisesRegex(RuntimeError, "expected 1 override"):
                validate_arm(
                    arm, root, log, scenario.workload, model_size="huge"
                )

            log.write_text(
                head + _size_line(NORMAL) + "Training completed\n" + applied * 16
            )
            validate_arm(arm, root, log, scenario.workload)


_METADATA = {
    "requested_gpu": "0",
    "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
    "torch_version": "test",
    "torchtitan_git_rev": "titan-rev",
    "benchmarks_git_rev": "bench-rev",
}


def _fake_process(size_line: str):
    def run(command, **kwargs):
        kwargs["stdout"].write(
            _compiled_line("default") + size_line + "Training completed\n"
        )
        # --dump-folder is not last under ac=none (the tyro subcommand
        # token trails it), so look the argument up by name.
        arm_dir = Path(command[command.index("--dump-folder") + 1])
        for iteration in (20, 40):
            trace = (
                arm_dir
                / f"profiling/traces/iteration_{iteration}/rank0_trace.json.gz"
            )
            trace.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(trace, "wt") as trace_file:
                json.dump({"traceEvents": []}, trace_file)
        return SimpleNamespace(returncode=0)

    return run


class ManifestAndResumeTests(unittest.TestCase):
    def _run(self, **request_kwargs):
        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", _METADATA),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            return execute_run(
                RunRequest(gpu="0", **request_kwargs),
                process_runner=_fake_process(_size_line(HUGE)),
                environment={"PATH": os.environ["PATH"]},
            )

    def test_huge_run_declares_no_regions_and_records_the_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            self._run(
                scenario_name="piper1b_attention",
                arm_name="baseline",
                out_dir=out_dir,
                ac_mode="none",
                model_size="huge",
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())

        self.assertEqual(manifest["schema_version"], 9)
        self.assertEqual(manifest["model_size"], "huge")
        self.assertEqual(manifest["model_shape"], HUGE.describe(seq_len=1024))
        # Rule 7's structural matcher cannot identify a 1-layer block graph,
        # so the run says so instead of claiming a region it cannot verify.
        self.assertEqual(manifest["regions"], [])
        command = manifest["commands"]["baseline"]
        self.assertEqual(command[command.index("--config") + 1], "qwen3_piper_1b")
        self.assertEqual(command[command.index("--config-arg") + 1], "size=huge")

    def test_normal_run_still_declares_the_eighty_invocation_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            with mock.patch(
                "benchmarks.e2e.runner.hardware_metadata",
                return_value=("test-gpu", _METADATA),
            ), mock.patch(
                "benchmarks.e2e.runner.resolve_cpu_pinning",
                return_value=CpuPinning((), "none: test"),
            ), mock.patch("benchmarks.e2e.runner.validate_arm"):
                execute_run(
                    RunRequest(
                        gpu="0",
                        scenario_name="piper1b_rope",
                        arm_name="baseline",
                        out_dir=out_dir,
                    ),
                    process_runner=_fake_process(_size_line(NORMAL)),
                    environment={"PATH": os.environ["PATH"]},
                )
            manifest = json.loads((out_dir / "manifest.json").read_text())
        self.assertEqual(
            [
                (region["name"], region["invocations_per_window"])
                for region in manifest["regions"]
            ],
            [("backward_block", 80), ("forward_block", 80)],
        )

    def test_resume_refuses_a_different_size_and_inherits_an_absent_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            self._run(
                scenario_name="piper1b_attention",
                arm_name="baseline",
                out_dir=out_dir,
                ac_mode="none",
                model_size="huge",
            )

            with self.assertRaisesRegex(Exception, "model_size"):
                self._run(
                    scenario_name=None,
                    arm_name="baseline",
                    resume_dir=out_dir,
                    ac_mode="none",
                    model_size="normal",
                )

            # Omitting --model-size on a resume inherits the recorded value.
            self._run(
                scenario_name=None,
                arm_name="baseline",
                resume_dir=out_dir,
                ac_mode="none",
            )

    def test_schema_eight_directories_still_resume_as_normal(self) -> None:
        scenario = scenario_by_name("piper1b_rope")
        selected = (scenario.arm("baseline"),)
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            write_manifest(
                out_dir,
                scenario,
                selected,
                {"baseline": ["cmd"]},
                "test-gpu",
                _METADATA,
                (),
                "default",
                "sac",
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())
            del manifest["model_size"]
            del manifest["model_shape"]
            manifest["schema_version"] = 8
            (out_dir / "manifest.json").write_text(json.dumps(manifest))

            from benchmarks.artifacts.manifests import _resume_mismatches

            self.assertEqual(
                _resume_mismatches(
                    manifest,
                    scenario,
                    selected,
                    "test-gpu",
                    _METADATA,
                    (),
                    "default",
                    "sac",
                    "normal",
                ),
                [],
            )
            self.assertIn(
                "model_size",
                _resume_mismatches(
                    manifest,
                    scenario,
                    selected,
                    "test-gpu",
                    _METADATA,
                    (),
                    "default",
                    "sac",
                    "huge",
                ),
            )


if __name__ == "__main__":
    unittest.main()
