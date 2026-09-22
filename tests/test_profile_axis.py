"""The ``--profile`` run axis, and the trace layout it must not move.

Off is the default. The run then writes no trace, the 40-step floor does
not apply, and every trace rule of ``benchmarks/e2e/validation.py`` is
skipped. On, the two engines write the layout an external analysis tool
reads, and that layout is pinned here character for character: a run
directory is an interface, and a rename would break a reader this
repository does not hold.
"""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts import layout
from benchmarks.artifacts.manifests import (
    MANIFEST_SCHEMA_VERSION,
    _resume_mismatches,
    manifest_data,
)
from benchmarks.e2e.engines import command_for_arm
from benchmarks.e2e.megatron_stock import profiling
from benchmarks.e2e.megatron_stock.flags import (
    BENCH_PROFILE,
    BENCH_PROFILE_SCHEDULE_FLAGS,
    stock_megatron_flags,
)
from benchmarks.e2e.parallelism import TRIVIAL_SPEC
from benchmarks.e2e.axes import RunAxes
from benchmarks.e2e.registry import DEFAULT_PROFILE, ENGINES, scenario_by_name
from benchmarks.e2e.runner import workload_with_overrides
from benchmarks.e2e.validation import validate_arm
from benchmarks.models.piper_qwen3.shape import shape_by_name
from tests.test_runner import _SAC_LINE, _SIZE_LINE, _compiled_line


_METADATA = {
    "nvidia_smi": "test",
    "cpu_pinning": "none: test",
    "torchtitan_git_rev": "abc",
    "benchmarks_git_rev": "def",
    "megatron_git_rev": "ghi",
}

# Megatron's own profiler tokens, which stock Megatron defaults off.
_MEGATRON_PROFILER_FLAGS = (
    "--profile",
    "--use-pytorch-profiler",
    "--profile-step-start",
    "--profile-step-end",
)

# TorchTitan's own, which the fork defaults off too.
_TITAN_PROFILER_FLAGS = (
    "--profiler.enable_profiling",
    "--profiler.profile_freq",
    "--profiler.profiler_active",
    "--profiler.profiler_warmup",
)


class TraceLayoutIsPinnedTests(unittest.TestCase):
    """The layout both engines write and an external tool reads.

    Pinned as literals rather than derived, because a derivation moves with
    the code that writes it and this is a published interface. A change
    here must be a deliberate edit of this test.
    """

    def test_the_stock_driver_names_the_window_and_the_file(self) -> None:
        self.assertEqual(profiling.TRACE_SUBDIR, "profiling/traces")
        self.assertEqual(profiling.WINDOW_DIR, "iteration_{step}")
        self.assertEqual(profiling.TRACE_NAME, "rank{rank}_trace.json.gz")

    def test_the_reader_globs_exactly_what_the_writer_names(self) -> None:
        self.assertEqual(layout.TRACE_FILE_GLOB, "rank*_trace.json.gz")
        with tempfile.TemporaryDirectory() as temporary:
            arm_dir = Path(temporary)
            written = (
                arm_dir
                / profiling.TRACE_SUBDIR
                / profiling.WINDOW_DIR.format(step=40)
                / profiling.TRACE_NAME.format(rank=3)
            )
            written.parent.mkdir(parents=True)
            written.write_bytes(b"")
            self.assertEqual(
                written.relative_to(arm_dir).as_posix(),
                "profiling/traces/iteration_40/rank3_trace.json.gz",
            )
            self.assertEqual(layout.trace_files(arm_dir), [written])
            self.assertEqual(layout.trace_files_by_rank(arm_dir), {3: [written]})
            self.assertEqual(layout.rank_of_trace(written), 3)


class StepFloorTests(unittest.TestCase):
    """40 steps hold two profiler windows, and nothing else needs them."""

    def test_a_profiled_run_keeps_the_forty_step_floor(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least 40"):
            workload_with_overrides(
                ENGINES, steps=12, profile=True, warmup_steps=None
            )

    def test_an_unprofiled_run_accepts_a_short_run(self) -> None:
        workload = workload_with_overrides(
            ENGINES, steps=12, profile=False, warmup_steps=2
        )
        self.assertEqual(workload.steps, 12)


class TitanArgvTests(unittest.TestCase):
    def _command(self, profile: bool) -> list[str]:
        return command_for_arm(
            ENGINES.workload,
            ENGINES.arm("titan_eager"),
            Path("/tmp/arm"),
            (),
            "none",
            profile=profile,
        )

    def test_the_profiler_flags_appear_only_under_profile(self) -> None:
        profiled = self._command(True)
        for flag in _TITAN_PROFILER_FLAGS:
            self.assertIn(flag, profiled)
        plain = self._command(False)
        for flag in _TITAN_PROFILER_FLAGS:
            self.assertNotIn(flag, plain)

    def test_nothing_else_moves(self) -> None:
        """The block leaves, and the rest of the argv is what it was."""
        profiled = [
            token
            for token in self._command(True)
            if token not in _TITAN_PROFILER_FLAGS
        ]
        workload = ENGINES.workload
        values = {
            str(workload.profile_freq),
            str(workload.profiler_active),
            str(workload.profiler_warmup),
        }
        self.assertEqual(
            [token for token in profiled if token not in values],
            self._command(False),
        )


class StockArgvTests(unittest.TestCase):
    def _flags(self, profile: bool) -> list[str]:
        return stock_megatron_flags(
            shape_by_name("1b"),
            ENGINES.workload,
            TRIVIAL_SPEC,
            arm_dir="/tmp/arm",
            model_size="1b",
            profile=profile,
        )

    def test_megatron_and_harness_profiler_flags_appear_only_under_profile(
        self,
    ) -> None:
        profiled = self._flags(True)
        for flag in _MEGATRON_PROFILER_FLAGS + BENCH_PROFILE_SCHEDULE_FLAGS:
            self.assertIn(flag, profiled)
        self.assertIn(BENCH_PROFILE, profiled)

        plain = self._flags(False)
        for flag in _MEGATRON_PROFILER_FLAGS + BENCH_PROFILE_SCHEDULE_FLAGS:
            self.assertNotIn(flag, plain)
        self.assertNotIn(BENCH_PROFILE, plain)

    def test_an_unprofiled_run_accepts_a_partial_profiler_cycle(self) -> None:
        """The refusal guards a window, and there is no window."""
        short = replace(ENGINES.workload, steps=12)
        with self.assertRaisesRegex(ValueError, "whole number of profiler"):
            stock_megatron_flags(
                shape_by_name("1b"),
                short,
                TRIVIAL_SPEC,
                arm_dir="/tmp/arm",
                model_size="1b",
                profile=True,
            )
        flags = stock_megatron_flags(
            shape_by_name("1b"),
            short,
            TRIVIAL_SPEC,
            arm_dir="/tmp/arm",
            model_size="1b",
            profile=False,
        )
        self.assertIn("--train-iters", flags)


class ValidationSkipsTheTraceRulesTests(unittest.TestCase):
    """Rules 5, 6 and 13 read a trace, and an unprofiled arm wrote none."""

    def _log(self, root: Path) -> Path:
        log = root / "titan_eager.log"
        log.write_text(_SAC_LINE + _SIZE_LINE + "Training completed\n")
        return log

    def test_a_traceless_arm_passes_without_profile_and_fails_with_it(
        self,
    ) -> None:
        arm = ENGINES.arm("titan_eager")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._log(root)
            validate_arm(arm, root, log, ENGINES.workload, profile=False)
            with self.assertRaisesRegex(RuntimeError, "profiler windows"):
                validate_arm(arm, root, log, ENGINES.workload, profile=True)

    def test_the_log_rules_still_run(self) -> None:
        """Rule 8 inverts on an eager arm, with or without a trace."""
        arm = ENGINES.arm("titan_eager")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = root / "titan_eager.log"
            log.write_text(
                _compiled_line("default")
                + _SAC_LINE
                + _SIZE_LINE
                + "Training completed\n"
            )
            with self.assertRaisesRegex(RuntimeError, "compiled the model"):
                validate_arm(arm, root, log, ENGINES.workload, profile=False)


class ManifestTests(unittest.TestCase):
    def _manifest(self, profile: bool) -> dict:
        scenario = scenario_by_name("engines")
        return manifest_data(
            scenario,
            (scenario.arm("titan_eager"),),
            {"titan_eager": ["cmd"]},
            "test-gpu",
            _METADATA,
            (),
            megatron_args=(),
            axes=RunAxes(
                ac_mode="none",
                model_size="1b",
                parallelism=TRIVIAL_SPEC,
                megatron_p2p_sync="off",
                megatron_nan_guard="off",
                megatron_precision="stock",
                profile=profile,
                warmup_steps=None if profile else 10,
            ),
        )

    def test_the_manifest_records_the_axis_under_schema_eighteen(
        self,
    ) -> None:
        recorded = self._manifest(True)
        self.assertEqual(recorded["schema_version"], MANIFEST_SCHEMA_VERSION)
        self.assertEqual(MANIFEST_SCHEMA_VERSION, 18)
        self.assertIs(recorded["profile"], True)
        self.assertIs(self._manifest(False)["profile"], False)
        # It survives the round trip a resume reads it back through.
        self.assertIs(json.loads(json.dumps(recorded))["profile"], True)

    def test_a_resume_refuses_a_mismatch_and_accepts_a_match(self) -> None:
        scenario = scenario_by_name("engines")
        arms = (scenario.arm("titan_eager"),)
        for recorded in (True, False):
            manifest = self._manifest(recorded)
            for requested in (True, False):
                mismatches = _resume_mismatches(
                    manifest,
                    scenario,
                    arms,
                    "test-gpu",
                    _METADATA,
                    (),
                    megatron_args=(),
                    axes=RunAxes(
                        ac_mode="none",
                        model_size="1b",
                        parallelism=TRIVIAL_SPEC,
                        megatron_p2p_sync="off",
                        megatron_nan_guard="off",
                        megatron_precision="stock",
                        profile=requested,
                        warmup_steps=None if requested else 10,
                    ),
                )
                if recorded == requested:
                    self.assertEqual(mismatches, [])
                else:
                    self.assertIn("profile", mismatches)


class TracelessEvaluationTests(unittest.TestCase):
    """A run without traces still publishes its throughput."""

    def _out_dir(self, root: Path) -> Path:
        manifest = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "profile": False,
            "warmup_steps": 3,
            "scenario": "engines",
            "hardware": "test-gpu",
            "workload": {
                "profile_freq": 20,
                "profiler_warmup": 5,
                "profiler_active": 5,
                "local_batch_size": 4,
                "seq_len": 1024,
            },
            "selected_arms": ["titan_eager"],
            "arms": [{"name": "titan_eager", "engine": "torchtitan"}],
            "parallelism": {"world_size": 1, "dp": 1, "pp": 1, "ep": 1},
        }
        (root / "manifest.json").write_text(json.dumps(manifest))
        (root / "titan_eager.log").write_text(
            "".join(
                f"step: {step} loss: 1.0 grad_norm: 2.0 memory: 3.00GiB "
                "tps: 1000\n"
                for step in range(2, 6)
            )
        )
        return root

    def test_the_evaluation_succeeds_and_publishes_no_kernel_time(
        self,
    ) -> None:
        from benchmarks.e2e.results import evaluate_run, render_evaluation

        with tempfile.TemporaryDirectory() as temporary:
            result = evaluate_run(self._out_dir(Path(temporary)))
        self.assertEqual(
            result.results["titan_eager"].stable_tokens_per_second, 1000
        )
        # Schema 6 publishes no kernel time in either profile mode.
        self.assertNotIn("gpu_time", result.to_dict())
        self.assertNotIn("gpu kernel time", render_evaluation(result))


class DefaultTests(unittest.TestCase):
    def test_the_axis_defaults_off(self) -> None:
        self.assertIs(DEFAULT_PROFILE, False)


if __name__ == "__main__":
    unittest.main()
