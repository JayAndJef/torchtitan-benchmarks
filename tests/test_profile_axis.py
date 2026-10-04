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
from dataclasses import dataclass, replace
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts import layout
from benchmarks.artifacts.manifests import (
    MANIFEST_SCHEMA_VERSION,
    load_manifest,
    resume_mismatches,
)
from benchmarks.e2e.engines.megatron_stock.driver import profiling
from benchmarks.e2e.engines.megatron_stock.flags import (
    BENCH_PROFILE,
    BENCH_PROFILE_SCHEDULE_FLAGS,
    stock_megatron_flags,
)
from benchmarks.e2e.checks import check_run, step_floor_refusals
from benchmarks.e2e.engines.api import Arm, Engine, EngineConfig
from benchmarks.e2e.engines.registry import ENGINES as ENGINE_REGISTRY
from benchmarks.e2e.engines.megatron_stock.config import MegatronStockConfig
from benchmarks.e2e.engines.megatron_stock.profiling import partial_cycle_refusal
from benchmarks.e2e.registry import DEFAULT_PROFILE, ENGINES, scenario_by_name
from tests.engine_helpers import (
    command,
    registered,
    run_spec,
    titan_step_line,
    validate,
    write_run_manifest,
)
from tests.test_runner import _SAC_LINE, _SIZE_LINE, _compiled_line


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
        refusals = step_floor_refusals(run_spec(profile=True, steps=12))
        self.assertEqual(len(refusals), 1)
        self.assertIn("at least 40", refusals[0])

    def test_an_unprofiled_run_accepts_a_short_run(self) -> None:
        self.assertEqual(
            step_floor_refusals(run_spec(profile=False, warmup_steps=2, steps=12)),
            [],
        )


@dataclass(frozen=True, kw_only=True)
class _TracelessConfig(EngineConfig):
    """The config of an engine that writes no profiler trace."""


def _traceless_engine() -> Engine:
    """An engine that passes its own check and writes no profiler trace."""
    engine = mock.Mock(spec=Engine)
    engine.name = "traceless"
    engine.config_type = _TracelessConfig
    engine.can_profile = False
    engine.check.return_value = []
    return engine


_TRACELESS_ARM = Arm(
    name="traceless_arm",
    description="an arm on an engine that writes no profiler trace",
    config=_TracelessConfig(),
)
"""An arm of the traceless engine."""


class ProfileSupportTests(unittest.TestCase):
    """The shared check refuses ``--profile`` for each arm whose engine writes no trace."""

    def _check(self, run, arms) -> None:
        with registered(_traceless_engine()):
            check_run(run, ENGINES, arms, device_count=1, resumed=None)

    def test_both_engines_write_profiler_traces(self) -> None:
        for engine in ENGINE_REGISTRY.values():
            with self.subTest(engine=engine.name):
                self.assertTrue(engine.can_profile)

    def test_a_profiled_run_refuses_the_arm_and_names_it_and_its_engine(
        self,
    ) -> None:
        with self.assertRaises(ValueError) as caught:
            self._check(
                run_spec(ac_mode="none", profile=True),
                (ENGINES.arm("titan_eager"), _TRACELESS_ARM),
            )
        self.assertEqual(
            str(caught.exception),
            "traceless_arm: the engine 'traceless' writes no profiler trace; "
            "drop --profile, or deselect the arm",
        )

    def test_the_refusal_is_listed_with_the_other_refusals(self) -> None:
        with self.assertRaises(ValueError) as caught:
            self._check(
                run_spec(ac_mode="none", profile=True, steps=12),
                (_TRACELESS_ARM,),
            )
        message = str(caught.exception)
        for part in ("at least 40", "traceless_arm: the engine 'traceless'"):
            with self.subTest(part=part):
                self.assertIn(part, message)

    def test_an_unprofiled_run_accepts_the_arm(self) -> None:
        self._check(run_spec(ac_mode="none", profile=False), (_TRACELESS_ARM,))


class TitanArgvTests(unittest.TestCase):
    def _command(self, profile: bool) -> list[str]:
        return command(
            run_spec(ac_mode="none", profile=profile),
            ENGINES.arm("titan_eager"),
            "/tmp/arm",
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
        window = ENGINES.window
        values = {str(window.freq), str(window.active), str(window.warmup)}
        self.assertEqual(
            [token for token in profiled if token not in values],
            self._command(False),
        )


class StockArgvTests(unittest.TestCase):
    def _flags(self, profile: bool) -> list[str]:
        return stock_megatron_flags(
            run_spec(profile=profile), MegatronStockConfig(), arm_dir="/tmp/arm"
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
        self.assertIsNotNone(
            partial_cycle_refusal("megatron_stock", run_spec(profile=True, steps=12))
        )
        self.assertIsNone(
            partial_cycle_refusal("megatron_stock", run_spec(profile=False, steps=12))
        )


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
            validate(run_spec(profile=False), arm, root, log)
            with self.assertRaisesRegex(RuntimeError, "profiler windows"):
                validate(run_spec(profile=True), arm, root, log)

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
                validate(run_spec(profile=False), arm, root, log)


class ManifestTests(unittest.TestCase):
    def _manifest(self, profile: bool) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            write_run_manifest(
                out_dir,
                run_spec(ac_mode="none", profile=profile),
                (ENGINES.arm("titan_eager"),),
            )
            return load_manifest(out_dir)

    def test_the_manifest_records_the_axis_in_the_run_block(self) -> None:
        recorded = self._manifest(True)
        self.assertEqual(recorded["schema_version"], MANIFEST_SCHEMA_VERSION)
        self.assertIs(recorded["run"]["profile"], True)
        self.assertIsNone(recorded["run"]["warmup_steps"])
        self.assertIs(self._manifest(False)["run"]["profile"], False)

    def test_a_resume_refuses_a_mismatch_and_accepts_a_match(self) -> None:
        arms = (ENGINES.arm("titan_eager"),)
        for recorded in (True, False):
            manifest = self._manifest(recorded)
            for requested in (True, False):
                mismatches = resume_mismatches(
                    manifest,
                    run=run_spec(ac_mode="none", profile=requested),
                    arms=arms,
                )
                if recorded == requested:
                    self.assertEqual(mismatches, [])
                else:
                    self.assertIn("run.profile", mismatches)


class TracelessEvaluationTests(unittest.TestCase):
    """A run without traces still publishes its throughput."""

    def _out_dir(self, root: Path) -> Path:
        write_run_manifest(
            root,
            run_spec(
                ac_mode="none",
                profile=False,
                warmup_steps=3,
                seq_len=1024,
                local_batch_size=4,
                steps=5,
            ),
            (ENGINES.arm("titan_eager"),),
        )
        (root / "titan_eager.log").write_text(
            "".join(titan_step_line(step) for step in range(1, 6))
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
        self.assertNotIn("gpu_time", result.to_dict())
        self.assertNotIn("gpu kernel time", render_evaluation(result))


class DefaultTests(unittest.TestCase):
    def test_the_axis_defaults_off(self) -> None:
        self.assertIs(DEFAULT_PROFILE, False)


if __name__ == "__main__":
    unittest.main()
