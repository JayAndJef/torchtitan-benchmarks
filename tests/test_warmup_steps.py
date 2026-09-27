"""``--warmup-steps``: the sample rule of an unprofiled run.

A profiled run samples the steps of each 20-step cycle the profiler is
idle on. A run without ``--profile`` has no profiler to sample around, so
it discards a warmup and measures every step after it. The two rules
produce two figures, and the axis is refused beside ``--profile`` rather
than ignored there.
"""

import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from click.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.manifests import (
    MANIFEST_SCHEMA_VERSION,
    load_manifest,
    resume_mismatches,
)
from benchmarks.cli.main import cli
from benchmarks.e2e.checks import step_floor_refusals
from benchmarks.e2e.parallelism import TRIVIAL_SPEC
from benchmarks.e2e.engines.api import RunSpec
from benchmarks.e2e.registry import (
    DEFAULT_WARMUP_STEPS,
    ENGINES,
    SEED,
    scenario_by_name,
)
from benchmarks.e2e.engines.api import StepSample
from benchmarks.e2e.results import evaluate_run, measured_samples, stable_samples
from tests.engine_helpers import titan_step_line, write_run_manifest
from benchmarks.models.piper_qwen3.shape import PIPER_1B


def _run(profile: bool, warmup_steps: int | None) -> RunSpec:
    """The run of these tests: the scenario data, one GPU, the 1b shape."""
    return RunSpec(
        shape=PIPER_1B,
        data=ENGINES.data,
        parallelism=TRIVIAL_SPEC,
        ac_mode="none",
        profile=profile,
        window=ENGINES.window,
        warmup_steps=warmup_steps,
        seed=SEED,
    )


def _samples(count: int) -> list[StepSample]:
    """One sample per step, numbered from 1, at ``100 * step`` tokens/s."""
    return [
        StepSample(
            rank=0,
            step=step,
            tokens_per_second=100 * step,
            peak_memory_gib=3.0,
            loss=1.0,
            grad_norm=2.0,
        )
        for step in range(1, count + 1)
    ]


def _tps(samples: list[StepSample]) -> list[float]:
    return [sample.tokens_per_second for sample in samples]


class DefaultTests(unittest.TestCase):
    def test_the_axis_defaults_to_ten(self) -> None:
        self.assertEqual(DEFAULT_WARMUP_STEPS, 10)


class MeasuredSamplesTests(unittest.TestCase):
    def test_every_step_after_the_warmup_is_a_sample(self) -> None:
        self.assertEqual(
            _tps(measured_samples(_samples(6), 2)), [300, 400, 500, 600]
        )

    def test_a_zero_warmup_measures_every_step(self) -> None:
        self.assertEqual(_tps(measured_samples(_samples(3), 0)), [100, 200, 300])

    def test_a_warmup_that_covers_the_run_measures_nothing(self) -> None:
        self.assertEqual(measured_samples(_samples(3), 3), [])

    def test_the_count_grows_with_the_run_where_the_profiled_rule_does_not(
        self,
    ) -> None:
        """The two rules are two figures, not one figure read two ways."""
        samples = _samples(40)
        self.assertEqual(len(measured_samples(samples, 10)), 30)
        self.assertEqual(len(stable_samples(samples, ENGINES.window)), 18)


class StepFloorTests(unittest.TestCase):
    def test_a_run_must_take_a_step_after_its_warmup(self) -> None:
        run = _run(False, 12)
        (refusal,) = step_floor_refusals(
            replace(run, data=replace(run.data, steps=12))
        )
        self.assertIn("must be more than the 12", refusal)

    def test_one_measured_step_is_enough(self) -> None:
        run = _run(False, 12)
        self.assertEqual(
            step_floor_refusals(
                replace(run, data=replace(run.data, steps=13))
            ),
            [],
        )


def _manifest(root: Path, profile: bool, warmup_steps: int | None) -> dict:
    scenario = scenario_by_name("engines")
    write_run_manifest(root, _run(profile, warmup_steps), (scenario.arm("titan_eager"),))
    return load_manifest(root)


class ManifestTests(unittest.TestCase):
    def test_the_manifest_records_the_count_under_schema_nineteen(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorded = _manifest(Path(temporary), False, 10)
        self.assertEqual(recorded["schema_version"], MANIFEST_SCHEMA_VERSION)
        self.assertEqual(MANIFEST_SCHEMA_VERSION, 19)
        self.assertEqual(recorded["run"]["warmup_steps"], 10)

    def test_a_profiled_run_records_null(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            recorded = _manifest(Path(temporary), True, None)
        self.assertIsNone(recorded["run"]["warmup_steps"])


class ResumeTests(unittest.TestCase):
    def test_a_resume_refuses_another_count(self) -> None:
        scenario = scenario_by_name("engines")
        arms = (scenario.arm("titan_eager"),)
        with tempfile.TemporaryDirectory() as temporary:
            manifest = _manifest(Path(temporary), False, 10)
        for requested, expected in ((10, []), (4, ["run.warmup_steps"])):
            self.assertEqual(
                resume_mismatches(manifest, run=_run(False, requested), arms=arms),
                expected,
            )


class CliRefusalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.runner = CliRunner()
        self.completed = SimpleNamespace(
            out_dir=Path("/tmp/output"),
            selected_arms=(ENGINES.arm("titan_eager"),),
        )

    def _run(self, *arguments: str):
        with mock.patch(
            "benchmarks.cli.e2e.execute_run", return_value=self.completed
        ) as execute, mock.patch("benchmarks.cli.e2e._evaluate"), mock.patch(
            "benchmarks.cli.e2e.record_evaluation_status"
        ):
            result = self.runner.invoke(cli, ["run", "2", *arguments])
        return result, execute

    def test_the_two_options_are_refused_together(self) -> None:
        result, _ = self._run("--profile", "--warmup-steps", "2")
        self.assertNotEqual(result.exit_code, 0)
        self.assertIn("applies only without it", result.output)

    def test_each_option_alone_is_accepted(self) -> None:
        result, execute = self._run("--profile")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIs(execute.call_args.args[0].axes.profile, True)
        self.assertIsNone(execute.call_args.args[0].axes.warmup_steps)

        result, execute = self._run("--warmup-steps", "2")
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(execute.call_args.args[0].axes.warmup_steps, 2)

    def test_an_unrequested_count_reaches_the_request_as_none(self) -> None:
        result, execute = self._run()
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertIsNone(execute.call_args.args[0].axes.warmup_steps)


class EvaluationPicksTheRuleTests(unittest.TestCase):
    """The manifest decides which rule the evaluation reads a run with."""

    def _build(
        self, root: Path, *, profile: bool, warmup_steps: int | None
    ) -> Path:
        run = _run(profile, warmup_steps)
        write_run_manifest(
            root,
            replace(run, data=replace(run.data, seq_len=1024)),
            (ENGINES.arm("titan_eager"),),
        )
        # Step 1 is fast, every later step is slow. The profiled rule drops
        # step 1 and keeps steps 2..10; a warmup of 1 keeps steps 2..12.
        (root / "titan_eager.log").write_text(
            "".join(
                titan_step_line(step, tps=9000 if step == 1 else 1000)
                for step in range(1, 13)
            )
        )
        return root

    def test_an_unprofiled_run_reads_the_measured_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._build(
                Path(temporary), profile=False, warmup_steps=1
            )
            summary = evaluate_run(root).results["titan_eager"]
        # Steps 2..12: eleven samples, where the profiled rule takes nine.
        self.assertEqual(summary.stable_sample_count, 11)
        self.assertEqual(summary.stable_tokens_per_second, 1000)

    def test_a_profiled_run_still_reads_the_profiler_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = self._build(Path(temporary), profile=True, warmup_steps=None)
            summary = evaluate_run(root).results["titan_eager"]
        # Steps 2..10 of the one cycle this log holds.
        self.assertEqual(summary.stable_sample_count, 9)


if __name__ == "__main__":
    unittest.main()
