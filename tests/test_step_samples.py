"""Tests for each engine's step reader: the samples it reads, and the lines it refuses."""

import json
import math
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.layout import logs_by_rank
from benchmarks.e2e.engines.api import DroppedLine, StepRead, StepSample
from benchmarks.e2e.engines.megatron_stock import steps as megatron_steps
from benchmarks.e2e.engines.registry import engine_named
from benchmarks.e2e.engines.torchtitan import steps as titan_steps
from tests.engine_helpers import titan_step_line

GOLDEN_RUNS = Path(__file__).resolve().parent / "fixtures" / "golden" / "runs"

RECORD_FIELDS = dict(
    step=7,
    tokens_per_second=41234,
    peak_memory_gib=12.3456,
    loss=2.123456789,
    grad_norm=0.7512345,
    tflops=123.456,
    mfu=12.4567,
)


class TorchTitanStepLineTests(unittest.TestCase):
    def test_the_fork_line_reads_as_one_sample(self) -> None:
        (sample,) = titan_steps.read_steps(
            3, titan_step_line(5, tps=12345, loss=2.5, grad_norm=0.75, memory=41.5)
        ).samples
        self.assertEqual(
            sample,
            StepSample(
                rank=3,
                step=5,
                tokens_per_second=12345,
                peak_memory_gib=41.5,
                loss=2.5,
                grad_norm=0.75,
                extras={"tflops": 12.5, "mfu": 1.26},
            ),
        )

    def test_the_rank_prefix_of_a_one_rank_log_is_read(self) -> None:
        (sample,) = titan_steps.read_steps(0, "[rank0]:" + titan_step_line(1)).samples
        self.assertEqual(sample.step, 1)

    def test_the_no_loss_value_reads_as_none(self) -> None:
        (sample,) = titan_steps.read_steps(0, titan_step_line(1, loss=-1.0)).samples
        self.assertIsNone(sample.loss)

    def test_a_non_finite_value_reads_as_itself(self) -> None:
        (sample,) = titan_steps.read_steps(
            0, titan_step_line(1, loss=float("nan"), grad_norm=float("-inf"))
        ).samples
        self.assertTrue(math.isnan(sample.loss))
        self.assertEqual(sample.grad_norm, float("-inf"))

    def test_an_unknown_mfu_is_absent_from_the_extras(self) -> None:
        line = titan_step_line(1).replace("mfu: 1.26%", "mfu: N/A")
        (sample,) = titan_steps.read_steps(0, line).samples
        self.assertEqual(dict(sample.extras), {"tflops": 12.5})

    def test_a_whole_step_line_with_another_rank_line_appended_reads(self) -> None:
        torn = titan_step_line(2).rstrip("\n") + "[rank1]:" + titan_step_line(2)
        read = titan_steps.read_steps(0, torn)
        self.assertEqual([sample.step for sample in read.samples], [2])
        self.assertEqual(read.dropped, ())

    def test_a_step_line_that_a_rank_prefix_cut_is_dropped(self) -> None:
        cut = titan_step_line(2)[:-60] + "[rank3]:USDT: profiler_stop\n"
        read = titan_steps.read_steps(0, titan_step_line(1) + cut + titan_step_line(3))
        self.assertEqual([sample.step for sample in read.samples], [1, 3])
        self.assertEqual(read.dropped, (DroppedLine(rank=0, line=2, step=2),))

    def test_a_line_of_another_rank_is_not_this_rank_step(self) -> None:
        self.assertEqual(
            titan_steps.read_steps(0, "[rank1]:" + titan_step_line(5)),
            StepRead(samples=()),
        )
        record = megatron_steps.step_record(**RECORD_FIELDS)
        text_line = MegatronTextLineTests.LINE
        for line in (record, text_line):
            with self.subTest(line=line[:12]):
                self.assertEqual(
                    megatron_steps.read_steps(0, "[rank1]:" + line), StepRead(samples=())
                )

    def test_a_step_line_in_an_appended_line_is_not_this_rank_step(self) -> None:
        text = "[rank0]:INFO starting [rank1]:" + titan_step_line(5)
        self.assertEqual(titan_steps.read_steps(0, text), StepRead(samples=()))
        cut = "[rank0]:INFO starting [rank1]:step: 5  loss[rank2]:x\n"
        self.assertEqual(titan_steps.read_steps(0, cut), StepRead(samples=()))

    def test_a_cut_step_line_raises(self) -> None:
        cut = titan_step_line(2)[:-40] + "\n"
        with self.assertRaisesRegex(ValueError, "rank 4 logs a TorchTitan step line"):
            titan_steps.read_steps(4, cut)

    def test_other_lines_are_not_step_lines(self) -> None:
        text = (
            "[titan] 2026-09-26 10:00:00,000 - root - INFO - Training starts at step 1\n"
            "[titan] 2026-09-26 10:00:00,000 - root - INFO - validate step:  1  loss: 1.0\n"
        )
        self.assertEqual(titan_steps.read_steps(0, text), StepRead(samples=()))

    def test_a_step_line_with_a_damaged_header_reads(self) -> None:
        line = titan_step_line(3).replace("2026-09-26 10:00:00", "2026-0\x04\x00")
        (sample,) = titan_steps.read_steps(0, line).samples
        self.assertEqual(sample.step, 3)

    def test_a_step_line_without_its_fields_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not parse"):
            titan_steps.read_steps(
                0, "step: 1 loss: 1.0 grad_norm: 2.0 memory: 3.00GiB tps: 1000\n"
            )


class MegatronStepRecordTests(unittest.TestCase):
    def test_the_record_round_trips(self) -> None:
        line = megatron_steps.step_record(**RECORD_FIELDS)
        self.assertTrue(line.startswith(megatron_steps.STEP_PREFIX))
        (sample,) = megatron_steps.read_steps(2, line).samples
        self.assertEqual(
            sample,
            StepSample(
                rank=2,
                step=7,
                tokens_per_second=41234,
                peak_memory_gib=12.35,
                loss=2.12346,
                grad_norm=0.7512,
                extras={"tflops": 123.46, "mfu": 12.46},
            ),
        )

    def test_a_rank_without_a_loss_records_none(self) -> None:
        line = megatron_steps.step_record(**{**RECORD_FIELDS, "loss": None})
        (sample,) = megatron_steps.read_steps(0, "[rank0]:" + line).samples
        self.assertIsNone(sample.loss)

    def test_a_non_finite_norm_round_trips(self) -> None:
        line = megatron_steps.step_record(
            **{**RECORD_FIELDS, "grad_norm": float("nan"), "loss": float("inf")}
        )
        (sample,) = megatron_steps.read_steps(0, line).samples
        self.assertTrue(math.isnan(sample.grad_norm))
        self.assertEqual(sample.loss, float("inf"))

    def test_another_rank_line_after_a_record_is_dropped(self) -> None:
        line = megatron_steps.step_record(**RECORD_FIELDS)
        (sample,) = megatron_steps.read_steps(0, line + "[rank3]:" + line).samples
        self.assertEqual(sample.step, 7)

    def test_a_record_that_a_rank_prefix_cut_is_dropped(self) -> None:
        line = megatron_steps.step_record(**RECORD_FIELDS)
        read = megatron_steps.read_steps(
            1, line[:40] + "[rank0]:USDT: profiler_start\n" + line + "\n"
        )
        self.assertEqual([sample.step for sample in read.samples], [7])
        self.assertEqual(read.dropped, (DroppedLine(rank=1, line=1, step=7),))

    def test_a_record_cut_before_its_step_drops_with_no_step(self) -> None:
        read = megatron_steps.read_steps(0, megatron_steps.STEP_PREFIX + '{"st[rank2]:x')
        self.assertEqual(read.dropped, (DroppedLine(rank=0, line=1, step=None),))

    def test_a_malformed_record_raises(self) -> None:
        line = megatron_steps.step_record(**RECORD_FIELDS)
        record = json.loads(line[len(megatron_steps.STEP_PREFIX) :])
        cases = {
            "cut": (line[:-10], "does not parse"),
            "trailing": (line + " extra", "does not parse"),
            "missing key": (
                megatron_steps.STEP_PREFIX
                + json.dumps({k: v for k, v in record.items() if k != "loss"}),
                "keys",
            ),
            "text value": (
                megatron_steps.STEP_PREFIX
                + json.dumps({**record, "tokens_per_second": "fast"}),
                "not a number",
            ),
            "list": (megatron_steps.STEP_PREFIX + "[1, 2]", "keys"),
        }
        for name, (text, message) in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, message):
                    megatron_steps.read_steps(0, text)


class MegatronTextLineTests(unittest.TestCase):
    """The text step line that the stored run directories hold."""

    LINE = (
        "step:  3  loss: 11.59900  grad_norm:  3.5000  memory: 41.50GiB(29.68%)  "
        "tps: 9,708  tflops: 1,123.29  mfu: 12.43%"
    )

    def test_the_text_line_reads_as_one_sample(self) -> None:
        (sample,) = megatron_steps.read_steps(1, "[rank1]:" + self.LINE).samples
        self.assertEqual(
            sample,
            StepSample(
                rank=1,
                step=3,
                tokens_per_second=9708,
                peak_memory_gib=41.5,
                loss=11.599,
                grad_norm=3.5,
                extras={"tflops": 1123.29, "mfu": 12.43},
            ),
        )

    def test_a_line_without_a_loss_reads_as_none(self) -> None:
        line = self.LINE.replace("loss: 11.59900  ", "")
        (sample,) = megatron_steps.read_steps(0, line).samples
        self.assertIsNone(sample.loss)

    def test_a_text_line_that_a_rank_prefix_cut_is_dropped(self) -> None:
        read = megatron_steps.read_steps(
            3, self.LINE + "\n" + self.LINE[:-40] + "[rank1]:USDT: profiler_start\n"
        )
        self.assertEqual(len(read.samples), 1)
        self.assertEqual(read.dropped, (DroppedLine(rank=3, line=2, step=3),))

    def test_a_cut_text_line_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Megatron step line"):
            megatron_steps.read_steps(0, self.LINE[:-20])


BASELINE_STEP_LINE = re.compile(
    r"step:\s*(\d+).*?memory:\s*([0-9.]+)GiB.*?tps:\s*([0-9,]+)"
)
"""The step-line pattern of the baseline evaluation, before the engine readers."""


class GoldenStepLineTests(unittest.TestCase):
    """Each rank of the stored run directories reads the step lines that the baseline pattern read."""

    def test_each_golden_rank_reads_every_step_line(self) -> None:
        engines = {
            "megatron_stock": engine_named("megatron_stock"),
            "titan_compiled": engine_named("torchtitan"),
            "titan_eager": engine_named("torchtitan"),
        }
        logs = sorted(GOLDEN_RUNS.glob("*/*.log"))
        self.assertEqual(len(logs), 10)
        for log in logs:
            by_rank = logs_by_rank(log.read_text(errors="replace"))
            for rank, text in by_rank.items():
                with self.subTest(log=f"{log.parent.name}/{log.name}", rank=rank):
                    read = engines[log.stem].read_steps(rank, text)
                    baseline = [
                        (int(match.group(1)), int(match.group(3).replace(",", "")))
                        for line in text.splitlines()
                        if (match := BASELINE_STEP_LINE.search(line))
                    ]
                    self.assertEqual(read.dropped, ())
                    self.assertEqual(
                        [(sample.step, sample.tokens_per_second) for sample in read.samples],
                        baseline,
                    )


if __name__ == "__main__":
    unittest.main()
