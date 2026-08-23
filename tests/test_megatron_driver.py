"""CPU tests for the Megatron driver's pipeline handling.

The driver itself needs two GPUs to run. What can be checked without one is
the arithmetic that decides what it runs: how a batch splits into
microbatches, which pipeline requests it refuses, and that every branch it
grew takes the single-rank value at ``--pp 1``.

That last property is the one this file exists for. Every megatron number
under ``out/`` was measured before the pipeline axis existed, and a driver
that repacked its batch, added a collective or renamed its trace file at
world size 1 would make none of them reproducible.
"""

import inspect
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.megatron import train


def _args(**overrides):
    argv = [
        "--seq-len",
        str(overrides.pop("seq_len", 1024)),
        "--steps",
        str(overrides.pop("steps", 40)),
        "--batch",
        str(overrides.pop("batch", 4)),
        "--seed",
        "42",
        "--profile-freq",
        "20",
        "--profiler-warmup",
        "5",
        "--profiler-active",
        "5",
        "--mode",
        "default",
    ]
    for name, value in overrides.items():
        argv.extend((f"--{name.replace('_', '-')}", str(value)))
    argv.append("/tmp/arm-dir")
    return train.parse_args(argv)


class DriverDefaultTests(unittest.TestCase):
    def test_the_three_pipeline_options_default_to_one_rank(self) -> None:
        """An argv built before this axis existed describes the same run."""
        args = _args()
        self.assertEqual(args.pp, 1)
        self.assertIsNone(args.pp_schedule)
        self.assertEqual(args.pp_microbatch_size, 1)


class PipelineSettingsTests(unittest.TestCase):
    def test_one_rank_keeps_the_whole_batch_in_one_pack(self) -> None:
        """The historical treatment: one THD pack, one microbatch.

        Neither engine microbatches without a pipeline, and repacking the
        batch would change the document layout every published megatron
        number was measured on.
        """
        self.assertEqual(train.pipeline_settings(_args()), (4, 1))

    def test_a_pipeline_splits_the_batch_into_microbatches(self) -> None:
        args = _args(pp=2, pp_schedule="1F1B", pp_microbatch_size=1)
        self.assertEqual(train.pipeline_settings(args), (1, 4))

    def test_a_wider_microbatch_gives_fewer_of_them(self) -> None:
        args = _args(pp=2, pp_schedule="1F1B", pp_microbatch_size=2)
        self.assertEqual(train.pipeline_settings(args), (2, 2))

    def test_a_batch_that_does_not_divide_is_refused(self) -> None:
        args = _args(batch=5, pp=2, pp_schedule="1F1B", pp_microbatch_size=2)
        with self.assertRaisesRegex(ValueError, "does not divide"):
            train.pipeline_settings(args)


class PipelineRefusalTests(unittest.TestCase):
    """What the driver declines, and why each refusal is a wrong number.

    ``benchmarks/e2e/parallelism.py`` refuses most of these for a run. The
    driver restates them because ``python -m benchmarks.e2e.megatron.train``
    holds no spec.
    """

    def test_one_rank_at_pp_one_is_accepted(self) -> None:
        train.refuse_unsupported_pipeline(_args(), 1)

    def test_two_ranks_at_1f1b_are_accepted(self) -> None:
        train.refuse_unsupported_pipeline(
            _args(pp=2, pp_schedule="1F1B"), 2
        )

    def test_more_ranks_than_stages_are_refused(self) -> None:
        """There is no data-parallel path here, so the extra rank is silent.

        Two ranks with one stage each would read the same tokens and never
        reduce their gradients, and every other check would pass.
        """
        with self.assertRaisesRegex(ValueError, "no data-parallel path"):
            train.refuse_unsupported_pipeline(_args(), 2)

    def test_fewer_ranks_than_stages_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not equal"):
            train.refuse_unsupported_pipeline(
                _args(pp=2, pp_schedule="1F1B"), 1
            )

    def test_a_schedule_at_pp_one_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "no pipeline to schedule"):
            train.refuse_unsupported_pipeline(_args(pp_schedule="1F1B"), 1)

    def test_a_microbatch_size_at_pp_one_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "not split"):
            train.refuse_unsupported_pipeline(_args(pp_microbatch_size=2), 1)

    def test_an_interleaved_schedule_is_refused_by_name(self) -> None:
        """Megatron-LM implements it; this driver does not.

        The registry declares Interleaved1F1B as megatron-supported on
        purpose, because that field records what the library implements. The
        failure belongs here, on the module that owns the missing model-chunk
        list.
        """
        with self.assertRaisesRegex(ValueError, "not implemented by this"):
            train.refuse_unsupported_pipeline(
                _args(pp=2, pp_schedule="Interleaved1F1B"), 2
            )

    def test_an_unregistered_schedule_is_refused_too(self) -> None:
        with self.assertRaisesRegex(ValueError, "not implemented by this"):
            train.refuse_unsupported_pipeline(
                _args(pp=2, pp_schedule="GPipe"), 2
            )


class SingleRankInertnessTests(unittest.TestCase):
    """Read off the source: every new collective is guarded on the world size.

    A collective at world size 1 is a synchronize inside the timed step, and
    the step time is the published number.
    """

    def _main_source(self) -> str:
        return inspect.getsource(train.main)

    def test_the_loss_broadcast_returns_early_at_one_rank(self) -> None:
        source = self._main_source()
        self.assertIn("if world_size == 1:\n            return loss", source)

    def test_the_norm_reduction_is_guarded(self) -> None:
        source = self._main_source()
        head = source.index("def clip_gradients")
        body = source[head : source.index("def trace_handler")]
        self.assertIn("if world_size > 1:", body)
        self.assertIn("ReduceOp.SUM", body)

    def test_the_trace_file_carries_the_rank(self) -> None:
        self.assertIn('f"rank{rank}_trace.json.gz"', self._main_source())

    def test_the_rank_variables_fall_back_to_the_single_rank_values(
        self,
    ) -> None:
        source = self._main_source()
        for line in (
            'rank = int(os.environ.get("RANK", "0"))',
            'world_size = int(os.environ.get("WORLD_SIZE", "1"))',
            'local_rank = int(os.environ.get("LOCAL_RANK", "0"))',
        ):
            self.assertIn(line, source)


if __name__ == "__main__":
    unittest.main()
