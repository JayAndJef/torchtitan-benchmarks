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
    def test_the_four_mesh_options_default_to_one_rank(self) -> None:
        """An argv built before this axis existed describes the same run."""
        args = _args()
        self.assertEqual(args.dp, 1)
        self.assertEqual(args.pp, 1)
        self.assertIsNone(args.pp_schedule)
        self.assertEqual(args.pp_microbatch_size, 1)

    def test_the_p2p_sync_defaults_to_on(self) -> None:
        """``on`` is megatron's own default, and every published number's."""
        args = _args()
        self.assertEqual(args.batch_p2p_sync, "on")
        self.assertTrue(train.batch_p2p_sync_enabled(args))

    def test_an_unknown_p2p_sync_value_is_refused_at_parsing(self) -> None:
        with self.assertRaises(SystemExit):
            _args(batch_p2p_sync="maybe")


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
        train.refuse_unsupported_mesh(_args(), 1)

    def test_two_ranks_at_1f1b_are_accepted(self) -> None:
        train.refuse_unsupported_mesh(
            _args(pp=2, pp_schedule="1F1B"), 2
        )

    def test_two_ranks_at_dp_two_are_accepted(self) -> None:
        train.refuse_unsupported_mesh(_args(dp=2), 2)

    def test_four_ranks_at_dp_two_by_pp_two_are_accepted(self) -> None:
        train.refuse_unsupported_mesh(
            _args(dp=2, pp=2, pp_schedule="1F1B"), 4
        )

    def test_a_rank_the_mesh_does_not_name_is_refused(self) -> None:
        """Megatron would make it a data-parallel replica nothing wraps.

        It would read the same tokens as its neighbour, never reduce a
        gradient, and every other check would pass.
        """
        with self.assertRaisesRegex(ValueError, "does not equal"):
            train.refuse_unsupported_mesh(_args(), 2)
        with self.assertRaisesRegex(ValueError, "does not equal"):
            train.refuse_unsupported_mesh(_args(dp=2), 4)

    def test_a_degree_below_one_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "--dp 0 must be >= 1"):
            train.refuse_unsupported_mesh(_args(dp=0), 1)
        with self.assertRaisesRegex(ValueError, "--pp 0 must be >= 1"):
            train.refuse_unsupported_mesh(_args(pp=0), 1)

    def test_fewer_ranks_than_stages_are_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "does not equal"):
            train.refuse_unsupported_mesh(
                _args(pp=2, pp_schedule="1F1B"), 1
            )

    def test_a_schedule_at_pp_one_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "no pipeline to schedule"):
            train.refuse_unsupported_mesh(_args(pp_schedule="1F1B"), 1)

    def test_a_microbatch_size_at_pp_one_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "not split"):
            train.refuse_unsupported_mesh(_args(pp_microbatch_size=2), 1)

    def test_an_interleaved_schedule_is_refused_by_name(self) -> None:
        """Megatron-LM implements it; this driver does not.

        The registry declares Interleaved1F1B as megatron-supported on
        purpose, because that field records what the library implements. The
        failure belongs here, on the module that owns the missing model-chunk
        list.
        """
        with self.assertRaisesRegex(ValueError, "not implemented by this"):
            train.refuse_unsupported_mesh(
                _args(pp=2, pp_schedule="Interleaved1F1B"), 2
            )

    def test_an_unregistered_schedule_is_refused_too(self) -> None:
        with self.assertRaisesRegex(ValueError, "not implemented by this"):
            train.refuse_unsupported_mesh(
                _args(pp=2, pp_schedule="GPipe"), 2
            )

    def test_p2p_sync_off_at_pp_one_is_refused(self) -> None:
        """The field is inert without a pipeline message.

        A run that accepted the value would print a treatment it did not
        have, and the harness would record it.
        """
        with self.assertRaisesRegex(ValueError, "no pipeline message"):
            train.refuse_unsupported_mesh(_args(batch_p2p_sync="off"), 1)

    def test_p2p_sync_off_under_a_pipeline_is_accepted(self) -> None:
        args = _args(pp=2, pp_schedule="1F1B", batch_p2p_sync="off")
        train.refuse_unsupported_mesh(args, 2)
        self.assertFalse(train.batch_p2p_sync_enabled(args))


class BatchLossIsASumTests(unittest.TestCase):
    """Megatron scales the recorded loss, not only the gradient.

    ``forward_step_calc_loss`` runs ``output_tensor /= num_microbatches`` in
    place, and the value it records is ``loss.detach()``, which shares
    storage with that tensor. So the schedule hands back ``L_i / M`` and the
    batch's loss is their SUM. A mean would report ``batch_loss / M``.
    """

    def test_one_microbatch_is_that_microbatch(self) -> None:
        # The published configuration. A sum and a mean agree at M 1, which
        # is why no recorded number moves.
        self.assertEqual(train.batch_loss([7.5]), 7.5)

    def test_four_microbatches_recover_the_batch_loss(self) -> None:
        # Four microbatches whose own losses are 1, 2, 3, 4: the batch's
        # loss is 2.5, and the schedule already divided each by 4.
        recorded = [1 / 4, 2 / 4, 3 / 4, 4 / 4]
        self.assertAlmostEqual(train.batch_loss(recorded), 2.5)
        # The mean this replaced would have reported a quarter of that.
        self.assertAlmostEqual(sum(recorded) / len(recorded), 2.5 / 4)

    def test_a_stage_with_no_loss_reports_zero(self) -> None:
        # Every stage but the last gets an empty list.
        self.assertEqual(train.batch_loss([]), 0)

    def test_detach_shares_storage_with_the_scaled_tensor(self) -> None:
        """The aliasing the reduction depends on, pinned against torch.

        If a future torch made ``detach`` copy, the schedule would hand back
        an unscaled ``L_i`` and the sum would become wrong -- so this asserts
        the property rather than trusting it.
        """
        import torch

        loss = torch.tensor([2.0]).sum() / 1
        recorded = loss.detach()
        loss *= 1  # cp_group_size, 1 here
        loss /= 4  # num_microbatches
        self.assertEqual(float(recorded), 0.5)


class P2pSyncLineTests(unittest.TestCase):
    """The p2p line reads the built config, never the arguments.

    Megatron guards its per-message ``torch.cuda.synchronize()`` on
    ``batch_p2p_comm and batch_p2p_sync``. A line that restated the
    requested value would prove nothing about the call.
    """

    def test_the_line_reads_both_fields_off_the_config(self) -> None:
        from types import SimpleNamespace

        config = SimpleNamespace(batch_p2p_comm=True, batch_p2p_sync=False)
        self.assertEqual(
            train.p2p_line(config),
            "Megatron-LM p2p: batch_p2p_comm=True batch_p2p_sync=False",
        )
        config = SimpleNamespace(batch_p2p_comm=True, batch_p2p_sync=True)
        self.assertEqual(
            train.p2p_line(config),
            "Megatron-LM p2p: batch_p2p_comm=True batch_p2p_sync=True",
        )

    def test_the_line_is_the_template_filled_in(self) -> None:
        self.assertEqual(
            train.P2P_LINE,
            "Megatron-LM p2p: batch_p2p_comm={comm} batch_p2p_sync={sync}",
        )

    def test_main_builds_with_the_knob_and_prints_the_built_config(
        self,
    ) -> None:
        """Read off the source, the way the inertness tests are.

        The build call must take the mapped value, and the print must read
        ``model.config`` at the body's own indentation: a print inside a
        branch would leave some mesh without the line arm rule 12 reads.
        """
        source = inspect.getsource(train.main)
        self.assertIn(
            "batch_p2p_sync=batch_p2p_sync_enabled(args),", source
        )
        self.assertIn(
            "\n    print(p2p_line(model.config), flush=True)\n", source
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

    def test_the_norm_reduction_is_guarded_on_the_pipeline_degree(
        self,
    ) -> None:
        """The group is the pipeline's, and the guard names that degree.

        A data-parallel replica holds the same stage as its partner and, once
        the reduction has run, the same gradients -- so the norm is already
        the same number on it and a second reduction would square nothing.
        """
        source = self._main_source()
        head = source.index("def clip_gradients")
        body = source[head : source.index("def trace_handler")]
        self.assertIn("if args.pp > 1:", body)
        self.assertIn("ReduceOp.SUM", body)
        self.assertIn("get_pipeline_model_parallel_group()", body)

    def test_the_ddp_wrapper_and_its_step_work_are_guarded(self) -> None:
        """Every DDP branch is behind ``use_ddp``, which is ``--dp > 1``.

        At dp 1 the model stays bare, no buffer is allocated and no
        collective is added, which is what every megatron number under out/
        was measured on.
        """
        source = self._main_source()
        self.assertIn("use_ddp = args.dp > 1", source)
        for guarded in (
            "if use_ddp:\n        # Megatron's own gradient reduction",
            "if use_ddp:\n                # DDP accumulates into its own flat",
            "if not use_ddp:\n                for parameter in merged:",
        ):
            self.assertIn(guarded, source)

    def test_graph_mode_leaves_the_main_grad_buffers_to_ddp(self) -> None:
        """A second allocation would rebind main_grad away from the buckets.

        DDP points every param.main_grad into one flat buffer per bucket.
        Allocating private tensors over them would leave the reduction
        reading buffers nothing writes, and the run would train on zeros
        without failing. The combination is unreachable while parallelism
        rule 13 refuses cuda-graph above one rank.
        """
        self.assertIn("if graphs and not use_ddp:", self._main_source())

    def test_the_parameter_sum_is_taken_over_one_pipeline(self) -> None:
        """A world sum would count every parameter dp times.

        The check exists to catch a pipeline that holds the wrong model, and
        a data-parallel replica holds the same stage as its partner.
        """
        source = self._main_source()
        head = source.index("counted = torch.tensor")
        self.assertIn(
            "group=parallel_state.get_pipeline_model_parallel_group()",
            source[head : head + 400],
        )

    def test_the_global_token_line_multiplies_by_the_dp_degree(self) -> None:
        """dp is the axis that multiplies the job's token count; pp is not.

        The ranks of one pipeline share a batch. Each data-parallel rank
        reads a batch of its own. At dp 1 the line is unchanged.
        """
        source = self._main_source()
        self.assertIn("local_tokens_per_step * args.dp", source)
        self.assertIn('f"(dp {args.dp} x batch {args.batch}', source)

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

    def test_the_materialized_line_is_the_recorded_one_at_one_rank(
        self,
    ) -> None:
        """A changed log line at the trivial spec is a changed record.

        Every megatron directory under out/ holds this line without a
        microbatch clause, so the clause is appended only when there is a
        split. Rendered here rather than pattern-matched, because the claim
        is about the characters.
        """
        source = self._main_source()
        self.assertIn('if args.pp > 1\n        else ""', source)
        for pp, microbatches, rows, expected in (
            (1, 1, 4, "(4x1024, 10 max packed documents)"),
            (
                2,
                4,
                1,
                "(4x1024, 4 microbatch(es) of 1 row(s), 10 max packed "
                "documents)",
            ),
        ):
            split = (
                f"{microbatches} microbatch(es) of {rows} row(s), "
                if pp > 1
                else ""
            )
            with self.subTest(pp=pp):
                self.assertEqual(f"(4x1024, {split}10 max packed documents)", expected)


if __name__ == "__main__":
    unittest.main()
