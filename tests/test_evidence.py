"""Tests for the harness facts: completion, model size, mesh and finite trajectories."""

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.engines.api import MeshObserved, RankEvidence
from benchmarks.e2e.engines.megatron_stock.evidence import (
    read_evidence as stock_evidence,
)
from benchmarks.e2e.engines.torchtitan.evidence import (
    read_evidence as titan_evidence,
)
from benchmarks.e2e.evidence import check_evidence, non_finite_refusals
from benchmarks.e2e.parallelism import ParallelismSpec
from benchmarks.e2e.registry import ENGINES
from benchmarks.models.piper_qwen3.shape import PIPER_1B
from tests.engine_helpers import run_spec, validate

DP2 = ParallelismSpec(dp=2)
TRIVIAL_MESH = MeshObserved(dp=1, pp=1, ep=1, zero=None)


def _evidence(
    rank: int,
    *,
    completed: bool = True,
    param_count: int | None = PIPER_1B.param_count,
    mesh: MeshObserved | None = TRIVIAL_MESH,
) -> RankEvidence:
    return RankEvidence(
        rank=rank, completed=completed, param_count=param_count, mesh=mesh
    )


class CompletionTests(unittest.TestCase):
    def test_a_clean_rank_passes(self) -> None:
        self.assertEqual(check_evidence(run_spec(), {0: _evidence(0)}), [])

    def test_a_missing_rank_fails(self) -> None:
        mesh = MeshObserved(dp=2, pp=1, ep=1, zero=0)
        self.assertIn(
            "rank 1 wrote nothing",
            check_evidence(
                run_spec(parallelism=DP2), {0: _evidence(0, mesh=mesh)}
            ),
        )

    def test_a_rank_outside_the_run_fails(self) -> None:
        self.assertIn(
            "rank 1 wrote output, and the run declares 1 rank(s)",
            check_evidence(run_spec(), {0: _evidence(0), 1: _evidence(1)}),
        )

    def test_an_unfinished_rank_fails(self) -> None:
        self.assertIn(
            "training did not complete on rank 0",
            check_evidence(run_spec(), {0: _evidence(0, completed=False)}),
        )


class ModelTests(unittest.TestCase):
    def test_a_wrong_parameter_count_fails(self) -> None:
        (refusal,) = check_evidence(
            run_spec(), {0: _evidence(0, param_count=PIPER_1B.param_count + 1)}
        )
        self.assertIn("did not apply on rank 0", refusal)

    def test_a_run_where_no_rank_states_the_count_fails(self) -> None:
        (refusal,) = check_evidence(run_spec(), {0: _evidence(0, param_count=None)})
        self.assertIn("no rank states a parameter count", refusal)

    def test_one_rank_that_states_the_count_is_enough(self) -> None:
        mesh = MeshObserved(dp=1, pp=2, ep=1, zero=None)
        spec = ParallelismSpec(pp=2, pp_schedule="1F1B")
        self.assertEqual(
            check_evidence(
                run_spec(parallelism=spec),
                {0: _evidence(0, mesh=mesh), 1: _evidence(1, param_count=None, mesh=mesh)},
            ),
            [],
        )


class MeshTests(unittest.TestCase):
    def test_a_dp2_log_recorded_as_dp1_fails(self) -> None:
        mesh = MeshObserved(dp=2, pp=1, ep=1, zero=0)
        (refusal,) = check_evidence(run_spec(), {0: _evidence(0, mesh=mesh)})
        self.assertEqual(
            refusal, "the run declares no data parallelism, and rank 0 records dp=2"
        )

    def test_a_dp2_log_of_either_engine_recorded_as_dp1_fails(self) -> None:
        for arm_name, log in (
            (
                "titan_eager",
                "Building device mesh with parallelism: pp=1, dp_replicate=2, "
                "dp_shard=1, cp=1, tp=1, ep=1\n",
            ),
            (
                "megatron_stock",
                "Megatron-LM stock parallelism: dp=2 pp=1 ep=1 schedule=1F1B "
                "microbatches=1 stages=1\n",
            ),
        ):
            with self.subTest(arm=arm_name), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "arm.log"
                path.write_text(
                    log
                    + f"size: {PIPER_1B.param_count:,} total parameters\n"
                    + "Training completed\n"
                )
                with self.assertRaisesRegex(
                    RuntimeError, "declares no data parallelism, and rank 0 records dp=2"
                ):
                    validate(
                        run_spec(ac_mode="none", profile=False),
                        ENGINES.arm(arm_name),
                        Path(temporary),
                        path,
                    )

    def test_a_pipeline_and_an_expert_degree_are_compared(self) -> None:
        refusals = check_evidence(
            run_spec(),
            {0: _evidence(0, mesh=MeshObserved(dp=1, pp=2, ep=2, zero=None))},
        )
        self.assertEqual(
            refusals,
            [
                "the run declares no pipeline, and rank 0 records pp=2",
                "the run declares no expert parallelism, and rank 0 records ep=2",
            ],
        )

    def test_the_zero_level_is_compared_above_one_data_parallel_rank(self) -> None:
        run = run_spec(parallelism=DP2)
        self.assertEqual(
            check_evidence(
                run,
                {
                    rank: _evidence(rank, mesh=MeshObserved(dp=2, pp=1, ep=1, zero=1))
                    for rank in (0, 1)
                },
            ),
            [
                "the run declares zero 0, and rank 0 records zero 1",
                "the run declares zero 0, and rank 1 records zero 1",
            ],
        )
        self.assertIn(
            "rank 0 does not state its ZeRO level at dp 2",
            check_evidence(
                run,
                {
                    rank: _evidence(
                        rank, mesh=MeshObserved(dp=2, pp=1, ep=1, zero=None)
                    )
                    for rank in (0, 1)
                },
            ),
        )

    def test_the_zero_level_is_not_compared_at_one_data_parallel_rank(self) -> None:
        self.assertEqual(
            check_evidence(
                run_spec(parallelism=ParallelismSpec(zero=1)), {0: _evidence(0)}
            ),
            [],
        )

    def test_a_rank_without_a_mesh_fails(self) -> None:
        self.assertIn(
            "rank 0 states no mesh",
            check_evidence(run_spec(), {0: _evidence(0, completed=False, mesh=None)}),
        )


class ReaderTests(unittest.TestCase):
    def test_an_empty_rank_states_nothing(self) -> None:
        for read in (titan_evidence, stock_evidence):
            with self.subTest(read=read.__module__):
                self.assertEqual(
                    read(2, "\n"),
                    RankEvidence(rank=2, completed=False, param_count=None, mesh=None),
                )

    def test_a_rank_that_states_no_mesh_ran_on_one_device(self) -> None:
        for read in (titan_evidence, stock_evidence):
            with self.subTest(read=read.__module__):
                self.assertEqual(read(0, "Training completed\n").mesh, TRIVIAL_MESH)

    def test_two_parameter_counts_raise(self) -> None:
        for read in (titan_evidence, stock_evidence):
            with self.subTest(read=read.__module__):
                with self.assertRaisesRegex(RuntimeError, "two values of its parameter count"):
                    read(
                        0,
                        "size: 1 total parameters\nsize: 2 total parameters\n",
                    )

    def test_a_titan_wrapper_that_contradicts_the_mesh_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "wraps dp_replicate=1, dp_shard=2"):
            titan_evidence(
                0,
                "Building device mesh with parallelism: pp=1, dp_replicate=2, "
                "dp_shard=1, cp=1, tp=1, ep=1\n"
                "piper1b data parallel: fully_shard applied "
                "(dp_replicate=1, dp_shard=2); 17 FSDP units\n",
            )

    def test_a_stock_wrapper_that_contradicts_the_mesh_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "wraps dp=4 ep=1"):
            stock_evidence(
                0,
                "Megatron-LM stock parallelism: dp=2 pp=1 ep=1 schedule=1F1B\n"
                "Megatron-LM stock data parallel: DistributedDataParallel over "
                "4 ranks (overlap_grad_reduce=False, grad_reduce_in_fp32=True, "
                "sharding_strategy=no_shard, expert_parallel=1, "
                "optimizer=ChainedOptimizer[DistributedOptimizer])\n",
            )


class NonFiniteTests(unittest.TestCase):
    def test_the_first_non_finite_value_of_each_rank_is_named(self) -> None:
        self.assertEqual(
            non_finite_refusals(
                {
                    0: "step: 1 loss: 2.0 grad_norm: 1.0\n",
                    1: "step: 1 loss: nan grad_norm: 1.0\n"
                    "step: 2 loss: nan grad_norm: inf\n",
                }
            ),
            [
                "rank 1 logged a non-finite loss at step 1 (nan)",
                "rank 1 logged a non-finite grad_norm at step 2 (inf)",
            ],
        )

    def test_a_diverged_rank_fails_validation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "arm.log"
            path.write_text(
                "step: 1 loss: nan grad_norm: 1.0\n"
                f"size: {PIPER_1B.param_count:,} total parameters\n"
                "Training completed\n"
            )
            with self.assertRaisesRegex(
                RuntimeError, "rank 0 logged a non-finite loss at step 1"
            ):
                validate(
                    run_spec(ac_mode="none", profile=False),
                    ENGINES.arm("megatron_stock"),
                    Path(temporary),
                    path,
                )


if __name__ == "__main__":
    unittest.main()
