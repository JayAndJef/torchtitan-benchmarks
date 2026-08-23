"""Validation under a pipeline split: one log holds every rank.

The runner gives one training subprocess one log file, so a two-rank run
writes both ranks into ``<arm>.log`` and torchrun prefixes each line with
the rank that wrote it. Every rule that reads the log was written when
"the log" and "the rank" were the same thing. Read against the whole file
they answer the weaker question -- *some* rank did this -- and arm rule 4 is
the case that matters: a kernel that silently degraded on rank 1 leaves rank
0's log clean.

The other half is the rank that wrote nothing at all. It appears in neither
the log split nor the trace grouping, so no rule fires for it and the
evaluation publishes a maximum over the survivors. The declared world size
is what makes it visible.
"""

import gzip
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.layout import logs_by_rank
from benchmarks.e2e.parallelism import ParallelismSpec, TRIVIAL_SPEC
from benchmarks.e2e.registry import PIPER_1B_ROPE, PIPER_1B_SWIGLU
from benchmarks.e2e.validation import validate_arm
from benchmarks.execution.environment import LOG_RANK_TEMPLATE
from benchmarks.execution.paths import RuntimePaths
from benchmarks.execution.environment import runtime_environment
from tests.test_runner import (
    PIPER_OPTIMIZED_SWIGLU_OVERRIDE,
    _SAC_LINE,
    _SIZE_LINE,
    _compiled_line,
)


PP2 = ParallelismSpec(pp=2, pp_schedule="1F1B")


def _prefixed(rank: int, text: str) -> str:
    """One rank's lines, as torchrun tees them."""
    return "".join(f"[rank{rank}]:{line}\n" for line in text.splitlines())


class LogsByRankTests(unittest.TestCase):
    def test_an_unprefixed_log_comes_back_whole(self) -> None:
        """The megatron arm's log at one rank: no torchrun, no prefix."""
        text = "Megatron-LM training loop (mode=default,\nTraining completed\n"
        self.assertEqual(logs_by_rank(text), {0: text})

    def test_a_one_rank_log_comes_back_whole_prefix_and_all(self) -> None:
        """Which is what makes this reader inert on every log written so far.

        A single-rank titan log already carries ``[rank0]:`` prefixes, and it
        also carries unprefixed lines that belong to no rank: the runner's
        header, the nvidia-smi block and run_train.sh's shell trace.
        Splitting would drop those and change what every rule reads.
        """
        text = "# scenario=x arm=y\n" + _prefixed(0, "Training completed")
        self.assertEqual(logs_by_rank(text), {0: text})

    def test_two_ranks_split_and_keep_only_their_own_lines(self) -> None:
        text = (
            "# scenario=x arm=y\n"
            + _prefixed(0, "first stage\nTraining completed")
            + _prefixed(1, "second stage\nTraining completed")
        )
        split = logs_by_rank(text)
        self.assertEqual(sorted(split), [0, 1])
        self.assertIn("first stage", split[0])
        self.assertNotIn("second stage", split[0])
        self.assertIn("second stage", split[1])
        # The launcher's own unprefixed lines belong to no rank.
        self.assertNotIn("scenario=x", split[0])
        self.assertNotIn("scenario=x", split[1])

    def test_a_silent_rank_leaves_one_entry(self) -> None:
        """And the caller then sees a rank set that is not the declared one."""
        self.assertEqual(
            sorted(logs_by_rank(_prefixed(0, "Training completed"))), [0]
        )

    def test_the_reader_parses_what_the_environment_asks_torchrun_to_write(
        self,
    ) -> None:
        """The template and the regex are one grammar stated twice.

        They are not one constant because the reader must also parse
        torchrun's own default, ``[${role_name}${local_rank}]:``, which
        every single-GPU log already carries through run_train.sh's
        ``--role rank``.
        """
        rendered = LOG_RANK_TEMPLATE.replace("${rank}", "7")
        text = f"{rendered}alpha\n{rendered.replace('7', '9')}beta\n"
        split = logs_by_rank(text)
        self.assertEqual(sorted(split), [7, 9])
        self.assertEqual(split[7], "alpha\n")


class RankLoggingEnvironmentTests(unittest.TestCase):
    def _environment(self, world_size: int) -> dict[str, str]:
        paths = RuntimePaths.resolve(environment={"PATH": "/usr/bin"})
        return runtime_environment(
            paths, "0", environment={"PATH": "/usr/bin"}, world_size=world_size
        )

    def test_one_rank_sets_neither_variable(self) -> None:
        """A single-GPU log stays byte for byte the log it always was."""
        environment = self._environment(1)
        self.assertNotIn("LOG_RANK", environment)
        self.assertNotIn("TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE", environment)

    def test_two_ranks_ask_for_every_rank_and_the_global_prefix(self) -> None:
        """TorchTitan's run_train.sh defaults LOG_RANK to 0.

        Without this, rank 1's output reaches no console and every per-rank
        rule below has nothing to read.
        """
        environment = self._environment(2)
        self.assertEqual(environment["LOG_RANK"], "0,1")
        self.assertEqual(
            environment["TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE"],
            LOG_RANK_TEMPLATE,
        )


class _ArmFixture:
    """A validated two-rank arm directory, so a test can spoil one rank."""

    def __init__(
        self, root: Path, *, ranks=(0, 1), windows=(20, 40), markers=()
    ) -> None:
        self.root = root
        for rank in ranks:
            for iteration in windows:
                trace = (
                    root
                    / "profiling"
                    / "traces"
                    / f"iteration_{iteration}"
                    / f"rank{rank}_trace.json.gz"
                )
                trace.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(trace, "wt") as handle:
                    handle.write("cudaLaunchKernel\n")
                    for marker in markers:
                        handle.write(marker + "\n")
        self.log = root / "baseline.log"

    def write(self, per_rank: dict[int, str]) -> None:
        self.log.write_text(
            "".join(_prefixed(rank, text) for rank, text in per_rank.items())
        )


_TITAN_TAIL = (
    _compiled_line("default").rstrip("\n")
    + "\n"
    + _SAC_LINE.rstrip("\n")
    + "\n"
    + _SIZE_LINE.rstrip("\n")
    + "\nTraining completed"
)


class PerRankLogRuleTests(unittest.TestCase):
    """Arm rules 1, 4 and 11, asked of each rank rather than of the file."""

    def test_a_clean_two_rank_log_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary))
            fixture.write({0: _TITAN_TAIL, 1: _TITAN_TAIL})
            validate_arm(
                PIPER_1B_ROPE.arm("baseline"),
                fixture.root,
                fixture.log,
                PIPER_1B_ROPE.workload,
                parallelism=PP2,
            )

    def test_a_fallback_on_rank_one_alone_fails_the_arm(self) -> None:
        """Arm rule 4, and the reason the split is worth its cost.

        An optimized kernel that degraded on the second stage only leaves
        rank 0's log clean, so the whole-file read passes it and the arm
        publishes a number measured by the stock path.
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary))
            fixture.write(
                {
                    0: _TITAN_TAIL,
                    1: "falling back to the PyTorch implementation\n"
                    + _TITAN_TAIL,
                }
            )
            with self.assertRaisesRegex(RuntimeError, "rank 1"):
                validate_arm(
                    PIPER_1B_ROPE.arm("baseline"),
                    fixture.root,
                    fixture.log,
                    PIPER_1B_ROPE.workload,
                    parallelism=PP2,
                )

    def test_a_rank_that_never_finished_fails_the_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary))
            fixture.write(
                {
                    0: _TITAN_TAIL,
                    1: _TITAN_TAIL.replace("Training completed", "killed"),
                }
            )
            with self.assertRaisesRegex(RuntimeError, "did not complete"):
                validate_arm(
                    PIPER_1B_ROPE.arm("baseline"),
                    fixture.root,
                    fixture.log,
                    PIPER_1B_ROPE.workload,
                    parallelism=PP2,
                )

    def test_the_override_count_is_the_whole_models_on_every_rank(self) -> None:
        """Arm rule 2 under a split, and the count does not halve.

        TorchTitan rewrites the config nodes and prints the parameter count
        while building the whole model, before ``pipelining_fn`` splits it.
        Both counts are therefore the whole model's on every rank.
        """
        arm = PIPER_1B_SWIGLU.arm("piper_optimized_triton")
        applied = (
            f"[Override] {PIPER_OPTIMIZED_SWIGLU_OVERRIDE}: "
            "model_spec.model.layers.0.moe ...\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(
                Path(temporary), markers=arm.trace_kernel_markers
            )
            whole = _TITAN_TAIL + "\n" + applied * 16
            fixture.write({0: whole, 1: whole})
            validate_arm(
                arm,
                fixture.root,
                fixture.log,
                PIPER_1B_SWIGLU.workload,
                parallelism=PP2,
            )

            half = _TITAN_TAIL + "\n" + applied * 8
            fixture.write({0: whole, 1: half})
            with self.assertRaisesRegex(RuntimeError, "expected 16 override"):
                validate_arm(
                    arm,
                    fixture.root,
                    fixture.log,
                    PIPER_1B_SWIGLU.workload,
                    parallelism=PP2,
                )


class RankCoverageTests(unittest.TestCase):
    """A rank that left nothing behind is refused, not skipped."""

    def test_a_rank_with_no_log_output_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary))
            fixture.write({0: _TITAN_TAIL})
            with self.assertRaisesRegex(RuntimeError, "wrote nothing"):
                validate_arm(
                    PIPER_1B_ROPE.arm("baseline"),
                    fixture.root,
                    fixture.log,
                    PIPER_1B_ROPE.workload,
                    parallelism=PP2,
                )

    def test_a_rank_with_no_traces_is_refused(self) -> None:
        """Arm rules 5 and 7 see the ranks that wrote something.

        A rank that wrote no trace is in neither, so without the declared
        world size the evaluation would take its maximum over the survivors.
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary), ranks=(0,))
            fixture.write({0: _TITAN_TAIL, 1: _TITAN_TAIL})
            with self.assertRaisesRegex(RuntimeError, "no trace"):
                validate_arm(
                    PIPER_1B_ROPE.arm("baseline"),
                    fixture.root,
                    fixture.log,
                    PIPER_1B_ROPE.workload,
                    parallelism=PP2,
                )

    def test_neither_check_fires_at_the_trivial_spec(self) -> None:
        """The default is the run this repo has always done.

        A single-rank arm carries one log and one rank's traces, and neither
        of the two checks above may narrow it.
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary), ranks=(0,))
            fixture.log.write_text(_TITAN_TAIL + "\n")
            validate_arm(
                PIPER_1B_ROPE.arm("baseline"),
                fixture.root,
                fixture.log,
                PIPER_1B_ROPE.workload,
                parallelism=TRIVIAL_SPEC,
            )


if __name__ == "__main__":
    unittest.main()
