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
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.layout import logs_by_rank
from benchmarks.e2e.megatron import train
from benchmarks.e2e.parallelism import (
    ParallelismSpec,
    TRIVIAL_SPEC,
    n_microbatches,
)
from benchmarks.e2e.registry import PIPER_1B_ROPE, PIPER_1B_SWIGLU
from benchmarks.e2e.validation import (
    ALL_REDUCE_MARKER,
    VALIDATION_PROFILES,
    validate_arm,
)
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
DP2 = ParallelismSpec(dp=2)


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

    def test_a_lone_non_zero_rank_keeps_its_own_number(self) -> None:
        """Rank 0 died before writing; rank 1 must not be renamed to it.

        The rank set is then {1}, which no declared world size matches, so
        validation fails the arm. Relabelling it 0 would instead hand a
        caller that does not validate -- ``evaluate`` on its own -- rank 1's
        rows under rank 0's name.
        """
        text = _prefixed(1, "Training completed")
        self.assertEqual(logs_by_rank(text), {1: text})

    def test_a_log_that_names_no_rank_is_still_rank_zero(self) -> None:
        # Every directory under out/ predates the prefix, and 0 is what it
        # was.
        text = "Training completed\n"
        self.assertEqual(logs_by_rank(text), {0: text})

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


def _write_traces(
    root: Path, rank: int, *, windows=(20, 40), markers=()
) -> None:
    """One rank's profiler windows, carrying the named kernels and no more.

    Per rank rather than per fixture, so a test can give two ranks different
    kernels -- which is what arm rule 13's "every rank" reading needs.
    """
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


class _ArmFixture:
    """A validated two-rank arm directory, so a test can spoil one rank."""

    def __init__(
        self, root: Path, *, ranks=(0, 1), windows=(20, 40), markers=()
    ) -> None:
        self.root = root
        for rank in ranks:
            _write_traces(root, rank, windows=windows, markers=markers)
        self.log = root / "baseline.log"

    def write(self, per_rank: dict[int, str]) -> None:
        self.log.write_text(
            "".join(_prefixed(rank, text) for rank, text in per_rank.items())
        )


def _titan_log(spec: ParallelismSpec = PP2) -> str:
    """One rank's titan output: every log line the rules read, and nothing else.

    The two parallelism lines are what arm rule 12 matches. They are built
    here from the same profile the validator uses, so this fixture cannot
    drift from the rule -- what it pins is that a rank whose OTHER lines are
    wrong still fails, not the wording of these two.
    """
    markers = "\n".join(
        VALIDATION_PROFILES["torchtitan"].parallelism_markers(
            spec, PIPER_1B_ROPE.workload
        )
    )
    return (
        _compiled_line("default").rstrip("\n")
        + "\n"
        + _SAC_LINE.rstrip("\n")
        + "\n"
        + _SIZE_LINE.rstrip("\n")
        + "\n"
        + markers
        + "\nTraining completed"
    )


_TITAN_TAIL = _titan_log()


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
            # The log a trivial run really writes. ``_TITAN_TAIL`` is the pp 2
            # log, and a trivial run cannot produce it: arm rule 12 refuses a
            # pipeline nobody requested.
            fixture.log.write_text(_titan_log(TRIVIAL_SPEC) + "\n")
            validate_arm(
                PIPER_1B_ROPE.arm("baseline"),
                fixture.root,
                fixture.log,
                PIPER_1B_ROPE.workload,
                parallelism=TRIVIAL_SPEC,
            )



class ArmRuleTwelveRefusesAnUnrequestedPipelineTests(unittest.TestCase):
    """The half of arm rule 12 that reads backwards.

    ``parallelism_markers`` proves the engine built the mesh that was asked
    for. It asks nothing when nothing was asked for, so a log from a real
    pipeline passed validation against the trivial spec and the run would
    have been published as single-GPU. This is the same inversion
    ``--compile-mode none`` applies to the compile marker.

    Both engines are checked, because both had the hole. Measured on real
    output before the rule was written: of 296 arm logs under ``out/``,
    exactly one matches either pattern, and it is a deliberate ``--pp 2``
    run.
    """

    def test_a_titan_pipeline_log_is_refused_at_the_trivial_spec(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary), ranks=(0,))
            fixture.log.write_text(_titan_log(PP2) + "\n")
            with self.assertRaisesRegex(RuntimeError, "declares no pipeline"):
                validate_arm(
                    PIPER_1B_ROPE.arm("baseline"),
                    fixture.root,
                    fixture.log,
                    PIPER_1B_ROPE.workload,
                    parallelism=TRIVIAL_SPEC,
                )

    def test_a_titan_trivial_log_still_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary), ranks=(0,))
            fixture.log.write_text(_titan_log(TRIVIAL_SPEC) + "\n")
            validate_arm(
                PIPER_1B_ROPE.arm("baseline"),
                fixture.root,
                fixture.log,
                PIPER_1B_ROPE.workload,
                parallelism=TRIVIAL_SPEC,
            )

    def test_the_megatron_pattern_reads_the_degree_not_the_line(self) -> None:
        """``pp=1`` is not a pipeline. Any other degree is."""
        pattern = VALIDATION_PROFILES["megatron"].pipelined_pattern
        trivial = "Megatron-LM parallelism: dp=1 pp=1 schedule=None"
        pipelined = "Megatron-LM parallelism: dp=1 pp=2 schedule=1F1B"
        deeper = "Megatron-LM parallelism: dp=1 pp=4 schedule=1F1B"
        self.assertIsNone(pattern.search(trivial))
        self.assertIsNotNone(pattern.search(pipelined))
        self.assertIsNotNone(pattern.search(deeper))

    def test_the_titan_pattern_does_not_match_a_one_gpu_mesh_line(self) -> None:
        pattern = VALIDATION_PROFILES["torchtitan"].pipelined_pattern
        self.assertIsNone(pattern.search(_titan_log(TRIVIAL_SPEC)))
        self.assertIsNotNone(pattern.search(_titan_log(PP2)))


class ArmRuleTwelveRefusesUnrequestedDataParallelismTests(unittest.TestCase):
    """The same inversion on the data-parallel axis, where it matters more.

    A pipeline rank and a single-GPU rank publish the same per-device
    throughput, so a pipeline published as one GPU misstates the mesh and
    not the rate. A data-parallel rank reads a batch of its own, so a
    ``dp 2`` run published under the trivial spec reads as roughly twice the
    true rate -- and every other rule passes, because the positive markers
    ask nothing when nothing was requested.
    """

    def test_a_titan_data_parallel_log_is_refused_at_the_trivial_spec(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary), ranks=(0,))
            fixture.log.write_text(_titan_log(DP2) + "\n")
            with self.assertRaisesRegex(
                RuntimeError, "declares no data parallelism"
            ):
                validate_arm(
                    PIPER_1B_ROPE.arm("baseline"),
                    fixture.root,
                    fixture.log,
                    PIPER_1B_ROPE.workload,
                    parallelism=TRIVIAL_SPEC,
                )

    def test_a_titan_trivial_log_still_validates(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary), ranks=(0,))
            fixture.log.write_text(_titan_log(TRIVIAL_SPEC) + "\n")
            validate_arm(
                PIPER_1B_ROPE.arm("baseline"),
                fixture.root,
                fixture.log,
                PIPER_1B_ROPE.workload,
                parallelism=TRIVIAL_SPEC,
            )

    def test_each_pattern_names_two_witnesses(self) -> None:
        """The engine's own mesh line, and this repo's wrapper line.

        The mesh line is logged whatever this repo's code does, so a degree
        above 1 shows there even in a run that never reached the wrapper.
        Either alone must fail the arm.
        """
        titan = VALIDATION_PROFILES["torchtitan"].data_parallel_pattern
        self.assertIsNotNone(
            titan.search(
                "Building device mesh with parallelism: pp=1, "
                "dp_replicate=2, dp_shard=1, cp=1, tp=1, ep=1"
            )
        )
        self.assertIsNotNone(
            titan.search(
                "piper1b data parallel: fully_shard applied "
                "(dp_replicate=2, dp_shard=1); 17 FSDP units"
            )
        )
        megatron = VALIDATION_PROFILES["megatron"].data_parallel_pattern
        self.assertIsNotNone(
            megatron.search(
                "Megatron-LM parallelism: dp=2 pp=1 schedule=None "
                "microbatches=1 stages=1"
            )
        )
        self.assertIsNotNone(
            megatron.search(
                "Megatron-LM data parallel: DistributedDataParallel over 2 "
                "ranks (overlap_grad_reduce=True, grad_reduce_in_fp32=False)"
            )
        )

    def test_a_shard_degree_is_data_parallelism_too(self) -> None:
        """ZeRO-3 is what an omitted shard-degree flag produces.

        ``parallelize_piper1b`` refuses it in the training process. This
        pattern is what stops such a log being published as single-GPU if
        the refusal is ever lifted.
        """
        titan = VALIDATION_PROFILES["torchtitan"].data_parallel_pattern
        self.assertIsNotNone(
            titan.search(
                "Building device mesh with parallelism: pp=1, "
                "dp_replicate=1, dp_shard=2, cp=1, tp=1, ep=1"
            )
        )

    def test_a_pipeline_only_log_is_not_data_parallel(self) -> None:
        """Read off the real ``pp 2, dp 1`` logs this harness has written.

        Both engines print their degrees on one line, so a pattern that
        matched the line rather than the degree would fail every honest
        pipeline run.
        """
        titan = VALIDATION_PROFILES["torchtitan"].data_parallel_pattern
        self.assertIsNone(
            titan.search(
                "Building device mesh with parallelism: pp=2, "
                "dp_replicate=1, dp_shard=1, cp=1, tp=1, ep=1"
            )
        )
        self.assertIsNone(titan.search(_titan_log(PP2)))
        megatron = VALIDATION_PROFILES["megatron"].data_parallel_pattern
        self.assertIsNone(
            megatron.search(
                "Megatron-LM parallelism: dp=1 pp=2 schedule=1F1B "
                "microbatches=4 stages=2"
            )
        )

    def test_a_double_digit_degree_is_not_read_as_one(self) -> None:
        """``dp=1`` must not match ``dp=12``, and the reverse."""
        megatron = VALIDATION_PROFILES["megatron"].data_parallel_pattern
        self.assertIsNotNone(
            megatron.search("Megatron-LM parallelism: dp=12 pp=1 schedule=None")
        )
        titan = VALIDATION_PROFILES["torchtitan"].data_parallel_pattern
        self.assertIsNotNone(titan.search("dp_replicate=10, dp_shard=1,"))
        self.assertIsNone(titan.search("dp_replicate=1, dp_shard=1,"))


class ArmRuleTwelveTests(unittest.TestCase):
    """Both engines must log the mesh they really built.

    Without this rule a run that ignored every ``--parallelism.*`` flag, or a
    megatron driver that read no ``RANK``, trains the whole model in one
    process and passes every other check.
    """

    def test_the_trivial_spec_asks_for_nothing(self) -> None:
        """There are no flags to ignore at one rank, and no old log to break.

        Every megatron directory under ``out/`` predates the driver's own
        parallelism line, and ``--resume`` re-validates what is on disk.
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary), ranks=(0,))
            fixture.log.write_text(
                _compiled_line("default") + _SAC_LINE + _SIZE_LINE
                + "Training completed\n"
            )
            validate_arm(
                PIPER_1B_ROPE.arm("baseline"),
                fixture.root,
                fixture.log,
                PIPER_1B_ROPE.workload,
            )

    def test_a_missing_mesh_line_fails_the_arm(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary))
            without = "\n".join(
                line
                for line in _titan_log().splitlines()
                if "Building device mesh" not in line
            )
            fixture.write({0: without, 1: _TITAN_TAIL})
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate_arm(
                    PIPER_1B_ROPE.arm("baseline"),
                    fixture.root,
                    fixture.log,
                    PIPER_1B_ROPE.workload,
                    parallelism=PP2,
                )

    def test_a_mesh_line_naming_another_mesh_fails_the_arm(self) -> None:
        """The line states what TorchTitan built, so a wrong one is the bug."""
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary))
            wrong = _titan_log().replace("pp=2", "pp=1")
            fixture.write({0: wrong, 1: wrong})
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate_arm(
                    PIPER_1B_ROPE.arm("baseline"),
                    fixture.root,
                    fixture.log,
                    PIPER_1B_ROPE.workload,
                    parallelism=PP2,
                )

    def test_a_wrong_microbatch_count_fails_the_arm(self) -> None:
        """The hazard this stage carries, on the titan side.

        Two engines that agree on the layer split and disagree on how many
        microbatches they move through it run two schedules under one label.
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary))
            wrong = _titan_log().replace("with 4 microbatches", "with 2 microbatches")
            fixture.write({0: wrong, 1: wrong})
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate_arm(
                    PIPER_1B_ROPE.arm("baseline"),
                    fixture.root,
                    fixture.log,
                    PIPER_1B_ROPE.workload,
                    parallelism=PP2,
                )

    def test_the_titan_markers_name_the_degrees_and_the_schedule(self) -> None:
        markers = VALIDATION_PROFILES["torchtitan"].parallelism_markers(
            PP2, PIPER_1B_ROPE.workload
        )
        self.assertEqual(
            markers,
            (
                "Building device mesh with parallelism: pp=2, "
                "dp_replicate=1, dp_shard=1, cp=1, tp=1, ep=1",
                "Using pipeline schedule 1F1B with 4 microbatches and 2 stages",
            ),
        )

    def test_the_titan_dp_marker_is_the_line_parallelize_prints(self) -> None:
        """The one titan marker that proves a gradient reduction.

        TorchTitan logs its mesh line from ``ParallelDims``, before
        ``parallelize_fn`` runs, so that line survives a run that skipped the
        data-parallel path entirely. ``parallelize_piper1b`` counts the FSDP
        units the delegate really built and prints its own line after the
        count. ``validation.py`` cannot import that constant -- it would pull
        torch into the parent -- so the string is stated twice and this test
        is the link.
        """
        from benchmarks.models.piper_qwen3.parallelize import (
            DATA_PARALLEL_LINE,
        )

        markers = VALIDATION_PROFILES["torchtitan"].parallelism_markers(
            DP2, PIPER_1B_ROPE.workload
        )
        self.assertIn(
            DATA_PARALLEL_LINE.format(replicate=2, shard=1), markers
        )

    def test_no_dp_marker_where_no_reduction_happens(self) -> None:
        """A pipeline rank reduces no gradient, so it prints no such line.

        Asking for the line there would fail an honest run, which is the
        direction a validation rule must never take.
        """
        for spec in (TRIVIAL_SPEC, PP2):
            with self.subTest(spec=spec):
                markers = VALIDATION_PROFILES[
                    "torchtitan"
                ].parallelism_markers(spec, PIPER_1B_ROPE.workload)
                self.assertEqual(
                    [m for m in markers if "data parallel" in m], []
                )

    def test_a_dp_run_that_skipped_fully_shard_fails_the_arm(self) -> None:
        """The named hazard of this stage, stated as one assertion.

        The mesh line is present and every other rule passes. Without the
        third marker the arm would publish roughly twice the true speed.
        """
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary))
            whole = _titan_log(DP2)
            unwrapped = "\n".join(
                line
                for line in whole.splitlines()
                if "data parallel" not in line
            )
            fixture.write({0: unwrapped, 1: unwrapped})
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate_arm(
                    PIPER_1B_ROPE.arm("baseline"),
                    fixture.root,
                    fixture.log,
                    PIPER_1B_ROPE.workload,
                    parallelism=DP2,
                )

    def test_every_rank_must_carry_an_all_reduce_under_dp(self) -> None:
        """Arm rule 13, and its reading is EVERY rank rather than any.

        At dp above 1 every rank sits in a data-parallel group of that size,
        so every rank reduces. That is what makes "every rank" provable here
        where arm rule 6's reading is still open.
        """
        for missing in (0, 1):
            with self.subTest(rank_without_the_marker=missing):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    fixture = _ArmFixture(root, ranks=())
                    for rank in (0, 1):
                        _write_traces(
                            root,
                            rank,
                            markers=()
                            if rank == missing
                            else (ALL_REDUCE_MARKER,),
                        )
                    log = _titan_log(DP2)
                    fixture.write({0: log, 1: log})
                    with self.assertRaisesRegex(
                        RuntimeError, f"rank {missing}'s profiler traces"
                    ):
                        validate_arm(
                            PIPER_1B_ROPE.arm("baseline"),
                            fixture.root,
                            fixture.log,
                            PIPER_1B_ROPE.workload,
                            parallelism=DP2,
                        )

    def test_a_pipeline_only_run_needs_no_all_reduce(self) -> None:
        """The rule reads ``dp``, not the world size.

        A pipeline synchronizes no gradient. Asking a pp-only run for an
        all-reduce would fail an honest run, which is the direction a
        validation rule must never take.
        """
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

    def test_a_pipeline_collective_cannot_satisfy_the_rule(self) -> None:
        """SendRecv and Broadcast are what a pipeline emits with no
        reduction.

        A marker of ``nccl`` alone would pass a dp run that reduced nothing,
        which is the one thing this rule exists to catch.
        """
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = _ArmFixture(root, ranks=())
            for rank in (0, 1):
                _write_traces(
                    root,
                    rank,
                    markers=(
                        "ncclDevKernel_SendRecv",
                        "ncclDevKernel_Broadcast_RING_LL",
                    ),
                )
            log = _titan_log(DP2)
            fixture.write({0: log, 1: log})
            with self.assertRaisesRegex(RuntimeError, "carry no"):
                validate_arm(
                    PIPER_1B_ROPE.arm("baseline"),
                    fixture.root,
                    fixture.log,
                    PIPER_1B_ROPE.workload,
                    parallelism=DP2,
                )

    def test_a_dp_run_with_the_marker_on_every_rank_passes(self) -> None:
        """And the suffix NCCL chose is not part of the marker.

        The algorithm and protocol depend on the message size and the
        topology, so pinning ``_RING_LL`` whole would fail an honest run
        whose buckets chose another.
        """
        for kernel in (
            "ncclDevKernel_AllReduce_Sum_bf16_RING_LL",
            "ncclDevKernel_AllReduce_Sum_bf16_TREE_LL128",
        ):
            with self.subTest(kernel=kernel):
                with tempfile.TemporaryDirectory() as temporary:
                    root = Path(temporary)
                    fixture = _ArmFixture(root, ranks=())
                    for rank in (0, 1):
                        _write_traces(root, rank, markers=(kernel,))
                    log = _titan_log(DP2)
                    fixture.write({0: log, 1: log})
                    validate_arm(
                        PIPER_1B_ROPE.arm("baseline"),
                        fixture.root,
                        fixture.log,
                        PIPER_1B_ROPE.workload,
                        parallelism=DP2,
                    )

    def test_the_megatron_marker_is_the_line_the_driver_prints(self) -> None:
        """The validator and the driver state one line in two places.

        MODE_LINE already carries that cost, and the same comment. This test
        is the link: the driver formats its own constants with the values
        ``pipeline_settings`` gives it, and the strings must be equal.

        The dp-only cell is the one that would have gone wrong quietly: the
        driver runs ONE microbatch at ``pp`` 1 and ``n_microbatches``
        describes the split a pipeline would make, so a validator that read
        the latter would fail every honest dp run.
        """
        for spec, schedule in ((PP2, "1F1B"), (DP2, None)):
            with self.subTest(spec=spec):
                args = SimpleNamespace(
                    batch=4, pp=spec.pp, pp_microbatch_size=1
                )
                _, microbatches = train.pipeline_settings(args)
                printed = [
                    train.PARALLELISM_LINE.format(
                        dp=spec.dp,
                        pp=spec.pp,
                        schedule=schedule,
                        microbatches=microbatches,
                        stages=spec.pp,
                    )
                ]
                if spec.dp > 1:
                    printed.append(
                        train.DATA_PARALLEL_LINE.format(
                            dp=spec.dp, overlap=True, fp32=False
                        )
                    )
                self.assertEqual(
                    VALIDATION_PROFILES["megatron"].parallelism_markers(
                        spec, PIPER_1B_ROPE.workload
                    ),
                    tuple(printed),
                )

    def test_both_engines_move_the_same_number_of_microbatches(self) -> None:
        """The named hazard of this stage, stated as one assertion.

        TorchTitan derives ``local_batch_size // pipeline_parallel_microbatch
        _size`` inside ``_build_pipeline_schedule``; the megatron driver
        derives its own count in ``pipeline_settings``. A disagreement runs
        two schedules under one label, and no correctness gate could see it.
        """
        for batch, microbatch_size in ((4, 1), (8, 2), (8, 1)):
            with self.subTest(batch=batch, microbatch_size=microbatch_size):
                spec = ParallelismSpec(
                    pp=2, pp_schedule="1F1B", pp_microbatch_size=microbatch_size
                )
                titan = n_microbatches(spec, local_batch_size=batch)
                _, megatron = train.pipeline_settings(
                    SimpleNamespace(
                        batch=batch, pp=2, pp_microbatch_size=microbatch_size
                    )
                )
                self.assertEqual(titan, megatron)

    def test_a_profile_that_can_prove_nothing_refuses_the_run(self) -> None:
        """An empty marker tuple is a refusal, never a pass.

        No profile returns one today. The day one does -- a data-parallel
        titan run has no schedule line, for instance -- the run must fail
        rather than publish a mesh nothing checked.
        """
        silent = replace(
            VALIDATION_PROFILES["torchtitan"],
            parallelism_markers=lambda spec, workload: (),
        )
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _ArmFixture(Path(temporary))
            fixture.write({0: _TITAN_TAIL, 1: _TITAN_TAIL})
            with mock.patch.dict(
                "benchmarks.e2e.validation.VALIDATION_PROFILES",
                {"torchtitan": silent},
            ):
                with self.assertRaisesRegex(RuntimeError, "logs nothing"):
                    validate_arm(
                        PIPER_1B_ROPE.arm("baseline"),
                        fixture.root,
                        fixture.log,
                        PIPER_1B_ROPE.workload,
                        parallelism=PP2,
                    )


if __name__ == "__main__":
    unittest.main()
