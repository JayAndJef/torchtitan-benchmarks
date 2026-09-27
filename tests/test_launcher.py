"""The shared launcher: the argv and the child environment of one launch."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.axes import RequestedAxes, RunRequest
from benchmarks.e2e.engines.api import Launch
from benchmarks.e2e.engines.megatron_stock.engine import MegatronStockEngine
from benchmarks.e2e.engines.registry import ENGINES
from benchmarks.e2e.registry import ENGINES as ENGINES_SCENARIO
from benchmarks.execution.affinity import CpuPinning
from benchmarks.execution.launcher import (
    LAUNCHER_KEYS,
    LOG_RANK_TEMPLATE,
    PINNING_DECLINED,
    build_command,
    command_line,
    torchrun_flags,
)
from benchmarks.e2e.runner import execute_run
from tests.engine_helpers import run_spec
from tests.test_golden import HARDWARE, HOST_ENVIRONMENT, METADATA, PINNING


PINNED = CpuPinning(
    ("numactl", "--cpunodebind=1", "--membind=1"),
    "numactl --cpunodebind=1 --membind=1",
)
"""A host pinning with a prefix."""

TARGET = ("-m", "some.module", "--flag")


def _build(launch: Launch, world_size: int = 1, base_env=None):
    return build_command(
        launch,
        world_size=world_size,
        gpu=",".join(str(rank) for rank in range(world_size)),
        pinning=PINNED,
        base_env=base_env if base_env is not None else {"PATH": "/usr/bin"},
    )


class ArgvTests(unittest.TestCase):
    def test_a_per_rank_launch_starts_torchrun_at_every_world_size(self) -> None:
        for world_size in (1, 2, 8):
            with self.subTest(world_size=world_size):
                launched = _build(
                    Launch(target=TARGET, processes="per_rank", pin=True), world_size
                )
                self.assertEqual(
                    launched.argv,
                    (
                        *PINNED.prefix,
                        sys.executable,
                        *torchrun_flags(world_size),
                        *TARGET,
                    ),
                )

    def test_the_torchrun_flags_name_every_rank(self) -> None:
        self.assertEqual(
            torchrun_flags(2),
            (
                "-m",
                "torch.distributed.run",
                "--nproc-per-node=2",
                "--rdzv-backend",
                "c10d",
                "--rdzv-endpoint",
                "localhost:0",
                "--local-ranks-filter",
                "0,1",
                "--role",
                "rank",
                "--tee",
                "3",
            ),
        )

    def test_a_single_launch_starts_no_torchrun(self) -> None:
        launched = _build(Launch(target=TARGET, processes="single", pin=True), 2)
        self.assertEqual(launched.argv, (*PINNED.prefix, sys.executable, *TARGET))

    def test_a_declined_pinning_drops_the_prefix_and_says_so(self) -> None:
        launched = _build(Launch(target=TARGET, processes="single", pin=False))
        self.assertEqual(launched.argv, (sys.executable, *TARGET))
        self.assertEqual(launched.cpu_pinning, PINNING_DECLINED)

    def test_an_accepted_pinning_records_the_host_description(self) -> None:
        launched = _build(Launch(target=TARGET, processes="single", pin=True))
        self.assertEqual(launched.cpu_pinning, PINNED.description)


class EnvironmentTests(unittest.TestCase):
    def test_the_launcher_keys_replace_a_host_value(self) -> None:
        launched = _build(
            Launch(target=TARGET, processes="per_rank", pin=True),
            2,
            base_env={
                "PATH": "/usr/bin",
                "PYTORCH_ALLOC_CONF": "max_split_size_mb:128",
                "CUDA_VISIBLE_DEVICES": "7",
            },
        )
        self.assertEqual(
            {key: launched.env[key] for key in LAUNCHER_KEYS},
            {
                "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
                "CUDA_VISIBLE_DEVICES": "0,1",
                "NGPU": "2",
                "LOG_RANK": "0,1",
                "TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE": LOG_RANK_TEMPLATE,
                "PYTORCH_ALLOC_CONF": "expandable_segments:True",
            },
        )
        self.assertEqual(launched.env["PATH"], "/usr/bin")

    def test_a_single_launch_gets_no_rank_logging(self) -> None:
        launched = _build(
            Launch(target=TARGET, processes="single", pin=True),
            base_env={
                "PATH": "/usr/bin",
                "LOG_RANK": "5",
                "TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE": "[x]:",
            },
        )
        self.assertNotIn("LOG_RANK", launched.env)
        self.assertNotIn("TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE", launched.env)

    def test_an_engine_key_reaches_the_child(self) -> None:
        launched = _build(
            Launch(target=TARGET, processes="single", pin=True, env={"NVTE_X": "1"})
        )
        self.assertEqual(launched.env["NVTE_X"], "1")

    def test_an_engine_key_that_the_launcher_owns_is_refused(self) -> None:
        launch = Launch(
            target=TARGET, processes="single", pin=True, env={"NGPU": "4"}
        )
        with self.assertRaisesRegex(ValueError, "NGPU, which the launcher owns"):
            _build(launch)
        with self.assertRaisesRegex(ValueError, "NGPU, which the launcher owns"):
            command_line(launch, world_size=1, pinning=PINNED)


class LaunchRecordTests(unittest.TestCase):
    def test_a_target_that_does_not_start_with_the_module_flag_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "starts with '-m'"):
            Launch(target=("./run_train.sh",), processes="per_rank", pin=True)

    def test_an_unknown_process_model_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "per_rank, single"):
            Launch(target=TARGET, processes="forked", pin=True)

    def test_every_engine_launches_one_process_per_rank_and_takes_the_pinning(
        self,
    ) -> None:
        run = run_spec(ac_mode="none")
        for arm in ENGINES_SCENARIO.arms:
            engine = ENGINES[type(arm.config)]
            with self.subTest(arm=arm.name):
                launch = engine.launch(run, arm, Path("/tmp/arm"))
                self.assertEqual(launch.processes, "per_rank")
                self.assertTrue(launch.pin)
                self.assertFalse(set(launch.env) & LAUNCHER_KEYS)


class RunnerRefusalTests(unittest.TestCase):
    def test_a_launcher_key_is_refused_before_any_arm_starts(self) -> None:
        """The runner builds every command line before it writes or starts anything."""
        colliding = Launch(
            target=TARGET, processes="per_rank", pin=True, env={"NGPU": "4"}
        )
        process_runner = mock.Mock()
        with tempfile.TemporaryDirectory() as temporary, mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=(HARDWARE, {"requested_gpu": "0", **METADATA}),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning", return_value=PINNING
        ), mock.patch.object(
            MegatronStockEngine, "launch", return_value=colliding
        ):
            out_dir = Path(temporary) / "run"
            with self.assertRaisesRegex(ValueError, "NGPU, which the launcher owns"):
                execute_run(
                    RunRequest(
                        gpu="0",
                        scenario_name="engines",
                        out_dir=out_dir,
                        axes=RequestedAxes(model_size="1b", profile=False),
                    ),
                    process_runner=process_runner,
                    environment=dict(HOST_ENVIRONMENT),
                )
            process_runner.assert_not_called()
            self.assertFalse(out_dir.exists())


if __name__ == "__main__":
    unittest.main()
