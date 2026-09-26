"""Golden fixtures that pin the e2e harness to its baseline behavior.

The fixtures under ``tests/fixtures/golden/`` record commit c4737f4 on
``master``, and they are the baseline of the engine refactor. Never write
them again from newer code. A change that the plan names goes into
``ACCEPTED_DIFFERENCES``, with the reason, and the test applies it to the
golden record before the comparison.

``launch/<case>.json`` holds, per arm, the argv, the child environment and
the working directory that ``execute_run`` hands to the process runner, and
the manifest that the run writes. ``runs/<name>/`` holds a run directory and
``expected_results.json``, which is the evaluation of that directory.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.axes import RequestedAxes, RunRequest
from benchmarks.e2e.parallelism import ParallelismSpec
from benchmarks.e2e.results import evaluate_run
from benchmarks.e2e.runner import execute_run
from benchmarks.execution.affinity import CpuPinning
from benchmarks.execution.paths import BENCH_DIR


GOLDEN_DIR = Path(__file__).resolve().parent / "fixtures" / "golden"
LAUNCH_DIR = GOLDEN_DIR / "launch"
RUNS_DIR = GOLDEN_DIR / "runs"
EXPECTED_RESULTS = "expected_results.json"

PINNING = CpuPinning(
    ("numactl", "--cpunodebind=1", "--membind=1"),
    "numactl --cpunodebind=1 --membind=1",
)
"""The CPU pinning of every golden launch, so the numactl prefix is on record."""

HOST_ENVIRONMENT = {
    "PATH": "/usr/bin:/bin",
    "HOME": "/home/golden",
    "BENCHMARK_CACHE_ROOT": "/golden/cache",
}
"""The parent environment of every golden launch."""

HARDWARE = "golden-h200"

METADATA = {
    "nvidia_smi": "0, NVIDIA H200, GPU-golden, 570.211.01",
    "torch_version": "2.14.0.dev20260729+cu130",
    "torchtitan_git_rev": "394510b4deb08fb549ab9aaca201906713ecd114",
    "benchmarks_git_rev": "c4737f4721f8e3fe8bb2d1e417ba5c48fed97e52",
    "megatron_git_rev": "59b72fa57f2059e858cb4bb5c094e62cc590754f",
    "te_version": "2.17.1",
    "cudnn_torch_build": "9.24.0",
    "cudnn_loader_resolves": "/usr/lib64/libcudnn.so.9.23.2",
}
"""The provenance block the stubbed host probe returns."""

LAUNCH_CASES: dict[str, dict[str, Any]] = {
    "dp1-profile": {"gpu": "0", "model_size": "1b", "profile": True},
    "dp1": {"gpu": "0", "model_size": "1b", "profile": False},
    "dp2-profile": {
        "gpu": "0,1",
        "model_size": "1b",
        "profile": True,
        "parallelism": {"dp": 2},
    },
    "dp2": {
        "gpu": "0,1",
        "model_size": "1b",
        "profile": False,
        "parallelism": {"dp": 2},
    },
    "dp1-pp2-profile": {
        "gpu": "0,1",
        "model_size": "1b",
        "profile": True,
        "batch": 4,
        "parallelism": {"pp": 2, "pp_schedule": "1F1B"},
    },
    "dp1-pp2": {
        "gpu": "0,1",
        "model_size": "1b",
        "profile": False,
        "batch": 4,
        "parallelism": {"pp": 2, "pp_schedule": "1F1B"},
    },
    "dp4-ep4-zero1-profile": {
        "gpu": "0,1,2,3",
        "model_size": "30b-a3b",
        "profile": True,
        "batch": 4,
        "parallelism": {"dp": 4, "ep": 4, "zero": 1},
    },
    "dp4-ep4-zero1": {
        "gpu": "0,1,2,3",
        "model_size": "30b-a3b",
        "profile": False,
        "batch": 4,
        "parallelism": {"dp": 4, "ep": 4, "zero": 1},
    },
    "dp1-torchtitan-arg": {
        "gpu": "0",
        "model_size": "1b",
        "profile": False,
        "torchtitan_args": ["--training.gc-freq", "50"],
    },
    "dp1-megatron-arg": {
        "gpu": "0",
        "model_size": "1b",
        "profile": False,
        "megatron_args": ["--moe-permute-fusion"],
    },
    "dp1-pp2-p2p-sync-on": {
        "gpu": "0,1",
        "model_size": "1b",
        "profile": False,
        "batch": 4,
        "parallelism": {"pp": 2, "pp_schedule": "1F1B"},
        "megatron_p2p_sync": "on",
    },
    "dp1-nan-guard-on": {
        "gpu": "0",
        "model_size": "1b",
        "profile": False,
        "megatron_nan_guard": "on",
    },
    "dp4-ep4-zero1-lean": {
        "gpu": "0,1,2,3",
        "model_size": "30b-a3b",
        "profile": False,
        "batch": 4,
        "parallelism": {"dp": 4, "ep": 4, "zero": 1},
        "megatron_precision": "lean",
    },
}
"""Every golden launch, as the operator asks for it; each one runs every arm."""


def _request(case: dict[str, Any], out_dir: Path) -> RunRequest:
    """The run request of one golden case."""
    parallelism = case.get("parallelism")
    torchtitan_args = case.get("torchtitan_args")
    megatron_args = case.get("megatron_args")
    return RunRequest(
        gpu=case["gpu"],
        scenario_name="engines",
        out_dir=out_dir,
        batch=case.get("batch"),
        torchtitan_args=None if torchtitan_args is None else tuple(torchtitan_args),
        megatron_args=None if megatron_args is None else tuple(megatron_args),
        axes=RequestedAxes(
            model_size=case["model_size"],
            profile=case["profile"],
            parallelism=None
            if parallelism is None
            else ParallelismSpec(**parallelism),
            megatron_p2p_sync=case.get("megatron_p2p_sync"),
            megatron_nan_guard=case.get("megatron_nan_guard"),
            megatron_precision=case.get("megatron_precision"),
        ),
    )


def _placeholders(out_dir: Path) -> tuple[tuple[str, str], ...]:
    """The host paths a golden record names by placeholder, longest first."""
    return (
        (str(out_dir), "<OUT>"),
        (str(Path(sys.executable).parent), "<PYTHON_BIN>"),
        (str(BENCH_DIR), "<REPO>"),
    )


def _neutral(value: Any, placeholders: tuple[tuple[str, str], ...]) -> Any:
    """``value`` with every host path replaced by its placeholder."""
    if isinstance(value, str):
        for path, placeholder in placeholders:
            value = value.replace(path, placeholder)
        return value
    if isinstance(value, list):
        return [_neutral(item, placeholders) for item in value]
    if isinstance(value, dict):
        return {key: _neutral(item, placeholders) for key, item in value.items()}
    return value


def capture_launch(case: dict[str, Any]) -> dict[str, Any]:
    """The argv, environment and manifest the current code gives one case."""
    launched: dict[str, dict[str, Any]] = {}

    def process_runner(command, *, cwd, env, stdout, **kwargs):
        launched[Path(stdout.name).stem] = {
            "argv": list(command),
            "env": dict(env),
            "cwd": str(cwd),
        }
        return SimpleNamespace(returncode=0)

    def probe(paths, gpu, label):
        return HARDWARE, {"requested_gpu": gpu, **METADATA}

    with tempfile.TemporaryDirectory() as temporary, mock.patch(
        "benchmarks.e2e.runner.hardware_metadata", side_effect=probe
    ), mock.patch(
        "benchmarks.e2e.runner.resolve_cpu_pinning", return_value=PINNING
    ), mock.patch(
        "benchmarks.e2e.runner.validate_arm"
    ):
        out_dir = Path(temporary) / "run"
        execute_run(
            _request(case, out_dir),
            process_runner=process_runner,
            environment=dict(HOST_ENVIRONMENT),
        )
        manifest = json.loads((out_dir / "manifest.json").read_text())
        return _neutral(
            {"request": case, "arms": launched, "manifest": manifest},
            _placeholders(out_dir),
        )


ACCEPTED_DIFFERENCES: tuple[Callable[[str, dict[str, Any]], None], ...] = ()
"""The changes the plan names, each applied in place to a golden launch record."""


def expected_launch(name: str) -> dict[str, Any]:
    """The golden record of one case, with every accepted difference applied."""
    record = json.loads((LAUNCH_DIR / f"{name}.json").read_text())
    for difference in ACCEPTED_DIFFERENCES:
        difference(name, record)
    return record


def evaluation(run_dir: Path) -> dict[str, Any]:
    """The evaluation of one run directory, with its own path as a placeholder."""
    result = evaluate_run(run_dir).to_dict()
    return _neutral(
        json.loads(json.dumps(result)), ((str(run_dir.resolve()), "<RUN>"),)
    )


class GoldenLaunchTest(unittest.TestCase):
    """Every golden case launches the argv and the environment it recorded."""

    def test_every_case_has_a_fixture_and_every_fixture_a_case(self) -> None:
        self.assertEqual(
            sorted(path.stem for path in LAUNCH_DIR.glob("*.json")),
            sorted(LAUNCH_CASES),
        )

    def test_every_case_matches_its_golden_record(self) -> None:
        for name, case in LAUNCH_CASES.items():
            expected = expected_launch(name)
            actual = capture_launch(case)
            with self.subTest(case=name, part="request"):
                self.assertEqual(actual["request"], expected["request"])
            with self.subTest(case=name, part="arms"):
                self.assertEqual(list(actual["arms"]), list(expected["arms"]))
            for arm, launched in expected["arms"].items():
                for part in ("argv", "env", "cwd"):
                    with self.subTest(case=name, arm=arm, part=part):
                        self.assertEqual(actual["arms"][arm][part], launched[part])
            with self.subTest(case=name, part="manifest"):
                self.assertEqual(actual["manifest"], expected["manifest"])


class GoldenEvaluationTest(unittest.TestCase):
    """Every golden run directory evaluates to its recorded results."""

    def test_every_run_directory_has_its_expected_results(self) -> None:
        runs = sorted(path for path in RUNS_DIR.iterdir() if path.is_dir())
        self.assertEqual(len(runs), 7)
        for run_dir in runs:
            with self.subTest(run=run_dir.name):
                self.assertTrue((run_dir / EXPECTED_RESULTS).is_file())

    def test_every_run_directory_evaluates_to_its_expected_results(self) -> None:
        for run_dir in sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()):
            with self.subTest(run=run_dir.name):
                expected = json.loads((run_dir / EXPECTED_RESULTS).read_text())
                self.assertEqual(evaluation(run_dir), expected)


if __name__ == "__main__":
    unittest.main()
