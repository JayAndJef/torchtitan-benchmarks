"""Golden fixtures that pin the e2e harness to its baseline behavior.

The fixtures under ``tests/fixtures/golden/`` record commit c4737f4 on
``master``, and they are the baseline of the engine refactor. Never write
them again from newer code. A change that the plan names goes into
``ACCEPTED_DIFFERENCES``, with the reason, and the test applies it to the
golden record before the comparison.

``launch/<case>.json`` holds, per arm, the argv, the child environment and
the working directory that ``execute_run`` hands to the process runner, and
the schema 18 manifest that the run wrote. The test reads that manifest
through ``upgrade_v18`` and compares it with the schema 19 manifest of the
current code. ``runs/<name>/`` holds a run directory and
``expected_results.json``, which is the evaluation of that directory.
"""

from __future__ import annotations

import json
import shlex
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.manifest_v18 import upgrade_v18
from benchmarks.e2e.axes import RequestedAxes, RunRequest
from benchmarks.e2e.overrides import parse_override
from benchmarks.e2e.parallelism import ParallelismSpec
from benchmarks.e2e.results import evaluate_run
from benchmarks.e2e.runner import execute_run
from benchmarks.execution.affinity import CpuPinning
from benchmarks.execution.launcher import LAUNCHER_KEYS
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


CASE_OVERRIDES = {
    "torchtitan_args": (
        "titan_compiled.extra_flags+={}",
        "titan_eager.extra_flags+={}",
    ),
    "megatron_args": ("megatron_stock.extra_flags+={}",),
    "megatron_p2p_sync": ("megatron_stock.p2p_sync={}",),
    "megatron_nan_guard": ("megatron_stock.nan_guard={}",),
    "megatron_precision": ("megatron_stock.precision={}",),
}
"""The ``--set`` values that replace each master option a golden case names."""


def _request(case: dict[str, Any], out_dir: Path) -> RunRequest:
    """The run request of one golden case."""
    parallelism = case.get("parallelism")
    overrides = []
    for key, templates in CASE_OVERRIDES.items():
        value = case.get(key)
        if value is None:
            continue
        text = shlex.join(value) if isinstance(value, list) else value
        overrides.extend(
            parse_override(template.format(text)) for template in templates
        )
    return RunRequest(
        gpu=case["gpu"],
        scenario_name="engines",
        out_dir=out_dir,
        batch=case.get("batch"),
        overrides=tuple(overrides),
        axes=RequestedAxes(
            model_size=case["model_size"],
            profile=case["profile"],
            parallelism=None
            if parallelism is None
            else ParallelismSpec(**parallelism),
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


def _world_size(record: dict[str, Any]) -> int:
    """The rank count of one golden launch."""
    return len(record["request"]["gpu"].split(","))


def _torchrun(world_size: int) -> list[str]:
    """The torchrun head that master gave a Megatron arm above one rank."""
    return [
        "<PYTHON_BIN>/python",
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={world_size}",
        "--rdzv-backend",
        "c10d",
        "--rdzv-endpoint",
        "localhost:0",
        "--local-ranks-filter",
        ",".join(str(rank) for rank in range(world_size)),
        "--role",
        "rank",
        "--tee",
        "3",
    ]


def _argvs(record: dict[str, Any]) -> list[tuple[str, list[str]]]:
    """Every argv of one golden launch, per arm and in the manifest."""
    return [
        *((arm, launched["argv"]) for arm, launched in record["arms"].items()),
        *record["manifest"]["commands"].items(),
    ]


def _titan_leaves_run_train_sh(record: dict[str, Any]) -> None:
    for _, argv in _argvs(record):
        if "./run_train.sh" in argv:
            at = argv.index("./run_train.sh")
            argv[at : at + 1] = [
                *_torchrun(_world_size(record)),
                "-m",
                "torchtitan.train",
            ]


def _megatron_uses_torchrun_at_one_rank(record: dict[str, Any]) -> None:
    if _world_size(record) != 1:
        return
    for _, argv in _argvs(record):
        if OLD_MEGATRON_DRIVER in argv:
            at = argv.index("<PYTHON_BIN>/python")
            argv[at : at + 1] = _torchrun(1)


def _launcher_sets_the_allocator_policy(record: dict[str, Any]) -> None:
    for launched in record["arms"].values():
        launched["env"]["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"


def _launcher_sets_rank_logging_at_one_rank(record: dict[str, Any]) -> None:
    if _world_size(record) != 1:
        return
    for launched in record["arms"].values():
        launched["env"]["LOG_RANK"] = "0"
        launched["env"]["TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE"] = "[rank${rank}]:"


def _titan_config_moves_into_the_engine_package(record: dict[str, Any]) -> None:
    for _, argv in _argvs(record):
        for index, token in enumerate(argv[:-1]):
            if token == "--module" and argv[index + 1] == OLD_TITAN_MODULE:
                argv[index + 1] = TITAN_MODULE
    workload = record["manifest"]["workload"]
    if workload["module"] == OLD_TITAN_MODULE:
        workload["module"] = TITAN_MODULE


def _megatron_driver_moves_into_the_engine_package(record: dict[str, Any]) -> None:
    for _, argv in _argvs(record):
        if OLD_MEGATRON_DRIVER in argv:
            argv[argv.index(OLD_MEGATRON_DRIVER)] = MEGATRON_DRIVER


EXECUTION_MODEL_SENTENCES = (
    " The manifest's execution_model reads plain-bf16 because it is "
    "composed from the parallelism spec; it describes the TorchTitan arms "
    "and not the Megatron one.",
    ". The manifest's execution_model says plain-bf16 and describes the "
    "other arms",
)
"""The description sentences that named the top-level execution model of schema 18."""


def _descriptions_drop_the_execution_model(record: dict[str, Any]) -> None:
    manifest = record["manifest"]
    for sentence in EXECUTION_MODEL_SENTENCES:
        manifest["description"] = manifest["description"].replace(sentence, "")
        for arm in manifest["arms"]:
            arm["description"] = arm["description"].replace(sentence, "")


OLD_MEGATRON_DRIVER = "benchmarks.e2e.megatron_stock.train"
MEGATRON_DRIVER = "benchmarks.e2e.engines.megatron_stock.driver.train"
OLD_TITAN_MODULE = "benchmarks.models.piper_qwen3"
TITAN_MODULE = "benchmarks.e2e.engines.torchtitan.plugins"


@dataclass(frozen=True)
class AcceptedDifference:
    """One change from the baseline that the plan names, and its reason."""

    reason: str
    apply: Callable[[dict[str, Any]], None] | None
    """Changes a golden launch record in place; ``None`` when the record cannot show it."""


ACCEPTED_DIFFERENCES = (
    AcceptedDifference(
        "Plan 2.6: a TorchTitan arm starts torchrun with the Megatron flags, "
        "then '-m torchtitan.train', in place of './run_train.sh'. The "
        "trainer arguments after it do not change. The shell trace of "
        "run_train.sh leaves the log.",
        _titan_leaves_run_train_sh,
    ),
    AcceptedDifference(
        "Plan 2.6: a per_rank launch uses torchrun at every world size, so "
        "the Megatron arm at one rank gains torchrun and the '[rank0]:' log "
        "prefix.",
        _megatron_uses_torchrun_at_one_rank,
    ),
    AcceptedDifference(
        "The launcher sets PYTORCH_ALLOC_CONF for every arm. run_train.sh "
        "set the same value for TorchTitan, and the Megatron bootstrap set "
        "it only when the host had no value; the launcher value now "
        "replaces a host value for Megatron too.",
        _launcher_sets_the_allocator_policy,
    ),
    AcceptedDifference(
        "The launcher sets LOG_RANK and the torchrun prefix template with "
        "torchrun at every world size. At one rank LOG_RANK is 0, which "
        "run_train.sh exported to the TorchTitan trainer, and the template "
        "gives the '[rank0]:' prefix that torchrun gave by default.",
        _launcher_sets_rank_logging_at_one_rank,
    ),
    AcceptedDifference(
        "run_train.sh exported TORCHFT_LIGHTHOUSE to the TorchTitan trainer. "
        "Only run_train.sh names it in the fork, and torchft is not "
        "installed, so no process reads it.",
        None,
    ),
    AcceptedDifference(
        "run_train.sh put '--module llama3 --config llama3_debugmodel' in "
        "front of the harness arguments. The fork's ConfigManager._load_config "
        "keeps the last --module and --config, and it imports the module "
        "only after the loop, so the harness values always won and "
        "llama3 was never loaded.",
        None,
    ),
    AcceptedDifference(
        "Plan 4: the TorchTitan config registry moves into the TorchTitan "
        "engine package, so '--module' and the recorded workload module "
        "name benchmarks.e2e.engines.torchtitan.plugins. The fork imports "
        "'<module>.config_registry' first, so the trainer finds the same "
        "config functions.",
        _titan_config_moves_into_the_engine_package,
    ),
    AcceptedDifference(
        "Plan 4: the stock Megatron driver moves into the driver package of "
        "the Megatron engine, so the '-m' target of the Megatron arm is "
        "benchmarks.e2e.engines.megatron_stock.driver.train.",
        _megatron_driver_moves_into_the_engine_package,
    ),
    AcceptedDifference(
        "Plan 2.9, refined in section C: each arm records its own execution "
        "model, so the scenario description and the Megatron arm description "
        "drop the sentence about the top-level execution model.",
        _descriptions_drop_the_execution_model,
    ),
)
"""The changes from the baseline that the plan names or that the review accepted."""


def _schema_19_facts(
    expected: dict[str, Any], actual: dict[str, Any], launched: dict[str, Any]
) -> None:
    """Copy into ``expected`` the two arm facts that schema 18 did not record.

    Plan 2.9 adds ``env_delta``: it must hold the launcher's keys, each with
    the value that the golden launch gave the child. Section C gives every arm an
    execution model, and schema 18 recorded one for the TorchTitan arms
    alone; ``tests/test_engines.py`` pins the Megatron strings.
    """
    actual_arms = {arm["name"]: arm for arm in actual["arms"]}
    for arm in expected["arms"]:
        found = actual_arms.get(arm["name"])
        if found is None:
            continue
        delta = found["env_delta"]
        if set(delta) != LAUNCHER_KEYS:
            raise AssertionError(
                f"{arm['name']}: env_delta holds {sorted(delta)}, and every "
                f"launch of both engines sets {sorted(LAUNCHER_KEYS)} alone"
            )
        env = launched[arm["name"]]["env"]
        wrong = {key: value for key, value in delta.items() if env.get(key) != value}
        if wrong:
            raise AssertionError(
                f"{arm['name']}: env_delta holds {wrong}, which the golden "
                "launch did not give the child"
            )
        arm["env_delta"] = delta
        if arm["execution_model"] is None:
            arm["execution_model"] = found["execution_model"]


def expected_launch(name: str) -> dict[str, Any]:
    """The golden record of one case, with every accepted difference applied."""
    record = json.loads((LAUNCH_DIR / f"{name}.json").read_text())
    for difference in ACCEPTED_DIFFERENCES:
        if difference.apply is not None:
            difference.apply(record)
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
                manifest = upgrade_v18(expected["manifest"])
                _schema_19_facts(manifest, actual["manifest"], expected["arms"])
                self.assertEqual(actual["manifest"], manifest)


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
