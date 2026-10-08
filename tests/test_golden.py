"""Golden fixtures that pin the e2e harness to its baseline behavior.

The fixtures under ``tests/fixtures/golden/`` record commit c4737f4 on
``master``, and they are the baseline of the engine refactor. Never write
them again from newer code. A change that the plan names goes into
``ACCEPTED_DIFFERENCES``, with the reason, and the test applies it to the
golden record before the comparison.

``launch/<case>.json`` holds, per arm, the argv, the child environment and
the working directory that ``execute_run`` hands to the process runner, and
the schema 18 manifest that the run wrote. The test reads that manifest
through ``current_manifest`` and compares it with the schema 20 manifest of
the current code. ``runs/<name>/`` holds a run directory and
``expected_results.json``, which is the evaluation of that directory.
"""

from __future__ import annotations

import json
import math
import shlex
import statistics
import sys
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.manifests import current_manifest, load_run_record
from benchmarks.e2e.axes import RequestedAxes, RunRequest
from benchmarks.e2e.overrides import parse_override
from benchmarks.e2e.parallelism import ParallelismSpec
from benchmarks.e2e.results import RESULTS_SCHEMA_VERSION, evaluate_run
from benchmarks.e2e.runner import check_request, execute_run
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
    "cudnn_loader_resolves": "92400 /venv/lib/python3.10/site-packages/nvidia/cudnn/lib",
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
            check_request(
                _request(case, out_dir),
                environment=dict(HOST_ENVIRONMENT),
            ),
            process_runner=process_runner,
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


def _torchrun_runs_unbuffered(record: dict[str, Any]) -> None:
    for _, argv in _argvs(record):
        if "torch.distributed.run" in argv:
            argv.insert(argv.index("torch.distributed.run") - 1, "-u")


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


ENGINE_BY_ARM = {
    "titan_compiled": "torchtitan",
    "titan_eager": "torchtitan",
    "megatron_stock": "megatron_stock",
}
"""The engine of each golden arm, which names its extras."""

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
    AcceptedDifference(
        "The launcher starts torchrun with '-u'. The tee threads of torchrun "
        "shared one buffered stream, and a thread race in it lost lines of "
        "the arm log and wrote NUL bytes in their place.",
        _torchrun_runs_unbuffered,
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


@dataclass(frozen=True)
class AcceptedEvaluationDifference:
    """One change from the baseline evaluation that the plan names, and its reason."""

    reason: str
    apply: Callable[[Path, dict[str, Any]], None]
    """Changes the golden results of one run directory in place."""


def _logged_tps(step_ms: float, *, tokens_per_step: int, pp: int) -> int:
    """The whole tokens/s that a step line logged, recovered from its step cost; a value that does not recover exactly raises."""
    tps = round(1000.0 * tokens_per_step / (step_ms * pp))
    if 1000.0 * tokens_per_step / (tps * pp) != step_ms:
        raise AssertionError(f"the step cost {step_ms} holds no whole tokens/s")
    return tps


def _schema_6_step_ms(series: list[float]) -> dict[str, Any]:
    """The schema 6 step cost of one series: the mean, the median and the nearest-rank p95."""
    return {
        "mean": statistics.fmean(series),
        "median": statistics.median(series),
        "p95": sorted(series)[max(math.ceil(0.95 * len(series)), 1) - 1],
        "series": series,
    }


def _profiled_rule_drops_step_2(run_dir: Path, record: dict[str, Any]) -> None:
    run = load_run_record(run_dir).run
    if not run.profile:
        return
    tokens_per_step = run.data.local_batch_size * run.data.seq_len
    for result in record["results"].values():
        for row in result["per_rank"]:
            # The series keeps step order, so step 2 is its first value.
            _, *series = row["step_ms"]["series"]
            row["stable_tokens_per_second"] = statistics.median(
                _logged_tps(ms, tokens_per_step=tokens_per_step, pp=run.parallelism.pp)
                for ms in series
            )
            row["stable_sample_count"] = len(series)
            row["step_ms"] = _schema_6_step_ms(series)
        published = min(
            result["per_rank"],
            key=lambda row: (row["stable_tokens_per_second"], row["rank"]),
        )
        result["stable_tokens_per_second"] = published["stable_tokens_per_second"]
        result["stable_sample_count"] = published["stable_sample_count"]
        result["step_ms"] = published["step_ms"]
        result["published_rank"] = published["rank"]


ACCEPTED_EVALUATION_DIFFERENCES = (
    AcceptedEvaluationDifference(
        "Plan A.3: the profiled rule drops step 2, the first sample of each "
        "rank, which runs slow in every arm. Each rank of a profiled run "
        "loses its first sample, and its figures and the published rank "
        "follow from the samples that remain.",
        _profiled_rule_drops_step_2,
    ),
)
"""The changes from the baseline evaluation that the plan names."""


@dataclass(frozen=True)
class AcceptedRefusal:
    """A golden run directory that the current code refuses to evaluate, and its reason."""

    reason: str
    arm: str
    rank: int
    """The rank that lacks a sampled step."""
    error: str
    """The start of the refusal."""


ACCEPTED_REFUSALS = {
    "dp4-ep4-titan-compiled-profile": AcceptedRefusal(
        "Plan A.4: the evaluation refuses an arm when a rank lacks a sampled "
        "step. A log line of NUL and other bytes lost step 42 of rank 3 "
        "with no warning, and the baseline published rank 3 from one sample "
        "fewer than the others.",
        arm="titan_compiled",
        rank=3,
        error="titan_compiled: rank 3 lacks sampled step 42; ",
    ),
}
"""The golden run directories that the plan refuses, by name."""


def expected_evaluation(run_dir: Path) -> dict[str, Any]:
    """The golden results of one run directory, with every accepted evaluation difference applied."""
    record = json.loads((run_dir / EXPECTED_RESULTS).read_text())
    for difference in ACCEPTED_EVALUATION_DIFFERENCES:
        difference.apply(run_dir, record)
    return record


def schema_6_view(actual: dict[str, Any]) -> dict[str, Any]:
    """The schema 6 record inside a schema 7 evaluation.

    Plan A.5 to A.7 regroup the figures: schema 7 publishes each statistic
    at its own slowest rank and keeps each rank's sampled steps. Schema 6
    published every figure at the rank with the lowest median tokens/s,
    which is the schema 7 ``median_rank`` of tokens/s, and it kept each
    rank's step costs as a bare series. The view drops every figure that
    schema 6 did not have; ``test_every_schema_7_figure_follows_from_the_steps``
    checks those.
    """
    if actual["schema_version"] != RESULTS_SCHEMA_VERSION:
        raise AssertionError(f"the evaluation records schema {actual['schema_version']}")
    results = {}
    for arm, result in actual["results"].items():
        rows = {
            row["rank"]: {
                "rank": row["rank"],
                "stable_tokens_per_second": row["tokens_per_second"]["median"],
                "stable_sample_count": len(row["steps"]),
                "step_ms": {
                    "mean": row["step_ms"]["mean"],
                    "median": row["step_ms"]["median"],
                    "p95": row["step_ms"]["p95"],
                    "series": [step["step_ms"] for step in row["steps"]],
                },
            }
            for row in result["per_rank"]
        }
        published = result["tokens_per_second"]["median_rank"]
        results[arm] = {
            "stable_tokens_per_second": result["tokens_per_second"]["median"],
            "stable_sample_count": result["sample_count"],
            "peak_memory_gib": result["peak_memory_gib"]["max"],
            "step_ms": rows[published]["step_ms"],
            "rank_reduction": "min_over_ranks",
            "published_rank": published,
            "per_rank": list(rows.values()),
        }
    return {**actual, "schema_version": 6, "results": results}


def sampled_steps_by_hand(run_dir: Path) -> list[int]:
    """The steps that the sample rule of the run takes, restated from the plan: steps 3 to 10 of each 20-step cycle, or every step after the warmup."""
    run = load_run_record(run_dir).run
    steps = range(1, run.data.steps + 1)
    if not run.profile:
        return [step for step in steps if step > run.warmup_steps]
    wait = run.window.freq - run.window.warmup - run.window.active
    return [
        step
        for step in steps
        if 2 <= (step - 1) % run.window.freq + 1 <= wait and step != 2
    ]


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
                manifest = current_manifest(expected["manifest"])
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
        """The schema 6 view of the evaluation matches the golden text byte for byte, with every accepted difference applied."""
        for run_dir in sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()):
            if run_dir.name in ACCEPTED_REFUSALS:
                continue
            with self.subTest(run=run_dir.name):
                actual = schema_6_view(evaluation(run_dir))
                self.assertEqual(
                    json.dumps(actual, indent=2, allow_nan=False) + "\n",
                    json.dumps(expected_evaluation(run_dir), indent=2, allow_nan=False)
                    + "\n",
                )

    def test_every_schema_7_figure_follows_from_the_steps(self) -> None:
        """Each rank holds the sampled steps of the plan's rule, each statistic follows from them, and each arm statistic is at its worst rank."""
        for run_dir in sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()):
            if run_dir.name in ACCEPTED_REFUSALS:
                continue
            run = load_run_record(run_dir).run
            tokens_per_step = run.data.local_batch_size * run.data.seq_len
            expected_steps = sampled_steps_by_hand(run_dir)
            for arm, result in evaluation(run_dir)["results"].items():
                with self.subTest(run=run_dir.name, arm=arm):
                    self._check_arm(
                        result,
                        engine=ENGINE_BY_ARM[arm],
                        expected_steps=expected_steps,
                        tokens_per_step=tokens_per_step,
                        pp=run.parallelism.pp,
                    )

    def _check_arm(
        self,
        result: dict[str, Any],
        *,
        engine: str,
        expected_steps: list[int],
        tokens_per_step: int,
        pp: int,
    ) -> None:
        self.assertEqual(result["sample_count"], len(expected_steps))
        self.assertEqual(result["rank_reduction"], "slowest_rank_per_statistic")
        by_rank = {row["rank"]: row for row in result["per_rank"]}
        for row in result["per_rank"]:
            steps = row["steps"]
            self.assertEqual([step["step"] for step in steps], expected_steps)
            for step in steps:
                self.assertEqual(
                    step["step_ms"],
                    1000.0 * tokens_per_step / (step["tokens_per_second"] * pp),
                )
            rates = [step["tokens_per_second"] for step in steps]
            self.assertEqual(row["tokens_per_second"]["median"], statistics.median(rates))
            self.assertAlmostEqual(
                row["tokens_per_second"]["mean"],
                len(rates) / sum(1 / rate for rate in rates),
                places=6,
            )
            memory = [step["peak_memory_gib"] for step in steps]
            self.assertEqual(row["peak_memory_gib"]["median"], statistics.median(memory))
            self.assertAlmostEqual(row["peak_memory_gib"]["mean"], sum(memory) / len(memory))
            self.assertGreaterEqual(row["peak_memory_gib"]["max"], max(memory))
            self.assertEqual(sorted(row["extras"]), ["mfu", "tflops"])
            for name, figure in row["extras"].items():
                values = [step["extras"][name] for step in steps]
                self.assertEqual(figure["median"], statistics.median(values))
                self.assertAlmostEqual(
                    figure["mean"], len(values) / sum(1 / value for value in values)
                )
        for figure, highest, names in (
            ("tokens_per_second", False, ("median", "mean")),
            ("step_ms", True, ("median", "mean", "p95")),
            ("peak_memory_gib", True, ("max", "median", "mean")),
        ):
            for name in names:
                values = {rank: row[figure][name] for rank, row in by_rank.items()}
                worst = max(values.values()) if highest else min(values.values())
                rank = result[figure][f"{name}_rank"]
                self.assertEqual(result[figure][name], worst, f"{figure} {name}")
                self.assertEqual(values[rank], worst, f"{figure} {name}_rank")
                self.assertEqual(rank, min(r for r, v in values.items() if v == worst))
        self.assertEqual(list(result["extras"]), [engine])
        for name, figure in result["extras"][engine].items():
            for statistic in ("median", "mean"):
                values = {
                    rank: row["extras"][name][statistic] for rank, row in by_rank.items()
                }
                self.assertEqual(figure[statistic], min(values.values()))
                self.assertEqual(
                    values[figure[f"{statistic}_rank"]], min(values.values())
                )

    def test_every_accepted_refusal_is_refused_and_on_record(self) -> None:
        """The golden record of a refused directory shows the lost sample: the rank holds one sample fewer than each other rank."""
        for name, refusal in ACCEPTED_REFUSALS.items():
            run_dir = RUNS_DIR / name
            with self.subTest(run=name):
                with self.assertRaises(ValueError) as caught:
                    evaluate_run(run_dir)
                self.assertTrue(
                    str(caught.exception).startswith(refusal.error),
                    str(caught.exception),
                )
                golden = json.loads((run_dir / EXPECTED_RESULTS).read_text())
                counts = {
                    row["rank"]: row["stable_sample_count"]
                    for row in golden["results"][refusal.arm]["per_rank"]
                }
                others = {
                    count for rank, count in counts.items() if rank != refusal.rank
                }
                self.assertEqual(others, {counts[refusal.rank] + 1})

    def test_an_unchanged_record_writes_the_golden_text(self) -> None:
        """The comparison above dumps the golden record again, so a dump with no difference applied must give the file's own text."""
        for run_dir in sorted(path for path in RUNS_DIR.iterdir() if path.is_dir()):
            with self.subTest(run=run_dir.name):
                text = (run_dir / EXPECTED_RESULTS).read_text()
                self.assertEqual(
                    json.dumps(json.loads(text), indent=2, allow_nan=False) + "\n",
                    text,
                )


if __name__ == "__main__":
    unittest.main()
