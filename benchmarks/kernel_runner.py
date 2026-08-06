"""Orchestrate kernel-isolation benchmark scenarios from the torch-free CLI.

Mirrors ``runner.py``: the parent resolves provenance and pinning, writes the
manifest, and spawns one pinned GPU worker per scenario so CUDA device
selection, NUMA binding, and the TE build's compiler environment all apply to
the measuring process. Scenarios are independent short runs, so unlike the
e2e sweep the parent continues past a failing scenario and reports all
outcomes.
"""

from __future__ import annotations

import datetime as dt
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping

from benchmarks.artifacts import atomic_write_json
from benchmarks.kernel_results import (
    KernelScenarioResult,
    load_kernel_results,
)
from benchmarks.kernels import (
    KernelScenario,
    Piper1BSpec,
    kernel_scenario_by_name,
    shape_summary,
    spec_with_overrides,
)
from benchmarks.runner import BENCH_DIR, EventHandler, RunEvent, run_timestamp
from benchmarks.runtime import (
    RuntimePaths,
    add_compiler_environment,
    hardware_metadata,
    resolve_cpu_pinning,
    runtime_environment,
)


KERNEL_MANIFEST_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class KernelRunRequest:
    gpu: str
    scenario_names: tuple[str, ...]
    n: int = 200
    warmup: int = 30
    burst: bool = False
    batch: int | None = None
    seq_len: int | None = None
    seed: int = 0
    hardware: str = "auto"
    out_dir: Path | None = None
    timestamp: str | None = None
    cache_root: Path | None = None
    compiler_env: Path | None = None


@dataclass(frozen=True)
class KernelScenarioOutcome:
    scenario: str
    out_dir: Path
    result: KernelScenarioResult | None
    correctness_failed: bool = False
    error: str | None = None

    @property
    def failed(self) -> bool:
        return self.correctness_failed or self.error is not None


def _emit(handler: EventHandler | None, kind: str, message: str) -> None:
    if handler is not None:
        handler(RunEvent(kind=kind, message=message))


def kernel_manifest_data(
    scenario: KernelScenario,
    spec: Piper1BSpec,
    request: KernelRunRequest,
    command: list[str],
    hardware: str,
    metadata: dict[str, str],
) -> dict:
    return {
        "schema_version": KERNEL_MANIFEST_SCHEMA_VERSION,
        "kind": "kernel",
        "scenario": scenario.name,
        "description": scenario.description,
        "spec": asdict(spec),
        "shapes": shape_summary(scenario.name, spec),
        "arms": [asdict(arm) for arm in scenario.arms],
        "baseline_arm": scenario.baseline_arm,
        "n": request.n,
        "warmup": request.warmup,
        "burst": request.burst,
        "seed": request.seed,
        "command": command,
        "hardware": hardware,
        "hardware_metadata": metadata,
        "created_at": dt.datetime.now(dt.timezone.utc).strftime("%FT%TZ"),
    }


def _log_tail(log_path: Path, lines: int = 12) -> str:
    try:
        content = log_path.read_text(errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join(content[-lines:])


def execute_kernel_run(
    request: KernelRunRequest,
    *,
    event_handler: EventHandler | None = None,
    process_runner=subprocess.run,
    environment: Mapping[str, str] | None = None,
) -> tuple[KernelScenarioOutcome, ...]:
    host_environment = dict(environment or os.environ)
    paths = RuntimePaths.resolve(
        cache_root=request.cache_root,
        compiler_env=request.compiler_env,
        environment=host_environment,
    )
    spec = spec_with_overrides(batch=request.batch, seq_len=request.seq_len)
    hardware, metadata = hardware_metadata(paths, request.gpu, request.hardware)
    pinning = resolve_cpu_pinning(request.gpu)
    metadata = {**metadata, "cpu_pinning": pinning.description}
    timestamp = request.timestamp or run_timestamp()
    base_environment = runtime_environment(
        paths, request.gpu, environment=host_environment
    )

    _emit(event_handler, "summary", f"GPU (PCI index): {request.gpu}")
    _emit(event_handler, "summary", metadata["nvidia_smi"])
    _emit(event_handler, "summary", f"cpu pinning: {pinning.description}")

    outcomes = []
    for name in request.scenario_names:
        scenario = kernel_scenario_by_name(name)
        out_dir = request.out_dir or (
            BENCH_DIR / "out" / timestamp / "kernels" / name / hardware
        )
        out_dir = out_dir.expanduser().resolve()
        _emit(event_handler, "arm", f"=== kernel scenario: {name} ===")

        if scenario.requires_gcc_toolset and paths.compiler_env is None:
            outcomes.append(
                KernelScenarioOutcome(
                    scenario=name,
                    out_dir=out_dir,
                    result=None,
                    error=(
                        f"{name}: needs a C++20 host compiler for the TE "
                        f"build; set --compiler-env/BENCH_COMPILER_ENV"
                    ),
                )
            )
            continue

        out_dir.mkdir(parents=True, exist_ok=False)
        command = list(pinning.prefix) + [
            sys.executable,
            "-m",
            "benchmarks.kernel_worker",
            "--scenario",
            name,
            "--out-dir",
            str(out_dir),
            "--hardware",
            hardware,
            "--n",
            str(request.n),
            "--warmup",
            str(request.warmup),
            "--seed",
            str(request.seed),
        ]
        if request.burst:
            command.append("--burst")
        if request.batch is not None:
            command.extend(("--batch", str(request.batch)))
        if request.seq_len is not None:
            command.extend(("--seq-len", str(request.seq_len)))

        atomic_write_json(
            out_dir / "manifest.json",
            kernel_manifest_data(
                scenario, spec, request, command, hardware, metadata
            ),
        )
        scenario_environment = base_environment
        if scenario.requires_gcc_toolset:
            # Containment: a broken compiler environment is this scenario's
            # failure, not grounds for abandoning the ones that do not need it.
            try:
                scenario_environment = add_compiler_environment(
                    base_environment, paths.compiler_env
                )
            except (OSError, ValueError, subprocess.SubprocessError) as error:
                outcome = KernelScenarioOutcome(
                    scenario=name,
                    out_dir=out_dir,
                    result=None,
                    error=f"{name}: cannot prepare the compiler environment: {error}",
                )
                _emit(event_handler, "error", f"ERROR {outcome.error}")
                outcomes.append(outcome)
                continue

        log_path = out_dir / "kernel_bench.log"
        _emit(event_handler, "command", " ".join(command))
        with log_path.open("w") as log:
            completed = process_runner(
                command,
                cwd=paths.bench_dir,
                env=scenario_environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )

        result = None
        error = None
        results_path = out_dir / "results.json"
        if results_path.exists():
            try:
                result = load_kernel_results(results_path)
            except ValueError as load_error:
                error = str(load_error)
        if completed.returncode not in (0, 3) and error is None:
            error = (
                f"{name}: worker exited with {completed.returncode}; "
                f"log tail:\n{_log_tail(log_path)}"
            )
        elif result is None and error is None:
            error = f"{name}: worker wrote no results; see {log_path}"

        outcome = KernelScenarioOutcome(
            scenario=name,
            out_dir=out_dir,
            result=result,
            correctness_failed=completed.returncode == 3,
            error=error,
        )
        # Report a failure the moment it happens. Scenarios continue past one
        # another, so holding this until the end would show a first-scenario
        # failure only after every later scenario had run.
        if outcome.error:
            _emit(event_handler, "error", f"ERROR {name}: {outcome.error}")
        elif outcome.correctness_failed:
            _emit(
                event_handler,
                "error",
                f"ERROR {name}: correctness gates failed; see the report below",
            )
        outcomes.append(outcome)
    return tuple(outcomes)
