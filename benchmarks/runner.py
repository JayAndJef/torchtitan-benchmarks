"""Run declarative torchtitan benchmark scenarios and validate their artifacts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
from dataclasses import asdict, replace
from pathlib import Path

from benchmarks.scenarios import Arm, Scenario, Workload, scenario_by_name


BENCH_DIR = Path(__file__).resolve().parent.parent
TITAN_DIR = Path(os.environ.get("TITAN_DIR", "/data/zejiaqi/torchtitan"))
DEFAULT_CACHE_ROOT = Path("/data/zejiaqi/tmp")


def parse_args(argv: list[str] | None = None) -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Run a declarative torchtitan benchmark scenario."
    )
    parser.add_argument(
        "gpu", nargs="?", help="PCI-order GPU index passed to CUDA_VISIBLE_DEVICES."
    )
    parser.add_argument("--scenario", default="piper1b_rope", help="Scenario name.")
    parser.add_argument("--arm", help="Run one arm instead of every arm in the scenario.")
    parser.add_argument(
        "--hardware",
        default="auto",
        help="Hardware label for output and provenance. 'auto' derives one from nvidia-smi.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Output directory. Defaults to out/<UTC timestamp>/<scenario>/<hardware>.",
    )
    parser.add_argument(
        "--list-scenarios", action="store_true", help="List scenario names and exit."
    )
    parser.add_argument("--seq-len", type=int, help="Override the scenario sequence length.")
    parser.add_argument("--steps", type=int, help="Override the scenario step count.")
    parser.add_argument("--batch", type=int, help="Override the scenario local batch size.")
    return parser.parse_known_args(argv)


def command_for_arm(
    workload: Workload, arm: Arm, arm_dir: Path, extra_args: list[str]
) -> list[str]:
    """Build the torchtitan command shared by all scenarios."""
    args = [
        "./run_train.sh",
        "--module",
        workload.module,
        "--config",
        arm.config or workload.config,
        "--training.seq-len",
        str(workload.seq_len),
        "--training.steps",
        str(workload.steps),
        "--training.local-batch-size",
        str(workload.local_batch_size),
        "--compile.enable",
        "--profiler.enable_profiling",
        "--profiler.profile_freq",
        str(workload.profile_freq),
        "--profiler.profiler_active",
        str(workload.profiler_active),
        "--profiler.profiler_warmup",
        str(workload.profiler_warmup),
    ]
    if workload.seed is not None:
        args.extend(("--debug.seed", str(workload.seed)))
    if arm.override_imports:
        args.extend(("--override.imports", ",".join(arm.override_imports)))
    return args + extra_args + ["--dump-folder", str(arm_dir)]


def _run_text(command: list[str], *, cwd: Path | None = None) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT, cwd=cwd)
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def hardware_metadata(gpu: str, hardware_label: str) -> tuple[str, dict[str, str]]:
    query = _run_text(
        [
            "nvidia-smi",
            "--id=" + gpu,
            "--query-gpu=index,name,uuid,driver_version",
            "--format=csv,noheader",
        ]
    ).strip()
    metadata = {
        "requested_gpu": gpu,
        "nvidia_smi": query,
        "torch_version": _run_text(
            [str(TITAN_DIR / ".venv/bin/python"), "-c", "import torch; print(torch.__version__)"],
            cwd=TITAN_DIR,
        ).strip(),
        "torchtitan_git_rev": _run_text(["git", "rev-parse", "HEAD"], cwd=TITAN_DIR).strip(),
        "benchmarks_git_rev": _run_text(["git", "rev-parse", "HEAD"], cwd=BENCH_DIR).strip(),
    }
    if hardware_label != "auto":
        return hardware_label, metadata
    name = query.split(",")[1].strip() if "," in query else f"gpu{gpu}"
    return re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower(), metadata


def output_dir(args: argparse.Namespace, scenario: Scenario, hardware: str) -> Path:
    if args.out is not None:
        return args.out.resolve()
    if env_out := os.environ.get("OUT"):
        return Path(env_out).resolve()
    timestamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return BENCH_DIR / "out" / timestamp / scenario.name / hardware


def workload_from_args(args: argparse.Namespace, scenario: Scenario) -> Workload:
    """Apply portable workload-size overrides without changing a scenario's arms."""
    workload = scenario.workload
    seq_len = args.seq_len if args.seq_len is not None else os.environ.get("SEQ")
    steps = args.steps if args.steps is not None else os.environ.get("STEPS")
    batch = args.batch if args.batch is not None else os.environ.get("BATCH")
    if seq_len is not None:
        workload = replace(workload, seq_len=int(seq_len))
    if steps is not None:
        workload = replace(workload, steps=int(steps))
    if batch is not None:
        workload = replace(workload, local_batch_size=int(batch))
    if workload.steps < workload.profile_freq * workload.min_trace_windows:
        raise ValueError(
            f"steps ({workload.steps}) must be at least "
            f"{workload.profile_freq * workload.min_trace_windows} to collect "
            f"{workload.min_trace_windows} profiler windows"
        )
    return workload


def runtime_environment(gpu: str) -> dict[str, str]:
    environment = os.environ.copy()
    pythonpath = environment.get("PYTHONPATH")
    environment.update(
        {
            "CUDA_DEVICE_ORDER": "PCI_BUS_ID",
            "CUDA_VISIBLE_DEVICES": gpu,
            "PYTHONPATH": f"{BENCH_DIR}{':' + pythonpath if pythonpath else ''}",
            "PATH": f"{TITAN_DIR / '.venv/bin'}:{environment['PATH']}",
            "TORCH_EXTENSIONS_DIR": environment.get(
                "TORCH_EXTENSIONS_DIR", str(DEFAULT_CACHE_ROOT / "torch_extensions")
            ),
            "TORCHINDUCTOR_CACHE_DIR": environment.get(
                "TORCHINDUCTOR_CACHE_DIR", str(DEFAULT_CACHE_ROOT / "inductor_cache")
            ),
            "TRITON_CACHE_DIR": environment.get(
                "TRITON_CACHE_DIR", str(DEFAULT_CACHE_ROOT / "triton_cache")
            ),
            "NGPU": "1",
        }
    )
    return environment


def add_gcc_toolset(environment: dict[str, str]) -> None:
    """Load the optional host compiler environment needed by the legacy TE arm."""
    toolset = Path("/opt/rh/gcc-toolset-13/enable")
    if not toolset.exists():
        return
    # Source the enable script on top of the runner's environment. A plain -c
    # shell is required: a login shell would rebuild PATH from the user's
    # profile (e.g. conda) and shadow the torchtitan venv python.
    output = subprocess.check_output(
        ["bash", "-c", f"source {toolset} && env -0"], env=environment
    )
    for entry in output.decode().split("\0"):
        if not entry:
            continue
        key, value = entry.split("=", 1)
        environment[key] = value


def trace_files(arm_dir: Path) -> list[Path]:
    return sorted(arm_dir.glob("profiling/traces*/iteration_*/rank0_trace.json.gz"))


def validate_arm(arm: Arm, arm_dir: Path, log_path: Path, workload: Workload) -> None:
    """Reject partial or wrongly configured runs before they reach analysis."""
    log = log_path.read_text(errors="replace")
    if "Training completed" not in log:
        raise RuntimeError(f"{arm.name}: training did not complete; see {log_path}")
    if arm.expected_override_count:
        override_count = len(re.findall(r"\[Override\]", log))
        if override_count != arm.expected_override_count:
            raise RuntimeError(
                f"{arm.name}: expected {arm.expected_override_count} override applications, "
                f"found {override_count}; see {log_path}"
            )
        for override_import in arm.override_imports:
            # torchtitan logs each application as "[Override] <import path>: <fqn> ..."
            if f"[Override] {override_import}:" not in log:
                raise RuntimeError(
                    f"{arm.name}: override {override_import!r} did not apply; see {log_path}"
                )
    if "falling back to the PyTorch" in log:
        raise RuntimeError(f"{arm.name}: override fell back to PyTorch; see {log_path}")
    traces = trace_files(arm_dir)
    if len(traces) < workload.min_trace_windows:
        raise RuntimeError(
            f"{arm.name}: expected at least {workload.min_trace_windows} profiler windows, "
            f"found {len(traces)} under {arm_dir}"
        )
    for marker in arm.trace_kernel_markers:
        if not any(marker in _run_text(["zgrep", "-F", marker, str(path)]) for path in traces):
            raise RuntimeError(
                f"{arm.name}: marker kernel {marker!r} absent from profiler traces"
            )


def write_manifest(
    out_dir: Path,
    scenario: Scenario,
    selected_arms: tuple[Arm, ...],
    commands: dict[str, list[str]],
    hardware: str,
    metadata: dict[str, str],
    extra_args: list[str],
) -> None:
    manifest = {
        "schema_version": 4,
        "scenario": scenario.name,
        "description": scenario.description,
        "hardware": hardware,
        "hardware_metadata": metadata,
        "workload": asdict(scenario.workload),
        "regions": [asdict(region) for region in scenario.regions],
        "arms": [asdict(arm) for arm in scenario.arms],
        "selected_arms": [arm.name for arm in selected_arms],
        "commands": commands,
        "extra_torchtitan_args": extra_args,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


def run(args: argparse.Namespace, extra_args: list[str]) -> Path:
    if extra_args[:1] == ["--"]:
        extra_args = extra_args[1:]
    if args.gpu is None:
        raise ValueError("gpu is required unless --list-scenarios is used")
    scenario = scenario_by_name(args.scenario)
    workload = workload_from_args(args, scenario)
    arms = (scenario.arm(args.arm),) if args.arm else scenario.arms
    hardware, metadata = hardware_metadata(args.gpu, args.hardware)
    out_dir = output_dir(args, scenario, hardware)
    out_dir.mkdir(parents=True, exist_ok=False)
    scenario = replace(scenario, workload=workload)
    commands = {
        arm.name: command_for_arm(scenario.workload, arm, out_dir / arm.name, extra_args)
        for arm in arms
    }
    write_manifest(out_dir, scenario, arms, commands, hardware, metadata, extra_args)

    print(f"GPU (PCI index): {args.gpu}")
    print(metadata["nvidia_smi"])
    print(f"scenario: {scenario.name}   hardware: {hardware}")
    print(f"arms: {' '.join(arm.name for arm in arms)}")
    print(f"output: {out_dir}")

    environment = runtime_environment(args.gpu)
    for arm in arms:
        if arm.requires_gcc_toolset:
            add_gcc_toolset(environment)

        arm_dir = out_dir / arm.name
        command = commands[arm.name]
        log_path = out_dir / f"{arm.name}.log"
        print(f"\n=== arm: {arm.name} ===")
        print(" ".join(command))
        with log_path.open("w") as log:
            log.write(
                f"# scenario={scenario.name} arm={arm.name} "
                f"gpu_pci_index={args.gpu} {dt.datetime.now(dt.timezone.utc):%FT%TZ}\n"
            )
            log.write(metadata["nvidia_smi"] + "\n")
            log.flush()
            completed = subprocess.run(
                command,
                cwd=TITAN_DIR,
                env=environment,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode:
            raise RuntimeError(
                f"{arm.name}: training exited with {completed.returncode}; see {log_path}"
            )
        validate_arm(arm, arm_dir, log_path, scenario.workload)
        print(f"{arm.name}: validated")
    return out_dir


def main(argv: list[str] | None = None) -> None:
    args, extra_args = parse_args(argv)
    if args.list_scenarios:
        from benchmarks.scenarios import SCENARIOS

        for scenario in SCENARIOS.values():
            print(f"{scenario.name}: {scenario.description}")
        return
    try:
        out_dir = run(args, extra_args)
    except (ValueError, RuntimeError) as error:
        raise SystemExit(f"FATAL: {error}") from error
    print("\nAll arms validated. Analyze with:")
    print(f"  {TITAN_DIR / '.venv/bin/python'} {BENCH_DIR / 'analysis/compare_arms.py'} {out_dir}")


if __name__ == "__main__":
    main()
