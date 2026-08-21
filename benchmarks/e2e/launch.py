"""Build the training launch command for one end-to-end benchmark arm.

Command construction is the seam between a declarative arm and the engine
that actually trains it: ``command_for_arm`` dispatches on ``arm.launcher``
and each branch delivers the scenario workload, the global run axes, and the
arm's own overrides in that engine's own spelling.
"""

from __future__ import annotations

import sys
from pathlib import Path

from benchmarks.e2e.registry import TORCH_COMPILE_MODE, Arm, Workload


def command_for_arm(
    workload: Workload,
    arm: Arm,
    arm_dir: Path,
    extra_args: list[str] | tuple[str, ...],
    compile_mode: str = "default",
    ac_mode: str = "sac",
    *,
    model_size: str = "1b",
) -> list[str]:
    """Build the training command for one arm, dispatching on its launcher.

    ``model_size`` is keyword-only: the six positional parameters are the
    historical signature and callers pass them positionally.
    """
    if arm.launcher == "megatron":
        return _megatron_command(
            workload, arm, arm_dir, extra_args, compile_mode, ac_mode, model_size
        )
    if arm.launcher != "torchtitan":
        raise ValueError(f"{arm.name}: unknown launcher {arm.launcher!r}")
    args = [
        "./run_train.sh",
        "--module",
        workload.module,
        "--config",
        arm.config or workload.config,
        # The fork's ConfigManager forwards --config-arg pairs as keyword
        # arguments to the config function, which resolves the name through
        # benchmarks.models.piper_qwen3.shape.shape_by_name.
        "--config-arg",
        f"size={model_size}",
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
    if workload.replay_dataloader:
        # The replay loader materializes exactly this many steps of samples
        # and hard-fails when the run asks for more, so it must track --steps.
        args.extend(("--dataloader.replay-steps", str(workload.steps)))
    if compile_mode != "default":
        args.extend(("--compile.mode", TORCH_COMPILE_MODE[compile_mode]))
    if workload.seed is not None:
        args.extend(("--debug.seed", str(workload.seed)))
    if arm.override_imports:
        args.extend(("--override.imports", ",".join(arm.override_imports)))
    args = args + list(extra_args) + ["--dump-folder", str(arm_dir)]
    if ac_mode == "none":
        # tyro subcommand token selecting activation_checkpoint=None; the
        # flag-style spelling does not exist for subcommand unions, and tyro
        # attributes any flags after the token to the (fieldless) subcommand,
        # so the token must come last.
        args.append("activation-checkpoint:none")
    return args


def _megatron_command(
    workload: Workload,
    arm: Arm,
    arm_dir: Path,
    extra_args: list[str] | tuple[str, ...],
    compile_mode: str,
    ac_mode: str,
    model_size: str = "1b",
) -> list[str]:
    """Launch command for the Megatron baseline driver.

    The driver replicates the titan workload treatment itself; the only
    parameters that cross the seam are the workload sizes, the seed, the
    profiler schedule, and the compile mode (mapped to Megatron's native
    CUDA-graph mechanism by the driver).
    """
    if extra_args:
        raise ValueError(
            f"{arm.name}: TorchTitan passthrough arguments cannot apply to a "
            f"megatron arm: {list(extra_args)}"
        )
    if ac_mode != "none":
        raise ValueError(
            f"{arm.name}: the megatron arm always runs without recompute; "
            f"ac mode {ac_mode!r} has no Megatron parity (use --ac none)"
        )
    if workload.seed is None:
        raise ValueError(
            f"{arm.name}: megatron arms require a seeded workload"
        )
    return [
        sys.executable,
        "-m",
        "benchmarks.e2e.megatron.train",
        "--seq-len",
        str(workload.seq_len),
        "--steps",
        str(workload.steps),
        "--batch",
        str(workload.local_batch_size),
        "--seed",
        str(workload.seed),
        "--profile-freq",
        str(workload.profile_freq),
        "--profiler-warmup",
        str(workload.profiler_warmup),
        "--profiler-active",
        str(workload.profiler_active),
        "--mode",
        compile_mode,
        "--model-size",
        model_size,
        str(arm_dir),
    ]
