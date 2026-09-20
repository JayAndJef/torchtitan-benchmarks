"""The engine records: one per training engine this harness can run.

``Arm.engine`` names a key of ``ENGINES``. The record joins the two halves
an arm needs -- the command builder that starts the engine, and the
validation profile that reads its log -- so the two cannot be paired
wrongly. The arm used to carry two independent strings, and any pair of them
was expressible.

This module is the source of that pairing. ``benchmarks.e2e.launch`` owns
the builders and ``benchmarks.e2e.validation`` owns the profiles; neither
reads this module, because a record holds both of them.

``command_for_arm`` lives here for the same reason: the dispatch needs the
records, so it cannot sit in ``launch.py`` below them.
"""

from __future__ import annotations

from pathlib import Path

from benchmarks.e2e.launch import megatron_stock_command, titan_command
from benchmarks.e2e.parallelism import ParallelismSpec, TRIVIAL_SPEC
from benchmarks.e2e.registry import (
    DEFAULT_MEGATRON_NAN_GUARD,
    DEFAULT_MEGATRON_P2P_SYNC,
    DEFAULT_MEGATRON_PRECISION,
)
from benchmarks.e2e.schema import Arm, Engine, Workload
from benchmarks.e2e.validation import (
    MEGATRON_STOCK_PROFILE,
    TORCHTITAN_PROFILE,
)


# Every engine an arm may name.
#
# Declared one by one, with ``is_megatron`` stated rather than read off the
# name. A name test fails open: an engine that spelled the library another
# way -- ``mcore``, ``nemo`` -- would walk past the parallelism rules and
# the three megatron run axes.
ENGINES: dict[str, Engine] = {
    "torchtitan": Engine(
        name="torchtitan",
        command=titan_command,
        validation=TORCHTITAN_PROFILE,
        is_megatron=False,
    ),
    "megatron_stock": Engine(
        name="megatron_stock",
        command=megatron_stock_command,
        validation=MEGATRON_STOCK_PROFILE,
        is_megatron=True,
    ),
}


def engine_for_arm(arm: Arm) -> Engine:
    """The engine record this arm names."""
    try:
        return ENGINES[arm.engine]
    except KeyError as error:
        raise ValueError(
            f"{arm.name}: unknown engine {arm.engine!r}. Available: "
            + ", ".join(sorted(ENGINES))
        ) from error


def command_for_arm(
    workload: Workload,
    arm: Arm,
    arm_dir: Path,
    extra_args: list[str] | tuple[str, ...],
    ac_mode: str = "sac",
    *,
    model_size: str = "1b",
    parallelism: ParallelismSpec = TRIVIAL_SPEC,
    megatron_p2p_sync: str = DEFAULT_MEGATRON_P2P_SYNC,
    megatron_nan_guard: str = DEFAULT_MEGATRON_NAN_GUARD,
    megatron_precision: str = DEFAULT_MEGATRON_PRECISION,
    profile: bool = True,
) -> list[str]:
    """Build the training command for one arm, through its engine record.

    ``model_size``, ``parallelism``, ``megatron_p2p_sync``,
    ``megatron_nan_guard`` and ``megatron_precision`` are keyword-only: the
    five positional parameters are the historical signature and callers
    pass them positionally.

    ``parallelism`` defaults to ``TRIVIAL_SPEC`` rather than being required,
    and the asymmetry with ``manifest_data`` -- which takes its parallelism
    with no default at all -- is deliberate. A manifest states what a run
    **was**, so a defaulted value there could publish a parallel run under a
    single-GPU claim. A command line is **built**, and this default builds
    the single-GPU command line, which is the identity: it cannot introduce
    a ``--parallelism.*`` token. ``_resolve_run`` passes the run's own spec
    explicitly either way.

    The three megatron values default to the identity as well: each adds no
    token to any argv, and each reaches the stock megatron builder alone.
    ``_resolve_run`` is what refuses a non-default value for a run that
    holds no megatron arm.

    ``profile`` defaults to ``True``, which is the identity again, by a
    different route: every command line this repository recorded before the
    axis existed carried the profiler flags. The run axis itself defaults
    to off (``DEFAULT_PROFILE``), and ``_resolve_run`` passes the run's own
    value. Under ``False`` both engines lose every profiler token and the
    run writes no trace.

    Every builder takes the same parameters, so this function passes one
    call through and branches on nothing.
    """
    return engine_for_arm(arm).command(
        workload,
        arm,
        arm_dir,
        extra_args,
        ac_mode,
        model_size,
        parallelism,
        megatron_p2p_sync,
        megatron_nan_guard,
        megatron_precision,
        profile,
    )
