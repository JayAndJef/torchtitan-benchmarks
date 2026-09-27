"""How TorchTitan builds the mesh of a run: its schedules, its data-parallel degrees and its execution model."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.e2e.parallelism import (
    ParallelismSpec,
    data_parallel_term,
    degree_terms,
    device_term,
)


@dataclass(frozen=True)
class TitanSchedule:
    """One pipeline schedule that TorchTitan runs."""

    titan_name: str
    """The value of ``--parallelism.pipeline-parallel-schedule``."""
    requires_uncompiled: bool
    """Whether the schedule raises on a compiled stage module."""


SCHEDULES: dict[str, TitanSchedule] = {
    "1F1B": TitanSchedule("1F1B", requires_uncompiled=False),
    "Interleaved1F1B": TitanSchedule("Interleaved1F1B", requires_uncompiled=False),
    "InterleavedZeroBubble": TitanSchedule(
        "InterleavedZeroBubble", requires_uncompiled=True
    ),
    "ZBVZeroBubble": TitanSchedule("ZBVZeroBubble", requires_uncompiled=True),
    "DualPipeV": TitanSchedule("DualPipeV", requires_uncompiled=True),
}
"""The TorchTitan schedule of each shared schedule name that TorchTitan runs."""


def titan_mesh(spec: ParallelismSpec) -> tuple[int, int]:
    """TorchTitan's ``(dp_replicate, dp_shard)`` for ``spec``."""
    if spec.zero == 0:
        return (spec.dp, 1)
    return (1, spec.dp)


def reshard_after_forward(spec: ParallelismSpec) -> str | None:
    """The ``--parallelism.fsdp-reshard-after-forward`` policy; ``None`` sends no flag."""
    return "never" if spec.zero == 1 else None


def skip_dp(spec: ParallelismSpec) -> bool:
    """Whether the parallelize plug-in skips TorchTitan's data-parallel path."""
    return spec.dp == 1 and spec.ep == 1


def execution_model(spec: ParallelismSpec) -> str:
    """How a TorchTitan arm holds the model state, as one manifest string."""
    data_parallel = "no-fsdp" if skip_dp(spec) else data_parallel_term(spec)
    return "-".join(
        (device_term(spec), "plain-bf16", data_parallel, *degree_terms(spec))
    )
