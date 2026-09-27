"""The settings of one stock Megatron-LM arm."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from benchmarks.e2e.engines.api import EngineConfig


@dataclass(frozen=True, kw_only=True)
class MegatronStockConfig(EngineConfig):
    """The settings of one stock Megatron-LM arm."""

    p2p_sync: Literal["on", "off"] = "off"
    """Whether Megatron synchronizes after a pipeline message."""
    nan_guard: Literal["on", "off"] = "off"
    """Whether Megatron keeps its NaN and Inf check."""
    precision: Literal["stock", "lean"] = "stock"
    """How Megatron holds the optimizer state."""
    trace_kernel_markers: tuple[str, ...] = ()
    """The kernel names that the traces of a profiled run must hold."""
