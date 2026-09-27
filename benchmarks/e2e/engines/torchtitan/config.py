"""The settings of one TorchTitan arm."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.e2e.engines.api import CompileMode, EngineConfig


@dataclass(frozen=True, kw_only=True)
class TorchTitanConfig(EngineConfig):
    """One TorchTitan arm: the fork's ``--module`` and ``--config``, and the treatment."""

    compile: CompileMode
    module: str = "benchmarks.e2e.engines.torchtitan.plugins"
    config: str = "qwen3_piper_1b_pretokenized"
    overrides_per_block: int = 0
    """The ``[Override]`` lines each transformer block prints."""
    override_imports: tuple[str, ...] = ()
    trace_kernel_markers: tuple[str, ...] = ()
    """Kernel names that the profiler traces of a profiled run must hold."""
    requires_gcc_toolset: bool = False
    """Whether the arm runs under the ``--compiler-env`` script."""
