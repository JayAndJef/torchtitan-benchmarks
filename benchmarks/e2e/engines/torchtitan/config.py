"""The settings of one TorchTitan arm."""

from __future__ import annotations

from dataclasses import dataclass

from benchmarks.e2e.engines.api import CompileMode, EngineConfig


@dataclass(frozen=True, kw_only=True)
class TorchTitanConfig(EngineConfig):
    """The settings of one TorchTitan arm."""

    compile: CompileMode
    """Whether TorchTitan compiles each transformer block."""
    module: str = "benchmarks.e2e.engines.torchtitan.plugins"
    """The package that the fork's ``--module`` flag imports; its ``config_registry`` holds ``config``."""
    config: str = "qwen3_piper_1b_pretokenized"
    """The config function that the fork's ``--config`` flag names."""
    overrides_per_block: int = 0
    """The ``[Override]`` lines that each transformer block prints."""
    override_imports: tuple[str, ...] = ()
    """The ``--override.imports`` targets of the arm."""
    trace_kernel_markers: tuple[str, ...] = ()
    """The kernel names that the traces of a profiled run must hold."""
    requires_gcc_toolset: bool = False
    """Whether the arm runs under the ``--compiler-env`` script."""
    packed_offsets: bool = False
    """Whether the loader sends the document offsets of each pipeline microbatch, which an override attention reads."""
