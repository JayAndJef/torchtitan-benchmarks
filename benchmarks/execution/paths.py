"""Where the repository, the TorchTitan checkout and the build caches of a run live."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


BENCH_DIR = Path(__file__).resolve().parents[2]
"""The repository root."""

TITAN_DIR = BENCH_DIR / "third_party" / "torchtitan"
"""The TorchTitan submodule checkout."""


@dataclass(frozen=True)
class RuntimePaths:
    """The directories and the compiler script of one run."""

    bench_dir: Path
    titan_dir: Path
    cache_root: Path
    compiler_env: Path | None

    @classmethod
    def resolve(
        cls,
        *,
        cache_root: Path | None = None,
        compiler_env: Path | None = None,
        environment: Mapping[str, str] | None = None,
    ) -> RuntimePaths:
        """The paths of a run: each argument, else its environment variable, else the fallback."""
        environment = environment or os.environ
        cache = cache_root or _optional_path(environment.get("BENCHMARK_CACHE_ROOT"))
        if cache is None:
            cache = Path(tempfile.gettempdir()) / "torchtitan-benchmarks"

        compiler = compiler_env or _optional_path(environment.get("BENCH_COMPILER_ENV"))
        if compiler is None:
            legacy_toolset = Path("/opt/rh/gcc-toolset-13/enable")
            compiler = legacy_toolset if legacy_toolset.exists() else None

        return cls(
            bench_dir=BENCH_DIR,
            titan_dir=TITAN_DIR,
            cache_root=cache.expanduser().resolve(),
            compiler_env=compiler.expanduser().resolve() if compiler else None,
        )


def _optional_path(value: str | None) -> Path | None:
    """``value`` as a path; an empty value gives ``None``."""
    return Path(value) if value else None
