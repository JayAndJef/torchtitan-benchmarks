"""Where the repository, its submodules, and a run's caches live.

Defined once, deliberately. ``BENCH_DIR`` is this file's own location walked
back to the repository root and ``TITAN_DIR`` is derived from it, not read
from the environment: there is no ``TITAN_DIR`` override and no
``TITAN_PYTHON``, so a run cannot be pointed at a torchtitan checkout that
the manifest's ``torchtitan_git_rev`` does not describe. The only thing that
invalidates ``parents[2]`` is moving this file, and
``tests/test_migration_contract.py`` asserts the walk against the directory
holding ``run_bench.sh`` because a stale ``.parent`` chain does not raise --
it silently relocates ``out/``, the caches, and the git-rev lookups.

These constants are split out from the rest of ``execution/`` because they
have consumers that are not the runners. ``benchmarks.artifacts.layout``
takes ``BENCH_DIR`` alone to root the default output directory, and
``benchmarks.e2e.megatron_stock.data`` -- which executes inside the *training*
subprocess rather than the supervisor -- takes ``TITAN_DIR`` alone to find
torchtitan's ``c4_test`` assets. Neither should have to import
subprocess-launching machinery to spell a path.

``RuntimePaths`` belongs here rather than beside ``runtime_environment``
because it is a record of resolved locations that the environment builder
then consumes; its own docstring has always said so. It resolves the cache
root and the compiler-env script from a caller-supplied mapping, defaulting
to ``os.environ`` only when none is passed, so a test or a resume can
resolve a run's paths without touching the real environment.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


BENCH_DIR = Path(__file__).resolve().parents[2]
TITAN_DIR = BENCH_DIR / "third_party" / "torchtitan"


@dataclass(frozen=True)
class RuntimePaths:
    """Filesystem locations needed to execute a benchmark."""

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
    return Path(value) if value else None
