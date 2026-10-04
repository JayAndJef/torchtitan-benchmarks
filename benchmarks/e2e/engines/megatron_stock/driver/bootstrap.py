"""The process setup that the driver runs before its first Megatron import.

This module imports no torch and no megatron.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import sysconfig
import typing
from pathlib import Path
from types import ModuleType

from benchmarks.models.piper_qwen3.megatron_bootstrap import (
    REPO_ROOT,
    add_megatron_to_path,
    configure_te_environment,
)

SHIMMED_NAME = "override"
"""The name that the shim adds to ``typing``."""

_IMPORT_FOR_ME = object()
"""The default of ``extensions``; ``None`` there means that ``typing_extensions`` is absent."""


def install_typing_override(
    *,
    typing_module: ModuleType | None = None,
    extensions: ModuleType | None | object = _IMPORT_FOR_ME,
) -> bool:
    """Add ``typing.override`` from ``typing_extensions`` when the interpreter lacks it, and return whether the shim added it."""
    target = typing_module if typing_module is not None else typing
    existing = getattr(target, SHIMMED_NAME, None)
    if callable(existing):
        return False
    if existing is not None:
        raise RuntimeError(
            f"typing.{SHIMMED_NAME} is {existing!r}, which is not callable, "
            "so megatron.training would bind it as a decorator and fail "
            "inside a class body; repair the interpreter rather than "
            "letting the shim overwrite a name it did not set"
        )
    if extensions is _IMPORT_FOR_ME:
        try:
            import typing_extensions
        except ImportError as error:
            raise RuntimeError(
                "this interpreter has no typing.override and "
                "typing_extensions is not installed, so megatron.training "
                "cannot import; install typing_extensions or run on Python "
                "3.12 or above"
            ) from error
        extensions = typing_extensions
    if extensions is None:
        raise RuntimeError(
            "this interpreter has no typing.override and no "
            "typing_extensions module was supplied, so megatron.training "
            "cannot import"
        )
    replacement = getattr(extensions, SHIMMED_NAME, None)
    if not callable(replacement):
        raise RuntimeError(
            "typing_extensions supplies no callable override decorator, so "
            "the shim cannot give megatron.training the name it reads; "
            "upgrade typing_extensions (4.4 added it)"
        )
    setattr(target, SHIMMED_NAME, replacement)
    return True


def ensure_dataset_helpers(megatron_dir: Path) -> tuple[Path, ...]:
    """Build Megatron's C++ dataset helper against torch's pybind11 headers, and return the files that it wrote."""
    datasets = megatron_dir / "megatron" / "core" / "datasets"
    source = datasets / "helpers.cpp"
    if not source.is_file():
        return ()

    suffixes = {sysconfig.get_config_var("EXT_SUFFIX") or ".so"}
    try:
        reported = subprocess.run(
            ["python3-config", "--extension-suffix"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if reported.returncode == 0 and reported.stdout.strip():
            suffixes.add(reported.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        # Without a system python3-config, make asks for the venv suffix alone.
        pass

    wanted = [datasets / f"helpers_cpp{suffix}" for suffix in sorted(suffixes)]
    stale = [
        target
        for target in wanted
        if not target.is_file()
        or target.stat().st_mtime < source.stat().st_mtime
    ]
    if not stale:
        return ()

    built = _compile_dataset_helper(source, stale[0])
    for other in stale[1:]:
        shutil.copyfile(built, other)
    return tuple(stale)


def _compile_dataset_helper(source: Path, target: Path) -> Path:
    """Compile ``helpers.cpp`` against the pybind11 headers of torch, without an import of torch."""
    spec = importlib.util.find_spec("torch")
    locations = list(getattr(spec, "submodule_search_locations", []) or [])
    if not locations:
        raise RuntimeError(
            "cannot locate torch, so the pybind11 headers megatron's dataset "
            "helper needs are unavailable"
        )
    torch_include = Path(locations[0]) / "include"
    python_include = sysconfig.get_paths()["include"]

    command = [
        os.environ.get("CXX", "g++"),
        "-O3",
        "-Wall",
        "-shared",
        "-std=c++17",
        "-fPIC",
        f"-I{python_include}",
        f"-I{torch_include}",
        str(source),
        "-o",
        str(target),
    ]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode != 0:
        raise RuntimeError(
            "failed to build megatron's dataset helper "
            f"({' '.join(command)}): {done.stderr.strip()[:400]}"
        )
    return target


WGRAD_MODULE = "fused_weight_gradient_mlp_cuda"
"""The apex extension Megatron imports for gradient accumulation fusion."""

WGRAD_SOURCE_DIR = REPO_ROOT / "third_party" / "apex-wgrad"
"""The vendored apex sources of ``WGRAD_MODULE``, and nothing else of apex."""

WGRAD_SOURCES = (
    "csrc/megatron/fused_weight_gradient_dense.cpp",
    "csrc/megatron/fused_weight_gradient_dense_cuda.cu",
    "csrc/megatron/fused_weight_gradient_dense_16bit_prec_cuda.cu",
)
"""The compiled sources, relative to ``WGRAD_SOURCE_DIR``, as apex lists them."""

WGRAD_HEADERS = ("csrc/type_shim.h",)
"""The one apex header the sources include."""

WGRAD_BUILD_DIR = REPO_ROOT / ".apex-wgrad"
"""Where ``tools/build_wgrad_ext.py`` writes the built module and its stamp."""

WGRAD_STAMP = WGRAD_BUILD_DIR / "stamp.json"
"""The build record: the torch version and the source digest it was built from."""


def wgrad_source_digest(source_dir: Path = WGRAD_SOURCE_DIR) -> str:
    """Return one sha256 over every vendored source and header, in a fixed order."""
    digest = hashlib.sha256()
    for relative in (*WGRAD_SOURCES, *WGRAD_HEADERS):
        digest.update(relative.encode())
        digest.update((source_dir / relative).read_bytes())
    return digest.hexdigest()


def wgrad_expected_stamp(source_dir: Path = WGRAD_SOURCE_DIR) -> dict[str, str]:
    """Return the stamp a current build must carry.

    It reads the torch version from the package metadata, so it imports no
    torch. A torch pin bump changes the version and so refuses the old build.
    """
    return {
        "module": WGRAD_MODULE,
        "torch_version": importlib.metadata.version("torch"),
        "source_sha256": wgrad_source_digest(source_dir),
    }


def add_wgrad_extension_to_path(
    build_dir: Path = WGRAD_BUILD_DIR,
    source_dir: Path = WGRAD_SOURCE_DIR,
) -> Path:
    """Put the built apex wgrad module on ``sys.path``, or refuse the run.

    Stock Megatron enables ``gradient_accumulation_fusion`` by default, and
    its ``ColumnParallelLinear`` output layer then needs ``WGRAD_MODULE``.
    Megatron raises at model build without it, after the arm has spent its
    startup. This check fails first, and it names the repair.

    A build for another torch version, or from other sources, is refused
    too. Its ABI or its kernels would not match the run.
    """
    repair = (
        "build it with: source /opt/rh/gcc-toolset-13/enable && "
        ".venv/bin/python tools/build_wgrad_ext.py (sync.sh runs it)"
    )
    stamp = build_dir / WGRAD_STAMP.name
    if not stamp.is_file():
        raise RuntimeError(f"{WGRAD_MODULE} is not built under {build_dir}; {repair}")
    recorded = json.loads(stamp.read_text())
    expected = wgrad_expected_stamp(source_dir)
    stale = {
        key: (recorded.get(key), value)
        for key, value in expected.items()
        if recorded.get(key) != value
    }
    if stale:
        raise RuntimeError(
            f"{WGRAD_MODULE} under {build_dir} is stale "
            f"(recorded, current: {stale}); {repair}"
        )
    if not any(build_dir.glob(f"{WGRAD_MODULE}*.so")):
        raise RuntimeError(
            f"{build_dir} has a stamp but no {WGRAD_MODULE} library; {repair}"
        )
    if str(build_dir) not in sys.path:
        sys.path.insert(0, str(build_dir))
    return build_dir


def prepare() -> Path:
    """Set up the process for Megatron in the necessary order, and return the Megatron-LM checkout."""
    install_typing_override()
    configure_te_environment()
    add_wgrad_extension_to_path()
    megatron_dir = add_megatron_to_path()
    ensure_dataset_helpers(megatron_dir)
    return megatron_dir
