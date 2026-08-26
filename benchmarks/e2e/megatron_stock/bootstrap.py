"""Make stock Megatron-LM importable in this venv, then put it on the path.

The tuned driver imports ``megatron.core`` alone. The stock driver imports
``megatron.training``, which reaches ``megatron/training/models/gpt.py``
and ``megatron/training/models/hybrid.py``. Both read ``override`` from
``typing``, and ``typing.override`` arrived in Python 3.12. This venv runs
Python 3.10, so the import fails. ``gpt.py`` is the file this driver's own
path reaches, through ``model_builder.py``.

``install_typing_override`` adds that one name from ``typing_extensions``,
which the venv already provides. It changes nothing else, and a submodule
edit would break the rule that ``third_party/`` is read-only.

Moving the venv to Python 3.12 was the alternative. It rebuilds every wheel,
rebuilds FlashAttention-3 from source, and makes every published number
incomparable. This work rejects it.

``ensure_dataset_helpers`` answers a second environment problem. Read its
own docstring for it.

This module imports no torch and no megatron.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sysconfig
import typing
from pathlib import Path
from types import ModuleType

from benchmarks.models.piper_qwen3.megatron_bootstrap import (
    add_megatron_to_path,
    configure_te_environment,
)

# The one name the shim adds. Python 3.12 and above supply it themselves.
SHIMMED_NAME = "override"

# Tells "import typing_extensions for me" apart from "there is no such
# module". A test needs the second case, and ``None`` alone cannot say it.
_IMPORT_FOR_ME = object()


def install_typing_override(
    *,
    typing_module: ModuleType | None = None,
    extensions: ModuleType | None | object = _IMPORT_FOR_ME,
) -> bool:
    """Add ``typing.override`` when the interpreter lacks it.

    Returns True when the shim added the name, and False when the
    interpreter already had it. The function is idempotent: a second call
    finds the name and returns False.

    **On Python 3.12 and above this function changes nothing.** It reads
    ``typing`` first and returns before it looks at ``typing_extensions``.

    **It never continues silently.** A missing ``typing_extensions``, one
    with no ``override``, or one whose ``override`` is not callable, raises
    ``RuntimeError``. Megatron would otherwise fail later inside a class
    body, with an error that names neither this shim nor the fix. The same
    reason makes the first check ``callable`` rather than ``hasattr``: a
    ``typing.override`` of ``None`` satisfies ``hasattr`` and breaks
    Megatron.

    The two keyword parameters exist for the tests. A caller in the driver
    passes neither, and gets the real ``typing`` module and a real import.
    """
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
    """Build megatron's C++ dataset helper before megatron's own make runs.

    ``megatron.training.initialize`` calls ``compile_helpers``, which runs
    ``make`` in ``megatron/core/datasets`` and then calls ``sys.exit(1)``
    when make fails. There is no flag that skips it. So a failure here
    stops the arm before it builds a model.

    The Makefile reads its include flags from ``python3 -m pybind11
    --includes`` and its output name from ``python3-config
    --extension-suffix``. Both run the **system** ``python3``. On this box
    that is Python 3.9 and it has no pybind11, so make compiles with no
    include path and stops at ``pybind11/pybind11.h: No such file or
    directory``.

    torch ships the pybind11 headers, so this function builds the extension
    itself with the correct flags. It writes the file under two names: the
    venv's own suffix, which this interpreter can import, and the suffix the
    system ``python3-config`` reports, which is the name make looks for. A
    present and current target makes make a no-op, so megatron's own step
    then succeeds without a compiler.

    Both names match ``*.so``, which the Megatron-LM checkout gitignores, so
    this writes no tracked file and edits nothing under ``third_party/``.

    Returns the files it guaranteed. An empty tuple means make already had a
    current target and this function did nothing.
    """
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
        # No system python3-config. The venv suffix is then the only name
        # make can ask for, and it is already in the set.
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
    """Compile ``helpers.cpp`` against torch's bundled pybind11 headers.

    This finds torch through ``importlib`` rather than importing it, because
    this module stays torch-free. The caller runs before the driver's first
    torch import.
    """
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


def prepare() -> Path:
    """Everything the stock driver does before its first megatron import.

    The order matters. The shim runs first, because a megatron import after
    ``add_megatron_to_path`` would otherwise fail on Python 3.10.
    ``configure_te_environment`` runs before any TransformerEngine import,
    because TE reads those variables at import time.

    Returns the Megatron-LM checkout the driver runs against.
    """
    install_typing_override()
    configure_te_environment()
    megatron_dir = add_megatron_to_path()
    ensure_dataset_helpers(megatron_dir)
    return megatron_dir
