"""Make stock Megatron-LM importable in this venv, then put it on the path.

The stock driver imports ``megatron.training``, which reaches two of
Megatron's own model modules. Both read ``override`` from ``typing``, and
``typing.override`` arrived in Python 3.12. This venv runs Python 3.10, so
the import fails.

``install_typing_override`` adds that one name from ``typing_extensions``,
which the venv already provides. It changes nothing else. The Megatron-LM
checkout is read-only, so a submodule edit is not an option.

Moving the venv to Python 3.12 was the alternative. It rebuilds every wheel,
rebuilds FlashAttention-3 from source, and makes every published number
incomparable. This work rejects it.

``ensure_dataset_helpers`` answers a second environment problem. Read its
own docstring for it. ``add_wgrad_extension_to_path`` answers a third: the
apex kernel that stock gradient accumulation fusion needs.

This module imports no torch and no megatron.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import sysconfig
import typing
from pathlib import Path
from types import ModuleType
from typing import MutableMapping

from benchmarks.models.piper_qwen3.megatron_bootstrap import (
    REPO_ROOT,
    add_megatron_to_path,
    configure_te_environment,
)

SHIMMED_NAME = "override"
"""The one name the shim adds. Python 3.12 and above supply it themselves."""

_IMPORT_FOR_ME = object()
"""Tells "import typing_extensions for me" from "there is no such module".

A test needs the second case, and ``None`` alone cannot say it.
"""


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
    ``make`` over Megatron's own dataset helpers and then calls
    ``sys.exit(1)`` when make fails. There is no flag that skips it. So a
    failure here stops the arm before it builds a model.

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
    this writes no tracked file and edits nothing in that checkout.

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


def install_allocator_defaults(
    environ: "MutableMapping[str, str]" = os.environ,
) -> None:
    """Set the allocator policy the TorchTitan arms get from run_train.sh.

    It has to precede the first torch import, because torch reads it when
    it initializes CUDA. Both engines of the scenario then run one
    allocator policy, which is the comparability property that matters.
    """
    environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")


def install_rendezvous_defaults(environ: "MutableMapping[str, str]" = os.environ) -> None:
    """Fill the rendezvous variables at one rank, and keep the launcher's.

    Megatron's ``_initialize_distributed`` calls ``init_process_group`` with
    no store, so torch reads ``MASTER_ADDR`` and ``MASTER_PORT`` from the
    environment. Above one rank ``torch.distributed.run`` sets both. At one
    rank nothing did, and the arm died before it trained a step. The two
    ``setdefault`` calls keep a value the launcher chose and fill the
    single-rank case only.
    """
    environ.setdefault("MASTER_ADDR", "127.0.0.1")
    if "MASTER_PORT" not in environ:
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            environ["MASTER_PORT"] = str(sock.getsockname()[1])


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
    add_wgrad_extension_to_path()
    megatron_dir = add_megatron_to_path()
    ensure_dataset_helpers(megatron_dir)
    return megatron_dir
