"""The hardware and source-revision facts recorded in every manifest.

``hardware_metadata`` produces the manifest's ``hardware_metadata`` block --
the requested GPU, the ``nvidia-smi`` identity line, the torch version, the
torchtitan/benchmarks/Megatron revisions and the TransformerEngine version --
and, as a second job, resolves ``--hardware auto`` into the slugified GPU
name used as the output-directory label. The two travel together because both
are read out of the same ``nvidia-smi`` query, and issuing it twice would let
the label and the recorded identity disagree.

Every lookup here shells out through ``run_text``, which returns
``"unavailable: <error>"`` rather than raising. That is deliberate:
collecting provenance must never be the thing that fails a run. The string is
honest about what happened, and because ``_resume_mismatches`` compares
``nvidia_smi``, ``torchtitan_git_rev``, ``benchmarks_git_rev`` and
``megatron_git_rev`` verbatim, an "unavailable" value is still a resume
boundary rather than a hole in one.

``cudnn_torch_build`` and ``cudnn_loader_resolves`` are recorded separately
and on purpose. TransformerEngine's ``DT_NEEDED`` entries carry no
``RUNPATH``, and torch loads its own cuDNN lazily, so on a host that ships
cuDNN in a system directory the loader binds that copy for TE rather than
the wheel torch is pinned against. Which cuDNN a megatron arm runs is
therefore decided by the host, not by the pin, and until 2026-08-20 no
manifest recorded it. The two fields are collected but **not** yet compared
by ``_resume_mismatches``: recording the boundary and gating on it are
separate decisions, and older manifests carry neither field.

Separate from ``environment.py`` because it changes for a different reason. A
new entry here is a manifest-schema decision -- ``megatron_git_rev`` and
``te_version`` were both added that way, and every one of these fields is
something a published number has to be cited with -- while a new entry there
is a subprocess-configuration decision. The Megatron revision is collected
for every run, pure-titan scenarios included, because
``benchmarks.models.piper_qwen3.megatron_bootstrap`` never imports Megatron
itself: the cost is one ``git rev-parse``.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

from benchmarks.execution.paths import RuntimePaths


def _megatron_git_rev() -> str:
    try:
        from benchmarks.models.piper_qwen3.megatron_bootstrap import megatron_git_rev
    except ImportError as error:
        return f"unavailable: {error}"
    return megatron_git_rev()


def _te_version() -> str:
    import importlib.metadata

    try:
        return importlib.metadata.version("transformer-engine")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable: transformer-engine not installed"


_CUDNN_LOADER_PROBE = """
import ctypes, ctypes.util, os
name = ctypes.util.find_library("cudnn") or "libcudnn.so.9"
try:
    ctypes.CDLL(name)
except OSError as error:
    print(f"unavailable: {error}")
    raise SystemExit
for line in open("/proc/self/maps"):
    if "libcudnn.so" in line:
        print(os.path.realpath(line.rsplit(" ", 1)[-1].strip()))
        break
else:
    print("unavailable: libcudnn not mapped after load")
"""


def _cudnn_torch_build() -> str:
    """The cuDNN version torch was compiled against.

    Read through ``getCompileVersion`` rather than ``backends.cudnn.version``,
    because the latter raises whenever the resolved runtime is older, which is
    the very case this field exists to record.
    """
    return run_text(
        [
            sys.executable,
            "-c",
            "import torch; print('.'.join(str(part) for part in "
            "torch._C._cudnn.getCompileVersion()))",
        ]
    ).strip()


def _cudnn_loader_resolves() -> str:
    """The cuDNN the dynamic loader binds, which is the one TE gets."""
    return run_text([sys.executable, "-c", _CUDNN_LOADER_PROBE]).strip()


def run_text(command: list[str], *, cwd: Path | None = None) -> str:
    """Run a metadata command, returning a diagnostic instead of failing."""
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, cwd=cwd
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def hardware_metadata(
    paths: RuntimePaths, gpu: str, hardware_label: str
) -> tuple[str, dict[str, str]]:
    """Collect the hardware and source provenance stored in the manifest."""
    query = run_text(
        [
            "nvidia-smi",
            "--id=" + gpu,
            "--query-gpu=index,name,uuid,driver_version",
            "--format=csv,noheader",
        ]
    ).strip()
    metadata = {
        "requested_gpu": gpu,
        "nvidia_smi": query,
        "torch_version": run_text(
            [sys.executable, "-c", "import torch; print(torch.__version__)"],
            cwd=paths.titan_dir,
        ).strip(),
        "torchtitan_git_rev": run_text(
            ["git", "rev-parse", "HEAD"], cwd=paths.titan_dir
        ).strip(),
        "benchmarks_git_rev": run_text(
            ["git", "rev-parse", "HEAD"], cwd=paths.bench_dir
        ).strip(),
        # Megatron provenance is recorded (and resume-gated) for every run:
        # megatron_bootstrap.py never imports megatron, so this stays cheap
        # even for pure-titan scenarios.
        "megatron_git_rev": _megatron_git_rev(),
        "te_version": _te_version(),
        "cudnn_torch_build": _cudnn_torch_build(),
        "cudnn_loader_resolves": _cudnn_loader_resolves(),
    }
    if hardware_label != "auto":
        return hardware_label, metadata
    name = query.split(",")[1].strip() if "," in query else f"gpu{gpu}"
    label = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return label, metadata
