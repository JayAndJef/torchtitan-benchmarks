"""The hardware facts and the source revisions that every manifest records."""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from benchmarks.execution.paths import RuntimePaths


def _megatron_git_rev() -> str:
    """The revision of the Megatron-LM checkout."""
    try:
        from benchmarks.models.piper_qwen3.megatron_bootstrap import megatron_git_rev
    except ImportError as error:
        return f"unavailable: {error}"
    return megatron_git_rev()


def _te_version() -> str:
    """The installed TransformerEngine version."""
    import importlib.metadata

    try:
        return importlib.metadata.version("transformer-engine")
    except importlib.metadata.PackageNotFoundError:
        return "unavailable: transformer-engine not installed"


_CUBLASLT_PROBE = """
import ctypes, os
import torch  # Loads the libcublasLt that TransformerEngine then binds.
def mapped():
    for line in open("/proc/self/maps"):
        if "libcublasLt.so" in line:
            return os.path.realpath(line.rsplit(" ", 1)[-1].strip())
path = mapped()
if path is None:
    try:
        ctypes.CDLL("libcublasLt.so.13")
    except OSError as error:
        print(f"unavailable: {error}")
        raise SystemExit
    path = mapped()
if path is None:
    print("unavailable: libcublasLt not mapped after load")
    raise SystemExit
library = ctypes.CDLL(path)
library.cublasLtGetVersion.restype = ctypes.c_size_t
print(f"{library.cublasLtGetVersion()} {path}")
"""


def _cublaslt_version() -> str:
    """The cuBLASLt version and path that the process binds after it imports torch, as TransformerEngine does."""
    return run_text([sys.executable, "-c", _CUBLASLT_PROBE]).strip()


CUDNN_LOADER_PROBE = """
import json, os
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # The TransformerEngine import touches no card.
def directories():
    found = set()
    for line in open("/proc/self/maps"):
        if "libcudnn" in line:
            found.add(os.path.dirname(os.path.realpath(line.rsplit(" ", 1)[-1].strip())))
    return sorted(found)
import torch
bundled = directories()
try:
    import transformer_engine.pytorch
except ImportError as error:
    print(f"unavailable: {error}")
    raise SystemExit
major, minor, patch = torch._C._cudnn.getCompileVersion()
try:
    runtime = str(torch.backends.cudnn.version())
except RuntimeError as error:
    runtime = f"raises: {error}"
print(json.dumps({
    "bundled": bundled,
    "loaded": directories(),
    "build": str(major * 10000 + minor * 100 + patch),
    "runtime": runtime,
}))
"""
"""A probe that prints, as one JSON line, the cuDNN directories after ``import torch`` and after the TransformerEngine import, torch's cuDNN build and ``torch.backends.cudnn.version()``."""


def _cudnn_torch_build() -> str:
    """The cuDNN version that torch was built against; ``getCompileVersion`` reads it, because ``backends.cudnn.version`` raises when the runtime is older."""
    return run_text(
        [
            sys.executable,
            "-c",
            "import torch; print('.'.join(str(part) for part in "
            "torch._C._cudnn.getCompileVersion()))",
        ]
    ).strip()


def cudnn_loader_resolves(output: str) -> str:
    """The cuDNN version and directories of one ``CUDNN_LOADER_PROBE`` output; a cuDNN beside torch's own, or another runtime version than torch's build, raises."""
    lines = output.splitlines()
    probes = [line for line in lines if line.startswith("{")]
    if not probes:
        if lines and lines[-1].startswith("unavailable:"):
            return lines[-1]
        raise ValueError(f"the cuDNN loader probe printed no JSON line: {output}")
    probe = json.loads(probes[-1])
    foreign = [path for path in probe["loaded"] if path not in probe["bundled"]]
    if foreign or probe["runtime"] != probe["build"]:
        raise ValueError(
            f"TransformerEngine maps the cuDNN in {', '.join(foreign) or 'none'} "
            f"beside torch's cuDNN in {', '.join(probe['bundled']) or 'none'}, and "
            f"torch.backends.cudnn.version() gives {probe['runtime']!r} against "
            f"the build {probe['build']}; run through ./run_bench.sh, which "
            "sources cudnn_env.sh"
        )
    return f"{probe['runtime']} {', '.join(probe['loaded'])}"


def _cudnn_loader_resolves() -> str:
    """The cuDNN version and directories that a process maps after it imports TransformerEngine."""
    return cudnn_loader_resolves(
        run_text([sys.executable, "-c", CUDNN_LOADER_PROBE]).strip()
    )


def run_text(command: list[str], *, cwd: Path | None = None) -> str:
    """The output of a provenance command; a failure gives ``unavailable: <error>`` and does not raise."""
    try:
        return subprocess.check_output(
            command, text=True, stderr=subprocess.STDOUT, cwd=cwd
        )
    except (OSError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"


def _device_names(gpu: str, query: str) -> list[str]:
    """The GPU model on each line of the ``nvidia-smi`` query, in index order; an empty list when the query does not give one line per device."""
    names = [
        line.split(",")[1].strip() for line in query.splitlines() if "," in line
    ]
    return names if len(names) == gpu.count(",") + 1 else []


def hardware_metadata(
    paths: RuntimePaths, gpu: str, hardware_label: str
) -> tuple[str, dict[str, str]]:
    """The hardware label and the ``hardware_metadata`` block of a manifest; a device set of two GPU models raises."""
    query = run_text(
        [
            "nvidia-smi",
            "--id=" + gpu,
            "--query-gpu=index,name,uuid,driver_version",
            "--format=csv,noheader",
        ]
    ).strip()
    names = _device_names(gpu, query)
    if len(set(names)) > 1:
        raise ValueError(
            f"device list {gpu!r} mixes GPU models ({', '.join(names)}); one "
            "run records one hardware label, so a mixed set is not one "
            "measurement"
        )
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
        # Recorded for every run, and cheap because nothing imports megatron.
        "megatron_git_rev": _megatron_git_rev(),
        "te_version": _te_version(),
        "cublaslt_version": _cublaslt_version(),
        "cudnn_torch_build": _cudnn_torch_build(),
        "cudnn_loader_resolves": _cudnn_loader_resolves(),
    }
    if hardware_label != "auto":
        return hardware_label, metadata
    # Not _device_names: --resume compares the label, so it must not move on a host whose nvidia-smi fails.
    name = query.split(",")[1].strip() if "," in query else f"gpu{gpu}"
    label = re.sub(r"[^a-zA-Z0-9]+", "-", name).strip("-").lower()
    return label, metadata
