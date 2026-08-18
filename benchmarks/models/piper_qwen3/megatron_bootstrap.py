"""Locate the Megatron-LM checkout without importing megatron.

The default is the pinned submodule at third_party/Megatron-LM; MEGATRON_DIR
overrides it for development against an external checkout. Megatron is not
pip-installed (its pyproject wants python >= 3.12; the venv is 3.10), so the
driver puts the checkout on sys.path instead.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
MEGATRON_SUBMODULE = REPO_ROOT / "third_party" / "Megatron-LM"


def megatron_dir(*, environment: dict[str, str] | None = None) -> Path:
    environment = environment if environment is not None else dict(os.environ)
    override = environment.get("MEGATRON_DIR")
    if override:
        path = Path(override).expanduser().resolve()
        if not (path / "megatron").is_dir():
            raise RuntimeError(
                f"MEGATRON_DIR={override} does not contain a megatron package"
            )
        return path
    if (MEGATRON_SUBMODULE / "megatron").is_dir():
        return MEGATRON_SUBMODULE
    raise RuntimeError(
        "Megatron-LM not found: initialize the submodule "
        "(git submodule update --init third_party/Megatron-LM) or set "
        "MEGATRON_DIR"
    )


def add_megatron_to_path() -> Path:
    path = megatron_dir()
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
    return path


def configure_te_environment() -> None:
    """Environment TE needs on this stack, set before importing it.

    The box runs torch cu13 wheels over a 12.8 driver via the cuda-compat
    shim, and the system ldconfig serves libcudart.so.12: TE's native
    "tuned" RMSNorm kernels fail to launch (CUDA invalid argument) in that
    mix, while its cuDNN-backed norm path works. Route norms through cuDNN
    and pin the cuDNN frontend to the cu13 runtime torch already loaded.
    """
    os.environ.setdefault("NVTE_NORM_FWD_USE_CUDNN", "1")
    os.environ.setdefault("NVTE_NORM_BWD_USE_CUDNN", "1")
    os.environ.setdefault("CUDNN_FRONTEND_CUDART_LIB_NAME", "libcudart.so.13")


def megatron_git_rev() -> str:
    """Provenance helper; never raises and never imports megatron."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=megatron_dir(),
            text=True,
            stderr=subprocess.STDOUT,
        ).strip()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        return f"unavailable: {error}"
