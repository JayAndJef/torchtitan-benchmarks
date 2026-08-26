"""Make stock Megatron-LM importable in this venv, then put it on the path.

The tuned driver imports ``megatron.core`` alone. The stock driver imports
``megatron.training``, which reaches
``megatron/training/models/hybrid.py``. That module reads ``override`` from
``typing``, and ``typing.override`` arrived in Python 3.12. This venv runs
Python 3.10, so the import fails.

``install_typing_override`` adds that one name from ``typing_extensions``,
which the venv already provides. It changes nothing else, and a submodule
edit would break the rule that ``third_party/`` is read-only.

Moving the venv to Python 3.12 was the alternative. It rebuilds every wheel,
rebuilds FlashAttention-3 from source, and makes every published number
incomparable. This work rejects it.

This module imports no torch and no megatron.
"""

from __future__ import annotations

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

    **It never continues silently.** A missing ``typing_extensions``, or one
    with no ``override``, raises ``RuntimeError``. Megatron would otherwise
    fail later with an import error that names neither this shim nor the
    fix.

    The two keyword parameters exist for the tests. A caller in the driver
    passes neither, and gets the real ``typing`` module and a real import.
    """
    target = typing_module if typing_module is not None else typing
    if hasattr(target, SHIMMED_NAME):
        return False
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
    if replacement is None:
        raise RuntimeError(
            "typing_extensions supplies no override decorator, so the shim "
            "cannot give megatron.training the name it reads; upgrade "
            "typing_extensions"
        )
    setattr(target, SHIMMED_NAME, replacement)
    return True


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
    return add_megatron_to_path()
