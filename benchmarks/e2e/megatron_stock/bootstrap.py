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
