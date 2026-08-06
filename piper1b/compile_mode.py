"""Apply a ``torch.compile`` mode's Inductor options process-wide.

TorchTitan compiles each ``TransformerBlock`` with the default mode and never
passes ``mode=``/``options=`` (``torchtitan/distributed/compile.py``), and its
``CompileConfig`` has no mode field. Setting the mode's option dict on the
global ``torch._inductor.config`` before the first lowering reproduces
``torch.compile(mode=...)`` for every compiled block without patching the
submodule.

The benchmark runner exports ``BENCH_COMPILE_MODE``; ``config_registry``
calls :func:`apply_compile_mode_from_env` at import, which torchtitan performs
while resolving ``--module piper1b`` -- before config parsing, overrides, model
construction, and the first Inductor lowering.

FlexAttention builds its own compiled callable with explicit options that pin
``triton.cudagraphs`` off, resolved at import time; the global config here does
not reach it, which is intended.
"""

import os


ENV_VAR = "BENCH_COMPILE_MODE"


def apply_compile_mode_from_env() -> None:
    """Apply the mode named by ``BENCH_COMPILE_MODE``, if any.

    Unset, empty, and ``default`` are silent no-ops. Any other mode is
    resolved by ``torch._inductor.list_mode_options``, which raises on an
    unknown name, and logs one ``[CompileMode]`` line that ``validate_arm``
    requires -- silent non-application fails the arm.
    """
    mode = os.environ.get(ENV_VAR, "default")
    if mode in ("", "default"):
        return

    # Imported lazily so CPU-only importers of this module do not pay for torch.
    import torch._inductor
    from torch._inductor import config as inductor_config

    options = torch._inductor.list_mode_options(mode)
    for key, value in options.items():
        *parents, leaf = key.split(".")
        target = inductor_config
        for parent in parents:
            target = getattr(target, parent)
        setattr(target, leaf, value)
    applied = " ".join(f"{key}={options[key]}" for key in sorted(options))
    print(f"[CompileMode] {mode}: {applied}", flush=True)
