"""Arm builders, one module per kernel family: the torch code each scenario times.

**Every implementation import in this package is deferred into the builder
that needs it.** A family module may import ``torch`` and the harness's own
light modules at module scope. It may not import ``torchtitan``, a
``benchmarks.models.piper_qwen3.components`` package, TransformerEngine,
Megatron, FlashAttention or Helion there.

The reason is per-arm process isolation. Each arm is built in its own
interpreter, so a process must pay only for the one arm it builds. A
module-scope import defeats that: it drags every arm's dependencies into
every arm's process, and the collisions are real rather than hypothetical.
FA3 and TransformerEngine cannot share a process at all -- a cuDNN soname
collision, recorded in CLAUDE.md -- and importing ``te_rope_override``
JIT-builds a CUDA extension that needs a C++20 compiler the run may not have
configured.

``tests/test_import_boundaries.py`` section 3 pins the rule.
"""
