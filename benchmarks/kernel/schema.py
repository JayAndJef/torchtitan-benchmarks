"""What a kernel-isolation benchmark *is*, separate from which ones exist.

The declaration vocabulary -- ``KernelWorkload``, ``CorrectnessCheck``,
``KernelArm``, ``KernelScenario``, the mode tuple -- plus the arithmetic that
follows from a (shape, workload) pair: what a run may ask for
(``resolve_shape_and_workload``), what it may not (``validate_shape_and_
workload``, ``routing_divides_evenly``), and the derived tensor shapes both
systems record (``shape_summary``). ``benchmarks.kernel.registry`` holds the
five scenarios themselves and imports from here; nothing here knows that
``rope`` or ``attention`` exist as objects.

**The split is an import-graph rule, not tidiness.**
``benchmarks.kernel.engine`` needs these types and must never import the
scenario *data*. Arm builders are dotted strings that ``resolve_symbol``
resolves inside the worker, so today's ``registry.py`` creates no edge to
``benchmarks.kernel.operations`` -- but the moment a scenario constant is
moved next to its family's builders ("colocate the family"), the chain
``engine -> registry -> operations.<family> -> torchtitan`` makes the engine
import every operation module and every model dependency behind them. The
engine importing only this module is what keeps that impossible, and what
lets a later change run each arm in its own process. A test in
``tests/test_import_boundaries.py`` pins both halves.

``shape_summary`` is the one member that is data-adjacent: it keys on
scenario names as bare strings, so adding a scenario edits this file and
``registry.py`` both. It lives here because its two callers are the manifest
writer in the torch-free parent and the result assembler in the engine, and
the second of those may not reach it through the scenario objects.

Torch-free, so ``./run_bench.sh scenarios`` and ``--help`` list arms without
CUDA. The one first-party import, ``benchmarks.models.piper_qwen3.shape``,
imports nothing but ``dataclasses``: it is the geometry registry both engines
already share, not a model package.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from benchmarks.models.piper_qwen3.shape import PiperShape, shape_by_name


MODES = ("forward", "backward", "forward_backward")


@dataclass(frozen=True)
class KernelWorkload:
    """What is run through the model, as opposed to what the model is.

    Kept separate from ``PiperShape`` so batch size is independent of
    geometry: the same shape is measurable at any batch, and a batch change
    is not a model change. Everything geometric -- dim, head counts, expert
    width, vocabulary, ``max_seq_len`` -- belongs to the shape, and no
    per-family shapes live in either: each scenario's inputs builder computes
    what it needs (swiglu rows = batch * seq_len * top_k, fused-qkv out
    features = (n_heads + 2 * n_kv_heads) * head_dim, lm_head tokens =
    batch * seq_len).
    """

    batch: int = 4
    seq_len: int = 1024


def validate_shape_and_workload(
    shape: PiperShape, workload: KernelWorkload
) -> None:
    """Raise ValueError on a shape/workload pair no scenario can run.

    Only constraints that genuinely cross the two live here. The purely
    geometric ones are structurally unrepresentable: ``PiperShape`` derives
    both head counts from ``dim`` and ``head_dim``, so
    ``n_heads * head_dim == dim`` and ``n_heads % n_kv_heads == 0`` cannot be
    violated and there is nothing to assert.
    """
    if workload.seq_len > shape.max_seq_len:
        raise ValueError(
            f"seq_len ({workload.seq_len}) exceeds max_seq_len "
            f"({shape.max_seq_len}); RoPE tables are sized by max_seq_len"
        )


def routing_divides_evenly(
    shape: PiperShape, workload: KernelWorkload
) -> bool:
    """Whether swiglu's synthetic workload routes rows evenly across experts.

    Scenario-scoped on purpose: only the swiglu inputs builder hands every
    expert an equal slice of ``batch * seq_len * top_k`` rows, so an uneven
    split is that scenario's problem and must not fail rope, qkv, lm_head or
    attention.
    """
    return (
        workload.batch * workload.seq_len * shape.top_k
    ) % shape.num_experts == 0


def resolve_shape_and_workload(
    *,
    model_size: str = "normal",
    batch: int | None = None,
    seq_len: int | None = None,
    max_seq_len: int | None = None,
) -> tuple[PiperShape, KernelWorkload]:
    """The geometry and the workload one kernel-bench run measures."""
    shape = shape_by_name(model_size)
    # max_seq_len first: validation rejects seq_len > max_seq_len, and the
    # attention sweep raises the ceiling precisely in order to run past it.
    if max_seq_len is not None:
        shape = replace(shape, max_seq_len=int(max_seq_len))
    workload = KernelWorkload()
    if batch is not None:
        workload = replace(workload, batch=int(batch))
    if seq_len is not None:
        workload = replace(workload, seq_len=int(seq_len))
    validate_shape_and_workload(shape, workload)
    return shape, workload


@dataclass(frozen=True)
class CorrectnessCheck:
    """One validity gate: compare named outputs against a reference.

    ``reference`` is another arm's name, or ``"fp64"`` for the scenario's
    ground-truth builder. ``kind`` selects the comparison: ``bitwise``
    (torch.equal), ``tolerance`` (max_abs / max_rel / max_rel_l2 bounds), or
    ``fp64_ulp`` (bf16 ULPs against the fp64 truth). Informational checks are
    recorded in results but never gate the run.

    Choosing a metric:

    - ``max_rel_l2`` (||a - b|| / ||b||) is the default. It is the only one
      that stays meaningful when the compared tensors differ in magnitude
      (weight gradients accumulate over thousands of rows and sit ~30x above
      activations, so one bf16 ULP there is a large absolute number) or when
      individual elements land near zero.
    - ``fp64_ulp`` reports the *mean* bf16 ULP error and suits elementwise
      kernels (RoPE), where it is the accuracy number this repo has always
      quoted (~0.24 ULP). It is a mean because the per-element maximum is
      meaningless wherever cancellation can drive a true value toward zero:
      dividing a negligible absolute error by that tiny magnitude reports
      thousands of ULPs for a numerically perfect kernel, and does so
      identically for the stock implementation.
    - ``bitwise`` where two implementations must agree exactly.
    """

    kind: str
    reference: str
    outputs: tuple[str, ...]
    max_mean_ulp: float | None = None
    max_abs: float | None = None
    max_rel: float | None = None
    max_rel_l2: float | None = None
    informational: bool = False


@dataclass(frozen=True)
class KernelArm:
    """One implementation in a head-to-head kernel comparison.

    ``builder`` is a dotted path ("module:function") resolved in the GPU
    worker; the function receives (shape, workload, inputs) and returns a
    BuiltArm whose per-mode closures are the timed operations.
    ``compare_to`` names the opponent arm for ratio and significance rows
    (None means the scenario baseline); comparisons only happen between arms
    sharing a mode.
    ``compiled`` records that the builder runs the arm under torch.compile,
    as production does; raw-kernel arms and floors stay eager on purpose.
    """

    name: str
    description: str
    builder: str
    modes: tuple[str, ...]
    compare_to: str | None = None
    correctness: tuple[CorrectnessCheck, ...] = ()
    requires_gcc_toolset: bool = False
    is_floor: bool = False
    compiled: bool = False


@dataclass(frozen=True)
class KernelScenario:
    """A kernel family and its comparable implementation arms.

    ``requires_balanced_routing`` marks a scenario whose inputs builder hands
    every expert an equal slice of the routed rows; the runner refuses to run
    it on a workload where they do not divide evenly, rather than silently
    rounding the split.
    """

    name: str
    description: str
    inputs_builder: str
    reference_builder: str | None
    arms: tuple[KernelArm, ...]
    baseline_arm: str
    requires_balanced_routing: bool = False

    def arm(self, name: str) -> KernelArm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise ValueError(
            f"Unknown arm {name!r} for kernel scenario {self.name!r}. "
            f"Available arms: {', '.join(arm.name for arm in self.arms)}"
        )

    @property
    def requires_gcc_toolset(self) -> bool:
        return any(arm.requires_gcc_toolset for arm in self.arms)


def shape_summary(
    scenario_name: str, shape: PiperShape, workload: KernelWorkload
) -> dict[str, object]:
    """Derived shapes recorded in the manifest for provenance."""
    batch, seq = workload.batch, workload.seq_len
    if scenario_name == "rope":
        return {
            "q": [batch, seq, shape.n_heads, shape.head_dim],
            "k": [batch, seq, shape.n_kv_heads, shape.head_dim],
            "positions": [batch, seq],
            "table_rows": shape.max_seq_len,
        }
    if scenario_name == "swiglu":
        rows = batch * seq * shape.top_k
        per_expert = rows // shape.num_experts
        return {
            "x": [rows, shape.dim],
            "tokens_per_expert": [per_expert] * shape.num_experts,
        }
    if scenario_name == "qkv":
        kv_out = shape.n_kv_heads * shape.head_dim
        return {
            "x": [batch, seq, shape.dim],
            "wq": [shape.n_heads * shape.head_dim, shape.dim],
            "wk": [kv_out, shape.dim],
            "wv": [kv_out, shape.dim],
            "wqkv": [shape.qkv_out_features, shape.dim],
        }
    if scenario_name == "lm_head":
        return {
            "hidden": [batch, seq, shape.dim],
            "weight": [shape.vocab_size, shape.dim],
            "tokens": batch * seq,
        }
    if scenario_name == "attention":
        return {
            "q": [batch, seq, shape.n_heads, shape.head_dim],
            "k": [batch, seq, shape.n_kv_heads, shape.head_dim],
            "v": [batch, seq, shape.n_kv_heads, shape.head_dim],
            "positions": [batch, seq],
            "packed_tokens": batch * seq,
            "max_seq_len": shape.max_seq_len,
        }
    raise ValueError(f"Unknown kernel scenario {scenario_name!r}")
