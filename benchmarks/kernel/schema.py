"""What a kernel-isolation benchmark *is*, separate from which ones exist.

The declaration vocabulary -- ``KernelWorkload``, ``CorrectnessCheck``,
``KernelArm``, ``KernelScenario``, ``KernelSpan``, the mode tuple -- plus the
arithmetic that follows from a (shape, workload) pair.
``benchmarks.kernel.registry`` holds the scenarios and
``benchmarks.kernel.spans`` holds the spans. Both import from here, and
this module names no scenario of its own.

The split is an import-graph rule, not tidiness.
``benchmarks.kernel.engine`` needs these types and must never import the
scenario data: the chain from the engine through the registry would make it
import every operation module and every model dependency behind them, and
per-arm process isolation rests on that not happening.
``tests/test_import_boundaries.py`` pins both halves.

``shape_summary`` is the one data-adjacent member. It keys on scenario names
as bare strings, so a new scenario edits this file as well as the registry.

The module is torch-free, so ``./run_bench.sh scenarios`` and ``--help``
list the arms without CUDA.
"""

from __future__ import annotations

import importlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Literal, get_args

from benchmarks.models.piper_qwen3.shape import PiperShape, shape_by_name


Mode = Literal["forward", "backward", "forward_backward"]
"""The timed operations an arm may expose, in the order the results print."""

MODES: tuple[Mode, ...] = get_args(Mode)
"""``Mode`` as a tuple, so the two cannot drift.

The union is a static aid alone, because this repository configures no type
checker. ``KernelScenario.__post_init__`` rejects an unknown mode when the
registry module loads.
"""


def resolve_symbol(path: str) -> Any:
    """Import ``module:function`` and return the function.

    Both sides resolve a dotted path: a worker resolves an arm's builder,
    and the parent resolves an arm's ``requirement``. Resolution by string
    keeps each side from importing what the other needs.
    """
    module_name, _, attribute = path.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


CORRECTNESS_FRAGMENT_KIND = "kernel_correctness_fragment"
"""The correctness fragment a worker writes and the parent reads."""

TIMING_FRAGMENT_KIND = "kernel_timing_fragment"
"""The timing fragment a worker writes and the parent reads.

Both names live in this declaration module rather than beside the reader,
so a timing worker does not import the merge and, with it, ``PiperShape``,
numpy and scipy.
"""

_FRAGMENT_SEPARATOR = "/"
"""The character an arm name may not carry into a filename."""

_FRAGMENT_REPLACEMENT = "-"
"""What replaces it.

A cross-engine scenario spells the engine and the profile as one name, such
as ``mcore/base``, and a fragment file is named after the arm that wrote it.
"""


def fragment_stem(arm_name: str) -> str:
    """The filename component for one arm, with no path separator left in it.

    A slash makes the fragment a file inside a directory no writer creates,
    so the worker writes nothing and the arm lands as ``failed`` with an
    unrecoverable cause. The substitution happens here, in the one place a
    name becomes a file, so the roster keeps the ``engine/profile``
    spelling. ``KernelScenario.__post_init__`` refuses two arms whose stems
    collide.
    """
    return arm_name.replace(_FRAGMENT_SEPARATOR, _FRAGMENT_REPLACEMENT)


def timing_fragment_path(
    fragments_dir: Path, arm_name: str, replicate: int
) -> Path:
    """Where one (arm, replicate)'s timing fragment lives.

    Both sides compute it. The parent plans and reads the files, and a
    worker that measures several replicates writes several of them. Neither
    side may learn the convention by string surgery on a path it was handed.
    """
    return (
        fragments_dir / f"timing__{fragment_stem(arm_name)}__r{replicate}.json"
    )


@dataclass(frozen=True)
class KernelWorkload:
    """What is run through the model, as opposed to what the model is.

    It stays apart from ``PiperShape``, so a batch change is not a model
    change. Everything geometric belongs to the shape, and each scenario's
    inputs builder computes whatever else it needs.
    """

    batch: int = 4
    seq_len: int = 1024


def validate_shape_and_workload(
    shape: PiperShape, workload: KernelWorkload
) -> None:
    """Raise ValueError on a shape/workload pair no scenario can run.

    Only a constraint that crosses the two lives here, because
    ``PiperShape`` already refuses a bad geometry. ``n_heads * head_dim ==
    dim`` is not an invariant, because a shape may write its own
    ``n_heads``, so every scenario spells the projection width as the
    product.
    """
    if workload.seq_len > shape.max_seq_len:
        raise ValueError(
            f"seq_len ({workload.seq_len}) exceeds max_seq_len "
            f"({shape.max_seq_len}); RoPE tables are sized by max_seq_len"
        )


def routing_divides_evenly(
    shape: PiperShape, workload: KernelWorkload
) -> bool:
    """Whether a scenario's synthetic workload routes rows evenly.

    It is scenario-scoped, because an uneven split must not fail a scenario
    that never splits rows. ``KernelScenario.requires_balanced_routing`` is
    the roster, so this function names no scenario. It tests the weaker of
    the two conditions a builder may need, and a builder that needs the
    stronger one asserts that itself.
    """
    return (
        workload.batch * workload.seq_len * shape.top_k
    ) % shape.num_experts == 0


DEFAULT_MODEL_SIZE = "30b-a3b"
"""The shape kernel-bench measures when the operator names none."""


def resolve_shape_and_workload(
    *,
    model_size: str = DEFAULT_MODEL_SIZE,
    batch: int | None = None,
    seq_len: int | None = None,
    max_seq_len: int | None = None,
) -> tuple[PiperShape, KernelWorkload]:
    """The geometry and the workload one kernel-bench run measures."""
    shape = shape_by_name(model_size)
    # max_seq_len first: a raised ceiling is what lets a long seq_len pass.
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
    ground-truth builder. ``kind`` selects the comparison: ``bitwise``,
    ``tolerance`` or ``fp64_ulp``. An informational check is recorded but
    never gates the run.

    ``max_rel_l2`` is the usual metric, because it alone stays meaningful
    when the compared tensors differ in magnitude or when elements land
    near zero. ``fp64_ulp`` reports the mean bf16 ULP error and suits an
    elementwise kernel such as RoPE, where this repository quotes about
    0.24 ULP; it is a mean because cancellation makes a per-element maximum
    meaningless. ``bitwise`` is for two implementations that must agree
    exactly.
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

    Attributes:
        builder: A dotted ``module:function`` path the GPU worker resolves.
            The function takes (shape, workload, inputs) and returns a
            BuiltArm whose per-mode closures are the timed operations.
        modes: Which operations the arm exposes. This declaration is the
            authority, and ``benchmarks.kernel.engine.run`` raises when the
            built arm disagrees.
        requirement: A dotted path to a parent-side, torch-free predicate
            called as ``predicate(shape, workload)``. It returns ``None``
            when the arm can run here, or the reason it cannot, and that
            reason reaches ``results.json`` as ``status_reason``. The
            parent checks it before it claims a GPU, and a builder that
            raises for an undeclared reason stays a failure.
        is_floor: Whether the arm is a bandwidth floor rather than an
            implementation. The parent needs it for the x-floor column.
        compiled: Whether the timed closure's entry point is a compiled
            callable, as production runs it. The test is where the compile
            sits, not who applied it: megatron-core binds ``jit_fuser`` to
            ``torch.compile`` at import, so an mcore arm whose entry point
            is one of those functions is compiled although no builder wraps
            it. A compiled megatron arm is still not the same treatment as
            a compiled titan arm, which takes a whole-module
            ``fullgraph=True`` compile, so a report may not pool the two.
        eager_reason: Why an eager arm is eager. ``compiled=False``
            requires one and ``compiled=True`` forbids one, because a
            harness choice, such as a bandwidth floor, and fidelity to what
            the engine runs in production read alike without it.
    """

    name: str
    description: str
    builder: str
    modes: tuple[Mode, ...]
    correctness: tuple[CorrectnessCheck, ...] = ()
    requires_gcc_toolset: bool = False
    requirement: str | None = None
    is_floor: bool = False
    compiled: bool = False
    eager_reason: str | None = None


@dataclass(frozen=True)
class KernelScenario:
    """A kernel family and its comparable implementation arms.

    Attributes:
        requires_balanced_routing: Marks a scenario whose inputs builder
            hands every expert an equal slice of the routed rows. The
            runner refuses a workload where they do not divide evenly,
            rather than round the split.
        comparisons: The ratio rows the results file publishes, as ordered
            ``(arm, opponent)`` pairs. ``None`` derives the single-anchor
            set: every arm except the anchor and the floors, against the
            anchor. An explicit tuple is exhaustive, and an empty tuple
            publishes no ratio at all.
    """

    name: str
    description: str
    inputs_builder: str
    reference_builder: str | None
    arms: tuple[KernelArm, ...]
    baseline_arm: str
    requires_balanced_routing: bool = False
    comparisons: tuple[tuple[str, str], ...] | None = None

    def __post_init__(self) -> None:
        # At import, so a mistyped opponent fails before any GPU work.
        self.arm(self.baseline_arm)
        for arm_name, opponent in self.comparisons or ():
            self.arm(arm_name)
            self.arm(opponent)
        # Colliding stems overwrite one timing file, so this runs at import.
        stems: dict[str, str] = {}
        for arm in self.arms:
            stem = fragment_stem(arm.name)
            if stem in stems:
                raise ValueError(
                    f"{self.name}: arms {stems[stem]!r} and {arm.name!r} both "
                    f"name the fragment file {stem!r}; one would overwrite the "
                    "other's samples"
                )
            stems[stem] = arm.name
        for arm in self.arms:
            # An empty or unknown mode tuple leaves the arm untimed, and a
            # floor reaches the merge with no samples and no complaint.
            if not arm.modes:
                raise ValueError(
                    f"{self.name}/{arm.name} declares no modes; an arm the "
                    "timing pass cannot time measures nothing"
                )
            unknown = [mode for mode in arm.modes if mode not in MODES]
            if unknown:
                raise ValueError(
                    f"{self.name}/{arm.name} declares unknown mode(s) "
                    f"{', '.join(sorted(unknown))}; expected one of "
                    f"{', '.join(MODES)}"
                )
            # Both directions are wrong alike: a treatment nobody stated.
            if not arm.compiled and not (arm.eager_reason or "").strip():
                raise ValueError(
                    f"{self.name}/{arm.name} is eager but declares no "
                    "eager_reason; a cross-engine ratio is a comparison of "
                    "two compile treatments, so each side must say what its "
                    "treatment is and why"
                )
            if arm.compiled and arm.eager_reason is not None:
                raise ValueError(
                    f"{self.name}/{arm.name} is compiled but declares "
                    f"eager_reason {arm.eager_reason!r}"
                )

    def arm(self, name: str) -> KernelArm:
        for arm in self.arms:
            if arm.name == name:
                return arm
        raise ValueError(
            f"Unknown arm {name!r} for kernel scenario {self.name!r}. "
            f"Available arms: {', '.join(arm.name for arm in self.arms)}"
        )

    def comparison_pairs(self) -> tuple[tuple[str, str], ...]:
        """The ``(arm, opponent)`` rows this scenario publishes."""
        if self.comparisons is not None:
            return self.comparisons
        return tuple(
            (arm.name, self.baseline_arm)
            for arm in self.arms
            if arm.name != self.baseline_arm and not arm.is_floor
        )

    @property
    def requires_gcc_toolset(self) -> bool:
        return any(arm.requires_gcc_toolset for arm in self.arms)


@dataclass(frozen=True)
class SpanParts:
    """Which arm of each enclosed scenario one span arm replaces.

    ``parts`` is positional: one arm name per entry of
    ``KernelSpan.scenarios``, in the model's own order through the cuts.
    The correspondence is declared, never inferred, because a span arm may
    replace arms of other names, and may name an implementation no enclosed
    scenario holds.
    """

    arm: str
    parts: tuple[str, ...]


@dataclass(frozen=True)
class KernelSpan:
    """One implementation that fuses across a scenario cut.

    A span is declared over an ordered scenario range, and its claim is the
    span against the sum of the scenarios it replaces. A span result holds
    two totals, which ``benchmarks.kernel.results.schema`` keeps in
    separate fields.

    A span composes a ``KernelScenario`` rather than subclasses one, so
    ``benchmarks.kernel.engine`` never learns that spans exist and
    ``isinstance(span, KernelScenario)`` is false.

    A span ratio carries a bias the declaration cannot remove. The parts
    total pays one host dispatch chain per enclosed scenario and the span
    pays one, and roughly 85% of a kernel number here is host dispatch, so
    the bias grows with the length of the range. The engine states it on
    every span.

    Attributes:
        measurement: The span's own head-to-head: its arms, its inputs
            builder, its gates and its within-span comparisons.
        scenarios: The ordered range.
        parts: What each arm replaces; see ``SpanParts``.
            ``validate_span_parts`` checks that each named part exists.
    """

    measurement: KernelScenario
    scenarios: tuple[str, ...]
    parts: tuple[SpanParts, ...]

    def __post_init__(self) -> None:
        # At import, so a mistyped range fails before any GPU work.
        if len(self.scenarios) < 2:
            raise ValueError(
                f"{self.name}: a span replaces at least two scenarios, got "
                f"{list(self.scenarios)}. A span over one scenario is that "
                "scenario, and its two totals would be one number twice"
            )
        seen: set[str] = set()
        for scenario_name in self.scenarios:
            if scenario_name in seen:
                raise ValueError(
                    f"{self.name}: scenario {scenario_name!r} appears twice "
                    "in the range; the parts sum would count one cut twice"
                )
            seen.add(scenario_name)
        declared: dict[str, SpanParts] = {}
        for entry in self.parts:
            # Raises on an arm the span does not declare.
            self.arm(entry.arm)
            if entry.arm in declared:
                raise ValueError(
                    f"{self.name}: arm {entry.arm!r} declares its parts "
                    "twice; a span arm has one claim, and two totals under "
                    "one name say which of them is it"
                )
            declared[entry.arm] = entry
            if len(entry.parts) != len(self.scenarios):
                raise ValueError(
                    f"{self.name}/{entry.arm}: {len(entry.parts)} part arm(s) "
                    f"for {len(self.scenarios)} scenario(s). The "
                    "correspondence is positional, so a short tuple drops a "
                    "cut from the sum without saying so"
                )
            for scenario_name, part_arm in zip(self.scenarios, entry.parts):
                if not part_arm.strip():
                    raise ValueError(
                        f"{self.name}/{entry.arm}: the part in "
                        f"{scenario_name!r} has no name"
                    )
        # Every arm, because a span arm with no parts states no claim.
        missing = [arm.name for arm in self.arms if arm.name not in declared]
        if missing:
            raise ValueError(
                f"{self.name}: arm(s) {', '.join(missing)} declare no parts. "
                "A span arm is compared against the sum of what it replaces, "
                "so an arm that names no parts publishes no claim"
            )

    # Read-only forwarding, so no caller reaches through ``.measurement``.
    @property
    def name(self) -> str:
        return self.measurement.name

    @property
    def description(self) -> str:
        return self.measurement.description

    @property
    def arms(self) -> tuple[KernelArm, ...]:
        return self.measurement.arms

    @property
    def baseline_arm(self) -> str:
        return self.measurement.baseline_arm

    @property
    def requires_gcc_toolset(self) -> bool:
        return self.measurement.requires_gcc_toolset

    def arm(self, name: str) -> KernelArm:
        return self.measurement.arm(name)

    def comparison_pairs(self) -> tuple[tuple[str, str], ...]:
        """The within-span ``(arm, opponent)`` rows, as a scenario's are.

        The span-vs-parts claim is separate, because one side of it sums
        over scenarios and has no arm of its own.
        """
        return self.measurement.comparison_pairs()

    def parts_for(self, arm_name: str) -> tuple[tuple[str, str], ...]:
        """``(scenario, arm)`` pairs one span arm replaces, in range order."""
        for entry in self.parts:
            if entry.arm == arm_name:
                return tuple(zip(self.scenarios, entry.parts))
        raise ValueError(
            f"{self.name}: arm {arm_name!r} declares no parts"
        )


def validate_span_parts(
    span: KernelSpan, scenarios: dict[str, KernelScenario]
) -> None:
    """Refuse a span whose parts do not exist in the scenarios it names.

    It is the other half of ``KernelSpan.__post_init__``, split out because
    it needs the scenario registry that ``schema.py`` may not import. The
    span registry calls it at import.

    The mode check is the subtle one. A parts sum is per mode, so a part
    that lacks a mode the span arm declares leaves that mode's total short
    of a term.
    """
    for entry in span.parts:
        span_arm = span.arm(entry.arm)
        for scenario_name, part_arm in zip(span.scenarios, entry.parts):
            scenario = scenarios.get(scenario_name)
            if scenario is None:
                raise ValueError(
                    f"{span.name}: unknown scenario {scenario_name!r} in the "
                    f"range. Available: {', '.join(sorted(scenarios))}"
                )
            part = scenario.arm(part_arm)
            absent = [
                mode for mode in span_arm.modes if mode not in part.modes
            ]
            if absent:
                raise ValueError(
                    f"{span.name}/{entry.arm} declares mode(s) "
                    f"{', '.join(sorted(absent))} that its part "
                    f"{scenario_name}/{part_arm} does not, so that mode's "
                    "parts total would sum fewer terms than the range holds"
                )


def shape_summary(
    scenario_name: str, shape: PiperShape, workload: KernelWorkload
) -> dict[str, object]:
    """Derived shapes recorded in the manifest for provenance.

    Where the two engines consume different layouts of the same elements,
    both layouts are recorded. A ``_thd``, ``_TBD`` or ``_TD`` entry is a
    free view of one contiguous allocation, taken outside every timed
    closure, so it is a label and not a measured difference.
    """
    batch, seq = workload.batch, workload.seq_len
    if scenario_name == "rope":
        return {
            "q_titan_BLNH": [batch, seq, shape.n_heads, shape.head_dim],
            "k_titan_BLNH": [batch, seq, shape.n_kv_heads, shape.head_dim],
            "q_mcore_THD": [batch * seq, shape.n_heads, shape.head_dim],
            "k_mcore_THD": [batch * seq, shape.n_kv_heads, shape.head_dim],
            # Two spellings of one packing. Only the token total derives here.
            "positions": [batch, seq],
            "cu_seqlens_total": batch * seq,
            # Two table sizes. Each covers every position.
            "titan_table_rows": shape.max_seq_len,
            "mcore_freqs": [seq, 1, 1, shape.head_dim],
            "rotated_rows": batch * seq * (shape.n_heads + shape.n_kv_heads),
        }
    if scenario_name == "lm_head":
        return {
            "hidden": [batch, seq, shape.dim],
            "weight": [shape.vocab_size, shape.dim],
            "tokens": batch * seq,
        }
    if scenario_name == "qk_norm":
        mcore_fused_row = (shape.heads_per_group + 2) * shape.head_dim
        return {
            # Not one tensor in two layouts here: megatron's key stays a
            # strided view of the fused buffer, and TE copies it when timed.
            "q_titan_BLNH": [batch, seq, shape.n_heads, shape.head_dim],
            "k_titan_BLNH": [batch, seq, shape.n_kv_heads, shape.head_dim],
            "qkv_mcore_fused_SBGR": [
                seq,
                batch,
                shape.n_kv_heads,
                mcore_fused_row,
            ],
            "q_mcore_SBNH": [seq, batch, shape.n_heads, shape.head_dim],
            "k_mcore_SBNH": [seq, batch, shape.n_kv_heads, shape.head_dim],
            # Derived, never a literal, so it cannot outlive the fact.
            # ``tests/test_kernel_qk_norm.py`` binds it to the real tensor.
            "k_mcore_is_a_strided_view": mcore_fused_row != shape.head_dim,
            "weight": [shape.head_dim],
            # q and k differ in row count, so the scenario times a pair.
            "rows": batch * seq * (shape.n_heads + shape.n_kv_heads),
            "reduction_length": shape.head_dim,
        }
    if scenario_name == "attention_core":
        return {
            # Not one tensor in two layouts here: megatron's value stays a
            # strided view of the fused buffer, and TE copies it when timed.
            "q_titan_BLNH": [batch, seq, shape.n_heads, shape.head_dim],
            "k_titan_BLNH": [batch, seq, shape.n_kv_heads, shape.head_dim],
            "v_titan_BLNH": [batch, seq, shape.n_kv_heads, shape.head_dim],
            "qkv_mcore_fused_TGR": [
                batch * seq,
                shape.n_kv_heads,
                (shape.heads_per_group + 2) * shape.head_dim,
            ],
            "q_mcore_THD": [batch * seq, shape.n_heads, shape.head_dim],
            "k_mcore_THD": [batch * seq, shape.n_kv_heads, shape.head_dim],
            "v_mcore_THD": [batch * seq, shape.n_kv_heads, shape.head_dim],
            "v_mcore_is_a_strided_view": True,
            # What each engine returns: one order, two spellings.
            "out_titan_BLNH": [batch, seq, shape.n_heads, shape.head_dim],
            "out_mcore_TD": [batch * seq, shape.n_heads * shape.head_dim],
            # Three spellings of one mask. Only the token total derives here.
            "positions": [batch, seq],
            "packed_tokens": batch * seq,
            "flex_block_size": 128,
            "flex_flash_block_size": [256, 128],
            # Pinned to the sequence length on both engines.
            "max_seqlen": seq,
            "max_seq_len": shape.max_seq_len,
        }
    if scenario_name == "attn_out_proj":
        # The product, not ``shape.dim``: a shape may write its own n_heads.
        in_features = shape.n_heads * shape.head_dim
        return {
            "x": [batch, seq, in_features],
            "x_thd": [batch * seq, 1, in_features],
            "weight": [shape.dim, in_features],
            "grad_out": [batch, seq, shape.dim],
            "tokens": batch * seq,
        }
    if scenario_name == "attn_residual":
        return {
            "attn_out": [batch, seq, shape.dim],
            "residual": [batch, seq, shape.dim],
            "grad_out": [batch, seq, shape.dim],
            "attn_out_thd": [batch * seq, 1, shape.dim],
            "tokens": batch * seq,
            # One flop per element, and two reads per write.
            "elements": batch * seq * shape.dim,
        }
    if scenario_name == "ffn_norm":
        return {
            "x": [batch, seq, shape.dim],
            "weight": [shape.dim],
            # Both engines flatten, so the layout never reaches the kernel.
            "rows": batch * seq,
            "row_width": shape.dim,
        }
    if scenario_name == "moe_residual":
        return {
            "x": [batch, seq, shape.dim],
            "residual": [batch, seq, shape.dim],
            "grad_out": [batch, seq, shape.dim],
            "x_thd": [batch * seq, 1, shape.dim],
            # The operation is elementwise, so this is the whole of its work.
            "elements": batch * seq * shape.dim,
        }
    if scenario_name == "final_norm":
        return {
            "x": [batch, seq, shape.dim],
            "weight": [shape.dim],
            # The reduction is over the last dimension, so these are rows.
            "rows": batch * seq,
            "reduction_length": shape.dim,
        }
    if scenario_name == "embedding_stage":
        # Both native output layouts, and no layout copy on either side.
        tokens = batch * seq
        return {
            "tokens": [batch, seq],
            "tokens_thd": [1, tokens],
            "weight": [shape.vocab_size, shape.dim],
            "out_titan_BLD": [batch, seq, shape.dim],
            "out_mcore_TBD": [tokens, 1, shape.dim],
            "grad_out": [batch, seq, shape.dim],
            "rows_gathered": tokens,
        }
    if scenario_name == "qkv_prep":
        kv_out = shape.n_kv_heads * shape.head_dim
        return {
            "x": [batch, seq, shape.dim],
            "x_thd": [batch * seq, 1, shape.dim],
            "norm_weight": [shape.dim],
            # All four weight forms, because each arm holds only its own:
            # three unfused matrices, and ``wqkv`` as their interleave.
            "wq": [shape.n_heads * shape.head_dim, shape.dim],
            "wk": [kv_out, shape.dim],
            "wv": [kv_out, shape.dim],
            "wqkv": [shape.qkv_out_features, shape.dim],
            # The split and the layout work are inside the timed region.
            "q": [batch, seq, shape.n_heads, shape.head_dim],
            "k": [batch, seq, shape.n_kv_heads, shape.head_dim],
            "v": [batch, seq, shape.n_kv_heads, shape.head_dim],
            "tokens": batch * seq,
        }
    if scenario_name == "lm_head_projection":
        # The ``cross_entropy`` branch labels megatron SBHD on purpose; this
        # one labels what the THD driver delivers. Neither is the other.
        return {
            "x_titan_BLD": [batch, seq, shape.dim],
            "x_mcore_TBD": [batch * seq, 1, shape.dim],
            "weight": [shape.vocab_size, shape.dim],
            "grad_out": [batch, seq, shape.vocab_size],
            "tokens": batch * seq,
            # The projection output dominates memory: 1.16 GiB of bf16
            # at the default workload against 8 MiB of input.
            "vocab_size": shape.vocab_size,
        }
    if scenario_name == "cross_entropy":
        # Each mcore arm materializes its [L, B, V] copy at build time, so
        # no timed closure pays for the harness's storage order.
        return {
            "logits_titan_BLV": [batch, seq, shape.vocab_size],
            "logits_mcore_SBV": [seq, batch, shape.vocab_size],
            # [B, L] on both sides; megatron transposes it itself.
            "labels": [batch, seq],
            "tokens": batch * seq,
            # The logits dominate memory: 1.16 GiB of bf16 at the default
            # workload against 32 KiB of labels.
            "vocab_size": shape.vocab_size,
        }
    if scenario_name == "moe_router":
        tokens = batch * seq
        return {
            "x": [batch, seq, shape.dim],
            "x_thd": [tokens, 1, shape.dim],
            "gate_weight": [shape.num_experts, shape.dim],
            # What each engine returns. The two differ in kind, not only in
            # layout, and both are small next to x, which dominates traffic.
            "probs_mcore_TE": [tokens, shape.num_experts],
            "routing_map_mcore_TE": [tokens, shape.num_experts],
            "scores_titan_BLE": [batch, seq, shape.num_experts],
            "topk_probs_titan_BLK": [batch, seq, shape.top_k],
            "topk_indices_titan_BLK": [batch, seq, shape.top_k],
            "tokens": tokens,
            "experts": shape.num_experts,
            "top_k": shape.top_k,
            # Negligible, so the scenario is bandwidth and dispatch bound.
            "gate_gemm_flops": 2 * tokens * shape.dim * shape.num_experts,
        }
    if scenario_name == "dispatch_permute":
        rows = batch * seq * shape.top_k
        return {
            "x_titan_BLD": [batch, seq, shape.dim],
            "x_mcore_TBD": [batch * seq, 1, shape.dim],
            # One routing decision in the two forms the engines consume.
            # Both are fp32 on both engines, so this cut has no precision gap.
            "topk_expert_ids_titan_TK": [batch * seq, shape.top_k],
            "topk_scores_titan_TK": [batch * seq, shape.top_k],
            "probs_mcore_TE": [batch * seq, shape.num_experts],
            "routing_map_mcore_TE": [batch * seq, shape.num_experts],
            # An identity dispatch returns batch * seq rows; this cut
            # returns batch * seq * top_k, which every arm's guard reads.
            "permuted_tokens": [rows, shape.dim],
            "permuted_probs": [rows],
            # Exact by construction; an uneven split skips the scenario loudly.
            "tokens_per_expert": (
                [rows // shape.num_experts] * shape.num_experts
            ),
            # A permute is a gather, so these are the whole of its device work.
            "permuted_elements": rows * shape.dim,
        }
    if scenario_name == "expert_mlp":
        rows = batch * seq * shape.top_k
        per_expert = rows // shape.num_experts
        return {
            # The permuted rows both engines consume, with no second layout.
            "x": [rows, shape.dim],
            "grad_out": [rows, shape.dim],
            # Read by the megatron arms alone, so the scenario stays
            # within-engine. Titan applies the probabilities in combine.
            "probs": [rows],
            # A synthetic even split. Only the counts have to match.
            "tokens_per_expert": [per_expert] * shape.num_experts,
            # All four weight forms, because each engine holds only its own.
            # ``benchmarks/models/piper_qwen3/megatron_weights.py`` maps them.
            "w1": [shape.num_experts, shape.moe_hidden_dim, shape.dim],
            "w2": [shape.num_experts, shape.dim, shape.moe_hidden_dim],
            "w3": [shape.num_experts, shape.moe_hidden_dim, shape.dim],
            "w13_titan_fused": [
                shape.num_experts,
                shape.moe_hidden_dim,
                2,
                shape.dim,
            ],
            "linear_fc1_mcore_per_expert": [
                2 * shape.moe_hidden_dim,
                shape.dim,
            ],
            "linear_fc2_mcore_per_expert": [shape.dim, shape.moe_hidden_dim],
        }
    if scenario_name == "moe_combine":
        tokens = batch * seq
        rows = tokens * shape.top_k
        return {
            # Read for its shape alone, so its values reach no number.
            "x": [batch, seq, shape.dim],
            "x_thd": [tokens, 1, shape.dim],
            # The canonical row order both engines' permutations produce.
            "expert_out": [rows, shape.dim],
            "grad_out": [tokens, shape.dim],
            # One decision in two forms. The builder refuses a mismatched pair.
            "routing_map": [tokens, shape.num_experts],
            "probs": [tokens, shape.num_experts],
            "topk_expert_ids": [tokens, shape.top_k],
            "tokens_per_expert": [rows // shape.num_experts]
            * shape.num_experts,
            # Read this many rows, write this many, add top_k into each.
            "rows_combined": rows,
            "rows_out": tokens,
        }
    raise ValueError(f"Unknown kernel scenario {scenario_name!r}")
