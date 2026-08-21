"""What a kernel-isolation benchmark *is*, separate from which ones exist.

The declaration vocabulary -- ``KernelWorkload``, ``CorrectnessCheck``,
``KernelArm``, ``KernelScenario``, ``KernelSpan``, the mode tuple -- plus the
arithmetic that
follows from a (shape, workload) pair: what a run may ask for
(``resolve_shape_and_workload``), what it may not (``validate_shape_and_
workload``, ``routing_divides_evenly``), and the derived tensor shapes both
systems record (``shape_summary``). ``benchmarks.kernel.registry`` holds the
scenarios themselves and ``benchmarks.kernel.spans`` holds the spans; both
import from here, and nothing here knows that ``rope`` or ``attention_core``
exist as objects.

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

import importlib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from benchmarks.models.piper_qwen3.shape import PiperShape, shape_by_name


MODES = ("forward", "backward", "forward_backward")


def resolve_symbol(path: str) -> Any:
    """Import ``module:function`` and return the function.

    Here rather than in the engine because both sides resolve a dotted path
    now: a worker resolves an arm's builder, and the **parent** resolves an
    arm's ``requirement``. Resolution by string is what keeps either side
    from importing what the other needs -- the parent must never import an
    ``operations/`` module, and the engine must never import the registry.
    Two copies of these three lines would let the two conventions drift.
    """
    module_name, _, attribute = path.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


# The two fragment shapes a worker writes and the parent reads, named once.
# They live here, in the declaration module both sides already import, rather
# than beside the reader: the producer is
# ``benchmarks.kernel.engine.run`` and the consumer is
# ``benchmarks.kernel.results.merge``, so naming them in the consumer made
# the producer import it -- and with it ``PiperShape``, numpy and scipy, in
# every timing worker.
CORRECTNESS_FRAGMENT_KIND = "kernel_correctness_fragment"
TIMING_FRAGMENT_KIND = "kernel_timing_fragment"

# The character an arm name may not carry into a filename, and what replaces
# it. Cross-engine scenarios spell the engine and the profile as one name --
# ``mcore/base`` against ``titan`` -- and a fragment file is named after the
# arm that wrote it.
_FRAGMENT_SEPARATOR = "/"
_FRAGMENT_REPLACEMENT = "-"


def fragment_stem(arm_name: str) -> str:
    """The filename component for one arm, with no path separator left in it.

    ``Path("fragments") / "timing__mcore/base__r0.json"`` is a file inside a
    ``timing__mcore`` **directory**, not a file with a slash in its name. No
    writer creates that directory, so the timing worker fails to write its
    fragment, the parent reads none, and the arm lands as ``failed`` with a
    cause no reader of the results can recover. Every cross-engine scenario in
    the partition would hit it.

    The separator is replaced here, in the one place a name becomes a file, so
    the roster keeps the ``engine/profile`` spelling that makes an arm
    self-describing. ``KernelScenario.__post_init__`` refuses two arms whose
    stems collide, which is the failure this substitution could otherwise
    introduce.
    """
    return arm_name.replace(_FRAGMENT_SEPARATOR, _FRAGMENT_REPLACEMENT)


def timing_fragment_path(
    fragments_dir: Path, arm_name: str, replicate: int
) -> Path:
    """Where one (arm, replicate)'s timing fragment lives.

    Here rather than in the parent because both sides now compute it. The
    parent plans and reads the files; a worker that measures several
    replicates of one arm writes several of them, and it must not learn the
    convention by string surgery on a path it was handed.
    """
    return (
        fragments_dir / f"timing__{fragment_stem(arm_name)}__r{replicate}.json"
    )


@dataclass(frozen=True)
class KernelWorkload:
    """What is run through the model, as opposed to what the model is.

    Kept separate from ``PiperShape`` so batch size is independent of
    geometry: the same shape is measurable at any batch, and a batch change
    is not a model change. Everything geometric -- dim, head counts, expert
    width, vocabulary, ``max_seq_len`` -- belongs to the shape, and no
    per-family shapes live in either: each scenario's inputs builder computes
    what it needs (expert_mlp rows = batch * seq_len * top_k, fused-qkv out
    features = (n_heads + 2 * n_kv_heads) * head_dim, cross_entropy tokens =
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
    """Whether a scenario's synthetic workload routes rows evenly.

    Scenario-scoped on purpose. Several inputs builders hand every expert an
    equal slice of ``batch * seq_len * top_k`` rows, and each of them declares
    ``requires_balanced_routing``; an uneven split is those scenarios'
    problem and must not fail a scenario that never splits rows at all.
    ``KernelScenario.requires_balanced_routing`` is the roster, and
    ``tests/test_kernels.py`` pins it exhaustively, so this function names no
    scenario itself.

    It tests the weaker of the two conditions such a builder may need. A
    builder that also needs ``batch * seq_len`` to divide by ``num_experts``
    -- ``dispatch_permute`` does, because its per-token expert offsets come
    from a permutation reduced modulo the expert count -- asserts that itself,
    with named numbers. Moving the stronger condition here would make
    ``expert_mlp`` skip workloads it can measure.
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
    # attention_core sweep raises the ceiling precisely in order to run
    # past it.
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
    ``compiled`` records that the timed closure's entry point is a compiled
    callable, as production runs it.

    **The test is where the compile sits, not who applied it.** Our builders
    apply ``torch.compile`` to every titan module arm, so for those two
    readings agree. They part on the megatron side, because megatron-core
    binds ``jit_fuser`` to ``torch.compile`` at import
    (``megatron/core/jit.py``) and decorates 41 of its own functions with it.
    ``attn_residual/mcore/base`` and ``moe_residual/mcore/base`` therefore
    declare ``compiled=True`` although no builder wraps them: the timed
    closure calls ``bias_dropout_add_fused_train``, which *is* the
    ``torch.compile`` wrapper, and the compile is the treatment the published
    row measures. Declaring them eager would put "eager against eager" in the
    manifest for a row whose whole delta is that compile.

    The counter-example is ``cross_entropy/mcore/ce_native``, which declares
    ``compiled=False``. Its entry point is a plain method, and the
    ``@jit_fuser`` regions are helpers nested inside it. The arm is eager
    where it is timed from.

    A megatron ``compiled=True`` arm is **not** the same treatment as a titan
    one, and a report may not pool them: megatron applied the compile at
    import over one function, and the harness applied a whole-module
    ``fullgraph=True`` compile over a titan arm. Each arm's ``description``
    says which.

    **A requirement that depends on the workload is declared, not
    discovered.** ``requires_gcc_toolset`` is a property of the *host*, and
    the parent can answer it before it knows the shape. ``requirement``
    covers the other case: an arm this shape or this workload cannot run --
    an unfused attention arm whose dense score tensor grows with the square
    of the sequence length is the case that forced it. It is a dotted
    ``module:function`` path, resolved by ``resolve_symbol`` **in the
    parent** and called as ``predicate(shape, workload)``. It returns
    ``None`` when the arm can run here, or the reason it cannot -- and that
    reason is what reaches ``results.json`` as ``status_reason``, so a
    reader learns why an arm is absent without holding the registry.

    The module it names must therefore be parent-side and torch-free, like
    this one and unlike every ``operations/`` module. A test asserts it.

    **Rejected: catch the builder's exception instead.** A ``try`` around
    the build turns a bug into a skipped arm: the roster shortens for a
    reason nobody declared, the real cause is a log line nobody reads, and
    the run still exits zero. A declared requirement names the condition in
    advance, records it in the manifest and in the results, and is checked
    before a GPU is claimed. A builder that raises for an undeclared reason
    stays a failure, because that is what it is.

    **An eager arm states why it is eager.** ``compiled=False`` requires an
    ``eager_reason``, and ``compiled=True`` forbids one. The reasons are not
    interchangeable, and a reader who cannot tell them apart will misread the
    row: ``qk_norm/copy_floor`` is eager because a bandwidth floor is not an
    implementation, while a megatron-core arm is eager because megatron
    compiles no whole layer end to end, and a TorchTitan module that sits
    outside every compiled region -- ``Decoder.norm`` and its siblings, which
    ``apply_compile`` never reaches -- is eager for the mirror image of that
    reason. The first is a harness choice. The other two are fidelity to what
    the engine does in production, and compiling either would be an arbitrary
    asymmetry the published ratio absorbs silently.

    **This declaration is the authority, and the builder must agree with
    it.** ``modes`` says which operations the arm exposes and ``is_floor``
    says whether it is a bandwidth floor rather than an implementation.
    ``benchmarks.kernel.engine.run`` raises when the built arm disagrees,
    instead of letting the builder decide silently. A mode the registry does
    not declare would otherwise be timed and published under a label nothing
    describes, and a floor known only to its builder cannot be read by the
    parent that computes the x-floor column. ``KernelScenario`` checks the
    other half at import: a declared mode must be one ``MODES`` holds, and
    the tuple must not be empty, or the timing pass runs nothing for the arm
    however well the builder agrees with it.
    """

    name: str
    description: str
    builder: str
    modes: tuple[str, ...]
    correctness: tuple[CorrectnessCheck, ...] = ()
    requires_gcc_toolset: bool = False
    requirement: str | None = None
    is_floor: bool = False
    compiled: bool = False
    eager_reason: str | None = None


@dataclass(frozen=True)
class KernelScenario:
    """A kernel family and its comparable implementation arms.

    ``requires_balanced_routing`` marks a scenario whose inputs builder hands
    every expert an equal slice of the routed rows; the runner refuses to run
    it on a workload where they do not divide evenly, rather than silently
    rounding the split.

    ``comparisons`` declares which ratio rows the results file publishes, as
    ordered ``(arm, opponent)`` pairs. ``None`` means "derive the usual
    single-anchor set": every arm except the anchor and the floors, each
    against the anchor. An explicit tuple is exhaustive, and the empty tuple
    is a scenario that publishes no ratio at all -- which a scenario holding
    two engines whose cut is not like-for-like must be able to say. It
    replaces the former per-arm ``compare_to``, which could name a different
    opponent but could not decline a row.
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
        # At import, so a mistyped opponent fails when the registry loads
        # rather than after a GPU has measured every arm.
        self.arm(self.baseline_arm)
        for arm_name, opponent in self.comparisons or ():
            self.arm(arm_name)
            self.arm(opponent)
        # And a mistyped mode with it. ``_seeded_build`` makes the declaration
        # authoritative over the builder, but it only asks the two to agree:
        # an arm declaring "fwd" and a builder supplying a "fwd" closure pass
        # it, and then ``run_timing_pass`` -- which iterates ``MODES`` --
        # never times the arm. A non-floor arm raises later in
        # ``_heaviest_mode``; a floor skips the memory pass and says nothing
        # at all, and reaches the merge with no samples. An empty tuple is the
        # same failure spelled differently.
        # Two arms whose fragment stems collide would overwrite each other's
        # timing file, and the second would be published under the first's
        # samples. Checked at import, because the alternative is to discover it
        # after a GPU has measured both.
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
            # Both directions, because both are wrong in the same way: a
            # treatment nobody stated. An unexplained eager arm reads as an
            # oversight when it is usually fidelity, and a reason attached to
            # a compiled arm describes a treatment the arm did not receive.
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

    ``parts`` is **positional**: one arm name per entry of
    ``KernelSpan.scenarios``, in that order. The order is the model's own
    order through the cuts, which is why a span declares an *ordered* range
    rather than a set.

    **The correspondence is declared, never inferred.** A span arm named
    ``titan`` does not necessarily replace an arm named ``titan`` in each
    enclosed scenario: the ``fused_linear_ce`` span replaces
    ``lm_head_projection/titan`` and ``cross_entropy/titan/full_logits``, two
    different names and neither of them the span arm's own. A span arm may
    also name an implementation no enclosed scenario has --
    ``mcore/fused_residual_rmsnorm`` is the fusion under test, and what it
    replaces is ``mcore/base`` at every cut it crosses. A name-based rule
    expresses neither case, and a name-based rule that silently found nothing
    would publish a span with no claim in it.
    """

    arm: str
    parts: tuple[str, ...]


@dataclass(frozen=True)
class KernelSpan:
    """One implementation that fuses across a scenario cut.

    A scenario cuts the model at one boundary and ranks the implementations
    at that cut. An implementation that fuses *across* a cut belongs to no
    single scenario, so a span is declared over an **ordered scenario range**
    and its claim is the span against the **sum of the scenarios it
    replaces**. A span result therefore holds two totals, and
    ``benchmarks.kernel.results.schema`` keeps them in separate fields.

    **A span composes a ``KernelScenario`` rather than subclassing one.**
    ``measurement`` is the span's own head-to-head: its arms, its inputs
    builder, its gates, its within-span comparisons. Two things follow.

    * ``benchmarks.kernel.engine`` never learns that spans exist. Both passes
      take a ``KernelScenario``, and a span hands them ``measurement``, which
      is one. The engine gains no import and no branch, which is what keeps
      per-arm process isolation intact.
    * ``isinstance(span, KernelScenario)`` is **false**. A span added to
      ``KERNEL_SCENARIOS`` by mistake would otherwise run as a bare scenario
      and publish one of its two totals under a name that promises both.

    **A span ratio carries a bias the declaration cannot remove.** The parts
    total pays one host dispatch chain per enclosed scenario and the span
    pays one, and roughly 85% of a kernel number here is host dispatch. So a
    ratio below 1.0 is fusion **plus** the N-1 chains the harness stopped
    paying, and the effect grows with the length of the range. It is a
    property of the range length rather than of what a span fuses, so the
    engine states it on every span -- printed under the table and recorded in
    the results file -- and a declaration cannot forget it.

    ``scenarios`` names the range. ``parts`` says what each arm replaces; see
    ``SpanParts``. The check that a named part arm *exists* in the named
    scenario needs both objects at once and therefore lives in
    ``validate_span_parts``, which the span registry calls at import --
    ``schema.py`` may not import ``registry.py``.
    """

    measurement: KernelScenario
    scenarios: tuple[str, ...]
    parts: tuple[SpanParts, ...]

    def __post_init__(self) -> None:
        # At import, so a mistyped range fails when the span registry loads
        # rather than after a GPU has measured every arm of six scenarios.
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
            # Raises on an arm the span does not declare, with the roster in
            # the message.
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
        # Every arm, because a span arm with no parts entry is an arm whose
        # claim cannot be stated -- and stating that claim is the whole of
        # what a span is for. An arm that wants no parts row belongs in a
        # scenario.
        missing = [arm.name for arm in self.arms if arm.name not in declared]
        if missing:
            raise ValueError(
                f"{self.name}: arm(s) {', '.join(missing)} declare no parts. "
                "A span arm is compared against the sum of what it replaces, "
                "so an arm that names no parts publishes no claim"
            )

    # The roster is the span's own, so a caller that needs it does not reach
    # through ``.measurement``. Read-only forwarding: a span never edits the
    # scenario it holds.
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

        Separate from the span-vs-parts claim, which is not an arm pair: one
        side of it is a sum over scenarios and has no arm of its own.
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

    The other half of ``KernelSpan.__post_init__``, split out because it
    needs the scenario registry and ``schema.py`` may not import it. The span
    registry calls this at import, so a mistyped part arm fails when the
    module loads rather than as a missing row after a GPU has run.

    The mode check is the subtle one. A span arm's parts sum is per mode, so
    a part that does not declare a mode the span arm declares contributes
    nothing to that mode's total -- and the row would then compare the span
    against a sum of fewer terms than the range holds. Caught here, the
    declaration is refused; caught at merge time, it is a warning about work
    already paid for.
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
    """Derived shapes recorded in the manifest for provenance."""
    batch, seq = workload.batch, workload.seq_len
    if scenario_name == "rope":
        return {
            # Both forms, because both are read. The THD pair is a view of the
            # BLNH pair and not a second allocation: [B, L, N, H] is contiguous,
            # so [B*L, N, H] is the same storage in the same order, and this
            # scenario charges no layout conversion to either engine.
            "q_titan_BLNH": [batch, seq, shape.n_heads, shape.head_dim],
            "k_titan_BLNH": [batch, seq, shape.n_kv_heads, shape.head_dim],
            "q_mcore_THD": [batch * seq, shape.n_heads, shape.head_dim],
            "k_mcore_THD": [batch * seq, shape.n_kv_heads, shape.head_dim],
            # The two spellings of one packing: positions is what titan's modules
            # index with, cu_seqlens is what megatron's THD path derives the same
            # positions from. The document count is a property of the seeded
            # draw, so only the token total is derivable here.
            "positions": [batch, seq],
            "cu_seqlens_total": batch * seq,
            # Titan precomputes a table of max_seq_len rows; megatron builds one
            # of rotary_seq_len rows per step, which our driver pins to seq_len.
            # Both cover every position, and neither number is the other's.
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
            # Both engine-native forms, because both are read, and they are
            # NOT the same tensors. Titan materializes two contiguous
            # tensors. Megatron's QKV GEMM writes one fused buffer and splits
            # it; the query leaves the buffer because megatron reshapes it,
            # and the key does not, so k_layernorm norms a non-contiguous
            # view. TE copies it inside the timed call, which is the half of
            # qkv_prep's deferral this scenario collects. The row width is
            # what makes that view strided, so it is recorded.
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
            # Derived from the two widths above, never asserted as a literal.
            # The key stays a view of the fused buffer, so it is strided
            # exactly while that buffer's row is wider than the key's own
            # slice. A literal True would keep claiming a strided key after
            # the inputs builder stopped producing one, which is the silent
            # failure the whole arm exists to prevent, one layer out.
            #
            # This module is torch-free and may not import an operations
            # module, so it cannot read the tensor itself. The binding to the
            # real tensor is a test:
            # ``tests/test_kernel_qk_norm.py`` compares this value against
            # ``not qk_norm_inputs(...).k_SBNH.is_contiguous()``.
            "k_mcore_is_a_strided_view": mcore_fused_row != shape.head_dim,
            "weight": [shape.head_dim],
            # The reduction runs over the last dimension alone, so every
            # leading dimension is a row count. This is the number that says
            # q and k are not the same size, and therefore why the scenario
            # times a pair rather than one module twice.
            "rows": batch * seq * (shape.n_heads + shape.n_kv_heads),
            "reduction_length": shape.head_dim,
        }
    if scenario_name == "attention_core":
        return {
            # Both engine-native forms, because both are read, and they are
            # NOT the same tensors. Titan materializes three contiguous
            # tensors. Megatron's QKV GEMM writes one fused buffer, and the
            # query and the key leave it when the norm and the rotation
            # write fresh tensors -- but the value gets neither, so it
            # reaches attention as a non-contiguous view. TE copies it
            # inside the timed call, which is the cost qkv_prep says
            # megatron defers to this scenario. The row width is what makes
            # that view strided, so it is recorded.
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
            # What each engine returns. Megatron's THD output folds the head
            # dimension in; the titan modules keep it split. The two hold the
            # same elements in the same order.
            "out_titan_BLNH": [batch, seq, shape.n_heads, shape.head_dim],
            "out_mcore_TD": [batch * seq, shape.n_heads * shape.head_dim],
            # The three spellings of one masking function. positions is what
            # the flex mask_mod indexes, cu_seqlens is what the THD kernels
            # read, and the document count is a property of the seeded draw
            # rather than of the shape, so only the token total is derivable
            # here.
            "positions": [batch, seq],
            "packed_tokens": batch * seq,
            "flex_block_size": 128,
            "flex_flash_block_size": [256, 128],
            # Pinned to the sequence length on both engines, matching what
            # the e2e megatron driver sends and what
            # create_varlen_metadata_for_document reports.
            "max_seqlen": seq,
            "max_seq_len": shape.max_seq_len,
        }
    if scenario_name == "attn_out_proj":
        # Spelled as the product rather than as ``shape.dim``. The two are
        # equal at every representable PiperShape -- ``n_heads`` is
        # ``dim // head_dim`` -- but megatron calls this quantity
        # ``query_projection_size`` and titan builds ``wo`` from the same
        # product, so the manifest records what both engines compute.
        in_features = shape.n_heads * shape.head_dim
        return {
            "x": [batch, seq, in_features],
            # Recorded next to ``x`` because the two arms consume different
            # leading dimensions of the same matrix: megatron reshapes the
            # core-attention output to (t, 1, h) before ``linear_proj`` and our
            # driver runs THD. Both flatten to [batch*seq, in_features] inside
            # the linear, so this is a label and not a measured difference --
            # and the manifest says so rather than leaving a reader to assume.
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
            # Recorded next to the canonical shapes because the mcore arms
            # consume a different leading layout of the same storage: our
            # megatron driver packs THD, so the layer receives a (t, b, h)
            # tensor. An add treats every leading dimension as a row index,
            # and the view is free on a contiguous tensor, so this is a label
            # and not a measured difference -- and the manifest says so
            # rather than leaving a reader to assume.
            "attn_out_thd": [batch * seq, 1, shape.dim],
            "tokens": batch * seq,
            # The scenario performs one flop per element and reads two
            # operands for every one it writes, so this number is what a
            # reader needs to see that both arms are at bandwidth or at
            # dispatch. No arm declares bytes_moved, because one count cannot
            # describe both of the declared modes.
            "elements": batch * seq * shape.dim,
        }
    if scenario_name == "ffn_norm":
        return {
            "x": [batch, seq, shape.dim],
            "weight": [shape.dim],
            # What the kernel actually sees. Both engines flatten the leading
            # dimensions, so the layout order does not reach the kernel.
            "rows": batch * seq,
            "row_width": shape.dim,
        }
    if scenario_name == "moe_residual":
        return {
            "x": [batch, seq, shape.dim],
            "residual": [batch, seq, shape.dim],
            "grad_out": [batch, seq, shape.dim],
            # Recorded next to the canonical shapes because the two engines
            # consume different leading dimensions of the same elements: our
            # megatron driver runs THD, so a hidden state reaching mlp_bda is
            # (t, 1, h). An add treats every leading dimension as a row
            # index, so this is a label rather than a measured difference --
            # and the manifest says so rather than leaving a reader to
            # assume it.
            "x_thd": [batch * seq, 1, shape.dim],
            # What the kernel actually sees. The operation is elementwise, so
            # the element count is the whole of its work.
            "elements": batch * seq * shape.dim,
        }
    if scenario_name == "final_norm":
        return {
            "x": [batch, seq, shape.dim],
            "weight": [shape.dim],
            # The reduction is over the last dimension, so the leading
            # dimensions are a row count and nothing else. This is why the
            # scenario charges no layout conversion to either engine.
            "rows": batch * seq,
            "reduction_length": shape.dim,
        }
    if scenario_name == "embedding_stage":
        # Both native output layouts are recorded, because both are
        # materialized and each engine reads its own. Megatron's driver packs
        # tokens as [1, batch*seq_len] (``benchmarks/e2e/megatron/data.py:54``,
        # THD), so its embedding returns [batch*seq_len, 1, dim] after a
        # transpose the size-1 leading dimension makes free; titan consumes
        # [batch, seq_len] and returns [batch, seq_len, dim]. The two hold the
        # same rows in the same order once both are put back into the canonical
        # shape, so -- as the ``attn_out_proj`` branch above already says of the
        # identical [T, 1, .] reshape -- this is a label and not a measured
        # difference. There is no ``layout_copy_bytes`` key because there is no
        # layout copy on either side.
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
        # ``x_thd`` is recorded next to ``x`` because the two engines consume
        # different leading dimensions of the same matrix: our megatron driver
        # runs THD, so hidden_states reaches ``linear_qkv`` as (t, 1, h). Both
        # flatten to [batch*seq, dim] inside the linear, so this is a label and
        # not a measured difference.
        #
        # All four weight shapes are recorded, not just the one each arm holds.
        # The three unfused matrices are what the inputs builder draws and what
        # the unfused titan arm loads; ``wqkv`` is their grouped interleave and
        # is what both the fused titan arm and megatron hold. A reader who sees
        # only one of the two forms cannot tell which arm a weight belongs to.
        #
        # q, k and v are recorded because the scenario's boundary is where it
        # ends, not only where it starts: the split and the layout work that
        # produces these three tensors are inside the timed region on both
        # engines.
        kv_out = shape.n_kv_heads * shape.head_dim
        return {
            "x": [batch, seq, shape.dim],
            "x_thd": [batch * seq, 1, shape.dim],
            "norm_weight": [shape.dim],
            "wq": [shape.n_heads * shape.head_dim, shape.dim],
            "wk": [kv_out, shape.dim],
            "wv": [kv_out, shape.dim],
            "wqkv": [shape.qkv_out_features, shape.dim],
            "q": [batch, seq, shape.n_heads, shape.head_dim],
            "k": [batch, seq, shape.n_kv_heads, shape.head_dim],
            "v": [batch, seq, shape.n_kv_heads, shape.head_dim],
            "tokens": batch * seq,
        }
    if scenario_name == "lm_head_projection":
        # Both native hidden-state layouts are recorded, because the two
        # engines really do consume different ones. Our megatron driver packs
        # THD as [1, batch*seq_len] (``benchmarks/e2e/megatron/data.py:54``),
        # so hidden states reach ``output_layer`` as [batch*seq_len, 1, dim].
        # Titan consumes [batch, seq_len, dim]. Both fold every leading
        # dimension inside the GEMM, so this is a label and not a measured
        # difference -- and the manifest says so rather than leaving a reader
        # to assume one shared tensor.
        #
        # Note the deliberate disagreement with the ``cross_entropy`` branch,
        # which labels megatron's side [seq, batch, V]. Scenario 16 builds its
        # logits from megatron's own SBHD convention; this scenario builds
        # them from what our THD driver actually delivers. Both are honest
        # about the run they describe, and neither is the other's shape.
        #
        # ``vocab_size`` is spelled out next to ``tokens`` because the output
        # of this projection is the largest object in the scenario by three
        # orders of magnitude -- 1.16 GiB of bf16 at the default workload
        # against 8 MiB of input -- and every memory number the scenario
        # publishes is read against it.
        return {
            "x_titan_BLD": [batch, seq, shape.dim],
            "x_mcore_TBD": [batch * seq, 1, shape.dim],
            "weight": [shape.vocab_size, shape.dim],
            "grad_out": [batch, seq, shape.vocab_size],
            "tokens": batch * seq,
            "vocab_size": shape.vocab_size,
        }
    if scenario_name == "cross_entropy":
        # Both logit layouts are recorded because the two engines really do
        # consume different ones and the manifest should say so rather than
        # leave a reader to assume a shared tensor. The inputs builder holds
        # the canonical [B, L, V] and each mcore arm materializes its [L, B, V]
        # copy at build time -- megatron runs SBHD, and a transpose inside a
        # timed closure would charge one engine for the harness's storage
        # order.
        #
        # ``labels`` is [B, L] on BOTH sides. Megatron's method takes (b, s)
        # and transposes it itself (``language_module.py:164,172``), and the
        # titan arms are charged the same preparation, so there is one label
        # shape and not two.
        #
        # ``vocab_size`` is spelled out next to ``tokens`` because the logit
        # tensor is the largest object in this scenario by three orders of
        # magnitude -- 1.16 GiB of bf16 at the default workload against 32 KiB
        # of labels -- and every memory number the scenario publishes is read
        # against it.
        return {
            "logits_titan_BLV": [batch, seq, shape.vocab_size],
            "logits_mcore_SBV": [seq, batch, shape.vocab_size],
            "labels": [batch, seq],
            "tokens": batch * seq,
            "vocab_size": shape.vocab_size,
        }
    if scenario_name == "moe_router":
        tokens = batch * seq
        return {
            "x": [batch, seq, shape.dim],
            # Recorded next to x because the two engines consume different
            # leading dimensions of the same elements: our megatron driver
            # runs THD, so the hidden state reaching MoELayer.route is
            # (t, 1, h). The gate GEMM flattens every leading dimension, so
            # this is a label and not a measured difference -- the same
            # thing the attn_out_proj and moe_residual branches already say
            # of the identical [T, 1, .] reshape.
            "x_thd": [tokens, 1, shape.dim],
            "gate_weight": [shape.num_experts, shape.dim],
            # What each engine RETURNS, both recorded, because the two
            # differ in kind and not only in layout. Megatron writes a dense
            # [T, E] probability tensor and a dense [T, E] boolean routing
            # map. Titan writes a [B, L, E] score tensor plus a [B, L, K]
            # probability tensor and a [B, L, K] int64 index tensor. At
            # E = 4 and K = 2 all of them are small next to x, which is what
            # both arms read and what dominates the traffic -- so the memory
            # column here is not a kernel statement, and the manifest
            # records the shapes rather than leaving a reader to assume one
            # form.
            "probs_mcore_TE": [tokens, shape.num_experts],
            "routing_map_mcore_TE": [tokens, shape.num_experts],
            "scores_titan_BLE": [batch, seq, shape.num_experts],
            "topk_probs_titan_BLK": [batch, seq, shape.top_k],
            "topk_indices_titan_BLK": [batch, seq, shape.top_k],
            "tokens": tokens,
            "experts": shape.num_experts,
            "top_k": shape.top_k,
            # Spelled out because the scenario's headline caption rests on
            # it being negligible: 33.5 MFLOP at the default workload,
            # against one 8 MiB read. This scenario is bandwidth-and-dispatch
            # bound, not compute bound, and this is the number that says so.
            "gate_gemm_flops": 2 * tokens * shape.dim * shape.num_experts,
        }
    if scenario_name == "dispatch_permute":
        rows = batch * seq * shape.top_k
        return {
            # The canonical hidden states, and the two views the engines
            # take of that one buffer. Both are free views of one contiguous
            # allocation, taken at build time, so neither engine is charged
            # a layout conversion inside a timed closure: megatron's
            # dispatch_preprocess immediately does hidden_states.view(-1, D)
            # (token_dispatcher.py:274) and titan's RoutedExperts.forward
            # does x_BLD.view(T, D) (moe.py:144).
            "x_titan_BLD": [batch, seq, shape.dim],
            "x_mcore_TBD": [batch * seq, 1, shape.dim],
            # The routing decision, in the two forms the engines consume.
            # They hold the same values: probs_TE[t, ids[t, k]] equals
            # scores_TK[t, k] exactly, and probs_TE is zero off-route. Both
            # are fp32 on both engines -- megatron's base profile sets
            # moe_router_dtype="fp32" and titan softmaxes its gate in fp32 --
            # so this scenario carries no precision difference. Scenario 9's
            # cross-engine row does; this one does not.
            "topk_expert_ids_titan_TK": [batch * seq, shape.top_k],
            "topk_scores_titan_TK": [batch * seq, shape.top_k],
            "probs_mcore_TE": [batch * seq, shape.num_experts],
            "routing_map_mcore_TE": [batch * seq, shape.num_experts],
            # What the cut produces, and the shape every downstream scenario
            # reads. The row count is the whole point of the guard every arm
            # carries: an identity dispatch returns batch * seq rows, and
            # this cut produces batch * seq * top_k.
            "permuted_tokens": [rows, shape.dim],
            "permuted_probs": [rows],
            # The split is exact by construction, so this is a list of equal
            # counts rather than a length. An uneven split skips the
            # scenario loudly; it is never capped or rounded.
            "tokens_per_expert": (
                [rows // shape.num_experts] * shape.num_experts
            ),
            # What the kernel actually moves. A permute is a gather, so the
            # copied elements are the whole of its device work, and the
            # x_floor column is read against this number.
            "permuted_elements": rows * shape.dim,
        }
    if scenario_name == "expert_mlp":
        rows = batch * seq * shape.top_k
        per_expert = rows // shape.num_experts
        return {
            # The permuted rows both engines consume, unchanged. Megatron
            # flattens [s, b, h] to [s*b, h] before it permutes and titan views
            # [b, l, d] as [t, d] before it dispatches, so neither engine pays
            # a layout conversion inside this cut and there is no second layout
            # to record.
            "x": [rows, shape.dim],
            "grad_out": [rows, shape.dim],
            # Read by the megatron arms ALONE. Titan's inner_experts never sees
            # the routing probabilities; its dispatcher applies them in
            # combine. This key is the asymmetry that makes the scenario
            # within-engine only, so the manifest records it rather than
            # leaving a reader to infer it from the comparison list.
            "probs": [rows],
            # The synthetic even split both engines receive, in place of a real
            # routing map. Values reach timing only through the probability
            # multiply, so the counts are what has to match, not the router.
            "tokens_per_expert": [per_expert] * shape.num_experts,
            # All four weight forms, because each engine holds only its own and
            # a reader who sees one cannot tell which arm a weight belongs to.
            # The three separate matrices are what the inputs builder draws and
            # what titan's GroupedExperts loads; w13 is the fused layout the
            # other three titan arms hold; the megatron pair is one gated fc1
            # and one fc2 per expert, and the map that produces them is
            # benchmarks/models/piper_qwen3/megatron_weights.py's experts
            # component.
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
            # The block input. Both engines read it inside the combine for
            # its shape alone -- titan allocates torch.zeros_like(x_TD) and
            # megatron restores self.hidden_shape -- so its values reach no
            # number.
            "x": [batch, seq, shape.dim],
            # Recorded next to x because our megatron driver runs THD, so a
            # hidden state reaching the dispatcher is (t, 1, h) and that is
            # the shape the mcore triple returns. Both hold the same
            # batch*seq rows, so this is a label rather than a measured
            # difference.
            "x_thd": [tokens, 1, shape.dim],
            # The tensor every arm combines, in the canonical (expert
            # ascending, token ascending) row order both engines'
            # permutations produce.
            "expert_out": [rows, shape.dim],
            "grad_out": [tokens, shape.dim],
            # The routing decision in the two forms the engines take. They
            # encode one decision; the inputs builder refuses to return a
            # pair that does not.
            "routing_map": [tokens, shape.num_experts],
            "probs": [tokens, shape.num_experts],
            "topk_expert_ids": [tokens, shape.top_k],
            "tokens_per_expert": [rows // shape.num_experts]
            * shape.num_experts,
            # What the kernel actually does: read this many rows, write this
            # many, and add top_k of the first into each of the second.
            "rows_combined": rows,
            "rows_out": tokens,
        }
    raise ValueError(f"Unknown kernel scenario {scenario_name!r}")
