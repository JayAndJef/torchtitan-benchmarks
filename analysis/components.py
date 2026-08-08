"""Per-component GPU-time attribution across arms.

Splits each arm's total GPU kernel time into ~12 model components and diffs
them, so an engine or kernel comparison can say *where* the time went rather
than only how much there was.

Usage:
    python analysis/components.py <arm_dir|trace> [<arm_dir|trace> ...]

An arm directory is ``out/<run>/<scenario>/<hw>/<arm>/``; every trace window
under it is pooled, which is what ``benchmarks/profile_regions.py`` does and
is required for the summed column total to equal that arm's ``results.json``
``gpu_time.kernel_ms_per_step``. A single ``.json`` or ``.json.gz`` trace is
also accepted.

Method
------
Each device event (kernel/memcpy/memset) is attributed to the host call that
launched it: ``args["correlation"]`` -> the ``cuda_runtime``/``cuda_driver``
launch -> the enclosing ``cpu_op`` frames. Two independent rules are computed
and cross-checked, because agreeing rules are the only cheap evidence the
attribution is sound:

``stack``    the time-nested ``cpu_op`` stack on the launching host thread,
             built by a single sweep over starts/ends/queries.
``ext_min``  the smallest-duration ``cpu_op`` sharing the launch's
             ``External id``.

Classification is then **frame-first**: it reads the autograd/module frame
names and only falls back to the kernel name or tensor shapes when the frame
is generic (``aten::mm``). Shape-first classification is a known trap -- see
``_GroupedLinearBackward`` in the notes below.

Classes partition the total: every device event lands in exactly one, so the
column sums to the harness number by construction.

Two bases are reported for every row:

``summed``  durations added up, identical to the harness definition.
``busy``    the interval union, which does not double-count kernels running
            concurrently on different CUDA streams. Megatron runs 5 streams
            to titan's 1, so summing overcounts it ~6.5% and can invert the
            sign of a per-component comparison. Gap shares use ``busy``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from analysis.analyze import busy_union
from benchmarks.artifacts import trace_files
from benchmarks.profile_regions import (
    KERNEL_CATEGORIES,
    LAUNCH_CATEGORIES,
    PROFILER_STEP_TAG,
    _load_events,
)


BACKWARD_FRAMES = ("CompiledFunctionBackward", "autograd::engine::evaluate_function")

# Piper-1B shape constants the shape-fallback rules key on. Exposed as flags
# so the tool is not silently wrong on a differently shaped model.
DEFAULT_VOCAB_SIZE = 151936
DEFAULT_NUM_EXPERTS = 4

# Display order, most-expensive-first for the piper-1B workload.
CLASS_ORDER = (
    "moe_expert_gemm",
    "cross_entropy",
    "lm_head_gemm",
    "swiglu_activation",
    "attention_core",
    "norm",
    "norm_rope_fused",
    "attn_projection_gemm",
    "moe_routing_permute",
    "other_elementwise",
    "rope",
    "optimizer",
    "embedding",
)

# Components that some engines cannot report separately because the compiler
# fused them into a neighbour. Printing 0.0 for these reads as a win; printing
# an upper bound and the host class is the honest rendering.
FUSED_INTO = {"rope": "norm"}

# Merged rows emitted under --merge-fused (the default): a component whose
# parts are only separable on one engine is rolled up so the comparison is
# like-for-like on both.
MERGED_ROWS = {
    "attention_block": ("attention_core", "rope", "norm", "norm_rope_fused")
}

MEGATRON_SNIFF = (
    "_GroupedLinear",
    "FusedAttnFunc",
    "VocabParallelCrossEntropy",
    "_LayerNormLinear",
)


def load_events(path: Path) -> list[dict]:
    """Read a chrome trace, gzipped or not."""
    if path.suffix == ".gz":
        return _load_events(path)
    try:
        with open(path) as handle:
            return json.load(handle)["traceEvents"]
    except (OSError, ValueError, KeyError) as error:
        raise ValueError(f"{path}: unreadable profiler trace: {error}") from error


def _stacks_for_queries(
    cpu_ops: list[dict], queries: list[tuple[float, int]]
) -> dict[int, tuple[int, ...]]:
    """Nested cpu_op stack containing each query timestamp, in one sweep.

    ``queries`` is (timestamp, key); the result maps key -> a tuple of indices
    into ``cpu_ops``, outermost first. Ordering at equal timestamps is
    start < query < end, so an op that begins or ends exactly at a launch is
    treated as containing it.
    """
    timeline: list[tuple[float, int, int]] = []
    for index, event in enumerate(cpu_ops):
        start = event["ts"]
        timeline.append((start, 0, index))
        timeline.append((start + event.get("dur", 0.0), 2, index))
    for timestamp, key in queries:
        timeline.append((timestamp, 1, key))
    timeline.sort(key=lambda item: (item[0], item[1]))

    found: dict[int, tuple[int, ...]] = {}
    stack: list[int] = []
    for _, kind, payload in timeline:
        if kind == 0:
            stack.append(payload)
        elif kind == 2:
            # cpu_ops nest properly, so this is normally the top of the stack;
            # scan down anyway rather than trust it.
            for position in range(len(stack) - 1, -1, -1):
                if stack[position] == payload:
                    del stack[position]
                    break
        else:
            found[payload] = tuple(stack)
    return found


@dataclass
class Attribution:
    """One device event and the host context that launched it."""

    event: dict
    frames: tuple[str, ...]
    inner: dict | None
    ext_min: dict | None

    @property
    def duration(self) -> float:
        return self.event.get("dur", 0.0)

    @property
    def kernel(self) -> str:
        return str(self.event.get("name", ""))

    @property
    def dims(self):
        source = self.inner if self.inner is not None else self.ext_min
        if source is None:
            return None
        return (source.get("args") or {}).get("Input Dims")

    @property
    def phase(self) -> str:
        if any(frame.startswith(BACKWARD_FRAMES) for frame in self.frames):
            return "backward"
        return "forward" if self.frames else "tail"


def attribute_window(path: Path) -> tuple[list[Attribution], int, int]:
    """Attribute one trace window. Returns (rows, profiled_steps, launches)."""
    events = load_events(path)

    device: list[dict] = []
    launches: dict[int, dict] = {}
    cpu_by_tid: dict[int, list[dict]] = defaultdict(list)
    cpu_by_ext: dict[int, list[dict]] = defaultdict(list)
    step_names: set[str] = set()
    launch_count = 0

    for event in events:
        if event.get("ph") != "X":
            continue
        category = event.get("cat")
        name = str(event.get("name", ""))
        if category in KERNEL_CATEGORIES:
            device.append(event)
        elif category in LAUNCH_CATEGORIES:
            correlation = (event.get("args") or {}).get("correlation")
            if correlation is not None:
                launches[correlation] = event
                launch_count += 1
        elif category == "cpu_op":
            cpu_by_tid[event.get("tid")].append(event)
            external = (event.get("args") or {}).get("External id")
            if external is not None:
                cpu_by_ext[external].append(event)
        if name.startswith(PROFILER_STEP_TAG):
            step_names.add(name)

    # Group the launch timestamps by host thread so each thread is swept once.
    queries_by_tid: dict[int, list[tuple[float, int]]] = defaultdict(list)
    launch_for: dict[int, dict] = {}
    for index, event in enumerate(device):
        launch = launches.get((event.get("args") or {}).get("correlation"))
        if launch is None:
            continue
        launch_for[index] = launch
        queries_by_tid[launch.get("tid")].append((launch["ts"], index))

    stacks: dict[int, tuple[int, ...]] = {}
    for tid, queries in queries_by_tid.items():
        stacks.update(_stacks_for_queries(cpu_by_tid.get(tid, []), queries))

    rows: list[Attribution] = []
    for index, event in enumerate(device):
        launch = launch_for.get(index)
        indices = stacks.get(index, ())
        ops = cpu_by_tid.get(launch.get("tid"), []) if launch is not None else []
        frames = tuple(ops[i]["name"] for i in indices)
        inner = ops[indices[-1]] if indices else None
        ext_min = None
        if launch is not None:
            external = (launch.get("args") or {}).get("External id")
            candidates = cpu_by_ext.get(external)
            if candidates:
                ext_min = min(candidates, key=lambda e: e.get("dur", 0.0))
        rows.append(Attribution(event, frames, inner, ext_min))

    return rows, len(step_names), launch_count


# --- classification -------------------------------------------------------


def _trailing_dim_is(dims, value: int) -> bool:
    """True if any operand's trailing dimension equals ``value``.

    Structural rather than substring: matching ``"[4,"`` in the repr of
    ``Input Dims`` also matches a leading batch of 4, which silently misfiles
    every activation tensor at batch 4.
    """
    if not isinstance(dims, list):
        return False
    for operand in dims:
        if isinstance(operand, list) and operand and operand[-1] == value:
            return True
    return False


def _any_dim_is(dims, value: int) -> bool:
    """True if a 2-D operand has ``value`` on either axis.

    Restricted to 2-D operands so a 3-D activation whose batch happens to
    equal the value (``bmm`` at batch 4, with num_experts 4) cannot match.
    Weights and their gradients -- the tensors that actually carry vocab_size
    and num_experts -- are always 2-D here.
    """
    if not isinstance(dims, list):
        return False
    for operand in dims:
        if isinstance(operand, list) and len(operand) == 2 and value in operand:
            return True
    return False


def classify_megatron(kernel: str, frames: tuple[str, ...], dims, shapes) -> str:
    joined = " | ".join(frames)
    lowered = kernel.lower()

    if "VocabParallelCrossEntropy" in joined:
        return "cross_entropy"
    if "LinearWithGradAccumulationAndAsyncCommunication" in joined:
        return "lm_head_gemm"
    if "_GroupedLinear" in joined:
        # Every kernel inside the grouped-GEMM frame belongs to the expert
        # GEMM, GEMM-named or not. Shape-first rules misfile 122.8 ms/step of
        # _GroupedLinearBackward here: it reports Input Dims as
        # [[98304, 1024], []], with the expert dimensions stripped.
        return "moe_expert_gemm"
    if "SwiGLUFunction" in joined or "GLUFunction" in joined:
        return "swiglu_activation"
    if "FusedAttnFunc" in joined:
        return "attention_core"
    if "FusedRoPEFunc" in joined:
        return "rope"
    if "_LayerNormLinear" in joined or "_Linear" in joined:
        # This frame holds BOTH the norm and the projection GEMM; split on the
        # kernel name, never on the frame alone.
        if kernel.startswith("ln_") or "layer_norm" in lowered or "rms" in lowered:
            return "norm"
        if kernel.startswith("nvjet") or "gemm" in lowered:
            return "attn_projection_gemm"
        return "other_elementwise"
    if "_OperationFuserAutogradFunction" in joined:
        return "norm" if kernel.startswith("ln_") else "other_elementwise"
    if any(
        marker in joined
        for marker in (
            "RouterGatingLinear",
            "te_moe::",
            "permute",
            "topk",
            "masked_select",
            "MaskedSelect",
            "Scatter",
            "nonzero",
            "sort",
        )
    ):
        return "moe_routing_permute"
    if (
        "adamw" in joined
        or "_foreach_" in joined
        or "clip_grad" in joined
        or (not frames and "multi_tensor_apply" in kernel)
    ):
        return "optimizer"
    if "Embedding" in joined or "embedding" in joined:
        return "embedding"
    return "other_elementwise"


def classify_titan(kernel: str, frames: tuple[str, ...], dims, shapes) -> str:
    joined = " | ".join(frames)
    lowered = kernel.lower()

    if "cross_entropy" in kernel or "cross_entropy" in joined:
        return "cross_entropy"
    if "_grouped_mm" in joined or "grouped_gemm" in lowered:
        return "moe_expert_gemm"
    if "aten::mm" in joined or "aten::addmm" in joined or "aten::bmm" in joined:
        if _any_dim_is(dims, shapes.vocab_size):
            return "lm_head_gemm"
        if _any_dim_is(dims, shapes.num_experts):
            # The router gate. Match num_experts on ANY axis, not just the
            # trailing one: the weight-grad GEMM is [[4, 49152], [49152, 1024],
            # [4, 1024]], where the expert axis leads. Matching the substring
            # "[4," instead would also match a leading batch of 4.
            return "moe_routing_permute"
        return "attn_projection_gemm"
    if (
        "flex_attention" in kernel
        # FlashAttention-3 (CUTLASS sm90) and torch's bundled FA2. Without
        # these the FA3 arm reports attention_core = 0.000 and dumps its
        # attention kernels into other_elementwise.
        or "FlashAttn" in kernel
        or "flash::" in kernel
        or "pytorch_flash" in kernel
        # TE fused attention, when a titan arm overrides to it.
        or "cudnn_generated_fort_native_sdpa" in kernel
        or "fmha" in kernel
    ):
        return "attention_core"
    if "silu" in lowered:
        return "swiglu_activation"
    if "rms_norm" in kernel:
        # Inductor fuses RoPE's rotate-half into the qk-norm kernel, so those
        # kernels are norm+RoPE and RoPE is not separately measurable. Kept as
        # its own class so `norm` stays comparable to megatron's norm (which
        # excludes RoPE) and RoPE renders as a bound, never as a free 0.0.
        return "norm_rope_fused" if is_fused_norm_rope(kernel) else "norm"
    if (
        any(
            marker in joined
            for marker in ("sort", "topk", "index_put", "scatter", "aminmax")
        )
        or "sort" in lowered
        or "index_put" in kernel
        or "indexing_backward" in kernel
        or "new_zeros" in kernel
        or "scatter" in kernel
        or "cub::" in kernel
    ):
        return "moe_routing_permute"
    if "adamw" in joined or "_foreach_" in joined or "multi_tensor_apply" in kernel:
        return "optimizer"
    if "Embedding" in joined or "embedding" in joined:
        return "embedding"
    return "other_elementwise"


def is_fused_norm_rope(kernel: str) -> bool:
    """A titan qk-norm kernel Inductor also folded the RoPE rotation into."""
    return "rms_norm" in kernel and ("_neg" in kernel or "neg_" in kernel)


REGION_MARKERS = ("Region: ", "## Call CompiledFxGraph", "Torch-Compiled Region")


def _loss_region_frames(
    rows: list[Attribution], assigned: list[str]
) -> frozenset[str]:
    """Compiled-region frames that contain the cross-entropy kernel.

    The loss's reduction epilogue (``triton_red_fused_sum_*``) is CE work but
    carries no CE-specific name, so it would land in other_elementwise. Find
    the enclosing region dynamically rather than hardcoding an Inductor region
    index like ``Region: 2/0``, which changes whenever the graph repartitions.
    """
    regions: set[str] = set()
    for row, component in zip(rows, assigned):
        if component != "cross_entropy":
            continue
        for frame in row.frames:
            if any(marker in frame for marker in REGION_MARKERS):
                regions.add(frame)
    return frozenset(regions)


@dataclass(frozen=True)
class ModelShapes:
    vocab_size: int = DEFAULT_VOCAB_SIZE
    num_experts: int = DEFAULT_NUM_EXPERTS


@dataclass
class ArmComponents:
    """Everything measured for one arm."""

    label: str
    engine: str
    steps: int = 0
    launches: int = 0
    summed_us: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    intervals: dict[str, list[dict]] = field(default_factory=lambda: defaultdict(list))
    phase_us: dict[tuple[str, str], float] = field(
        default_factory=lambda: defaultdict(float)
    )
    detail: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    all_events: list[dict] = field(default_factory=list)
    streams: set = field(default_factory=set)
    fused_bound_us: float = 0.0
    # attribution health
    total_events: int = 0
    no_launch_us: float = 0.0
    no_launch_events: int = 0
    no_frames_us: float = 0.0
    no_frames_events: int = 0
    rule_agree: int = 0
    rule_compared: int = 0

    def ms_per_step(self, value_us: float) -> float:
        return value_us / self.steps / 1000.0 if self.steps else 0.0

    @property
    def summed_total_us(self) -> float:
        return sum(self.summed_us.values())

    @property
    def busy_total_us(self) -> float:
        return busy_union(self.all_events)

    def busy_us(self, component: str) -> float:
        return busy_union(self.intervals.get(component, []))


def analyze_arm(paths: list[Path], label: str, shapes: ModelShapes) -> ArmComponents:
    """Pool every window of one arm into a component breakdown."""
    arm = ArmComponents(label=label, engine="unknown")

    for path in paths:
        rows, steps, launches = attribute_window(path)
        arm.steps += steps
        arm.launches += launches

        if arm.engine == "unknown":
            arm.engine = "megatron" if _sniff_megatron(rows) else "titan"
        classifier = (
            classify_megatron if arm.engine == "megatron" else classify_titan
        )

        assigned = [
            classifier(row.kernel, row.frames, row.dims, shapes) for row in rows
        ]

        # Second pass: sweep the loss region's un-named epilogue kernels into
        # cross-entropy. Cheap, and it keeps the CE row whole.
        loss_regions = _loss_region_frames(rows, assigned)
        if loss_regions:
            for index, row in enumerate(rows):
                if assigned[index] == "other_elementwise" and any(
                    frame in loss_regions for frame in row.frames
                ):
                    assigned[index] = "cross_entropy"

        for row, component in zip(rows, assigned):
            duration = row.duration
            arm.total_events += 1
            arm.all_events.append(row.event)
            arm.streams.add((row.event.get("pid"), row.event.get("tid")))

            if row.inner is None and row.ext_min is None:
                arm.no_launch_us += duration
                arm.no_launch_events += 1
            elif not row.frames:
                arm.no_frames_us += duration
                arm.no_frames_events += 1

            # Cross-check the two attribution rules wherever both fired.
            if row.inner is not None and row.ext_min is not None:
                arm.rule_compared += 1
                if row.inner.get("name") == row.ext_min.get("name"):
                    arm.rule_agree += 1

            arm.summed_us[component] += duration
            arm.counts[component] += 1
            arm.intervals[component].append(row.event)
            arm.phase_us[(component, row.phase)] += duration
            arm.detail[component][row.kernel[:70]] += duration

    arm.fused_bound_us = arm.summed_us.get("norm_rope_fused", 0.0)
    return arm


def _sniff_megatron(rows: list[Attribution]) -> bool:
    for row in rows[:5000]:
        joined = " | ".join(row.frames)
        if any(marker in joined for marker in MEGATRON_SNIFF):
            return True
    return False


# --- reporting ------------------------------------------------------------


def resolve_inputs(target: Path) -> tuple[list[Path], Path | None]:
    """Return (trace paths, arm dir) for an arm directory or a single trace."""
    if target.is_dir():
        paths = trace_files(target)
        if not paths:
            paths = sorted(target.glob("**/*.json.gz")) + sorted(
                target.glob("**/*.json")
            )
        if not paths:
            raise SystemExit(f"{target}: no profiler traces found")
        return paths, target
    if not target.exists():
        raise SystemExit(f"{target}: no such file or directory")
    return [target], None


def provenance(arm_dir: Path | None) -> dict:
    if arm_dir is None:
        return {}
    manifest = arm_dir.parent / "manifest.json"
    if not manifest.exists():
        return {}
    try:
        data = json.loads(manifest.read_text())
    except (OSError, ValueError):
        return {}
    hardware = data.get("hardware_metadata") or {}
    workload = data.get("workload") or {}
    return {
        "scenario": data.get("scenario"),
        "compile_mode": data.get("compile_mode"),
        "ac_mode": data.get("ac_mode"),
        "batch": workload.get("batch"),
        "seq_len": workload.get("seq_len"),
        "steps": workload.get("steps"),
        "torch_version": hardware.get("torch_version"),
        "torchtitan_git_rev": hardware.get("torchtitan_git_rev"),
        "benchmarks_git_rev": hardware.get("benchmarks_git_rev"),
        "megatron_git_rev": hardware.get("megatron_git_rev"),
        "te_version": hardware.get("te_version"),
    }


def render(arms: list[ArmComponents], meta: dict, merge_fused: bool, by_phase: bool):
    if meta:
        print("===== provenance =====")
        for key, value in meta.items():
            if value is not None:
                print(f"  {key:22s} {value}")
        print()

    base = arms[0]
    width = max(15, max(len(arm.label) for arm in arms) + 2)
    paired = len(arms) == 2

    print("===== components (ms/step, summed) =====")
    header = f"{'component':<22}" + "".join(f"{arm.label:>{width}s}" for arm in arms)
    if paired:
        header += f"{'delta':>11s}{'ratio':>8s}{'gap%':>8s}"
    print(header)

    gap_busy = (arms[1].busy_total_us - base.busy_total_us) if paired else 0.0

    def cell(arm: ArmComponents, component: str) -> str:
        value = arm.ms_per_step(arm.summed_us.get(component, 0.0))
        if component in FUSED_INTO and arm.engine == "titan":
            # Never print 0.0 for a component the compiler fused away: that
            # reads as "free" when it is really "not separately measurable".
            return f"{'<=' + format(arm.ms_per_step(arm.fused_bound_us), '.3f'):>{width}s}"
        return f"{value:{width}.3f}"

    for component in CLASS_ORDER:
        if not any(arm.summed_us.get(component) for arm in arms):
            continue
        line = f"{component:<22}" + "".join(cell(arm, component) for arm in arms)
        if paired:
            other = arms[1]
            base_value = base.ms_per_step(base.summed_us.get(component, 0.0))
            other_value = other.ms_per_step(other.summed_us.get(component, 0.0))
            ratio = f"{other_value / base_value:8.3f}" if base_value else f"{'--':>8s}"
            busy_delta = other.ms_per_step(other.busy_us(component)) - base.ms_per_step(
                base.busy_us(component)
            )
            share = (
                f"{100.0 * busy_delta / base.ms_per_step(gap_busy):+8.1f}"
                if gap_busy
                else f"{'--':>8s}"
            )
            line += f"{other_value - base_value:+11.3f}{ratio}{share}"
        print(line)

    if merge_fused:
        for name, parts in MERGED_ROWS.items():
            line = f"{name + ' (merged)':<22}" + "".join(
                f"{arm.ms_per_step(sum(arm.summed_us.get(p, 0.0) for p in parts)):{width}.3f}"
                for arm in arms
            )
            print(line)

    if paired:
        print(
            "\n  delta/ratio are summed-basis; gap% is the BUSY-basis share of the\n"
            "  total gap (summed here would overstate a multi-stream engine)."
        )
    elif len(arms) > 2:
        print("\n  (comparison columns are shown only for exactly two arms)")

    print()
    print("===== totals =====")
    for arm in arms:
        summed = arm.ms_per_step(arm.summed_total_us)
        busy = arm.ms_per_step(arm.busy_total_us)
        overcount = 100.0 * (summed - busy) / summed if summed else 0.0
        ratio_summed = (
            summed / base.ms_per_step(base.summed_total_us) if base.summed_total_us else 0.0
        )
        ratio_busy = (
            busy / base.ms_per_step(base.busy_total_us) if base.busy_total_us else 0.0
        )
        print(
            f"  {arm.label:<22} summed {summed:9.4f}  busy {busy:9.4f}  "
            f"overcount {overcount:5.2f}%  streams {len(arm.streams):2d}  "
            f"steps {arm.steps:3d}  x{ratio_summed:.4f} summed / x{ratio_busy:.4f} busy"
        )

    print()
    print("===== attribution health =====")
    for arm in arms:
        total = arm.summed_total_us or 1.0
        agreement = (
            100.0 * arm.rule_agree / arm.rule_compared if arm.rule_compared else float("nan")
        )
        print(
            f"  {arm.label:<22} engine={arm.engine:<9} events={arm.total_events:7d}  "
            f"no-launch {100 * arm.no_launch_us / total:5.3f}% "
            f"({arm.no_launch_events} ev)  no-frames "
            f"{100 * arm.no_frames_us / total:5.3f}% ({arm.no_frames_events} ev)  "
            f"two-rule agreement {agreement:6.2f}% of {arm.rule_compared}"
        )
        if arm.engine == "titan" and arm.fused_bound_us:
            print(
                f"  {'':<22} fused: rope is inside norm; bound "
                f"<= {arm.ms_per_step(arm.fused_bound_us):.3f} ms/step"
            )

    if by_phase:
        print()
        print("===== by phase (ms/step) =====")
        for arm in arms:
            print(f"  -- {arm.label}")
            for component in CLASS_ORDER:
                parts = {
                    phase: arm.ms_per_step(arm.phase_us.get((component, phase), 0.0))
                    for phase in ("forward", "backward", "tail")
                }
                if not any(parts.values()):
                    continue
                print(
                    f"     {component:<22} fwd {parts['forward']:8.3f}  "
                    f"bwd {parts['backward']:8.3f}  tail {parts['tail']:8.3f}"
                )


def render_detail(arms: list[ArmComponents], top: int) -> None:
    for arm in arms:
        print()
        print(f"===== {arm.label}: top kernels per component =====")
        for component in CLASS_ORDER:
            if not arm.summed_us.get(component):
                continue
            print(f"  -- {component}: {arm.ms_per_step(arm.summed_us[component]):.3f}")
            for kernel, value in arm.detail[component].most_common(top):
                print(f"       {arm.ms_per_step(value):8.3f}  {kernel}")


def to_json(arms: list[ArmComponents], meta: dict) -> dict:
    return {
        "schema_version": 1,
        "provenance": meta,
        "arms": [
            {
                "label": arm.label,
                "engine": arm.engine,
                "steps": arm.steps,
                "streams": len(arm.streams),
                "launches_per_step": arm.launches / arm.steps if arm.steps else None,
                "summed_ms_per_step": arm.ms_per_step(arm.summed_total_us),
                "busy_ms_per_step": arm.ms_per_step(arm.busy_total_us),
                "fused": (
                    {"rope": {"into": "norm", "bound_ms_per_step": arm.ms_per_step(arm.fused_bound_us)}}
                    if arm.engine == "titan" and arm.fused_bound_us
                    else {}
                ),
                "attribution_health": {
                    "events": arm.total_events,
                    "no_launch_pct": 100 * arm.no_launch_us / (arm.summed_total_us or 1),
                    "no_frames_pct": 100 * arm.no_frames_us / (arm.summed_total_us or 1),
                    "two_rule_agreement_pct": (
                        100 * arm.rule_agree / arm.rule_compared
                        if arm.rule_compared
                        else None
                    ),
                    "two_rule_compared": arm.rule_compared,
                },
                "components": {
                    component: {
                        "summed_ms_per_step": arm.ms_per_step(
                            arm.summed_us.get(component, 0.0)
                        ),
                        "busy_ms_per_step": arm.ms_per_step(arm.busy_us(component)),
                        "kernels_per_step": (
                            arm.counts.get(component, 0) / arm.steps if arm.steps else 0
                        ),
                        "mean_kernel_us": (
                            arm.summed_us.get(component, 0.0)
                            / arm.counts[component]
                            if arm.counts.get(component)
                            else 0.0
                        ),
                        "share_pct": (
                            100
                            * arm.summed_us.get(component, 0.0)
                            / arm.summed_total_us
                            if arm.summed_total_us
                            else 0.0
                        ),
                        "fused_into": FUSED_INTO.get(component)
                        if arm.engine == "titan"
                        else None,
                    }
                    for component in CLASS_ORDER
                },
            }
            for arm in arms
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Per-component GPU-time attribution across arms."
    )
    parser.add_argument("targets", nargs="+", type=Path)
    parser.add_argument("--labels", help="comma-separated column headers")
    parser.add_argument("--json", dest="json_path", type=Path)
    parser.add_argument("--merge-fused", dest="merge_fused", action="store_true", default=True)
    parser.add_argument("--no-merge-fused", dest="merge_fused", action="store_false")
    parser.add_argument("--by-phase", action="store_true")
    parser.add_argument("--detail", type=int, default=0, metavar="N",
                        help="also print the top N kernels in each component")
    parser.add_argument("--vocab-size", type=int, default=DEFAULT_VOCAB_SIZE)
    parser.add_argument("--num-experts", type=int, default=DEFAULT_NUM_EXPERTS)
    args = parser.parse_args(argv)

    labels = args.labels.split(",") if args.labels else []
    shapes = ModelShapes(vocab_size=args.vocab_size, num_experts=args.num_experts)

    arms: list[ArmComponents] = []
    meta: dict = {}
    for index, target in enumerate(args.targets):
        paths, arm_dir = resolve_inputs(target)
        label = labels[index] if index < len(labels) else target.name
        if not meta:
            meta = provenance(arm_dir)
        arms.append(analyze_arm(paths, label, shapes))

    render(arms, meta, args.merge_fused, args.by_phase)
    if args.detail:
        render_detail(arms, args.detail)
    if args.json_path:
        args.json_path.write_text(json.dumps(to_json(arms, meta), indent=2))
        print(f"\nwrote {args.json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
