"""The two GPU passes a kernel worker runs, and the order inside each.

Every kernel measurement is one of exactly two passes, and each is a whole
worker process:

* ``run_correctness_pass`` gates every arm against the others and against
  the fp64 reference. It runs first, and once per scenario. It builds the
  arms **one at a time**, keeping each arm's outputs and dropping the arm.
* ``run_timing_pass`` builds **one** arm and times it for one replicate. It
  runs once per (arm, replicate).

Each returns a JSON-ready fragment rather than a result. The parent merges
the fragments (``benchmarks.kernel.results.merge``), because the parent is
the only process that sees them all -- floor ratios and anchor comparisons
are cross-arm work, and no timing worker holds more than one arm.

**Why correctness is one process and timing is many.** Half the arms name
another arm as their correctness reference, so a check needs both sides at
once and gating cannot be split per arm without splitting the checks
themselves. That split lands with the first genuinely incompatible pair,
where it also first gets exercised. Timing has no such coupling: an arm's
samples depend on nothing but that arm.

**One process is not one resident arm.** A check compares *tensors*, so this
pass keeps each arm's outputs and frees the arm itself before building the
next. Isolation and residency are separate questions, and only the first one
needs a process. Memory is therefore bounded by the largest single arm plus
every arm's outputs, not by the sum of the arms -- which is what a scenario
whose arms do not fit together in one device needs.

**Re-seeding is per arm build, not per process.** Inputs are rebuilt in
every worker from the same seed and are bit-identical, because the inputs
builder takes its own generator. Arm *weights* are not: builders consume the
global RNG, so in one process arm 2 saw the state arm 1 left behind, while in
its own process it would see a fresh one. ``_seeded_build`` re-seeds before
every build, which makes an arm's weights identical whether it was built
alone or alongside its neighbours -- so the arm the correctness pass gates is
the arm the timing pass measures. Values do not change timings (a GEMM costs
what it costs), but a gate that ran against different weights than the
measurement is a claim nobody can check.

Nothing in this package imports ``benchmarks.kernel.operations``, a model
*implementation*, or ``benchmarks.kernel.registry``. Arms arrive as
already-resolved ``BuiltArm`` values through ``resolve_symbol``, and the
scenario declarations arrive as ``benchmarks.kernel.schema`` types rather
than through ``benchmarks.kernel.registry``, so the engine's import graph
stays independent of which kernel families exist and of everything they
depend on -- which is what lets each arm run in its own process at all.
``tests/test_import_boundaries.py`` asserts both edges are absent.

The one first-party module under ``benchmarks.models`` the engine does reach
is ``piper_qwen3.shape``, through ``benchmarks.kernel.schema``. It imports
nothing but ``dataclasses``: it is the geometry registry both engines share,
not a model package in the sense above.

``benchmarks.kernel.results.merge`` is the other module this one deliberately
does not import at module scope. The merge is parent-side, and it reaches
numpy and scipy through ``engine.statistics``, so the single call into it --
inside ``run_kernel_scenario``, which the runner never invokes -- is a
deferred import. The two fragment-kind constants both sides need live in
``benchmarks.kernel.schema`` for the same reason: naming them in the reader
made the writer import it.

Both passes re-assert the balanced-routing invariant that
``benchmarks.kernel.runner`` also checks, and do so before the CUDA check:
the runner's loud skip is the friendly path, not the guard, and a direct
caller (``python -m benchmarks.kernel.worker``, the GPU smoke test) must not
be able to measure an expert split that does not cover the rows it built.
"""

from __future__ import annotations

import gc
import importlib
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.engine.correctness import run_correctness
from benchmarks.kernel.engine.measurement import (
    burst_ladder,
    burst_samples,
    memory_pass,
)
from benchmarks.kernel.engine.phases import phase
from benchmarks.kernel.results.schema import KernelScenarioResult
from benchmarks.kernel.schema import (
    CORRECTNESS_FRAGMENT_KIND,
    KernelArm,
    KernelScenario,
    KernelWorkload,
    MODES,
    routing_divides_evenly,
    TIMING_FRAGMENT_KIND,
)

if TYPE_CHECKING:
    # Annotation-only, and the guard is a habit rather than a saving here:
    # benchmarks.kernel.schema imports this same module at run time, and it
    # imports nothing but dataclasses. What the engine must never import is a
    # model *implementation*. Arms reach it as already-resolved BuiltArm
    # values via resolve_symbol, never as imports, which is what lets each arm
    # run in its own process without the engine dragging in every model's
    # dependencies.
    from benchmarks.models.piper_qwen3.shape import PiperShape


@dataclass(frozen=True)
class RunOptions:
    """Everything the measurement passes need that the scenario does not say.

    ``replicates`` / ``burst_k`` / ``warmup_calls`` replace the former
    ``n`` / ``warmup`` pair rather than reinterpreting it. The old fields
    counted round-robin *cycles*, and results.json printed them as such, so
    reusing the names for a different quantity would put a false statement in
    every published table.

    ``burst_k`` is one value for every arm in the scenario, and its default
    of 16 is **not yet validated on hardware**. The right value is the
    smallest at which no arm's per-call time is still falling, which only the
    ``--burst`` ladder can show. Run it and revisit this default.
    """

    replicates: int = 5
    samples_per_replicate: int = 40
    burst_k: int = 16
    warmup_calls: int = 30
    burst: bool = False
    seed: int = 0
    memory_iters: int = 5
    bursts: tuple[int, ...] = (1, 4, 16, 64)
    burst_iters: int = 50


def resolve_symbol(path: str) -> Any:
    module_name, _, attribute = path.partition(":")
    module = importlib.import_module(module_name)
    return getattr(module, attribute)


def _heaviest_mode(arm: BuiltArm) -> str:
    for mode in reversed(MODES):
        if mode in arm.calls:
            return mode
    raise ValueError(f"{arm.name}: no modes")


def _prepare(
    scenario: KernelScenario,
    shape: PiperShape,
    workload: KernelWorkload,
    options: RunOptions,
) -> dict[str, Any]:
    """Assert the invariants, seed the device, and build the shared inputs.

    Called by both passes, so every worker process re-establishes the same
    starting state. The inputs generator is separate from the global RNG, so
    a later ``_seeded_build`` cannot disturb the tensors built here.
    """
    # Asserted here as well as in the runner, and before the device check so
    # it holds on any host: the runner's loud skip is the friendly path, not
    # the guard. `python -m benchmarks.kernel.worker` and direct callers reach
    # this function without passing through it, and an unbalanced split is the
    # violation that can measure rather than fail -- swiglu_inputs hands every
    # expert an equal slice that does not cover the rows it just built.
    if scenario.requires_balanced_routing and not routing_divides_evenly(
        shape, workload
    ):
        rows = workload.batch * workload.seq_len * shape.top_k
        raise ValueError(
            f"{scenario.name}: {rows} routed rows (batch {workload.batch} x "
            f"seq {workload.seq_len} x top_k {shape.top_k}) do not divide "
            f"evenly among {shape.num_experts} experts"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("kernel benchmarks require a CUDA device")
    device = torch.device("cuda")
    # Split from the build below because they are different costs with
    # different fixes: this one is the driver handing the process a context,
    # and it is paid once per process no matter what the process then does.
    with phase("cuda_init", torch.cuda.synchronize):
        torch.cuda.init()
    with phase("inputs_build", torch.cuda.synchronize):
        torch.manual_seed(options.seed)
        generator = torch.Generator(device=device)
        generator.manual_seed(options.seed)
        return resolve_symbol(scenario.inputs_builder)(
            shape, workload, device, generator
        )


def _seeded_build(
    arm: KernelArm,
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: dict[str, Any],
    seed: int,
) -> BuiltArm:
    """Re-seed the global RNG, then build one arm from its dotted path.

    The re-seed makes an arm's parameters depend on the arm alone, never on
    how many arms were built before it. See this module's docstring.

    The gate below is where ``KernelArm`` becomes the authority. Every
    consumer downstream reads the declared modes: the manifest lists them,
    the merge pairs an arm against its opponent mode by mode, and a reader
    ranks implementations by them. A builder that returned one extra closure
    would publish a timed operation the registry never described, and one
    that dropped a closure would leave a declared mode silently absent from
    the table. Both are wrong in the same way, so both raise here.
    """
    torch.manual_seed(seed)
    built = resolve_symbol(arm.builder)(shape, workload, inputs)
    if set(built.calls) != set(arm.modes):
        raise ValueError(
            f"{arm.name}: the builder exposes modes {sorted(built.calls)}, "
            f"but the scenario declares {sorted(arm.modes)}"
        )
    return built


def _detached(
    outputs: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """The same tensors, with no autograd graph behind them.

    ``correctness_outputs`` returns activations that still carry a
    ``grad_fn``, and a graph holds every tensor its backward saved. Dropping
    the arm would then free the module and keep the activations, which is
    most of what the arm allocated. Detaching breaks that edge. It shares
    storage, so it copies nothing, and a gate reads values only.
    """
    return {name: tensor.detach() for name, tensor in outputs.items()}


def _sync() -> None:
    """Close a phase boundary on the device as well as on the host.

    Guarded, because ``gate_outputs`` is deliberately runnable without a CUDA
    device and a phase boundary must not be what makes it require one.
    """
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _release() -> None:
    """Return one arm's memory to the allocator before the next is built.

    ``del`` drops the reference; a reference cycle through the module keeps
    the storage until ``gc`` runs, and a freed block stays in torch's caching
    allocator until it is emptied. Both matter here: the next build asks for
    a block of the same size, and at the huge shape there is no headroom to
    hold two.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gate_outputs(
    scenario: KernelScenario,
    shape: PiperShape,
    workload: KernelWorkload,
    inputs: dict[str, Any],
    seed: int,
    skip: frozenset[str] = frozenset(),
) -> dict[str, dict[str, torch.Tensor]]:
    """Every arm's correctness outputs, with one arm resident at a time.

    Separate from ``run_correctness_pass`` because that function needs a CUDA
    device before it reaches this loop, and the residency property this loop
    exists for is testable without one.

    The invariant: when arm N is built, arms 1..N-1 are already collected.
    Only their output tensors survive, and those are a small fraction of what
    an arm allocates -- at the huge shape a swiglu arm holds an 11.8 GiB bf16
    expert copy and grows a weight gradient of the same size, against
    activations measured in tens of MiB.
    """
    outputs: dict[str, dict[str, torch.Tensor]] = {}
    for arm in scenario.arms:
        if arm.name in skip:
            continue
        with phase(f"gate_build:{arm.name}", _sync):
            built = _seeded_build(arm, shape, workload, inputs, seed)
            outputs[arm.name] = _detached(built.correctness_outputs())
            del built
            _release()
    return outputs


def _environment() -> dict[str, Any]:
    return {
        "device": torch.cuda.get_device_name(0),
        "torch_version": torch.__version__,
    }


def run_correctness_pass(
    scenario: KernelScenario,
    shape: PiperShape,
    workload: KernelWorkload,
    options: RunOptions,
    skip: frozenset[str] = frozenset(),
) -> dict[str, Any]:
    """Gate every arm of the scenario. One process, once per scenario.

    ``skip`` names the arms this host cannot run, so they are neither built
    nor gated. The parent decides the set and closes it over correctness
    references, so a *built* arm always has its reference beside it.

    **One arm is resident at a time.** A gate compares tensors, not modules,
    so each arm is built, asked for its outputs, and dropped before the next
    one is built. The pass held every arm at once until this changed, and
    ``swiglu`` at the huge shape exhausted a 139 GiB device here while each
    of its three arms fits alone: three bf16 expert copies of about 11.8 GiB,
    plus a weight gradient of the same size per arm.

    This does not remove the reason a *pair* of arms may still need separate
    processes. TransformerEngine and the FA3 varlen path cannot share an
    interpreter at all -- a cuDNN soname collision -- and freeing memory does
    not make an import succeed. That split lands with the first scenario that
    declares such a pair.
    """
    inputs = _prepare(scenario, shape, workload, options)
    with phase("reference_build", _sync):
        fp64_reference = (
            resolve_symbol(scenario.reference_builder)(shape, workload, inputs)
            if scenario.reference_builder
            else None
        )
    arm_outputs = gate_outputs(
        scenario, shape, workload, inputs, options.seed, skip
    )
    with phase("gates", _sync):
        rows, all_passed = run_correctness(
            scenario, arm_outputs, fp64_reference
        )
    return {
        "kind": CORRECTNESS_FRAGMENT_KIND,
        "scenario": scenario.name,
        "rows": [asdict(row) for row in rows],
        "all_passed": all_passed,
        # Recorded by the pass that always runs, and runs first.
        "environment": _environment(),
    }


def run_timing_pass(
    scenario: KernelScenario,
    arm_name: str,
    replicate: int,
    shape: PiperShape,
    workload: KernelWorkload,
    options: RunOptions,
) -> dict[str, Any]:
    """Time one arm for one replicate. One process, once per (arm, replicate).

    ``peak_memory_gib`` and the ``--burst`` ladder are properties of the arm
    rather than of a replicate, so replicate 0 measures them and every later
    replicate reports ``None``. The merge reads them from replicate 0 alone.
    """
    inputs = _prepare(scenario, shape, workload, options)
    declaration = scenario.arm(arm_name)
    with phase("arm_build", torch.cuda.synchronize):
        arm = _seeded_build(declaration, shape, workload, inputs, options.seed)

    modes: dict[str, list[float]] = {}
    for mode in MODES:
        values = burst_samples(
            arm,
            mode,
            options.burst_k,
            options.samples_per_replicate,
            options.warmup_calls,
        )
        if values:
            modes[mode] = values

    fragment: dict[str, Any] = {
        "kind": TIMING_FRAGMENT_KIND,
        "scenario": scenario.name,
        "arm": arm_name,
        "replicate": replicate,
        "modes": modes,
        # The one arm property the parent needs and cannot read from the
        # declaration; the parent never sees a BuiltArm. Whether the arm is a
        # floor is declared, so the merge reads it from the registry instead.
        "bytes_moved": arm.bytes_moved,
        "peak_memory_gib": None,
        "burst_us_per_call": None,
    }
    if replicate == 0:
        if not declaration.is_floor:
            with phase("memory_pass"):
                fragment["peak_memory_gib"] = memory_pass(
                    arm, _heaviest_mode(arm), options.memory_iters
                )
        if options.burst:
            # Every declared mode, not just "forward". The old pass read
            # arm.calls["forward"] directly, which left lm_head -- whose arms
            # declare forward_backward only -- with no way to run the
            # diagnostic at all.
            with phase("burst_ladder"):
                fragment["burst_us_per_call"] = {
                    mode: burst_ladder(
                        arm, mode, options.bursts, options.burst_iters
                    )
                    for mode in MODES
                    if mode in arm.calls
                }
    return fragment


def run_kernel_scenario(
    scenario: KernelScenario,
    shape: PiperShape,
    workload: KernelWorkload,
    options: RunOptions,
    hardware: str,
) -> KernelScenarioResult:
    """Both passes and the merge, composed in a single process.

    The production path is ``benchmarks.kernel.runner``, which runs the same
    two functions in separate processes. This is the in-process composition,
    for the GPU smoke test and for direct callers, and it runs the identical
    passes in the identical replicate-major order -- so what it exercises is
    what a real run executes, minus the process boundaries.

    It builds every arm in one process, exactly as the correctness pass does,
    so it is valid only for a scenario whose arms can co-tenant. The runner
    never calls it.

    The merge is imported here rather than at module scope. It is parent-side
    work, and it reaches ``PiperShape``, numpy and scipy at its own module
    scope; importing it above would put all three back into every timing
    worker's graph, for a function no worker calls.
    """
    from benchmarks.kernel.results.merge import merge_kernel_fragments

    correctness = run_correctness_pass(scenario, shape, workload, options)
    # Replicate-major, so drift is shared across arms rather than charged to
    # whichever arm was timed while it happened. Within one replicate the
    # arms run in declaration order; the next replicate repeats the sweep.
    # This is what survives of the old round-robin's paired-sample property
    # once an arm is timed as a burst instead of a single call, and it is the
    # order the runner spawns its workers in.
    timings = [
        run_timing_pass(
            scenario, arm.name, replicate, shape, workload, options
        )
        for replicate in range(options.replicates)
        for arm in scenario.arms
    ]
    return merge_kernel_fragments(
        scenario=scenario,
        shape=shape,
        workload=workload,
        hardware=hardware,
        replicates=options.replicates,
        samples_per_replicate=options.samples_per_replicate,
        burst_k=options.burst_k,
        warmup_calls=options.warmup_calls,
        seed=options.seed,
        correctness=correctness,
        timings=timings,
    )
