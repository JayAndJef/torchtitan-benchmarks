"""Evaluating the validity gates an arm declares, before anything is timed.

The other half of the engine: ``measurement.py`` answers "how fast", this
answers "is it the same function". ``run_correctness`` walks every arm's
declared ``CorrectnessCheck`` tuple, materializes each side's named outputs
once (arms are cached, because an arm that is another arm's reference is run
by both), and ``_check_rows`` turns one check into one ``CorrectnessResult``
per output and metric. A failing gate aborts the run at worker exit 3 with
results.json still written, so the failure is inspectable rather than merely
reported.

Split from ``run.py`` because the three comparison kinds are where the
numerical judgment lives, and that judgment has its own rate of change: the
choice of ``max_rel_l2`` over max/ULP metrics for reductions, the mean rather
than maximum ULP, the ``clamp_min`` guards, the ``informational`` escape
hatch that records a metric without gating on it. ``CorrectnessCheck``'s
docstring in ``benchmarks.kernel.schema`` states the rules; this module is
the only thing that implements them, and adding a fourth kind touches these
120 lines and nothing else in the engine.

The one subtlety worth restating: ``informational`` checks report
``passed=None`` rather than ``passed=True``, because ``run_correctness``
folds the column with ``all(... if row.passed is not None)``. A check
recorded as passing would be indistinguishable from one that was enforced.
"""

from __future__ import annotations

import torch

from benchmarks.kernel.engine.arm import BuiltArm
from benchmarks.kernel.results.schema import CorrectnessResult
from benchmarks.kernel.schema import CorrectnessCheck, KernelScenario


def _tensor_pair(
    arm_outputs: dict[str, torch.Tensor],
    reference_outputs: dict[str, torch.Tensor],
    arm_name: str,
    output: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    if output not in arm_outputs:
        raise RuntimeError(
            f"{arm_name}: correctness_outputs did not produce {output!r}"
        )
    if output not in reference_outputs:
        raise RuntimeError(
            f"{arm_name}: reference did not produce {output!r}"
        )
    return arm_outputs[output], reference_outputs[output]


def _check_rows(
    arm_name: str,
    check: CorrectnessCheck,
    arm_outputs: dict[str, torch.Tensor],
    reference_outputs: dict[str, torch.Tensor],
) -> list[CorrectnessResult]:
    rows = []
    for output in check.outputs:
        value, truth = _tensor_pair(
            arm_outputs, reference_outputs, arm_name, output
        )
        if check.kind == "bitwise":
            equal = value.shape == truth.shape and torch.equal(value, truth)
            difference = (
                (value.float() - truth.float()).abs().max().item()
                if value.shape == truth.shape
                else float("inf")
            )
            rows.append(
                CorrectnessResult(
                    arm=arm_name,
                    reference=check.reference,
                    kind=check.kind,
                    output=output,
                    metric="max_abs",
                    value=difference,
                    threshold=0.0,
                    passed=None if check.informational else equal,
                    informational=check.informational,
                )
            )
        elif check.kind == "fp64_ulp":
            truth64 = truth.double()
            ulp = torch.ldexp(
                torch.ones_like(truth64),
                torch.floor(
                    torch.log2(truth64.abs().clamp_min(1e-30))
                ).long()
                - 7,
            )
            mean_ulp = ((value.double() - truth64).abs() / ulp).mean().item()
            rows.append(
                CorrectnessResult(
                    arm=arm_name,
                    reference=check.reference,
                    kind=check.kind,
                    output=output,
                    metric="mean_ulp",
                    value=mean_ulp,
                    threshold=check.max_mean_ulp,
                    passed=None
                    if check.informational
                    else mean_ulp <= float(check.max_mean_ulp),
                    informational=check.informational,
                )
            )
        elif check.kind == "tolerance":
            delta = (value.float() - truth.float()).abs()
            if check.max_abs is not None:
                max_abs = delta.max().item()
                rows.append(
                    CorrectnessResult(
                        arm=arm_name,
                        reference=check.reference,
                        kind=check.kind,
                        output=output,
                        metric="max_abs",
                        value=max_abs,
                        threshold=check.max_abs,
                        passed=None
                        if check.informational
                        else max_abs <= float(check.max_abs),
                        informational=check.informational,
                    )
                )
            if check.max_rel is not None:
                max_rel = (
                    (delta / truth.float().abs().clamp_min(1e-6))
                    .max()
                    .item()
                )
                rows.append(
                    CorrectnessResult(
                        arm=arm_name,
                        reference=check.reference,
                        kind=check.kind,
                        output=output,
                        metric="max_rel",
                        value=max_rel,
                        threshold=check.max_rel,
                        passed=None
                        if check.informational
                        else max_rel <= float(check.max_rel),
                        informational=check.informational,
                    )
                )
            if check.max_rel_l2 is not None:
                norm = truth.float().norm()
                rel_l2 = (
                    (delta.norm() / norm).item()
                    if norm
                    else delta.norm().item()
                )
                rows.append(
                    CorrectnessResult(
                        arm=arm_name,
                        reference=check.reference,
                        kind=check.kind,
                        output=output,
                        metric="rel_l2",
                        value=rel_l2,
                        threshold=check.max_rel_l2,
                        passed=None
                        if check.informational
                        else rel_l2 <= float(check.max_rel_l2),
                        informational=check.informational,
                    )
                )
        else:
            raise ValueError(f"unknown correctness kind {check.kind!r}")
    return rows


def run_correctness(
    scenario: KernelScenario,
    built: dict[str, BuiltArm],
    fp64_reference: dict[str, torch.Tensor] | None,
) -> tuple[list[CorrectnessResult], bool]:
    outputs_cache: dict[str, dict[str, torch.Tensor]] = {}

    def outputs_for(name: str) -> dict[str, torch.Tensor]:
        if name == "fp64":
            if fp64_reference is None:
                raise RuntimeError(
                    f"{scenario.name}: fp64 reference requested but the "
                    f"scenario declares no reference_builder"
                )
            return fp64_reference
        if name not in outputs_cache:
            outputs_cache[name] = built[name].correctness_outputs()
        return outputs_cache[name]

    rows: list[CorrectnessResult] = []
    for arm in scenario.arms:
        if arm.name not in built:
            # Skipped by the caller: this host cannot run it. Its own gates
            # go with it.
            continue
        for check in arm.correctness:
            if check.reference != "fp64" and check.reference not in built:
                # Never dropped quietly. The parent closes its skip set over
                # correctness references precisely so this cannot happen, so
                # reaching it means an arm would be timed with nothing
                # checking it.
                raise ValueError(
                    f"{scenario.name}: {arm.name} is gated against "
                    f"{check.reference!r}, which was not built"
                )
            rows.extend(
                _check_rows(
                    arm.name,
                    check,
                    outputs_for(arm.name),
                    outputs_for(check.reference),
                )
            )
    all_passed = all(row.passed for row in rows if row.passed is not None)
    return rows, all_passed
