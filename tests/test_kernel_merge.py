"""Direct tests for the parent-side merge, with no runner around it.

Every other merge assertion in this suite reaches ``merge_kernel_fragments``
through ``execute_kernel_run``, and that hides the guard the merge exists
for. The runner spawns replicate-major, so one arm's fragments always arrive
in index order, ``_by_arm`` inserts them in that order, and
``_ordered_replicates`` never sees the input it was written to reject.
Replacing its index lookup with ``list(fragments.values())`` left the whole
suite green.

**The pairing carries the statistic.** The headline number is a bootstrap CI
on per-replicate log-ratios: for each replicate the merge forms one ratio
from the arm's samples over the anchor's samples *of that same replicate*,
then bootstraps across those ratios. A replicate is a sweep in time, so drift
that moves a whole replicate moves numerator and denominator together and
cancels. Pair the arm's replicate 2 against the anchor's replicate 0 and the
cancellation is gone, while the number still looks ordinary.

So the fragments below arrive in a deliberately scrambled order, and every
(arm, replicate) carries its own value. A mis-pairing changes the answer
rather than reproducing it -- which the shared CLI fixture cannot do, because
its samples vary with the replicate and not with the arm, so every ratio in
it is 1.0 however the replicates are paired.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kernel.registry import kernel_scenario_by_name
from benchmarks.kernel.results.merge import merge_kernel_fragments
from benchmarks.kernel.schema import (
    CORRECTNESS_FRAGMENT_KIND,
    resolve_shape_and_workload,
    TIMING_FRAGMENT_KIND,
)


SCENARIO = "qkv"
ANCHOR = "baseline"
ARM = "fused_qkv"
REPLICATES = 3


def correctness_fragment(kind: str = CORRECTNESS_FRAGMENT_KIND) -> dict:
    return {
        "kind": kind,
        "scenario": SCENARIO,
        "rows": [],
        "all_passed": True,
        "environment": {"device": "Test GPU", "torch_version": "test"},
    }


def timing_fragment(
    arm: str,
    replicate: int,
    value: float,
    kind: str = TIMING_FRAGMENT_KIND,
) -> dict:
    """One worker's output, with a value that identifies (arm, replicate)."""
    declaration = kernel_scenario_by_name(SCENARIO).arm(arm)
    return {
        "kind": kind,
        "scenario": SCENARIO,
        "arm": arm,
        "replicate": replicate,
        "modes": {mode: [value, value] for mode in declaration.modes},
        "bytes_moved": None,
        "peak_memory_gib": 1.5 if replicate == 0 else None,
        "burst_us_per_call": None,
    }


def merge(
    timings: list[dict],
    correctness: dict | None = None,
    replicates_per_process: int = 1,
):
    shape, workload = resolve_shape_and_workload()
    return merge_kernel_fragments(
        scenario=kernel_scenario_by_name(SCENARIO),
        shape=shape,
        workload=workload,
        hardware="test-gpu",
        replicates=REPLICATES,
        samples_per_replicate=2,
        burst_k=16,
        warmup_calls=1,
        seed=0,
        correctness=correctness or correctness_fragment(),
        timings=timings,
        replicates_per_process=replicates_per_process,
    )


def every_fragment() -> list[dict]:
    """A complete, well-paired set: both arms, every replicate."""
    return [
        timing_fragment(arm, replicate, 100.0 + replicate)
        for replicate in range(REPLICATES)
        for arm in (ANCHOR, ARM)
    ]


class ReplicatePairingTests(unittest.TestCase):
    def test_replicates_pair_by_index_and_not_by_arrival(self) -> None:
        """The anchor's replicate r faces the arm's replicate r.

        The anchor runs 10, 20, 30 us and the arm 11, 22, 33, so a correct
        pairing gives 1.1 in every replicate: one point estimate, and a
        spread of zero. The arm's fragments arrive rotated by one, which
        under arrival-order pairing gives 2.2, 1.65 and 0.367 instead.
        """
        anchor = {index: 10.0 * (index + 1) for index in range(REPLICATES)}
        arm = {index: 11.0 * (index + 1) for index in range(REPLICATES)}
        # Interleaved, and the arm rotated: it arrives 1, 2, 0.
        arrival = [
            (ANCHOR, 0),
            (ARM, 1),
            (ANCHOR, 1),
            (ARM, 2),
            (ANCHOR, 2),
            (ARM, 0),
        ]
        values = {ANCHOR: anchor, ARM: arm}
        timings = [
            timing_fragment(name, replicate, values[name][replicate])
            for name, replicate in arrival
        ]
        self.assertNotEqual(
            [f["replicate"] for f in timings if f["arm"] == ARM],
            list(range(REPLICATES)),
            "the arm's fragments must not arrive in index order, or this "
            "test asserts nothing",
        )

        result = merge(timings)
        row = next(
            row
            for row in result.comparisons
            if row["arm"] == ARM and row["mode"] == "forward"
        )
        self.assertAlmostEqual(row["ratio"], 1.1)
        self.assertAlmostEqual(row["replicate_ratio_spread"], 0.0)
        self.assertEqual(row["replicates"], REPLICATES)
        self.assertAlmostEqual(row["ratio_ci_low"], 1.1)
        self.assertAlmostEqual(row["ratio_ci_high"], 1.1)
        # The stored samples carry the boundaries in index order too, which
        # is what lets a reader re-derive the ratio from results.json.
        self.assertEqual(
            result.arms[ARM].modes["forward"].replicates_us,
            ((11.0, 11.0), (22.0, 22.0), (33.0, 33.0)),
        )
        self.assertEqual(
            result.arms[ANCHOR].modes["forward"].replicates_us,
            ((10.0, 10.0), (20.0, 20.0), (30.0, 30.0)),
        )

    def test_two_fragments_for_one_replicate_raise(self) -> None:
        """A duplicate is a wiring bug: one worker owns one (arm, replicate),
        and silently keeping either copy would drop the other's sweep."""
        timings = [
            timing_fragment(name, replicate, 10.0)
            for name in (ANCHOR, ARM)
            for replicate in range(REPLICATES)
        ]
        timings.append(timing_fragment(ARM, 1, 99.0))
        with self.assertRaisesRegex(
            ValueError, "two fragments for replicate 1"
        ):
            merge(timings)


class FragmentKindTests(unittest.TestCase):
    """The kinds are a wiring check, not a data condition, so they raise."""

    def test_a_correctness_fragment_of_the_wrong_kind_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "kernel_correctness_fragment"):
            merge([], correctness=correctness_fragment(TIMING_FRAGMENT_KIND))

    def test_a_timing_fragment_of_the_wrong_kind_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "kernel_timing_fragment"):
            merge(
                [timing_fragment(ANCHOR, 0, 10.0, CORRECTNESS_FRAGMENT_KIND)]
            )


class WithinProcessIntervalTests(unittest.TestCase):
    """No degraded statistic ships under an honest field name.

    Above one replicate per process an arm's replicates are consecutive
    measurements of one build in one interpreter, so every statistic taken
    over them describes one process rather than several. The interval narrows
    and the spread narrows, without the ratio having become better known. The
    merge therefore renames each of them, exactly as the schema history
    renames every field whose meaning changes: a downstream reader of
    ``ratio_ci_low`` must find nothing rather than a narrower number.
    """

    def test_one_replicate_per_process_keeps_the_honest_names(self) -> None:
        result = merge(every_fragment(), replicates_per_process=1)
        row = result.comparisons[0]
        self.assertIn("ratio_ci_low", row)
        self.assertIn("ratio_ci_high", row)
        self.assertNotIn("within_process_ratio_ci_low", row)
        self.assertEqual(
            result.methodology["arm_isolation"],
            "one_process_per_arm_replicate",
        )
        self.assertEqual(result.methodology["replicates_per_process"], 1)

    def test_a_batched_run_renames_the_interval(self) -> None:
        honest = merge(every_fragment(), replicates_per_process=1)
        batched = merge(every_fragment(), replicates_per_process=3)
        row = batched.comparisons[0]
        self.assertNotIn("ratio_ci_low", row)
        self.assertNotIn("ratio_ci_high", row)
        # The interval is renamed, not discarded: the same numbers, under a
        # name that says what they are.
        self.assertEqual(
            row["within_process_ratio_ci_low"],
            honest.comparisons[0]["ratio_ci_low"],
        )
        self.assertEqual(
            row["within_process_ratio_ci_high"],
            honest.comparisons[0]["ratio_ci_high"],
        )
        # The point estimate is unaffected -- only the interval's meaning
        # changed.
        self.assertEqual(
            row["median_ratio"], honest.comparisons[0]["median_ratio"]
        )

    def test_the_replicate_spread_is_renamed_with_the_interval(self) -> None:
        """The spread is degraded by the same mechanism, so it moves too.

        ``replicate_ratio_spread`` is ``exp(max) - exp(min)`` over the same
        per-replicate log-ratios the interval bootstraps, which makes it the
        round-to-round spread the note says a shared process does not
        improve. Left behind by the rename it would publish a narrowed
        number under a name that says nothing about the process boundary --
        the exact failure the rename exists to prevent.
        """
        honest = merge(every_fragment(), replicates_per_process=1)
        batched = merge(every_fragment(), replicates_per_process=3)
        row = batched.comparisons[0]
        self.assertNotIn("replicate_ratio_spread", row)
        self.assertEqual(
            row["within_process_replicate_ratio_spread"],
            honest.comparisons[0]["replicate_ratio_spread"],
        )

    def test_a_batched_run_says_so_in_the_methodology(self) -> None:
        result = merge(every_fragment(), replicates_per_process=3)
        self.assertEqual(
            result.methodology["arm_isolation"],
            "one_process_per_arm_replicate_block",
        )
        self.assertEqual(result.methodology["replicates_per_process"], 3)
        self.assertIn(
            "within_process_ratio_ci_low",
            result.methodology["replicates_per_process_note"],
        )

# ``ffn_norm`` is the scenario that exercises every compile treatment at once:
# ``copy_floor`` is eager because a bandwidth floor is not an implementation,
# ``mcore/base`` is eager because megatron compiles no whole layer, and
# ``titan`` is compiled. ``qkv`` above cannot distinguish them -- both its arms
# are compiled -- so a merge that dropped the treatment entirely would leave
# these tests green if they used it.
TREATMENT_SCENARIO = "ffn_norm"
TREATMENT_ANCHOR = "mcore/base"


def treatment_fragments(skip: str | None = None) -> list[dict]:
    """Every (arm, replicate) of ``ffn_norm``, optionally missing one arm."""
    scenario = kernel_scenario_by_name(TREATMENT_SCENARIO)
    return [
        {
            "kind": TIMING_FRAGMENT_KIND,
            "scenario": TREATMENT_SCENARIO,
            "arm": arm.name,
            "replicate": replicate,
            "modes": {mode: [100.0, 101.0] for mode in arm.modes},
            "bytes_moved": None,
            "peak_memory_gib": 1.5 if replicate == 0 else None,
            "burst_us_per_call": None,
        }
        for replicate in range(REPLICATES)
        for arm in scenario.arms
        if arm.name != skip
    ]


def merge_treatment(timings: list[dict]):
    shape, workload = resolve_shape_and_workload()
    correctness = correctness_fragment()
    correctness["scenario"] = TREATMENT_SCENARIO
    return merge_kernel_fragments(
        scenario=kernel_scenario_by_name(TREATMENT_SCENARIO),
        shape=shape,
        workload=workload,
        hardware="test-gpu",
        replicates=REPLICATES,
        samples_per_replicate=2,
        burst_k=16,
        warmup_calls=1,
        seed=0,
        correctness=correctness,
        timings=timings,
        replicates_per_process=1,
    )


class CompileTreatmentTests(unittest.TestCase):
    """Schema 6 carries the declared compile treatment into every arm.

    A cross-engine ratio is a comparison of two compile treatments, not of
    two kernels: every megatron arm runs eager because megatron compiles no
    whole layer, and every titan module arm runs compiled because that is
    what it faces end to end. A row that does not name both sides cannot be
    read, so the treatment must reach ``results.json`` rather than staying in
    the registry.
    """

    def test_the_treatment_comes_from_the_declaration_and_not_the_worker(
        self,
    ) -> None:
        """The declaration is the authority. A timing worker builds exactly
        one arm and never sees a second, so it cannot own the roster -- the
        same reason ``is_floor`` is read here and ``BuiltArm`` carries
        neither."""
        fragments = treatment_fragments()
        for fragment in fragments:
            self.assertNotIn("compiled", fragment)
            self.assertNotIn("eager_reason", fragment)
        result = merge_treatment(fragments)
        scenario = kernel_scenario_by_name(TREATMENT_SCENARIO)
        for declaration in scenario.arms:
            arm = result.arms[declaration.name]
            self.assertEqual(arm.compiled, declaration.compiled)
            self.assertEqual(arm.eager_reason, declaration.eager_reason)

    def test_an_eager_arm_publishes_why_it_is_eager(self) -> None:
        """``compiled=False`` alone reads as an oversight. The three eager
        arms here are eager for three different reasons, and only the reason
        tells them apart."""
        result = merge_treatment(treatment_fragments())
        floor = result.arms["copy_floor"]
        anchor = result.arms[TREATMENT_ANCHOR]
        self.assertFalse(floor.compiled)
        self.assertFalse(anchor.compiled)
        self.assertIn("bandwidth floor", floor.eager_reason)
        self.assertIn("megatron", anchor.eager_reason)
        self.assertNotEqual(floor.eager_reason, anchor.eager_reason)
        compiled = result.arms["titan"]
        self.assertTrue(compiled.compiled)
        self.assertIsNone(compiled.eager_reason)

    def test_an_arm_that_never_measured_still_names_its_treatment(
        self,
    ) -> None:
        """An arm reaches the file whether it measured or not. Its treatment
        is declared, not measured, so losing the worker must not lose it --
        otherwise a reader cannot tell a compiled arm that failed from an
        eager one."""
        result = merge_treatment(treatment_fragments(skip="titan"))
        lost = result.arms["titan"]
        self.assertEqual(lost.status, "failed")
        self.assertEqual(lost.modes, {})
        self.assertTrue(lost.compiled)
        self.assertIsNone(lost.eager_reason)

    def test_the_scenario_description_reaches_the_results_file(self) -> None:
        """The caption belongs beside the numbers. A reader of
        ``results.json`` has no registry in hand."""
        result = merge_treatment(treatment_fragments())
        scenario = kernel_scenario_by_name(TREATMENT_SCENARIO)
        self.assertEqual(result.description, scenario.description)
        self.assertIn("megatron", result.description)


if __name__ == "__main__":
    unittest.main()
