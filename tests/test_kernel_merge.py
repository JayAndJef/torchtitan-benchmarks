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


def merge(timings: list[dict], correctness: dict | None = None):
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
    )


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


if __name__ == "__main__":
    unittest.main()
