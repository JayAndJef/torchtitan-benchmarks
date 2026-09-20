"""The global run axes, and the manifest keys that record them.

``RunAxes`` groups the eight axes into one record, so every reader takes
them together. This module holds the property that grouping is worth
having: the manifest records one key per field, and a new axis that nothing
recorded fails here rather than in a published directory.
"""

import sys
import unittest
from dataclasses import fields
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.manifests import AXIS_KEYS, manifest_data
from benchmarks.e2e.parallelism import TRIVIAL_SPEC, describe
from benchmarks.e2e.registry import scenario_by_name
from benchmarks.e2e.schema import RequestedAxes, RunAxes


_METADATA = {
    "requested_gpu": "0",
    "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
    "torch_version": "test",
    "torchtitan_git_rev": "titan-rev",
    "benchmarks_git_rev": "bench-rev",
    "megatron_git_rev": "megatron-rev",
}

_AXES = RunAxes(
    ac_mode="none",
    model_size="1b",
    parallelism=TRIVIAL_SPEC,
    megatron_p2p_sync="off",
    megatron_nan_guard="off",
    megatron_precision="stock",
    profile=False,
    warmup_steps=10,
)


def _manifest() -> dict:
    scenario = scenario_by_name("engines")
    return manifest_data(
        scenario,
        (scenario.arm("titan_eager"),),
        {"titan_eager": ["cmd"]},
        "test-gpu",
        _METADATA,
        (),
        axes=_AXES,
    )


class AxisKeyTests(unittest.TestCase):
    """Every axis is recorded, and every recorded key is an axis."""

    def test_the_key_list_names_every_field_and_no_other(self) -> None:
        """A new axis that nothing recorded fails here.

        The manifest is what a reader compares two runs by, so an axis the
        manifest does not carry is a comparability boundary that silently
        does not hold.
        """
        self.assertEqual(
            set(AXIS_KEYS), {field.name for field in fields(RunAxes)}
        )

    def test_the_keys_are_flat_and_the_manifest_writes_each_one(self) -> None:
        """External readers address the keys by name, at the top level."""
        recorded = _manifest()
        for key in AXIS_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, recorded)
        self.assertNotIn("axes", recorded)

    def test_each_key_carries_the_value_that_was_asked_for(self) -> None:
        recorded = _manifest()
        self.assertEqual(recorded["ac_mode"], "none")
        self.assertEqual(recorded["model_size"], "1b")
        self.assertEqual(recorded["megatron_p2p_sync"], "off")
        self.assertEqual(recorded["megatron_nan_guard"], "off")
        self.assertEqual(recorded["megatron_precision"], "stock")
        self.assertIs(recorded["profile"], False)
        self.assertEqual(recorded["warmup_steps"], 10)
        # The one key that is a described block rather than the field
        # itself: a reader needs the derived degrees beside the six the
        # spec holds.
        self.assertEqual(
            recorded["parallelism"], describe(TRIVIAL_SPEC, local_batch_size=4)
        )


class AxisRecordTests(unittest.TestCase):
    """The requested record and the resolved one name the same axes."""

    def test_the_two_records_hold_the_same_field_names(self) -> None:
        self.assertEqual(
            [field.name for field in fields(RequestedAxes)],
            [field.name for field in fields(RunAxes)],
        )

    def test_a_requested_axis_may_be_left_out(self) -> None:
        """``None`` is what a resume reads as "inherit the recorded value"."""
        requested = RequestedAxes()
        for field in fields(RequestedAxes):
            with self.subTest(axis=field.name):
                self.assertIsNone(getattr(requested, field.name))

    def test_no_resolved_axis_has_a_default(self) -> None:
        """A defaulted field would record a treatment the run did not have."""
        with self.assertRaises(TypeError):
            RunAxes()


if __name__ == "__main__":
    unittest.main()
