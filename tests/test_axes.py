"""The global run axes, and the manifest keys that record them.

``RunAxes`` groups the eight axes into one record, so every reader takes
them together. This module holds the property that grouping is worth
having: the manifest records one key per field, and a new axis that nothing
recorded fails here rather than in a published directory.

It also holds the two roster agreements the harness used to re-check on
every run: each ``click.Choice`` equals the axis tuple it came from, and
every ZeRO level the spec admits has a row in each table that reads one.
Both are pure functions of the registry, so a test settles them once.
"""

import sys
import unittest
from dataclasses import fields
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.manifests import AXIS_KEYS, manifest_data
from benchmarks.cli.e2e import run_command
from benchmarks.e2e.megatron_stock.flags import (
    DATA_PARALLEL_OPTIMIZERS,
    DATA_PARALLEL_OVERLAP,
    DATA_PARALLEL_WRAPPERS,
    SHARDING_FLAGS_BY_VALUE,
    SHARDING_STRATEGIES,
)
from benchmarks.e2e.parallelism import (
    PP_SCHEDULE_CHOICES,
    TRIVIAL_SPEC,
    describe,
)
from benchmarks.e2e.registry import (
    AC_MODES,
    MEGATRON_NAN_GUARD_MODES,
    MEGATRON_P2P_SYNC_MODES,
    MEGATRON_PRECISION_MODES,
    scenario_by_name,
)
from benchmarks.e2e.axes import RequestedAxes, RunAxes
from benchmarks.e2e.parallelism import ZERO_MODES
from benchmarks.models.piper_qwen3.shape import MODEL_SIZE_CHOICES


# The roster every ``--flag`` with a closed value set takes, keyed by the
# option's own parameter name. The CLI must offer exactly these values:
# an option that offered more would accept a value no table below has a
# row for, and one that offered fewer would refuse a legal run.
_CHOICE_BY_OPTION: dict[str, tuple] = {
    "ac_mode": AC_MODES,
    "model_size": MODEL_SIZE_CHOICES,
    "pp_schedule": PP_SCHEDULE_CHOICES,
    "zero": ZERO_MODES,
    "megatron_p2p_sync": MEGATRON_P2P_SYNC_MODES,
    "megatron_nan_guard": MEGATRON_NAN_GUARD_MODES,
    "megatron_precision": MEGATRON_PRECISION_MODES,
}

# Every table keyed by a ZeRO level. Each one answers a question about a
# level, so a level with no row builds the wrong argv or the wrong marker.
_ZERO_TABLES = {
    "SHARDING_FLAGS_BY_VALUE": SHARDING_FLAGS_BY_VALUE,
    "DATA_PARALLEL_WRAPPERS": DATA_PARALLEL_WRAPPERS,
    "DATA_PARALLEL_OVERLAP": DATA_PARALLEL_OVERLAP,
    "DATA_PARALLEL_OPTIMIZERS": DATA_PARALLEL_OPTIMIZERS,
    "SHARDING_STRATEGIES": SHARDING_STRATEGIES,
}


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


class ChoiceRosterTests(unittest.TestCase):
    """Each ``click.Choice`` is the axis tuple, and nothing narrower."""

    def test_every_choice_option_offers_its_own_axis_tuple(self) -> None:
        """A copied roster could drift from the tuple the code reads.

        The readers downstream -- the marker builders, the flag tables --
        take the value as given, because the CLI already refused every
        other one. That only holds while the two rosters agree.
        """
        found = {}
        for parameter in run_command.params:
            if isinstance(parameter.type, click.Choice):
                found[parameter.name] = tuple(parameter.type.choices)
        self.assertEqual(
            set(found), set(_CHOICE_BY_OPTION), "a choice option is unpinned"
        )
        for name, choices in found.items():
            with self.subTest(option=name):
                self.assertEqual(choices, tuple(_CHOICE_BY_OPTION[name]))


class ZeroTableTests(unittest.TestCase):
    """Every declared ZeRO level has a row in every table that reads one."""

    def test_each_level_has_a_row_in_each_table(self) -> None:
        """A missing row was a refusal the flag builder made every run.

        ``ParallelismSpec`` admits exactly ``ZERO_MODES``, so a level with
        no row here would reach a table as a ``KeyError`` deep inside the
        argv build. The test moves that failure to the registry.
        """
        for table_name, table in _ZERO_TABLES.items():
            with self.subTest(table=table_name):
                self.assertEqual(set(table), set(ZERO_MODES))


if __name__ == "__main__":
    unittest.main()
