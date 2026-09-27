"""The run-wide values, and the manifest block that records them.

The manifest records one key per ``RunSpec`` field, so a new run-wide value
that nothing records fails here. Each ``click.Choice`` equals the tuple it
came from, and every ZeRO level has a row in each table that reads one.
"""

import sys
import unittest
from dataclasses import fields
from pathlib import Path

import click

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.manifests import run_json
from benchmarks.cli.e2e import run_command
from benchmarks.e2e.axes import RequestedAxes
from benchmarks.e2e.engines.api import RunSpec
from benchmarks.e2e.engines.megatron_stock.flags import (
    DATA_PARALLEL_OPTIMIZERS,
    DATA_PARALLEL_WRAPPERS,
    SHARDING_FLAGS_BY_VALUE,
    SHARDING_STRATEGIES,
)
from benchmarks.e2e.parallelism import (
    PP_SCHEDULE_CHOICES,
    TRIVIAL_SPEC,
    ZERO_MODES,
    describe,
)
from benchmarks.e2e.registry import AC_MODES
from benchmarks.models.piper_qwen3.shape import MODEL_SIZE_CHOICES, PIPER_1B
from tests.engine_helpers import run_spec


# The roster every ``--flag`` with a closed value set takes, keyed by the
# option's own parameter name. The CLI must offer exactly these values:
# an option that offered more would accept a value no table below has a
# row for, and one that offered fewer would refuse a legal run.
_CHOICE_BY_OPTION: dict[str, tuple] = {
    "ac_mode": AC_MODES,
    "model_size": MODEL_SIZE_CHOICES,
    "pp_schedule": PP_SCHEDULE_CHOICES,
    "zero": ZERO_MODES,
}

# Every table keyed by a ZeRO level. Each one answers a question about a
# level, so a level with no row builds the wrong argv or the wrong marker.
_ZERO_TABLES = {
    "SHARDING_FLAGS_BY_VALUE": SHARDING_FLAGS_BY_VALUE,
    "DATA_PARALLEL_WRAPPERS": DATA_PARALLEL_WRAPPERS,
    "DATA_PARALLEL_OPTIMIZERS": DATA_PARALLEL_OPTIMIZERS,
    "SHARDING_STRATEGIES": SHARDING_STRATEGIES,
}


class RunBlockTests(unittest.TestCase):
    """The manifest's ``run`` block records every ``RunSpec`` field."""

    def test_the_block_names_every_field_and_no_other(self) -> None:
        self.assertEqual(
            list(run_json(run_spec(profile=False))),
            [field.name for field in fields(RunSpec)],
        )

    def test_each_key_carries_the_value_that_was_asked_for(self) -> None:
        recorded = run_json(run_spec(ac_mode="none", profile=False))
        self.assertEqual(recorded["ac_mode"], "none")
        self.assertEqual(recorded["shape"], PIPER_1B.describe(seq_len=4096))
        self.assertIs(recorded["profile"], False)
        self.assertEqual(recorded["warmup_steps"], 10)
        self.assertEqual(
            recorded["parallelism"], describe(TRIVIAL_SPEC, local_batch_size=4)
        )


class RequestedAxesTests(unittest.TestCase):
    def test_a_requested_axis_may_be_left_out(self) -> None:
        """``None`` is what a resume reads as "inherit the recorded value"."""
        requested = RequestedAxes()
        for field in fields(RequestedAxes):
            with self.subTest(axis=field.name):
                self.assertIsNone(getattr(requested, field.name))


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
