"""The engine records, and the two facts stated in more than one place.

``benchmarks/e2e/engines.py`` is the source: one record per engine, pairing
a command builder with a validation profile. Two other modules state a
piece of the same pairing, because they sit below ``engines.py`` and cannot
read the records back:

* ``benchmarks/e2e/validation.py`` maps an engine name to its profile, so
  ``validate_arm`` can choose one; and
* ``benchmarks/e2e/parallelism.py`` holds ``MEGATRON_ENGINES``, so the
  parallelism rules can ask whether a run holds a Megatron arm.

Each of those is pinned equal to the records here. Without these tests a
new engine could take one module's edit and not the other's, and the run
would be validated or refused under the wrong engine's terms.

This module also holds the two arm-to-profile agreements ``validate_arm``
used to refuse at run time. Both are pure functions of the registry, so a
test settles them once instead of every run asking again.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.engines import ENGINES, command_for_arm, engine_for_arm
from benchmarks.e2e.launch import megatron_stock_command, titan_command
from benchmarks.e2e.parallelism import MEGATRON_ENGINES
from benchmarks.e2e.registry import SCENARIOS
from benchmarks.e2e.schema import Arm, ParallelismSpec
from benchmarks.e2e.validation import (
    MEGATRON_STOCK_PROFILE,
    TORCHTITAN_PROFILE,
    profile_for_engine,
)


# The meshes arm rule 12 is asked about: every shape above one rank the
# harness can express, plus one that carries a schedule.
_MESHES = (
    ParallelismSpec(dp=2),
    ParallelismSpec(pp=2, pp_schedule="1F1B"),
    ParallelismSpec(dp=2, pp=2, pp_schedule="1F1B"),
    ParallelismSpec(dp=2, ep=2, zero=1),
)


class EngineRecordTests(unittest.TestCase):
    def test_each_record_names_itself(self):
        """``ENGINES[k].name`` is ``k``, so a record cannot lie about its key."""
        for name, engine in ENGINES.items():
            with self.subTest(engine=name):
                self.assertEqual(engine.name, name)

    def test_each_engine_carries_its_named_profile(self):
        self.assertIs(ENGINES["torchtitan"].validation, TORCHTITAN_PROFILE)
        self.assertIs(
            ENGINES["megatron_stock"].validation, MEGATRON_STOCK_PROFILE
        )

    def test_each_engine_carries_its_named_command_builder(self):
        self.assertIs(ENGINES["torchtitan"].command, titan_command)
        self.assertIs(
            ENGINES["megatron_stock"].command, megatron_stock_command
        )

    def test_the_validation_lookup_agrees_with_the_records(self):
        """The profile ``validate_arm`` picks is the profile on the record."""
        self.assertEqual(
            {name: engine.validation for name, engine in ENGINES.items()},
            {name: profile_for_engine(name) for name in ENGINES},
        )

    def test_the_megatron_set_agrees_with_the_records(self):
        self.assertEqual(
            MEGATRON_ENGINES,
            frozenset(
                name for name, engine in ENGINES.items() if engine.is_megatron
            ),
        )

    def test_an_unknown_engine_is_refused_by_name(self):
        arm = Arm(
            name="invented",
            description="an arm naming an engine nobody registered",
            compile="none",
            engine="mcore",
        )
        with self.assertRaisesRegex(ValueError, "unknown engine 'mcore'"):
            engine_for_arm(arm)


class ArmDeclarationTests(unittest.TestCase):
    def test_every_arm_names_a_registered_engine(self):
        offenders = []
        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                if arm.engine not in ENGINES:
                    offenders.append(f"{scenario.name}/{arm.name}: {arm.engine}")
        self.assertEqual(
            offenders,
            [],
            "an arm names an engine that holds no record:\n  "
            + "\n  ".join(offenders),
        )

    def test_every_compiled_arm_names_an_engine_that_can_prove_it(self):
        """Rule 8 reads a log line, so the engine must print one.

        ``validate_arm`` used to refuse this pairing on every run. The
        pairing is a property of the registry alone, so it is settled
        here: an arm that declares ``compile="torch"`` names an engine
        whose profile carries a ``compile_marker``.
        """
        offenders = []
        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                profile = ENGINES[arm.engine].validation
                if arm.compile == "torch" and profile.compile_marker is None:
                    offenders.append(
                        f"{scenario.name}/{arm.name}: engine {arm.engine}"
                    )
        self.assertEqual(
            offenders,
            [],
            "an arm asks for torch.compile and its engine proves nothing:\n  "
            + "\n  ".join(offenders),
        )

    def test_every_engine_proves_every_mesh_above_one_rank(self):
        """Arm rule 12 needs a marker, so every profile must name one.

        ``validate_arm`` used to refuse an empty tuple on every run. Which
        lines a profile names is a property of the registry, so it is
        settled here instead. The p2p half is deliberately left out: an
        empty tuple there is honest, because a TorchTitan arm never
        receives that value.
        """
        workload = SCENARIOS["engines"].workload
        for name, engine in ENGINES.items():
            for spec in _MESHES:
                for precision in ("stock", "lean"):
                    with self.subTest(
                        engine=name, spec=spec, precision=precision
                    ):
                        markers = engine.validation.parallelism_markers(
                            spec, workload, precision
                        )
                        self.assertTrue(markers)
                        for marker in markers:
                            self.assertTrue(marker.strip())

    def test_the_dispatcher_calls_the_arm_s_own_builder(self):
        """One call goes through the record, and nothing branches."""
        calls = []

        def builder(*args, **kwargs):
            calls.append(args)
            return ["built"]

        scenario = SCENARIOS["engines"]
        arm = scenario.arm("megatron_stock")
        engine = ENGINES[arm.engine]
        ENGINES[arm.engine] = type(engine)(
            name=engine.name,
            command=builder,
            validation=engine.validation,
            is_megatron=engine.is_megatron,
        )
        try:
            command = command_for_arm(
                scenario.workload, arm, Path("/tmp/arm"), (), "none"
            )
        finally:
            ENGINES[arm.engine] = engine
        self.assertEqual(command, ["built"])
        self.assertEqual(len(calls), 1)


if __name__ == "__main__":
    unittest.main()
