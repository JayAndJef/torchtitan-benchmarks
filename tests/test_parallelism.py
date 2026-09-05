"""CPU-only tests for the parallelism run axis.

``benchmarks/e2e/parallelism.py`` owns the axis itself, while the CLI and
runner thread it through the harness. This file checks the module-level
contract: every one of the seventeen validator rules is exercised in both
directions, the four derivations are pinned, and the schedule registry is
checked against the PyTorch classes it names. Plumbing and runtime validation
have their own test modules.

Two hazards drive the shape of the file. A rule that **admits an illegal
mesh** publishes a number under a topology the run did not have. A rule that
**refuses a legal one** costs a measurement that should have happened -- the
pp=2 / batch 4 / 1F1B milestone is the case that must pass, and the
single-GPU ``--batch 1`` run is the case an over-eager rule 12 would take
away.
"""

import ast
import dataclasses
import inspect
import json
import os
import re
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.parallelism import (
    DEFAULT_DENSE_SHARDING,
    DENSE_SHARDING_MODES,
    MAX_PP,
    MAX_WORLD_SIZE,
    MEGATRON_FSDP_LAUNCHERS,
    MEGATRON_LAUNCHERS,
    NAN_GUARD_LAUNCHERS,
    REPLICATE_ONLY_LAUNCHERS,
    PP_SCHEDULE_CHOICES,
    PP_SCHEDULES,
    TRIVIAL_SPEC,
    ParallelismSpec,
    PipelineSchedule,
    describe,
    execution_model,
    n_microbatches,
    skip_dp,
    titan_mesh,
    validate_parallelism,
)
from benchmarks.e2e.registry import (
    COMPILE_MODES,
    EXECUTION_MODEL,
    SCENARIOS,
    Workload,
)
from benchmarks.models.piper_qwen3.shape import (
    PIPER_SHAPES,
    PiperShape,
    shape_by_name,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

SHAPE_1B = shape_by_name("1b")  # 16 layers, 4 experts
SHAPE_9B = shape_by_name("9b")  # 24 layers, 4 experts
SHAPE_HUGE = shape_by_name("huge")  # 1 layer


def workload(local_batch_size: int = 4) -> Workload:
    """A workload carrying only what the validator reads from one."""
    return Workload(
        module="benchmarks.models.piper_qwen3",
        config="qwen3_piper_1b",
        seq_len=1024,
        steps=40,
        local_batch_size=local_batch_size,
    )


def check(
    spec: ParallelismSpec,
    *,
    shape: PiperShape = SHAPE_1B,
    batch: int = 4,
    compile_mode: str = "default",
    engines: tuple[str, ...] = ("torchtitan",),
    device_count: int | None = None,
) -> None:
    """Validate ``spec``; ``device_count`` defaults to its own world size.

    Defaulting the device count is what lets every rule other than rule 1 be
    tested without rule 1 firing first.
    """
    validate_parallelism(
        spec,
        shape=shape,
        workload=workload(batch),
        compile_mode=compile_mode,
        engines=engines,
        device_count=spec.world_size if device_count is None else device_count,
    )


PP2 = ParallelismSpec(pp=2, pp_schedule="1F1B")

# The cell the stock-Megatron suite runs: two data-parallel replicas of a
# four-stage pipeline, on eight devices. Rule 12 asks it for 8 microbatches,
# and the two batch settings below are the two ways to reach exactly 8.
DP2_PP4 = ParallelismSpec(dp=2, pp=4, pp_schedule="1F1B")
DP2_PP4_MICRO4 = ParallelismSpec(
    dp=2, pp=4, pp_schedule="1F1B", pp_microbatch_size=4
)


# --------------------------------------------------------------------------
# The spec itself
# --------------------------------------------------------------------------


class ParallelismSpecTest(unittest.TestCase):
    def test_the_trivial_spec_is_the_all_defaults_spec(self):
        self.assertEqual(TRIVIAL_SPEC, ParallelismSpec())
        self.assertEqual(TRIVIAL_SPEC.dp, 1)
        self.assertEqual(TRIVIAL_SPEC.pp, 1)
        self.assertEqual(TRIVIAL_SPEC.ep, 1)
        self.assertIsNone(TRIVIAL_SPEC.pp_schedule)
        self.assertEqual(TRIVIAL_SPEC.pp_microbatch_size, 1)
        self.assertEqual(TRIVIAL_SPEC.dense_sharding, "replicate")
        self.assertEqual(TRIVIAL_SPEC.dense_sharding, DEFAULT_DENSE_SHARDING)
        self.assertEqual(TRIVIAL_SPEC.world_size, 1)

    def test_the_world_size_multiplies_dp_by_pp_and_ignores_ep(self):
        """EP borrows ranks from the DP axis; it never asks for more."""
        self.assertEqual(ParallelismSpec(dp=2, pp=2).world_size, 4)
        self.assertEqual(ParallelismSpec(dp=4, ep=4).world_size, 4)
        self.assertEqual(ParallelismSpec(dp=4, ep=2, pp=1).world_size, 4)

    def test_the_spec_holds_no_tp_or_cp_field(self):
        """A field nobody can set would misread as a supported axis."""
        fields = set(ParallelismSpec.__dataclass_fields__)
        self.assertEqual(fields & {"tp", "cp"}, set())

    def test_a_degree_below_one_is_refused_at_construction(self):
        """The precondition the seventeen rules assume.

        Without it ``dp=-1, pp=-1`` has world size 1 and walks past rule 1
        on a one-GPU box, which is exactly the illegal mesh the rules exist
        to refuse.
        """
        for kwargs in (
            {"dp": 0},
            {"dp": -1},
            {"pp": 0},
            {"ep": 0},
            {"pp_microbatch_size": 0},
            {"dp": -1, "pp": -1},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    ParallelismSpec(**kwargs)

    def test_a_positive_degree_is_accepted(self):
        self.assertEqual(ParallelismSpec(dp=4).dp, 4)
        self.assertEqual(ParallelismSpec(pp_microbatch_size=2).pp_microbatch_size, 2)

    def test_the_two_declared_dense_sharding_modes(self):
        """The roster, and that the default is one of its members."""
        self.assertEqual(DENSE_SHARDING_MODES, ("replicate", "shard"))
        self.assertIn(DEFAULT_DENSE_SHARDING, DENSE_SHARDING_MODES)
        self.assertEqual(DEFAULT_DENSE_SHARDING, "replicate")

    def test_each_declared_dense_sharding_mode_is_accepted(self):
        for mode in DENSE_SHARDING_MODES:
            with self.subTest(dense_sharding=mode):
                self.assertEqual(
                    ParallelismSpec(dp=2, dense_sharding=mode).dense_sharding,
                    mode,
                )

    def test_an_unknown_dense_sharding_mode_is_refused_at_construction(self):
        """``titan_mesh`` and ``execution_model`` are total functions over a
        spec and both branch on this value, so a spec carrying a string
        neither branch knows must not exist. A validator rule would be too
        late: both functions run on specs the validator never sees.
        """
        for mode in ("", "Shard", "replicated", "zero3", None):
            with self.subTest(dense_sharding=mode):
                with self.assertRaisesRegex(
                    ValueError, "Unknown dense sharding mode"
                ):
                    ParallelismSpec(dp=2, dense_sharding=mode)

    def test_the_spec_is_frozen(self):
        """``FrozenInstanceError``, not any ``Exception``.

        A bare ``Exception`` also passes on the ``AttributeError`` a renamed
        field raises, so it would stay green on a spec that was no longer
        frozen at all.
        """
        with self.assertRaises(dataclasses.FrozenInstanceError):
            TRIVIAL_SPEC.dp = 2  # type: ignore[misc]


# --------------------------------------------------------------------------
# The schedule registry
# --------------------------------------------------------------------------


class PipelineScheduleRegistryTest(unittest.TestCase):
    def test_the_five_declared_schedules_and_their_flags(self):
        expected = {
            "1F1B": (True, 1, False),
            "Interleaved1F1B": (True, 2, False),
            "InterleavedZeroBubble": (False, 2, True),
            "ZBVZeroBubble": (False, 2, True),
            "DualPipeV": (False, 2, True),
        }
        self.assertEqual(set(PP_SCHEDULES), set(expected))
        for name, (supported, stages, uncompiled) in expected.items():
            with self.subTest(schedule=name):
                schedule = PP_SCHEDULES[name]
                self.assertIsInstance(schedule, PipelineSchedule)
                self.assertEqual(schedule.name, name)
                self.assertEqual(schedule.megatron_supported, supported)
                self.assertEqual(schedule.stages_per_rank, stages)
                self.assertEqual(schedule.requires_uncompiled, uncompiled)

    def test_every_schedule_carries_a_description(self):
        for name, schedule in PP_SCHEDULES.items():
            with self.subTest(schedule=name):
                self.assertTrue(schedule.description.strip())

    def test_the_registry_key_is_the_schedule_name(self):
        for name, schedule in PP_SCHEDULES.items():
            self.assertEqual(name, schedule.name)

    def test_the_choices_equal_the_registry_keys(self):
        self.assertEqual(PP_SCHEDULE_CHOICES, tuple(PP_SCHEDULES))

    def test_the_choices_are_derived_rather_than_written_out(self):
        """Structural, because equality alone cannot tell the two apart.

        A hand-written tuple that happens to match today passes the test
        above and then goes stale the moment somebody registers a schedule.
        So read the source: the assignment must be a call, not a literal.
        """
        source = (REPO_ROOT / "benchmarks" / "e2e" / "parallelism.py").read_text()
        tree = ast.parse(source)
        assignments = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AnnAssign)
            and isinstance(node.target, ast.Name)
            and node.target.id == "PP_SCHEDULE_CHOICES"
        ]
        self.assertEqual(len(assignments), 1)
        value = assignments[0].value
        self.assertNotIsInstance(
            value,
            (ast.Tuple, ast.List, ast.Set),
            "PP_SCHEDULE_CHOICES is written out; derive it from PP_SCHEDULES",
        )
        self.assertIsInstance(value, ast.Call)


try:  # pragma: no cover - import-availability branch
    from torch.distributed.pipelining import schedules as torch_schedules
except Exception:  # pragma: no cover - CPU-only hosts without torch
    torch_schedules = None


@unittest.skipIf(
    torch_schedules is None, "torch.distributed.pipelining is unavailable"
)
class ScheduleNamesMatchPyTorchTest(unittest.TestCase):
    """Every declared property is a claim about a PyTorch class. Check it.

    ``titan_name`` reaches TorchTitan as
    ``--parallelism.pipeline-parallel-schedule``, which forwards it to
    ``get_schedule_class``. A wrong string fails inside the training
    subprocess, minutes into a run, and a wrong ``stages_per_rank`` or
    ``requires_uncompiled`` fails later still -- or not at all, in the case
    of rule 7 admitting a split PyTorch will not do.
    """

    def test_every_titan_name_is_a_key_of_pytorchs_own_map(self):
        """Case-sensitively: ``get_schedule_class`` lowercases its argument,
        so calling it would accept ``1f1b`` and prove nothing about the
        spelling a manifest records."""
        source = inspect.getsource(torch_schedules.get_schedule_class)
        for name, schedule in PP_SCHEDULES.items():
            with self.subTest(schedule=name):
                self.assertIn(f'"{schedule.titan_name}":', source)

    def test_stages_per_rank_matches_pytorchs_single_stage_classification(self):
        """TorchTitan derives the same value: ``stages_per_rank = 1 if
        is_single_stage_schedule else 2`` (``pipeline_parallel.py``)."""
        for name, schedule in PP_SCHEDULES.items():
            with self.subTest(schedule=name):
                cls = torch_schedules.get_schedule_class(schedule.titan_name)
                single = issubclass(cls, torch_schedules.PipelineScheduleSingle)
                self.assertEqual(schedule.stages_per_rank, 1 if single else 2)

    def test_requires_uncompiled_matches_the_classes_that_check_compilation(self):
        """Exactly the classes calling ``_check_torch_compile_compatibility``."""
        for name, schedule in PP_SCHEDULES.items():
            with self.subTest(schedule=name):
                cls = torch_schedules.get_schedule_class(schedule.titan_name)
                checks = "_check_torch_compile_compatibility" in inspect.getsource(cls)
                self.assertEqual(schedule.requires_uncompiled, checks)


# --------------------------------------------------------------------------
# The four derivations
# --------------------------------------------------------------------------


# The dp/ep pairs the budget allows. ``ep`` divides ``dp`` in every one, so
# rule 9 admits them all and the mesh tests may read them under either
# parity.
LEGAL_DP_EP_PAIRS = (
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (8, 1),
    (2, 2),
    (4, 2),
    (4, 4),
    (8, 2),
    (8, 4),
)


class TitanMeshTest(unittest.TestCase):
    def test_the_trivial_spec_still_resolves_to_one_by_one(self):
        """The mesh every published number was measured under."""
        self.assertEqual(titan_mesh(TRIVIAL_SPEC), (1, 1))

    def test_replicate_gives_the_whole_width_to_the_replicate_degree(self):
        """``dp_shard`` 1 is HSDP over a shard group of one rank, which
        shards nothing. That is the pairing a cross-engine DP row needs
        against Megatron's DDP.

        **The expert degree does not move it.** An earlier revision returned
        ``(dp // ep, ep)`` at ``ep > 1``, which moved the dense treatment
        between an ``ep 1`` control cell and its ``ep 2`` twin and made the
        expert row carry two changes.
        """
        for dp, ep in LEGAL_DP_EP_PAIRS:
            with self.subTest(dp=dp, ep=ep):
                self.assertEqual(
                    titan_mesh(ParallelismSpec(dp=dp, ep=ep)), (dp, 1)
                )

    def test_shard_gives_the_whole_width_to_the_shard_degree(self):
        """Pure FSDP over the data-parallel width, at every expert degree.

        Megatron-FSDP v1 shards the dense parameters over ``dp_cp`` and the
        experts over ``expt_dp``. At ``dp 4, ep 2`` that is dense over 4 and
        experts over 2, and ``(1, 4)`` gives TorchTitan the same two numbers:
        ``efsdp = dp_shard // ep`` is 2 there.
        """
        for dp, ep in LEGAL_DP_EP_PAIRS:
            with self.subTest(dp=dp, ep=ep):
                self.assertEqual(
                    titan_mesh(
                        ParallelismSpec(dp=dp, ep=ep, dense_sharding="shard")
                    ),
                    (1, dp),
                )

    def test_the_expert_mesh_degree_stays_whole_under_shard(self):
        """``efsdp = dp_shard * cp * tp // ep`` at cp = tp = 1. Spec rule 9
        keeps ``dp`` divisible by ``ep``, so the division is exact and the
        degree is at least 1."""
        for dp, ep in LEGAL_DP_EP_PAIRS:
            with self.subTest(dp=dp, ep=ep):
                _, shard = titan_mesh(
                    ParallelismSpec(dp=dp, ep=ep, dense_sharding="shard")
                )
                self.assertEqual(shard % ep, 0)
                self.assertGreaterEqual(shard // ep, 1)

    def test_the_mesh_product_is_the_data_parallel_width(self):
        """The invariant that survives both branches."""
        for mode in DENSE_SHARDING_MODES:
            for dp, ep in LEGAL_DP_EP_PAIRS:
                with self.subTest(dense_sharding=mode, dp=dp, ep=ep):
                    replicate, shard = titan_mesh(
                        ParallelismSpec(dp=dp, ep=ep, dense_sharding=mode)
                    )
                    self.assertEqual(replicate * shard, dp)

    def test_the_planned_sharded_cell_at_expert_degree_one(self):
        """**The cell the suite runs, named rather than derived.**

        ``dp 2 x pp 4`` with no expert split is the geometry of the
        dense-sharding control pair. The two parities must give two meshes
        there, and the sharded one must be pure FSDP over the whole
        data-parallel width, because that is what Megatron-FSDP does.

        **The sweep above already covers this pair, and that is the point.**
        ``LEGAL_DP_EP_PAIRS`` holds ``(2, 1)``, so no mutant of ``titan_mesh``
        fails here alone. What this case guards is the ROSTER: a later edit
        that trimmed the pair list would stop covering the geometry the suite
        runs, and the sweep would go green while this case failed. It also
        lets a reader find the run cell by name.
        """
        for mode, mesh in (("replicate", (2, 1)), ("shard", (1, 2))):
            with self.subTest(dense_sharding=mode):
                spec = ParallelismSpec(
                    dp=2,
                    pp=4,
                    pp_schedule="1F1B",
                    pp_microbatch_size=4,
                    ep=1,
                    dense_sharding=mode,
                )
                self.assertEqual(titan_mesh(spec), mesh)

    def test_the_pipeline_degree_does_not_reach_the_mesh(self):
        self.assertEqual(titan_mesh(ParallelismSpec(dp=2, pp=2)), (2, 1))
        self.assertEqual(
            titan_mesh(
                ParallelismSpec(dp=2, pp=2, dense_sharding="shard")
            ),
            (1, 2),
        )


class SkipDpTest(unittest.TestCase):
    def test_true_only_when_neither_dp_nor_ep_is_split(self):
        self.assertTrue(skip_dp(TRIVIAL_SPEC))
        self.assertTrue(skip_dp(ParallelismSpec(pp=2, pp_schedule="1F1B")))
        self.assertFalse(skip_dp(ParallelismSpec(dp=2)))
        self.assertFalse(skip_dp(ParallelismSpec(dp=2, ep=2)))


class NMicrobatchesTest(unittest.TestCase):
    def test_the_batch_divided_by_the_microbatch_size(self):
        self.assertEqual(n_microbatches(TRIVIAL_SPEC, local_batch_size=4), 4)
        self.assertEqual(
            n_microbatches(
                ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=2),
                local_batch_size=8,
            ),
            4,
        )


class ExecutionModelTest(unittest.TestCase):
    def test_the_trivial_spec_string_is_the_one_every_manifest_records(self):
        """Pinned as a literal, because every manifest written since schema
        7 carries this string. Composing it from parts must reproduce it."""
        self.assertEqual(
            execution_model(TRIVIAL_SPEC), "single-gpu-plain-bf16-no-fsdp"
        )

    def test_the_trivial_string_agrees_with_the_registry_constant(self):
        """The other half of the same fact: the constant in use today."""
        self.assertEqual(execution_model(TRIVIAL_SPEC), EXECUTION_MODEL)

    def test_a_parallel_spec_names_its_axes(self):
        self.assertEqual(
            execution_model(PP2), "2-gpu-plain-bf16-no-fsdp-pp2-1F1B"
        )
        self.assertEqual(
            execution_model(ParallelismSpec(dp=2)), "2-gpu-plain-bf16-dp2"
        )
        self.assertEqual(
            execution_model(ParallelismSpec(dp=2, pp=2, pp_schedule="1F1B")),
            "4-gpu-plain-bf16-dp2-pp2-1F1B",
        )
        self.assertEqual(
            execution_model(ParallelismSpec(dp=2, ep=2)),
            "2-gpu-plain-bf16-dp2-ep2",
        )

    def test_the_sharded_parity_reaches_the_string(self):
        """It is a comparability boundary, so a manifest has to carry it.

        The default parity adds nothing, which is what keeps every string
        this repo has already recorded exactly where it was.
        """
        self.assertEqual(
            execution_model(ParallelismSpec(dp=2, dense_sharding="shard")),
            "2-gpu-plain-bf16-dp2-shard",
        )
        self.assertEqual(
            execution_model(
                ParallelismSpec(dp=2, ep=2, dense_sharding="shard")
            ),
            "2-gpu-plain-bf16-dp2-shard-ep2",
        )
        self.assertEqual(
            execution_model(
                ParallelismSpec(
                    dp=2, pp=4, pp_schedule="1F1B", dense_sharding="shard"
                )
            ),
            "8-gpu-plain-bf16-dp2-shard-pp4-1F1B",
        )

    def test_the_two_parities_give_two_strings(self):
        """A run that sharded must not record the string a replicated run
        records. The two answer different questions and the manifest is
        where a reader meets the difference."""
        for dp in (2, 4, 8):
            with self.subTest(dp=dp):
                self.assertNotEqual(
                    execution_model(ParallelismSpec(dp=dp)),
                    execution_model(
                        ParallelismSpec(dp=dp, dense_sharding="shard")
                    ),
                )

    def test_the_eight_gpu_cell_names_its_two_axes(self):
        """The manifest string for the stock-Megatron cell."""
        self.assertEqual(
            execution_model(DP2_PP4), "8-gpu-plain-bf16-dp2-pp4-1F1B"
        )
        self.assertEqual(
            execution_model(ParallelismSpec(pp=4, pp_schedule="1F1B")),
            "4-gpu-plain-bf16-no-fsdp-pp4-1F1B",
        )
        self.assertEqual(
            execution_model(ParallelismSpec(dp=8)), "8-gpu-plain-bf16-dp8"
        )

    def test_the_parallel_parts_name_degrees_and_not_mechanisms(self):
        """One manifest carries one execution_model for a whole run, and a
        cross-engine run holds arms of both engines. A term only TorchTitan's
        code produces -- ``fsdp2``, a resolved mesh degree -- would be false
        for the megatron arm beside it. ``titan_mesh``'s resolution belongs
        in ``describe``, under names that say whose it is.

        **The forbidden set names the mesh spelling, not the word.** It
        listed the bare words ``replicate`` and ``shard`` while neither could
        appear. ``dense_sharding`` is a declared parity that BOTH engines
        honor, so ``dp2-shard`` is true of a TorchTitan arm and of a Megatron
        arm alike, and the word alone is no longer the thing to refuse. What
        must stay out is a resolved mesh DEGREE -- ``replicate2``,
        ``shard1`` -- which is one engine's and is false for the other's arm.

        The pattern refuses either word followed by a digit, rather than the
        two numbers ``titan_mesh`` returns for this spec. Refusing only the
        correct numbers would admit a wrong mesh spelling: at ``dp 2`` under
        ``shard`` the mesh is ``(1, 2)``, so a string reading ``dp2-shard1``
        would carry ``shard1`` past a check that forbade only ``shard2``.
        """
        for spec in (
            ParallelismSpec(dp=2),
            ParallelismSpec(dp=4),
            ParallelismSpec(dp=2, ep=2),
            ParallelismSpec(dp=2, pp=2, pp_schedule="1F1B"),
            ParallelismSpec(dp=2, dense_sharding="shard"),
            ParallelismSpec(dp=4, ep=2, dense_sharding="shard"),
        ):
            with self.subTest(spec=spec):
                rendered = execution_model(spec)
                for engine_term in ("fsdp2", "ddp"):
                    self.assertNotIn(engine_term, rendered)
                self.assertIsNone(
                    re.search(r"(replicate|shard)\d", rendered),
                    f"{rendered} spells out a resolved mesh degree",
                )
                self.assertIn(f"dp{spec.dp}", rendered)

    def test_every_spec_gives_a_distinct_string(self):
        specs = (
            TRIVIAL_SPEC,
            PP2,
            ParallelismSpec(dp=2),
            ParallelismSpec(dp=4),
            ParallelismSpec(dp=2, pp=2, pp_schedule="1F1B"),
            ParallelismSpec(pp=2, pp_schedule="Interleaved1F1B"),
            ParallelismSpec(dp=2, ep=2),
            ParallelismSpec(dp=2, dense_sharding="shard"),
            ParallelismSpec(dp=2, ep=2, dense_sharding="shard"),
        )
        rendered = [execution_model(spec) for spec in specs]
        self.assertEqual(len(set(rendered)), len(rendered))


class DescribeTest(unittest.TestCase):
    def test_the_trivial_record(self):
        self.assertEqual(
            describe(TRIVIAL_SPEC, local_batch_size=4),
            {
                "dp": 1,
                "pp": 1,
                "ep": 1,
                "pp_schedule": None,
                "pp_microbatch_size": 1,
                "dense_sharding": "replicate",
                "world_size": 1,
                "dp_replicate": 1,
                "dp_shard": 1,
                "n_microbatches": 4,
            },
        )

    def test_the_pp2_record(self):
        self.assertEqual(
            describe(PP2, local_batch_size=4),
            {
                "dp": 1,
                "pp": 2,
                "ep": 1,
                "pp_schedule": "1F1B",
                "pp_microbatch_size": 1,
                "dense_sharding": "replicate",
                "world_size": 2,
                "dp_replicate": 1,
                "dp_shard": 1,
                "n_microbatches": 4,
            },
        )

    def test_a_sharded_record_carries_the_parity_and_the_mesh_it_resolves_to(
        self,
    ):
        """Two facts, not one. The parity is what the operator asked for and
        the mesh is what TorchTitan builds from it; neither derives the other
        for a reader who does not hold this module."""
        spec = ParallelismSpec(dp=4, ep=2, dense_sharding="shard")
        record = describe(spec, local_batch_size=8)
        self.assertEqual(record["dense_sharding"], "shard")
        self.assertEqual(record["dp_replicate"], 1)
        self.assertEqual(record["dp_shard"], 4)
        replicated = describe(
            ParallelismSpec(dp=4, ep=2), local_batch_size=8
        )
        self.assertEqual(replicated["dense_sharding"], "replicate")
        self.assertEqual(replicated["dp_replicate"], 4)
        self.assertEqual(replicated["dp_shard"], 1)

    def test_the_record_is_json_safe(self):
        for spec in (
            TRIVIAL_SPEC,
            PP2,
            ParallelismSpec(dp=2, pp=2, pp_schedule="1F1B"),
            ParallelismSpec(dp=2, pp=2, pp_schedule="1F1B", dense_sharding="shard"),
        ):
            with self.subTest(spec=spec):
                payload = describe(spec, local_batch_size=8)
                self.assertEqual(json.loads(json.dumps(payload)), payload)


# --------------------------------------------------------------------------
# The validator, rule by rule, in both directions
# --------------------------------------------------------------------------


class Rule01WorldSizeMatchesDevicesTest(unittest.TestCase):
    def test_a_matching_mesh_passes(self):
        check(TRIVIAL_SPEC, device_count=1)
        check(PP2, device_count=2)
        check(ParallelismSpec(dp=2, pp=2, pp_schedule="1F1B"), device_count=4)

    def test_a_mesh_that_does_not_fill_the_devices_is_refused(self):
        with self.assertRaisesRegex(ValueError, "world size"):
            check(PP2, device_count=1)

    def test_a_mesh_larger_than_the_devices_is_refused(self):
        with self.assertRaisesRegex(ValueError, "world size"):
            check(TRIVIAL_SPEC, device_count=2)

    def test_ep_does_not_excuse_a_short_device_list(self):
        """The failure the ``ep`` borrow rule invites: ``dp=2, ep=2`` is two
        ranks, not four, and not one."""
        with self.assertRaisesRegex(ValueError, "world size"):
            check(ParallelismSpec(dp=2, ep=2), device_count=4)


class Rule02BudgetTest(unittest.TestCase):
    """Rule 2 has two halves, and both raise a ``ValueError``.

    A test that asserts the raise alone cannot say which half ran, so every
    test here matches the message of the half it means to fire. That matters
    because the caps moved: a ``pp 4`` spec fired the pp half before, and it
    passes now. A loose test would still be green and would prove nothing.

    **The two caps are now equal, so the halves are no longer independent.**
    ``world_size`` is ``dp * pp`` and is never below ``pp``, so every spec
    the pipeline half refuses is one the world-size half would refuse too.
    The pipeline half runs first for that reason, and the class below pins
    the order.
    """

    def test_the_declared_caps_are_the_ones_this_pass_targets(self):
        """Pinned as literals, because the messages below name them.

        Eight devices is what this host holds. Eight stages is the deepest
        pipeline those devices hold, and the suite runs that cell. Lifting
        either is a deliberate act, and it edits this test.
        """
        self.assertEqual(MAX_WORLD_SIZE, 8)
        self.assertEqual(MAX_PP, 8)

    def test_the_largest_budgeted_mesh_passes(self):
        """Both ways to fill eight devices.

        ``dp 2 x pp 4`` needs 8 microbatches for rule 12, so it takes batch
        8 rather than the default 4. The pure data-parallel mesh has no
        pipeline and rule 12 does not read it.
        """
        check(ParallelismSpec(dp=MAX_WORLD_SIZE))
        check(DP2_PP4, batch=8)

    def test_a_world_size_above_the_budget_is_refused(self):
        """``dp 16`` satisfies rule 1, because ``check`` gives it 16
        devices, so the budget is what refuses it. The message says so."""
        with self.assertRaisesRegex(
            ValueError, r"world size 16 exceeds the 8-GPU budget"
        ):
            check(ParallelismSpec(dp=MAX_WORLD_SIZE * 2))

    def test_the_boundary_of_the_world_size_cap_from_both_sides(self):
        """Eight devices pass and nine do not.

        ``dp 9`` has no pipeline, so no other rule reads it. Rule 1 passes
        because ``check`` gives it 9 devices.
        """
        check(ParallelismSpec(dp=MAX_WORLD_SIZE))
        with self.assertRaisesRegex(
            ValueError, r"world size 9 exceeds the 8-GPU budget"
        ):
            check(ParallelismSpec(dp=MAX_WORLD_SIZE + 1))

    def test_a_pipeline_deeper_than_the_maximum_is_refused(self):
        """``pp 16`` is above both caps, and the pipeline half is what names
        it.

        The spec has moved twice. It was ``pp 4`` while the cap was 2, then
        ``pp 8`` while the cap was 4, and both of those are legal now. The
        message moves with it, which is what keeps the test honest about
        which half fired.
        """
        with self.assertRaisesRegex(
            ValueError, r"pipeline degree 16 exceeds the supported maximum 8"
        ):
            check(ParallelismSpec(pp=MAX_PP * 2, pp_schedule="1F1B"))

    def test_the_boundary_of_the_pipeline_cap_from_both_sides(self):
        """``pp 8`` passes and ``pp 9`` does not.

        ``pp 8`` needs 16 microbatches for rule 12, so it takes batch 16.
        ``pp 9`` is world size 9, above the budget, and 16 layers do not
        divide into 9 stages -- so two other refusals are available. The
        message is what says rule 2's pipeline half fired first.
        """
        check(ParallelismSpec(pp=MAX_PP, pp_schedule="1F1B"), batch=16)
        with self.assertRaisesRegex(
            ValueError, r"pipeline degree 9 exceeds the supported maximum 8"
        ):
            check(ParallelismSpec(pp=MAX_PP + 1, pp_schedule="1F1B"), batch=18)

    def test_the_pipeline_half_is_tested_before_the_world_size_half(self):
        """**The order is what keeps the pipeline half reachable.**

        ``world_size`` is ``dp * pp`` and is never below ``pp``, so once the
        two caps are equal every spec above MAX_PP is also above
        MAX_WORLD_SIZE. Testing the world size first would make this message
        unreachable, and every deep-pipeline refusal would name the GPU
        budget instead of the cap somebody has to lift.

        ``pp 9`` and ``pp 16`` are both above both caps. Each must name the
        pipeline, and neither may name the budget.
        """
        for pp in (MAX_PP + 1, MAX_PP * 2):
            with self.subTest(pp=pp):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"pipeline degree {pp} exceeds the supported maximum 8",
                ) as raised:
                    check(
                        ParallelismSpec(pp=pp, pp_schedule="1F1B"),
                        batch=2 * pp,
                    )
                self.assertNotIn("GPU budget", str(raised.exception))

    def test_the_pipeline_message_no_longer_claims_an_engine_reason(self):
        """The old message said the two engines count layers the same way
        only at ``pp <= 2``. That was false: ``launch.py`` sends
        ``--parallelism.pipeline-parallel-first-stage-less-layers 0`` and its
        twin at every ``pp > 1``, which makes the two conventions agree at
        every degree. The cap is a plan, not an engine limit.

        The positive match comes first, and it has to. ``pp 16`` at batch 4
        also fails the world-size half and rule 11, and neither of those
        messages contains either phrase -- so two ``assertNotIn`` checks
        alone would pass with rule 2's pp half deleted.
        """
        with self.assertRaisesRegex(
            ValueError, r"pipeline degree 16 exceeds the supported maximum 8"
        ) as raised:
            check(ParallelismSpec(pp=MAX_PP * 2, pp_schedule="1F1B"))
        self.assertNotIn("layer-counting", str(raised.exception))
        self.assertNotIn("pp <= 2", str(raised.exception))


class Rule03ScheduleAccompaniesAPipelineTest(unittest.TestCase):
    def test_no_pipeline_and_no_schedule_passes(self):
        check(TRIVIAL_SPEC)

    def test_a_pipeline_with_a_schedule_passes(self):
        check(PP2)

    def test_a_schedule_without_a_pipeline_is_refused(self):
        with self.assertRaisesRegex(ValueError, "no pipeline to schedule"):
            check(ParallelismSpec(pp_schedule="1F1B"))

    def test_a_pipeline_without_a_schedule_is_refused(self):
        with self.assertRaisesRegex(ValueError, "needs a pipeline schedule"):
            check(ParallelismSpec(pp=2))

    def test_a_microbatch_size_without_a_pipeline_is_refused(self):
        """The schedule's twin. TorchTitan reads
        ``pipeline_parallel_microbatch_size`` only inside
        ``_build_pipeline_schedule``, which runs only at pp > 1, so a value
        set at pp 1 is recorded in the manifest and delivered to nothing --
        and ``--resume`` gates on that record."""
        with self.assertRaisesRegex(ValueError, "at pp 1"):
            check(ParallelismSpec(pp_microbatch_size=2))
        with self.assertRaisesRegex(ValueError, "at pp 1"):
            check(ParallelismSpec(pp_microbatch_size=4), batch=8)

    def test_a_microbatch_size_with_a_pipeline_passes(self):
        check(
            ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=2),
            batch=8,
        )


class Rule04ScheduleIsRegisteredTest(unittest.TestCase):
    def test_a_registered_name_passes(self):
        check(PP2)

    def test_an_unregistered_name_is_refused(self):
        with self.assertRaisesRegex(ValueError, "Unknown pipeline schedule"):
            check(ParallelismSpec(pp=2, pp_schedule="GPipe"))

    def test_a_case_variant_is_refused(self):
        """PyTorch's own lookup is case-insensitive; ours is not, because the
        manifest records the string and two spellings would not compare."""
        with self.assertRaisesRegex(ValueError, "Unknown pipeline schedule"):
            check(ParallelismSpec(pp=2, pp_schedule="1f1b"))


class Rule05MegatronSupportsTheScheduleTest(unittest.TestCase):
    def test_a_supported_schedule_reaches_a_megatron_run(self):
        check(PP2, engines=("torchtitan", "megatron"))

    def test_a_pytorch_only_schedule_reaches_a_titan_only_run(self):
        """Rule 5 reads the launchers, not the scenario name, so a run with
        no megatron arm keeps the schedule."""
        check(
            ParallelismSpec(pp=2, pp_schedule="ZBVZeroBubble"),
            batch=8,
            compile_mode="none",
            engines=("torchtitan",),
        )

    def test_interleaved_reaches_a_megatron_run_and_the_driver_refuses_it(self):
        """The one schedule where "Megatron-LM implements it" and "this
        repo's driver runs it" disagree, pinned so nobody closes the gap by
        writing a false ``megatron_supported=False``.

        Megatron-LM implements Interleaved1F1B, so a cross-engine row is
        possible in principle and rule 5 -- which asks the library's
        question -- lets the spec through. ``benchmarks/e2e/megatron/
        train.py`` has no model-chunk list and raises. That is the
        declaration-without-a-builder pattern the kernel spans use: the
        failure lands where the missing work lives.
        """
        self.assertTrue(PP_SCHEDULES["Interleaved1F1B"].megatron_supported)
        check(
            ParallelismSpec(pp=2, pp_schedule="Interleaved1F1B"),
            batch=8,
            engines=("torchtitan", "megatron"),
        )

    def test_a_pytorch_only_schedule_is_refused_beside_a_megatron_arm(self):
        for name in ("InterleavedZeroBubble", "ZBVZeroBubble", "DualPipeV"):
            with self.subTest(schedule=name):
                with self.assertRaisesRegex(ValueError, "not implemented by"):
                    check(
                        ParallelismSpec(pp=2, pp_schedule=name),
                        batch=8,
                        compile_mode="none",
                        engines=("torchtitan", "megatron"),
                    )


class MegatronLauncherSetTest(unittest.TestCase):
    """Rule 5 asks "does this run drive Megatron-LM", and this set answers it.

    The rule tested one launcher name by equality until a second Megatron-LM
    launcher arrived. The second one then walked past the rule, and a refusal
    inside a command builder covered the hole instead. These tests make the
    classification a declaration rather than a spelling.
    """

    def test_every_registry_launcher_is_classified(self):
        """The guard the declared set needs.

        A launcher that is neither ``torchtitan`` nor a member of
        ``MEGATRON_LAUNCHERS`` has never been classified, so nobody has
        decided whether rule 5 applies to it. Fail here, where the decision
        is one edit, rather than inside a training subprocess.
        """
        launchers = {
            arm.launcher
            for scenario in SCENARIOS.values()
            for arm in scenario.arms
        }
        unclassified = launchers - MEGATRON_LAUNCHERS - {"torchtitan"}
        self.assertEqual(
            unclassified,
            set(),
            "add each launcher to MEGATRON_LAUNCHERS, or to the titan side, "
            "before rule 5 has to read it",
        )

    def test_the_set_holds_only_launchers_the_registry_declares(self):
        """The other direction. A name nobody uses is a name that went
        stale, and rule 5 would then read a set that describes no arm."""
        launchers = {
            arm.launcher
            for scenario in SCENARIOS.values()
            for arm in scenario.arms
        }
        self.assertEqual(MEGATRON_LAUNCHERS - launchers, set())

    def test_the_set_does_not_hold_the_titan_launcher(self):
        self.assertNotIn("torchtitan", MEGATRON_LAUNCHERS)

    def test_the_nan_guard_set_names_the_stock_launcher_alone(self):
        """``--megatron-nan-guard`` reaches these launchers and no other.

        A subset of the Megatron set, because the guard is Megatron's; and
        the tuned driver is not in it, because it never calls
        ``validate_result`` and has no guard under either value. A launcher
        the registry does not declare would be a stale name here.
        """
        self.assertEqual(NAN_GUARD_LAUNCHERS, frozenset({"megatron_stock"}))
        self.assertTrue(NAN_GUARD_LAUNCHERS <= MEGATRON_LAUNCHERS)
        self.assertNotIn("megatron", NAN_GUARD_LAUNCHERS)
        launchers = {
            arm.launcher
            for scenario in SCENARIOS.values()
            for arm in scenario.arms
        }
        self.assertEqual(NAN_GUARD_LAUNCHERS - launchers, set())

    def test_rule_five_reads_every_member_of_the_set(self):
        """Each Megatron-LM launcher alone must trip rule 5.

        A membership test that read only the first name would pass with any
        one launcher present, so ask each of them on its own.
        """
        for launcher in sorted(MEGATRON_LAUNCHERS):
            with self.subTest(launcher=launcher):
                with self.assertRaisesRegex(
                    ValueError, r"not implemented by Megatron-LM"
                ):
                    check(
                        ParallelismSpec(pp=2, pp_schedule="ZBVZeroBubble"),
                        batch=8,
                        compile_mode="none",
                        engines=("torchtitan", launcher),
                    )

    def test_a_launcher_outside_the_set_keeps_a_pytorch_only_schedule(self):
        """The rule must not become "anything that is not torchtitan".

        A third engine would then inherit Megatron's restriction and lose a
        legal cell for a reason that is not about it.
        """
        check(
            ParallelismSpec(pp=2, pp_schedule="ZBVZeroBubble"),
            batch=8,
            compile_mode="none",
            engines=("torchtitan", "some-other-engine"),
        )


class Rule06UncompiledScheduleTest(unittest.TestCase):
    def test_an_uncompiled_mode_carries_a_zero_bubble_schedule(self):
        check(
            ParallelismSpec(pp=2, pp_schedule="ZBVZeroBubble"),
            batch=8,
            compile_mode="none",
        )

    def test_a_compiled_mode_is_refused_for_a_zero_bubble_schedule(self):
        with self.assertRaisesRegex(ValueError, "uncompiled compile mode"):
            check(
                ParallelismSpec(pp=2, pp_schedule="ZBVZeroBubble"),
                batch=8,
                compile_mode="default",
            )

    def test_the_two_targeted_schedules_run_compiled(self):
        """1F1B and Interleaved1F1B never call
        ``_check_torch_compile_compatibility``, so the milestone is a fully
        compiled run."""
        check(PP2, compile_mode="default")
        check(
            ParallelismSpec(pp=2, pp_schedule="Interleaved1F1B"),
            batch=8,
            compile_mode="default",
        )


class Rule07LayersDivideIntoStagesTest(unittest.TestCase):
    def test_sixteen_layers_over_two_stages_passes(self):
        check(PP2, shape=SHAPE_1B)

    def test_sixteen_layers_over_four_interleaved_stages_passes(self):
        check(
            ParallelismSpec(pp=2, pp_schedule="Interleaved1F1B"),
            shape=SHAPE_1B,
            batch=8,
        )

    def test_a_one_layer_shape_is_refused_at_pp_two(self):
        with self.assertRaisesRegex(ValueError, "does not divide evenly"):
            check(PP2, shape=SHAPE_HUGE)

    def test_the_two_suite_shapes_divide_into_four_stages(self):
        """16 and 24 layers over ``pp 4`` with 1F1B, which is 4 stages.

        Both are the shapes the stock-Megatron suite runs. TorchTitan's own
        splitter gives [4, 4, 4, 4] and [6, 6, 6, 6] at weight 0, which is
        the split Megatron gives, so the rule and the engines agree.
        """
        for shape in (SHAPE_1B, SHAPE_9B):
            with self.subTest(model_size=shape.name):
                check(DP2_PP4, shape=shape, batch=8, device_count=8)

    def test_a_one_layer_shape_is_refused_at_pp_four(self):
        """The refusal has to name four stages, not two. Rule 2 admits
        ``pp 4`` now, so rule 7 is what stops a 1-layer shape there."""
        with self.assertRaisesRegex(
            ValueError, r"does not divide evenly into 4 pipeline stages"
        ):
            check(
                ParallelismSpec(pp=4, pp_schedule="1F1B"),
                shape=SHAPE_HUGE,
                batch=8,
            )

    def test_every_registered_shape_at_pp_eight(self):
        """The deepest pipeline the budget holds, shape by shape.

        Four shapes divide: 1b (16 layers, 2 a stage), 9b (24, 3), 30b-a3b
        (48, 6) and 48b (32, 4). Three do not: large has 4 layers, huge and
        giant have 1. Rule 7 is the only rule that reads the layer count, so
        it is the one that decides which shapes the depth-8 cell can run.

        Batch 16 is what rule 12 asks for at eight stages.
        """
        divides = {"1b", "9b", "30b-a3b", "48b"}
        spec = ParallelismSpec(pp=8, pp_schedule="1F1B")
        for name, shape in PIPER_SHAPES.items():
            with self.subTest(model_size=name):
                if name in divides:
                    check(spec, shape=shape, batch=16, device_count=8)
                else:
                    with self.assertRaisesRegex(
                        ValueError,
                        r"does not divide evenly into 8 pipeline stages",
                    ):
                        check(spec, shape=shape, batch=16, device_count=8)

    def test_eight_interleaved_stages_at_pp_four(self):
        """``pp 4`` with a two-stage schedule asks for 8 stages.

        16 and 24 layers both divide by 8. A 4-layer shape does not, and
        rule 7 refuses it at any batch, because rule 7 runs before rules 10
        to 12. Batch 16 is what the two passing shapes need: eight stages
        ask rule 12 for 16 microbatches.
        """
        interleaved = ParallelismSpec(pp=4, pp_schedule="Interleaved1F1B")
        for shape in (SHAPE_1B, SHAPE_9B):
            with self.subTest(model_size=shape.name):
                check(interleaved, shape=shape, batch=16)
        four_layers = PiperShape.derived(name="probe", dim=128, n_layers=4)
        with self.assertRaisesRegex(
            ValueError, r"does not divide evenly into 8 pipeline stages"
        ):
            check(interleaved, shape=four_layers, batch=16)

    def test_a_layer_count_that_misses_the_interleaved_stage_count(self):
        """Two layers cover ``pp * stages_per_rank`` = 2 but not 4."""
        odd = PiperShape.derived(name="probe", dim=128, n_layers=2)
        check(PP2, shape=odd)
        with self.assertRaisesRegex(ValueError, "does not divide evenly"):
            check(
                ParallelismSpec(pp=2, pp_schedule="Interleaved1F1B"),
                shape=odd,
                batch=8,
            )


class Rule08ExpertsDivideTest(unittest.TestCase):
    """Reachable since rule 14 stopped refusing every expert degree.

    Rule 8 runs before rule 14, so an illegal expert count is named by rule
    8 under either parity. A legal one passes under ``shard`` and reaches
    rule 14 under ``replicate``.
    """

    def test_a_legal_expert_split_passes_under_the_sharded_parity(self):
        check(
            ParallelismSpec(dp=2, ep=2, dense_sharding="shard"),
            shape=SHAPE_1B,
        )

    def test_a_legal_expert_split_reaches_rule_fourteen_under_replicate(self):
        with self.assertRaisesRegex(ValueError, "needs --dense-sharding shard"):
            check(ParallelismSpec(dp=2, ep=2), shape=SHAPE_1B)

    def test_more_expert_ranks_than_experts_is_refused(self):
        two_experts = PiperShape.derived(
            name="probe", dim=128, n_layers=4, num_experts=2
        )
        for mode in DENSE_SHARDING_MODES:
            with self.subTest(dense_sharding=mode):
                with self.assertRaisesRegex(ValueError, "exceeds shape"):
                    check(
                        ParallelismSpec(dp=4, ep=4, dense_sharding=mode),
                        shape=two_experts,
                    )

    def test_experts_that_do_not_divide_by_the_degree_are_refused(self):
        for mode in DENSE_SHARDING_MODES:
            with self.subTest(dense_sharding=mode):
                with self.assertRaisesRegex(ValueError, "do not divide evenly"):
                    check(
                        ParallelismSpec(dp=3, ep=3, dense_sharding=mode),
                        shape=SHAPE_1B,
                    )


class Rule09ExpertDegreeDividesDataParallelTest(unittest.TestCase):
    def test_an_expert_degree_that_divides_dp_passes_under_shard(self):
        check(ParallelismSpec(dp=4, ep=2, dense_sharding="shard"))

    def test_an_expert_degree_that_divides_dp_reaches_rule_fourteen(self):
        with self.assertRaisesRegex(ValueError, "needs --dense-sharding shard"):
            check(ParallelismSpec(dp=4, ep=2))

    def test_an_expert_degree_that_does_not_divide_dp_is_refused(self):
        for mode in DENSE_SHARDING_MODES:
            with self.subTest(dense_sharding=mode):
                with self.assertRaisesRegex(
                    ValueError, "does not divide the data"
                ):
                    check(ParallelismSpec(dp=3, ep=2, dense_sharding=mode))


class Rule10BatchDividesIntoMicrobatchesTest(unittest.TestCase):
    def test_an_exact_split_passes(self):
        check(
            ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=2),
            batch=8,
        )

    def test_an_inexact_split_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not divide evenly into"):
            check(
                ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=3),
                batch=4,
            )

    def test_a_microbatch_size_larger_than_the_batch_is_refused(self):
        """It would make ``n_microbatches`` zero in the manifest."""
        with self.assertRaisesRegex(ValueError, "does not divide evenly into"):
            check(
                ParallelismSpec(pp=2, pp_schedule="1F1B", pp_microbatch_size=8),
                batch=4,
            )


class Rule11MicrobatchesDivideByPipelineDegreeTest(unittest.TestCase):
    def test_four_microbatches_over_two_ranks_passes(self):
        check(PP2, batch=4)

    def test_an_odd_microbatch_count_is_refused_at_pp_two(self):
        with self.assertRaisesRegex(ValueError, "do not divide evenly across"):
            check(PP2, batch=3)

    def test_the_rule_is_vacuous_without_a_pipeline(self):
        check(TRIVIAL_SPEC, batch=3)

    def test_seven_microbatches_at_pp_four_is_refused_by_this_rule(self):
        """**Seven is below rule 12's floor of eight and rule 11 fires
        first.**

        Rule 11 runs before rule 12, so a count that fails both is reported
        as an uneven split across the pipeline degree. Read the message
        before you conclude that a batch was too small: the repair differs.
        Seven microbatches needs a batch that divides by 4; four
        microbatches needs a larger batch.
        """
        with self.assertRaisesRegex(
            ValueError,
            r"7 microbatches do not divide evenly across pipeline degree 4",
        ):
            check(DP2_PP4, batch=7, device_count=8)


class Rule12MicrobatchesCoverTheWarmupTest(unittest.TestCase):
    def test_the_milestone_cell_passes(self):
        """**pp 2, batch 4, 1F1B.** Four microbatches against the four that
        ``2 * pp * stages_per_rank`` asks for. This is the cell the first
        cross-engine pipeline number comes from; a rule that refused it
        would take the milestone away."""
        self.assertEqual(n_microbatches(PP2, local_batch_size=4), 4)
        check(PP2, batch=4)

    def test_too_few_microbatches_is_refused(self):
        with self.assertRaisesRegex(ValueError, "is below the"):
            check(PP2, batch=2)

    def test_interleaved_needs_a_larger_batch(self):
        interleaved = ParallelismSpec(pp=2, pp_schedule="Interleaved1F1B")
        with self.assertRaisesRegex(ValueError, "is below the"):
            check(interleaved, batch=4)
        check(interleaved, batch=8)

    def test_the_pp_four_cell_passes_at_exactly_eight_microbatches(self):
        """Both batch settings the stock-Megatron suite may use.

        ``pp 4`` with 1F1B is 4 stages, so rule 12 asks for 8 microbatches.
        Batch 32 with microbatch 4 gives 8, and batch 8 with microbatch 1
        gives 8. Both sit exactly on the floor.
        """
        self.assertEqual(
            n_microbatches(DP2_PP4_MICRO4, local_batch_size=32), 8
        )
        self.assertEqual(n_microbatches(DP2_PP4, local_batch_size=8), 8)
        check(DP2_PP4_MICRO4, batch=32, device_count=8)
        check(DP2_PP4, batch=8, device_count=8)

    def test_the_pp_four_cell_is_refused_below_eight_microbatches(self):
        """The floor from the other side, at both batch settings.

        Four microbatches divides by ``pp 4``, so rule 11 passes and rule 12
        is what refuses it. The message names the 8 the cell needs.
        """
        for spec, batch in ((DP2_PP4, 4), (DP2_PP4_MICRO4, 16)):
            with self.subTest(microbatch=spec.pp_microbatch_size, batch=batch):
                self.assertEqual(
                    n_microbatches(spec, local_batch_size=batch), 4
                )
                with self.assertRaisesRegex(
                    ValueError,
                    r"4 microbatches is below the 8 that pp 4 x 1 stage",
                ):
                    check(spec, batch=batch, device_count=8)

    def test_the_default_batch_does_not_reach_pp_four(self):
        """The ``Workload`` default is batch 4, so a bare ``--pp 4`` fails
        rule 12 even now that rule 2 admits the degree. An operator has to
        raise the batch on purpose."""
        with self.assertRaisesRegex(ValueError, r"is below the 8 that pp 4"):
            check(ParallelismSpec(pp=4, pp_schedule="1F1B"), batch=4)

    def test_the_pp_eight_floor_is_sixteen_microbatches(self):
        """**16 passes and 8 fails. The boundary is not 16 against 15.**

        Eight stages ask for ``2 * 8`` microbatches. 15 microbatches does
        not divide by 8, so it fires rule 11 -- a different rule, with a
        different repair -- and would prove nothing about this one. Batch 16
        with microbatch 2 gives 8, which divides by 8 and reaches rule 12.
        """
        passing = ParallelismSpec(
            pp=8, pp_schedule="1F1B", pp_microbatch_size=2
        )
        self.assertEqual(n_microbatches(passing, local_batch_size=32), 16)
        check(passing, batch=32, device_count=8)
        self.assertEqual(n_microbatches(passing, local_batch_size=16), 8)
        with self.assertRaisesRegex(
            ValueError,
            r"8 microbatches is below the 16 that pp 8 x 1 stage",
        ):
            check(passing, batch=16, device_count=8)

    def test_fifteen_microbatches_at_pp_eight_fires_rule_eleven(self):
        """Named so that nobody writes 15 into the test above.

        Rule 11 runs first and reports an uneven split across the pipeline
        degree. The repair differs: 15 needs a count that divides by 8, and
        8 needs a larger batch.
        """
        with self.assertRaisesRegex(
            ValueError,
            r"15 microbatches do not divide evenly across pipeline degree 8",
        ):
            check(
                ParallelismSpec(
                    pp=8, pp_schedule="1F1B", pp_microbatch_size=2
                ),
                batch=30,
                device_count=8,
            )

    def test_a_single_gpu_run_at_batch_one_is_not_refused(self):
        """The guard that keeps rule 12 a pipeline rule.

        ``--batch`` takes any positive integer and there is no pipeline at
        pp 1, so an unconditional rule 12 would refuse a legal single-GPU
        run that this repo can do today.
        """
        check(TRIVIAL_SPEC, batch=1)
        check(TRIVIAL_SPEC, batch=2)


class Rule13CudaGraphTest(unittest.TestCase):
    def test_a_single_gpu_run_keeps_every_compile_mode(self):
        for mode in ("default", "cuda-graph", "none"):
            with self.subTest(compile_mode=mode):
                check(TRIVIAL_SPEC, compile_mode=mode)

    def test_a_pipeline_run_is_refused_under_cuda_graph(self):
        with self.assertRaisesRegex(ValueError, "cuda-graph"):
            check(PP2, compile_mode="cuda-graph")

    def test_a_data_parallel_run_is_refused_under_cuda_graph(self):
        with self.assertRaisesRegex(ValueError, "cuda-graph"):
            check(ParallelismSpec(dp=2), compile_mode="cuda-graph")

    def test_a_parallel_run_keeps_the_other_two_modes(self):
        check(PP2, compile_mode="default")
        check(PP2, compile_mode="none")


class Rule14ExpertParallelismNeedsTheShardedParityTest(unittest.TestCase):
    """TorchTitan cannot split the experts and replicate the dense
    parameters, so the two engines compare under an expert degree only when
    both shard. The rule refuses the other combination and names the flag.
    """

    def test_the_trivial_expert_degree_passes_under_either_parity(self):
        check(TRIVIAL_SPEC)
        check(ParallelismSpec(dp=2, ep=1))
        check(ParallelismSpec(dp=2, ep=1, dense_sharding="shard"))

    def test_every_expert_split_passes_under_the_sharded_parity(self):
        for spec in (
            ParallelismSpec(dp=2, ep=2, dense_sharding="shard"),
            ParallelismSpec(dp=4, ep=2, dense_sharding="shard"),
            ParallelismSpec(dp=4, ep=4, dense_sharding="shard"),
            ParallelismSpec(
                dp=2,
                pp=4,
                pp_schedule="1F1B",
                ep=2,
                dense_sharding="shard",
            ),
        ):
            with self.subTest(spec=spec):
                check(spec, batch=8, device_count=spec.world_size)

    def test_every_expert_split_is_refused_under_the_replicated_parity(self):
        for spec in (
            ParallelismSpec(dp=2, ep=2),
            ParallelismSpec(dp=4, ep=2),
            ParallelismSpec(dp=4, ep=4),
        ):
            with self.subTest(spec=spec):
                with self.assertRaisesRegex(
                    ValueError,
                    r"expert degree \d+ needs --dense-sharding shard",
                ):
                    check(spec)

    def test_the_message_names_the_flag_that_repairs_it(self):
        """A refusal that named no repair would leave the operator to read
        this module. The parity is a flag, so the message says which one."""
        with self.assertRaisesRegex(
            ValueError, "--dense-sharding shard"
        ) as raised:
            check(ParallelismSpec(dp=2, ep=2))
        self.assertIn("'replicate'", str(raised.exception))


class Rule15ShardNeedsADataParallelWidthTest(unittest.TestCase):
    """``shard`` at ``dp`` 1 shards one copy over one rank, which is what a
    replicated run already holds. The manifest would then record a parity
    the run did not have.
    """

    def test_the_sharded_parity_passes_above_dp_one(self):
        for dp in (2, 4, 8):
            with self.subTest(dp=dp):
                check(ParallelismSpec(dp=dp, dense_sharding="shard"))

    def test_the_sharded_parity_is_refused_at_dp_one(self):
        for spec in (
            ParallelismSpec(dense_sharding="shard"),
            ParallelismSpec(pp=2, pp_schedule="1F1B", dense_sharding="shard"),
            ParallelismSpec(pp=8, pp_schedule="1F1B", dense_sharding="shard"),
        ):
            with self.subTest(spec=spec):
                with self.assertRaisesRegex(
                    ValueError,
                    r"--dense-sharding shard needs a data-parallel degree "
                    r"above 1",
                ):
                    check(spec, batch=16)

    def test_the_depth_eight_cell_forces_the_replicated_parity(self):
        """``pp 8`` fills the budget, so ``dp`` is 1 and this rule then
        refuses ``shard``. The depth-8 cells are replicated by arithmetic
        rather than by choice."""
        check(ParallelismSpec(pp=8, pp_schedule="1F1B"), batch=16)
        with self.assertRaisesRegex(ValueError, "needs a data-parallel degree"):
            check(
                ParallelismSpec(
                    pp=8, pp_schedule="1F1B", dense_sharding="shard"
                ),
                batch=16,
            )

    def test_the_replicated_parity_is_untouched_at_dp_one(self):
        """Every number this repo has published was measured at ``dp`` 1
        under ``replicate``. This rule may not reach one of those."""
        check(TRIVIAL_SPEC)
        check(PP2)


class Rule16TheTunedMegatronDriverTakesNeitherTest(unittest.TestCase):
    """``benchmarks/e2e/megatron/train.py`` implements neither the sharded
    parity nor an expert degree, and its command line carries no flag for
    either. So a spec that asks for one would be ignored by that arm and
    recorded by the manifest anyway.

    **Both halves became reachable in this pass.** Rule 14 refused every
    expert degree before it, and the parity did not exist. Without this rule
    a cross-engine row would put a sharded, expert-split TorchTitan arm
    against a replicated, unsplit Megatron arm, under one manifest claiming
    both sides had the same mesh.
    """

    TUNED = ("torchtitan", "megatron")
    STOCK = ("torchtitan", "megatron_stock")

    def test_the_set_names_the_tuned_driver_alone(self):
        """It is a fact about one driver's source, not about Megatron-LM.
        ``megatron_stock`` hands the run to Megatron's own ``pretrain``,
        which implements both."""
        self.assertEqual(REPLICATE_ONLY_LAUNCHERS, frozenset({"megatron"}))
        self.assertIn("megatron", MEGATRON_LAUNCHERS)
        self.assertNotIn("megatron_stock", REPLICATE_ONLY_LAUNCHERS)
        self.assertNotIn("torchtitan", REPLICATE_ONLY_LAUNCHERS)

    # Every launcher the registry runs, and whether its driver implements
    # the sharded parity and an expert degree. Written out rather than
    # derived: the answer is a fact about each driver's source, and a
    # derivation would agree with the set by construction and check nothing.
    IMPLEMENTS_NEITHER = {
        "torchtitan": False,
        "megatron": True,
        "megatron_stock": False,
    }

    def test_the_set_is_the_declared_classification(self):
        """A launcher this set does not name is one rule 16 admits, so a new
        launcher must not pass quietly.

        The table above is the classification, and this asserts the set is
        exactly its true half. ``MegatronLauncherSetTest`` already checks
        that every registry launcher is a known one; the point here is that
        an author who adds a launcher there still has to answer this
        question, and cannot make the suite green by editing one set.
        """
        launchers = {
            arm.launcher
            for scenario in SCENARIOS.values()
            for arm in scenario.arms
        }
        self.assertEqual(
            set(self.IMPLEMENTS_NEITHER),
            launchers,
            "classify every registry launcher in IMPLEMENTS_NEITHER: say "
            "whether its driver implements the sharded parity and an "
            "expert degree, then make REPLICATE_ONLY_LAUNCHERS agree",
        )
        self.assertEqual(
            REPLICATE_ONLY_LAUNCHERS,
            frozenset(
                name
                for name, neither in self.IMPLEMENTS_NEITHER.items()
                if neither
            ),
            "REPLICATE_ONLY_LAUNCHERS disagrees with the table above",
        )

    def test_the_tuned_arm_refuses_an_expert_degree(self):
        with self.assertRaisesRegex(
            ValueError, r"expert degree 2 is not implemented by the megatron"
        ):
            check(
                ParallelismSpec(dp=2, ep=2, dense_sharding="shard"),
                engines=self.TUNED,
            )

    def test_the_tuned_arm_refuses_the_sharded_parity(self):
        with self.assertRaisesRegex(
            ValueError,
            r"--dense-sharding shard is not implemented by the megatron",
        ):
            check(
                ParallelismSpec(dp=2, dense_sharding="shard"),
                engines=self.TUNED,
            )

    def test_the_expert_half_names_the_expert_degree_and_not_the_parity(self):
        """It is checked first on purpose. Rule 14 already ties an expert
        degree to the sharded parity, so the shard half alone would refuse
        every expert spec that reached here -- under a message about
        sharding, which is not what the operator asked for."""
        with self.assertRaisesRegex(ValueError, "expert degree") as raised:
            check(
                ParallelismSpec(dp=2, ep=2, dense_sharding="shard"),
                engines=self.TUNED,
            )
        self.assertNotIn("--dense-sharding shard is not", str(raised.exception))

    def test_the_stock_arm_takes_both(self):
        """Stock Megatron implements both, so the rule must not reach it."""
        check(ParallelismSpec(dp=2, dense_sharding="shard"), engines=self.STOCK)
        check(
            ParallelismSpec(dp=2, ep=2, dense_sharding="shard"),
            engines=self.STOCK,
        )

    def test_each_half_names_the_repair_under_its_own_message(self):
        """Rule 14's own comment sets the standard: the refusal names the
        flag that repairs it.

        **Each half is matched by its own opening words.** A bare
        ``assertIn("--arm", ...)`` passes when one half is deleted, because
        the surviving half catches the other spec and its message also holds
        the word. Measured: with the expert half removed, both cases below
        fell through to the shard half and both still passed.

        It matches ``run --arm`` rather than ``--arm``. ``run-all`` carries
        PASSTHROUGH_CONTEXT and forwards an unknown flag to the training
        subprocess, so a message naming the bare flag would send a
        ``run-all`` operator to a flag that command ignores.
        """
        for spec, opening in (
            (
                ParallelismSpec(dp=2, ep=2, dense_sharding="shard"),
                r"expert degree 2 is not implemented",
            ),
            (
                ParallelismSpec(dp=2, dense_sharding="shard"),
                r"--dense-sharding shard is not implemented",
            ),
        ):
            with self.subTest(spec=spec):
                with self.assertRaisesRegex(ValueError, opening) as raised:
                    check(spec, engines=self.TUNED)
                self.assertIn("run --arm", str(raised.exception))

    def test_a_titan_only_roster_passes_this_rule(self):
        """The state the repair reaches. ``engines`` is the launcher set of
        the arms the run will really start, so a roster without the tuned
        driver passes.

        **This says the rule admits it. It does not say the run succeeds.**
        ``parallelize_piper1b`` now admits an explicit shard degree, so the
        subprocess no longer refuses such a roster -- but no sharded arm has
        run on a GPU, so this rule is the only thing under test here.
        ``ResolveRunTests.test_an_arm_subset_narrows_the_engine_set_spec_rule_16_reads``
        checks the other half of the claim, that ``--arm`` really narrows
        the set.
        """
        check(ParallelismSpec(dp=2, dense_sharding="shard"))
        check(ParallelismSpec(dp=2, ep=2, dense_sharding="shard"))

    def test_the_tuned_arm_keeps_every_mesh_it_can_reach(self):
        """The rule may not refuse a mesh the tuned driver already runs.
        Every recorded megatron cell is replicated with no expert split, so
        rule 16 has to admit all of them.

        **The last row is planned, not recorded.** Every manifest under
        ``out/`` that names the ``megatron`` launcher is ``piper1b_megatron``
        at ``pp2``, ``dp2`` or ``dp2 x pp2``. ``DP2_PP4`` has run on
        ``megatron_stock``, which is a different launcher and a different
        scenario. It is swept here because rule 16 must not refuse it, not
        because the tuned arm has measured it.
        """
        for spec, batch, devices in (
            (TRIVIAL_SPEC, 4, 1),
            (PP2, 4, 2),
            (ParallelismSpec(dp=2), 4, 2),
            (ParallelismSpec(dp=2, pp=2, pp_schedule="1F1B"), 4, 4),
            (DP2_PP4, 8, 8),
        ):
            with self.subTest(spec=spec):
                check(
                    spec,
                    batch=batch,
                    device_count=devices,
                    engines=self.TUNED,
                )


class Rule17MegatronFsdpCannotHoldAPipelineTest(unittest.TestCase):
    """Megatron-FSDP factors the GLOBAL world size into ``dp_cp x ep x tp``
    and has no pipeline term, so it builds a mesh only at ``pp`` 1.

    **This rule exists because a run measured it.** On 2026-08-28 a
    ``--dense-sharding shard --dp 2 --pp 4`` cell died on all eight ranks in
    20 seconds inside ``einops.rearrange``, before the wrapper existed. The
    error names no flag of ours and no repair, so the refusal has to happen
    here instead.

    **The rule is engine-scoped.** TorchTitan shards under a pipeline
    through ``fully_shard``, so a titan-only roster passes.
    """

    STOCK = ("torchtitan", "megatron_stock")
    TITAN = ("torchtitan",)

    def test_the_set_names_the_stock_driver_alone(self):
        """``megatron`` shards not at all and rule 16 refuses it earlier;
        ``torchtitan`` shards through ``fully_shard``, which holds a
        pipeline."""
        self.assertEqual(MEGATRON_FSDP_LAUNCHERS, frozenset({"megatron_stock"}))
        self.assertNotIn("megatron", MEGATRON_FSDP_LAUNCHERS)
        self.assertNotIn("torchtitan", MEGATRON_FSDP_LAUNCHERS)

    def test_the_two_sets_are_disjoint(self):
        """A launcher cannot both shard through Megatron-FSDP and implement
        no sharding at all. Rule 16 would refuse it first, so a launcher in
        both sets would make rule 17 unreachable for it."""
        self.assertEqual(
            MEGATRON_FSDP_LAUNCHERS & REPLICATE_ONLY_LAUNCHERS,
            frozenset(),
        )

    def test_the_sharded_pipeline_is_refused(self):
        with self.assertRaisesRegex(
            ValueError,
            r"--dense-sharding shard with pp 4 is not buildable by the "
            r"megatron_stock driver",
        ):
            check(
                ParallelismSpec(
                    dp=2, pp=4, pp_schedule="1F1B", dense_sharding="shard"
                ),
                engines=self.STOCK,
                batch=32,
            )

    def test_the_message_names_the_missing_factor(self):
        """An operator who reads only the message must learn that the
        missing factor is the pipeline degree, and that ``--pp 1`` is the
        repair. The einops error says neither."""
        with self.assertRaises(ValueError) as raised:
            check(
                ParallelismSpec(
                    dp=2, pp=4, pp_schedule="1F1B", dense_sharding="shard"
                ),
                engines=self.STOCK,
                batch=32,
            )
        message = str(raised.exception)
        self.assertIn("no pipeline term", message)
        self.assertIn("the missing factor is exactly pp 4", message)
        self.assertIn("The product is 2 and the world is 8", message)
        self.assertIn("Use --pp 1", message)

    def test_an_expert_split_under_a_pipeline_is_refused(self):
        """Rule 14 ties an expert degree to the sharded parity, so every
        expert spec reaches this rule as a sharded one."""
        with self.assertRaisesRegex(
            ValueError, r"--dense-sharding shard with pp 4 is not buildable"
        ):
            check(
                ParallelismSpec(
                    dp=2,
                    pp=4,
                    ep=2,
                    pp_schedule="1F1B",
                    dense_sharding="shard",
                ),
                engines=self.STOCK,
                batch=32,
            )

    def test_the_sharded_mesh_without_a_pipeline_passes(self):
        """``--dp 8 --pp 1`` satisfies the pattern at eight ranks: the
        product is 8 and the world is 8."""
        check(
            ParallelismSpec(dp=8, dense_sharding="shard"),
            engines=self.STOCK,
            batch=32,
            device_count=8,
        )

    def test_the_expert_split_without_a_pipeline_passes(self):
        """At ``ep`` 2 the product is ``4 x 2 x 1``, which is still 8."""
        check(
            ParallelismSpec(dp=8, ep=2, dense_sharding="shard"),
            engines=self.STOCK,
            batch=32,
            device_count=8,
        )

    def test_the_replicated_pipeline_passes(self):
        """The rule reads the parity, not the pipeline alone. A replicated
        run builds no Megatron-FSDP mesh."""
        check(
            ParallelismSpec(
                dp=2, pp=4, pp_schedule="1F1B", dense_sharding="replicate"
            ),
            engines=self.STOCK,
            batch=32,
            device_count=8,
        )

    def test_the_titan_arms_alone_pass(self):
        """TorchTitan shards under a pipeline through ``fully_shard``, so
        ``run --arm`` narrowing to the titan arms takes this axis off the
        Megatron driver."""
        check(
            ParallelismSpec(
                dp=2, pp=4, pp_schedule="1F1B", dense_sharding="shard"
            ),
            engines=self.TITAN,
            batch=32,
            device_count=8,
        )


class TheEightGpuCellTest(unittest.TestCase):
    """The whole ``dp 2 x pp 4`` cell, rule by rule rather than by half.

    Rule 2 is what used to refuse this mesh. Every other rule has to admit
    it for its own reason, and this class asks each of them at eight
    devices, at both shapes the suite runs, and at both batch settings.
    """

    def test_the_cell_passes_at_eight_devices_for_both_engines(self):
        for shape in (SHAPE_1B, SHAPE_9B):
            for engines in (("torchtitan",), ("torchtitan", "megatron")):
                for spec, batch in ((DP2_PP4, 8), (DP2_PP4_MICRO4, 32)):
                    with self.subTest(
                        model_size=shape.name,
                        engines=engines,
                        batch=batch,
                    ):
                        check(
                            spec,
                            shape=shape,
                            batch=batch,
                            engines=engines,
                            device_count=8,
                        )

    def test_the_cell_still_has_to_fill_the_devices(self):
        """Rule 1 is unchanged. Eight ranks need eight devices."""
        for device_count in (4, 7, 9):
            with self.subTest(device_count=device_count):
                with self.assertRaisesRegex(
                    ValueError,
                    r"world size 8 \(dp 2 x pp 4\) does not match the "
                    rf"{device_count} device",
                ):
                    check(DP2_PP4, batch=8, device_count=device_count)

    def test_the_cell_is_refused_under_cuda_graph(self):
        """Rule 13 reads the world size, so it grows with the mesh."""
        with self.assertRaisesRegex(
            ValueError, r"refused at world size 8"
        ):
            check(
                DP2_PP4, batch=8, device_count=8, compile_mode="cuda-graph"
            )

    def test_the_cell_keeps_the_other_two_compile_modes(self):
        for mode in ("default", "none"):
            with self.subTest(compile_mode=mode):
                check(DP2_PP4, batch=8, device_count=8, compile_mode=mode)

    def test_the_cell_records_eight_ranks_and_eight_microbatches(self):
        self.assertEqual(DP2_PP4.world_size, 8)
        self.assertEqual(
            describe(DP2_PP4_MICRO4, local_batch_size=32),
            {
                "dp": 2,
                "pp": 4,
                "ep": 1,
                "pp_schedule": "1F1B",
                "pp_microbatch_size": 4,
                "dense_sharding": "replicate",
                "world_size": 8,
                "dp_replicate": 2,
                "dp_shard": 1,
                "n_microbatches": 8,
            },
        )

    def test_the_cell_asks_torchtitan_to_replicate_rather_than_shard(self):
        """``dp_shard`` 1 replicates. The harness must send it, because
        TorchTitan resolves an omitted shard degree to every remaining
        rank, which at eight ranks is ZeRO-3 under a replication label."""
        self.assertEqual(titan_mesh(DP2_PP4), (2, 1))
        self.assertFalse(skip_dp(DP2_PP4))


class CapsThatMovedTest(unittest.TestCase):
    """Specs whose verdict changed when the caps rose, named one by one.

    A cap refuses a whole class of spec, so lifting one can hand a spec to
    a later rule, to no rule at all, or leave it refused by a rule that was
    never reached. All three happen here. A test that only asserted "this
    raises" would stay green through every one of them and say nothing.
    """

    def test_a_five_rank_data_parallel_mesh_is_now_accepted(self):
        """World size 5 was above the old 4-GPU budget. Nothing else reads
        it, so it now passes outright."""
        check(ParallelismSpec(dp=5))
        check(ParallelismSpec(dp=6))
        check(ParallelismSpec(dp=7))
        check(ParallelismSpec(dp=8))

    def test_pipeline_degree_three_is_now_refused_by_rule_seven(self):
        """**The verdict is the same and the reason is not.**

        ``pp 3`` used to fail the cap. The cap admits it now, and 16 layers
        do not divide into 3 stages, so rule 7 refuses it instead. Read the
        message: the repair is a shape with 24 layers, not a smaller
        degree.
        """
        with self.assertRaisesRegex(
            ValueError,
            r"'1b' has 16 layers, which does not divide evenly into 3",
        ):
            check(ParallelismSpec(pp=3, pp_schedule="1F1B"), batch=6)

    def test_pipeline_degree_three_passes_on_a_shape_that_divides(self):
        """24 layers divide into 3 stages, so nothing refuses the spec.
        Rule 12 asks for 6 microbatches and rule 11 asks that 6 divide by
        3."""
        check(
            ParallelismSpec(pp=3, pp_schedule="1F1B"),
            shape=SHAPE_9B,
            batch=6,
        )

    def test_pipeline_degree_four_is_no_longer_refused_by_the_cap(self):
        """The change the earlier lift existed to make."""
        check(ParallelismSpec(pp=4, pp_schedule="1F1B"), batch=8)

    def test_pipeline_degree_eight_is_no_longer_refused_by_the_cap(self):
        """The change this lift exists to make.

        One pipeline of eight stages fills the whole budget, so ``dp`` is 1
        and nothing else reads the mesh. Rule 12 asks for 16 microbatches at
        eight stages, which batch 16 gives.
        """
        check(ParallelismSpec(pp=8, pp_schedule="1F1B"), batch=16)

    def test_pipeline_degree_five_to_seven_is_now_refused_by_rule_seven(self):
        """**The verdict is the same and the reason is not.**

        Each used to fail the cap. The cap admits all three now, and 16
        layers divide into none of 5, 6 or 7 stages, so rule 7 refuses them
        instead. Read the message: the repair is a shape whose layer count
        divides, not a smaller degree.
        """
        for pp in (5, 6, 7):
            with self.subTest(pp=pp):
                with self.assertRaisesRegex(
                    ValueError,
                    rf"'1b' has 16 layers, which does not divide evenly "
                    rf"into {pp}",
                ):
                    check(
                        ParallelismSpec(pp=pp, pp_schedule="1F1B"),
                        batch=2 * pp,
                    )


class PreconditionsOnTheBorrowedArgumentsTest(unittest.TestCase):
    """The two values this module reads but does not own.

    Both are checked before the numbered rules, because a rule that reads an
    unchecked value can fail open.
    """

    def test_every_declared_compile_mode_is_accepted(self):
        for mode in COMPILE_MODES:
            with self.subTest(compile_mode=mode):
                check(TRIVIAL_SPEC, compile_mode=mode)

    def test_an_unknown_compile_mode_is_refused(self):
        """Rules 6 and 13 read the mode in OPPOSITE directions: rule 6
        refuses anything outside the uncompiled set and so fails safe on an
        unknown name, while rule 13 refuses only names inside the cudagraph
        set and so fails OPEN on one. Without this precondition,
        ``reduce-overhead`` -- the torch-level spelling of ``cuda-graph``,
        which schema <= 7 manifests record -- would slip a graph-capturing
        run past rule 13."""
        for mode in ("reduce-overhead", "max-autotune", "", "eager"):
            with self.subTest(compile_mode=mode):
                with self.assertRaisesRegex(ValueError, "Unknown compile mode"):
                    check(PP2, compile_mode=mode)

    def test_a_batch_below_one_is_refused(self):
        """Neither ``Workload`` nor ``workload_with_overrides`` bounds it --
        ``--batch`` takes a bare int -- and it is the one integer the
        microbatch arithmetic divides."""
        for batch in (0, -4):
            with self.subTest(batch=batch):
                with self.assertRaisesRegex(ValueError, "must be >= 1"):
                    check(TRIVIAL_SPEC, batch=batch)


class TheSingleGpuRunStaysLegalTest(unittest.TestCase):
    """Nothing this repo can run today may become unrunnable.

    The trivial spec is what every published number was measured under, so
    it has to survive every other axis: each registered shape, each compile
    mode, each engine roster, and any batch size.
    """

    def test_every_registered_shape_passes_at_the_trivial_spec(self):
        for name in PIPER_SHAPES:
            with self.subTest(model_size=name):
                check(TRIVIAL_SPEC, shape=PIPER_SHAPES[name])

    def test_every_compile_mode_and_engine_roster_passes(self):
        for mode in COMPILE_MODES:
            for engines in ((), ("torchtitan",), ("torchtitan", "megatron")):
                with self.subTest(compile_mode=mode, engines=engines):
                    check(TRIVIAL_SPEC, compile_mode=mode, engines=engines)

    def test_any_batch_size_passes(self):
        for batch in (1, 2, 3, 4, 8, 48):
            with self.subTest(batch=batch):
                check(TRIVIAL_SPEC, batch=batch)


class ValidatorInterfaceTest(unittest.TestCase):
    def test_the_engine_set_may_be_any_iterable(self):
        """It is built from ``{arm.launcher for arm in arms}`` at the call
        site, and a one-shot iterator must not read differently."""
        check(PP2, engines=iter(("torchtitan", "megatron")))
        check(PP2, engines=frozenset({"megatron"}))
        check(PP2, engines=())

    def test_a_valid_spec_returns_none(self):
        self.assertIsNone(
            validate_parallelism(
                PP2,
                shape=SHAPE_1B,
                workload=workload(4),
                compile_mode="default",
                engines=("torchtitan", "megatron"),
                device_count=2,
            )
        )


# --------------------------------------------------------------------------
# The import budget
# --------------------------------------------------------------------------

_IMPORT_PROBE = """
import importlib
import json
import sys

importlib.import_module("benchmarks.e2e.parallelism")
heavy = ["torch", "torchtitan", "megatron", "transformer_engine", "triton"]
print(json.dumps(sorted(m for m in heavy if m in sys.modules)))
"""


class ImportBudgetTest(unittest.TestCase):
    def test_the_module_imports_no_torch(self):
        """A subprocess, because this suite imports torch elsewhere and an
        in-process ``sys.modules`` check would always see it.

        The megatron driver reads a spec, and it defers its own torch import
        so ``--help`` costs nothing; a torch import here would undo that.
        """
        env = dict(os.environ)
        pythonpath = env.get("PYTHONPATH")
        env["PYTHONPATH"] = str(REPO_ROOT) + (
            os.pathsep + pythonpath if pythonpath else ""
        )
        completed = subprocess.run(
            [sys.executable, "-c", _IMPORT_PROBE],
            cwd=REPO_ROOT,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(
            completed.returncode,
            0,
            f"probe failed:\n{completed.stdout}\n{completed.stderr}",
        )
        self.assertEqual(json.loads(completed.stdout.splitlines()[-1]), [])

    def test_the_module_imports_only_the_declared_dependencies(self):
        """Static: the module scope may reach the stdlib, the shape registry
        and the compile-mode constants, and nothing else."""
        source = (REPO_ROOT / "benchmarks" / "e2e" / "parallelism.py").read_text()
        tree = ast.parse(source)
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module)
        self.assertEqual(
            {name for name in imported if name.startswith("benchmarks")},
            {
                "benchmarks.e2e.registry",
                "benchmarks.models.piper_qwen3.shape",
            },
        )


if __name__ == "__main__":
    unittest.main()
