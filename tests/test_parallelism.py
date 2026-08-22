"""CPU-only tests for the parallelism run axis.

``benchmarks/e2e/parallelism.py`` lands before anything imports it, so this
file is the whole check on it: every one of the fourteen validator rules is
exercised in both directions, the four derivations are pinned, and the
schedule registry is checked against the PyTorch classes it names.

Two hazards drive the shape of the file. A rule that **admits an illegal
mesh** publishes a number under a topology the run did not have. A rule that
**refuses a legal one** costs a measurement that should have happened -- the
pp=2 / batch 4 / 1F1B milestone is the case that must pass, and the
single-GPU ``--batch 1`` run is the case an over-eager rule 12 would take
away.
"""

import ast
import inspect
import json
import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.parallelism import (
    MAX_PP,
    MAX_WORLD_SIZE,
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
from benchmarks.e2e.registry import COMPILE_MODES, EXECUTION_MODEL, Workload
from benchmarks.models.piper_qwen3.shape import (
    PIPER_SHAPES,
    PiperShape,
    shape_by_name,
)


REPO_ROOT = Path(__file__).resolve().parent.parent

SHAPE_1B = shape_by_name("1b")  # 16 layers, 4 experts
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
        """The precondition the fourteen rules assume.

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

    def test_the_spec_is_frozen(self):
        with self.assertRaises(Exception):
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


class TitanMeshTest(unittest.TestCase):
    def test_every_legal_dp_ep_pair_up_to_world_size_four(self):
        """``(dp_replicate, dp_shard)`` at each pair the budget allows.

        ``ep > 1`` borrows the shard axis, so the replicate degree is what
        is left of ``dp``. Rule 14 refuses these specs today; the mesh
        arithmetic is what the expert-parallel stage will inherit, so it is
        pinned now.
        """
        expected = {
            (1, 1): (1, 1),
            (2, 1): (2, 1),
            (3, 1): (3, 1),
            (4, 1): (4, 1),
            (2, 2): (1, 2),
            (4, 2): (2, 2),
            (4, 4): (1, 4),
        }
        for (dp, ep), mesh in expected.items():
            with self.subTest(dp=dp, ep=ep):
                self.assertEqual(titan_mesh(ParallelismSpec(dp=dp, ep=ep)), mesh)

    def test_the_mesh_product_is_the_data_parallel_width(self):
        for dp, ep in ((1, 1), (2, 1), (2, 2), (4, 2), (4, 4)):
            with self.subTest(dp=dp, ep=ep):
                replicate, shard = titan_mesh(ParallelismSpec(dp=dp, ep=ep))
                self.assertEqual(replicate * shard, dp)

    def test_the_pipeline_degree_does_not_reach_the_mesh(self):
        self.assertEqual(titan_mesh(ParallelismSpec(dp=2, pp=2)), (2, 1))


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

    def test_the_parallel_parts_name_degrees_and_not_mechanisms(self):
        """One manifest carries one execution_model for a whole run, and a
        cross-engine run holds arms of both engines. A term only TorchTitan's
        code produces -- ``fsdp2``, a shard degree -- would be false for the
        megatron arm beside it. ``titan_mesh``'s resolution belongs in
        ``describe``, under names that say whose it is."""
        for spec in (
            ParallelismSpec(dp=2),
            ParallelismSpec(dp=4),
            ParallelismSpec(dp=2, ep=2),
            ParallelismSpec(dp=2, pp=2, pp_schedule="1F1B"),
        ):
            with self.subTest(spec=spec):
                rendered = execution_model(spec)
                for engine_term in ("fsdp2", "replicate", "shard", "ddp"):
                    self.assertNotIn(engine_term, rendered)
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
                "world_size": 2,
                "dp_replicate": 1,
                "dp_shard": 1,
                "n_microbatches": 4,
            },
        )

    def test_the_record_is_json_safe(self):
        for spec in (
            TRIVIAL_SPEC,
            PP2,
            ParallelismSpec(dp=2, pp=2, pp_schedule="1F1B"),
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
    def test_the_largest_budgeted_mesh_passes(self):
        check(ParallelismSpec(dp=MAX_WORLD_SIZE))
        check(ParallelismSpec(dp=2, pp=MAX_PP, pp_schedule="1F1B"))

    def test_a_world_size_above_the_budget_is_refused(self):
        with self.assertRaisesRegex(ValueError, "budget"):
            check(ParallelismSpec(dp=MAX_WORLD_SIZE * 2))

    def test_a_pipeline_deeper_than_two_is_refused(self):
        """World size 4 satisfies rule 1 and the budget, so this isolates
        the pp half of rule 2."""
        with self.assertRaisesRegex(ValueError, "pipeline degree"):
            check(ParallelismSpec(pp=4, pp_schedule="1F1B"))


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
    """Dead behind rule 14. A legal expert split must reach rule 14's
    message, and an illegal one must be named by rule 8 first."""

    def test_a_legal_expert_split_reaches_rule_fourteen(self):
        with self.assertRaisesRegex(ValueError, "not supported yet"):
            check(ParallelismSpec(dp=2, ep=2), shape=SHAPE_1B)

    def test_more_expert_ranks_than_experts_is_refused(self):
        two_experts = PiperShape.derived(
            name="probe", dim=128, n_layers=4, num_experts=2
        )
        with self.assertRaisesRegex(ValueError, "exceeds shape"):
            check(ParallelismSpec(dp=4, ep=4), shape=two_experts)

    def test_experts_that_do_not_divide_by_the_degree_are_refused(self):
        with self.assertRaisesRegex(ValueError, "do not divide evenly"):
            check(ParallelismSpec(dp=3, ep=3), shape=SHAPE_1B)


class Rule09ExpertDegreeDividesDataParallelTest(unittest.TestCase):
    def test_an_expert_degree_that_divides_dp_reaches_rule_fourteen(self):
        with self.assertRaisesRegex(ValueError, "not supported yet"):
            check(ParallelismSpec(dp=4, ep=2))

    def test_an_expert_degree_that_does_not_divide_dp_is_refused(self):
        with self.assertRaisesRegex(ValueError, "does not divide the data"):
            check(ParallelismSpec(dp=3, ep=2))


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


class Rule14ExpertParallelismIsRefusedTest(unittest.TestCase):
    def test_the_trivial_expert_degree_passes(self):
        check(TRIVIAL_SPEC)
        check(ParallelismSpec(dp=2, ep=1))

    def test_any_expert_split_is_refused(self):
        for spec in (
            ParallelismSpec(dp=2, ep=2),
            ParallelismSpec(dp=4, ep=2),
            ParallelismSpec(dp=4, ep=4),
        ):
            with self.subTest(spec=spec):
                with self.assertRaisesRegex(ValueError, "not supported yet"):
                    check(spec)


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
