"""CPU-only tests for the --model-size axis.

Covers the shape arithmetic (pinned against the numbers a real run logs),
the closure of every scenario arm's config over every registered size, the
derivation of the override counts from the shape, and the
manifest/resume plumbing.
"""

import gzip
import inspect
import json
import os
import sys
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.manifests import load_manifest, resume_mismatches
from benchmarks.e2e.engines.registry import engine_for
from benchmarks.e2e.parallelism import TRIVIAL_SPEC
from benchmarks.e2e.registry import (
    SCENARIOS,
    scenario_by_name,
)
from benchmarks.e2e.axes import RequestedAxes, RunRequest
from benchmarks.e2e.runner import check_request, execute_run
from benchmarks.execution.affinity import CpuPinning
from tests.engine_helpers import command, run_spec, validate, write_run_manifest
from benchmarks.models.piper_qwen3.shape import (
    canonical_size_name,
    MODEL_SIZE_ALIASES,
    MODEL_SIZE_CHOICES,
    PIPER_1B,
    PIPER_30B_A3B,
    PIPER_30B_A3B_CUT,
    PIPER_SHAPES,
    PiperShape,
    shape_by_name,
)
from tests.test_runner import _SAC_LINE, _compiled_line


def _counts_from_the_tensor_list(shape) -> tuple[int, int, int]:
    """Count the parameters tensor by tensor, as a state dict enumerates them.

    A second route to the numbers ``PiperShape``'s closed form returns, and
    the route that lets a new shape be derived rather than transcribed. Every
    width here is written from the tensor it belongs to, so a head-count or an
    expert-count error reaches the total instead of cancelling. The test below
    checks this helper against the four numbers a real ``normal`` run logs,
    which is what gives it the authority to derive the counts of the shapes no
    run has produced.
    """
    tables = 2 * shape.vocab_size * shape.dim  # tok_embeddings + lm_head
    per_layer_dense = (
        shape.dim  # attention_norm
        + shape.qkv_out_features * shape.dim  # fused qkv
        + 2 * shape.head_dim  # q_norm + k_norm
        + shape.n_heads * shape.head_dim * shape.dim  # wo
        + shape.dim  # ffn_norm
    )
    router = shape.num_experts * shape.dim
    one_expert = 3 * shape.moe_hidden_dim * shape.dim  # w1 + w2 + w3
    dense = tables + shape.dim + shape.n_layers * per_layer_dense
    sparse = shape.n_layers * (router + shape.num_experts * one_expert)
    active = dense + shape.n_layers * (router + shape.top_k * one_expert)
    return dense, sparse, active


def _flops_from_the_tensor_list(shape, seq_len: int) -> int:
    """The tflops/MFU denominator, off the same tensor list."""
    matmul = _counts_from_the_tensor_list(shape)[2] - shape.vocab_size * shape.dim
    attention = 6 * shape.n_layers * shape.n_heads * 2 * shape.head_dim * seq_len
    return 6 * matmul + attention


class ShapeArithmeticTests(unittest.TestCase):
    def test_normal_matches_the_numbers_a_real_run_logs(self) -> None:
        # These four are the exact values torchtitan prints
        # ("Total parameter count: dense D, sparse S, vision 0, active A")
        # at the 1b shape. They were duplicated by hand in the megatron
        # baseline before schema 9.
        self.assertEqual(PIPER_1B.param_count, 1_066_241_024)
        self.assertEqual(PIPER_1B.nparams_dense, 361_532_416)
        self.assertEqual(PIPER_1B.nparams_sparse, 704_708_608)
        self.assertEqual(PIPER_1B.nparams_active, 713_919_488)
        self.assertEqual(PIPER_1B.num_flops_per_token(1024), 3_551_348_736)

    def test_normal_geometry(self) -> None:
        self.assertEqual(
            (PIPER_1B.dim, PIPER_1B.n_layers, PIPER_1B.n_heads, PIPER_1B.n_kv_heads),
            (1024, 16, 16, 8),
        )
        self.assertEqual(PIPER_1B.moe_hidden_dim, 3584)
        self.assertEqual(PIPER_1B.qkv_out_features, 2048)
        self.assertEqual(PIPER_1B.heads_per_group, 2)

    def test_the_tensor_by_tensor_count_reproduces_a_real_run(self) -> None:
        # The helper earns its authority here, against the same logged
        # numbers the test above pins, and then derives the shapes no run has
        # produced.
        self.assertEqual(
            _counts_from_the_tensor_list(PIPER_1B),
            (361_532_416, 704_708_608, 713_919_488),
        )
        self.assertEqual(
            _flops_from_the_tensor_list(PIPER_1B, 1024), 3_551_348_736
        )

    def test_every_registered_shape_agrees_with_the_tensor_list(self) -> None:
        """Every shape's counts, derived a second way rather than pasted.

        A shape with no logged run has nothing to pin it against, so this
        is what stands in for one: a count that walks the parameter tensors must
        agree with the closed form at every registered size.
        """
        for name, shape in PIPER_SHAPES.items():
            with self.subTest(size=name):
                dense, sparse, active = _counts_from_the_tensor_list(shape)
                self.assertEqual(shape.nparams_dense, dense)
                self.assertEqual(shape.nparams_sparse, sparse)
                self.assertEqual(shape.nparams_active, active)
                self.assertEqual(shape.param_count, dense + sparse)
                self.assertEqual(
                    shape.num_flops_per_token(1024),
                    _flops_from_the_tensor_list(shape, 1024),
                )

    def test_the_registry_is_declared_smallest_to_largest(self) -> None:
        """The order the module docstring states, pinned.

        The names do not carry it, so the declaration order is the
        statement, and this is what keeps a later insertion from breaking it.
        """
        self.assertEqual(tuple(PIPER_SHAPES), ("1b", "30b-a3b-20l", "30b-a3b"))
        counts = [shape.param_count for shape in PIPER_SHAPES.values()]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(len(set(counts)), len(counts))

    def test_the_geometry_guards_refuse_an_impossible_shape(self) -> None:
        with self.assertRaisesRegex(ValueError, "multiple of head_dim"):
            PiperShape.derived(name="bad", dim=1000, n_layers=1)
        with self.assertRaisesRegex(ValueError, "n_layers"):
            PiperShape.derived(name="bad", dim=1024, n_layers=0)
        # n_kv_heads is recorded now, so a ratio that leaves a partial query
        # group is a shape error rather than an unreachable one.
        with self.assertRaisesRegex(ValueError, "multiple of n_kv_heads"):
            PiperShape(
                name="bad",
                dim=1024,
                n_layers=1,
                head_dim=64,
                n_kv_heads=5,
                num_experts=4,
            )
        with self.assertRaisesRegex(ValueError, "num_experts"):
            PiperShape.derived(name="bad", dim=1024, n_layers=1, num_experts=0)

    def test_the_family_constructor_is_not_how_a_shape_is_registered(self) -> None:
        """``derived`` carries the piper-1B rules, which 30B-A3B breaks.

        The rules give ``head_dim`` 64, half as many kv heads as query heads,
        and 4 experts. Qwen3-30B-A3B has ``head_dim`` 128, 8:1 grouped-query
        attention and 128 experts, so a registered shape that took these
        would build the wrong attention and the wrong expert count under the
        right name.
        """
        probe = PiperShape.derived(name="probe", dim=2048, n_layers=1)
        self.assertEqual(
            (probe.head_dim, probe.n_heads, probe.n_kv_heads, probe.num_experts),
            (64, 32, 16, 4),
        )
        # 30B-A3B is dim 2048 too, and agrees with none of those but n_heads.
        self.assertEqual(PIPER_30B_A3B.dim, probe.dim)
        self.assertEqual(PIPER_30B_A3B.head_dim, 128)
        self.assertEqual(PIPER_30B_A3B.n_kv_heads, 4)
        self.assertEqual(PIPER_30B_A3B.num_experts, 128)

        # A structural guard, not a value comparison. Comparing a registered
        # shape to a PiperShape rebuilt from its own fields is a tautology for
        # a frozen dataclass, so this reads the source instead: the registry
        # must construct every entry through the plain constructor.
        source = Path(
            inspect.getsourcefile(PiperShape) or ""
        ).read_text(encoding="utf-8")
        self.assertNotIn("PiperShape.derived(", source)
        self.assertIn("PIPER_30B_A3B = PiperShape(", source)

    def test_the_two_promoted_fields_default_to_their_derivations(self) -> None:
        """``n_heads`` and ``moe_hidden_dim`` are fields since 2026-09-05.

        Left at ``None`` they take ``dim // head_dim`` and ``dim * 7 // 2``,
        which is what every shape registered before then satisfied, so no
        such shape moved a number. ``PINNED_SHAPES`` is what proves that;
        this pins the default itself, on a probe and on the registry.
        """
        probe = PiperShape(
            name="probe", dim=1024, n_layers=1, head_dim=64, n_kv_heads=8,
            num_experts=4,
        )
        self.assertEqual((probe.n_heads, probe.moe_hidden_dim), (16, 3584))
        self.assertIsInstance(probe.n_heads, int)
        self.assertIsInstance(probe.moe_hidden_dim, int)
        self.assertEqual(PIPER_1B.n_heads, PIPER_1B.dim // PIPER_1B.head_dim)
        self.assertEqual(PIPER_1B.moe_hidden_dim, PIPER_1B.dim * 7 // 2)

    def test_explicit_values_are_kept_and_counted(self) -> None:
        """A shape that writes its own values is counted from them.

        The Qwen3 30B-A3B geometry: 32 heads of 128 at dim 2048, so ``wo``
        is ``[2048, 4096]`` and the fused qkv is ``(32 + 8) * 128`` wide; and
        an expert width of 768 where the derivation says 7168. Both the
        closed form and the tensor-by-tensor helper must read the fields.
        """
        shape = PiperShape(
            name="probe", dim=2048, n_layers=1, head_dim=128, n_kv_heads=4,
            num_experts=128, n_heads=32, moe_hidden_dim=768, top_k=8,
        )
        self.assertEqual((shape.n_heads, shape.moe_hidden_dim), (32, 768))
        self.assertEqual(shape.n_heads * shape.head_dim, 2 * shape.dim)
        self.assertEqual(shape.qkv_out_features, 40 * 128)
        self.assertEqual(shape.heads_per_group, 8)
        dense, sparse, active = _counts_from_the_tensor_list(shape)
        self.assertEqual(
            (shape.nparams_dense, shape.nparams_sparse, shape.nparams_active),
            (dense, sparse, active),
        )
        # One expert is 3 * 768 * 2048, not 3 * 7168 * 2048.
        self.assertEqual(shape._experts, 128 * 3 * 768 * 2048)
        described = shape.describe(seq_len=1024)
        self.assertEqual(
            (described["n_heads"], described["moe_hidden_dim"]), (32, 768)
        )

    def test_the_divisibility_guards_apply_to_the_derivation_alone(self) -> None:
        """A written value has no quotient to keep exact.

        The old guards refused any dim that ``head_dim`` does not divide and
        any odd dim, because the derivations needed both. They still refuse
        those when the value is derived, and they let a written value
        through; a written zero is refused on its own.
        """
        with self.assertRaisesRegex(ValueError, "multiple of head_dim"):
            PiperShape(
                name="bad", dim=1000, n_layers=1, head_dim=64, n_kv_heads=1,
                num_experts=4,
            )
        odd = PiperShape(
            name="odd", dim=1000, n_layers=1, head_dim=64, n_kv_heads=2,
            num_experts=4, n_heads=16,
        )
        self.assertEqual((odd.n_heads, odd.moe_hidden_dim), (16, 3500))
        with self.assertRaisesRegex(ValueError, "must be even"):
            PiperShape(
                name="bad", dim=1001, n_layers=1, head_dim=7, n_kv_heads=1,
                num_experts=1,
            )
        written = PiperShape(
            name="odd", dim=1001, n_layers=1, head_dim=7, n_kv_heads=1,
            num_experts=1, moe_hidden_dim=10,
        )
        self.assertEqual((written.n_heads, written.moe_hidden_dim), (143, 10))
        for field in ("n_heads", "moe_hidden_dim"):
            with self.subTest(field=field):
                with self.assertRaisesRegex(ValueError, f"{field} must be >= 1"):
                    PiperShape(
                        name="bad", dim=1024, n_layers=1, head_dim=64,
                        n_kv_heads=1, num_experts=4, **{field: 0},
                    )

    def test_30b_a3b_is_transcribed_and_breaks_both_derivations(self) -> None:
        """Every geometry field against piper's case '30B-A3B'.

        The two written fields disagree with the derived defaults, which is
        why they are fields; and the counts are the helper's, which the
        pinned table records and the selection report agrees with.
        """
        shape = PIPER_SHAPES["30b-a3b"]
        self.assertIs(shape, PIPER_30B_A3B)
        self.assertEqual(
            (shape.dim, shape.n_layers, shape.n_heads, shape.n_kv_heads,
             shape.head_dim, shape.moe_hidden_dim, shape.num_experts,
             shape.top_k, shape.vocab_size, shape.rope_theta),
            (2048, 48, 32, 4, 128, 768, 128, 8, 151_936, 1_000_000.0),
        )
        self.assertNotEqual(shape.n_heads, shape.dim // shape.head_dim)
        self.assertNotEqual(shape.moe_hidden_dim, shape.dim * 7 // 2)
        self.assertEqual(shape.heads_per_group, 8)
        self.assertEqual(shape.qkv_out_features, 5120)
        # Not piper's 262144: the harness ceiling and the RoPE cache size.
        self.assertEqual(shape.max_seq_len, 4096)
        self.assertEqual(
            _counts_from_the_tensor_list(shape),
            (1_528_510_464, 29_003_612_160, 3_353_032_704),
        )
        self.assertEqual(shape.param_count, 30_532_122_624)

    def test_describe_is_json_safe(self) -> None:
        for shape in PIPER_SHAPES.values():
            described = shape.describe(seq_len=1024)
            self.assertEqual(json.loads(json.dumps(described)), described)
            self.assertEqual(described["name"], shape.name)

    def test_parity_gate_is_shape_data(self) -> None:
        # These are the per-shape logit-parity tolerances, recorded data with
        # no current consumer.
        self.assertEqual(PIPER_1B.parity_gate, 2e-2)
        self.assertEqual(
            PiperShape.derived(name="probe", dim=1024, n_layers=2).parity_gate, 2e-2
        )
        for shape in PIPER_SHAPES.values():
            self.assertEqual(
                shape.describe(seq_len=1024)["parity_gate"], shape.parity_gate
            )



# Every registered shape, transcribed. ``describe`` is what the manifest
# records and what a reader compares two runs by, so a change to any number
# here changes the meaning of every run already on disk under that name. The
# table is deliberately a literal: it must be read against the model config
# the shape claims to be, never regenerated from the code it guards.
PINNED_SHAPES: dict[str, dict[str, object]] = {
    "1b": {
        "dim": 1024,
        "n_layers": 16,
        "n_heads": 16,
        "n_kv_heads": 8,
        "head_dim": 64,
        "moe_hidden_dim": 3584,
        "num_experts": 4,
        "top_k": 2,
        "vocab_size": 151_936,
        "rope_theta": 1_000_000.0,
        "max_seq_len": 4096,
        "parity_gate": 2e-2,
        "param_count": 1_066_241_024,
        "nparams_dense": 361_532_416,
        "nparams_sparse": 704_708_608,
        "nparams_active": 713_919_488,
        "num_flops_per_token": 3_551_348_736,
    },
    # Qwen3-30B-A3B, geometry transcribed from examples/models/qwen3.py case
    # '30B-A3B'. The five computed values are the tensor-by-tensor helper's,
    # not the selection report's; the report's total and active agree.
    "30b-a3b": {
        "dim": 2048,
        "n_layers": 48,
        "n_heads": 32,
        "n_kv_heads": 4,
        "head_dim": 128,
        "moe_hidden_dim": 768,
        "num_experts": 128,
        "top_k": 8,
        "vocab_size": 151_936,
        "rope_theta": 1_000_000.0,
        "max_seq_len": 4096,
        "parity_gate": 2e-2,
        "param_count": 30_532_122_624,
        "nparams_dense": 1_528_510_464,
        "nparams_sparse": 29_003_612_160,
        "nparams_active": 3_353_032_704,
        "num_flops_per_token": 20_667_125_760,
    },
    # Qwen3-30B-A3B cut to 20 layers; every other field is 30b-a3b's.
    "30b-a3b-20l": {
        "dim": 2048,
        "n_layers": 20,
        "n_heads": 32,
        "n_kv_heads": 4,
        "head_dim": 128,
        "moe_hidden_dim": 768,
        "num_experts": 128,
        "top_k": 8,
        "vocab_size": 151_936,
        "rope_theta": 1_000_000.0,
        "max_seq_len": 4096,
        "parity_gate": 2e-2,
        "param_count": 13_084_744_704,
        "nparams_dense": 999_906_304,
        "nparams_sparse": 12_084_838_400,
        "nparams_active": 1_760_123_904,
        "num_flops_per_token": 9_700_386_816,
    },
}


class PinnedShapeTests(unittest.TestCase):
    """The whole registry, number by number.

    Runs exist on disk against these shapes, and a run is comparable to
    another only within one ``model_size``. A silent change to any value here
    would therefore publish two different models under one name, which no
    other test in this file can see: the arithmetic tests check that the
    closed form agrees with the tensor list, and both would move together.
    """

    def test_every_registered_shape_matches_its_pinned_geometry(self) -> None:
        for name, expected in PINNED_SHAPES.items():
            with self.subTest(size=name):
                described = PIPER_SHAPES[name].describe(seq_len=1024)
                for key, value in expected.items():
                    self.assertEqual(described[key], value, f"{name}.{key}")

    def test_every_registered_shape_is_pinned(self) -> None:
        """A new shape must write its numbers down before it can ship.

        The table above is the only place a shape's geometry is stated
        twice, and stating it twice is the point: the second statement is
        transcribed from the model config the shape claims to be, so a
        derivation that is wrong for that model cannot pass both.

        That holds for the geometry half only. ``param_count``, the three
        ``nparams_*`` values and ``num_flops_per_token`` appear in no model
        config and had to be computed, so for those five the table is a
        regression pin rather than an independent statement. The independent
        route for them is ``_counts_from_the_tensor_list`` above, which walks
        the parameter tensors and which every registered shape must match.
        """
        self.assertEqual(set(PINNED_SHAPES), set(PIPER_SHAPES))


class StageParamCountTests(unittest.TestCase):
    """The per-stage split, which is what makes arm rule 11 work under PP.

    A pipelined rank builds a slice of the model, so the whole model's count
    is no longer what it can assert. These pin the split both engines use:
    the embedding and the output head are not layers, so every stage holds
    ``n_layers // pipeline_degree`` of them and the two end stages carry a
    table each.
    """

    def test_one_stage_is_the_whole_model(self) -> None:
        for name, shape in PIPER_SHAPES.items():
            with self.subTest(size=name):
                self.assertEqual(
                    shape.stage_param_count(pipeline_degree=1, stage_index=0),
                    shape.param_count,
                )

    def test_the_stages_sum_to_the_whole_model(self) -> None:
        """The property a driver asserts across the world, at every degree.

        A split that lost or double-counted a tensor would show here and
        nowhere else: each stage's own count would still look plausible.
        """
        for name, shape in PIPER_SHAPES.items():
            for degree in (1, 2, 4):
                if shape.n_layers % degree:
                    continue
                with self.subTest(size=name, degree=degree):
                    self.assertEqual(
                        sum(
                            shape.stage_param_count(
                                pipeline_degree=degree, stage_index=stage
                            )
                            for stage in range(degree)
                        ),
                        shape.param_count,
                    )

    def test_the_two_end_stages_carry_the_tables(self) -> None:
        """Written from the tensor list, not from the closed form.

        At pp 2 the difference between the two stages is exactly the final
        norm: the first holds the embedding table and the last holds the
        output head plus that norm, and both tables are ``vocab_size x dim``.
        """
        shape = PIPER_1B
        layers_each = shape.n_layers // 2
        per_layer = (
            shape.dim
            + shape.qkv_out_features * shape.dim
            + 2 * shape.head_dim
            + shape.n_heads * shape.head_dim * shape.dim
            + shape.dim
            + shape.num_experts * shape.dim
            + shape.num_experts * 3 * shape.moe_hidden_dim * shape.dim
        )
        table = shape.vocab_size * shape.dim
        self.assertEqual(
            shape.stage_param_count(pipeline_degree=2, stage_index=0),
            layers_each * per_layer + table,
        )
        self.assertEqual(
            shape.stage_param_count(pipeline_degree=2, stage_index=1),
            layers_each * per_layer + table + shape.dim,
        )

    def test_30b_a3b_splits_at_pp_four_and_pp_eight(self) -> None:
        """48 layers: 12 a stage at pp 4, 6 at pp 8; the tables at the ends.

        The depth-8 pipeline is the deepest eight GPUs hold, and this is
        the first real shape above 1b that reaches it with more than four
        layers a stage. Written from the tensor list, as the 1b test is.
        """
        shape = PIPER_30B_A3B
        per_layer = (
            shape.dim
            + shape.qkv_out_features * shape.dim
            + 2 * shape.head_dim
            + shape.n_heads * shape.head_dim * shape.dim
            + shape.dim
            + shape.num_experts * shape.dim
            + shape.num_experts * 3 * shape.moe_hidden_dim * shape.dim
        )
        table = shape.vocab_size * shape.dim
        for degree, layers_each in ((4, 12), (8, 6)):
            with self.subTest(pp=degree):
                self.assertEqual(shape.n_layers // degree, layers_each)
                stages = [
                    shape.stage_param_count(
                        pipeline_degree=degree, stage_index=stage
                    )
                    for stage in range(degree)
                ]
                self.assertEqual(sum(stages), shape.param_count)
                self.assertEqual(stages[0], layers_each * per_layer + table)
                self.assertEqual(
                    stages[-1], layers_each * per_layer + table + shape.dim
                )
                for middle in stages[1:-1]:
                    self.assertEqual(middle, layers_each * per_layer)
        # 128 experts divide every expert degree eight GPUs can hold.
        for expert_degree in (2, 4, 8):
            with self.subTest(ep=expert_degree):
                self.assertEqual(
                    sum(
                        shape.stage_param_count(
                            pipeline_degree=8,
                            stage_index=stage,
                            expert_degree=expert_degree,
                        )
                        for stage in range(8)
                    ),
                    shape.nparams_dense
                    + shape.n_layers
                    * (shape._router + shape._experts // expert_degree),
                )

    def test_an_uneven_split_raises_rather_than_rounding(self) -> None:
        # A one-layer shape is what parallelism rule 7 refuses at pp 2. This
        # is the same refusal, one level down.
        one_layer = PiperShape.derived(name="probe", dim=1024, n_layers=1)
        with self.assertRaisesRegex(ValueError, "do not divide evenly"):
            one_layer.stage_param_count(pipeline_degree=2, stage_index=0)

    def test_a_stage_outside_the_pipeline_raises(self) -> None:
        for degree, stage in ((2, 2), (2, -1), (0, 0)):
            with self.subTest(degree=degree, stage=stage):
                with self.assertRaises(ValueError):
                    PIPER_1B.stage_param_count(
                        pipeline_degree=degree, stage_index=stage
                    )



class StageParamCountUnderAnExpertDegreeTest(unittest.TestCase):
    """A rank under an expert degree holds ``num_experts // ep`` of the
    routed experts, so the guard has to divide that term or it refuses an
    honest run. It did refuse one, on 2026-08-28.
    """

    def test_the_default_is_one_and_reproduces_the_whole_model(self):
        """Every recorded run so far passed no expert degree, so the default
        must reproduce what those runs asserted, exactly."""
        for shape in PIPER_SHAPES.values():
            with self.subTest(shape=shape.name):
                self.assertEqual(
                    shape.stage_param_count(pipeline_degree=1, stage_index=0),
                    shape.stage_param_count(
                        pipeline_degree=1, stage_index=0, expert_degree=1
                    ),
                )

    def test_only_the_expert_term_divides(self):
        """The router is the gate that chooses an expert, so every rank
        needs the whole of it. No dense parameter divides either."""
        for shape in PIPER_SHAPES.values():
            if shape.num_experts % 2:
                continue
            with self.subTest(shape=shape.name):
                whole = shape.stage_param_count(
                    pipeline_degree=1, stage_index=0
                )
                split = shape.stage_param_count(
                    pipeline_degree=1, stage_index=0, expert_degree=2
                )
                self.assertEqual(
                    whole - split, shape.n_layers * shape._experts // 2
                )

    def test_the_1b_run_that_was_refused_now_agrees(self):
        """The measured count from the refused run, transcribed. ``1b`` at
        ``ep 2``, one stage."""
        self.assertEqual(
            PIPER_1B.stage_param_count(
                pipeline_degree=1, stage_index=0, expert_degree=2
            ),
            713_919_488,
        )

    def test_the_1b_match_with_active_params_is_a_coincidence(self):
        """It holds at ``1b`` because ``num_experts // ep`` is ``top_k``
        there. It does not hold at ``30b-a3b``, and nobody may substitute one
        for the other."""
        self.assertEqual(
            PIPER_1B.stage_param_count(
                pipeline_degree=1, stage_index=0, expert_degree=2
            ),
            PIPER_1B.nparams_active,
        )
        self.assertNotEqual(
            PIPER_30B_A3B.stage_param_count(
                pipeline_degree=1, stage_index=0, expert_degree=2
            ),
            PIPER_30B_A3B.nparams_active,
        )

    def test_an_expert_count_that_does_not_divide_is_refused(self):
        with self.assertRaisesRegex(
            ValueError, r"experts do not divide evenly into 3"
        ):
            PIPER_1B.stage_param_count(
                pipeline_degree=1, stage_index=0, expert_degree=3
            )

    def test_a_degree_below_one_is_refused(self):
        with self.assertRaisesRegex(ValueError, r"expert_degree 0 must be"):
            PIPER_1B.stage_param_count(
                pipeline_degree=1, stage_index=0, expert_degree=0
            )

    def test_the_split_still_sums_to_the_whole_model_at_ep_one(self):
        """Rule 11's other half: the sum over the stages is ``param_count``.
        The expert term must not disturb it at the default."""
        for shape in PIPER_SHAPES.values():
            for degree in (1, 2, 4):
                if shape.n_layers % degree:
                    continue
                with self.subTest(shape=shape.name, pp=degree):
                    self.assertEqual(
                        sum(
                            shape.stage_param_count(
                                pipeline_degree=degree, stage_index=i
                            )
                            for i in range(degree)
                        ),
                        shape.param_count,
                    )


class ModelSizeAliasTests(unittest.TestCase):
    """``normal`` is a second name of ``1b``, and a manifest records the canonical name."""

    def test_the_retired_name_resolves_to_the_same_shape(self) -> None:
        self.assertIs(shape_by_name("normal"), PIPER_1B)
        self.assertIs(shape_by_name("1b"), PIPER_1B)
        self.assertEqual(canonical_size_name("normal"), "1b")
        self.assertEqual(canonical_size_name("1b"), "1b")

    def test_the_alias_is_not_a_registry_entry(self) -> None:
        """``PIPER_SHAPES`` enumerates shapes; the CLI and the tests count it.

        An alias held there would make one shape appear twice -- in
        ``click.Choice``, in the declaration-order test, and in every sweep
        that iterates the registry.
        """
        self.assertNotIn("normal", PIPER_SHAPES)
        self.assertEqual(set(MODEL_SIZE_ALIASES), {"normal"})
        self.assertIn("normal", MODEL_SIZE_CHOICES)
        self.assertEqual(
            len(MODEL_SIZE_CHOICES), len(PIPER_SHAPES) + len(MODEL_SIZE_ALIASES)
        )

    def test_an_unknown_size_is_still_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown model size"):
            shape_by_name("enormous")
        # A total function: an unknown name passes through rather than
        # raising, so the resume comparison can normalise any recorded string.
        self.assertEqual(canonical_size_name("enormous"), "enormous")

    def _manifest(self, model_size: str) -> dict:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            write_run_manifest(
                out_dir,
                run_spec(model_size, profile=False),
                (scenario_by_name("engines").arm("titan_compiled"),),
            )
            return load_manifest(out_dir)

    def test_a_fresh_manifest_records_the_canonical_name(self) -> None:
        self.assertEqual(self._manifest("normal")["run"]["shape"]["name"], "1b")

    def test_a_manifest_recording_either_name_resumes_against_the_other(
        self,
    ) -> None:
        selected = (scenario_by_name("engines").arm("titan_compiled"),)
        for recorded_name, requested in (
            ("normal", "1b"),
            ("1b", "normal"),
            ("normal", "normal"),
            ("1b", "1b"),
        ):
            with self.subTest(recorded=recorded_name, requested=requested):
                self.assertEqual(
                    resume_mismatches(
                        self._manifest(recorded_name),
                        run=run_spec(requested, profile=False),
                        arms=selected,
                    ),
                    [],
                )
        self.assertEqual(
            resume_mismatches(
                self._manifest("normal"),
                run=run_spec("30b-a3b", profile=False),
                arms=selected,
            ),
            ["run.shape"],
        )


class ConfigSizeClosureTests(unittest.TestCase):
    def test_the_fork_parses_every_titan_argv_through_its_module_flag(self) -> None:
        """The fork's own ``ConfigManager`` resolves ``--module`` and parses the whole argv.

        ``--module`` names the plug-in package, and the fork imports its
        ``config_registry``. A parse of the argv the engine builds proves the
        trainer finds the config and accepts every other argument.
        """
        from torchtitan.config.manager import ConfigManager

        from benchmarks.e2e.engines.torchtitan.flags import trainer_args
        from benchmarks.e2e.engines.torchtitan.plugins.replay import (
            PretokenizedReplayDataLoader,
        )
        from benchmarks.e2e.parallelism import ParallelismSpec

        mesh = ParallelismSpec(dp=2, pp=2, ep=2, zero=1, pp_schedule="1F1B")
        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                if engine_for(arm).name != "torchtitan":
                    continue
                for spec, profile in ((TRIVIAL_SPEC, False), (mesh, True)):
                    with self.subTest(arm=arm.name, spec=spec):
                        run = run_spec(
                            "30b-a3b-20l",
                            parallelism=spec,
                            profile=profile,
                            ac_mode="none",
                            local_batch_size=8,
                        )
                        argv = trainer_args(run, arm.config, Path("/tmp/arm"))
                        config = ConfigManager().parse_args(list(argv))
                        self.assertEqual(
                            config.model_spec.model.dim, PIPER_30B_A3B_CUT.dim
                        )
                        self.assertIsInstance(
                            config.dataloader, PretokenizedReplayDataLoader.Config
                        )
                        self.assertEqual(
                            config.dataloader.replay_steps, run.data.steps
                        )
                        self.assertEqual(
                            config.compile.enable,
                            arm.config.compile.value == "torch",
                        )

    def test_every_scenario_arm_builds_at_every_size(self) -> None:
        """Every arm's config resolves and accepts every registered size.

        This is the closure the runner depends on: it emits ``--config <name>
        --config-arg size=<size>`` for whatever the arm names, so a config
        that did not accept the keyword -- or accepted it and ignored it --
        would publish a run under the wrong shape.
        """
        import inspect

        import benchmarks.e2e.engines.torchtitan.plugins.config_registry as registry

        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                if engine_for(arm).name != "torchtitan":
                    continue
                name = arm.config.config
                with self.subTest(scenario=scenario.name, arm=arm.name):
                    factory = getattr(registry, name, None)
                    self.assertTrue(
                        callable(factory), f"{name} is not a config factory"
                    )
                    parameter = inspect.signature(factory).parameters.get("size")
                    self.assertIsNotNone(parameter, f"{name} takes no size")
                    self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                    self.assertEqual(parameter.default, "1b")
                    self._assert_every_size_lands(factory, name)

    def _assert_every_size_lands(self, factory, name: str) -> None:
        for size in tuple(PIPER_SHAPES):
            shape = PIPER_SHAPES[size]
            # One config gates its attention backend on the host GPU
            # (qwen3_piper_1b_flex_flash wants sm90+), which this CPU-only
            # suite does not have. Skipping it would retire the only check on
            # a config that accepts ``size`` and ignores it, so pretend the
            # capability instead: the fork imports has_cuda_capability inside
            # get_attention_config, so patching its source module reaches it,
            # and nothing here runs a kernel.
            with mock.patch(
                "torchtitan.tools.utils.has_cuda_capability", return_value=True
            ):
                model = factory(size=size).model_spec.model
            self.assertEqual(model.dim, shape.dim, f"{name} at {size}")
            self.assertEqual(len(model.layers), shape.n_layers, f"{name} at {size}")
            # Every field the shape records, checked where it lands. dim and
            # the layer count alone would pass a config that ignored the head
            # geometry or the expert roster -- which is the whole class of
            # error that made head_dim, n_kv_heads and num_experts recorded
            # fields.
            attention = model.layers[0].attention
            moe = model.layers[0].moe
            self.assertEqual(
                (attention.n_heads, attention.n_kv_heads, attention.head_dim),
                (shape.n_heads, shape.n_kv_heads, shape.head_dim),
                f"{name} at {size}",
            )
            experts = moe.routed_experts.inner_experts
            self.assertEqual(
                (moe.num_experts, moe.router.top_k, experts.hidden_dim),
                (shape.num_experts, shape.top_k, shape.moe_hidden_dim),
                f"{name} at {size}",
            )
            self.assertEqual(
                model.vocab_size, shape.vocab_size, f"{name} at {size}"
            )

    def test_the_private_builders_require_an_explicit_shape(self) -> None:
        """No default shape on the builders every public config calls.

        All eleven call sites pass ``shape=`` today, so a default could only
        ever be reached by a future omission -- and would then build the
        normal geometry silently, under whatever size the run asked for. That
        is the exact silent fallback the ``size`` keyword replaced.
        """
        import inspect

        from benchmarks.models.piper_qwen3.titan_model import _piper_1b_model

        parameter = inspect.signature(_piper_1b_model).parameters["shape"]
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameter.default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            _piper_1b_model(fuse_qkv=True)

    def test_the_megatron_builder_requires_an_explicit_shape(self) -> None:
        """The same rule on the other engine's builder, for the same reason.

        ``build_model`` is the megatron twin of ``_piper_1b_model``: both
        construct the same geometry from the same ``PiperShape``, and the
        whole point of that sharing is that a size cannot drift between the
        engines. A default here would reintroduce the drift on one side, and
        a caller that builds the model outside the harness never reaches
        validation rule 11's parameter-count check. Signature inspection
        only; this imports no megatron.
        """
        import inspect

        from benchmarks.models.piper_qwen3.megatron_model import build_model

        parameter = inspect.signature(build_model).parameters["shape"]
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_size_round_trips_through_the_config_argument(self) -> None:
        from benchmarks.e2e.engines.torchtitan.plugins.config_registry import (
            qwen3_piper_1b_pretokenized,
        )

        self.assertEqual(qwen3_piper_1b_pretokenized(size="30b-a3b-20l").model_spec.model.dim, 2048)
        self.assertEqual(qwen3_piper_1b_pretokenized(size="normal").model_spec.model.dim, 1024)
        # The default is the normal shape, so an unparameterized call is the
        # historical config.
        self.assertEqual(qwen3_piper_1b_pretokenized().model_spec.model.dim, PIPER_1B.dim)
        with self.assertRaisesRegex(ValueError, "Unknown model size"):
            qwen3_piper_1b_pretokenized(size="enormous")

    def test_built_models_carry_the_requested_shape(self) -> None:
        from benchmarks.e2e.engines.torchtitan.plugins.config_registry import (
            qwen3_piper_1b_pretokenized,
        )

        normal = qwen3_piper_1b_pretokenized().model_spec.model
        self.assertEqual(normal.dim, PIPER_1B.dim)
        self.assertEqual(len(normal.layers), PIPER_1B.n_layers)

        cut = qwen3_piper_1b_pretokenized(size="30b-a3b-20l").model_spec.model
        self.assertEqual(cut.dim, PIPER_30B_A3B_CUT.dim)
        self.assertEqual(len(cut.layers), PIPER_30B_A3B_CUT.n_layers)
        self.assertEqual(cut.vocab_size, PIPER_30B_A3B_CUT.vocab_size)
        # The load_balance_coeff=None fixup must survive the parameterization.
        self.assertIsNone(cut.layers[0].moe.load_balance_coeff)

    def test_pretokenized_configs_pass_the_size_down_to_their_delegate(self) -> None:
        from benchmarks.e2e.engines.torchtitan.plugins.config_registry import (
            qwen3_piper_1b_pretokenized,
        )

        for factory in (qwen3_piper_1b_pretokenized,):
            for size, shape in PIPER_SHAPES.items():
                with self.subTest(config=factory.__name__, size=size):
                    config = factory(size=size)
                    self.assertEqual(config.model_spec.model.dim, shape.dim)
                    # replay_steps tracks the config's own step count.
                    self.assertEqual(
                        config.dataloader.replay_steps, config.training.steps
                    )


class CommandTests(unittest.TestCase):
    def test_titan_command_delivers_the_size_as_a_config_argument(self) -> None:
        scenario = scenario_by_name("engines")
        arm = scenario.arm("titan_compiled")
        argv = command(run_spec("30b-a3b-20l"), arm, "/tmp/arm")
        # The config name is the arm's, unmangled: the shape rides alongside.
        self.assertEqual(
            argv[argv.index("--config") + 1],
            "qwen3_piper_1b_pretokenized",
        )
        self.assertEqual(argv[argv.index("--config-arg") + 1], "size=30b-a3b-20l")
        self.assertFalse(
            [token for token in argv if token.endswith("_30b-a3b-20l")]
        )
        # replay_steps must track --training.steps or the loader hard-fails.
        self.assertEqual(
            argv[argv.index("--dataloader.replay-steps") + 1],
            argv[argv.index("--training.steps") + 1],
        )

    def test_the_default_size_is_delivered_explicitly_too(self) -> None:
        scenario = scenario_by_name("engines")
        argv = command(run_spec(), scenario.arm("titan_compiled"), "/tmp/arm")
        self.assertEqual(
            argv[argv.index("--config") + 1], "qwen3_piper_1b_pretokenized"
        )
        self.assertEqual(argv[argv.index("--config-arg") + 1], "size=1b")

    def test_every_titan_arm_gets_the_replay_flag(self) -> None:
        scenario = scenario_by_name("engines")
        for arm in scenario.arms:
            if engine_for(arm).name != "torchtitan":
                continue
            with self.subTest(arm=arm.name):
                argv = command(run_spec(steps=12, profile=False), arm, "/tmp/arm")
                self.assertEqual(
                    argv[argv.index("--dataloader.replay-steps") + 1], "12"
                )

    def test_megatron_command_carries_the_model_size(self) -> None:
        scenario = scenario_by_name("engines")
        argv = command(
            run_spec("30b-a3b-20l", ac_mode="none"),
            scenario.arm("megatron_stock"),
            "/tmp/arm",
        )
        self.assertEqual(
            argv[argv.index("--bench-model-size") + 1], "30b-a3b-20l"
        )


def _size_line(shape) -> str:
    return (
        "[titan] - root - INFO - Model qwen3 piper_1B "
        f"size: {shape.param_count:,} total parameters\n"
    )


class ValidationRuleElevenTests(unittest.TestCase):
    def _fixture(self, root: Path) -> Path:
        for iteration in ("iteration_20", "iteration_40"):
            trace = root / "profiling" / "traces" / iteration / "rank0_trace.json.gz"
            trace.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(trace, "wt") as trace_file:
                trace_file.write("cudaLaunchKernel\n")
        return root / "baseline.log"

    def test_a_log_without_the_size_marker_fails(self) -> None:
        scenario = scenario_by_name("engines")
        arm = scenario.arm("titan_compiled")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._fixture(root)
            head = _compiled_line("default") + _SAC_LINE

            log.write_text(head + "Training completed\n")
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate(run_spec(), arm, root, log)

            log.write_text(head + _size_line(PIPER_1B) + "Training completed\n")
            validate(run_spec(), arm, root, log)

            # The normal-size marker must not satisfy a 30b-a3b-20l run.
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate(run_spec("30b-a3b-20l"), arm, root, log)

            log.write_text(
                head + _size_line(PIPER_30B_A3B_CUT) + "Training completed\n"
            )
            validate(run_spec("30b-a3b-20l"), arm, root, log)

    def test_override_count_scales_with_the_layer_count(self) -> None:
        from tests.test_runner import OVERRIDE_ARM

        scenario = scenario_by_name("engines")
        arm = OVERRIDE_ARM
        applied = (
            f"[Override] {arm.config.override_imports[0]}: "
            "model_spec.model.layers.0.moe ...\n"
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for iteration in ("iteration_20", "iteration_40"):
                trace = (
                    root / "profiling" / "traces" / iteration / "rank0_trace.json.gz"
                )
                trace.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(trace, "wt") as trace_file:
                    trace_file.write("_combined_silu_and_mul_forward_kernel\n")
                    trace_file.write("_combined_silu_and_mul_backward_kernel\n")
            log = root / "arm.log"
            head = _compiled_line("default") + _SAC_LINE

            cut = _size_line(PIPER_30B_A3B_CUT)
            log.write_text(head + cut + "Training completed\n" + applied * 20)
            validate(run_spec("30b-a3b-20l"), arm, root, log)

            log.write_text(head + cut + "Training completed\n" + applied * 16)
            with self.assertRaisesRegex(RuntimeError, "expected 20 override"):
                validate(run_spec("30b-a3b-20l"), arm, root, log)

            log.write_text(
                head + _size_line(PIPER_1B) + "Training completed\n" + applied * 16
            )
            validate(run_spec(), arm, root, log)


_METADATA = {
    "requested_gpu": "0",
    "nvidia_smi": "0, Test GPU, GPU-uuid, driver",
    "torch_version": "test",
    "torchtitan_git_rev": "titan-rev",
    "benchmarks_git_rev": "bench-rev",
}


def _fake_process(size_line: str):
    def run(command, **kwargs):
        kwargs["stdout"].write(
            _compiled_line("default") + size_line + "Training completed\n"
        )
        # --dump-folder is not last under ac=none (the tyro subcommand
        # token trails it), so look the argument up by name.
        arm_dir = Path(command[command.index("--dump-folder") + 1])
        for iteration in (20, 40):
            trace = (
                arm_dir
                / f"profiling/traces/iteration_{iteration}/rank0_trace.json.gz"
            )
            trace.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(trace, "wt") as trace_file:
                json.dump({"traceEvents": []}, trace_file)
        return SimpleNamespace(returncode=0)

    return run


_AXIS_KEYWORDS = tuple(field.name for field in fields(RequestedAxes))


class ManifestAndResumeTests(unittest.TestCase):
    def _run(self, **request_kwargs):
        axes = RequestedAxes(
            **{
                name: request_kwargs.pop(name)
                for name in list(request_kwargs)
                if name in _AXIS_KEYWORDS
            }
        )
        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", _METADATA),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            return execute_run(
                check_request(
                    RunRequest(gpu="0", axes=axes, **request_kwargs),
                    environment={"PATH": os.environ["PATH"]},
                ),
                process_runner=_fake_process(_size_line(PIPER_30B_A3B_CUT)),
            )

    def test_a_30b_a3b_20l_run_records_the_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            self._run(
                scenario_name="engines",
                arm_names=("titan_compiled",),
                out_dir=out_dir,
                ac_mode="none",
                model_size="30b-a3b-20l",
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())

        self.assertEqual(manifest["schema_version"], 20)
        self.assertEqual(
            manifest["run"]["shape"], PIPER_30B_A3B_CUT.describe(seq_len=4096)
        )
        (arm,) = manifest["arms"]
        command = arm["command"]
        self.assertEqual(
            command[command.index("--config") + 1], "qwen3_piper_1b_pretokenized"
        )
        self.assertEqual(
            command[command.index("--config-arg") + 1], "size=30b-a3b-20l"
        )

    def test_resume_refuses_a_different_size_and_inherits_an_absent_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            self._run(
                scenario_name="engines",
                arm_names=("titan_compiled",),
                out_dir=out_dir,
                ac_mode="none",
                model_size="30b-a3b-20l",
            )

            with self.assertRaisesRegex(ValueError, "run.shape"):
                self._run(
                    scenario_name=None,
                    arm_names=("titan_compiled",),
                    resume_dir=out_dir,
                    ac_mode="none",
                    model_size="normal",
                )

            # Omitting --model-size on a resume inherits the recorded value.
            self._run(
                scenario_name=None,
                arm_names=("titan_compiled",),
                resume_dir=out_dir,
                ac_mode="none",
            )


if __name__ == "__main__":
    unittest.main()
