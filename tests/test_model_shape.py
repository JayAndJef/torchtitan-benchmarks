"""CPU-only tests for the --model-size axis.

Covers the shape arithmetic (pinned against the numbers a real run logs),
the closure of every scenario arm's config over every registered size, the
derivation of regions and override counts from the shape, and the
manifest/resume plumbing.
"""

import gzip
import inspect
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.artifacts.manifests import (
    _resume_mismatches,
    manifest_data,
    write_manifest,
)
from benchmarks.e2e.launch import command_for_arm
from benchmarks.e2e.parallelism import TRIVIAL_SPEC
from benchmarks.e2e.registry import (
    SCENARIOS,
    piper_block_regions,
    scenario_by_name,
)
from benchmarks.e2e.runner import RunRequest, execute_run
from benchmarks.e2e.validation import validate_arm
from benchmarks.execution.affinity import CpuPinning
from benchmarks.models.piper_qwen3.shape import (
    canonical_size_name,
    GIANT,
    HUGE,
    LARGE,
    MODEL_SIZE_ALIASES,
    MODEL_SIZE_CHOICES,
    PIPER_1B,
    PIPER_9B,
    PIPER_48B,
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
        # and the constant tools/megatron_parity_check.py asserts. They were
        # duplicated by hand in the megatron baseline before schema 9.
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
        self.assertTrue(PIPER_1B.supports_block_regions)

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

        The two new shapes have no run to pin them against, so this is what
        stands in for one: a count that walks the parameter tensors must
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

    def test_large_geometry_follows_the_piper_1b_family_rules(self) -> None:
        """Recorded values now, and this is what they must record.

        This shape is synthetic. It was built by applying the
        piper-1B rules at a larger width, so its head geometry is 2:1
        grouped-query attention at ``head_dim`` 64. Real piper runs
        4:1 from 9B up. Do not read this shape as piper at scale.
        """
        self.assertEqual((LARGE.dim, LARGE.n_layers), (4096, 4))
        self.assertEqual(LARGE.n_heads, LARGE.dim // 64)
        self.assertEqual(LARGE.n_kv_heads, LARGE.n_heads // 2)
        self.assertEqual(LARGE.moe_hidden_dim, LARGE.dim * 7 // 2)
        self.assertEqual(LARGE.qkv_out_features, 2 * LARGE.dim)
        self.assertEqual(LARGE.head_dim, PIPER_1B.head_dim)
        self.assertEqual(LARGE.vocab_size, PIPER_1B.vocab_size)
        # Four layers, so this is the only shape above normal whose block
        # graphs stay identifiable and whose runs validation rule 7 guards.
        self.assertTrue(LARGE.supports_block_regions)

    def test_giant_geometry_follows_the_piper_1b_family_rules(self) -> None:
        """Recorded values now, and this is what they must record.

        This shape is synthetic. It was built by applying the
        piper-1B rules at a larger width, so its head geometry is 2:1
        grouped-query attention at ``head_dim`` 64. Real piper runs
        4:1 from 9B up. Do not read this shape as piper at scale.
        """
        self.assertEqual((GIANT.dim, GIANT.n_layers), (16384, 1))
        self.assertEqual(GIANT.n_heads, GIANT.dim // 64)
        self.assertEqual(GIANT.n_kv_heads, GIANT.n_heads // 2)
        self.assertEqual(GIANT.moe_hidden_dim, GIANT.dim * 7 // 2)
        self.assertEqual(GIANT.qkv_out_features, 2 * GIANT.dim)
        self.assertEqual(GIANT.head_dim, PIPER_1B.head_dim)
        self.assertEqual(GIANT.vocab_size, PIPER_1B.vocab_size)
        # One layer, so the same region argument the huge shape makes.
        self.assertFalse(GIANT.supports_block_regions)
        # 6753/dim is 0.41 here, further below 1.0 than huge's 0.55.
        self.assertLess(
            2 * GIANT.vocab_size / (45 * GIANT.dim),
            2 * HUGE.vocab_size / (45 * HUGE.dim),
        )

    def test_the_layer_count_keeps_the_tables_out_of_the_way(self) -> None:
        """Why large is four layers and not one.

        At dim 4096 one layer is 45*D^2 against 2*V*D of tables, a ratio of
        1.65, so a 1-layer model there is mostly embedding table and the
        benchmark measures the lm_head and the cross entropy. Four layers
        move 71% of the parameters into the layer stack.
        """
        one_layer = PiperShape.derived(name="probe", dim=LARGE.dim, n_layers=1)
        self.assertGreater(self._table_fraction(one_layer), 0.6)
        self.assertLess(self._table_fraction(LARGE), 0.3)

    def test_three_shapes_share_the_1b_parameter_split(self) -> None:
        # 1b, large and giant all hold n_layers*dim at 16384, so all three
        # carry the same split between the two tables and the layer stack.
        # The other three do not hold that product: huge because the memory
        # ceiling chose its dim, 9b and 48b because piper chose theirs.
        for shape in (PIPER_1B, LARGE, GIANT):
            with self.subTest(size=shape.name):
                self.assertEqual(shape.n_layers * shape.dim, 16384)
                self.assertAlmostEqual(
                    self._table_fraction(shape), 0.292, places=3
                )
        self.assertNotEqual(HUGE.n_layers * HUGE.dim, 16384)

    def test_the_registry_is_declared_smallest_to_largest(self) -> None:
        """The order the module docstring states, pinned.

        The names do not carry it -- ``huge`` < ``giant`` is not self-evident
        -- so the declaration order is the statement, and this is what keeps
        a later insertion from breaking it.
        """
        self.assertEqual(
            tuple(PIPER_SHAPES),
            ("1b", "large", "9b", "huge", "giant", "48b"),
        )
        counts = [shape.param_count for shape in PIPER_SHAPES.values()]
        self.assertEqual(counts, sorted(counts))
        self.assertEqual(len(set(counts)), len(counts))

    @staticmethod
    def _table_fraction(shape) -> float:
        """Embedding plus lm_head, as a fraction of every parameter."""
        return 2 * shape.vocab_size * shape.dim / shape.param_count

    def test_huge_geometry_follows_the_piper_1b_family_rules(self) -> None:
        """Recorded values now, and this is what they must record.

        This shape is synthetic. It was built by applying the
        piper-1B rules at a larger width, so its head geometry is 2:1
        grouped-query attention at ``head_dim`` 64. Real piper runs
        4:1 from 9B up. Do not read this shape as piper at scale.
        """
        self.assertEqual(HUGE.n_layers, 1)
        self.assertEqual(HUGE.n_heads, HUGE.dim // 64)
        self.assertEqual(HUGE.n_kv_heads, HUGE.n_heads // 2)
        self.assertEqual(HUGE.moe_hidden_dim, HUGE.dim * 7 // 2)
        self.assertEqual(HUGE.qkv_out_features, 2 * HUGE.dim)
        self.assertEqual(HUGE.head_dim, PIPER_1B.head_dim)
        self.assertEqual(HUGE.vocab_size, PIPER_1B.vocab_size)
        # The reason the huge shape exists: at one layer the embedding tables
        # must not dominate. embedding+lm_head / one layer = 6753/dim.
        self.assertLess(2 * HUGE.vocab_size / (45 * HUGE.dim), 1.0)
        self.assertFalse(HUGE.supports_block_regions)

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
        """``derived`` carries the piper-1B rules, which 9B and 48B break.

        The rules give ``head_dim`` 64, half as many kv heads as query heads,
        and 4 experts. Real piper runs 4:1 grouped-query attention and 8
        experts above 1B, so a registered shape that took these would build
        the wrong attention and the wrong expert count under the right name.
        """
        probe = PiperShape.derived(name="probe", dim=2048, n_layers=1)
        self.assertEqual(
            (probe.head_dim, probe.n_heads, probe.n_kv_heads, probe.num_experts),
            (64, 32, 16, 4),
        )
        # Piper 9B is dim 2048 too, and agrees with none of those but n_heads.
        self.assertEqual(PIPER_9B.dim, probe.dim)
        self.assertEqual(PIPER_9B.n_kv_heads, 8)
        self.assertEqual(PIPER_9B.num_experts, 8)

        # A structural guard, not a value comparison. Comparing a registered
        # shape to a PiperShape rebuilt from its own fields is a tautology for
        # a frozen dataclass, so this reads the source instead: the registry
        # must construct every entry through the plain constructor.
        source = Path(
            inspect.getsourcefile(PiperShape) or ""
        ).read_text(encoding="utf-8")
        self.assertNotIn("PiperShape.derived(", source)
        self.assertIn("PIPER_9B = PiperShape(", source)

    def test_describe_is_json_safe(self) -> None:
        for shape in PIPER_SHAPES.values():
            described = shape.describe(seq_len=1024)
            self.assertEqual(json.loads(json.dumps(described)), described)
            self.assertEqual(described["name"], shape.name)

    def test_block_region_support_is_derived_from_the_layer_count(self) -> None:
        # Not a per-shape flag anyone can set wrong: a 1-layer block graph is
        # not structurally identifiable, at any dim.
        self.assertTrue(PIPER_1B.supports_block_regions)
        self.assertFalse(HUGE.supports_block_regions)
        self.assertFalse(
            PiperShape.derived(name="probe", dim=1024, n_layers=1).supports_block_regions
        )
        self.assertTrue(
            PiperShape.derived(name="probe", dim=1024, n_layers=2).supports_block_regions
        )

    def test_parity_gate_is_shape_data(self) -> None:
        # tools/megatron_parity_check.py reads these; the huge gate is wider
        # only because bf16 accumulation scales with the reduction length.
        self.assertEqual(PIPER_1B.parity_gate, 2e-2)
        self.assertEqual(HUGE.parity_gate, 5e-2)
        self.assertEqual(LARGE.parity_gate, 3e-2)
        self.assertEqual(GIANT.parity_gate, 6e-2)
        self.assertEqual(
            PiperShape.derived(name="probe", dim=1024, n_layers=2).parity_gate, 2e-2
        )
        for shape in PIPER_SHAPES.values():
            self.assertEqual(
                shape.describe(seq_len=1024)["parity_gate"], shape.parity_gate
            )

    def test_the_unverified_gates_follow_the_measured_ones(self) -> None:
        """The two new gates are estimates, and this states the estimate.

        Nothing has measured large or giant. The two measured shapes fit
        rel_l2 = 5.5e-3 * sqrt(dim/1024) to within 7% (normal 5.5e-3 at dim
        1024, huge 2.03e-2 at dim 12288 against 1.9e-2 predicted). Each new
        gate sits above that prediction by at least the 2.46x margin huge
        keeps over its own measurement, and below 4x it, so neither gate is
        so wide that it would pass a real layout error.
        """
        for shape in (LARGE, GIANT, PIPER_9B, PIPER_48B):
            with self.subTest(size=shape.name):
                predicted = 5.5e-3 * (shape.dim / PIPER_1B.dim) ** 0.5
                self.assertGreater(shape.parity_gate, 2.4 * predicted)
                self.assertLess(shape.parity_gate, 4.0 * predicted)
        # A wider shape never gets a tighter gate. Ordered by dim, not by the
        # declaration: the law reads dim alone, and the registry is ordered by
        # parameter count, which puts 24-layer 9b above 4-layer large.
        by_dim = sorted(PIPER_SHAPES.values(), key=lambda shape: shape.dim)
        gates = [shape.parity_gate for shape in by_dim]
        self.assertEqual(gates, sorted(gates))
        # large and 48b share a dim, so the law must give them one gate.
        self.assertEqual(LARGE.dim, PIPER_48B.dim)
        self.assertEqual(LARGE.parity_gate, PIPER_48B.parity_gate)


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
        "max_seq_len": 2048,
        "supports_block_regions": True,
        "parity_gate": 2e-2,
        "param_count": 1_066_241_024,
        "nparams_dense": 361_532_416,
        "nparams_sparse": 704_708_608,
        "nparams_active": 713_919_488,
        "num_flops_per_token": 3_551_348_736,
    },
    "large": {
        "dim": 4096,
        "n_layers": 4,
        "n_heads": 64,
        "n_kv_heads": 32,
        "head_dim": 64,
        "moe_hidden_dim": 14_336,
        "num_experts": 4,
        "top_k": 2,
        "vocab_size": 151_936,
        "rope_theta": 1_000_000.0,
        "max_seq_len": 2048,
        "supports_block_regions": True,
        "parity_gate": 3e-2,
        "param_count": 4_264_661_504,
        "nparams_dense": 1_446_023_680,
        "nparams_sparse": 2_818_637_824,
        "nparams_active": 2_855_375_360,
        "num_flops_per_token": 13_599_599_616,
    },
    # Piper 9B, transcribed from examples/models/qwen3.py case '9B'.
    "9b": {
        "dim": 2048,
        "n_layers": 24,
        "n_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 64,
        "moe_hidden_dim": 7168,
        "num_experts": 8,
        "top_k": 2,
        "vocab_size": 151_936,
        "rope_theta": 1_000_000.0,
        "max_seq_len": 2048,
        "supports_block_regions": True,
        "parity_gate": 2e-2,
        "param_count": 9_330_201_600,
        "nparams_dense": 874_091_520,
        "nparams_sparse": 8_456_110_080,
        "nparams_active": 2_988_413_952,
        "num_flops_per_token": 16_667_473_920,
    },
    "huge": {
        "dim": 12_288,
        "n_layers": 1,
        "n_heads": 192,
        "n_kv_heads": 96,
        "head_dim": 64,
        "moe_hidden_dim": 43_008,
        "num_experts": 4,
        "top_k": 2,
        "vocab_size": 151_936,
        "rope_theta": 1_000_000.0,
        "max_seq_len": 2048,
        "supports_block_regions": False,
        "parity_gate": 5e-2,
        "param_count": 10_528_837_760,
        "nparams_dense": 4_187_000_960,
        "nparams_sparse": 6_341_836_800,
        "nparams_active": 7_357_943_936,
        "num_flops_per_token": 33_096_721_152,
    },
    "giant": {
        "dim": 16_384,
        "n_layers": 1,
        "n_heads": 256,
        "n_kv_heads": 128,
        "head_dim": 64,
        "moe_hidden_dim": 57_344,
        "num_experts": 4,
        "top_k": 2,
        "vocab_size": 151_936,
        "rope_theta": 1_000_000.0,
        "max_seq_len": 2048,
        "supports_block_regions": False,
        "parity_gate": 6e-2,
        "param_count": 17_058_349_184,
        "nparams_dense": 5_783_994_496,
        "nparams_sparse": 11_274_354_688,
        "nparams_active": 11_421_204_608,
        "num_flops_per_token": 53_792_637_696,
    },
    # Piper 48B, transcribed from examples/models/qwen3.py case '48B'.
    "48b": {
        "dim": 4096,
        "n_layers": 32,
        "n_heads": 32,
        "n_kv_heads": 8,
        "head_dim": 128,
        "moe_hidden_dim": 14_336,
        "num_experts": 8,
        "top_k": 2,
        "vocab_size": 151_936,
        "rope_theta": 1_000_000.0,
        "max_seq_len": 2048,
        "supports_block_regions": True,
        "parity_gate": 3e-2,
        "param_count": 47_685_316_608,
        "nparams_dense": 2_587_111_424,
        "nparams_sparse": 45_098_205_184,
        "nparams_active": 13_862_449_152,
        "num_flops_per_token": 81_051_328_512,
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

    def test_an_uneven_split_raises_rather_than_rounding(self) -> None:
        # HUGE holds one layer, which is why parallelism rule 7 refuses it at
        # pp 2. This is the same refusal, one level down.
        with self.assertRaisesRegex(ValueError, "do not divide evenly"):
            HUGE.stage_param_count(pipeline_degree=2, stage_index=0)

    def test_a_stage_outside_the_pipeline_raises(self) -> None:
        for degree, stage in ((2, 2), (2, -1), (0, 0)):
            with self.subTest(degree=degree, stage=stage):
                with self.assertRaises(ValueError):
                    PIPER_1B.stage_param_count(
                        pipeline_degree=degree, stage_index=stage
                    )


class ModelSizeAliasTests(unittest.TestCase):
    """``normal`` is the retired name of ``1b``, and it must keep working.

    Measured under ``out/`` on 2026-08-21: 42 e2e manifests record
    ``"model_size": "normal"``, and 88 more record no size at all and are
    defined to resume as that shape. ``--resume`` compares the recorded
    string against the requested one, so a rename without an alias would
    refuse a resume that should succeed for 130 of the 144 e2e runs on disk.
    """

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

    def test_a_fresh_manifest_records_the_canonical_name(self) -> None:
        scenario = scenario_by_name("piper1b_rope")
        selected = (scenario.arm("baseline"),)
        recorded = manifest_data(
            scenario,
            selected,
            {"baseline": ["cmd"]},
            "test-gpu",
            _METADATA,
            (),
            "default",
            "sac",
            "normal",
            parallelism=TRIVIAL_SPEC,
        )
        self.assertEqual(recorded["model_size"], "1b")
        self.assertEqual(recorded["model_shape"]["name"], "1b")

    def test_a_manifest_recording_either_name_resumes_against_the_other(
        self,
    ) -> None:
        """The site the rename could most easily have broken.

        ``_resume_mismatches`` compares the recorded size against the
        requested one. Both sides must normalise, or a run recorded before the
        rename would be refused for naming its own shape.
        """
        scenario = scenario_by_name("piper1b_rope")
        selected = (scenario.arm("baseline"),)
        for recorded_name, requested in (
            ("normal", "1b"),
            ("1b", "normal"),
            ("normal", "normal"),
            ("1b", "1b"),
        ):
            with self.subTest(recorded=recorded_name, requested=requested):
                manifest = manifest_data(
                    scenario,
                    selected,
                    {"baseline": ["cmd"]},
                    "test-gpu",
                    _METADATA,
                    (),
                    "default",
                    "sac",
                    "1b",
                    parallelism=TRIVIAL_SPEC,
                )
                # Written by hand, because manifest_data canonicalises: an
                # on-disk manifest from before the rename says "normal".
                manifest["model_size"] = recorded_name
                self.assertEqual(
                    _resume_mismatches(
                        manifest,
                        scenario,
                        selected,
                        "test-gpu",
                        _METADATA,
                        (),
                        "default",
                        "sac",
                        requested,
                        parallelism=TRIVIAL_SPEC,
                    ),
                    [],
                )
        # A genuinely different size is still refused.
        manifest = manifest_data(
            scenario,
            selected,
            {"baseline": ["cmd"]},
            "test-gpu",
            _METADATA,
            (),
            "default",
            "sac",
            "normal",
            parallelism=TRIVIAL_SPEC,
        )
        self.assertEqual(
            _resume_mismatches(
                manifest,
                scenario,
                selected,
                "test-gpu",
                _METADATA,
                (),
                "default",
                "sac",
                "huge",
                parallelism=TRIVIAL_SPEC,
            ),
            ["model_size"],
        )


class ConfigSizeClosureTests(unittest.TestCase):
    def test_every_scenario_arm_builds_at_every_size(self) -> None:
        """Every arm's config resolves and accepts every registered size.

        This is the closure the runner depends on: it emits ``--config <name>
        --config-arg size=<size>`` for whatever the arm names, so a config
        that did not accept the keyword -- or accepted it and ignored it --
        would publish a run under the wrong shape.
        """
        import inspect

        import benchmarks.models.piper_qwen3.config_registry as registry

        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                if arm.launcher != "torchtitan":
                    continue
                name = arm.config or scenario.workload.config
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
            # fields, and the class no run at 9b or 48b has yet exercised.
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

        from benchmarks.models.piper_qwen3.config_registry import (
            _piper_1b_model,
            _piper_1b_trainer,
        )

        for builder in (_piper_1b_model, _piper_1b_trainer):
            with self.subTest(builder=builder.__name__):
                parameter = inspect.signature(builder).parameters["shape"]
                self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
                self.assertIs(parameter.default, inspect.Parameter.empty)
        with self.assertRaises(TypeError):
            _piper_1b_model(fuse_qkv=True)
        with self.assertRaises(TypeError):
            _piper_1b_trainer(fuse_qkv=True, loss_kind="full_logits")

    def test_the_megatron_builder_requires_an_explicit_shape(self) -> None:
        """The same rule on the other engine's builder, for the same reason.

        ``build_model`` is the megatron twin of ``_piper_1b_model``: both
        construct the same geometry from the same ``PiperShape``, and the
        whole point of that sharing is that a size cannot drift between the
        engines. A default here would reintroduce the drift on one side --
        and it would land in the arm with the least protection, since
        ``tools/megatron_parity_check.py`` builds the model outside the
        harness and so never reaches validation rule 11's parameter-count
        check. Signature inspection only; this imports no megatron.
        """
        import inspect

        from benchmarks.models.piper_qwen3.megatron_model import build_model

        parameter = inspect.signature(build_model).parameters["shape"]
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameter.default, inspect.Parameter.empty)

    def test_size_round_trips_through_the_config_argument(self) -> None:
        from benchmarks.models.piper_qwen3.config_registry import qwen3_piper_1b

        self.assertEqual(qwen3_piper_1b(size="huge").model_spec.model.dim, 12288)
        self.assertEqual(qwen3_piper_1b(size="normal").model_spec.model.dim, 1024)
        # The default is the normal shape, so an unparameterized call is the
        # historical config.
        self.assertEqual(qwen3_piper_1b().model_spec.model.dim, PIPER_1B.dim)
        with self.assertRaisesRegex(ValueError, "Unknown model size"):
            qwen3_piper_1b(size="enormous")

    def test_built_models_carry_the_requested_shape(self) -> None:
        from benchmarks.models.piper_qwen3.config_registry import qwen3_piper_1b

        normal = qwen3_piper_1b().model_spec.model
        self.assertEqual(normal.dim, PIPER_1B.dim)
        self.assertEqual(len(normal.layers), PIPER_1B.n_layers)

        huge = qwen3_piper_1b(size="huge").model_spec.model
        self.assertEqual(huge.dim, HUGE.dim)
        self.assertEqual(len(huge.layers), HUGE.n_layers)
        self.assertEqual(huge.vocab_size, HUGE.vocab_size)
        # The load_balance_coeff=None fixup must survive the parameterization.
        self.assertIsNone(huge.layers[0].moe.load_balance_coeff)

    def test_pretokenized_configs_pass_the_size_down_to_their_delegate(self) -> None:
        from benchmarks.models.piper_qwen3.config_registry import (
            qwen3_piper_1b_piper_optimized_te_ce_pretokenized,
            qwen3_piper_1b_pretokenized,
        )

        for factory in (
            qwen3_piper_1b_pretokenized,
            qwen3_piper_1b_piper_optimized_te_ce_pretokenized,
        ):
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
        scenario = scenario_by_name("piper1b_megatron")
        arm = scenario.arm("titan_stock")
        command = command_for_arm(
            scenario.workload, arm, Path("/tmp/arm"), (), model_size="huge"
        )
        # The config name is the arm's, unmangled: the shape rides alongside.
        self.assertEqual(
            command[command.index("--config") + 1],
            "qwen3_piper_1b_pretokenized",
        )
        self.assertEqual(
            command[command.index("--config-arg") + 1], "size=huge"
        )
        self.assertFalse([token for token in command if token.endswith("_huge")])
        # replay_steps must track --training.steps or the loader hard-fails.
        self.assertEqual(
            command[command.index("--dataloader.replay-steps") + 1],
            command[command.index("--training.steps") + 1],
        )

    def test_the_default_size_is_delivered_explicitly_too(self) -> None:
        scenario = scenario_by_name("piper1b_rope")
        command = command_for_arm(
            scenario.workload, scenario.arm("baseline"), Path("/tmp/arm"), ()
        )
        self.assertEqual(
            command[command.index("--config") + 1], "qwen3_piper_1b"
        )
        self.assertEqual(
            command[command.index("--config-arg") + 1], "size=1b"
        )

    def test_non_replay_scenarios_do_not_get_the_replay_flag(self) -> None:
        scenario = scenario_by_name("piper1b_rope")
        command = command_for_arm(
            scenario.workload, scenario.arm("baseline"), Path("/tmp/arm"), ()
        )
        self.assertNotIn("--dataloader.replay-steps", command)

    def test_megatron_command_carries_the_model_size(self) -> None:
        scenario = scenario_by_name("piper1b_megatron")
        command = command_for_arm(
            scenario.workload,
            scenario.arm("baseline"),
            Path("/tmp/arm"),
            (),
            "default",
            "none",
            model_size="huge",
        )
        self.assertEqual(command[command.index("--model-size") + 1], "huge")

    def test_megatron_driver_accepts_the_flag(self) -> None:
        from benchmarks.e2e.megatron.train import parse_args

        parsed = parse_args(
            [
                "--seq-len", "1024", "--steps", "80", "--batch", "4",
                "--seed", "42", "--profile-freq", "20",
                "--profiler-warmup", "5", "--profiler-active", "5",
                "--mode", "default", "--model-size", "huge", "/tmp/arm",
            ]
        )
        self.assertEqual(parsed.model_size, "huge")
        self.assertEqual(parse_args(
            [
                "--seq-len", "1024", "--steps", "40", "--batch", "4",
                "--seed", "42", "--profile-freq", "20",
                "--profiler-warmup", "5", "--profiler-active", "5",
                "--mode", "default", "/tmp/arm",
            ]
        ).model_size, "1b")


class RegionDerivationTests(unittest.TestCase):
    def test_the_factory_reproduces_the_historical_counts(self) -> None:
        self.assertEqual(
            [
                (r.name, r.phase, r.invocations_per_window)
                for r in piper_block_regions(n_layers=16, profiler_active=5)
            ],
            [("backward_block", "backward", 80), ("forward_block", "forward", 80)],
        )

    def test_one_layer_regions_would_collide_so_huge_declares_none(self) -> None:
        # At one layer the block graph runs profiler_active times per window,
        # which the loss-side partitions also do -- the reason
        # supports_block_regions is False rather than rescaled to 5.
        self.assertEqual(
            piper_block_regions(n_layers=1, profiler_active=5)[0]
            .invocations_per_window,
            5,
        )
        self.assertFalse(HUGE.supports_block_regions)


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
        scenario = scenario_by_name("piper1b_rope")
        arm = scenario.arm("baseline")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            log = self._fixture(root)
            head = _compiled_line("default") + _SAC_LINE

            log.write_text(head + "Training completed\n")
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate_arm(arm, root, log, scenario.workload)

            log.write_text(head + _size_line(PIPER_1B) + "Training completed\n")
            validate_arm(arm, root, log, scenario.workload)

            # The normal-size marker must not satisfy a huge-size run.
            with self.assertRaisesRegex(RuntimeError, "did not apply"):
                validate_arm(
                    arm, root, log, scenario.workload, model_size="huge"
                )

            log.write_text(head + _size_line(HUGE) + "Training completed\n")
            validate_arm(arm, root, log, scenario.workload, model_size="huge")

    def test_override_count_scales_with_the_layer_count(self) -> None:
        scenario = scenario_by_name("piper1b_swiglu")
        arm = scenario.arm("piper_optimized_triton")
        applied = (
            f"[Override] {arm.override_imports[0]}: "
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

            log.write_text(head + _size_line(HUGE) + "Training completed\n" + applied)
            validate_arm(
                arm, root, log, scenario.workload, model_size="huge"
            )

            log.write_text(
                head + _size_line(HUGE) + "Training completed\n" + applied * 16
            )
            with self.assertRaisesRegex(RuntimeError, "expected 1 override"):
                validate_arm(
                    arm, root, log, scenario.workload, model_size="huge"
                )

            log.write_text(
                head + _size_line(PIPER_1B) + "Training completed\n" + applied * 16
            )
            validate_arm(arm, root, log, scenario.workload)


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


class ManifestAndResumeTests(unittest.TestCase):
    def _run(self, **request_kwargs):
        with mock.patch(
            "benchmarks.e2e.runner.hardware_metadata",
            return_value=("test-gpu", _METADATA),
        ), mock.patch(
            "benchmarks.e2e.runner.resolve_cpu_pinning",
            return_value=CpuPinning((), "none: test"),
        ):
            return execute_run(
                RunRequest(gpu="0", **request_kwargs),
                process_runner=_fake_process(_size_line(HUGE)),
                environment={"PATH": os.environ["PATH"]},
            )

    def test_huge_run_declares_no_regions_and_records_the_shape(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            self._run(
                scenario_name="piper1b_attention",
                arm_names=("baseline",),
                out_dir=out_dir,
                ac_mode="none",
                model_size="huge",
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())

        self.assertEqual(manifest["schema_version"], 12)
        self.assertEqual(manifest["model_size"], "huge")
        self.assertEqual(manifest["model_shape"], HUGE.describe(seq_len=1024))
        # Rule 7's structural matcher cannot identify a 1-layer block graph,
        # so the run says so instead of claiming a region it cannot verify.
        self.assertEqual(manifest["regions"], [])
        command = manifest["commands"]["baseline"]
        self.assertEqual(command[command.index("--config") + 1], "qwen3_piper_1b")
        self.assertEqual(command[command.index("--config-arg") + 1], "size=huge")

    def test_normal_run_still_declares_the_eighty_invocation_regions(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            with mock.patch(
                "benchmarks.e2e.runner.hardware_metadata",
                return_value=("test-gpu", _METADATA),
            ), mock.patch(
                "benchmarks.e2e.runner.resolve_cpu_pinning",
                return_value=CpuPinning((), "none: test"),
            ), mock.patch("benchmarks.e2e.runner.validate_arm"):
                execute_run(
                    RunRequest(
                        gpu="0",
                        scenario_name="piper1b_rope",
                        arm_names=("baseline",),
                        out_dir=out_dir,
                    ),
                    process_runner=_fake_process(_size_line(PIPER_1B)),
                    environment={"PATH": os.environ["PATH"]},
                )
            manifest = json.loads((out_dir / "manifest.json").read_text())
        self.assertEqual(
            [
                (region["name"], region["invocations_per_window"])
                for region in manifest["regions"]
            ],
            [("backward_block", 80), ("forward_block", 80)],
        )

    def test_resume_refuses_a_different_size_and_inherits_an_absent_one(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "run"
            self._run(
                scenario_name="piper1b_attention",
                arm_names=("baseline",),
                out_dir=out_dir,
                ac_mode="none",
                model_size="huge",
            )

            with self.assertRaisesRegex(Exception, "model_size"):
                self._run(
                    scenario_name=None,
                    arm_names=("baseline",),
                    resume_dir=out_dir,
                    ac_mode="none",
                    model_size="normal",
                )

            # Omitting --model-size on a resume inherits the recorded value.
            self._run(
                scenario_name=None,
                arm_names=("baseline",),
                resume_dir=out_dir,
                ac_mode="none",
            )

    def test_schema_eight_directories_still_resume_as_the_1b_shape(self) -> None:
        scenario = scenario_by_name("piper1b_rope")
        selected = (scenario.arm("baseline"),)
        with tempfile.TemporaryDirectory() as temporary:
            out_dir = Path(temporary)
            write_manifest(
                out_dir,
                scenario,
                selected,
                {"baseline": ["cmd"]},
                "test-gpu",
                _METADATA,
                (),
                "default",
                "sac",
                "normal",
                parallelism=TRIVIAL_SPEC,
            )
            manifest = json.loads((out_dir / "manifest.json").read_text())
            del manifest["model_size"]
            del manifest["model_shape"]
            manifest["schema_version"] = 8
            (out_dir / "manifest.json").write_text(json.dumps(manifest))

            from benchmarks.artifacts.manifests import _resume_mismatches

            self.assertEqual(
                _resume_mismatches(
                    manifest,
                    scenario,
                    selected,
                    "test-gpu",
                    _METADATA,
                    (),
                    "default",
                    "sac",
                    "normal",
                    parallelism=TRIVIAL_SPEC,
                ),
                [],
            )
            self.assertIn(
                "model_size",
                _resume_mismatches(
                    manifest,
                    scenario,
                    selected,
                    "test-gpu",
                    _METADATA,
                    (),
                    "default",
                    "sac",
                    "huge",
                    parallelism=TRIVIAL_SPEC,
                ),
            )


if __name__ == "__main__":
    unittest.main()
