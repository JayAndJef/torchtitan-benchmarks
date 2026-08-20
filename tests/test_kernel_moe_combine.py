"""CPU tests for the ``moe_combine`` cross-engine kernel scenario.

Everything here runs without a GPU. The scenario makes four claims a CPU can
check, and one it cannot.

* **The two engines compute different functions**, because the routing
  probabilities are applied on opposite sides of this cut. That is why the
  scenario publishes no cross-engine ratio and declares no cross-engine gate,
  and the reference builder's two differently-named truths are where the fact
  becomes executable.
* **Both engines combine the same rows in the same order.** The canonical
  ``(expert ascending, token ascending)`` order is derived from the routing
  map, and titan's own argsort reproduces it -- checked here on real tensors
  rather than argued in a docstring.
* **The titan arm computes the scenario's operation**, including the score
  multiply, under ``torch.compile(fullgraph=True)``. It is the one arm a CPU
  can build, and it is built and run.
* **The build-time guard refuses a combine that does not unpermute.** The
  guard is exercised against stand-in dispatchers: one correct, one identity,
  one that restores the wrong order, and one whose ``token_combine`` is not
  inert.

**No test here imports megatron**, and that is a cost decision rather than an
oversight. ``megatron/core/__init__.py`` imports TransformerEngine, so the
cheapest ``from megatron.core...`` costs about 9 s and pulls a GPU library
into a suite that otherwise runs in seconds. The pinned sources are read as
**text** instead, which is what a submodule bump has to invalidate anyway;
``tests/test_kernel_moe_residual.py`` and ``tests/test_lm_head_losses.py`` do
the same.

What that leaves untested on CPU is ``_assert_mcore_dispatcher``, whose first
statement is a megatron import, and the three megatron arms themselves. Each
guard runs on every mcore build, so a wiring error raises there rather than
reaching a number. **Nothing in this scenario has ever run on a GPU.**
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.execution.paths import TITAN_DIR
from benchmarks.kernel.operations.moe_combine import (
    _assert_mcore_combine,
    _canonical_combine,
    _canonical_row_order,
    _combine_arm,
    _require_grads,
    build_moe_combine_copy_floor,
    build_moe_combine_titan,
    COMBINE_GUARD_REL_L2,
    COPY_FLOOR_ARM,
    DISPATCHER_ALLTOALL,
    MCORE_ALLTOALL_ARM,
    MCORE_BASE_ARM,
    MCORE_NO_PERMUTE_FUSION_ARM,
    moe_combine_inputs,
    moe_combine_reference,
    MoeCombineInputs,
    NO_PERMUTE_FUSION,
    TITAN_ARM,
    TITAN_COMM_BACKEND,
)
from benchmarks.kernel.schema import fragment_stem, KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE
from benchmarks.models.piper_qwen3.megatron_bootstrap import megatron_dir
from benchmarks.models.piper_qwen3.shape import PiperShape, shape_by_name

# Small enough to run in milliseconds, and wide enough that a lost row or a
# transposed view shows up rather than cancelling. ``dim`` must be a multiple
# of ``2 * head_dim``; 256 is the smallest that is.
TINY = PiperShape(name="tiny", dim=256, n_layers=2, vocab_size=64)
TINY_WORKLOAD = KernelWorkload(batch=2, seq_len=8)

MEGATRON = megatron_dir()
TOKEN_DISPATCHER = MEGATRON / "megatron/core/transformer/moe/token_dispatcher.py"
MOE_UTILS = MEGATRON / "megatron/core/transformer/moe/moe_utils.py"
MOE_LAYER = MEGATRON / "megatron/core/transformer/moe/moe_layer.py"
MAPPINGS = MEGATRON / "megatron/core/tensor_parallel/mappings.py"
TITAN_DISPATCHER = TITAN_DIR / "torchtitan/models/common/token_dispatcher.py"
TITAN_MOE = TITAN_DIR / "torchtitan/models/common/moe.py"
TITAN_CONFIG_UTILS = TITAN_DIR / "torchtitan/models/common/config_utils.py"


def _inputs(shape=TINY, workload=TINY_WORKLOAD, seed=0) -> MoeCombineInputs:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return moe_combine_inputs(shape, workload, torch.device("cpu"), generator)


def _rel_l2(value: torch.Tensor, truth: torch.Tensor) -> float:
    delta = (value.double() - truth.double()).norm()
    return (delta / truth.double().norm()).item()


def _source(path: Path) -> str:
    if not path.exists():
        raise unittest.SkipTest(f"{path} is absent; the submodule is not checked out")
    return path.read_text()


def _closed_over(closure, name: str):
    """One named free variable of a closure, for the floor's own buffer.

    By name rather than by index: ``__closure__`` is ordered by
    ``co_freevars``, so an index would silently read a different tensor the
    moment a variable is renamed.
    """
    cells = dict(zip(closure.__code__.co_freevars, closure.__closure__))
    return cells[name].cell_contents


class _StandInDispatcher:
    """A minimal allgather-shaped dispatcher, for the guard tests alone.

    It reproduces the three phases at ``world_size=1``: an unpermute in
    ``combine_preprocess``, an identity ``token_combine``, and a view in
    ``combine_postprocess``. ``row_token`` is the order it restores, so a test
    can hand it the wrong one and watch the guard refuse the arm.
    """

    def __init__(self, row_token, hidden_shape, inert=True, unpermute=True):
        self.row_token = row_token
        self.hidden_shape = hidden_shape
        self.inert = inert
        self.unpermute = unpermute
        self.tp_size = 1
        self.ep_size = 1

    def combine_preprocess(self, hidden_states):
        if not self.unpermute:
            return hidden_states
        tokens = self.hidden_shape[0]
        index = self.row_token.reshape(-1, 1).expand(-1, hidden_states.shape[-1])
        out = torch.zeros(
            (tokens, hidden_states.shape[-1]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        return out.scatter_add(0, index, hidden_states)

    def token_combine(self, hidden_states):
        return hidden_states if self.inert else hidden_states * 2

    def combine_postprocess(self, hidden_states):
        if not self.unpermute:
            # The pass-through case in full: cutting on ``MoELayer.combine``
            # would have handed back the routed rows untouched, so the shape
            # restore never runs either.
            return hidden_states
        return hidden_states.view(self.hidden_shape)

    def combine(self, expert_out):
        return self.combine_postprocess(
            self.token_combine(self.combine_preprocess(expert_out))
        )


class ProfileTests(unittest.TestCase):
    """The two deltas this scenario declares, and the base they sit on."""

    def test_the_base_profile_is_the_fused_allgather_dispatcher(self) -> None:
        """Both published rows are deltas against these two fields."""
        self.assertIs(BASE.config_overrides["moe_permute_fusion"], True)
        self.assertEqual(
            BASE.config_overrides["moe_token_dispatcher_type"], "allgather"
        )

    def test_each_variant_moves_exactly_one_field(self) -> None:
        """A delta that moved a second field would confound its row.

        Each arm is one flag from the anchor by construction. Anything else in
        the diff makes the published ratio a comparison of two configurations
        rather than of two implementations of one combine.
        """
        for profile, field, value in (
            (NO_PERMUTE_FUSION, "moe_permute_fusion", False),
            (DISPATCHER_ALLTOALL, "moe_token_dispatcher_type", "alltoall"),
        ):
            with self.subTest(profile=profile.name):
                differing = {
                    key
                    for key in set(BASE.config_overrides)
                    | set(profile.config_overrides)
                    if BASE.config_overrides.get(key)
                    != profile.config_overrides.get(key)
                }
                self.assertEqual(differing, {field})
                self.assertEqual(profile.config_overrides[field], value)

    def test_the_alltoall_variant_keeps_the_permutation_fusion_on(self) -> None:
        """Two deltas in one arm would confound the two published rows.

        It also protects the scenario's device-time declaration: the unfused
        ``sort_chunks_by_idxs`` calls ``split_sizes.tolist()``, and only the
        alltoall class reaches that function from a combine phase.
        """
        self.assertIs(DISPATCHER_ALLTOALL.config_overrides["moe_permute_fusion"], True)

    def test_both_variants_are_recordable_provenance(self) -> None:
        """A profile must reach a manifest, so it stays JSON-safe data."""
        for profile, word in (
            (NO_PERMUTE_FUSION, "fused_unpermute"),
            (DISPATCHER_ALLTOALL, "inert"),
        ):
            with self.subTest(profile=profile.name):
                described = profile.describe()
                self.assertEqual(described["name"], profile.name)
                self.assertIn(word, described["description"])


class RowOrderTests(unittest.TestCase):
    """The linchpin: one row order, and both engines produce it."""

    def test_the_canonical_order_is_expert_major_and_token_ascending(self) -> None:
        routing_map = torch.tensor(
            [[True, False, True], [False, True, True], [True, True, False]]
        )
        row_token, row_expert = _canonical_row_order(routing_map)
        self.assertEqual(row_expert.tolist(), [0, 0, 1, 1, 2, 2])
        self.assertEqual(row_token.tolist(), [0, 2, 1, 2, 0, 1])

    def test_titans_argsort_reproduces_the_order_derived_from_the_map(self) -> None:
        """Stated in the module docstring, executed here.

        megatron argsorts ``routing_map.T`` and titan argsorts
        ``topk_expert_ids``. The two flat indices are different quantities, so
        the agreement is a fact about the data rather than about the code, and
        it is the whole reason no arm needs a gather.
        """
        inputs = _inputs()
        titan_order = torch.argsort(
            inputs.topk_expert_ids_TK.reshape(-1), stable=True
        )
        self.assertTrue(
            torch.equal(titan_order // TINY.top_k, inputs.row_token_N)
        )

    def test_every_row_is_one_routed_token_expert_pair(self) -> None:
        inputs = _inputs()
        pairs = {
            (int(t), int(e))
            for t, e in zip(inputs.row_token_N, inputs.row_expert_N)
        }
        self.assertEqual(len(pairs), inputs.row_token_N.numel())
        for token, expert in pairs:
            self.assertTrue(bool(inputs.routing_map_TE[token, expert]))

    def test_the_sorted_scores_are_the_probs_of_those_pairs(self) -> None:
        """``titan_scores_grad`` is gated against a truth built from these."""
        inputs = _inputs()
        expected = inputs.probs_TE[inputs.row_token_N, inputs.row_expert_N]
        self.assertTrue(torch.equal(inputs.scores_sorted_N, expected))


class InputsTests(unittest.TestCase):
    def test_the_shared_tensors_have_the_declared_shapes(self) -> None:
        inputs = _inputs()
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        rows = tokens * TINY.top_k
        self.assertEqual(
            tuple(inputs.x_BLD.shape),
            (TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len, TINY.dim),
        )
        self.assertEqual(tuple(inputs.expert_out_ND.shape), (rows, TINY.dim))
        self.assertEqual(tuple(inputs.grad_out_TD.shape), (tokens, TINY.dim))
        self.assertEqual(
            tuple(inputs.routing_map_TE.shape), (tokens, TINY.num_experts)
        )
        self.assertEqual(tuple(inputs.probs_TE.shape), (tokens, TINY.num_experts))
        self.assertEqual(
            tuple(inputs.topk_expert_ids_TK.shape), (tokens, TINY.top_k)
        )
        self.assertEqual(inputs.scores_sorted_N.numel(), rows)

    def test_the_two_engine_forms_encode_one_routing_decision(self) -> None:
        """megatron reads the dense pair and titan reads the top-k pair.

        A scenario whose two sides routed differently would still combine
        correctly on each side and would still pass every gate, because each
        engine is gated against its own truth. Only this catches it.
        """
        inputs = _inputs()
        rebuilt = torch.zeros_like(inputs.routing_map_TE).scatter(
            1, inputs.topk_expert_ids_TK, True
        )
        self.assertTrue(torch.equal(rebuilt, inputs.routing_map_TE))
        self.assertTrue(
            torch.equal(
                inputs.probs_TE.gather(1, inputs.topk_expert_ids_TK),
                inputs.topk_scores_TK,
            )
        )

    def test_the_routing_is_exactly_balanced(self) -> None:
        inputs = _inputs()
        rows = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len * TINY.top_k
        per_expert = rows // TINY.num_experts
        self.assertEqual(
            inputs.tokens_per_expert_E.tolist(), [per_expert] * TINY.num_experts
        )

    def test_each_token_reaches_top_k_distinct_experts(self) -> None:
        inputs = _inputs()
        self.assertTrue(
            torch.equal(
                inputs.routing_map_TE.sum(dim=1),
                torch.full((inputs.routing_map_TE.shape[0],), TINY.top_k),
            )
        )

    def test_bytes_moved_is_the_minimum_traffic_of_a_combine(self) -> None:
        """One read of every routed row and one write of every token row."""
        inputs = _inputs()
        expected = (
            inputs.expert_out_ND.numel() + inputs.grad_out_TD.numel()
        ) * inputs.expert_out_ND.element_size()
        self.assertEqual(inputs.bytes_moved, expected)

    def test_one_seed_rebuilds_the_inputs_bit_identically(self) -> None:
        """Every worker rebuilds these, and the gates compare across workers."""
        first, second = _inputs(seed=7), _inputs(seed=7)
        self.assertTrue(torch.equal(first.expert_out_ND, second.expert_out_ND))
        self.assertTrue(torch.equal(first.topk_scores_TK, second.topk_scores_TK))
        self.assertTrue(torch.equal(first.x_BLD, second.x_BLD))

    def test_an_uneven_split_fails_loudly_with_named_numbers(self) -> None:
        """Capped or rounded, this would measure a split nothing declares."""
        with self.assertRaisesRegex(ValueError, "do not divide evenly"):
            _inputs(workload=KernelWorkload(batch=1, seq_len=3))

    def test_a_top_k_above_the_expert_count_is_refused(self) -> None:
        """A token would reach one expert twice and the order would collide."""
        wide = PiperShape(
            name="wide", dim=256, n_layers=2, vocab_size=64, num_experts=2, top_k=4
        )
        with self.assertRaisesRegex(ValueError, "exceeds num_experts"):
            _inputs(shape=wide, workload=KernelWorkload(batch=1, seq_len=2))


class ReferenceTests(unittest.TestCase):
    def test_the_reference_names_one_truth_per_engine(self) -> None:
        """Different names because the two engines compute different functions.

        A shared ``out`` would force one side to be gated against the other's
        function, and the difference is a row-by-row scaling rather than a
        tolerance.
        """
        reference = moe_combine_reference(TINY, TINY_WORKLOAD, _inputs())
        self.assertEqual(
            sorted(reference),
            [
                "mcore_expert_out_grad",
                "mcore_out",
                "titan_expert_out_grad",
                "titan_out",
                "titan_scores_grad",
            ],
        )

    def test_the_megatron_truth_applies_no_probability(self) -> None:
        inputs = _inputs()
        reference = moe_combine_reference(TINY, TINY_WORKLOAD, inputs)
        expected = torch.zeros(
            (TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len, TINY.dim),
            dtype=torch.float64,
        ).index_add(0, inputs.row_token_N, inputs.expert_out_ND.double())
        self.assertTrue(
            torch.allclose(reference["mcore_out"].reshape(-1, TINY.dim), expected)
        )

    def test_the_titan_truth_is_the_megatron_truth_with_the_probabilities(
        self,
    ) -> None:
        """The scenario's central claim, as arithmetic rather than as prose."""
        inputs = _inputs()
        reference = moe_combine_reference(TINY, TINY_WORKLOAD, inputs)
        weighted = inputs.expert_out_ND.double() * inputs.scores_sorted_N.double(
        ).unsqueeze(-1)
        expected = torch.zeros(
            (TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len, TINY.dim),
            dtype=torch.float64,
        ).index_add(0, inputs.row_token_N, weighted)
        self.assertTrue(
            torch.allclose(reference["titan_out"].reshape(-1, TINY.dim), expected)
        )
        self.assertFalse(
            torch.allclose(reference["titan_out"], reference["mcore_out"])
        )

    def test_the_gradients_differ_by_the_same_factor(self) -> None:
        """megatron's combine routes the gradient; titan's scales it.

        This is what a cross-engine gate would have to reconcile, and it is
        why the scenario declares none.
        """
        inputs = _inputs()
        reference = moe_combine_reference(TINY, TINY_WORKLOAD, inputs)
        mcore = reference["mcore_expert_out_grad"]
        titan = reference["titan_expert_out_grad"]
        self.assertTrue(
            torch.allclose(titan, mcore * inputs.scores_sorted_N.double().unsqueeze(-1))
        )

    def test_the_score_gradient_is_the_row_wise_inner_product(self) -> None:
        inputs = _inputs()
        reference = moe_combine_reference(TINY, TINY_WORKLOAD, inputs)
        expected = (
            inputs.expert_out_ND.double()
            * inputs.grad_out_TD.double()[inputs.row_token_N]
        ).sum(dim=-1)
        self.assertTrue(torch.allclose(reference["titan_scores_grad"], expected))


class ArmClosureTests(unittest.TestCase):
    """The closures every implementation arm shares, over a stand-in call."""

    def _arm(self, inputs: MoeCombineInputs, native_thd=False):
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        hidden_shape = (tokens, 1, TINY.dim) if native_thd else (tokens, TINY.dim)
        dispatcher = _StandInDispatcher(inputs.row_token_N, hidden_shape)
        return _combine_arm(
            name="stand_in",
            call=dispatcher.combine,
            inputs=inputs,
            grad_native=inputs.grad_out_TD.reshape(hidden_shape),
            canonical=tuple(inputs.x_BLD.shape),
            output_prefix="mcore",
        )

    def test_the_arm_declares_forward_and_forward_backward_only(self) -> None:
        """No isolated backward: TE's fused unpermute is not re-runnable over
        a retained graph, and dropping the mode from every arm keeps them
        comparable. Backward cost stays recoverable as the difference."""
        self.assertEqual(
            sorted(self._arm(_inputs()).calls), ["forward", "forward_backward"]
        )

    def test_the_forward_closure_builds_a_graph(self) -> None:
        """The forward mode carries autograd, as production does."""
        out = self._arm(_inputs()).calls["forward"]()
        self.assertIsNotNone(out.grad_fn)

    def test_a_thd_arm_returns_canonical_outputs(self) -> None:
        """The mcore arms run ``[t, 1, D]`` and the gates compare ``[B, L, D]``.

        Without this the gate would fail on shape rather than on numbers, and
        a reshape that reordered elements would pass it.
        """
        inputs = _inputs()
        outputs = self._arm(inputs, native_thd=True).correctness_outputs()
        reference = moe_combine_reference(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(
            sorted(outputs), ["mcore_expert_out_grad", "mcore_out"]
        )
        self.assertEqual(tuple(outputs["mcore_out"].shape), tuple(inputs.x_BLD.shape))
        for name in outputs:
            with self.subTest(output=name):
                self.assertLess(_rel_l2(outputs[name], reference[name]), 2e-2)

    def test_a_repeated_round_trip_does_not_accumulate_a_gradient(self) -> None:
        """Every timed call must measure one backward, not a growing sum."""
        arm = self._arm(_inputs())
        round_trip = arm.calls["forward_backward"]
        round_trip()
        first = arm.correctness_outputs()["mcore_expert_out_grad"].clone()
        for _ in range(3):
            round_trip()
        self.assertTrue(
            torch.equal(first, arm.correctness_outputs()["mcore_expert_out_grad"])
        )

    def test_no_arm_marks_the_shared_expert_outputs(self) -> None:
        """``_combine_arm`` clones before it calls ``requires_grad_``.

        Were it to mark the shared tensor in place, the floor built after it
        in the same interpreter would start building a graph per call and
        would stop being a floor -- silently, and only in the correctness
        pass, which is the one pass that builds every arm together.
        """
        inputs = _inputs()
        self._arm(inputs)
        build_moe_combine_titan(TINY, TINY_WORKLOAD, inputs)
        self.assertFalse(inputs.expert_out_ND.requires_grad)
        self.assertFalse(inputs.scores_sorted_N.requires_grad)


class FloorTests(unittest.TestCase):
    def test_the_floor_declares_forward_alone_and_the_shared_traffic(self) -> None:
        """A floor for the forward+backward traffic would be an invention."""
        inputs = _inputs()
        arm = build_moe_combine_copy_floor(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(arm.name, COPY_FLOOR_ARM)
        self.assertEqual(sorted(arm.calls), ["forward"])
        self.assertEqual(arm.bytes_moved, inputs.bytes_moved)
        self.assertEqual(arm.correctness_outputs(), {})

    def test_the_floor_performs_the_combines_additions_contiguously(self) -> None:
        """Same traffic, same adds, no indirection. That is the whole claim."""
        inputs = _inputs()
        arm = build_moe_combine_copy_floor(TINY, TINY_WORKLOAD, inputs)
        self.assertIsNone(arm.calls["forward"]())
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        expected = inputs.expert_out_ND.view(tokens, TINY.top_k, TINY.dim).sum(dim=1)
        floor_out = _closed_over(arm.calls["forward"], "out")
        self.assertTrue(torch.equal(floor_out, expected))

    def test_the_floor_builds_no_autograd_graph(self) -> None:
        """It is a bandwidth reference, so it must not pay for a graph."""
        inputs = _inputs()
        arm = build_moe_combine_copy_floor(TINY, TINY_WORKLOAD, inputs)
        arm.calls["forward"]()
        self.assertIsNone(_closed_over(arm.calls["forward"], "out").grad_fn)


class TitanArmTests(unittest.TestCase):
    """The one arm a CPU can build, built and run.

    It is the only executable evidence outside a GPU that the compiled closure
    computes the scenario's operation. All three megatron arms need a device, a
    process group and TransformerEngine.
    """

    def test_the_arm_declares_the_shared_modes_and_bytes(self) -> None:
        inputs = _inputs()
        arm = build_moe_combine_titan(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(arm.name, TITAN_ARM)
        self.assertEqual(sorted(arm.calls), ["forward", "forward_backward"])
        self.assertEqual(arm.bytes_moved, inputs.bytes_moved)

    def test_the_arm_runs_and_matches_its_own_reference(self) -> None:
        inputs = _inputs()
        arm = build_moe_combine_titan(TINY, TINY_WORKLOAD, inputs)
        arm.calls["forward_backward"]()
        outputs = arm.correctness_outputs()
        reference = moe_combine_reference(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(
            sorted(outputs),
            ["titan_expert_out_grad", "titan_out", "titan_scores_grad"],
        )
        for name in outputs:
            with self.subTest(output=name):
                self.assertLess(_rel_l2(outputs[name], reference[name]), 2e-2)

    def test_the_arm_applies_the_probabilities_and_megatron_does_not(self) -> None:
        """The reason for the missing cross-engine row, measured on tensors.

        If titan's combine ever stopped scoring, this arm's output would move
        onto megatron's truth and the scenario's whole account would be wrong
        while every one of its own gates still passed.
        """
        inputs = _inputs()
        arm = build_moe_combine_titan(TINY, TINY_WORKLOAD, inputs)
        reference = moe_combine_reference(TINY, TINY_WORKLOAD, inputs)
        combined = arm.correctness_outputs()["titan_out"]
        self.assertLess(_rel_l2(combined, reference["titan_out"]), 2e-2)
        self.assertGreater(_rel_l2(combined, reference["mcore_out"]), 0.1)

    def test_the_score_gradient_exists_and_is_reset_between_calls(self) -> None:
        """The multiply is inside the timed region, and this is what says so."""
        inputs = _inputs()
        arm = build_moe_combine_titan(TINY, TINY_WORKLOAD, inputs)
        first = arm.correctness_outputs()["titan_scores_grad"].clone()
        self.assertTrue(torch.isfinite(first).all())
        self.assertGreater(first.abs().max().item(), 0.0)
        for _ in range(3):
            arm.calls["forward_backward"]()
        self.assertTrue(
            torch.equal(first, arm.correctness_outputs()["titan_scores_grad"])
        )


class GuardTests(unittest.TestCase):
    """``_assert_mcore_combine`` raises. Each case is a way it must."""

    def _run(self, dispatcher, inputs):
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        _assert_mcore_combine(
            dispatcher.combine,
            "stand_in",
            inputs,
            dispatcher,
            dispatcher.hidden_shape,
            tokens,
            TINY.dim,
        )

    def test_a_faithful_combine_passes(self) -> None:
        """Negative control: without it every assertion below is vacuous."""
        inputs = _inputs()
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        self._run(
            _StandInDispatcher(inputs.row_token_N, (tokens, 1, TINY.dim)), inputs
        )

    def test_an_identity_combine_is_refused(self) -> None:
        """The failure the plan names: it would read as a spectacular win."""
        inputs = _inputs()
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        dispatcher = _StandInDispatcher(
            inputs.row_token_N, (tokens, 1, TINY.dim), unpermute=False
        )
        with self.assertRaisesRegex(RuntimeError, "unpermute did not run"):
            self._run(dispatcher, inputs)

    def test_a_combine_that_restores_the_wrong_order_is_refused(self) -> None:
        """Right shape, right magnitude, wrong tokens.

        No correctness gate would catch it either, because each engine is
        gated against its own truth and the arm's own dispatch produced the
        order it is inverting.
        """
        inputs = _inputs()
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        shuffled = inputs.row_token_N.flip(0)
        dispatcher = _StandInDispatcher(shuffled, (tokens, 1, TINY.dim))
        with self.assertRaisesRegex(RuntimeError, "not combining the rows"):
            self._run(dispatcher, inputs)

    def test_a_token_combine_that_is_not_inert_is_refused(self) -> None:
        """At tp_size 1 and ep_size 1 the collective is documented inert.

        A phase that moved data there would mean the process group is larger
        than this harness supports, or that the class started doing local work
        the scenario has not accounted for.
        """
        inputs = _inputs()
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        dispatcher = _StandInDispatcher(
            inputs.row_token_N, (tokens, 1, TINY.dim), inert=False
        )
        with self.assertRaisesRegex(RuntimeError, "token_combine changed its input"):
            self._run(dispatcher, inputs)

    def test_the_guard_threshold_is_the_gate_threshold(self) -> None:
        """A permutation that is wrong is wrong by about 100%, not by 3%."""
        self.assertEqual(COMBINE_GUARD_REL_L2, 2e-2)

    def test_the_canonical_combine_is_the_unweighted_scatter_add(self) -> None:
        inputs = _inputs()
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        expected = torch.zeros((tokens, TINY.dim), dtype=torch.float32).index_add(
            0, inputs.row_token_N, inputs.expert_out_ND.float()
        )
        self.assertTrue(
            torch.allclose(_canonical_combine(inputs, TINY.dim, tokens), expected)
        )

    def test_a_missing_gradient_is_named_rather_than_dereferenced(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "mcore_expert_out_grad"):
            _require_grads(
                "stand_in",
                {"mcore_out": torch.zeros(1), "mcore_expert_out_grad": None},
            )

    def test_a_complete_set_passes_through_unchanged(self) -> None:
        outputs = {"mcore_out": torch.zeros(1), "mcore_expert_out_grad": torch.ones(1)}
        self.assertIs(_require_grads("stand_in", outputs), outputs)


class PinnedSourceTests(unittest.TestCase):
    """The lines on the pinned revisions the arms depend on.

    Each is a fact one of the docstrings states. A submodule bump that moves
    one is exactly the event that has to re-open this scenario, so the failure
    is the signal rather than the nuisance.
    """

    def test_the_allgather_token_combine_is_still_guarded_by_the_world_size(
        self,
    ) -> None:
        """The trap the whole cut exists to avoid. At one rank it is nothing."""
        source = _source(TOKEN_DISPATCHER)
        body = source[source.index("class MoEAllGatherTokenDispatcher") :]
        body = body[: body.index("class MoEAlltoAllTokenDispatcher")]
        combine = body[body.index("    def token_combine(") :]
        combine = combine[: combine.index("    def combine_postprocess(")]
        self.assertIn("if self.tp_size > 1 or self.ep_size > 1:", combine)
        self.assertIn("return hidden_states", combine)

    def test_the_moe_layer_combine_is_only_that_method(self) -> None:
        """So cutting on ``MoELayer.combine`` would have measured a zero."""
        source = _source(MOE_LAYER)
        body = source[source.index("    def combine(self, output: torch.Tensor):") :]
        body = body[: body.index("    def postprocess(")]
        self.assertIn("output = self.token_dispatcher.token_combine(output)", body)
        self.assertNotIn("combine_preprocess", body)
        self.assertNotIn("combine_postprocess", body)

    def test_the_work_sits_in_the_neighbouring_phases(self) -> None:
        """``combine_preprocess`` unpermutes under allgather; the layer calls
        it from ``routed_experts_compute`` and ``combine_postprocess`` from
        ``postprocess``. All three are the cut."""
        source = _source(MOE_LAYER)
        self.assertIn(
            "output = self.token_dispatcher.combine_preprocess(expert_output)", source
        )
        self.assertIn(
            "output = self.token_dispatcher.combine_postprocess(output)", source
        )

    def test_the_internal_api_boundary_is_the_one_we_avoid(self) -> None:
        """The cut is on the phase API because this method is not public."""
        source = _source(MOE_LAYER)
        self.assertIn(
            "@internal_api\n    def routed_experts_compute(", source
        )

    def test_each_dispatcher_unpermutes_in_its_own_phase(self) -> None:
        """Why the cut names all three phases instead of one."""
        source = _source(TOKEN_DISPATCHER)
        allgather = source[source.index("class MoEAllGatherTokenDispatcher") :]
        allgather = allgather[: allgather.index("class MoEAlltoAllTokenDispatcher")]
        preprocess = allgather[allgather.index("    def combine_preprocess(") :]
        preprocess = preprocess[: preprocess.index("    def token_combine(")]
        self.assertIn("unpermuted_local_hidden = unpermute(", preprocess)

        alltoall = source[source.index("class MoEAlltoAllTokenDispatcher") :]
        alltoall = alltoall[: alltoall.index("class _DispatchManager")]
        postprocess = alltoall[alltoall.index("    def combine_postprocess(") :]
        postprocess = postprocess[: postprocess.index("    def _maybe_update_cuda")]
        self.assertIn("output = unpermute(", postprocess)
        self.assertIn("sort_chunks_by_idxs(", alltoall)

    def test_the_permutation_fusion_flag_is_read_at_the_unpermute_call(self) -> None:
        """The ``no_permute_fusion`` delta, delivered by the config alone."""
        source = _source(TOKEN_DISPATCHER)
        self.assertGreaterEqual(source.count("fused=self.config.moe_permute_fusion"), 2)

    def test_unpermute_still_branches_on_fused_and_falls_to_scatter_add(self) -> None:
        source = _source(MOE_UTILS)
        body = source[source.index("def unpermute(") :]
        body = body[: body.index("def sort_chunks_by_idxs(")]
        self.assertIn("return fused_unpermute(", body)
        self.assertIn("output_tokens.scatter_add_(", body)

    def test_the_unfused_permute_still_orders_rows_expert_then_token(self) -> None:
        """The canonical row order this module derives from the routing map."""
        source = _source(MOE_UTILS)
        body = source[source.index("def permute(") :]
        body = body[: body.index("def unpermute(")]
        self.assertIn("routing_map = routing_map.bool().T.contiguous()", body)
        self.assertIn(
            "flat_sorted = routing_map.reshape(-1).argsort(descending=True, stable=True)",
            body,
        )
        self.assertIn("sorted_indices = flat_sorted % num_tokens", body)

    def test_the_unfused_chunk_sort_would_read_the_device(self) -> None:
        """Why ``dispatcher_alltoall`` keeps the permutation fusion on."""
        source = _source(MOE_UTILS)
        body = source[source.index("def sort_chunks_by_idxs(") :]
        self.assertIn("input = torch.split(input, split_sizes.tolist(), dim=0)", body)

    def test_the_all_to_all_collective_is_inert_at_one_rank(self) -> None:
        """The COLLECTIVE, which is not the same claim as the CLASS."""
        source = _source(MAPPINGS)
        body = source[source.index("class _AllToAll(torch.autograd.Function):") :]
        self.assertIn("world_size = group.size()", body)
        self.assertIn("if world_size == 1:\n            return input", body)

    def test_the_dispatcher_class_comes_from_the_config_field(self) -> None:
        """The ``dispatcher_alltoall`` delta, read once in ``MoELayer``."""
        source = _source(MOE_LAYER)
        self.assertIn('if config.moe_token_dispatcher_type == "allgather":', source)
        self.assertIn("self.token_dispatcher = MoEAllGatherTokenDispatcher(", source)
        self.assertIn('elif config.moe_token_dispatcher_type == "alltoall":', source)
        self.assertIn("self.token_dispatcher = MoEAlltoAllTokenDispatcher(", source)

    def test_no_combine_phase_carries_the_jit_fuser(self) -> None:
        """Which is what makes all three mcore arms eager rather than compiled.

        The single decorator in the module sits on the flex dispatcher, a class
        that asserts ``tp_size * ep_size > 1`` and is never built here.
        """
        source = _source(TOKEN_DISPATCHER)
        self.assertEqual(source.count("@jit_fuser"), 1)
        flex = source[source.index("class MoEFlexTokenDispatcher") :]
        self.assertIn("@jit_fuser\n    def dispatch_preprocess(", flex)

    def test_titan_still_calls_combine_where_the_scenario_cuts(self) -> None:
        source = _source(TITAN_MOE)
        self.assertIn("out_TD = self.token_dispatcher.combine(", source)
        # The trailing reshape sits AFTER the call, so it is outside the cut.
        body = source[source.index("out_TD = self.token_dispatcher.combine(") :]
        self.assertIn("return out_TD.view(B, -1, D)", body[: body.index("def ", 10)])

    def test_titans_combine_scores_before_it_scatters(self) -> None:
        """The multiply megatron has already done, on the other side of 11/12."""
        source = _source(TITAN_DISPATCHER)
        body = source[source.index("class LocalTokenDispatcher(") :]
        body = body[: body.index("class BaseEPTokenDispatcher(")]
        self.assertIn('"""Score and scatter_add routed expert outputs.', body)
        self.assertIn("* metadata.topk_scores_experts_sorted_N.reshape(-1, 1)", body)
        self.assertIn("out_TD = deterministic_scatter_add(", body)

    def test_titans_expert_sorted_order_is_a_stable_argsort(self) -> None:
        source = _source(TITAN_DISPATCHER)
        self.assertIn(
            "token_indices_experts_sorted_N = torch.argsort(\n"
            "            topk_expert_ids_TK.view(-1), stable=True\n"
            "        )",
            source,
        )

    def test_the_production_dispatcher_delegates_to_the_local_combine(self) -> None:
        """Why the arm builds ``AllToAllTokenDispatcher`` and not the base."""
        source = _source(TITAN_DISPATCHER)
        body = source[source.index("class AllToAllTokenDispatcher(") :]
        combine = body[body.index("    def combine(") :]
        self.assertIn("if self.ep_mesh is None:", combine)
        self.assertIn("return LocalTokenDispatcher.combine(", combine)

    def test_the_standard_comm_backend_builds_that_class(self) -> None:
        source = _source(TITAN_CONFIG_UTILS)
        self.assertIn('elif comm_backend == "standard":', source)
        body = source[source.index('elif comm_backend == "standard":') :]
        self.assertIn("return AllToAllTokenDispatcher.Config(", body)

    def test_our_config_asks_for_that_backend(self) -> None:
        """So the arm holds the class this model actually builds."""
        registry = (
            Path(__file__).resolve().parent.parent
            / "benchmarks/models/piper_qwen3/config_registry.py"
        )
        self.assertIn(
            f'moe_comm_backend="{TITAN_COMM_BACKEND}"', registry.read_text()
        )


class RegisteredShapeTests(unittest.TestCase):
    def test_the_inputs_build_at_every_registered_model_size(self) -> None:
        """``--model-size`` is single-valued but not fixed, so both must work."""
        for name in ("normal", "huge"):
            with self.subTest(size=name):
                shape = shape_by_name(name)
                workload = KernelWorkload(batch=1, seq_len=2)
                generator = torch.Generator(device="cpu")
                generator.manual_seed(0)
                inputs = moe_combine_inputs(
                    shape, workload, torch.device("cpu"), generator
                )
                self.assertEqual(inputs.x_BLD.shape[-1], shape.dim)
                self.assertEqual(
                    inputs.expert_out_ND.shape[0], 2 * shape.top_k
                )
                self.assertEqual(
                    inputs.tokens_per_expert_E.numel(), shape.num_experts
                )


class ArmNameTests(unittest.TestCase):
    """**Not a registry test, and it cannot be one yet.**

    The ``moe_combine`` scenario is not in ``benchmarks/kernel/registry.py``:
    it lands as a declaration fragment that a merge agent pastes, because
    several scenarios are written in parallel and one file cannot take four
    concurrent edits. So these assertions compare the module's own constants
    against the names the fragment spells. **After the merge, replace them
    with assertions against** ``KERNEL_SCENARIOS["moe_combine"]``, which is
    the only version of this test that can catch a divergence.
    """

    def test_the_arm_names_are_the_declared_roster(self) -> None:
        self.assertEqual(
            [
                COPY_FLOOR_ARM,
                MCORE_BASE_ARM,
                MCORE_NO_PERMUTE_FUSION_ARM,
                MCORE_ALLTOALL_ARM,
                TITAN_ARM,
            ],
            [
                "copy_floor",
                "mcore/base",
                "mcore/no_permute_fusion",
                "mcore/dispatcher_alltoall",
                "titan",
            ],
        )

    def test_every_arm_name_becomes_a_distinct_fragment_file(self) -> None:
        """A slash is a directory separator, and no writer creates one."""
        names = (
            COPY_FLOOR_ARM,
            MCORE_BASE_ARM,
            MCORE_NO_PERMUTE_FUSION_ARM,
            MCORE_ALLTOALL_ARM,
            TITAN_ARM,
        )
        stems = [fragment_stem(name) for name in names]
        self.assertEqual(len(set(stems)), len(names))
        for stem in stems:
            self.assertNotIn("/", stem)


if __name__ == "__main__":
    unittest.main()
