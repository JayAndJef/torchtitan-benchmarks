"""CPU tests for the ``dispatch_permute`` cross-engine kernel scenario.

Everything here runs without a GPU. The scenario makes three claims a CPU can
check, and they are the three a wrong number would come from.

* **Both engines permute into the same order.** TorchTitan sorts flattened
  expert ids with a stable *ascending* argsort; megatron sorts a transposed,
  flattened routing map with a stable *descending* one. The two constructions
  look nothing alike. ``PermutationOrderTests`` transcribes both from the
  pinned sources and shows they agree with a third derivation that is neither
  engine's.
* **A norm-based gate cannot police a permutation.**
  ``PermutationOrderTests`` computes the relative L2 error of a single swapped
  pair at the default workload and shows it lands below the gate every
  neighbouring scenario uses. That arithmetic is why the permutation gate is
  bitwise.
* **The megatron side is the identity one phase away from the cut.**
  ``PinnedSourceTests`` reads the guard at ``token_dispatcher.py:282`` and the
  bypass at ``mappings.py:433-435`` out of the pinned tree, and
  ``GuardTests`` shows ``_require_a_real_permutation`` refuses an arm that
  returns its input.

**No test here imports megatron**, and that is a cost decision rather than an
oversight. ``megatron/core/__init__.py`` imports TransformerEngine, so the
cheapest possible ``from megatron.core...`` costs about 9 s and pulls a GPU
library into a suite that otherwise runs in seconds. The pinned sources are
read as **text** instead, which is what a submodule bump has to invalidate
anyway; ``tests/test_lm_head_losses.py`` and ``tests/test_kernel_moe_residual.
py`` do the same.

What that leaves untested on CPU is ``_build_dispatch_permute_mcore`` itself,
whose first statement is ``initialize_megatron_single_rank``. It runs on every
mcore build, so a wiring error raises there rather than reaching a number.
"""

import functools
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.execution.paths import TITAN_DIR
from benchmarks.kernel.operations.dispatch_permute import (
    _canonical_counts,
    _require_a_real_permutation,
    _require_gradients,
    _require_titan_dispatch_contract,
    _titan_dispatch,
    ALLGATHER_DISPATCHER,
    ALLTOALL_DISPATCHER,
    build_dispatch_permute_copy_floor,
    build_dispatch_permute_titan,
    COPY_FLOOR_ARM,
    dispatch_permute_inputs,
    dispatch_permute_reference,
    DispatchPermuteInputs,
    DISPATCHER_ALLTOALL,
    MCORE_BASE_ARM,
    MCORE_DISPATCHER_ALLTOALL_ARM,
    MCORE_NO_PERMUTE_FUSION_ARM,
    NO_PERMUTE_FUSION,
    titan_dispatcher,
    TITAN_ARM,
)
from benchmarks.kernel.schema import fragment_stem, KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE
from benchmarks.models.piper_qwen3.megatron_bootstrap import megatron_dir
from benchmarks.models.piper_qwen3.shape import PIPER_SHAPES, PiperShape

# Small enough to build in milliseconds, and wide enough that a lost row or a
# transposed view shows up rather than cancelling. 32 tokens and 4 experts keep
# the split exact, which is the scenario's own precondition.
TINY = PiperShape(name="tiny", dim=256, n_layers=2, vocab_size=64)
TINY_WORKLOAD = KernelWorkload(batch=2, seq_len=16)

MEGATRON = megatron_dir()
TOKEN_DISPATCHER = MEGATRON / "megatron/core/transformer/moe/token_dispatcher.py"
MOE_UTILS = MEGATRON / "megatron/core/transformer/moe/moe_utils.py"
MOE_LAYER = MEGATRON / "megatron/core/transformer/moe/moe_layer.py"
MAPPINGS = MEGATRON / "megatron/core/tensor_parallel/mappings.py"
TRANSFORMER_LAYER = MEGATRON / "megatron/core/transformer/transformer_layer.py"
PACKED_SEQ_PARAMS = MEGATRON / "megatron/core/packed_seq_params.py"
TITAN_MOE = TITAN_DIR / "torchtitan/models/common/moe.py"
TITAN_DISPATCHER = TITAN_DIR / "torchtitan/models/common/token_dispatcher.py"


def _inputs(shape=TINY, workload=TINY_WORKLOAD, seed=0) -> DispatchPermuteInputs:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return dispatch_permute_inputs(shape, workload, torch.device("cpu"), generator)


@functools.lru_cache(maxsize=1)
def _shared_titan_build():
    """One compiled titan build for the whole module, and why one is enough.

    ``build_dispatch_permute_titan`` applies ``torch.compile(fullgraph=True)``,
    and that single compilation is effectively the whole cost of this file:
    about nine seconds, against about thirty milliseconds for every other test
    here together. Two classes need the built arm. A build in each of them
    makes whichever class runs first pay the nine seconds, so the alphabetical
    order of the class names decides where the cost lands -- and a third caller
    would put it somewhere else again.

    **Sharing one build between tests is safe, and that is a property of this
    arm rather than a hope.** The arm holds three independent leaf pairs -- one
    for ``forward``, one for ``forward_backward`` and one for the gate -- so no
    mode's gradient state can reach another mode's. Both ``forward_backward``
    and ``correctness_outputs`` clear their own leaves before they run, which
    ``test_a_repeated_round_trip_does_not_accumulate_a_gradient`` pins. And the
    inputs stay read-only, because every closure clones its leaves, which
    ``test_no_arm_makes_the_shared_inputs_require_grad`` pins.

    Returns the inputs beside the arm. They are the inputs the arm closed over,
    so a caller cannot gate the arm against a second draw by accident.
    """
    inputs = _inputs()
    return inputs, build_dispatch_permute_titan(TINY, TINY_WORKLOAD, inputs)


def _rel_l2(value: torch.Tensor, truth: torch.Tensor) -> float:
    delta = (value.double() - truth.double()).norm()
    return (delta / truth.double().norm()).item()


def _source(path: Path) -> str:
    if not path.exists():
        raise unittest.SkipTest(f"{path} is absent; the submodule is not checked out")
    return path.read_text()


def _squeezed(text: str) -> str:
    """The same source with every run of whitespace collapsed to one space.

    Line breaks inside a call are formatting, and a reformat must not fail a
    test whose subject is the code's behaviour.
    """
    return " ".join(text.split())


class ProfileTests(unittest.TestCase):
    """The two deltas, and that each is a delta rather than a restatement."""

    def test_the_base_profile_declares_the_fusion_on_and_allgather(self) -> None:
        self.assertIs(BASE.config_overrides["moe_permute_fusion"], True)
        self.assertEqual(
            BASE.config_overrides["moe_token_dispatcher_type"], "allgather"
        )

    def test_the_fusion_variant_turns_exactly_one_field_off(self) -> None:
        changed = {
            key
            for key, value in NO_PERMUTE_FUSION.config_overrides.items()
            if BASE.config_overrides.get(key) != value
        }
        self.assertEqual(changed, {"moe_permute_fusion"})
        self.assertIs(
            NO_PERMUTE_FUSION.config_overrides["moe_permute_fusion"], False
        )

    def test_the_dispatcher_variant_changes_exactly_one_field(self) -> None:
        changed = {
            key
            for key, value in DISPATCHER_ALLTOALL.config_overrides.items()
            if BASE.config_overrides.get(key) != value
        }
        self.assertEqual(changed, {"moe_token_dispatcher_type"})
        self.assertEqual(
            DISPATCHER_ALLTOALL.config_overrides["moe_token_dispatcher_type"],
            "alltoall",
        )

    def test_the_dispatcher_variant_keeps_the_fusion_on(self) -> None:
        """The two deltas are independent, and only one may move per arm.

        ``mcore/dispatcher_alltoall`` declares ``permute_fusion=True`` to its
        contract guard, so a profile that also turned the fusion off would fail
        the build -- but it would fail it in a GPU worker, and this catches it
        on CPU.
        """
        self.assertIs(
            DISPATCHER_ALLTOALL.config_overrides["moe_permute_fusion"], True
        )

    def test_both_variants_are_recordable_provenance(self) -> None:
        for profile in (NO_PERMUTE_FUSION, DISPATCHER_ALLTOALL):
            with self.subTest(profile=profile.name):
                record = profile.describe()
                self.assertEqual(record["name"], profile.name)
                self.assertTrue(record["description"])
                self.assertIn("moe_permute_fusion", record["config_overrides"])

    def test_the_alltoall_description_says_the_collective_is_inert(self) -> None:
        """The caption the plan requires, on the object the manifest records.

        A reader who sees this arm's row without it takes a local permute and
        sync comparison for a communication result.
        """
        text = DISPATCHER_ALLTOALL.description.lower()
        self.assertIn("inert", text)
        self.assertIn("world_size=1", text)
        self.assertIn("never communication", text)


class InputsTests(unittest.TestCase):
    def test_the_inputs_carry_both_engines_routing_forms(self) -> None:
        inputs = _inputs()
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        self.assertEqual(
            tuple(inputs.x_BLD.shape),
            (TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len, TINY.dim),
        )
        self.assertEqual(inputs.x_BLD.dtype, torch.bfloat16)
        self.assertEqual(
            tuple(inputs.topk_expert_ids_TK.shape), (tokens, TINY.top_k)
        )
        self.assertEqual(
            tuple(inputs.probs_TE.shape), (tokens, TINY.num_experts)
        )
        self.assertEqual(inputs.probs_TE.dtype, torch.float32)
        self.assertEqual(inputs.routing_map_TE.dtype, torch.bool)

    def test_the_two_probability_forms_hold_the_same_values(self) -> None:
        """The dense ``[T, E]`` megatron reads is the sparse ``[T, K]`` titan
        reads, at the routed positions and zero elsewhere."""
        inputs = _inputs()
        gathered = inputs.probs_TE.gather(1, inputs.topk_expert_ids_TK)
        self.assertTrue(torch.equal(gathered, inputs.topk_scores_TK))
        self.assertTrue(
            torch.equal(
                inputs.probs_TE == 0.0,
                ~inputs.routing_map_TE,
            )
        )

    def test_the_routing_map_agrees_with_the_expert_ids(self) -> None:
        inputs = _inputs()
        rebuilt = torch.zeros_like(inputs.routing_map_TE).scatter_(
            1, inputs.topk_expert_ids_TK, True
        )
        self.assertTrue(torch.equal(rebuilt, inputs.routing_map_TE))

    def test_every_token_reaches_top_k_distinct_experts(self) -> None:
        inputs = _inputs()
        per_token = inputs.routing_map_TE.sum(dim=1)
        self.assertTrue(bool((per_token == TINY.top_k).all()))

    def test_the_split_is_exactly_balanced(self) -> None:
        inputs = _inputs()
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        expected = tokens * TINY.top_k // TINY.num_experts
        self.assertEqual(
            inputs.tokens_per_expert_E.tolist(), [expected] * TINY.num_experts
        )

    def test_the_expected_order_is_expert_major_and_token_ascending(self) -> None:
        inputs = _inputs()
        experts = inputs.expected_slot_expert_N
        tokens_per = inputs.tokens_per_expert_E.tolist()
        self.assertTrue(bool((experts.diff() >= 0).all()))
        start = 0
        for expert, count in enumerate(tokens_per):
            block = inputs.expected_slot_token_N[start : start + count]
            self.assertTrue(bool((experts[start : start + count] == expert).all()))
            self.assertTrue(bool((block.diff() > 0).all()))
            start += count

    def test_the_expected_order_is_not_the_identity(self) -> None:
        """The property that makes an identity arm detectable at all."""
        inputs = _inputs()
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        self.assertFalse(
            torch.equal(
                inputs.expected_slot_token_N[:tokens], torch.arange(tokens)
            )
        )

    def test_the_gradients_match_the_two_permuted_outputs(self) -> None:
        inputs = _inputs()
        slots = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len * TINY.top_k
        self.assertEqual(tuple(inputs.grad_permuted_ND.shape), (slots, TINY.dim))
        self.assertEqual(inputs.grad_permuted_ND.dtype, torch.bfloat16)
        self.assertEqual(tuple(inputs.grad_probs_N.shape), (slots,))
        self.assertEqual(inputs.grad_probs_N.dtype, torch.float32)

    def test_copy_bytes_is_one_read_and_one_write_of_the_permuted_buffer(
        self,
    ) -> None:
        inputs = _inputs()
        slots = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len * TINY.top_k
        self.assertEqual(inputs.copy_bytes, 2 * slots * TINY.dim * 2)

    def test_one_seed_rebuilds_the_inputs_bit_identically(self) -> None:
        first, second = _inputs(seed=3), _inputs(seed=3)
        for name in (
            "x_BLD",
            "topk_expert_ids_TK",
            "topk_scores_TK",
            "probs_TE",
            "routing_map_TE",
            "tokens_per_expert_E",
            "expected_slot_token_N",
            "grad_permuted_ND",
            "grad_probs_N",
        ):
            with self.subTest(tensor=name):
                self.assertTrue(
                    torch.equal(getattr(first, name), getattr(second, name))
                )

    def test_a_different_seed_draws_a_different_routing(self) -> None:
        self.assertFalse(
            torch.equal(
                _inputs(seed=0).topk_expert_ids_TK,
                _inputs(seed=7).topk_expert_ids_TK,
            )
        )

    def test_unbalanced_slots_fail_loudly_with_named_numbers(self) -> None:
        shape = PiperShape(name="odd", dim=256, n_layers=2, vocab_size=64)
        workload = KernelWorkload(batch=1, seq_len=3)
        with self.assertRaises(ValueError) as raised:
            _inputs(shape=shape, workload=workload)
        message = str(raised.exception)
        self.assertIn("6 routed slots", message)
        self.assertIn("4 experts", message)
        self.assertIn("remainder is 2", message)

    def test_unbalanced_tokens_fail_loudly_with_named_numbers(self) -> None:
        """Stronger than the scenario-level invariant, and separately named.

        ``batch=1, seq_len=6`` gives 12 slots, which divide by 4 experts, so
        ``routing_divides_evenly`` accepts it. The token count does not divide,
        and this construction needs it to.
        """
        shape = PiperShape(name="odd", dim=256, n_layers=2, vocab_size=64)
        workload = KernelWorkload(batch=1, seq_len=6)
        with self.assertRaises(ValueError) as raised:
            _inputs(shape=shape, workload=workload)
        message = str(raised.exception)
        self.assertIn("6 tokens", message)
        self.assertIn("remainder is 2", message)

    def test_top_k_above_the_expert_count_fails(self) -> None:
        shape = PiperShape(
            name="narrow", dim=256, n_layers=2, vocab_size=64, num_experts=2, top_k=4
        )
        with self.assertRaises(ValueError) as raised:
            _inputs(shape=shape)
        self.assertIn("top_k 4 exceeds num_experts 2", str(raised.exception))


class PermutationOrderTests(unittest.TestCase):
    """The scenario's central claim, transcribed from both pinned sources.

    Neither transcription imports its engine. Each is the algorithm read off
    the file that ``PinnedSourceTests`` separately checks is still there, so a
    submodule bump that changes either one fails there and the transcription
    below stops being evidence.
    """

    def _titan_order(self, inputs: DispatchPermuteInputs) -> torch.Tensor:
        """``LocalTokenDispatcher._local_reorder``, ``token_dispatcher.py:93-99``."""
        flat = torch.argsort(inputs.topk_expert_ids_TK.reshape(-1), stable=True)
        return flat // TINY.top_k

    def _mcore_order(self, inputs: DispatchPermuteInputs) -> torch.Tensor:
        """``moe_utils.permute``, the torch path, ``moe_utils.py:467-475``."""
        tokens = inputs.routing_map_TE.shape[0]
        num_out_tokens = int(inputs.tokens_per_expert_E.sum())
        routing_map = inputs.routing_map_TE.bool().T.contiguous()
        flat_sorted = routing_map.reshape(-1).argsort(descending=True, stable=True)
        return flat_sorted[:num_out_tokens] % tokens

    def test_titan_sorts_into_the_expected_order(self) -> None:
        inputs = _inputs()
        self.assertTrue(
            torch.equal(self._titan_order(inputs), inputs.expected_slot_token_N)
        )

    def test_megatron_sorts_into_the_same_order(self) -> None:
        inputs = _inputs()
        self.assertTrue(
            torch.equal(self._mcore_order(inputs), inputs.expected_slot_token_N)
        )

    def test_the_two_engines_agree_at_several_seeds(self) -> None:
        """One agreement could be an artifact of one routing draw."""
        for seed in (0, 1, 2, 5, 11):
            with self.subTest(seed=seed):
                inputs = _inputs(seed=seed)
                self.assertTrue(
                    torch.equal(
                        self._titan_order(inputs), self._mcore_order(inputs)
                    )
                )

    def test_megatron_permutes_the_probabilities_into_the_same_order(self) -> None:
        """``token_dispatcher.py:328-330``, the by-hand probability permute."""
        inputs = _inputs()
        local_probs = inputs.probs_TE.T.contiguous().masked_select(
            inputs.routing_map_TE.T.contiguous()
        )
        expected = inputs.probs_TE[
            inputs.expected_slot_token_N, inputs.expected_slot_expert_N
        ]
        self.assertTrue(torch.equal(local_probs, expected))

    def test_a_norm_gate_sees_one_misplaced_row_only_at_a_small_workload(
        self,
    ) -> None:
        """The arithmetic that makes the permutation gate bitwise.

        Swapping one pair of rows changes the relative L2 error by about
        ``2 / sqrt(N)``, not ``sqrt(2 / N)``. At the default workload
        (batch 4, sequence 1024, top_k 2, so ``N = 8192``) that is 2.2e-2,
        just **above** the 2e-2 gate. Doubling the batch halves ``sqrt(N)``'s
        growth rate against the gate and drops it to 1.6e-2, **below** it.

        So a tolerance gate catches a misplaced row at one batch size and
        misses it at the next. That workload dependence is the argument for
        the bitwise gate, and it is what this test pins. The row width is the
        model dim, not a narrow stand-in: a narrow row does not shrink the
        ratio, it widens its spread, and at width 8 this measurement lands
        below the gate for about half of all seeds. At the model dim both
        assertions below hold for every one of 60 seeds measured: batch 4
        spans 0.0211 to 0.0232, and batch 8 spans 0.0150 to 0.0164.
        """
        shape = PIPER_SHAPES["normal"]
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        measured = {}
        for batch in (4, 8):
            slots = batch * 1024 * shape.top_k
            truth = torch.randn((slots, shape.dim), generator=generator)
            swapped = truth.clone()
            swapped[[0, 1]] = swapped[[1, 0]]
            measured[batch] = _rel_l2(swapped, truth)
            self.assertFalse(torch.equal(swapped, truth))
            # The closed form the docstring states. One swapped pair is a
            # single draw, not an average, so it deviates by a few percent;
            # 10% covers the worst of 60 seeds measured at this width.
            closed_form = 2 / math.sqrt(slots)
            self.assertLess(abs(measured[batch] - closed_form) / closed_form, 0.10)
        self.assertGreater(measured[4], 2e-2)
        self.assertLess(measured[8], 2e-2)


class ReferenceTests(unittest.TestCase):
    def test_the_forward_reference_is_an_exact_gather(self) -> None:
        inputs = _inputs()
        reference = dispatch_permute_reference(TINY, TINY_WORKLOAD, inputs)
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        x_TD = inputs.x_BLD.reshape(tokens, TINY.dim)
        self.assertEqual(reference["permuted_tokens"].dtype, torch.bfloat16)
        self.assertTrue(
            torch.equal(
                reference["permuted_tokens"], x_TD[inputs.expected_slot_token_N]
            )
        )
        self.assertEqual(reference["permuted_probs"].dtype, torch.float32)

    def test_the_gradient_reference_is_fp64(self) -> None:
        inputs = _inputs()
        reference = dispatch_permute_reference(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(reference["x_grad"].dtype, torch.float64)
        self.assertEqual(reference["probs_grad"].dtype, torch.float64)

    def test_the_token_gradient_accumulates_every_slot_of_a_token(self) -> None:
        """A token routed to ``top_k`` experts receives ``top_k`` slots back."""
        inputs = _inputs()
        reference = dispatch_permute_reference(TINY, TINY_WORKLOAD, inputs)
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        by_hand = torch.zeros((tokens, TINY.dim), dtype=torch.float64)
        for slot, token in enumerate(inputs.expected_slot_token_N.tolist()):
            by_hand[token] += inputs.grad_permuted_ND[slot].double()
        self.assertTrue(
            torch.allclose(
                reference["x_grad"].reshape(tokens, TINY.dim), by_hand
            )
        )

    def test_the_probability_gradient_is_dense_and_zero_off_route(self) -> None:
        inputs = _inputs()
        reference = dispatch_permute_reference(TINY, TINY_WORKLOAD, inputs)
        grad = reference["probs_grad"]
        self.assertEqual(tuple(grad.shape), tuple(inputs.probs_TE.shape))
        self.assertTrue(bool((grad[~inputs.routing_map_TE] == 0.0).all()))

    def test_the_reference_counts_are_the_shared_ones(self) -> None:
        inputs = _inputs()
        reference = dispatch_permute_reference(TINY, TINY_WORKLOAD, inputs)
        self.assertTrue(
            torch.equal(reference["tokens_per_expert"], inputs.tokens_per_expert_E)
        )


class TitanArmTests(unittest.TestCase):
    """One build for every test here, because the build compiles.

    The build comes from ``_shared_titan_build``, which explains why one build
    serves the whole module and why the tests cannot see each other's state.
    The reference is derived per class, because it is arithmetic on the inputs
    and costs nothing.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.inputs, cls.arm = _shared_titan_build()
        cls.reference = dispatch_permute_reference(
            TINY, TINY_WORKLOAD, cls.inputs
        )

    def test_the_production_dispatcher_is_the_alltoall_class_unwired(
        self,
    ) -> None:
        """The config builds the EP class; one rank takes the local branch."""
        dispatcher = titan_dispatcher(TINY)
        self.assertEqual(type(dispatcher).__name__, "AllToAllTokenDispatcher")
        self.assertIsNone(dispatcher.ep_mesh)
        self.assertEqual(dispatcher.sp_size, 1)
        self.assertEqual(dispatcher.num_experts, TINY.num_experts)
        self.assertEqual(dispatcher.top_k, TINY.top_k)

    def test_the_contract_guard_refuses_a_wired_expert_mesh(self) -> None:
        dispatcher = titan_dispatcher(TINY)
        dispatcher.ep_mesh = object()
        with self.assertRaises(RuntimeError) as raised:
            _require_titan_dispatch_contract(dispatcher, TINY)
        self.assertIn("ep_mesh", str(raised.exception))

    def test_the_contract_guard_refuses_a_sequence_parallel_dispatcher(
        self,
    ) -> None:
        dispatcher = titan_dispatcher(TINY)
        dispatcher.sp_size = 2
        with self.assertRaises(RuntimeError) as raised:
            _require_titan_dispatch_contract(dispatcher, TINY)
        self.assertIn("sp_size", str(raised.exception))

    def test_the_contract_guard_refuses_a_foreign_geometry(self) -> None:
        dispatcher = titan_dispatcher(TINY)
        dispatcher.top_k = TINY.top_k + 1
        with self.assertRaises(RuntimeError) as raised:
            _require_titan_dispatch_contract(dispatcher, TINY)
        self.assertIn("top_k", str(raised.exception))

    def test_the_uncompiled_call_returns_the_megatron_triple(self) -> None:
        """Three tensors, in megatron's order, so the gate compares like rows."""
        inputs = self.inputs
        batch, seq = TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len
        permuted, counts, probs = _titan_dispatch(
            titan_dispatcher(TINY),
            inputs.x_BLD,
            inputs.topk_scores_TK.view(batch, seq, TINY.top_k),
            inputs.topk_expert_ids_TK.view(batch, seq, TINY.top_k),
            inputs.probs_TE.view(batch, seq, TINY.num_experts),
        )
        self.assertEqual(
            tuple(permuted.shape),
            (batch * seq * TINY.top_k, TINY.dim),
        )
        self.assertEqual(tuple(counts.shape), (TINY.num_experts,))
        self.assertEqual(probs.dim(), 1)

    def test_the_arm_declares_forward_and_forward_backward_only(self) -> None:
        self.assertEqual(self.arm.name, TITAN_ARM)
        self.assertEqual(sorted(self.arm.calls), ["forward", "forward_backward"])

    def test_the_arm_carries_no_byte_count(self) -> None:
        """Only the floor does. See ``build_dispatch_permute_copy_floor``."""
        self.assertIsNone(self.arm.bytes_moved)

    def test_the_permutation_matches_the_reference_bitwise(self) -> None:
        outputs = self.arm.correctness_outputs()
        for name in ("permuted_tokens", "permuted_probs", "tokens_per_expert"):
            with self.subTest(output=name):
                self.assertTrue(
                    torch.equal(outputs[name], self.reference[name])
                )

    def test_both_gradients_land_inside_the_declared_tolerance(self) -> None:
        outputs = self.arm.correctness_outputs()
        for name in ("x_grad", "probs_grad"):
            with self.subTest(output=name):
                self.assertLess(_rel_l2(outputs[name], self.reference[name]), 2e-2)

    def test_the_probability_gradient_is_canonicalized_to_the_dense_form(
        self,
    ) -> None:
        """Titan's leaf is ``[T, K]``; the gate's canonical form is ``[T, E]``."""
        grad = self.arm.correctness_outputs()["probs_grad"]
        self.assertEqual(tuple(grad.shape), tuple(self.inputs.probs_TE.shape))
        self.assertTrue(bool((grad[~self.inputs.routing_map_TE] == 0.0).all()))

    def test_a_repeated_round_trip_does_not_accumulate_a_gradient(self) -> None:
        first = self.arm.correctness_outputs()["x_grad"].clone()
        self.arm.calls["forward_backward"]()
        second = self.arm.correctness_outputs()["x_grad"]
        self.assertTrue(torch.equal(first, second))

    def test_no_arm_makes_the_shared_inputs_require_grad(self) -> None:
        """Every closure clones its leaves, so one arm cannot poison another."""
        self.assertIsNotNone(self.arm)
        self.assertFalse(self.inputs.x_BLD.requires_grad)
        self.assertFalse(self.inputs.topk_scores_TK.requires_grad)
        self.assertFalse(self.inputs.probs_TE.requires_grad)

    def test_the_forward_closure_builds_a_graph(self) -> None:
        permuted, _, probs = self.arm.calls["forward"]()
        self.assertIsNotNone(permuted.grad_fn)
        self.assertIsNotNone(probs.grad_fn)


class FloorTests(unittest.TestCase):
    def test_the_floor_declares_forward_alone_and_carries_the_traffic(
        self,
    ) -> None:
        inputs = _inputs()
        floor = build_dispatch_permute_copy_floor(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(floor.name, COPY_FLOOR_ARM)
        self.assertEqual(sorted(floor.calls), ["forward"])
        self.assertEqual(floor.bytes_moved, inputs.copy_bytes)

    def test_the_floor_copies_the_permuted_buffer_and_gates_nothing(self) -> None:
        inputs = _inputs()
        floor = build_dispatch_permute_copy_floor(TINY, TINY_WORKLOAD, inputs)
        self.assertIsNone(floor.calls["forward"]())
        self.assertEqual(floor.correctness_outputs(), {})

    def test_the_floor_moves_the_permuted_buffer_and_not_the_input(self) -> None:
        """A floor over ``[T, D]`` would understate the traffic by ``top_k``."""
        inputs = _inputs()
        floor = build_dispatch_permute_copy_floor(TINY, TINY_WORKLOAD, inputs)
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        self.assertEqual(
            floor.bytes_moved,
            2 * tokens * TINY.top_k * TINY.dim * inputs.x_BLD.element_size(),
        )


class GuardTests(unittest.TestCase):
    """``_require_a_real_permutation``: the guard the plan makes mandatory."""

    def setUp(self) -> None:
        self.inputs = _inputs()
        self.tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        self.x_TD = self.inputs.x_BLD.reshape(self.tokens, TINY.dim)
        self.order = self.inputs.expected_slot_token_N
        self.expert = self.inputs.expected_slot_expert_N

    def _honest(self):
        return (
            self.x_TD[self.order],
            self.inputs.tokens_per_expert_E,
            self.inputs.probs_TE[self.order, self.expert],
        )

    def _check(self, call) -> None:
        _require_a_real_permutation(
            call, self.inputs, TINY, TINY_WORKLOAD, "probe"
        )

    def test_an_honest_permutation_passes(self) -> None:
        self._check(self._honest)

    def test_an_identity_dispatch_is_refused(self) -> None:
        """The failure the whole guard exists for.

        At ``world_size=1`` ``token_dispatch`` returns its input, so a cut one
        phase away from ``permute`` measures nothing and reads as a win.
        """
        with self.assertRaises(RuntimeError) as raised:
            self._check(
                lambda: (
                    self.x_TD,
                    self.inputs.tokens_per_expert_E,
                    self.inputs.probs_TE[self.order, self.expert],
                )
            )
        message = str(raised.exception)
        self.assertIn("identity", message)
        self.assertIn("reads as a win", message)

    def test_a_permutation_that_returns_its_own_buffer_is_refused(self) -> None:
        """Covers ``top_k == 1``, where the row count alone cannot see it."""
        one_slot = PiperShape(
            name="single", dim=256, n_layers=2, vocab_size=64, top_k=1
        )
        inputs = _inputs(shape=one_slot)
        tokens = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len
        x_TD = inputs.x_BLD.reshape(tokens, one_slot.dim)
        with self.assertRaises(RuntimeError) as raised:
            _require_a_real_permutation(
                lambda: (
                    x_TD,
                    inputs.tokens_per_expert_E,
                    inputs.probs_TE[
                        inputs.expected_slot_token_N,
                        inputs.expected_slot_expert_N,
                    ],
                ),
                inputs,
                one_slot,
                TINY_WORKLOAD,
                "probe",
            )
        self.assertIn("its own input buffer", str(raised.exception))

    def test_a_swapped_pair_of_rows_is_refused(self) -> None:
        """The case a ``max_rel_l2`` gate would accept at the real workload."""
        permuted = self.x_TD[self.order].clone()
        permuted[[0, 1]] = permuted[[1, 0]]
        with self.assertRaises(RuntimeError) as raised:
            self._check(
                lambda: (
                    permuted,
                    self.inputs.tokens_per_expert_E,
                    self.inputs.probs_TE[self.order, self.expert],
                )
            )
        self.assertIn("wrong places", str(raised.exception))

    def test_probabilities_on_another_permutation_are_refused(self) -> None:
        probs = self.inputs.probs_TE[self.order, self.expert].flip(0)
        with self.assertRaises(RuntimeError) as raised:
            self._check(
                lambda: (
                    self.x_TD[self.order],
                    self.inputs.tokens_per_expert_E,
                    probs,
                )
            )
        self.assertIn("same permutation", str(raised.exception))

    def test_a_two_dimensional_probability_tensor_is_refused(self) -> None:
        probs = self.inputs.probs_TE[self.order, self.expert].reshape(-1, 1)
        with self.assertRaises(RuntimeError) as raised:
            self._check(
                lambda: (
                    self.x_TD[self.order],
                    self.inputs.tokens_per_expert_E,
                    probs,
                )
            )
        self.assertIn("flat one slot per row", str(raised.exception))

    def test_counts_that_disagree_with_the_shared_map_are_refused(self) -> None:
        counts = self.inputs.tokens_per_expert_E.clone()
        counts[0] += 1
        with self.assertRaises(RuntimeError) as raised:
            self._check(
                lambda: (
                    self.x_TD[self.order],
                    counts,
                    self.inputs.probs_TE[self.order, self.expert],
                )
            )
        self.assertIn("per-expert counts", str(raised.exception))

    def test_host_side_counts_are_accepted_after_canonicalization(self) -> None:
        """Megatron returns them on the host; titan returns them on the device.

        ``token_dispatcher.py:317`` ends in ``.cpu()``. The comparison happens
        outside every timed closure, so the move costs no published sample.
        """
        counts = self.inputs.tokens_per_expert_E.to("cpu")
        canonical = _canonical_counts(counts, self.inputs.tokens_per_expert_E)
        self.assertEqual(canonical.device, self.inputs.tokens_per_expert_E.device)
        self.assertEqual(canonical.dtype, self.inputs.tokens_per_expert_E.dtype)
        self._check(
            lambda: (
                self.x_TD[self.order],
                counts,
                self.inputs.probs_TE[self.order, self.expert],
            )
        )

    def test_a_missing_gradient_is_named_rather_than_dereferenced(self) -> None:
        leaf = torch.zeros(2, requires_grad=True)
        with self.assertRaises(RuntimeError) as raised:
            _require_gradients("probe", {"x_grad": leaf})
        self.assertIn("x_grad", str(raised.exception))

    def test_a_complete_set_of_gradients_passes(self) -> None:
        leaf = torch.zeros(2, requires_grad=True)
        leaf.sum().backward()
        _require_gradients("probe", {"x_grad": leaf})


class PinnedSourceTests(unittest.TestCase):
    """Every claim this scenario makes about the pinned trees, as text.

    A submodule bump that moves any of these fails here, which is where a
    reader can act on it, rather than in a GPU worker or -- worse -- not at all.
    """

    def test_token_dispatch_is_still_guarded_by_the_parallel_sizes(self) -> None:
        """The identity that decides where this scenario cuts."""
        source = _squeezed(_source(TOKEN_DISPATCHER))
        self.assertIn(
            "def token_dispatch(self, hidden_states, probs): "
            '"""Gathers tokens from all TP*EP ranks using AllGather.""" '
            "# Permute the tokens across the expert parallel devices. "
            "if self.tp_size > 1 or self.ep_size > 1:",
            source,
        )

    def test_the_all_to_all_collective_still_bypasses_at_one_rank(self) -> None:
        source = _squeezed(_source(MAPPINGS))
        self.assertIn(
            "world_size = group.size() "
            "# Bypass the function if we are using only 1 GPU. "
            "if world_size == 1: return input",
            source,
        )

    def test_the_permute_still_runs_in_dispatch_postprocess(self) -> None:
        source = _squeezed(_source(TOKEN_DISPATCHER))
        self.assertIn(
            "permute( hidden_states, self.local_map, "
            "num_out_tokens=tokens_per_expert.sum().item(), "
            "fused=self.config.moe_permute_fusion, )",
            source,
        )

    def test_dispatch_postprocess_still_copies_the_counts_to_the_host(
        self,
    ) -> None:
        """The blocking sync plan rule 5 makes this scenario declare."""
        source = _squeezed(_source(TOKEN_DISPATCHER))
        self.assertIn(
            "tokens_per_expert = self.local_map.sum(dim=0).long().cpu()", source
        )

    def test_the_allgather_dispatcher_still_permutes_probabilities_by_hand(
        self,
    ) -> None:
        source = _squeezed(_source(TOKEN_DISPATCHER))
        self.assertIn(
            "self.local_probs = self.local_probs.T.contiguous().masked_select( "
            "self.local_map.T.contiguous() )",
            source,
        )

    def test_the_alltoall_dispatcher_still_fuses_the_probabilities_in(
        self,
    ) -> None:
        """The delta ``mcore/dispatcher_alltoall`` measures, in one line."""
        source = _squeezed(_source(TOKEN_DISPATCHER))
        self.assertIn(
            ") = permute( hidden_states, self.routing_map, probs=probs, "
            "num_out_tokens=self.num_out_tokens, "
            "fused=self.config.moe_permute_fusion,",
            source,
        )

    def test_permute_still_branches_on_the_fusion_argument_alone(self) -> None:
        """Why the config field is decisive here, unlike ``moe_grouped_gemm``."""
        source = _squeezed(_source(MOE_UTILS))
        self.assertIn("if fused and probs is None:", source)
        self.assertIn("if fused and probs is not None:", source)
        self.assertIn(
            "flat_sorted = routing_map.reshape(-1).argsort(descending=True, "
            "stable=True) flat_sorted = flat_sorted[:num_out_tokens] "
            "sorted_indices = flat_sorted % num_tokens",
            source,
        )
        self.assertIn(
            "permuted_input = tokens.index_select(0, sorted_indices)", source
        )

    def test_the_three_phases_are_still_public_on_the_dispatcher(self) -> None:
        source = _source(TOKEN_DISPATCHER)
        for phase in (
            "def dispatch_preprocess(",
            "def token_dispatch(",
            "def dispatch_postprocess(",
        ):
            with self.subTest(phase=phase):
                self.assertIn(f"    @abstractmethod\n    {phase}", source)

    def test_routed_experts_compute_is_still_internal_api(self) -> None:
        """Why the third phase is reached through the dispatcher, not the layer."""
        source = _source(MOE_LAYER)
        self.assertIn(
            "    @internal_api\n    def routed_experts_compute(", source
        )

    def test_the_layer_still_exposes_preprocess_and_dispatch(self) -> None:
        source = _squeezed(_source(MOE_LAYER))
        self.assertIn(
            "hidden_states, probs = self.token_dispatcher.dispatch_preprocess( "
            "hidden_states, routing_map, probs )",
            source,
        )
        self.assertIn(
            "return self.token_dispatcher.token_dispatch(hidden_states, probs)",
            source,
        )

    def test_the_layer_still_maps_the_type_name_onto_the_class(self) -> None:
        source = _squeezed(_source(MOE_LAYER))
        for name, klass in (
            ("allgather", ALLGATHER_DISPATCHER),
            ("alltoall", ALLTOALL_DISPATCHER),
        ):
            with self.subTest(dispatcher=name):
                self.assertIn(
                    f'if config.moe_token_dispatcher_type == "{name}": '
                    f"self.token_dispatcher = {klass}(".replace('"', "'"),
                    source.replace('"', "'"),
                )

    def test_the_only_jit_fuser_here_is_on_a_dispatcher_we_cannot_build(
        self,
    ) -> None:
        """Why both mcore arms declare ``eager_reason`` and not ``compiled``."""
        source = _source(TOKEN_DISPATCHER)
        decorated = source.count("\n    @jit_fuser")
        self.assertEqual(decorated, 1)
        flex = source.index("class MoEFlexTokenDispatcher")
        self.assertGreater(source.index("\n    @jit_fuser"), flex)

    def test_the_moe_unflatten_glue_is_still_inert_without_tokens_per_sample(
        self,
    ) -> None:
        source = _squeezed(_source(TRANSFORMER_LAYER))
        self.assertIn(
            "if ( not self.is_moe_layer or packed_seq_params is None or "
            "getattr(packed_seq_params, 'tokens_per_sample', None) is None ): "
            "return hidden_states, padding_mask, None",
            source,
        )
        self.assertIn(
            "def _maybe_reflatten_from_moe(self, output, packed_seq_params, "
            "mbs): \"\"\"Re-flatten MoE output back to [mbs*S, 1, H] for the "
            'residual add.""" if mbs is None: return output',
            source,
        )

    def test_tokens_per_sample_still_defaults_to_none(self) -> None:
        self.assertIn(
            "tokens_per_sample: int = None", _source(PACKED_SEQ_PARAMS)
        )

    def test_nothing_in_this_repository_sets_tokens_per_sample(self) -> None:
        """The other half of the inertness claim, and the half that can rot."""
        root = Path(__file__).resolve().parent.parent
        setters = [
            path
            for directory in ("benchmarks", "tools")
            for path in sorted((root / directory).rglob("*.py"))
            if "tokens_per_sample=" in path.read_text()
        ]
        self.assertEqual(setters, [])

    def test_titan_still_builds_the_routing_map_where_this_scenario_cuts(
        self,
    ) -> None:
        source = _squeezed(_source(TITAN_MOE))
        self.assertIn(
            "routing_map_BLE = torch.zeros_like(scores_BLE, dtype=torch.bool)"
            ".scatter_( -1, topk_expert_ids_BLK, True, ) "
            "num_local_tokens_per_expert_E = routing_map_BLE.sum(dim=(0, 1))",
            source,
        )

    def test_titan_still_calls_the_dispatcher_with_the_four_arguments(
        self,
    ) -> None:
        source = _squeezed(_source(TITAN_MOE))
        self.assertIn(
            ") = self.token_dispatcher.dispatch( x_TD, topk_scores_TK, "
            "topk_expert_ids_TK, num_local_tokens_per_expert_E, )",
            source,
        )

    def test_titan_still_sorts_with_a_stable_ascending_argsort(self) -> None:
        source = _squeezed(_source(TITAN_DISPATCHER))
        self.assertIn(
            "token_indices_experts_sorted_N = torch.argsort( "
            "topk_expert_ids_TK.view(-1), stable=True )",
            source,
        )
        self.assertIn(
            "token_indices_experts_sorted_N = token_indices_experts_sorted_N "
            "// self.top_k routed_input_ND = x_TD[token_indices_experts_sorted_N]",
            source,
        )

    def test_the_titan_dispatcher_still_falls_back_at_one_expert_rank(
        self,
    ) -> None:
        source = _squeezed(_source(TITAN_DISPATCHER))
        self.assertIn(
            "# EP=1: fall back to local dispatch (no all-to-all needed) "
            "if self.ep_mesh is None: return LocalTokenDispatcher.dispatch(",
            source,
        )

    def test_titan_still_skips_the_usage_counter_without_load_balancing(
        self,
    ) -> None:
        """The in-place mutation this scenario would otherwise time."""
        source = _squeezed(_source(TITAN_MOE))
        self.assertIn(
            "if self.load_balance_coeff is not None: with torch.no_grad(): "
            "self.tokens_per_expert_E.add_(num_local_tokens_per_expert_E)",
            source,
        )


class RegisteredShapeTests(unittest.TestCase):
    def test_the_inputs_build_at_every_registered_model_size(self) -> None:
        """A tiny workload, because ``huge`` allocates by ``dim`` here."""
        workload = KernelWorkload(batch=1, seq_len=8)
        for name, shape in PIPER_SHAPES.items():
            with self.subTest(size=name):
                inputs = _inputs(shape=shape, workload=workload)
                slots = 8 * shape.top_k
                self.assertEqual(
                    inputs.tokens_per_expert_E.tolist(),
                    [slots // shape.num_experts] * shape.num_experts,
                )


class ArmNameTests(unittest.TestCase):
    """The names this module spells, and what a merge must keep true.

    While the scenario is a declaration fragment
    (``reports/20260819-partc/decl/dispatch_permute.decl.py``) these can only
    compare this module's constants against themselves. After the merge,
    rewrite the class against ``KERNEL_SCENARIOS["dispatch_permute"]`` so it
    catches a divergence between the module and the registry instead.
    """

    NAMES = (
        COPY_FLOOR_ARM,
        MCORE_BASE_ARM,
        MCORE_NO_PERMUTE_FUSION_ARM,
        MCORE_DISPATCHER_ALLTOALL_ARM,
        TITAN_ARM,
    )

    def test_the_five_names_are_unique_and_survive_the_fragment_stem(
        self,
    ) -> None:
        self.assertEqual(len(set(self.NAMES)), len(self.NAMES))
        stems = [fragment_stem(name) for name in self.NAMES]
        self.assertEqual(len(set(stems)), len(stems))
        for stem in stems:
            with self.subTest(stem=stem):
                self.assertNotIn("/", stem)

    def test_each_builder_returns_the_name_the_fragment_declares(self) -> None:
        """The two builders a CPU can run; the mcore three need a device."""
        inputs, titan = _shared_titan_build()
        self.assertEqual(
            build_dispatch_permute_copy_floor(TINY, TINY_WORKLOAD, inputs).name,
            COPY_FLOOR_ARM,
        )
        self.assertEqual(titan.name, TITAN_ARM)

    def test_the_two_megatron_variants_are_named_after_their_profiles(
        self,
    ) -> None:
        self.assertEqual(
            MCORE_NO_PERMUTE_FUSION_ARM, f"mcore/{NO_PERMUTE_FUSION.name}"
        )
        self.assertEqual(
            MCORE_DISPATCHER_ALLTOALL_ARM, f"mcore/{DISPATCHER_ALLTOALL.name}"
        )


if __name__ == "__main__":
    unittest.main()
