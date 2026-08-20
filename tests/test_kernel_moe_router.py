"""CPU tests for the ``moe_router`` cross-engine kernel scenario.

Everything here runs without a GPU. It does **not** all run without megatron:
``UnflattenGuardTests`` and ``TeFusionReachabilityTests`` import
``megatron.core``, and through it TransformerEngine, and skip themselves when
the submodule is absent. The sibling ``tests/test_kernel_ffn_norm.py`` can
claim "without megatron" and this module cannot, because two of its guards are
statements about megatron's own source. Four things are testable without a
GPU, and each is something a wrong scenario gets wrong quietly:

* **The two engines compute one function.** TorchTitan takes the softmax over
  all experts, selects the top k and renormalizes; megatron selects on the
  logits and takes the softmax over the k selected. The cross-engine row rests
  entirely on those two producing the same probabilities and the same
  gradients. ``EngineOrderingTests`` runs the real upstream
  ``TokenChoiceTopKRouter`` against this module's fp64 reference, which is
  written in megatron's ordering.
* **The two profile deltas are real deltas.** ``mcore/router_bf16`` exists
  because the base profile already asks for fp32, so the delta that earns an
  arm is bf16. An earlier draft of the plan declared an ``mcore/router_fp32``
  arm that changed nothing. ``ProfileDeltaTests`` pins the direction.
* **The guards raise.** Every build guard in the module exists because the
  failure it catches is numerically valid and therefore invisible to a
  correctness gate. Each one is exercised in both directions.
* **The shared timed closures behave**, exercised over a stand-in CPU module
  in place of the two real routers.

The megatron side itself needs a CUDA device, a process group and TE, so no
test here builds one. What a CPU can still check about that side is the shape
of the guards, which is what these tests do with stand-in objects.
"""

import sys
import unittest
import warnings
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from benchmarks.kernel.operations.moe_router import (
    _assert_mcore_gating_gemm_takes_the_te_path,
    _assert_mcore_router,
    _assert_te_router_fusion_is_reachable,
    _assert_titan_router_config,
    _assert_titan_router_is_fp32,
    _assert_unflatten_is_an_early_return,
    _canonical_outputs,
    _release_untimed_moe_submodules,
    _router_arm,
    build_moe_router_copy_floor,
    LAYER,
    MCORE_ARM_PROFILES,
    MCORE_BASE_ARM_NAME,
    MCORE_BF16_ARM_NAME,
    MCORE_FUSION_ARM_NAME,
    MCORE_WEIGHT_NAME,
    mcore_moe_layer_path,
    mcore_transformer_layer_path,
    moe_router_inputs,
    moe_router_reference,
    MoeRouterInputs,
    ROUTER_BF16,
    ROUTER_FUSION,
    TITAN_WEIGHT_NAME,
    WEIGHT_COMPONENT,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE
from benchmarks.models.piper_qwen3.shape import PiperShape

# Small enough to run in milliseconds, wide enough that the gate GEMM reduces
# over a real row and the top-2 of 4 selection is a real selection.
TINY = PiperShape(name="tiny", dim=256, n_layers=2, vocab_size=64)
TINY_WORKLOAD = KernelWorkload(batch=2, seq_len=16)
TOKENS = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len

# Stand-in classes whose *names* are what the megatron guard reads. It only
# ever reads attributes, so a namespace runs the same code path a real
# ``TopKRouter`` would -- the failures it catches are configuration, not
# arithmetic, and none of them needs a GPU to express.
_FakeTopKRouter = type("TopKRouter", (SimpleNamespace,), {})
_FakeInferenceRouter = type("InferenceTopKRouter", (SimpleNamespace,), {})
_FakeMoELayer = type("MoELayer", (SimpleNamespace,), {})


def _inputs(shape=TINY, workload=TINY_WORKLOAD, seed=0) -> MoeRouterInputs:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return moe_router_inputs(shape, workload, torch.device("cpu"), generator)


def _rel_l2(value: torch.Tensor, truth: torch.Tensor) -> float:
    delta = (value.double() - truth.double()).norm()
    norm = truth.double().norm()
    return (delta / norm).item() if norm else delta.item()


class _McoreOrderingRouter(nn.Module):
    """Megatron's ordering: top-k on the logits, then softmax over the k.

    Not a mock of megatron. It is the arithmetic ``moe_utils.py:900-949``
    performs, written once here so the shared closures and the canonical
    conversion can be exercised without a GPU.
    """

    def __init__(self, dim: int, num_experts: int, top_k: int) -> None:
        super().__init__()
        self.gate = nn.Linear(dim, num_experts, bias=False)
        self.top_k = top_k

    def forward(self, x: torch.Tensor):
        logits = self.gate(x)
        top_logits, top_indices = torch.topk(logits, self.top_k, dim=-1)
        top_probs = torch.softmax(top_logits, dim=-1)
        probs = torch.zeros_like(logits).scatter(-1, top_indices, top_probs)
        routing_map = torch.zeros_like(logits).scatter(
            -1, top_indices, torch.ones_like(top_probs)
        )
        return probs, routing_map


def _titan_router_config(shape: PiperShape):
    """The production TorchTitan MoE node, built at the toy shape."""
    from benchmarks.models.piper_qwen3.config_registry import _piper_1b_model

    return _piper_1b_model(fuse_qkv=True, shape=shape).layers[LAYER].moe


def _titan_dense(
    module: nn.Module, x: torch.Tensor, shape: PiperShape
) -> tuple[torch.Tensor, torch.Tensor]:
    """Titan's ``[*, K]`` router outputs as a dense ``[T, E]`` pair.

    The same conversion ``build_moe_router_titan``'s ``canonical`` performs,
    and for the same reason: the dense form is the only one in which the two
    engines' selections are directly comparable.
    """
    with warnings.catch_warnings():
        # torch.autocast(device_type="cpu", dtype=torch.float32) warns and
        # disables itself; the module and its input are fp32 here anyway, so
        # the arithmetic is the arithmetic the CUDA path runs.
        warnings.simplefilter("ignore")
        top_scores, top_indices, _ = module(x, None)
    tokens = x.shape[0] * x.shape[1]
    flat_scores = top_scores.reshape(tokens, shape.top_k)
    flat_indices = top_indices.reshape(tokens, shape.top_k)
    zeros = torch.zeros((tokens, shape.num_experts), dtype=flat_scores.dtype)
    return (
        zeros.scatter(1, flat_indices, flat_scores),
        zeros.scatter(1, flat_indices, torch.ones_like(flat_scores)),
    )


def _megatron_available() -> bool:
    """Whether the pinned megatron submodule can be imported on this host.

    Two tests reach a megatron symbol -- the ``PackedSeqParams`` the unflatten
    guard builds, and the TE-fusion binding. Neither needs a GPU, but both
    need the submodule checked out, so both skip rather than fail on a tree
    that has not initialized it.
    """
    from benchmarks.models.piper_qwen3.megatron_bootstrap import (
        add_megatron_to_path,
    )

    try:
        add_megatron_to_path()
        import megatron.core.packed_seq_params  # noqa: F401
    except Exception:
        return False
    return True


class InputsTests(unittest.TestCase):
    def test_the_shared_tensors_have_the_shapes_every_arm_consumes(self) -> None:
        inputs = _inputs()
        self.assertEqual(
            tuple(inputs.x.shape),
            (TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len, TINY.dim),
        )
        self.assertEqual(inputs.x.dtype, torch.bfloat16)
        self.assertEqual(
            tuple(inputs.grad_probs.shape),
            (TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len, TINY.num_experts),
        )
        self.assertEqual(inputs.grad_probs.dtype, torch.float32)
        # fp32, so every arm rounds the same values to bf16 exactly once.
        self.assertEqual(
            tuple(inputs.gate_weight.shape), (TINY.num_experts, TINY.dim)
        )
        self.assertEqual(inputs.gate_weight.dtype, torch.float32)

    def test_inputs_rebuild_bit_identically_from_the_same_seed(self) -> None:
        """Every worker rebuilds these, so a drift would desynchronize arms."""
        first, second = _inputs(seed=3), _inputs(seed=3)
        self.assertTrue(torch.equal(first.x, second.x))
        self.assertTrue(torch.equal(first.gate_weight, second.gate_weight))
        self.assertFalse(torch.equal(first.x, _inputs(seed=4).x))

    def test_a_megatron_arm_moves_fewer_bytes_than_the_floor(self) -> None:
        """The claim that makes ``x_floor`` a lower bound rather than an
        estimate on the megatron side: the floor reads and writes x, and that
        router reads x and writes a tensor orders of magnitude smaller."""
        inputs = _inputs()
        x_bytes = inputs.x.numel() * inputs.x.element_size()
        self.assertEqual(inputs.floor_bytes_moved, 2 * x_bytes)
        self.assertGreater(inputs.mcore_bytes_moved, x_bytes)
        self.assertLess(inputs.mcore_bytes_moved, inputs.floor_bytes_moved)

    def test_titan_is_charged_for_the_fp32_copy_its_autocast_makes(
        self,
    ) -> None:
        """The two engines must NOT share one figure. Titan's autocast casts
        both ``F.linear`` operands up, so it materializes a full fp32 copy of
        the hidden state that megatron never makes; one shared number would
        put that asymmetry inside the GB/s and x_floor columns instead of in
        front of the reader."""
        inputs = _inputs()
        x_bytes = inputs.x.numel() * inputs.x.element_size()
        # Read x, write the fp32 copy, read it back in the mm.
        self.assertGreaterEqual(inputs.titan_bytes_moved, 5 * x_bytes)
        self.assertGreater(inputs.titan_bytes_moved, inputs.mcore_bytes_moved)
        # And it is the one arm that moves MORE than the floor, which is why
        # a titan x_floor below 1.0 is a real number rather than an impossible
        # one.
        self.assertGreater(inputs.titan_bytes_moved, inputs.floor_bytes_moved)

    def test_each_arm_carries_its_own_engines_byte_budget(self) -> None:
        """A regression guard on the wiring, not on the arithmetic. The floor
        and the two engines are three different figures, and an arm handed
        the wrong one publishes a GB/s column for another arm's traffic."""
        inputs = _inputs()
        floor = build_moe_router_copy_floor(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(floor.bytes_moved, inputs.floor_bytes_moved)
        self.assertNotEqual(inputs.titan_bytes_moved, inputs.mcore_bytes_moved)

    def test_a_top_k_wider_than_the_expert_count_fails_with_both_numbers(
        self,
    ) -> None:
        """Loud, and named. ``torch.topk`` would otherwise raise inside a
        timed closure, where the reason reaches no result file."""
        impossible = replace(TINY, num_experts=2, top_k=3)
        with self.assertRaises(ValueError) as caught:
            _inputs(shape=impossible)
        message = str(caught.exception)
        self.assertIn("top_k (3)", message)
        self.assertIn("num_experts (2)", message)


class ReferenceTests(unittest.TestCase):
    def test_the_reference_routing_satisfies_the_invariants_it_gates_on(
        self,
    ) -> None:
        reference = moe_router_reference(TINY, TINY_WORKLOAD, _inputs())
        probs, routing_map = reference["probs"], reference["routing_map"]
        self.assertEqual(tuple(probs.shape), (TOKENS, TINY.num_experts))
        # The routing map is a mask, and probs is zero exactly off it.
        self.assertTrue(
            torch.equal(routing_map, (probs != 0).to(routing_map.dtype))
        )
        # Both bitwise-gated rows are fp32, so torch.equal can compare them
        # with an arm's own fp32 rows at all.
        self.assertEqual(routing_map.dtype, torch.float32)
        self.assertTrue(
            torch.allclose(
                reference["prob_row_sums"],
                torch.ones(TOKENS, dtype=torch.float64),
            )
        )
        self.assertTrue(
            torch.equal(
                reference["selected_count"],
                torch.full((TOKENS,), float(TINY.top_k)),
            )
        )

    def test_selected_count_is_float32_because_it_is_gated_bitwise(
        self,
    ) -> None:
        """``torch.equal`` compares dtypes, so an fp64 reference row would
        fail against every arm's fp32 row for a reason no reader could see."""
        reference = moe_router_reference(TINY, TINY_WORKLOAD, _inputs())
        self.assertEqual(reference["selected_count"].dtype, torch.float32)

    def test_the_reference_quantizes_the_gate_to_bfloat16_first(self) -> None:
        """An fp64 truth built from the unrounded gate would charge every arm
        for a cast none of them performs."""
        inputs = _inputs()
        rounded = replace(
            inputs, gate_weight=inputs.gate_weight.to(torch.bfloat16).float()
        )
        self.assertFalse(torch.equal(inputs.gate_weight, rounded.gate_weight))
        first = moe_router_reference(TINY, TINY_WORKLOAD, inputs)
        second = moe_router_reference(TINY, TINY_WORKLOAD, rounded)
        self.assertTrue(torch.equal(first["logits"], second["logits"]))

    def test_the_reference_gradient_flows_only_through_selected_experts(
        self,
    ) -> None:
        """The property that makes ``x_grad`` tie-sensitive, and therefore the
        reason it is not gated across a precision boundary: a token routed
        differently carries a different gradient."""
        inputs = _inputs()
        reference = moe_router_reference(TINY, TINY_WORKLOAD, inputs)
        mask = reference["routing_map"].reshape(
            TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len, TINY.num_experts
        )
        masked = replace(inputs, grad_probs=inputs.grad_probs * mask)
        off_map = moe_router_reference(TINY, TINY_WORKLOAD, masked)
        self.assertLess(_rel_l2(off_map["x_grad"], reference["x_grad"]), 1e-12)


class EngineOrderingTests(unittest.TestCase):
    """The claim the whole cross-engine row rests on.

    TorchTitan runs softmax over E, top-k, renormalize. Megatron runs top-k,
    softmax over k. The reference is written in megatron's ordering, so
    running the real upstream titan module against it is a proof of the
    identity rather than a restatement of it.
    """

    def _paired(self):
        inputs = _inputs()
        reference = moe_router_reference(TINY, TINY_WORKLOAD, inputs)
        module = _titan_router_config(TINY).router.build()
        # The same gate both arms load, rounded once to bf16 as both engines
        # store it, then promoted so CPU arithmetic is representative.
        module.load_state_dict(
            {"gate.weight": inputs.gate_weight.to(torch.bfloat16).float()}
        )
        return inputs, reference, module, inputs.x.float()

    def test_titans_ordering_produces_megatrons_probabilities(self) -> None:
        _, reference, module, x = self._paired()
        probs, routing_map = _titan_dense(module, x, TINY)
        self.assertTrue(
            torch.equal(routing_map.float(), reference["routing_map"])
        )
        self.assertLess(_rel_l2(probs, reference["probs"]), 1e-6)

    def test_titans_ordering_produces_megatrons_gradients(self) -> None:
        """Titan's softmax runs over all E experts, so its graph carries a
        path through the experts it did not select. That path cancels
        algebraically -- the renormalization is invariant to a common scale --
        and this is what says it also cancels numerically."""
        inputs, reference, module, x = self._paired()
        leaf = x.clone().requires_grad_()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            top_scores, top_indices, _ = module(leaf, None)
        grad = inputs.grad_probs.gather(-1, top_indices)
        torch.autograd.backward(top_scores, grad)
        self.assertLess(_rel_l2(leaf.grad, reference["x_grad"]), 1e-5)
        self.assertLess(
            _rel_l2(module.gate.weight.grad, reference["gate_weight_grad"]),
            1e-5,
        )

    def test_route_norm_off_would_break_the_identity(self) -> None:
        """Why ``_assert_titan_router_config`` refuses ``route_norm=False``:
        without it titan publishes an unnormalized weight against megatron's
        normalized one, and no row sum is 1 any more."""
        _, reference, module, x = self._paired()
        module.route_norm = False
        probs, _ = _titan_dense(module, x, TINY)
        self.assertGreater(_rel_l2(probs, reference["probs"]), 1e-2)


class CanonicalOutputTests(unittest.TestCase):
    def _valid(self):
        routing_map = torch.zeros(TOKENS, TINY.num_experts)
        routing_map[:, : TINY.top_k] = 1.0
        return {
            "probs": torch.rand(TOKENS, TINY.num_experts),
            "routing_map": routing_map,
            "logits": torch.rand(TOKENS, TINY.num_experts),
            "x_grad": torch.rand(TOKENS, TINY.dim),
            "gate_weight_grad": torch.rand(TINY.num_experts, TINY.dim),
        }

    def test_the_two_derived_rows_are_derived_and_not_supplied(self) -> None:
        tensors = self._valid()
        outputs = _canonical_outputs(
            arm="stand_in", shape=TINY, workload=TINY_WORKLOAD, **tensors
        )
        self.assertTrue(
            torch.allclose(
                outputs["prob_row_sums"], tensors["probs"].sum(dim=-1)
            )
        )
        self.assertTrue(
            torch.equal(
                outputs["selected_count"],
                torch.full((TOKENS,), float(TINY.top_k)),
            )
        )

    def test_the_gradient_is_reshaped_to_the_canonical_input_shape(
        self,
    ) -> None:
        outputs = _canonical_outputs(
            arm="stand_in", shape=TINY, workload=TINY_WORKLOAD, **self._valid()
        )
        self.assertEqual(
            tuple(outputs["x_grad"].shape),
            (TINY_WORKLOAD.batch, TINY_WORKLOAD.seq_len, TINY.dim),
        )
        for name in ("logits", "probs", "routing_map", "selected_count"):
            with self.subTest(output=name):
                self.assertEqual(outputs[name].dtype, torch.float32)

    def test_a_routing_of_the_wrong_shape_is_named_and_refused(self) -> None:
        tensors = self._valid()
        tensors["probs"] = torch.rand(TOKENS, TINY.num_experts + 1)
        with self.assertRaises(RuntimeError) as caught:
            _canonical_outputs(
                arm="stand_in", shape=TINY, workload=TINY_WORKLOAD, **tensors
            )
        self.assertIn("canonical probs", str(caught.exception))

    def test_a_missing_gradient_is_named_rather_than_an_attribute_error(
        self,
    ) -> None:
        tensors = self._valid()
        tensors["gate_weight_grad"] = None
        with self.assertRaises(RuntimeError) as caught:
            _canonical_outputs(
                arm="stand_in", shape=TINY, workload=TINY_WORKLOAD, **tensors
            )
        self.assertIn("gate_weight_grad", str(caught.exception))


class SharedClosureTests(unittest.TestCase):
    """The closures every arm runs, over a stand-in in place of two routers."""

    def _arm(self):
        inputs = _inputs()
        module = _McoreOrderingRouter(TINY.dim, TINY.num_experts, TINY.top_k)
        with torch.no_grad():
            module.gate.weight.copy_(
                inputs.gate_weight.to(torch.bfloat16).float()
            )
        x = inputs.x.float().reshape(TOKENS, TINY.dim)
        grad = inputs.grad_probs.float().reshape(TOKENS, TINY.num_experts)

        def canonical(x_grad, result):
            probs, routing_map = result
            return _canonical_outputs(
                arm="stand_in",
                shape=TINY,
                workload=TINY_WORKLOAD,
                probs=probs.detach(),
                routing_map=routing_map.detach(),
                logits=module.gate(x).detach(),
                x_grad=x_grad,
                gate_weight_grad=module.gate.weight.grad,
            )

        arm = _router_arm(
            name="stand_in",
            owner=module,
            call=module,
            backward_target=lambda result: result[0],
            canonical=canonical,
            x_native=x,
            grad_native=grad,
            bytes_moved=inputs.mcore_bytes_moved,
            notes={"router": "stand_in"},
        )
        return inputs, module, arm

    def test_the_arm_exposes_exactly_the_two_declared_modes(self) -> None:
        _, _, arm = self._arm()
        self.assertEqual(sorted(arm.calls), ["forward", "forward_backward"])

    def test_forward_backward_is_repeatable_and_does_not_accumulate(
        self,
    ) -> None:
        """Every timed call clears the leaf and the parameters first, so a
        burst of k calls measures k identical operations rather than a growing
        gradient."""
        _, module, arm = self._arm()
        arm.calls["forward_backward"]()
        first = module.gate.weight.grad.clone()
        arm.calls["forward_backward"]()
        self.assertTrue(torch.allclose(module.gate.weight.grad, first))

    def test_each_mode_runs_on_its_own_leaf(self) -> None:
        """Three leaves, so a burst in one mode cannot leave a gradient on the
        tensor another mode is about to read."""
        _, _, arm = self._arm()
        arm.calls["forward_backward"]()
        result = arm.calls["forward"]()
        self.assertEqual(len(result), 2)

    def test_correctness_outputs_agree_with_the_fp64_reference(self) -> None:
        """End to end over the shared machinery: the stand-in runs megatron's
        ordering, so its canonical outputs must land on the reference."""
        inputs, _, arm = self._arm()
        reference = moe_router_reference(TINY, TINY_WORKLOAD, inputs)
        outputs = arm.correctness_outputs()
        for name in ("logits", "probs", "x_grad", "gate_weight_grad"):
            with self.subTest(output=name):
                self.assertLess(_rel_l2(outputs[name], reference[name]), 1e-5)
        self.assertTrue(
            torch.equal(outputs["selected_count"], reference["selected_count"])
        )

    def test_the_floor_declares_forward_only_and_copies(self) -> None:
        inputs = _inputs()
        floor = build_moe_router_copy_floor(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(list(floor.calls), ["forward"])
        self.assertEqual(floor.bytes_moved, inputs.floor_bytes_moved)
        # A floor has no gate: it computes nothing to compare.
        self.assertEqual(floor.correctness_outputs(), {})
        floor.calls["forward"]()


class ProfileDeltaTests(unittest.TestCase):
    """Both variants change something, and each changes what its name says."""

    def test_router_fusion_turns_on_a_flag_the_base_leaves_at_its_default(
        self,
    ) -> None:
        self.assertNotIn("moe_router_fusion", BASE.config_overrides)
        self.assertIs(ROUTER_FUSION.config_overrides["moe_router_fusion"], True)

    def test_the_bf16_delta_removes_fp32_rather_than_adding_it(self) -> None:
        """The error the plan records as having shipped once: an arm named for
        a value the base profile already sets measures nothing at all."""
        self.assertEqual(BASE.config_overrides["moe_router_dtype"], "fp32")
        self.assertIsNone(ROUTER_BF16.config_overrides["moe_router_dtype"])

    def test_each_variant_differs_from_the_base_in_exactly_one_field(
        self,
    ) -> None:
        for profile, field in (
            (ROUTER_FUSION, "moe_router_fusion"),
            (ROUTER_BF16, "moe_router_dtype"),
        ):
            with self.subTest(profile=profile.name):
                keys = set(profile.config_overrides) | set(BASE.config_overrides)
                differing = {
                    key
                    for key in keys
                    if profile.config_overrides.get(key)
                    != BASE.config_overrides.get(key)
                }
                self.assertEqual(differing, {field})

    def test_every_mcore_arm_names_the_profile_it_builds(self) -> None:
        for arm, profile in MCORE_ARM_PROFILES.items():
            with self.subTest(arm=arm):
                self.assertTrue(arm.startswith("mcore/"))
                self.assertEqual(arm.split("/", 1)[1], profile.name)


class TitanGuardTests(unittest.TestCase):
    def test_the_production_node_is_the_one_this_arm_measures(self) -> None:
        """Without this, the arm could time a router the registry never puts
        in front of the experts and every other test would still pass."""
        moe = _titan_router_config(TINY)
        self.assertIsNone(moe.load_balance_coeff)
        _assert_titan_router_config(moe.router, TINY)

    def test_each_behavioural_field_is_refused_when_it_moves(self) -> None:
        router = _titan_router_config(TINY).router
        for field, wrong in (
            ("route_norm", False),
            ("score_func", "sigmoid"),
            ("route_scale", 2.0),
            ("top_k", TINY.top_k + 1),
            ("num_experts", TINY.num_experts + 1),
            ("_debug_force_load_balance", True),
            ("num_expert_groups", 2),
        ):
            with self.subTest(field=field):
                with self.assertRaises(RuntimeError) as caught:
                    _assert_titan_router_config(
                        replace(router, **{field: wrong}), TINY
                    )
                self.assertIn(field, str(caught.exception))

    def test_a_gate_of_the_wrong_width_is_refused(self) -> None:
        router = _titan_router_config(TINY).router
        wider = replace(
            router,
            gate=replace(router.gate, out_features=TINY.num_experts + 1),
        )
        with self.assertRaises(RuntimeError) as caught:
            _assert_titan_router_config(wider, TINY)
        self.assertIn("gate.out_features", str(caught.exception))

    def test_the_fp32_guard_accepts_fp32_and_refuses_anything_lower(
        self,
    ) -> None:
        """The run-time half of this scenario's headline caption: titan's
        router computes its gate under ``torch.autocast(float32)``, and the
        cross-engine row claims both engines route in fp32."""
        fp32 = torch.zeros(2, 2)
        _assert_titan_router_is_fp32(fp32, fp32)
        with self.assertRaises(RuntimeError) as caught:
            _assert_titan_router_is_fp32(fp32, fp32.bfloat16())
        self.assertIn("topk_scores_BLK", str(caught.exception))
        with self.assertRaises(RuntimeError) as caught:
            _assert_titan_router_is_fp32(fp32.bfloat16(), fp32)
        self.assertIn("scores_BLE", str(caught.exception))


def _fake_mcore_router(arm: str, **overrides):
    """A stand-in router and MoE layer shaped like the ones the guard reads."""
    profile = MCORE_ARM_PROFILES[arm]
    config = SimpleNamespace(
        num_moe_experts=TINY.num_experts,
        moe_router_topk=TINY.top_k,
        moe_router_score_function="softmax",
        moe_router_pre_softmax=False,
        moe_router_topk_scaling_factor=None,
        moe_router_num_groups=None,
        moe_router_group_topk=None,
        moe_z_loss_coeff=None,
        moe_input_jitter_eps=None,
        moe_expert_capacity_factor=None,
        moe_router_force_load_balancing=False,
        moe_router_force_biased=None,
        cuda_graph_impl="none",
        moe_router_dtype=profile.config_overrides.get("moe_router_dtype"),
        moe_router_fusion=profile.config_overrides.get(
            "moe_router_fusion", False
        ),
    )
    for name, value in overrides.items():
        if hasattr(config, name):
            setattr(config, name, value)
    router = _FakeTopKRouter(
        config=config,
        routing_type=overrides.get("routing_type", "none"),
        enable_expert_bias=overrides.get("enable_expert_bias", False),
        expert_bias=overrides.get("expert_bias", None),
        bias=overrides.get("bias", None),
        router_replay=overrides.get("router_replay", None),
        training=overrides.get("training", True),
        weight=torch.zeros(TINY.num_experts, TINY.dim),
        is_aux_loss_enabled=lambda: overrides.get("aux_loss", False),
    )
    return _FakeMoELayer(router=router), router


class McoreGuardTests(unittest.TestCase):
    def test_a_correct_layer_passes_and_reports_its_delta(self) -> None:
        for arm, profile in MCORE_ARM_PROFILES.items():
            with self.subTest(arm=arm):
                moe_layer, router = _fake_mcore_router(arm)
                notes = _assert_mcore_router(arm, moe_layer, router, TINY)
                self.assertEqual(notes["profile"], profile.name)
                self.assertIs(notes["compiled"], False)
                self.assertEqual(
                    notes["moe_router_dtype"],
                    profile.config_overrides.get("moe_router_dtype"),
                )

    def test_a_delta_that_did_not_reach_the_config_is_refused(self) -> None:
        """The check plan section C.3 says a verifier hunts for. A variant arm
        whose flag never landed measures the base under the variant's name,
        and no correctness gate can see it: both are numerically valid."""
        moe_layer, router = _fake_mcore_router(
            MCORE_FUSION_ARM_NAME, moe_router_fusion=False
        )
        with self.assertRaises(RuntimeError) as caught:
            _assert_mcore_router(MCORE_FUSION_ARM_NAME, moe_layer, router, TINY)
        self.assertIn("moe_router_fusion", str(caught.exception))

        moe_layer, router = _fake_mcore_router(
            MCORE_BF16_ARM_NAME, moe_router_dtype="fp32"
        )
        with self.assertRaises(RuntimeError) as caught:
            _assert_mcore_router(MCORE_BF16_ARM_NAME, moe_layer, router, TINY)
        self.assertIn("moe_router_dtype", str(caught.exception))

    def test_the_base_arm_is_refused_if_the_fp32_request_disappears(
        self,
    ) -> None:
        """The mirror of the test above, and the reason the base is the
        cross-engine anchor: it must be the fp32 side."""
        moe_layer, router = _fake_mcore_router(
            MCORE_BASE_ARM_NAME, moe_router_dtype=None
        )
        with self.assertRaises(RuntimeError):
            _assert_mcore_router(MCORE_BASE_ARM_NAME, moe_layer, router, TINY)

    def test_a_profile_that_never_states_its_precision_is_refused(
        self,
    ) -> None:
        """The hole a defaulted lookup left open. megatron's dataclass default
        for ``moe_router_dtype`` is also ``None``, so a key DELETED from the
        base profile would have given expected None against a built config of
        None, passed, and routed ``mcore/base`` in bf16 under an fp32 label --
        the exact silent precision change this scenario is about."""
        silent = replace(
            BASE,
            name="no_dtype",
            config_overrides={
                key: value
                for key, value in BASE.config_overrides.items()
                if key != "moe_router_dtype"
            },
        )
        moe_layer, router = _fake_mcore_router(MCORE_BASE_ARM_NAME)
        router.config.moe_router_dtype = None
        with mock.patch.dict(
            MCORE_ARM_PROFILES, {MCORE_BASE_ARM_NAME: silent}
        ):
            with self.assertRaises(RuntimeError) as caught:
                _assert_mcore_router(
                    MCORE_BASE_ARM_NAME, moe_layer, router, TINY
                )
        message = str(caught.exception)
        self.assertIn("does not state", message)
        self.assertIn("moe_router_dtype", message)

    def test_deterministic_algorithms_are_refused(self) -> None:
        """A process global that would land inside the fusion row. With it on,
        megatron's UNFUSED score function builds its dense routing through
        index_put_ rather than scatter, and the fused arm never reaches that
        code at all, so the flag moves two of the three arms and not the
        third."""
        moe_layer, router = _fake_mcore_router(MCORE_BASE_ARM_NAME)
        with mock.patch.object(
            torch, "are_deterministic_algorithms_enabled", lambda: True
        ):
            with self.assertRaises(RuntimeError) as caught:
                _assert_mcore_router(
                    MCORE_BASE_ARM_NAME, moe_layer, router, TINY
                )
        self.assertIn("deterministic", str(caught.exception))

    def test_every_optional_routing_stage_is_refused_when_it_is_live(
        self,
    ) -> None:
        """Each of these adds work titan's router has no counterpart for, and
        each is numerically valid, so only a guard can see it."""
        for override in (
            {"moe_z_loss_coeff": 1e-3},
            {"moe_input_jitter_eps": 1e-2},
            {"moe_expert_capacity_factor": 1.0},
            {"moe_router_topk_scaling_factor": 2.0},
            {"moe_router_group_topk": 2},
            {"moe_router_num_groups": 2},
            {"moe_router_force_load_balancing": True},
            {"moe_router_force_biased": 0.5},
            {"moe_router_pre_softmax": True},
            {"moe_router_score_function": "sigmoid"},
            {"routing_type": "aux_loss"},
            {"enable_expert_bias": True},
            {"expert_bias": torch.zeros(TINY.num_experts)},
            {"bias": torch.zeros(TINY.num_experts)},
            {"router_replay": object()},
            {"aux_loss": True},
            {"training": False},
            {"cuda_graph_impl": "transformer_engine"},
        ):
            with self.subTest(**{k: str(v) for k, v in override.items()}):
                moe_layer, router = _fake_mcore_router(
                    MCORE_BASE_ARM_NAME, **override
                )
                with self.assertRaises(RuntimeError):
                    _assert_mcore_router(
                        MCORE_BASE_ARM_NAME, moe_layer, router, TINY
                    )

    def test_an_inference_router_is_refused(self) -> None:
        """``InferenceTopKRouter`` returns a compact ``[tokens, topk]`` index
        routing, not the dense map this scenario canonicalizes."""
        moe_layer, router = _fake_mcore_router(MCORE_BASE_ARM_NAME)
        inference = _FakeInferenceRouter(**vars(router))
        moe_layer.router = inference
        with self.assertRaises(RuntimeError) as caught:
            _assert_mcore_router(
                MCORE_BASE_ARM_NAME, moe_layer, inference, TINY
            )
        self.assertIn("InferenceTopKRouter", str(caught.exception))

    def test_a_layer_that_is_not_a_moe_layer_is_refused(self) -> None:
        moe_layer, router = _fake_mcore_router(MCORE_BASE_ARM_NAME)
        dense = type("MLP", (SimpleNamespace,), {})(router=router)
        with self.assertRaises(RuntimeError) as caught:
            _assert_mcore_router(MCORE_BASE_ARM_NAME, dense, router, TINY)
        self.assertIn("MoELayer", str(caught.exception))

    def test_a_gate_matrix_of_the_wrong_shape_is_refused(self) -> None:
        moe_layer, router = _fake_mcore_router(MCORE_BASE_ARM_NAME)
        router.weight = torch.zeros(TINY.num_experts, TINY.dim + 1)
        with self.assertRaises(RuntimeError) as caught:
            _assert_mcore_router(MCORE_BASE_ARM_NAME, moe_layer, router, TINY)
        self.assertIn(mcore_moe_layer_path(), str(caught.exception))


class UnflattenGuardTests(unittest.TestCase):
    """The activation unflatten is an early return, and that is proved."""

    @classmethod
    def setUpClass(cls) -> None:
        if not _megatron_available():
            raise unittest.SkipTest(
                "the pinned Megatron-LM submodule is not checked out"
            )

    def _layer(self, unflatten):
        layer = nn.Module()
        layer.register_parameter("anchor", nn.Parameter(torch.zeros(1)))
        layer.is_moe_layer = True
        layer.config = SimpleNamespace(hidden_size=TINY.dim)
        layer._maybe_unflatten_for_moe = unflatten
        return layer

    def test_an_identity_unflatten_is_accepted(self) -> None:
        layer = self._layer(lambda hidden, mask, packed: (hidden, mask, None))
        _assert_unflatten_is_an_early_return(layer, MCORE_BASE_ARM_NAME)

    def test_an_unflatten_that_transposes_is_refused(self) -> None:
        """It sits inside this scenario's declared cut, so a live copy has to
        be charged to both engines rather than to megatron alone."""
        layer = self._layer(
            lambda hidden, mask, packed: (hidden.clone(), mask, 2)
        )
        with self.assertRaises(RuntimeError) as caught:
            _assert_unflatten_is_an_early_return(layer, MCORE_BASE_ARM_NAME)
        self.assertIn("re-cut", str(caught.exception))

    def test_a_dense_layer_is_refused_before_the_unflatten_is_examined(
        self,
    ) -> None:
        """The early return is also satisfied by a layer with no experts, so
        the MoE check has to come first or the guard would pass vacuously."""
        layer = self._layer(lambda hidden, mask, packed: (hidden, mask, None))
        layer.is_moe_layer = False
        with self.assertRaises(RuntimeError) as caught:
            _assert_unflatten_is_an_early_return(layer, MCORE_BASE_ARM_NAME)
        self.assertIn("not an MoE layer", str(caught.exception))

    def test_a_renamed_unflatten_is_refused(self) -> None:
        layer = self._layer(lambda hidden, mask, packed: (hidden, mask, None))
        del layer._maybe_unflatten_for_moe
        with self.assertRaises(RuntimeError) as caught:
            _assert_unflatten_is_an_early_return(layer, MCORE_BASE_ARM_NAME)
        self.assertIn("_maybe_unflatten_for_moe", str(caught.exception))


class ReleaseTests(unittest.TestCase):
    def _moe_layer(self) -> nn.Module:
        layer = nn.Module()
        layer.router = nn.Linear(TINY.dim, TINY.num_experts, bias=False)
        layer.experts = nn.Linear(TINY.dim, TINY.moe_hidden_dim, bias=False)
        return layer

    def test_the_experts_are_dropped_and_their_size_is_reported(self) -> None:
        """``memory_pass`` reads an absolute peak, so anything still resident
        is charged to the arm."""
        layer = self._moe_layer()
        freed = _release_untimed_moe_submodules(layer, MCORE_BASE_ARM_NAME)
        self.assertEqual(freed, TINY.dim * TINY.moe_hidden_dim)
        self.assertFalse(hasattr(layer, "experts"))
        # The router survives, because it is what the arm times.
        self.assertTrue(hasattr(layer, "router"))

    def test_a_dropped_expert_stack_raises_rather_than_reading_as_none(
        self,
    ) -> None:
        """``del`` rather than ``= None``: a megatron revision that starts
        touching ``self.experts`` must fail loudly, not read a None some other
        branch could take for 'this layer has no experts'."""
        layer = self._moe_layer()
        _release_untimed_moe_submodules(layer, MCORE_BASE_ARM_NAME)
        with self.assertRaises(AttributeError):
            layer.experts  # noqa: B018

    def test_a_layer_with_no_experts_is_refused(self) -> None:
        layer = nn.Module()
        layer.router = nn.Linear(TINY.dim, TINY.num_experts, bias=False)
        with self.assertRaises(RuntimeError):
            _release_untimed_moe_submodules(layer, MCORE_BASE_ARM_NAME)


def _titan_state(shape: PiperShape) -> dict[str, torch.Tensor]:
    """Titan's state-dict names and shapes, for the cross-engine weight map.

    ``weight_transfers`` reads a whole titan state dict, so the router key
    alone cannot run it. Everything but the router is zeros of the right
    shape, which keeps the test about the router pairing rather than about the
    other components' layouts -- those have their own tests in
    ``tests/test_megatron_weights.py``.
    """
    state = {
        "tok_embeddings.weight": torch.zeros(shape.vocab_size, shape.dim),
        "lm_head.weight": torch.zeros(shape.vocab_size, shape.dim),
        "norm.weight": torch.zeros(shape.dim),
    }
    q_out = shape.n_heads * shape.head_dim
    kv_out = shape.n_kv_heads * shape.head_dim
    for layer in range(shape.n_layers):
        prefix = f"layers.{layer}"
        state.update(
            {
                f"{prefix}.attention.qkv_linear.wq.weight": torch.zeros(
                    q_out, shape.dim
                ),
                f"{prefix}.attention.qkv_linear.wk.weight": torch.zeros(
                    kv_out, shape.dim
                ),
                f"{prefix}.attention.qkv_linear.wv.weight": torch.zeros(
                    kv_out, shape.dim
                ),
                f"{prefix}.attention_norm.weight": torch.zeros(shape.dim),
                f"{prefix}.attention.wo.weight": torch.zeros(
                    shape.dim, q_out
                ),
                f"{prefix}.attention.q_norm.weight": torch.zeros(
                    shape.head_dim
                ),
                f"{prefix}.attention.k_norm.weight": torch.zeros(
                    shape.head_dim
                ),
                f"{prefix}.ffn_norm.weight": torch.zeros(shape.dim),
                # Distinct per layer, so a map that yielded the wrong layer's
                # gate would be visible rather than plausible.
                TITAN_WEIGHT_NAME.format(layer=layer): torch.full(
                    (shape.num_experts, shape.dim), float(layer + 1)
                ),
                f"{prefix}.moe.routed_experts.inner_experts.w1_EFD": torch.zeros(
                    shape.num_experts, shape.moe_hidden_dim, shape.dim
                ),
                f"{prefix}.moe.routed_experts.inner_experts.w2_EDF": torch.zeros(
                    shape.num_experts, shape.dim, shape.moe_hidden_dim
                ),
                f"{prefix}.moe.routed_experts.inner_experts.w3_EFD": torch.zeros(
                    shape.num_experts, shape.moe_hidden_dim, shape.dim
                ),
            }
        )
    return state


class WeightMapTests(unittest.TestCase):
    """This scenario quotes the cross-engine map; it reimplements nothing."""

    def test_the_map_still_pairs_the_two_names_this_module_names(self) -> None:
        from benchmarks.models.piper_qwen3.megatron_weights import (
            weight_transfers,
        )

        state = _titan_state(TINY)
        rows = [
            (name, tensor)
            for component, name, tensor in weight_transfers(state, TINY)
            if component == WEIGHT_COMPONENT
        ]
        self.assertEqual(len(rows), TINY.n_layers)
        for layer, (name, tensor) in enumerate(rows):
            with self.subTest(layer=layer):
                self.assertEqual(name, MCORE_WEIGHT_NAME.format(layer=layer))
                # A gate matrix needs no reshape, so the map is the identity.
                self.assertTrue(
                    torch.equal(
                        tensor, state[TITAN_WEIGHT_NAME.format(layer=layer)]
                    )
                )

    def test_the_navigation_paths_agree_with_the_map(self) -> None:
        self.assertTrue(
            MCORE_WEIGHT_NAME.format(layer=LAYER).startswith(
                mcore_moe_layer_path() + "."
            )
        )
        self.assertTrue(
            mcore_moe_layer_path().startswith(
                mcore_transformer_layer_path() + "."
            )
        )


class GatingGemmPathTests(unittest.TestCase):
    """The branch ``mcore_bytes_moved`` describes, and the one it does not.

    ``RouterGatingLinearFunction.forward`` has two branches that compute the
    same numbers at different costs, so no correctness gate separates them.
    The guard is the only thing that can.
    """

    @classmethod
    def setUpClass(cls) -> None:
        if not _megatron_available():
            raise unittest.SkipTest(
                "the pinned Megatron-LM submodule is not checked out"
            )

    def test_a_gate_gemm_that_would_upcast_both_operands_is_refused(
        self,
    ) -> None:
        """The cost claim this scenario is built on, made checkable.

        With ``te_general_gemm`` unbound, ``RouterGatingLinearFunction`` takes
        the ``torch.mm`` fallback, which upcasts both operands and makes the
        fp32 copy of the hidden state that ``mcore_bytes_moved`` declares
        megatron does not make. The numbers stay right and the byte count goes
        about 5x wrong, so only a guard can see it.
        """
        from megatron.core.transformer.moe import moe_utils

        _, router = _fake_mcore_router(MCORE_BASE_ARM_NAME)
        with mock.patch.object(moe_utils, "te_general_gemm", None):
            with self.assertRaises(RuntimeError) as caught:
                _assert_mcore_gating_gemm_takes_the_te_path(
                    MCORE_BASE_ARM_NAME, router
                )
        message = str(caught.exception)
        self.assertIn("te_general_gemm", message)
        self.assertIn("mcore_bytes_moved", message)

    def test_an_fp64_router_is_refused_even_with_te_present(self) -> None:
        """The second half of megatron's own condition, which selects the same
        fallback. No profile sets fp64 today; one that did would upcast with
        TransformerEngine present and loaded."""
        _, router = _fake_mcore_router(MCORE_BASE_ARM_NAME)
        router.config.moe_router_dtype = "fp64"
        with self.assertRaises(RuntimeError) as caught:
            _assert_mcore_gating_gemm_takes_the_te_path(
                MCORE_BASE_ARM_NAME, router
            )
        self.assertIn("fp64", str(caught.exception))

    def test_the_declared_arms_pass_the_gate_gemm_guard(self) -> None:
        """Non-vacuity: the guard must accept what the registry declares.

        A guard that refused every arm would also pass the two tests above.
        """
        for arm in MCORE_ARM_PROFILES:
            with self.subTest(arm=arm):
                _, router = _fake_mcore_router(arm)
                _assert_mcore_gating_gemm_takes_the_te_path(arm, router)


class TeFusionReachabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if not _megatron_available():
            raise unittest.SkipTest(
                "the pinned Megatron-LM submodule is not checked out"
            )

    def test_a_missing_te_kernel_is_refused_at_build_time(self) -> None:
        """``topk_routing_with_score_function`` already raises rather than
        degrading, so this guard exists to fail with the reason at build time
        instead of from inside the first timed closure."""
        from megatron.core.transformer.moe import moe_utils

        with mock.patch.object(
            moe_utils, "fused_topk_with_score_function", None
        ):
            with self.assertRaises(RuntimeError) as caught:
                _assert_te_router_fusion_is_reachable(MCORE_FUSION_ARM_NAME)
        self.assertIn("TE >= 2.7.0.dev", str(caught.exception))


if __name__ == "__main__":
    unittest.main()
