"""CPU tests for the ``expert_mlp`` kernel scenario.

Everything here runs without a GPU, without megatron and without
TransformerEngine. That rules out the two things the scenario's timed arms
really are -- ``torch._grouped_mm`` needs a device, and every megatron arm
needs a process group and TE -- so the tests aim at what is left, which is
also where a wrong scenario goes wrong quietly:

* the shared inputs, including the two count forms and the routing
  probabilities that only one engine reads;
* the fp64 truth, checked against an independent batched formulation and
  against two identities the weighted branch must satisfy;
* the timed closures and the weight-gradient accessors, exercised over
  stand-in modules; and
* every guard, which is the only part of the megatron side a CPU can reach
  and the part that decides whether a published number means what its label
  says.

The stand-in modules are not a second implementation of anything under test.
They exist so ``_titan_arm``'s leaves, gradient resets and output names can be
exercised; the arithmetic they carry is checked against the fp64 reference,
which is itself checked against a formulation written differently.
"""

import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from benchmarks.kernel.operations.common import WEIGHT_STD
from benchmarks.kernel.operations.expert_mlp import (
    _assert_mcore_experts,
    _assert_te_activation_ran,
    _fused_weight_grads,
    _mcore_weight_grads,
    _require_even_routing,
    _stock_weight_grads,
    _titan_arm,
    expert_mlp_inputs,
    expert_mlp_reference,
    ExpertMlpInputs,
    GROUPED_EXPERT_CLASS,
    MCORE_FC1_WEIGHT_NAME,
    MCORE_FC2_WEIGHT_NAME,
    MCORE_LAYER,
    MCORE_WEIGHT_COMPONENT,
    mcore_expert_parameters,
    mcore_experts_path,
    NO_BIAS_ACTIVATION_FUSION_PROFILE,
    NO_GROUPED_GEMM_PROFILE,
    TE_ACTIVATION_FUNC_PROFILE,
    TITAN_STATE_KEYS,
    UNGROUPED_EXPERT_CLASS,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.mcore_profiles import BASE
from benchmarks.models.piper_qwen3.megatron_weights import weight_transfers
from benchmarks.models.piper_qwen3.shape import PiperShape, shape_by_name

# Small enough to run in milliseconds in fp64, wide enough that every expert
# gets several rows and the gated width is not degenerate.
TINY = PiperShape(name="tiny", dim=256, n_layers=2, vocab_size=64)
TINY_WORKLOAD = KernelWorkload(batch=2, seq_len=8)


def _inputs(shape=TINY, workload=TINY_WORKLOAD, seed=0) -> ExpertMlpInputs:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return expert_mlp_inputs(shape, workload, torch.device("cpu"), generator)


def _rel_l2(value: torch.Tensor, truth: torch.Tensor) -> float:
    delta = (value.double() - truth.double()).norm()
    return (delta / truth.double().norm()).item()


def _batched_expert_mlp(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    w3: torch.Tensor,
    num_experts: int,
) -> torch.Tensor:
    """The expert MLP as three ``bmm`` calls over an expert-major view.

    An independent formulation of what ``expert_mlp_reference`` computes with
    a Python loop of per-expert slices, and the one ``torch._grouped_mm``
    actually performs when every group is the same size. Written differently
    on purpose: a reference proved against a copy of itself proves nothing.
    """
    rows, dim = x.shape
    grouped = x.reshape(num_experts, rows // num_experts, dim)
    gate = torch.bmm(grouped, w1.transpose(-2, -1))
    up = torch.bmm(grouped, w3.transpose(-2, -1))
    activation = torch.nn.functional.silu(gate) * up
    return torch.bmm(activation, w2.transpose(-2, -1)).reshape(rows, dim)


class _StockStandIn(nn.Module):
    """A ``GroupedExperts``-shaped module that needs no CUDA.

    Same three parameter names and same call signature, so ``_titan_arm`` and
    ``_stock_weight_grads`` are exercised exactly as they are in a worker.
    ``torch._grouped_mm`` is device-only, so the real module cannot stand here.
    """

    def __init__(self, shape: PiperShape) -> None:
        super().__init__()
        hidden = shape.moe_hidden_dim
        self.num_experts = shape.num_experts
        self.w1_EFD = nn.Parameter(
            torch.empty(shape.num_experts, hidden, shape.dim)
        )
        self.w2_EDF = nn.Parameter(
            torch.empty(shape.num_experts, shape.dim, hidden)
        )
        self.w3_EFD = nn.Parameter(
            torch.empty(shape.num_experts, hidden, shape.dim)
        )

    def forward(self, x_RD, counts):
        del counts
        return _batched_expert_mlp(
            x_RD, self.w1_EFD, self.w2_EDF, self.w3_EFD, self.num_experts
        )


class _FusedStandIn(nn.Module):
    """The same module with w1 and w3 in one ``(E, F, 2, D)`` parameter.

    The layout all three fused titan arms use, so ``_fused_weight_grads``'s
    slice-0-is-gate convention is exercised over a real autograd gradient.
    """

    def __init__(self, shape: PiperShape) -> None:
        super().__init__()
        hidden = shape.moe_hidden_dim
        self.num_experts = shape.num_experts
        self.w13 = nn.Parameter(
            torch.empty(shape.num_experts, hidden, 2, shape.dim)
        )
        self.w2_EDF = nn.Parameter(
            torch.empty(shape.num_experts, shape.dim, hidden)
        )

    def forward(self, x_RD, counts):
        del counts
        return _batched_expert_mlp(
            x_RD,
            self.w13[:, :, 0, :],
            self.w2_EDF,
            self.w13[:, :, 1, :],
            self.num_experts,
        )


def _loaded(module: nn.Module, inputs: ExpertMlpInputs) -> nn.Module:
    with torch.no_grad():
        if hasattr(module, "w13"):
            module.w13.copy_(
                torch.stack(
                    [
                        inputs.stock_state["w1_EFD"],
                        inputs.stock_state["w3_EFD"],
                    ],
                    dim=2,
                )
            )
        else:
            module.w1_EFD.copy_(inputs.stock_state["w1_EFD"])
            module.w3_EFD.copy_(inputs.stock_state["w3_EFD"])
        module.w2_EDF.copy_(inputs.stock_state["w2_EDF"])
    module.to(torch.bfloat16)
    return module


class InputsTests(unittest.TestCase):
    def test_the_rows_are_the_routed_rows_of_the_workload(self) -> None:
        inputs = _inputs()
        rows = TINY_WORKLOAD.batch * TINY_WORKLOAD.seq_len * TINY.top_k
        self.assertEqual(tuple(inputs.x.shape), (rows, TINY.dim))
        self.assertEqual(inputs.x.dtype, torch.bfloat16)
        self.assertEqual(tuple(inputs.grad_out.shape), (rows, TINY.dim))
        self.assertEqual(inputs.grad_out.dtype, torch.bfloat16)
        self.assertEqual(
            sorted(inputs.stock_state), sorted(TITAN_STATE_KEYS)
        )

    def test_the_two_count_forms_agree_and_differ_only_in_device_and_dtype(
        self,
    ) -> None:
        """megatron reads a host tensor, titan a device one. See the module."""
        inputs = _inputs()
        self.assertEqual(inputs.counts_device.dtype, torch.int32)
        self.assertEqual(inputs.counts_host.dtype, torch.int64)
        self.assertEqual(inputs.counts_host.device.type, "cpu")
        self.assertTrue(
            torch.equal(
                inputs.counts_device.cpu().to(torch.int64), inputs.counts_host
            )
        )
        self.assertEqual(
            int(inputs.counts_host.sum()), inputs.x.shape[0]
        )

    def test_each_token_s_probabilities_sum_to_one_and_are_not_constant(
        self,
    ) -> None:
        """A constant would let an arm that dropped them pass every gate."""
        inputs = _inputs()
        self.assertEqual(tuple(inputs.probs.shape), (inputs.x.shape[0],))
        self.assertEqual(inputs.probs.dtype, torch.float32)
        pairs = inputs.probs.reshape(-1, TINY.top_k).sum(dim=-1)
        self.assertTrue(torch.allclose(pairs, torch.ones_like(pairs), atol=1e-6))
        self.assertGreater(float(inputs.probs.std()), 0.1)

    def test_the_shared_weights_are_fp32_and_scaled(self) -> None:
        """Loaded in fp32 and cast once, so every arm rounds the same values."""
        inputs = _inputs()
        for key in TITAN_STATE_KEYS:
            with self.subTest(weight=key):
                self.assertEqual(inputs.stock_state[key].dtype, torch.float32)
                self.assertAlmostEqual(
                    float(inputs.stock_state[key].std()), WEIGHT_STD, places=2
                )

    def test_one_seed_rebuilds_the_inputs_bit_identically(self) -> None:
        """Every worker rebuilds these, and the gates compare across workers."""
        first, second = _inputs(seed=7), _inputs(seed=7)
        self.assertTrue(torch.equal(first.x, second.x))
        self.assertTrue(torch.equal(first.grad_out, second.grad_out))
        self.assertTrue(torch.equal(first.probs, second.probs))
        for key in TITAN_STATE_KEYS:
            self.assertTrue(
                torch.equal(first.stock_state[key], second.stock_state[key])
            )

    def test_an_uneven_routing_split_raises_and_names_both_numbers(
        self,
    ) -> None:
        """Loudly, and never capped or rounded. See ``_require_even_routing``."""
        odd = PiperShape(name="odd", dim=256, n_layers=2, num_experts=3)
        with self.assertRaises(ValueError) as caught:
            _require_even_routing(odd, KernelWorkload(batch=1, seq_len=5))
        message = str(caught.exception)
        self.assertIn("10 routed rows", message)
        self.assertIn("3 experts", message)

    def test_the_inputs_builder_raises_on_the_same_pair(self) -> None:
        """A direct caller must not get a silently capped split either."""
        odd = PiperShape(name="odd", dim=256, n_layers=2, num_experts=3)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        with self.assertRaisesRegex(ValueError, "do not divide evenly"):
            expert_mlp_inputs(
                odd,
                KernelWorkload(batch=1, seq_len=5),
                torch.device("cpu"),
                generator,
            )


class ReferenceTests(unittest.TestCase):
    def test_the_unweighted_truth_matches_an_independent_batched_form(
        self,
    ) -> None:
        """Three ``bmm`` calls against the reference's per-expert loop."""
        inputs = _inputs()
        reference = expert_mlp_reference(TINY, TINY_WORKLOAD, inputs)
        expected = _batched_expert_mlp(
            inputs.x.double(),
            inputs.stock_state["w1_EFD"].to(torch.bfloat16).double(),
            inputs.stock_state["w2_EDF"].to(torch.bfloat16).double(),
            inputs.stock_state["w3_EFD"].to(torch.bfloat16).double(),
            TINY.num_experts,
        )
        self.assertLess(_rel_l2(reference["out"], expected), 1e-12)

    def test_the_weighted_truth_is_the_unweighted_one_scaled_by_the_probs(
        self,
    ) -> None:
        """fc2 is bias-free and the probability is a per-row scalar.

        The reference applies the probability where megatron does, between the
        activation and fc2, rather than to the output. In exact arithmetic the
        two are equal, and this is the check that the reference did put it on
        the activation of the right rows.
        """
        inputs = _inputs()
        reference = expert_mlp_reference(TINY, TINY_WORKLOAD, inputs)
        scaled = reference["out"] * inputs.probs.double().unsqueeze(-1)
        self.assertLess(_rel_l2(reference["out_weighted"], scaled), 1e-12)

    def test_the_probability_gradient_is_the_unweighted_row_inner_product(
        self,
    ) -> None:
        """``d/dp_r [p_r f(x_r)] . g_r`` is ``f(x_r) . g_r``.

        Independent of how the reference computes it, and the sharpest check
        the megatron arms get: it is identically zero for any implementation
        that dropped the routing probabilities.
        """
        inputs = _inputs()
        reference = expert_mlp_reference(TINY, TINY_WORKLOAD, inputs)
        expected = (reference["out"] * inputs.grad_out.double()).sum(dim=-1)
        self.assertLess(
            _rel_l2(reference["probs_grad_weighted"], expected), 1e-12
        )

    def test_the_two_branches_produce_different_input_gradients(self) -> None:
        """A scenario that gated the wrong branch would be caught here."""
        inputs = _inputs()
        reference = expert_mlp_reference(TINY, TINY_WORKLOAD, inputs)
        self.assertGreater(
            _rel_l2(reference["x_grad_weighted"], reference["x_grad"]), 0.1
        )

    def test_the_truth_holds_the_weights_the_arms_hold(self) -> None:
        """The fp64 truth quantizes to bf16 first, as ``qkv_prep`` does.

        A truth built from the unrounded fp32 weights would charge every arm
        for an input cast none of them performs.
        """
        inputs = _inputs()
        quantized = expert_mlp_reference(TINY, TINY_WORKLOAD, inputs)["out"]
        pre_rounded = ExpertMlpInputs(
            x=inputs.x,
            grad_out=inputs.grad_out,
            probs=inputs.probs,
            counts_device=inputs.counts_device,
            counts_host=inputs.counts_host,
            stock_state={
                key: value.to(torch.bfloat16).float()
                for key, value in inputs.stock_state.items()
            },
        )
        self.assertTrue(
            torch.equal(
                quantized,
                expert_mlp_reference(TINY, TINY_WORKLOAD, pre_rounded)["out"],
            )
        )
        self.assertGreater(
            float(
                (
                    inputs.stock_state["w1_EFD"]
                    - inputs.stock_state["w1_EFD"].to(torch.bfloat16).float()
                )
                .abs()
                .max()
            ),
            0.0,
        )


class TitanClosureTests(unittest.TestCase):
    """The closures all four titan arms share, over stand-in modules."""

    def _arm(self, inputs: ExpertMlpInputs, fused: bool = False):
        module = _loaded(
            _FusedStandIn(TINY) if fused else _StockStandIn(TINY), inputs
        )
        grads = _fused_weight_grads if fused else _stock_weight_grads
        return _titan_arm("stand_in", module, module, inputs, grads)

    def test_a_titan_arm_declares_all_three_modes(self) -> None:
        """megatron cannot expose an isolated backward; titan still does."""
        arm = self._arm(_inputs())
        self.assertEqual(
            sorted(arm.calls), ["backward", "forward", "forward_backward"]
        )

    def test_the_named_outputs_match_the_unweighted_fp64_truth(self) -> None:
        inputs = _inputs()
        outputs = self._arm(inputs).correctness_outputs()
        reference = expert_mlp_reference(TINY, TINY_WORKLOAD, inputs)
        self.assertEqual(
            sorted(outputs), ["out", "w1_grad", "w2_grad", "w3_grad", "x_grad"]
        )
        for name in ("out", "x_grad", "w1_grad", "w2_grad", "w3_grad"):
            with self.subTest(output=name):
                self.assertLess(_rel_l2(outputs[name], reference[name]), 2e-2)

    def test_the_fused_layout_reports_the_same_weight_gradients(self) -> None:
        """Slice 0 is the gate and slice 1 the up projection, for all three."""
        inputs = _inputs()
        stock = self._arm(inputs).correctness_outputs()
        fused = self._arm(inputs, fused=True).correctness_outputs()
        for name in ("w1_grad", "w2_grad", "w3_grad"):
            with self.subTest(output=name):
                self.assertLess(_rel_l2(fused[name], stock[name]), 2e-2)

    def test_a_swapped_fused_slice_would_be_caught(self) -> None:
        """The unpacking is load-bearing, so its inverse must fail the gate.

        Run against the fp64 truth rather than against the other slice.
        Asserting only that the two gradients differ is much weaker than the
        claim: two independently drawn weights give different gradients
        whether or not the unpacking is right, so that assertion passes for a
        module whose w13 slices are swapped as well.
        """
        inputs = _inputs()
        module = _loaded(_FusedStandIn(TINY), inputs)
        arm = _titan_arm(
            "stand_in", module, module, inputs, _fused_weight_grads
        )
        outputs = arm.correctness_outputs()
        truth = expert_mlp_reference(TINY, TINY_WORKLOAD, inputs)

        # The honest unpacking passes the gate the registry declares.
        for name in ("w1_grad", "w3_grad"):
            self.assertLess(_rel_l2(outputs[name], truth[name]), 2e-2, name)

        # Its inverse fails it, which is what makes the unpacking a claim.
        swapped = {"w1_grad": outputs["w3_grad"], "w3_grad": outputs["w1_grad"]}
        for name in ("w1_grad", "w3_grad"):
            self.assertGreater(_rel_l2(swapped[name], truth[name]), 2e-2, name)

    def test_a_repeated_round_trip_does_not_accumulate_a_gradient(self) -> None:
        """Every timed call must measure one backward, not a growing sum."""
        inputs = _inputs()
        arm = self._arm(inputs)
        round_trip = arm.calls["forward_backward"]
        round_trip()
        first = arm.correctness_outputs()["x_grad"].clone()
        for _ in range(3):
            round_trip()
        self.assertTrue(
            torch.equal(first, arm.correctness_outputs()["x_grad"])
        )

    def test_the_retained_backward_can_run_more_than_once(self) -> None:
        """The isolated mode re-runs one graph, which is why megatron drops it."""
        arm = self._arm(_inputs())
        for _ in range(3):
            arm.calls["backward"]()

    def test_no_titan_closure_reads_the_routing_probabilities(self) -> None:
        """The scenario's asymmetry: titan applies them in combine, not here."""
        inputs = _inputs()
        other = ExpertMlpInputs(
            x=inputs.x,
            grad_out=inputs.grad_out,
            probs=torch.zeros_like(inputs.probs),
            counts_device=inputs.counts_device,
            counts_host=inputs.counts_host,
            stock_state=inputs.stock_state,
        )
        first = self._arm(inputs).correctness_outputs()
        second = self._arm(other).correctness_outputs()
        for name in first:
            with self.subTest(output=name):
                self.assertTrue(torch.equal(first[name], second[name]))


# ---------------------------------------------------------------------------
# The megatron guards
# ---------------------------------------------------------------------------


class TEGroupedMLP(nn.Module):
    """A stand-in whose *class name* is what the guard reads."""

    def __init__(self, shape: PiperShape, config, activation=None) -> None:
        super().__init__()
        hidden = shape.moe_hidden_dim
        self.config = config
        self.num_local_experts = shape.num_experts
        self.linear_fc1 = nn.Module()
        self.linear_fc2 = nn.Module()
        for expert in range(shape.num_experts):
            self.linear_fc1.register_parameter(
                f"weight{expert}",
                nn.Parameter(torch.zeros(2 * hidden, shape.dim)),
            )
            self.linear_fc2.register_parameter(
                f"weight{expert}",
                nn.Parameter(torch.zeros(shape.dim, hidden)),
            )
        self.activation_func = (
            activation if activation is not None else torch.nn.functional.silu
        )


class SequentialMLP(nn.Module):
    """The ungrouped stand-in: ``E`` separate expert modules."""

    def __init__(self, shape: PiperShape, config, activation=None) -> None:
        super().__init__()
        hidden = shape.moe_hidden_dim
        self.config = config
        self.num_local_experts = shape.num_experts
        self.local_experts = nn.ModuleList()
        for _ in range(shape.num_experts):
            expert = nn.Module()
            expert.linear_fc1 = nn.Linear(shape.dim, 2 * hidden, bias=False)
            expert.linear_fc2 = nn.Linear(hidden, shape.dim, bias=False)
            self.local_experts.append(expert)
        self.activation_func = (
            activation if activation is not None else torch.nn.functional.silu
        )


class _TransformerEngineSwiGLU(nn.Module):
    """Stands in for ``transformer_engine.pytorch.ops.SwiGLU``.

    The guard reads ``type(module).__module__``, so the stand-in has to claim
    the origin rather than the name.
    """

    __module__ = "transformer_engine.pytorch.ops.basic.swiglu"

    def forward(self, x):
        gate, up = torch.chunk(x, 2, dim=-1)
        return torch.nn.functional.silu(gate) * up


def _fake_config(profile, **overrides):
    """A ``TransformerConfig`` stand-in carrying what the guard reads."""
    values = {
        "moe_apply_probs_on_input": False,
        "moe_mlp_glu_interleave_size": None,
        "use_transformer_engine_op_fuser": False,
        "fp8": None,
        "fp4": None,
        "moe_latent_size": None,
        "recompute_granularity": None,
        "recompute_modules": None,
        "use_te_activation_func": False,
    }
    values.update(profile.config_overrides)
    values.update(overrides)
    return types.SimpleNamespace(**values)


def _grouped(profile=BASE, shape=TINY, activation=None, **overrides):
    return TEGroupedMLP(shape, _fake_config(profile, **overrides), activation)


def _ungrouped(profile=NO_GROUPED_GEMM_PROFILE, shape=TINY, **overrides):
    return SequentialMLP(shape, _fake_config(profile, **overrides))


class McoreGuardTests(unittest.TestCase):
    def test_the_base_layer_passes_and_reports_what_it_is(self) -> None:
        notes = _assert_mcore_experts("arm", _grouped(), BASE, TINY)
        self.assertEqual(notes["module"], GROUPED_EXPERT_CLASS)
        self.assertTrue(notes["moe_grouped_gemm"])
        self.assertFalse(notes["use_te_activation_func"])

    def test_the_ungrouped_layer_passes_under_its_own_profile(self) -> None:
        notes = _assert_mcore_experts(
            "arm", _ungrouped(), NO_GROUPED_GEMM_PROFILE, TINY
        )
        self.assertEqual(notes["module"], UNGROUPED_EXPERT_CLASS)
        self.assertFalse(notes["moe_grouped_gemm"])

    def test_a_grouped_arm_that_built_the_ungrouped_class_is_refused(
        self,
    ) -> None:
        """The TE fallback would make two arms the same module.

        ``TESpecProvider.grouped_mlp_modules`` builds ``SequentialMLP`` when
        ``TEColumnParallelGroupedLinear`` is absent, so without this guard
        ``mcore/base`` and ``mcore/no_grouped_gemm`` would publish a ratio of
        one as "the grouped kernel buys nothing".
        """
        layer = _ungrouped(profile=BASE)
        with self.assertRaisesRegex(
            RuntimeError, "TEColumnParallelGroupedLinear"
        ):
            _assert_mcore_experts("arm", layer, BASE, TINY)

    def test_an_ungrouped_arm_that_built_the_grouped_class_is_refused(
        self,
    ) -> None:
        """The config field alone was inert under a hand-built layer spec."""
        layer = _grouped(profile=NO_GROUPED_GEMM_PROFILE)
        with self.assertRaisesRegex(RuntimeError, "not SequentialMLP"):
            _assert_mcore_experts("arm", layer, NO_GROUPED_GEMM_PROFILE, TINY)

    def test_a_grouped_gemm_flag_that_did_not_take_is_refused(self) -> None:
        """FUSION_FIELDS does not cover it, so declared_mismatches cannot."""
        layer = _ungrouped(
            profile=NO_GROUPED_GEMM_PROFILE, moe_grouped_gemm=True
        )
        with self.assertRaisesRegex(RuntimeError, "moe_grouped_gemm"):
            _assert_mcore_experts("arm", layer, NO_GROUPED_GEMM_PROFILE, TINY)

    def test_a_te_activation_flag_that_did_not_take_is_refused(self) -> None:
        layer = _grouped(
            profile=TE_ACTIVATION_FUNC_PROFILE,
            activation=_TransformerEngineSwiGLU(),
            use_te_activation_func=False,
        )
        with self.assertRaisesRegex(RuntimeError, "use_te_activation_func"):
            _assert_mcore_experts(
                "arm", layer, TE_ACTIVATION_FUNC_PROFILE, TINY
            )

    def test_the_te_activation_arm_passes_with_a_te_module(self) -> None:
        notes = _assert_mcore_experts(
            "arm",
            _grouped(
                profile=TE_ACTIVATION_FUNC_PROFILE,
                activation=_TransformerEngineSwiGLU(),
            ),
            TE_ACTIVATION_FUNC_PROFILE,
            TINY,
        )
        self.assertTrue(notes["use_te_activation_func"])

    def test_the_te_activation_arm_is_refused_without_a_module(self) -> None:
        """``TEGroupedMLP`` falls back to ``config.activation_func``."""
        with self.assertRaisesRegex(RuntimeError, "not a module"):
            _assert_mcore_experts(
                "arm",
                _grouped(profile=TE_ACTIVATION_FUNC_PROFILE),
                TE_ACTIVATION_FUNC_PROFILE,
                TINY,
            )

    def test_a_module_from_somewhere_other_than_te_is_refused(self) -> None:
        class LocalSwiGLU(nn.Module):
            def forward(self, x):
                return x

        with self.assertRaisesRegex(RuntimeError, "transformer_engine"):
            _assert_mcore_experts(
                "arm",
                _grouped(
                    profile=TE_ACTIVATION_FUNC_PROFILE,
                    activation=LocalSwiGLU(),
                ),
                TE_ACTIVATION_FUNC_PROFILE,
                TINY,
            )

    def test_a_plain_arm_carrying_a_module_activation_is_refused(self) -> None:
        """megatron's fused branch tests ``activation_func == F.silu``."""
        with self.assertRaisesRegex(RuntimeError, "declares the plain path"):
            _assert_mcore_experts(
                "arm",
                _grouped(activation=_TransformerEngineSwiGLU()),
                BASE,
                TINY,
            )

    def test_the_te_activation_arm_is_refused_beside_the_fused_swiglu(
        self,
    ) -> None:
        """The two-flag delta: TransformerConfig refuses the pair itself."""
        layer = _grouped(
            profile=TE_ACTIVATION_FUNC_PROFILE,
            activation=_TransformerEngineSwiGLU(),
            bias_activation_fusion=True,
        )
        with self.assertRaisesRegex(
            RuntimeError, "bias_activation_fusion"
        ):
            _assert_mcore_experts(
                "arm", layer, TE_ACTIVATION_FUNC_PROFILE, TINY
            )

    def test_a_fusion_flag_that_did_not_take_is_refused(self) -> None:
        """The declared_mismatches half, in the direction that handicaps."""
        layer = _grouped(
            profile=NO_BIAS_ACTIVATION_FUSION_PROFILE,
            bias_activation_fusion=True,
        )
        with self.assertRaisesRegex(RuntimeError, "did not take"):
            _assert_mcore_experts(
                "arm", layer, NO_BIAS_ACTIVATION_FUSION_PROFILE, TINY
            )

    def test_moving_the_probabilities_to_the_input_is_refused(self) -> None:
        """It resets them to ones, which is a different function."""
        with self.assertRaisesRegex(RuntimeError, "moe_apply_probs_on_input"):
            _assert_mcore_experts(
                "arm",
                _grouped(moe_apply_probs_on_input=True),
                BASE,
                TINY,
            )

    def test_the_operation_fuser_path_is_refused(self) -> None:
        """``TEGroupedMLP.forward`` takes ``_fused_forward`` instead."""
        with self.assertRaisesRegex(
            RuntimeError, "use_transformer_engine_op_fuser"
        ):
            _assert_mcore_experts(
                "arm",
                _grouped(use_transformer_engine_op_fuser=True),
                BASE,
                TINY,
            )

    def test_the_activation_recompute_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "activation recompute"):
            _assert_mcore_experts(
                "arm",
                _grouped(
                    recompute_granularity="selective",
                    recompute_modules=["moe_act"],
                ),
                BASE,
                TINY,
            )

    def test_a_partial_expert_roster_is_refused(self) -> None:
        """An expert-parallel split would measure a fraction of the layer."""
        layer = _grouped()
        layer.num_local_experts = TINY.num_experts - 1
        with self.assertRaisesRegex(RuntimeError, "local experts"):
            _assert_mcore_experts("arm", layer, BASE, TINY)

    def test_a_weight_of_the_wrong_shape_is_refused(self) -> None:
        """A wrong navigation hands back another linear of the same model."""
        layer = _grouped()
        layer.linear_fc1.weight0 = nn.Parameter(torch.zeros(3, TINY.dim))
        with self.assertRaisesRegex(RuntimeError, "fc1 weight is"):
            _assert_mcore_experts("arm", layer, BASE, TINY)


class ExpertParameterTests(unittest.TestCase):
    def test_both_class_spellings_yield_the_same_pair_count(self) -> None:
        for layer in (_grouped(), _ungrouped()):
            with self.subTest(module=type(layer).__name__):
                pairs = mcore_expert_parameters(layer, TINY)
                self.assertEqual(len(pairs), TINY.num_experts)
                for fc1, fc2 in pairs:
                    self.assertEqual(
                        tuple(fc1.shape), (2 * TINY.moe_hidden_dim, TINY.dim)
                    )
                    self.assertEqual(
                        tuple(fc2.shape), (TINY.dim, TINY.moe_hidden_dim)
                    )

    def test_a_packed_grouped_weight_raises_rather_than_loading_none(
        self,
    ) -> None:
        """TE's ``single_grouped_weight`` sets every ``weight{i}`` to None."""
        layer = _grouped()
        layer.linear_fc1.weight0 = None
        with self.assertRaisesRegex(RuntimeError, "single_grouped_weight"):
            mcore_expert_parameters(layer, TINY)

    def test_the_weight_gradients_are_returned_in_titan_s_layout(self) -> None:
        """The inverse of the weight map's ``cat([w1[e], w3[e]])``."""
        hidden = TINY.moe_hidden_dim
        layer = _grouped()
        for expert, (fc1, fc2) in enumerate(
            mcore_expert_parameters(layer, TINY)
        ):
            fc1.grad = torch.full_like(fc1, float(expert))
            fc1.grad[hidden:] += 0.5
            fc2.grad = torch.full_like(fc2, float(expert) + 10.0)
        grads = _mcore_weight_grads(layer, TINY)
        self.assertEqual(
            tuple(grads["w1_grad_weighted"].shape),
            (TINY.num_experts, hidden, TINY.dim),
        )
        self.assertEqual(
            tuple(grads["w2_grad_weighted"].shape),
            (TINY.num_experts, TINY.dim, hidden),
        )
        for expert in range(TINY.num_experts):
            self.assertAlmostEqual(
                float(grads["w1_grad_weighted"][expert].mean()), float(expert)
            )
            self.assertAlmostEqual(
                float(grads["w3_grad_weighted"][expert].mean()),
                float(expert) + 0.5,
            )

    def test_a_missing_gradient_reports_none_rather_than_raising(self) -> None:
        """``_require_grads`` is what names it, and it names all three."""
        grads = _mcore_weight_grads(_grouped(), TINY)
        self.assertEqual(set(grads.values()), {None})


class TeActivationProbeTests(unittest.TestCase):
    """The hook probe, which proves the module runs rather than exists."""

    class _RunsIt(nn.Module):
        def __init__(self, activation):
            super().__init__()
            self.activation_func = activation

        def forward(self, x, counts, probs):
            del counts, probs
            return self.activation_func(torch.cat([x, x], dim=-1)), None

    class _SkipsIt(nn.Module):
        def __init__(self, activation):
            super().__init__()
            self.activation_func = activation

        def forward(self, x, counts, probs):
            del counts, probs
            return x, None

    def test_a_layer_that_calls_the_module_passes(self) -> None:
        inputs = _inputs()
        _assert_te_activation_ran(
            "arm", self._RunsIt(_TransformerEngineSwiGLU()), inputs
        )

    def test_a_layer_that_holds_it_but_never_calls_it_is_refused(self) -> None:
        """Being picked and being reached are two different claims."""
        inputs = _inputs()
        with self.assertRaisesRegex(RuntimeError, "never ran"):
            _assert_te_activation_ran(
                "arm", self._SkipsIt(_TransformerEngineSwiGLU()), inputs
            )

    def test_the_probe_removes_its_hook(self) -> None:
        """It runs at build time; a surviving hook would reach a timed call."""
        layer = self._RunsIt(_TransformerEngineSwiGLU())
        _assert_te_activation_ran("arm", layer, _inputs())
        self.assertEqual(len(layer.activation_func._forward_hooks), 0)


class ProfileDeltaTests(unittest.TestCase):
    def _delta(self, profile) -> dict:
        return {
            key: value
            for key, value in profile.config_overrides.items()
            if BASE.config_overrides.get(key, "<absent>") != value
        }

    def test_the_te_activation_profile_is_a_two_flag_delta(self) -> None:
        """One flag would build a config megatron refuses outright."""
        self.assertEqual(
            self._delta(TE_ACTIVATION_FUNC_PROFILE),
            {"use_te_activation_func": True, "bias_activation_fusion": False},
        )

    def test_the_other_two_profiles_change_exactly_one_flag_each(self) -> None:
        self.assertEqual(
            self._delta(NO_BIAS_ACTIVATION_FUSION_PROFILE),
            {"bias_activation_fusion": False},
        )
        self.assertEqual(
            self._delta(NO_GROUPED_GEMM_PROFILE), {"moe_grouped_gemm": False}
        )

    def test_every_delta_keeps_the_base_geometry_and_precision(self) -> None:
        """A profile carries behaviour; geometry comes from the shape."""
        for profile in (
            NO_BIAS_ACTIVATION_FUSION_PROFILE,
            TE_ACTIVATION_FUNC_PROFILE,
            NO_GROUPED_GEMM_PROFILE,
        ):
            with self.subTest(profile=profile.name):
                self.assertEqual(
                    set(profile.config_overrides),
                    set(BASE.config_overrides) | {"use_te_activation_func"}
                    if profile is TE_ACTIVATION_FUNC_PROFILE
                    else set(BASE.config_overrides),
                )
                self.assertEqual(
                    profile.config_overrides["params_dtype"], "bfloat16"
                )


class WeightMapTests(unittest.TestCase):
    """This module restates the map's concatenation; the map stays authority."""

    def _state(self, shape: PiperShape) -> dict:
        hidden = shape.moe_hidden_dim
        state = {
            "tok_embeddings.weight": torch.zeros(shape.vocab_size, shape.dim),
            "lm_head.weight": torch.zeros(shape.vocab_size, shape.dim),
            "norm.weight": torch.zeros(shape.dim),
        }
        for layer in range(shape.n_layers):
            prefix = f"layers.{layer}"
            state[f"{prefix}.attention.qkv_linear.wq.weight"] = torch.zeros(
                shape.n_heads * shape.head_dim, shape.dim
            )
            for name in ("wk", "wv"):
                state[f"{prefix}.attention.qkv_linear.{name}.weight"] = (
                    torch.zeros(shape.n_kv_heads * shape.head_dim, shape.dim)
                )
            state[f"{prefix}.attention_norm.weight"] = torch.zeros(shape.dim)
            state[f"{prefix}.attention.wo.weight"] = torch.zeros(
                shape.dim, shape.dim
            )
            for name in ("q_norm", "k_norm"):
                state[f"{prefix}.attention.{name}.weight"] = torch.zeros(
                    shape.head_dim
                )
            state[f"{prefix}.ffn_norm.weight"] = torch.zeros(shape.dim)
            state[f"{prefix}.moe.router.gate.weight"] = torch.zeros(
                shape.num_experts, shape.dim
            )
            base = f"{prefix}.moe.routed_experts.inner_experts"
            state[f"{base}.w1_EFD"] = torch.randn(
                shape.num_experts, hidden, shape.dim
            )
            state[f"{base}.w2_EDF"] = torch.randn(
                shape.num_experts, shape.dim, hidden
            )
            state[f"{base}.w3_EFD"] = torch.randn(
                shape.num_experts, hidden, shape.dim
            )
        return state

    def test_the_map_still_names_and_builds_what_this_module_loads(
        self,
    ) -> None:
        """fc1 is ``cat([w1, w3])`` and fc2 is ``w2``, per expert."""
        state = self._state(TINY)
        base = f"layers.{MCORE_LAYER}.moe.routed_experts.inner_experts"
        found = {
            name: tensor
            for component, name, tensor in weight_transfers(state, TINY)
            if component == MCORE_WEIGHT_COMPONENT
        }
        for expert in range(TINY.num_experts):
            fc1 = MCORE_FC1_WEIGHT_NAME.format(layer=MCORE_LAYER, expert=expert)
            fc2 = MCORE_FC2_WEIGHT_NAME.format(layer=MCORE_LAYER, expert=expert)
            with self.subTest(expert=expert):
                self.assertTrue(
                    torch.equal(
                        found[fc1],
                        torch.cat(
                            [
                                state[f"{base}.w1_EFD"][expert],
                                state[f"{base}.w3_EFD"][expert],
                            ],
                            dim=0,
                        ),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        found[fc2], state[f"{base}.w2_EDF"][expert]
                    )
                )

    def test_the_navigation_path_is_derived_from_the_mapped_name(self) -> None:
        """So the module tree this walks and the map cannot drift apart."""
        self.assertEqual(
            MCORE_FC1_WEIGHT_NAME.format(layer=3, expert=0),
            f"{mcore_experts_path(3)}.linear_fc1.weight0",
        )


class RegisteredShapeTests(unittest.TestCase):
    def test_the_inputs_build_at_the_normal_shape(self) -> None:
        shape = shape_by_name("normal")
        workload = KernelWorkload(batch=1, seq_len=2)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        inputs = expert_mlp_inputs(
            shape, workload, torch.device("cpu"), generator
        )
        rows = workload.batch * workload.seq_len * shape.top_k
        self.assertEqual(tuple(inputs.x.shape), (rows, shape.dim))
        self.assertEqual(
            tuple(inputs.stock_state["w1_EFD"].shape),
            (shape.num_experts, shape.moe_hidden_dim, shape.dim),
        )
        self.assertEqual(tuple(inputs.probs.shape), (rows,))

    def test_the_huge_shape_is_not_built_here_and_the_reason_is_its_size(
        self,
    ) -> None:
        """``--model-size huge`` is recorded as untested, not attempted.

        The shared fp32 weight state is three ``(E, F, D)`` tensors, and at the
        huge shape that is tens of gibibytes before an arm materializes its own
        bf16 copy. Building it in a CPU unit test would make the suite unusable
        and would prove nothing a GPU has not already refused to run. This
        asserts the figure the module docstring quotes, from the registered
        shape, so a shape change moves the warning with it.
        """
        shape = shape_by_name("huge")
        per_tensor_bytes = (
            shape.num_experts * shape.moe_hidden_dim * shape.dim * 4
        )
        self.assertAlmostEqual(3 * per_tensor_bytes / 2**30, 23.6, places=1)
        # And the fp64 reference's weight gradients, which the correctness
        # worker holds resident while it builds all eight arms.
        self.assertAlmostEqual(3 * per_tensor_bytes * 2 / 2**30, 47.3, places=1)


if __name__ == "__main__":
    unittest.main()
