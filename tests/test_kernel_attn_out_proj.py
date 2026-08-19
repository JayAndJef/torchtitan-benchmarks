"""CPU tests for the ``attn_out_proj`` kernel scenario.

The two arm builders need a GPU, megatron and torchtitan, so nothing here
builds one. What is checkable on a CPU is everything that decides whether a
GPU run means what the table says it means:

* the seeded inputs, and the fp64 truth the gates measure against;
* the agreement between this module's megatron attribute path and
  ``benchmarks.models.piper_qwen3.megatron_weights``, which is the repository's
  authority on which titan weight equals which megatron weight;
* the guard that refuses to time a module of the wrong class, the wrong shape,
  or the wrong parallel mode; and
* the shape canonicalization, which is what lets the correctness engine
  subtract a megatron THD output from a titan BSD output at all.

The last one is worth stating plainly. ``engine/correctness.py`` pairs the two
sides by output *name* and subtracts. Torch aligns shapes from the right, so a
canonicalization that returned a ``[t, 1, D]`` tensor against a ``[B, L, D]``
one raises at every batch above 1 -- measured: ``batch=4`` gives "The size of
tensor a (32) must match the size of tensor b (4)". At ``batch == 1`` the two
*do* broadcast, silently, to ``[L, L, D]``, and the gate then compares an arm
against a rearrangement of itself. So the hazard is real but narrow, and it
lands exactly where a cheap smoke run would put it. The test below runs both
native layouts through the same helper and requires identical canonical
outputs.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from benchmarks.kernel.operations.attn_out_proj import (
    MCORE_LAYER,
    MCORE_MODULE_CLASS,
    MCORE_WEIGHT_COMPONENT,
    MCORE_WEIGHT_NAME,
    TITAN_WEIGHT_NAME,
    _assert_mcore_linear_proj,
    _navigate,
    _projection_arm,
    attn_out_proj_inputs,
    attn_out_proj_reference,
    mcore_module_path,
)
from benchmarks.kernel.schema import KernelWorkload
from benchmarks.models.piper_qwen3.megatron_weights import (
    COMPONENTS,
    weight_transfers,
)
from benchmarks.models.piper_qwen3.shape import PiperShape

# Small enough to run in milliseconds, and still a real GQA geometry: 2 query
# heads over 1 kv group. ``n_heads * head_dim`` is 128, which equals ``dim``
# at every representable shape, so the tests spell the product out rather
# than reuse ``dim``.
TINY = PiperShape(name="tiny", dim=128, n_layers=2, vocab_size=64)
WORKLOAD = KernelWorkload(batch=2, seq_len=4)


def _inputs(shape: PiperShape = TINY, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return attn_out_proj_inputs(
        shape, WORKLOAD, torch.device("cpu"), generator
    )


class InputsTests(unittest.TestCase):
    def test_shapes_and_dtypes_are_what_both_engines_consume(self) -> None:
        inputs = _inputs()
        in_features = TINY.n_heads * TINY.head_dim
        self.assertEqual(
            tuple(inputs.x.shape),
            (WORKLOAD.batch, WORKLOAD.seq_len, in_features),
        )
        self.assertEqual(
            tuple(inputs.grad_out.shape),
            (WORKLOAD.batch, WORKLOAD.seq_len, TINY.dim),
        )
        self.assertEqual(tuple(inputs.weight.shape), (TINY.dim, in_features))
        # Activations are bf16 because both engines run plain bf16. The weight
        # stays fp32 so each arm casts it once, the way qkv does.
        self.assertEqual(inputs.x.dtype, torch.bfloat16)
        self.assertEqual(inputs.grad_out.dtype, torch.bfloat16)
        self.assertEqual(inputs.weight.dtype, torch.float32)

    def test_the_generator_alone_decides_the_values(self) -> None:
        """Two arms in two processes must see the same tensors.

        Every timing worker rebuilds the inputs from a seeded generator. If
        anything here read the global RNG, one arm would face different data
        than the other and the published ratio would be a ratio of two
        problems.
        """
        torch.manual_seed(1)
        first = _inputs(seed=7)
        torch.manual_seed(2)
        second = _inputs(seed=7)
        for name in ("x", "grad_out", "weight"):
            self.assertTrue(
                torch.equal(getattr(first, name), getattr(second, name)), name
            )
        different = _inputs(seed=8)
        self.assertFalse(torch.equal(first.x, different.x))


class ReferenceTests(unittest.TestCase):
    def test_the_reference_is_the_fp64_projection_and_both_gradients(
        self,
    ) -> None:
        inputs = _inputs()
        reference = attn_out_proj_reference(TINY, WORKLOAD, inputs)
        self.assertEqual(
            sorted(reference), ["out", "weight_grad", "x_grad"]
        )
        weight64 = inputs.weight.to(torch.bfloat16).double()
        x64 = inputs.x.double()
        grad64 = inputs.grad_out.double()
        torch.testing.assert_close(
            reference["out"], x64 @ weight64.T, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            reference["x_grad"], grad64 @ weight64, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            reference["weight_grad"],
            grad64.reshape(-1, TINY.dim).T @ x64.reshape(-1, weight64.shape[1]),
            rtol=1e-12,
            atol=1e-12,
        )
        self.assertEqual(reference["out"].dtype, torch.float64)

    def test_the_reference_rounds_the_weight_to_bfloat16_first(self) -> None:
        """Otherwise both arms pay for the input cast in the gate.

        The truth has to be the truth for the weight the arms hold. A value
        below is chosen so bf16 rounds it: the two projections then differ,
        and only one of them is the one to gate against.
        """
        inputs = _inputs()
        # 1 + 2**-10 needs 11 mantissa bits; bf16 has 8, so it rounds to 1.
        exact = torch.full_like(inputs.weight, 1.0 + 2.0**-10)
        inputs.weight = exact
        reference = attn_out_proj_reference(TINY, WORKLOAD, inputs)
        rounded = inputs.x.double() @ exact.to(torch.bfloat16).double().T
        unrounded = inputs.x.double() @ exact.double().T
        torch.testing.assert_close(
            reference["out"], rounded, rtol=1e-12, atol=1e-12
        )
        self.assertFalse(torch.allclose(rounded, unrounded))


class WeightMapAgreementTests(unittest.TestCase):
    """This scenario must not reimplement the titan-to-megatron map."""

    def test_the_module_path_drops_only_the_weight_suffix(self) -> None:
        self.assertEqual(
            mcore_module_path(),
            f"decoder.layers.{MCORE_LAYER}.self_attention.linear_proj",
        )
        self.assertEqual(
            mcore_module_path() + ".weight",
            MCORE_WEIGHT_NAME.format(layer=MCORE_LAYER),
        )

    def test_the_component_this_scenario_measures_is_a_declared_one(
        self,
    ) -> None:
        self.assertIn(MCORE_WEIGHT_COMPONENT, COMPONENTS)

    def test_the_map_pairs_titan_wo_with_megatron_linear_proj(self) -> None:
        """The names this module navigates are the names the map declares.

        ``weight_transfers`` is a generator over the whole model, so the test
        stops at the first ``attn_out`` entry rather than build every tensor.
        """
        state = _partial_titan_state(TINY)
        component, megatron_name, tensor = _first_attn_out(state, TINY)
        self.assertEqual(component, MCORE_WEIGHT_COMPONENT)
        self.assertEqual(
            megatron_name, MCORE_WEIGHT_NAME.format(layer=MCORE_LAYER)
        )
        titan_name = TITAN_WEIGHT_NAME.format(layer=MCORE_LAYER)
        self.assertIn(titan_name, state)
        # The same object, not an equal one: the map applies no reshape and no
        # transpose to this weight, which is why one seeded tensor can be
        # copied into both engines without a layout step.
        self.assertIs(tensor, state[titan_name])

    def test_the_shared_weight_has_the_shape_both_engines_expect(self) -> None:
        inputs = _inputs()
        state = _partial_titan_state(TINY)
        titan_weight = state[TITAN_WEIGHT_NAME.format(layer=MCORE_LAYER)]
        self.assertEqual(tuple(inputs.weight.shape), tuple(titan_weight.shape))


def _partial_titan_state(shape: PiperShape) -> dict[str, torch.Tensor]:
    """Just the titan weights ``weight_transfers`` reads before ``attn_out``.

    The generator yields the embedding, the lm head and the final norm, then
    per layer the grouped QKV, the attention norm and the output projection.
    Anything after that is never reached, so it is never built.
    """
    in_features = shape.n_heads * shape.head_dim
    kv_features = shape.n_kv_heads * shape.head_dim

    def tensor(*size: int) -> torch.Tensor:
        return torch.randn(size, dtype=torch.float32)

    state = {
        "tok_embeddings.weight": tensor(shape.vocab_size, shape.dim),
        "lm_head.weight": tensor(shape.vocab_size, shape.dim),
        "norm.weight": tensor(shape.dim),
    }
    for layer in range(shape.n_layers):
        prefix = f"layers.{layer}"
        state[f"{prefix}.attention.qkv_linear.wq.weight"] = tensor(
            in_features, shape.dim
        )
        state[f"{prefix}.attention.qkv_linear.wk.weight"] = tensor(
            kv_features, shape.dim
        )
        state[f"{prefix}.attention.qkv_linear.wv.weight"] = tensor(
            kv_features, shape.dim
        )
        state[f"{prefix}.attention_norm.weight"] = tensor(shape.dim)
        state[f"{prefix}.attention.wo.weight"] = tensor(shape.dim, in_features)
    return state


def _first_attn_out(state: dict[str, torch.Tensor], shape: PiperShape):
    for entry in weight_transfers(state, shape):
        if entry[0] == MCORE_WEIGHT_COMPONENT:
            return entry
    raise AssertionError("weight_transfers yields no attn_out entry")


class _FakeGptModel:
    """A ``GPTModel`` reduced to the attribute path this scenario walks."""

    def __init__(self, module: object, layers: int = 4) -> None:
        layer_objects = [_FakeLayer(None) for _ in range(layers)]
        layer_objects[MCORE_LAYER] = _FakeLayer(module)
        self.decoder = _FakeDecoder(layer_objects)


class _FakeDecoder:
    def __init__(self, layers: list) -> None:
        self.layers = layers


class _FakeLayer:
    def __init__(self, self_attention: object) -> None:
        self.self_attention = _FakeAttention(self_attention)


class _FakeAttention:
    def __init__(self, linear_proj: object) -> None:
        self.linear_proj = linear_proj


class TERowParallelLinear(nn.Module):
    """A stand-in with the three attributes the guard reads.

    Named for the class the guard demands, because the guard compares class
    names. A real ``TERowParallelLinear`` needs megatron, TransformerEngine
    and a CUDA device, none of which this test has.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        parallel_mode: str = "row",
        tp_size: int = 1,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.parallel_mode = parallel_mode
        self.tp_size = tp_size


class ColumnParallelLinear(TERowParallelLinear):
    pass


class NavigationTests(unittest.TestCase):
    def test_the_path_reaches_the_module_through_the_numeric_index(
        self,
    ) -> None:
        target = TERowParallelLinear(TINY.dim, TINY.n_heads * TINY.head_dim)
        model = _FakeGptModel(target)
        self.assertIs(_navigate(model, mcore_module_path()), target)

    def test_a_path_that_does_not_exist_raises(self) -> None:
        model = _FakeGptModel(TERowParallelLinear(4, 4))
        with self.assertRaises(AttributeError):
            _navigate(model, "decoder.layers.0.self_attention.linear_qkv")


class McoreGuardTests(unittest.TestCase):
    def _module(self, **kwargs) -> TERowParallelLinear:
        return TERowParallelLinear(
            TINY.dim, TINY.n_heads * TINY.head_dim, **kwargs
        )

    def test_a_correct_module_passes_and_records_no_reduce_at_one_rank(
        self,
    ) -> None:
        """The row-parallel collective is what could make this arm inert.

        TransformerEngine runs it only at ``tp_size > 1``
        (``transformer_engine/pytorch/module/linear.py:512``). At one rank
        both engines are one GEMM, which is the comparison the scenario
        claims. The guard records the fact so a reader does not have to
        assume it.
        """
        notes = _assert_mcore_linear_proj(self._module(), TINY)
        self.assertEqual(notes["module"], MCORE_MODULE_CLASS)
        self.assertEqual(notes["parallel_mode"], "row")
        self.assertEqual(notes["tp_size"], 1)
        self.assertFalse(notes["row_parallel_reduce_runs"])
        self.assertFalse(notes["compiled"])

    def test_a_module_of_another_class_is_refused(self) -> None:
        wrong = ColumnParallelLinear(TINY.dim, TINY.n_heads * TINY.head_dim)
        with self.assertRaisesRegex(RuntimeError, MCORE_MODULE_CLASS):
            _assert_mcore_linear_proj(wrong, TINY)

    def test_a_module_of_the_wrong_shape_is_refused(self) -> None:
        """A wrong attribute path lands on some other linear of the model."""
        with self.assertRaisesRegex(RuntimeError, "linear_proj.weight"):
            _assert_mcore_linear_proj(
                TERowParallelLinear(TINY.dim, TINY.dim * 3), TINY
            )

    def test_a_module_that_is_not_row_parallel_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "parallel_mode"):
            _assert_mcore_linear_proj(self._module(parallel_mode="column"), TINY)

    def test_more_than_one_rank_is_refused_rather_than_noted(self) -> None:
        """A note is not evidence: ``BuiltArm.notes`` reaches no artifact.

        At ``tp_size > 1`` TE enqueues a row-parallel all-reduce inside the
        timed call and the titan arm has no counterpart for it, so the ratio
        stops comparing two GEMMs. The guard raises instead of recording a
        fact that ``results.json`` never carries.
        """
        with self.assertRaisesRegex(RuntimeError, "tp_size"):
            _assert_mcore_linear_proj(self._module(tp_size=2), TINY)

    def test_one_rank_passes_and_reports_no_collective(self) -> None:
        notes = _assert_mcore_linear_proj(self._module(tp_size=1), TINY)
        self.assertFalse(notes["row_parallel_reduce_runs"])


class ProjectionArmTests(unittest.TestCase):
    """The helper both arms are built from, run on a plain ``nn.Linear``."""

    def _arm(self, native: str, return_owner: bool = False):
        torch.manual_seed(0)
        in_features, out_features = 8, 4
        batch, seq = 2, 3
        module = nn.Linear(in_features, out_features, bias=False)
        x = torch.arange(
            batch * seq * in_features, dtype=torch.float32
        ).reshape(batch, seq, in_features) / 100.0
        grad = torch.ones(batch, seq, out_features)
        if native == "mcore":
            x_native = x.reshape(batch * seq, 1, in_features)
            grad_native = grad.reshape(batch * seq, 1, out_features)
        else:
            x_native, grad_native = x, grad
        arm = _projection_arm(
            name=native,
            weight_owner=module,
            call=lambda leaf: module(leaf),
            x_native=x_native,
            grad_native=grad_native,
            canonical_in=(batch, seq, in_features),
            canonical_out=(batch, seq, out_features),
        )
        return (arm, module) if return_owner else arm

    def test_the_arm_exposes_the_two_declared_modes_and_no_third(self) -> None:
        """There is no isolated ``backward`` mode, on either side.

        TransformerEngine's ``_Linear.backward`` clears its saved context
        (``linear.py:1365``), so the retained-graph trick the other module
        scenarios use cannot run twice. Both arms drop the mode, which keeps
        them comparable; backward cost stays recoverable as
        ``forward_backward`` minus ``forward``.
        """
        arm = self._arm("titan")
        self.assertEqual(sorted(arm.calls), ["forward", "forward_backward"])

    def test_both_native_layouts_produce_identical_canonical_outputs(
        self,
    ) -> None:
        """THD and BSD are two labels for one GEMM, and the gates need both
        canonicalized before they subtract."""
        titan = self._arm("titan").correctness_outputs()
        mcore = self._arm("mcore").correctness_outputs()
        self.assertEqual(sorted(titan), ["out", "weight_grad", "x_grad"])
        self.assertEqual(sorted(titan), sorted(mcore))
        for name in titan:
            self.assertEqual(
                tuple(titan[name].shape), tuple(mcore[name].shape), name
            )
            torch.testing.assert_close(titan[name], mcore[name])

    def test_the_canonical_shapes_are_the_ones_the_reference_reports(
        self,
    ) -> None:
        outputs = self._arm("mcore").correctness_outputs()
        self.assertEqual(tuple(outputs["out"].shape), (2, 3, 4))
        self.assertEqual(tuple(outputs["x_grad"].shape), (2, 3, 8))
        self.assertEqual(tuple(outputs["weight_grad"].shape), (4, 8))

    def test_forward_backward_clears_gradients_between_calls(self) -> None:
        """A timed closure runs thousands of times in one burst.

        Without the reset the weight gradient would grow without bound, and
        the arm would time an addition that grows a tensor rather than the
        projection.

        Read the gradient off the module directly. ``correctness_outputs``
        calls ``_reset_grads`` itself and then runs its own forward and
        backward, so a value read through it depends only on that last call
        and holds whatever ``forward_backward`` did or did not accumulate.
        """
        arm, owner = self._arm("titan", return_owner=True)

        arm.calls["forward_backward"]()
        after_one = owner.weight.grad.clone()
        for _ in range(3):
            arm.calls["forward_backward"]()
        after_four = owner.weight.grad.clone()

        torch.testing.assert_close(after_one, after_four)
        # And prove the test can fail: four unreset backwards would be 4x.
        self.assertFalse(
            torch.allclose(after_one * 4, after_four),
            "the gradient did not accumulate, so this test proves nothing",
        )

    def test_forward_returns_the_projection(self) -> None:
        arm = self._arm("titan")
        self.assertEqual(tuple(arm.calls["forward"]().shape), (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
