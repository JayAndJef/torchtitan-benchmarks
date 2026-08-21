"""CPU tests for the ``qkv_prep`` kernel scenario.

The three arm builders need a GPU, megatron and torchtitan, so nothing here
builds one. What is checkable on a CPU is everything that decides whether a
GPU run means what the published table says it means:

* the seeded inputs, and the fp64 truth every gate measures against;
* the agreement between this module's megatron attribute path and
  ``benchmarks.models.piper_qwen3.megatron_weights``, which is the repository's
  authority on which titan weight equals which megatron weight;
* the guards that refuse to time a module of the wrong class, the wrong norm,
  the wrong shape or the wrong parallel mode;
* the removal of the qk norms, which is what keeps this scenario from
  measuring scenario 3's territory a second time; and
* the shape canonicalization, which is what lets the correctness engine
  subtract a megatron THD output from a titan BSD output at all.

The last one is worth stating plainly. ``engine/correctness.py`` pairs the two
sides by output *name* and subtracts. Torch aligns shapes from the right, so a
canonicalization that returned a ``[t, 1, N, H]`` tensor against a
``[B, L, N, H]`` one raises at every batch above 1, and at ``batch == 1`` the
two broadcast silently. The test below runs both native layouts through the
same helper and requires identical canonical outputs.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn

from benchmarks.kernel.operations.qkv_prep import (
    MCORE_ARM_NAME,
    MCORE_LAYER,
    MCORE_MODULE_CLASS,
    MCORE_NORM_WEIGHT_NAME,
    MCORE_WEIGHT_COMPONENT,
    MCORE_WEIGHT_NAME,
    NORM_EPS,
    QKV_STATE_KEYS,
    TITAN_ARM_NAME,
    TITAN_NORM_WEIGHT_NAME,
    TITAN_UNFUSED_ARM_NAME,
    TITAN_WEIGHT_NAMES,
    _assert_mcore_qkv_prep,
    _drop_qk_layernorms,
    _navigate,
    _qkv_prep_arm,
    _require_grads,
    _TitanQkvPrep,
    _weight_grad_outputs,
    mcore_attention_path,
    qkv_prep_inputs,
    qkv_prep_reference,
)
from benchmarks.kernel.schema import KernelWorkload, fragment_stem
from benchmarks.models.piper_qwen3.megatron_weights import (
    COMPONENTS,
    grouped_qkv,
    weight_transfers,
)
from benchmarks.models.piper_qwen3.shape import PiperShape

# Small enough to run in milliseconds, and still a real GQA geometry: 2 query
# heads over 1 kv group, so the grouped interleave is not the identity.
TINY = PiperShape.derived(name="tiny", dim=128, n_layers=2, vocab_size=64)
WORKLOAD = KernelWorkload(batch=2, seq_len=4)


def _inputs(shape: PiperShape = TINY, seed: int = 0):
    generator = torch.Generator().manual_seed(seed)
    return qkv_prep_inputs(shape, WORKLOAD, torch.device("cpu"), generator)


class InputsTests(unittest.TestCase):
    def test_shapes_and_dtypes_are_what_all_three_arms_consume(self) -> None:
        inputs = _inputs()
        batch, seq = WORKLOAD.batch, WORKLOAD.seq_len
        self.assertEqual(tuple(inputs.x.shape), (batch, seq, TINY.dim))
        self.assertEqual(
            tuple(inputs.grad_q.shape),
            (batch, seq, TINY.n_heads, TINY.head_dim),
        )
        self.assertEqual(
            tuple(inputs.grad_k.shape),
            (batch, seq, TINY.n_kv_heads, TINY.head_dim),
        )
        self.assertEqual(
            tuple(inputs.grad_v.shape), tuple(inputs.grad_k.shape)
        )
        self.assertEqual(sorted(inputs.weight_state), sorted(QKV_STATE_KEYS))
        self.assertEqual(
            tuple(inputs.weight_state["wq.weight"].shape),
            (TINY.n_heads * TINY.head_dim, TINY.dim),
        )
        self.assertEqual(
            tuple(inputs.weight_state["wk.weight"].shape),
            (TINY.n_kv_heads * TINY.head_dim, TINY.dim),
        )
        self.assertEqual(tuple(inputs.norm_weight.shape), (TINY.dim,))
        # Activations are bf16 because both engines run plain bf16. Every
        # parameter stays fp32 so each arm rounds it exactly once, the way
        # ``expert_mlp`` and ``ffn_norm`` do.
        self.assertEqual(inputs.x.dtype, torch.bfloat16)
        self.assertEqual(inputs.grad_q.dtype, torch.bfloat16)
        self.assertEqual(inputs.weight_state["wq.weight"].dtype, torch.float32)
        self.assertEqual(inputs.norm_weight.dtype, torch.float32)

    def test_the_norm_gain_is_not_exactly_ones(self) -> None:
        """A gain of ones cannot show a gain an arm never loaded.

        Both engines initialize an RMSNorm gain to exactly ones, so an arm
        whose ``load`` silently did nothing would still pass every gate. The
        shared gain is therefore ``1 + N(0, WEIGHT_STD)``, near 1.0 but not
        equal to it.
        """
        gain = _inputs().norm_weight
        self.assertFalse(torch.equal(gain, torch.ones_like(gain)))
        self.assertLess((gain - 1.0).abs().max().item(), 0.5)

    def test_the_declared_epsilon_comes_from_the_mcore_profile(self) -> None:
        self.assertEqual(_inputs().eps, NORM_EPS)
        self.assertEqual(NORM_EPS, 1e-6)

    def test_the_generator_alone_decides_the_values(self) -> None:
        """Three arms in three processes must see the same tensors.

        Every timing worker rebuilds the inputs from a seeded generator. If
        anything here read the global RNG, one arm would face different data
        than another and the published ratio would be a ratio of two problems.
        """
        torch.manual_seed(1)
        first = _inputs(seed=7)
        torch.manual_seed(2)
        second = _inputs(seed=7)
        for name in ("x", "grad_q", "grad_k", "grad_v", "norm_weight"):
            self.assertTrue(
                torch.equal(getattr(first, name), getattr(second, name)), name
            )
        for key in QKV_STATE_KEYS:
            self.assertTrue(
                torch.equal(first.weight_state[key], second.weight_state[key]),
                key,
            )
        self.assertFalse(torch.equal(first.x, _inputs(seed=8).x))


def _expected_reference(shape: PiperShape, inputs) -> dict[str, torch.Tensor]:
    """The fp64 truth, recomputed here rather than reused from the module."""
    batch, seq = WORKLOAD.batch, WORKLOAD.seq_len
    x = inputs.x.double()
    gain = inputs.norm_weight.to(torch.bfloat16).double()
    hidden = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + inputs.eps) * gain

    def project(key: str, heads: int) -> torch.Tensor:
        weight = inputs.weight_state[key].to(torch.bfloat16).double()
        return (hidden @ weight.T).view(batch, seq, heads, shape.head_dim)

    return {
        "q_out": project("wq.weight", shape.n_heads),
        "k_out": project("wk.weight", shape.n_kv_heads),
        "v_out": project("wv.weight", shape.n_kv_heads),
    }


class ReferenceTests(unittest.TestCase):
    def test_the_reference_names_every_output_the_gates_compare(self) -> None:
        reference = qkv_prep_reference(TINY, WORKLOAD, _inputs())
        self.assertEqual(
            sorted(reference),
            [
                "k_out",
                "norm_weight_grad",
                "q_out",
                "qkv_weight_grad",
                "v_out",
                "x_grad",
            ],
        )
        self.assertEqual(reference["q_out"].dtype, torch.float64)

    def test_the_reference_is_the_fp64_norm_then_projection(self) -> None:
        inputs = _inputs()
        reference = qkv_prep_reference(TINY, WORKLOAD, inputs)
        expected = _expected_reference(TINY, inputs)
        for name, value in expected.items():
            torch.testing.assert_close(
                reference[name], value, rtol=1e-12, atol=1e-12
            )

    def test_the_reference_rounds_every_parameter_to_bfloat16_first(
        self,
    ) -> None:
        """Otherwise every arm pays for the input cast inside the gate.

        The truth has to be the truth for the parameters the arms hold.
        ``1 + 2**-10`` needs 11 mantissa bits and bf16 has 8, so it rounds to
        1.0; the two projections then differ, and only one of them is the one
        to gate against.
        """
        inputs = _inputs()
        unrepresentable = 1.0 + 2.0**-10
        inputs.norm_weight = torch.full_like(
            inputs.norm_weight, unrepresentable
        )
        for key in QKV_STATE_KEYS:
            inputs.weight_state[key] = torch.full_like(
                inputs.weight_state[key], unrepresentable
            )
        reference = qkv_prep_reference(TINY, WORKLOAD, inputs)
        torch.testing.assert_close(
            reference["q_out"],
            _expected_reference(TINY, inputs)["q_out"],
            rtol=1e-12,
            atol=1e-12,
        )
        # And prove the rounding matters: the unrounded projection differs.
        x = inputs.x.double()
        gain = inputs.norm_weight.double()
        hidden = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + inputs.eps) * gain
        )
        unrounded = (
            hidden @ inputs.weight_state["wq.weight"].double().T
        ).view(*reference["q_out"].shape)
        self.assertFalse(torch.allclose(reference["q_out"], unrounded))

    def test_the_weight_gradient_is_reported_in_the_grouped_interleave(
        self,
    ) -> None:
        """The one layout all three arms can be compared in.

        Megatron holds one ``linear_qkv.weight`` and titan's fused module
        holds one ``wqkv.weight`` in the same interleave, so the grouped form
        is the only shape the three arms share. The reference builds it with
        ``megatron_weights.grouped_qkv``, which proves its own inverse on
        every call, so nothing here reimplements the reshape.
        """
        inputs = _inputs()
        reference = qkv_prep_reference(TINY, WORKLOAD, inputs)
        grouped = reference["qkv_weight_grad"]
        self.assertEqual(tuple(grouped.shape), (TINY.qkv_out_features, TINY.dim))
        view = grouped.view(
            TINY.n_kv_heads, TINY.heads_per_group + 2, TINY.head_dim, TINY.dim
        )
        # The three slices the interleave is built from must come back, which
        # is also what a wrong reshape here would break.
        self.assertEqual(
            tuple(view[:, : TINY.heads_per_group].reshape(-1, TINY.dim).shape),
            tuple(inputs.weight_state["wq.weight"].shape),
        )
        rebuilt = grouped_qkv(
            view[:, : TINY.heads_per_group].reshape(-1, TINY.dim),
            view[:, TINY.heads_per_group].reshape(-1, TINY.dim),
            view[:, TINY.heads_per_group + 1].reshape(-1, TINY.dim),
            TINY,
        )
        self.assertTrue(torch.equal(rebuilt, grouped))

    def test_the_gradients_are_the_autograd_gradients_of_that_function(
        self,
    ) -> None:
        """A hand-written backward would be a second implementation to prove.

        This recomputes the same fp64 forward here and differentiates it, so
        the test fails if the reference ever stops differentiating the exact
        expression it publishes.
        """
        inputs = _inputs()
        reference = qkv_prep_reference(TINY, WORKLOAD, inputs)
        x = inputs.x.double().detach().requires_grad_()
        gain = (
            inputs.norm_weight.to(torch.bfloat16)
            .double()
            .detach()
            .requires_grad_()
        )
        hidden = (
            x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + inputs.eps) * gain
        )
        heads = (TINY.n_heads, TINY.n_kv_heads, TINY.n_kv_heads)
        outputs = []
        for key, count in zip(QKV_STATE_KEYS, heads):
            weight = inputs.weight_state[key].to(torch.bfloat16).double()
            outputs.append(
                (hidden @ weight.T).view(
                    WORKLOAD.batch, WORKLOAD.seq_len, count, TINY.head_dim
                )
            )
        torch.autograd.backward(
            tuple(outputs),
            (
                inputs.grad_q.double(),
                inputs.grad_k.double(),
                inputs.grad_v.double(),
            ),
        )
        torch.testing.assert_close(
            reference["x_grad"], x.grad, rtol=1e-12, atol=1e-12
        )
        torch.testing.assert_close(
            reference["norm_weight_grad"], gain.grad, rtol=1e-12, atol=1e-12
        )


class ArmNameTests(unittest.TestCase):
    def test_every_arm_name_survives_becoming_a_filename(self) -> None:
        """Two of the three names hold a slash, which a fragment path cannot.

        ``fragment_path`` writes ``fragments/timing__{arm}__r{n}.json``, so a
        raw slash would make the timing worker write into a directory nobody
        creates and the arm would land as ``failed``. ``fragment_stem`` is what
        makes these names legal, and the stems must still differ.
        """
        names = (MCORE_ARM_NAME, TITAN_ARM_NAME, TITAN_UNFUSED_ARM_NAME)
        stems = [fragment_stem(name) for name in names]
        self.assertEqual(len(set(stems)), len(names))
        for stem in stems:
            self.assertNotIn("/", stem)


class WeightMapAgreementTests(unittest.TestCase):
    """This scenario must not reimplement the titan-to-megatron map."""

    def test_the_attention_path_drops_only_the_linear_and_weight_suffix(
        self,
    ) -> None:
        self.assertEqual(
            mcore_attention_path(),
            f"decoder.layers.{MCORE_LAYER}.self_attention",
        )
        self.assertEqual(
            mcore_attention_path() + ".linear_qkv.weight",
            MCORE_WEIGHT_NAME.format(layer=MCORE_LAYER),
        )
        self.assertEqual(
            mcore_attention_path() + ".linear_qkv.layer_norm_weight",
            MCORE_NORM_WEIGHT_NAME.format(layer=MCORE_LAYER),
        )

    def test_the_component_this_scenario_measures_is_a_declared_one(
        self,
    ) -> None:
        self.assertIn(MCORE_WEIGHT_COMPONENT, COMPONENTS)

    def test_the_map_yields_both_qkv_parameters_this_arm_writes(self) -> None:
        """The names this module navigates are the names the map declares.

        The map yields two entries under the ``qkv`` tag per layer: the
        grouped projection matrix, and the attention-input norm gain, which is
        tagged ``qkv`` precisely because megatron holds no separate module for
        it. Both are what this arm copies into.
        """
        names = list(_qkv_entries(_partial_titan_state(TINY), TINY))
        self.assertIn(MCORE_WEIGHT_NAME.format(layer=MCORE_LAYER), names)
        self.assertIn(MCORE_NORM_WEIGHT_NAME.format(layer=MCORE_LAYER), names)

    def test_the_map_reads_the_titan_names_this_module_declares(self) -> None:
        state = _partial_titan_state(TINY)
        for template in TITAN_WEIGHT_NAMES + (TITAN_NORM_WEIGHT_NAME,):
            self.assertIn(template.format(layer=MCORE_LAYER), state)
        entries = _qkv_entries(state, TINY)
        # The norm gain is copied straight across: the same object, not an
        # equal one, so no layout step hides between the two engines.
        self.assertIs(
            entries[MCORE_NORM_WEIGHT_NAME.format(layer=MCORE_LAYER)],
            state[TITAN_NORM_WEIGHT_NAME.format(layer=MCORE_LAYER)],
        )
        # The projection is not: it is the grouped interleave of the three.
        self.assertTrue(
            torch.equal(
                entries[MCORE_WEIGHT_NAME.format(layer=MCORE_LAYER)],
                grouped_qkv(
                    *(
                        state[name.format(layer=MCORE_LAYER)]
                        for name in TITAN_WEIGHT_NAMES
                    ),
                    TINY,
                ),
            )
        )

    def test_the_shared_weights_have_the_shapes_the_map_expects(self) -> None:
        inputs = _inputs()
        state = _partial_titan_state(TINY)
        for key, template in zip(QKV_STATE_KEYS, TITAN_WEIGHT_NAMES):
            self.assertEqual(
                tuple(inputs.weight_state[key].shape),
                tuple(state[template.format(layer=MCORE_LAYER)].shape),
                key,
            )
        self.assertEqual(
            tuple(inputs.norm_weight.shape),
            tuple(state[TITAN_NORM_WEIGHT_NAME.format(layer=MCORE_LAYER)].shape),
        )


def _qkv_entries(
    state: dict[str, torch.Tensor], shape: PiperShape
) -> dict[str, torch.Tensor]:
    """The ``qkv`` transfers of layer ``MCORE_LAYER``, and nothing after them.

    ``weight_transfers`` is a generator over the whole model, so the walk
    stops as soon as both entries are in hand rather than building every
    tensor of every later component.
    """
    entries: dict[str, torch.Tensor] = {}
    for component, name, tensor in weight_transfers(state, shape):
        if component != MCORE_WEIGHT_COMPONENT:
            continue
        entries[name] = tensor
        if len(entries) == 2:
            break
    if len(entries) != 2:
        raise AssertionError("weight_transfers yields no qkv pair")
    return entries


def _partial_titan_state(shape: PiperShape) -> dict[str, torch.Tensor]:
    """Just the titan weights ``weight_transfers`` reads before ``qk_norm``.

    The generator yields the embedding, the lm head and the final norm, then
    per layer the grouped QKV, the attention norm and the output projection.
    ``_qkv_entries`` stops before anything after that, so nothing after that
    is built.
    """
    q_features = shape.n_heads * shape.head_dim
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
            q_features, shape.dim
        )
        state[f"{prefix}.attention.qkv_linear.wk.weight"] = tensor(
            kv_features, shape.dim
        )
        state[f"{prefix}.attention.qkv_linear.wv.weight"] = tensor(
            kv_features, shape.dim
        )
        state[f"{prefix}.attention_norm.weight"] = tensor(shape.dim)
        state[f"{prefix}.attention.wo.weight"] = tensor(shape.dim, q_features)
    return state


class TELayerNormColumnParallelLinear(nn.Module):
    """A stand-in with the attributes the guard reads.

    Named for the class the guard demands, because the guard compares class
    names. A real ``TELayerNormColumnParallelLinear`` needs megatron,
    TransformerEngine and a CUDA device, none of which this test has.
    """

    def __init__(
        self,
        out_features: int,
        in_features: int,
        *,
        norm_features: int | None = None,
        normalization: str = "RMSNorm",
        zero_centered_gamma: bool = False,
        eps: float = NORM_EPS,
        parallel_mode: str = "column",
        tp_size: int = 1,
        with_layer_norm_weight: bool = True,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        if with_layer_norm_weight:
            self.layer_norm_weight = nn.Parameter(
                torch.zeros(
                    in_features if norm_features is None else norm_features
                )
            )
        self.normalization = normalization
        self.zero_centered_gamma = zero_centered_gamma
        self.eps = eps
        self.parallel_mode = parallel_mode
        self.tp_size = tp_size


class TEColumnParallelLinear(TELayerNormColumnParallelLinear):
    """The unfused twin: a linear with no norm inside it."""


class _FakeConfig:
    def __init__(self, test_mode: bool = False) -> None:
        self.test_mode = test_mode


class _FakeAttention:
    """A ``SelfAttention`` reduced to what the guard and the drop touch."""

    def __init__(
        self,
        linear_qkv: object,
        *,
        world_size: int = 1,
        test_mode: bool = False,
        qk_norms: bool = True,
    ) -> None:
        self.linear_qkv = linear_qkv
        self.world_size = world_size
        self.config = _FakeConfig(test_mode)
        self.q_layernorm = object() if qk_norms else None
        self.k_layernorm = object() if qk_norms else None


class _FakeLayer:
    def __init__(self, attention: object) -> None:
        self.self_attention = attention


class _FakeDecoder:
    def __init__(self, layers: list) -> None:
        self.layers = layers


class _FakeGptModel:
    """A ``GPTModel`` reduced to the attribute path this scenario walks."""

    def __init__(self, attention: object, layers: int = 4) -> None:
        layer_objects = [_FakeLayer(None) for _ in range(layers)]
        layer_objects[MCORE_LAYER] = _FakeLayer(attention)
        self.decoder = _FakeDecoder(layer_objects)


def _module(**kwargs) -> TELayerNormColumnParallelLinear:
    return TELayerNormColumnParallelLinear(
        TINY.qkv_out_features, TINY.dim, **kwargs
    )


class NavigationTests(unittest.TestCase):
    def test_the_path_reaches_the_block_through_the_numeric_index(self) -> None:
        attention = _FakeAttention(_module())
        model = _FakeGptModel(attention)
        self.assertIs(_navigate(model, mcore_attention_path()), attention)

    def test_a_path_that_does_not_exist_raises(self) -> None:
        model = _FakeGptModel(_FakeAttention(_module()))
        with self.assertRaises(AttributeError):
            _navigate(model, "decoder.layers.0.mlp")


class McoreGuardTests(unittest.TestCase):
    def test_a_correct_block_passes_and_records_its_treatment(self) -> None:
        notes = _assert_mcore_qkv_prep(
            _FakeAttention(_module()), TINY, NORM_EPS
        )
        self.assertEqual(notes["module"], MCORE_MODULE_CLASS)
        self.assertEqual(notes["normalization"], "RMSNorm")
        self.assertEqual(notes["parallel_mode"], "column")
        self.assertEqual(notes["tp_size"], 1)
        self.assertFalse(notes["compiled"])

    def test_a_module_of_another_class_is_refused(self) -> None:
        """An unfused linear would put the norm outside the timed region.

        The whole partition argument for putting the norm inside this scenario
        is that megatron fuses it into ``linear_qkv``. A build that did not
        would measure a different cut under this label.
        """
        wrong = TEColumnParallelLinear(TINY.qkv_out_features, TINY.dim)
        with self.assertRaisesRegex(RuntimeError, MCORE_MODULE_CLASS):
            _assert_mcore_qkv_prep(_FakeAttention(wrong), TINY, NORM_EPS)

    def test_a_layernorm_is_refused_where_titan_runs_an_rmsnorm(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "RMSNorm"):
            _assert_mcore_qkv_prep(
                _FakeAttention(_module(normalization="LayerNorm")),
                TINY,
                NORM_EPS,
            )

    def test_zero_centered_gamma_is_refused(self) -> None:
        """TE would then compute with ``1 + gamma``.

        Copying titan's gain into that parameterization shifts every element
        by one, and no correctness gate names the cause.
        """
        with self.assertRaisesRegex(RuntimeError, "zero_centered_gamma"):
            _assert_mcore_qkv_prep(
                _FakeAttention(_module(zero_centered_gamma=True)),
                TINY,
                NORM_EPS,
            )

    def test_a_different_epsilon_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "eps"):
            _assert_mcore_qkv_prep(
                _FakeAttention(_module(eps=1e-5)), TINY, NORM_EPS
            )

    def test_a_module_of_the_wrong_shape_is_refused(self) -> None:
        """A wrong path lands on another linear, and the gate would widen."""
        wrong = TELayerNormColumnParallelLinear(TINY.dim * 3, TINY.dim)
        with self.assertRaisesRegex(RuntimeError, "linear_qkv.weight"):
            _assert_mcore_qkv_prep(_FakeAttention(wrong), TINY, NORM_EPS)

    def test_a_module_without_a_layer_norm_weight_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "layer_norm_weight"):
            _assert_mcore_qkv_prep(
                _FakeAttention(_module(with_layer_norm_weight=False)),
                TINY,
                NORM_EPS,
            )

    def test_a_module_that_is_not_column_parallel_is_refused(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "parallel_mode"):
            _assert_mcore_qkv_prep(
                _FakeAttention(_module(parallel_mode="row")), TINY, NORM_EPS
            )

    def test_more_than_one_rank_is_refused_rather_than_noted(self) -> None:
        """A note is not evidence: ``BuiltArm.notes`` reaches no artifact.

        Above one rank the timed call holds tensor-parallel collectives, and
        the ``num_query_groups < world_size`` branch adds an all-gather inside
        ``get_query_key_value_tensors``. The titan arms have no counterpart
        for either.
        """
        with self.assertRaisesRegex(RuntimeError, "tp_size"):
            _assert_mcore_qkv_prep(
                _FakeAttention(_module(tp_size=2)), TINY, NORM_EPS
            )
        with self.assertRaisesRegex(RuntimeError, "world_size"):
            _assert_mcore_qkv_prep(
                _FakeAttention(_module(), world_size=2), TINY, NORM_EPS
            )

    def test_test_mode_is_refused(self) -> None:
        """It runs two distributed all-gathers inside the timed call."""
        with self.assertRaisesRegex(RuntimeError, "test_mode"):
            _assert_mcore_qkv_prep(
                _FakeAttention(_module(), test_mode=True), TINY, NORM_EPS
            )


class QkNormRemovalTests(unittest.TestCase):
    def test_both_norms_are_dropped(self) -> None:
        """They belong to scenario 3, and the partition allows no overlap.

        ``get_query_key_value_tensors`` applies them before it returns, and
        titan's ``qkv_linear`` returns pre-norm q/k/v, so leaving them in
        would both double-count and make the two sides compute different
        functions.
        """
        attention = _FakeAttention(_module())
        _drop_qk_layernorms(attention)
        self.assertIsNone(attention.q_layernorm)
        self.assertIsNone(attention.k_layernorm)

    def test_a_model_that_never_had_them_is_refused(self) -> None:
        """The piper model has qk norms; a model without them is not it."""
        attention = _FakeAttention(_module(), qk_norms=False)
        with self.assertRaisesRegex(RuntimeError, "qk_layernorm"):
            _drop_qk_layernorms(attention)


class MissingGradientTests(unittest.TestCase):
    def test_a_missing_gradient_is_named_rather_than_thrown_as_attribute(
        self,
    ) -> None:
        with self.assertRaisesRegex(RuntimeError, "norm_weight_grad"):
            _weight_grad_outputs("arm", torch.zeros(2), None)

    def test_present_gradients_pass_through_unchanged(self) -> None:
        qkv, norm = torch.zeros(2), torch.ones(3)
        outputs = _weight_grad_outputs("arm", qkv, norm)
        self.assertIs(outputs["qkv_weight_grad"], qkv)
        self.assertIs(outputs["norm_weight_grad"], norm)
        present = {"a": torch.zeros(1)}
        self.assertIs(_require_grads("arm", present), present)


class _ThreeWayProjection(nn.Module):
    """A stand-in ``qkv_linear``: three projections and the head split."""

    def __init__(self, dim: int, n_heads: int, n_kv: int, head_dim: int) -> None:
        super().__init__()
        self.head_dim = head_dim
        self.n_heads, self.n_kv = n_heads, n_kv
        self.wq = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(dim, n_kv * head_dim, bias=False)
        self.wv = nn.Linear(dim, n_kv * head_dim, bias=False)

    def forward(self, x: torch.Tensor):
        lead = tuple(x.shape[:-1])
        return (
            self.wq(x).view(*lead, self.n_heads, self.head_dim),
            self.wk(x).view(*lead, self.n_kv, self.head_dim),
            self.wv(x).view(*lead, self.n_kv, self.head_dim),
        )


class TitanCompositionTests(unittest.TestCase):
    def test_the_wrapper_runs_the_norm_then_the_projection(self) -> None:
        """It composes two production modules and adds no arithmetic.

        The block runs ``self.qkv_linear(self.attention_norm(x))``, and the
        wrapper exists only so the whole cut compiles as one graph, which is
        the treatment the block gives it.
        """
        torch.manual_seed(0)
        norm = nn.RMSNorm(8, eps=NORM_EPS)
        with torch.no_grad():
            norm.weight.copy_(torch.rand(8) + 0.5)
        projection = _ThreeWayProjection(8, 2, 1, 4)
        module = _TitanQkvPrep(norm, projection)
        x = torch.randn(2, 3, 8)
        for produced, expected in zip(module(x), projection(norm(x))):
            torch.testing.assert_close(produced, expected)
        # Both children are registered, so ``_reset_grads`` reaches every
        # parameter the arm accumulates into.
        self.assertEqual(
            sorted(name for name, _ in module.named_parameters()),
            [
                "attention_norm.weight",
                "qkv_linear.wk.weight",
                "qkv_linear.wq.weight",
                "qkv_linear.wv.weight",
            ],
        )


class QkvPrepArmTests(unittest.TestCase):
    """The helper all three arms are built from, on plain torch modules."""

    def _arm(
        self,
        native: str,
        return_owner: bool = False,
        isolated_backward: bool = False,
    ):
        torch.manual_seed(0)
        dim, n_heads, n_kv, head_dim = 8, 2, 1, 4
        batch, seq = 2, 3
        norm = nn.RMSNorm(dim, eps=NORM_EPS)
        with torch.no_grad():
            norm.weight.copy_(torch.rand(dim) + 0.5)
        module = _TitanQkvPrep(
            norm, _ThreeWayProjection(dim, n_heads, n_kv, head_dim)
        )
        x = torch.arange(batch * seq * dim, dtype=torch.float32).reshape(
            batch, seq, dim
        ) / 100.0
        grads = (
            torch.ones(batch, seq, n_heads, head_dim),
            torch.ones(batch, seq, n_kv, head_dim) * 0.5,
            torch.ones(batch, seq, n_kv, head_dim) * 0.25,
        )
        if native == "mcore":
            x_native = x.reshape(batch * seq, 1, dim)
            grads_native = tuple(
                grad.reshape(batch * seq, 1, *grad.shape[2:]) for grad in grads
            )
        else:
            x_native, grads_native = x, grads

        def weight_grads():
            return _weight_grad_outputs(
                native,
                module.qkv_linear.wq.weight.grad,
                module.attention_norm.weight.grad,
            )

        arm = _qkv_prep_arm(
            name=native,
            owner=module,
            call=module,
            x_native=x_native,
            grads_native=grads_native,
            canonical_in=(batch, seq, dim),
            canonical_out=(
                (batch, seq, n_heads, head_dim),
                (batch, seq, n_kv, head_dim),
                (batch, seq, n_kv, head_dim),
            ),
            weight_grads=weight_grads,
            isolated_backward=isolated_backward,
        )
        return (arm, module) if return_owner else arm

    def test_the_megatron_arm_has_no_isolated_backward(self) -> None:
        """TE's ``LayerNormLinear`` backward clears its saved statistics
        (``layernorm_linear.py:1113-1114``), so the retained-graph trick the
        ``rope`` and ``expert_mlp`` scenarios use cannot run twice on it.
        The default is therefore the two-mode arm, and the megatron builder
        takes it; its backward cost stays recoverable as ``forward_backward``
        minus ``forward``.
        """
        self.assertEqual(
            sorted(self._arm("mcore").calls), ["forward", "forward_backward"]
        )

    def test_a_titan_arm_adds_a_backward_that_re_runs(self) -> None:
        """The mode the retired ``qkv`` scenario measured, restored.

        ``KernelArm.modes`` is per arm, so a titan arm may declare
        ``backward`` while the megatron arm does not. Two properties matter
        and neither is the declaration: the closure must exist, because
        ``_seeded_build`` refuses an arm whose calls do not match its
        declared modes exactly; and it must survive being called more than
        once, because the timing pass calls it once per burst over one
        retained graph. A closure that consumed its graph would raise on the
        second call rather than measure anything.
        """
        arm, module = self._arm(
            "titan", return_owner=True, isolated_backward=True
        )
        self.assertEqual(
            sorted(arm.calls),
            ["backward", "forward", "forward_backward"],
        )
        for call in range(2):
            with self.subTest(call=call):
                arm.calls["backward"]()
                self.assertIsNotNone(module.attention_norm.weight.grad)
                self.assertIsNotNone(module.qkv_linear.wq.weight.grad)

    def test_both_native_layouts_produce_identical_canonical_outputs(
        self,
    ) -> None:
        """THD and BSD are two labels for the same rows.

        The gates pair the two sides by output name and subtract, and torch
        aligns shapes from the right, so a canonicalization that left the
        megatron layout alone would raise above batch 1 and broadcast
        silently at batch 1.
        """
        titan = self._arm("titan").correctness_outputs()
        mcore = self._arm("mcore").correctness_outputs()
        self.assertEqual(
            sorted(titan),
            [
                "k_out",
                "norm_weight_grad",
                "q_out",
                "qkv_weight_grad",
                "v_out",
                "x_grad",
            ],
        )
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
        self.assertEqual(tuple(outputs["q_out"].shape), (2, 3, 2, 4))
        self.assertEqual(tuple(outputs["k_out"].shape), (2, 3, 1, 4))
        self.assertEqual(tuple(outputs["v_out"].shape), (2, 3, 1, 4))
        self.assertEqual(tuple(outputs["x_grad"].shape), (2, 3, 8))

    def test_forward_backward_clears_gradients_between_calls(self) -> None:
        """A timed closure runs thousands of times in one burst.

        Without the reset every weight gradient would grow without bound, and
        the arm would time an addition that grows a tensor rather than the
        projection.

        Read the gradient off the module directly. ``correctness_outputs``
        calls ``_reset_grads`` itself and then runs its own forward and
        backward, so a value read through it holds only that last call.
        """
        arm, owner = self._arm("titan", return_owner=True)
        arm.calls["forward_backward"]()
        after_one = owner.qkv_linear.wq.weight.grad.clone()
        for _ in range(3):
            arm.calls["forward_backward"]()
        after_four = owner.qkv_linear.wq.weight.grad.clone()
        torch.testing.assert_close(after_one, after_four)
        self.assertFalse(
            torch.allclose(after_one * 4, after_four),
            "the gradient did not accumulate, so this test proves nothing",
        )

    def test_forward_returns_the_three_tensors_the_scenario_ends_at(
        self,
    ) -> None:
        produced = self._arm("titan").calls["forward"]()
        self.assertEqual(len(produced), 3)
        self.assertEqual(tuple(produced[0].shape), (2, 3, 2, 4))
        self.assertEqual(tuple(produced[1].shape), (2, 3, 1, 4))


class SplitMaterializationTests(unittest.TestCase):
    """Pin the materialization asymmetry the scenario description declares.

    The scenario times both engines' split, and the two do different amounts
    of copying: megatron materializes only ``query`` and hands ``key`` and
    ``value`` on as strided views, while titan materializes all three. That
    difference is inside the published cross-engine ratio, so the description
    states it and these tests keep the statement true.

    What is under test is the **shape arithmetic**, not torchtitan's module.
    These tests replay the three lines at ``models/common/attention.py:814``
    and ``:819-821`` and TE's ``pytorch/utils.py:437`` on plain CPU tensors at
    the ``normal`` geometry. Building the real modules needs a GPU, megatron
    and a compiler; the claim in the docstring is about ``torch.split``,
    ``reshape`` and ``contiguous``, and that is exactly what runs here.
    """

    # The normal shape: dim 1024, head_dim 64 -> 16 heads, 8 kv heads, so a
    # group holds 2 query heads plus one k and one v.
    N_KV, HEADS_PER_GROUP, HEAD_DIM = 8, 2, 64
    BATCH, SEQ = 4, 1024

    def _packed(self) -> torch.Tensor:
        """The buffer ``FusedQKVLinear.forward`` views at ``attention.py:801``."""
        return torch.zeros(
            self.BATCH,
            self.SEQ,
            self.N_KV,
            self.HEADS_PER_GROUP + 2,
            self.HEAD_DIM,
            dtype=torch.bfloat16,
        )

    @staticmethod
    def _shares_storage(a: torch.Tensor, b: torch.Tensor) -> bool:
        return a.untyped_storage().data_ptr() == b.untyped_storage().data_ptr()

    def test_the_query_reshape_copies_and_its_contiguous_is_a_no_op(
        self,
    ) -> None:
        """``xq``'s copy is charged to ``reshape``, not to ``.contiguous()``.

        ``xq`` is a strided slice of the packed buffer, and the reshape merges
        the kv-group axis with the query-head axis -- two dimensions that are
        not adjacent in memory once the slice narrowed the one between them.
        So the reshape cannot return a view.
        """
        packed = self._packed()
        b, s = packed.shape[0], packed.shape[1]
        xq, _, _ = torch.split(packed, [self.HEADS_PER_GROUP, 1, 1], dim=-2)

        reshaped = xq.reshape(b, s, -1, self.HEAD_DIM)
        self.assertFalse(
            self._shares_storage(reshaped, packed),
            "the query reshape returned a view, so the copy this scenario "
            "charges titan is not where the module docstring says it is",
        )
        self.assertTrue(reshaped.is_contiguous())
        self.assertTrue(
            self._shares_storage(reshaped.contiguous(), reshaped),
            "the trailing .contiguous() copied, so it is not the no-op the "
            "module docstring calls it",
        )

    def test_the_key_and_value_copies_are_charged_to_contiguous(self) -> None:
        """``xk``/``xv`` reshape to a view; the ``.contiguous()`` copies.

        Their reshape only drops a size-1 dimension, which is a squeeze and
        always a view.
        """
        packed = self._packed()
        b, s = packed.shape[0], packed.shape[1]
        _, xk, xv = torch.split(packed, [self.HEADS_PER_GROUP, 1, 1], dim=-2)

        for name, tensor in (("xk", xk), ("xv", xv)):
            with self.subTest(tensor=name):
                reshaped = tensor.reshape(b, s, -1, self.HEAD_DIM)
                self.assertTrue(
                    self._shares_storage(reshaped, packed),
                    f"{name}'s reshape copied, so the split materializes "
                    "more than the three tensors the description counts",
                )
                self.assertFalse(reshaped.is_contiguous())
                self.assertFalse(
                    self._shares_storage(reshaped.contiguous(), packed),
                    f"{name}'s .contiguous() returned a view, so titan does "
                    "not materialize it and the byte figures are wrong",
                )

    def test_a_plain_split_hands_on_views_which_is_megatrons_k_and_v(
        self,
    ) -> None:
        """TE's ``SplitAlongDim`` is ``torch.split`` off the FP8 path.

        ``pytorch/utils.py:437-440`` returns ``torch.split(...)`` directly for
        a non-FP8 tensor, and megatron reshapes only ``query``
        (``transformer/attention.py:1908``). So megatron's ``key`` and
        ``value`` reach the next scenario as strided views it never paid to
        materialize, and the 8 MiB the description attributes to megatron is
        the query alone.
        """
        packed = self._packed()
        query, key, value = torch.split(
            packed, [self.HEADS_PER_GROUP, 1, 1], dim=-2
        )
        for name, tensor in (("key", key), ("value", value)):
            with self.subTest(tensor=name):
                self.assertTrue(self._shares_storage(tensor, packed))
                self.assertFalse(tensor.is_contiguous())

        materialized = query.reshape(
            query.size(0), query.size(1), -1, self.HEAD_DIM
        )
        self.assertFalse(self._shares_storage(materialized, packed))
        self.assertEqual(
            materialized.numel() * materialized.element_size(),
            8 * 1024 * 1024,
            "megatron's one materialization is not the 8 MiB the scenario "
            "description publishes",
        )


if __name__ == "__main__":
    unittest.main()
