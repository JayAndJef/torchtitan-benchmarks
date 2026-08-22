"""What ``benchmarks/models/piper_qwen3/megatron_model.py`` promises a caller.

The file builds one bare megatron-core ``GPTModel`` from a ``PiperShape`` and
an ``McoreProfile``. Two other suites already cover parts of it:
``tests/test_model_shape.py`` pins the ``shape`` argument and
``tests/test_mcore_profiles.py`` pins the ``profile`` argument. This module
covers ``blank_parts``, the spec surgery that lets a kernel scenario build
only the part of the transformer layer it times.

Every test here runs on the CPU and builds no model. ``_blank_layer_parts``
is a pure dataclass transformation, so it can be exercised with the real
``TransformerLayerSubmodules`` and sentinel values in its slots -- which is
what proves the field names are megatron's own rather than this repo's.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import re
import sys
import textwrap
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.models.piper_qwen3 import megatron_model  # noqa: E402

MEGATRON_MODEL_SOURCE = Path(megatron_model.__file__).read_text()

# The nine parts of a transformer layer, and the identity class megatron's own
# dataclass declares as each one's default. This is the roster
# ``_blank_layer_parts`` reads at run time; stating it here is what turns a
# megatron bump that renames or re-defaults a slot into a failed test rather
# than into a silently smaller blanking set.
EXPECTED_DEFAULTS = {
    "input_layernorm": "IdentityOp",
    "self_attention": "IdentityOp",
    "self_attn_bda": "IdentityFuncOp",
    "pre_cross_attn_layernorm": "IdentityOp",
    "cross_attention": "IdentityOp",
    "cross_attn_bda": "IdentityFuncOp",
    "pre_mlp_layernorm": "IdentityOp",
    "mlp": "IdentityOp",
    "mlp_bda": "IdentityFuncOp",
}


def _megatron_layer_types():
    """The megatron classes the surgery operates on, or a skip.

    **Called from inside a test, never at module scope.**
    ``add_megatron_to_path`` inserts the checkout at ``sys.path[0]``, and
    Megatron-LM ships a ``tests/`` directory of its own, so from then on
    ``import tests.<anything>`` resolves to megatron's tests rather than to
    this suite's. ``unittest discover`` imports every module before it runs
    any test, so a module-scope call breaks the later modules that import
    from this suite's own package.
    """
    from benchmarks.models.piper_qwen3.megatron_bootstrap import (
        add_megatron_to_path,
    )

    try:
        add_megatron_to_path()
        from megatron.core.transformer.identity_op import (
            IdentityFuncOp,
            IdentityOp,
        )
        from megatron.core.transformer.spec_utils import ModuleSpec
        from megatron.core.transformer.transformer_block import (
            TransformerBlockSubmodules,
        )
        from megatron.core.transformer.transformer_layer import (
            TransformerLayerSubmodules,
        )
    except Exception as error:  # pragma: no cover - host dependent
        raise unittest.SkipTest(f"megatron is not importable here: {error}")
    return (
        IdentityOp,
        IdentityFuncOp,
        ModuleSpec,
        TransformerBlockSubmodules,
        TransformerLayerSubmodules,
    )


def _code_of(function) -> str:
    """One function's source with its docstring removed.

    Several tests below assert that a name appears nowhere in a function.
    The docstrings here name the very things the code must not do, so a
    plain source read would find them as prose.
    """
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    body = tree.body[0].body
    if (
        isinstance(body[0], ast.Expr)
        and isinstance(body[0].value, ast.Constant)
        and isinstance(body[0].value.value, str)
    ):
        body = body[1:]
    return "\n".join(ast.unparse(node) for node in body)


class _Sentinel:
    """Stands in for a real part. Distinguishable, and allocates nothing."""

    def __init__(self, name: str) -> None:
        self.name = name

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<part {self.name}>"


def _block_spec(layers: int = 2):
    """A block spec whose every layer part is a distinct sentinel.

    Megatron's own derivation appends the **same** layer-spec object once per
    layer, so the fixture does too. A surgery that mutated in place, or that
    replaced one entry and left the rest, is visible against this.
    """
    (
        _identity,
        _identity_func,
        module_spec,
        block_submodules,
        layer_submodules,
    ) = _megatron_layer_types()
    submodules = layer_submodules(
        **{name: _Sentinel(name) for name in EXPECTED_DEFAULTS}
    )
    layer_spec = module_spec(module=_Sentinel("TransformerLayer"))
    layer_spec.submodules = submodules
    return block_submodules(layer_specs=[layer_spec] * layers, layer_norm=None)


class BlankPartsSignatureTests(unittest.TestCase):
    def test_the_default_builds_every_part(self) -> None:
        """An omitted argument must keep the historical model.

        Every number this repo has published from a megatron kernel arm was
        taken on a model with all nine parts built. A default that blanked
        anything would silently move that boundary.
        """
        parameter = inspect.signature(megatron_model.build_model).parameters[
            "blank_parts"
        ]
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertEqual(parameter.default, ())

    def test_the_e2e_driver_and_the_parity_tool_never_blank(self) -> None:
        """Both need the whole model, and neither may acquire the argument.

        ``benchmarks/e2e/megatron/train.py`` trains the model and
        ``tools/megatron_parity_check.py`` matches its logits against
        TorchTitan's. A blanked part would make either one measure a
        different network under the same label.
        """
        root = Path(__file__).resolve().parent.parent
        for relative in (
            "benchmarks/e2e/megatron/train.py",
            "tools/megatron_parity_check.py",
        ):
            with self.subTest(caller=relative):
                self.assertNotIn(
                    "blank_parts", (root / relative).read_text()
                )


class BlankLayerPartsTests(unittest.TestCase):
    def test_megatron_declares_the_nine_parts_this_repo_blanks(self) -> None:
        """The roster and the two identity classes are megatron's, not ours.

        ``_blank_layer_parts`` writes no replacement value of its own: it
        reads each field's declared default. So this asserts the declaration
        rather than a mapping written here.
        """
        *_, layer_submodules = _megatron_layer_types()
        defaults = {
            field.name: field.default.__name__
            for field in dataclasses.fields(layer_submodules)
            if isinstance(field.default, type)
        }
        self.assertEqual(defaults, EXPECTED_DEFAULTS)

    def test_blanking_replaces_the_named_part_and_nothing_else(self) -> None:
        identity, *_ = _megatron_layer_types()
        spec = _block_spec()
        blanked = megatron_model._blank_layer_parts(spec, ("mlp",))
        parts = blanked.layer_specs[0].submodules
        self.assertIs(parts.mlp, identity)
        for name in EXPECTED_DEFAULTS:
            if name == "mlp":
                continue
            with self.subTest(part=name):
                self.assertIs(
                    getattr(parts, name),
                    getattr(spec.layer_specs[0].submodules, name),
                )

    def test_a_bias_dropout_add_slot_blanks_to_the_function_identity(
        self,
    ) -> None:
        """``IdentityFuncOp`` returns a function; ``IdentityOp`` returns a
        tensor. A bda slot is called for a function, so the two are not
        interchangeable, and reading the declared default is what keeps them
        apart without a mapping written here.
        """
        identity, identity_func, *_ = _megatron_layer_types()
        blanked = megatron_model._blank_layer_parts(
            _block_spec(), ("self_attn_bda", "mlp_bda", "self_attention")
        )
        parts = blanked.layer_specs[0].submodules
        self.assertIs(parts.self_attn_bda, identity_func)
        self.assertIs(parts.mlp_bda, identity_func)
        self.assertIs(parts.self_attention, identity)

    def test_every_layer_gets_its_own_blanked_copy(self) -> None:
        """Megatron appends one spec object per layer, and they are the same
        object. A surgery that replaced the list entry rather than each
        entry's contents would blank one layer and leave the rest.
        """
        identity, *_ = _megatron_layer_types()
        blanked = megatron_model._blank_layer_parts(_block_spec(3), ("mlp",))
        self.assertEqual(len(blanked.layer_specs), 3)
        for index, layer_spec in enumerate(blanked.layer_specs):
            with self.subTest(layer=index):
                self.assertIs(layer_spec.submodules.mlp, identity)

    def test_the_derived_spec_is_left_unmutated(self) -> None:
        """The caller keeps a usable original, and no other reader is
        surprised. ``dataclasses.replace`` copies; an in-place write would
        not.
        """
        spec = _block_spec()
        before = spec.layer_specs[0].submodules.mlp
        megatron_model._blank_layer_parts(spec, ("mlp",))
        self.assertIs(spec.layer_specs[0].submodules.mlp, before)

    def test_the_layer_module_class_is_untouched(self) -> None:
        """Blanking removes parts. It never chooses a different layer."""
        spec = _block_spec()
        blanked = megatron_model._blank_layer_parts(spec, ("mlp",))
        self.assertIs(
            blanked.layer_specs[0].module, spec.layer_specs[0].module
        )

    def test_a_field_that_is_not_an_identity_slot_is_refused(self) -> None:
        """``sharded_state_dict_keys_map`` is a dict, and blanking it would
        put an ``IdentityOp`` where a mapping belongs.
        """
        with self.assertRaisesRegex(ValueError, "sharded_state_dict_keys_map"):
            megatron_model._blank_layer_parts(
                _block_spec(), ("sharded_state_dict_keys_map",)
            )

    def test_a_name_the_layer_does_not_declare_is_refused(self) -> None:
        """A nested name is the mistake this catches. ``experts`` lives
        inside the mlp part, not beside it, so blanking reaches it only
        through the part that owns it.
        """
        with self.assertRaisesRegex(ValueError, "experts"):
            megatron_model._blank_layer_parts(_block_spec(), ("experts",))

    def test_a_block_spec_with_no_layer_is_refused(self) -> None:
        _, _, _, block_submodules, _ = _megatron_layer_types()
        with self.assertRaisesRegex(ValueError, "holds no layer"):
            megatron_model._blank_layer_parts(
                block_submodules(layer_specs=[], layer_norm=None), ("mlp",)
            )


class BuildModelWiringTests(unittest.TestCase):
    """Source-level pins on how ``build_model`` uses the surgery.

    ``build_model`` allocates a whole model on a CUDA device, so a CPU test
    cannot call it. These read the file instead.
    """

    def test_the_surgery_runs_on_the_derived_spec(self) -> None:
        """The derivation stays megatron's. The surgery is applied to its
        result, never in place of it.
        """
        derivation = MEGATRON_MODEL_SOURCE.index(
            "spec = get_gpt_decoder_block_spec("
        )
        build = MEGATRON_MODEL_SOURCE.index("    model = GPTModel(")
        between = MEGATRON_MODEL_SOURCE[derivation:build]
        self.assertIn("spec = _blank_layer_parts(spec, blank_parts)", between)

    def test_the_built_model_is_checked_before_it_is_returned(self) -> None:
        """The blanked parts have no reader, because their whole point is
        that nobody reads them. So the only place the blanking can be
        checked is here, where it always runs.
        """
        call = "_assert_parts_are_blank(model, blank_parts)"
        self.assertIn(call, MEGATRON_MODEL_SOURCE)
        self.assertGreater(
            MEGATRON_MODEL_SOURCE.rindex("    return model"),
            MEGATRON_MODEL_SOURCE.index(call),
        )

    def test_the_surgery_writes_no_module_class_of_its_own(self) -> None:
        """The whole argument for this change is that it authors no spec.

        ``_blank_layer_parts`` may name the two identity classes only by
        reading a declared default, so its body must contain no assignment
        of a module class and no import of one.
        """
        body = _code_of(megatron_model._blank_layer_parts)
        self.assertIn("field.default", body)
        for forbidden in ("IdentityFuncOp", "SelfAttention", "MoELayer"):
            with self.subTest(symbol=forbidden):
                self.assertNotIn(forbidden, body)

    def test_the_config_is_built_before_the_surgery_and_never_by_it(
        self,
    ) -> None:
        """The declined alternative was ``num_moe_experts=None``, which
        changes the config. That field, ``moe_grouped_gemm`` and
        ``qk_layernorm`` are the three megatron turns into a module-class
        choice and then checks nowhere, so a config edit is silent in both
        directions. Blanking edits the derived spec only.
        """
        body = _code_of(megatron_model._blank_layer_parts)
        for field in ("num_moe_experts", "moe_grouped_gemm", "qk_layernorm"):
            with self.subTest(field=field):
                self.assertNotIn(field, body)
        self.assertLess(
            MEGATRON_MODEL_SOURCE.index("config = TransformerConfig(**kwargs)"),
            MEGATRON_MODEL_SOURCE.index("spec = _blank_layer_parts("),
        )

    def test_the_blank_check_reads_decoder_layer_zero_only(self) -> None:
        """Every caller reads layer 0, and blanking applies to every layer,
        so one layer is the whole check. A future index would mean a caller
        this rule was never written for.
        """
        body = _code_of(megatron_model._assert_parts_are_blank)
        self.assertIn("model.decoder.layers[0]", body)
        self.assertEqual(re.findall(r"layers\[(\d+)\]", body), ["0"])


class OperationsBlankingTests(unittest.TestCase):
    """Which kernel builder blanks which part, pinned per module.

    ``blank_parts`` is safe only when the scenario never reads the part it
    removes, and nothing in the code can tell which cut a builder times. So
    the mapping is stated here, and every module that builds a megatron model
    must appear in exactly one of the two sets below.
    """

    # The cut these scenarios time sits outside the mlp part, so they blank
    # it. See ``benchmarks/kernel/operations/common.py``'s ``MCORE_BLANK_MLP``
    # for why the timed module keeps its exact weights.
    BLANKS_THE_MLP = {
        "attention_core",
        "attn_out_proj",
        "attn_residual",
        "cross_entropy",
        "ffn_norm",
        "lm_head_projection",
        "embedding_stage",
        "final_norm",
        "moe_residual",
        "qk_norm",
        "qkv_prep",
        "rope",
    }

    # These four time a cut **inside** the mlp part: the experts, the router,
    # and the token dispatcher twice. They build the whole layer.
    CUTS_INSIDE_THE_MLP = {
        "dispatch_permute",
        "expert_mlp",
        "moe_combine",
        "moe_router",
    }

    @staticmethod
    def _operations() -> dict[str, str]:
        directory = (
            Path(__file__).resolve().parent.parent
            / "benchmarks/kernel/operations"
        )
        return {
            path.stem: path.read_text()
            for path in sorted(directory.glob("*.py"))
            if "build_model(" in path.read_text()
        }

    def test_every_megatron_builder_is_classified(self) -> None:
        """A new builder must state which set it belongs to.

        Without this a module added later would build the whole layer by
        default and nobody would notice the omission.
        """
        self.assertEqual(
            set(self._operations()),
            self.BLANKS_THE_MLP | self.CUTS_INSIDE_THE_MLP,
        )
        self.assertEqual(
            self.BLANKS_THE_MLP & self.CUTS_INSIDE_THE_MLP, set()
        )

    def test_the_cuts_outside_the_mlp_blank_it(self) -> None:
        sources = self._operations()
        for name in sorted(self.BLANKS_THE_MLP):
            with self.subTest(module=name):
                self.assertIn("blank_parts=MCORE_BLANK_MLP", sources[name])
                self.assertIn("    MCORE_BLANK_MLP,", sources[name])

    def test_the_cuts_inside_the_mlp_build_the_whole_layer(self) -> None:
        """The decisive rule. A builder that timed the experts or the router
        against a blanked mlp would raise, but a builder that timed the
        layer's own parts against a blanked one would measure a different
        model in silence.
        """
        sources = self._operations()
        for name in sorted(self.CUTS_INSIDE_THE_MLP):
            with self.subTest(module=name):
                self.assertNotIn("blank_parts", sources[name])

    def test_no_builder_names_a_part_of_its_own(self) -> None:
        """One constant carries the justification, so a reader finds it once.

        A literal tuple at a call site would put the choice next to the call
        and the reason nowhere.
        """
        for name, source in sorted(self._operations().items()):
            with self.subTest(module=name):
                self.assertNotIn("blank_parts=(", source)

    def test_the_constant_names_the_mlp_part_alone(self) -> None:
        from benchmarks.kernel.operations.common import MCORE_BLANK_MLP

        self.assertEqual(MCORE_BLANK_MLP, ("mlp",))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
