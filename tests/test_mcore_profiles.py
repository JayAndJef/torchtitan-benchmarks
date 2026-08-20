"""The mcore profile registry, and the extraction it has to preserve.

``tools/megatron_parity_check.py`` cannot guard this. It is forward-only,
eval-mode, and computes its loss with ``F.cross_entropy`` on logits rather
than through ``compute_language_model_loss``, so it would pass unchanged if
the extraction dropped ``cross_entropy_fusion_impl``,
``cross_entropy_loss_fusion``, ``gradient_accumulation_fusion`` or
``bias_dropout_fusion`` -- and it never exercises the cuda-graph branch at
all. A frozen literal does guard it: any flag that moves has to move here too,
in the same commit, where a reader sees it.

Every test here is CPU-only and imports no torch, which is the property the
registry exists to have.
"""

import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.models.piper_qwen3.mcore_profiles import (
    ATTENTION_BACKEND_FIELDS,
    ATTENTION_BACKENDS,
    BASE,
    FUSION_FIELDS,
    MCORE_PROFILES,
    McoreProfile,
    declared_mismatches,
    derive,
    profile_by_name,
    transformer_config_kwargs,
)
from benchmarks.models.piper_qwen3.shape import HUGE, NORMAL

# What benchmarks/models/piper_qwen3/megatron_model.py wrote inline before the
# profile registry existed, at the normal shape. Every megatron number this
# repo has published was measured on exactly this payload. Changing a value
# here is changing the model; changing it without saying so in the commit
# message is how a run stops being comparable to the ones before it.
BASE_KWARGS_AT_NORMAL = {
    # geometry, from PiperShape
    "num_layers": 16,
    "hidden_size": 1024,
    "num_attention_heads": 16,
    "num_query_groups": 8,
    "kv_channels": 64,
    "ffn_hidden_size": 3584,
    "num_moe_experts": 4,
    "moe_router_topk": 2,
    "moe_ffn_hidden_size": 3584,
    # behaviour, from the profile
    "normalization": "RMSNorm",
    "layernorm_epsilon": 1e-6,
    "add_bias_linear": False,
    "add_qkv_bias": False,
    "gated_linear_unit": True,
    "activation_func": "silu",
    "qk_layernorm": True,
    "hidden_dropout": 0.0,
    "attention_dropout": 0.0,
    "moe_layer_freq": 1,
    "moe_router_score_function": "softmax",
    "moe_router_pre_softmax": False,
    "moe_router_load_balancing_type": "none",
    "moe_aux_loss_coeff": 0.0,
    "moe_router_enable_expert_bias": False,
    "moe_router_dtype": "fp32",
    "moe_token_dispatcher_type": "allgather",
    "moe_grouped_gemm": True,
    "apply_rope_fusion": True,
    "bias_activation_fusion": True,
    "bias_dropout_fusion": True,
    "cross_entropy_loss_fusion": True,
    "cross_entropy_fusion_impl": "te",
    "moe_permute_fusion": True,
    "bf16": True,
    "params_dtype": "bfloat16",
    "pipeline_dtype": "bfloat16",
    "gradient_accumulation_fusion": False,
}


class BaseProfileExtractionTests(unittest.TestCase):
    def test_the_base_profile_builds_the_config_it_always_built(self) -> None:
        """The extraction is a refactor, so the payload must not move."""
        self.assertEqual(
            transformer_config_kwargs(shape=NORMAL, profile=BASE),
            BASE_KWARGS_AT_NORMAL,
        )

    def test_the_cuda_graph_branch_is_frozen_too(self) -> None:
        """The branch the parity check never reaches, so nothing else pins it.

        ``use_te_rng_tracker`` is the subtle one: TE's attention asserts on
        the tracker type inside a captured graph, so dropping it turns every
        cuda-graph megatron run into a crash at capture time -- or worse, a
        silent fall back to eager if a future rev only warns.
        """
        kwargs = transformer_config_kwargs(
            shape=NORMAL,
            profile=BASE,
            cuda_graph_impl="local",
            cuda_graph_modules=("moe_router", "moe_preprocess"),
            use_cpu_initialization=True,
        )
        self.assertEqual(
            kwargs,
            {
                **BASE_KWARGS_AT_NORMAL,
                "use_cpu_initialization": True,
                "cuda_graph_impl": "local",
                "cuda_graph_modules": ["moe_router", "moe_preprocess"],
                "use_te_rng_tracker": True,
            },
        )

    def test_no_graph_modules_means_no_graph_modules_key(self) -> None:
        """An empty tuple must not reach megatron as an empty list.

        The old inline block guarded the key on truthiness, so an empty list
        was never passed. Passing one would ask the graph manager to capture
        a named set of zero modules rather than to use its own default.
        """
        kwargs = transformer_config_kwargs(
            shape=NORMAL, profile=BASE, cuda_graph_impl="local"
        )
        self.assertNotIn("cuda_graph_modules", kwargs)
        self.assertTrue(kwargs["use_te_rng_tracker"])

    def test_the_geometry_moves_with_the_shape_and_nothing_else_does(
        self,
    ) -> None:
        """A profile carries no geometry, so huge differs only in geometry."""
        at_huge = transformer_config_kwargs(shape=HUGE, profile=BASE)
        moved = {
            key
            for key in at_huge
            if at_huge[key] != BASE_KWARGS_AT_NORMAL[key]
        }
        self.assertEqual(
            moved,
            {
                "num_layers",
                "hidden_size",
                "num_attention_heads",
                "num_query_groups",
                "ffn_hidden_size",
                "moe_ffn_hidden_size",
            },
        )
        self.assertEqual(at_huge["hidden_size"], HUGE.dim)
        self.assertEqual(at_huge["num_attention_heads"], HUGE.n_heads)

    def test_the_kwargs_are_a_fresh_dict_per_call(self) -> None:
        """The correctness pass builds several arms in one interpreter.

        A caller that mutated the returned dict -- resolving ``"silu"`` to
        ``F.silu`` in place, which ``build_model`` does -- would otherwise
        leave a torch object inside the shared profile and contaminate every
        later build in that process.
        """
        first = transformer_config_kwargs(shape=NORMAL, profile=BASE)
        first["activation_func"] = "mutated"
        first["injected"] = True
        second = transformer_config_kwargs(shape=NORMAL, profile=BASE)
        self.assertEqual(second, BASE_KWARGS_AT_NORMAL)
        self.assertEqual(BASE.config_overrides["activation_func"], "silu")

    def test_the_layer_spec_derivation_inputs_are_all_on_the_config(
        self,
    ) -> None:
        """megatron builds the layer spec by reading exactly these fields.

        ``get_gpt_decoder_block_spec`` turns ``num_moe_experts``,
        ``moe_grouped_gemm`` and ``qk_layernorm`` into module-class choices
        (``gpt_layer_specs.py:592-594``). Dropping ``num_moe_experts`` would
        not raise -- the derivation would take its dense branch and build a
        model with no experts at all, under the MoE label.
        """
        at_normal = transformer_config_kwargs(shape=NORMAL, profile=BASE)
        self.assertEqual(at_normal["num_moe_experts"], NORMAL.num_experts)
        self.assertTrue(at_normal["moe_grouped_gemm"])
        self.assertTrue(at_normal["qk_layernorm"])
        # moe_layer_freq drives the dense/MoE pattern; 1 means every layer.
        self.assertEqual(at_normal["moe_layer_freq"], 1)
        self.assertEqual(
            transformer_config_kwargs(shape=HUGE, profile=BASE)[
                "num_moe_experts"
            ],
            HUGE.num_experts,
        )


class ProfileValidationTests(unittest.TestCase):
    def test_a_class_selecting_delta_is_one_field(self) -> None:
        """The delta that used to need two halves kept in agreement by hand.

        ``moe_grouped_gemm`` picks the expert module class. While the builder
        wrote its own layer spec, this profile had to set the value twice, and
        setting it once built the grouped kernel under the ungrouped label --
        which no correctness gate can catch, because both implementations are
        numerically right. The derivation reads one field, so the two cannot
        disagree and there is nothing left to enforce.
        """
        profile = derive(
            BASE,
            name="no_grouped_gemm",
            description="per-expert GEMMs instead of the grouped kernel",
            config_overrides={"moe_grouped_gemm": False},
        )
        self.assertFalse(
            transformer_config_kwargs(shape=NORMAL, profile=profile)[
                "moe_grouped_gemm"
            ]
        )
        changed = {
            key
            for key in profile.config_overrides
            if profile.config_overrides[key] != BASE.config_overrides[key]
        }
        self.assertEqual(changed, {"moe_grouped_gemm"})

    def test_a_torch_value_in_a_profile_is_refused(self) -> None:
        """The registry is parent-side, so its values must survive json.

        Writing ``F.silu`` here rather than ``"silu"`` would import torch into
        the parent and break the manifest write, both far from the cause.
        """
        with self.assertRaisesRegex(ValueError, "JSON-safe"):
            McoreProfile(
                name="torchy",
                description="x",
                config_overrides={"activation_func": lambda value: value},
            )

    def test_an_unknown_encoded_name_is_refused(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a known activation"):
            derive(
                BASE, name="x", description="x",
                config_overrides={"activation_func": "relu"},
            )
        with self.assertRaisesRegex(ValueError, "not a known dtype"):
            derive(
                BASE, name="x", description="x",
                config_overrides={"params_dtype": "fp8"},
            )
        with self.assertRaisesRegex(ValueError, "not a known attention"):
            derive(
                BASE, name="x", description="x",
                config_overrides={"attention_backend": "cudnn"},
            )

    def test_every_attention_backend_name_is_a_megatron_enum_member(
        self,
    ) -> None:
        """The names the registry offers are megatron's own, spelled its way.

        This module imports no torch and no megatron, so it reads the enum
        out of the pinned submodule as text. A submodule bump that renames or
        drops a member fails here, on CPU, rather than inside a GPU worker
        that has already built a model.
        ``megatron_model.build_model`` asserts the same equality against the
        imported enum, which is the run-time half of this check.
        """
        source = (
            Path(__file__).resolve().parent.parent
            / "third_party"
            / "Megatron-LM"
            / "megatron"
            / "core"
            / "transformer"
            / "enums.py"
        ).read_text()
        body = source.split("class AttnBackend", 1)[1].split("class ", 1)[0]
        members = re.findall(r"^\s{4}(\w+) = \d+$", body, flags=re.MULTILINE)
        self.assertEqual(sorted(members), sorted(ATTENTION_BACKENDS))

    def test_a_named_attention_backend_reaches_the_config_kwargs(self) -> None:
        """The name travels as a name, and nothing here turns it into an enum.

        ``ATTENTION_BACKEND_FIELDS`` is what ``megatron_model.build_model``
        walks to resolve it. A field added to a profile but missing from that
        tuple would arrive at ``TransformerConfig`` as a string, which
        megatron compares against enum members and never matches.
        """
        profile = derive(
            BASE,
            name="pinned",
            description="pinned",
            config_overrides={"attention_backend": "fused"},
        )
        kwargs = transformer_config_kwargs(shape=NORMAL, profile=profile)
        self.assertEqual(kwargs["attention_backend"], "fused")
        self.assertIn("attention_backend", ATTENTION_BACKEND_FIELDS)
        self.assertNotIn("attention_backend", BASE.config_overrides)

    def test_a_delta_names_only_what_it_changes(self) -> None:
        """A delta that restated 28 flags would hide its own difference."""
        profile = derive(
            BASE,
            name="ce_native",
            description="megatron's own fp32 multi-pass cross-entropy",
            config_overrides={"cross_entropy_fusion_impl": "native"},
        )
        changed = {
            key
            for key in profile.config_overrides
            if profile.config_overrides[key] != BASE.config_overrides[key]
        }
        self.assertEqual(changed, {"cross_entropy_fusion_impl"})
        self.assertEqual(BASE.config_overrides["cross_entropy_fusion_impl"], "te")

    def test_the_registry_resolves_by_name_and_says_so_when_it_cannot(
        self,
    ) -> None:
        self.assertIs(profile_by_name("base"), BASE)
        self.assertIn("base", MCORE_PROFILES)
        with self.assertRaisesRegex(ValueError, "Unknown mcore profile"):
            profile_by_name("nope")

    def test_the_manifest_record_is_json_safe_and_complete(self) -> None:
        import json

        payload = BASE.describe()
        self.assertEqual(json.loads(json.dumps(payload)), payload)
        self.assertEqual(payload["name"], "base")
        self.assertEqual(
            payload["config_overrides"]["cross_entropy_fusion_impl"], "te"
        )
        # One surface. A second mapping here would mean the builder went back
        # to writing its own layer spec, and the duplication with it.
        self.assertEqual(
            set(payload), {"name", "description", "config_overrides"}
        )
        # A copy, so a manifest writer cannot edit the registry.
        payload["config_overrides"]["bf16"] = False
        self.assertTrue(BASE.config_overrides["bf16"])


class DeclaredStateTests(unittest.TestCase):
    """What the megatron driver checks after it builds the config."""

    def _built(self, **changes) -> dict:
        built = {
            name: BASE.config_overrides[name] for name in FUSION_FIELDS
        }
        built.update(changes)
        return built

    def test_a_matching_config_reports_nothing(self) -> None:
        self.assertEqual(declared_mismatches(BASE, self._built()), [])

    def test_a_fusion_that_came_out_off_is_reported(self) -> None:
        """The original hazard, and the one that already cost a wrong verdict.

        Building TransformerConfig directly bypasses megatron's argparse
        layer, where its real defaults live, so an unset flag takes the
        dataclass value. That ran the unfused chunk/silu/mul/copy path and
        cost 11.9 GPU ms/step -- an accidental handicap reported as an engine
        property.
        """
        wrong = declared_mismatches(
            BASE, self._built(bias_activation_fusion=False)
        )
        self.assertEqual(len(wrong), 1)
        self.assertIn("bias_activation_fusion", wrong[0])
        self.assertIn("declares True", wrong[0])
        self.assertIn("config has False", wrong[0])

    def test_a_delta_that_did_not_take_is_reported(self) -> None:
        """The new hazard, and the reason the check reads the declaration.

        The old check asserted every fusion was on, so it could only ever
        catch the first case -- and it would have rejected half the Part C
        roster outright. A variant whose flag silently failed to apply now
        fails the run instead of publishing the base implementation under the
        variant's name.
        """
        no_permute = derive(
            BASE,
            name="no_permute_fusion",
            description="the torch permute path instead of TE's fused one",
            config_overrides={"moe_permute_fusion": False},
        )
        # The delta took: the config carries what the profile declares.
        took = declared_mismatches(
            no_permute, self._built(moe_permute_fusion=False)
        )
        self.assertEqual(took, [])
        # The delta did not take: the config still holds the base value. The
        # old all-on check called this state correct.
        still_on = declared_mismatches(no_permute, self._built())
        self.assertEqual(len(still_on), 1)
        self.assertIn("declares False", still_on[0])

    def test_the_cross_entropy_implementation_is_checked_not_just_logged(
        self,
    ) -> None:
        """It is a string, not a boolean, and it is the largest loss lever.

        'te' is TE's online-softmax kernel at 14.9 GPU ms/step; 'native' is
        megatron's fp32 multi-pass one at 88. An all-boolean check could not
        see the difference at all.
        """
        wrong = declared_mismatches(
            BASE, self._built(cross_entropy_fusion_impl="native")
        )
        self.assertEqual(len(wrong), 1)
        self.assertIn("'te'", wrong[0])
        self.assertIn("'native'", wrong[0])

    def test_every_checked_field_is_declared_by_the_base_profile(self) -> None:
        """An undeclared field is skipped, so a typo would check nothing.

        The failure mode is silence: the run passes, the log line prints, and
        the flag it was supposed to guard is unguarded.
        """
        undeclared = [
            name
            for name in FUSION_FIELDS
            if name not in BASE.config_overrides
        ]
        self.assertEqual(undeclared, [])


class BuilderContractTests(unittest.TestCase):
    def test_the_megatron_builder_requires_an_explicit_profile(self) -> None:
        """The same rule the shape already has, one step further out.

        An omitted profile builds the base behaviour under whatever label the
        caller asked for -- a wrong number rather than a missing one, and one
        no correctness gate can see, because every profile is numerically
        valid. Signature inspection only; this imports no megatron.
        """
        import inspect

        from benchmarks.models.piper_qwen3.megatron_model import build_model

        parameter = inspect.signature(build_model).parameters["profile"]
        self.assertEqual(parameter.kind, inspect.Parameter.KEYWORD_ONLY)
        self.assertIs(parameter.default, inspect.Parameter.empty)


if __name__ == "__main__":
    unittest.main()
