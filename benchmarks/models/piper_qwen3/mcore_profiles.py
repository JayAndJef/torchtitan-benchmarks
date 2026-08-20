"""Megatron-core behavioural configuration as data, one profile per variant.

``benchmarks/models/piper_qwen3/megatron_model.py`` used to write every
``TransformerConfig`` flag inline, which made "megatron, but with the native
cross-entropy" inexpressible: there was no name for the variant and no way to
hand one to the builder. A profile is that name plus that delta.

**A profile is one flat dict of ``TransformerConfig`` values, and that is the
only surface.** ``megatron_model.build_model`` hands the built config to
``get_gpt_decoder_block_spec``, megatron's own config-to-spec derivation. That
function reads ``num_moe_experts``, ``moe_grouped_gemm`` and ``qk_layernorm``
off the config and turns each into a module-class choice
(``gpt_layer_specs.py:592-594``). So every setting is written once, here.

**Do not reintroduce a hand-built layer spec.** This module used to carry a
second ``spec_kwargs`` mapping, because the builder called the inner factory
``get_gpt_layer_with_transformer_engine_spec`` directly. That factory takes no
config, so ``moe_grouped_gemm`` and ``qk_layernorm`` had to be typed twice and
kept in agreement by hand. Megatron polices that agreement for ``qk_layernorm``
(``attention.py:1711`` raises) but for ``moe_grouped_gemm`` nowhere -- both
disagreement directions are silent there, and both publish one implementation
under the other's label. A layer spec is a generated artifact; writing it by
hand is what created the duplication, and the guard rail that went with it.

**A profile carries no geometry.** Every shape knob comes from
``benchmarks.models.piper_qwen3.shape.PiperShape``, which is the single source
of truth both engines build from. Duplicating a geometry field here would let
the two engines drift, which is the one property that module exists to
prevent.

**A profile is JSON-serializable, and the manifest records it verbatim.**
Torch values are encoded as names -- ``"silu"``, ``"bfloat16"`` -- and
``megatron_model.build_model`` resolves them. That keeps this module torch-free
and parent-side, so the harness can name, record and diff a profile without
importing the ML stack. ``__post_init__`` runs ``json.dumps`` over both
mappings, so a torch object written into a profile by accident fails at
declaration rather than at manifest-write time.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from benchmarks.models.piper_qwen3.shape import PiperShape

# Encoded torch values, and the only names build_model resolves. A profile may
# not name anything outside these; build_model raises on an unknown name too,
# so a hand-built profile cannot reach a run either.
ACTIVATION_FUNCS = ("silu", "gelu")
DTYPES = ("bfloat16", "float16", "float32")

ACTIVATION_FUNC_FIELDS = ("activation_func",)
DTYPE_FIELDS = ("params_dtype", "pipeline_dtype")

# The fields a run reports and checks against what its profile declares.
# Megatron's real defaults live in its argparse layer, which constructing
# TransformerConfig directly bypasses, so a flag left unset silently takes the
# dataclass value -- once costing 11.9 GPU ms/step of unfused SwiGLU. These
# are the flags where that has bitten or would bite hardest.
#
# cross_entropy_fusion_impl is last and is not a boolean: which CE ran is the
# single largest lever in the loss path, so it belongs in the same line.
FUSION_FIELDS = (
    "bias_activation_fusion",
    "bias_dropout_fusion",
    "cross_entropy_loss_fusion",
    "moe_permute_fusion",
    "apply_rope_fusion",
    "cross_entropy_fusion_impl",
)


@dataclass(frozen=True)
class McoreProfile:
    """One named megatron-core behavioural configuration."""

    name: str
    description: str
    config_overrides: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        try:
            json.dumps(self.config_overrides)
        except TypeError as error:
            raise ValueError(
                f"profile {self.name!r}: config_overrides must be JSON-safe, "
                "so encode a torch value as its name (activation_func="
                '"silu", params_dtype="bfloat16"). This module is '
                f"parent-side and imports no torch. {error}"
            ) from error
        for name in ACTIVATION_FUNC_FIELDS:
            value = self.config_overrides.get(name)
            if value is not None and value not in ACTIVATION_FUNCS:
                raise ValueError(
                    f"profile {self.name!r}: {name}={value!r} is not a known "
                    f"activation name. Available: {', '.join(ACTIVATION_FUNCS)}"
                )
        for name in DTYPE_FIELDS:
            value = self.config_overrides.get(name)
            if value is not None and value not in DTYPES:
                raise ValueError(
                    f"profile {self.name!r}: {name}={value!r} is not a known "
                    f"dtype name. Available: {', '.join(DTYPES)}"
                )
    def describe(self) -> dict[str, Any]:
        """Flat JSON-safe provenance record for the manifest."""
        return {
            "name": self.name,
            "description": self.description,
            "config_overrides": dict(self.config_overrides),
        }


def derive(
    base: McoreProfile,
    *,
    name: str,
    description: str,
    config_overrides: dict[str, Any] | None = None,
) -> McoreProfile:
    """A new profile: ``base`` with these keys replaced.

    A delta names only what it changes, so a reader sees the difference and
    not a second copy of 28 flags. ``McoreProfile.__post_init__`` still
    validates the merged result, so a delta cannot smuggle in a torch value or
    an unknown encoded name.
    """
    return McoreProfile(
        name=name,
        description=description,
        config_overrides={**base.config_overrides, **(config_overrides or {})},
    )


# The behavioural half of what megatron_model.py wrote inline before this
# module existed. Extracted verbatim: the kwargs this profile produces are
# byte-for-byte the ones that built every megatron number published so far,
# which tests/test_mcore_profiles.py pins against a frozen literal.
#
# Two comments from that block are load-bearing and travel with the flags they
# explain, because deleting either is how the handicap comes back.
BASE = McoreProfile(
    name="base",
    description=(
        "megatron-core + TransformerEngine at its own best: every fusion "
        "megatron's training entrypoint enables, plus TE's cross-entropy"
    ),
    config_overrides={
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
        # Post-softmax top-k: top-k on logits then softmax over the selected
        # k, algebraically identical to Qwen3's softmax-then-topk-then-renorm.
        "moe_router_pre_softmax": False,
        "moe_router_load_balancing_type": "none",
        "moe_aux_loss_coeff": 0.0,
        "moe_router_enable_expert_bias": False,
        "moe_router_dtype": "fp32",
        "moe_token_dispatcher_type": "allgather",
        # This one and qk_layernorm above are read TWICE by megatron: the
        # derivation turns each into a module-class choice (TEGroupedMLP vs
        # the per-expert MLP; a real norm vs IdentityOp), and other code reads
        # the same field at run time. Both reads come off this one value, so
        # they cannot disagree -- which is exactly what a hand-built layer
        # spec used to make possible.
        "moe_grouped_gemm": True,
        "apply_rope_fusion": True,
        # TransformerConfig's dataclass defaults are NOT megatron's defaults:
        # the ones a real run gets are set in the argparse layer we bypass by
        # constructing the config directly. --no-bias-swiglu-fusion and
        # --no-bias-dropout-fusion are both action="store_false" (so argparse
        # defaults them True) and argument_utils.py forwards the swiglu one as
        # bias_activation_fusion. Leaving them at the dataclass False ran the
        # unfused chunk/silu/mul/copy path in moe/experts.py and cost 11.9 GPU
        # ms/step -- an accidental handicap, not a property of the engine.
        "bias_activation_fusion": True,
        "bias_dropout_fusion": True,
        # Opt-in in megatron, but shipped in NVIDIA's own Qwen3 example
        # configs, and both are parity-safe pure kernel fusions (measured:
        # loss trajectories match the unfused run to ~1e-3).
        "cross_entropy_loss_fusion": True,
        # 'te' routes the loss through transformer_engine.pytorch.
        # parallel_cross_entropy -- the SAME implementation our te_fused_ce
        # arm wraps (vendored at benchmarks/models/piper_qwen3/components/
        # lm_head/te_cross_entropy.py), and the one piper_optimized_te_ce
        # further optimizes. Choosing it makes the loss path an
        # apples-to-apples kernel comparison instead of an algorithm one.
        #
        # 'native' is megatron's own jit_fuser CE: it upcasts the whole
        # [tokens, 151936] logits tensor to fp32 and makes ~6 full-tensor
        # traversals (max, sub, exp, sum, div) whose fp32 softmax it also
        # keeps resident for backward -- 88 GPU ms/step at batch 48 versus
        # 14.9 for the online-softmax TE-family kernel, which streams the
        # row in two bf16 passes and writes the gradient in place.
        #
        # NOTE: megatron's *training entrypoint* refuses this combination
        # (arguments.py ~1630) citing known stability issues. The core config
        # only warns (model_parallel_config.py ~536), and we construct the
        # config directly, so we get it. That is a deliberate choice to
        # measure megatron's fastest available loss path; it is NOT the
        # configuration a stock `pretrain_gpt.py` user runs. Report numbers
        # from this setting as "megatron with its fastest available CE", and
        # flip back to "native" for "megatron as NVIDIA ships it".
        "cross_entropy_fusion_impl": "te",
        "moe_permute_fusion": True,
        "bf16": True,
        "params_dtype": "bfloat16",
        "pipeline_dtype": "bfloat16",
        # The one performance default we knowingly decline: the fused wgrad
        # path needs apex-style main_grad buffers no DDP wrapper provides
        # here, so a bare backward would never populate .grad.
        "gradient_accumulation_fusion": False,
    },
)


# The one delta two kernel scenarios share, and the reason it lives here
# rather than in either of them. ``bias_dropout_fusion`` is a single
# ``TransformerConfig`` field, and megatron reads it at two call sites:
# ``self_attn_bda`` (``transformer_layer.py:684``), which the ``attn_residual``
# scenario cuts, and ``mlp_bda`` (``:980``), which ``moe_residual`` cuts. Two
# local copies would be one object described twice, and a reader would have to
# diff them to learn that.
#
# ``BASE`` sets the flag True because megatron's own argparse layer does:
# ``--no-bias-dropout-fusion`` is ``action="store_false"``, and the
# ``TransformerConfig`` dataclass default is the opposite of what a real
# megatron run gets. Turning the flag off is therefore a deviation from
# megatron, not a return to its default.
#
# It stays out of ``MCORE_PROFILES`` deliberately. That dict is the roster of
# profiles other systems run, and the e2e megatron arm runs ``base``. The
# ``cross_entropy`` scenario keeps its two variants out of the roster for the
# same reason.
NO_BIAS_DROPOUT_FUSION: McoreProfile = derive(
    BASE,
    name="no_bias_dropout_fusion",
    description=(
        "megatron with the bias-dropout-add fusion off: both bda call sites "
        "resolve to bias_dropout_add_unfused, which builds a Python closure "
        "per call and then dispatches the same arithmetic eagerly, instead "
        "of the @jit_fuser-compiled bias_dropout_add_fused_train. The device "
        "work is one bf16 add either way, so this profile isolates the cost "
        "of the compiled region and nothing else"
    ),
    config_overrides={"bias_dropout_fusion": False},
)


MCORE_PROFILES: dict[str, McoreProfile] = {
    profile.name: profile for profile in (BASE,)
}


def profile_by_name(name: str) -> McoreProfile:
    try:
        return MCORE_PROFILES[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown mcore profile {name!r}. Available: "
            + ", ".join(MCORE_PROFILES)
        ) from error


def declared_mismatches(
    profile: McoreProfile, built: dict[str, Any]
) -> list[str]:
    """Where a built config disagrees with what ``profile`` declares.

    Lives here rather than in the megatron driver so it is torch-free and
    testable on CPU: the driver's copy could only be exercised by a real
    megatron run, which is the least convenient place to find a typo.

    Both directions are failures. A flag declared on that came out off is the
    dataclass-default handicap that once cost 11.9 GPU ms/step of unfused
    SwiGLU. A flag declared off that came out on is a delta that did not take,
    which publishes the base implementation under the variant's name.

    Fields the profile does not declare are not checked -- it has nothing to
    say about them -- which is why ``FUSION_FIELDS`` and the base profile must
    agree. A test pins that.
    """
    return sorted(
        f"{name}: profile declares {profile.config_overrides[name]!r}, "
        f"config has {built.get(name)!r}"
        for name in FUSION_FIELDS
        if name in profile.config_overrides
        and built.get(name) != profile.config_overrides[name]
    )


def geometry_config_kwargs(shape: PiperShape) -> dict[str, Any]:
    """The ``TransformerConfig`` fields that come from the shape, not a profile."""
    return {
        "num_layers": shape.n_layers,
        "hidden_size": shape.dim,
        "num_attention_heads": shape.n_heads,
        "num_query_groups": shape.n_kv_heads,
        "kv_channels": shape.head_dim,
        # Required field; unused (every layer is MoE) but kept at the expert
        # width so nothing dense-sized is ever allocated from it.
        "ffn_hidden_size": shape.moe_hidden_dim,
        "num_moe_experts": shape.num_experts,
        "moe_router_topk": shape.top_k,
        "moe_ffn_hidden_size": shape.moe_hidden_dim,
    }


def transformer_config_kwargs(
    *,
    shape: PiperShape,
    profile: McoreProfile,
    cuda_graph_impl: str | None = None,
    cuda_graph_modules: tuple[str, ...] = (),
    use_cpu_initialization: bool = False,
) -> dict[str, Any]:
    """Every ``TransformerConfig`` keyword, with torch values still encoded.

    Geometry, then the profile, then the runtime knobs -- the three categories
    the old inline block held in one literal. Returns a fresh dict per call:
    the correctness pass builds several arms in one process, so a mutated
    profile mapping would contaminate the next build.

    Torch-free by construction, which is what lets the frozen-literal test
    assert the whole payload on CPU without importing megatron.
    """
    kwargs: dict[str, Any] = geometry_config_kwargs(shape)
    kwargs.update(profile.config_overrides)
    if use_cpu_initialization:
        kwargs["use_cpu_initialization"] = True
    if cuda_graph_impl is not None:
        kwargs["cuda_graph_impl"] = cuda_graph_impl
        if cuda_graph_modules:
            kwargs["cuda_graph_modules"] = list(cuda_graph_modules)
        # TE's attention asserts on the RNG tracker type inside captured
        # graphs; its own tracker is the supported one.
        kwargs["use_te_rng_tracker"] = True
    return kwargs
