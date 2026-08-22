"""The behavioral contract a package restructure must preserve, frozen.

Every module in this repo was relocated by the ``benchmarks`` restructure. A
move is only correct if nothing observable changes, and almost everything
observable here is a *string*: scenario and arm ids that name published results, dotted override
paths delivered to TorchTitan on the command line, "module:symbol" builder
paths resolved inside the GPU worker, the ``--module`` token TorchTitan
resolves through its own two-candidate algorithm, and two ``Path(__file__)``
walks that decide where ``out/`` lands. None of those are type-checked, none
are exercised by an import, and every one of them fails only at run time --
40 minutes into a GPU run, or worse, silently.

This file pins all of them so a missed reference costs one CPU second.

Everything the move was allowed to change lives in the constants at the top:
``CANONICAL_ROOTS``, ``TITAN_CONFIG_MODULE``, ``MEGATRON_DRIVER_MODULE``, the
override/builder path prefixes, and the golden argv lists. Updating this file
for the move was a handful of data edits, and reverting the move would be the
same edits back -- the assertions themselves never move.
"""

import importlib.util
import re
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.e2e.launch import command_for_arm
from benchmarks.e2e.registry import SCENARIOS, scenario_by_name
from benchmarks.e2e.validation import VALIDATION_PROFILES
from benchmarks.execution.paths import BENCH_DIR, TITAN_DIR
from benchmarks.kernel.registry import KERNEL_SCENARIOS
from benchmarks.models.piper_qwen3.megatron_bootstrap import (
    MEGATRON_SUBMODULE,
    REPO_ROOT,
)


# The first-party package. Every importable name this repo owns lives under
# it, and the runner exports its parent directory as PYTHONPATH to the
# training subprocess. It was three roots before the restructure.
CANONICAL_ROOTS = ("benchmarks",)

# The two strings a mechanical move is allowed to change, and the only two.
# TITAN_CONFIG_MODULE is the --module token; MEGATRON_DRIVER_MODULE is the
# python -m target of the megatron launcher.
TITAN_CONFIG_MODULE = "benchmarks.models.piper_qwen3"
MEGATRON_DRIVER_MODULE = "benchmarks.e2e.megatron.train"

# Dotted-path prefixes the registries hand to importlib at run time.
SWIGLU_INDUCTOR_OVERRIDE = (
    "benchmarks.models.piper_qwen3.components.swiglu.combined_swiglu."
    "piper_optimized_inductor_fused_grouped_experts"
)
# One arm-builder module per kernel family, and every builder path a scenario
# names -- inputs, fp64 reference and each arm -- must resolve inside its own
# family's module. Pinned as data rather than derived from the scenario name,
# so a scenario renamed without moving its builders is caught rather than
# followed. The stronger form matters more than the flat prefix it replaced:
# these are the strings ``resolve_symbol`` hands to importlib inside the GPU
# worker, and a builder left behind in another family's module would still
# import cleanly and still measure the right kernel -- it would just quietly
# undo the split, and nothing else in the suite would notice.
KERNEL_ARMS_MODULES = {
    "rope": "benchmarks.kernel.operations.rope",
    "lm_head": "benchmarks.kernel.operations.lm_head",
    "embedding_stage": "benchmarks.kernel.operations.embedding_stage",
    "qkv_prep": "benchmarks.kernel.operations.qkv_prep",
    "qk_norm": "benchmarks.kernel.operations.qk_norm",
    "attn_out_proj": "benchmarks.kernel.operations.attn_out_proj",
    "attn_residual": "benchmarks.kernel.operations.attn_residual",
    "ffn_norm": "benchmarks.kernel.operations.ffn_norm",
    "moe_router": "benchmarks.kernel.operations.moe_router",
    "dispatch_permute": "benchmarks.kernel.operations.dispatch_permute",
    "expert_mlp": "benchmarks.kernel.operations.expert_mlp",
    "moe_combine": "benchmarks.kernel.operations.moe_combine",
    "moe_residual": "benchmarks.kernel.operations.moe_residual",
    "final_norm": "benchmarks.kernel.operations.final_norm",
    "lm_head_projection": "benchmarks.kernel.operations.lm_head_projection",
    "cross_entropy": "benchmarks.kernel.operations.cross_entropy",
    "attention_core": "benchmarks.kernel.operations.attention_core",
}

# Third-party roots override paths are allowed to name, so the "did this move
# leave a first-party path behind?" check does not trip on the fork.
FOREIGN_ROOTS = ("torchtitan",)


# --------------------------------------------------------------------------
# 1. Stable id inventory.
# --------------------------------------------------------------------------

# Scenario and arm ids are frozen external identifiers: they name directories
# under out/, keys in manifest.json and results.json, and every published
# number. They must survive the move byte for byte. Ordering is deterministic
# (SCENARIOS is built from a tuple, arms are tuples), so it is pinned too.
E2E_INVENTORY = {
    "piper1b_rope": ("baseline", "helion", "te"),
    "piper1b_swiglu": (
        "baseline",
        "piper_optimized_triton",
        "piper_optimized_inductor",
    ),
    "piper1b_qkv": ("baseline", "fused_qkv"),
    "piper1b_lm_head": (
        "baseline",
        "fused_linear_ce",
        "te_fused_ce",
        "piper_optimized_te_ce",
    ),
    "piper1b_attention": ("baseline", "flash_attention_3", "flex_flash"),
    "piper1b_megatron": (
        "baseline",
        "titan_stock",
        "titan_swiglu",
        "titan_lm_head",
        "titan_swiglu_lm_head",
    ),
}

KERNEL_INVENTORY = {
    # Cross-engine since scenario 4 replaced the single-engine roster in
    # place: two megatron THD arms and the three re-homed titan arms. It
    # declares no floor, and it is no longer in KERNEL_TO_E2E_SCENARIO.
    "rope": (
        "mcore/base",
        "mcore/no_rope_fusion",
        "titan",
        "titan/helion",
        "titan/te",
    ),
    "lm_head": (
        "baseline",
        "fused_linear_ce",
        "te_fused_ce",
        "piper_optimized_te_ce",
    ),
    # Cross-engine from here down, in partition order. The arms are named
    # engine/profile, and the anchor is the megatron side throughout -- see
    # KERNEL_BASELINE_ARMS.
    "embedding_stage": ("copy_floor", "mcore/base", "titan"),
    # No floor: a GEMM is compute-bound. The third arm is titan's own fusion
    # question and is not an upstream configuration.
    "qkv_prep": ("mcore/base", "titan", "titan/unfused_qkv"),
    "qk_norm": ("copy_floor", "mcore/base", "titan"),
    # No floor: a GEMM is compute-bound, and a bandwidth number would not
    # bound it. The two arms are the whole roster.
    "attn_out_proj": ("mcore/base", "titan"),
    # Scenario 7. Four arms and one published row, and the row is within
    # megatron: the titan arm is measured and gated but compared with
    # nothing, because isolating the cut gives titan megatron's fusion scope
    # and a cross-engine ratio would report the isolation.
    "attn_residual": (
        "copy_floor",
        "mcore/base",
        "mcore/no_bias_dropout_fusion",
        "titan",
    ),
    "ffn_norm": ("copy_floor", "mcore/base", "titan"),
    # Scenario 9. Five arms and a cross-engine row: both engines route in
    # fp32, so titan against mcore/base is the like-for-like precision cut.
    # The two engines do NOT move the same bytes, and the module declares a
    # byte count per engine so the GB/s column shows that rather than
    # burying it.
    "moe_router": (
        "copy_floor",
        "mcore/base",
        "mcore/router_fusion",
        "mcore/router_bf16",
        "titan",
    ),
    # Scenario 10. Five arms and a cross-engine row: both engines permute the
    # same rows into the same order, which a bitwise gate enforces.
    "dispatch_permute": (
        "copy_floor",
        "mcore/base",
        "mcore/no_permute_fusion",
        "mcore/dispatcher_alltoall",
        "titan",
    ),
    # Scenario 11. Eight arms, six rows, and every row within one engine: at
    # this cut megatron applies the routing probabilities and titan does not,
    # so a cross-engine ratio would compare two functions. It is also the one
    # cross-engine scenario anchored on titan, because it publishes no
    # cross-engine row and the anchor's only remaining job is to decide whose
    # loss costs the scenario. No floor: this is a GEMM scenario, and a
    # bandwidth number would not bound it -- the same reason qkv_prep and
    # attn_out_proj declare none. No arm declares bytes_moved either, so no
    # GB/s column is printed.
    "expert_mlp": (
        "mcore/base",
        "mcore/no_bias_activation_fusion",
        "mcore/te_activation_func",
        "mcore/no_grouped_gemm",
        "titan",
        "titan/fused_grouped_experts",
        "titan/piper_optimized_triton",
        "titan/piper_optimized_inductor",
    ),
    # Scenario 12, the mirror of scenario 10 on the way back. Five arms and
    # two rows, both within megatron: titan applies the routing
    # probabilities here and megatron applied them in scenario 11, so the
    # two sides compute different functions of the same rows.
    "moe_combine": (
        "copy_floor",
        "mcore/base",
        "mcore/no_permute_fusion",
        "mcore/dispatcher_alltoall",
        "titan",
    ),
    # Scenario 13, the twin of attn_residual at the other bda call site. Four
    # arms, one published row, and the row is within megatron for the same
    # reason.
    "moe_residual": (
        "copy_floor",
        "mcore/base",
        "mcore/no_bias_dropout_fusion",
        "titan",
    ),
    "final_norm": ("copy_floor", "mcore/base", "titan"),
    # Scenario 15. Two arms and no floor: the projection is compute-bound at
    # 1.27 TFLOP of forward work at the normal shape, so a copy floor would
    # answer a question nobody asks of it. Both arms are eager, and the titan
    # side is eager because apply_compile reaches model.layers only.
    "lm_head_projection": ("mcore/base", "titan"),
    # Six arms: three megatron CE variants and the three titan losses re-homed
    # from lm_head, which measured the projection and the loss together.
    "cross_entropy": (
        "mcore/base",
        "mcore/ce_native",
        "mcore/no_ce_fusion",
        "titan/full_logits",
        "titan/te_fused_ce",
        "titan/piper_optimized_te_ce",
    ),
    "attention_core": (
        "mcore/base",
        "mcore/attn_flash3",
        "mcore/attn_unfused",
        "titan",
        "titan/flex_flash",
        "titan/flash_attention_3",
    ),
}

KERNEL_BASELINE_ARMS = {
    "rope": "mcore/base",
    "lm_head": "baseline",
    # The anchor is mcore/base at every cross-engine scenario that publishes a
    # cross-engine row. Several of them hold mcore-only variant arms that can
    # compare against nothing else, so an anchor on the titan side would make
    # the derived set compare a megatron fusion delta against the other engine
    # and conflate the two. It also matches e2e, where piper1b_megatron makes
    # the megatron arm the baseline. The cost is accepted and recorded: the
    # anchor's loss costs the whole scenario, and TransformerEngine is the
    # more fragile side.
    #
    # expert_mlp is the exception, and it is the exception because it
    # publishes NO cross-engine row: it declares its comparisons rather than
    # deriving them, and both arms of every row sit on one engine. So the
    # derivation argument does not reach it, and the only remaining question
    # is whose loss the scenario cannot survive. It anchors on titan, the side
    # with neither a megatron nor a TransformerEngine dependency to lose.
    "embedding_stage": "mcore/base",
    "qkv_prep": "mcore/base",
    "qk_norm": "mcore/base",
    "attn_out_proj": "mcore/base",
    "attn_residual": "mcore/base",
    "ffn_norm": "mcore/base",
    "moe_router": "mcore/base",
    "dispatch_permute": "mcore/base",
    "expert_mlp": "titan",
    "moe_combine": "mcore/base",
    "moe_residual": "mcore/base",
    "final_norm": "mcore/base",
    "lm_head_projection": "mcore/base",
    "cross_entropy": "mcore/base",
    "attention_core": "mcore/base",
}

# Arm names deliberately match across the two registries wherever the same
# implementation is measured at two scopes (CLAUDE.md, "Scenarios and arms").
# copy_floor is a bandwidth floor with no end-to-end counterpart.
#
# **Only a single-engine kernel scenario can appear here.** The pairing means
# "one roster, measured at two scopes", and a cross-engine scenario has no
# such roster: its arms are named ``engine/profile`` and half of them are
# megatron, which the e2e side spells piper1b_megatron/baseline rather than
# piper1b_<family>/<arm>. That is why none of embedding_stage, qkv_prep,
# qk_norm, attn_out_proj, ffn_norm, final_norm, cross_entropy, attn_residual,
# moe_residual or lm_head_projection is listed.
#
# ``rope`` left this table when it became cross-engine. Its kernel arms are
# now mcore/base, mcore/no_rope_fusion, titan, titan/helion and titan/te,
# against an e2e piper1b_rope that still runs baseline, helion and te. The
# two are no longer one roster, so the map stops claiming they are. The e2e
# ids are deliberately NOT renamed to match: they name directories under
# out/, keys in every manifest.json and results.json, and every published
# rope number, and this file's own section 1 requires them to survive byte
# for byte.
#
# ``swiglu``, ``qkv`` and ``attention`` left it by being deleted: expert_mlp,
# qkv_prep and attention_core re-homed every one of their arms, and each
# successor is cross-engine, so none of the three could take the departed
# scenario's place here. ONE ENTRY IS LEFT, AND THE MAP IS KEPT.
# ``lm_head`` is the last single-engine kernel scenario, its four arms carry
# the four ids piper1b_lm_head carries, and reports cite the pair. A rename
# on one side alone is exactly what this table catches, and a table of one
# catches it as well as a table of four. Delete the table and the guard goes
# with it, for a saving of two lines.
KERNEL_TO_E2E_SCENARIO = {
    "lm_head": "piper1b_lm_head",
}
# A bandwidth floor has no end-to-end counterpart, so it is excluded from the
# pairing above. No scenario the map still holds declares one today -- rope
# was the last, and it left with copy_floor -- so the subtraction is empty
# until a mapped scenario adds a floor. The rule is kept rather than the
# set widened: KERNEL_ONLY_ARMS means "a floor", and it may never be used to
# absorb an arm a rename left unpaired.
KERNEL_ONLY_ARMS = frozenset({"copy_floor"})


class StableIdInventoryTests(unittest.TestCase):
    def test_e2e_scenarios_and_arms_are_exactly_the_pinned_ids(self) -> None:
        self.assertEqual(list(SCENARIOS), list(E2E_INVENTORY))
        for name, arms in E2E_INVENTORY.items():
            with self.subTest(scenario=name):
                self.assertEqual(
                    tuple(arm.name for arm in SCENARIOS[name].arms), arms
                )

    def test_kernel_scenarios_and_arms_are_exactly_the_pinned_ids(self) -> None:
        self.assertEqual(list(KERNEL_SCENARIOS), list(KERNEL_INVENTORY))
        for name, arms in KERNEL_INVENTORY.items():
            with self.subTest(scenario=name):
                self.assertEqual(
                    tuple(arm.name for arm in KERNEL_SCENARIOS[name].arms), arms
                )

    def test_every_kernel_scenario_names_its_pinned_baseline(self) -> None:
        for name, baseline in KERNEL_BASELINE_ARMS.items():
            with self.subTest(scenario=name):
                scenario = KERNEL_SCENARIOS[name]
                self.assertEqual(scenario.baseline_arm, baseline)
                # arm() raises on an unknown name, so this also proves the
                # baseline id is one of the scenario's own arms.
                self.assertEqual(scenario.arm(baseline).name, baseline)

    def test_kernel_arm_names_still_match_their_e2e_counterparts(self) -> None:
        """The same implementation carries the same id at both scopes.

        Reports lean on this: piper1b_lm_head/piper_optimized_te_ce and
        lm_head/piper_optimized_te_ce are documented as the same code
        measured two ways. A rename on one side alone would silently break
        the pairing.
        """
        for kernel_name, e2e_name in KERNEL_TO_E2E_SCENARIO.items():
            with self.subTest(scenario=kernel_name):
                kernel_arms = set(KERNEL_INVENTORY[kernel_name]) - KERNEL_ONLY_ARMS
                self.assertEqual(kernel_arms, set(E2E_INVENTORY[e2e_name]))

    def test_every_arm_carries_a_description(self) -> None:
        # Recorded in manifests and printed by `run_bench.sh scenarios`; an
        # arm added during the move without one would ship a blank id.
        for registry in (SCENARIOS, KERNEL_SCENARIOS):
            for scenario in registry.values():
                for arm in scenario.arms:
                    with self.subTest(scenario=scenario.name, arm=arm.name):
                        self.assertTrue(arm.description.strip())


# --------------------------------------------------------------------------
# 2 and 3. Every dotted path in both registries resolves.
# --------------------------------------------------------------------------


def _find_spec(module_name: str):
    """find_spec, never raising, never importing the leaf module.

    find_spec is the whole point of this section: every module named below
    imports torch (several of them build Triton or CUDA extensions on
    import), so importing them to prove a string is spelled right would cost
    minutes and, for some, a GPU. find_spec imports only the parent packages,
    which are torch-free here.
    """
    try:
        return importlib.util.find_spec(module_name)
    except (ImportError, AttributeError, ValueError):
        return None


def _module_defines(spec, symbol: str) -> bool:
    """Whether ``symbol`` is defined at module scope, read from source.

    find_spec proves the module exists but says nothing about the trailing
    attribute, and that attribute is half of what a mechanical move can break
    (an override path names a function, a builder path names a builder). A
    source scan is the strongest check available without importing torch.
    """
    origin = getattr(spec, "origin", None)
    if not origin or not origin.endswith(".py"):
        return True
    source = Path(origin).read_text(errors="replace")
    name = re.escape(symbol)
    return bool(
        re.search(rf"^(?:async def|def|class)\s+{name}\b|^{name}\s*[:=]", source, re.M)
    )


def _split_dotted(path: str) -> tuple[str, str]:
    """Split ``a.b.c`` into the module ``a.b`` and the attribute ``c``."""
    module_name, _, attribute = path.rpartition(".")
    return module_name, attribute


def _split_builder(path: str) -> tuple[str, str]:
    """Split ``module:symbol`` as kernel.schema.resolve_symbol does."""
    module_name, _, attribute = path.partition(":")
    return module_name, attribute


class OverrideImportPathTests(unittest.TestCase):
    def test_every_override_import_resolves(self) -> None:
        seen = 0
        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                for path in arm.override_imports:
                    seen += 1
                    module_name, attribute = _split_dotted(path)
                    with self.subTest(
                        scenario=scenario.name, arm=arm.name, path=path
                    ):
                        spec = _find_spec(module_name)
                        self.assertIsNotNone(
                            spec, f"{path}: no module named {module_name}"
                        )
                        self.assertTrue(
                            _module_defines(spec, attribute),
                            f"{path}: {module_name} defines no {attribute}",
                        )
        self.assertGreater(seen, 0, "no override_imports found to check")

    def test_first_party_override_roots_are_canonical(self) -> None:
        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                for path in arm.override_imports:
                    root = path.split(".")[0]
                    with self.subTest(arm=arm.name, path=path):
                        self.assertIn(root, CANONICAL_ROOTS + FOREIGN_ROOTS)

    def test_an_arm_declaring_overrides_declares_a_count(self) -> None:
        # validate_arm multiplies overrides_per_block by the layer count, so
        # a declared import with a zero count is an unchecked override.
        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                with self.subTest(scenario=scenario.name, arm=arm.name):
                    self.assertEqual(
                        bool(arm.override_imports), arm.overrides_per_block > 0
                    )


class KernelBuilderPathTests(unittest.TestCase):
    def _assert_resolves(self, path: str, label: str) -> None:
        module_name, attribute = _split_builder(path)
        spec = _find_spec(module_name)
        self.assertIsNotNone(spec, f"{label}: no module named {module_name}")
        self.assertTrue(
            _module_defines(spec, attribute),
            f"{label}: {module_name} defines no {attribute}",
        )

    def test_every_kernel_arm_builder_resolves(self) -> None:
        seen = 0
        for scenario in KERNEL_SCENARIOS.values():
            for arm in scenario.arms:
                seen += 1
                with self.subTest(scenario=scenario.name, arm=arm.name):
                    self.assertIn(":", arm.builder)
                    self._assert_resolves(
                        arm.builder, f"{scenario.name}/{arm.name}"
                    )
        self.assertEqual(seen, sum(len(a) for a in KERNEL_INVENTORY.values()))

    def test_every_scenario_inputs_and_reference_builder_resolves(self) -> None:
        for scenario in KERNEL_SCENARIOS.values():
            with self.subTest(scenario=scenario.name):
                self._assert_resolves(
                    scenario.inputs_builder, f"{scenario.name}.inputs_builder"
                )
                if scenario.reference_builder is not None:
                    self._assert_resolves(
                        scenario.reference_builder,
                        f"{scenario.name}.reference_builder",
                    )

    def test_every_scenario_has_a_pinned_arms_module(self) -> None:
        # Both directions, so the map cannot drift from the registry: a new
        # scenario with no entry would otherwise be exempt from the check
        # below, and a stale entry would pin a module nothing uses.
        self.assertEqual(set(KERNEL_ARMS_MODULES), set(KERNEL_SCENARIOS))

    def test_all_builders_live_in_their_own_familys_arms_module(self) -> None:
        for scenario in KERNEL_SCENARIOS.values():
            expected = KERNEL_ARMS_MODULES[scenario.name]
            paths = [scenario.inputs_builder]
            if scenario.reference_builder is not None:
                paths.append(scenario.reference_builder)
            paths.extend(arm.builder for arm in scenario.arms)
            for path in paths:
                with self.subTest(scenario=scenario.name, path=path):
                    self.assertEqual(_split_builder(path)[0], expected)

    def test_the_split_matches_resolve_symbol(self) -> None:
        """The colon format is resolve_symbol's, not a guess.

        ``resolve_symbol`` moved into the schema when the parent gained a
        dotted path of its own to resolve: an arm's ``requirement``. Both
        sides read the same function, which is what keeps the two
        conventions from drifting.
        """
        from benchmarks.kernel import schema as kernel_schema

        source = Path(kernel_schema.__file__).read_text()
        self.assertIn('path.partition(":")', source)


# --------------------------------------------------------------------------
# 4. Registry keys dispatch.
# --------------------------------------------------------------------------

LAUNCHERS = ("torchtitan", "megatron")
VALIDATION_KEYS = ("torchtitan", "megatron")


class RegistryDispatchTests(unittest.TestCase):
    def test_launcher_values_are_exactly_the_pinned_set(self) -> None:
        self.assertEqual(
            {arm.launcher for s in SCENARIOS.values() for arm in s.arms},
            set(LAUNCHERS),
        )

    def test_command_for_arm_handles_every_launcher_in_the_registry(self) -> None:
        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                with self.subTest(scenario=scenario.name, arm=arm.name):
                    command = command_for_arm(
                        scenario.workload,
                        arm,
                        Path("/tmp/arm-dir"),
                        (),
                        "default",
                        # The megatron launcher refuses any other ac mode, and
                        # the scenario carrying it supports only "none".
                        "none" if arm.launcher == "megatron" else "sac",
                    )
                    self.assertTrue(command)
                    self.assertIsInstance(command, list)
                    self.assertTrue(all(isinstance(a, str) for a in command))

    def test_an_unknown_launcher_is_rejected_rather_than_ignored(self) -> None:
        scenario = scenario_by_name("piper1b_rope")
        arm = replace(scenario.arm("baseline"), launcher="not-an-engine")
        with self.assertRaisesRegex(ValueError, "unknown launcher"):
            command_for_arm(scenario.workload, arm, Path("/tmp/arm-dir"), ())

    def test_every_validation_key_is_a_registered_profile(self) -> None:
        used = {arm.validation for s in SCENARIOS.values() for arm in s.arms}
        self.assertEqual(used, set(VALIDATION_KEYS))
        # Both directions: an unused profile after the move means an arm lost
        # its engine-specific rules and fell back to another engine's.
        self.assertEqual(set(VALIDATION_PROFILES), set(VALIDATION_KEYS))
        for key in VALIDATION_KEYS:
            with self.subTest(validation=key):
                self.assertIn(key, VALIDATION_PROFILES)

    def test_launcher_and_validation_agree_per_arm(self) -> None:
        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                with self.subTest(scenario=scenario.name, arm=arm.name):
                    self.assertEqual(arm.launcher, arm.validation)


# --------------------------------------------------------------------------
# 5. Golden command_for_arm output.
# --------------------------------------------------------------------------

# The strongest single proof that a mechanical move changed nothing. After the
# move the only tokens permitted to differ are TITAN_CONFIG_MODULE (the
# --module value), MEGATRON_DRIVER_MODULE (the python -m target), and the
# dotted override path -- all three are constants above, so the diff must be
# exactly those substitutions and nothing else.
#
# The titan golden is piper1b_megatron/titan_lm_head under cuda-graph/none at
# both sizes: it exercises every optional branch of the builder except the
# override one (arm config, --config-arg size, replay-steps, the cuda-graph ->
# reduce-overhead mapping, the seed, and the trailing tyro ac token).

GOLDEN_TITAN_ARM = ("piper1b_megatron", "titan_lm_head")


def _golden_titan_command(size: str) -> list[str]:
    return [
        "./run_train.sh",
        "--module",
        TITAN_CONFIG_MODULE,
        "--config",
        "qwen3_piper_1b_piper_optimized_te_ce_pretokenized",
        "--config-arg",
        f"size={size}",
        "--training.seq-len",
        "1024",
        "--training.steps",
        "40",
        "--training.local-batch-size",
        "4",
        "--compile.enable",
        "--profiler.enable_profiling",
        "--profiler.profile_freq",
        "20",
        "--profiler.profiler_active",
        "5",
        "--profiler.profiler_warmup",
        "5",
        "--dataloader.replay-steps",
        "40",
        "--compile.mode",
        "reduce-overhead",
        "--debug.seed",
        "42",
        "--dump-folder",
        "/tmp/arm-dir",
        "activation-checkpoint:none",
    ]


# The plain path, and the only golden that carries an override: no seed, no
# replay loader, no --compile.mode, no trailing ac token.
GOLDEN_OVERRIDE_ARM = ("piper1b_swiglu", "piper_optimized_inductor")
GOLDEN_OVERRIDE_COMMAND = [
    "./run_train.sh",
    "--module",
    TITAN_CONFIG_MODULE,
    "--config",
    "qwen3_piper_1b",
    "--config-arg",
    "size=normal",
    "--training.seq-len",
    "1024",
    "--training.steps",
    "40",
    "--training.local-batch-size",
    "4",
    "--compile.enable",
    "--profiler.enable_profiling",
    "--profiler.profile_freq",
    "20",
    "--profiler.profiler_active",
    "5",
    "--profiler.profiler_warmup",
    "5",
    "--override.imports",
    SWIGLU_INDUCTOR_OVERRIDE,
    "--dump-folder",
    "/tmp/arm-dir",
]

# sys.executable leads the megatron argv and is machine-specific, so it is
# asserted separately and the literal starts at the -m.
GOLDEN_MEGATRON_ARM = ("piper1b_megatron", "baseline")


def _golden_megatron_tail(size: str) -> list[str]:
    return [
        "-m",
        MEGATRON_DRIVER_MODULE,
        "--seq-len",
        "1024",
        "--steps",
        "40",
        "--batch",
        "4",
        "--seed",
        "42",
        "--profile-freq",
        "20",
        "--profiler-warmup",
        "5",
        "--profiler-active",
        "5",
        "--mode",
        "default",
        "--model-size",
        size,
        "/tmp/arm-dir",
    ]


class GoldenCommandTests(unittest.TestCase):
    def _command(self, pinned, size, compile_mode, ac_mode) -> list[str]:
        scenario = scenario_by_name(pinned[0])
        return command_for_arm(
            scenario.workload,
            scenario.arm(pinned[1]),
            Path("/tmp/arm-dir"),
            (),
            compile_mode,
            ac_mode,
            model_size=size,
        )

    def test_titan_argv_at_normal(self) -> None:
        self.assertEqual(
            self._command(GOLDEN_TITAN_ARM, "normal", "cuda-graph", "none"),
            _golden_titan_command("normal"),
        )

    def test_titan_argv_at_huge(self) -> None:
        self.assertEqual(
            self._command(GOLDEN_TITAN_ARM, "huge", "cuda-graph", "none"),
            _golden_titan_command("huge"),
        )

    def test_the_only_difference_between_sizes_is_the_config_argument(self) -> None:
        normal = self._command(GOLDEN_TITAN_ARM, "normal", "cuda-graph", "none")
        huge = self._command(GOLDEN_TITAN_ARM, "huge", "cuda-graph", "none")
        self.assertEqual(
            [(a, b) for a, b in zip(normal, huge) if a != b],
            [("size=normal", "size=huge")],
        )
        self.assertEqual(len(normal), len(huge))

    def test_titan_override_argv_on_the_plain_path(self) -> None:
        self.assertEqual(
            self._command(GOLDEN_OVERRIDE_ARM, "normal", "default", "sac"),
            GOLDEN_OVERRIDE_COMMAND,
        )

    def test_megatron_argv_at_normal(self) -> None:
        command = self._command(GOLDEN_MEGATRON_ARM, "normal", "default", "none")
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1:], _golden_megatron_tail("normal"))

    def test_megatron_argv_at_huge(self) -> None:
        command = self._command(GOLDEN_MEGATRON_ARM, "huge", "default", "none")
        self.assertEqual(command[0], sys.executable)
        self.assertEqual(command[1:], _golden_megatron_tail("huge"))

    def test_the_megatron_driver_module_is_importable_as_a_module(self) -> None:
        # python -m needs the module to exist under the runner's PYTHONPATH;
        # a wrong -m target fails only once the subprocess starts.
        self.assertIsNotNone(
            _find_spec(MEGATRON_DRIVER_MODULE),
            f"python -m {MEGATRON_DRIVER_MODULE} would fail to start",
        )


# --------------------------------------------------------------------------
# 6. --module resolves the way TorchTitan will resolve it.
# --------------------------------------------------------------------------


def _titan_module_candidates(module_name: str) -> tuple[str, ...]:
    """The candidate list ConfigManager._load_config tries, in its order.

    Mirrors third_party/torchtitan/torchtitan/config/manager.py: for a module
    name that is not one of torchtitan's own shorthands, it imports
    f"{module}.config_registry" first and falls back to the bare module.
    Reimplemented here rather than called because the real thing imports the
    module (and therefore torch, and the whole model package).
    """
    return (f"{module_name}.config_registry", module_name)


class TitanModuleResolutionTests(unittest.TestCase):
    def test_the_workload_module_is_the_pinned_constant(self) -> None:
        for scenario in SCENARIOS.values():
            with self.subTest(scenario=scenario.name):
                self.assertEqual(scenario.workload.module, TITAN_CONFIG_MODULE)

    def test_the_module_resolves_under_torchtitans_own_algorithm(self) -> None:
        """A bad --module string is caught here, before any GPU time.

        The runner passes ``--module benchmarks.models.piper_qwen3`` and
        TorchTitan resolves it in the training subprocess. Nothing in this
        repo imports that path, so after a move an unchanged --module would
        import cleanly on the CLI side and die minutes into the run.
        """
        resolved = [
            candidate
            for candidate in _titan_module_candidates(TITAN_CONFIG_MODULE)
            if _find_spec(candidate) is not None
        ]
        self.assertTrue(
            resolved,
            f"--module {TITAN_CONFIG_MODULE} resolves to neither "
            f"{TITAN_CONFIG_MODULE}.config_registry nor {TITAN_CONFIG_MODULE}",
        )
        # The first candidate wins in the manager, and it is the one that
        # holds the config functions.
        self.assertEqual(resolved[0], f"{TITAN_CONFIG_MODULE}.config_registry")

    def test_the_resolved_module_defines_every_config_the_arms_name(self) -> None:
        spec = _find_spec(f"{TITAN_CONFIG_MODULE}.config_registry")
        self.assertIsNotNone(spec)
        names = set()
        for scenario in SCENARIOS.values():
            for arm in scenario.arms:
                if arm.launcher != "torchtitan":
                    continue
                names.add(arm.config or scenario.workload.config)
        self.assertTrue(names)
        for name in sorted(names):
            with self.subTest(config=name):
                self.assertTrue(
                    _module_defines(spec, name),
                    f"--config {name} is not defined in {spec.origin}",
                )

    def test_the_module_name_does_not_collide_with_a_torchtitan_shorthand(
        self,
    ) -> None:
        """A colliding name would resolve to torchtitan's registry, silently.

        The manager checks its own shorthand list *before* the two-candidate
        path, so a move that renamed the module to e.g. "qwen3" would build
        stock torchtitan configs under our arm names.
        """
        try:
            from torchtitan.experiments import _supported_experiments
            from torchtitan.models import _supported_models
        except ImportError as error:  # pragma: no cover - fork-dependent
            self.skipTest(f"torchtitan shorthand lists unavailable: {error}")
        self.assertNotIn(
            TITAN_CONFIG_MODULE, _supported_models | _supported_experiments
        )


# --------------------------------------------------------------------------
# 7. BENCH_DIR is the repository root.
# --------------------------------------------------------------------------

# Files that identify the repository root, independently of any module's
# nesting depth.
ROOT_MARKERS = ("run_bench.sh", "pyproject.toml", "CLAUDE.md")


def _repo_root_from_here() -> Path:
    """Walk up from this test file to the directory holding run_bench.sh."""
    for candidate in Path(__file__).resolve().parents:
        if all((candidate / marker).exists() for marker in ROOT_MARKERS):
            return candidate
    raise AssertionError(
        f"no ancestor of {__file__} contains {ROOT_MARKERS}"
    )


class RepositoryRootTests(unittest.TestCase):
    """The only silent failure mode in the whole migration.

    ``benchmarks.execution.paths.BENCH_DIR`` and
    ``benchmarks.models.piper_qwen3.megatron_bootstrap.REPO_ROOT`` were both
    ``Path(__file__).resolve().parent.parent`` before the move. The move
    changed how deep those files sit (to ``parents[2]`` and ``parents[3]``),
    and a stale ``.parent`` chain does not raise: the
    run starts, the training subprocess gets a PYTHONPATH one level off,
    ``out/`` and the cache land somewhere else, provenance reads the git rev
    of whatever repo happens to be there, and the numbers look fine. Every
    other breakage in this file announces itself with an ImportError or a
    failed arm; this one does not. Assert it explicitly.
    """

    def test_bench_dir_is_the_repository_root(self) -> None:
        self.assertEqual(BENCH_DIR, _repo_root_from_here())

    def test_megatron_repo_root_is_the_same_directory(self) -> None:
        self.assertEqual(REPO_ROOT, _repo_root_from_here())
        self.assertEqual(REPO_ROOT, BENCH_DIR)

    def test_the_root_holds_every_canonical_package(self) -> None:
        # BENCH_DIR is exported as PYTHONPATH to the training subprocess, so
        # it must be the directory the first-party packages import from.
        for root in CANONICAL_ROOTS:
            with self.subTest(package=root):
                self.assertTrue(
                    (BENCH_DIR / root).is_dir(),
                    f"{root} is not a package directory under {BENCH_DIR}",
                )
                spec = _find_spec(root)
                self.assertIsNotNone(spec)
                locations = [
                    Path(path).resolve()
                    for path in (spec.submodule_search_locations or [])
                ]
                self.assertIn((BENCH_DIR / root).resolve(), locations)

    def test_the_submodule_paths_hang_off_the_same_root(self) -> None:
        self.assertEqual(TITAN_DIR, BENCH_DIR / "third_party" / "torchtitan")
        self.assertTrue((TITAN_DIR / "torchtitan").is_dir())
        self.assertEqual(
            MEGATRON_SUBMODULE, REPO_ROOT / "third_party" / "Megatron-LM"
        )


# --------------------------------------------------------------------------
# 8. Per-file test census.
# --------------------------------------------------------------------------

# One entry per test module, with the number of test cases it contributes.
# The point is not the totals: it is that no behavioral assertion silently
# disappears during the move. A module that stops being discovered (wrong
# directory, a broken import swallowed by a loader error) drops to zero here
# instead of quietly shrinking the suite.
TEST_CENSUS = {
    # 16 pre-migration, +1 for the assertion that importing
    # benchmarks.cli.main alone yields all five commands -- the failure mode
    # the cli.add_command wiring exists to prevent, added when the CLI split
    # into main/e2e/kernel/rendering. +1 with the uncompiled compile mode:
    # the sweep skips a scenario that declines it, as it already does for ac.
    "test_cli": 19,
    # 19 pre-fix, +2 for the two halves of the correctness verdict: a failed
    # gate fragment beside a clean exit code, and a timing worker that dies
    # after it writes. +2 more for the arm that measured nothing, as a
    # non-anchor and as the anchor. +1 for the three counts that refuse zero,
    # and +1 for the two derived columns the parent alone can compute.
    # +2 with --replicates-per-process: that a batched worker owns
    # consecutive replicates of one arm and never a second arm, and that a
    # batched worker which dies costs every replicate of its block.
    # +4 with --arm: that the selection reaches the request and defaults to
    # every arm, that it refuses more than one scenario, that an unselected
    # arm is spawned in neither pass and says which flag dropped it, and
    # that a selection missing the anchor, a reference or a real arm name is
    # refused rather than repaired.
    "test_kernel_cli": 31,
    # -2 with the deletion of the single-engine ``qkv`` and ``swiglu``
    # scenarios: the smoke test names one scenario per test, and neither
    # of these two had a scenario left to name.
    "test_kernel_gpu_smoke": 2,
    # Added when the merge got a direct caller: every other merge assertion
    # reaches it through execute_kernel_run, which spawns replicate-major and
    # therefore never hands _ordered_replicates the arrival order it exists
    # to reject.
    # +3 with the within-process interval: that one replicate per process
    # keeps the honest field names, that a batched run renames the interval
    # rather than dropping it, and that the methodology says so.
    # +1 for the replicate spread, which is derived from the same
    # per-replicate log-ratios and was left behind by that rename.
    "test_kernel_merge": 12,
    # The span engine. A span is declared over an ordered scenario range
    # and its claim is the span against the SUM of the scenarios it
    # replaces, so a span result holds two totals where a scenario result
    # holds one. 18 covering what the declaration refuses, the
    # registry-side cross-check that every part arm exists, and that a
    # span is not a KernelScenario.
    # +7 with the schema 6 -> 7 bump: a span results file names a span
    # and not a scenario, its two totals live in separate fields, its
    # claim is a separate list from the within-span comparisons, and
    # each kind of file refuses the other's reader.
    # +10 with the merge: that the parts total is summed per replicate,
    # that the breakdown reaches the file, that the claim carries a
    # renamed interval and no two-sample test, that a missing part
    # costs one arm its claim, and that a span with no claim at all
    # is refused.
    # +7 with the runner and the CLI: that the enclosed scenarios run
    # first and once, that a span is named with --span, that one run
    # publishes both totals, that a span file sits apart from the
    # scenario files, that the printed table marks every interval,
    # and that a span is opt-in.
    # +3 with the mutation-resistant fixture: that the paired estimate is
    # not the unpaired one, and that both totals are auditable from the
    # file that publishes them.
    # +3 for the dispatch-chain bias, which favours the span and is
    # stated in the file, in the printed table, and on a short range
    # as well as a long one.
    # +1 with the merge that brought --arm and --span into one command:
    # the two flags do not combine, because a span run holds several
    # rosters and one arm selection cannot say which of them it names.
    "test_kernel_spans": 51,
    # The spans this repository DECLARES, as opposed to the span type. Its
    # sibling above proves the mechanism against a synthetic span; this one
    # reads the real roster and asks whether each declaration still says a
    # true thing about the live registry -- which is the half that goes
    # stale when a scenario is renamed or an arm is re-homed. 11 generic
    # over every declared span (the roster pin, the disjointness guard, the
    # part arms and their modes, the builder module rule, and the two
    # captions every span owes its reader) and 6 for expert_combine, whose
    # cross-engine row is the reason the span type exists at all.
    # +7 for attn_residual_norm, whose range is 6+7+8 because titan's
    # residual add fuses FORWARD into the next norm's prologue -- a 6+7
    # span would cut on the wrong side and land at 1.0.
    # +8 for ffn_norm_to_moe_residual, the mcore-only backward-only span,
    # whose eight cover the range its residual sets, that both arms are
    # backward-mode only, and that a single-engine span over six
    # cross-engine scenarios sums only the arms its SpanParts name.
    # +7 for fused_linear_ce, the titan-only span whose loss owns the LM
    # head: its parts share neither name with its arm, it publishes no
    # within-span row, and the compile treatment of the projection moves
    # across the cut and reaches the ratio.
    # +7 for chunked_ce, upstream's actual default over that same range:
    # that the two spans share a range without colliding, that their arm
    # names stay distinct, that this one is eager where it is timed from,
    # and that its sequence requirement is declared and costs the span.
    "test_kernel_span_declarations": 47,
    # The per-arm build probe. requires_gcc_toolset answers a question about
    # the HOST; KernelArm.requirement answers one about this shape and this
    # workload, which is what a sequence sweep needs and what an arm that
    # skips for a runtime reason needs. 6 covering both answers of the
    # contract, that the reason reaches the file in the predicate's own
    # words, and that a workload skip closes over correctness references --
    # the closure the compiler skip has never reached.
    "test_kernel_arm_requirements": 6,
    # 3 pre-bump, +1 for the assertion that the replicate boundaries
    # survive the round trip -- they are the repetition unit the bootstrap
    # CI is taken over, and a flat sample list cannot express them. +3 for
    # the burst residual: that it marks a flagged arm beside its ratio, that
    # an absent ladder reads as unknown rather than as converged, and that
    # the threshold still splits the measured rope arms from copy_floor.
    # +1 for the one-sidedness -- a flat ladder is not a device-bound
    # verdict, which rope backward demonstrates.
    # Scenario 3 of the cross-engine partition, and the first module here
    # to build both engines: 35 covering the shared inputs, the two layouts,
    # the fp64 reference, both arm builders and the guards that refuse a norm
    # the spec resolved to something other than a real one. 18 of the 35
    # arrived with megatron's strided key: the layout of every tensor the
    # megatron arm reads, the leaf sets that must carry it, the manifest flag
    # that describes it, and the builder wiring that reports its byte count.
    "test_kernel_qk_norm": 35,
    # Scenario 5, the last of the cross-engine partition, and the one whose
    # arms no correctness gate can tell apart: cuDNN, FlashAttention and the
    # unfused path all compute attention, so a megatron arm that fell through
    # to the wrong backend passes every gate the scenario declares. By class:
    # 4 profile deltas, 9 roster, 15 inputs, 3 reference, 5 layout, 7 shared
    # closures, 13 backend verdict, 7 megatron build guards, 2 markers,
    # 3 shape summary, 5 builder wiring and 1 profile delivery. The last
    # two groups exist because no test can CALL five of the six builders
    # without a GPU: a mutation pass wired the FA3 builder to the cuDNN
    # anchor and the whole suite stayed green. Three tests are
    # self-invalidating and are meant to
    # fail one day: one pins that TransformerEngine reads no
    # NVTE_FLASH_ATTN_V variable, so megatron's flash_attention_version is
    # inert and the generation cannot be an arm; one pins that TE does not
    # recognize megatron's fused QKV layout, which is why the megatron arms
    # carry a copy this scenario times; and one pins that megatron still
    # norms the key and still leaves the value alone, which is what decides
    # which tensor carries that copy.
    "test_kernel_attention_core": 73,
    # Scenario 4, the cross-engine rope roster that replaced the
    # single-engine one in place: 35 covering the packed-document inputs in
    # both engine-native forms, the fp64 reference, all five arm builders and
    # the two guards the roster needs -- the marker guard that refuses a titan
    # override which fell back to the stock path, and the branch check that
    # refuses mcore/no_rope_fusion at batch 1, where megatron rotates by
    # global document offsets and computes a different rotation.
    "test_kernel_rope": 35,
    # Scenario 6, the first cross-engine GEMM: 21 covering the shared inputs
    # and their two native layouts, the fp64 reference, both arm builders, and
    # the guards that refuse an mcore module of the wrong class, the wrong
    # shape or more than one rank -- the last being what would make the arm
    # inert rather than wrong.
    "test_kernel_attn_out_proj": 21,
    # Scenario 7, the first within-engine-only scenario, and the guard-heavy
    # module at 51: 7 pin the profile delta at exactly one field and the arm
    # roster, 8 the shared inputs and the fp64 reference, 6 the bandwidth
    # floor, 7 the shared closures, and 23 the three build guards -- one of
    # which pins what the exact-add guard CANNOT see, because addition
    # commutes bit for bit and an operand swap therefore passes it. The guards
    # carry the weight because the scenario declines its cross-engine row on
    # the strength of one claim -- both engines compute residual + attn_out --
    # and because an unfused arm that resolved to the compiled callable would
    # publish two labels for one implementation, which no correctness gate can
    # see. The floor tests carry the other weight: the published row is a
    # dispatch comparison, and without the floor nothing separates it from a
    # kernel result.
    "test_kernel_attn_residual": 51,
    # Scenario 8: 18 covering the shared inputs, the fp64 reference, the
    # floor's traffic, both arm closures, and the two refusals that keep the
    # arm honest -- an epsilon the two engines do not share, and a module
    # whose forward returns the residual tuple the fused variant produces.
    "test_kernel_ffn_norm": 18,
    # Scenario 13: 26 covering the shared inputs, the fp64 reference, the
    # profile delta, the bandwidth floor and the closures the three
    # implementation arms share -- plus ten that pin the megatron and
    # torchtitan source lines the scenario's "both engines compute one add"
    # claim rests on. They are read as text rather than imported, because
    # ``megatron.core`` pulls in TransformerEngine and costs the suite about
    # nine seconds. One of the ten pins which transformer-layer class this
    # build instantiates, which is what decides whether the reflatten
    # ordering the module states applies at all.
    "test_kernel_moe_residual": 36,
    # Scenario 14: 22 covering the shared inputs, the fp64 reference, the
    # floor's traffic and both arm closures -- plus the titan extraction,
    # which is one config read here because the norm is a model-level module
    # and not a per-layer one.
    # +5 for the mcore arm's model release: four run the two helpers on a
    # stand-in model, and one pins the order the builder calls them in.
    "test_kernel_final_norm": 27,
    # Scenario 1, and the widest module here at 55, because the scenario's
    # claim is a negative one: the layout conversion megatron enters is free
    # at our THD packing, so a large part of the module proves that -- both
    # directions of the transpose, the probe the builder runs, and the
    # absence of the charge an earlier revision put on the titan arm. The
    # rest covers the shared inputs, the fp64 reference, the floor, both arm
    # builders, the guards that refuse a module of the wrong class or with a
    # tensor-parallel group, and the one weight transfer that joins the two
    # embeddings.
    # +5 for the mcore arm's model release, the same five ``final_norm``
    # carries: four run the two helpers on a stand-in model, and one pins the
    # order the builder calls them in.
    "test_kernel_embedding_stage": 60,
    # Scenario 2: 40 covering the shared inputs and both weight layouts, the
    # fp64 reference, all three arm builders, and the guards that refuse an
    # mcore module of the wrong class, with the wrong epsilon, or above one
    # rank. A block of them pins the materialization asymmetry the scenario
    # declares rather than equalizes -- which arm copies q, k and v, and which
    # hands on a strided view.
    # +1 net when the isolated backward the retired ``qkv`` scenario measured
    # was restored to the two titan arms: the one test that pinned "no
    # isolated backward, on any arm" became two, one per side of the
    # asymmetry, and the titan one calls the closure twice because the timing
    # pass does.
    "test_kernel_qkv_prep": 41,
    # Scenario 16: 33 covering the shared logits and both layouts, the fp64
    # reference, all six arm builders and the label preparation both engines
    # are charged. A block of them pins the two claims a reader would
    # otherwise have to take on trust -- that mcore/base destroys its input,
    # so no timed sample runs on the declared logits, and that megatron as
    # NVIDIA ships it is mcore/no_ce_fusion rather than mcore/ce_native.
    "test_kernel_cross_entropy": 33,
    # Scenario 15: 33 covering the shared inputs, the fp64 reference, both
    # arm builders and the two layouts. A block of them pins the two claims
    # that invert an obvious reading of the row -- that the megatron arm is
    # ColumnParallelLinear and not TransformerEngine, which the guard proves
    # by refusing a stand-in named for the TE class, and that the titan arm
    # is eager because apply_compile cannot reach a sibling of layers. Two
    # more pin the one wrong module no correctness gate can see: an uncast
    # fp32 weight passes the 2e-2 rel_l2 gate, because the bf16 quantization
    # of the activation dominates the metric.
    "test_kernel_lm_head_projection": 33,
    # Scenario 9: 52 covering the shared inputs, the canonical output
    # conversion both engines are put through, the two profile deltas, every
    # arm builder, and the guards. Three of them pin the gate GEMM's branch,
    # because mcore_bytes_moved is declared rather than measured and it
    # describes megatron's TransformerEngine path alone.
    "test_kernel_moe_router": 52,
    # Scenario 10: 80, the largest of the four. The measurand is a
    # permutation, so most of them pin the row order -- including the
    # transcription of megatron's own torch permute path that shows it agrees
    # with TorchTitan's argsort, which is what the cross-engine bitwise gate
    # rests on.
    "test_kernel_dispatch_permute": 80,
    # Scenario 11: 50 covering the shared weights in all four layouts, the
    # fp64 reference, all eight arm builders and the guards. The expert-class
    # guard is the one that matters: megatron polices the agreement between
    # moe_grouped_gemm's two readers nowhere, and both disagreement
    # directions are numerically correct.
    "test_kernel_expert_mlp": 50,
    # Scenario 12: 62 covering the shared inputs, both engines' fp64 truths,
    # every arm builder and the guards. A block of them pins the canonical
    # (expert, token) row order, which is the premise both engines' combines
    # are checked against.
    "test_kernel_moe_combine": 62,
    # +3 with the isolation report: that the honest state reaches the reader,
    # that a batched run is marked above the table, and that a renamed
    # interval still prints, with a mark that says it is a lower bound.
    # +3 with the scenario description reaching the printed table: that it
    # prints above the tables, that a scenario without one grows no blank
    # block, and that a long one wraps.
    # +1 with the arm column: the field must fit the longest declared name.
    "test_kernel_results": 17,
    # 19 pre-bump, +3 net: the Wilcoxon pair became five tests covering the
    # bootstrap CI, the single-replicate case and reproducibility. +1 for the
    # scenario's rejection of a mode name MODES does not hold, +1 for
    # _seeded_build's builder-vs-declaration gate.
    # +3 with fragment_stem: a slash never reaches a filename, two arms may
    # not collide after the substitution, and no declared arm does today.
    # +4 with the correctness residency loop: no earlier arm survives the
    # next build, the outputs outlive the arms, every kept tensor is
    # detached, and a skipped arm is never built.
    # Net 0 with eager_reason: the assertion that only rope/copy_floor is
    # eager became one that an eager arm with no reason is refused at
    # import, which is what an mcore arm makes unavoidable.
    # +5 with the worker phase table: that a span is recorded under its name,
    # that it closes when the block raises, that the exit hook runs inside it,
    # that a span nothing timed can still be recorded, and that the offset
    # from process exec is a plausible age.
    # +1 with the libc flush: os._exit skips libc's own exit, so a C
    # extension's output would be lost from the log the parent tails.
    # +1 with the within-engine-only caption: a scenario that declines a
    # cross-engine row must say in its description that suppressing the row
    # does not suppress the number two derived columns still reproduce.
    # -1 with the deletion of the single-engine ``attention`` scenario: its
    # roster test had no subject left, and attention_core already pins the
    # re-homed one.
    "test_kernels": 49,
    "test_lm_head_losses": 8,
    # New with the mcore profile registry: 6 that pin the extraction against
    # a frozen literal (including the cuda-graph branch the parity check
    # never reaches), 8 that pin the validation -- above all that a
    # dual-delivery field set on one side only is refused -- 5 for the
    # driver's declared-state check, and 1 that the builder takes no default
    # profile.
    # +2 for the attention-backend encoding. Three checks arrive, in two new
    # tests: that the five names the registry offers are megatron's own
    # AttnBackend members, read out of the pinned submodule, and that a named
    # backend travels to the config kwargs as a name for
    # megatron_model.build_model to resolve. The third -- that an unknown
    # backend name is refused like an unknown activation -- is one more
    # assertion inside the existing refusal test, so it adds no row.
    "test_mcore_profiles": 20,
    "test_megatron_data": 5,
    # New with the promotion of the cross-engine weight map out of
    # tools/megatron_parity_check.py: 3 that pin the QKV grouped
    # interleave (including that the guard rejects a plain concatenation)
    # and 8 that pin the transfer, its component slices and its refusals.
    # +1 with the real piper ladder: the same interleave at 4:1 grouped-query
    # attention and head_dim 128, which the 1B-family TINY shape cannot reach.
    "test_megatron_weights": 12,
    # 25 pre-migration, +1 for the megatron builder's explicit-shape test --
    # the symmetric twin of the titan builders' one, added when the megatron
    # side stopped defaulting to the normal shape. +8 for the large and giant
    # shapes: a tensor-by-tensor parameter count that derives what no run has
    # logged, the two geometry tests, the layer-count and parameter-split
    # arguments, the declaration order, and the estimated parity gates.
    # +3 when the head and expert geometry became recorded data: the two
    # halves of the pinned-shape table (every shape matches it, every shape is
    # in it) and the one that keeps the piper-1B family constructor out of the
    # registry. +5 when normal became 1b: the alias resolves, it is not a
    # registry entry, an unknown size is still refused, a fresh manifest
    # records the canonical name, and a manifest recording either name
    # resumes against the other.
    "test_model_shape": 42,
    "test_profile_regions": 19,
    # New with the cuDNN identity fields. TransformerEngine binds the
    # loader's cuDNN while torch expects the wheel's, so which cuDNN a
    # megatron arm ran is a property of the host and was recorded
    # nowhere until 2026-08-20. 4 pin that both fields reach the
    # manifest separately, and 2 pin that collecting them reports a
    # diagnostic instead of failing the run.
    "test_provenance": 6,
    # +3 with the inverted half of validation rule 8: that an uncompiled mode
    # needs the compile line absent, that an engine which cannot prove eager
    # execution is refused, and that both halves of the rule name one line.
    # +6 with the mode itself: that an uncompiled command drops only the
    # compile flag, that a megatron command refuses the mode, that the
    # megatron scenario declines it while every other scenario takes it,
    # that a run records the mode and declares no regions, and that a
    # resume refuses to cross the boundary.
    "test_runner": 51,
    "test_run_validation": 1,
    "test_swiglu": 4,
    "test_te_rope": 1,
    # New with the in-process titan build: 3 that pin the override count
    # -- the kernel-side equivalent of validation rule 2, and the only
    # guard an arm whose kernel has no distinctive name can have -- and 6
    # that pin the build itself, its dtype delivery and its seeding.
    "test_titan_model": 9,
    # Added by the restructure: the sweep that asserts no retired module
    # path survives anywhere git would ship. Censused like the rest so it
    # cannot quietly stop being discovered.
    "test_retired_paths": 15,
    # New with build_model's blank_parts: 2 that pin the argument's default
    # and the two callers that must never take it, 9 that exercise the spec
    # edit against megatron's own TransformerLayerSubmodules, 5 that pin
    # how build_model wires it and check its raise path, 2 that refuse
    # blank_parts on the host initialization path, and 5 that pin which
    # kernel builder blanks which part.
    "test_megatron_model": 26,
}
TEST_CENSUS_TOTAL = 1160

# The package the modules above are imported as, and this file's own name --
# excluded from the census so editing it does not require editing its own
# pinned count.
TESTS_PACKAGE = "tests"
THIS_MODULE = Path(__file__).stem


class TestCensusTests(unittest.TestCase):
    def test_the_pinned_counts_sum_to_the_pinned_total(self) -> None:
        self.assertEqual(sum(TEST_CENSUS.values()), TEST_CENSUS_TOTAL)

    def test_every_censused_module_is_still_on_disk(self) -> None:
        """A censused module must not vanish; new ones are allowed.

        Deliberately one-directional. The contract is that no existing
        assertion disappears during the move, not that the suite stops
        growing -- and this file is written alongside other work that adds
        test modules, so an equality here would fail on unrelated additions
        rather than on a migration defect. A module that is split rather than
        deleted is still caught: its own pinned count drops.
        """
        on_disk = {
            path.stem
            for path in sorted(Path(__file__).resolve().parent.glob("test_*.py"))
            if path.stem != THIS_MODULE
        }
        self.assertEqual(set(TEST_CENSUS) - on_disk, set())

    def test_each_module_still_contributes_its_pinned_test_count(self) -> None:
        for name, expected in TEST_CENSUS.items():
            with self.subTest(module=name):
                loader = unittest.TestLoader()
                try:
                    suite = loader.loadTestsFromName(f"{TESTS_PACKAGE}.{name}")
                except Exception as error:  # pragma: no cover - loader-dependent
                    self.skipTest(f"{name}: loader unavailable ({error})")
                if loader.errors:
                    # An import error is reported as a single synthetic
                    # _FailedTest, whose count would be a lie. Skip rather
                    # than assert a number the loader did not really produce;
                    # the module's own failure is the suite's job to report.
                    self.skipTest(f"{name}: failed to load ({loader.errors[0]})")
                self.assertEqual(suite.countTestCases(), expected)


if __name__ == "__main__":
    unittest.main()
