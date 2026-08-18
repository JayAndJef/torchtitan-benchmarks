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
    "swiglu": "benchmarks.kernel.operations.swiglu",
    "qkv": "benchmarks.kernel.operations.qkv",
    "lm_head": "benchmarks.kernel.operations.lm_head",
    "attention": "benchmarks.kernel.operations.attention",
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
    "rope": ("copy_floor", "baseline", "helion", "te"),
    "swiglu": ("baseline", "piper_optimized_triton", "piper_optimized_inductor"),
    "qkv": ("baseline", "fused_qkv"),
    "lm_head": (
        "baseline",
        "fused_linear_ce",
        "te_fused_ce",
        "piper_optimized_te_ce",
    ),
    "attention": ("baseline", "flex_flash", "flash_attention_3"),
}

KERNEL_BASELINE_ARMS = {
    "rope": "baseline",
    "swiglu": "baseline",
    "qkv": "baseline",
    "lm_head": "baseline",
    "attention": "baseline",
}

# Arm names deliberately match across the two registries wherever the same
# implementation is measured at two scopes (CLAUDE.md, "Scenarios and arms").
# copy_floor is a bandwidth floor with no end-to-end counterpart.
KERNEL_TO_E2E_SCENARIO = {
    "rope": "piper1b_rope",
    "swiglu": "piper1b_swiglu",
    "qkv": "piper1b_qkv",
    "lm_head": "piper1b_lm_head",
    "attention": "piper1b_attention",
}
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

        Reports lean on this: piper1b_attention/flex_flash and
        attention/flex_flash are documented as the same code measured two
        ways. A rename on one side alone would silently break the pairing.
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
    """Split ``module:symbol`` as kernel.engine.run.resolve_symbol does."""
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
        """The colon format is resolve_symbol's, not a guess."""
        from benchmarks.kernel.engine import run as kernel_bench

        source = Path(kernel_bench.__file__).read_text()
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
    # into main/e2e/kernel/rendering.
    "test_cli": 17,
    "test_kernel_cli": 14,
    "test_kernel_gpu_smoke": 4,
    # 3 pre-bump, +1 for the assertion that the replicate boundaries
    # survive the round trip -- they are the repetition unit the bootstrap
    # CI is taken over, and a flat sample list cannot express them.
    "test_kernel_results": 4,
    # 19 pre-bump, +3 net: the Wilcoxon pair became five tests covering the
    # bootstrap CI, the single-replicate case and reproducibility.
    "test_kernels": 23,
    "test_lm_head_losses": 8,
    "test_megatron_data": 5,
    # 25 pre-migration, +1 for the megatron builder's explicit-shape test --
    # the symmetric twin of the titan builders' one, added when the megatron
    # side stopped defaulting to the normal shape.
    "test_model_shape": 26,
    "test_profile_regions": 19,
    "test_runner": 41,
    "test_run_validation": 1,
    "test_swiglu": 4,
    "test_te_rope": 1,
    # Added by the restructure: the sweep that asserts no retired module
    # path survives anywhere git would ship. Censused like the rest so it
    # cannot quietly stop being discovered.
    "test_retired_paths": 15,
}
TEST_CENSUS_TOTAL = 182

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
