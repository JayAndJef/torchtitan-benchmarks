"""CPU-only tests for the kernel-benchmark registry and statistics."""

import contextlib
import importlib
import io
import sys
import unittest
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmarks.kernel.engine.statistics import kernel_comparison
from benchmarks.kernel.registry import KERNEL_SCENARIOS, kernel_scenario_by_name
from benchmarks.kernel.schema import (
    KernelWorkload,
    MODES,
    fragment_stem,
    resolve_shape_and_workload,
    routing_divides_evenly,
    shape_summary,
)


class RegistryTests(unittest.TestCase):
    def test_every_scenario_is_internally_consistent(self) -> None:
        """The roster itself is pinned in ``tests/test_migration_contract.py``,
        which owns the stable-id inventory; a literal list here would be the
        same pin written twice and edited sixteen times. What this checks is
        the shape of each declaration: the anchor is one of the arms, every
        mode is a real mode, and every correctness reference resolves."""
        for scenario in KERNEL_SCENARIOS.values():
            self.assertIn(
                scenario.baseline_arm, [arm.name for arm in scenario.arms]
            )
            for arm in scenario.arms:
                self.assertTrue(set(arm.modes) <= set(MODES), arm.name)
                for check in arm.correctness:
                    self.assertIn(
                        check.kind, ("bitwise", "tolerance", "fp64_ulp")
                    )
                    if check.reference == "fp64":
                        self.assertIsNotNone(scenario.reference_builder)
                    else:
                        scenario.arm(check.reference)

    def test_every_comparison_pair_shares_a_mode(self) -> None:
        """A declared row that shares no mode would write nothing at all."""
        for scenario in KERNEL_SCENARIOS.values():
            pairs = scenario.comparison_pairs()
            self.assertTrue(pairs, scenario.name)
            for arm_name, opponent in pairs:
                arm = scenario.arm(arm_name)
                self.assertFalse(arm.is_floor, arm_name)
                self.assertNotEqual(arm_name, opponent)
                self.assertTrue(
                    set(arm.modes) & set(scenario.arm(opponent).modes),
                    f"{scenario.name}: {arm_name} vs {opponent}",
                )

    def test_a_scenario_derives_its_comparisons_or_declares_them(self) -> None:
        """This replaces an assertion that *every* scenario left
        ``comparisons`` at ``None``. That was true of the five single-engine
        scenarios and stopped being true at ``attn_out_proj``, which declares
        its one row so that the direction of a cross-engine ratio is stated
        rather than inherited. The universal was never the property worth
        keeping; the two branches of ``comparison_pairs`` are.

        So each branch is checked against a scenario that takes it. ``rope``
        derives: every arm faces the anchor and the anchor is not compared
        with itself. ``qk_norm`` derives too and additionally shows that a
        floor is left out, which ``rope`` cannot show since it retired its
        own. ``attn_out_proj`` declares: the published rows are its tuple,
        verbatim and in order."""
        rope = kernel_scenario_by_name("rope")
        self.assertIsNone(rope.comparisons)
        self.assertEqual(
            rope.comparison_pairs(),
            (
                ("mcore/no_rope_fusion", "mcore/base"),
                ("titan", "mcore/base"),
                ("titan/helion", "mcore/base"),
                ("titan/te", "mcore/base"),
            ),
        )

        floored = kernel_scenario_by_name("qk_norm")
        self.assertIsNone(floored.comparisons)
        self.assertIn("copy_floor", [arm.name for arm in floored.arms])
        self.assertEqual(
            floored.comparison_pairs(), (("titan", "mcore/base"),)
        )

        cross_engine = kernel_scenario_by_name("attn_out_proj")
        self.assertEqual(
            cross_engine.comparisons, (("titan", "mcore/base"),)
        )
        self.assertEqual(
            cross_engine.comparison_pairs(), cross_engine.comparisons
        )

        # The explicit form is exhaustive, not additive: a declared tuple is
        # the whole published set, so the derivation must not run beside it.
        # Written against rope, where the derived set has two rows and this
        # one has none of them.
        self.assertEqual(
            replace(
                rope, comparisons=(("titan/te", "titan/helion"),)
            ).comparison_pairs(),
            (("titan/te", "titan/helion"),),
        )

    def test_a_scenario_may_declare_that_it_publishes_no_ratio(self) -> None:
        scenario = kernel_scenario_by_name("qkv")
        self.assertEqual(replace(scenario, comparisons=()).comparison_pairs(), ())
        with self.assertRaisesRegex(ValueError, "Unknown arm"):
            replace(scenario, comparisons=(("fused_qkv", "nope"),))

    def test_a_declared_mode_must_be_one_the_timing_pass_runs(self) -> None:
        """The declaration is authoritative, so it has to be legal itself.

        ``_seeded_build`` asks only that the builder and the declaration
        agree. An arm declaring ``fwd`` with a builder supplying a ``fwd``
        closure passes that gate, and then ``run_timing_pass`` -- which
        iterates ``MODES`` -- times it in no mode at all. A non-floor arm
        raises later in ``_heaviest_mode``; a floor skips the memory pass and
        reaches the merge silently, with nothing in it.
        """
        scenario = kernel_scenario_by_name("qkv")
        with self.assertRaisesRegex(ValueError, "unknown mode"):
            replace(
                scenario,
                arms=tuple(
                    replace(arm, modes=("fwd",)) for arm in scenario.arms
                ),
            )
        with self.assertRaisesRegex(ValueError, "declares no modes"):
            replace(
                scenario,
                arms=tuple(replace(arm, modes=()) for arm in scenario.arms),
            )

    def test_attention_arms(self) -> None:
        scenario = KERNEL_SCENARIOS["attention"]
        self.assertEqual(
            [arm.name for arm in scenario.arms],
            ["baseline", "flex_flash", "flash_attention_3"],
        )
        # No isolated backward: see _attention_arm's docstring.
        for arm in scenario.arms:
            self.assertEqual(arm.modes, ("forward", "forward_backward"))
            self.assertTrue(arm.compiled, arm.name)

    def test_builder_paths_resolve_without_importing_torch(self) -> None:
        # Both declaration modules must stay torch-free; the dotted paths just
        # need to be well-formed module:function references, each one naming
        # its *own* family's arm module. A builder left in another family's
        # module would import and measure correctly, so nothing but this
        # notices. tests/test_migration_contract.py pins the five literal
        # module names; here the point is that they track the scenario.
        for name in ("benchmarks.kernel.registry", "benchmarks.kernel.schema"):
            self.assertNotIn("torch", vars(importlib.import_module(name)))
        for scenario in KERNEL_SCENARIOS.values():
            expected = f"benchmarks.kernel.operations.{scenario.name}"
            references = [scenario.inputs_builder] + [
                arm.builder for arm in scenario.arms
            ]
            if scenario.reference_builder:
                references.append(scenario.reference_builder)
            for reference in references:
                module, _, function = reference.partition(":")
                self.assertEqual(module, expected)
                self.assertTrue(function.isidentifier(), reference)

    def test_an_unexplained_eager_arm_is_refused_at_import(self) -> None:
        """This replaces an assertion that the eager set was exactly
        ``{("rope", "copy_floor")}``. That set was true while every arm was
        TorchTitan's, and it is false the moment a megatron-core arm exists:
        megatron compiles no whole layer, so an mcore arm is eager by design
        and there will be one in every cross-engine scenario. Widening the
        literal to name them would encode the roster in a test and say
        nothing about why any of them is eager.

        The property worth keeping is that no arm is eager by accident, and
        the place to keep it is the declaration -- so a new arm cannot reach
        a GPU without a reason, rather than reaching one and failing a test
        afterwards. A companion test that walked the registry asserting the
        same thing would never fail: an offending arm cannot be imported."""
        rope = kernel_scenario_by_name("rope")
        eager, compiled = rope.arm("mcore/base"), rope.arm("titan")
        # The anchor is the eager arm here, and it stays in both tuples:
        # __post_init__ resolves the anchor first, so dropping it would raise
        # "Unknown arm" and the assertions below would pass for the wrong
        # reason. It is a megatron arm rather than the retired floor, which
        # is the case this docstring says now matters.
        with self.assertRaisesRegex(ValueError, "declares no eager_reason"):
            replace(
                rope, arms=(replace(eager, eager_reason=None), compiled)
            )
        with self.assertRaisesRegex(ValueError, "is compiled but declares"):
            replace(
                rope, arms=(eager, replace(compiled, eager_reason="because"))
            )

    def test_every_arm_has_a_description(self) -> None:
        from benchmarks.e2e.registry import SCENARIOS

        for registry in (KERNEL_SCENARIOS, SCENARIOS):
            for scenario in registry.values():
                for arm in scenario.arms:
                    self.assertTrue(
                        arm.description.strip(),
                        f"{scenario.name}/{arm.name} needs a description",
                    )

    def test_only_te_requires_gcc_toolset(self) -> None:
        self.assertTrue(kernel_scenario_by_name("rope").requires_gcc_toolset)
        for name in ("swiglu", "qkv", "lm_head"):
            self.assertFalse(
                kernel_scenario_by_name(name).requires_gcc_toolset
            )

    def test_unknown_names_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown kernel scenario"):
            kernel_scenario_by_name("nope")
        with self.assertRaisesRegex(ValueError, "Unknown arm"):
            kernel_scenario_by_name("rope").arm("nope")

    def test_the_balanced_routing_flag_is_set_exactly_where_it_is_needed(
        self,
    ) -> None:
        """Every scenario that hands each expert an equal slice, and no other.

        The flag is what makes an uneven expert split skip the scenario
        LOUDLY -- named numbers, a recorded error, a nonzero exit
        (``benchmarks/kernel/runner.py``) -- instead of routing rows the arms
        never built. A scenario that needs it and does not set it crashes
        inside its inputs builder; one that sets it and does not need it
        skips workloads it could have measured. The literal is exhaustive on
        purpose, so both directions are a failure here.
        """
        balanced = {
            scenario.name
            for scenario in KERNEL_SCENARIOS.values()
            if scenario.requires_balanced_routing
        }
        self.assertEqual(
            balanced,
            {"swiglu", "dispatch_permute", "expert_mlp", "moe_combine"},
        )


class ShapeAndWorkloadTests(unittest.TestCase):
    def test_shape_arithmetic(self) -> None:
        shape, workload = resolve_shape_and_workload()
        self.assertEqual((workload.batch, workload.seq_len), (4, 1024))
        swiglu = shape_summary("swiglu", shape, workload)
        self.assertEqual(swiglu["x"], [8192, 1024])
        self.assertEqual(swiglu["tokens_per_expert"], [2048] * 4)
        qkv = shape_summary("qkv", shape, workload)
        self.assertEqual(qkv["wqkv"], [2048, 1024])
        self.assertEqual(qkv["wk"], [512, 1024])
        lm_head = shape_summary("lm_head", shape, workload)
        self.assertEqual(lm_head["tokens"], 4096)
        self.assertEqual(lm_head["weight"], [151936, 1024])
        rope = shape_summary("rope", shape, workload)
        # Both engine-native forms, and the THD pair is a view of the BLNH
        # pair rather than a second allocation.
        self.assertEqual(rope["q_titan_BLNH"], [4, 1024, 16, 64])
        self.assertEqual(rope["k_titan_BLNH"], [4, 1024, 8, 64])
        self.assertEqual(rope["q_mcore_THD"], [4096, 16, 64])
        self.assertEqual(rope["k_mcore_THD"], [4096, 8, 64])
        self.assertEqual(rope["cu_seqlens_total"], 4096)
        self.assertEqual(rope["mcore_freqs"], [1024, 1, 1, 64])
        self.assertEqual(rope["rotated_rows"], 4096 * 24)

    def test_model_size_selects_the_geometry_not_the_workload(self) -> None:
        shape, workload = resolve_shape_and_workload(model_size="huge")
        self.assertEqual(shape.name, "huge")
        self.assertEqual(shape.dim, 12288)
        # The workload is untouched by the size: batch is not a model property.
        self.assertEqual(workload, KernelWorkload())
        self.assertEqual(
            shape_summary("lm_head", shape, workload)["hidden"],
            [4, 1024, 12288],
        )
        with self.assertRaisesRegex(ValueError, "Unknown model size"):
            resolve_shape_and_workload(model_size="enormous")

    def test_workload_overrides_apply(self) -> None:
        shape, workload = resolve_shape_and_workload(batch=1, seq_len=2048)
        self.assertEqual(
            shape_summary("swiglu", shape, workload)["x"], [4096, 1024]
        )

    def test_seq_len_is_bounded_by_the_shapes_ceiling(self) -> None:
        with self.assertRaisesRegex(ValueError, "exceeds max_seq_len"):
            resolve_shape_and_workload(seq_len=4096)

    def test_max_seq_len_override_replaces_the_shape_before_the_check(
        self,
    ) -> None:
        """Ordering is load-bearing: the attention sweep raises the ceiling
        precisely so it can then set seq_len above the old one."""
        shape, workload = resolve_shape_and_workload(
            seq_len=4096, max_seq_len=4096
        )
        self.assertEqual(shape.max_seq_len, 4096)
        self.assertEqual(workload.seq_len, 4096)
        # replace() on the registered shape, not a mutation of it.
        self.assertEqual(shape.name, "normal")
        self.assertEqual(resolve_shape_and_workload()[0].max_seq_len, 2048)
        self.assertEqual(
            shape_summary("attention", shape, workload)["max_seq_len"], 4096
        )

    def test_routing_needs_an_even_row_count(self) -> None:
        """rows = batch * seq_len * top_k, and both shapes carry top_k 2 with
        4 experts, so the split fails exactly when batch * seq_len is odd."""
        shape, workload = resolve_shape_and_workload(batch=3, seq_len=1025)
        self.assertFalse(routing_divides_evenly(shape, workload))  # 6150 rows
        # An odd batch alone is fine at the default even seq_len: 3 * 1024 * 2
        # = 6144 rows, which 4 experts do split.
        self.assertTrue(
            routing_divides_evenly(*resolve_shape_and_workload(batch=3))
        )
        self.assertTrue(
            routing_divides_evenly(*resolve_shape_and_workload(batch=4))
        )

    def test_split_covers_every_former_spec_field(self) -> None:
        """The twelve fields the single flat spec used to carry are all still
        reachable, each from exactly one of the two objects."""
        shape, workload = resolve_shape_and_workload()
        geometry = (
            "dim",
            "n_heads",
            "n_kv_heads",
            "head_dim",
            "num_experts",
            "top_k",
            "moe_hidden_dim",
            "vocab_size",
            "max_seq_len",
        )
        for name in geometry:
            self.assertTrue(hasattr(shape, name), name)
            # No facade: the workload must not answer geometry questions.
            self.assertFalse(hasattr(workload, name), name)
        for name in ("batch", "seq_len"):
            self.assertTrue(hasattr(workload, name), name)
            self.assertFalse(hasattr(shape, name), name)
        # The twelfth: the old spec's "theta" is PiperShape's "rope_theta".
        self.assertEqual(shape.rope_theta, 1_000_000.0)
        self.assertFalse(hasattr(shape, "theta"))


class BalancedRoutingInvariantTests(unittest.TestCase):
    """The invariant must hold at every entry point, not just the runner's.

    ``execute_kernel_run`` skips an unbalanced swiglu scenario loudly before
    it spawns a worker, but that is the friendly path, not the guard:
    ``python -m benchmarks.kernel.worker`` and direct ``run_kernel_scenario``
    callers (``tests/test_kernel_gpu_smoke.py``) never pass through it. Left
    unchecked there, ``swiglu_inputs`` builds ``batch * seq_len * top_k`` rows
    and then splits them into ``num_experts`` equal blocks that do not cover
    them -- a different workload measured under the scenario's name.
    """

    def test_run_kernel_scenario_rejects_an_uneven_split(self) -> None:
        from benchmarks.kernel.engine.run import RunOptions, run_kernel_scenario

        shape, workload = resolve_shape_and_workload(batch=3, seq_len=1025)
        with self.assertRaises(ValueError) as caught:
            run_kernel_scenario(
                kernel_scenario_by_name("swiglu"),
                shape,
                workload,
                RunOptions(),
                "test",
            )
        message = str(caught.exception)
        # The numbers, not just a complaint: 3 x 1025 x 2 rows over 4 experts.
        self.assertIn("6150 routed rows", message)
        self.assertIn("batch 3 x seq 1025 x top_k 2", message)
        self.assertIn("4 experts", message)
        # Raised before the CUDA check, so a CPU-only host -- where this test
        # runs -- diagnoses the workload rather than the missing device.


class RunOptionsCallSiteTests(unittest.TestCase):
    """Every ``RunOptions(...)`` in the tree names fields that exist.

    The GPU smoke test skips itself without CUDA, so a renamed option there
    stays green on every CPU host and fails only on the machine that can
    actually measure. That is how the round-robin ``n``/``warmup`` pair
    survived its own removal into a committed tree. This test reads the call
    sites with ``ast`` instead of executing them, so a CPU host catches it.
    """

    def _call_sites(self):
        import ast

        root = Path(__file__).resolve().parent.parent
        for path in sorted(root.glob("tests/*.py")) + sorted(
            root.glob("benchmarks/**/*.py")
        ):
            tree = ast.parse(path.read_text(), filename=str(path))
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Call)
                    and getattr(node.func, "id", None) == "RunOptions"
                ):
                    yield path.relative_to(root), node

    def test_every_call_site_uses_real_fields(self) -> None:
        import dataclasses

        from benchmarks.kernel.engine.run import RunOptions

        fields = {f.name for f in dataclasses.fields(RunOptions)}
        sites = list(self._call_sites())
        self.assertTrue(sites, "no RunOptions call sites found to check")
        violations = []
        for path, node in sites:
            unknown = {kw.arg for kw in node.keywords if kw.arg} - fields
            if unknown:
                violations.append(f"{path}:{node.lineno}: {sorted(unknown)}")
        self.assertEqual(
            violations,
            [],
            "RunOptions constructed with fields it does not have:\n  "
            + "\n  ".join(violations),
        )


def _stub_builder(modes: tuple[str, ...]):
    """A builder that exposes exactly ``modes`` and computes nothing.

    ``_seeded_build`` resolves a builder by dotted path, so the stubs below
    are module-level names this file hands back to it as
    ``"<this module>:<name>"``. They need no device: the gate under test
    compares two key sets.
    """

    def build(shape, workload, inputs):
        from benchmarks.kernel.engine.arm import BuiltArm

        return BuiltArm(
            name="stub",
            calls={mode: (lambda: None) for mode in modes},
            correctness_outputs=lambda: {},
        )

    return build


_stub_forward_only = _stub_builder(("forward",))
_stub_forward_and_backward = _stub_builder(("forward", "backward"))


# Residency probe. ``_residency_builder`` records, at each build, how many
# arms it handed back earlier are still alive. The gate loop must leave that
# count at zero for every build after the first.
_RESIDENT: list = []
_STILL_ALIVE: list[int] = []


def _residency_builder(shape, workload, inputs):
    import gc
    import weakref

    import torch

    from benchmarks.kernel.engine.arm import BuiltArm

    gc.collect()
    _STILL_ALIVE.append(sum(1 for ref in _RESIDENT if ref() is not None))
    built = BuiltArm(
        name="stub",
        calls={"forward": (lambda: None)},
        # A real arm's outputs carry a grad_fn, and the graph behind one
        # holds everything backward saved. The gate loop detaches for that
        # reason, so the probe returns a tensor that needs detaching too.
        correctness_outputs=lambda: {
            "out": torch.zeros(1, requires_grad=True) * 2
        },
    )
    _RESIDENT.append(weakref.ref(built))
    return built


class CorrectnessResidencyTests(unittest.TestCase):
    """One arm is resident at a time, and the pass keeps only the outputs.

    ``swiglu`` at the huge shape exhausted a 139 GiB device in the gate pass
    while each of its three arms fits alone. The pass held every arm at once;
    a check only ever needs two output *tensors*. This pins the fix, because
    the failure it prevents needs a GPU and a 10 B-parameter shape to
    reproduce and would otherwise be untested.
    """

    def _scenario(self, arms: int):
        from benchmarks.kernel.schema import KernelArm, KernelScenario

        return KernelScenario(
            name="residency",
            description="stub",
            inputs_builder=f"{__name__}:_residency_builder",
            reference_builder=None,
            baseline_arm="arm0",
            arms=tuple(
                KernelArm(
                    name=f"arm{index}",
                    description="stub",
                    builder=f"{__name__}:_residency_builder",
                    modes=("forward",),
                    eager_reason="a stub, not an implementation",
                )
                for index in range(arms)
            ),
        )

    def _run(self, arms: int, skip=frozenset()):
        from benchmarks.kernel.engine.run import gate_outputs

        _RESIDENT.clear()
        _STILL_ALIVE.clear()
        shape, workload = resolve_shape_and_workload()
        return gate_outputs(
            self._scenario(arms), shape, workload, {}, 0, skip
        )

    def test_no_earlier_arm_survives_the_next_build(self) -> None:
        outputs = self._run(4)
        self.assertEqual(sorted(outputs), ["arm0", "arm1", "arm2", "arm3"])
        self.assertEqual(
            _STILL_ALIVE,
            [0, 0, 0, 0],
            "an earlier arm was still alive when the next one was built; "
            "peak memory is then the sum of the arms, not the largest one",
        )

    def test_the_outputs_outlive_the_arms_that_made_them(self) -> None:
        outputs = self._run(2)
        for name, tensors in outputs.items():
            self.assertEqual(sorted(tensors), ["out"], name)
            self.assertEqual(tensors["out"].shape, (1,))

    def test_every_kept_tensor_is_detached(self) -> None:
        """A live graph would keep the activations the arm allocated."""
        for name, tensors in self._run(2).items():
            for output, tensor in tensors.items():
                self.assertIsNone(tensor.grad_fn, f"{name}/{output}")
                self.assertFalse(tensor.requires_grad, f"{name}/{output}")

    def test_a_skipped_arm_is_never_built(self) -> None:
        outputs = self._run(3, skip=frozenset({"arm1"}))
        self.assertEqual(sorted(outputs), ["arm0", "arm2"])
        self.assertEqual(len(_STILL_ALIVE), 2)


class SeededBuildContractTests(unittest.TestCase):
    """Where ``KernelArm`` becomes the authority over ``BuiltArm``.

    Every consumer downstream reads the *declared* modes: the manifest lists
    them, the merge pairs an arm against its opponent mode by mode, and a
    reader ranks implementations by them. A builder with one extra closure
    publishes a timed operation the registry never described; one with a
    closure missing leaves a declared mode absent from the table. Both are
    wrong in the same way, and both raise.
    """

    def _arm(self, builder: str, modes: tuple[str, ...]):
        from benchmarks.kernel.schema import KernelArm

        return KernelArm(
            name="stub",
            description="stub",
            builder=f"{__name__}:{builder}",
            modes=modes,
            eager_reason="a stub, not an implementation",
        )

    def _build(self, arm):
        from benchmarks.kernel.engine.run import _seeded_build

        shape, workload = resolve_shape_and_workload()
        return _seeded_build(arm, shape, workload, {}, 0)

    def test_the_builder_must_expose_exactly_the_declared_modes(self) -> None:
        extra = self._arm("_stub_forward_and_backward", ("forward",))
        with self.assertRaises(ValueError) as caught:
            self._build(extra)
        self.assertIn("['backward', 'forward']", str(caught.exception))
        self.assertIn("['forward']", str(caught.exception))

        missing = self._arm("_stub_forward_only", ("forward", "backward"))
        with self.assertRaises(ValueError) as caught:
            self._build(missing)
        self.assertIn("['forward']", str(caught.exception))
        self.assertIn("['backward', 'forward']", str(caught.exception))

        agreed = self._arm("_stub_forward_only", ("forward",))
        self.assertEqual(set(self._build(agreed).calls), {"forward"})


def _replicates(values: list[float], per_replicate: int) -> list[list[float]]:
    """Split a flat sample list into equal replicates."""
    return [
        values[start : start + per_replicate]
        for start in range(0, len(values), per_replicate)
    ]


class KernelComparisonTests(unittest.TestCase):
    def test_shifted_distributions_are_detected(self) -> None:
        base = _replicates([100.0 + 0.1 * (i % 7) for i in range(50)], 10)
        arm = _replicates([110.0 + 0.1 * (i % 5) for i in range(50)], 10)
        row = kernel_comparison(base, arm)
        self.assertAlmostEqual(row["median_ratio"], 1.1, places=1)
        self.assertLess(row["welch_p"], 1e-6)
        self.assertLess(row["mwu_p"], 1e-6)
        self.assertGreater(row["cohens_d"], 2.0)
        self.assertAlmostEqual(row["ratio"], 1.1, places=1)
        self.assertEqual(row["replicates"], 5)
        # A consistent shift puts the whole interval above 1.0.
        self.assertGreater(row["ratio_ci_low"], 1.0)

    def test_wilcoxon_is_gone(self) -> None:
        """It needed per-cycle pairing the round-robin provided.

        At the replicate level it cannot work either: the exact two-sided
        minimum p at n=5 is 0.0625, so it can never reject. Printing it would
        imply a test that cannot produce a result.
        """
        row = kernel_comparison(_replicates([100.0] * 20, 4), _replicates([100.0] * 20, 4))
        self.assertNotIn("wilcoxon_p", row)
        self.assertNotIn("arm_faster_fraction", row)

    def test_identical_samples_give_a_unit_ratio_and_a_degenerate_ci(self) -> None:
        values = _replicates([100.0] * 20, 4)
        row = kernel_comparison(values, [list(r) for r in values])
        self.assertAlmostEqual(row["median_ratio"], 1.0)
        self.assertEqual(row["cohens_d"], 0.0)
        self.assertAlmostEqual(row["ratio"], 1.0)
        self.assertAlmostEqual(row["ratio_ci_low"], 1.0)
        self.assertAlmostEqual(row["ratio_ci_high"], 1.0)
        self.assertAlmostEqual(row["replicate_ratio_spread"], 0.0)

    def test_a_single_replicate_reports_no_interval(self) -> None:
        """One replicate is a point estimate, and must not pretend otherwise."""
        row = kernel_comparison([[10.0, 11.0]], [[20.0, 22.0]])
        self.assertEqual(row["replicates"], 1)
        self.assertAlmostEqual(row["ratio"], 2.0)
        self.assertIsNone(row["ratio_ci_low"])
        self.assertIsNone(row["ratio_ci_high"])

    def test_the_bootstrap_is_reproducible_from_the_same_samples(self) -> None:
        """A fixed seed, so a results file can be re-analyzed without drift."""
        base = _replicates([100.0 + (i % 9) for i in range(45)], 9)
        arm = _replicates([104.0 + (i % 5) for i in range(45)], 9)
        first = kernel_comparison(base, arm)
        second = kernel_comparison(base, arm)
        self.assertEqual(first["ratio_ci_low"], second["ratio_ci_low"])
        self.assertEqual(first["ratio_ci_high"], second["ratio_ci_high"])

    def test_pooled_sd_is_df_weighted(self) -> None:
        base = [10.0, 12.0, 14.0, 16.0]
        arm = [11.0, 13.0]
        row = kernel_comparison([base], [arm])
        import statistics

        sd_base = statistics.stdev(base)
        sd_arm = statistics.stdev(arm)
        pooled = (
            ((len(base) - 1) * sd_base**2 + (len(arm) - 1) * sd_arm**2)
            / (len(base) + len(arm) - 2)
        ) ** 0.5
        expected = (statistics.mean(arm) - statistics.mean(base)) / pooled
        self.assertAlmostEqual(row["cohens_d"], expected)


class FragmentNameTests(unittest.TestCase):
    """An arm name becomes a filename exactly once, and must survive it.

    The cross-engine roster spells an arm ``mcore/base``. Without the
    substitution a fragment path nests into a directory nothing creates, the
    worker fails to write, and the arm reaches the results as ``failed`` with
    no recoverable cause.
    """

    def test_a_slash_never_reaches_the_filename(self) -> None:
        from benchmarks.kernel.schema import (
            fragment_stem,
            timing_fragment_path as fragment_path,
        )

        self.assertEqual(fragment_stem("mcore/base"), "mcore-base")
        self.assertEqual(fragment_stem("titan"), "titan")
        path = fragment_path(Path("/tmp/fragments"), "mcore/base", 0)
        self.assertEqual(path.parent, Path("/tmp/fragments"))
        self.assertEqual(path.name, "timing__mcore-base__r0.json")

    def test_colliding_stems_are_refused_at_declaration(self) -> None:
        from benchmarks.kernel.schema import KernelArm, KernelScenario

        def arm(name: str) -> KernelArm:
            return KernelArm(
                name=name,
                description="d",
                builder="m:f",
                modes=("forward",),
                eager_reason="a stub, not an implementation",
            )

        with self.assertRaises(ValueError) as caught:
            KernelScenario(
                name="clash",
                description="d",
                inputs_builder="m:i",
                reference_builder=None,
                arms=(arm("mcore/base"), arm("mcore-base")),
                baseline_arm="mcore/base",
            )
        self.assertIn("overwrite", str(caught.exception))

    def test_every_declared_arm_has_a_distinct_stem(self) -> None:
        for scenario in KERNEL_SCENARIOS.values():
            stems = [fragment_stem(a.name) for a in scenario.arms]
            self.assertEqual(len(stems), len(set(stems)), scenario.name)


class PhaseTableTests(unittest.TestCase):
    """The worker's wall-clock attribution, which is stdlib-only by design.

    A phase table costs nothing and explains where a worker's twenty seconds
    go. What it must never do is need a device or a torch import to record a
    span, because the first span it records ends at the first torch import.
    """

    def setUp(self) -> None:
        from benchmarks.kernel.engine import phases

        self.phases = phases
        phases.reset()
        self.addCleanup(phases.reset)

    def test_a_span_is_recorded_under_its_name(self) -> None:
        with self.phases.phase("first"):
            pass
        with self.phases.phase("second"):
            pass
        self.assertEqual(
            [entry["phase"] for entry in self.phases.phases()],
            ["first", "second"],
        )
        for entry in self.phases.phases():
            self.assertGreaterEqual(entry["seconds"], 0.0)

    def test_a_span_closes_even_when_the_block_raises(self) -> None:
        """A failed build must still leave its own cost attributed.

        Without this the phase table of every worker that died would be the
        table of a worker that never got started, which is the case a reader
        most wants to see.
        """
        with self.assertRaises(RuntimeError):
            with self.phases.phase("failed"):
                raise RuntimeError("boom")
        self.assertEqual(
            [entry["phase"] for entry in self.phases.phases()], ["failed"]
        )

    def test_the_exit_hook_runs_inside_the_span(self) -> None:
        """The synchronize a setup boundary passes is charged to that phase.

        Kernel launches are asynchronous, so a boundary that closed before the
        device caught up would charge one phase's device work to the next --
        usually the arm build, which is the number the table exists to
        attribute.
        """
        seen: list[str] = []
        with self.phases.phase("boundary", lambda: seen.append("hook")):
            seen.append("body")
        self.assertEqual(seen, ["body", "hook"])
        self.assertEqual(len(self.phases.phases()), 1)

    def test_a_span_this_module_did_not_time_can_be_recorded(self) -> None:
        self.phases.record("process_startup", 1.25)
        self.assertEqual(
            self.phases.phases(), [{"phase": "process_startup", "seconds": 1.25}]
        )

    def test_the_process_start_offset_is_a_plausible_age(self) -> None:
        offset = self.phases.process_start_offset()
        if offset is None:
            self.skipTest("no /proc on this platform")
        self.assertGreater(offset, 0.0)
        self.assertLess(offset, 24 * 3600)


class WorkerExitTests(unittest.TestCase):
    """``_exit_now`` ends the process, so it owns every buffer in it.

    ``os._exit`` skips the interpreter unwind, which is the whole point: it
    also skips libc's ``exit()``, and with it the flush of libc's own output
    streams. Python's two flushes do not reach them. The parent redirects a
    worker's stdout to a file, so libc's stdout is block-buffered, and any C
    extension that printed through it would lose its output at the exit codes
    whose only report to the operator is a tail of that log.
    """

    @unittest.skipUnless(
        sys.platform.startswith("linux"), "libc.so.6 is Linux-specific"
    )
    def test_c_level_output_survives_the_exit(self) -> None:
        import subprocess

        root = str(Path(__file__).resolve().parent.parent)
        script = "\n".join(
            [
                "import ctypes, sys",
                f"sys.path.insert(0, {root!r})",
                "from benchmarks.kernel.worker import _exit_now",
                'ctypes.CDLL("libc.so.6").printf(b"LIBC-LINE\\n")',
                'sys.stdout.write("PYTHON-LINE\\n")',
                "_exit_now(0)",
            ]
        )
        # capture_output gives the child a pipe, which is block-buffered
        # exactly as the run's log file is.
        completed = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("PYTHON-LINE", completed.stdout)
        self.assertIn("LIBC-LINE", completed.stdout)


class WorkerArgumentTests(unittest.TestCase):
    """The worker's own argument guards, which no other test reaches.

    Every runner test drives a fake process, so the real parser runs only on
    the hardware. ``--fragment`` was unconditionally required until the timing
    pass started writing a file per replicate; it is now conditional, and
    these branches are the only thing between a mis-wired parent and a
    ``None`` path. Each guard is checked by its message, so a branch that
    fires for the wrong reason fails here rather than passing as "some error".
    """

    def error_for(self, *argv: str) -> str:
        """Parse ``argv`` and return the message argparse exits with."""
        from benchmarks.kernel.worker import parse_args

        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as caught:
                parse_args(list(argv))
        self.assertEqual(caught.exception.code, 2)
        return stderr.getvalue()

    def test_a_timing_pass_needs_a_fragments_directory(self) -> None:
        message = self.error_for(
            "--scenario", "qkv",
            "--mode", "timing",
            "--arm", "baseline",
            "--replicate", "0",
        )
        self.assertIn("--mode timing requires --fragments-dir", message)

    def test_a_timing_pass_needs_an_arm_and_a_replicate(self) -> None:
        message = self.error_for(
            "--scenario", "qkv",
            "--mode", "timing",
            "--fragments-dir", "/tmp/fragments",
        )
        self.assertIn("--mode timing requires --arm and --replicate", message)

    def test_a_block_covers_at_least_one_replicate(self) -> None:
        message = self.error_for(
            "--scenario", "qkv",
            "--mode", "timing",
            "--arm", "baseline",
            "--replicate", "0",
            "--fragments-dir", "/tmp/fragments",
            "--replicate-count", "0",
        )
        self.assertIn("--replicate-count must be >= 1", message)

    def test_a_correctness_pass_needs_a_fragment_path(self) -> None:
        message = self.error_for("--scenario", "qkv", "--mode", "correctness")
        self.assertIn("--mode correctness requires --fragment", message)

    def test_a_well_formed_timing_argv_is_accepted(self) -> None:
        """The control. Without it a parser that refused everything would
        satisfy all four guards above."""
        from benchmarks.kernel.worker import parse_args

        args = parse_args(
            [
                "--scenario", "qkv",
                "--mode", "timing",
                "--arm", "baseline",
                "--replicate", "2",
                "--replicate-count", "3",
                "--fragments-dir", "/tmp/fragments",
            ]
        )
        self.assertEqual(args.arm, "baseline")
        self.assertEqual(args.replicate, 2)
        self.assertEqual(args.replicate_count, 3)
        self.assertEqual(args.fragments_dir, Path("/tmp/fragments"))


if __name__ == "__main__":
    unittest.main()
