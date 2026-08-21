"""CPU tests for the ``cross_entropy`` kernel scenario.

What is testable without a GPU is the half of the scenario that decides
whether the numbers mean anything: the shared inputs, the fp64 reference both
engines are gated against, the layout charge the scenario applies to the titan
side, the arm shape every builder returns, and the two megatron profiles that
make ``mcore/ce_native`` and ``mcore/no_ce_fusion`` different arms rather than
three spellings of one.

Neither engine's real loss runs here. The titan losses need Triton and the
megatron ones need CUDA, TransformerEngine and a process group, so their
guards live in the builders (``_assert_profile_took``,
``_assert_te_cross_entropy_available``) where they run on the box that
measures. ``_assert_profile_took`` is the exception and is covered directly:
it is torch-free arithmetic over a mapping, and it is the only thing that
distinguishes the three megatron arms.

The tolerances below mirror what the declaration gates on. A test that
invented its own would pass while the run failed.
"""

import sys
import types
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
import torch.nn.functional as F

from benchmarks.kernel.engine.run import resolve_symbol
from benchmarks.kernel.operations import cross_entropy as ce
from benchmarks.kernel.schema import MODES, KernelWorkload, fragment_stem
from benchmarks.models.piper_qwen3.mcore_profiles import BASE, FUSION_FIELDS
from benchmarks.models.piper_qwen3.shape import PiperShape

# Small enough to run in well under a second, and every mechanism under test
# is the one a GPU arm uses. The vocabulary stays long enough that a softmax
# over it is a real reduction rather than a pair of terms.
TINY = PiperShape.derived(name="tiny", dim=256, n_layers=2, vocab_size=64)
WORKLOAD = KernelWorkload(batch=2, seq_len=8)

# A row count the reference's chunk loop cannot cover in whole blocks, used to
# prove the tail is written rather than left at whatever ``torch.empty`` held.
RAGGED_WORKLOAD = KernelWorkload(batch=2, seq_len=5)
RAGGED_CHUNK = 4

# The two names every arm's ``correctness_outputs`` produces and every gate
# compares. The reference must produce exactly these and nothing else, or
# ``benchmarks/kernel/engine/correctness.py`` raises on the missing side.
GATED_OUTPUTS = ("loss", "logits_grad")

# The modes every arm declares. There is deliberately no isolated "backward":
# every implementation here mutates the buffer it saved, so a retained-graph
# re-run would rescale the same tensor twice. See the module docstring of
# benchmarks/kernel/operations/cross_entropy.py.
DECLARED_MODES = ("forward", "forward_backward")

# The dotted paths the registry declaration names. The engine resolves these
# strings inside the worker, so a renamed builder is a run-time failure on the
# GPU rather than an import error here.
DECLARED_BUILDER_PATHS = (
    "benchmarks.kernel.operations.cross_entropy:cross_entropy_inputs",
    "benchmarks.kernel.operations.cross_entropy:cross_entropy_reference",
    "benchmarks.kernel.operations.cross_entropy"
    ":build_cross_entropy_mcore_base",
    "benchmarks.kernel.operations.cross_entropy"
    ":build_cross_entropy_mcore_ce_native",
    "benchmarks.kernel.operations.cross_entropy"
    ":build_cross_entropy_mcore_no_ce_fusion",
    "benchmarks.kernel.operations.cross_entropy"
    ":build_cross_entropy_titan_full_logits",
    "benchmarks.kernel.operations.cross_entropy"
    ":build_cross_entropy_titan_te_fused_ce",
    "benchmarks.kernel.operations.cross_entropy"
    ":build_cross_entropy_titan_piper_optimized_te_ce",
)

DECLARED_ARM_NAMES = (
    ce.MCORE_BASE_ARM,
    ce.MCORE_CE_NATIVE_ARM,
    ce.MCORE_NO_CE_FUSION_ARM,
    ce.TITAN_FULL_LOGITS_ARM,
    ce.TITAN_TE_FUSED_CE_ARM,
    ce.TITAN_PIPER_ARM,
)


def _inputs(workload: KernelWorkload = WORKLOAD) -> ce.CrossEntropyInputs:
    generator = torch.Generator()
    generator.manual_seed(0)
    return ce.cross_entropy_inputs(
        TINY, workload, torch.device("cpu"), generator
    )


def _fp64_truth(
    inputs: ce.CrossEntropyInputs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The loss and gradient an unchunked fp64 autograd pass produces."""
    leaf = inputs.logits.double().detach().requires_grad_()
    loss = (
        F.cross_entropy(
            leaf.flatten(0, 1),
            inputs.labels.flatten(0, 1),
            reduction="sum",
        )
        / inputs.valid_tokens
    )
    loss.backward()
    return loss.detach(), leaf.grad


class SharedInputsTests(unittest.TestCase):
    def test_logits_are_contiguous_bf16_at_the_titan_layout(self) -> None:
        inputs = _inputs()
        self.assertEqual(
            tuple(inputs.logits.shape),
            (WORKLOAD.batch, WORKLOAD.seq_len, TINY.vocab_size),
        )
        self.assertEqual(inputs.logits.dtype, torch.bfloat16)
        # The mcore arms transpose this at build time. A non-contiguous
        # shared tensor would make that transpose a second copy and charge
        # megatron for the harness's storage order.
        self.assertTrue(inputs.logits.is_contiguous())

    def test_labels_are_int64_over_the_whole_vocabulary(self) -> None:
        inputs = _inputs()
        self.assertEqual(
            tuple(inputs.labels.shape), (WORKLOAD.batch, WORKLOAD.seq_len)
        )
        self.assertEqual(inputs.labels.dtype, torch.int64)
        self.assertGreaterEqual(int(inputs.labels.min()), 0)
        self.assertLess(int(inputs.labels.max()), TINY.vocab_size)

    def test_every_token_is_valid(self) -> None:
        # No -100 anywhere, so both engines normalize by the same count and
        # the fp64 reference needs no ignore-index branch.
        inputs = _inputs()
        self.assertEqual(
            inputs.valid_tokens, float(WORKLOAD.batch * WORKLOAD.seq_len)
        )
        self.assertEqual(int((inputs.labels == -100).sum()), 0)

    def test_inputs_rebuild_bit_identically_from_one_seed(self) -> None:
        # Every worker rebuilds the inputs rather than receiving them, so an
        # arm timed in one process must see the tensor the gate saw in
        # another.
        first, second = _inputs(), _inputs()
        self.assertTrue(torch.equal(first.logits, second.logits))
        self.assertTrue(torch.equal(first.labels, second.labels))


class ReferenceTests(unittest.TestCase):
    def test_reference_produces_exactly_the_gated_outputs(self) -> None:
        inputs = _inputs()
        reference = ce.cross_entropy_reference(TINY, WORKLOAD, inputs)
        self.assertEqual(set(reference), set(GATED_OUTPUTS))

    def test_loss_matches_an_unchunked_fp64_pass(self) -> None:
        inputs = _inputs()
        reference = ce.cross_entropy_reference(TINY, WORKLOAD, inputs)
        expected, _ = _fp64_truth(inputs)
        torch.testing.assert_close(
            reference["loss"], expected.float(), rtol=1e-6, atol=1e-6
        )

    def test_gradient_matches_an_unchunked_fp64_pass(self) -> None:
        inputs = _inputs()
        reference = ce.cross_entropy_reference(TINY, WORKLOAD, inputs)
        _, expected = _fp64_truth(inputs)
        torch.testing.assert_close(
            reference["logits_grad"], expected.float(), rtol=1e-6, atol=1e-9
        )

    def test_gradient_is_fp32_at_the_canonical_shape(self) -> None:
        # fp32 rather than fp64 on purpose: the correctness engine casts both
        # sides to fp32 before subtracting, and holding fp64 would double the
        # 2.3 GiB this tensor costs at the real shape for no accuracy.
        inputs = _inputs()
        reference = ce.cross_entropy_reference(TINY, WORKLOAD, inputs)
        self.assertEqual(reference["logits_grad"].dtype, torch.float32)
        self.assertEqual(
            tuple(reference["logits_grad"].shape),
            (WORKLOAD.batch, WORKLOAD.seq_len, TINY.vocab_size),
        )

    def test_the_row_chunk_does_not_change_the_result(self) -> None:
        inputs = _inputs()
        with mock.patch.object(ce, "REFERENCE_ROW_CHUNK", 1):
            one_row = ce.cross_entropy_reference(TINY, WORKLOAD, inputs)
        with mock.patch.object(ce, "REFERENCE_ROW_CHUNK", 4096):
            all_rows = ce.cross_entropy_reference(TINY, WORKLOAD, inputs)
        torch.testing.assert_close(
            one_row["loss"], all_rows["loss"], rtol=1e-6, atol=1e-6
        )
        torch.testing.assert_close(
            one_row["logits_grad"],
            all_rows["logits_grad"],
            rtol=1e-6,
            atol=1e-9,
        )

    def test_a_ragged_tail_is_written_rather_than_left_uninitialized(
        self,
    ) -> None:
        # The gradient buffer is torch.empty, so a chunk loop that missed the
        # tail would publish whatever the allocator held -- and would still
        # pass every shape assertion above.
        inputs = _inputs(RAGGED_WORKLOAD)
        rows = RAGGED_WORKLOAD.batch * RAGGED_WORKLOAD.seq_len
        self.assertNotEqual(rows % RAGGED_CHUNK, 0)
        with mock.patch.object(ce, "REFERENCE_ROW_CHUNK", RAGGED_CHUNK):
            reference = ce.cross_entropy_reference(
                TINY, RAGGED_WORKLOAD, inputs
            )
        _, expected = _fp64_truth(inputs)
        torch.testing.assert_close(
            reference["logits_grad"], expected.float(), rtol=1e-6, atol=1e-9
        )


class LabelPreparationTests(unittest.TestCase):
    """The layout charge the scenario applies to both engines."""

    def test_prepared_labels_hold_the_same_values_and_shape(self) -> None:
        labels = _inputs().labels
        prepared = ce._prepared_labels(labels)
        self.assertEqual(prepared.shape, labels.shape)
        self.assertTrue(torch.equal(prepared, labels))

    def test_preparation_materializes_the_transposed_copy(self) -> None:
        # This is the kernel megatron runs at language_module.py:172. A view
        # would charge titan nothing and put a megatron-only dispatch cost
        # into the published ratio.
        labels = _inputs().labels
        prepared = ce._prepared_labels(labels)
        self.assertNotEqual(
            prepared.data_ptr(), labels.data_ptr(), "no copy was made"
        )
        self.assertTrue(prepared.transpose(0, 1).is_contiguous())

    def test_prepared_labels_still_need_a_contiguous_pass(self) -> None:
        # The second small copy: the loss's own flatten, or the TE-family
        # ``target.stride(-1) != 1`` guard, materializes it. Two layout
        # kernels per call on each engine is the symmetry the scenario claims.
        prepared = ce._prepared_labels(_inputs().labels)
        self.assertNotEqual(prepared.stride(-1), 1)
        self.assertFalse(prepared.is_contiguous())


class ArmShapeTests(unittest.TestCase):
    """What ``_cross_entropy_arm`` guarantees, on a stand-in loss."""

    def _tracked_arm(self, leaf, loss_call, to_canonical=lambda g: g):
        created: list[torch.Tensor] = []

        def make_leaf() -> torch.Tensor:
            tensor = leaf()
            created.append(tensor)
            return tensor

        return created, ce._cross_entropy_arm(
            "toy", make_leaf, loss_call, to_canonical
        )

    def test_calls_are_exactly_the_declared_modes(self) -> None:
        # benchmarks/kernel/engine/run.py:_seeded_build raises when these
        # disagree with the registry, so the declaration and the builder are
        # checked against each other here too.
        inputs = _inputs()
        _, arm = self._tracked_arm(
            lambda: inputs.logits.clone().requires_grad_(),
            lambda logits: logits.float().sum(),
        )
        self.assertEqual(set(arm.calls), set(DECLARED_MODES))
        for mode in DECLARED_MODES:
            self.assertIn(mode, MODES)

    def test_correctness_outputs_name_the_gated_tensors(self) -> None:
        inputs = _inputs()
        _, arm = self._tracked_arm(
            lambda: inputs.logits.clone().requires_grad_(),
            lambda logits: logits.float().sum(),
        )
        outputs = arm.correctness_outputs()
        self.assertEqual(set(outputs), set(GATED_OUTPUTS))
        self.assertEqual(outputs["loss"].dtype, torch.float32)
        self.assertEqual(
            tuple(outputs["logits_grad"].shape), tuple(inputs.logits.shape)
        )

    def test_each_mode_and_the_gate_own_a_separate_leaf(self) -> None:
        inputs = _inputs()
        created, arm = self._tracked_arm(
            lambda: inputs.logits.clone().requires_grad_(),
            lambda logits: logits.float().sum(),
        )
        self.assertEqual(len(created), len(DECLARED_MODES))
        arm.correctness_outputs()
        self.assertEqual(len(created), len(DECLARED_MODES) + 1)
        self.assertEqual(
            len({tensor.data_ptr() for tensor in created}), len(created)
        )
        # And none of them is the shared tensor the other arms read.
        for tensor in created:
            self.assertNotEqual(tensor.data_ptr(), inputs.logits.data_ptr())

    def test_the_gate_starts_from_pristine_logits(self) -> None:
        # mcore/base overwrites its input with the gradient, so a gate that
        # reused a timed leaf would compare the second call's arithmetic
        # against the reference. The stand-in loss below destroys its input
        # the same way; two gate calls must still agree.
        inputs = _inputs()

        def destructive(logits: torch.Tensor) -> torch.Tensor:
            loss = logits.float().sum()
            logits.data.fill_(0.0)
            return loss

        _, arm = self._tracked_arm(
            lambda: inputs.logits.clone().requires_grad_(), destructive
        )
        first = arm.correctness_outputs()["loss"].clone()
        second = arm.correctness_outputs()["loss"].clone()
        self.assertTrue(torch.equal(first, second))
        self.assertNotEqual(float(first), 0.0)

    def test_to_canonical_maps_the_mcore_layout_back(self) -> None:
        inputs = _inputs()
        transposed = inputs.logits.transpose(0, 1).contiguous()
        _, arm = self._tracked_arm(
            lambda: transposed.clone().requires_grad_(),
            lambda logits: logits.float().sum(),
            lambda gradient: gradient.transpose(0, 1),
        )
        outputs = arm.correctness_outputs()
        self.assertEqual(
            tuple(outputs["logits_grad"].shape), tuple(inputs.logits.shape)
        )


class MegatronProfileTests(unittest.TestCase):
    def test_ce_native_changes_only_the_fusion_implementation(self) -> None:
        changed = {
            key
            for key, value in ce.CE_NATIVE_PROFILE.config_overrides.items()
            if BASE.config_overrides.get(key) != value
        }
        self.assertEqual(changed, {"cross_entropy_fusion_impl"})
        self.assertEqual(
            ce.CE_NATIVE_PROFILE.config_overrides[
                "cross_entropy_fusion_impl"
            ],
            "native",
        )
        self.assertTrue(
            ce.CE_NATIVE_PROFILE.config_overrides["cross_entropy_loss_fusion"]
        )

    def test_no_ce_fusion_changes_only_the_fusion_flag(self) -> None:
        changed = {
            key
            for key, value in ce.NO_CE_FUSION_PROFILE.config_overrides.items()
            if BASE.config_overrides.get(key) != value
        }
        self.assertEqual(changed, {"cross_entropy_loss_fusion"})
        self.assertFalse(
            ce.NO_CE_FUSION_PROFILE.config_overrides[
                "cross_entropy_loss_fusion"
            ]
        )

    def test_the_variants_are_named_and_keep_every_other_flag(self) -> None:
        for profile in (ce.CE_NATIVE_PROFILE, ce.NO_CE_FUSION_PROFILE):
            with self.subTest(profile=profile.name):
                self.assertNotEqual(profile.name, BASE.name)
                self.assertTrue(profile.description)
                self.assertEqual(
                    set(profile.config_overrides), set(BASE.config_overrides)
                )

    def test_both_changed_fields_are_fields_the_guard_checks(self) -> None:
        # declared_mismatches only reads FUSION_FIELDS, so a variant whose
        # delta sits outside that tuple could silently fail to take.
        self.assertIn("cross_entropy_fusion_impl", FUSION_FIELDS)
        self.assertIn("cross_entropy_loss_fusion", FUSION_FIELDS)

    def test_assert_profile_took_accepts_an_agreeing_config(self) -> None:
        config = types.SimpleNamespace(
            **ce.CE_NATIVE_PROFILE.config_overrides
        )
        ce._assert_profile_took(ce.CE_NATIVE_PROFILE, config)

    def test_assert_profile_took_rejects_a_delta_that_did_not_take(
        self,
    ) -> None:
        # The variant asked for megatron's native CE and got TE's: the base
        # implementation published under the variant's name, which no
        # correctness gate can see because both are numerically right.
        config = types.SimpleNamespace(**BASE.config_overrides)
        with self.assertRaisesRegex(
            RuntimeError, "cross_entropy_fusion_impl"
        ):
            ce._assert_profile_took(ce.CE_NATIVE_PROFILE, config)

    def test_assert_profile_took_rejects_a_flag_that_came_out_off(
        self,
    ) -> None:
        # The other direction, and the one that once cost 11.9 GPU ms/step.
        overrides = dict(BASE.config_overrides)
        overrides["bias_activation_fusion"] = False
        with self.assertRaisesRegex(RuntimeError, "bias_activation_fusion"):
            ce._assert_profile_took(BASE, types.SimpleNamespace(**overrides))


class ShippedDefaultTests(unittest.TestCase):
    """Which of the three megatron arms is "megatron as NVIDIA ships it"."""

    def test_no_ce_fusion_is_the_arm_that_matches_megatron_s_default(
        self,
    ) -> None:
        # megatron's only default in the tree is cross_entropy_loss_fusion =
        # False (model_parallel_config.py:320), and megatron/training/
        # arguments.py declares no add_argument for it on this rev -- the only
        # mentions are the assert at :1631-1634 -- so no argparse default
        # overrides the dataclass one. The shipped path is therefore the
        # unfused one, and ce_native is the fused non-TE path.
        #
        # CLAUDE.md attaches "as NVIDIA ships it" to "native", which is right
        # for the narrower te-versus-native choice the e2e arm makes and wrong
        # for a roster holding all three. This pins the relationship the arm
        # descriptions state, so a future edit cannot quietly move the label.
        self.assertIs(
            ce.NO_CE_FUSION_PROFILE.config_overrides["cross_entropy_loss_fusion"],
            False,
        )
        self.assertIs(
            ce.CE_NATIVE_PROFILE.config_overrides["cross_entropy_loss_fusion"],
            True,
        )
        self.assertEqual(
            ce.CE_NATIVE_PROFILE.config_overrides["cross_entropy_fusion_impl"],
            "native",
        )
        self.assertEqual(
            BASE.config_overrides["cross_entropy_fusion_impl"], "te"
        )


class _FakeGptModel(nn.Module):
    """Enough of a ``GPTModel`` for the release path: submodules and the
    three attributes ``compute_language_model_loss`` reads."""

    def __init__(
        self,
        *,
        tp_group: object | None = object(),
        submodules: tuple[str, ...] | None = None,
    ) -> None:
        super().__init__()
        # Every name the release drops, so the assertion exercises the drop
        # rather than passing on an attribute that was never there.
        if submodules is None:
            submodules = ce.MCORE_RELEASED_SUBMODULES
        for name in submodules:
            setattr(self, name, nn.Linear(2, 2))
        self.config = object()
        self.pg_collection = object()
        self.tp_group = tp_group


class ModelReleaseTests(unittest.TestCase):
    def test_release_drops_every_parameter_holding_submodule(self) -> None:
        model = _FakeGptModel()
        ce._release_model_parameters(model)
        for name in ce.MCORE_RELEASED_SUBMODULES:
            with self.subTest(submodule=name):
                self.assertIsNone(getattr(model, name, None))
        self.assertEqual(list(model.parameters()), [])

    def test_release_tolerates_a_submodule_the_model_does_not_have(
        self,
    ) -> None:
        # GPTModel builds mtp only under mtp_process, which no profile here
        # sets, so the real model carries no mtp at this configuration. Naming
        # it must stay free: a future MTP profile would otherwise leave a
        # second set of parameters resident and inflate peak_memory_gib.
        self.assertIn("mtp", ce.MCORE_RELEASED_SUBMODULES)
        model = _FakeGptModel(submodules=("embedding", "decoder"))
        ce._release_model_parameters(model)
        self.assertEqual(list(model.parameters()), [])

    def test_release_keeps_what_the_loss_method_reads(self) -> None:
        model = _FakeGptModel()
        ce._release_model_parameters(model)
        for attribute in ce.MCORE_LOSS_ATTRIBUTES:
            with self.subTest(attribute=attribute):
                self.assertIsNotNone(getattr(model, attribute, None))

    def test_release_refuses_when_a_required_attribute_is_absent(self) -> None:
        # A megatron bump that renamed tp_group would otherwise leave the arm
        # to raise from inside its own timed closure.
        model = _FakeGptModel(tp_group=None)
        with self.assertRaisesRegex(RuntimeError, "tp_group"):
            ce._release_model_parameters(model)


class DeclarationTests(unittest.TestCase):
    def test_every_declared_builder_path_resolves(self) -> None:
        for path in DECLARED_BUILDER_PATHS:
            with self.subTest(path=path):
                self.assertTrue(callable(resolve_symbol(path)))

    def test_arm_names_and_their_fragment_stems_are_unique(self) -> None:
        # Two arms whose stems collide would overwrite each other's timing
        # fragment, and the second would be published under the first's
        # samples. KernelScenario checks this at import; the roster is checked
        # here so a rename is caught before the registry merge.
        self.assertEqual(len(set(DECLARED_ARM_NAMES)), len(DECLARED_ARM_NAMES))
        stems = [fragment_stem(name) for name in DECLARED_ARM_NAMES]
        self.assertEqual(len(set(stems)), len(stems))
        for stem in stems:
            self.assertNotIn("/", stem)

    def test_every_arm_name_says_which_engine_it_is(self) -> None:
        for name in DECLARED_ARM_NAMES:
            with self.subTest(arm=name):
                engine, separator, profile = name.partition("/")
                self.assertEqual(separator, "/")
                self.assertIn(engine, {"mcore", "titan"})
                self.assertTrue(profile)


if __name__ == "__main__":
    unittest.main()
