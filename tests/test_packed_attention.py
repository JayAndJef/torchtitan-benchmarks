"""The packed-document FA3 override: the offset builder, the config swap, the block checks and GPU parity with Flex."""

import dataclasses
import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from torchtitan.config import derive
from torchtitan.protocols.module import Module

from benchmarks.models.piper_qwen3.components.attention.packed import (
    PackedGQAttention,
    microbatch_offsets,
)
from benchmarks.models.piper_qwen3.shape import PIPER_30B_A3B, PiperShape
from benchmarks.models.piper_qwen3.titan_model import (
    _piper_1b_model,
    apply_config_overrides,
    build_titan_model,
)

FA3_OVERRIDE = (
    "benchmarks.models.piper_qwen3.components.attention.fa3_override."
    "packed_fa3_attention"
)

TINY = PiperShape.derived(name="tiny", dim=256, n_layers=2, vocab_size=64)

GQA_PROBE = dataclasses.replace(
    PIPER_30B_A3B, name="gqa-probe", n_layers=2, num_experts=8, top_k=2, vocab_size=1024
)
"""The attention geometry of 30b-a3b (32 query heads, 4 kv heads, head_dim 128) at two layers."""


def positions_from_docs(rows: list[list[int]]) -> torch.Tensor:
    """The positions of rows of packed documents with the given lengths."""
    return torch.stack(
        [torch.cat([torch.arange(n) for n in row]) for row in rows]
    )


def exact_offsets(rows: list[list[int]]) -> list[int]:
    """The offsets of the documents of ``rows``, packed into one microbatch."""
    starts, offset = [], 0
    for row in rows:
        for n in row:
            starts.append(offset)
            offset += n
    return starts + [offset]


class MicrobatchOffsetsTests(unittest.TestCase):
    def test_rows_with_uneven_document_counts(self) -> None:
        rows = [[3, 5], [8], [1, 1, 6], [2, 2, 2, 2]]
        offsets = microbatch_offsets(positions_from_docs(rows), 4)
        self.assertEqual(offsets.dtype, torch.int32)
        self.assertEqual(offsets.tolist(), [exact_offsets(rows)])

    def test_each_microbatch_starts_at_zero_and_a_shorter_row_pads_at_the_end(self) -> None:
        """The pipeline splitter cuts dim 0, so each microbatch reads its own row."""
        rows = [[3, 5], [1, 1, 1, 5], [8], [6, 2]]
        offsets = microbatch_offsets(positions_from_docs(rows), 2)
        self.assertEqual(
            offsets.tolist(),
            [exact_offsets(rows[:2]), exact_offsets(rows[2:]) + [16, 16, 16]],
        )
        first, second = offsets.chunk(2, dim=0)
        self.assertEqual(tuple(first.shape), (1, 7))
        self.assertEqual(tuple(second.shape), (1, 7))

    def test_a_row_that_does_not_start_a_document_raises(self) -> None:
        positions = positions_from_docs([[4, 4], [8]])
        positions[1, :4] += 1
        with self.assertRaisesRegex(ValueError, "does not start a document"):
            microbatch_offsets(positions, 1)

    def test_rows_that_do_not_divide_into_microbatches_raise(self) -> None:
        with self.assertRaisesRegex(ValueError, "do not divide into microbatches"):
            microbatch_offsets(positions_from_docs([[8], [8], [8]]), 2)


def _positional_names(forward) -> list[str]:
    """The positional names that ``Module._cache_pos_arg_names`` reads from a class with this ``forward``."""
    probe = type("Probe", (), {"forward": forward, "_pos_arg_list": None})()
    return Module._cache_pos_arg_names(probe)


class OverrideTests(unittest.TestCase):
    def test_the_override_replaces_one_attention_per_block(self) -> None:
        sentinel = object()
        config = _piper_1b_model(fuse_qkv=True, shape=TINY)
        config.layers[0].attention.inner_attention.sharding_config = sentinel
        lines = apply_config_overrides(config, [FA3_OVERRIDE], expected=TINY.n_layers)
        for layer, line in enumerate(lines):
            self.assertIn(f"layers.{layer}.attention", line)
        for layer in config.layers:
            attention = layer.attention
            self.assertIs(type(attention), PackedGQAttention.Config)
            self.assertEqual(
                type(attention.inner_attention).__qualname__, "PackedFA3Attention.Config"
            )
            self.assertIs(attention.inner_attention.sharding_config, sentinel)

    def test_the_block_refuses_missing_or_malformed_offsets(self) -> None:
        """The checks run before the inner kernel, so a Flex inner builds the block on the CPU."""
        attention = derive(
            _piper_1b_model(fuse_qkv=True, shape=TINY).layers[0].attention,
            PackedGQAttention.Config,
        ).build()
        positions = positions_from_docs([[4, 4], [8]])
        x = torch.zeros(2, 8, TINY.dim)
        offsets = microbatch_offsets(positions, 2)
        for masks, error, message in (
            (None, ValueError, "--dataloader.offset-rows"),
            (offsets.long(), TypeError, "must be int32"),
            (microbatch_offsets(positions, 1), ValueError, r"must be \[1, W\]"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(error, message):
                    attention(x, masks, positions)

    def test_importing_the_override_marks_the_offsets_length_dynamic(self) -> None:
        module = importlib.import_module(FA3_OVERRIDE.rpartition(".")[0])
        entries = torch.compiler.config.dynamic_sources.split(",")
        self.assertEqual(entries.count(module.OFFSETS_LENGTH_SOURCE), 1)
        with torch.compiler.config.patch(dynamic_sources="L['x']:0"):
            module.mark_offsets_length_dynamic()
            module.mark_offsets_length_dynamic()
            self.assertEqual(
                torch.compiler.config.dynamic_sources,
                f"L['x']:0,{module.OFFSETS_LENGTH_SOURCE}",
            )

    def test_the_offsets_length_entry_matches_the_offsets_alone(self) -> None:
        from torch._dynamo.variables.builder import is_dynamic_source

        module = importlib.import_module(FA3_OVERRIDE.rpartition(".")[0])
        with torch.compiler.config.patch(dynamic_sources=module.OFFSETS_LENGTH_SOURCE):
            self.assertTrue(is_dynamic_source("L['attention_masks']", 1))
            self.assertFalse(is_dynamic_source("L['attention_masks']", 0))
            self.assertFalse(is_dynamic_source("L['x']", 1))
            self.assertFalse(is_dynamic_source("L['positions']", 1))

    def test_a_torch_without_dynamic_sources_refuses_the_override(self) -> None:
        module = importlib.import_module(FA3_OVERRIDE.rpartition(".")[0])
        with mock.patch.object(torch.compiler, "config", SimpleNamespace()):
            with self.assertRaisesRegex(RuntimeError, "no torch.compiler.config.dynamic_sources"):
                module.mark_offsets_length_dynamic()

    def test_the_fa3_forward_names_the_positional_inputs_of_the_parent(self) -> None:
        """The fork's input redistribution maps positional inputs by these names, so a ``*args`` forward loses them above one data-parallel rank."""
        from torchtitan.models.common.attention import VarlenAttention

        module = importlib.import_module(FA3_OVERRIDE.rpartition(".")[0])
        names = _positional_names(module.PackedFA3Attention.forward)
        self.assertEqual(names, ["q_BLNH", "k_BLNH", "v_BLNH"])
        self.assertEqual(names, _positional_names(VarlenAttention.forward))

    @unittest.skipIf(torch.cuda.is_available(), "a GPU may have FA3")
    def test_the_fa3_override_refuses_to_build_without_fa3(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "needs FA3"):
            build_titan_model(
                shape=TINY,
                overrides=[FA3_OVERRIDE],
                overrides_per_block=1,
                device="cpu",
                dtype=torch.float32,
            )


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.float() - b.float()).norm() / b.float().norm()).item()


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() == (9, 0),
    "needs an sm90 GPU",
)
class GpuParityTests(unittest.TestCase):
    """The packed FA3 kernel against stock FlexAttention, on the same weights and the same packed batch."""

    def _run(self, overrides: list[str]):
        model = build_titan_model(
            shape=GQA_PROBE, overrides=overrides, overrides_per_block=len(overrides)
        )
        rows = [[700, 1300, 48, 2048], [4096], [17, 4000, 79], [1, 1, 4094]]
        cpu_positions = positions_from_docs(rows)
        positions = cpu_positions.cuda()
        tokens = torch.randint(
            0, GQA_PROBE.vocab_size, positions.shape, device="cuda",
            generator=torch.Generator("cuda").manual_seed(0),
        )
        masks = (
            microbatch_offsets(cpu_positions, len(rows)).cuda()
            if overrides
            else model.get_attention_masks(positions)
        )
        logits = model(tokens, positions=positions, attention_masks=masks)
        logits.float().square().mean().backward()
        grads = {
            name: p.grad for name, p in model.named_parameters() if "attention" in name
        }
        return logits.detach(), grads

    def test_fa3_matches_flex(self) -> None:
        flex_logits, flex_grads = self._run([])
        logits, grads = self._run([FA3_OVERRIDE])
        self.assertLess(_rel(logits, flex_logits), 2e-2)
        self.assertEqual(grads.keys(), flex_grads.keys())
        for name, grad in grads.items():
            self.assertLess(_rel(grad, flex_grads[name]), 5e-2, name)


if __name__ == "__main__":
    unittest.main()
