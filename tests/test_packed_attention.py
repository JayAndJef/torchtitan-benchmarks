"""The packed-document attention overrides: the offset builder, the config swap and GPU parity with Flex."""

import dataclasses
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.models.piper_qwen3.components.attention.packed import (
    MAX_DOCUMENTS,
    PackedGQAttention,
    cu_seqlens_from_positions,
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
CUDNN_OVERRIDE = (
    "benchmarks.models.piper_qwen3.components.attention.cudnn_override."
    "packed_cudnn_attention"
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


def expected_offsets(rows: list[list[int]], max_docs: int) -> list[int]:
    starts, offset = [], 0
    for row in rows:
        for n in row:
            starts.append(offset)
            offset += n
    return starts + [offset] * (max_docs + 1 - len(starts))


class CuSeqlensTests(unittest.TestCase):
    def test_rows_with_uneven_document_counts(self) -> None:
        rows = [[3, 5], [8], [1, 1, 6], [2, 2, 2, 2]]
        cu = cu_seqlens_from_positions(positions_from_docs(rows), 12)
        self.assertEqual(cu.dtype, torch.int32)
        self.assertEqual(cu.tolist(), expected_offsets(rows, 12))

    def test_exactly_max_documents(self) -> None:
        rows = [[2, 2], [1, 3]]
        cu = cu_seqlens_from_positions(positions_from_docs(rows), 4)
        self.assertEqual(cu.tolist(), [0, 2, 4, 5, 8])

    def test_more_documents_than_the_cap_raises(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "more documents than max_documents"):
            cu_seqlens_from_positions(positions_from_docs([[1, 1, 1, 1, 4]]), 4)

    def test_a_row_that_does_not_start_a_document_raises(self) -> None:
        positions = positions_from_docs([[4, 4]])
        positions[0, :4] += 1
        with self.assertRaisesRegex(RuntimeError, "each row must start a document"):
            cu_seqlens_from_positions(positions, 4)

    def test_each_pipeline_microbatch_gets_its_own_offsets(self) -> None:
        """The pipeline splitter cuts positions along the batch, and each half rebases to 0."""
        rows = [[3, 5], [8], [5, 3], [6, 2]]
        first, second = positions_from_docs(rows).chunk(2, dim=0)
        self.assertEqual(
            cu_seqlens_from_positions(first, 4).tolist(), expected_offsets(rows[:2], 4)
        )
        self.assertEqual(
            cu_seqlens_from_positions(second, 4).tolist(), expected_offsets(rows[2:], 4)
        )

    def test_the_cap_covers_the_replay_stream(self) -> None:
        """At batch 4 and sequence length 4096 the c4_test stream holds at most 23 documents per microbatch."""
        self.assertGreaterEqual(MAX_DOCUMENTS, 23)


class OverrideTests(unittest.TestCase):
    def test_each_override_replaces_one_attention_per_block(self) -> None:
        sentinel = object()
        for target, inner in (
            (FA3_OVERRIDE, "PackedFA3Attention"),
            (CUDNN_OVERRIDE, "PackedCuDNNAttention"),
        ):
            with self.subTest(target=target):
                config = _piper_1b_model(fuse_qkv=True, shape=TINY)
                config.layers[0].attention.inner_attention.sharding_config = sentinel
                lines = apply_config_overrides(config, [target], expected=TINY.n_layers)
                for layer, line in enumerate(lines):
                    self.assertIn(f"layers.{layer}.attention", line)
                for layer in config.layers:
                    attention = layer.attention
                    self.assertIs(type(attention), PackedGQAttention.Config)
                    self.assertEqual(attention.max_documents, MAX_DOCUMENTS)
                    self.assertEqual(type(attention.inner_attention).__qualname__, f"{inner}.Config")
                    self.assertIs(attention.inner_attention.sharding_config, sentinel)

    def test_the_cudnn_override_builds_on_cpu(self) -> None:
        model = build_titan_model(
            shape=TINY,
            overrides=[CUDNN_OVERRIDE],
            overrides_per_block=1,
            device="cpu",
            dtype=torch.float32,
        )
        attention = model.layers["0"].attention
        self.assertIsInstance(attention, PackedGQAttention)
        self.assertEqual(type(attention.inner_attention).__name__, "PackedCuDNNAttention")

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
    """Each packed kernel against stock FlexAttention, on the same weights and the same packed batch."""

    def _run(self, overrides: list[str]):
        model = build_titan_model(
            shape=GQA_PROBE, overrides=overrides, overrides_per_block=len(overrides)
        )
        rows = [[700, 1300, 48, 2048], [4096], [17, 4000, 79], [1, 1, 4094]]
        positions = positions_from_docs(rows).cuda()
        tokens = torch.randint(
            0, GQA_PROBE.vocab_size, positions.shape, device="cuda",
            generator=torch.Generator("cuda").manual_seed(0),
        )
        masks = model.get_attention_masks(positions) if not overrides else None
        logits = model(tokens, positions=positions, attention_masks=masks)
        logits.float().square().mean().backward()
        grads = {
            name: p.grad for name, p in model.named_parameters() if "attention" in name
        }
        return logits.detach(), grads

    def test_fa3_and_cudnn_match_flex(self) -> None:
        flex_logits, flex_grads = self._run([])
        for target in (FA3_OVERRIDE, CUDNN_OVERRIDE):
            with self.subTest(target=target):
                logits, grads = self._run([target])
                self.assertLess(_rel(logits, flex_logits), 2e-2)
                self.assertEqual(grads.keys(), flex_grads.keys())
                for name, grad in grads.items():
                    self.assertLess(_rel(grad, flex_grads[name]), 5e-2, name)


if __name__ == "__main__":
    unittest.main()
