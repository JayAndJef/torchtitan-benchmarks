"""CUDA tests that the packed FA3 override runs FA3 kernels, eager and compiled, and never a cuDNN attention kernel."""

import importlib.util
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from benchmarks.models.piper_qwen3.components.attention.packed import microbatch_offsets
from benchmarks.models.piper_qwen3.titan_model import build_titan_model
from tests.test_packed_attention import FA3_OVERRIDE, GQA_PROBE, positions_from_docs


def _fa3_status() -> str | None:
    """Why this host cannot run the tests, or None when it can."""
    if not torch.cuda.is_available():
        return "needs CUDA"
    if torch.cuda.get_device_capability() != (9, 0):
        return "needs an sm90 GPU"
    if importlib.util.find_spec("flash_attn_interface") is None:
        return "needs flash_attn_interface"
    return None


SKIP_REASON = _fa3_status()

FA3_FORWARD = "FlashAttnFwdSm90"
FA3_BACKWARD = "FlashAttnBwdSm90"
CUDNN = "cudnn"
"""A substring of the cuDNN SDPA kernel names, compared in lower case."""

ROWS = [[700, 1300, 48, 2048], [4096]]
OTHER_ROWS = [[17, 4000, 79], [1, 1, 4094]]
"""Two packed microbatches with a different offsets length each."""


def _kernels(step) -> list[str]:
    """The names of the CUDA kernels that ``step`` launches under the profiler."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as profiler:
        step()
        torch.cuda.synchronize()
    return [
        event.name
        for event in profiler.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
    ]


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class FA3KernelTests(unittest.TestCase):
    """One block of a model with the override, forward and backward."""

    @classmethod
    def setUpClass(cls) -> None:
        model = build_titan_model(shape=GQA_PROBE, overrides=[FA3_OVERRIDE], overrides_per_block=1)
        cls.block = model.layers["0"]

    def _step(self, module, rows: list[list[int]]):
        """A closure that runs one forward and backward of ``module`` on ``rows``."""
        cpu_positions = positions_from_docs(rows)
        positions = cpu_positions.cuda()
        offsets = microbatch_offsets(cpu_positions, len(rows)).cuda()
        generator = torch.Generator("cuda").manual_seed(0)
        x = torch.randn(
            (*positions.shape, GQA_PROBE.dim), device="cuda", generator=generator
        ).bfloat16()

        def step() -> None:
            self.block.zero_grad(set_to_none=True)
            module(x, offsets, positions).float().square().mean().backward()

        return step

    def _assert_fa3(self, kernels: list[str]) -> None:
        self.assertTrue(any(FA3_FORWARD in name for name in kernels), kernels)
        self.assertTrue(any(FA3_BACKWARD in name for name in kernels), kernels)
        self.assertEqual([name for name in kernels if CUDNN in name.lower()], [])

    def test_eager_runs_fa3(self) -> None:
        step = self._step(self.block, ROWS)
        step()
        self._assert_fa3(_kernels(step))

    def test_compiled_with_a_dynamic_offsets_length_runs_fa3(self) -> None:
        """The block compiles as the fork's ``apply_compile`` compiles it, and the import of the override marks the offsets length dynamic."""
        torch._dynamo.reset()
        compiled = torch.compile(self.block, fullgraph=True)
        with torch._dynamo.config.patch(capture_scalar_outputs=True, error_on_recompile=True):
            for rows in (ROWS, OTHER_ROWS):
                with self.subTest(documents=sum(len(row) for row in rows)):
                    step = self._step(compiled, rows)
                    step()
                    self._assert_fa3(_kernels(step))


if __name__ == "__main__":
    unittest.main()
