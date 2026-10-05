"""CUDA tests for the TE cuBLASLt grouped GEMM expert override: parity, custom-op checks, host syncs, compile and the config swap."""

import importlib.util
import sys
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from benchmarks.models.piper_qwen3.shape import PiperShape

TE_OVERRIDE = (
    "benchmarks.models.piper_qwen3.components.moe.te_grouped_experts."
    "te_grouped_experts"
)

TINY = PiperShape.derived(name="tiny", dim=256, n_layers=2, vocab_size=64)

E, D, H = 4, 256, 384
"""The expert count, the model width and the expert hidden width of the module tests."""

TOLERANCE = 1e-2
"""The largest relative L2 error of a bf16 result against its reference."""

SYNC_EVENTS = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize")
"""The CUDA runtime calls that block the host on the device."""


def _te_status() -> str | None:
    """Why this host cannot run the tests, or None when it can."""
    if not torch.cuda.is_available():
        return "needs CUDA"
    if torch.cuda.get_device_capability() != (9, 0):
        return "needs an sm90 GPU"
    if importlib.util.find_spec("transformer_engine") is None:
        return "needs TransformerEngine"
    import transformer_engine.pytorch  # noqa: F401
    import transformer_engine_torch as tex

    if tex.get_cublasLt_version() < 130400:
        return f"needs cuBLASLt >= 130400, has {tex.get_cublasLt_version()}"
    return None


SKIP_REASON = _te_status()


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.double(), b.double()
    return ((a - b).norm() / b.norm()).item()


def _experts(cls):
    """A ``cls`` module of E experts on the GPU with seeded bf16 weights."""
    module = cls(cls.Config(dim=D, hidden_dim=H, num_experts=E)).cuda().bfloat16()
    generator = torch.Generator("cuda").manual_seed(0)
    with torch.no_grad():
        for _, p in sorted(module.named_parameters()):
            p.copy_(torch.randn(p.shape, device="cuda", generator=generator) * 0.05)
    return module


def _inputs(splits: list[int]):
    """Rows for ``splits``, their split tensor and an output gradient, seeded."""
    rows = sum(splits)
    generator = torch.Generator("cuda").manual_seed(1)
    x = torch.randn(rows, D, device="cuda", generator=generator).bfloat16()
    dy = torch.randn(rows, D, device="cuda", generator=generator).bfloat16()
    return x, torch.tensor(splits, device="cuda", dtype=torch.int64), dy


def _run(module, x, splits, dy, dynamic_rows: bool = False):
    """The output and the gradients of x, w1, w2 and w3 after one backward."""
    module.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_()
    if dynamic_rows:
        torch._dynamo.mark_dynamic(x, 0)
    out = module(x, splits)
    out.backward(dy)
    return {
        "out": out.detach(),
        "dx": x.grad,
        "dw1": module.w1_EFD.grad,
        "dw2": module.w2_EDF.grad,
        "dw3": module.w3_EFD.grad,
    }


def _profiled(module, x, splits, dy):
    """The results of ``_run``, the profiler events and the names of the CUDA kernels it launched."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        with torch.profiler.record_function("te_step"):
            result = _run(module, x, splits, dy)
    events = prof.events()
    kernels = [e.name for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
    return result, events, kernels


def _fp64(module, x, splits: list[int], dy):
    """The same results in fp64, one expert at a time."""
    w1, w2, w3 = (
        p.detach().double().requires_grad_()
        for p in (module.w1_EFD, module.w2_EDF, module.w3_EFD)
    )
    x = x.detach().double().requires_grad_()
    parts, start = [], 0
    for e, n in enumerate(splits):
        xe = x[start : start + n]
        parts.append((F.silu(xe @ w1[e].T) * (xe @ w3[e].T)) @ w2[e].T)
        start += n
    out = torch.cat(parts)
    out.backward(dy.double())
    return {"out": out.detach(), "dx": x.grad, "dw1": w1.grad, "dw2": w2.grad, "dw3": w3.grad}


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class TEGroupedExpertsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from torchtitan.models.common.moe import GroupedExperts

        from benchmarks.models.piper_qwen3.components.moe import te_grouped_experts

        cls.ops = te_grouped_experts
        cls.reference = _experts(GroupedExperts)
        cls.te = _experts(te_grouped_experts.TEGroupedExperts)
        cls.te.load_state_dict(cls.reference.state_dict())

    def _check_te_path(self, kernels: list[str]) -> None:
        """Three forward and six backward GEMMs ran on TE's grouped path, and none on torch's CUTLASS grouped kernel."""
        self.assertEqual(sum("setup_grouped_gemm_kernel" in n for n in kernels), 9)
        self.assertEqual(sum("_ptrGroup_" in n for n in kernels), 9)
        self.assertEqual([n for n in kernels if "GroupProblemShape" in n], [])

    def _check_parity(self, splits: list[int]) -> None:
        x, split_tensor, dy = _inputs(splits)
        reference, _, reference_kernels = _profiled(self.reference, x, split_tensor, dy)
        self.assertEqual(sum("GroupProblemShape" in n for n in reference_kernels), 9)
        self.assertEqual([n for n in reference_kernels if "_ptrGroup_" in n], [])
        actual, _, kernels = _profiled(self.te, x, split_tensor, dy)
        self._check_te_path(kernels)
        exact = _fp64(self.reference, x, splits, dy)
        for name in ("out", "dx", "dw1", "dw2", "dw3"):
            with self.subTest(splits=splits, result=name):
                self.assertEqual(actual[name].shape, reference[name].shape)
                self.assertEqual(actual[name].dtype, reference[name].dtype)
                to_grouped = _rel(actual[name], reference[name])
                to_fp64 = _rel(actual[name], exact[name])
                print(
                    f"splits={splits} {name}: rel L2 to _grouped_mm {to_grouped:.2e}, "
                    f"to fp64 {to_fp64:.2e}, _grouped_mm to fp64 "
                    f"{_rel(reference[name], exact[name]):.2e}"
                )
                self.assertLessEqual(to_grouped, TOLERANCE)
                self.assertLessEqual(to_fp64, TOLERANCE)
        for e, n in enumerate(splits):
            if n == 0:
                for name in ("dw1", "dw2", "dw3"):
                    self.assertTrue(
                        bool((actual[name][e] == 0).all()),
                        f"{name} of empty expert {e} is not zero",
                    )

    def test_parity_with_empty_and_single_row_experts(self) -> None:
        self._check_parity([0, 37, 129, 1])

    def test_parity_with_every_row_in_one_expert(self) -> None:
        self._check_parity([0, 0, 167, 0])

    def test_an_empty_expert_gets_a_zero_weight_gradient_from_unwritten_memory(self) -> None:
        x, splits, _ = _inputs([0, 37, 129, 1])
        dy = torch.randn(x.shape[0], H, device="cuda").bfloat16()
        w = self.te.w1_EFD.detach()

        def nan_like(t: torch.Tensor, **kwargs) -> torch.Tensor:
            return torch.full_like(t, float("nan"), **kwargs)

        with mock.patch.object(torch, "empty_like", nan_like):
            dx, dw = self.ops.te_grouped_mm_backward(dy, x, w, splits)
        self.assertTrue(bool((dw[0] == 0).all()))
        self.assertFalse(bool(dw.isnan().any()))
        self.assertFalse(bool(dx.isnan().any()))

    def test_the_ops_refuse_operands_that_te_cannot_read(self) -> None:
        x, splits, _ = _inputs([0, 37, 129, 1])
        w = self.te.w1_EFD.detach()
        strided = w.transpose(1, 2).contiguous().transpose(1, 2)
        dy = torch.randn(x.shape[0], H, device="cuda").bfloat16()
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            self.ops.te_grouped_mm(x, strided, splits)
        with self.assertRaisesRegex(RuntimeError, "int64 splits"):
            self.ops.te_grouped_mm(x, w, splits.int())
        with self.assertRaisesRegex(RuntimeError, "got x"):
            self.ops.te_grouped_mm(x.float(), w, splits)
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            self.ops.te_grouped_mm_backward(dy, x, strided, splits)
        with self.assertRaisesRegex(RuntimeError, "got dy"):
            self.ops.te_grouped_mm_backward(dy.float(), x, w, splits)
        with self.assertRaisesRegex(RuntimeError, "got dy"):
            self.ops.te_grouped_mm_backward(dy[:-1], x, w, splits)

    def test_zero_rows(self) -> None:
        x, splits, dy = _inputs([0, 0, 0, 0])
        result = _run(self.te, x, splits, dy)
        self.assertEqual(tuple(result["out"].shape), (0, D))
        self.assertEqual(tuple(result["dx"].shape), (0, D))
        for name in ("dw1", "dw2", "dw3"):
            self.assertTrue(bool((result[name] == 0).all()), name)

    def test_opcheck(self) -> None:
        x, splits, _ = _inputs([0, 37, 129, 1])
        w = self.te.w1_EFD.detach().clone().requires_grad_()
        dy = torch.randn(x.shape[0], H, device="cuda").bfloat16()
        torch.library.opcheck(self.ops.te_grouped_mm, (x.requires_grad_(), w, splits))
        torch.library.opcheck(
            self.ops.te_grouped_mm_backward, (dy, x.detach(), w.detach(), splits)
        )

    def test_no_host_sync(self) -> None:
        x, splits, dy = _inputs([0, 37, 129, 1])
        _run(self.te, x, splits, dy)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            _, events, kernels = _profiled(self.te, x, splits, dy)
        finally:
            torch.cuda.set_sync_debug_mode(0)
        (step,) = [
            e
            for e in events
            if e.name == "te_step" and e.device_type == torch.autograd.DeviceType.CPU
        ]
        inside = [
            e.name
            for e in events
            if step.time_range.start <= e.time_range.start <= step.time_range.end
        ]
        self.assertEqual([n for n in inside if n in SYNC_EVENTS], [])
        self.assertEqual([n for n in kernels if "DtoH" in n], [])
        self._check_te_path(kernels)

    def test_a_stride_zero_output_gradient(self) -> None:
        x, splits, _ = _inputs([0, 37, 129, 1])
        torch._dynamo.reset()
        for module in (self.te, torch.compile(self.te, fullgraph=True)):
            self.te.zero_grad(set_to_none=True)
            leaf = x.detach().clone().requires_grad_()
            module(leaf, splits).sum().backward()
            actual = {
                "dx": leaf.grad,
                "dw1": self.te.w1_EFD.grad,
                "dw2": self.te.w2_EDF.grad,
                "dw3": self.te.w3_EFD.grad,
            }
            expected = _run(self.te, x, splits, torch.ones_like(x))
            for name, value in actual.items():
                with self.subTest(compiled=module is not self.te, result=name):
                    self.assertLessEqual(_rel(value, expected[name]), TOLERANCE)

    def test_compiled_with_a_dynamic_row_count(self) -> None:
        torch._dynamo.reset()
        compiled = torch.compile(self.te, fullgraph=True)
        with torch._dynamo.config.patch(error_on_recompile=True):
            for index, splits in enumerate(([0, 37, 129, 1], [5, 64, 0, 200])):
                x, split_tensor, dy = _inputs(splits)
                eager = _run(self.te, x, split_tensor, dy)
                actual = _run(compiled, x, split_tensor, dy, dynamic_rows=index == 0)
                for name, value in eager.items():
                    with self.subTest(splits=splits, result=name):
                        self.assertLessEqual(_rel(actual[name], value), TOLERANCE)

    def test_compiled_with_an_unbacked_row_count(self) -> None:
        torch._dynamo.reset()

        def step(x_full, splits):
            rows = splits.sum().item()
            torch._check(rows >= 0)
            torch._check(rows <= x_full.shape[0])
            return self.te(x_full[:rows], splits)

        x, splits, dy = _inputs([0, 37, 129, 1])
        padded = torch.cat([x, torch.zeros_like(x[:16])]).requires_grad_()
        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            out = torch.compile(step, fullgraph=True)(padded, splits)
        out.backward(dy)
        eager = _run(self.te, x, splits, dy)
        self.assertLessEqual(_rel(out.detach(), eager["out"]), TOLERANCE)
        self.assertLessEqual(_rel(padded.grad[: x.shape[0]], eager["dx"]), TOLERANCE)
        self.assertTrue(bool((padded.grad[x.shape[0] :] == 0).all()))


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class TEGroupedExpertsOverrideTests(unittest.TestCase):
    def test_the_override_swaps_each_grouped_experts_config(self) -> None:
        from torchtitan.models.common.moe import GroupedExperts

        from benchmarks.models.piper_qwen3.components.moe.te_grouped_experts import (
            TEGroupedExperts,
        )
        from benchmarks.models.piper_qwen3.titan_model import (
            _piper_1b_model,
            apply_config_overrides,
        )

        config = _piper_1b_model(fuse_qkv=True, shape=TINY)
        old = {fqn: cfg for fqn, cfg, _, _ in config.traverse(GroupedExperts.Config)}
        lines = apply_config_overrides(config, [TE_OVERRIDE], expected=TINY.n_layers)
        for line in lines:
            self.assertIn("GroupedExperts.Config -> TEGroupedExperts.Config", line)
        new = {fqn: cfg for fqn, cfg, _, _ in config.traverse(GroupedExperts.Config)}
        self.assertEqual(len(old), TINY.n_layers)
        self.assertEqual(list(new), list(old))
        for fqn, cfg in new.items():
            with self.subTest(fqn=fqn):
                self.assertIs(type(old[fqn]), GroupedExperts.Config)
                self.assertIs(type(cfg), TEGroupedExperts.Config)
                for field in ("dim", "hidden_dim", "num_experts"):
                    self.assertEqual(getattr(cfg, field), getattr(old[fqn], field), field)
                for field in ("param_init", "sharding_config"):
                    self.assertIs(getattr(cfg, field), getattr(old[fqn], field), field)

    def test_the_built_model_keeps_each_parameter(self) -> None:
        from benchmarks.models.piper_qwen3.components.moe.te_grouped_experts import (
            TEGroupedExperts,
        )
        from benchmarks.models.piper_qwen3.titan_model import build_titan_model

        stock = build_titan_model(shape=TINY)
        swapped = build_titan_model(
            shape=TINY, overrides=[TE_OVERRIDE], overrides_per_block=1
        )
        experts = [m for m in swapped.modules() if isinstance(m, TEGroupedExperts)]
        self.assertEqual(len(experts), TINY.n_layers)
        stock_params = dict(stock.named_parameters())
        swapped_params = dict(swapped.named_parameters())
        self.assertEqual(list(swapped_params), list(stock_params))
        for name, param in swapped_params.items():
            with self.subTest(parameter=name):
                self.assertEqual(param.shape, stock_params[name].shape)
                self.assertEqual(param.dtype, stock_params[name].dtype)
                self.assertTrue(torch.equal(param, stock_params[name]))


if __name__ == "__main__":
    unittest.main()
