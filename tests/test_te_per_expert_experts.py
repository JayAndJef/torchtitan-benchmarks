"""CUDA tests for the TE per-expert GEMM override and the host count dispatcher: parity, custom-op checks, kernels, compile, the config swap and a 2-GPU dispatch."""

import importlib.util
import os
import socket
import sys
import tempfile
import unittest
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from benchmarks.models.piper_qwen3.shape import PiperShape

MOE = "benchmarks.models.piper_qwen3.components.moe"
EXPERTS_OVERRIDE = f"{MOE}.te_per_expert_experts.te_per_expert_experts"
DISPATCHER_OVERRIDE = f"{MOE}.host_count_dispatcher.host_count_dispatcher"

TRACE_MARKER = "engine_bench::te_per_expert_mm"
"""The trace marker of the experts scenario's per-expert arm; tests/test_runner.py ties it to the op module."""

TINY = PiperShape.derived(name="tiny", dim=256, n_layers=2, vocab_size=64)

E, D, H = 4, 256, 384
"""The expert count, the model width and the expert hidden width of the module tests."""

TOLERANCE = 1e-2
"""The largest relative L2 error of a bf16 result against its reference."""

SPLIT_SETS = {
    "empty_and_single_row": [0, 37, 129, 1],
    "balanced": [64, 64, 64, 64],
    "skewed": [3, 200, 17, 41],
    "all_but_one_empty": [0, 0, 167, 0],
}
"""The rows per expert of each parity case."""

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
    """Rows for ``splits``, the host and device count tensors and an output gradient, seeded."""
    rows = sum(splits)
    generator = torch.Generator("cuda").manual_seed(1)
    x = torch.randn(rows, D, device="cuda", generator=generator).bfloat16()
    dy = torch.randn(rows, D, device="cuda", generator=generator).bfloat16()
    host = torch.tensor(splits, dtype=torch.int64)
    return x, host, host.cuda(), dy


def _run(module, x, counts, dy):
    """The output and the gradients of x, w1, w2 and w3 after one backward."""
    module.zero_grad(set_to_none=True)
    x = x.detach().clone().requires_grad_()
    out = module(x, counts)
    out.backward(dy)
    return {
        "out": out.detach(),
        "dx": x.grad,
        "dw1": module.w1_EFD.grad,
        "dw2": module.w2_EDF.grad,
        "dw3": module.w3_EFD.grad,
    }


def _profiled(module, x, counts, dy):
    """The results of ``_run``, the profiler events and the names of the CUDA kernels it launched."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as prof:
        with torch.profiler.record_function("te_step"):
            result = _run(module, x, counts, dy)
    events = prof.events()
    kernels = [e.name for e in events if e.device_type == torch.autograd.DeviceType.CUDA]
    return result, events, kernels


def _trace_text(prof) -> str:
    """The Chrome trace of ``prof`` as text, which the e2e trace marker rule searches."""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "trace.json"
        prof.export_chrome_trace(str(path))
        return path.read_text()


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


def _gemm_kernels(kernels: list[str]) -> list[str]:
    """The GEMM kernels among ``kernels``."""
    return [n for n in kernels if "nvjet" in n or "gemm" in n.lower()]


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class TEPerExpertExpertsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        from torchtitan.models.common.moe import GroupedExperts

        from benchmarks.models.piper_qwen3.components.moe import te_per_expert_experts

        cls.ops = te_per_expert_experts
        cls.reference = _experts(GroupedExperts)
        cls.te = _experts(te_per_expert_experts.TEPerExpertExperts)
        cls.te.load_state_dict(cls.reference.state_dict())

    def _check_per_expert_path(self, kernels: list[str], splits: list[int]) -> None:
        """Each of the nine GEMMs ran as one cuBLAS kernel per non-empty expert, and no grouped kernel ran."""
        gemms = _gemm_kernels(kernels)
        print(f"splits={splits} GEMM kernels: {sorted(set(gemms))}")
        self.assertEqual(len(gemms), 9 * sum(n > 0 for n in splits), gemms)
        # cuBLAS can pick its own sm75 CUTLASS kernel for a one-row GEMM.
        self.assertEqual(
            [n for n in gemms if "nvjet" not in n and "cutlass_75_tensorop" not in n], []
        )
        self.assertTrue(any("nvjet" in n for n in gemms), gemms)
        self.assertEqual([n for n in kernels if "_ptrGroup_" in n], [])
        self.assertEqual([n for n in kernels if "GroupProblemShape" in n], [])
        self.assertEqual([n for n in kernels if "setup_grouped_gemm" in n], [])

    def _check_parity(self, splits: list[int]) -> None:
        x, host, device, dy = _inputs(splits)
        reference, _, reference_kernels = _profiled(self.reference, x, device, dy)
        self.assertEqual(sum("GroupProblemShape" in n for n in reference_kernels), 9)
        actual, _, kernels = _profiled(self.te, x, host, dy)
        self._check_per_expert_path(kernels, splits)
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

    def test_parity(self) -> None:
        for case, splits in SPLIT_SETS.items():
            with self.subTest(case=case):
                self._check_parity(splits)

    def test_an_empty_expert_gets_a_zero_weight_gradient_in_reused_garbage_memory(self) -> None:
        """Freed 0xFF bytes (bf16 NaN) back the weight gradient, and each empty expert still reads exactly zero."""
        w = self.te.w1_EFD.detach()
        for splits in ([0, 37, 129, 1], [0, 0, 167, 0], [13, 0, 0, 0]):
            with self.subTest(splits=splits):
                x, host, _, _ = _inputs(splits)
                dy = torch.randn(x.shape[0], H, device="cuda").bfloat16()
                torch.cuda.synchronize()
                garbage = [
                    torch.full((w.numel() * w.element_size(),), 0xFF, dtype=torch.uint8, device="cuda")
                    for _ in range(16)
                ]
                pointers = {g.data_ptr() for g in garbage}
                del garbage
                dx, dw = self.ops.te_per_expert_mm_backward(dy, x, w, host)
                torch.cuda.synchronize()
                self.assertIn(dw.data_ptr(), pointers, "dw did not reuse the garbage memory")
                for e, n in enumerate(splits):
                    if n == 0:
                        self.assertTrue(bool((dw[e] == 0).all()), f"dw of empty expert {e}")
                self.assertFalse(bool(dw.isnan().any()))
                self.assertFalse(bool(dx.isnan().any()))

    def test_zero_rows(self) -> None:
        x, host, _, dy = _inputs([0, 0, 0, 0])
        result = _run(self.te, x, host, dy)
        self.assertEqual(tuple(result["out"].shape), (0, D))
        self.assertEqual(tuple(result["dx"].shape), (0, D))
        for name in ("dw1", "dw2", "dw3"):
            self.assertTrue(bool((result[name] == 0).all()), name)

    def test_the_ops_refuse_operands_that_te_cannot_read(self) -> None:
        x, host, device, _ = _inputs([0, 37, 129, 1])
        w = self.te.w1_EFD.detach()
        strided = w.transpose(1, 2).contiguous().transpose(1, 2)
        dy = torch.randn(x.shape[0], H, device="cuda").bfloat16()
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            self.ops.te_per_expert_mm(x, strided, host)
        with self.assertRaisesRegex(RuntimeError, "int64 host counts"):
            self.ops.te_per_expert_mm(x, w, device)
        with self.assertRaisesRegex(RuntimeError, "int64 host counts"):
            self.ops.te_per_expert_mm(x, w, host.int())
        with self.assertRaisesRegex(RuntimeError, "got counts"):
            self.ops.te_per_expert_mm(x, w, host + 1)
        with self.assertRaisesRegex(RuntimeError, "got x"):
            self.ops.te_per_expert_mm(x.float(), w, host)
        with self.assertRaisesRegex(RuntimeError, "contiguous"):
            self.ops.te_per_expert_mm_backward(dy, x, strided, host)
        with self.assertRaisesRegex(RuntimeError, "got dy"):
            self.ops.te_per_expert_mm_backward(dy.float(), x, w, host)
        with self.assertRaisesRegex(RuntimeError, "got dy"):
            self.ops.te_per_expert_mm_backward(dy[:-1], x, w, host)

    def test_the_module_refuses_device_counts(self) -> None:
        x, _, device, _ = _inputs([0, 37, 129, 1])
        with self.assertRaisesRegex(RuntimeError, "host_count_dispatcher"):
            self.te(x, device)

    def test_opcheck(self) -> None:
        x, host, _, _ = _inputs([0, 37, 129, 1])
        w = self.te.w1_EFD.detach().clone().requires_grad_()
        dy = torch.randn(x.shape[0], H, device="cuda").bfloat16()
        torch.library.opcheck(self.ops.te_per_expert_mm, (x.requires_grad_(), w, host))
        torch.library.opcheck(
            self.ops.te_per_expert_mm_backward, (dy, x.detach(), w.detach(), host)
        )

    def test_no_host_sync(self) -> None:
        x, host, _, dy = _inputs([0, 37, 129, 1])
        _run(self.te, x, host, dy)
        torch.cuda.synchronize()
        torch.cuda.set_sync_debug_mode("error")
        try:
            _, events, kernels = _profiled(self.te, x, host, dy)
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
        self._check_per_expert_path(kernels, [0, 37, 129, 1])
        names = [e.name for e in events if e.device_type == torch.autograd.DeviceType.CPU]
        self.assertEqual(names.count(TRACE_MARKER), 3)
        self.assertEqual(names.count(f"{TRACE_MARKER}_backward"), 3)

    def test_compiled_with_an_unbacked_row_count_and_host_counts(self) -> None:
        """One fullgraph compile serves every split set: the row count is unbacked and the counts live on the host."""
        torch._dynamo.reset()

        def step(x_full, counts, total):
            rows = total.item()
            torch._check(rows >= 0)
            torch._check(rows <= x_full.shape[0])
            return self.te(x_full[:rows], counts)

        padded_rows = 320
        with torch._dynamo.config.patch(capture_scalar_outputs=True, error_on_recompile=True):
            compiled = torch.compile(step, fullgraph=True)
            for splits in ([0, 37, 129, 1], [5, 64, 0, 200], [0, 0, 167, 0]):
                with self.subTest(splits=splits):
                    x, host, _, dy = _inputs(splits)
                    padded = torch.cat(
                        [x, torch.zeros(padded_rows - x.shape[0], D, device="cuda").bfloat16()]
                    ).requires_grad_()
                    self.te.zero_grad(set_to_none=True)
                    with torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ]
                    ) as prof:
                        out = compiled(padded, host, torch.tensor(x.shape[0]))
                        out.backward(dy)
                    trace = _trace_text(prof)
                    self.assertIn(f'"{TRACE_MARKER}"', trace)
                    self.assertIn(f'"{TRACE_MARKER}_backward"', trace)
                    actual = {
                        "out": out.detach(),
                        "dx": padded.grad[: x.shape[0]],
                        "dw1": self.te.w1_EFD.grad,
                        "dw2": self.te.w2_EDF.grad,
                        "dw3": self.te.w3_EFD.grad,
                    }
                    self.assertTrue(bool((padded.grad[x.shape[0] :] == 0).all()))
                    eager = _run(self.te, x, host, dy)
                    for name, value in eager.items():
                        self.assertLessEqual(_rel(actual[name], value), TOLERANCE, name)


@unittest.skipIf(not torch.cuda.is_available(), "needs CUDA")
class HostTokenCountsTests(unittest.TestCase):
    def test_the_op_returns_the_splits_and_the_rows_of_each_local_expert(self) -> None:
        from benchmarks.models.piper_qwen3.components.moe.host_count_dispatcher import (
            host_token_counts,
        )

        local = torch.tensor([3, 0, 5, 1, 2, 2, 0, 7], device="cuda")
        received = torch.tensor([[4, 0, 9, 1], [2, 0, 0, 6]], device="cuda")
        input_splits, output_splits, rows = host_token_counts(local, received)
        for tensor in (input_splits, output_splits, rows):
            self.assertEqual(tensor.device.type, "cpu")
            self.assertEqual(tensor.dtype, torch.int64)
        self.assertEqual(input_splits.tolist(), [9, 11])
        self.assertEqual(output_splits.tolist(), [14, 8])
        self.assertEqual(rows.tolist(), [6, 0, 9, 7])
        torch.library.opcheck(host_token_counts, (local, received))
        with self.assertRaisesRegex(RuntimeError, "int64 counts"):
            host_token_counts(local.int(), received)
        with self.assertRaisesRegex(RuntimeError, "local counts"):
            host_token_counts(local[:4], received)


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
class OverrideTests(unittest.TestCase):
    def test_both_overrides_swap_each_config_once_per_layer(self) -> None:
        from torchtitan.models.common.moe import GroupedExperts
        from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher

        from benchmarks.models.piper_qwen3.components.moe.host_count_dispatcher import (
            HostCountDispatcher,
        )
        from benchmarks.models.piper_qwen3.components.moe.te_per_expert_experts import (
            TEPerExpertExperts,
        )
        from benchmarks.models.piper_qwen3.titan_model import (
            _piper_1b_model,
            apply_config_overrides,
        )

        config = _piper_1b_model(fuse_qkv=True, shape=TINY)
        old_experts = {f: c for f, c, _, _ in config.traverse(GroupedExperts.Config)}
        old_dispatchers = {
            f: c for f, c, _, _ in config.traverse(AllToAllTokenDispatcher.Config)
        }
        lines = apply_config_overrides(
            config, [DISPATCHER_OVERRIDE, EXPERTS_OVERRIDE], expected=2 * TINY.n_layers
        )
        self.assertEqual(
            sum("GroupedExperts.Config -> TEPerExpertExperts.Config" in l for l in lines),
            TINY.n_layers,
        )
        self.assertEqual(
            sum(
                "AllToAllTokenDispatcher.Config -> HostCountDispatcher.Config" in l
                for l in lines
            ),
            TINY.n_layers,
        )
        self.assertEqual(len(old_experts), TINY.n_layers)
        self.assertEqual(len(old_dispatchers), TINY.n_layers)
        new_experts = {f: c for f, c, _, _ in config.traverse(GroupedExperts.Config)}
        new_dispatchers = {
            f: c for f, c, _, _ in config.traverse(AllToAllTokenDispatcher.Config)
        }
        self.assertEqual(list(new_experts), list(old_experts))
        self.assertEqual(list(new_dispatchers), list(old_dispatchers))
        for fqn, cfg in new_experts.items():
            self.assertIs(type(cfg), TEPerExpertExperts.Config)
            for field in ("dim", "hidden_dim", "num_experts"):
                self.assertEqual(getattr(cfg, field), getattr(old_experts[fqn], field))
            for field in ("param_init", "sharding_config"):
                self.assertIs(getattr(cfg, field), getattr(old_experts[fqn], field))
        for fqn, cfg in new_dispatchers.items():
            self.assertIs(type(cfg), HostCountDispatcher.Config)
            for field in ("num_experts", "top_k"):
                self.assertEqual(getattr(cfg, field), getattr(old_dispatchers[fqn], field))

    def test_the_built_model_keeps_each_parameter(self) -> None:
        from benchmarks.models.piper_qwen3.components.moe.te_per_expert_experts import (
            TEPerExpertExperts,
        )
        from benchmarks.models.piper_qwen3.titan_model import build_titan_model

        stock = build_titan_model(shape=TINY)
        swapped = build_titan_model(
            shape=TINY,
            overrides=[DISPATCHER_OVERRIDE, EXPERTS_OVERRIDE],
            overrides_per_block=2,
        )
        experts = [m for m in swapped.modules() if isinstance(m, TEPerExpertExperts)]
        self.assertEqual(len(experts), TINY.n_layers)
        stock_params = dict(stock.named_parameters())
        swapped_params = dict(swapped.named_parameters())
        self.assertEqual(list(swapped_params), list(stock_params))
        for name, param in swapped_params.items():
            with self.subTest(parameter=name):
                self.assertEqual(param.shape, stock_params[name].shape)
                self.assertTrue(torch.equal(param, stock_params[name]))


EP = 2
"""The expert-parallel degree of the dispatcher test."""

NUM_EXPERTS, TOP_K, TOKENS = 8, 2, 96
"""The global expert count, the routed experts per token and the tokens per rank of the dispatcher test."""

EMPTY_EXPERTS = (3, 6)
"""The global experts that no rank routes to: one on each rank."""


def _dispatch_inputs(rank: int):
    """Seeded tokens, scores, expert ids and counts of one rank; no rank routes to experts 3 and 6, and rank 1 sends most rows to expert 0."""
    generator = torch.Generator("cuda").manual_seed(100 + rank)
    x = torch.randn(TOKENS, D, device="cuda", generator=generator).bfloat16()
    weights = torch.ones(NUM_EXPERTS, device="cuda")
    weights[list(EMPTY_EXPERTS)] = 0
    if rank == 1:
        weights[0] = 20
    ids = torch.multinomial(
        weights.expand(TOKENS, -1), TOP_K, replacement=False, generator=generator
    )
    scores = torch.rand(TOKENS, TOP_K, device="cuda", generator=generator)
    counts = torch.zeros(TOKENS, NUM_EXPERTS, dtype=torch.bool, device="cuda")
    counts = counts.scatter_(-1, ids, True).sum(dim=0)
    return x, scores, ids, counts


def _count_syncs(fn):
    """The result of ``fn`` and the number of blocking CUDA calls it made."""
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("warn")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn()
    finally:
        torch.cuda.set_sync_debug_mode(0)
    syncs = [w for w in caught if "synchroniz" in str(w.message)]
    return result, len(syncs)


def _dispatcher_worker(rank: int, port: int) -> None:
    """One rank of the 2-GPU dispatcher test; an assertion raises in the parent."""
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh

    from torchtitan.models.common.moe import GroupedExperts
    from torchtitan.models.common.token_dispatcher import AllToAllTokenDispatcher

    from benchmarks.models.piper_qwen3.components.moe.host_count_dispatcher import (
        HostCountDispatcher,
    )
    from benchmarks.models.piper_qwen3.components.moe.te_per_expert_experts import (
        TEPerExpertExperts,
    )

    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=EP)
    try:
        mesh = init_device_mesh("cuda", (EP,), mesh_dim_names=("ep",))
        stock = AllToAllTokenDispatcher.Config(num_experts=NUM_EXPERTS, top_k=TOP_K).build()
        host = HostCountDispatcher.Config(num_experts=NUM_EXPERTS, top_k=TOP_K).build()
        for dispatcher in (stock, host):
            dispatcher.wire_meshes(ep_mesh=mesh, tp_mesh=None)
        x, scores, ids, counts = _dispatch_inputs(rank)

        (s_routed, s_counts, _), stock_syncs = _count_syncs(
            lambda: stock.dispatch(x, scores, ids, counts)
        )
        (h_routed, h_counts, _), host_syncs = _count_syncs(
            lambda: host.dispatch(x, scores, ids, counts)
        )
        print(
            f"rank {rank}: device counts {s_counts.tolist()}, host counts "
            f"{h_counts.tolist()}, blocking calls stock {stock_syncs} host {host_syncs}",
            flush=True,
        )
        assert stock_syncs == 1, stock_syncs
        assert host_syncs == 1, host_syncs
        assert h_counts.device.type == "cpu" and h_counts.dtype == torch.int64
        assert s_counts.device.type == "cuda"
        assert torch.equal(h_counts, s_counts.cpu()), (h_counts, s_counts)
        assert torch.equal(h_routed, s_routed)
        local_experts = NUM_EXPERTS // EP
        empty = [e - rank * local_experts for e in EMPTY_EXPERTS if e // local_experts == rank]
        assert empty and all(int(h_counts[e]) == 0 for e in empty), h_counts

        # The whole MoE layer, stock against the two overrides, eager and compiled.
        reference = GroupedExperts(
            GroupedExperts.Config(dim=D, hidden_dim=H, num_experts=local_experts)
        ).cuda().bfloat16()
        generator = torch.Generator("cuda").manual_seed(7 + rank)
        with torch.no_grad():
            for _, p in sorted(reference.named_parameters()):
                p.copy_(torch.randn(p.shape, device="cuda", generator=generator) * 0.05)
        te = TEPerExpertExperts(
            TEPerExpertExperts.Config(dim=D, hidden_dim=H, num_experts=local_experts)
        ).cuda().bfloat16()
        te.load_state_dict(reference.state_dict())
        dy = torch.randn(TOKENS, D, device="cuda", generator=generator).bfloat16()

        def layer(dispatcher, experts, x_TD):
            routed, rows, metadata = dispatcher.dispatch(x_TD, scores, ids, counts)
            out = experts(routed, rows)
            return dispatcher.combine(
                out,
                metadata,
                x_TD,
                num_local_tokens_after_padding=TOKENS,
                local_seq_len_after_padding=TOKENS,
            )

        def run(fn, dispatcher, experts):
            experts.zero_grad(set_to_none=True)
            leaf = x.detach().clone().requires_grad_()
            out = fn(dispatcher, experts, leaf)
            out.backward(dy)
            return [out.detach(), leaf.grad, experts.w1_EFD.grad, experts.w2_EDF.grad, experts.w3_EFD.grad]

        expected = run(layer, stock, reference)
        torch._dynamo.reset()
        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            compiled = torch.compile(layer, fullgraph=True)
            for label, fn in (("eager", layer), ("compiled", compiled)):
                actual = run(fn, host, te)
                for name, a, b in zip(("out", "dx", "dw1", "dw2", "dw3"), actual, expected):
                    error = _rel(a, b)
                    print(f"rank {rank} {label} {name}: rel L2 to stock {error:.2e}", flush=True)
                    assert error <= TOLERANCE, (label, name, error)
                for e in range(local_experts):
                    if int(h_counts[e]) == 0:
                        for grad in actual[2:]:
                            assert bool((grad[e] == 0).all()), (label, e)
            _, compiled_syncs = _count_syncs(
                lambda: compiled(host, te, x.detach().clone().requires_grad_())
            )
            print(f"rank {rank} compiled layer forward: blocking calls {compiled_syncs}", flush=True)
            assert compiled_syncs == 1, compiled_syncs
        dist.barrier()
    finally:
        dist.destroy_process_group()


@unittest.skipIf(SKIP_REASON is not None, SKIP_REASON or "")
@unittest.skipIf(torch.cuda.device_count() < EP, f"needs {EP} GPUs")
class HostCountDispatcherTests(unittest.TestCase):
    def test_two_ranks_match_the_stock_dispatcher_with_one_blocking_copy(self) -> None:
        import torch.multiprocessing as mp

        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        mp.spawn(_dispatcher_worker, args=(port,), nprocs=EP, join=True)


if __name__ == "__main__":
    unittest.main()
