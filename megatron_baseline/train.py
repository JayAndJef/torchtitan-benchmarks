"""Megatron-LM baseline training driver.

Launched by the benchmark runner as `python -m megatron_baseline.train ...
<arm_dir>`. Replicates the TorchTitan arms' workload treatment exactly (the
parity contract): same data stream (bit-identical, see data.py), same
optimizer and LR schedule, same grad clipping, same loss normalization, same
profiler schedule and trace layout, and TorchTitan-shaped step log lines so
the harness's metrics parsing works unchanged. The engines differ only in
their kernels and runtime.

Mode handling: --mode default runs Megatron's forward_backward schedule
eagerly; --mode cuda-graph wraps it in Megatron's FullCudaGraphWrapper
(whole fwd+bwd captured as one graph, optimizer eager). The driver logs a
`(mode=...)` line the validation profile matches, plus which graph
implementation actually ran.

Everything megatron-related is imported inside main() so the module itself
imports (for tests and constants) without megatron or TE installed.
"""

from __future__ import annotations

import argparse
import gc
import os
import socket
import time
from pathlib import Path

# Same allocator configuration titan's run_train.sh exports; must be set
# before torch initializes CUDA. Variable THD document counts otherwise
# fragment the caching allocator and step time degrades over the run.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

# The log-line contract with benchmarks/artifacts.py's megatron validation
# profile and benchmarks/metrics.py's STEP_METRICS regex. Keep in sync.
MODE_LINE = "Megatron-LM training loop (mode={mode}, cuda_graph_impl={impl})"
FUSION_LINE = "Megatron fusions: {state}"

# Megatron's per-layer partial-capture recipe for MoE models: the router and
# dispatch preprocessing are graphed (MoETransformerLayer's partial mode);
# expert GEMMs stay eager because their shapes are routing-dependent, and at
# this rev the local impl has no attention-scope branch for MoE layers, so
# attention stays eager too. Whole-iteration capture is architecturally
# impossible here -- the token dispatcher must D2H-copy tokens_per_expert
# for the grouped GEMM's host-side splits, which capture forbids (verified:
# capture aborts on that copy). Net: 64 graph launches/step (16 layers x
# router+preprocess x fwd+bwd) with attention/experts eager -- far thinner
# coverage than titan's whole-block graphs; documented wherever graph-mode
# numbers appear.
CUDA_GRAPH_IMPL = "local"
CUDA_GRAPH_MODULES = ("moe_router", "moe_preprocess")
TRAINING_COMPLETED = "Training completed"

# TorchTitan's flops estimate for this model, reused so tflops/mfu are
# computed on the same denominator by both engines (display only).
NUM_FLOPS_PER_TOKEN = 3_551_348_736
H100_CLASS_BF16_PEAK_FLOPS = 989e12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile-freq", type=int, required=True)
    parser.add_argument("--profiler-warmup", type=int, required=True)
    parser.add_argument("--profiler-active", type=int, required=True)
    parser.add_argument("--mode", choices=("default", "cuda-graph"), required=True)
    parser.add_argument("arm_dir", type=Path)
    return parser.parse_args(argv)


def lr_lambda_for(steps: int, warmup_steps: int = 2):
    """TorchTitan's LR profile: linear warmup over warmup_steps, one stable
    step, linear decay to zero over steps - warmup_steps."""
    decay_steps = steps - warmup_steps
    stable_steps = steps + 1 - warmup_steps - decay_steps

    def lr_lambda(scheduler_epoch: int) -> float:
        step = scheduler_epoch + 1  # 1-based training step
        if step <= warmup_steps:
            return step / warmup_steps
        if step <= warmup_steps + stable_steps:
            return 1.0
        completed = step - warmup_steps - stable_steps
        return max(0.0, 1.0 - completed / decay_steps)

    return lr_lambda


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    graphs = args.mode == "cuda-graph"
    impl = (
        f"{CUDA_GRAPH_IMPL}:{'+'.join(CUDA_GRAPH_MODULES)}" if graphs else "none"
    )
    print(MODE_LINE.format(mode=args.mode, impl=impl), flush=True)

    from megatron_baseline.location import (
        add_megatron_to_path,
        configure_te_environment,
        megatron_git_rev,
    )

    megatron_path = add_megatron_to_path()
    print(f"Megatron-LM at {megatron_path} rev {megatron_git_rev()}", flush=True)
    configure_te_environment()

    import torch
    import transformer_engine

    print(f"Transformer Engine {transformer_engine.__version__}", flush=True)

    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_free_port()))
    torch.distributed.init_process_group(backend="nccl", rank=0, world_size=1)
    torch.cuda.set_device(0)

    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    parallel_state.initialize_model_parallel()
    torch.manual_seed(args.seed)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    # Graph mode needs TE's RNG tracker (TE attention asserts on the tracker
    # type inside captured graphs); default mode keeps the stock tracker.
    model_parallel_cuda_manual_seed(
        args.seed, te_rng_tracker=graphs, use_cudagraphable_rng=graphs
    )

    from megatron_baseline.data import materialize_titan_samples, thd_batches
    from megatron_baseline.model import build_model

    samples = materialize_titan_samples(
        seq_len=args.seq_len, num_samples=args.steps * args.batch
    )
    batches = thd_batches(samples, batch_size=args.batch)
    # Static shapes across steps (required for whole-iteration graph capture,
    # kept identical in default mode for cross-mode parity): pad cu_seqlens
    # to one common length with zero-length trailing segments, and use the
    # constant seq_len as max_seqlen everywhere.
    total_tokens = args.batch * args.seq_len
    max_documents = max(batch.cu_seqlens.numel() for batch in batches)
    step_data = []
    for batch in batches:
        pad = max_documents - batch.cu_seqlens.numel()
        cu_seqlens = torch.cat(
            [
                batch.cu_seqlens,
                torch.full((pad,), total_tokens, dtype=torch.int32),
            ]
        )
        step_data.append(
            {
                "tokens": batch.tokens,
                "labels": batch.labels,
                "cu_seqlens": cu_seqlens,
            }
        )
    print(
        f"Materialized {len(step_data)} steps of c4_test batches "
        f"({args.batch}x{args.seq_len}, {max_documents - 1} max packed documents)",
        flush=True,
    )

    model = build_model(
        seq_len=args.seq_len,
        cuda_graph_impl=CUDA_GRAPH_IMPL if graphs else None,
        cuda_graph_modules=CUDA_GRAPH_MODULES if graphs else (),
    )
    if graphs:
        # The captured backward accumulates graphed-module weight grads into
        # param.main_grad (megatron.core cuda_graphs), which mcore DDP would
        # normally provide. Bare-model equivalent: persistent bf16 buffers.
        for parameter in model.parameters():
            parameter.main_grad = torch.zeros_like(parameter)
    num_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Model qwen3 piper_1B (megatron) size: {num_params:,} total parameters")

    # Megatron's real defaults live in its argparse layer, which building
    # TransformerConfig directly bypasses; running the dataclass defaults once
    # cost 11.9 GPU ms/step of unfused SwiGLU. Assert rather than trust, and
    # log the state so a regression is visible in the arm log.
    fusions = {
        name: getattr(model.config, name)
        for name in (
            "bias_activation_fusion",
            "bias_dropout_fusion",
            "cross_entropy_loss_fusion",
            "moe_permute_fusion",
            "apply_rope_fusion",
        )
    }
    # Not a boolean: which CE implementation ran is the single largest lever
    # in the loss path (te = TE's online-softmax kernel, native = megatron's
    # fp32 multi-pass one), so it goes in the log next to the flags.
    ce_impl = getattr(model.config, "cross_entropy_fusion_impl", "unknown")
    missing = sorted(name for name, on in fusions.items() if not on)
    if missing:
        raise RuntimeError(
            "megatron fusions unexpectedly disabled: "
            + ", ".join(missing)
            + " -- see megatron_baseline/model.py"
        )
    print(
        FUSION_LINE.format(
            state=" ".join(f"{k}={v}" for k, v in fusions.items())
            + f" cross_entropy_fusion_impl={ce_impl}"
        )
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=8e-4,
        betas=(0.9, 0.95),
        eps=1e-8,
        weight_decay=0.1,
        fused=True,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_for(args.steps)
    )

    def forward_step(data_iterator, model):
        data = next(data_iterator)
        tokens = data["tokens"].to("cuda", non_blocking=True)
        labels = data["labels"].to("cuda", non_blocking=True)
        cu_seqlens = data["cu_seqlens"].to("cuda", non_blocking=True)
        packed = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_kv=cu_seqlens,
            max_seqlen_q=args.seq_len,
            max_seqlen_kv=args.seq_len,
        )
        token_losses = model(
            tokens,
            position_ids=None,
            attention_mask=None,
            labels=labels,
            packed_seq_params=packed,
        )

        def loss_func(output_tensor):
            # Mean CE per token, exactly titan's sum / global_valid_tokens
            # (every token is valid: packing never pads or masks).
            loss = output_tensor.sum() / output_tensor.numel()
            return loss, {"lm loss": loss.detach()}

        return token_losses, loss_func

    forward_backward_func = get_forward_backward_func()

    def run_step(step_index: int) -> float:
        losses = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=iter([step_data[step_index]]),
            model=model,
            num_microbatches=1,
            seq_length=total_tokens,
            micro_batch_size=1,
            forward_only=False,
        )
        return float(losses[0]["lm loss"])

    parameters = list(model.parameters())

    def clip_gradients() -> float:
        # Mirrors titan's clip: pre-clip total L2 norm is what gets logged.
        grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(
            grads, norm_type=2.0, error_if_nonfinite=False, foreach=True
        )
        torch.nn.utils.clip_grads_with_norm_(
            parameters, max_norm=1.0, total_norm=total_norm, foreach=True
        )
        return float(total_norm)

    def trace_handler(prof) -> None:
        trace_dir = (
            args.arm_dir / "profiling" / "traces" / f"iteration_{prof.step_num}"
        )
        trace_dir.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_dir / "rank0_trace.json.gz"))
        print(f"Dumping profiler traces at step {prof.step_num}", flush=True)

    args.arm_dir.mkdir(parents=True, exist_ok=True)
    gc.disable()
    gc.collect()

    device_total = torch.cuda.get_device_properties(0).total_memory
    wait = args.profile_freq - args.profiler_warmup - args.profiler_active
    schedule = torch.profiler.schedule(
        wait=wait, warmup=args.profiler_warmup, active=args.profiler_active
    )
    torch.cuda.reset_peak_memory_stats()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=schedule,
        on_trace_ready=trace_handler,
        record_shapes=True,
    ) as prof:
        last_time = time.perf_counter()
        for step in range(1, args.steps + 1):
            loss = run_step(step - 1)
            merged = []
            if graphs:
                # Post-capture, graphed modules deliver weight grads via
                # main_grad and leave .grad unset (eager warmup steps still
                # use .grad). Point .grad at main_grad for those so clipping
                # and the optimizer see every gradient.
                for parameter in parameters:
                    if parameter.grad is None:
                        parameter.grad = parameter.main_grad
                        merged.append(parameter)
            grad_norm = clip_gradients()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            for parameter in merged:
                # zero_grad dropped the .grad reference; the captured graph
                # keeps accumulating into main_grad, which must be zeroed in
                # place for the next replay.
                parameter.main_grad.zero_()

            now = time.perf_counter()
            tps = round(args.batch * args.seq_len / (now - last_time))
            last_time = now
            reserved = torch.cuda.max_memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            tflops = NUM_FLOPS_PER_TOKEN * tps / 1e12
            mfu = 100 * tflops / (H100_CLASS_BF16_PEAK_FLOPS / 1e12)
            print(
                f"step: {step:2}  loss: {loss:8.5f}  "
                f"grad_norm: {grad_norm:7.4f}  "
                f"memory: {reserved / 2**30:5.2f}GiB"
                f"({reserved / device_total * 100:.2f}%)  "
                f"tps: {tps:,}  tflops: {tflops:,.2f}  mfu: {mfu:.2f}%",
                flush=True,
            )
            prof.step()

    torch.distributed.destroy_process_group()
    print(TRAINING_COMPLETED, flush=True)


if __name__ == "__main__":
    main()
