"""Megatron-LM baseline training driver.

Launched by the benchmark runner as `python -m benchmarks.e2e.megatron.train
... <arm_dir>`. Replicates the TorchTitan arms' workload treatment exactly (the
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

Pipeline handling: the driver reads RANK, WORLD_SIZE and LOCAL_RANK from the
environment, each falling back to the single-rank value, so the same module
runs under `python -m` and under `torchrun`. At --pp above 1 it builds one
stage per rank, splits each step's batch into microbatches, and lets
Megatron's own 1F1B schedule move them. **There is no data-parallel path
here yet**: the world size has to equal the pipeline degree, and a driver
that accepted more ranks than stages would give two ranks the same data and
never reduce their gradients.

At --pp 1 every branch below takes the value it always took: one pack of
`--batch` rows, one microbatch, `pre_process` and `post_process` both true,
and no collective at all.

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

# The log-line contract with benchmarks/e2e/validation.py's megatron
# validation profile and benchmarks/e2e/results.py's STEP_METRICS regex.
# Keep in sync.
MODE_LINE = "Megatron-LM training loop (mode={mode}, cuda_graph_impl={impl})"
FUSION_LINE = "Megatron fusions: {state}"
# What arm rule 12 matches. Every rank prints it, and it states what this
# process really resolved rather than what it was asked for: the degrees come
# from the environment and the microbatch count from pipeline_settings. A run
# that ignored the flags therefore cannot produce it.
PARALLELISM_LINE = (
    "Megatron-LM parallelism: dp={dp} pp={pp} schedule={schedule} "
    "microbatches={microbatches} stages={stages}"
)

# Megatron's per-layer partial-capture recipe for MoE models: the router and
# dispatch preprocessing are graphed (MoETransformerLayer's partial mode);
# expert GEMMs stay eager because their shapes are routing-dependent, and at
# this rev the local impl has no attention-scope branch for MoE layers, so
# attention stays eager too. Whole-iteration capture is architecturally
# impossible here -- the token dispatcher must D2H-copy tokens_per_expert
# for the grouped GEMM's host-side splits, which capture forbids (verified:
# capture aborts on that copy). Net: n_layers x 2 modules x fwd+bwd graph
# launches per step (64 at the normal 16-layer shape, 4 at the 1-layer huge
# shape) with attention/experts eager -- far thinner coverage than titan's
# whole-block graphs, and thinner still at one layer; documented wherever
# graph-mode numbers appear.
CUDA_GRAPH_IMPL = "local"
CUDA_GRAPH_MODULES = ("moe_router", "moe_preprocess")
TRAINING_COMPLETED = "Training completed"

# The one pipeline schedule this driver implements. Megatron-LM implements
# Interleaved1F1B as well, and benchmarks/e2e/parallelism.py registers it --
# but an interleaved run needs a list of model chunks, a list of data
# iterators and a virtual pipeline degree, none of which this driver builds.
# It is refused here by name rather than declared unsupported in the
# registry: the failure then lands on the module that owns the missing work,
# which is the pattern the kernel spans already use.
SUPPORTED_PP_SCHEDULE = "1F1B"

H100_CLASS_BF16_PEAK_FLOPS = 989e12


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    from benchmarks.models.piper_qwen3.shape import MODEL_SIZE_CHOICES

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seq-len", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--profile-freq", type=int, required=True)
    parser.add_argument("--profiler-warmup", type=int, required=True)
    parser.add_argument("--profiler-active", type=int, required=True)
    parser.add_argument("--mode", choices=("default", "cuda-graph"), required=True)
    parser.add_argument(
        "--model-size", choices=MODEL_SIZE_CHOICES, default="1b"
    )
    # The pipeline split. All three default to the single-rank run, so an
    # argv built before this axis existed still describes the same run.
    parser.add_argument("--pp", type=int, default=1)
    parser.add_argument("--pp-schedule", default=None)
    parser.add_argument("--pp-microbatch-size", type=int, default=1)
    parser.add_argument("arm_dir", type=Path)
    return parser.parse_args(argv)


def pipeline_settings(args: argparse.Namespace) -> tuple[int, int]:
    """``(rows per microbatch, microbatches per step)`` for this run.

    At ``--pp 1`` this is ``(--batch, 1)``: one pack of the whole batch and
    no split, which is what this driver has always done and what every
    published megatron number was measured on. Neither engine microbatches
    without a pipeline.

    At ``--pp`` above 1 the batch splits into ``--pp-microbatch-size`` rows
    per microbatch, which is TorchTitan's
    ``pipeline_parallel_microbatch_size`` under its own name. The two engines
    then move the same number of microbatches through the same schedule.

    The split is a repack, not a reshape of the computation: attention never
    crosses a row boundary in either arrangement, because ``thd_batches``
    marks every row's documents in ``cu_seqlens`` and TorchTitan's mask is
    block-diagonal per document. A microbatch is fewer rows in one pack.
    """
    if args.pp == 1:
        return args.batch, 1
    if args.batch % args.pp_microbatch_size:
        raise ValueError(
            f"batch {args.batch} does not divide into microbatches of "
            f"{args.pp_microbatch_size} rows"
        )
    return args.pp_microbatch_size, args.batch // args.pp_microbatch_size


def tokens_per_second(
    local_tokens_per_step: int, elapsed_seconds: float, pipeline_degree: int
) -> int:
    """Tokens per second PER DEVICE, which is the published figure.

    TorchTitan reports the same quantity: ``metrics.py`` divides a rank's own
    token count by ``non_data_parallel_size``, which is ``cp * tp * pp``. The
    ranks of one pipeline share a batch, so this rank's tokens have to be
    divided by the pipeline degree. The data-parallel degree is absent from
    the divisor because each data-parallel rank reads a batch of its own.

    **At ``pipeline_degree`` 1 the divisor is 1**, so the value is exactly
    the value this driver has always printed and no published number moves.
    """
    if pipeline_degree < 1:
        raise ValueError(f"pipeline degree {pipeline_degree} must be >= 1")
    return round(local_tokens_per_step / (elapsed_seconds * pipeline_degree))


def refuse_unsupported_pipeline(args: argparse.Namespace, world_size: int) -> None:
    """Reject a pipeline request this driver cannot honor, before it builds.

    ``benchmarks/e2e/parallelism.py`` refuses most of these for a run. They
    are restated here because ``python -m benchmarks.e2e.megatron.train`` is
    a supported entry point that holds no spec, and because a wrong answer
    here is a wrong number rather than a crash: a driver that ignored ``--pp``
    would train the whole model on every rank and publish it under a
    pipeline label.
    """
    if args.pp < 1:
        raise ValueError(f"--pp {args.pp} must be >= 1")
    if world_size != args.pp:
        raise ValueError(
            f"WORLD_SIZE {world_size} does not equal --pp {args.pp}: this "
            "driver has no data-parallel path, so every rank is a pipeline "
            "stage"
        )
    if args.pp == 1:
        if args.pp_schedule is not None:
            raise ValueError(
                f"--pp-schedule {args.pp_schedule!r} was given at --pp 1, "
                "where there is no pipeline to schedule"
            )
        if args.pp_microbatch_size != 1:
            raise ValueError(
                f"--pp-microbatch-size {args.pp_microbatch_size} was given at "
                "--pp 1, where the batch is not split"
            )
    elif args.pp_schedule != SUPPORTED_PP_SCHEDULE:
        raise ValueError(
            f"--pp-schedule {args.pp_schedule!r} is not implemented by this "
            f"driver, which runs {SUPPORTED_PP_SCHEDULE!r} only; an "
            "interleaved schedule needs a model-chunk list, a data-iterator "
            "list and a virtual pipeline degree, and none of them is built"
        )


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
    from benchmarks.models.piper_qwen3.mcore_profiles import (
        BASE,
        FUSION_FIELDS,
        declared_mismatches,
    )
    from benchmarks.models.piper_qwen3.shape import shape_by_name

    args = parse_args(argv)
    shape = shape_by_name(args.model_size)
    # torchrun sets all three; a bare `python -m` sets none, and the
    # fallbacks are then exactly the values this driver used to hardcode.
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    refuse_unsupported_pipeline(args, world_size)
    microbatch_rows, num_microbatches = pipeline_settings(args)
    # The e2e megatron arm measures megatron at its own best, which is the
    # base profile. A profile axis belongs to kernel-bench, where one arm per
    # profile is the unit; an e2e run has one megatron arm and no such axis.
    # Resolved here rather than taken from argv for that reason.
    profile = BASE
    num_flops_per_token = shape.num_flops_per_token(args.seq_len)
    graphs = args.mode == "cuda-graph"
    impl = (
        f"{CUDA_GRAPH_IMPL}:{'+'.join(CUDA_GRAPH_MODULES)}" if graphs else "none"
    )
    print(MODE_LINE.format(mode=args.mode, impl=impl), flush=True)
    print(
        PARALLELISM_LINE.format(
            # No data-parallel path here, and refuse_unsupported_pipeline has
            # already refused a world size that is not the pipeline degree.
            dp=1,
            pp=args.pp,
            schedule=args.pp_schedule,
            microbatches=num_microbatches,
            # 1F1B is the one schedule this driver runs, and it gives each
            # rank one stage.
            stages=args.pp,
        ),
        flush=True,
    )
    # The MFU/tflops denominator, printed so the report has an audit trail.
    print(
        f"num_flops_per_token: {num_flops_per_token:,} "
        f"(shape={shape.name}, seq_len={args.seq_len})",
        flush=True,
    )

    from benchmarks.models.piper_qwen3.megatron_bootstrap import (
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

    # Both are already set under torchrun, so setdefault leaves the
    # rendezvous the launcher chose and only fills in the single-rank case.
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", str(_free_port()))
    torch.distributed.init_process_group(
        backend="nccl", rank=rank, world_size=world_size
    )
    torch.cuda.set_device(local_rank)

    from megatron.core import parallel_state
    from megatron.core.packed_seq_params import PackedSeqParams
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func

    parallel_state.initialize_model_parallel(
        pipeline_model_parallel_size=args.pp
    )
    torch.manual_seed(args.seed)
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    # Graph mode needs TE's RNG tracker (TE attention asserts on the tracker
    # type inside captured graphs); default mode keeps the stock tracker.
    model_parallel_cuda_manual_seed(
        args.seed, te_rng_tracker=graphs, use_cudagraphable_rng=graphs
    )

    from benchmarks.e2e.megatron.data import materialize_titan_samples, thd_batches
    from benchmarks.models.piper_qwen3.megatron_model import build_model

    # Every rank drains the same stream and keeps the whole batch. Under a
    # pipeline split the ranks of one pipeline read the SAME tokens -- the
    # batch is not divided between them, it is passed along them -- so no
    # per-rank slice belongs here. A data-parallel degree would need one, and
    # this driver has none.
    samples = materialize_titan_samples(
        seq_len=args.seq_len, num_samples=args.steps * args.batch
    )
    batches = thd_batches(samples, batch_size=microbatch_rows)
    # Static shapes across steps (required for whole-iteration graph capture,
    # kept identical in default mode for cross-mode parity): pad cu_seqlens
    # to one common length with zero-length trailing segments, and use the
    # constant seq_len as max_seqlen everywhere. The maximum is taken over
    # every microbatch of the whole run, so one length serves them all.
    tokens_per_microbatch = microbatch_rows * args.seq_len
    max_documents = max(batch.cu_seqlens.numel() for batch in batches)
    microbatch_data = []
    for batch in batches:
        pad = max_documents - batch.cu_seqlens.numel()
        cu_seqlens = torch.cat(
            [
                batch.cu_seqlens,
                torch.full((pad,), tokens_per_microbatch, dtype=torch.int32),
            ]
        )
        microbatch_data.append(
            {
                "tokens": batch.tokens,
                "labels": batch.labels,
                "cu_seqlens": cu_seqlens,
            }
        )
    # One entry per step, each holding this step's microbatches in order. At
    # --pp 1 every entry holds exactly one, which is what the schedule was
    # handed before microbatches existed.
    step_data = [
        microbatch_data[start : start + num_microbatches]
        for start in range(0, len(microbatch_data), num_microbatches)
    ]
    # The tokens this rank reads per step, and the tokens the whole job
    # retires per step. They are equal here because the ranks of one pipeline
    # share a batch; a data-parallel degree would multiply the global figure
    # and leave the local one alone. Printed so a reader can recover the
    # job's rate from the per-device rate the step lines carry.
    local_tokens_per_step = args.batch * args.seq_len
    print(
        f"tokens_per_step_global: {local_tokens_per_step} "
        f"(dp 1 x batch {args.batch} x seq_len {args.seq_len})",
        flush=True,
    )
    print(
        f"Materialized {len(step_data)} steps of c4_test batches "
        f"({args.batch}x{args.seq_len}, {num_microbatches} microbatch(es) of "
        f"{microbatch_rows} row(s), "
        f"{max_documents - 1} max packed documents)",
        flush=True,
    )

    # This rank's own stage. Megatron divides config.num_layers by the degree
    # in get_num_layers_to_build, so the derived block spec already holds
    # this stage's layers alone; pre_process and post_process decide which
    # end modules come with them.
    model = build_model(
        seq_len=args.seq_len,
        shape=shape,
        profile=profile,
        cuda_graph_impl=CUDA_GRAPH_IMPL if graphs else None,
        cuda_graph_modules=CUDA_GRAPH_MODULES if graphs else (),
        pipeline_model_parallel_size=args.pp,
        pre_process=parallel_state.is_pipeline_first_stage(),
        post_process=parallel_state.is_pipeline_last_stage(),
    )
    if graphs:
        # The captured backward accumulates graphed-module weight grads into
        # param.main_grad (megatron.core cuda_graphs), which mcore DDP would
        # normally provide. Bare-model equivalent: persistent bf16 buffers.
        for parameter in model.parameters():
            parameter.main_grad = torch.zeros_like(parameter)
    num_params = sum(parameter.numel() for parameter in model.parameters())
    stage = parallel_state.get_pipeline_model_parallel_rank()
    expected_local = shape.stage_param_count(
        pipeline_degree=args.pp, stage_index=stage
    )
    # The "size: N total parameters" substring is validate_arm's arm rule 11
    # marker (both engines print it); keep the wording. It names the WHOLE
    # model on every rank, exactly as TorchTitan's own line does -- titan
    # prints that count before its pipeline split, so the two engines make
    # the same claim in the same words.
    print(
        f"Model qwen3 piper_1B/{shape.name} (megatron) "
        f"size: {shape.param_count:,} total parameters"
    )
    # The local half, on its own line and deliberately without the words
    # "total parameters": arm rule 11 greps for the declared total, and a
    # stage count that happened to equal another shape's total must not be
    # able to satisfy it.
    print(
        f"Model qwen3 piper_1B/{shape.name} (megatron) local size: "
        f"{num_params:,} parameters (stage {stage} of {args.pp})",
        flush=True,
    )
    # A hard failure inside the process, not just a log-grep failure: a
    # megatron shape that silently disagreed with
    # benchmarks.models.piper_qwen3.shape would
    # otherwise be published as a comparison of two different models.
    assert num_params == expected_local, (
        f"megatron built {num_params:,} parameters on stage {stage} of "
        f"{args.pp} but shape {shape.name!r} declares {expected_local:,}"
    )
    # And the sum over the world is the declared total. This is the half a
    # single rank cannot check: every stage could hold a plausible count and
    # the pipeline still hold the wrong model, or hold one layer twice.
    if world_size > 1:
        counted = torch.tensor([num_params], dtype=torch.int64, device="cuda")
        torch.distributed.all_reduce(counted)
        if int(counted.item()) != shape.param_count:
            raise RuntimeError(
                f"the {world_size} stages hold {int(counted.item()):,} "
                f"parameters between them, and shape {shape.name!r} declares "
                f"{shape.param_count:,}"
            )

    # Megatron's real defaults live in its argparse layer, which building
    # TransformerConfig directly bypasses; running the dataclass defaults once
    # cost 11.9 GPU ms/step of unfused SwiGLU. Assert rather than trust, and
    # log the state so a regression is visible in the arm log.
    #
    # The check is against what the PROFILE declares, not against a fixed
    # all-on list. Both directions are real failures now: a fusion declared on
    # that came out off is the old dataclass-default handicap, and a fusion
    # declared off that came out on is a delta that did not take -- which
    # would publish the base implementation under the variant's name.
    fusions = {
        name: getattr(model.config, name, None) for name in FUSION_FIELDS
    }
    wrong = declared_mismatches(profile, fusions)
    if wrong:
        raise RuntimeError(
            f"megatron profile {profile.name!r} did not take: "
            + "; ".join(wrong)
            + " -- see benchmarks/models/piper_qwen3/mcore_profiles.py"
        )
    print(
        FUSION_LINE.format(
            state=f"profile={profile.name} "
            + " ".join(f"{k}={v}" for k, v in fusions.items())
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
    last_stage_rank = (
        parallel_state.get_pipeline_model_parallel_last_rank()
        if world_size > 1
        else rank
    )

    def run_step(step_index: int) -> float:
        losses = forward_backward_func(
            forward_step_func=forward_step,
            data_iterator=iter(step_data[step_index]),
            model=model,
            num_microbatches=num_microbatches,
            seq_length=tokens_per_microbatch,
            micro_batch_size=1,
            forward_only=False,
        )
        # Only the last stage computes a loss; every other stage gets an
        # empty list. Megatron divides each microbatch's gradient by the
        # microbatch count itself, so the mean over the list is the batch's
        # own loss, and at one microbatch it is that microbatch's value.
        if losses:
            return sum(float(loss["lm loss"]) for loss in losses) / len(losses)
        return 0.0

    def broadcast_loss(loss: float) -> float:
        """Move the last stage's loss to every rank, so rank 0 can log it.

        The step line rank 0 prints is what ``benchmarks/e2e/results.py``
        parses, and under a pipeline split rank 0 is the first stage, which
        never sees a loss. Skipped entirely at world size 1, where the value
        is already local and the collective would add a synchronize inside
        the timed step.
        """
        if world_size == 1:
            return loss
        carrier = torch.tensor([loss], dtype=torch.float64, device="cuda")
        torch.distributed.broadcast(carrier, src=last_stage_rank)
        return float(carrier.item())

    parameters = list(model.parameters())

    def clip_gradients() -> float:
        # Mirrors titan's clip: pre-clip total L2 norm is what gets logged.
        grads = [p.grad for p in parameters if p.grad is not None]
        total_norm = torch.nn.utils.get_total_norm(
            grads, norm_type=2.0, error_if_nonfinite=False, foreach=True
        )
        if world_size > 1:
            # A gradient norm is a property of the whole model, and a
            # pipeline stage holds part of it. TorchTitan reduces the same
            # way over its pp mesh (distributed/utils.py: square, all-reduce
            # SUM, root), so without this the two engines would clip against
            # different norms and the logged value would be one stage's.
            total_norm = total_norm**2.0
            torch.distributed.all_reduce(
                total_norm,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_pipeline_model_parallel_group(),
            )
            total_norm = total_norm ** (1.0 / 2.0)
        torch.nn.utils.clip_grads_with_norm_(
            parameters, max_norm=1.0, total_norm=total_norm, foreach=True
        )
        return float(total_norm)

    def trace_handler(prof) -> None:
        trace_dir = (
            args.arm_dir / "profiling" / "traces" / f"iteration_{prof.step_num}"
        )
        # Every rank writes into one directory, under its own name, which is
        # the layout TorchTitan already writes and the one
        # trace_files_by_rank reads.
        trace_dir.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(trace_dir / f"rank{rank}_trace.json.gz"))
        print(f"Dumping profiler traces at step {prof.step_num}", flush=True)

    args.arm_dir.mkdir(parents=True, exist_ok=True)
    gc.disable()
    gc.collect()

    device_total = torch.cuda.get_device_properties(local_rank).total_memory
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
            loss = broadcast_loss(run_step(step - 1))
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
            tps = tokens_per_second(
                local_tokens_per_step, now - last_time, args.pp
            )
            last_time = now
            reserved = torch.cuda.max_memory_reserved()
            torch.cuda.reset_peak_memory_stats()
            tflops = num_flops_per_token * tps / 1e12
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
