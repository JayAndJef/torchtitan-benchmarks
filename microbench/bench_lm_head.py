"""Head-only piper-1B linear/cross-entropy benchmark and chunk tuner."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F

from piper1b.te_cross_entropy import parallel_cross_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument(
        "--fused-chunks",
        type=int,
        nargs="*",
        default=[128, 256, 512, 1024],
    )
    parser.add_argument("--include-auto", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(42)
    device = torch.device("cuda")
    num_tokens = args.batch * args.seq_len
    hidden = torch.randn(
        num_tokens,
        args.dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = (
        torch.randn(
            args.vocab_size,
            args.dim,
            device=device,
            dtype=torch.bfloat16,
        )
        / args.dim**0.5
    ).requires_grad_()
    labels = torch.randint(args.vocab_size, (num_tokens,), device=device)

    def reset_grads() -> None:
        hidden.grad = None
        weight.grad = None

    def full_logits() -> torch.Tensor:
        logits = F.linear(hidden, weight)
        loss = F.cross_entropy(logits.float(), labels, reduction="sum") / num_tokens
        loss.backward()
        return loss

    def chunked_ce() -> torch.Tensor:
        total = torch.zeros((), device=device, dtype=torch.float32)
        for hidden_chunk, label_chunk in zip(hidden.split(512), labels.split(512)):
            logits = F.linear(hidden_chunk, weight)
            loss = (
                F.cross_entropy(logits.float(), label_chunk, reduction="sum")
                / num_tokens
            )
            total += loss.detach()
            loss.backward()
        return total

    def fused_linear_ce(
        chunk_size: int | None, *, chunking_method: str | None = None
    ) -> torch.Tensor:
        options = torch.nn.LinearCrossEntropyOptions(
            batch_chunk_size=chunk_size,
            chunking_method=chunking_method,
        )
        loss = F.linear_cross_entropy(
            hidden,
            weight,
            labels,
            reduction="sum",
            options=options,
        ) / num_tokens
        loss.backward()
        return loss

    def te_fused_ce(num_chunks: int) -> torch.Tensor:
        total = torch.zeros((), device=device, dtype=torch.float32)
        hidden_3d = hidden.reshape(args.batch, args.seq_len, args.dim)
        labels_2d = labels.reshape(args.batch, args.seq_len)
        if args.seq_len % num_chunks != 0:
            raise ValueError("seq_len must be divisible by the number of TE chunks")
        chunk_size = args.seq_len // num_chunks
        for hidden_chunk, label_chunk in zip(
            hidden_3d.split(chunk_size, dim=1),
            labels_2d.split(chunk_size, dim=1),
        ):
            logits = F.linear(hidden_chunk, weight)
            loss = parallel_cross_entropy(
                logits,
                label_chunk,
                reduce_loss=False,
            ).sum() / num_tokens
            total += loss.detach()
            loss.backward()
        return total

    def benchmark(name: str, function) -> dict[str, float | int | str]:
        for _ in range(args.warmup):
            reset_grads()
            function()
        torch.cuda.synchronize()
        elapsed_ms = []
        peaks_gib = []
        output = None
        for _ in range(args.iters):
            reset_grads()
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            output = function()
            end.record()
            torch.cuda.synchronize()
            elapsed_ms.append(start.elapsed_time(end))
            peaks_gib.append(torch.cuda.max_memory_allocated() / 2**30)
        assert output is not None
        return {
            "name": name,
            "mean_ms": statistics.mean(elapsed_ms),
            "median_ms": statistics.median(elapsed_ms),
            "peak_gib": max(peaks_gib),
            "loss": float(output.detach()),
        }

    rows = [
        benchmark("baseline_full_logits", full_logits),
        benchmark("chunked", chunked_ce),
    ]
    if args.include_auto:
        rows.append(
            benchmark(
                "fused_linear_ce_auto",
                lambda: fused_linear_ce(None, chunking_method="auto"),
            )
        )
    rows.extend(
        benchmark(
            f"fused_linear_ce_{chunk_size}",
            lambda chunk_size=chunk_size: fused_linear_ce(chunk_size),
        )
        for chunk_size in args.fused_chunks
    )
    if num_tokens not in args.fused_chunks:
        rows.append(
            benchmark(
                "fused_linear_ce_full",
                lambda: fused_linear_ce(num_tokens),
            )
        )
    rows.extend(
        (
            benchmark("te_fused_ce", lambda: te_fused_ce(8)),
            benchmark("te_fused_ce_full", lambda: te_fused_ce(1)),
        )
    )
    print(
        json.dumps(
            {
                "gpu": torch.cuda.get_device_name(),
                "shape": {
                    "batch": args.batch,
                    "seq_len": args.seq_len,
                    "dim": args.dim,
                    "vocab_size": args.vocab_size,
                },
                "rows": rows,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
