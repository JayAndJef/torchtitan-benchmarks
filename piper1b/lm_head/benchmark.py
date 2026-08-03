"""Head-only full-token piper-1B linear/cross-entropy benchmark."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn.functional as F

from piper1b.lm_head.te_cross_entropy import parallel_cross_entropy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=10)
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

    def cross_entropy_sum(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.float(), targets, reduction="sum")

    # TorchTitan compiles its standard CrossEntropyLoss leaf independently.
    compiled_cross_entropy_sum = torch.compile(cross_entropy_sum)

    def baseline() -> torch.Tensor:
        logits = F.linear(hidden, weight)
        loss = compiled_cross_entropy_sum(logits, labels) / num_tokens
        loss.backward()
        return loss

    def fused_linear_ce() -> torch.Tensor:
        options = torch.nn.LinearCrossEntropyOptions(
            batch_chunk_size=None,
            chunking_method=None,
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

    def te_fused_ce() -> torch.Tensor:
        logits = F.linear(hidden, weight).reshape(
            args.batch, args.seq_len, args.vocab_size
        )
        loss = parallel_cross_entropy(
            logits,
            labels.reshape(args.batch, args.seq_len),
            reduce_loss=False,
        ).sum() / num_tokens
        loss.backward()
        return loss

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
        benchmark("baseline", baseline),
        benchmark("fused_linear_ce", fused_linear_ce),
        benchmark("te_fused_ce", te_fused_ce),
    ]
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
