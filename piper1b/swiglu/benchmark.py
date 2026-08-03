"""Compare TorchTitan and combined-layout SwiGLU kernels at Piper shape."""

import statistics
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from piper1b.swiglu.combined_swiglu import (
    combined_silu_and_mul_backward_kernel,
    combined_silu_and_mul_forward_kernel,
)
from torchtitan.overrides.fused_swiglu import (
    silu_and_mul_backward_kernel,
    silu_and_mul_forward_kernel,
)


NUM_ROWS = 8192
HIDDEN_DIM = 3584
NUM_ITERS = 50
NUM_WARMUP = 10


def _median_us(fn) -> float:
    for _ in range(NUM_WARMUP):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(NUM_ITERS):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000)
    return statistics.median(samples)


def main() -> None:
    generator = torch.Generator(device="cuda").manual_seed(42)
    gate_up = torch.randn(
        NUM_ROWS,
        HIDDEN_DIM * 2,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    grad_out = torch.randn(
        NUM_ROWS,
        HIDDEN_DIM,
        device="cuda",
        dtype=torch.bfloat16,
        generator=generator,
    )
    offsets = torch.tensor(
        [2048, 4096, 6144, 8192],
        device="cuda",
        dtype=torch.int32,
    )
    gate, up = gate_up.reshape(NUM_ROWS, HIDDEN_DIM, 2).unbind(-1)

    def current_forward():
        return silu_and_mul_forward_kernel(gate, up, offsets)

    def combined_forward():
        return combined_silu_and_mul_forward_kernel(gate_up, offsets)

    def current_backward():
        return silu_and_mul_backward_kernel(grad_out, gate, up, offsets)

    def combined_backward():
        return combined_silu_and_mul_backward_kernel(grad_out, gate_up, offsets)

    print(torch.cuda.get_device_name())
    for name, fn in (
        ("TorchTitan forward", current_forward),
        ("combined forward", combined_forward),
        ("TorchTitan backward", current_backward),
        ("combined backward", combined_backward),
    ):
        print(f"{name:24s} {_median_us(fn):8.2f} us")

    grad_gate, grad_up = current_backward()
    expected_grad = torch.stack((grad_gate, grad_up), dim=-1).reshape_as(gate_up)
    print("forward exact:", torch.equal(current_forward(), combined_forward()))
    print("backward exact:", torch.equal(expected_grad, combined_backward()))


if __name__ == "__main__":
    main()
