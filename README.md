# torchtitan-benchmarks: RoPE implementation benchmark (inductor vs Helion vs TransformerEngine)

End-to-end and kernel-level comparison of three RoPE implementations under
torchtitan, on the piper Qwen3-1B model ported to torchtitan's config system.
Out-of-tree: needs zero edits to the torchtitan checkout (`--module` accepts a
fully qualified module path, and overrides activate by import path).

Validated against torchtitan checkout `b5eb9d92` and its `.venv`
(torch 2.14.0.dev20260729+cu130). The config port imports private torchtitan
helpers (`_build_qwen3_moe_layers` etc.), so expect breakage across torchtitan
versions -- re-check `piper1b/config_registry.py` first.

## The three arms

| arm      | RoPE | mechanism |
|----------|------|-----------|
| baseline | stock `CosSinRoPE` | inductor fuses the rotation into neighboring kernels (QK-norm epilogue) |
| helion   | `torchtitan.overrides.helion_rope.helion_cos_sin_rope` | fused Helion kernel via `torch.library.custom_op` |
| te       | `te_rope_override.te_rope` (this repo) | verbatim port of TransformerEngine's `fused_rope_{forward,backward}_kernel` via `torch.library.custom_op` |

**TE arm caveat (benchmark-only):** TE's kernel has no per-token position
gather; the override IGNORES torchtitan's per-token `positions` (which reset at
document boundaries). Perf is representative of TE's kernel design; the loss
trajectory diverges from the other arms on packed documents. Never use it for
real training.

## Model: piper Qwen3-1B port

From `/data/zejiaqi/piper/examples/models/qwen3.py` case `'1B'`: dim 1024,
16 layers, 16 q / 8 kv heads, head_dim 64, MoE (4 experts, top_k 2,
inter_dim 3584), qk_norm, rope theta 1e6, max_seq_len 2048, vocab 151936, no
weight tying. Known deltas (identical across arms): `route_norm=True`
(torchtitan's builder hardcodes it; piper uses False); c4_test tokenizer
(vocab 2020) against the 151936-row embedding, so losses are not comparable to
real Qwen3 training. Measured peak memory at batch 4 / seq 1024: 16.85 GiB.

## Run

```bash
# Pick an IDLE GPU from nvidia-smi first. The index is PCI order.
./run_bench.sh <gpu-index>                 # all three arms, 40 steps each
./run_bench.sh <gpu-index> --arm te        # one arm
BATCH=1 ./run_bench.sh <gpu-index>         # piper per-microbatch RoPE shape (1,1024)
MODULE=qwen3 CONFIG=qwen3_debugmodel SEQ=4096 ./run_bench.sh <gpu-index>
                                           # the original 2026-07-30 debugmodel comparison
```

Traces land in `out/<timestamp>/<arm>/profiling/traces/iteration_{20,40}/`.
Profiler windows: freq 20, warmup 5, active 5 -> two 5-step windows per run.

Analyze:

```bash
/data/zejiaqi/torchtitan/.venv/bin/python analysis/compare_arms.py out/<timestamp>
```

## Kernel-level microbenchmarks (`microbench/`)

Standalone TE-vs-Helion-vs-copy-floor benchmarks (run with the torchtitan venv
python, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx>`, and
`source /opt/rh/gcc-toolset-13/enable` for the extension build):

- `bench_rope.py` -- correctness gate + timing vs copy floor, 3 shapes
- `significance.py` -- interleaved A/B, n=200, Welch/MWU/Wilcoxon
- `piper_size.py` / `piper_burst.py` -- piper-1B shapes; burst separates CPU dispatch from kernel time
- `accuracy_fp64.py` -- fp64 ground-truth accuracy of both kernels
- `ablation.py` -- TE partner-load ablation (diagnostic, wrong-by-design output)

## Findings so far (2026-07-31, RTX A6000 unless noted)

Standalone kernels (see `/data/zejiaqi/tmp/te_bench/*.log` for raw runs):

- Large shapes (e.g. q `8x4096x16x128`): Helion at 1.016x the copy floor --
  effectively optimal; TE 11.5% slower (scalar 2-byte access, shown by
  ablation). On Blackwell (from 2026-07-30 traces) Helion hits 1518 GB/s.
- piper-1B microbatch shape (`1x1024x16x64`): reverses -- TE 3.3x faster;
  Helion's shipped config buckets (tuned on GB200/head_dim-128/large tokens)
  leave it 3.4x off the floor, plus ~20 us/call CPU dispatch.
- Numerics: Helion and TE agree to 1 bf16 ULP on ~0.001% of elements; vs fp64
  ground truth their error stats are identical (mean 0.254 ULP). Neither is
  bitwise-identical to the other or to stock.
- e2e on qwen3_debugmodel (Blackwell, 2026-07-30 traces in torchtitan
  `outputs/profiling/`): the Helion override is a net LOSS under
  `--compile.enable` (+5% forward region) because the baseline gets RoPE fused
  into the QK-norm kernel for ~30 us marginal cost, while any custom op is a
  fusion barrier.

The e2e piper-1B three-arm comparison is what `run_bench.sh` produces; see
`out/<ts>/` and the session notes.
