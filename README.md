# torchtitan-benchmarks

End-to-end and kernel-level implementation benchmarks under torchtitan, using
the piper Qwen3-1B model ported to torchtitan's config system. End-to-end
benchmarks are declarative scenarios: a scenario owns its workload and arms,
while the shared runner owns launch mechanics, run provenance, and validation.
Out-of-tree: needs zero edits to the torchtitan checkout (`--module` accepts a
fully qualified module path, and overrides activate by import path).

Validated against torchtitan checkout `b5eb9d92` and its `.venv`
(torch 2.14.0.dev20260729+cu130). The config port imports private torchtitan
helpers (`_build_qwen3_moe_layers` etc.), so expect breakage across torchtitan
versions -- re-check `piper1b/config_registry.py` first.

## End-to-end scenarios

Both scenarios run the same piper-1B workload (batch 4, seq len 1024,
40 steps, `--compile.enable`) and collect two profiler windows (5 warmup +
5 active steps at iterations 20 and 40). A scenario only declares its arms;
the runner turns every arm into the identical `./run_train.sh` invocation in
the torchtitan checkout, appending only that arm's `--override.imports`
value and an arm-specific `--dump-folder`:

```
./run_train.sh --module piper1b --config qwen3_piper_1b \
    --training.seq-len 1024 --training.steps 40 --training.local-batch-size 4 \
    --compile.enable --profiler.enable_profiling --profiler.profile_freq 20 \
    --profiler.profiler_active 5 --profiler.profiler_warmup 5 \
    [--override.imports <arm override>] --dump-folder <out>/<arm>
```

The `piper1b` module is imported out-of-tree: the runner puts this repo on
`PYTHONPATH`, and torchtitan's `--module` accepts a fully qualified module
path, so `piper1b/config_registry.py` registers `qwen3_piper_1b` without
touching the torchtitan checkout.

| scenario | arms | purpose |
|----------|------|---------|
| `piper1b_rope` | `baseline`, `helion`, `te` | Existing RoPE comparison. |
| `piper1b_swiglu` | `baseline`, `fused_grouped_experts` | MoE SwiGLU comparison. |

`piper1b_swiglu` intentionally activates only
`torchtitan.overrides.fused_swiglu.fused_grouped_experts`. Piper-1B contains
routed MoE experts, not dense `FeedForward` blocks, so the dense
`fused_swiglu` override would not apply. The fused arm uses one grouped gate/up
projection and torchtitan's fused SiLU-and-multiply Triton custom op.

### RoPE arms

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
# Existing RoPE comparison (default scenario).
./run_bench.sh <gpu-index>
./run_bench.sh <gpu-index> --scenario piper1b_rope --arm te

# New SwiGLU comparison: stock MoE experts vs FusedGroupedExperts.
./run_bench.sh <gpu-index> --scenario piper1b_swiglu
./run_bench.sh <gpu-index> --scenario piper1b_swiglu --arm fused_grouped_experts

# Adapt the workload to hardware capacity while preserving comparable arms.
./run_bench.sh <gpu-index> --scenario piper1b_swiglu --batch 1

# Stable hardware label for cross-machine comparisons (default: GPU name).
./run_bench.sh <gpu-index> --scenario piper1b_rope --hardware a6000-node1
./run_bench.sh --list-scenarios
```

Pass extra torchtitan arguments after `--`. The runner requires at least 40
steps because its profiler schedule collects and validates two trace windows.
`SEQ`, `STEPS`, `BATCH`, and `OUT` environment variables remain supported.

After every arm the runner validates completion (`Training completed` in the
log), the exact override application count (16 per override on piper-1B, one
per transformer block), absence of PyTorch fallbacks, and at least two
profiler trace windows before declaring the arm good.

### Output layout and manifest

```
out/<timestamp>/<scenario>/<hardware>/
  manifest.json          # run configuration and provenance
  <arm>.log              # full training log per arm
  <arm>/profiling/traces/iteration_{20,40}/rank0_trace.json.gz
```

`manifest.json` records the scenario name and description, hardware label
and metadata (nvidia-smi identity, torch version, torchtitan and benchmark
git revisions), the resolved workload, the declared compiled regions, all
scenario arms, the arms actually selected for this run, the full
`run_train.sh` command per selected arm, and any extra pass-through
torchtitan arguments.

### Analyze

```bash
/data/zejiaqi/torchtitan/.venv/bin/python analysis/compare_arms.py \
    out/<timestamp>/<scenario>/<hardware>
```

The measurement is the GPU span of whole compiled regions (`## Call
CompiledFxGraph` GPU annotations). Regions are matched structurally: each
compiled graph's direction is read from the CPU side of the trace (backward
graph calls nest inside `CompiledFunctionBackward` autograd frames), and the
transformer-block graph is the one with exactly 80 invocations per profiler
window (16 blocks x 5 active steps -- the embedding/loss-side partitions run
40 and 5 times). The analyzer validates that count for every window and
fails if a compiler partitioning change breaks the graph-to-region mapping,
then pools both windows and reports per region:
n, mean, median, standard deviation, arm-vs-baseline delta and ratio,
Welch's t-test, Mann-Whitney U, and Cohen's d, plus loss trajectories as a
sanity check.

**Interpretation rule:** these are timings of full compiled forward/backward
transformer-block regions. They are *not* timings of an individual fused
Inductor kernel -- generated kernel names are unstable across torch versions
and arms, so per-kernel attribution is intentionally out of scope. Also note
the fused/override arms change the computation (they are not bitwise
identical to baseline), so final performance claims need a longer
representative C4 convergence comparison, not just these region timings.

### Tests (CPU-only)

```bash
python3 -m unittest discover -s tests
```

Covers scenario/command/manifest construction and the trace analyzer over
synthetic profiler-trace fixtures (extraction, rank mapping, pooling,
statistics, and failure on malformed or repartitioned traces).

## Legacy kernel-level microbenchmarks (`microbench/`)

Legacy standalone RoPE tooling (TE vs Helion vs copy floor). These scripts
are kept as-is for kernel-level investigation and are **not part of the
end-to-end benchmark results above** -- they measure isolated kernels with
CUDA events, not compiled training regions. Run with the torchtitan venv
python, `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<idx>`, and
`source /opt/rh/gcc-toolset-13/enable` for the extension build:

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
