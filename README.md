# torchtitan-benchmarks

End-to-end and kernel-level implementation benchmarks under torchtitan, using
the piper Qwen3-1B model ported to torchtitan's config system. End-to-end
benchmarks are declarative scenarios: a scenario owns its workload and arms,
while the shared runner owns launch mechanics, run provenance, and validation.
Out-of-tree: needs zero edits to the torchtitan checkout (`--module` accepts a
fully qualified module path, and overrides activate by import path).

Validated against torchtitan checkout `b5eb9d92`
(torch 2.14.0.dev20260729+cu130). The config port imports private torchtitan
helpers (`_build_qwen3_moe_layers` etc.), so expect breakage across torchtitan
versions -- re-check `piper1b/config_registry.py` first.

## End-to-end scenarios

All scenarios run the same piper-1B workload (batch 4, seq len 1024,
40 steps, `--compile.enable`) and collect two profiler windows (5 warmup +
5 active steps at iterations 20 and 40). A scenario declares its arms, including
an optional arm-specific trainer config or `--override.imports` value. The runner
turns each arm into the shared `./run_train.sh` invocation in the torchtitan
checkout with an arm-specific `--dump-folder`:

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
| `piper1b_qkv` | `baseline`, `fused_qkv` | Separate Q/K/V projections versus fused QKV. |
| `piper1b_lm_head` | `baseline`, `fused_linear_ce`, `te_fused_ce`, `piper_optimized_te_ce` | Four full-token LM-head/loss paths. |

The QKV scenario uses `qwen3_piper_1b_unfused_qkv` for its baseline and the
existing `qwen3_piper_1b` config for the fused arm. Other scenarios retain the
existing fused-QKV config for every arm. The QKV workload also pins
`--debug.seed 42` so both arms start from the same initialization without
enabling slower deterministic kernels during the performance run.

The LM-head scenario pins fused QKV in every arm and changes only the final
projection/loss path:

| arm | implementation |
|-----|----------------|
| `baseline` | Piper-style full lm-head logits followed by standard cross entropy. |
| `fused_linear_ce` | PyTorch `F.linear_cross_entropy` over all 4,096 local tokens. This loss owns the lm-head through TorchTitan's `LossWithLMHead` protocol, but does not inherit from or use `ChunkedLossWrapper`. |
| `te_fused_ce` | The normal model lm-head followed directly by TransformerEngine's fused Triton cross entropy. This is a regular `BaseLoss`, like the baseline CE. |
| `piper_optimized_te_ce` | A TE-derived single-GPU kernel that combines online softmax and cross entropy, normalizes in FP32 registers, and writes the saved logits gradient directly in the logits dtype. Piper's BF16 path therefore avoids TE's intermediate statistics and token-count buffers, FP32 gradient buffer, scaling pass, and FP32-to-BF16 handoff. |

The TE arm ports the pure-Triton kernels from TransformerEngine commit
`bffde8f4a0a4eea9036dc753e28269247e5de69d` under
`LICENSE.transformer-engine`; it does not require the TransformerEngine
package. The vendored kernel file is byte-for-byte identical and the two
wrapper files differ only in their local import paths. TE fuses softmax, loss,
and logits-gradient generation, but not the lm-head linear itself. All four
arms process the same 4,096 local tokens without outer sequence chunking. The
fused-linear arm is currently a single-GPU path; TP/PP integration is deferred.
The Piper-optimized TE arm is also single-GPU and requires its internally
normalized loss to be the terminal loss passed directly to `backward()`;
arbitrary downstream loss scaling would require another gradient pass.
All four loss implementations honor TorchTitan's loss compile configuration.
The optimized kernel explicitly keeps its normalization scalar in FP32;
otherwise TorchInductor promotes the compiled per-logit multiply to FP64 on
the tested build, which substantially regresses A6000 performance.

The isolated benchmark compiles each loss wrapper independently, matching
TorchTitan's `--compile.enable` behavior. Production-shape medians from 10
measured iterations were:

| arm | A6000 ms / GiB | Blackwell ms / GiB |
|-----|-----------------|--------------------|
| Piper baseline | 37.59 / 2.93 | 16.31 / 2.98 |
| PyTorch fused linear-CE | 74.23 / 4.11 | 31.30 / 4.16 |
| TE fused CE | 53.27 / 4.95 | 23.12 / 5.00 |
| Piper-optimized TE CE | 37.74 / 2.93 | 15.65 / 2.98 |

The optimized TE arm ties baseline time on A6000 and is 4.1% faster on
Blackwell in isolation, while matching baseline memory on both. Relative to
the preceding uncompiled-wrapper measurement, optimized latency is unchanged
on A6000 (37.56 -> 37.74 ms) and improves by 9.3% on Blackwell (17.25 ->
15.65 ms); the other arms remain within 1%. The 40-step end-to-end result was:

| arm | A6000 stable tps / peak GiB | Blackwell stable tps / peak GiB |
|-----|------------------------------|---------------------------------|
| Piper baseline | 10,977.5 / 20.04 | 23,484.5 / 20.05 |
| PyTorch fused linear-CE | 9,896.0 / 21.19 | 21,558.5 / 21.21 |
| TE fused CE | 10,457.5 / 21.19 | 21,882.5 / 21.21 |
| Piper-optimized TE CE | 10,994.0 / 20.04 | 22,714.0 / 20.05 |

The optimized arm improves over reference TE by 5.1% end-to-end on A6000 and
3.8% on Blackwell. It ties baseline on A6000 and trails it by 3.3% in the
Blackwell end-to-end run. The latter is not attributable to the loss: the
unchanged forward and backward transformer-block regions were also about 3.4%
slower in the optimized arm's sequential run. The optimized trace contains one
combined softmax/CE kernel, with a median duration of about 6.31 ms on A6000
and 2.34 ms on Blackwell; there is no separate online-softmax or
gradient-scaling kernel. The optimized arm saves about 1.15 GiB end-to-end on
A6000 and 1.16 GiB on Blackwell relative to reference TE.

The optimized BF16 logits gradient is bitwise identical to the reference TE
FP32 gradient after reference normalization and BF16 conversion on both GPUs,
including ignored labels and vocabulary size 151,936. Its 40-step loss and
gradient-norm trajectory matches the reference TE arm at every printed step.

PyTorch's BF16 fused-linear scalar loss is visibly quantized. The arm remains
finite and reduces loss, but it amplifies small numerical
differences into a higher gradient-norm trajectory on this high-learning-rate
MoE workload (step 40: 3.28 A6000 / 3.11 Blackwell, versus 1.62 / 1.54 for
Piper). A production-shape one-step check found no scaling error: versus full
logits, full-token fused linear-CE had dH relative L2 error 0.41% with cosine
0.999991 and dW relative L2 error 0.037% with cosine effectively 1.0. The TE
and standard CE arms retain FP32 loss values; full-token TE closely tracks the
baseline loss and gradient norms.

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
| te       | `piper1b.rope.te_rope_override.te_rope` (this repo) | verbatim TE block rotation with a Piper BSHD entry point that gathers per-token positions, exposed via `torch.library.custom_op` |

The TE-derived entry point loads TorchTitan's int64 position for each
`[batch, sequence]` row before invoking TE's unchanged rotation routine. A
compiled packed-document regression test verifies forward and backward against
stock `CosSinRoPE` across multiple position resets on both tested GPU
architectures.

## Model: piper Qwen3-1B port

From piper's `examples/models/qwen3.py` case `'1B'`: dim 1024,
16 layers, 16 q / 8 kv heads, head_dim 64, MoE (4 experts, top_k 2,
inter_dim 3584), qk_norm, rope theta 1e6, max_seq_len 2048, vocab 151936, no
weight tying. Known deltas (identical across arms): `route_norm=True`
(torchtitan's builder hardcodes it; piper uses False); c4_test tokenizer
(vocab 2020) against the 151936-row embedding, so losses are not comparable to
real Qwen3 training. Current per-arm peak-memory results are reported above.

## Run

Install the small benchmark-only CLI and statistics dependencies into the
TorchTitan environment with `pip install -r requirements.txt`.

```bash
# List scenarios and their arms.
./run_bench.sh scenarios

# Canonical full comparison: run, validate, and evaluate every arm.
./run_bench.sh run-all <gpu-index> --scenario piper1b_rope
./run_bench.sh run-all <gpu-index> --scenario piper1b_lm_head \
    --hardware <stable-label>

# Execute only, optionally selecting one arm or adapting the workload.
./run_bench.sh run <gpu-index> --scenario piper1b_qkv --arm fused_qkv
./run_bench.sh run <gpu-index> --scenario piper1b_swiglu --batch 1

# Evaluate an existing output directory without rerunning training.
./run_bench.sh evaluate out/<timestamp>/<scenario>/<hardware>

# Resume an interrupted all-arm run. Valid arms are not rerun.
./run_bench.sh run-all <gpu-index> --resume \
    out/<timestamp>/<scenario>/<hardware>
```

The original `./run_bench.sh <gpu-index> ...` and `--list-scenarios` forms
remain compatibility aliases. Pass extra TorchTitan arguments after `--`.
The runner requires at least 40 steps because its profiler schedule collects
and validates two trace windows. `SEQ`, `STEPS`, `BATCH`, `OUT`, and
`TITAN_DIR` environment variables remain supported. Cache roots and an
optional compiler setup script can be selected with `--cache-root` and
`--compiler-env`; command help lists their environment-variable equivalents.

After every arm the runner validates completion (`Training completed` in the
log), the exact override application count (16 per override on piper-1B, one
per transformer block), absence of PyTorch fallbacks, and at least two
profiler trace windows before declaring the arm good.

### Output layout and manifest

```
out/<timestamp>/<scenario>/<hardware>/
  manifest.json          # run configuration and provenance
  run_state.json         # atomic per-arm status and attempt history
  results.json           # machine-readable metrics and significance tests
  <arm>.log              # full training log per arm
  <arm>/profiling/traces/iteration_{20,40}/rank0_trace.json.gz
  attempts/...           # preserved artifacts from retried incomplete arms
```

`manifest.json` records the scenario name and description, hardware label
and metadata (nvidia-smi identity, torch version, torchtitan and benchmark
git revisions), the resolved workload, the declared compiled regions, all
scenario arms, the arms actually selected for this run, the full
`run_train.sh` command per selected arm, and any extra pass-through
torchtitan arguments.

### Evaluate

```bash
./run_bench.sh evaluate out/<timestamp>/<scenario>/<hardware>
```

`run-all` performs this evaluation automatically. The former
`python analysis/compare_arms.py ...` command remains a compatibility entry
point. Both print a summary containing stable tokens/sec, peak allocated
memory, and forward/backward compiled-region timing for every arm. They also
print each baseline comparison with Welch's t-test, Mann-Whitney U, and
Cohen's d, and write the same data to `results.json`.

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

The analyzer also reports peak training memory and median tokens/s from the
post-compile steps before each profiler warmup window. For LM-head attribution,
use the CUDA-event full-token microbenchmark:

```bash
python piper1b/lm_head/benchmark.py
```

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

Covers the Click interface, scenario/command/manifest construction, resumable
execution, metric serialization, and the trace analyzer over synthetic
profiler-trace fixtures (extraction, phase mapping, pooling, statistics, and
failure on malformed or repartitioned traces).

## Kernel-level microbenchmarks

Standalone RoPE tooling under `piper1b/rope/` (TE vs Helion vs copy floor). These scripts
are kept as-is for kernel-level investigation and are **not part of the
end-to-end benchmark results above** -- they measure isolated kernels with
CUDA events, not compiled training regions. The TE extension requires a
C++20-capable host compiler:

- `benchmark.py` -- correctness gate + timing vs copy floor, 3 shapes
- `significance.py` -- interleaved A/B, n=200, Welch/MWU/Wilcoxon
- `piper_size.py` / `piper_burst.py` -- piper-1B shapes; burst separates CPU dispatch from kernel time
- `accuracy_fp64.py` -- fp64 ground-truth accuracy of both kernels
- `ablation.py` -- TE partner-load ablation (diagnostic, wrong-by-design output)
- `piper1b/lm_head/benchmark.py` -- full-token baseline/PyTorch-fused/TE-fused head timing and memory

## Findings so far (2026-08-03, RTX A6000 unless noted)

Standalone kernel results:

- Large shapes (e.g. q `8x4096x16x128`): Helion at 1.016x the copy floor --
  effectively optimal; TE 11.5% slower (scalar 2-byte access, shown by
  ablation). On Blackwell (from 2026-07-30 traces) Helion hits 1518 GB/s.
- Piper-1B microbatch shape (`1x1024x16/8x64`): the position-correct TE Q+K
  path takes 29.70 us on A6000 and 31.20 us on Blackwell, versus Helion's
  98.30 and 100.80 us. TE is 3.2-3.3x faster and within 4% of the copy floor.
  At the end-to-end batch-4 shape, TE remains 22% faster than Helion on A6000
  and 3.2x faster on Blackwell in isolation. In saturated 64-call bursts at
  batch 1, TE takes 12.82/12.27 us per call on A6000/Blackwell versus Helion's
  66.02/67.72 us, so the gap is not solely Python dispatch.
- Numerics: Helion and TE agree to 1 bf16 ULP on ~0.001% of elements; vs fp64
  ground truth their error stats are identical (mean 0.254 ULP). Neither is
  generally bitwise-identical to the other or to stock. The compiled packed
  reset regression shape is bitwise identical to stock in forward and backward.
- End-to-end Piper-1B does not inherit the standalone win. On A6000, stable TPS
  is 10,985.5 baseline, 10,975.5 Helion, and 10,963.0 TE. On Blackwell it is
  23,302.5 baseline, 22,690.5 Helion, and 22,498.5 TE. TE's forward block is
  3.25% slower than baseline on A6000 and 2.89% slower on Blackwell because the
  baseline rotation fuses into QK norm while both custom ops remain fusion
  barriers.
