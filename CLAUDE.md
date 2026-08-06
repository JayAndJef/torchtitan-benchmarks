# torchtitan-benchmarks: Agent Guide

Out-of-tree benchmarks for the Piper Qwen3-1B TorchTitan port. One CLI,
two kinds of measurement:

1. **Declarative end-to-end scenarios** -- `./run_bench.sh run` / `run-all`,
   driven by `benchmarks/scenarios.py`. Runs real TorchTitan training,
   validates, and evaluates.
2. **Declarative kernel-isolation benchmarks** -- `./run_bench.sh
   kernel-bench`, driven by `benchmarks/kernels.py`. Times competing kernel
   implementations head-to-head on synthetic tensors at Piper-1B shapes.

Never present kernel numbers as end-to-end results, or vice versa: a kernel
that wins in isolation can be irrelevant once Inductor fuses the graph around
it.

## Environment

One environment, owned by this repo. There is no `TITAN_DIR` and no
`TITAN_PYTHON`; both were removed.

```bash
git clone --recurse-submodules <repo> && cd torchtitan-benchmarks
uv sync                      # creates .venv, installs pinned torch + submodule
```

- TorchTitan is a git submodule at `third_party/torchtitan`, installed editable.
  `benchmarks/runtime.py:18` hardcodes it as `TITAN_DIR`.
- Every command runs under `.venv/bin/python`. `run_bench.sh` execs it directly;
  `runtime.py` derives the training subprocess interpreter from `sys.executable`,
  so the CLI and training always share one environment.
- torch is pinned to an exact nightly (`2.14.0.dev20260729+cu130`). The nightly
  index retains roughly 60 days, so the pin will eventually need bumping. A bump
  changes the numbers -- rerun baselines, do not compare across it.
- Requires a driver reporting CUDA 13.0+ (`nvidia-smi` header). On drivers that
  report less (this box's 570.211.01 reports 12.8), `run_bench.sh` sources
  `cuda_compat.sh`, which stages NVIDIA's forward-compat userspace driver under
  `.cuda-compat/` (gitignored; downloaded from the rhel9 repo when absent) and
  prepends it to `LD_LIBRARY_PATH`. Manual sessions: `source ./cuda_compat.sh`.
  The cu130 wheels cover `sm_75` through `sm_120`: Ampere and Hopper both work
  unchanged.

## Repository map

| path | contents |
|---|---|
| `benchmarks/` | Click CLI plus both systems: e2e (`scenarios/runner/metrics`) and kernel (`kernels/kernel_arms/kernel_bench/kernel_runner/kernel_worker/kernel_stats/kernel_results`) |
| `piper1b/config_registry.py` | The `--module piper1b` config port; all registered `--config` names |
| `piper1b/rope/` | TE RoPE override + `te_rope_standalone.cu` |
| `piper1b/swiglu/` | Combined-SwiGLU Triton kernels and override |
| `piper1b/lm_head/` | Vendored TE cross-entropy, Piper-optimized CE, losses |
| `analysis/` | Two argv-driven trace diagnostics (`analyze.py`, `per_block.py`) |
| `tests/` | CPU + GPU unit tests |
| `third_party/torchtitan/` | Pinned submodule |
| `out/` | Run outputs (gitignored) |
| `reports/` | Local investigation notes (gitignored). Put conclusions here, not in docs. |

## End-to-end scenarios

### Commands

```bash
./run_bench.sh scenarios                    # list scenarios and arms
./run_bench.sh run <gpu> [OPTIONS] [-- TORCHTITAN_ARGS]
./run_bench.sh run-all <gpu> [OPTIONS] [-- TORCHTITAN_ARGS]
./run_bench.sh evaluate <out_dir> [--arm NAME]... [--results PATH]
```

`<gpu>` is a PCI index. The runner always sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`
and `NGPU=1`, so runs are single-GPU and the index is stable.

- `run` executes and validates only. `run-all` also evaluates and writes
  `results.json`.
- `run` accepts `--arm NAME` to execute a single arm. `run-all` does not; it
  always runs every arm in the scenario.
- `run-all` accepts `--resume <out_dir>`; `--resume` and `--out` are mutually
  exclusive.
- `run-all --all-scenarios` sweeps every scenario in sequence, sharing one
  timestamp so they group under `out/<timestamp>/`. It is **fail-fast**: the
  first failing arm aborts the sweep and later scenarios never run. It cannot be
  combined with `--scenario`, `--out`, `--resume`, or `--results` (note `--out`
  also trips on an exported `OUT`). Budget roughly 25 minutes for all 11 arms.

Shared options, with env equivalents:

| flag | env | default |
|---|---|---|
| `--scenario` | -- | `piper1b_rope` |
| `--hardware` | -- | `auto` (slugified GPU name) |
| `--out` | `OUT` | `out/<UTC timestamp>/<scenario>/<hardware>` |
| `--seq-len` | `SEQ` | workload value (1024) |
| `--steps` | `STEPS` | workload value (40) |
| `--batch` | `BATCH` | workload value (4) |
| `--cache-root` | `BENCHMARK_CACHE_ROOT` | `$TMPDIR/torchtitan-benchmarks` |
| `--compiler-env` | `BENCH_COMPILER_ENV` | `/opt/rh/gcc-toolset-13/enable` if present |

### The 40-step floor

`Workload` defaults are `profile_freq=20`, `profiler_warmup=5`,
`profiler_active=5`, `min_trace_windows=2`. `workload_with_overrides` enforces
`steps >= profile_freq * min_trace_windows`, so **40 is the minimum** and
anything less raises `ValueError`. Two profiler windows are required by both
trace validation and region pooling. Do not lower `--steps` to save time.

### Scenarios and arms

Arm names match the kernel scenarios wherever the same implementation is
measured, so `piper1b_swiglu/piper_optimized_triton` and
`swiglu/piper_optimized_triton` are the same code at two scopes. Every arm
in both registries carries a one-line `description`; `./run_bench.sh
scenarios` prints them and manifests record them.

All four share `PIPER_1B_REGIONS`: `forward_block` and `backward_block`, each
80 invocations per window (16 layers x 5 active steps).

| scenario | arm | mechanism |
|---|---|---|
| `piper1b_rope` | `baseline` | stock `CosSinRoPE` |
| | `helion` | override `torchtitan.overrides.helion_rope.helion_cos_sin_rope` |
| | `te` | override `piper1b.rope.te_rope_override.te_rope`, needs gcc-13 |
| `piper1b_swiglu` | `baseline` | stock `GroupedExperts` |
| | `piper_optimized_triton` | override `piper1b.swiglu.combined_swiglu.piper_optimized_triton_fused_grouped_experts` |
| | `piper_optimized_inductor` | override `piper1b.swiglu.combined_swiglu.piper_optimized_inductor_fused_grouped_experts` |
| `piper1b_qkv` | `baseline` | config `qwen3_piper_1b_unfused_qkv` |
| | `fused_qkv` | config `qwen3_piper_1b` |
| `piper1b_lm_head` | `baseline` | config `qwen3_piper_1b_full_logits` |
| | `fused_linear_ce` | config `qwen3_piper_1b_fused_linear_ce` |
| | `te_fused_ce` | config `qwen3_piper_1b_te_fused_ce` |
| | `piper_optimized_te_ce` | config `qwen3_piper_1b_piper_optimized_te_ce` |

`piper1b_qkv` and `piper1b_lm_head` set `seed=42` because their arms differ in
model structure; the RoPE and SwiGLU scenarios do not.

### Output layout

```
out/<timestamp>/<scenario>/<hardware>/
  manifest.json     # schema 5: workload, regions, arms, commands, hardware_metadata
  run_state.json    # per-arm status, attempts, evaluation status
  results.json      # schema 3: throughput, memory, gpu_time, region stats, significance
  <arm>.log         # training stdout+stderr
  <arm>/profiling/traces/iteration_*/rank0_trace.json.gz
  attempts/<ts>/<arm>/   # archived artifacts from a failed prior attempt
```

`manifest.json` `hardware_metadata` records `requested_gpu`, `nvidia_smi`,
`cpu_pinning`, `torch_version`, `torchtitan_git_rev`, `benchmarks_git_rev`.
Always cite `torchtitan_git_rev` and `torch_version` when reporting numbers.

### CPU pinning

The training step is host-bound at benchmark sizes, so unpinned runs measure
scheduler placement, not kernels. The runner therefore binds each training
process to the GPU's own NUMA node with `numactl --cpunodebind --membind`,
resolved from the GPU's PCI bus id via sysfs. When that cannot be resolved
(no `numactl`, unknown bus id, or the device reports no NUMA affinity) the
run proceeds unpinned and `cpu_pinning` records why. Pinned and unpinned runs
are not comparable; `--resume` refuses to mix them.

### Validation

`validate_arm` (`benchmarks/artifacts.py`) fails an arm on any of:

1. Missing `<arm>.log`, or log lacking `Training completed`.
2. `[Override]` line count != `arm.expected_override_count` (16 for override
   arms: one per transformer block).
3. A declared `override_imports` entry with no matching `[Override] <path>:` line.
4. Log contains `falling back to the PyTorch` -- an optimized kernel silently
   degraded. This fails the arm regardless of anything else.
5. Fewer than `min_trace_windows` trace files.
6. A declared `trace_kernel_markers` string absent from every trace.
7. `pooled_window_metrics` structural failure -- a declared region did not match
   exactly one same-phase compiled graph with the expected invocation count.
   This means Inductor repartitioned the graph; the region mapping is invalid.

Rule 4 and rule 7 are the ones that catch silent wrongness. Never work around
them by relaxing the check.

### Resume

`run-all --resume <out_dir>` re-validates each arm against what is on disk,
skips those that already pass, archives partial artifacts under `attempts/`,
and re-runs the rest. It aborts if any of these changed since the manifest was
written: scenario, workload, selected arms, hardware label, extra TorchTitan
args, `nvidia_smi`, `cpu_pinning`, `torchtitan_git_rev`, `benchmarks_git_rev`.
A different GPU or a different commit will not resume -- that is intentional.

### Evaluation

Requires `baseline` among the arms. Reports:

- **stable tokens/s** -- median over steps 2..10 of each 20-step cycle, excluding
  step 1 (startup noise) and steps 11..20 (profiler overhead).
- **peak memory** over all steps.
- **GPU kernel time** -- per-step summed kernel/memcpy/memset durations from the
  traces (`gpu_time` in `results.json`): total, within declared regions, the
  remainder, and the ratio vs baseline. This is the host-speed-immune metric;
  **compare kernels with it**, not with tokens/s.
- **per-region span and kernel time** -- each declared region measured two ways:
  the annotation span (first kernel to last, includes host-idle gaps) and the
  summed kernel time inside it. Span distributions carry Welch's t-test,
  Mann-Whitney U, and Cohen's d against baseline.
- **host launch latency** -- mean host-side duration of kernel-launch calls
  (runtime and driver APIs) per arm.
  The workload is host-bound at benchmark sizes, so tokens/s tracks this, not
  kernel quality. Evaluation warns when it spreads more than 1.15x across
  arms: that run's tokens/s and span comparisons are contaminated.
- loss and grad-norm trajectories, as a sanity check only.

The significance numbers are **distribution diagnostics within a single run**,
not independent repeated-run tests: invocations share steps and layer structure.
`results.json` states this in `significance_methodology`. Do not report them as
evidence that one kernel is faster than another across runs.

## Kernel-isolation benchmarks

`run_bench.sh kernel-bench` times competing implementations of one kernel
family head-to-head on synthetic tensors at Piper-1B shapes. Same CLI, same
provenance discipline, same NUMA pinning as the end-to-end runs -- but the
numbers answer a different question and **must never be presented as
end-to-end results** (a kernel that wins in isolation can be irrelevant, or
even absent, once Inductor fuses the surrounding graph).

```bash
./run_bench.sh kernel-bench <gpu> [OPTIONS]
```

| flag | default | meaning |
|---|---|---|
| `--scenario` (repeatable) | all four | subset of kernel scenarios |
| `--n` | 200 | interleaved measurement cycles |
| `--warmup` | 30 | warmup cycles per mode |
| `--burst` | off | adds the 1/4/16/64 dispatch-cost diagnostic |
| `--batch` / `--seq-len` | 4 / 1024 | `Piper1BSpec` overrides (seq <= 2048) |
| `--seed` | 0 | input generator seed |
| `--hardware` | `auto` | provenance label |
| `--out` | `out/<ts>/kernels/<scenario>/<hardware>` | single `--scenario` only |
| `--cache-root` / `--compiler-env` | as e2e | `rope` needs the compiler env |

Unlike `run-all --all-scenarios`, a failing scenario does not abort the rest;
every scenario is reported and the command exits nonzero if any failed.
Deliberately ignores the `OUT`/`SEQ`/`BATCH` env vars -- flags only, so an
e2e shell cannot leak settings into a kernel run.

### Scenarios and arms

| scenario | arms | modes | notes |
|---|---|---|---|
| `rope` | `copy_floor`, `baseline`*, `helion`, `te` | fwd, bwd | `te` needs gcc-13; GB/s and x-floor reported |
| `swiglu` | `baseline`*, `piper_optimized_triton`, `piper_optimized_inductor` | fwd, bwd, fwd+bwd | whole expert layer only; both Piper arms fuse the w13 GEMM and differ in the activation (custom Triton op vs plain ops left to Inductor) |
| `qkv` | `baseline`*, `fused_qkv` | fwd, bwd, fwd+bwd | weights transferred via the fused state-dict merge hook |
| `lm_head` | `baseline`*, `fused_linear_ce`, `te_fused_ce`, `piper_optimized_te_ce` | fwd+bwd | losses compiled; peak memory is the secondary metric |

`*` = scenario baseline. `benchmarks/kernels.py` is the registry: add an arm
by appending a `KernelArm` with a builder path, and a scenario by appending a
`KernelScenario`. Builders live in `benchmarks/kernel_arms.py` and return a
`BuiltArm` whose `calls` map a mode to a zero-argument timed closure and
whose `correctness_outputs` returns named tensors for the gates.

### Method

- Module-scope arms (all rope modules, the swiglu layer arms, both
  qkv arms) run under `torch.compile(fullgraph=True)`, because that is what
  they face end-to-end: eager isolation races custom ops against
  materialization costs Inductor deletes, which inverts verdicts (the
  swiglu combined layout wins eager, loses compiled). `copy_floor` is the
  one deliberately eager arm: a bandwidth floor, not an implementation.
  lm_head losses are built with the production
  `CompileConfig(components=["loss"])`. `KernelArm.compiled` records the
  treatment in the manifest. The worker sets
  `torch._functorch.config.donated_buffer = False`: retained-graph backward
  timing re-runs compiled backward graphs, which buffer donation forbids.
  This changes backward memory reuse, not the generated kernels.
- Arms are timed **round-robin**: one cycle runs every arm once between
  adjacent entries of a preallocated CUDA event matrix, with a single
  synchronize at the end. Drift hits all arms equally, so the per-cycle
  deltas are paired and the Welch/MWU/Wilcoxon/Cohen's d numbers **are**
  inferential for that run (unlike the e2e span diagnostics).
- Verified not to distort: interleaved and isolated timings of the same
  kernels agree within 0.7%.
- The event-timed medians are **wall time**: host dispatch and host-idle
  gaps included. For small kernels this dominates -- the rope arms are
  90%+ dispatch (device work ~11-13 us inside 131-286 us walls). Dispatch
  cost is real on this host-bound workload, but kernel-speed claims need
  profiler-summed device time or `--burst` amortization, not the wall
  median.
- Python's garbage collector is paused during the timed region. A collection
  starves the launch queue and lands as idle time inside whichever arm's
  interval is open; pausing it cut the swiglu module sd from ~63 us to
  ~1.4 us and removed every 2x outlier, medians unchanged.
- The first cycle after the warmup synchronize is discarded (empty queue,
  systematically high).
- No L2 flush: interleaving equalizes cache state across arms.
- Backward is measured separately wherever the arm exposes a backward entry
  point; module arms retain the graph and re-run `torch.autograd.backward`,
  so only backward kernels are timed. `lm_head` is fwd+bwd only because
  `FusedLinearCrossEntropyLoss` runs its backward inside `__call__`.
- Correctness runs before timing and fails the run loudly (worker exit 3).

### Choosing a correctness metric

- **`max_rel_l2`** (`||a-b|| / ||b||`) is the default and the only safe choice
  when magnitudes differ: weight gradients accumulate over thousands of rows
  and sit ~30x above activations, so one bf16 ULP there is a large absolute
  number. bf16 kernels land at ~2e-3; gates sit at 2e-2.
- **`fp64_ulp`** reports the **mean** bf16 ULP (~0.24 for RoPE) and suits
  elementwise kernels only. Never gate a reduction on max-ULP: cancellation
  drives individual dot-product outputs toward zero, and dividing their
  negligible error by that tiny magnitude reports thousands of ULPs for a
  numerically perfect kernel -- including the stock one.
- **`bitwise`** where implementations must agree exactly. Currently only
  the fused-vs-unfused QKV outputs use it, and informationally: compiled
  GEMM epilogues broke bit-identity, so the gate records equality without
  enforcing it while the rel_l2 gates still enforce closeness.

### Silent-fallback guard

`HelionCosSinRoPE` and `TECosSinRoPE` fall back to the *numerically correct*
stock path when their eligibility checks fail, so correctness gates cannot
catch a mis-timed arm. Their builders profile one call and refuse to continue
unless the arm's marker kernel (`_helion__rope_cos_sin_fwd`,
`fused_rope_forward_positions_kernel`) actually appears.

### Output layout

```
out/<timestamp>/kernels/<scenario>/<hardware>/
  manifest.json      # schema 1: spec, shapes, arms, n/warmup/seed, command, provenance
  results.json       # schema 1: per-arm per-mode summaries + raw samples, comparisons, correctness
  kernel_bench.log   # worker stdout+stderr
```

Raw per-cycle samples are kept in `results.json` so a run can be re-analyzed
without re-measuring.

### Trace diagnostics

| script | measures | args |
|---|---|---|
| `analysis/analyze.py` | Two-trace diff: device/host totals, per-kernel movers | 2 positional trace paths |
| `analysis/per_block.py` | Per-compiled-region GPU time, paired by size rank | 2 positional trace paths |

These take uncompressed Chrome traces, but runs write `.json.gz`. Decompress
first:

```bash
gunzip -c out/<...>/baseline/profiling/traces/iteration_40/rank0_trace.json.gz > /tmp/a.json
```

### CUDA extension builds

`piper1b/rope/te_rope_standalone.cu` is JIT-built via
`torch.utils.cpp_extension.load` when `te_rope_override.py` is imported. It
needs a C++20 host compiler; the stock one usually is not. Both the scenario
runner (`requires_gcc_toolset`) and `kernel-bench` handle this automatically
via `BENCH_COMPILER_ENV`, defaulting to `/opt/rh/gcc-toolset-13/enable`. To
import the override by hand, enable it yourself first:

```bash
source /opt/rh/gcc-toolset-13/enable
```

`piper1b/swiglu/` and `piper1b/lm_head/` are pure Triton/PyTorch and need no
compiler setup.

## Model config and override mechanisms

`--module piper1b` resolves through TorchTitan's config manager to
`piper1b.config_registry`, found via `PYTHONPATH` (the runner sets it to the
repo root). `--config <name>` is `getattr(config_registry, name)()`.

`qwen3_piper_1b`: dim 1024, 16 layers, 16 heads / 8 KV heads, head_dim 64, flex
attention, qk_norm; MoE on every layer with 4 experts, top_k 2, inter_dim 3584,
`load_balance_coeff=None`; RoPE theta 1e6, max_seq_len 2048; vocab 151936, no
weight tying. Trains on `c4_test` (tokenizer vocab 2020) against the full
151936-row embedding, so **losses are not comparable to real Qwen3 training** --
they are a convergence sanity check only.

An arm changes behavior one of two ways:

- **A different registered config** (`arm.config`). Used when the difference is
  structural: fused vs unfused QKV, or the loss/lm-head strategy.
- **`--override.imports <dotted.path>`** (`arm.override_imports`). Swaps config
  nodes in place after construction. Each replacement logs
  `[Override] <module>.<function>: <fqn> <Old> -> <New>`, which is exactly what
  `validate_arm` counts.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

GPU tests skip themselves when CUDA is unavailable. `test_te_rope.py`
additionally requires g++ >= 13 and JIT-builds the CUDA extension on import.
`test_lm_head_losses.py` includes a SHA-256 check that the vendored TE sources
are unmodified except for import rewrites -- if you touch
`piper1b/lm_head/te_*.py`, that test is supposed to fail.

## Bumping the TorchTitan submodule

`config_registry.py` imports private Qwen3 helpers. After any bump, verify all
of these still exist with unchanged behavior:

- `_build_qwen3_moe_layers`, `_EMBEDDING_INIT`, `_output_linear_init`,
  `_qwen3_norm` from `torchtitan.models.qwen3`
- `torchtitan.config.override` (`override`, `derive`) and the `[Override]` log
  line format that `validate_arm` regexes
- `CosSinRoPE` and `_maybe_check_max_pos` from `torchtitan.models.common.rope`
- `GroupedExperts` from `torchtitan.models.common.moe`

The kernel scenarios additionally depend on:

- `HelionCosSinRoPE` from `torchtitan.overrides.helion_rope` and the op
  `torchtitan::helion_cossin_rope_bwd`, plus the marker kernel name
  `_helion__rope_cos_sin_fwd` the fallback guard greps for
- `FusedGroupedExperts`, `silu_and_mul_forward_kernel`,
  `silu_and_mul_backward_kernel` from `torchtitan.overrides.fused_swiglu` --
  no longer benchmarked; only `tests/test_swiglu.py` imports them as the
  bitwise ground truth for the combined-layout kernels
- `QKVLinear` / `FusedQKVLinear` / `Linear` from `torchtitan.models.common`,
  and the fused module's state-dict merge hook (arms rely on
  `fused.load_state_dict(unfused.state_dict())` producing bit-identical
  weights)
- `CrossEntropyLoss` from `torchtitan.components.loss`

Also recheck the documented deltas vs Piper: the builder hardcodes
`route_norm=True` (Piper wants `False`), experts are `GroupedExperts` rather
than `BmmExperts`, and the `load_balance_coeff = None` fixup is applied
post-hoc and silently stops mattering if the builder default changes.

`piper1b_lm_head` depends on TorchTitan commit `43d328ad`, which generalized the
trainer's LM-head handoff to the `LossWithLMHead` protocol. Only
`FusedLinearCrossEntropyLoss` implements it; the TE arms do not.

## Operating rules

- Check `nvidia-smi` for a free GPU before starting. Runs are single-GPU and a
  shared GPU invalidates timings.
- Use at least 40 steps. The runner enforces this; do not try to route around it.
- Numbers are only comparable within one `torch_version` and one
  `torchtitan_git_rev`. Both are in every manifest -- check them before
  comparing against an older run in `out/`.
- Put investigation notes and hardware-specific results in `reports/`, which is
  gitignored. Keep them out of `README.md` and this file.
- After changing anything in `benchmarks/`, run the test suite. It is CPU-only
  and takes about two seconds.
