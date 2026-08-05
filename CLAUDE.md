# torchtitan-benchmarks: Agent Guide

Out-of-tree benchmarks for the Piper Qwen3-1B TorchTitan port. Two distinct
systems live here and are operated differently:

1. **Declarative end-to-end scenarios** -- `./run_bench.sh`, driven by
   `benchmarks/`. Runs real TorchTitan training, validates, and evaluates.
2. **Standalone per-kernel microbenchmarks** -- individual scripts under
   `piper1b/*/`. Mostly hardcoded, no shared CLI. A refactor to fold these into
   the CLI is planned; until then treat them as one-off scripts.

Never present microbenchmark numbers as end-to-end results, or vice versa.

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
- Requires a driver reporting CUDA 13.0+ (`nvidia-smi` header). The cu130 wheels
  cover `sm_75` through `sm_120`: Ampere and Hopper both work unchanged.

## Repository map

| path | contents |
|---|---|
| `benchmarks/` | Scenario definitions, Click CLI, runner, validation, metrics, reporting |
| `piper1b/config_registry.py` | The `--module piper1b` config port; all registered `--config` names |
| `piper1b/rope/` | RoPE overrides and 6 standalone microbenchmarks + `te_rope_standalone.cu` |
| `piper1b/swiglu/` | Combined-SwiGLU Triton kernels, override, 1 microbenchmark |
| `piper1b/lm_head/` | Vendored TE cross-entropy, Piper-optimized CE, losses, 1 microbenchmark |
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

All four share `PIPER_1B_REGIONS`: `forward_block` and `backward_block`, each
80 invocations per window (16 layers x 5 active steps).

| scenario | arm | mechanism |
|---|---|---|
| `piper1b_rope` | `baseline` | stock `CosSinRoPE` |
| | `helion` | override `torchtitan.overrides.helion_rope.helion_cos_sin_rope` |
| | `te` | override `piper1b.rope.te_rope_override.te_rope`, needs gcc-13 |
| `piper1b_swiglu` | `baseline` | stock `GroupedExperts` |
| | `fused_grouped_experts` | override `piper1b.swiglu.combined_swiglu.piper_optimized_fused_grouped_experts` |
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
  manifest.json     # schema 4: workload, regions, arms, commands, hardware_metadata
  run_state.json    # per-arm status, attempts, evaluation status
  results.json      # schema 2: throughput, memory, region stats, significance
  <arm>.log         # training stdout+stderr
  <arm>/profiling/traces/iteration_*/rank0_trace.json.gz
  attempts/<ts>/<arm>/   # archived artifacts from a failed prior attempt
```

`manifest.json` `hardware_metadata` records `requested_gpu`, `nvidia_smi`,
`torch_version`, `torchtitan_git_rev`, `benchmarks_git_rev`. Always cite
`torchtitan_git_rev` and `torch_version` when reporting numbers.

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
7. `pooled_region_samples` structural failure -- a declared region did not match
   exactly one same-phase compiled graph with the expected invocation count.
   This means Inductor repartitioned the graph; the region mapping is invalid.

Rule 4 and rule 7 are the ones that catch silent wrongness. Never work around
them by relaxing the check.

### Resume

`run-all --resume <out_dir>` re-validates each arm against what is on disk,
skips those that already pass, archives partial artifacts under `attempts/`,
and re-runs the rest. It aborts if any of these changed since the manifest was
written: scenario, workload, selected arms, hardware label, extra TorchTitan
args, `nvidia_smi`, `torchtitan_git_rev`, `benchmarks_git_rev`. A different GPU
or a different commit will not resume -- that is intentional.

### Evaluation

Requires `baseline` among the arms. Reports:

- **stable tokens/s** -- median over steps 2..10 of each 20-step cycle, excluding
  step 1 (startup noise) and steps 11..20 (profiler overhead).
- **peak memory** over all steps.
- **per-region GPU span** stats pooled across windows, with Welch's t-test,
  Mann-Whitney U, and Cohen's d against baseline.
- loss and grad-norm trajectories, as a sanity check only.

The significance numbers are **distribution diagnostics within a single run**,
not independent repeated-run tests: invocations share steps and layer structure.
`results.json` states this in `significance_methodology`. Do not report them as
evidence that one kernel is faster than another across runs.

## Per-kernel microbenchmarks

These are standalone scripts, not part of the CLI. Run them with
`.venv/bin/python`. All require a GPU except the `analysis/` tools.

| script | measures | args |
|---|---|---|
| `piper1b/rope/benchmark.py` | TE vs Helion RoPE speed + correctness, 3 large shapes | none, all hardcoded |
| `piper1b/rope/significance.py` | Same, n=200 interleaved A/B with Welch/MWU/Wilcoxon/Cohen's d | none |
| `piper1b/rope/piper_size.py` | TE vs Helion at Piper-1B shapes, n=200 with MWU/Wilcoxon | none |
| `piper1b/rope/piper_burst.py` | Whether single-call timing is CPU-dispatch-bound (bursts of 1/4/16/64) | none |
| `piper1b/rope/accuracy_fp64.py` | Helion and TE accuracy vs fp64 ground truth, in bf16 ULPs | none |
| `piper1b/rope/ablation.py` | Where TE's bandwidth deficit comes from, via a partner-load ablation kernel | none |
| `piper1b/swiglu/benchmark.py` | Stock vs combined-layout SwiGLU Triton kernels, fwd/bwd + exactness | none |
| `piper1b/lm_head/benchmark.py` | 4 lm-head/CE implementations, compiled, fwd+bwd | argparse |
| `analysis/analyze.py` | Two-trace diff: device/host totals, per-kernel movers | 2 positional trace paths |
| `analysis/per_block.py` | Per-compiled-region GPU time, paired by size rank | 2 positional trace paths |

Only `piper1b/lm_head/benchmark.py` has real flags:

```bash
.venv/bin/python piper1b/lm_head/benchmark.py \
    --batch 4 --seq-len 1024 --dim 1024 --vocab-size 151936 --warmup 2 --iters 10
```

It prints a JSON blob to stdout; redirect it if you want to keep it. Everything
else prints tables to stdout and writes nothing.

To change what any other script measures you must **edit the source**. The
shapes live in module-level constants near the top (`SHAPES`, `NUM_ROWS`,
`HIDDEN_DIM`, `N`, `WARMUP`, or literal `run_shape(...)` calls at the bottom).

The `analysis/` tools take uncompressed Chrome traces, but runs write
`.json.gz`. Decompress first:

```bash
gunzip -c out/<...>/baseline/profiling/traces/iteration_40/rank0_trace.json.gz > /tmp/a.json
```

### Known hazards in these scripts

These are real defects to be aware of, and worth fixing in the planned CLI
refactor:

- **The `piper1b/rope/*` scripts import the wrong TorchTitan.** Each does
  `sys.path.insert(0, "/data/zejiaqi/torchtitan")`, which takes priority over
  the editable install. If that directory exists they measure against an
  unpinned checkout, not `third_party/torchtitan`. Verify with
  `python -c "import torchtitan; print(torchtitan.__file__)"` inside the script's
  import path before trusting a RoPE microbenchmark number.
- **Absolute cache paths are hardcoded** (`/data/zejiaqi/tmp/torch_extensions`
  and siblings) in the same scripts, so they will not build on another machine
  without editing or presetting `TORCH_EXTENSIONS_DIR`.
- Several rope scripts share the extension build name `te_rope_standalone` and
  build directory. A stale build from one script is reused by another.

### CUDA extension builds

`piper1b/rope/te_rope_standalone.cu` is JIT-built via
`torch.utils.cpp_extension.load` by every rope script and by
`te_rope_override.py` on import. It needs a C++20 host compiler; the stock one
usually is not. The scenario runner handles this automatically for the `te` arm
via `requires_gcc_toolset` and `BENCH_COMPILER_ENV`. For standalone scripts,
enable it yourself first:

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
