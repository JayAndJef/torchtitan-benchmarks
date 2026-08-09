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
./sync.sh                    # two-pass uv sync; builds the TE torch binding
```

- `./sync.sh` wraps `uv sync`: `transformer-engine[pytorch,core-cu13]` (a
  default dependency group, needed by the Megatron arm) compiles its torch
  binding without build isolation against the pinned nightly, and the wrapper
  provides the venv's bundled NVIDIA headers plus the gcc-13 toolset. A plain
  `uv sync` works only after the TE wheel is already in uv's cache.
- Three default dependency groups: `megatron` (TE), `flash3` (FlashAttention-3,
  a CUTLASS sm90a source build) and `fa4` (FlashAttention-4 plus the CuTe DSL).
  `fa4` needs no build pass and no compiler -- FA4 generates its kernels at
  compile time and ships pure-Python wheels -- but `quack-kernels` is pinned to
  0.5.3 because only 0.5.x agrees with flash-attn-4's exact
  `nvidia-cutlass-dsl==4.6.0.dev0`; 0.6.x pins 4.6.0/4.6.1/4.6.2 and will not
  resolve. Installing it also downgrades `protobuf` to 6.x (a cutlass-dsl
  constraint), which TensorBoard tolerates.
- **Any resolution change rebuilds FA3 from source**, because it is
  `no-build-isolation` and therefore builds inside the project env, so uv
  cannot reuse the cached wheel. Budget 15-40 min for adding or bumping any
  dependency, and never benchmark while it runs -- the build saturates the
  host and this workload is host-bound.
- TorchTitan is a git submodule at `third_party/torchtitan`, installed editable.
  `benchmarks/runtime.py:18` hardcodes it as `TITAN_DIR`.
- Megatron-LM is a git submodule at `third_party/Megatron-LM` (pinned
  `59b72fa57`, core 0.20.0). It is **not** pip-installed (its pyproject wants
  python >= 3.12); `megatron_baseline/location.py` puts it on `sys.path` in
  the driver process only. `MEGATRON_DIR` overrides the location for
  development checkouts; the manifest records the resolved rev.
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
| `piper1b/model_shape.py` | `PiperShape` + the `normal`/`huge` registry; both engines' single source of geometry |
| `piper1b/config_registry.py` | The `--module piper1b` config port; all registered `--config` names |
| `piper1b/rope/` | TE RoPE override + `te_rope_standalone.cu` |
| `piper1b/swiglu/` | Combined-SwiGLU Triton kernels and override |
| `piper1b/lm_head/` | Vendored TE cross-entropy, Piper-optimized CE, losses |
| `piper1b/pretokenized_data.py` | Replay dataloader: drains the c4_test pipeline at init (megatron scenario) |
| `megatron_baseline/` | Everything Megatron: location/provenance, Qwen3-1B GPTModel builder, THD data, training driver |
| `analysis/` | Three argv-driven trace diagnostics (`analyze.py`, `per_block.py`, `components.py`) |
| `tools/` | `megatron_parity_check.py` (GPU logit-parity gate between the engines, `--model-size` aware); `run_matrix.sh` (shared-box matrix supervisor) |
| `tests/` | CPU + GPU unit tests |
| `third_party/torchtitan/` | Pinned submodule (our fork) |
| `third_party/Megatron-LM/` | Pinned submodule (upstream NVIDIA, sys.path only) |
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
  also trips on an exported `OUT`). Budget roughly 45 minutes for all 20 arms
  at `--ac none`, 35 for the 15 at `--ac sac` (megatron is `none`-only).

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
| `--compile-mode` | `COMPILE_MODE` | `default` |
| `--ac` | `AC_MODE` | `sac` |
| `--model-size` | `MODEL_SIZE` | `normal` |

### Compile modes

`--compile-mode` picks the compile treatment for the whole run -- every arm
in it, any scenario. E2e only; `kernel-bench` has no such flag and always
compiles at the default mode. Since manifest schema 8 the axis is
engine-neutral and has exactly two values:

| mode | TorchTitan arms get | megatron arm gets |
|---|---|---|
| `default` | per-block `torch.compile`, mode default | TE modules uncompiled and uncaptured (its `@jit_fuser` regions still compile) |
| `cuda-graph` | per-block `torch.compile(mode="reduce-overhead")` | Megatron's local per-layer partial graphs |

`cuda-graph` is the renamed `reduce-overhead` (schema <= 7 manifests record
the torch-level name); the two max-autotune modes were **removed** after the
full matrix showed them to be GPU-time regressions at these shapes
(`reports/20260807-mode-matrix-plain-bf16.md`). `TORCH_COMPILE_MODE` in
`benchmarks/artifacts.py` maps the harness name to the `--compile.mode`
value delivered to the fork, which applies it to each block's
`torch.compile` (`CompileConfig.mode`, applied in `distributed/compile.py`).
Per-block scope is the whole point: a global `torch._inductor.config`
mutation would also reach every other `torch.compile` in the process, and
one of them -- `attention._compiled_create_block_mask` -- returns a
BlockMask that is built once per step and read by all 16 blocks. Capturing
that hands the model tensors a later replay overwrites, and training dies
with "accessing tensor output of CUDAGraphs that has been overwritten". Do
not reintroduce a global.

`apply_compile` logs `Compiling each TransformerBlock with torch.compile
(mode=<torch-level mode>)`, and validation requires the line to name the
requested mode (rules 8 and 9 below), so a mode that silently failed to
apply cannot be reported as a measurement of that mode. The megatron arm
logs its own `Megatron-LM training loop (mode=...)` line, matched by its
validation profile.

Under `cuda-graph`, titan blocks capture cleanly: plain bf16 modules with no
FSDP wrapper (see the execution-model note below), one-time captures (32
recordings at warmup, 16 forward + 16 backward), steady state 32 graph
replays per step. Rules 7 and 9 remain the arbiters -- if either fails on a
cudagraph run, something real regressed (historically: an input mutation
blocking forward capture, or drifting static-input addresses forcing
per-step re-capture). Note that `cudaGraphLaunch` is not counted as a kernel
launch by `gpu_time`, so a cudagraph arm reports a much smaller
`launch_count` and evaluation may warn about launch-latency spread.

### AC modes

`--ac` picks the activation-checkpointing treatment for the whole run:
`sac` (TorchTitan's per-op SelectiveAC -- the historical treatment, implied
by schema <= 7 manifests) or `none` (checkpointing disabled, delivered as
the trailing tyro subcommand token `activation-checkpoint:none`). Validation
requires the `Applied SelectiveAC` log line to match the requested mode.
Scenarios may restrict the axis via `Scenario.supported_ac_modes`:
`piper1b_megatron` supports only `none` (Megatron's recompute options are
not parity with per-op SAC, and the megatron arm itself always runs without
recompute -- `--ac` never affects it). `run-all --all-scenarios` skips
unsupported scenario x ac combinations; a direct `--scenario` request
errors.

**Numbers are only comparable within one `compile_mode` and one `ac_mode`.**
Cite both alongside `torch_version` and `torchtitan_git_rev`; the manifest
records them and `--resume` refuses to mix either. Measured across the full
2x2 matrix: `--ac none` cuts titan GPU kernel time ~15% (SAC's recompute is
pure GPU cost at these sizes) for ~2.5 GiB more peak memory -- see
`reports/20260807/compile-ac-matrix.md`.

### Model sizes

`--model-size` is the **third global run axis**, exactly parallel to
`--compile-mode` and `--ac`: one shape for every arm in the run, recorded in
the manifest, gated by `--resume`, and a hard comparability boundary. The
shapes live in `piper1b/model_shape.py` as frozen `PiperShape` dataclasses
and are the single source of truth for *both* engines --
`piper1b/config_registry.py` and `megatron_baseline/model.py` build from the
same object, so a size cannot drift between them. That module imports
nothing but `dataclasses`.

| | `normal` | `huge` |
|---|---|---|
| dim | 1024 | 12288 |
| n_layers | 16 | 1 |
| n_heads / n_kv_heads | 16 / 8 | 192 / 96 |
| head_dim | 64 | 64 |
| MoE inter_dim (3.5x dim) | 3584 | 43008 |
| experts / top_k | 4 / 2 | 4 / 2 |
| vocab / rope theta | 151936 / 1e6 | 151936 / 1e6 |
| param_count | 1,066,241,024 | 10,528,837,760 |
| dense / sparse / active | 361,532,416 / 704,708,608 / 713,919,488 | 4,187,000,960 / 6,341,836,800 / 7,357,943,936 |
| num_flops_per_token @1024 | 3,551,348,736 | 33,096,721,152 |
| per-block regions | yes (80/80) | **no** |

Everything except `dim` and `n_layers` is derived
(`n_heads = dim/head_dim`, `n_kv_heads = n_heads/2`,
`moe_hidden_dim = dim*7/2`), and the parameter/flops formulas mirror
torchtitan's `get_moe_model_nparams_and_flops`. `tests/test_model_shape.py`
pins the five normal-size numbers against what a real run logs; they were
previously duplicated by hand in `megatron_baseline/`.

**Config naming**: `<config>` at `normal`, `<config>_<size>` otherwise, so
`--model-size huge` runs `qwen3_piper_1b_pretokenized_huge`. Every public
config has an explicit `_huge` `def` (not a `globals()` loop -- explicit
names stay greppable and carry a `__name__`), and a test asserts the closure
over every (scenario, arm, size) triple.

**Why huge is 1 layer at a large dim, and not 16 layers.** Embedding +
lm_head are `2*V*D` parameters and one layer is `45*D^2`, so their ratio is
`6753/D`. A 1-layer model at dim 1024 would be 87% embedding table and the
cuda-graph comparison would be measuring the lm_head and the CE, not a
transformer block. At dim 12288 the ratio inverts to 0.55x: the single layer
is 64% of the parameters and the great majority of the FLOPs.

**Why 12288 and not larger.** Megatron under `--compile-mode cuda-graph` is
the binding constraint, because `megatron_baseline/train.py` allocates a bf16
`main_grad` for *every* parameter under graph mode -- 10 B/param of state
against titan's 8 (params 2 + grads 2 + fused-AdamW m,v 4). Measured on an
idle H200 (139.81 GiB usable) with the megatron cuda-graph driver: dim 10240
peaks at 92.8 GiB, 12288 at 120.1, 13312 at 136.6 (2.3% headroom -- rejected),
14336 OOMs. The acceptance rule is <= 125 GiB. Full ladder including the OOM
rungs: `reports/20260809/`.

**The huge shape declares no regions, deliberately.** `PIPER_1B_REGIONS`
identifies a block graph by its invocations per window
(`n_layers * profiler_active`), and that count is the *identity*: measured
on a real 16-layer trace the forward graphs run {5, 80, 5} times and the
backward graphs {5, 80}, so 80 is unique to the block graphs. At one layer
the block graph also runs 5 times, colliding with two forward and one
backward partition, and `pooled_window_metrics` would raise "found 3". There
is no invocation count that identifies a 1-layer block graph and adding a
tiebreak would be relaxing validation rule 7 -- so `PiperShape.huge` sets
`supports_block_regions=False` and `_resolve_run` writes `regions: []`,
exactly as `piper1b_megatron` already does and for the same honest reason.
Rule 7 therefore does not guard huge runs; rules 8, 9 and 11 do. Cross-mode
metrics (total GPU kernel time, tokens/s, launch latency, peak memory) are
unaffected.

Rejected alternatives, for the record: a copy-pasted `_huge` scenario
(duplicates arm definitions, cannot apply to other scenarios, and the size
would not be a comparability boundary); an env var read inside
`config_registry.py` (hidden global state -- the recorded command would no
longer identify the model); overriding `--model-spec.model.dim` through tyro
(the layer list is nested dataclasses with per-layer depth-scaled inits;
there is no single field to move).

**Out of scope**: `kernel-bench` has no `--model-size`;
`benchmarks/kernels.py`'s `Piper1BSpec` keeps its own fixed shapes.
`analysis/components.py` needs no change -- it keys only on `--vocab-size`
and `--num-experts`, both unchanged by the huge shape.

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

The five titan scenarios share `PIPER_1B_REGIONS`: `forward_block` and
`backward_block`, each 80 invocations per window (16 layers x 5 active
steps). `piper1b_megatron` declares no regions (region pooling rides on
Inductor's compiled-graph annotations around whole transformer blocks, which
the megatron arm honestly lacks) -- its cross-engine metrics are total GPU
kernel time, tokens/s, launch latency, and peak memory.

**"Eager megatron" is shorthand, and it is imprecise.** Megatron-core sets
`jit_fuser = torch.compile` on torch >= 2.2 (`megatron/core/jit.py:17-24`,
enabled at import) and decorates 41 functions with it across 12 modules --
cross-entropy, bias-swiglu, bias-dropout, the router, norms. Those regions go
through Inductor in every run. What megatron does *not* do is compile whole
transformer layers: there is no `torch_compile` knob in
`transformer_config.py`, `model_parallel_config.py`, or
`megatron/training/arguments.py`, and `TEDotProductAttention` and the other
TE extension modules carry no compile at all. So the real axis is
**selectively-compiled megatron vs whole-block-compiled titan**, which is an
engine design difference -- NVIDIA's answer to whole-layer overhead is
hand-written TE kernels plus CUDA graphs, which `--compile-mode cuda-graph`
measures separately. Unlike the fusion defaults, there is no switch here we
are declining to set.

| scenario | arm | mechanism |
|---|---|---|
| `piper1b_rope` | `baseline` | TorchTitan `CosSinRoPE` |
| | `helion` | override `torchtitan.overrides.helion_rope.helion_cos_sin_rope` |
| | `te` | override `piper1b.rope.te_rope_override.te_rope`, needs gcc-13 |
| `piper1b_swiglu` | `baseline` | TorchTitan `GroupedExperts` |
| | `piper_optimized_triton` | override `piper1b.swiglu.combined_swiglu.piper_optimized_triton_fused_grouped_experts` |
| | `piper_optimized_inductor` | override `piper1b.swiglu.combined_swiglu.piper_optimized_inductor_fused_grouped_experts` |
| `piper1b_qkv` | `baseline` | config `qwen3_piper_1b_unfused_qkv` |
| | `fused_qkv` | config `qwen3_piper_1b` |
| `piper1b_lm_head` | `baseline` | config `qwen3_piper_1b_full_logits` |
| | `fused_linear_ce` | config `qwen3_piper_1b_fused_linear_ce` |
| | `te_fused_ce` | config `qwen3_piper_1b_te_fused_ce` |
| | `piper_optimized_te_ce` | config `qwen3_piper_1b_piper_optimized_te_ce` |
| `piper1b_attention` | `baseline` | `FlexAttention` (Inductor Triton template) |
| | `flash_attention_3` | config `qwen3_piper_1b_varlen`; needs the `flash3` group |
| | `flex_flash` | config `qwen3_piper_1b_flex_flash`; same FlexAttention module and BlockMask, lowered to FA4 CuTe kernels; needs the `fa4` group |
| `piper1b_megatron` | `baseline` | `launcher="megatron"`: Megatron-LM + TE bare GPTModel (see the Megatron section) |
| | `titan_stock` | config `qwen3_piper_1b_pretokenized` (fused qkv, stock kernels) |
| | `titan_swiglu` | pretokenized config + the `piper_optimized_inductor` swiglu override |
| | `titan_lm_head` | config `qwen3_piper_1b_piper_optimized_te_ce_pretokenized` |
| | `titan_swiglu_lm_head` | te_ce pretokenized config + the swiglu override |

`piper1b_attention` swaps only the inner attention, so it needs no `seed=42`
(the backend does not change parameter structure). Its `flash_attention_3`
arm pins `FlashAttnFwdSm90` / `FlashAttnBwdSm90` as trace markers because FA3
degrades to FA2 rather than failing: seeing `pytorch_flash::` names instead
would mean the arm measured FA2 under an FA3 label.

The three arms differ along **two** axes, which is why there are three.
`flash_attention_3` changes both the kernel family (Triton -> CUTLASS) and the
masking mechanism (BlockMask -> THD `cu_seqlens`); `flex_flash` keeps the
FlexAttention module, its mask_mod and its BlockMask and changes **only** the
lowering, to FA4 CuTe DSL kernels. So `baseline` vs `flex_flash` is the clean
kernel-family comparison, and `flex_flash` vs `flash_attention_3` isolates the
masking mechanism. Its markers (`FlashAttentionForwardSm90` /
`FlashAttentionBackwardSm90`) were captured by profiling: FA4's kernel names
are generated by the CuTe DSL and appear nowhere in the torch source. FA4
spells "FlashAttention" out where FA3 abbreviates it, so neither arm's markers
can be satisfied by the other's kernels -- `analysis/components.py` carries a
separate `flash_attncute` rule for the same reason.

**`flex_flash` is expected to lose on Hopper, and that is not a verdict on
FA4.** FlexAttention's packed-interval mask optimization is gated on compute
capability 10/11 (Blackwell), so on sm90 partial blocks evaluate the mask per
lane -- a path torch's own perf hint calls "much slower" (`TORCH_LOGS=
"+perf_hints"` to see it). Independently, at `batch > 1` torchtitan's
packed-document mask_mod emits a per-KV-lane gather the interval analyzer
refuses anyway. Report an FA4 number here as "FA4 running the generic per-lane
mask path on sm90", never as "FA4 is slower than the Triton template".

**There is deliberately no TE attention arm here, and there cannot be one.**
TransformerEngine wraps `DotProductAttention.forward` in
`torch.compiler.disable` itself (`transformer_engine/pytorch/jit.py`), so
Dynamo refuses to inline it: any titan arm calling TE attention dies with
"Skip inlining `torch.compiler.disable()`d function" the moment
`apply_compile` wraps the block containing it. This is upstream NVIDIA's
choice, not a gap in our integration, and it is why megatron runs TE eagerly.
Getting a titan+TE arm would mean excluding that block from compilation,
which changes the treatment and makes the arm incomparable to the others.
TE attention is still measured head-to-head in the `attention` kernel
scenario, where each arm carries its own compile treatment.

`piper1b_qkv`, `piper1b_lm_head`, and `piper1b_megatron` set `seed=42`
because their arms differ in model structure; the RoPE and SwiGLU scenarios
do not. The megatron scenario's `_pretokenized` configs swap the dataloader
for `piper1b/pretokenized_data.py`'s replay loader (all 40 steps of c4_test
batches materialized at init, ~zero per-step data-host cost, matching the
megatron driver's treatment). `replay_steps` must be >= `--training.steps`
or the loader hard-fails at exhaustion, so it tracks the run: the workload
sets `replay_dataloader=True` and `command_for_arm` delivers
`--dataloader.replay-steps` next to `--training.steps`.

### Output layout

```
out/<timestamp>/<scenario>/<hardware>/
  manifest.json     # schema 9: workload, regions, arms, commands, compile_mode, ac_mode, model_size, model_shape, execution_model, hardware_metadata
  run_state.json    # per-arm status, attempts, evaluation status
  results.json      # schema 3: throughput, memory, gpu_time, region stats, significance
  <arm>.log         # training stdout+stderr
  <arm>/profiling/traces/iteration_*/rank0_trace.json.gz
  attempts/<ts>/<arm>/   # archived artifacts from a failed prior attempt
```

`manifest.json` `hardware_metadata` records `requested_gpu`, `nvidia_smi`,
`cpu_pinning`, `torch_version`, `torchtitan_git_rev`, `benchmarks_git_rev`,
`megatron_git_rev`, `te_version`. Always cite `torchtitan_git_rev`,
`torch_version`, `compile_mode`, and `ac_mode` when reporting numbers --
plus `megatron_git_rev` and `te_version` for the megatron scenario.

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

1. Missing `<arm>.log`, or log lacking the profile's completion marker
   (`Training completed` for both engines).
2. `[Override]` line count != `arm.overrides_per_block * shape.n_layers`
   (16 at the normal size, 1 at huge: one per transformer block).
3. A declared `override_imports` entry with no matching `[Override] <path>:` line.
4. A profile `failure_marker` phrase in the log (`falling back to the
   PyTorch` for titan arms -- an optimized kernel silently degraded). This
   fails the arm regardless of anything else.
5. Fewer than `min_trace_windows` trace files.
6. A declared `trace_kernel_markers` string absent from every trace (the
   megatron arm pins the cuDNN fused-attention kernel name here, guarding a
   silent TE fallback to unfused attention).
7. `pooled_window_metrics` structural failure -- a declared region did not match
   exactly one same-phase compiled graph with the expected invocation count.
   This means Inductor repartitioned the graph; the region mapping is invalid.
8. The engine's mode line names a mode other than the requested one, or is
   missing entirely: `Compiling each TransformerBlock with torch.compile
   (mode=<torch-level name>)` for titan arms, `Megatron-LM training loop
   (mode=...)` for the megatron arm.
9. `cudaGraphLaunch` absent from every trace under `cuda-graph` -- graphs
   declined to capture (either engine), so the arm is not measuring the
   mode it claims.
10. The `Applied SelectiveAC` line's presence contradicts the requested
    `--ac` mode (titan arms only).
11. `size: <N> total parameters` for the requested shape's exact parameter
    count is absent from the log. Both engines print it verbatim; without
    this rule a run whose `--config` mapping or `--model-size` silently fell
    back to another shape would pass every other rule and be published under
    the wrong size. This is the only structural guard the huge shape has in
    place of rule 7.

Engine differences live in the `ValidationProfile` registry
(`VALIDATION_PROFILES`), selected by `Arm.validation`; rules 2/3/5/6/9/11
are shared. Rules 4, 7, 8, 9, 10, and 11 are the ones that catch silent
wrongness. Never work around them by relaxing the check.

### Resume

`run-all --resume <out_dir>` re-validates each arm against what is on disk,
skips those that already pass, archives partial artifacts under `attempts/`,
and re-runs the rest. It aborts if any of these changed since the manifest was
written: scenario, workload, selected arms, hardware label, extra TorchTitan
args, `compile_mode`, `ac_mode`, `model_size`, `nvidia_smi`, `cpu_pinning`,
`torchtitan_git_rev`, `benchmarks_git_rev`, `megatron_git_rev`. A different
GPU or a different commit will not resume -- that is intentional. Omitting
`--compile-mode`, `--ac`, or `--model-size` on a resume inherits the
recorded value; passing a different one is refused. Schema <= 8 manifests
carry no `model_size` and resume as `normal`; schema <= 7 manifests cannot be
resumed by this code at all (they record pre-rename mode names and imply
`ac=sac`).

### Evaluation

Requires `baseline` among the arms. Reports:

- **stable tokens/s** -- median over steps 2..10 of each 20-step cycle, excluding
  step 1 (startup noise) and steps 11..20 (profiler overhead).
- **peak memory** over all steps.
- **GPU kernel time** -- per-step summed kernel/memcpy/memset durations from the
  traces (`gpu_time` in `results.json`): total, within declared regions, the
  remainder, and the ratio vs baseline. It is the host-speed-immune metric,
  so prefer it to tokens/s -- but it is **not autotuning-immune**, and on
  arms that differ in one component it is not evidence. See the next
  section before ranking anything with it.

### Total kernel time cannot rank arms that differ in one component

**On arms that differ in ONE component of the same model, total GPU kernel
time is not evidence.** The arms share the rest of the model by
construction, but Inductor can pick different configs for that shared code
between arms, and the resulting drift routinely exceeds the effect under
test. Rank such arms with `analysis/components.py` on the component that
actually differs, and quote the total only as the arm's step cost.

Measured three times on 2026-08-09, twice producing a published claim that
had to be retracted:

| case | total says | `components.py` says |
|---|---|---|
| `piper1b_attention` normal/sac/default, `flex_flash` | 0.9816 -- a 1.8% win | `attention_core` **1.17x worse**; the "win" was -1.11 ms/step of `moe_expert_gemm` in arms whose MoE code is identical |
| `piper1b_attention` huge/none/default, `flash_attention_3` | 0.9645 -- a 3.5% win | `attention_core` **1.78x worse**; -11.14 ms/step across components that cannot differ against +1.64 ms of real attention |
| the same normal cell run twice, byte-identical code, 1 h apart | 0.9816, then 1.0103 | a 2.9% swing straddling 1.0, i.e. the sign of the conclusion is not reproducible |

Across all three, and across a 12x change in `dim`, `attention_core` gave
the identical ordering every time: baseline < `flex_flash` (1.17-1.19x) <
`flash_attention_3` (1.64-1.78x). **The totals agree with the components in
three of four cells -- the metric is right most of the time, which is
exactly what makes it dangerous.**

**Cross-*engine* comparisons are the exception.** Megatron and TorchTitan
share no code, so every component legitimately differs and the total is the
right metric for `piper1b_megatron`'s `baseline` vs the titan arms.

**Measured single-run noise floors**, so you know what a table can resolve:

| cell | drift in components that cannot differ |
|---|---|
| normal (`piper1b_rope`, repeat runs) | ~0.6% (`helion` 0.9899 -> 0.9959 on byte-identical code) |
| huge `piper1b_megatron` | 1.5% (4.0 ms/step of 265) |
| huge `piper1b_attention` | 4.2% (11.1 ms/step of 268) |

A consequence worth stating plainly: at the normal shape the `piper1b_rope`
arms (`helion` 0.9899-0.9959, `te` 1.0065-1.0121) sit **inside** the floor
and are indistinguishable from the baseline and from each other. Report them
that way rather than ranking them.

`components.py` works on `default`-mode cells only. Under `cuda-graph`,
graph replay erases the per-op CPU frames it classifies on, so the whole
captured block lands in `other_elementwise` -- while attribution health
still prints 100% agreement, so the tool does **not** warn you. No
per-component claim can be made about a cuda-graph cell.
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
| `--scenario` (repeatable) | all five | subset of kernel scenarios |
| `--n` | 200 | interleaved measurement cycles |
| `--warmup` | 30 | warmup cycles per mode |
| `--burst` | off | adds the 1/4/16/64 dispatch-cost diagnostic |
| `--batch` / `--seq-len` | 4 / 1024 | `Piper1BSpec` overrides (seq <= `max_seq_len`) |
| `--max-seq-len` | 2048 | raises the seq ceiling; needed to sweep `attention` past 2048 |
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
| `attention` | `baseline`*, `flex_flash`, `flash_attention_3` | fwd, fwd+bwd | inner attention only, packed-document causal masking; FA3 needs the `flash3` group, `flex_flash` the `fa4` group |

The `attention` scenario measures **inner attention only** -- the level at
which the implementations are substitutable, and the level that keeps it from
re-measuring the projection work `qkv` already covers. All three arms consume
the same q/k/v and the same synthetic packed-document boundaries, delivered in
the three mask forms the backends need -- a flex `BlockMask` at the default 128
block size, the same mask at the `(256, 128)` blocks the FLASH backend wants,
and THD `cu_seqlens` -- all built once in the inputs builder, because
`create_varlen_metadata_for_document` contains a device-to-host sync and
`create_block_mask` is itself a compiled call, so none may run inside a timed
closure. Nothing validates the FLASH block size on the torch side: it is
forwarded verbatim into FA4's block-sparse tensors, so a mismatch surfaces
inside FA4 rather than as a torch-level error.

One asymmetry is deliberate and recorded rather than hidden: **`baseline` is
not wrapped in `torch.compile`.** `FlexAttention` already holds a class-level
compile of `flex_attention`, so wrapping it again risks a double compile or a
graph break around its spmd context. `flex_flash` is the same module and gets
the same treatment; only `flash_attention_3` is wrapped explicitly. All three
arms are therefore compiled, just by different mechanisms, and
`KernelArm.compiled` records it.

**There is no TE arm in either attention scenario.** Two independent blockers,
both verified: TE and the FA3 varlen path cannot share a process (the cuDNN
soname collision above), which rules TE out of the single-process h2h; and
TE wraps `DotProductAttention.forward` in `torch.compiler.disable`
(`transformer_engine/pytorch/jit.py`), which the fork's
`fullgraph=True` per-block compile (`distributed/compile.py:58`) refuses,
ruling it out of e2e. The second is not absolute -- with graph breaks allowed
TE would run, splitting each block into ~3 graphs -- but that is a different
compile treatment from the baseline and so not a like-for-like arm.

The fp64 reference is computed **per (row, kv group)**. A one-shot
`[B, n_heads, L, L]` fp64 score tensor is 8.6 GiB at batch 4 / seq 4096 and
69 GiB at batch 32, so the obvious implementation OOMs exactly at the shapes
worth measuring. Gate the arms with `max_rel_l2` only: attention is a
reduction, and CLAUDE.md's rule against max/ULP metrics on reductions applies.

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

`flex_flash` is the one arm with the *opposite* failure mode, and it still
carries a guard. `BACKEND="FLASH"` hard-raises when `flash_attn.cute` is
missing rather than degrading to a Triton template, so an arm that runs at all
ran FA4 -- there is no silent fallback anywhere in that lowering. The
`FlashAttentionForwardSm90` guard therefore protects against the *reverse*
mistake: a future refactor quietly dropping `kernel_options` and leaving the
arm measuring the baseline under an FA4 label.

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
| `analysis/components.py` | Per-component GPU time (~12 classes), summed and overlap-corrected | 1..N arm dirs or traces, plus options |

`analyze.py` and `per_block.py` take uncompressed Chrome traces, but runs write
`.json.gz`. Decompress first:

```bash
gunzip -c out/<...>/baseline/profiling/traces/iteration_40/rank0_trace.json.gz > /tmp/a.json
```

`components.py` does **not** need that -- it reads `.json.gz` directly and
takes whole arm directories, pooling every window the way the harness does:

```bash
./run_bench.sh  # (any run) then:
D=out/<ts>/piper1b_megatron/<hw>
.venv/bin/python analysis/components.py $D/baseline $D/titan_swiglu_lm_head \
    [--by-phase] [--detail 10] [--json breakdown.json]
```

It splits each arm's kernel time into MoE expert GEMM, cross-entropy, lm_head
GEMM, SwiGLU, attention core, norm, attention projection GEMM, MoE
routing/permute, RoPE, optimizer, embedding, and other elementwise. Three
properties make it trustworthy, and all three are asserted in
`tests/test_components.py`:

- **The classes partition the total.** The summed column equals that arm's
  `results.json` `gpu_time.kernel_ms_per_step` exactly; a test re-derives it
  from the traces for every arm of a real run.
- **Classification is frame-first**, reading autograd/module frame names
  before kernel names and shapes. Shape-first is a trap: megatron reports
  `_GroupedLinearBackward`'s `Input Dims` as `[[98304,1024],[]]` with the
  expert dims stripped, which misfiles 122.8 ms/step of MoE backward as a
  projection GEMM. Note also that `_LayerNormLinear` frames contain *both*
  norm and GEMM kernels, split by kernel name.
- **Two independent attribution rules are cross-checked** (the nested `cpu_op`
  stack on the launching thread, and the smallest-duration `cpu_op` sharing
  the launch's `External id`). The printed agreement rate is 100% on real
  traces; any drop means the attribution is suspect.

Two reporting rules the tool enforces, both of which previously produced wrong
published conclusions:

- **Compare on the busy (interval-union) basis, not the summed one, whenever
  the arms differ in stream count.** Megatron runs 5 CUDA streams to titan's
  1, so summing overcounts it ~6.5% and *inverts the sign* of the MoE
  expert-GEMM comparison. `share_of_gap` is computed on the busy basis for
  this reason.
- **A fused component is never printed as `0.0`.** Inductor folds titan's RoPE
  into the qk-norm kernels, so RoPE renders as `<= 4.661 (fused: norm)` and
  gets a `norm_rope_fused` row of its own, keeping `norm` comparable to
  megatron's. `--merge-fused` (default) also emits an `attention_block` row
  that is like-for-like across engines.

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

**Execution model: single GPU, plain bf16, no FSDP.** The ModelSpec's
`parallelize_fn` is `piper1b/parallelize.py:parallelize_piper1b`, which
delegates to the fork's `parallelize_qwen3` with `skip_dp=True` (AC and
per-block compile applied, FSDP skipped) and hard-errors at `world_size > 1`
or any `training.dtype` other than `bfloat16`. `training.dtype="bfloat16"`
puts params, grads, and optimizer states in bf16 with no fp32 masters --
matching piper's own execution and the treatment kernel-bench already gives
its modules. This is the only bf16 mechanism in the run (there is no
autocast and no mixed-precision wrapper), which is why the wrapper enforces
it: the TE RoPE arm requires bf16 activations, and an fp32 swiglu baseline
would pass every validation rule while measuring the wrong thing. Manifests
record `execution_model`; runs from schema <= 6 used FSDP2 mixed precision
and are not comparable.

An arm changes behavior one of two ways:

- **A different registered config** (`arm.config`). Used when the difference is
  structural: fused vs unfused QKV, or the loss/lm-head strategy.
- **`--override.imports <dotted.path>`** (`arm.override_imports`). Swaps config
  nodes in place after construction. Each replacement logs
  `[Override] <module>.<function>: <fqn> <Old> -> <New>`, which is exactly what
  `validate_arm` counts.

## The Megatron baseline arm

`piper1b_megatron`'s `baseline` arm trains the same Qwen3-1B model with
Megatron-LM + TransformerEngine instead of TorchTitan. All Megatron
knowledge lives in `megatron_baseline/`; the harness connects only through
`Arm(launcher="megatron", validation="megatron")` and
`megatron_baseline.location` for provenance. The runner launches
`python -m megatron_baseline.train` with the workload sizes, seed, profiler
schedule, and compile mode; the driver replicates the titan treatment
itself (fused AdamW on every param, titan's LR lambda, pre-clip-norm
logging, sum/valid-tokens loss, gc handling, identical torch.profiler
schedule and trace layout, titan-shaped step log lines).

Faithfulness guarantees, all verified:

- **Same model**: bare megatron-core `GPTModel` built from the same
  `piper1b.model_shape.PiperShape` the TorchTitan config uses -- exactly
  1,066,241,024 bf16 params at `normal`, 10,528,837,760 at `huge`.
  `tools/megatron_parity_check.py [--model-size SIZE]` transfers titan
  weights into the megatron layout and matches logits on a real batch -- run
  it after touching `megatron_baseline/model.py` or bumping either
  submodule. The gate is per-shape (`_PARITY_GATE`): 2e-2 at normal
  (measured 5.5e-3), 5e-2 at huge (measured 2.03e-2). The wider huge gate is
  bf16 accumulation, not slack, and it is evidenced rather than assumed --
  `--fp32-reference` runs the same weights in fp32 and shows titan's own
  bf16 output sits 3.25e-2 from it against megatron's 3.29e-2 (ratio 1.011),
  i.e. the engines agree with each other better than either agrees with
  fp32. The QKV grouped-interleave is proved separately and *bitwise* by
  `_assert_qkv_roundtrip`, so a layout bug cannot hide inside a widened
  gate. Never widen one without both.
- **Same data and masking**: `megatron_baseline/data.py` drains torchtitan's
  own c4_test dataset class (bit-identical stream to the titan arms'
  replay loader; tested) and packs each batch's rows into TE THD form with
  `cu_seqlens` at the `positions == 0` document boundaries, reproducing
  titan's block-diagonal causal flex mask. cu_seqlens are padded to a
  constant length and `max_seqlen` pinned to seq_len in both modes (static
  shapes for graph capture without changing the computation).
- **Same precision**: plain bf16 params/grads/optimizer states, no fp32
  masters, no autocast, no fp8. No recompute ever (`--ac` never affects
  this arm).
- **Megatron at its own best**: every fusion megatron's training entrypoint
  would enable is enabled explicitly. This matters because building
  `TransformerConfig` directly bypasses `megatron/training/arguments.py`,
  where those defaults actually live -- the dataclass defaults are `False`
  where argparse defaults them `True` (`--no-bias-swiglu-fusion` is
  `action="store_false"`, forwarded as `bias_activation_fusion`). Running
  the dataclass defaults once cost 11.9 GPU ms/step of unfused SwiGLU and
  produced a bogus engine verdict. `train.py` now asserts the fusion state
  and logs `Megatron fusions: ...`, and the arm pins `_mul_silu_split` /
  `_permute_kernel` as trace markers. `gradient_accumulation_fusion` is the
  one performance default deliberately declined (its fused wgrad path needs
  apex-style `main_grad` buffers we have no DDP wrapper to provide); its
  cost is unmeasured.
- **Cross-entropy implementation is a reporting-sensitive choice.**
  `cross_entropy_fusion_impl="te"` routes the loss through
  `transformer_engine.pytorch.parallel_cross_entropy` -- the same
  implementation the `te_fused_ce` titan arm wraps and that
  `piper_optimized_te_ce` optimizes -- making the loss path a kernel
  comparison rather than an algorithm comparison. Megatron's *training
  entrypoint* refuses this combination (`arguments.py` ~1630, "known
  stability issues"); the core config only warns (`model_parallel_config.py`
  ~536) and we construct the config directly, so we get it. The alternative,
  `"native"`, is megatron's own jit_fuser CE: it upcasts the full
  `[tokens, 151936]` logits to fp32 and makes ~6 full-tensor traversals,
  costing 88 GPU ms/step at batch 48 versus 14.9 for the TE-family kernel --
  73% of the entire engine gap. **Always say which one a number came from**:
  `"te"` = megatron's fastest available loss path, `"native"` = megatron as
  NVIDIA ships it. The driver logs `cross_entropy_fusion_impl=` on the
  `Megatron fusions:` line.
- **No distributed machinery**: single-rank process group + megatron init
  only; no Megatron DDP wrapper, no MegatronOptimizer.

Environment notes: TE's native tuned RMSNorm kernels fail to launch on this
box's cuda-compat stack, so `configure_te_environment` routes norms through
TE's cuDNN backend (`NVTE_NORM_*_USE_CUDNN=1`) -- keep it in any process
that imports TE here. Without apex, megatron's standalone norms are torch
RMSNorm (its own spec fallback); the qkv-input norm fuses into the TE
linear.

Under `--compile-mode cuda-graph` the arm uses Megatron's per-layer partial
capture (`MoETransformerLayer`, `cuda_graph_modules=("moe_router",
"moe_preprocess")`): `n_layers x 2 modules x fwd+bwd` graph replays/step (64
at the normal shape, **4** at the 1-layer huge shape), with attention and
expert GEMMs eager -- whole-iteration capture is impossible for dynamic MoE (the token
dispatcher D2H-copies `tokens_per_expert`, which capture forbids), and this
rev's local impl has no attention scope for MoE layers. **Megatron's graph
coverage is therefore far thinner than titan's whole-block graphs; say so
next to any cuda-graph-mode comparison.** Graphed-module weight grads land
in manually attached `main_grad` buffers, merged into `.grad` before
clipping each step.

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

The submodule is our fork (`JayAndJef/torchtitan`, with `pytorch/torchtitan` as
`upstream`) and carries benchmark-required commits that upstream does not have.
Rebasing onto upstream must preserve them:

- `Generalize loss-owned LM head integration` -- the `LossWithLMHead` protocol
  `piper1b_lm_head`'s `fused_linear_ce` arm needs.
- `CompileConfig.mode` plus its use in `apply_compile` -- what `--compile-mode`
  rides on, including the `(mode=...)` suffix on the "Compiling each
  TransformerBlock" log line that `validate_arm` matches.
- `Skip expert-usage accumulation when load balancing is off` (the
  `load_balance_coeff is not None` gate around `tokens_per_expert_E.add_`
  in `models/common/moe.py`) -- without it the in-place input mutation makes
  Inductor refuse to capture every forward block graph under the cudagraph
  compile modes.

`piper1b/parallelize.py` additionally relies on `parallelize_qwen3`'s
`skip_dp` kwarg and its ordering guarantee: AC, then `apply_compile` (which
emits the `(mode=...)` log line), then the early return *before* mesh
resolution and `apply_fsdp_to_decoder`. Note also that this fork applies no
autocast at world size 1 -- `training.mixed_precision_param` is consumed
only by `apply_fsdp_to_decoder` -- so `training.dtype` is the whole dtype
story; if a future bump adds single-device autocast, revisit it.

`config_registry.py` imports private Qwen3 helpers. After any bump, verify all
of these still exist with unchanged behavior:

- `_build_qwen3_moe_layers`, `_EMBEDDING_INIT`, `_output_linear_init`,
  `_qwen3_norm` from `torchtitan.models.qwen3`
- `torchtitan.config.override` (`override`, `derive`) and the `[Override]` log
  line format that `validate_arm` regexes
- `CosSinRoPE` and `_maybe_check_max_pos` from `torchtitan.models.common.rope`
- the trainer's `Model <name> <flavor> size: N total parameters` log line,
  which validation rule 11 matches
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
- Numbers are only comparable within one `torch_version`, one
  `torchtitan_git_rev`, one `benchmarks_git_rev`, one `compile_mode`, one
  `ac_mode`, and one `model_size` (plus one `megatron_git_rev`/`te_version`
  for the megatron scenario). All are in
  every manifest -- check them before comparing against an older run in
  `out/` (manifests written before schema 6 predate the compile-mode flag
  and are `default`; before schema 8 they record the old torch-level mode
  names -- `reduce-overhead` data is comparable to `cuda-graph` for titan
  arms -- and imply `ac=sac`). Manifests before schema 7 predate the FSDP
  removal (they ran under FSDP2 mixed precision with fp32 masters) and are
  not comparable to schema-7+ runs at all -- different init RNG, different
  optimizer numerics.
- Put investigation notes and hardware-specific results in `reports/`, which is
  gitignored. Keep them out of `README.md` and this file.
- After changing anything in `benchmarks/`, run the test suite. It is CPU-only
  and takes about two seconds.
- On a shared box, drive multi-cell matrices with `tools/run_matrix.sh`
  rather than a loop of `run-all`s. It refuses to start on a dirty tree
  (`benchmarks_git_rev` would mislabel the run), holds a `flock`, waits for
  a genuinely idle GPU before each cell, and runs a watchdog *during* each
  cell that flags foreign compute PIDs (any session id but its own),
  unaccounted GPU memory, and host-load spikes. A flagged cell is moved
  aside and redone on the next pass, because a contaminated run still writes
  a `results.json` and the bad numbers would otherwise be permanent. Never
  report a cell it marked `CONTAMINATED`.
