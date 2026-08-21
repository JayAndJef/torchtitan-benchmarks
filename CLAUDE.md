# torchtitan-benchmarks: Agent Guide

Out-of-tree benchmarks for the Piper Qwen3-1B TorchTitan port. One CLI,
three kinds of measurement:

1. **Declarative end-to-end scenarios** -- `./run_bench.sh run` / `run-all`,
   driven by `benchmarks/e2e/registry.py`. Runs real TorchTitan training,
   validates, and evaluates.
2. **Declarative kernel-isolation scenarios** -- `./run_bench.sh
   kernel-bench`, driven by `benchmarks/kernel/registry.py`. Times competing
   kernel implementations head-to-head on synthetic tensors at Piper-1B shapes.
   A scenario cuts the model at one boundary and ranks the implementations
   there.
3. **Declarative kernel spans** -- `./run_bench.sh kernel-bench --span`,
   driven by `benchmarks/kernel/spans.py`. A span is an implementation that
   fuses **across** a scenario cut, so it belongs to no single scenario. Its
   claim is the span against the **sum of the scenarios it replaces**. The
   roster is empty at this rev; read the module.

Never present kernel numbers as end-to-end results, or vice versa: a kernel
that wins in isolation can be irrelevant once Inductor fuses the graph around
it. Never present a span total as a scenario total either: the two answer
different questions, and the span statistic is not the scenario statistic.


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
  `benchmarks/execution/paths.py` hardcodes it as `TITAN_DIR`, derived from
  `BENCH_DIR` on the line above. It is defined once: the megatron data
  module imports this one rather than redefining it.
- Megatron-LM is a git submodule at `third_party/Megatron-LM` (pinned
  `59b72fa57`, core 0.20.0). It is **not** pip-installed (its pyproject wants
  python >= 3.12); `benchmarks/models/piper_qwen3/megatron_bootstrap.py` puts it
  on `sys.path` in the driver process only. `MEGATRON_DIR` overrides the location
  for development checkouts; the manifest records the resolved rev.
- Every command runs under `.venv/bin/python`. `run_bench.sh` execs it directly;
  `benchmarks/e2e/launch.py` derives the training subprocess interpreter from
  `sys.executable`, so the CLI and training always share one environment.
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

`benchmarks/` is the **one** first-party package. Everything importable lives
under `benchmarks.`; the two argv-driven trace scripts live in `tools/` beside
the other operator scripts. The three former top-level packages are gone --
their names are listed once, in the provenance note below, and nowhere else.

| path | contents |
|---|---|
| `benchmarks/cli/` | `main.py` (the Click group, `scenarios`, and the `add_command` wiring), `e2e.py` (`run`/`run-all`/`evaluate` and their shared option block), `kernel.py` (`kernel-bench`), `rendering.py` (the `RunEvent` renderer both families share), plus `__main__.py`, which is what `python -m benchmarks.cli` runs. Commands are declared with plain `@click.command` and attached in `main.py`, so importing `main` is what populates the group. `scenarios` prints three rosters: the e2e scenarios, the kernel scenarios, and the kernel spans -- the span heading prints even when the roster is empty, so a reader learns `--span` exists |

| `benchmarks/e2e/registry.py` | Scenario/arm/workload declarations, the compile-mode and AC-mode tables, `EXECUTION_MODEL` |
| `benchmarks/e2e/runner.py` | Executes and resumes a scenario; `RunRequest`/`RunResult` |
| `benchmarks/e2e/launch.py` | Builds the training subprocess command line for each arm (both engines) |
| `benchmarks/e2e/validation.py` | `validate_arm` and the `ValidationProfile` registry |
| `benchmarks/e2e/results.py` | Evaluation, region comparison, `results.json`, and its renderer |
| `benchmarks/e2e/data/piper_qwen3.py` | Replay dataloader: drains the c4_test pipeline at init (megatron scenario) |
| `benchmarks/e2e/megatron/` | The Megatron-LM training driver (`train.py`) and its THD data pipeline (`data.py`) |
| `benchmarks/kernel/schema.py` | What a kernel benchmark *is*: `KernelScenario`/`KernelArm`/`CorrectnessCheck`/`KernelWorkload`, the span types `KernelSpan`/`SpanParts`/`validate_span_parts`, plus `resolve_symbol`, `resolve_shape_and_workload` and `shape_summary` |
| `benchmarks/kernel/registry.py` | The kernel scenarios themselves (17 at this rev, 71 arms), declared with those types. Re-derive the counts; do not quote them |
| `benchmarks/kernel/spans.py` | The kernel spans. Parent-side and torch-free, exactly as the scenario registry is. `KERNEL_SPANS` is **empty** at this rev: the mechanism landed before any declaration. It imports the scenario registry to check that each named part arm exists, which is why it is a separate module |
| `benchmarks/kernel/runner.py`, `worker.py` | Kernel-bench supervisor and the per-pass subprocess it launches, one per (arm, replicate) plus one for correctness. The runner also owns `measurement_plan`, which orders a run's units: every enclosed scenario ahead of its span |
| `benchmarks/kernel/engine/` | `arm.py` (the `BuiltArm` contract), `measurement.py` (burst timing, memory and the burst ladder), `correctness.py` (the gates), `run.py` (orchestration, and the timing pass: `build_timing_arm`/`time_replicate`/`arm_extras`, composed once in `time_replicate_block`), `phases.py` (the stdlib-only wall-clock attribution every fragment carries) and `statistics.py`. **The engine does not know that spans exist**: both passes take a `KernelScenario`, and a span hands them its own `measurement`, which is one |
| `benchmarks/kernel/operations/` | Arm builders, one module per scenario and named after it, plus `common.py` |
| `benchmarks/kernel/results/` | `schema.py` (kernel `results.json`), `merge.py` (parent-side assembly of the workers' fragments, for a scenario and for a span), `span_statistics.py` (the span-versus-parts estimator, parent-side because one of its two sides is a sum over units the engine never sees) and `reporting.py` |

| `benchmarks/models/piper_qwen3/shape.py` | `PiperShape` + the `normal`/`huge` registry; both engines' single source of geometry |
| `benchmarks/models/piper_qwen3/config_registry.py` | The `--module benchmarks.models.piper_qwen3` config port; all registered `--config` names |
| `benchmarks/models/piper_qwen3/parallelize.py` | The ModelSpec `parallelize_fn` (single-GPU, plain bf16, no FSDP) |
| `benchmarks/models/piper_qwen3/mcore_profiles.py` | Megatron behaviour as data: one `McoreProfile` per variant, torch-free and parent-side |
| `benchmarks/models/piper_qwen3/megatron_bootstrap.py` | Megatron location/provenance and the TE environment setup |
| `benchmarks/models/piper_qwen3/megatron_model.py` | The Qwen3-1B megatron-core `GPTModel` builder; takes a shape and a profile |
| `benchmarks/models/piper_qwen3/megatron_weights.py` | The titan-to-megatron per-parameter map, tagged by component so a caller can take a slice |
| `benchmarks/models/piper_qwen3/titan_model.py` | The in-process titan build, and the override count that replaces the `[Override]` log check |
| `benchmarks/models/piper_qwen3/components/rope/` | TE RoPE override + `te_rope_standalone.cu` |
| `benchmarks/models/piper_qwen3/components/swiglu/` | Combined-SwiGLU Triton kernels and override |
| `benchmarks/models/piper_qwen3/components/lm_head/` | Vendored TE cross-entropy, Piper-optimized CE, losses |
| `benchmarks/traces/` | `schema.py` (the `Region` declaration) and `extraction.py` (trace parsing, window and region pooling) |
| `benchmarks/artifacts/` | On-disk artifacts: `layout.py` (output layout, `trace_files`, `atomic_write_json` -- the only JSON writer), `manifests.py` (the manifest schema and the resume predicate; the one module here coupled to `e2e/`), `run_state.py` (the per-arm ledger) and `summaries.py` (`SampleSummary`, shared by both systems) |
| `benchmarks/execution/` | Subprocess execution: `paths.py` (`BENCH_DIR`/`TITAN_DIR`, `RuntimePaths`), `environment.py` (the child's env vars), `affinity.py` (NUMA pinning), `provenance.py` (`hardware_metadata`, including the two cuDNN fields -- see "Which cuDNN a megatron arm runs"), `events.py` (`RunEvent`, `ProcessRunner`) |

| `tools/` | `megatron_parity_check.py` (GPU logit-parity gate between the engines, `--model-size` aware); `run_matrix.sh` (shared-box matrix supervisor), `collect_matrix.py` (merges a matrix tree into one JSON), `test_watchdog_attribution.sh` (proves the supervisor's process-ancestry check), and the two argv-driven trace diagnostics `analyze.py` and `per_block.py` |
| `tests/` | CPU + GPU unit tests. Deliberately **flat** -- every module does `sys.path.insert(0, <repo root>)` at a fixed depth, and `unittest discover -s tests` needs no `__init__.py` |
| `third_party/torchtitan/` | Pinned submodule (our fork) |
| `third_party/Megatron-LM/` | Pinned submodule (upstream NVIDIA, sys.path only) |
| `out/` | Run outputs (gitignored) |
| `reports/` | Local investigation notes (gitignored). Put conclusions here, not in docs. |

### Provenance boundary: artifacts written before the restructure

The layout above landed as a single flag-day commit, `9f6a69b`. It
retired three top-level packages -- `piper1b`, `megatron_baseline` and
`analysis` -- into `benchmarks/` and `tools/`, and flattened the old
`benchmarks.kernel_*` modules into `benchmarks/kernel/`. **This section is the
only place in the repo's documentation where any retired module name still
appears**, and here they appear as history, not as instructions; the
retired-path audit allowlists it for that reason.

Every manifest, `results.json` and report written **before** the flag day
records the old module names, and they are still on disk: 197 `manifest.json`
files under `out/`, of which 142 record `"module": "piper1b"`, 55 record
`benchmarks.kernel_arms:<builder>` kernel-arm paths, and 21 record
`python -m megatron_baseline.train` in `commands`.

**Those strings are inert history.** No module of any of those names exists any
more, and nothing imports one: manifest `commands` is a write-only provenance
field, and the recorded builder paths are read as data, never resolved. An old
run directory therefore still **decodes** -- `load_manifest`, `load_run` and
`load_kernel_results` read every pre-restructure schema they read before, and
`tests/test_legacy_artifacts.py` pins that with byte-frozen fixtures.

What an old directory is **not** is **resumable**. `run-all --resume` aborts on
a changed `benchmarks_git_rev`, and the flag day changes it -- so a run started
before `9f6a69b` cannot be continued after it, by design. Nothing here
makes the old *numbers* wrong; a pure file move changes no measurement. It makes
them a separate `benchmarks_git_rev`, which was already a comparability
boundary.

Do not "fix" a retired dotted path inside a recorded artifact. The fixtures
under `tests/fixtures/legacy/` are byte-frozen evidence of runs that actually
happened, the retired strings are the payload under test, and the retired-path
audit allowlists them for exactly that reason.

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
`benchmarks/e2e/registry.py` maps the harness name to the `--compile.mode`
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
shapes live in `benchmarks/models/piper_qwen3/shape.py` as frozen `PiperShape`
dataclasses and are the single source of truth for *both* engines --
`benchmarks/models/piper_qwen3/config_registry.py` and
`benchmarks/models/piper_qwen3/megatron_model.py` build from the same object,
so a size cannot drift between them. That module imports nothing but
`dataclasses`.

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
previously duplicated by hand in the Megatron builder.
`supports_block_regions` is derived too (`n_layers > 1`, see below), and
`parity_gate` -- the tolerance `tools/megatron_parity_check.py` enforces --
is per-shape data on the same dataclass rather than a lookup table beside
the checker.

**Delivering a shape**: a shape is **one entry in `PIPER_SHAPES`**, handed to
torchtitan as `--config-arg size=<name>` alongside `--config`, so
`--model-size huge` runs `--config qwen3_piper_1b_pretokenized --config-arg
size=huge`. Every public config takes a `size: str = "normal"` keyword and
resolves it through `shape_by_name`, which raises on an unknown size; there
is one `def` per config no matter how many shapes exist. A test asserts that
every (scenario, arm) config resolves and accepts every registered size.

**Why huge is 1 layer at a large dim, and not 16 layers.** Embedding +
lm_head are `2*V*D` parameters and one layer is `45*D^2`, so their ratio is
`6753/D`. A 1-layer model at dim 1024 would be 87% embedding table and the
cuda-graph comparison would be measuring the lm_head and the CE, not a
transformer block. At dim 12288 the ratio inverts to 0.55x: the single layer
is 64% of the parameters and the great majority of the FLOPs.

**Why 12288 and not larger.** Megatron under `--compile-mode cuda-graph` is
the binding constraint, because `benchmarks/e2e/megatron/train.py` allocates a bf16
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
tiebreak would be relaxing validation rule 7 -- so `supports_block_regions`
is `n_layers > 1`, False at huge, and `_resolve_run` writes `regions: []`,
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
there is no single field to move). The `<config>_<size>` name-mangling that
shipped first was itself later rejected in favor of `--config-arg`: it cost
ten twin `_huge` `def`s and made every future shape cost ten more, and it
put the geometry in the config *name* rather than in an explicit argument.

**Shared with `kernel-bench`**: the kernel benchmarks take the same
`--model-size` (single-valued -- kernel-bench does not sweep sizes) and draw
their geometry from the same `PIPER_SHAPES` entry, so a shape cannot drift
between the two systems either. Batch and sequence length are **not** model
shape: they live in `benchmarks/kernel/schema.py`'s `KernelWorkload`, and
`resolve_shape_and_workload` pairs the two and validates the pair.

### The 40-step floor

`Workload` defaults are `profile_freq=20`, `profiler_warmup=5`,
`profiler_active=5`, `min_trace_windows=2`. `workload_with_overrides` enforces
`steps >= profile_freq * min_trace_windows`, so **40 is the minimum** and
anything less raises `ValueError`. Two profiler windows are required by both
trace validation and region pooling. Do not lower `--steps` to save time.

### Scenarios and arms

**Arm names match the kernel scenarios at one scenario only.** The pairing is
declared data rather than a naming habit: `tests/test_migration_contract.py`'s
`KERNEL_TO_E2E_SCENARIO` maps `lm_head` onto `piper1b_lm_head`, so
`piper1b_lm_head/piper_optimized_te_ce` and `lm_head/piper_optimized_te_ce`
are the same code at two scopes. The map held four entries until 2026-08-20.
Three left it in two different ways, and the difference matters:

- **`rope` left by becoming cross-engine.** Its kernel arms are now
  `mcore/base`, `mcore/no_rope_fusion`, `titan`, `titan/helion` and
  `titan/te`, against an e2e `piper1b_rope` that still runs `baseline`,
  `helion` and `te`.
- **`swiglu`, `qkv` and `attention` left by being deleted.** Each was
  superseded by a cross-engine scenario that re-homed every one of its arms:
  `swiglu` into `expert_mlp`, `qkv` into `qkv_prep`, `attention` into
  `attention_core`. The e2e scenarios `piper1b_swiglu`, `piper1b_qkv` and
  `piper1b_attention` are untouched and still run.

The e2e ids are deliberately not renamed to match a kernel roster -- they name
directories under `out/` and every published number. Every arm in both
registries carries a one-line `description`; `./run_bench.sh scenarios` prints
them and manifests record them.


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
| | `te` | override `benchmarks.models.piper_qwen3.components.rope.te_rope_override.te_rope`, needs gcc-13 |
| `piper1b_swiglu` | `baseline` | TorchTitan `GroupedExperts` |
| | `piper_optimized_triton` | override `benchmarks.models.piper_qwen3.components.swiglu.combined_swiglu.piper_optimized_triton_fused_grouped_experts` |
| | `piper_optimized_inductor` | override `benchmarks.models.piper_qwen3.components.swiglu.combined_swiglu.piper_optimized_inductor_fused_grouped_experts` |
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
can be satisfied by the other's kernels.

**`flex_flash` is expected to lose on Hopper, and that is not a verdict on
FA4.** FlexAttention's packed-interval mask optimization is gated on compute
capability 10/11 (Blackwell), so on sm90 partial blocks evaluate the mask per
lane -- a path torch's own perf hint calls "much slower" (`TORCH_LOGS=
"+perf_hints"` to see it). Independently, at `batch > 1` torchtitan's
packed-document mask_mod emits a per-KV-lane gather the interval analyzer
refuses anyway. Report an FA4 number here as "FA4 running the generic per-lane
mask path on sm90", never as "FA4 is slower than the Triton template".

**There is deliberately no TE attention arm in this e2e scenario, and there
cannot be one.** TransformerEngine wraps `DotProductAttention.forward` in
`torch.compiler.disable` itself (`transformer_engine/pytorch/jit.py`), so
Dynamo refuses to inline it: any titan arm calling TE attention dies with
"Skip inlining `torch.compiler.disable()`d function" the moment
`apply_compile` wraps the block containing it. This is upstream NVIDIA's
choice, not a gap in our integration, and it is why megatron runs TE eagerly.
Getting a titan+TE arm would mean excluding that block from compilation,
which changes the treatment and makes the arm incomparable to the others.

**The kernel side is the other way round, and the old text here said the
opposite.** `kernel-bench` runs no compile treatment it did not choose, so TE
attention is measurable in isolation, and the `attention_core` scenario
measures three TE backends: `mcore/base` (cuDNN FusedAttention),
`mcore/attn_flash3` (FlashAttention 3) and `mcore/attn_unfused` (TE's own
torch implementation). The sentence "no kernel scenario measures TE
attention" was true of the deleted single-engine `attention` scenario and is
false now. (`attention_core` also collides by name with the `attention_core`
component label in "Total kernel time cannot rank arms that differ in one
component". The label there is the removed trace classifier's, and it names
no scenario.)


`piper1b_qkv`, `piper1b_lm_head`, and `piper1b_megatron` set `seed=42`
because their arms differ in model structure; the RoPE and SwiGLU scenarios
do not. The megatron scenario's `_pretokenized` configs swap the dataloader
for `benchmarks/e2e/data/piper_qwen3.py`'s replay loader (all 40 steps of c4_test
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

`validate_arm` (`benchmarks/e2e/validation.py`) fails an arm on any of:

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
test. Quote the total only as the arm's step cost.

**No in-tree tool currently attributes per-component GPU time.** A
`components.py` trace classifier did, and it was removed (before the package
restructure; it never moved into `tools/`).

For the **same-engine** case, rank the implementations in the matching
`kernel-bench` scenario instead. **Only `lm_head` still pairs by name**:
`KERNEL_TO_E2E_SCENARIO` holds that one entry, so
`piper1b_lm_head/piper_optimized_te_ce` and `lm_head/piper_optimized_te_ce`
are the same code at two scopes. For the other four titan e2e scenarios the
matching kernel scenario is now **cross-engine**, and its titan arms carry
`engine/profile` names:

| e2e scenario | kernel scenario | the titan arms to read |
|---|---|---|
| `piper1b_rope` | `rope` | `titan`, `titan/helion`, `titan/te` |
| `piper1b_swiglu` | `expert_mlp` | `titan`, `titan/fused_grouped_experts`, `titan/piper_optimized_triton`, `titan/piper_optimized_inductor` |
| `piper1b_qkv` | `qkv_prep` | `titan`, `titan/unfused_qkv` |
| `piper1b_attention` | `attention_core` | `titan`, `titan/flex_flash`, `titan/flash_attention_3` |

**Two of those four re-homings changed what the number means, and both say so
in their own `description`.** `qkv_prep` puts the attention-input norm inside
the cut, on both engines, because megatron fuses the RMSNorm into
`linear_qkv`'s GEMM prologue and exposes no entry point that runs either half
alone -- so a `qkv_prep` number is the norm plus the projection, and a `qkv`
number under `out/` is the projection alone. `expert_mlp` publishes each Piper
arm against `titan/fused_grouped_experts`, which is TorchTitan's own w13
fusion, rather than against unfused experts -- so the Piper arms are no longer
credited with a fusion upstream already ships. Neither is a like-for-like
successor of the retired number.

Remember also that a kernel-isolation number is **not device time**: for small
kernels it is dominated by host dispatch, and `--burst` amortization does not
remove that (see "Method" under Kernel-isolation benchmarks). A kernel-speed
claim needs profiler-summed device time, which nothing in this repo currently
measures. Do not apply the replacement more loosely than the tool it replaces.

The **cross-engine** case -- attributing a megatron-vs-titan gap to particular
components -- is what the 16 cross-engine kernel scenarios are for. They are
declarations at this rev and nothing more: **no `results.json` under `out/`
contains the string `mcore/`**, so there is no cross-engine component number
to cite yet. Only one of the 16, `attention_core`, has ever been built at all;
see "No cross-engine arm has produced a number" below.


Measured three times on 2026-08-09, twice producing a published claim that
had to be retracted:

| case | total says | per-component attribution said |
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

Under `cuda-graph` the question is unanswerable **from these traces**: graph
replay erases the per-op CPU frames a frame-based attribution classifies on,
so the whole captured block arrives as one undifferentiated lump. Kernel
names do survive the replay, so attributing by name is not forbidden in
principle -- but nothing in-tree does it, and a name-only classifier gives up
the frame evidence that made the removed tool trustworthy. No per-component
claim can be made about a cuda-graph cell.
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
| `--scenario` (repeatable) | every scenario | subset of kernel scenarios |
| `--replicates` | 5 | sweeps of every arm; the unit the CI is taken over |
| `--replicates-per-process` | 1 | consecutive replicates of one arm per worker; above 1 the CI is renamed (see "Startup cost") |
| `--samples-per-replicate` | 40 | timed bursts per arm per mode, per replicate |
| `--burst-k` | 16 | calls per timed burst; one value for every arm |
| `--warmup-calls` | 30 | untimed calls per arm per mode, before each replicate |
| `--burst` | off | adds the 1/4/16/64 dispatch-cost diagnostic |
| `--model-size` | `normal` | shape from `PIPER_SHAPES`; single-valued, no sweep |
| `--batch` / `--seq-len` | 4 / 1024 | `KernelWorkload` overrides (seq <= `max_seq_len`) |
| `--max-seq-len` | 2048 | raises the shape's seq ceiling; needed to sweep `attention` past 2048 |
| `--seed` | 0 | input generator seed |
| `--hardware` | `auto` | provenance label |
| `--out` | `out/<ts>/kernels/<scenario>/<hardware>` | single `--scenario` only |
| `--cache-root` / `--compiler-env` | as e2e | `rope` needs the compiler env |

Unlike `run-all --all-scenarios`, a failing scenario does not abort the rest;
every scenario is reported and the command exits nonzero if any failed.
Deliberately ignores the `OUT`/`SEQ`/`BATCH` env vars -- flags only, so an
e2e shell cannot leak settings into a kernel run.

Four scenarios need their synthetic rows to route evenly
(`batch * seq_len * top_k` divisible by `num_experts`), and each declares it
with `requires_balanced_routing`: `swiglu`, `dispatch_permute`, `expert_mlp`
and `moe_combine`. A shape/workload pair that breaks that skips those four
**loudly** -- named numbers, a recorded error, a nonzero exit -- rather than
capping or rounding anything. The check is per scenario, so every other
scenario is unaffected and still runs. The same invariant is re-asserted
inside `run_kernel_scenario`, so `python -m benchmarks.kernel.worker` and any
other direct caller raise instead of measuring an expert split that does not
cover the rows they built.

**`--model-size huge` is probed on four scenarios only.** On an H200 on
2026-08-19, `qkv`, `attention` and `lm_head` completed at `huge` with every
arm `ok`, and `swiglu` ran out of memory and wrote no `results.json`
(`out/20260819T010252Z/kernels/`). No cross-engine arm has run at either
size. Treat every other scenario at `huge` as untested rather than as
working.

The `swiglu` run died inside `run_correctness_pass`, in the gate's fp32
upcast, with 134 GiB of the device's 139.81 GiB already in use. The
arithmetic behind that: `swiglu_inputs` allocates three fp32
`(4, 43008, 12288)` expert tensors (7.9 GiB each, 23.6 GiB of state dict,
held for the whole run), and each of the three arms then loads its own bf16
copy (11.8 GiB) and grows a bf16 weight gradient of the same size in
backward. That run predates the residency change: `run_correctness_pass`
now builds one arm at a time and drops it before the next, so the pass is
bounded by the largest single arm rather than by their sum. Nobody has
re-run `swiglu` at `huge` since. Measure it before reporting anything about
it.

### Scenarios and arms

**The registry declares 19 scenarios and 73 arms.** 15 of the 19 are
cross-engine: they put megatron-core beside TorchTitan at one cut of the
model. The other 4 -- `swiglu`, `qkv`, `lm_head` and `attention` -- are
single-engine, and they do **not** share one fate. `swiglu` is recorded as
superseded by `expert_mlp` and deliberately not deleted
(`reports/20260819-partc/MERGE-half2-result.md`). Nothing supersedes
`attention`: no declared scenario cuts inner attention cross-engine, and the
`attention_core` scenario that would is in flight in another worktree and is
not in this registry. Do not write that a later commit removes the four
together; no in-tree document says so.

**Every count in this section is the count at this HEAD**, and
`attention_core` moves all of them at once -- scenarios, arms, the
cross-engine total, the comparison and correctness tallies below. Re-derive
rather than quote:

```bash
.venv/bin/python -c "from benchmarks.kernel.registry import KERNEL_SCENARIOS as K; print(len(K), sum(len(s.arms) for s in K.values()))"
```

The table below details the 4 holdovers plus `rope`. **It is 5 of the 19, and
the registry is the authority.** The paragraph after the table names the other
14.

| scenario | arms | modes | notes |
|---|---|---|---|
| `rope` | `mcore/base`*, `mcore/no_rope_fusion`, `titan`, `titan/helion`, `titan/te` | fwd, bwd | **cross-engine**: megatron's THD path eager against three compiled titan modules. `titan/te` alone needs gcc-13, and is skipped by name without it. GB/s is reported; **x-floor is not**, because the scenario declares no floor |
| `swiglu` | `baseline`*, `piper_optimized_triton`, `piper_optimized_inductor` | fwd, bwd, fwd+bwd | whole expert layer only; both Piper arms fuse the w13 GEMM and differ in the activation (custom Triton op vs plain ops left to Inductor) |
| `qkv` | `baseline`*, `fused_qkv` | fwd, bwd, fwd+bwd | weights transferred via the fused state-dict merge hook |
| `lm_head` | `baseline`*, `fused_linear_ce`, `te_fused_ce`, `piper_optimized_te_ce` | fwd+bwd | losses compiled; peak memory is the secondary metric |
| `attention` | `baseline`*, `flex_flash`, `flash_attention_3` | fwd, fwd+bwd | inner attention only, packed-document causal masking; FA3 needs the `flash3` group, `flex_flash` the `fa4` group |

**The table above is the four single-engine holdovers, plus `rope`. It is not
the whole registry.** `benchmarks/kernel/registry.py` declares 14 further
cross-engine scenarios, in partition order: `embedding_stage`, `qkv_prep`,
`qk_norm`, `attn_out_proj`, `attn_residual`, `ffn_norm`, `moe_router`,
`dispatch_permute`, `expert_mlp`, `moe_combine`, `moe_residual`, `final_norm`,
`lm_head_projection` and `cross_entropy`. Their arms are named `engine` or
`engine/profile`, except the `copy_floor` bandwidth arms. The anchor is
`mcore/base` in every one except `expert_mlp`, which anchors on `titan`
because it publishes no cross-engine row. `./run_bench.sh scenarios` prints
every scenario with its description. Read the registry rather than this table
for that half.

**No cross-engine arm has produced a number on this box yet.** Every
`results.json` under `out/` names one of the five scenarios in the table
above, and none of them contains the string `mcore/base`. Treat the 15
cross-engine scenarios as declarations until a run says otherwise.

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

**There is no TE arm in either attention scenario, and one of the two reasons
given for that has been refuted.** The file used to claim TE and the FA3
varlen path cannot share a process, citing a cuDNN soname collision. That was
measured on 2026-08-20 and **it does not hold**: both libraries import and run
real attention kernels in one process, in both import orders, dense and THD
varlen alike, under the conditions `run_bench.sh` creates. Evidence:
`reports/20260820-te-fa3-coexist.md`. So process sharing blocks nothing, and
a kernel-bench TE attention arm is not ruled out by it.

The other blocker stands, and it is an e2e blocker only: TE wraps
`DotProductAttention.forward` in `torch.compiler.disable`
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

`*` = scenario baseline. `benchmarks/kernel/registry.py` is the registry: add an
arm by appending a `KernelArm` with a builder path, and a scenario by appending
a `KernelScenario` (both declared by `benchmarks/kernel/schema.py`). Builders
live in that family's module under `benchmarks/kernel/operations/`, one per
scenario and named after it (spelled `benchmarks.kernel.operations.<scenario>:<fn>`
in a builder path), and return a `BuiltArm` whose `calls` map a mode to a
zero-argument timed closure and whose `correctness_outputs` returns named
tensors for the gates.

**The declaration is the authority, and the builder must agree with it.**
`KernelArm.modes` says which operations exist and `KernelArm.is_floor` says
whether the arm is a bandwidth floor rather than an implementation.
`_seeded_build` raises when `BuiltArm.calls` does not match the declared
modes exactly, and the merge reads `is_floor` from the registry, so
`BuiltArm` carries neither. This is what lets a reader check the roster
without a GPU: an undeclared mode would be timed and published under a label
nothing describes, and a floor known only to its builder could not produce
the x-floor column, which the parent computes.

**Which comparisons exist is declared too.** `KernelScenario.comparisons` is
a tuple of `(arm, opponent)` pairs. Left `None` -- as 7 of the 19 scenarios
leave it -- it derives the usual set: every non-floor arm against the anchor.
An explicit tuple is exhaustive, and the empty tuple declares a scenario that
publishes no ratio at all, which a scenario whose two sides are not a
like-for-like cut must be able to say. It replaces the per-arm `compare_to`,
which could redirect a row but could not decline one.

**A builder path is a string, and must stay one.** `benchmarks/kernel/engine/`
imports `schema.py`, never `registry.py`, and never an `operations/` module:
arms reach it only as already-resolved `BuiltArm` values via `resolve_symbol`.
So do not move a scenario constant next to its family's builders -- that
"colocate the family" move recreates `engine -> registry ->
operations.<family> -> torchtitan` and drags every kernel family and its model
dependencies into the engine's import graph, which is what per-arm process
isolation cannot have. `tests/test_import_boundaries.py` section 3 asserts both
halves; `tests/test_migration_contract.py` pins each scenario's builders to its
own family module.

### Method

- Module-scope arms (the three titan rope modules, the swiglu layer arms,
  both qkv arms) run under `torch.compile(fullgraph=True)`, because that is
  what they face end-to-end: eager isolation races custom ops against
  materialization costs Inductor deletes, which inverts verdicts (the
  swiglu combined layout wins eager, loses compiled). **The treatment is per
  arm, and the engine does not imply it.** An arm is eager only where
  `KernelArm.eager_reason` says why: every `copy_floor`, because a bandwidth
  floor is not an implementation; most megatron-core arms, because megatron
  compiles no whole layer; and the three titan arms whose module sits
  outside every compiled region end-to-end (`embedding_stage/titan`,
  `final_norm/titan`, `lm_head_projection/titan` -- `apply_compile` walks
  `model.layers` alone and each of those modules is a sibling of `layers`).
  Two megatron arms run the other way and declare `compiled=True`:
  `attn_residual/mcore/base` and `moe_residual/mcore/base`, whose timed
  closure calls megatron's own `@jit_fuser` function directly, so the
  compile is the engine's choice rather than the harness's.
  lm_head losses are built with the production
  `CompileConfig(components=["loss"])`. `KernelArm.compiled` records the
  treatment in the manifest. The worker sets
  `torch._functorch.config.donated_buffer = False`: retained-graph backward
  timing re-runs compiled backward graphs, which buffer donation forbids.
  This changes backward memory reuse, not the generated kernels.
- The measurand is the **burst-amortized per-call cost under back-to-back
  dispatch**: synchronize, record a start event, launch `--burst-k` calls
  back to back, record an end event, synchronize, divide by k. The former
  wall median is **gone**.
- **That number is not device time, and on several arms it is mostly not.**
  A CUDA event pair measures an interval on the stream, and the host has to
  keep the stream fed. Where the host cannot enqueue faster than the device
  drains, the interval holds the host stalls too. A burst amortizes the
  fixed per-burst synchronize; it does not amortize per-call dispatch, so
  no `k` removes them. Measured on an H200, rope forward, us per call at
  k=1/4/16/64 -- **on the retired single-engine rope roster**, whose arm
  names the cross-engine replacement no longer uses, and which nothing
  re-measures. **The cross-engine rope arms cannot be compared against these
  figures**, and not only because the names changed: two corrections reach the
  measurand. The arms now take packed-document positions where they took
  `arange`, and their timed closures now run the training graph where they ran
  an inference graph that saved no activations. Both are fixes, and both move
  the number:
  `copy_floor` 27.66/16.16/13.43/13.18, `baseline`
  125.49/88.96/76.91/73.10, `helion` 301.23/256.53/234.87/226.01, `te`
  186.64/138.87/122.60/114.40. `copy_floor` converges onto the ~11-13 us of
  device work these shapes carry, so the method works where an arm is
  device-bound; the three module arms sit 6-19x above that floor and are
  still falling at 64, so **roughly 85% of every published rope number is
  host dispatch**. `qkv` is dispatch-heavy in absolute terms too, but both
  its arms are, so the effect largely cancels in the ratio.
- **A dispatch-bound arm carries a k-dependent ratio, so its ranking is not
  a kernel result.** On the same retired rope roster, `helion` against
  `baseline` is 2.40x at k=1, 3.05x at k=16 and 3.09x at k=64. Run `--burst`; the merge derives a `residual`
  column from the top two rungs of the ladder and flags any arm whose
  per-call time is still falling there. Report a flagged row as a
  comparison of dispatch cost, never as a kernel-speed claim.
- **That test is one-sided, and an unflagged arm is not device-bound.** A
  ladder can plateau at a dispatch cost bursting cannot amortize. Rope
  backward does exactly that, again on the retired roster: `baseline` reads
  187/156/157/159 us across
  k=1/4/16/64, flat from k=4 on, against ~12 us of device work. Flat means
  `k` stopped buying amortization, not that the number became device time.
  The residual is also a difference of two medians and carries their
  noise -- the same rope forward ladder an hour apart put `copy_floor` at
  1.9% then 4.8%, and `te` at 7.2% then 14.2% -- so read a value near the
  2% threshold as undecided. **A kernel-speed claim needs profiler-summed
  device time, which nothing in this repo currently measures.**
- **OPEN QUESTION: 32 Inductor compile-worker processes run during the
  timed region, and nobody has ruled out that they contaminate every kernel
  number in this repo.** `torch._inductor.config.compile_threads` defaults to
  32 on this box with `worker_start_method="subprocess"`, so the first
  `torch.compile` in a worker starts 32 subprocesses that each `import
  torch`. They spawn during `arm_build` and are still importing torch when
  the shorter arms reach their timed region, and this workload is
  host-dispatch bound, so host jitter lands inside the measured interval.
  Setting `compile_threads=1` removes the pool. Measured at n=3, median
  us/call with the per-run standard deviation: `qkv/fused_qkv/forward`
  248.68 +/- **9.42** against 233.37 +/- **0.82**; `rope/helion/forward`
  261.27 +/- **29.07** against 244.64 +/- **6.47**. The standard deviation
  falls 3x to 11x on the dispatch-bound arms and the device-bound arm is
  unmoved -- which is what 32 concurrent `import torch` processes would do.
  The medians move in both directions at n=3, so **only the variance change
  is claimed**. **This is unresolved, and it is not resolvable by argument.**
  Three repeats is not enough, and every number above was taken on a box at
  load average 28 to 81, which is itself a source of the jitter under test.
  If the effect is real it reaches every kernel number this repo has ever
  published, because every one of them was taken with the pool running.
  **What would settle it**: a full replicate count on an **idle** box, the
  same scenario measured with the pool and without it, reporting the spread
  and not only the median. Until somebody runs that, treat the spread of any
  kernel number here as partly a property of the harness. Note that
  `compile_threads=1` is separately *rejected as a speed change* -- see
  "Startup cost" -- and that rejection is not an answer to this question.
- **One `--burst-k` for every arm in a scenario.** A per-arm k makes arms
  incomparable: a k=64 arm overlaps 64 launches with device work and a k=4
  arm overlaps 4, and the residual bias runs in the same direction as the
  effect under test. The schema records k.
- The repetition unit is the **replicate**. One replicate sweeps every arm
  once, in declaration order, and the sweep repeats, so drift that moves a
  whole replicate cancels in the ratio. **The headline statistic is a
  bootstrap CI on per-replicate log-ratios.** Welch, MWU and Cohen's d run
  on the pooled samples and are within-run distribution diagnostics only --
  consecutive bursts of one closure are correlated, so their independence
  assumption is not met. **Wilcoxon was removed**: it needed the per-cycle
  pairing the old round-robin provided, and at 5 replicates its exact
  two-sided minimum p is 0.0625, so it can never reject.
- Python's garbage collector is paused during the timed region. A collection
  starves the launch queue and lands as idle time inside whichever arm's
  interval is open; pausing it cut the swiglu module sd from ~63 us to
  ~1.4 us and removed every 2x outlier, medians unchanged.
- The first burst after the warmup synchronize is discarded (empty queue,
  systematically high).
- No L2 flush, and **no equalization is claimed**. The old text claimed
  interleaving equalized cache state across arms; timing one arm at a time
  makes that false, so it is withdrawn rather than carried. The known bias:
  an arm whose working set fits in L2 benefits from bursting more than one
  whose does not.
- Backward is measured separately wherever the arm exposes a backward entry
  point; module arms retain the graph and re-run `torch.autograd.backward`,
  so only backward kernels are timed. `lm_head` is fwd+bwd only because
  `FusedLinearCrossEntropyLoss` runs its backward inside `__call__`.
- **Every arm is timed in its own process. The gates still run in one.** A
  scenario is one correctness worker plus `replicates x arms` timing
  workers, spawned sequentially by `benchmarks/kernel/runner.py` in
  replicate-major order. (Above `--replicates-per-process 1` the sweep is
  block-major instead, and there are `blocks x arms` workers; see "Startup
  cost". No value of that flag ever puts two arms in one process.) Each
  worker writes a JSON fragment per replicate under
  `fragments/`, and the parent merges them
  (`benchmarks/kernel/results/merge.py`). The timing split is what keeps one
  arm's dependencies out of another arm's interpreter during measurement, and
  it is why the parent computes the ratios: no timing worker sees a second
  arm. **`run_correctness_pass` is the exception**: it gates *every* arm of
  the scenario in one interpreter, because 35 of the 73 arms name another
  arm as their correctness reference and a check needs both sides at once.
  One arm is resident at a time -- each is built, asked for its outputs, and
  dropped before the next is built -- so the pass is bounded by the largest
  single arm rather than by their sum
  (`benchmarks/kernel/engine/run.py`). The scenario that was expected to
  force a per-arm split -- one holding both a TransformerEngine arm and an
  FA3 arm -- does not: the two were measured to coexist in one process
  (`reports/20260820-te-fa3-coexist.md`). The split may still be wanted, but
  nothing is waiting on it.
  `benchmarks/kernel/engine/run.py`'s `run_kernel_scenario` composes the same
  two passes in a single process for the GPU smoke test; the runner never
  calls it.
- **Re-seeding is per arm build, not per process.** Inputs rebuild
  bit-identically in every worker (the inputs builder owns its generator),
  but builders consume the global RNG, so an arm built second in one process
  would otherwise get different weights than the same arm built alone.
  `_seeded_build` re-seeds before every build, so the arm the correctness
  pass gates is the arm the timing pass measures.
- Correctness runs before timing and fails the run loudly (worker exit 3).
  **No timing worker launches after a failed gate**: a gate failure is the
  result, and measuring an arm already known to be wrong wastes the GPU.
  `results.json` is still written, carrying the gates and every arm at
  `status: skipped`.
- **A lost worker never becomes a quiet number.** A timing worker that writes
  no fragment is reported as it happens and the sweep continues. Its arm then
  lands in `results.json` at `status: failed` with the reason, plus a recorded
  warning, and the scenario exits nonzero. Losing the *anchor* arm writes no
  results at all -- every comparison is a ratio against it, so the
  alternative is a table whose missing ratios look like a scenario that
  declared none.
- **Requirements belong to the arm, not to the scenario.** Without a C++20
  host compiler, rope loses `titan/te` and still measures its other four
  arms; the former scenario-level `requires_gcc_toolset` check threw away
  the whole scenario. `resolve_arm_skips` decides the set in the parent, closes it
  over correctness references (an arm whose reference is skipped is skipped
  too -- timing an arm nothing checked is the wrongness the gates exist for),
  and delivers it to the correctness worker as `--skip-arm NAME`. A skipped
  arm is spawned in neither pass. The scenario-level property survives for
  its one honest use: asking whether anything here needs the compiler at all,
  which is what decides whether `add_compiler_environment` runs. That call
  shells out to bash and is now resolved **once per run**.
- **Every declared arm reaches `results.json`, measured or not**, carrying
  `status` `ok`, `skipped` or `failed` and the reason. At schema 3 an arm
  this host could not run and an arm the registry never declared were both
  simply absent, so a reader could not tell a short roster from a complete
  one. The skip of an *anchor* is the exception that costs the scenario.

### Startup cost

**Every timing number in this section was measured on a contended box**
(load average 28 to 81, several agents on the same host). They are the
record of what was tried. **None of them is a confirmed result**, and none
may be cited. Re-measure on an idle box before you quote any of them.

Per-arm process isolation is not free. A timing worker pays about 15.7 s
before the arm it measures exists: process start, the torch import, the
engine import, and the arm build. At the default settings a scenario pays
that once per (arm, replicate). Three mechanisms were built against it and
one was dropped; what survives is the phase table, the exit path, and
`--replicates-per-process`.

**The phase table** (`benchmarks/kernel/engine/phases.py`) attributes a
worker's wall clock to named spans, and every fragment carries the table of
the process that wrote it. It is stdlib-only, it adds no synchronize, and it
touches neither the CUDA events nor the arithmetic, so the published samples
are the same numbers between the same two points. `process_start_offset()`
reads `/proc/self/stat` field 22 against `/proc/uptime`, which is what makes
`process_startup` -- the time before `main()` ran at all -- measurable from
inside the worker. Use the table to decide where a slow run spends its time;
do not read it as a measurement of the kernel.

**The exit path.** `benchmarks/kernel/worker.py` ends at `_exit_now(code)`,
which flushes both streams and calls `os._exit`. This skips the interpreter's
unwind, whose cost is dominated by joining Inductor's 32 compile
subprocesses. Four facts make it safe, and `_exit_now`'s docstring records
them: no measurement remains at that point; the fragment is already on disk,
because `atomic_write_json` writes and renames before the exit; the compile
caches are already on disk, which was checked by comparing the cache trees
byte for byte both ways (13 Inductor and 65 Triton files either way); and the
compile workers are not orphaned, because each takes `--parent` and exits when
it is reparented. `main()` still returns a code, so an importer decides its
own exit.

**`--replicates-per-process`** lets one worker measure a block of consecutive
replicates of one arm. It defaults to 1, and **1 is the value for anything
published**.

The reason is the statistic. One replicate is a sweep in time, so drift that
moves a whole replicate moves the arm and the anchor together and cancels in
the ratio. Replicates inside one process are consecutive measurements of one
build in one interpreter, so that cancellation is gone. The interval narrows
without the ratio becoming better known. Above 1, therefore:

- `results.json` publishes `within_process_ratio_ci_low`/`_high` and carries
  no `ratio_ci_low`/`_high`. The field is renamed, not reinterpreted, so a
  reader of the honest name finds nothing rather than a narrower number.
- `methodology.arm_isolation` reads `one_process_per_arm_replicate_block`,
  and `methodology.replicates_per_process` records the value.
- The printed table states the isolation on every run, marks the interval
  with `~`, and prints a warning that names the renamed fields.

**Its acceptance gate is outstanding.** The flag lives on one condition: more
than a 2x end-to-end speedup on an idle box. If the verified speedup is under
2x, remove the flag.

**Two architectural claims are contested, and the contest is unsettled.** One
agent measured that a fresh process does not reduce the spread of a
measurement, and that replicate-major pairing cancels nothing -- the
paired-ratio coefficient of variation sat above each arm's own in all three
modes, which is what independent noise predicts, because variances add.

**That is a measurement, and what follows is not.** Its own author states the
limit that makes it provisional: *"40 replicates on one arm pair in one
scenario at one shape is not enough evidence to change the default that
published numbers are taken at."* That limit needs no counter-argument and
cannot be disputed. It is the reason to record the claims as pending.

One hypothesis, labelled as one: heavy contention may also decorrelate two
arms' medians and inflate a paired CV, and both measurements were taken at
load average 28 to 81. **That mechanism is untested here**, and the same
agent's only *measured* load effect runs the other way -- an 8-replicate
sweep taken while host load fell monotonically gave correlations of +0.34 to
+0.82, so a monotone load trend *creates* common-mode correlation rather than
destroying it.

**Record both claims as pending, not as refuted.** An idle box settles them.

**Rejected, each for a stated reason:**

- **A fork pool of pre-warmed interpreters.** Its author could not
  demonstrate timing neutrality: `rope/te/backward` read 12% high pooled in
  one experiment and 2% low in another, and two dispatch-bound measurements
  that disagree about the sign are not a result. It was never run on
  `attention` (the scenario whose isolation matters most, holding both FA3
  and FA4) or on `lm_head`. It targets the same wall clock
  `--replicates-per-process` removes, and it costs a second launch path.
- **A fork server**, for the same reasons and a fortiori: it never landed,
  and its own author called it not ready.
- **`compile_threads=1`.** It removes the 32-worker pool and most of the
  teardown, but it is **not measurement-neutral**: the sample standard
  deviation fell 3x to 11x on dispatch-bound arms. A knob that changes the
  spread of the measurand cannot be set for speed. **Rejecting it as a
  speedup settles nothing about the numbers already published.** The same
  measurement is evidence about the status quo, where the pool runs during
  every timed region; that reading is an open question and lives in "Method"
  above.
- **Building an arm ahead of the previous arm's measurement.** The overlap
  shifted the measurand by 4% to 9%.
- **Caching a built arm on disk.** Same objection as the pool, with a larger
  surface.

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
- **`bitwise`** where implementations must agree exactly. Seven scenarios
  declare it (`qkv`, `embedding_stage`, `qkv_prep`, `attn_residual`,
  `moe_router`, `dispatch_permute`, `moe_residual`), and whether a given
  check *gates* depends on what it compares. It stays informational
  wherever a compiled region may legitimately round differently from an
  eager one -- the fused-vs-unfused QKV outputs are the original case -- and
  the rel_l2 gates enforce closeness there. It **enforces** on a permutation
  or a routing decision, in `dispatch_permute` and on `moe_router`'s
  `selected_count`. A permutation is a gather, so a wrong one moves right
  values to wrong places, and both engines build the permuted rows by a pure
  copy: exact equality is achievable, and it is the metric that sees a
  misplacement whatever the shape.

  **Do not justify that with the `sqrt(2/N)` figure two other files carry.**
  Swapping one pair of `N` rows gives `max_rel_l2 ~ 2/sqrt(N)`, not
  `sqrt(2/N)`. At the default `dispatch_permute` workload (`N = 8192`) that
  is 2.21e-2 measured, **above** the 2e-2 gate, not below it. So a tolerance
  gate would catch a single swap here by a 10% margin, and would stop
  catching it as `N` grows. `benchmarks/kernel/registry.py` and
  `benchmarks/kernel/operations/dispatch_permute.py` state the wrong form and
  the wrong conclusion; neither is editable from here. The bitwise gate is
  right for the reason above, not for that one.

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
  manifest.json      # schema 6: model_size, model_shape, workload, shapes, arms, skipped_arms, replicates/replicates_per_process/burst_k/warmup_calls/seed, commands, provenance
  results.json       # schema 6: every declared arm with a status and its declared compile treatment, per-mode summaries + per-replicate samples, comparisons, correctness, warnings
  kernel_bench.log   # every worker's stdout+stderr, in spawn order
  fragments/
    correctness.json         # the gate pass
    timing__<arm>__r<N>.json # one per (arm, replicate)
```

Every fragment also carries a `phases` table: the named wall-clock spans of
the process that wrote it. A worker that measures several replicates writes
one fragment per replicate, and each of them carries that process's table, so
the same table appears more than once. Inside one table the timing spans
carry the replicate index (`samples:r0:forward`), so a batched worker's
repeated groups stay distinguishable. It is provenance about the run, not
about the kernel.

Raw per-replicate samples are kept in `results.json` so a run can be
re-analyzed without re-measuring, and the fragments are kept so a merge can
be redone without re-measuring either. `model_shape` is the same `describe()`
payload the e2e manifest records, so the two systems state model identity
identically.

Schema history, and why each step renamed rather than reinterpreted a field:
both went 1 -> 2 when the flat `spec` split into `model_size` /
`model_shape` / `workload`. Both went 2 -> 3 when the measurand changed --
`n`/`warmup` counted round-robin cycles and were printed as such, so they
became `replicates` / `samples_per_replicate` / `burst_k` / `warmup_calls`,
and `samples_us` became `replicates_us` because the replicate boundaries are
the repetition unit the statistics run over. The manifest alone went 3 -> 4
when a scenario stopped being one worker invocation: `command` became
`commands`, one argv per pass. The results file went 3 -> 4 when every
declared arm started reaching it with a `status`: at schema 3 an arm this
host could not run was absent, which is indistinguishable from an arm the
registry never declared. The manifest alone went 4 -> 5 when `skipped_arms`
arrived: `arms` is the registry roster, so it names arms the host never
launched, and a manifest-only reader had to diff it against `commands` to
find them. The manifest alone went 5 -> 6 when `replicates_per_process`
arrived, because a manifest that does not state it cannot say whether its
run's intervals are across processes or within one. The results file went
4 -> 5 for the other half of the same change: above one replicate per
process a comparison row carries
`within_process_ratio_ci_low`/`_high` and **no** `ratio_ci_low`/`_high`, so
the shape of the file itself changes and a bump is required rather than an
addition. The rename is the point -- those replicates share a build and an
interpreter, so the interval is a lower bound, and a reader of the honest
name must find nothing rather than a narrower number. The results file went
5 -> 6 when it started carrying the **declared compile treatment** of every
arm (`compiled` and `eager_reason`) and the scenario's own `description`.
A cross-engine ratio compares two compile treatments and not two kernels,
and neither treatment follows from the engine: most megatron arms run eager
and two do not, most titan arms run compiled and three do not. A reader of
`results.json` holds no registry, so a row that does not name both sides
cannot be read. The fields were added to the dataclasses one
commit before they reached `to_dict`, so a schema-6 file briefly had the
shape of a schema-5 file; a round-trip test now pins both directions. The
results loader enforces exact schema equality,
so older files are rejected rather than half-read; the manifest is
write-only provenance and has no loader.

### Trace diagnostics

| script | measures | args |
|---|---|---|
| `tools/analyze.py` | Two-trace diff: device/host totals, per-kernel movers | 2 positional trace paths |
| `tools/per_block.py` | Per-compiled-region GPU time, paired by size rank | 2 positional trace paths |

Both take uncompressed Chrome traces, but runs write `.json.gz`. Decompress
first:

```bash
gunzip -c out/<...>/baseline/profiling/traces/iteration_40/rank0_trace.json.gz > /tmp/a.json
```

Three facts about trace arithmetic, each of which previously produced a wrong
published conclusion, and all of which outlive the tool that found them:

- **Summed kernel time overcounts whenever streams overlap.** Megatron runs
  5 CUDA streams to titan's 1, so summing overstates it ~6.5% -- enough to
  *invert the sign* of a cross-engine MoE expert-GEMM comparison. Compare on
  the busy (interval-union) basis whenever the arms differ in stream count;
  `analyze.py`'s `busy_union` computes it.
- **A fused component reads as zero, not as absent.** Inductor folds titan's
  RoPE into the qk-norm kernels, so a per-kernel split reports 0.0 RoPE and a
  `norm` number that is not comparable to megatron's separate norm. Never
  publish a fused component as `0.0`.
- **Megatron strips expert dims from its trace shapes.** `_GroupedLinearBackward`
  reports `Input Dims` as `[[98304,1024],[]]`, so anything classifying by shape
  misfiles 122.8 ms/step of MoE backward as a projection GEMM -- read
  autograd/module frame names before kernel names and shapes. Note also that
  `_LayerNormLinear` frames contain *both* norm and GEMM kernels.

### CUDA extension builds

`benchmarks/models/piper_qwen3/components/rope/te_rope_standalone.cu` is
JIT-built via `torch.utils.cpp_extension.load` when `te_rope_override.py` is
imported, which locates the `.cu` as its own directory sibling -- keep the two
together if either ever moves again. It
needs a C++20 host compiler; the stock one usually is not. Both the scenario
runner (`requires_gcc_toolset`) and `kernel-bench` handle this automatically
via `BENCH_COMPILER_ENV`, defaulting to `/opt/rh/gcc-toolset-13/enable`. To
import the override by hand, enable it yourself first:

```bash
source /opt/rh/gcc-toolset-13/enable
```

The sibling `components/swiglu/` and `components/lm_head/` packages are pure
Triton/PyTorch and need no compiler setup.

## Model config and override mechanisms

`--module benchmarks.models.piper_qwen3` resolves through TorchTitan's config
manager to `benchmarks.models.piper_qwen3.config_registry`, found via
`PYTHONPATH` (the runner sets it to the repo root). `--config <name>` is
`getattr(config_registry, name)`, called with any `--config-arg KEY=VALUE` pairs
as keyword arguments -- the runner always passes `size=<model-size>`.

**Why the file is still called `config_registry.py`, and must stay that way.**
The fork's `ConfigManager._load_config` takes an unrecognized `--module` down
the fully-qualified branch and tries **two** candidates in order:

```python
for candidate in (f"{module_name}.config_registry", module_name):
```

So `--module benchmarks.models.piper_qwen3` resolves on **candidate 1**,
`benchmarks.models.piper_qwen3.config_registry`, against the **unmodified**
fork -- no torchtitan patch is involved in this, unlike `--config-arg`. Rename
the file and candidate 1 fails; the manager then imports the *package* on
candidate 2, `getattr` finds no `qwen3_piper_1b*` there, and every run dies at
config load with a confusing "not found in benchmarks.models.piper_qwen3"
message. Equally, do **not** pass the full path to `config_registry` as
`--module`: it would resolve on candidate 2 instead and report a different
`module_path` in that error. The name is a contract with the fork, not a
stylistic choice. A test reimplements the two-candidate tuple and asserts
candidate 1 wins.

Because the `--module` value is imported inside the *training subprocess*,
`benchmarks/__init__.py`, `benchmarks/models/__init__.py` and
`benchmarks/models/piper_qwen3/__init__.py` all execute there. Keep all three
import-light and Torch-free.

`qwen3_piper_1b`: dim 1024, 16 layers, 16 heads / 8 KV heads, head_dim 64, flex
attention, qk_norm; MoE on every layer with 4 experts, top_k 2, inter_dim 3584,
`load_balance_coeff=None`; RoPE theta 1e6, max_seq_len 2048; vocab 151936, no
weight tying. Trains on `c4_test` (tokenizer vocab 2020) against the full
151936-row embedding, so **losses are not comparable to real Qwen3 training** --
they are a convergence sanity check only.

**Execution model: single GPU, plain bf16, no FSDP.** The ModelSpec's
`parallelize_fn` is
`benchmarks/models/piper_qwen3/parallelize.py:parallelize_piper1b`, which
delegates to the fork's `parallelize_qwen3` with `skip_dp=True` (AC and
per-block compile applied, FSDP skipped) and hard-errors at `world_size > 1`
or any `training.dtype` other than `bfloat16`. `training.dtype="bfloat16"`
puts params, grads, and optimizer states in bf16 with no fp32 masters --
matching piper's own execution and the treatment kernel-bench already gives
its modules. It is the only *framework-level* dtype mechanism in the run:
there is no mixed-precision wrapper and no autocast in `parallelize_qwen3`.
**It is not the whole dtype story.** Upstream's MoE router wraps its gate
GEMM in `torch.autocast(dtype=float32)`
(`models/common/moe.py:292`, deliberate -- the comment cites expert
load-balancing stability), so the gate runs fp32 on every layer of this
model, and autocast casts *both* operands up, which materializes an fp32
copy of the hidden state. That copy is real work in every e2e run this repo
has ever done, and nothing isolated it until the `moe_router` kernel
scenario. Read the wrapper's enforcement narrowly, as what it is: the TE RoPE arm requires bf16 activations, and an fp32 swiglu baseline
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
Megatron-LM + TransformerEngine instead of TorchTitan. Megatron knowledge lives
in two places: the driver and its data pipeline in `benchmarks/e2e/megatron/`,
and the model builder plus the submodule bootstrap in
`benchmarks/models/piper_qwen3/` (`megatron_model.py`, `mcore_profiles.py`,
`megatron_bootstrap.py`) -- the split follows the rest of the tree, where a
model definition sits under `models/` and an execution driver under `e2e/`.

**Configuration is data.** `build_model` takes a `PiperShape` for the
geometry and an `McoreProfile` for the behaviour, and adds nothing of its
own. A profile is one flat dict of `TransformerConfig` values. The registry
is torch-free and encodes torch values as names (`"silu"`, `"bfloat16"`), so
the parent can name, record and diff a profile without the ML stack. Both
`build_model` arguments are required: an omitted one builds the default
under another arm's label, which is a wrong number rather than a missing
one.

**The layer spec is derived, never written.** `build_model` hands the built
config to `get_gpt_decoder_block_spec`, which is megatron's own
config-to-spec derivation and the path its training entrypoint takes. That
function reads `num_moe_experts`, `moe_grouped_gemm` and `qk_layernorm` off
the config and turns each into a module-class choice
(`gpt_layer_specs.py:592-594`), so every setting is written once.

Do **not** go back to calling the inner factory
`get_gpt_layer_with_transformer_engine_spec` directly. It receives no config,
so those three settings must then be written a second time and kept in
agreement by hand -- which the profile used to do with a `spec_kwargs`
mapping and a `DUAL_DELIVERY_FIELDS` guard, both now deleted. Megatron
checks the agreement for `qk_layernorm` (`attention.py:1711` raises) but for
`moe_grouped_gemm` **nowhere**: both disagreement directions are silent, and
each builds one expert implementation and publishes it under the other's
label. No correctness gate can catch that, because both are numerically
right. A layer spec is a generated artifact; hand-writing it is what created
the hazard.

The harness connects only through `Arm(launcher="megatron",
validation="megatron")` and
`benchmarks.models.piper_qwen3.megatron_bootstrap` for provenance. The runner
launches `python -m benchmarks.e2e.megatron.train` with the workload sizes,
seed, profiler
schedule, and compile mode; the driver replicates the titan treatment
itself (fused AdamW on every param, titan's LR lambda, pre-clip-norm
logging, sum/valid-tokens loss, gc handling, identical torch.profiler
schedule and trace layout, titan-shaped step log lines).

Faithfulness guarantees, all verified:

- **Same model**: bare megatron-core `GPTModel` built from the same
  `benchmarks.models.piper_qwen3.shape.PiperShape` the TorchTitan config uses --
  exactly 1,066,241,024 bf16 params at `normal`, 10,528,837,760 at `huge`.
  `tools/megatron_parity_check.py [--model-size SIZE]` transfers titan
  weights into the megatron layout and matches logits on a real batch -- run
  it after touching `benchmarks/models/piper_qwen3/megatron_model.py`,
  `mcore_profiles.py`, `megatron_weights.py`, `titan_model.py`, or after
  bumping either submodule. It owns neither the build nor the map any more:
  it builds both engines through the same helpers a cross-engine kernel arm
  uses, which is what makes it their numerics check rather than a parallel
  implementation. The gate is per-shape and lives on the shape itself
  (`PiperShape.parity_gate`): 2e-2 at normal
  (measured 5.5e-3), 5e-2 at huge (measured 2.03e-2). The wider huge gate is
  bf16 accumulation, not slack, and it is evidenced rather than assumed --
  `--fp32-reference` runs the same weights in fp32 and shows titan's own
  bf16 output sits 3.25e-2 from it against megatron's 3.29e-2 (ratio 1.011),
  i.e. the engines agree with each other better than either agrees with
  fp32. The QKV grouped-interleave is proved separately and *bitwise* by
  `megatron_weights.assert_qkv_roundtrip`, so a layout bug cannot hide
  inside a widened
  gate. Never widen one without both.
- **Same data and masking**: `benchmarks/e2e/megatron/data.py` drains torchtitan's
  own c4_test dataset class (bit-identical stream to the titan arms'
  replay loader; tested) and packs each batch's rows into TE THD form with
  `cu_seqlens` at the `positions == 0` document boundaries, reproducing
  titan's block-diagonal causal flex mask. cu_seqlens are padded to a
  constant length and `max_seqlen` pinned to seq_len in both modes (static
  shapes for graph capture without changing the computation).
- **Same precision**: plain bf16 params/grads/optimizer states, no fp32
  masters, no autocast, no fp8. No recompute ever (`--ac` never affects
  this arm). **One exception, and it is symmetric: the MoE router runs
  fp32 on both engines.** Megatron asks for it explicitly
  (`moe_router_dtype="fp32"`, `mcore_profiles.py`); titan gets it from
  upstream's `torch.autocast(dtype=float32)` around the gate
  (`models/common/moe.py:292`). The precision matches. The **cost** does
  not: autocast converts its arguments, so titan materializes an fp32 copy
  of the `[B, L, D]` hidden state, while megatron's
  `RouterGatingLinearFunction` hands TE bf16 operands and asks only for an
  fp32 output (`moe/moe_utils.py:1348-1391`, written for that purpose).
  Read "no autocast" here as a statement about the *wrapper*, never about
  the model -- three passages in this file previously said it the loose way
  and one of them sent a plan to the wrong conclusion.
- **Megatron at its own best**: every fusion megatron's training entrypoint
  would enable is enabled explicitly. This matters because building
  `TransformerConfig` directly bypasses `megatron/training/arguments.py`,
  where those defaults actually live -- the dataclass defaults are `False`
  where argparse defaults them `True` (`--no-bias-swiglu-fusion` is
  `action="store_false"`, forwarded as `bias_activation_fusion`). Running
  the dataclass defaults once cost 11.9 GPU ms/step of unfused SwiGLU and
  produced a bogus engine verdict. Those flags are now the `base` profile in
  `mcore_profiles.py`, and `train.py` asserts the built config against what
  the profile *declares* rather than against a fixed all-on list. Both
  directions fail the run: a flag declared on that came out off is the old
  handicap, and a flag declared off that came out on is a variant that did
  not take. It logs `Megatron fusions: profile=<name> ...`, and the arm pins
  `_mul_silu_split` / `_permute_kernel` as trace markers.
  `gradient_accumulation_fusion` is the
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
`benchmarks/models/piper_qwen3/components/lm_head/te_*.py`, that test is
supposed to fail. The check normalizes each vendored file's local import lines
back to their upstream spelling before hashing, which is why the package move
did not require regenerating a single hash -- but it also means the rewritten
import lines must match the strings the test substitutes, character for
character.

## Bumping the TorchTitan submodule

The submodule is our fork (`JayAndJef/torchtitan`, with `pytorch/torchtitan` as
`upstream`), pinned on the `bench/torchtitan-benchmarks` branch, which
`.gitmodules` records. Pin that branch, not the fork's `main` -- `main` trails
it and does not contain every commit below. The fork carries benchmark-required
commits that upstream does not have; rebasing onto upstream must preserve them:

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
- `Give varlen metadata static shapes across steps` (`models/common/
  attention.py`) -- pads packed `cu_seqlens` to a fixed multiple with
  trailing full-offset entries and pins `max_seqlen` to `seq_len`. The
  document count varies per batch, so without it the varlen metadata changes
  shape every step; under `--compile-mode cuda-graph` that moves the captured
  inputs and forces a re-record, and the FA3 arms measured ~24 steps of
  repeated re-recording on an H200 (throughput dipping to 3k tok/s) before
  converging. It also removes a per-forward device-to-host sync.
- `--config-arg KEY=VALUE` in `ConfigManager._load_config` (`config/
  manager.py`), which calls the config function with those pairs as keyword
  arguments instead of no arguments -- what `--model-size` rides on. Without
  it the harness cannot deliver a shape to a config function at all, and the
  registry is back to one `def` per (config, size) pair.

**Also check, on every bump: `third_party/torchtitan/benchmarks/` must not
acquire an `__init__.py`.** The fork ships a `benchmarks/` directory of
markdown reports, and since the restructure that name collides with our root
package. The training subprocess runs with `cwd=third_party/torchtitan`
(`benchmarks/e2e/runner.py`, `cwd=paths.titan_dir`), and `python -m` puts the CWD at
`sys.path[0]` -- *ahead* of the repo root we prepend to `PYTHONPATH`. It
resolves to ours today only because that directory contains no `__init__.py`:
it is a namespace portion, and a regular package found on a later path entry
beats an earlier namespace portion. Add an `__init__.py` there -- which an
upstream rebase could do without anyone noticing -- and it becomes a regular
package that wins outright, silently stealing the name from the training
subprocess and breaking `--module benchmarks.models.piper_qwen3` and every
`--override.imports` path with it. The old top-level package name had no such
exposure, so this hazard is new. A regression test guards it; do not delete
that test to make a bump green. (The kernel worker is not exposed: it runs with
`cwd=<repo root>`, so our package is `sys.path[0]` there.)

`benchmarks/models/piper_qwen3/parallelize.py` additionally relies on
`parallelize_qwen3`'s
`skip_dp` kwarg and its ordering guarantee: AC, then `apply_compile` (which
emits the `(mode=...)` log line), then the early return *before* mesh
resolution and `apply_fsdp_to_decoder`. Note also that this fork applies no
*parallelism* autocast at world size 1 -- `training.mixed_precision_param`
is consumed only by `apply_fsdp_to_decoder`. That is not the same as "no
autocast": `models/common/moe.py:292` wraps the router gate in
`torch.autocast(dtype=float32)`, upstream and on purpose. Check that line on
every bump, because it decides whether the router is a precision difference
against megatron or a like-for-like one -- megatron routes fp32 too
(`moe_router_dtype`), so today it is like-for-like on precision and roughly
5x apart on bytes moved.

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
- `FusedGroupedExperts`, `silu_and_mul_op`, `silu_and_mul_forward_kernel`,
  `silu_and_mul_backward_kernel` from `torchtitan.overrides.fused_swiglu`.
  `FusedGroupedExperts` is the `expert_mlp/titan/fused_grouped_experts` arm
  (`benchmarks/kernel/operations/expert_mlp.py`), which exists so both Piper
  arms are published against the w13 fusion TorchTitan already ships rather
  than against unfused experts. It calls `silu_and_mul_op`, which calls the
  two kernels. `tests/test_swiglu.py` also imports them as the bitwise
  ground truth for the combined-layout kernels
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
  optimizer numerics. The package restructure moved `benchmarks_git_rev` again
  without changing any measurement; see "Provenance boundary" above for what an
  older directory can and cannot still be used for.
- Put investigation notes and hardware-specific results in `reports/`, which is
  gitignored. Keep them out of `README.md` and this file.
- After changing anything in `benchmarks/`, run the test suite. It is CPU-only
  and takes about two seconds.
- **Commit in single, self-contained steps, as the work happens.** One commit
  is one logical change that leaves the tree green on its own. Do not
  accumulate a whole task in the working tree and land it as one commit --
  a multi-file, multi-concern commit cannot be reviewed, bisected, or
  partially reverted, and `benchmarks_git_rev` stops being a useful label
  when a single rev spans several unrelated changes. Split by *concern*, not
  by file count: a refactor whose parts are independently true is several
  commits (e.g. the submodule/fork patch; the shape-registry data move; the
  spec split; each schema bump; the removal of a retired tool), each with its
  own tests passing. Run the suite before each one, not just at the end. If a
  step only makes sense alongside the next, they are one commit -- but that
  is the exception, not the default. Prefer landing a reviewed prefix of the
  work over holding all of it back.
- On a shared box, drive multi-cell matrices with `tools/run_matrix.sh`
  rather than a loop of `run-all`s. It refuses to start on a dirty tree
  (`benchmarks_git_rev` would mislabel the run), holds a `flock`, waits for
  a genuinely idle GPU before each cell, and runs a watchdog *during* each
  cell that flags foreign compute PIDs (any session id but its own),
  unaccounted GPU memory, and host-load spikes. A flagged cell is moved
  aside and redone on the next pass, because a contaminated run still writes
  a `results.json` and the bad numbers would otherwise be permanent. Never
  report a cell it marked `CONTAMINATED`.
