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
   claim is the span against the **sum of the scenarios it replaces**. Five
   spans and eight span arms at this rev. **No span has a builder**, so
   `--span` measures the enclosed scenarios and then fails in the span's own
   correctness worker; the declarations are the specification those builders
   must meet.

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
- **`HF_HOME` can point at a directory another user owns, and a run then dies
  before it trains one step.** The datasets library takes a builder lock inside
  the cache before it reads a row, so the arm fails with a
  `PermissionError` on `<hash>_builder.lock`. On this box `HF_HOME` is
  `/m-coriander/coriander/hf`, owned by another user. Both entry points now
  export `HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/hf-datasets}"`
  and create it: `tools/run_matrix.sh` always did, and `run_bench.sh` does
  since 2026-08-23. An explicit `HF_DATASETS_CACHE` still wins.

  **The failure is intermittent by nature, so do not read a passing run as
  proof the cache is writable.** It fires only when the config resolves to a
  cache entry that does not exist yet. A run that reuses an entry another user
  already built reads it happily and never takes a write lock. So a config
  change, a dataset bump or a new tokenizer can surface this on a box where
  every previous run worked. A manual session that calls
  `python -m benchmarks.cli` directly bypasses both wrappers and must export
  the variable itself.

## Repository map

`benchmarks/` is the **one** first-party package. Everything importable lives
under `benchmarks.`; the two argv-driven trace scripts live in `tools/` beside
the other operator scripts. The three former top-level packages are gone --
their names are listed once, in the provenance note below, and nowhere else.

| path | contents |
|---|---|
| `benchmarks/cli/` | `main.py` (the Click group, `scenarios`, and the `add_command` wiring), `e2e.py` (`run`/`run-all`/`evaluate` and their shared option block), `kernel.py` (`kernel-bench`), `rendering.py` (the `RunEvent` renderer both families share), plus `__main__.py`, which is what `python -m benchmarks.cli` runs. Commands are declared with plain `@click.command` and attached in `main.py`, so importing `main` is what populates the group. `scenarios` prints three rosters: the e2e scenarios, the kernel scenarios, and the kernel spans |
| `benchmarks/e2e/registry.py` | Scenario/arm/workload declarations, the compile-mode and AC-mode tables, `EXECUTION_MODEL` |
| `benchmarks/e2e/parallelism.py` | The parallelism run axis: `ParallelismSpec`, the `PP_SCHEDULES` registry, the four derivations and `validate_parallelism`'s seventeen rules. Parent-side and torch-free |
| `benchmarks/e2e/runner.py` | Executes and resumes a scenario; `RunRequest`/`RunResult` |
| `benchmarks/e2e/launch.py` | Builds the training subprocess command line for each arm. Three launchers: `torchtitan`, `megatron` and `megatron_stock` |
| `benchmarks/e2e/validation.py` | `validate_arm` and the `ValidationProfile` registry. Three profiles: `torchtitan`, `megatron` and `megatron_stock` |
| `benchmarks/e2e/results.py` | Evaluation, region comparison, `results.json`, and its renderer |
| `benchmarks/e2e/data/piper_qwen3.py` | Replay dataloader: drains the c4_test pipeline at init (megatron scenario) |
| `benchmarks/e2e/megatron/` | The **tuned** Megatron-LM training driver (`train.py`) and its THD data pipeline (`data.py`). It replicates the TorchTitan treatment step by step |
| `benchmarks/e2e/megatron_stock/` | The **stock** Megatron-LM arm: `megatron.training.pretrain` through `pretrain_gpt`'s own providers, with one substituted dataset provider. `flags.py` is the whole command line as data and is parent-side; `bootstrap.py`, `profiling.py` and `train.py`'s step-log shim are the three in-process substitutions that make it fit the harness without a `third_party/` edit |
| `benchmarks/kernel/schema.py` | What a kernel benchmark *is*: `KernelScenario`/`KernelArm`/`CorrectnessCheck`/`KernelWorkload`, the span types `KernelSpan`/`SpanParts`/`validate_span_parts`, plus `resolve_symbol`, `resolve_shape_and_workload` and `shape_summary` |
| `benchmarks/kernel/registry.py` | The kernel scenarios themselves (17 at this rev, 71 arms), declared with those types. Re-derive the counts; do not quote them |
| `benchmarks/kernel/spans.py` | The kernel spans. Parent-side and torch-free, exactly as the scenario registry is. `KERNEL_SPANS` holds five spans and eight span arms at this rev, none of them with a builder. It imports the scenario registry to check that each named part arm exists, which is why it is a separate module |
| `benchmarks/kernel/runner.py`, `worker.py` | Kernel-bench supervisor and the per-pass subprocess it launches, one per (arm, replicate) plus one for correctness. The runner also owns `measurement_plan`, which orders a run's units: every enclosed scenario ahead of its span |
| `benchmarks/kernel/engine/` | `arm.py` (the `BuiltArm` contract), `measurement.py` (burst timing, memory and the burst ladder), `correctness.py` (the gates), `run.py` (orchestration, and the timing pass: `build_timing_arm`/`time_replicate`/`arm_extras`, composed once in `time_replicate_block`), `phases.py` (the stdlib-only wall-clock attribution every fragment carries) and `statistics.py`. **The engine does not know that spans exist**: both passes take a `KernelScenario`, and a span hands them its own `measurement`, which is one |
| `benchmarks/kernel/operations/` | Arm builders, one module per scenario and named after it, plus `common.py` |
| `benchmarks/kernel/results/` | `schema.py` (kernel `results.json`), `merge.py` (parent-side assembly of the workers' fragments, for a scenario and for a span), `span_statistics.py` (the span-versus-parts estimator, parent-side because one of its two sides is a sum over units the engine never sees) and `reporting.py` |
| `benchmarks/models/piper_qwen3/shape.py` | `PiperShape` + the four-entry `PIPER_SHAPES` registry (`normal`, `large`, `huge`, `giant`, in ascending parameter count); both engines' single source of geometry |
| `benchmarks/models/piper_qwen3/config_registry.py` | The `--module benchmarks.models.piper_qwen3` config port; all registered `--config` names |
| `benchmarks/models/piper_qwen3/parallelize.py` | The ModelSpec `parallelize_fn`: plain bf16, and `fully_shard` only where the delivered mesh asks for a data-parallel degree |
| `benchmarks/models/piper_qwen3/mcore_profiles.py` | Megatron behaviour as data: one `McoreProfile` per variant, torch-free and parent-side |
| `benchmarks/models/piper_qwen3/megatron_bootstrap.py` | Megatron location/provenance and the TE environment setup |
| `benchmarks/models/piper_qwen3/megatron_model.py` | The Qwen3-1B megatron-core `GPTModel` builder; takes a shape, a profile, and the layer parts to leave out |
| `benchmarks/models/piper_qwen3/megatron_weights.py` | The titan-to-megatron per-parameter map, tagged by component so a caller can take a slice |
| `benchmarks/models/piper_qwen3/titan_model.py` | The in-process titan build, and the override count that replaces the `[Override]` log check |
| `benchmarks/models/piper_qwen3/components/rope/` | TE RoPE override + `te_rope_standalone.cu` |
| `benchmarks/models/piper_qwen3/components/swiglu/` | Combined-SwiGLU Triton kernels and override |
| `benchmarks/models/piper_qwen3/components/lm_head/` | Vendored TE cross-entropy, Piper-optimized CE, losses |
| `benchmarks/traces/` | `schema.py` (the `Region` declaration) and `extraction.py` (trace parsing, window and region pooling) |
| `benchmarks/artifacts/` | On-disk artifacts: `layout.py` (output layout, `trace_files`, `atomic_write_json` -- the only JSON writer), `manifests.py` (the manifest schema and the resume predicate; the one module here coupled to `e2e/`), `run_state.py` (the per-arm ledger) and `summaries.py` (`SampleSummary`, shared by both systems) |
| `benchmarks/execution/` | Subprocess execution: `paths.py` (`BENCH_DIR`/`TITAN_DIR`, `RuntimePaths`), `devices.py` (`parse_devices`, the `<gpu>` positional read as a device set), `environment.py` (the child's env vars), `affinity.py` (NUMA pinning, one node per device), `provenance.py` (`hardware_metadata`, including the two cuDNN fields -- see "Which cuDNN a megatron arm runs"), `events.py` (`RunEvent`, `ProcessRunner`) |
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
records the old module names, and they are still on disk: 142 `manifest.json`
files under `out/` record `"module": "piper1b"`, 55 record
`benchmarks.kernel_arms:<builder>` kernel-arm paths, and 21 record
`python -m megatron_baseline.train` in `commands`. Those three counts are
fixed, because no new run writes a retired name. The **total** number of
manifests under `out/` is not fixed, grows with every run, and is
deliberately not stated here.


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

**One directory there holds two kinds of file, and the difference is
labelled.** `tests/fixtures/legacy/e2e_results3/` is the `baseline` arm of
`out/20260807T175156Z/piper1b_qkv/nvidia-h200`. Its `manifest.json` and
`results.json` are verbatim copies, like every other fixture here. Its two
trace windows are a **projection** of that run's real traces: every event
that reaches a state-mutating branch of `trace_window_metrics` is kept,
carrying only the fields those branches read, which takes each window from
about 35,450 events and 1.0 MB to 11,775 events and 175 KB. A full trace
cannot be checked in, and a synthetic one would prove nothing about a real
run. The projection is not taken on trust: `tests/test_parallel_traces.py`
re-evaluates the arm from those traces and compares field by field against
the verbatim `results.json`, so a dropped event that mattered fails the
test. Every one of the four files carries a digest in
`tests/test_retired_paths.py`, and none of them may be edited.

## End-to-end scenarios

### Commands

```bash
./run_bench.sh scenarios                    # list scenarios and arms
./run_bench.sh run <gpu> --scenario NAME [--arm NAME]... [OPTIONS] [-- TORCHTITAN_ARGS]
./run_bench.sh run-all <gpu> --scenario NAME [OPTIONS] [-- TORCHTITAN_ARGS]
./run_bench.sh evaluate <out_dir> [--arm NAME]... [--results PATH]
```

`<gpu>` is a PCI index, or a comma-separated set of them (`0,1`). The runner
always sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, so the index is stable, and it
sets `NGPU` to the requested world size. `benchmarks/execution/devices.py`'s
`parse_devices` is what says whether the string is a legal device set; it
accepts decimal indices separated by single commas and refuses everything
else, including a repeat. **A `GPU-<uuid>` is refused**, which both
`nvidia-smi --id=` and `CUDA_VISIBLE_DEVICES` accept and which is the only
spelling that selects a MIG instance -- widen the grammar before running on
a MIG host. The string itself is never rewritten: it reaches
`CUDA_VISIBLE_DEVICES` and `hardware_metadata.requested_gpu` exactly as
typed. `kernel-bench` refuses more than one device.

`run 0,1` used to run **one** GPU -- `NGPU` was 1, so training took the
first visible device while the manifest recorded `"0,1"`. That is a wrong
recorded fact rather than a missing one, and it now fails.

**`run 0,1 --pp 2 --pp-schedule 1F1B --compile-mode default --ac none`
starts two ranks, on both engines.** `_resolve_run`'s blanket refusal of
every world size above 1 is gone. What refuses an unimplemented mesh is the
seventeen rules of `benchmarks/e2e/parallelism.py` plus the engines
themselves, and each failure lands on the module that owns the missing work:
`parallelize_piper1b` refuses a tensor or context degree, a dropped
shard-degree flag and a mesh that replicates and shards at once, and the
megatron driver refuses a schedule it does not implement (it runs `1F1B`
alone). A bare `run 0,1` still dies at parallelism rule 1, which
compares `dp * pp` against the device count: "parallelism world size 1 (dp 1
x pp 1) does not match the 2 device(s) requested". That lands before any
host probe.

**Seven individual-arm multi-rank runs have executed and passed
`validate_arm`.** On 2026-08-23 `piper1b_megatron` ran `baseline` (megatron)
and `titan_stock` at each of `pp2`, `dp2`, and `dp2 x pp2`; on 2026-08-24
`piper1b_rope/baseline` ran at `dp2` on TorchTitan. The `pp2` traces are the
evidence behind the NCCL classifier and arm rule 6 discussion below. The
data-parallel runs additionally printed the same `grad_norm` on ranks that
read different slices, which is stronger evidence of gradient reduction
than a trace marker alone.

**What that does not cover**: no `run-all` or `--resume` at a mesh, no
evaluation across several parallel arms, and no repeated cell on an idle
host. The runs are correctness gates, not citable timings. Arm rule 7 has
run under `fully_shard` for one arm of one region-declaring scenario; the
other four such scenarios and every override arm remain unmeasured under a
data-parallel degree.

**How each engine starts its ranks differs, and only one of them is the
harness's own work.** The titan arms run `./run_train.sh`, which already
calls `torchrun --nproc_per_node=${NGPU} --local-ranks-filter ${LOG_RANK}
--role rank --tee 3`; `benchmarks/execution/environment.py` sets `NGPU` to
the spec's world size and, **above one rank only**, sets `LOG_RANK` to every
rank and `TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE` to `[rank${rank}]:`. The
megatron driver has no such wrapper, so `benchmarks/e2e/launch.py` builds
the launcher itself: `python -m torch.distributed.run` with the same flags.
**At one rank it builds neither**, so the megatron argv stays
`python -m benchmarks.e2e.megatron.train ...`, token for token.

- `run` executes and validates only. `run-all` also evaluates and writes
  `results.json`.
- Both need a `--scenario`. There is no default, and an omitted one fails the
  run. `run-all` is the one exception, and only when `--all-scenarios` or
  `--resume` supplies the name instead.
- `run` accepts repeatable `--arm NAME` options to execute an ordered subset;
  omitting them executes every arm. Duplicate and unknown names fail before a
  host probe or output-directory creation. `run-all` does not accept the
  selector; it always runs every arm in the scenario.

- `run-all` accepts `--resume <out_dir>`; `--resume` and `--out` are mutually
  exclusive.
- `run-all --all-scenarios` sweeps every scenario in sequence, sharing one
  timestamp so they group under `out/<timestamp>/`. It is **fail-fast**: the
  first failing arm aborts the sweep and later scenarios never run. It cannot be
  combined with `--scenario`, `--out`, `--resume`, or `--results` (note `--out`
  also trips on an exported `OUT`). Seven scenarios hold 22 arms; 15 of them
  accept `--ac sac`, because both megatron scenarios are `none`-only. Budget
  roughly 45 minutes for the `--ac none` sweep and 35 for the `sac` one, and
  re-derive the arm counts from the registry rather than quoting them.

Shared options, with env equivalents:

| flag | env | default |
|---|---|---|
| `--scenario` | -- | **none; required** (see below) |
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
| `--dp` | -- | 1 |
| `--pp` | -- | 1 |
| `--ep` | -- | 1 |
| `--pp-schedule` | -- | none |
| `--pp-microbatch-size` | -- | 1 |
| `--dense-sharding` | -- | `replicate` |
| `--megatron-p2p-sync` | -- | `on` |
| `--megatron-nan-guard` | -- | `on` |

**The six parallelism options take no environment variable, and the three
axes above them do.** Each parallelism value has to agree with the `<gpu>`
positional, and a positional has no environment form; an exported `PP=2`
would make a plain `run 0 --scenario X` fail its own world-size check. The
degrees, the schedule registry and the seventeen rules that refuse an illegal
set live in `benchmarks/e2e/parallelism.py`; read that module, not this
table, for what a combination means. `--megatron-p2p-sync` and
`--megatron-nan-guard` have no environment variable for a different
reason: each changes a Megatron treatment, and an exported value would
reach every megatron cell of a shell session and move a recorded fact
without a flag on the command line.

**The budget is `MAX_WORLD_SIZE = 8` and `MAX_PP = 8`**
(`benchmarks/e2e/parallelism.py`). `MAX_WORLD_SIZE` went from 4 to 8 for
the `piper_megatron_stock` suite, which runs `--dp 2 --pp 4` on eight
devices. `MAX_PP` went 2, then 4, then 8: the last lift is for the
depth-8 pipeline, the deepest split eight GPUs can hold, and `pp 8`
forces `dp 1`. Neither number is a property of an engine. Both declare
what somebody plans to run, and lifting either is one edit at those two
names.

**The cap lift made rule 2's own pipeline half unreachable, and the
repair is an order swap rather than a deletion.** The world size is
`dp * pp`, which is never less than `pp`, so once `MAX_PP` equals
`MAX_WORLD_SIZE` every spec above the pipeline cap is also above the world
budget. Rule 2 now tests the pipeline degree first, so a `pp 9` spec reads
the message that names the cap somebody has to lift rather than the
world-size message. Both halves refuse every spec they refused before; the
rule chooses the more specific of two true messages.

The earlier `pp 2` cap gave a false reason -- that the two engines count
layers the same way only at `pp <= 2`. They agree at every degree, because
`benchmarks/e2e/launch.py` always sends
`--parallelism.pipeline-parallel-first-stage-less-layers 0` and its `last`
twin; without those TorchTitan splits 16 layers over 4 stages as
`[4, 5, 4, 3]` where Megatron gives `[4, 4, 4, 4]`. Read the comment block
above the two constants for the measured splits.

**One recorded gap widens with the cap, and it is not closed.**
`benchmarks/e2e/results.py`'s `loss_visible_rank` is
`(world_size // pp) * (pp - 1)`. That is right for the two schedules this
repo runs and wrong for the two V-shaped ones, `ZBVZeroBubble` and
`DualPipeV`, which give rank 0 the last stage. Spec rule 5 refuses a
V-shaped schedule only beside a megatron arm, and spec rule 6 narrows the
exposure without closing it: both V-shaped schedules set
`requires_uncompiled`, so such a run also has to ask for `--compile-mode
none`. The lift to `MAX_PP = 8` adds four `(dp, pp)` pairs to the gap --
`pp` 5, 6, 7 and 8, each at `dp 1`, because nothing else fits eight ranks.
**No run has ever used a V-shaped schedule.** Read the comment above the two constants before you
run one, and repair `loss_visible_rank` rather than the cap.

**`--dense-sharding {replicate,shard}` names how the run holds the DENSE
parameters** -- every parameter that is not a routed expert weight.
`replicate` gives each rank a whole copy; `shard` splits one copy between
the ranks of the data-parallel axis. It defaults to `replicate`, which is
the treatment every number this repo has published was measured under, and
it is a **comparability boundary**: the manifest records it inside the
`parallelism` record, so `--resume` gates it for free and two parities are
never one series. It is a field on `ParallelismSpec`, not a run axis of its
own, because it is a treatment of the data-parallel axis exactly as
`--pp-schedule` is a treatment of the pipeline axis.

**The value is what makes an expert degree legal, and the reason is a
TorchTitan constraint rather than a preference.** TorchTitan cannot split
the experts while it keeps the dense parameters replicated:
`apply_fsdp_to_decoder` sends every non-expert parameter to `Shard(0)` on
the dense mesh, and the expert mesh degree is
`efsdp = dp_shard * cp * tp // ep`, which needs `dp_shard >= ep`. Megatron
holds either parity. So the two engines compare under an expert degree only
when **both** shard, and three rules enforce it:

- **Spec rule 14** refuses `ep > 1` under `replicate`, and names
  `--dense-sharding shard` as the repair. It used to refuse every `ep > 1`.
- **Spec rule 15** refuses `shard` at `dp 1`. The shard degree is 1 there
  whatever the flag says, so the manifest would record a parity the run did
  not have.
- **Spec rule 16** refuses both the sharded parity and an expert degree to
  the **tuned** megatron driver, which implements neither. `run --arm`
  narrows the engine set, so a TorchTitan-only subset passes.

**What each value delivers.** TorchTitan gets
`--parallelism.data-parallel-replicate-degree` and its shard twin from
`titan_mesh`, which returns `(dp, 1)` under `replicate` and `(1, dp)` under
`shard`, at **every** expert degree -- it reads the declared parity and
never infers one from `ep`. The stock Megatron arm gets nothing extra under
`replicate` and five flags under `shard`: `--use-megatron-fsdp`,
`--megatron-fsdp-version 1`, `--data-parallel-sharding-strategy
optim_grads_params`, `--use-distributed-optimizer` and `--ckpt-format
fsdp_dtensor`. Two of the five restate a Megatron default on purpose, so a
submodule bump that moved either default changes a recorded argv rather than
a silent run.

**The two engines still reshard differently, and no flag here removes it.**
TorchTitan's `get_fsdp_reshard_after_forward_policy` returns `not
pp_enabled` at the default policy, so under a pipeline TorchTitan keeps the
gathered parameters through the step where Megatron-FSDP reshards. **State
that beside every sharded number.** A titan peak-memory figure that barely
moves under `shard` is expected; a **Megatron** figure that does not move is
a real failure.

**`results.json` records no run axis at all, so two cells of one mesh are
indistinguishable without their manifests.** The file carries no
`compile_mode`, no `ac_mode`, no `model_size`, no `parallelism` and no
`execution_model`. Cells 1 and 2 of the planned matrix differ in **nothing
else** -- same scenario, shape, mesh, compile mode and ac mode -- so anyone
who tables two `results.json` files side by side pools a replicated cell
with a sharded one and reports the difference as noise. **Carry the
manifest with every sharded number.**

**The gap is older and wider than this axis. `dense_sharding` is the fourth
axis it covers, not the first -- but it is the one where it bites
hardest.** The other three announce themselves in the numbers: a different
`model_size` changes the parameter count, `cuda-graph` changes
`launch_count` by orders of magnitude, and `--ac none` moves peak memory by
about 2.5 GiB. A reader who pooled two of those would see something is
wrong. A sharded cell is expected to differ from its replicated twin only
slightly, so pooling those two looks like a repeat measurement and the
difference reads as noise. That is why the caption matters here and not
merely for tidiness.

**The sharded parity cannot be built under a pipeline, and this is
measured.** On 2026-08-28 a `--dense-sharding shard --dp 2 --pp 4` cell
died on all eight ranks in 20 seconds, inside `einops.rearrange`, before
Megatron built the wrapper. **Spec rule 17 now refuses that combination
parent-side**, so no GPU is claimed for it.

Megatron-FSDP factors the **global** world size into terms that carry no
pipeline degree. Both of its mesh builders do it:

```
mcore_fsdp_adapter.py:810   "(dp_cp ep tp) -> ep dp_cp tp"
mcore_fsdp_adapter.py:739   "(outer_fsdp_dp fsdp ep tp) -> ep outer_fsdp_dp fsdp tp"
```

The product must equal the world size, so both hold only at `pp` 1. At
`dp 2, pp 4` the product is 2 against a world of 8, and the missing factor
is exactly `pp`.

**It blocks every model, not only a mixture of experts.** The failing call
is the unconditional one at `:455`, which builds the dense mesh. Only the
expert mesh at `:445` is gated on `num_moe_experts is not None`, and the
error reports `ep: 1`, which that gated call cannot produce -- it passes
`ep_size=ep_group.size()`. The HSDP builder omits `pp` too, so
`--outer-dp-sharding-strategy` is no escape.

**Our five flags are not the fault, and Megatron parses all of them.** The
run reaches `use_megatron_fsdp=True, megatron_fsdp_version=1,
data_parallel_sharding_strategy='optim_grads_params',
use_distributed_optimizer=True` and dies one layer lower, building the
mesh. A submodule bump that gives both patterns a pipeline term removes
rule 17.

**The reachable sharded mesh is `--dp 8 --pp 1`**, where the product is 8
against a world of 8, and `--dp 8 --pp 1 --ep 2` gives `4 x 2 x 1`. That
is a different mesh from every `pp 4` cell, so **never divide a number
taken under it by a `pp 4` number.**

**Beyond that refusal, nothing has run.** No sharded cell and no expert
cell has completed on a GPU on either engine. The value, the remaining
rules, the flags and the log markers are declared and tested on the CPU;
read them as a specification until a `results.json` says otherwise.

**`--megatron-p2p-sync {on,off}` names whether Megatron synchronizes the
device after each batched pipeline send and receive.** It maps to
`ModelParallelConfig.batch_p2p_sync`. `on` is stock Megatron: the field
defaults to `True`, and stock Megatron exposes no flag for it. `off` skips
the `torch.cuda.synchronize()` that `p2p_communication.py` runs when
`batch_p2p_comm and batch_p2p_sync` holds. The value reaches the two
megatron launchers alone -- `--batch-p2p-sync off` for the tuned driver,
`--bench-batch-p2p-sync off` for the stock one -- and a TorchTitan argv is
the same under either value. Two refusals land in `_resolve_run`, before
any host probe: `off` at `pp` 1, because there is no pipeline message to
synchronize, and `off` in a run that selects no megatron arm, because the
value would reach nothing. `run --arm` narrows the engine set, so a
megatron-only subset passes, and `run-all --all-scenarios` skips a
scenario with no megatron arm under `off`. The manifest records the value
as `megatron_p2p_sync`, and `--resume` gates it. It is not part of
`execution_model` and not a field of `ParallelismSpec`: it is a treatment
of the pipeline messages, not a degree. **Every number this repo has
published was measured at `on`.** The measured effect of `off`, and the
caveats that go with it, live in `reports/20260901-p2p-sync-ab.md`; read
that report before you cite a number taken under `off`.

**`--megatron-nan-guard {on,off}` names whether stock Megatron checks
every loss and every gradient bucket for NaN and Inf.** It maps to
Megatron's own `check_for_nan_in_loss_and_grad`. `on` is stock Megatron,
and the field defaults to `True`. `off` sends Megatron's own
`--no-check-for-nan-in-loss-and-grad` to the stock launcher, so a stock
user can type the same argv; there is no `--bench-` flag for it. At
Megatron-LM 59b72fa5 that one field gates two host waits: `pretrain_gpt.py`'s
`loss_func` evaluates the loss twice per microbatch through
`rerun_state_machine.validate_result`, and `training.py` copies the field
into `ddp_config.check_for_nan_in_grad`, under which
`param_and_grad_buffer.py`'s `check_grads` evaluates every bucket's
gradient norm twice per step. Each evaluation reads a device bool and
synchronizes the stream. `--rerun-mode disabled`, which the stock argv
already sends, removes neither: `validate_result` still evaluates the
rejection function under `RerunMode.DISABLED`. Four tests pin those facts
against the submodule source.

The value reaches the stock megatron launcher alone. The tuned driver
(`benchmarks/e2e/megatron/train.py`) never calls `validate_result` and
has no guard under either value, so `off` is refused whenever that arm is
selected; a run with no stock megatron arm is refused too, because the
value would reach nothing. Both refusals land in `_resolve_run` before
any host probe, each names its repair, and `run-all --all-scenarios`
prints the same reason and skips the scenario. Unlike the p2p option it is
legal at every mesh, because the guard runs at `pp` 1 and at `dp` 1. The
manifest records the value as `megatron_nan_guard` (schema 14), `--resume`
gates it, and schema <= 13 directories read as `on`. It is not part of
`execution_model` and not a field of `ParallelismSpec`.

**Every number this repo has published was measured at `on`.** The effect
of `off` was measured on 2026-09-05 (`reports/20260905-host-sync-ab.md`)
at the `1b` shape, `--dp 1 --pp 4`, batch 32, four H200s, with the p2p
sync already off: the NaN guard alone is worth **+12.1% in tokens/s** on
the stock megatron arm (condition D against C, two clean cells each), and
it is almost the whole of the +12.9% the report's headline condition gives.
**A stock arm under `off` is "stock minus the NaN detector", and a report
must say so beside every such number**: it is a fifth deliberate
difference on top of the four in "The stock Megatron arm". The 2026-08-30
report refused to send the flag until a validation rule caught a NaN loss;
that rule now exists on the evaluation side and runs under both values --
see "Evaluation" below.

**`--scenario` has no default, and an omitted one fails the run.** A default
scenario can only be reached by an omission, and it would then measure one
scenario under whatever label the operator assumed -- a wrong result rather
than a missing one. The same argument governs `build_model` in
`benchmarks/models/piper_qwen3/megatron_model.py`. The rule lives in **one**
place, `_resolve_run` in `benchmarks/e2e/runner.py`, and no name is written
anywhere as a default. Two callers legitimately pass no `--scenario`, and
both supply the scenario themselves: `run-all --all-scenarios` names each
scenario in turn, and `run-all --resume` reads the name from the manifest.
`RunRequest.scenario_name` therefore stays `str | None` with a `None`
default, which means "not requested" exactly as `--compile-mode`, `--ac` and
`--model-size` do.

### Compile modes

`--compile-mode` picks the compile treatment for the whole run -- every arm
in it, any scenario. E2e only; `kernel-bench` has no such flag and always
compiles at the default mode. Since manifest schema 8 the axis is
engine-neutral. It has three values, and `default` remains the default:

| mode | TorchTitan arms get | megatron arm gets |
|---|---|---|
| `default` | per-block `torch.compile`, mode default | TE modules uncompiled and uncaptured (its `@jit_fuser` regions still compile) |
| `cuda-graph` | per-block `torch.compile(mode="reduce-overhead")` | Megatron's local per-layer partial graphs |
| `none` | no `torch.compile` at all: no per-block compile, no compiled loss | **unsupported whenever the arm is selected** |

**`none` answers "what does per-block compile buy end to end?", and nothing
else did.** Every titan number this repo published before it is compiled per
block. The mode omits `--compile.enable`, because `CompileConfig.enable` is
`False` in the fork and there is no negation to pass; every compiled mode
builds the command line it built before the mode existed.

Two things change under `none`, and both are inversions rather than
relaxations. **The run declares no regions**, because region pooling reads
Inductor's compiled-graph annotations and an eager run emits none -- the same
honest reason the 1-layer shapes and `piper1b_megatron` declare none. So rule
7 guards nothing here, and rules 8, 10 and 11 do. **Rule 8 reads the other
way**: `validate_arm` requires the compile log line to be *absent*, so a run
that silently compiled cannot be published as eager.

**The complete `piper1b_megatron` roster declines `none`**, through
`Scenario.supported_compile_modes`, exactly as it declines `--ac sac`. An
explicit `run --arm ...` subset may use `none` only when every selected arm is
TorchTitan; this is how one output directory measures `baseline` plus
`titan_stock` at `default` and another measures only `titan_stock` at `none`
without mislabelling Megatron. Selecting any Megatron arm keeps the hard
error. The mode names a titan treatment -- whole-block `torch.compile` -- and
Megatron never has one, so there is nothing to turn off. Turning megatron's
own fusion off instead was measured on 2026-08-22 and rejected:
`@jit_fuser` binds `torch.compile` at decoration time and `import
megatron.core` already imports six consumers, so a later `disable_jit_fuser()`
leaves `bias_swiglu`, `swiglu` and `weighted_swiglu` as dynamo wrappers. No
import order of ours wins, TransformerEngine's hand-written kernels would
remain anyway, and handicapping megatron to match a titan treatment is the
mistake recorded below, where fusions off cost 11.9 GPU ms/step.

**`piper1b_attention/flex_flash` cannot honor `none`, and validation says
so.** `kernel_options={"BACKEND": "FLASH"}` is an Inductor lowering hint, so
without compile the arm runs the same eager FlexAttention as `baseline`. Rule
6 fails the arm on its absent FA4 marker. The scenario is not restricted,
because its other two arms are honest uncompiled measurements.

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
**both megatron scenarios** support only `none` (Megatron's recompute
options are not parity with per-op SAC, and a megatron arm always runs
without recompute -- `--ac` never affects one). `Scenario.supported_compile_modes` is
its twin on the compile axis, and `piper_megatron_stock` narrows it to
`default` alone. `run-all --all-scenarios` skips a scenario that
declines either mode; a direct `--scenario` request errors.

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

**Seven shapes are registered**, in ascending order of the parameter count.
`normal` is the retired name of `1b` and still resolves to it, through
`MODEL_SIZE_ALIASES`; the manifests of every run before schema 9 record
`normal`.

| | `1b` | `large` | `9b` | `huge` | `giant` | `30b-a3b` | `48b` |
|---|---|---|---|---|---|---|---|
| real or synthetic | **real** | synthetic | **real** | synthetic | synthetic | **real** | **real** |
| dim | 1024 | 4096 | 2048 | 12288 | 16384 | 2048 | 4096 |
| n_layers | 16 | 4 | 24 | 1 | 1 | 48 | 32 |
| n_heads / n_kv_heads | 16 / 8 | 64 / 32 | 32 / 8 | 192 / 96 | 256 / 128 | **32 / 4** | 32 / 8 |
| head_dim | 64 | 64 | 64 | 64 | 64 | **128** | **128** |
| MoE inter_dim (3.5x dim unless written) | 3584 | 14336 | 7168 | 43008 | 57344 | **768** | 14336 |
| experts / top_k | 4 / 2 | 4 / 2 | 8 / 2 | 4 / 2 | 4 / 2 | **128 / 8** | 8 / 2 |
| vocab / rope theta | 151936 / 1e6 | 151936 / 1e6 | 151936 / 1e6 | 151936 / 1e6 | 151936 / 1e6 | 151936 / 1e6 | 151936 / 1e6 |
| param_count | 1,066,241,024 | 4,264,661,504 | 9,330,201,600 | 10,528,837,760 | 17,058,349,184 | 30,532,122,624 | 47,685,316,608 |
| dense / sparse / active | 361,532,416 / 704,708,608 / 713,919,488 | 1,446,023,680 / 2,818,637,824 / 2,855,375,360 | 874,091,520 / 8,456,110,080 / 2,988,413,952 | 4,187,000,960 / 6,341,836,800 / 7,357,943,936 | 5,783,994,496 / 11,274,354,688 / 11,421,204,608 | 1,528,510,464 / 29,003,612,160 / 3,353,032,704 | 2,587,111,424 / 45,098,205,184 / 13,862,449,152 |
| num_flops_per_token @1024 | 3,551,348,736 | 13,599,599,616 | 16,667,473,920 | 33,096,721,152 | 53,792,637,696 | 20,667,125,760 | 81,051,328,512 |
| per-block regions | yes (80/80) | yes (20/20) | yes (120/120) | **no** | **no** | yes (240/240) | yes (160/160) |
| `parity_gate` | 2e-2 | 3e-2 | 2e-2 | 5e-2 | 6e-2 | 2e-2 | 3e-2 |
| measured? | yes | **no** | **no** | yes | **no** | **no** | **no** |

**Four of the seven are real piper models, and three are benchmark
inventions.** `1b`, `9b`, `30b-a3b` and `48b` are transcribed field for
field from the piper checkout's `examples/models/qwen3.py`. `large`, `huge`
and `giant` are ours: each took a dim and a layer count for a benchmark
reason, then applied the piper-1B rules to everything else. **A synthetic
shape is not piper at scale.** Piper holds `n_heads` at 32 and `n_kv_heads`
at 8 from 9B up, which is 4:1 grouped-query attention; the synthetic shapes
pin `head_dim` at 64 and take the derived `n_heads = dim/head_dim`, so
`huge` carries 192 query heads over 96 kv heads. Real piper never approaches
that. `large` and `48b` share a dim and an expert width and agree on nothing
else, so `large` does not approximate `48b`.
`benchmarks/models/piper_qwen3/shape.py` is the authority; read its module
docstring before you cite a shape.

**`30b-a3b` is Qwen3-30B-A3B as Piper's registry declares it, and Piper
never ran it.** Its geometry is the model's: 32 heads of 128 at dim 2048,
so `n_heads * head_dim` is twice `dim`; 128 experts of width 768 at top-8,
where 3.5x dim would be 7168. Those two values are why `n_heads` and
`moe_hidden_dim` became fields. The numbers in its column are the
tensor-by-tensor helper's in `tests/test_model_shape.py`, and the
2026-09-05 selection report's total and active agree with them. **It is
unmeasured and unparity-checked** -- no e2e scenario, no kernel scenario
and no `tools/megatron_parity_check.py` run exists at it, so its
`parity_gate` is the default and not a measurement. **TorchTitan's RoPE
cache still caps it at seq 2048**: `config_registry.py` sizes
`CosSinRoPE` from `shape.max_seq_len`, which stays at the 2048 default
rather than the 262144 Piper declares, and nothing here widens that cache.
It does not fit one H200 (227.5 GiB of state at titan's 8 B/param). It
divides evenly at `pp` 4 and `pp` 8, and its experts divide every expert
degree eight GPUs hold.

**Five of the seven have never run.** `large`, `9b`, `giant`, `30b-a3b` and
`48b` have no scenario, no parity check and no `results.json`. Three
consequences follow, and each is an open question rather than a setting:

- **Four parity gates are unverified.** `large` and `giant` are fitted, not
  measured: the two measured shapes fit `rel_l2 = 5.5e-3 * sqrt(dim/1024)` to
  within 7%, and each gate sits above that prediction by the margin `huge`
  keeps over its own measurement. `9b` and `30b-a3b` take the default 2e-2,
  and `48b` 3e-2. Run `tools/megatron_parity_check.py --model-size <name>`
  before any parity claim at any of them.
- **Validation rule 7 above `1b` is untested and could collide.** Regions are
  derived per shape, so `large` asks for 4 layers x 5 active steps = **20**
  invocations per window, `9b` asks for 120, `48b` for 160 and `30b-a3b` for
  240. The uniqueness argument behind rule 7 was measured on a 16-layer
  trace, where the forward graphs ran {5, 80, 5} times and the backward
  graphs {5, 80}; 80 is unique there. Nobody has looked at a 4-layer, a
  24-layer, a 32-layer or a 48-layer trace. If another same-phase partition
  runs the same number of times, `pooled_window_metrics` raises and the arm
  fails rule 7. Treat such a run as unproven on that rule until a trace says
  otherwise.
- **Memory is unproven above `huge`.** `giant` is declared from a memory
  estimate, so its first run can run out of memory. `9b` may fit one H200;
  `30b-a3b` and `48b` cannot, and the `PIPER_30B_A3B` and `PIPER_48B`
  constants carry that arithmetic.

`large` is 4 layers rather than 1 for the reason `huge` is 1 rather than 16,
applied in the other direction: at dim 4096 the layer-to-table ratio is 1.65,
so a 1-layer model would be 62% embedding table and the benchmark would
measure the lm_head and the cross entropy. Four layers put `n_layers * dim`
at 16384, which is the product `1b` carries, so `large` reproduces the `1b`
parameter split and keeps `supports_block_regions` True. `9b` and `48b` do
not hold that product, because piper never chose it.

`dim`, `n_layers`, `head_dim`, `n_kv_heads` and `num_experts` are **fields**,
because the registered shapes disagree about each of them. `n_heads` and
`moe_hidden_dim` are **fields with a derived default** since 2026-09-05:
left unset they take `dim/head_dim` and `dim*7/2`, which every shape
registered before then satisfies, and `30b-a3b` writes both. `top_k` is a
default that `30b-a3b` alone writes; `vocab_size`, `rope_theta` and
`max_seq_len` are defaults, because they all agree (`max_seq_len` is the
harness ceiling and the RoPE cache size, not a model's context length).
The parameter/flops formulas mirror torchtitan's
`get_moe_model_nparams_and_flops`. **The five fields that used to be
derivations are the ones the real ladder broke**: `n_kv_heads = n_heads // 2`
returns 16 at `9b` and at `48b` where piper carries 8, `head_dim` is 128 at
`48b`, `num_experts` is 8 at both, and at `30b-a3b` the derived `n_heads`
would be 16 against 32 and the derived expert width 7168 against 768. Each
wrong value builds a different model and publishes it under the requested
name.

`tests/test_model_shape.py` pins every registered shape's numbers in
`PINNED_SHAPES` and derives the counts rather than transcribing them: a
helper counts the parameters tensor by tensor, the test proves the helper
against the `1b` numbers, and every registered shape must then agree with the
helper. The pinned table is a deliberate second statement of the geometry,
transcribed from the model config the shape claims to be.

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

**The 1-layer shapes declare no regions, deliberately.** `piper_block_regions`
identifies a block graph by its invocations per window
(`n_layers * profiler_active`), and that count is the *identity*: measured
on a real 16-layer trace the forward graphs run {5, 80, 5} times and the
backward graphs {5, 80}, so 80 is unique to the block graphs. At one layer
the block graph also runs 5 times, colliding with two forward and one
backward partition, and `pooled_window_metrics` would raise "found 3". There
is no invocation count that identifies a 1-layer block graph and adding a
tiebreak would be relaxing validation rule 7 -- so `supports_block_regions`
is `n_layers > 1`, False at `huge` and at `giant`, and `_resolve_run` writes
`regions: []`, exactly as `piper1b_megatron` already does and for the same
honest reason. Rule 7 therefore does not guard a run at either of those two
shapes; rules 8, 9 and 11 do. Cross-mode
metrics (total GPU kernel time, tokens/s, launch latency, peak memory) are
unaffected.

**The four multi-layer shapes above `1b` pass that test arithmetically and
none has been checked against a trace.** `large` asks for 20 invocations per
window, `9b` for 120, `48b` for 160 and `30b-a3b` for 240. None of the four
is 5, so none hits the collision above. Whether each count is *unique* in its
own trace is the part nobody has measured. See "Seven shapes are registered"
above.


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
enabled at import) and decorates 41 functions with it across 14 modules --
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
| `piper_megatron_stock` | `baseline` | `launcher="megatron_stock"`: stock `megatron.training.pretrain` (see "The stock Megatron arm") |
| | `titan_stock` | the same config as `piper1b_megatron/titan_stock` |

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

**The kernel side is different, and the old text here stated the opposite.** `kernel-bench` runs no compile treatment it did not choose, so TE
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
  manifest.json     # schema 14: workload, regions, arms, commands, compile_mode, ac_mode, model_size, model_shape, parallelism (with dense_sharding), megatron_p2p_sync, megatron_nan_guard, throughput_definition, execution_model, hardware_metadata
  run_state.json    # per-arm status, attempts, evaluation status
  results.json      # schema 5: throughput (per rank), memory, gpu_time, region stats, significance
  <arm>.log         # training stdout+stderr
  <arm>/profiling/traces/iteration_*/rank<n>_trace.json.gz
  attempts/<ts>/<arm>/   # archived artifacts from a failed prior attempt
```

`manifest.json` `hardware_metadata` records `requested_gpu`, `nvidia_smi`,
`cpu_pinning`, `torch_version`, `torchtitan_git_rev`, `benchmarks_git_rev`,
`megatron_git_rev`, `te_version`, `cudnn_torch_build` and
`cudnn_loader_resolves`. Always cite `torchtitan_git_rev`,
`torch_version`, `compile_mode`, and `ac_mode` when reporting numbers --
plus `megatron_git_rev`, `te_version` and the two cuDNN fields for the
megatron scenario, `megatron_p2p_sync` beside any megatron number taken
above `pp` 1, and `megatron_nan_guard` beside any stock megatron number.

### Which cuDNN a megatron arm runs is a host property

TransformerEngine binds cuDNN **in Python**, before the dynamic loader
resolves any `DT_NEEDED` entry. `transformer_engine/common/__init__.py:345`
tries the system copy first, and its last resort at `:330` is
`ctypes.CDLL("libcudnn.so", RTLD_GLOBAL)` -- the **unversioned** name. Torch
also loads its own cuDNN lazily. So on a host that ships cuDNN in a system
directory, TE gets **that** copy rather than the wheel torch is pinned
against. Which cuDNN a megatron arm ran is therefore decided by the host,
not by the pin, and until 2026-08-20 no manifest recorded it.

`hardware_metadata` now records both halves: `cudnn_torch_build` is the
version torch was compiled against, read through `getCompileVersion` because
`backends.cudnn.version()` **raises** in exactly the case the field exists to
record; `cudnn_loader_resolves` is the real path the dynamic loader binds,
probed in a subprocess. On this box the two disagree -- torch expects
**9.24.0** and the loader binds `/usr/lib64` **9.23.2**, which `rpm -qf`
names as `libcudnn9-cuda-12`, a **CUDA 12 build inside a cu13 process**.

**The version changes no value, and that was measured rather than assumed.**
A direct comparison on 2026-08-21 ran the `attention_core` gates under 9.23.2
and under 9.24.0 (`reports/20260821-cudnn-version-comparison.md`). Every gate
row matches to the float64 bit pattern, the raw bytes of all eight output
tensors hash identically, and TE selects the same backend either way. So the
cuDNN version is **not a numerical comparability boundary**, and no published
megatron figure is numerically wrong because of it.

**What stays open is speed.** Nobody has timed the two versions against each
other. So cite `cudnn_loader_resolves` beside any timing number that goes
through TransformerEngine, and read a cuDNN difference as an unmeasured
effect on speed rather than as a difference in the values.

The two fields are collected but **not** resume-gated, which is consistent
with the above: recording a fact and refusing to cross it are separate
decisions, and there is no numerical boundary here to refuse.

**Two variables are necessary to move the whole process to the pinned
cuDNN.**
`LD_LIBRARY_PATH` alone does **not** work: the wheel directory ships only
`libcudnn.so.9`, so the unversioned `CDLL` above skips it and takes
`/usr/lib64` anyway, leaving torch reporting 9.24.0 while TE still runs
9.23.2 -- a split in the opposite direction, which is worse than doing
nothing. `CUDNN_PATH` **and** `LD_LIBRARY_PATH` together were measured to
leave zero `/usr/lib64/libcudnn` mappings.

### CPU pinning


The training step is host-bound at benchmark sizes, so unpinned runs measure
scheduler placement, not kernels. The runner therefore binds each training
process to the GPU's own NUMA node with `numactl --cpunodebind --membind`,
resolved from the GPU's PCI bus id via sysfs, once per requested device.
When that cannot be resolved (no `numactl`, unknown bus id, the device
reports no NUMA affinity, or the requested devices sit on **different**
nodes) the run proceeds unpinned and `cpu_pinning` records why. One
`numactl` prefix leads one command line, so a set that spans nodes cannot be
pinned without binding every rank to one node. Pinned and unpinned runs
are not comparable; `--resume` refuses to mix them.

### Validation

`validate_arm` (`benchmarks/e2e/validation.py`) fails an arm on any of:

1. Missing `<arm>.log`, or log lacking the profile's completion marker
   (`Training completed` for both engines).
2. `[Override]` line count != `arm.overrides_per_block * shape.n_layers`
   (one per transformer block, so 16 / 4 / 24 / 1 / 1 / 48 / 32 across the
   seven shapes).

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
   (mode=...)` for the megatron arm. Under `--compile-mode none` the rule
   inverts: `ValidationProfile.compiled_marker` (`with torch.compile`) must
   be **absent**, so a run that silently compiled cannot pass as eager. A
   profile that declares no marker cannot prove eager execution, and an
   uncompiled run reaching it is refused.
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
    place of rule 7. Under a pipeline split each engine additionally prints
    its own stage's count on a **separate** line and asserts it against
    `PiperShape.stage_param_count`, and the driver sums the counted
    parameters across the world and asserts the total. Rule 11 itself still
    matches the whole-model line, which every rank prints.
12. The engine's `parallelism` line is absent or names another mesh. Both
    engines log what they really built -- TorchTitan's `Building device mesh
    with parallelism: pp=..., dp_replicate=..., dp_shard=..., cp=1, tp=1,
    ep=...` plus, above `pp` 1, its `Using pipeline schedule <name> with <N>
    microbatches and <S> stages`; the megatron driver its own `Megatron-LM
    parallelism: dp=... pp=... schedule=... microbatches=... stages=...`.
    **This is the rule that catches the hazard of this axis**: a run that
    ignored the `--parallelism.*` flags, or a driver that read no `RANK`,
    passes every other rule while training something else, and two engines
    that agree on the split but disagree on the microbatch count would
    publish two different schedules under one label. The markers are a
    per-profile callable of the spec and the workload, because every value
    in them comes from those two. An **empty** tuple means the engine proves
    nothing about this spec, and `validate_arm` then refuses the run -- the
    same shape as `compiled_marker` under `--compile-mode none`. The rule is
    consulted only above one rank, where there is a mesh to get wrong.

    Above `dp` 1 each engine's marker set gains its own data-parallel line,
    and that line is the axis's real proof. TorchTitan's
    `parallelize_piper1b` prints `piper1b data parallel: fully_shard applied
    (dp_replicate=R, dp_shard=S)` **after** counting the FSDP units it got
    and raising when there are none; the megatron driver prints
    `Megatron-LM data parallel: DistributedDataParallel over N ranks (...)`
    after the wrapper exists. Neither line can be printed by a run that
    skipped the path.

    Above `pp` 1 each megatron profile also asks every rank for its
    driver's p2p line, against the requested `--megatron-p2p-sync` value.
    The tuned driver prints `Megatron-LM p2p: batch_p2p_comm=True
    batch_p2p_sync=<bool>` and the stock driver `Megatron-LM stock p2p:
    batch_p2p_comm=True batch_p2p_sync=<bool>`, each read off the config
    the driver BUILT rather than off its arguments. `batch_p2p_comm` is
    pinned to `True`, because Megatron's guard is `batch_p2p_comm and
    batch_p2p_sync`, and a run with the first field False would skip the
    sync under either label. The line is a second profile callable,
    `p2p_markers`, of the spec and the value. The TorchTitan profile
    returns an empty tuple at every mesh, because the option never reaches
    it; that empty tuple is not the refusal above, which reads the mesh
    markers alone.

    The stock profile also asks every rank, **at every mesh**, for its
    driver's nan guard line against the requested `--megatron-nan-guard`
    value: `Megatron-LM stock nan guard:
    check_for_nan_in_loss_and_grad=<bool>`, printed from the value
    Megatron PARSED. Nothing in the driver sets the field, so a run whose
    argv lost the token prints `True` under an `off` label and a run whose
    Megatron turned the field off by itself prints `False` under `on`;
    either fails. The callable, `nan_guard_markers`, takes the value alone
    and no spec, because the guard runs at `pp` 1 and at `dp` 1. The
    TorchTitan profile returns an empty tuple. The tuned megatron profile
    returns an empty tuple at `on` and **refuses** `off`, because that
    driver has no guard and no line of its log could prove the treatment.
13. A rank of a `dp > 1` run whose traces carry no all-reduce kernel
    (`ncclDevKernel_AllReduce`). **Two ranks that never reduce their
    gradients train two models and report roughly twice the true
    throughput**, and every other rule passes. This one reads **every**
    rank, which is provable here where rule 6's reading is not: at `dp`
    above 1 every rank sits in a data-parallel group of that size, so every
    rank reduces.
    The marker is the **spec's**, never an `Arm.trace_kernel_markers` entry
    -- a static declaration would fail every single-GPU run of the same arm.

    **It is a necessary condition and not a sufficient one, measured.** The
    `--pp 2, --dp 1` trace already carries
    `ncclDevKernel_AllReduce_Sum_bf16_RING_LL` five times per window, from
    the gradient-norm reduction over the pipeline group, and above `dp` 1
    both engines also all-reduce the loss on every logged step. So this rule
    says a collective ran; rule 12's data-parallel line says which mechanism
    built it. Cite the two together, and do not read rule 13 alone as proof
    that gradients were reduced.

    **The marker passed real megatron runs at `dp2` and at `dp2 x pp2` for
    the normal shape. It remains untested at other shapes and bucketings,
    and it can fail an honest run.** mcore reduces its buckets inside a
    `_coalescing_manager`, and a grouped NCCL launch can surface as
    `ncclDevKernel_Generic` rather than naming the operation. That is the
    safe failure direction, but read a rule 13 failure on the megatron arm
    as a question about the marker string first, and settle it from that
    arm's own trace. Widening it to a bare `nccl` is not the repair -- a
    pipeline's `SendRecv` would then satisfy it.

**The log rules run once per rank.** One `<arm>.log` holds every rank's
output, so a rule read against the whole file asks "did *some* rank do
this". Rule 4 is the sharpest case: a kernel that silently degraded on rank
1 alone leaves rank 0's log clean. `benchmarks/artifacts/layout.py`'s
`logs_by_rank` splits the file on the `[rank<n>]:` prefix and returns a
single-rank log **whole**, so a one-GPU arm is checked against exactly the
text it was checked against before. Rules 1, 2, 3, 4, 8, 10, 11 and 12 run
per rank; rules 5 and 7 run per rank's traces.

**A rank that wrote nothing is in neither split, so no rule would fire for
it.** Two coverage checks close that, and both are guarded on world size
above 1: the log rank set and the trace rank set must each equal
`range(world_size)`. A rank with no trace is a rank no per-step figure
measures, and evaluation would then publish a maximum over the survivors.

**Rules 6 and 9 read every rank's traces as one set, and that is
deliberate.** Rule 9 is unreachable under a mesh, because parallelism rule
13 refuses cuda-graph above world size 1. **Rule 6 is reachable and its
reading is an open question**: under PP a stage holds some of the layers, so
a marker kernel can be legitimately absent from a rank -- "every rank" would
fail an honest run, and "any rank" passes a run where rank 0 silently
degraded.

**One PP2 megatron trace per rank has now been read, and the reading still
does not change.** Both stages of that 16-layer run carry the cuDNN
fused-attention kernel, `_mul_silu_split` and `_permute_kernel`, so "every
rank" would cost **that arm at that shape** nothing. It is one arm at one
shape. A stage that holds no layer of the kind a marker names would still
lack it, so the general repair is a per-arm declaration of which ranks carry
which marker, not a blanket "every rank". Do not weaken this rule to make a
hypothetical run pass, and do not tighten it on one arm's evidence.

**What that costs today is concrete, and it is a coverage hole rather than
a wrong number.** "Any rank" means one stage satisfies the marker for the
whole arm. So at `--pp 2` a silent TransformerEngine fallback to unfused
attention on the stage that is not checked passes a guard this file lists
under "the ones that catch silent wrongness", and the same holds for the
FA3 and FA4 markers. **A pipelined arm therefore has strictly less
fallback coverage than the same arm at one rank.** Before publishing any
pipelined number that rests on a marker, read every rank's traces by hand
and say that you did.

**Every lift of `MAX_PP` thins that coverage again, and it is now as thin
as eight ranks allow.** At `pp` 2 a marker was satisfied by one of two
stages. At `pp` 4 it is one of four. At `MAX_PP = 8` it is **one of
eight**: a fallback on the other seven publishes under the fused label,
and `gpu_time` is a **maximum over ranks**, so a degraded stage is exactly
the rank that sets the published figure. The rule reads seven eighths of
the run's stages not at all.

**The fraction is what moved, not the rule.** "Any rank" always meant one
stage speaks for the arm. `pp 8` makes the odds of that stage being the
degraded one seven in eight rather than one in two, so the same rule that
was a coverage hole at `pp 2` is close to no coverage at `pp 8`. It is
still the safe direction only in the sense that it cannot fail an honest
run; it can pass a wrong one.

`piper_megatron_stock/baseline` declares two such markers
(`cudnn_generated_fort_native_sdpa` and `_mul_silu_split`), and the
scenario's *planned* matrix runs `--dp 2 --pp 4` and a `1 x 8` pipeline --
planned in a document, because no in-tree tool sweeps it. **Read
every rank's traces by hand on the first cell of each mesh** -- eight ranks
at `dp 2 x pp 4`, and eight again at `pp 8`, where all eight are stages of
one pipeline. Do not carry a `pp 4` reading forward to a `pp 8` cell: a
deeper split gives each stage fewer layers, so a stage can honestly lack a
marker the shallower split put on every stage. **The `dp 2 x pp 4` cell has
had that reading**, and both declared markers appear on all eight ranks of
both windows -- so "any rank" costs that arm nothing at that mesh. No other
mesh has been read.

**Arm rule 7 holds at `--dp 2, --pp 1`, measured.** A run without a pipeline
still declares its regions, so the 80-invocations-per-window identity has to
survive `fully_shard` wrapping compiled blocks and turning their parameters
into DTensors. On 2026-08-24 an operator ran `piper1b_rope/baseline` at
`--dp 2 --ac none --compile-mode default`; it declared `forward_block` and
`backward_block` at 80 each and passed. So the wrap does not make Inductor
repartition the block graph, and the region identity is unchanged.

**That is one arm, of one scenario, at one shape.** The other four
region-declaring scenarios are unmeasured under a data-parallel degree, and
so is every arm that carries an override. Should one fail, the failure is
the safe direction: do not relax the rule to make it pass, read its trace
first.

**Under `--pp 2` a run declares no regions, so rule 7 guards nothing and
rules 8 to 12 do.** `piper_block_regions` identifies a block graph by its
invocations per window, `n_layers * profiler_active`, and that count *is*
the identity. A rank of a two-stage pipeline holds half the layers and runs
each once per microbatch, so it reaches a different count -- and not the
same count on every rank under an interleaved schedule. Deriving a per-rank
count would be rule 7 rewritten rather than applied. This is the same
honest reason the 1-layer shapes, `piper1b_megatron` and `--compile-mode
none` declare none.

Engine differences live in the `ValidationProfile` registry
(`VALIDATION_PROFILES`), selected by `Arm.validation`; rules 2/3/5/6/9/11/13
are shared. Rules 4, 7, 8, 9, 10, 11, 12 and 13 are the ones that catch
silent wrongness. Never work around them by relaxing the check.

**There are three profiles: `torchtitan`, `megatron` and `megatron_stock`.**
The third is the stock arm's, and every marker it carries spells the word
"stock", so no line of the tuned driver can satisfy it and no line of the
stock driver can satisfy the tuned profile. It sets `compiled_marker=None`
and `check_regions=False` for the reasons the tuned profile gives, and its
scenario declines every uncompiled mode. Its two mesh lines are the
`megatron_stock` driver's own:

```
Megatron-LM stock training loop (mode=<mode>, main_params_dtype=..., ...)
Megatron-LM stock parallelism: dp=<dp> pp=<pp> ep=<ep> schedule=1F1B microbatches=<m> stages=<pp>
Megatron-LM stock data parallel: <wrapper> over <dp> ranks (overlap_grad_reduce=..., grad_reduce_in_fp32=..., sharding_strategy=..., expert_parallel=<ep>)
```

**The microbatch count in that line is `microbatch_geometry`'s, not
`n_microbatches`'s, and it is 1 at `pp` 1.** One Megatron sample is one
packed sequence of `rows * seq_len` tokens rather than a batch of rows, so
the harness sends `--micro-batch-size 1` and `--global-batch-size
microbatches * dp`. Without a pipeline neither engine splits the batch, so
the whole local batch is one pack. `benchmarks/e2e/megatron_stock/flags.py`
builds both the argv and the marker's count from that one function, which is
what keeps them from drifting apart.

**The stock data-parallel line observes the wrapper, and nothing declares
it.** `install_data_parallel_marker` replaces
`megatron.training.training.setup_model_and_optimizer`, reads the model it
returns, and **raises when no chunk carries a `_BaseDataParallel`**. It then
prints the wrapper's own class name, `overlap_grad_reduce`,
`grad_reduce_in_fp32`, the sharding strategy and the expert group's real
width. `parallelism_lines` deliberately prints no copy of that line: a second
copy derived from the arguments would satisfy arm rule 12 on its own, and a
run whose wrapper went missing would pass the rule the shim exists to
enforce. Arm rule 13 cannot make up that difference here, because stock
Megatron all-reduces the reported loss over the data-parallel group on every
step.

**Five things about that line, and each is a trap somebody already fell
into:**

- **The isinstance is `_BaseDataParallel`, not `DistributedDataParallel`.**
  Megatron picks one of three wrapper classes from the arguments, and
  `FullyShardedDataParallelV1` is a **sibling** of `DistributedDataParallel`
  rather than a subclass -- both derive directly from `_BaseDataParallel`.
  The narrow check raised on an honest sharded run. Widening it loses what
  the old check proved, so the class name is printed to restore it.
- **`FullyShardedDataParallel` is a FACTORY FUNCTION, not a class.** Its own
  docstring says so. `isinstance(chunk, FullyShardedDataParallel)` raises
  `TypeError`. Do not write it.
- **The word `DistributedDataParallel` is not hardcoded in the line.** Under
  `shard` it would be a lie, so the class name is templated in and it is
  what says which memory strategy ran.
- **`sharding_strategy` is the strategy the run ACTS on, not the raw
  field.** Megatron's argparse defaults `data_parallel_sharding_strategy` to
  `optim_grads_params` and copies it into every `ddp_config`, but
  `megatron/core/optimizer/__init__.py` reads it only under
  `use_megatron_fsdp`. So the raw field says `optim_grads_params` on a
  replicated run that shards nothing, and the line reports `no_shard` there.
- **`overlap_grad_reduce` MOVES under `shard`, and the marker table derives
  it.** `MegatronFSDP.__init__` sets
  `self.ddp_config.overlap_grad_reduce = True` on the config it was handed
  -- the reference, not a copy -- whenever the strategy is `optim_grads` or
  `optim_grads_params`. So the marker pins `False` under `replicate` and
  `True` under `shard`. **Do not repair this by sending
  `--overlap-grad-reduce`**: `resolve_ddp_bucket_size` reads that value
  before the wrapper exists, so the flag would move the gradient bucket size
  and change the run rather than the record.

**The expert degree is proved in two halves, because no one line can prove
it.** `PARALLELISM_LINE` prints `ep=` from the arguments, and it prints in
`main()` **before** `pretrain()` runs, so no process group exists there and
an accessor would return 0. `install_data_parallel_marker` therefore reads
the **built** expert group after `setup_model_and_optimizer` returns and
**raises** when it disagrees with the argument. The argument is never
trusted on its own; the observation moves to the one place in the run where
the group exists.

### Resume

`run-all --resume <out_dir>` re-validates each arm against what is on disk,
skips those that already pass, archives partial artifacts under `attempts/`,
and re-runs the rest. It aborts if any of these changed since the manifest was
written: scenario, workload, selected arms, hardware label, extra TorchTitan
args, `compile_mode`, `ac_mode`, `model_size`, `parallelism`,
`megatron_p2p_sync`, `megatron_nan_guard`, `nvidia_smi`, `cpu_pinning`,
`torchtitan_git_rev`, `benchmarks_git_rev`, `megatron_git_rev`. A
different GPU or a different commit will not resume -- that is
intentional. Omitting `--compile-mode`, `--ac`, `--model-size`,
`--megatron-p2p-sync` or `--megatron-nan-guard` on a resume inherits the
recorded value; passing a different one is refused. Schema <= 12 manifests
carry no `megatron_p2p_sync` and resume as `on`, because no run before
schema 13 could turn the sync off; schema <= 13 manifests carry no
`megatron_nan_guard` and resume as `on` for the same reason. Schema <= 8
manifests carry no
`model_size` and resume as `normal`; schema <= 7 manifests cannot be
resumed by this code at all (they record pre-rename mode names and imply
`ac=sac`).

**`parallelism` does not inherit, and the asymmetry is deliberate.** The
five values above are single strings, so a resume can read one back and
rebuild the run from it. A spec is six fields that together decide every
arm's command line, and `--resume` compares no command line -- so a
reconstruction that dropped one field would relaunch the arms differently
and the gate would not see it. Omitting the flags on a resume therefore asks
for the trivial spec, which matches every schema <= 9 directory (they carry
no `parallelism` key and are read through the trivial record) and is refused
against any other.

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

  **`baseline_kernel_ratio` divides each side's OWN busiest rank**, so the
  two rank indices need not match. That is the right comparison of step
  costs -- the schedule holds the ranks together -- and it is not one
  component against itself once those two ranks hold different partitions
  of the model. Evaluation **warns and names both rank indices** whenever
  they differ; a single-GPU run has one rank on each side and never raises
  it. Captioned rather than pinned to a rank index: pinning would divide two
  ranks nobody chose for being busy, and it would move a figure every
  single-GPU run has already published.

**Every tokens/s figure is PER DEVICE, and the published one is the
MINIMUM over ranks.** Both engines divide one rank's own token count by
`cp * tp * pp`: the ranks of one pipeline share a batch, and each
data-parallel rank reads a batch of its own, so the data-parallel degree is
absent from the divisor. The manifest records this in
`throughput_definition`, because tensor and context parallelism would each
move the divisor again and an old directory cannot otherwise say which
definition produced its numbers. **At one rank the divisor is 1**, so every
figure this repo has published is unmoved.

The reduction is a minimum for the reason the trace reduction is a maximum:
a schedule holds the ranks in step, so the mesh runs at the pace of its
slowest participant and a mean would report a rate no device achieved. A
rank whose log holds no stable sample sorts last rather than winning as a
zero -- "no sample" is a measurement that did not happen. `results.json`
carries `rank_reduction`, `published_rank`, `ranks`, `per_rank` and
`tokens_per_second_global` (the published figure times the world size)
beside it, and evaluation **warns when the ranks spread more than 1.15x**,
in the style of the launch-latency warning: that wide, one rank is starved
or the ranks are not running one job.

**The loss and grad-norm trajectories come from one rank, not from every
rank concatenated.** TorchTitan computes the loss on the last pipeline stage
and every rank still prints a step line. A rank without that stage does not
print a smaller loss -- it prints a **sentinel**: `trainer.py` sets `loss =
torch.tensor([-1.0])` there, and at `dp 1` that reaches the step line
unreduced, so rank 0 of a pp2 run logs `loss: -1.00000`. Pooling it would
not add noise; it would add a constant that is not a loss.
`loss_visible_rank` is TorchTitan's own `_get_metrics_rank` arithmetic,
`(world_size // pp) * (pp - 1)`, and the megatron driver satisfies it by a
different route: it broadcasts the last stage's loss, so every rank prints
the real one. It is right for the two schedules this repo runs and **not**
for `ZBVZeroBubble`, which returns the loss on rank 0 and which parallelism
rule 5 refuses for any run holding a megatron arm.

**Every trace figure is read per rank, and the published one is the
MAXIMUM over ranks. It is never the mean.** One rank is one process on one
GPU, and its own profiler windows are the only set a per-step figure may be
pooled over: pooling two ranks gives a mean across ranks, which is neither
one rank's cost nor the step's total. A parallel schedule holds the ranks in
step, so the step is as long as its busiest participant.
`benchmarks/traces/extraction.py`'s `per_rank_pooled_metrics` is the entry
point, and `pooled_window_metrics` **refuses** a call carrying two ranks
rather than averaging them. Validation rules 5 and 7 are per rank for the
same reason.

`gpu_time` therefore also carries `published_rank` (the rank every scalar in
the row came from -- the row is one rank's, not a per-field maximum, because
`other = kernel - regions` taken from two ranks can go negative), `ranks`,
`per_rank` and `kernel_ms_per_step_summed_over_ranks`. **A single-GPU run
holds one rank, so every one of these figures equals what schema 3
recorded**; that is proved against a real recorded run in
`tests/test_parallel_traces.py`.

Three further per-step columns exist because the summed kernel total stops
answering the question once a run has more than one rank:

- `collective_ms_per_step` and `compute_ms_per_step`. An NCCL kernel carries
  `cat: "kernel"`, so it joins the total like any other -- but a blocking
  collective's duration includes waiting for a peer, so counting it as
  compute absorbs the pipeline bubble as work and makes one rank's figure
  depend on another rank's speed. `compute` is the total without them, and
  it is the value that compares to a single-GPU run. The region columns
  exclude them for the same reason, so `region_kernel_ms_per_step` never
  exceeds `compute_ms_per_step` and a collective sharing a region's stream
  cannot read as that region's work. **The NCCL name prefix is confirmed
  against a real pipeline trace**: each rank of the `--pp 2` megatron run
  carries 35 collective device kernels in its window, of 21,885 and 21,581
  kernel-category events, every one named `ncclDevKernel_*` and every one
  categorised as a kernel. The three names there are `..._SendRecv`,
  `..._Broadcast_RING_LL` and `..._AllReduce_Sum_bf16_RING_LL`. **That exact
  name census was performed on the pipeline trace, not on the data-parallel
  traces.** A data-parallel trace could carry an additional name, so re-check
  the constant before citing a data-parallel split.
- `busy_kernel_ms_per_step`. The interval-union basis. Summing double-counts
  whatever overlaps, which already overstates megatron's five streams
  against titan's one by about 6.5%, and a collective adds a stream of its
  own. Both bases are recorded; neither is chosen for the reader. On a
  single-stream arm the two agree to within float accumulation order -- 5e-4
  us in 402,484 on the checked-in fixture -- so a small difference there is
  not evidence of overlap.
- `wall_ms_per_step`, from the profiler's **host** `ProfilerStep#`
  annotation. A bubble runs no kernel, so no kernel-derived total sees it;
  `bubble = wall - busy` is what explains a pipeline result. The profiler
  emits each step twice, on the host and on the device, and only the host
  span is the step's wall clock -- taking the longer of the two would be
  right on a host-bound run and wrong on a device-bound one.

**A rank the profiler never measured is refused, not ranked as zero.** A
rank whose windows carry no `ProfilerStep` annotation has no per-step cost,
and treating that as 0 would drop it from the maximum -- possibly the
busiest rank. `busiest_rank` raises on a run that mixes stepped and stepless
ranks, exactly as `pooled_window_metrics` already raises on that mixture
between one rank's own windows. A run where **no** rank has steps is not a
mixture and keeps its old answer.

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

**Three of those four rows changed what the number means, so an old number
and its successor are not comparable.** `rope`'s titan arms now take
packed-document positions where they took `arange`, and their timed closures
now run the training graph where they ran an inference graph that saved no
activations. `qkv_prep` puts the attention-input norm inside
the cut, on both engines, because megatron fuses the RMSNorm into
`linear_qkv`'s GEMM prologue and exposes no entry point that runs either half
alone -- so a `qkv_prep` number is the norm plus the projection, and a `qkv`
number under `out/` is the projection alone. `expert_mlp` publishes each Piper
arm against `titan/fused_grouped_experts`, which is TorchTitan's own w13
fusion, rather than against unfused experts -- so the Piper arms are no longer
credited with a fusion upstream already ships. `attention_core` is the one row
whose titan arms measure the same cut the retired scenario measured.

Remember also that a kernel-isolation number is **not device time**: for small
kernels it is dominated by host dispatch, and `--burst` amortization does not
remove that (see "Method" under Kernel-isolation benchmarks). A kernel-speed
claim needs profiler-summed device time, which nothing in this repo currently
measures. Do not apply the replacement more loosely than the tool it replaces.

The **cross-engine** case -- attributing a megatron-vs-titan gap to particular
components -- is what the 16 cross-engine kernel scenarios are for. **Eight of
them have run. Eight have never been built.** The eight that ran cover one cut
each, at one shape and one sequence length, except `attention_core`, which was
swept. So there are eight cross-engine cuts to cite, and together they are
still not a component breakdown of a training step: they omit eight cuts, and
isolated component times are not additive. See "What has been measured" below.


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
- loss and grad-norm trajectories, as a sanity check only. **A `nan` or an
  `inf` on any rank's step line fails the arm before anything is
  published**, and the message names the rank and the step
  (`refuse_non_finite_trajectories` in `benchmarks/e2e/results.py`). It
  reads every rank, not only the published one, because no rank prints a
  non-finite value on purpose: TorchTitan's rank without the loss prints
  the `-1.0` sentinel, the stock driver omits the field, and the stock
  driver's `grad_norm: nan` on a skipped step is unreachable for a bf16
  run with no grad scaler. It is the one non-finite check the harness
  owns, and it runs on every arm under both `--megatron-nan-guard` values.

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
| `--arm` (repeatable) | every arm | subset of **one** scenario's arms; needs exactly one `--scenario` and refuses `--span` |
| `--span` (repeatable) | **none** | a kernel span, plus every scenario it replaces; refuses `--out` and `--arm` |
| `--replicates` | 5 | sweeps of every arm; the unit the CI is taken over |
| `--replicates-per-process` | 1 | consecutive replicates of one arm per worker; above 1 the CI is renamed (see "Startup cost") |
| `--samples-per-replicate` | 40 | timed bursts per arm per mode, per replicate |
| `--burst-k` | 16 | calls per timed burst; one value for every arm |
| `--warmup-calls` | 30 | untimed calls per arm per mode, before each replicate |
| `--burst` | off | adds the 1/4/16/64 dispatch-cost diagnostic |
| `--model-size` | `normal` | shape from `PIPER_SHAPES`; single-valued, no sweep |
| `--batch` / `--seq-len` | 4 / 1024 | `KernelWorkload` overrides (seq <= `max_seq_len`) |
| `--max-seq-len` | 2048 | raises the shape's seq ceiling; needed to sweep `attention_core` past 2048 |
| `--seed` | 0 | input generator seed |
| `--hardware` | `auto` | provenance label |
| `--out` | `out/<ts>/kernels/<scenario>/<hardware>` | single `--scenario`, and no `--span` |
| `--cache-root` / `--compiler-env` | as e2e | `rope` needs the compiler env |

`--replicates`, `--replicates-per-process`, `--samples-per-replicate` and
`--burst-k` all take `IntRange(min=1)`. Each of the four used to accept a
zero, and each broke differently and late.

Unlike `run-all --all-scenarios`, a failing scenario does not abort the rest;
every scenario is reported and the command exits nonzero if any failed.
Deliberately ignores the `OUT`/`SEQ`/`BATCH` env vars -- flags only, so an
e2e shell cannot leak settings into a kernel run.

**`--scenario` and `--span` default differently, and the asymmetry is
deliberate.** `--scenario` defaults to every scenario; `--span` defaults to
none. A span drags every scenario it encloses into the run, so a default of
"every span" would silently change what a bare invocation costs. An explicit
`--span` with no `--scenario` measures that span and its range, and nothing
else.

**`--arm` measures a subset of one scenario's arms.** Repeat it per arm, and
pair it with exactly one `--scenario`; an arm name belongs to one roster, so
a selection across two scenarios would mean a different thing in each. The
selection **must** name the anchor arm, because every comparison is a ratio
against it, and every correctness reference the selected arms use, because a
gate needs both sides in one process. A selection that omits either is
**refused, not repaired**: adding an arm the operator did not ask for changes
what the run measures. An unknown arm name raises rather than quietly
measuring a smaller set.

Every arm the selection leaves out still reaches `results.json` and the
manifest as `skipped`, with a reason that **names the flag**. The skip map
therefore carries two kinds of reason now, and a reader must not take one for
the other: the operator's choice (`--arm did not select it; this run measures
...`) and the host's capability (no C++20 compiler, or a declared
`KernelArm.requirement` this shape or workload fails). The selection is
resolved first, so an arm nobody asked for keeps that reason rather than a
capability reason it never had to meet.

Three scenarios need their synthetic rows to route evenly
(`batch * seq_len * top_k` divisible by `num_experts`), and each declares it
with `requires_balanced_routing`: `dispatch_permute`, `expert_mlp` and
`moe_combine`. A shape/workload pair that breaks that skips those three
**loudly** -- named numbers, a recorded error, a nonzero exit -- rather than
capping or rounding anything. The check is per scenario, so every other
scenario is unaffected and still runs. The same invariant is re-asserted
inside `run_kernel_scenario`, so `python -m benchmarks.kernel.worker` and any
other direct caller raise instead of measuring an expert split that does not
cover the rows they built.

**`--model-size huge` has been probed on four scenarios, and three of them no
longer exist.** On an H200 on 2026-08-19, `qkv`, `attention` and `lm_head`
completed at `huge` with every arm `ok`, and `swiglu` ran out of memory and
wrote no `results.json` (`out/20260819T010252Z/kernels/`). Of those four only
`lm_head` is still a scenario. **No cross-engine arm has run at `huge`**, and
no successor scenario has been probed there; the one cross-engine run so far
is at `normal`. Treat every scenario except `lm_head` at `huge` as untested
rather than as working.


The `swiglu` run died inside `run_correctness_pass`, in the gate's fp32
upcast, with 134 GiB of the device's 139.81 GiB already in use. **Keep that
as history and do not carry its arithmetic forward.** `swiglu_inputs` is a
deleted symbol, its successor `expert_mlp` has 8 arms rather than 3, and four
of those build a whole megatron `GPTModel` -- so re-pointing the name without
re-doing the sum would publish a memory figure for a roster that does not
exist. `benchmarks/kernel/operations/expert_mlp.py` carries its own `huge`
arithmetic, computed for its own eight arms; cite that. The run also predates
the residency change: `run_correctness_pass` now builds one arm at a time and
drops it before the next, so the pass is bounded by the largest single arm
rather than by their sum. **`expert_mlp` at `huge` is untested.** Measure it
before reporting anything about it.

### A megatron kernel arm builds only the layer parts it times

A scenario times one cut. It does not need the rest of the transformer layer
to exist. `build_model` therefore takes `blank_parts`, and each named part
becomes megatron's own `IdentityOp`, which allocates nothing. **Twelve of the
sixteen megatron builders pass `("mlp",)`** through
`benchmarks/kernel/operations/common.py`'s `MCORE_BLANK_MLP`. The other four
keep the whole layer because they time a cut inside the mlp: `expert_mlp`,
`moe_router`, `dispatch_permute` and `moe_combine`.
`tests/test_megatron_model.py` pins the two sets and requires a new builder
to name its own.

**This edits the derived spec. It writes no spec.** `get_gpt_decoder_block_spec`
derives the spec exactly as before, `_blank_layer_parts` replaces one field
of the derived result with that field's own declared default, and megatron
then builds the layer through its own constructor. **The config is not
touched**, so `num_moe_experts`, `moe_grouped_gemm` and `qk_layernorm` -- the
three fields megatron turns into a module-class choice and then checks
nowhere -- keep the values the profile and the shape gave them. A dense-layer
build (`num_moe_experts=None`) would move one of those three and was declined
for that reason; blanking moves none of them.

**The timed module keeps its exact weights.** Megatron builds the nine layer
parts in the order its dataclass declares them, and the mlp is the eighth. So
the embedding, the whole self-attention part and the pre-mlp norm of decoder
layer 0 are built first. The parts built after the mlp draw nothing the mlp
moves. The decoder's final norm is a constant fill. The output layer draws
from megatron's model-parallel RNG state, and the mlp touches neither that
state nor the position of anything in it: the expert weights take the
expert-parallel state and the router gate takes the host generator. Every
affected arm also either overwrites the parameters it measures or measures a
cut that holds none. **The argument holds for the CUDA initialization path
only**, so `build_model` refuses `blank_parts` together with
`use_cpu_initialization`, which draws every weight from one host generator.

**The saving is a transient build peak, not the published memory column.**
Every one of these builders releases the model before `memory_pass` runs, so
`peak_memory_gib` is unchanged and no number under `out/` needs a different
reading. Build peak, by arithmetic over `PiperShape` and not by measurement:

| shape | full build | with the mlp blanked | saved |
|---|---|---|---|
| `1b` | 1.99 GiB | 0.67 GiB | 66.1% |
| `large` | 7.94 | 2.69 | 66.1% |
| `9b` | 17.38 | 1.63 | 90.6% |
| `huge` | 19.61 | 7.80 | 60.2% |
| `giant` | 31.77 | 10.77 | 66.1% |
| `48b` | 88.82 | 4.82 | 94.6% |

**None of this has run on a GPU.** The CPU tests exercise the edit against
megatron's own ``TransformerLayerSubmodules``, whose field names and identity
defaults they also pin. They build no spec and no model, because both need a
device. No arm has been built with a part left out.

### Scenarios and arms

**The registry declares 17 scenarios and 71 arms.** 16 of the 17 are
cross-engine: they put megatron-core beside TorchTitan at one cut of the
model. **`lm_head` is the one single-engine scenario left.** It stays because
its `fused_linear_ce` arm has no successor: `FusedLinearCrossEntropyLoss` owns
the LM head, so the arm fits neither `lm_head_projection` nor
`cross_entropy`, and the `fused_linear_ce` **span** that would hold it is not
declared yet.

**Three single-engine holdovers were deleted on 2026-08-20**, each once a
cross-engine scenario had re-homed every one of its arms: `swiglu` into
`expert_mlp`, `qkv` into `qkv_prep`, `attention` into `attention_core`. A
re-homed titan arm is the same code under an `engine/profile` name, so
`swiglu/piper_optimized_triton` is now
`expert_mlp/titan/piper_optimized_triton`. Two of the successors changed what
the number means -- see the table under "Total kernel time cannot rank arms
that differ in one component" -- so an old number and its successor are not
comparable.

**Every count in this section is the count at this HEAD.** Any scenario that
arrives or leaves moves all of them at once -- scenarios, arms, the
cross-engine total, the comparison and correctness tallies below. Re-derive
rather than quote:

```bash
.venv/bin/python -c "
from benchmarks.kernel.registry import KERNEL_SCENARIOS as K
from benchmarks.kernel.spans import KERNEL_SPANS
print(len(K), sum(len(s.arms) for s in K.values()), len(KERNEL_SPANS))"
```

The table below details `rope` and `lm_head`. **It is 2 of the 17, and the
registry is the authority.** The paragraph after the table names the other 15.

| scenario | arms | modes | notes |
|---|---|---|---|
| `rope` | `mcore/base`*, `mcore/no_rope_fusion`, `titan`, `titan/helion`, `titan/te` | fwd, bwd | **cross-engine**: megatron's THD path eager against three compiled titan modules. `titan/te` alone needs gcc-13, and is skipped by name without it. GB/s is reported; **x-floor is not**, because the scenario declares no floor |
| `lm_head` | `baseline`*, `fused_linear_ce`, `te_fused_ce`, `piper_optimized_te_ce` | fwd+bwd | losses compiled; peak memory is the secondary metric. The last single-engine scenario |

**The table above is not the whole registry.**
`benchmarks/kernel/registry.py` declares 15 further cross-engine scenarios, in
partition order: `embedding_stage`, `qkv_prep`, `qk_norm`, `attn_out_proj`,
`attn_residual`, `ffn_norm`, `moe_router`, `dispatch_permute`, `expert_mlp`,
`moe_combine`, `moe_residual`, `final_norm`, `lm_head_projection`,
`cross_entropy` and `attention_core`. Their arms are named `engine` or
`engine/profile`, except the `copy_floor` bandwidth arms. The anchor is
`mcore/base` in every one except `expert_mlp`, which anchors on `titan`
because it publishes no cross-engine row. `./run_bench.sh scenarios` prints
every scenario with its description. Read the registry rather than this table
for that half.

**What has been measured.** Eight of the 16 cross-engine scenarios have run.
**Every count in this section moves with every run. Re-derive them:**

```bash
find out -path '*kernels*' -name results.json | wc -l
find out -path '*kernels*' -name results.json -exec grep -l "mcore/" {} + | wc -l
```

At this writing there are 73 kernel `results.json` under `out/`, and 21 of
them contain `mcore/`.

| scenario | arms `ok` | measured at |
|---|---|---|
| `qkv_prep` | 3 of 3 | `large`, batch 4, seq 2048 |
| `qk_norm` | 3 of 3 | `large`, batch 4, seq 2048 |
| `attn_out_proj` | 2 of 2 | `large`, batch 4, seq 2048 |
| `ffn_norm` | 3 of 3 | `large`, batch 4, seq 2048 |
| `moe_router` | 5 of 5 | `large`, batch 4, seq 2048 |
| `expert_mlp` | 8 of 8 | `large`, batch 4, seq 2048 |
| `final_norm` | 3 of 3 | `large`, batch 4, seq 2048 |
| `attention_core` | 2 of 6 | `normal`/`large`/`huge` at seq 2048 and 8192, plus `normal` at seq 16384 |

Seven of the eight ran **every** declared arm, so their gates are complete.
`attention_core` is the exception, and only because an `--arm` selection asked
for two of its six. 29 of the 71 declared arms have ever reported `ok`.

**Eight cross-engine scenarios have still never been built**: `rope`,
`embedding_stage`, `attn_residual`, `dispatch_permute`, `moe_combine`,
`moe_residual`, `lm_head_projection` and `cross_entropy`. No arm was
constructed, no `_assert_mcore_*` guard ran, and no gate compared a real
tensor. The correct reading of one of those is "a declaration whose builders
have never executed", which is weaker than "untested" and much weaker than
"measured". The single-engine `lm_head` has not run either.

`reports/20260821-cross-engine-first-numbers.md` holds the raw tables and the
grading of every row that ran. Read it before citing any of them: four of its
six group-B comparisons are usable, and two are noise.

**The longest-sequence `attention_core` cell**
(`out/20260821T042516Z/kernels/attention_core/nvidia-h200/`) is a **two-arm
selection**: `--arm mcore/base --arm titan` at
`--seq-len 16384`, batch 4, `normal` shape, 5 replicates. The other four arms
carry `status: skipped` with the selection as the reason, so the file holds 2
comparison rows and no ratio against FA3, FA4 or the unfused arm. `titan`
against `mcore/base`:

| mode | titan / mcore/base | 95% CI | medians (us/call) |
|---|---|---|---|
| forward | **1.1226** | [1.1196, 1.1241] | 2071.7 against 1847.0 |
| forward_backward | **1.1421** | [1.1357, 1.1494] | 7936.9 against 6951.5 |

Titan is slower on both. **The number is quotable because the sequence
length made both arms device-dominated**: the `--burst` ladder puts every
residual inside +/-0.7%, far below the 2% flag, where the same two arms at
seq 1024 are dispatch-bound and their ratio would move with `k`. Read it as
one cell of one scenario at one long sequence, not as an engine verdict.

`attention_core` was the first cross-engine scenario built and gated, on
2026-08-20 (`reports/20260820-attention_core-firstrun.md`): all six arms
built and all 24 enforcing gates passed. The other seven followed on
2026-08-21, in one correctness sweep of 27 arms and 204 gates with zero
failures.


The `attention_core` scenario measures **inner attention only** -- the level
at which the implementations are substitutable, and the level that keeps it
from re-measuring the projection work `qkv_prep` and `attn_out_proj` already
cover. Its six arms are `mcore/base` (TransformerEngine's cuDNN
FusedAttention), `mcore/attn_flash3` (TE resolving to FlashAttention 3 on
sm90), `mcore/attn_unfused` (TE's own torch implementation), `titan`
(FlexAttention's Triton template), `titan/flex_flash` (the same FlexAttention
module lowered to FA4 CuTe kernels) and `titan/flash_attention_3` (FA3 varlen
through `torch.nn.attention.varlen`).

Every arm consumes the same q/k/v **values** and the same synthetic
packed-document boundaries. Those boundaries reach the arms in the three mask
forms the backends need: a flex `BlockMask` at the default 128 block size, the
same mask at the `(256, 128)` blocks the FLASH backend wants, and THD
`cu_seqlens`. The inputs builder builds all three once. No timed closure may
build one, because `create_varlen_metadata_for_document` contains a
device-to-host sync and `create_block_mask` is itself a compiled call. Nothing validates the FLASH
block size on the torch side: it is forwarded verbatim into FA4's
block-sparse tensors, so a mismatch surfaces inside FA4 rather than as a
torch-level error.

**The two engines get the same values in different memory layouts, on
purpose.** Titan gets three contiguous tensors, which is what its projection
materializes. Megatron gets a contiguous query, a contiguous key, and one
non-contiguous strided view -- the **value**, which is neither normed nor
rotated and therefore never leaves the fused QKV buffer. TE does not
recognize that layout and copies the value inside every timed megatron call.
So a megatron number here is attention plus megatron's own layout adaptation
for the value. The key's equal half belongs to `qk_norm`, which hands its own
megatron arm the matching strided key, so neither scenario double-books it.

**The compile treatment differs by arm, and every ratio is a comparison of
treatments.** All three megatron arms are eager, because megatron compiles no
whole transformer layer. `titan` and `titan/flex_flash` are compiled by
FlexAttention's own class-level `torch.compile`, which carries
`max_autotune` **and** `coordinate_descent_tuning`;
`titan/flash_attention_3` gets a plain `torch.compile(fullgraph=True)` with no
autotune. The Triton template is therefore the **only autotuned arm in the
scenario**, and a row against it is not a kernel-quality claim on its own.
`KernelArm.compiled` and `eager_reason` record the treatment per arm.

**Which kernel each megatron arm runs is pinned by its profile and enforced
by a guard**, because every backend computes the same function and no
correctness gate can tell them apart. `_assert_te_selected_backend` reads
TE's own recorded decision. On 2026-08-20 the three arms selected
`FusedAttention`, `FlashAttention 3.0.0` and `UnfusedDotProductAttention`
respectively. **`mcore/base` is the cuDNN arm**, because cuDNN is what
TransformerEngine resolves to on Hopper -- so the anchor of this scenario is
a cuDNN kernel, and every cuDNN caveat below applies to it.

**There is no megatron FlashAttention-4 arm**, and there cannot be one here:
TE prefers FA3 on sm90 whenever both are installed and no megatron setting
reaches past that. `titan/flex_flash` therefore has no megatron opponent and
is published against `titan`, which isolates the lowering. Expect it to lose
on Hopper for a reason that is not about FA4: FlexAttention's packed-interval
mask optimization is gated on compute capability 10/11, so partial blocks
evaluate the mask per lane here.

**`mcore/attn_unfused` can exhaust the device, and its OOM would take the
whole scenario.** It materializes the score matrix over the **segment** count,
which is `cu_seqlens.numel() - 1` and not the real document count -- 127 at
every workload, because `cu_seqlens` is padded to 128 entries. One bf16 score
tensor is 4.0 GiB at seq 1024, 15.9 GiB at 2048 and 63.5 GiB at 4096 on the
normal shape, and 47.6 GiB at seq 1024 on `huge`; the arm holds three of
those. A sweep past seq 2048 will OOM, and the correctness pass has no
per-arm exception handling, so the whole scenario goes with it. At the
default workload it does not: measured peak 13.47 GiB against 139.81.

**The TE/FA3 coexistence question is settled, and its answer is not the one
the file used to give.** The old text claimed TE and the FA3 varlen path
cannot share a process, citing a cuDNN soname collision. Measured on
2026-08-20, **that is refuted**: both libraries import and run real attention
kernels in one interpreter, in both import orders
(`reports/20260820-te-fa3-coexist.md`). What does block them on this host is
narrower and real -- torch's cuDNN **version bookkeeping**. See "The
correctness pass needs a per-arm split on some hosts" below.

The fp64 reference is computed **per (row, kv group)**. A one-shot
`[B, n_heads, L, L]` fp64 score tensor is 8.6 GiB at batch 4 / seq 4096 and
69 GiB at batch 32, so the obvious implementation OOMs exactly at the shapes
worth measuring. Gate the arms with `max_rel_l2` only: attention is a
reduction, and this file's rule against max/ULP metrics on reductions
applies.

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
a tuple of `(arm, opponent)` pairs. Left `None` -- as 4 of the 17 scenarios
leave it -- it derives the usual set: every non-floor arm against the anchor.
An explicit tuple is exhaustive, and the empty tuple declares a scenario that
publishes no ratio at all, which a scenario whose two sides are not a
like-for-like cut must be able to say. It replaces the per-arm `compare_to`,
which could redirect a row but could not decline one. The 17 scenarios
publish 41 comparison rows between them; no scenario declares the empty tuple
today.

**A second kind of per-arm requirement exists.** `KernelArm.requirement` is a
dotted `module:function` path, resolved **in the parent** and called as
`predicate(shape, workload)`. It returns `None` when the arm can run here, or
the reason it cannot -- and that reason reaches `results.json` as
`status_reason`, so a reader learns why an arm is absent without holding the
registry. It differs from `requires_gcc_toolset` in what decides it:
`requires_gcc_toolset` is a property of the **host**, answerable before the
shape is known, and `requirement` is a property of this **shape and
workload**. The module it names must be parent-side and torch-free, like the
schema; a test asserts that. **No arm declares one at this rev.** The case
that forced the field is an unfused attention arm whose score tensor grows
with the square of the sequence length, and it is undeclared. Catching the
builder's exception instead was rejected: a `try` around the build turns a
bug into a skipped arm, shortens the roster for a reason nobody declared, and
still exits zero.

**A builder path is a string, and must stay one.** `benchmarks/kernel/engine/`
imports `schema.py`, never `registry.py`, and never an `operations/` module:
arms reach it only as already-resolved `BuiltArm` values via `resolve_symbol`.
So do not move a scenario constant next to its family's builders -- that
"colocate the family" move recreates `engine -> registry ->
operations.<family> -> torchtitan`. That chain puts every kernel family and
its model dependencies into the engine's import graph. Per-arm process
isolation cannot have that. `tests/test_import_boundaries.py` section 3 asserts both
halves; `tests/test_migration_contract.py` pins each scenario's builders to its
own family module.

#### The correctness pass needs a per-arm split on some hosts

`run_correctness_pass` gates every arm of a scenario in **one** interpreter,
because a gate needs both sides at once. On this box that is what
`attention_core` cannot do without a workaround.

The mechanism is the one "Which cuDNN a megatron arm runs is a host property"
describes: TE binds the host's cuDNN, so `torch.backends.cudnn.version()`
raises, because torch requires `runtime_minor >= compile_minor`.
`torch.nn.attention.varlen` asks for that version and the answer is
`lru_cache`d, so one raise is enough -- torchtitan's `VarlenAttention` cannot
build in a process that has imported TE.

**The blocker is the correctness pass alone.** A timing worker holds one arm,
so TE never sits beside the varlen path there. cuDNN is also the only library
that splits this way: `libcublas`, `libcublasLt`, `libcudart` and `libnccl`
all resolve to the venv wheels in the same process, because torch loads those
eagerly.

Two ways past it, and they are **not** equivalent, so say which one a number
came from:

- `PYTORCH_SKIP_CUDNN_COMPATIBILITY_CHECK=1` leaves TE on 9.23.2 and only
  stops torch refusing to answer. This is what the first `attention_core`
  gate pass used. The flag reaches **every** worker, because the child
  environment is built from `os.environ`, so a shell that exports it
  publishes every number under it.
- `CUDNN_PATH` **and** `LD_LIBRARY_PATH` together move the whole process to
  the pinned 9.24.0. `LD_LIBRARY_PATH` on its own does not, and leaves a
  split in the opposite direction; the section above gives the reason.

A third option -- initializing torch's cuDNN before TE imports -- is
**untested and predicted to be bad**: it should leave TE running a 9.24.0
graph engine against 9.23.2 ops. The cuDNN comparison measured four
configurations and this is not one of them, and the binding order above works
against it. Treat it as a prediction, and do not use it.

A per-arm correctness split would remove the need for either workaround,
because TE and the varlen path would never share an interpreter. A repair to
the environment removes it too. Neither is done; the choice is recorded rather
than made.


#### The GB/s and x_floor columns are not always read against 1.0

Two derived columns are computed by the parent for every scenario that
declares a `copy_floor`. `gbps` is `bytes_moved / median`. `x_floor` is
`median / floor_median` for every non-floor arm that shares a mode with the
floor, so it reads **no** `bytes_moved` at all.

That last fact is what a reader misses. When two arms of one scenario declare **different**
`bytes_moved`, an arm running at exactly the floor's bandwidth no longer
reads `x_floor` 1.0. Two scenarios do that today, and the correction factor
is `arm_bytes / floor_bytes`:

| scenario | arm | `bytes_moved` at the default workload | reads against |
|---|---|---|---|
| `qk_norm` | `copy_floor`, `titan` | 25,165,824 (24 MiB) | 1.00 |
| `qk_norm` | `mcore/base` | 33,554,432 (32 MiB) | **1.33** |
| `moe_router` | `copy_floor` | 16,777,216 | 1.00 |
| `moe_router` | the three `mcore` arms | 8,470,528 | **0.50** |
| `moe_router` | `titan` | 42,106,880 | **2.51** |

Divide by the factor to recover the usual "near 1 means bandwidth-bound"
reading. **`qk_norm` states its factor in the scenario `description`, so a
reader of `results.json` or of the printed table finds it. `moe_router` does
not** -- its description says only that each arm carries its own
`bytes_moved` "so the GB/s and x_floor columns show the asymmetry instead of
absorbing it", which is true and gives a reader no factor. The `moe_router`
spread is the wider of the two, at 5x between its extremes.

**Why `qk_norm/mcore/base` moves 8 MiB more.** Megatron's `k_layernorm`
receives the key as a **strided view** into the fused QKV buffer, which is
what megatron really produces: the query leaves the buffer because the
reshape that merges the group dimension has to copy, and the key does not.
TransformerEngine's `RMSNorm` calls `input_.contiguous()` inside the timed
closure, so the copy is real work -- one 4 MiB read plus one 4 MiB write
under the read-plus-write convention `qk_bytes` itself uses. That is the half
of `qkv_prep`'s deferred copy which used to be timed in **no** scenario;
`attention_core` times the value's half. **The 4 MiB and the 8 MiB are two
conventions and both are right -- never put them side by side unnamed.**

**One column understates one arm outright, and no factor fixes it.**
`moe_combine` declares a single `bytes_moved` of 25,165,824 (25.2 MB) for
every arm. It describes what an **unpermute** moves -- read the routed rows,
write one row per token. Titan's combine additionally scales every routed row
by its probability, which megatron applied one cut earlier and does not
repeat here, so titan makes **at least one more pass** over the `[rows, dim]`
tensor: at least 41,943,040 B (41.9 MB) against the declared 25.2. By how
much the titan figure is low is **not measured** -- it depends on whether
Inductor fuses the cast, the multiply and the cast back into one pass or
materializes an fp32 copy. **Read no bandwidth achievement off the
`moe_combine` titan row.**

### Spans

**A span is an implementation that fuses across a scenario cut.** It belongs
to no single scenario, so it is declared over an **ordered scenario range**,
and its claim is the span against the **sum of the scenarios it replaces**. A
span result therefore holds two totals.

**`KERNEL_SPANS` declares five spans and eight span arms at this rev**, and
**not one of them has a builder**. Every arm's `builder` names a module under
`benchmarks/kernel/operations/` that nobody has written, so `--span` measures
the enclosed scenarios and then raises in the span's correctness worker at
`resolve_symbol`. The declarations are the specification those builders must
meet. The mechanism landed one commit ahead of the declarations, deliberately:
a span needs a runner that can launch one and a merge that can hold two totals.
**Re-derive the roster rather than trusting this count:**

```bash
.venv/bin/python -c "
from benchmarks.kernel.spans import KERNEL_SPANS
print(len(KERNEL_SPANS), sorted(KERNEL_SPANS))"
```

**No span has ever been measured.** The engine has never run on a GPU, and
every number in its tests is synthetic.

How a span is declared. `benchmarks/kernel/spans.py` holds the roster;
`KernelSpan` and `SpanParts` live in `benchmarks/kernel/schema.py` beside the
scenario types.

- `measurement` is the span's own head-to-head, and it is a `KernelScenario`.
  A span **composes** one rather than subclassing it, so
  `isinstance(span, KernelScenario)` is False and a span added to
  `KERNEL_SCENARIOS` by mistake cannot run as a bare scenario. It is also
  what keeps `benchmarks/kernel/engine/` free of any knowledge that spans
  exist: both passes take a `KernelScenario`, and a span hands them
  `measurement`.
- `scenarios` is the range, in the model's own order, and it must hold at
  least two entries with no repeat.
- `parts` says what each span arm replaces, **positionally**: one arm name per
  entry of `scenarios`, in that order. The correspondence is declared and
  never inferred, because a span arm named `titan` does not necessarily
  replace an arm named `titan` at each cut, and a span arm may name an
  implementation no enclosed scenario has.
- Every span arm must declare a `parts` entry. A span exists to state one
  claim. An arm with no parts row cannot state it.
- `validate_span_parts` runs at **import**, so a part arm that does not exist
  fails when the module loads rather than as an absent row after a GPU has
  measured every arm of the span and of every scenario it encloses.

How a span runs. `--span NAME` is repeatable and defaults to none. Asking for
one span can add several scenarios to the run: `measurement_plan` puts every
enclosed scenario **ahead of** its span, and each scenario once however many
spans enclose it. That ordering is what makes the two sides share a run.
`--out` refuses a span outright, because a span run is always several units
and they would all resolve to one directory; `--arm` refuses it too, because
an arm name belongs to one roster.

What a span publishes. A span writes a `kernel_span` results file whose
`arms` holds the span's **own** measurement, whose `parts` holds the other
side keyed by span arm, whose `comparisons` keeps its scenario meaning (arm
against arm, both measured inside the span), and whose `parts_comparisons`
holds the span-versus-parts claim under its own name. A reader who opens one
and finds the other has opened the wrong field.

Three things about a span number that must be said next to it:

1. **The statistic is unpaired.** See "Method" -- the interval is
   `unpaired_ratio_ci_low`/`_high` and it cancels no drift.
2. **The ratio is biased in the span's favour**, by one host dispatch chain
   per enclosed scenario beyond the first. See "Method".
3. **An incomplete parts side costs one row, not the whole file.** A part
   that lost its measurement costs that span arm its parts total; a part that
   measured fewer modes than it declared costs that mode's row, and both are
   recorded as warnings. The loud failure is kept for the case where **no**
   span arm carries a parts total: that file would state the span's own
   number and no claim about it. That is a scenario under a span's name.

### Method

- Module-scope arms (the three titan rope modules, `expert_mlp`'s four titan
  arms, `qkv_prep`'s two) run under `torch.compile(fullgraph=True)`, because
  that is what they face end-to-end: eager isolation races custom ops against
  materialization costs Inductor deletes, which inverts verdicts (the
  combined SwiGLU layout wins eager, loses compiled). **The treatment is per
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
  host dispatch**. The retired `qkv` scenario was dispatch-heavy in absolute
  terms too, but both its arms were, so the effect largely cancelled in the
  ratio. That reading transfers to `qkv_prep`'s titan pair in principle, and
  **it is not a repeat**: `qkv_prep` puts the attention-input norm inside the
  cut, so the arms are not the ones the figure was taken on.
- **A span ratio carries the same 85% as a bias in its own favour.** The
  parts side pays **one host dispatch chain per enclosed scenario**, and the
  span pays one. So a parts total over N scenarios holds N-1 extra chains
  that no fusion removed -- the harness stopped paying them because it timed
  one closure instead of N. The published span/parts ratio is therefore
  **smaller** than fusion alone would make it, which is the direction that
  supports the claim a span exists to make, and the effect grows with the
  length of the range. It is a property of the range length rather than of
  what a span fuses, so the engine states it on every span -- printed under
  the table and recorded in every span `results.json` -- and nothing corrects
  for it. Separating the two would need profiler-summed device time, which
  nothing here measures.

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
  us/call with the per-run standard deviation, **on two arms that no longer
  exist under those names**: `qkv/fused_qkv/forward` (the retired `qkv`
  scenario) 248.68 +/- **9.42** against 233.37 +/- **0.82**, and
  `rope/helion/forward` (the retired single-engine rope roster) 261.27 +/-
  **29.07** against 244.64 +/- **6.47**. The standard deviation
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
- **That whole bullet is a *scenario* statement. The span statistic is
  unpaired.** `measurement_plan` runs each unit to completion before the next
  one starts, so a span's replicate `r` and an enclosed scenario's replicate
  `r` are separated by every worker in between and share nothing but the
  number. Any permutation of the parts' indices would be as justified as the
  identity. So a span's point estimate is a ratio of two medians, each taken
  over its own side, and the interval is an **unpaired** bootstrap that
  resamples each side independently. It is published as
  `unpaired_ratio_ci_low`/`_high` and never under the scenario name: it
  cancels no drift, and the two must not be read as the same statistic.
  **Rejected: an interleave of the units, which would make the pairing real.**
  It would make a scenario's own numbers depend on whether a span asked for
  it.
- Python's garbage collector is paused during the timed region. A collection
  starves the launch queue and lands as idle time inside whichever arm's
  interval is open; pausing it cut the sd of the retired `swiglu` scenario's
  module arm from ~63 us to ~1.4 us and removed every 2x outlier, medians
  unchanged.

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
  the scenario in one interpreter, because 36 of the 71 arms name another
  arm as their correctness reference and a check needs both sides at once.
  One arm is resident at a time -- each is built, asked for its outputs, and
  dropped before the next is built -- so the pass is bounded by the largest
  single arm rather than by their sum
  (`benchmarks/kernel/engine/run.py`).
- **One declared scenario now needs a per-arm correctness split, and the
  reason is not the one anybody predicted.** TransformerEngine and the FA3
  varlen path were expected to collide over kernels or sonames; measured,
  they do not (`reports/20260820-te-fa3-coexist.md`). What collides is
  torch's cuDNN **version bookkeeping**, and it stops `attention_core`'s
  titan FA3 arm from building after a megatron arm in the same interpreter.
  See "The correctness pass needs a per-arm split on some hosts" above. The
  split is not built; the workaround is an environment flag.
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
- **Requirements belong to the arm, not to the scenario, and there are now
  two kinds.** `requires_gcc_toolset` is a property of the host:
  without a C++20 host compiler, rope loses `titan/te` and still measures its
  other four arms, where the former scenario-level check threw away the whole
  scenario. `KernelArm.requirement` is the other kind -- a parent-side
  predicate called with `(shape, workload)` -- and no arm declares one yet.
  `resolve_arm_skips` decides the set in the parent, before a GPU is claimed;
  it resolves the operator's `--arm` choice first, then the capability
  probes, then closes the set over correctness references (an arm whose
  reference is skipped is skipped too -- timing an arm nothing checked is the
  wrongness the gates exist for). It delivers the result to the correctness
  worker as `--skip-arm NAME`, and a skipped arm is spawned in neither pass.
  The scenario-level property survives for its one honest use: asking whether
  anything here needs the compiler at all, which is what decides whether
  `add_compiler_environment` runs. That call shells out to bash and is now
  resolved **once per run**.
- **Every declared arm reaches `results.json`, measured or not**, carrying
  `status` `ok`, `skipped` or `failed` and the reason. At schema 3 an arm
  this host could not run and an arm the registry never declared were both
  simply absent, so a reader could not tell a short roster from a complete
  one. **Read the reason, not only the status**: a `skipped` arm the operator
  left out with `--arm` and a `skipped` arm this host cannot build are
  different facts, and the reason string is what separates them. The skip of
  an *anchor* is the exception that costs the scenario.

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
- A **span**'s parts row is published under `unpaired_ratio_ci_*` either way
  and does not also take the `within_process_` rename;
  `methodology.replicates_per_process` records the value in both files.

**Its acceptance gate is still outstanding.** The flag lives on one
condition: more than a 2x end-to-end speedup on an idle box. Nobody has run
that measurement. If the verified speedup is under 2x, remove the flag. The
measurement it rests on was taken on the retired `qkv` scenario, on a
contended box, and may not be cited.

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
  that disagree about the sign are not a result. It was never run on the
  scenario whose isolation matters most, or on `lm_head`. That scenario is
  now `attention_core`, which holds FA3, FA4 **and** a TransformerEngine arm
  in one correctness pass -- so the isolation argument is stronger than it
  was, not weaker. The pool targets the same wall clock
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
- **`bitwise`** where implementations must agree exactly. Six scenarios
  declare it (`embedding_stage`, `qkv_prep`, `attn_residual`, `moe_router`,
  `dispatch_permute`, `moe_residual`), and whether a given check *gates*
  depends on what it compares. It stays informational
  wherever a compiled region may legitimately round differently from an
  eager one -- the fused-vs-unfused QKV outputs are the original case, and
  `qkv_prep/titan/unfused_qkv` still carries that informational check against
  `qkv_prep/titan` -- and the rel_l2 gates enforce closeness there. It
  **enforces** on a permutation or a routing decision, in `dispatch_permute`
  and on `moe_router`'s `selected_count`. A permutation is a gather, so a
  wrong one moves right values to wrong places, and both engines build the
  permuted rows by a pure copy: exact equality is achievable, and it is the
  metric that sees a misplacement whatever the shape.

  **The arithmetic behind that is now stated correctly in the source, and
  this file's warning about it is withdrawn.** Swapping one pair of `N` rows
  gives `max_rel_l2 ~ 2/sqrt(N)`, which is 2.2e-2 at the default
  `dispatch_permute` workload (`N = 8192`) -- **above** the 2e-2 gate rather
  than below it, and **falling** as the workload grows, to 1.6e-2 at batch 8.
  So a tolerance gate would catch a misplaced row at one batch size and miss
  it at the next, while the flags that set batch and sequence length are the
  operator's to change. `benchmarks/kernel/registry.py` and
  `benchmarks/kernel/operations/dispatch_permute.py` both carry that form and
  that conclusion today; the earlier `sqrt(2/N)` spelling is gone from both.

### Silent-fallback guard

`HelionCosSinRoPE` and `TECosSinRoPE` fall back to the *numerically correct*
stock path when their eligibility checks fail, so correctness gates cannot
catch a mis-timed arm. Their builders profile one call and refuse to continue
unless the arm's marker kernel (`_helion__rope_cos_sin_fwd`,
`fused_rope_forward_positions_kernel`) actually appears.

`attention_core/titan/flex_flash` is the one arm with the *opposite* failure
mode, and it still carries a guard. `BACKEND="FLASH"` hard-raises when
`flash_attn.cute` is missing rather than degrading to a Triton template, so an
arm that runs at all ran FA4 -- there is no silent fallback anywhere in that
lowering. The `FlashAttentionForwardSm90` marker guard therefore protects
against the *reverse* mistake: a future refactor quietly dropping
`kernel_options` and leaving the arm measuring the baseline under an FA4
label. The sibling `titan/flash_attention_3` pins `FlashAttnFwdSm90` for the
usual reason: FA3 degrades to FA2 rather than failing.

**The megatron arms of `attention_core` are guarded by a different mechanism,
and it must not be described as a marker guard.** Every attention backend
computes the same function, so no correctness gate separates them, and
megatron's own `flash_attention_version` field writes environment variables
TransformerEngine 2.17.1 reads nowhere. `_assert_te_selected_backend`
therefore reads TE's own recorded decision about the arm's own call. That is
stronger than a trace marker, and there is deliberately **no cuDNN
kernel-name marker** anywhere.

### Output layout

```
out/<timestamp>/kernels/<scenario>/<hardware>/
  manifest.json      # schema 7: kind, unit_kind, scenario, span, model_size, model_shape, workload, shapes, arms, skipped_arms, replicates/replicates_per_process/burst_k/warmup_calls/seed, commands, provenance
  results.json       # schema 7: kind, every declared arm with a status and its declared compile treatment, per-mode summaries + per-replicate samples, comparisons, correctness, warnings
  kernel_bench.log   # every worker's stdout+stderr, in spawn order
  fragments/
    correctness.json         # the gate pass
    timing__<arm>__r<N>.json # one per (arm, replicate)

out/<timestamp>/kernels/spans/<span>/<hardware>/
  manifest.json      # the same schema 7, plus span_scenarios and parts
  results.json       # kind "kernel_span": arms, parts, comparisons, parts_comparisons
  ...                # the same log and fragments
```

**A span writes one directory deeper, and that is load-bearing.** Every
statement in this file that globs `out/*/kernels/*/*/` is a **scenarios-only**
statement: the shallow pattern cannot reach a span, which is what the extra
directory is for. A span total and a scenario total answer different
questions and must never be pooled by a path pattern. A **recursive** walk
(`rglob`, `find -name results.json`) meets both shapes, and there the `kind`
field is the only thing that separates them. `--out` refuses a span for the
same reason: a span run is several units and they would all land in one
directory.

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
shape of a schema-5 file; a round-trip test now pins both directions.

**Both files went 6 -> 7 together, when a second kind of unit arrived.** The
rename that makes it a bump rather than an addition is the value of `kind`.
It was `"kernel"` on every file, and every file was a scenario -- so the word
named the family **and** one member of it, which is exactly how `n` and
`warmup` went wrong at schema 3. It is now `"kernel_scenario"` or
`"kernel_span"`. The wrong reading this prevents is concrete: a recursive
walk of `out/` meets both shapes and has `kind` as its only way to tell them
apart, so a reader keyed on `kind == "kernel"` would take a span total for a
scenario total and sum it beside the scenarios the span replaces,
double-counting the same work. Keeping `"kernel"` for scenarios and adding
`"kernel_span"` was **rejected**: it lets such a reader keep working while
silently skipping every span, which is the half-read the exact-equality
loader exists to prevent. The manifest carries the same rename plus
`unit_kind`, `span`, `span_scenarios` and `parts`; a span manifest writes its
name in `span` and leaves `scenario` null, because a span name looked up in
`KERNEL_SCENARIOS` finds nothing.

The results loader enforces exact schema equality,
so older files are rejected rather than half-read, and each `from_dict` also
refuses the other kind's file; the manifest is
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
  the busy (interval-union) basis whenever the arms differ in stream count.
  The arithmetic is `benchmarks/traces/extraction.py`'s `busy_union`, which
  `results.json` records as `busy_kernel_ms_per_step` beside the summed
  total; `analyze.py`'s function of the same name is the chrome-event
  adapter for it, not a second implementation.
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

**Execution model at the trivial spec: single GPU, plain bf16, no FSDP.**
The ModelSpec's `parallelize_fn` is
`benchmarks/models/piper_qwen3/parallelize.py:parallelize_piper1b`, which
delegates to the fork's `parallelize_qwen3` (AC and per-block compile
applied) and hard-errors on any `training.dtype` other than `bfloat16`.

**`skip_dp` is a function of the delivered mesh, not a constant.**
`skip_data_parallel(parallel_dims)` is `dp_replicate * dp_shard == 1`, so a
one-rank run and a pure pipeline still take the early return that skips
`fully_shard` -- the treatment every published titan number was measured
under, unchanged. Above that product the delegate runs `fully_shard`, and
`parallelize_piper1b` then **counts the FSDP units it got back and raises
when there are none**, because a delegate that silently skipped the wrap
would leave the ranks reducing nothing. It logs `piper1b data parallel:
fully_shard applied (dp_replicate=R, dp_shard=S)`, which is arm rule 12's
data-parallel marker.

**Under `--dense-sharding replicate` a DP spec asks for
`dp_replicate=dp, dp_shard=1`**: the parameters are replicated, not
sharded, so the model each rank holds is still the plain-bf16 model above.
**The delegate logs it as `Applied HSDP to the model`**, because it takes
the two-dimensional mesh branch whenever a replicate degree exists; the
shard group is one rank wide, so nothing is sharded. Say "replication at
shard degree 1", never "TorchTitan calls it DDP" -- its config names the
flag that way and its log does not.

**Under `shard` the same spec asks for `dp_replicate=1, dp_shard=dp`**, so
the delegate really shards and the model is no longer the plain-bf16 model
above. That is the point of the value, and it is why it is a comparability
boundary.

The harness always sends the shard-degree flag for either spec, because
`data_parallel_shard_degree` defaults to **-1** and an omitted flag would
turn `--dp 2` into ZeRO-3 silently. `parallelize_piper1b` guards the same
hazard from the other side, and reads the **raw configured value** to do
it: it refuses a negative degree, which is the only unambiguous "nobody
sent the flag". A guard written on the resolved mesh could not tell an
honest sharded run from a dropped flag.

**Its parallel refusals are per axis, and each names its own reason.** One
`world_size != 1` check stood there before, and it refused every axis for
one axis's reason. It now refuses `tp > 1` and `cp > 1` (the harness cannot
express either degree, so the manifest could not record such a run), a
**negative** `data_parallel_shard_degree` (nobody sent the flag, so the run
would shard while the manifest recorded whatever parity the harness asked
for) and a mesh that replicates **and** shards (no `dense_sharding` value
names HSDP). **It no longer refuses a shard degree above 1**, because
`--dense-sharding shard` asks for exactly that. **A pipeline rank passes**,
because it
holds a slice of the layers, needs no gradient synchronization, and
therefore keeps exactly the plain-bf16 model above. The module reads the
`ParallelDims` TorchTitan builds from the command line, never
`benchmarks/e2e/parallelism.py`: it runs inside the training subprocess,
where the harness's own spec is neither present nor needed.

**Gradients are SUMMED over the data-parallel mesh, not averaged, and that
is correct.** `apply_fsdp_to_decoder` calls
`disable_fsdp_gradient_division`, and TorchTitan divides its loss by the
**global** valid-token count instead (`trainer.py` all-reduces
`local_valid_tokens` over the batch mesh first). Megatron reaches the same
gradient the other way -- its loss divides by the **local** count and its
DDP scales each rank by `1/dp` before summing -- and the two agree because
every rank holds `batch x seq_len` tokens with no padding. Never "fix" one
side to look like the other. Note also that FSDP2 reduces in **fp32**
(`mixed_precision_reduce` is `Literal["float32"]` in the fork) where
megatron reduces in bf16 (`grad_reduce_in_fp32=False`); neither keeps an
fp32 gradient, and neither setting is reachable from the harness.

`training.dtype="bfloat16"`
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
scenario. Read the wrapper's enforcement narrowly, as what it is: the TE RoPE
arm requires bf16 activations, and an fp32 expert layer would pass every
validation rule while measuring the wrong thing. (The arm that would show it
is `expert_mlp/titan`; the `swiglu/baseline` this sentence used to name is
deleted.) Manifests
record `execution_model`; runs from schema <= 6 used FSDP2 mixed precision
and are not comparable.

**`execution_model` names the mesh the run had, and is no longer a
constant.** `benchmarks/e2e/parallelism.py`'s `execution_model` composes it
from the run's own spec: the device count, the parameter treatment, the
data-parallel treatment, then the pipeline and expert axes when they are
not trivial. **The trivial spec still returns
`single-gpu-plain-bf16-no-fsdp`, character for character** -- that string is
a fixed point every manifest since schema 7 carries, `EXECUTION_MODEL` in
`benchmarks/e2e/registry.py` is what pins it, and a test compares the two.
A pipelined run records `2-gpu-plain-bf16-no-fsdp-pp2-1F1B`.

**The data-parallel part carries the dense-sharding value.** It is `dp<N>`
under `replicate` and `dp<N>-shard` under `shard`, so
`--dp 2 --pp 4 --ep 2 --dense-sharding shard` records
`8-gpu-plain-bf16-dp2-shard-pp4-1F1B-ep2` and the same mesh under
`replicate` records `8-gpu-plain-bf16-dp2-pp4-1F1B-ep2`. Two runs of one
mesh under different parities therefore no longer read alike.

**The parts name degrees, not mechanisms**, because one manifest carries
one `execution_model` for a whole run and a cross-engine run holds arms of
both engines: `dp2` is true of both, where `fsdp2-replicate2-shard1` would
describe TorchTitan's path and misdescribe Megatron's. `shard` passes that
test for the same reason -- both engines shard under it, TorchTitan through
`fully_shard` and Megatron through Megatron-FSDP. The field is **not**
resume-gated -- `parallelism` is, and this is derived from it, so gating
both would refuse the same run twice.

An arm changes behavior one of two ways:

- **A different registered config** (`arm.config`). Used when the difference is
  structural: fused vs unfused QKV, or the loss/lm-head strategy.
- **`--override.imports <dotted.path>`** (`arm.override_imports`). Swaps config
  nodes in place after construction. Each replacement logs
  `[Override] <module>.<function>: <fqn> <Old> -> <New>`, which is exactly what
  `validate_arm` counts.

## The Megatron baseline arm

**This section is the TUNED arm.** The stock arm is a different arm of a
different scenario; read "The stock Megatron arm" below for it.

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
  (`PiperShape.parity_gate`): 2e-2 at `normal`
  (measured 5.5e-3), 5e-2 at `huge` (measured 2.03e-2), and 3e-2 at `large`
  and 6e-2 at `giant`, **both of which are fitted predictions that no parity
  check has tested**. The wider huge gate is
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

  **Above `--dp` 1 each rank reads its own slice**, through the same
  `dp_rank`/`dp_world_size` arguments the titan replay loader takes, so the
  two engines put the same tokens on the same rank and no token is trained
  on twice. Two consequences follow. The `max_documents` padding target is
  taken over the **global** sample set, by an all-reduce of each rank's own
  maximum, because a per-rank maximum would give the ranks different
  `cu_seqlens` lengths and their static shapes would stop matching. And the
  padding length is the only thing that reduction shares -- the tokens
  themselves never cross ranks.
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
  apex-style `main_grad` buffers, which only the `--dp` above 1 path
  provides; enabling it would make the fusion a property of the mesh rather
  than of the profile, and the arms would stop being comparable across
  degrees); its cost is unmeasured.
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
- **Distributed machinery is the mesh's, and no more**: a process group and
  megatron's own `initialize_model_parallel`, plus a
  `DistributedDataParallel` **only above `--dp` 1**. There is no
  MegatronOptimizer at any degree. At one rank it is exactly what it always
  was: the driver falls back to rank 0 of a world of 1 when torchrun set no
  variables, and every published megatron number is from such a run.

  Above one rank the driver reads `RANK`, `WORLD_SIZE` and `LOCAL_RANK`,
  binds the device by local rank, passes `pipeline_model_parallel_size` to
  `initialize_model_parallel`, and builds only its own stage
  (`pre_process`/`post_process` reach `GPTModel`, and megatron's own
  `get_num_layers_to_build` decides the layer count). It runs
  `forward_backward_pipelining_without_interleaving`, which is **`1F1B` and
  nothing else** -- an interleaved schedule needs a model-chunk list, a
  data-iterator list and a virtual pipeline degree, none of which is built,
  so the driver raises rather than running a schedule under the wrong name.
  The mesh check is `WORLD_SIZE == --dp x --pp`, and it names both degrees.

  **Above `--dp` 1 the driver wraps the model in megatron's own
  `DistributedDataParallel`** with `overlap_grad_reduce=True` and
  `grad_reduce_in_fp32=False`, sets `config.finalize_model_grads_func` and
  `config.no_sync_func`, and calls `zero_grad_buffer()` at the top of every
  step. The schedule is what invokes `finalize_model_grads`, at both its
  pipelined and non-pipelined entry points, and that is what runs the
  reduction. DDP delivers weight gradients through `param.main_grad` and
  leaves `.grad` unset, so the step loop points `.grad` at `main_grad`
  before clipping, exactly as the cuda-graph path already did. It does
  **not** zero `main_grad` afterwards: the next step's `zero_grad_buffer`
  does that over the whole bucket, and zeroing a view of a buffer whose
  reduction has not been waited on would race it. **The two paths never
  meet today** -- parallelism rule 13 refuses cuda-graph above one rank --
  and the guard is written for the day that rule lifts.

  The loss lives on the last stage. The driver broadcasts it over the
  **pipeline** group and then takes the mean over the **data-parallel**
  group, so every rank prints the same real number and it means what
  TorchTitan's `global_avg_loss` means. Both groups matter as soon as `dp`
  is above 1: each pipeline has its own last rank, so a broadcast on the
  default group would have the two pipelines naming different sources for
  one collective. Gradient clipping all-reduces the squared norm over the
  pipeline group and takes no data-parallel term, because after the
  reduction every rank of a dp group holds the same gradients. The
  parameter-count check sums over the pipeline group for the same reason:
  over the world it would report `dp x param_count`. Each rank writes its
  own `rank<n>_trace.json.gz`.

  **`--pp 2`, `--dp 2`, and their four-rank composition have run.** On
  2026-08-23 the `baseline` arm completed and passed `validate_arm` at
  `--pp 2 --pp-schedule 1F1B`, at `--dp 2`, and at
  `--dp 2 --pp 2 --pp-schedule 1F1B`. In both data-parallel cells every rank
  passed arm rule 13, and ranks reading different data slices printed the
  same `grad_norm`. These are correctness results only: the host was loaded,
  no cell was repeated on an idle box, and no timing number is citable.

Environment notes: TE's native tuned RMSNorm kernels fail to launch on this
box's cuda-compat stack, so `configure_te_environment` routes norms through
TE's cuDNN backend (`NVTE_NORM_*_USE_CUDNN=1`) -- keep it in any process
that imports TE here. Without apex, megatron's standalone norms are torch
RMSNorm (its own spec fallback); the qkv-input norm fuses into the TE
linear.

**That routing makes the cuDNN identity a caption obligation, and the
obligation is about speed.** Every norm number from this arm, and from every
cross-engine kernel scenario that norms, is "TE norms via cuDNN backend" and
not TE's fastest norm. Which cuDNN serves that backend is a host property --
see "Which cuDNN a megatron arm runs is a host property" above -- and the
version changes no value, so the caption qualifies the timing and not the
numbers. `megatron_bootstrap.py` also sets
`CUDNN_FRONTEND_CUDART_LIB_NAME=libcudart.so.13` for the same class of
mismatch, arrived at independently. A host with a native CUDA 13 driver must
repeat the measurement.


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

## The stock Megatron arm

`piper_megatron_stock`'s `baseline` arm runs **stock Megatron-LM**: it hands
the run to `megatron.training.pretrain` and to `pretrain_gpt`'s own
providers, and substitutes **one** argument -- the dataset provider, so both
engines read the same c4_test stream rank for rank. The model builder, the
optimizer, the learning-rate schedule, the distributed setup, the forward
step, the embedding-rank rule and the training loop all stay Megatron's.

**This is not the tuned arm, and it does not replace it.** The tuned arm
(`piper1b_megatron/baseline`, `benchmarks/e2e/megatron/train.py`) replicates
the TorchTitan treatment step by step, and stays the "Megatron at its own
best" comparison. The stock arm asks a different question: what does a stock
user get?

### The four deliberate differences

**The claim is a systems-throughput claim about two configured engines.** It
is not a numerical-equivalence claim, and it is not a claim about the
Megatron engine as such. Four differences are deliberate, each one moves the
number, and **the report must state all four beside every number**:

1. **The stock arm keeps fp32 master weights and reduces gradients in
   fp32.** `--bf16` alone does that (`arguments.py`: with the default
   `--main-grads-dtype fp32` it sets `accumulate_allreduce_grads_in_fp32`),
   which is about 18 bytes of optimizer state per parameter against
   TorchTitan's 8. The titan arm and the tuned arm run plain bf16.
2. **The stock arm runs Megatron's unfused native cross entropy.** The tuned
   arm runs the TransformerEngine cross entropy. The native path upcasts the
   whole `[tokens, 151936]` logit tensor to fp32 and makes about six
   full-tensor passes. It is expensive: the tuned arm measured it at 88 GPU
   ms/step against 14.9 at batch 48. **That measurement is the tuned arm's
   and not this one's**; read "Cross-entropy implementation is a
   reporting-sensitive choice" above for it.
3. **The stock arm keeps `--init-method-std 0.01`.** The titan arm keeps
   TorchTitan's own initialization. **No weight transfer happens**, so the
   two arms do not start from the same parameters.
4. **The stock arm applies no permutation fusion**, because stock Megatron
   defaults `--moe-permute-fusion` off. The arm therefore does **not** pin
   `_permute_kernel` as a trace marker, where the tuned arm does.

**`--megatron-nan-guard off` adds a fifth difference, and it is not a
default.** Under `off` the stock arm runs without Megatron's NaN/Inf
checks, which is "stock minus the NaN detector", and the manifest's
`megatron_nan_guard` says so. State it beside the four above whenever a
number was taken under `off`; every published number of this scenario
was taken at `on`.

**`execution_model` says `plain-bf16` and describes the other arm.** The
field is composed from the parallelism spec, and
`single-gpu-plain-bf16-no-fsdp` is a fixed point every manifest since schema
7 carries; moving it to describe one arm would move a published constant. So
the difference is stated in three places a reader meets instead: the arm's
`description`, the scenario's `description`, and this section. Never read a
stock-arm number as plain bf16.

### The package

`benchmarks/e2e/megatron_stock/`, one file per concern:

| file | contents |
|---|---|
| `bootstrap.py` | `install_typing_override`, then `prepare()`, which also calls `configure_te_environment` and `add_megatron_to_path` |
| `data.py` | `StockReplayIterator` and the provider `--dataloader-type external` passes through |
| `flags.py` | `stock_megatron_flags`, the whole command line as data. Parent-side and torch-free |
| `model_builder.py` | `BenchGPTModelConfig` and `CountingGPTModelBuilder`, which print the two parameter lines |
| `profiling.py` | `install_profiler_shim` and `assert_windows_written` |
| `train.py` | The driver, and the log-line contract with `validation.py` |

**No file under `third_party/` is edited.** Three shims run in this process
instead, and each answers a way stock Megatron does not fit the harness:

- **`typing.override`.** `megatron.training` reaches
  `megatron/training/models/gpt.py`, which reads `override` from `typing`;
  that name arrived in Python 3.12 and this venv runs 3.10.
  `install_typing_override` adds the one name from `typing_extensions`.
  Moving the venv to 3.12 was the alternative: it rebuilds every wheel,
  rebuilds FlashAttention-3 from source, and makes every published number
  incomparable.
- **The profiler.** Stock Megatron writes **one** window per rank, to
  `{tensorboard_dir}/../torch_profile/rank-<n>.json.gz`, with `repeat=1`.
  The harness reads
  `<arm>/profiling/traces/iteration_*/rank<n>_trace.json.gz` and arm rule 5
  needs two windows. `install_profiler_shim` replaces
  `torch.profiler.profile` on `training.py` before `pretrain()` reaches it
  and gives Megatron the right schedule and the right path. **The repair is
  never to lower the rule.** `assert_windows_written` refuses a run whose
  shim was called anything but once, and a run that wrote fewer windows than
  the workload declares, so a shim that did not install fails both guards.

  **`--steps` must be a whole number of profiler cycles for this arm, and
  `stock_megatron_flags` refuses anything else.** Megatron steps the
  profiler at the top of every pass and stops it at the bottom of one, so
  an iteration after the stop transits a dead Kineto session. Ending the
  profiler early only delays that; ending it at `--train-iters` trades it
  for a truncated window, which `assert_windows_written` does not catch
  because it refuses a short count and not a short window. A whole number
  of cycles makes `--profile-step-end` equal `--train-iters`, so no
  iteration follows the stop. With the 40-step floor and `profile_freq` 20,
  the accepted values are 40, 60, 80 and so on. **The refusal is
  parent-side**, so a bad `--steps` fails before a GPU is claimed.

  **The schedule carries `skip_first=1`, and that is a comparability
  property rather than a detail.** The two engines step the profiler at
  opposite ends of the loop body -- Megatron first, TorchTitan after
  `train_step` -- so the same training step ran one schedule index apart.
  `stable_tps` samples steps 2 to 10 of every cycle, and the
  `NONE -> WARMUP` transition, which runs torch's `prepare_trace` and its
  CUPTI setup, landed on step 10 for Megatron and step 11 for TorchTitan.
  Two of the eighteen samples behind the published median therefore carried
  profiler setup on one engine only. No validation rule reads the profiler
  phase, so nothing saw it. The offset puts both engines on the same action
  at every step. Its cost is that the last window is closed by Megatron's
  own `prof.stop()`, which writes the window in full; a stop that does not
  fire loses that window and `assert_windows_written` refuses the run.
- **The step line.** Megatron's own `training_log` prints an `iteration ...
  elapsed time per iteration (ms)` line on the last rank only.
  `install_step_log_shim` prints the line `benchmarks/e2e/results.py`
  parses, in the shape the tuned driver and the TorchTitan trainer both
  print.

**`--bench-` names the harness flags**, and `train.py` adds them through
Megatron's own `extra_args_provider` hook, so Megatron's parser owns them
and an unknown one fails at parse time rather than being ignored.

### The data contract

`data.py` reshapes `benchmarks/e2e/megatron/data.py`'s
`materialize_titan_samples` and adds nothing to it, so the stock arm reads
the byte-identical stream the tuned arm and the titan replay loader read.
Three rules govern the iterator, and each answers a way a run can be wrong:

1. **Every rank builds one.** `--dataloader-inter-document-masking` makes
   the middle pipeline stages read the batch too, for the `cu_seqlens` the
   attention needs.
2. **The key is the data-parallel rank, never the global rank.** The stages
   of one pipeline train one model on one batch, so they must read the same
   tokens in the same order; `mpu.get_data_parallel_rank()` returns the same
   value on every stage of one pipeline.
3. **Exhaustion raises.** A wrap would train a second epoch under the first
   epoch's label, and no validation rule would see it.

**One microbatch is one packed sequence, never a batch of rows.** Megatron
flattens an `(m, S)` microbatch to `(1, m*S)` whenever `cu_seqlens` is
present, and then allocates its pipeline receive buffer as `(S, m, H)` --
the same element count and a different layout, so the next stage would read
a permuted activation. The iterator therefore concatenates `rows_per_sample`
titan rows into one `(1, rows * seq_len)` sample and the harness sends
`--micro-batch-size 1`. The attention is unchanged: `cu_seqlens` already
marks every document, and every titan row starts at position 0, so a row
boundary is a document boundary.

### What is not settled

**At one rank the stock driver fills `MASTER_ADDR` and `MASTER_PORT`
itself.** `torch.distributed.run` sets both above one rank. At one rank
nothing did, and Megatron's `_initialize_distributed` calls
`init_process_group` with no store, so the `env://` rendezvous failed
before a step trained. `install_rendezvous_defaults` in
`benchmarks/e2e/megatron_stock/train.py` mirrors the tuned driver's two
`setdefault` calls, since 2026-09-02. No single-GPU cell of this scenario
exists before that date.

**One mesh has run.**
`out/20260826T172258Z/piper_megatron_stock/nvidia-h200` holds both arms at
the `1b` shape, `--dp 2 --pp 4`, both `completed`, with a `results.json`.
Re-derive that from `out/` rather than quoting it.

**That manifest is schema 11, so it carries no `dense_sharding` key, and
an absent key may not be read as `replicate`.** That is the whole reason
the schema went to 12.

**Each engine's half has to be read from that engine's own evidence, and
one field does not cover both.** The record's `dp_replicate: 2,
dp_shard: 1` is the **TorchTitan** mesh -- `describe`'s own docstring says
so, and says Megatron is told neither value -- so it settles the titan arm
and asserts nothing about the megatron one.

**The megatron half rests on `commands["baseline"]`, which carries none of
the five sharding flags.** That is sufficient on its own:
`megatron/training/training.py` picks the wrapper with
`elif args.use_megatron_fsdp: DP = FullyShardedDataParallel` and
`else: DP = DDP`, so an argv without the flag reaches the plain
`DistributedDataParallel` branch.

**Read that cell's `Megatron-LM stock data parallel:` line with care, and
never as a comparison against a class name it could not print.** At
`509c716`, the rev that manifest records, `DATA_PARALLEL_LINE` **hardcoded
the literal string `DistributedDataParallel`**. Templating the class name
into that line is a change made after the cell ran, so the word in that log
names no observation and discriminates nothing.

**What the line does prove is that it printed at all.** At that same rev
`install_data_parallel_marker` filtered the model chunks on
`isinstance(chunk, DistributedDataParallel)` -- the narrow class -- and
**raised** when nothing matched. `FullyShardedDataParallelV1` derives from
`_BaseDataParallel` beside `DistributedDataParallel` rather than from it,
so it could not have satisfied that filter. A printed line is therefore an
observation that the wrapper really was a `DistributedDataParallel`. Do not
paraphrase this as "the log names the wrapper": at that rev it did not.

Its log also carries the pre-schema-12 marker shape, with no `ep=` field
and no `sharding_strategy=` field, so the block quoted above is not the
block that cell wrote. No published number is at risk: `evaluate` re-reads
no marker, and `--resume` already refuses the directory on the changed
`benchmarks_git_rev`.

**Every other mesh of this scenario is still a declaration.** No sharded
cell, no expert cell and no depth-8 pipeline has executed on either arm.
Read those as never-built kernel scenarios are read: report what they
declare, never what they measure.

Four items were expected rather than measured. **The first is now
settled**; the first run of each further mesh settles the rest:

- **The two trace markers: SETTLED, and on every rank.** Both
  `cudnn_generated_fort_native_sdpa` and `_mul_silu_split` appear in
  **all eight ranks of both profiler windows** of the `dp 2 x pp 4` cell
  above, at 320 and 640 per window per rank. `_permute_kernel` is 0 on
  every rank, which is correct and is why this arm does not declare it:
  stock Megatron defaults `--moe-permute-fusion` off. Because the counts
  are uniform across the eight ranks, arm rule 6's "any rank" reading costs
  this arm nothing **at this mesh and this shape**. It is one cell; a `pp 8`
  split gives each stage fewer layers, so repeat the reading there.
- Arm rule 13's `ncclDevKernel_AllReduce` marker under Megatron's DDP
  bucketing at world 8. A grouped launch can surface as
  `ncclDevKernel_Generic`. Read every rank's trace before you widen it, and
  never widen it to a bare `nccl`.

**Arm rule 13 proves nothing on this arm, and the wrapper check is what
carries the axis.** Stock Megatron all-reduces the reported loss over the
data-parallel group on every last-stage rank every step, and above `pp` 1
the gradient-norm reduction puts a collective on every rank anyway. So
`ncclDevKernel_AllReduce` appears whether or not a gradient moved. What
closes the axis is `install_data_parallel_marker`, which raises when no
model chunk carries a `_BaseDataParallel`. Treat that one function as
load-bearing: the whole data-parallel axis of this arm rests on it, and
under `--dense-sharding shard` it is also the only observed proof that
Megatron sharded -- it reads the strategy off the wrapper, not off the
argv.
- Whether Megatron's `--lr-decay-iters 40` decays over the 38 post-warmup
  steps, as TorchTitan does. The rate does not change the throughput, so a
  mismatch is a reporting defect.
- Whether the `9b` shape fits at batch 32 on this box. The fallback is batch
  8 with microbatch 1, which gives the same 8 microbatches.

**Eight GPUs on this box span two NUMA nodes, so every eight-rank cell runs
unpinned.** `cpu_pinning` records the reason. An unpinned run is not
comparable to a pinned one; say so beside the number.

**`tools/run_matrix.sh` cannot drive this scenario, and the gap is wider
than the scenario name.** It sweeps `--compile-mode` and `--model-size`
only. It sends no `--dp`, `--pp`, `--ep` or `--dense-sharding` at all, so
it cannot express one cell of the planned matrix on any scenario, and it
never names `piper_megatron_stock`. An operator following the rule to
"drive multi-cell matrices with `tools/run_matrix.sh`" therefore has no
tool for this axis and runs the cells by hand, losing the dirty-tree
refusal, the `flock`, the idle-GPU wait and the contamination watchdog.
Extending it is separate work, and it is the largest piece of unbuilt work
this axis leaves behind.

## Comparing against Piper's artifact

Piper's published Megatron baseline loses ZeRO-1 whenever the expert degree
equals the data-parallel degree. Five handicaps sit on its TorchTitan arm.
**Read the `piper-comparison` skill before you compare any number of ours
against Piper's paper** (arXiv 2606.11169). The skill holds the defect, the
evidence, and the rules for what may be said.

## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

The last full run at this rev discovered 1813 tests and skipped 11. Re-derive
those counts rather than quoting them; `tests/test_migration_contract.py`
carries `TEST_CENSUS` and `TEST_CENSUS_TOTAL`, and the total is the **sum of
the dict**, recomputed at every commit that changes a count. Never add
deltas. The census does not cover every module -- `test_import_boundaries`,
`test_legacy_artifacts` and `test_migration_contract` are outside it -- so
`TEST_CENSUS_TOTAL` is smaller than the suite's own count, by design.

GPU tests skip themselves when CUDA is unavailable. `test_te_rope.py`
additionally requires g++ >= 13 and JIT-builds the CUDA extension on import.

**Three tests in `test_megatron_data.py` skip unless the datasets cache is
writable**, and they are not minor ones: they hold the disjoint per-rank slice
that the data-parallel axis rests on, the global padding target, and the
bit-identical stream parity against titan's loader. The skip names its own
cause and the fix. `run_bench.sh` exports a writable `HF_DATASETS_CACHE`; a
bare `unittest discover` does not, so run the suite with the same variable
when you touch the megatron data path:

```bash
HF_DATASETS_CACHE="$HOME/.cache/hf-datasets" .venv/bin/python -m unittest discover -s tests
```

See the `HF_HOME` bullet under "Environment" for why the shared cache fails
only sometimes.

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
  weights). These are `qkv_prep`'s two titan arms since the single-engine
  `qkv` scenario was deleted
- `FlexAttention`, `VarlenAttention`, `VarlenMetadata`,
  `create_attention_mask`, `create_varlen_metadata_for_document`,
  `get_causal_mask_mod` and
  `get_efficient_causal_mask_mod_for_packed_document` from
  `torchtitan.models.common.attention` -- the whole `attention_core` titan
  side, plus the mask forms every arm of that scenario shares. The FA3 and
  FA4 marker strings `FlashAttnFwdSm90` and `FlashAttentionForwardSm90` are
  pinned by the arms' guards and are generated by CUTLASS and the CuTe DSL,
  so they appear nowhere in the torch source
- `AllToAllTokenDispatcher` and `make_token_dispatcher_config` from
  `torchtitan.models.common.token_dispatcher` / `config_utils`, and
  `CosSinRoPE` and `_qwen3_norm` for the rope and norm cuts
- `CrossEntropyLoss` from `torchtitan.components.loss`

Also recheck the documented deltas vs Piper: the builder hardcodes
`route_norm=True` (Piper wants `False`), experts are `GroupedExperts` rather
than `BmmExperts`, and the `load_balance_coeff = None` fixup is applied
post-hoc and silently stops mattering if the builder default changes.

`piper1b_lm_head` depends on TorchTitan commit `43d328ad`, which generalized the
trainer's LM-head handoff to the `LossWithLMHead` protocol. Only
`FusedLinearCrossEntropyLoss` implements it; the TE arms do not.

## Operating rules

- Check `nvidia-smi` for a free GPU before starting -- **one per rank**, and
  the rank count is `--dp x --pp`, so either at 2 needs two GPUs. A shared
  GPU invalidates timings.
- Use at least 40 steps. The runner enforces this; do not try to route around it.
- Numbers are only comparable within one `torch_version`, one
  `torchtitan_git_rev`, one `benchmarks_git_rev`, one `compile_mode`, one
  `ac_mode`, one `model_size`, one `parallelism` record, one
  `megatron_p2p_sync` value and one `megatron_nan_guard` value (plus one
  `megatron_git_rev`/`te_version` for either megatron scenario). A pipelined
  run also declares no regions, so it carries no `forward_block` or
  `backward_block` row a single-GPU run could be compared against.
  `cudnn_loader_resolves` is a **speed** axis only: the version changes no value, measured, so cite it beside a timing
  number and never call two runs numerically incomparable for it. All are in
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
- After changing anything in `benchmarks/`, run the full test suite. GPU-gated
  tests skip themselves when their prerequisites are unavailable; re-derive
  the test and skip counts from the run.
- **Do not let "declared" become "measured".** Much of the kernel registry has
  never executed: 8 of the 16 cross-engine scenarios have never had an arm
  built, the single-engine `lm_head` has not run, 29 of the 71 declared arms
  have ever reported `ok`, and no span has been measured at all. Report what
  such a scenario declares, never what it measures. Re-derive the counts; the
  command is under "What has been measured".


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
