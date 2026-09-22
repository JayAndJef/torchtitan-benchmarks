# torchtitan-benchmarks: Agent Guide

## 1. What this repository measures

Two measurement systems share one CLI. Never mix their numbers.

**End-to-end throughput.** One scenario, `engines`, trains the same
Qwen3 MoE model on one pre-tokenized c4_test stream and publishes tokens/s,
step time and peak memory. `--model-size` picks the shape. It has three
arms:

| arm | engine | treatment |
|---|---|---|
| `titan_compiled` | `torchtitan` | whole-block `torch.compile` |
| `titan_eager` | `torchtitan` | the same model, blocks eager |
| `megatron_stock` | `megatron_stock` | stock `megatron.training.pretrain` |

**Kernel isolation.** `kernel-bench` times competing implementations of one
model component head-to-head on synthetic tensors. A kernel that wins in
isolation can be irrelevant once the compiler fuses the graph around it.

A kernel number is never an end-to-end number, and an end-to-end number is
never a kernel number. State which system produced a figure.

The `engines` scenario carries four deliberate differences, and each one
moves the number. The Megatron arm keeps fp32 master weights, reduces
gradients in fp32, runs Megatron's unfused native cross entropy, keeps
`--init-method-std 0.01` with no weight transfer, and applies no permutation
fusion. State all four beside every cross-engine number.

## 2. Environment

One environment owns this repository. Every command runs under
`.venv/bin/python`, and `run_bench.sh` execs that interpreter directly.

```bash
git clone --recurse-submodules <repo> && cd torchtitan-benchmarks
./sync.sh
```

`sync.sh` wraps `uv sync` in two passes. The first pass installs torch and
the NVIDIA header wheels. The second pass builds the groups that compile
without build isolation against the pinned nightly. It also links
`tools/pre-push.sh` into the common git hook directory, so every worktree
shares one pre-push hook.

- Three default dependency groups: `megatron` (TransformerEngine), `flash3`
  (a CUTLASS sm90a source build) and `fa4`.
- **Any resolution change rebuilds FA3 from source.** Budget 15 to 40
  minutes for a dependency change. Never benchmark while it runs, because
  the build saturates the host and this workload is host-bound.
- Skip the long build with `./sync.sh --no-group flash3`.
- torch is pinned to an exact nightly. The nightly index keeps roughly 60
  days, so the pin needs a bump eventually. A bump changes the numbers.
  Rerun the baselines and do not compare across it.
- `run_bench.sh` sources `cuda_compat.sh`. On a driver that reports less
  than CUDA 13.0, that script stages NVIDIA's forward-compat userspace
  driver under `.cuda-compat/` and prepends it to `LD_LIBRARY_PATH`. It is
  a no-op on a driver that already reports 13.0 or more.
- TorchTitan is a submodule at `third_party/torchtitan`, installed editable.
  It is our fork, pinned on the `bench/torchtitan-benchmarks` branch.
- Megatron-LM is a submodule at `third_party/Megatron-LM`. It is **not**
  pip-installed. `benchmarks/models/piper_qwen3/megatron_bootstrap.py` puts
  it on `sys.path`.
- `run_bench.sh` exports `HF_DATASETS_CACHE`, because a shared `HF_HOME` may
  belong to another user.

## 3. Repository map

| path | contents |
|---|---|
| `benchmarks/cli/` | The Click CLI: `benchmarks/cli/e2e.py`, `benchmarks/cli/kernel.py`, and the group in `benchmarks/cli/main.py`. |
| `benchmarks/e2e/` | The end-to-end system: `benchmarks/e2e/schema.py`, `benchmarks/e2e/axes.py`, `benchmarks/e2e/registry.py`, `benchmarks/e2e/parallelism.py`, `benchmarks/e2e/launch.py`, `benchmarks/e2e/validation.py`, `benchmarks/e2e/engines.py`, `benchmarks/e2e/runner.py`, `benchmarks/e2e/results.py`. |
| `benchmarks/e2e/megatron_stock/` | The stock Megatron-LM driver and its command line. |
| `benchmarks/e2e/data/piper_qwen3.py` | The pre-tokenized replay dataloader. |
| `benchmarks/artifacts/` | `manifest.json`, `run_state.json`, the output layout and the atomic JSON writer. |
| `benchmarks/traces/extraction.py` | Chrome-trace parsing, used under `--profile` alone. |
| `benchmarks/execution/` | Subprocess environment, device parsing, CPU pinning and provenance. |
| `benchmarks/kernel/` | The kernel-isolation system: registry, spans, runner, worker, timing engine, results. |
| `benchmarks/models/piper_qwen3/` | The model port: `benchmarks/models/piper_qwen3/shape.py`, `benchmarks/models/piper_qwen3/config_registry.py`, `benchmarks/models/piper_qwen3/parallelize.py`, the Megatron model builder and the kernel components. |
| `tools/` | `tools/run_matrix.sh`, `tools/collect_matrix.py`, `tools/pre-push.sh`, and the knowledge-base scripts. |
| `tests/` | The CPU and GPU test suite. |
| `third_party/torchtitan/` | Our TorchTitan fork, pinned. |
| `third_party/Megatron-LM/` | Upstream Megatron-LM, pinned, on `sys.path` only. |
| `out/` | Run outputs. Gitignored. |
| `reports/` | Investigation notes. Gitignored. Put conclusions here. |

## 4. The `run` command

```bash
./run_bench.sh scenarios                    # list scenarios and arms
./run_bench.sh run <gpu> [OPTIONS] [-- TORCHTITAN_ARGS]
./run_bench.sh evaluate <out_dir> [--arm NAME]... [--results PATH]
```

`<gpu>` is a PCI index, or a comma list of them. The runner sets
`CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES` and `NGPU`, so the
index is stable and the world size follows the request.

`run` executes, validates and evaluates. It runs every scenario unless
`--scenario` narrows the set, and it is fail-fast: the first failing arm
stops it. Pass extra TorchTitan arguments after `--`.

Named scenarios run one at a time, in the order given, and a name may
repeat. Each repeat is another run of that scenario. The second and later
runs of one name take a `-run<n>` suffix on the scenario directory, so they
do not overwrite each other. `manifest.json` records the plain name.

**The default shape does not fit one GPU.** `30b-a3b` is the paper headline
shape: 30,532,122,624 parameters, about 227.5 GiB for the TorchTitan arms
against an H200's 139.81 GiB. The harness does not refuse a one-GPU run of
it, because the parallelism rules read layer counts and not memory; the
training process runs out of memory instead. Pass `--model-size 1b` for a
one-GPU run.

The default therefore needs a mesh that splits it. `--pp 8` also needs
`--batch 16`, because parallelism rules 11 and 12 ask for a microbatch
count that divides the pipeline degree and is at least twice the stage
count.

### Flag table

Each row of the table below is `| flag | env | default | meaning |`. The
flag cell and the default cell each hold one backticked literal. A default
cell of `--` means the option has no default value. `tests/test_docs.py`
parses this table and compares each default against the code.

| flag | env | default | meaning |
|---|---|---|---|
| `--scenario` | -- | `--` | Scenario to run, in the order given; repeat a name to run it again. Omit to run every scenario. |
| `--arm` | -- | `--` | Arm subset; repeat per arm. It applies to every selected scenario. |
| `--resume` | -- | `--` | Resume an output directory and retry the incomplete arms. |
| `--results` | -- | `--` | JSON destination for the evaluation. |
| `--hardware` | -- | `auto` | Provenance label; `auto` uses the GPU name. |
| `--out` | `OUT` | `--` | Output directory. |
| `--seq-len` | `SEQ` | `4096` | Sequence length. |
| `--steps` | `STEPS` | `40` | Training steps per arm. |
| `--batch` | `BATCH` | `4` | Local batch size. |
| `--cache-root` | `BENCHMARK_CACHE_ROOT` | `--` | Root of the build caches. |
| `--compiler-env` | `BENCH_COMPILER_ENV` | `--` | Shell script that enables the host compiler. |
| `--ac` | `AC_MODE` | `none` | Activation checkpointing: `sac` or `none`. |
| `--model-size` | `MODEL_SIZE` | `30b-a3b` | Model shape from `benchmarks/models/piper_qwen3/shape.py`. |
| `--dp` | -- | `1` | Data-parallel degree. |
| `--pp` | -- | `1` | Pipeline-parallel degree. |
| `--ep` | -- | `1` | Expert-parallel degree. |
| `--pp-schedule` | -- | `--` | Pipeline schedule; required above one pipeline rank. |
| `--pp-microbatch-size` | -- | `1` | Rows per pipeline microbatch. |
| `--zero` | -- | `0` | ZeRO level for the dense parameters: `0` or `1`. |
| `--megatron-p2p-sync` | -- | `off` | Whether Megatron synchronizes after a pipeline message. |
| `--megatron-nan-guard` | -- | `off` | Whether stock Megatron keeps its NaN and Inf check. |
| `--megatron-precision` | -- | `stock` | How stock Megatron holds the optimizer state. |
| `--profile` | -- | `off` | Whether the run collects profiler traces. |
| `--warmup-steps` | `WARMUP_STEPS` | `10` | Steps an unprofiled run discards before it measures. |

The six parallelism options and the three Megatron options take no
environment variable. Each one has to agree with the `<gpu>` positional or
with `--arm`, and a positional has no environment form.

### The comparability rule

Numbers are comparable only within one value of each of these: `model_size`,
`ac_mode`, `zero`, the parallelism spec, `megatron_p2p_sync`,
`megatron_nan_guard`, `megatron_precision`, `profile`, `warmup_steps`,
`torch_version`, `torchtitan_git_rev`, `benchmarks_git_rev` and
`megatron_git_rev`. Every one of them sits in `manifest.json`. Check them
before you compare against an older run.

The Megatron sync option and the Megatron NaN-guard option both default to
`off`. Every number published before that flip was measured under `on`.
State the value beside any Megatron number.

The lean precision value changes the numerics. It holds bf16 Adam moments
and accumulates gradients in bf16. Read the loss trajectories beside any
lean number. At one data-parallel rank the distributed optimizer also does
bucket bookkeeping for no saving. Both effects are unmeasured.

### The profile axis and the step floor

`--profile` is off by default, and off is the treatment a published
throughput number wants. The profiler costs GPU and host time on every
window.

- Off: the run writes no trace. The evaluation samples every step after the
  warmup, so `--steps` must be **more than** `--warmup-steps`.
- On: both engines write `<arm>/profiling/traces/iteration_*/`, every trace
  rule applies, and `--steps` must be at least 40. Two profiler windows are
  the floor, and `--warmup-steps` is refused beside `--profile`.

An exported `WARMUP_STEPS` refuses every profiled run. Unset it first.

### Resume

`run --resume <out_dir>` re-validates each arm against the files on disk,
skips the arms that pass, archives the partial artifacts under
`attempts/<timestamp>/<arm>/`, and re-runs the rest.

- `--resume` and `--out` cannot be combined.
- `--resume` needs exactly one selected scenario, as `--out` and `--results`
  do.
- An omitted axis inherits the recorded value. A different value is refused.
- A resume does **not** inherit the parallelism spec. Omitted parallelism
  flags ask for the single-GPU spec.
- The resume refuses a change of scenario, workload, selected arms, hardware
  label, extra TorchTitan arguments, any of the eight run axes, `nvidia_smi`,
  `cpu_pinning`, `torchtitan_git_rev`, `benchmarks_git_rev` or
  `megatron_git_rev`.

### Output layout

```
out/<timestamp>/<scenario>/<hardware>/
  manifest.json     # schema 17
  run_state.json    # per-arm status, attempts, evaluation status
  results.json      # schema 6
  <arm>.log         # training stdout and stderr, every rank
  <arm>/profiling/traces/iteration_*/rank<n>_trace.json.gz   # --profile only
  attempts/<ts>/<arm>/   # archived artifacts of a failed attempt
```

The manifest records the scenario, the arms, one command line per arm, the
eight run axes as flat keys, the model shape, the execution model, the
throughput definition and `hardware_metadata`. The reader accepts schema 17
alone and refuses any other version by name.

The `tps` figure of a step line, and `stable_tokens_per_second`, are **per
device**. Both engines divide one rank's token count by `cp * tp * pp`. The
data-parallel degree is absent from that divisor, because each data-parallel
rank reads a batch of its own. The manifest records the definition as
`throughput_definition`.

### CPU pinning

The training step is host-bound at these sizes, so an unpinned run measures
scheduler placement. The runner binds each training process to its GPU's own
NUMA node with `numactl --cpunodebind --membind`, resolved from the PCI bus
id through sysfs. When that fails the run proceeds unpinned and
`cpu_pinning` records why. Pinned and unpinned runs are not comparable, and
`--resume` refuses to mix them.

## 5. Validation

`benchmarks.e2e.validation:validate_arm` gates every arm before its numbers
are published. Each log rule runs **once per rank**, because one `<arm>.log`
holds every rank's output. Above one rank, a rank that wrote no log and a
rank that wrote no trace each fail the arm.

The numbering below is the numbering the code uses. Rules 7 and 9 are
deleted and their numbers stay empty, because messages and tests name the
rules that remain.

| rule | what it refuses | needs the profile axis |
|---|---|---|
| 1 | a missing log, or a log without the engine's completion marker | no |
| 2 | an `[Override]` count other than `overrides_per_block * n_layers` | no |
| 3 | a declared `override_imports` entry with no matching line | no |
| 4 | a silent-fallback phrase in the log | no |
| 5 | fewer trace files than `min_trace_windows`, per rank | yes |
| 6 | a declared `trace_kernel_markers` string absent from every trace | yes |
| 7 | DELETED with the compiled-region measurements | -- |
| 8 | a compile log line that contradicts the arm's own compile value | no |
| 9 | DELETED with the cuda-graph compile mode | -- |
| 10 | a SelectiveAC line that contradicts `--ac` | no |
| 11 | a missing `size: <N> total parameters` line for the shape | no |
| 12 | a log that does not state the mesh, the sync value, the NaN-guard value or the precision the run asked for | no |
| 13 | a rank whose traces carry no all-reduce kernel, above one data-parallel rank | yes |

Rule 8 reads both ways. The marker must be present when the arm declares
`compile="torch"` and absent when it declares `compile="none"`. Never relax
the absence half, or an arm that silently compiled publishes as eager.

Rule 12 also reads two patterns the other way round. A log that records a
pipeline fails at one pipeline rank, and a log that records data parallelism
fails at one data-parallel rank. The second one guards the worse mistake: a
two-way data-parallel run published as one GPU reads as roughly twice the
true rate.

Without the profile axis the arm writes no trace, so rules 5, 6 and 13 are
skipped whole. Rule 12 then carries the data-parallel axis alone. Cite a
data-parallel number from an unprofiled run as resting on that log line.

Rule 13 matches the kernel name `ncclDevKernel_AllReduce`. Megatron issues
its bucket reductions inside a coalescing manager, and a grouped NCCL launch
can surface under a generic name. Read a Megatron rule 13 failure as a
question about the marker string, and settle it with the arm's own trace.
Widening the marker to a bare `nccl` is not the repair. The marker is
necessary and never sufficient: an all-reduce proves a collective ran, never
which one. Read rule 12 and rule 13 together.

The engine differences live in two profiles, `TORCHTITAN_PROFILE` and
`MEGATRON_STOCK_PROFILE`. `benchmarks/e2e/engines.py` puts each one on an
engine record beside that engine's command builder, so an arm cannot take
one engine's argv and another engine's log rules.

## 6. Evaluation

`run` always evaluates, and `evaluate <out_dir>` re-evaluates a finished
directory. **Evaluation reads the logs alone.** Both engines print every
published figure on a step line, so a directory evaluates the same way under
either profile value.

`results.json` is schema 6. Per arm it carries `stable_tokens_per_second`,
`stable_sample_count`, `peak_memory_gib`, `step_ms` with `mean`, `median`,
`p95` and `series`, plus `rank_reduction`, `published_rank` and a `per_rank`
list. The file also carries `losses`, `gradient_norms` and `warnings`.

`benchmarks.e2e.results:step_ms` derives the step cost from the throughput:

```
step_ms = 1000 * local_batch_size * seq_len / (tps * pp)
```

The published rank is the **slowest** rank, never the mean. A parallel
schedule locks the ranks together at every step boundary, so the mesh runs
at the pace of its slowest rank. `peak_memory_gib` is the maximum over the
ranks. The 95th percentile uses the nearest-rank method, so it is always a
measured step.

The sample rule follows the recorded axis.
`benchmarks.e2e.results:measured_tps` takes every step after the warmup in
an unprofiled run. `benchmarks.e2e.results:stable_tps` takes the steps of
each profiler cycle that carry no profiler cost. The two figures are
different figures, not two readings of one.

`benchmarks.e2e.results:refuse_non_finite_trajectories` fails an arm whose
step lines carry a `nan` or an `inf`, on any rank. A run that diverged
publishes no throughput.

The evaluation warns when tokens/s spreads more than 1.15x across ranks, and
it repeats the ZeRO-level warnings the runner printed.

**What is deliberately absent.** There is no baseline arm, no ratio, no GPU
kernel time, no per-region measurement, no launch latency and no
significance test. The three arms share no implementation, so a reader
compares two absolute rows.

Send a kernel-level question to `kernel-bench`. Send a trace-level question
to the external trace-anatomy tool, which reads the trace layout a profiled
run writes.

### A known gap: V-shaped pipeline schedules

`benchmarks.e2e.results:loss_visible_rank` returns
`(world_size // pp) * (pp - 1)`. That is right for `1F1B` and
`Interleaved1F1B`. It is wrong for `ZBVZeroBubble` and `DualPipeV`, where
rank 0 holds the last stage and the loss. Parallelism rule 5 refuses a
V-shaped schedule only beside a Megatron arm, so a TorchTitan-only run may
ask for one. No run has ever used one. Repair `loss_visible_rank` before you
run one. Do not lift `MAX_PP` instead.

## 7. The stock Megatron arm

`benchmarks/e2e/megatron_stock/` holds everything about Megatron. The
harness connects through the arm's `engine` name alone.

| module | job |
|---|---|
| `benchmarks/e2e/megatron_stock/bootstrap.py` | Sets the process environment the driver needs before torch, adds `typing.override` for Python 3.10, then puts Megatron on `sys.path`. |
| `benchmarks/e2e/megatron_stock/flags.py` | The whole Megatron command line, as data, plus the harness parser and the run refusals. Torch-free, so a CPU test reads it. |
| `benchmarks/e2e/megatron_stock/markers.py` | The log lines this arm prints, and the functions that format one. |
| `benchmarks/e2e/megatron_stock/data.py` | Drains TorchTitan's own c4_test dataset class and feeds it as an external dataloader. |
| `benchmarks/e2e/megatron_stock/model_builder.py` | Builds the stock GPT model and prints the parameter count rule 11 reads. |
| `benchmarks/e2e/megatron_stock/dp_marker.py` | The data-parallel line, printed from the wrapper Megatron really built. |
| `benchmarks/e2e/megatron_stock/step_log.py` | The step line the evaluation parses, and the shim that prints it. |
| `benchmarks/e2e/megatron_stock/profiling.py` | Gives Megatron the profiler schedule and the trace path the harness reads. |
| `benchmarks/e2e/megatron_stock/train.py` | Reproduces the stock training entry point and calls `pretrain`. |

The driver substitutes **one** provider, the dataset provider. The model
builder, the optimizer, the schedule, the distributed setup, the forward
step and the training loop all stay Megatron's. No file of the Megatron-LM
checkout is edited; four shims run in the driver process instead.

Faithfulness guarantees that hold today:

- **Same shape.** Both engines build from the same shape record in
  `benchmarks/models/piper_qwen3/shape.py`.
- **Same data.** The driver drains TorchTitan's own dataset class with
  TorchTitan's own tokenizer. The test suite asserts that the stream is
  bit-identical to the TorchTitan replay loader's.
- **Same batch mapping at one pipeline rank.** The flag list sends
  `--micro-batch-size 1`, and the geometry packs the whole local batch into
  one Megatron sample there. So that ratio is not biased by the mapping.
- **No recompute, ever.** The `--ac` axis never reaches this arm.
- **Stock precision.** The arm is **not** plain bf16. Under `--bf16` alone
  Megatron keeps fp32 master weights and fp32 Adam moments and reduces
  gradients in fp32, which is about 18 bytes of state per parameter against
  TorchTitan's 8. That is the stock treatment and the scenario keeps it.
  The manifest's `execution_model` says plain bf16, because it is composed
  from the parallelism spec; it describes the TorchTitan arms alone.

The driver prints one training-loop line that names `main_params_dtype`,
`main_grads_dtype`, `use_precision_aware_optimizer`, `exp_avg_dtype`,
`exp_avg_sq_dtype`, `accumulate_allreduce_grads_in_fp32`,
`cross_entropy_loss_fusion` and `moe_token_dispatcher_type`. Rule 12 matches
that prefix and the four precision fields.

The arm declines `--cross-entropy-loss-fusion`, so it runs Megatron's own
unfused native cross entropy. That path upcasts the full logits to fp32 and
traverses them several times, and it is a large part of the engine gap. Say
so beside any loss-path claim. The arm also declines
`--moe-permute-fusion`, `--overlap-grad-reduce` and
`--overlap-param-gather`.

The three Megatron run axes are `--megatron-p2p-sync`,
`--megatron-nan-guard` and `--megatron-precision`. Each one reaches the
stock Megatron command alone, and each is refused parent-side when it
reaches no arm of the selection. The sync value `on` also needs more than
one pipeline rank, and the lean precision also needs `--zero 1`.

## 8. Parallelism

`benchmarks/e2e/parallelism.py` owns the degrees, the schedules and the
rules. One parallelism spec describes a whole run, so every arm shares it.
`--dp`, `--pp`, `--ep`, `--zero`, `--pp-schedule` and `--pp-microbatch-size`
build it.

The world size is `dp * pp`. **Expert parallelism does not multiply it.**
Both engines take the expert ranks out of the data-parallel axis. Tensor and
context parallelism are deliberately absent.

`--zero 0` keeps a whole copy of the dense parameters on every rank.
`--zero 1` shards the optimizer states: Megatron gets
`--use-distributed-optimizer`, and TorchTitan gets the whole data-parallel
width as its shard degree plus `fsdp-reshard-after-forward never`.

`validate_parallelism` refuses a spec before any host probe. The numbering
below is the code's own, and rules 6, 13, 15, 16 and 17 are deleted.

| rule | refuses |
|---|---|
| 1 | a world size other than the number of devices requested |
| 2 | a pipeline degree above 8, then a world size above 8 |
| 3 | a schedule or a microbatch size at one pipeline rank, and a missing schedule above it |
| 4 | a schedule name the registry does not declare |
| 5 | a schedule Megatron-LM does not implement, beside a Megatron arm |
| 7 | a layer count that does not divide the total stage count |
| 8 | an expert degree above the shape's expert count, or one that does not divide it |
| 9 | an expert degree that does not divide the data-parallel degree |
| 10 | a batch that does not divide into whole microbatches |
| 11 | a microbatch count that does not divide the pipeline degree |
| 12 | fewer microbatches than twice the total stage count, above one pipeline rank |
| 14 | an expert degree above 1 under `--zero 0` |

Rule 6 left this function, because compile is an arm property and a spec
alone cannot answer it. `benchmarks.e2e.runner:_resolve_run` now refuses a
schedule whose `requires_uncompiled` is true when a selected arm declares
`compile="torch"`, and the message names the arm. Three of the five
registered schedules raise on a compiled stage module.

Two legal meshes warn rather than refuse.
`benchmarks.e2e.parallelism:zero_warnings` states both, the runner prints
them, and `results.json` records them.

- A sharded level at one data-parallel rank: the shard degree is 1 whatever
  the level says, so the run holds the dense parameters as a replicated run
  holds them.
- `--zero 1` at one pipeline rank: one microbatch puts the gradient
  reduce-scatter inside the only backward pass, so the **TorchTitan arms**
  hold ZeRO-2 rather than the ZeRO-1 shape the level names. Megatron holds
  ZeRO-1 at every mesh. Do not read the two engines of that cell as one
  ZeRO level.

Read a registered schedule as a declaration, never as a measurement. Only
`1F1B` is targeted. Real two-way pipeline, two-way data-parallel and
combined runs have passed on both engines, and one eight-rank cell has
completed on the stock arm. No run has used an eight-stage pipeline, and no
run has used `--zero 1`.

## 9. Kernel-isolation benchmarks

```bash
./run_bench.sh kernel-bench <gpu> [OPTIONS]
```

The registry declares 17 scenarios, 71 arms and 5 spans. 16 of the scenarios
are cross-engine: each cuts the model at one component and puts
megatron-core beside TorchTitan there. `./run_bench.sh scenarios` prints
every scenario, arm and span with its description, and
`benchmarks/kernel/registry.py` is the authority.

| flag | default | meaning |
|---|---|---|
| `--scenario` | all 17 | Scenario subset; repeat per scenario. |
| `--arm` | every arm | Arm subset within one scenario; repeat per arm. |
| `--span` | none | Span to measure; repeat per span. |
| `--replicates` | `5` | Passes over every arm; the unit the interval is taken over. |
| `--replicates-per-process` | `1` | Consecutive replicates of one arm per worker process. |
| `--samples-per-replicate` | `40` | Timed bursts per arm per mode. |
| `--burst-k` | `16` | Calls per timed burst. |
| `--warmup-calls` | `30` | Untimed calls per arm per mode. |
| `--burst` | off | Adds the 1/4/16/64 dispatch-cost diagnostic. |
| `--model-size` | `30b-a3b` | Model shape; sizes the geometry alone. |
| `--batch` | none | Batch size run through the model. |
| `--seq-len` | none | Sequence length run through the model. |
| `--max-seq-len` | none | Raises the shape's sequence ceiling. |
| `--seed` | `0` | Input seed. |
| `--hardware` | `auto` | Provenance label. |
| `--out` | none | Output directory; valid with a single scenario alone. |
| `--cache-root` | none | Root of the build caches. |
| `--compiler-env` | none | Shell script that enables the host compiler. |

`kernel-bench` takes flags only. It reads no `OUT`, `SEQ` or `BATCH`, so an
end-to-end shell cannot leak a setting into a kernel measurement. A failing
scenario does not abort the rest; the command exits nonzero if any failed.

Method, and the rules a reader needs:

- Each arm is **timed in its own process**, so one arm's dependencies never
  reach another's. The parent merges the fragments into the results file.
- The correctness pass is the exception. It gates the whole roster in one
  process, because many arms name another arm as their reference. It fails
  the run loudly, and no arm is timed after a failed gate.
- Every declared arm reaches the results file with a status of `ok`,
  `skipped` or `failed`, and a skip names the flag that caused it.
- `--arm` must name the scenario's anchor arm and every correctness
  reference the selection uses. A selection that omits one is refused, not
  repaired.
- Each number is the burst-amortized per-call cost under back-to-back
  dispatch. It is **not** device time: where the host cannot keep the stream
  fed, the interval holds host stalls too. Run `--burst` to see which arms
  those are.
- A **span** is an implementation that fuses across a scenario cut. Its
  claim is the span against the **sum** of the scenarios it replaces, and
  asking for one span adds every enclosed scenario to the run. Two cautions
  belong beside every span number. The parts side pays one host dispatch
  chain per enclosed scenario against the span's one, so the ratio is biased
  in the span's favour, by more as the range grows. And the interval is
  **unpaired**, published as `unpaired_ratio_ci_*`, so it must not be read
  as a scenario interval.
- **No span can be measured yet.** Every span arm names a builder module
  that nobody has written. The declarations are the specification those
  builders must meet.

Output goes to `out/<timestamp>/kernels/<scenario>/<hardware>/`, holding a
manifest, a results file and `kernel_bench.log`. A span writes one directory
deeper, under `out/<timestamp>/kernels/spans/<span>/<hardware>/`, so a
shallow glob of the scenarios cannot reach one. The raw per-replicate
samples stay in the results file, so a run can be re-analyzed without
re-measuring.

**One cross-engine number exists so far.** It covers two arms of
`attention_core` at sequence length 16384, where titan is about 1.12x slower
forward and 1.14x slower forward plus backward. Fifteen of the sixteen
cross-engine scenarios have never had an arm built. Read those as
declarations until a run says otherwise.

## 10. Tests and CI

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests
```

The suite is CPU-only in about a minute. The GPU tests skip themselves when
CUDA is unavailable. Run it after any change under `benchmarks/`.

`tools/pre-push.sh` runs that same command before a push leaves the machine.
`sync.sh` links it into the common git hook directory, so every worktree
shares it. Bypass it with `git push --no-verify` or `PRE_PUSH_SKIP=1`.

`.github/workflows/tests.yml` runs a named module list rather than
`discover`. Three properties of this repository make discovery unrunnable on
a hosted runner: torch is pinned to an exact nightly, the engines are
submodules the job does not fetch, and the kernel tests need a GPU. The
workflow runs `tests/test_axes.py`, `tests/test_cli.py`,
`tests/test_docstrings.py`, `tests/test_engines.py`,
`tests/test_import_boundaries.py`, `tests/test_parallelism_plumbing.py`,
`tests/test_schema.py` and `tests/test_warmup_steps.py`. Each one imports
the standard library, `click`, `numpy` and `scipy` alone, and
`tests/test_tools.py` checks the list against the sources.

Four structural tests deserve naming:

- `tests/test_docstrings.py` bans an out-of-tree reference in a docstring or
  a comment under the e2e, artifacts, traces, cli and execution packages. An
  upstream file moves between revisions, so a path to one is wrong as soon
  as a submodule is bumped. It also bans a `#` comment above a dataclass
  field and above a module-level constant; both belong in a docstring.
- `tests/test_schema.py` pins the e2e layering. `benchmarks/e2e/schema.py`
  holds `Workload`, `Arm` and `Scenario` alone and imports the standard
  library alone. Every other record belongs to the module that builds it,
  and every module may import only modules earlier in the declared order.
- `tests/test_import_boundaries.py` enumerates every module under
  `benchmarks/`, as parent-side or worker-side. **Every module you add or
  delete edits that list, in the same commit.** It also bans a relative
  import, which the two layering checks above cannot see.
- `tests/test_docs.py` checks this file and `README.md`: every path exists,
  every documented flag is a real parameter, every documented identifier
  imports, and every default in the table above matches the code.

## 11. Operating rules

- Check `nvidia-smi` for a free GPU first. A shared GPU invalidates the
  timings.
- Drive a multi-cell matrix with `tools/run_matrix.sh`, never a loop of
  `run` calls. It refuses a dirty tree, holds a lock, waits for an idle GPU
  before each cell, and runs a watchdog during the cell. A cell is a
  **quoted string of `run` flags**, one per array entry, and the script
  word-splits it. A cell string may hold no quote and no space inside a
  value. `MATRIX_CELLS` replaces the built-in list with newline-separated
  cells. `DRY_RUN=1` prints each command and runs nothing. `GPU` is
  required.
- The watchdog flags foreign compute processes, unaccounted GPU memory and
  host-load spikes. It moves a flagged cell aside, so the next pass redoes
  it. **Never report a cell it marked `CONTAMINATED`.**
- `tools/collect_matrix.py <root>` merges a matrix tree into one table, one
  row per arm. It reads the manifest and the results file alone, and it
  refuses a results file another schema wrote.
- Put investigation notes and hardware-specific results in `reports/`, which
  is gitignored. Keep them out of `README.md` and out of this file.
- Keep commits small. Write each message as one sentence in Simplified
  Technical English.

## 12. Bumping the TorchTitan submodule

The submodule is our fork, pinned on the `bench/torchtitan-benchmarks`
branch. Pin that branch, not the fork's `main`. The fork carries commits
upstream does not have, and a rebase must preserve the two this code still
needs:

- **`--config-arg KEY=VALUE`**, which the fork's config manager forwards as
  a keyword to the config function. `benchmarks/e2e/launch.py` sends
  `--config-arg size=<name>`, so `--model-size` rides on it.
- **Static varlen metadata across steps**, which pads the packed sequence
  offsets to a fixed multiple and pins the maximum sequence length.
  `benchmarks/kernel/operations/attention_core.py` depends on that padding,
  and it also removes a per-forward device-to-host sync.

`benchmarks/models/piper_qwen3/parallelize.py` additionally needs
`parallelize_qwen3`'s `skip_dp` keyword and its ordering guarantee:
activation checkpointing, then the per-block compile, then the early return
before mesh resolution.

`benchmarks/models/piper_qwen3/config_registry.py` imports private Qwen3
helpers. After a bump, verify that each of these still exists with unchanged
behaviour:

- `_build_qwen3_moe_layers`, `_EMBEDDING_INIT`, `_output_linear_init`,
  `_qwen3_norm` and `Qwen3Model` from `torchtitan.models.qwen3`
- `CosSinRoPE`, `Embedding` and `Linear` from `torchtitan.models.common`,
  and `decoder_vocab_size` from its config helpers
- `Qwen3StateDictAdapter`, `ModelSpec`, `Trainer`, `CheckpointManager`,
  `CrossEntropyLoss`, `LRSchedulersContainer`, `MetricsProcessor`,
  `default_adamw`, `TrainingConfig`, `SelectiveAC`, `pipeline_llm` and
  `HuggingFaceTextDataLoader`
- the trainer's `size: <N> total parameters` log line, which rule 11 matches
- the per-block compile log line, which rule 8 matches through the substring
  `with torch.compile`
- `override` and `derive` from the config override module, and the
  `[Override]` log-line format rules 2 and 3 read

The kernel scenarios additionally depend on `HelionCosSinRoPE`,
`FusedGroupedExperts`, `GroupedExperts`, `QKVLinear`, `FusedQKVLinear`,
`FlexAttention` and `create_varlen_metadata_for_document`.

Three fork features are no longer load-bearing, and the list above drops
them. The compile-mode field served the deleted compile-mode axis; the
launcher now sends `--compile.enable` alone. The loss-owned LM head protocol
served the deleted fused linear cross-entropy arm; no module imports it
today. The expert-usage accumulation gate served graph capture, which is
also deleted. Keep the commits on the branch, and do not treat them as
requirements.

Also recheck the documented deltas against Piper: the builder hardcodes
`route_norm=True` where Piper wants `False`, the experts are
`GroupedExperts` rather than Piper's `BmmExperts`, and the
`load_balance_coeff = None` fixup is applied after the build. That fixup
silently stops mattering if the builder default changes.
