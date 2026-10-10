# engine-bench: Agent Guide

## 1. What this repository measures

Two measurement systems share one CLI. Never mix their numbers.

**End-to-end throughput.** Four scenarios train the same Qwen3 MoE model
on one pre-tokenized c4_test stream. Each publishes tokens/s, step time and
peak memory. `--model-size` selects the shape. The `engines` scenario has
three arms:

| arm | engine | treatment |
|---|---|---|
| `titan_compiled` | `torchtitan` | whole-block `torch.compile` |
| `titan_eager` | `torchtitan` | the same model, blocks eager |
| `megatron_stock` | `megatron_stock` | stock `megatron.training.pretrain` |

The `attention` scenario has three arms. It reuses `titan_compiled` and
`megatron_stock` unchanged, and it adds one arm that replaces the attention
kernel of `titan_compiled` through an override:

| arm | engine | attention kernel |
|---|---|---|
| `titan_compiled` | `torchtitan` | compiled FlexAttention with a BlockMask |
| `titan_compiled_fa3` | `torchtitan` | FA3 varlen |
| `megatron_stock` | `megatron_stock` | TransformerEngine's cuDNN attention |

The FA3 arm sets `packed_offsets`, so the replay loader computes the exact
document offsets of each pipeline microbatch on the CPU. The offsets have no
cap and no device assert. Their width changes per batch, so the override
marks that length dynamic. Each block compiles once, with a dynamic offsets
length.

The `experts` scenario has three arms. It reuses `titan_compiled` and
`megatron_stock` unchanged, and it adds one arm that replaces the expert
GEMMs of `titan_compiled` through overrides:

| arm | engine | expert GEMM |
|---|---|---|
| `titan_compiled` | `torchtitan` | `torch._grouped_mm`, one CUTLASS grouped kernel |
| `titan_compiled_te_per_expert` | `torchtitan` | one TransformerEngine cuBLAS GEMM per expert |
| `megatron_stock` | `megatron_stock` | one TransformerEngine cuBLAS GEMM per expert |

The per-expert arm moves the expert GEMM path alone against
`titan_compiled`. The two TorchTitan arms share their init, seed and data,
so their routing matches. Megatron routes from another init, so its row is
a reference.

- `titan_compiled_te_per_expert` imports two overrides.
  `benchmarks/models/piper_qwen3/components/moe/host_count_dispatcher.py`
  copies the whole count matrix in the dispatcher's one blocking copy, and
  it returns the rows of each local expert on the host.
  `benchmarks/models/piper_qwen3/components/moe/te_per_expert_experts.py`
  runs TE's legacy grouped GEMM, the path of Megatron's `GroupedLinear`.
  TorchTitan has no fp32 `main_grad`, so the arm writes a fresh bf16
  weight gradient and does not fuse the accumulation as Megatron does.
  cuBLAS picks the kernel from the rows of each expert, so the arm's trace
  marker is not a kernel name. It is the profiler range that each op opens,
  `engine_bench::te_per_expert_mm`.
- The engine's check refuses either per-expert override without the other,
  and either one at ep 1. It also refuses an arm with either override that
  selects the `spmd_types` backend, because their custom ops have no SPMD
  type rule.

The `stacked` scenario has three arms. It reuses `titan_compiled` and
`megatron_stock` unchanged, and it adds one arm that moves two axes
together:

| arm | engine | attention kernel | expert GEMM |
|---|---|---|---|
| `titan_compiled` | `torchtitan` | compiled FlexAttention with a BlockMask | `torch._grouped_mm` |
| `titan_compiled_fa3_te_per_expert` | `torchtitan` | FA3 varlen | one TransformerEngine cuBLAS GEMM per expert |
| `megatron_stock` | `megatron_stock` | TransformerEngine's cuDNN attention | one TransformerEngine cuBLAS GEMM per expert |

The stacked arm moves the attention kernel and the expert GEMM path
against `titan_compiled`. It answers whether the two gains add. FA3 alone
is the `attention` arm `titan_compiled_fa3`. The per-expert GEMM alone is
the `experts` arm `titan_compiled_te_per_expert`. The stacked arm imports
the three overrides of those two arms, sets `packed_offsets`, and carries
the trace markers of both. It imports the per-expert overrides, so it needs
`--ep 2` or more.

Parallelism rules 9 and 14 also apply to both per-expert arms: the expert
degree must divide `--dp`, and it needs `--zero 1`. So two GPUs take
`--dp 2 --ep 2 --zero 1`, and `--ep 2` alone is refused. `--model-size 1b`
fits that mesh. The stacked comparison ran at `--model-size 30b-a3b-20l
--dp 4 --ep 4 --zero 1 --batch 4 --profile --steps 80`.

`--arm` applies to every selected scenario, so an arm name that one of them
lacks needs `--scenario`.

**Kernel isolation.** `kernel-bench` times competing implementations of one
model component head-to-head on synthetic tensors. A kernel that wins in
isolation can be irrelevant once the compiler fuses the graph around it.

A kernel number is never an end-to-end number, and an end-to-end number is
never a kernel number. State which system produced a figure.

The `engines`, `attention`, `experts` and `stacked` scenarios carry four
deliberate differences by default, and each one moves the number. The
Megatron arm keeps fp32 master weights and reduces gradients in fp32. It runs Megatron's unfused native cross entropy.
It keeps `--init-method-std 0.01` with no weight transfer. It applies no
permutation fusion. A `megatron_stock.extra_flags` value can remove the two
fusion differences, and the manifest records it. State the four differences
and each arm's config beside every cross-engine number.

## 2. Environment

One environment owns this repository. Every command runs under
`.venv/bin/python`, and `run_bench.sh` execs that interpreter directly.

```bash
git clone --recurse-submodules <repo> && cd engine-bench
./sync.sh
```

`sync.sh` wraps `uv sync` in two passes. The first pass installs torch and
the NVIDIA header wheels. The second pass builds the groups that compile
without build isolation against the pinned torch. It also links
`tools/pre-push.sh` into the common git hook directory, so every worktree
shares one pre-push hook.

- Three default dependency groups: `megatron` (TransformerEngine), `flash3`
  (a CUTLASS sm90a source build) and `fa4`.
- A build of FA3 from source takes 15 to 40 minutes. Never benchmark
  during the build, because it saturates the host.
- **uv caches a wheel that builds without isolation by its source alone**,
  and ignores the torch version. So `sync.sh` appends the torch pin to
  `UV_CACHE_DIR`, and a pin bump rebuilds TE's torch binding and FA3. A
  plain `uv sync` uses the shared cache and can install the old torch's
  builds. Each pin's cache holds several GiB. After a bump, delete the
  old pin's cache when no venv still syncs from it.
- Skip the long build with `./sync.sh --no-group flash3`.
- torch is pinned to the stable `2.14.1` build from PyPI, which uses CUDA
  13.0. A stable release stays on PyPI, so the pin has no index expiry.
  `torch.__version__`, and so the manifest's `torch_version`, reads
  `2.14.1+cu130`.
- The `flash3` group pins nvcc, crt and nvvm at 13.0.88 and cccl at
  13.0.85. All four must come from CUDA 13.0, the CUDA version of the torch
  runtime.
- No number compares across a torch change. After a change, rerun the
  baselines.
- A torch bump that edits torch's `tail_logfile` stops every per-rank
  launch. Then port `benchmarks.execution.torchrun:tail_whole_lines` and its
  source hash.
- `run_bench.sh` sources `cuda_compat.sh`. On a kernel driver below r580,
  that script stages NVIDIA's CUDA 13.0 forward-compat userspace driver
  under `.cuda-compat/<rpm>/` and prepends it to `LD_LIBRARY_PATH`.
- `run_bench.sh` then sources `cudnn_env.sh`. That script sets
  `CUDNN_HOME` to torch's wheel cuDNN and puts its `lib` directory first on
  `LD_LIBRARY_PATH`. Without it, TransformerEngine maps the system cuDNN
  beside torch's, and `torch.backends.cudnn.version()` raises.
- TorchTitan is a submodule at `third_party/torchtitan`, installed editable.
  It is our fork, pinned on the `bench/engine-bench` branch.
- Megatron-LM is a submodule at `third_party/Megatron-LM`. It is **not**
  pip-installed. `benchmarks/models/piper_qwen3/megatron_bootstrap.py` puts
  it on `sys.path`.
- `run_bench.sh` exports `HF_DATASETS_CACHE`, because a shared `HF_HOME` may
  belong to another user.
- Stock Megatron's gradient accumulation fusion needs apex's
  `fused_weight_gradient_mlp_cuda`. `sync.sh` runs `tools/build_wgrad_ext.py`,
  which builds that one extension from `third_party/apex-wgrad/` into
  `.apex-wgrad/`. The stock driver refuses a missing build, or a build for
  another torch version. Rebuild it after a torch pin bump, and pass
  `--check` to test it on a GPU.

## 3. Repository map

| path | contents |
|---|---|
| `benchmarks/cli/` | The Click CLI: `benchmarks/cli/e2e.py`, `benchmarks/cli/kernel.py`, and the group in `benchmarks/cli/main.py`. The group attaches each command, so no command module imports the group. |
| `benchmarks/e2e/` | The shared end-to-end code: the scenario table in `benchmarks/e2e/registry.py`, the run request, the `--set` parser, the flag-pattern matcher, the run checks, the parallelism rules, the harness facts, the validation helpers, the runner and the evaluation. |
| `benchmarks/e2e/engines/` | The engine interface in `benchmarks/e2e/engines/api.py`, the engine registry in `benchmarks/e2e/engines/registry.py`, and one package per engine. `benchmarks/e2e/engines/torchtitan/plugins/` holds the modules that the TorchTitan trainer imports through `--module`. |
| `benchmarks/e2e/data/c4_replay.py` | The pre-tokenized c4_test stream that both engines read. |
| `benchmarks/artifacts/` | `manifest.json` and its schema 18 and 19 readers, `run_state.json`, the output layout and the atomic JSON writer. |
| `benchmarks/traces/extraction.py` | Chrome-trace parsing, used under `--profile` alone. |
| `benchmarks/execution/` | The launcher in `benchmarks/execution/launcher.py`, the torchrun module in `benchmarks/execution/torchrun.py`, the subprocess environment, device parsing, CPU pinning, provenance and the progress events. A runner prints nothing: it emits events, and `benchmarks/cli/rendering.py` prints them. |
| `benchmarks/kernel/` | The kernel-isolation system: registry, spans, runner, worker, timing engine, results. |
| `benchmarks/models/piper_qwen3/` | The model port: `benchmarks/models/piper_qwen3/shape.py`, the TorchTitan model config in `benchmarks/models/piper_qwen3/titan_model.py`, the megatron-core model builder and the kernel components. |
| `tools/` | The matrix job template `tools/matrix_job.sbatch` and its cell runner `tools/run_matrix_cell.sh`, `tools/collect_matrix.py`, `tools/pre-push.sh`, and the knowledge-base scripts. |
| `tests/` | The CPU and GPU test suite. |
| `third_party/torchtitan/` | Our TorchTitan fork, pinned. |
| `third_party/Megatron-LM/` | Upstream Megatron-LM, pinned, on `sys.path` only. |
| `out/` | Run outputs. Gitignored. |
| `reports/` | Investigation notes. Gitignored. Put conclusions here. |

## 4. Engines

An engine is one package under `benchmarks/e2e/engines/`, with two
halves. The **parent side** is a subclass of
`benchmarks.e2e.engines.api:Engine`, and it runs in the harness process.
The **worker side** runs in the training processes: the fork's
`torchtitan.train` with our plug-ins, or our driver around Megatron's
`pretrain`.

The type of an arm's config selects its engine.
`benchmarks.e2e.engines.registry:ENGINES` maps each subclass of
`benchmarks.e2e.engines.api:EngineConfig` to one engine. So an arm cannot
take one engine's argv and another engine's rules.

The `Engine` interface:

| member | what it gives |
|---|---|
| `name` | The engine name that the manifest records. |
| `config_type` | The config class that selects the engine. |
| `can_profile` | Whether the engine writes profiler traces. |
| `check` | Every reason that one arm cannot run; an empty list means that it can run. |
| `launch` | A `Launch`: the training processes of one arm. It does no I/O. |
| `execution_model` | One manifest string that says how the arm holds the model state. |
| `warnings` | What a reader must not conclude from the arm's numbers at this mesh. |
| `read_evidence` | A `RankEvidence` from the log of one rank: completion, parameter count and mesh. |
| `read_steps` | A `StepRead` from the log of one rank: the step samples and the dropped step lines. |
| `validate` | Every engine rule that the arm's logs or traces break. |

A `Launch` holds these fields:

| field | meaning |
|---|---|
| `target` | The arguments after the interpreter. The first one is `-m`. |
| `processes` | `per_rank` starts one process per rank under torchrun. `single` starts one process. |
| `pin` | Whether the engine accepts the CPU pinning prefix. |
| `env` | The engine's own environment keys. |
| `host_compiler` | Whether the processes need the `--compiler-env` script. |
| `cwd` | The working directory of the processes. |

### What crosses to a training process

Only argv strings and environment strings cross from the harness to a
training process. The harness writes no file for an engine to read. An
engine that needs a generated file builds it in its own process from its
arguments. The shape crosses as its name, and each rank calls
`benchmarks.models.piper_qwen3.shape:shape_by_name`. No test enforces this
convention, so a reviewer checks it.

### The shared launcher

`benchmarks.execution.launcher:build_command` turns a `Launch` into the
command line and the child environment. The command line is the pinning
prefix, then the interpreter, then the torchrun flags, then the target. A
`per_rank` launch uses torchrun at every world size, also at one rank.

The torchrun flags start with `-u`, and they run torchrun through
`benchmarks.execution.torchrun`. Torchrun tees each rank into the arm log
through a thread of its own, and all the threads write to one stream. With
a buffered stream, a thread race in CPython 3.10 loses lines and writes NUL
bytes in their place. With `-u`, each write of a thread is one call.
Torchrun already starts its workers with `-u`, so the flag now also applies
to the torchrun agent.

A worker can write one line in two calls: `print` writes the text, then the
newline. A tee thread can read the file between the two calls. Torch's tee
then writes the text alone, and the line of another rank joins it.
`benchmarks.execution.torchrun:tail_whole_lines` holds a partial line until
its newline arrives. It also reads a line that a worker wrote just before
it exited, which torch's tee can lose. The module pins the SHA-256 of the
source of torch's tee function, and it refuses any other source.
Keep `-u` and the module, and `tests/test_rank_log.py` checks both.

The launcher owns six environment keys: `CUDA_DEVICE_ORDER`,
`CUDA_VISIBLE_DEVICES`, `NGPU`, `LOG_RANK`,
`TORCHELASTIC_LOG_LINE_PREFIX_TEMPLATE` and `PYTORCH_ALLOC_CONF`. The
runner refuses a launch that sets one of them, before any arm starts.

Each engine decides its own pinning through `pin`. A launch that declines
the prefix records `declined by engine` as its CPU pinning.

### How to add an engine

A Piper package follows these steps, in this order.

1. Make the package `benchmarks/e2e/engines/<name>/`, with a
   docstring-only `__init__` module.
2. Write the config: a frozen, keyword-only subclass of `EngineConfig`.
   Give each field a `Literal`, an `Enum`, a `bool`, an `int`, a `str` or a
   tuple of `str` type, because `--set` converts a value from the
   annotation.
3. Write the worker side. It reads argv and environment strings alone.
   Each rank prints a completion line, its parameter count, its mesh, and
   one step line or step record per step.
4. Write the engine: a subclass of `Engine` with each member of the table
   above. Its `check` refuses each `extra_flags` token that it does not
   classify, and its `launch` sends the other tokens last. Keep
   `can_profile` false until `validate` checks the traces.
5. Add one instance to the engine tuple in
   `benchmarks/e2e/engines/registry.py`, and declare the arms in
   `benchmarks/e2e/registry.py`.
6. In the commit that adds a module, add it to `LAYER_ORDER` in
   `tests/test_schema.py` and to the lists in
   `tests/test_import_boundaries.py`.

## 5. The `run` command

```bash
./run_bench.sh scenarios                    # list scenarios and arms (or scenarios e2e, kernel, detail <name>)
./run_bench.sh run <gpu> [OPTIONS]
./run_bench.sh evaluate <out_dir> [--arm NAME]... [--results PATH]
```

`<gpu>` is a PCI index, or a comma list of them. It takes decimal indices
and single commas alone, and the manifest records it as typed. The runner
sets `CUDA_DEVICE_ORDER=PCI_BUS_ID`, `CUDA_VISIBLE_DEVICES` and `NGPU`, so
the index is stable and the world size follows the request.

Inside a Slurm job, the job sees only its own cards, and their indices
start at 0. So give `0`, or `0,1` and so on, and never a physical index.

`run` executes, validates and evaluates. It runs every scenario unless
`--scenario` narrows the set. It checks every selected scenario before the
first arm starts, so one refused scenario stops the whole command. It stops
at the first arm that fails.

At ep 1 the `experts` scenario refuses `titan_compiled_te_per_expert`, and
the `stacked` scenario refuses `titan_compiled_fa3_te_per_expert`. So a
one-GPU run names its scenarios with `--scenario`. A one-GPU run of
`experts` or `stacked` also names its arms with `--arm`.

Named scenarios run one at a time, in the order given. A name may repeat,
and each repeat is another run with a `-run<n>` suffix on its scenario
directory. `manifest.json` records the plain name.

**The default shape does not fit one GPU.** `30b-a3b` holds about
227.5 GiB of TorchTitan state against an H200's 139.81 GiB. The harness
reads layer counts and not memory, so a one-GPU run of it runs out of
memory. Pass `--model-size 1b` for a one-GPU run. `--pp 8` also needs
`--batch 16`. Parallelism rules 11 and 12 ask for a microbatch count that
divides the pipeline degree and is at least twice the stage count.

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
| `--set` | -- | `--` | One field of one arm's engine config, as `<arm>.<field>=<value>` or `<arm>.<field>+=<value>`; repeat per field. |
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
| `--profile` | -- | `off` | Whether the run collects profiler traces. |
| `--warmup-steps` | `WARMUP_STEPS` | `10` | Steps an unprofiled run discards before it measures. |

The six parallelism options and `--set` take no environment variable,
because each one must agree with the `<gpu>` positional or with `--arm`.

### Arm settings: `--set`

`--set <arm>.<field>=<value>` sets one field of one arm's engine config.
`--set <arm>.<field>+=<value>` appends to a list field. The fields differ
per engine: `benchmarks/e2e/engines/torchtitan/config.py` and
`benchmarks/e2e/engines/megatron_stock/config.py` declare them.

- The arm must be a selected arm, and the field must exist.
- A choice field takes one of its values. A switch takes `on`, `off`,
  `true` or `false`. A list field, such as `extra_flags`, takes `+=` alone.

Five `run` options became `--set` values. The run refuses each old option,
and the message gives the `--set` spelling of its value.

| old option | `--set` value |
|---|---|
| `--megatron-p2p-sync` | `megatron_stock.p2p_sync=<value>` |
| `--megatron-nan-guard` | `megatron_stock.nan_guard=<value>` |
| `--megatron-precision` | `megatron_stock.precision=<value>` |
| `--megatron-arg` | `megatron_stock.extra_flags+=<flags>` |
| `--torchtitan-arg` | `titan_compiled.extra_flags+=<flags>` or `titan_eager.extra_flags+=<flags>`, once for each selected arm |

### Engine passthrough

A perf flag passes. The engine's check refuses a flag that changes a
recorded fact.

An arm's `extra_flags` reach that arm alone. Shell rules split one value, so
`--set "megatron_stock.extra_flags+=--moe-token-dispatcher-type flex"` adds
two tokens. The tokens go after the harness flags, and both parsers keep
the last value of a repeated flag.

Each engine holds three flag tables, in
`benchmarks/e2e/engines/torchtitan/flags.py` and
`benchmarks/e2e/engines/megatron_stock/flags.py`. The owned table maps each
harness option or `--set` field to the flags it sets. The pinned table holds
the flags that keep the two engines on the same work: the optimizer, the
routing, the data, the step lines and the timed steps. The perf table holds
the flags that a passthrough may set. The engine's check refuses a
passthrough flag in the owned or the pinned table, and the message names
the owner.

- An unlisted Megatron flag passes, so every fusion flag that Megatron
  offers passes. Section 7 names the stock omissions that a passthrough can
  add.
- The TorchTitan check refuses an unlisted flag. So a field that a fork
  bump adds fails until someone classifies it.
- `tests/test_passthrough.py` checks that the tables do not overlap and
  that each emitted flag has one class.

A passthrough cannot turn off a store-true flag that a builder sends,
unless the engine has a negative form of it.

### The comparability rule

Numbers are comparable only within one value of each of these: each key of
the manifest's `run` block, each arm's `config`, the CPU pinning,
`torch_version`, `cublaslt_version`, `cudnn_loader_resolves`,
`torchtitan_git_rev`, `benchmarks_git_rev` and `megatron_git_rev`. Check
each one before you compare against an older run.

Before `cudnn_env.sh`, a TransformerEngine process ran its cuDNN attention
on the system cuDNN 9.23.2, unless its job put the cuDNN of torch first. Job
9393 ran on the system cuDNN. Job 9406 put the cuDNN of torch first. So a
Megatron attention number from before `cudnn_env.sh` is not comparable until
its job script or its manifest's `cudnn_loader_resolves` path shows which
cuDNN it ran. A manifest from before
`cudnn_env.sh` records `cudnn_loader_resolves` as one library
path, with no leading version number.

The Megatron `p2p_sync` and `nan_guard` fields both default to `off`. Every
Megatron number published before that flip had both at `on`. State the
value beside any Megatron number.

The `lean` value of the Megatron `precision` field changes the numerics. It
holds bf16 Adam moments and accumulates gradients in bf16. Read the loss
trajectories beside any lean number. At one data-parallel rank the
distributed optimizer also does bucket bookkeeping for no saving. Both
effects are unmeasured.

### The profile axis and the step floor

`--profile` is off by default, and off is the treatment that a published
throughput number wants. The profiler costs GPU and host time on every
window.

- Off: the run writes no trace. The evaluation samples every step after the
  warmup, so `--steps` must be **more than** `--warmup-steps`.
- On: each engine writes `<arm>/profiling/traces/iteration_*/`, every trace
  rule applies, and `--steps` must be at least 40, for two profiler
  windows. The run refuses `--warmup-steps`. It also refuses each arm whose
  engine sets `can_profile` false, and the message names the arm and the
  engine.

An exported `WARMUP_STEPS` refuses every profiled run. Unset it first.

### Resume

`run --resume <out_dir>` re-validates each arm against the files on disk,
skips the arms that pass, archives the partial artifacts under
`attempts/<timestamp>/<arm>/`, and runs the rest again.

- `--resume` cannot be combined with `--out`, and it needs exactly one
  selected scenario.
- An omitted run option and each arm config inherit the recorded value.
- The resume refuses a change of scenario, selected arms, hardware label,
  any key of the `run` block, any field of an arm config, or any key that
  `benchmarks.artifacts.manifests:HOST_KEYS` names. A refused `--set` names
  `<arm>.config.<field>`.
- A resume does **not** inherit the parallelism spec. Omitted parallelism
  flags ask for the single-GPU spec.
- A resume reads manifest schema 20 alone, and it refuses schemas 18 and 19
  by name.

### Output layout

```
out/<timestamp>/<scenario>/<hardware>/
  manifest.json     # schema 20
  run_state.json    # per-arm status, attempts, evaluation status
  results.json      # schema 7
  <arm>.log         # training stdout and stderr, every rank
  <arm>/profiling/traces/iteration_*/rank<n>_trace.json.gz   # --profile only
  attempts/<ts>/<arm>/   # archived artifacts of a failed attempt
```

`manifest.json` is schema 20. It records the scenario, the `run` block, one
record per selected arm, the throughput definition and
`hardware_metadata`. An arm record holds the engine, the config type name,
the config, the execution model, the command line, the environment keys
that the launcher and the engine set, and the CPU pinning.

`evaluate` and `tools/collect_matrix.py` read schemas 18, 19 and 20.
`benchmarks.artifacts.manifest_v18:upgrade_v18` converts a schema 18
manifest to schema 19. `benchmarks.artifacts.manifest_v19:upgrade_v19`
converts a schema 19 manifest to schema 20. It gives each TorchTitan arm
`packed_offsets` false, because no older arm sent loader offsets. The reader
refuses any other version by name.

A provenance probe that fails records `unavailable: <error>` and does not
stop the run. A device list of two GPU models stops it. A TransformerEngine
process that maps a second cuDNN beside torch's, or a cuDNN version other
than torch's build, also stops it. The probe runs in the environment of a
training process, and again under the `--compiler-env` script when the run
has one. This check runs before `kernel-bench` too. `cudnn_loader_resolves`
records the cuDNN version and directory.

The tokens/s of a step sample, and each tokens/s figure of `results.json`,
are **per device**. Both engines divide one rank's token count by `cp * tp * pp`. The
data-parallel degree is absent from that divisor, because each
data-parallel rank reads a batch of its own. The manifest records the
definition as `throughput_definition`.

### CPU pinning

The training step is host-bound at these sizes, so an unpinned run measures
scheduler placement. For each engine whose launch accepts it, the runner
binds each training process to the NUMA node of its GPU with
`numactl --cpunodebind --membind`. The runner finds the node from the PCI
bus id through sysfs. When that fails, or when the devices sit on two
nodes, the run proceeds unpinned and `cpu_pinning` records why. Pinned and
unpinned runs are not comparable.

A Slurm job can hold part of the node's CPUs, or none of them. When the job
holds part of them, the runner binds to that part with
`numactl --physcpubind --membind`. When the job holds none of them, the run
proceeds unpinned. Both cases record a different `cpu_pinning`, so they are
not comparable with a run on the whole node. `--resume` refuses to mix them, and
`results.json` warns when the arms of one run mix them.

## 6. Validation and evaluation

### Validation

`benchmarks.e2e.validation:validate_arm` gates every arm before the harness
publishes its numbers. It splits `<arm>.log` by rank, because one log holds
every rank's output. The split drops NUL bytes first, because a log that
torchrun wrote without `-u` can put NUL bytes in front of a rank prefix.
The arm's engine reads one `RankEvidence` from each rank's log. Then
`validate_arm` checks the harness facts and runs the engine's `validate`.
It raises one error that lists every failure. A step line that does not
parse, and a step that does not follow the previous step of its rank, also
fail the arm.

The four harness facts live in `benchmarks/e2e/evidence.py`. They apply to
every engine:

- **Completion:** each rank below the world size writes output and
  completes. A missing rank fails, and so does a rank outside the run.
- **Model:** each stated parameter count equals the shape's count, and at
  least one rank states it.
- **Mesh:** each rank's stated `dp`, `pp` and `ep` equal the run's spec.
  Above one data-parallel rank the stated ZeRO level must also equal the
  spec. A log that states no mesh reads as one device. So a two-way
  data-parallel run cannot publish as one GPU, which would read as roughly
  twice the true rate.
- **Finite trajectories:** no rank's step samples carry a `nan` or an `inf`
  loss or gradient norm.

Each engine's `validate` holds its own rules. The code names each rule by
its message.

The TorchTitan rules, per rank:

- **Compile:** the per-block compile line is present when the arm asks for
  `torch.compile`, and absent when the arm runs eager.
- **SelectiveAC:** the SelectiveAC line agrees with `--ac`.
- **Overrides:** an arm with overrides prints `overrides_per_block *
  n_layers` `[Override]` lines, and one line for each `override_imports`
  entry.
- **Fallback:** the log holds no silent-fallback phrase.
- **Mesh lines:** above one rank, the log holds the mesh line, the
  data-parallel line above one data-parallel rank, and the schedule line
  above one pipeline rank.

The stock Megatron rules, per rank:

- **Mesh lines:** above one rank, the log holds the mesh line, the
  data-parallel line above one data-parallel rank, and the p2p line above
  one pipeline rank.
- **NaN guard:** the log holds the NaN-guard line of the `nan_guard` value.
- **Precision:** the log holds the training-loop line and the four
  precision fields of the `precision` value.

The compile rule reads both ways. Never relax the absence half, or an arm
that silently compiled publishes as eager.

Under `--profile` both engines also call
`benchmarks.e2e.validation:trace_refusals`, which checks three trace rules:

- **Windows:** each rank wrote at least `min_windows` traces. Above one
  rank, each rank wrote a trace.
- **Kernel markers:** each `trace_kernel_markers` string of the arm config
  appears in some trace.
- **All-reduce:** above one data-parallel rank, each rank's traces hold the
  kernel `ncclDevKernel_AllReduce`.

Without the profile axis the arm writes no trace, and the trace rules do not
run. The mesh fact then carries the data-parallel axis alone. Cite a
data-parallel number from an unprofiled run as resting on that log line.

A grouped NCCL launch in Megatron can show a generic kernel name. Settle
a Megatron all-reduce failure with the arm's own trace, and do not widen
the marker to a bare `nccl`. An all-reduce proves that a collective ran,
never which one, so read the mesh fact and the all-reduce rule together.

### Evaluation

`run` always evaluates, and `evaluate <out_dir>` evaluates a finished
directory again. **Evaluation reads the logs alone**, so a directory
evaluates the same way under either profile value.

The engine's `read_steps` turns the log of one rank into a `StepRead`. It
holds one `StepSample` per step: the rank, the step, the tokens/s, the peak
memory, the loss, the gradient norm and the `extras`. The loss is `None` on
a rank that holds no loss. `extras` holds the other figures of the step,
under the names that the engine gives them.

- TorchTitan prints its own step line. A pipeline rank that holds no loss
  prints the loss `-1`, and the reader reads it as no loss.
- The stock Megatron driver prints one JSON step record per rank and step,
  after the prefix `bench-step: `. The reader also reads the text step line
  that the stored run directories hold.

A log that torchrun wrote before `benchmarks.execution.torchrun` can join
the lines of two ranks. Its tee wrote a partial line when a worker wrote
one line in two calls. The readers still read such a log:

- When the prefix of another rank cuts a step line, the reader drops the
  step line. `results.json` records a warning that names the arm, the rank,
  the step and the log line. Validation does not fail on it.
- When a whole step line comes first, the reader reads it.
- **A known gap:** when a whole step line of one rank comes after the text
  of another rank on one log line, the reader loses the step with no
  warning. A log that the launcher writes now holds no such line.

A lost step that the sample rule takes refuses the arm, as the lost-step
refusal below states. A lost step outside the sample rule changes no figure.

The sample rule follows the recorded axis.
`benchmarks.e2e.results:measured_samples` takes every step after the warmup
in an unprofiled run. `benchmarks.e2e.results:stable_samples` takes the
steps of each profiler cycle that carry no profiler cost, without step 2.
Step 2 is the first step of that rule, and it runs slow in every measured
arm. So an 80-step profiled run samples 35 steps: 3 to 10, 22 to 30, 42 to
50 and 62 to 70. The two rules give different figures, not two readings of
one figure.

**The lost-step refusal.** `benchmarks.e2e.results:sampled_steps` lists the
steps that the rule takes from the run's step count.
`benchmarks.e2e.results:refuse_lost_steps` refuses the arm when a rank lacks
one of them, before the evaluation computes a figure. The error names the
arm, the rank and the step, and the evaluation writes no `results.json`.
So every rank holds the same sampled steps.

`results.json` is schema 7, the value of
`benchmarks.e2e.results:RESULTS_SCHEMA_VERSION`. Per arm it carries these
keys:

- `sample_count`: the sampled steps of each rank.
- `tokens_per_second`: the `median` and the `mean`, each with its rank as
  `median_rank` and `mean_rank`.
- `step_ms`: the same four keys, plus the `p95` and its `p95_rank`.
- `peak_memory_gib`: the same four keys, plus the `max` over every step and
  rank and its `max_rank`. The median and the mean read the per-step peaks
  of the sampled steps.
- `extras`: the engine's name, and under it the same four keys for each
  extra figure. Do not compare an extra across engines, because each engine
  computes its own.
- `rank_reduction`: `slowest_rank_per_statistic`.
- `per_rank`: each rank's own statistics, and a `steps` table with the
  step, the tokens/s, the step time, the peak memory and the extras of each
  sampled step.

The file also carries `losses`, `gradient_norms` and `warnings`.

Each figure has two statistics of equal weight. The median is
`statistics.median` of the per-step values. The mean of a rate is the total
over the total time: for tokens/s, the total tokens over the total step
time. `benchmarks.e2e.results:rate_mean` computes it as the harmonic mean of
the per-step rates, because each step of one rank holds the same token
count. TFLOPS and MFU are rates too, so they take the same mean.

The mean of a step time or of a memory figure is the arithmetic mean. A
host stall pulls the mean tokens/s below its median and the mean step time
above its median, so state which statistic a figure is.

`benchmarks.e2e.results:step_ms` derives the step cost from the throughput:

```
step_ms = 1000 * local_batch_size * seq_len / (tps * pp)
```

Each statistic is published at the **slowest** rank for that statistic,
never as a mean over the ranks. A parallel schedule locks the ranks together
at every step boundary, so the mesh runs at the pace of its slowest rank.
The slowest rank has the lowest tokens/s or extra, and the highest step time
or memory. The evaluation finds it separately for each figure and each
statistic, so the median and the mean can name two ranks. A tie goes to the
lower rank.

The 95th percentile uses the nearest-rank method, so it is always a measured
step.

`benchmarks.e2e.results:refuse_non_finite_trajectories` fails an arm whose
step samples carry a `nan` or an `inf` on any rank. A run that diverged
publishes no throughput. The stock driver records a `nan` gradient norm on
a step that Megatron skipped, so the refusal also refuses a skipped step.

The evaluation warns when the median tokens/s spreads more than 1.15x across
ranks, and when the arms of one run mix pinned and unpinned processes. It
also repeats the warnings that the runner printed.

**What is deliberately absent.** There is no baseline arm, no ratio, no GPU
kernel time, no per-region measurement, no launch latency and no
significance test. A reader compares two absolute rows.

Send a kernel-level question to `kernel-bench`. Send a trace-level question
to the external trace-anatomy tool, which reads the trace layout that a
profiled run writes.

### A known gap: V-shaped pipeline schedules

`benchmarks.e2e.results:loss_visible_rank` returns
`(world_size // pp) * (pp - 1)`. That is right for `1F1B` and
`Interleaved1F1B`. It is wrong for `ZBVZeroBubble` and `DualPipeV`, where
rank 0 holds the last stage and the loss. The Megatron engine's check
refuses a V-shaped schedule, so a TorchTitan-only run may ask for one. No
run has ever used one. Repair `loss_visible_rank` before you run one. Do
not lift `MAX_PP` instead.

## 7. The stock Megatron arm

`benchmarks/e2e/engines/megatron_stock/` holds everything about Megatron,
and the harness reaches it through the arm's config type alone. The parent
side sits at the top of the package, and the worker side sits in
`benchmarks/e2e/engines/megatron_stock/driver/`. Each module's docstring
states its job.

The driver substitutes **one** provider, the dataset provider. The model
builder, the optimizer, the schedule, the distributed setup, the forward
step and the training loop all stay Megatron's. The driver edits no file of
the Megatron-LM checkout. Four shims run in the driver process instead.

These faithfulness guarantees hold today:

- **Same shape.** Both engines build from the same shape record in
  `benchmarks/models/piper_qwen3/shape.py`.
- **Same data.** Both engines read the samples that
  `benchmarks.e2e.data.c4_replay:materialize` drains from TorchTitan's own
  dataset class with TorchTitan's own tokenizer. The test suite asserts that
  the loaders of both engines give that class's tokens bit for bit. Each
  engine reads the whole run at startup, so no measured step pays a data
  cost, and each engine raises when a step asks for a sample past the
  last one.
- **No recompute, ever.** The engine's check refuses `--ac sac`.
- **Stock precision.** The arm is **not** plain bf16. Under `--bf16` alone
  Megatron keeps fp32 master weights and fp32 Adam moments and reduces
  gradients in fp32. That is about 18 bytes of state per parameter against
  TorchTitan's 8. That is the stock treatment, and the scenario keeps it.
  The arm's `execution_model` in the manifest names the three state dtypes.

The driver prints one training-loop line that names the state dtypes, the
cross-entropy fusion and the token dispatcher. The precision rule matches
that line and its four precision fields.

The stock recipe omits `--cross-entropy-loss-fusion`, so by default it runs
Megatron's own unfused native cross entropy. That path upcasts the full
logits to fp32 and reads them several times, and it is a large part of the
engine gap. Say so beside any loss-path claim. The stock recipe also omits
`--moe-permute-fusion`, `--overlap-grad-reduce` and
`--overlap-param-gather`. A `megatron_stock.extra_flags` value can add each
of them, and `--use-flash-attn` too. `--overlap-param-gather` needs
`--zero 1`. The mesh rule takes the expected `overlap_grad_reduce` value
from the arm's own flags. A passthrough cannot turn off `--moe-grouped-gemm`,
because Megatron has no negative form of it.

Of the three Megatron treatments, the `p2p_sync` value `on` needs more
than one pipeline rank, and the `lean` precision needs `--zero 1`.

The recipe follows Piper's stock command line, with three deliberate
differences. It sends `--moe-router-dtype fp32`, because TorchTitan routes
in fp32 too. It sends no attention-backend flag, so TransformerEngine
selects the cuDNN kernel. It sends no distributed optimizer at `--zero 0`,
so both engines replicate their parameters there. It keeps the deprecated
`--use-mcore-models`, because Piper sends it. Nobody has checked whether
Megatron decays the learning rate over the same steps as TorchTitan.
Read `OptimizerParamScheduler` before you report a learning rate.

Eight facts explain the driver:

- **One sample is one packed sequence.** Megatron flattens an `(m, S)`
  microbatch to `(1, m*S)`, but it sizes the pipeline receive buffer as
  `(S, m, H)`. So a microbatch of several rows reaches the next stage
  permuted, and nothing raises. The flag list therefore sends
  `--micro-batch-size 1`, and the driver packs the rows of one microbatch
  into one sample. `cu_seqlens` marks every document. The driver pads
  `cu_seqlens` to the packed length, as Megatron's dataset does.
- **The driver does not restart a rank.** It omits the
  `inprocess_restart.maybe_wrap_for_inprocess_restart` wrap of Megatron's
  GPT entry point, because a restarted rank publishes a number that no run
  asked for.
- **The data key is the data-parallel rank.** The stages of one pipeline
  read the same tokens, and every rank builds an iterator, because the
  middle stages read `cu_seqlens` too. An exhausted stream raises.
- **The p2p value has no Megatron flag.** `--bench-batch-p2p-sync off`
  carries it, and the driver sets `args.batch_p2p_sync` before Megatron
  builds its config.
- **The data-parallel line comes from the built wrapper.** The all-reduce
  rule cannot prove data parallelism, because Megatron all-reduces the loss
  on every step. Both ZeRO levels build a `DistributedDataParallel`
  wrapper, so the line names the optimizers inside Megatron's
  `ChainedOptimizer`, and their classes separate the two levels.
- **The profiler schedule skips its first step.** Megatron steps the
  profiler at the top of its loop and TorchTitan at the bottom.
  `skip_first=1` puts both engines in the same profiler state on each
  sampled step. Megatron also steps the profiler after it stops it, so the
  engine refuses a profiled run that ends inside a profiler cycle.
- **The driver builds Megatron's dataset helper** against torch's pybind11
  headers, because Megatron's `make` runs the system `python3`, which has no
  pybind11 on this host.
- **Python 3.10 has no `typing.override`.** Megatron imports it, so the
  driver adds it from `typing_extensions`. A move to Python 3.12 would
  make every published number incomparable.

## 8. Parallelism

`benchmarks/e2e/parallelism.py` owns the degrees, the schedule names and
the shared rules. One parallelism spec describes a whole run, so every arm
shares it. `--dp`, `--pp`, `--ep`, `--zero`, `--pp-schedule` and
`--pp-microbatch-size` build it.

The world size is `dp * pp`. **Expert parallelism does not multiply it.**
Both engines take the expert ranks out of the data-parallel axis. Tensor
and context parallelism are deliberately absent.

`--zero 0` keeps a whole copy of the dense parameters on every rank.
`--zero 1` shards the optimizer states: Megatron gets
`--use-distributed-optimizer`, and TorchTitan gets the whole data-parallel
width as its shard degree plus `fsdp-reshard-after-forward never`.

TorchTitan always gets its shard degree explicitly, because it reads an
omitted shard degree as every remaining rank. Its pipeline flags set both
`less-layers` values to 0, so its stages split the layers evenly, as
Megatron's stages do.

`parallelism_refusals` lists the rules that a spec breaks.
`benchmarks.e2e.checks:check_run` collects these, the other shared
refusals and each engine's `check` into one error, before any host probe.
The numbers below are the numbers of the code. Rules 5, 6, 13, 15, 16 and
17 are deleted.

| rule | refuses |
|---|---|
| 1 | a world size other than the number of devices requested |
| 2 | a pipeline degree above 8, then a world size above 8 |
| 3 | a schedule or a microbatch size at one pipeline rank, and a missing schedule above it |
| 4 | a schedule name that `PP_SCHEDULES` does not declare |
| 7 | a layer count that does not divide the total stage count |
| 8 | an expert degree above the shape's expert count, or one that does not divide it |
| 9 | an expert degree that does not divide the data-parallel degree |
| 10 | a batch that does not divide into whole microbatches |
| 11 | a microbatch count that does not divide the pipeline degree |
| 12 | fewer microbatches than twice the total stage count, above one pipeline rank |
| 14 | an expert degree above 1 under `--zero 0` |

Each engine declares the schedules that it runs, and its `check` refuses
the others. The Megatron engine refuses a schedule that Megatron-LM or its
driver does not implement. The TorchTitan engine declares its schedules in
`benchmarks/e2e/engines/torchtitan/mesh.py`. It refuses a schedule whose
`requires_uncompiled` is true for an arm that asks for `torch.compile`.
Three of the five registered schedules raise on a compiled stage module.

Two legal meshes warn and do not refuse. The runner prints both warnings,
and `results.json` records them.
`benchmarks.e2e.parallelism:zero_warnings` states the first one. The
TorchTitan engine's `warnings` states the second one.

- A sharded level at one data-parallel rank: the shard degree is 1, so the
  run holds the dense parameters as a replicated run holds them.
- `--zero 1` at one pipeline rank: one microbatch puts the gradient
  reduce-scatter inside the only backward pass, so the **TorchTitan arms**
  hold ZeRO-2 and not the ZeRO-1 shape that the level names. Megatron holds
  ZeRO-1 at every mesh. Do not read the two engines of that cell as one
  ZeRO level.

Read a registered schedule as a declaration, never as a measurement. Only
`1F1B` is targeted.

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
`discover`. Two properties of this repository make discovery unrunnable on
a hosted runner: the engines are submodules the job does not fetch, and the
kernel tests need a GPU. The workflow runs `tests/test_axes.py`,
`tests/test_cli.py`, `tests/test_docstrings.py`, `tests/test_engines.py`,
`tests/test_evidence.py`, `tests/test_golden.py`,
`tests/test_import_boundaries.py`, `tests/test_launcher.py`,
`tests/test_overrides.py`, `tests/test_parallelism_plumbing.py`,
`tests/test_schema.py`, `tests/test_step_samples.py` and
`tests/test_warmup_steps.py`. Each one imports the standard library,
`click`, `numpy` and `scipy` alone, and `tests/test_tools.py` checks the
list against the sources.

Four structural tests deserve naming:

- `tests/test_docstrings.py` bans an out-of-tree reference in a docstring or
  a comment under the e2e, artifacts, traces, cli and execution packages. An
  upstream file moves between revisions, so a path to one is wrong as soon
  as a submodule is bumped. It also bans a `#` comment above a dataclass
  field and above a module-level constant; both belong in a docstring.
- `tests/test_schema.py` pins the e2e layering. `benchmarks/e2e/schema.py`
  holds `Scenario` alone, and it imports the standard library and
  `benchmarks/e2e/engines/api.py` alone. Every other record belongs to the
  module that builds it,
  and every module may import only modules earlier in the declared order.
- `tests/test_import_boundaries.py` enumerates every module under
  `benchmarks/`, as parent-side or worker-side. **Every module you add or
  delete edits that list, in the same commit.** It also bans a relative
  import, which the two layering checks above cannot see.
- `tests/test_docs.py` checks this file and `README.md`: every path exists,
  every documented flag is a real parameter, every documented identifier
  imports, and every default in the table above matches the code.

## 11. Operating rules

- Run GPU work as a Slurm job. Read the admin skill `run-gpu-job` before
  you submit. It sets the partitions and the time limits.
- Run a multi-cell matrix as one Slurm job, never as a loop of `run`
  calls. Make a new output root under `out/`. Copy `tools/matrix_job.sbatch`
  into it, and edit the copy. The template is a working example: the seven
  dp4 x ep4 cells of the stacked comparison, with `titan_compiled` first and
  last.
- The copy holds the `#SBATCH` lines, `ROOT` (the output root), the
  job-local device list, the shared flags and one `cell <name> <run flags>`
  line per cell. A cell name holds only letters, digits, `_` and `-`.
- Each cell needs exactly one `--scenario`. The `dp * pp` of each cell must
  equal the number of cards. The job word-splits the flags, so a value may
  hold no quote and no space.
- The `--time` of the job must cover the sum of the run times of the cells.
  It must also cover up to `IDLE_MAX_WAIT` seconds of load gate per cell.
  A copy can lower `IDLE_MAX_WAIT`, as the template does.
- Submit the copy from the repository root, on a clean tree. The output
  root must exist, because `sbatch` does not make the directory of its log.

  ```bash
  MATRIX_REV=$(git rev-parse HEAD) sbatch -o out/<root>/slurm-%j.out out/<root>/matrix.sbatch
  ```

- In the job, `tools/run_matrix_cell.sh` runs each cell into
  `out/<root>/<name>`. It adds the output option itself, so a cell must not
  send one. A failed cell does not stop the job.
- Before each cell, the runner checks the commit, the clean tree and the
  cards of the device list. It also checks that the job holds some CPU on
  the NUMA node of each card. Then it waits for an idle host, and it runs
  the watchdog during the cell.
- The runner reads these environment variables:

  | variable | default | meaning |
  |---|---|---|
  | `MATRIX_REV` | required | The commit at submission. The job refuses another `HEAD`. |
  | `IDLE_LOAD` | `80` | The 1-minute load average that the load gate waits for. |
  | `IDLE_SETTLE` | `3` | The consecutive idle samples that open the gate. |
  | `IDLE_POLL` | `20` | The seconds between two gate samples. |
  | `IDLE_MAX_WAIT` | `1200` | The seconds after which the cell runs anyway, flagged. |
  | `CONTENDED_LOAD` | `150` | The 1-minute load average that condemns the cell. |
  | `FOREIGN_MEM_MIB` | `2000` | The MiB of GPU memory that no process of ours explains. More condemns the cell. |
  | `WATCH_INTERVAL` | `15` | The seconds between two watchdog samples. |

- Each cell writes one line `STATUS <name> <state>` to
  `out/<root>/sweep.log`, and the state to `out/<root>/<name>.status`. Read
  the `.status` files, or the last `STATUS` line of each cell:

  | state | meaning |
  |---|---|
  | `OK` | The run passed, and the watchdog flagged nothing. |
  | `OK(load-flagged)` | The run passed, but the load gate timed out before it. |
  | `OK(existing:<state>)` | The cell was done, so nothing ran. This state goes to `sweep.log` alone. |
  | `FAIL(rc=N)` | The run exited with `N`. The runner renames the cell as failed. |
  | `FAIL(no-results)` | The run exited with 0 and wrote no `results.json`. The runner renames the cell as failed. |
  | `CONTAMINATED` | The watchdog flagged the cell, or it could not read the cards. The runner renames the cell as contaminated. |
  | `PLACEMENT` | The job holds no CPU on the node of some card, or that node is unknown. Nothing ran. |
  | `ERROR` | A check failed. Nothing ran. |

- A cell is done when it has a `results.json` and an OK state. The runner
  renames a failed cell to `<name>.failed-<stamp>`, and a contaminated cell
  to `<name>.contaminated-<stamp>`. The log, the watch file and the exit
  code get the same name. The stamp holds the UTC time and the job id.
- A cell that is not done, but has a directory or a log, gets the failed
  name before its run. So a cell that a time limit stopped runs again.
- To retry the cells that are not OK, submit the same copy again. The
  runner skips each done cell. Retry only a coincidental failure, and
  retry a cell at most 3 times.
- `ERROR`, `PLACEMENT` and a `FAIL` that repeats with the same message are
  deterministic. Fix the copy or the code, and do not resubmit the same
  copy.
- One `ERROR` is a coincidence: the runner refuses to rename a cell onto a
  name that exists. A resubmit clears it, because the next job id gives a
  new name.
- The watchdog flags foreign compute processes, unaccounted GPU memory,
  host-load spikes and a card query that fails in two consecutive samples.
  **Never report a cell it marked `CONTAMINATED`.**
- An `OK(load-flagged)` cell ran on a busy host. Check its step times by
  hand before you report it.
- A job has one time limit, and `main` caps it at 2 hours. All cells of
  the job share it. When the cells need more, split them into two copies.
- `tools/collect_matrix.py <root>` merges a matrix tree into one table, one
  row per arm. It reads the manifest and the results file alone, and it
  refuses a results file another schema wrote.
- Put investigation notes and hardware-specific results in `reports/`, which
  is gitignored. Keep them out of `README.md` and out of this file.
- Keep commits small. Write each message as one sentence in Simplified
  Technical English.

## 12. Bumping the TorchTitan submodule

The submodule is our fork, pinned on the `bench/engine-bench`
branch. Pin that branch, not the fork's `main`. The fork carries commits
upstream does not have, and a rebase must preserve the two this code still
needs:

- **`--config-arg KEY=VALUE`**, which the fork's config manager forwards as
  a keyword to the config function. `benchmarks/e2e/engines/torchtitan/flags.py` sends
  `--config-arg size=<name>`, so `--model-size` rides on it.
- **Static varlen metadata across steps**, which pads the packed sequence
  offsets to a fixed multiple and pins the maximum sequence length.
  `benchmarks/kernel/operations/attention_core.py` depends on that padding,
  and it also removes a per-forward device-to-host sync.

`benchmarks/e2e/engines/torchtitan/plugins/parallelize.py` additionally needs
`parallelize_qwen3`'s `skip_dp` keyword and its ordering guarantee:
activation checkpointing, then the per-block compile, then the early return
before mesh resolution. At one data-parallel rank the function skips FSDP,
so the model holds plain bf16 parameters and `training.dtype` is the only
bf16 mechanism. Above one rank it applies `fully_shard`, because TorchTitan
has no DDP class, and it prints the `piper1b data parallel` line after it
counts the FSDP units. That line, and not TorchTitan's mesh line, proves
the wrap, because TorchTitan logs the mesh before the function runs.

`benchmarks/e2e/engines/torchtitan/plugins/config_registry.py` and
`benchmarks/models/piper_qwen3/titan_model.py` import private Qwen3
helpers. After a bump, verify that each of these still exists with
unchanged behaviour:

- `_build_qwen3_moe_layers`, `_EMBEDDING_INIT`, `_output_linear_init`,
  `_qwen3_norm` and `Qwen3Model` from `torchtitan.models.qwen3`
- `CosSinRoPE`, `Embedding` and `Linear` from `torchtitan.models.common`,
  and `decoder_vocab_size` from its config helpers
- `Qwen3StateDictAdapter`, `ModelSpec`, `Trainer`, `CheckpointManager`,
  `CrossEntropyLoss`, `LRSchedulersContainer`, `MetricsProcessor`,
  `default_adamw`, `TrainingConfig`, `SelectiveAC`, `pipeline_llm` and
  `HuggingFaceTextDataLoader`
- the trainer's `size: <N> total parameters` log line, which the model fact reads
- the per-block compile log line, which the compile rule matches through
  the substring `with torch.compile`
- `override` and `derive` from the config override module, and the
  `[Override]` log-line format that the override rules read

The kernel scenarios additionally depend on `HelionCosSinRoPE`,
`FusedGroupedExperts`, `GroupedExperts`, `QKVLinear`, `FusedQKVLinear`,
`FlexAttention` and `create_varlen_metadata_for_document`.

`benchmarks/models/piper_qwen3/components/moe/te_per_expert_experts.py`
copies the forward of `GroupedExperts`, and it imports `get_spmd_backend`.
After a bump, compare that forward with the fork's forward again.

`benchmarks/models/piper_qwen3/components/moe/host_count_dispatcher.py`
subclasses `AllToAllTokenDispatcher`. After a bump, verify these points:

- `_sync_token_count_exchange` keeps its signature and its one blocking
  copy, and `dispatch` still calls it once.
- `dispatch` still returns the routed rows, the counts of each local
  expert and the metadata. `RoutedExperts.forward` passes those counts to
  `inner_experts` alone.
- The standard communication backend still builds
  `AllToAllTokenDispatcher.Config` exactly.

Three fork features are no longer load-bearing, and the list above drops
them. The compile-mode field served the deleted compile-mode axis; the
TorchTitan engine sends `--compile.enable` alone. The loss-owned LM head
protocol served the deleted fused linear cross-entropy arm; no module
imports it today. The expert-usage accumulation gate served graph capture, which is
also deleted. Keep the commits on the branch, and do not treat them as
requirements.

Also recheck the documented deltas against Piper: the builder hardcodes
`route_norm=True` where Piper wants `False`, the experts are
`GroupedExperts` rather than Piper's `BmmExperts`, and the
`load_balance_coeff = None` fixup is applied after the build. That fixup
silently stops mattering if the builder default changes.
