# torchtitan-benchmarks

End-to-end and kernel benchmarks for the Piper Qwen3-1B TorchTitan port. The
repository is out-of-tree: it registers the `benchmarks.models.piper_qwen3`
module and its experiment overrides without modifying the TorchTitan
checkout.

`AGENTS.md` is the operating guide. It holds the flag tables, the validation
rules, the parallelism rules and the reporting rules. This file is the short
introduction.

TorchTitan is pinned as a submodule at `third_party/torchtitan`. The port
imports private TorchTitan Qwen3 helpers, so bumping the submodule means
revalidating `benchmarks/models/piper_qwen3/config_registry.py`. Every run
records the revision it used in `manifest.json`.

## Layout

Everything importable lives in the single `benchmarks/` package; `tools/`
holds the argv-driven operator scripts.

| path | purpose |
|---|---|
| `benchmarks/cli/` | Click CLI and the `python -m benchmarks.cli` entry point. |
| `benchmarks/e2e/` | End-to-end system: scenario registry, runner, subprocess launch, validation, results. The stock Megatron-LM driver sits under `benchmarks/e2e/megatron_stock/`. |
| `benchmarks/kernel/` | Kernel-isolation system: scenario registry, span registry, runner and worker, timing engine, arm builders, results. |
| `benchmarks/models/piper_qwen3/` | Piper Qwen3-1B config port, model shape, and the benchmark-local kernel components. Also the Megatron model builder and the submodule bootstrap. |
| `benchmarks/traces/` | Chrome-trace parsing, read under `--profile` alone. |
| `benchmarks/artifacts/` | Manifest and run-state IO, output layout, shared sample summaries. |
| `benchmarks/execution/` | Subprocess environment: repository paths, CPU pinning, hardware metadata. |
| `tools/` | The shared-box matrix supervisor, the matrix collector and the pre-push hook. |
| `tests/` | CLI, runner, artifact, metric and kernel correctness tests. |
| `third_party/torchtitan/` | Pinned TorchTitan submodule; installed editable into `.venv`. |
| `third_party/Megatron-LM/` | Pinned Megatron-LM submodule; placed on `sys.path`, not pip-installed. |

The model configuration is registered as `qwen3_piper_1b`. The port
represents Piper's 1B routed-MoE Qwen3 variant.

## The scenario

There is one end-to-end scenario, `engines`. It trains one model on one
pre-tokenized c4_test stream under three arms:

| arm | comparison |
|---|---|
| `titan_compiled` | TorchTitan with whole-block `torch.compile`. |
| `titan_eager` | The same model and stream, with the blocks eager. |
| `megatron_stock` | Stock Megatron-LM through its own `pretrain` entry point. |

The scenario definition lives in `benchmarks/e2e/registry.py`. It declares
the workload, each arm's compile treatment and engine, and the expected
trace markers. Scenario and arm names are stable identifiers.

The Megatron arm is not plain bf16, and three other deliberate differences
move its number. `AGENTS.md` lists all four; state them beside any
cross-engine result.

## Requirements

- An NVIDIA driver reporting CUDA 13.0 or newer in the `nvidia-smi` header.
  The locked wheels are cu130 builds covering `sm_75` through `sm_120`, so
  Ampere and Hopper work without changes. On a driver that reports less,
  `run_bench.sh` sources `cuda_compat.sh`, which stages NVIDIA's
  forward-compat userspace driver under `.cuda-compat/` and prepends it to
  `LD_LIBRARY_PATH`.
- `uv`, and a host compiler satisfying C++20 for the standalone CUDA
  extensions.

torch is pinned to an exact nightly. The PyTorch nightly index retains
roughly sixty days of builds, so the pin eventually stops resolving and has
to be bumped. A bump changes the numbers, so rerun the baselines rather than
comparing across it.

## Setup

```bash
git clone --recurse-submodules https://github.com/JayAndJef/torchtitan-benchmarks
cd torchtitan-benchmarks
./sync.sh
```

For a clone made without `--recurse-submodules`:

```bash
git submodule update --init --recursive
./sync.sh
```

`sync.sh` wraps `uv sync` in two passes: the first installs torch and the
NVIDIA header wheels, the second builds the dependency groups that compile
without build isolation against the pinned nightly. A plain `uv sync` works
only once those wheels are already in uv's cache. `sync.sh` also installs
the pre-push hook, which runs the CPU test suite before a push.

The result is a `.venv` with the pinned torch, the TorchTitan submodule
installed editable, and the benchmark dependencies. `run_bench.sh` uses that
interpreter, and the same one launches training, so the CLI and the training
process cannot diverge.

## Run

```bash
# List every scenario and arm, end-to-end and kernel.
./run_bench.sh scenarios

# Run, validate and evaluate every arm. One command does all three.
./run_bench.sh run <gpu-index>

# Run one arm only; repeat --arm once per selected arm.
./run_bench.sh run <gpu-index> --scenario engines --arm titan_eager

# Collect profiler traces. It needs at least 40 steps.
./run_bench.sh run <gpu-index> --profile --steps 40

# Run a two-GPU pipeline.
./run_bench.sh run 0,1 --pp 2 --pp-schedule 1F1B

# Evaluate a completed run without training again.
./run_bench.sh evaluate out/<timestamp>/<scenario>/<hardware>

# Resume an interrupted run. The arms that already validate are kept.
./run_bench.sh run <gpu-index> --resume out/<timestamp>/<scenario>/<hardware>
```

`run` always evaluates, and it is fail-fast. Pass extra TorchTitan arguments
after `--`. `--out`, `--resume` and `--results` each name one directory, so
each needs exactly one selected scenario.

The run axes are `--ac`, `--model-size`, `--dp`, `--pp`, `--ep`, `--zero`,
`--pp-schedule`, `--pp-microbatch-size`, `--megatron-p2p-sync`,
`--megatron-nan-guard`, `--megatron-precision`, `--profile` and
`--warmup-steps`. Results are comparable only within one value of each.
`--help` shows the defaults and the environment-variable equivalents, and
`AGENTS.md` holds the full table.

## Outputs and evaluation

Each run writes:

```text
out/<timestamp>/<scenario>/<hardware>/
  manifest.json          # workload, axes, commands, source revisions, hardware metadata
  run_state.json         # resumable arm and evaluation status
  results.json           # tokens/s, step time, peak memory, trajectories, warnings
  <arm>.log              # training output, every rank
  <arm>/profiling/traces/iteration_*/rank<n>_trace.json.gz   # --profile only
```

Training runs are bound to the GPU's NUMA node with `numactl` when
available, so host scheduling does not decide throughput; the manifest
records the binding as `cpu_pinning`.

Evaluation reads the logs alone and publishes absolute numbers per arm:
tokens/s at the slowest rank, that rank's step time, peak allocated memory,
and the loss and gradient-norm trajectories as a sanity check. There is no
baseline arm and no ratio. Send kernel-level questions to `kernel-bench`,
and trace-level questions to the external trace-anatomy tool.

Keep local benchmark reports and investigation notes under `reports/`; the
directory is intentionally ignored. Do not put hardware-specific results or
research conclusions in this README.

## Kernel-isolation benchmarks

`kernel-bench` times competing implementations of one model component
head-to-head on synthetic tensors at Piper shapes. These are not end-to-end
results: a kernel that wins in isolation can be irrelevant once the compiler
fuses the graph around it.

```bash
./run_bench.sh kernel-bench <gpu>                        # all scenarios
./run_bench.sh kernel-bench <gpu> --scenario expert_mlp  # one scenario
./run_bench.sh kernel-bench <gpu> --burst                # dispatch-cost diagnostic
```

The registry declares 17 scenarios, 71 arms and 5 spans, and
`benchmarks/kernel/registry.py` is the authority. 16 of the scenarios are
cross-engine: each cuts the model at one component and puts megatron-core
beside TorchTitan there. `./run_bench.sh scenarios` prints every one with
its description.

A **span** is a third kind of unit. A scenario cuts the model at one
boundary and ranks the implementations there; a span is an implementation
that fuses *across* a cut, so it is declared over an ordered scenario range
and compared against the sum of the scenarios it replaces. `--span NAME`
measures one, plus every scenario it replaces, in the same run. **No span
can be measured yet**: every span arm names a builder module that nobody has
written, so the declarations are the specification those builders must meet.

Two properties of a span number are not properties of a scenario number, and
both belong beside it. The interval is an **unpaired** bootstrap, published
as `unpaired_ratio_ci_*`, because nothing pairs a span's replicate with a
part's. And the parts side pays one host dispatch chain per enclosed
scenario against the span's one, so a span ratio is biased in the span's own
favour, by more as the range grows.

Each number is the burst-amortized per-call cost under back-to-back
dispatch, repeated over replicate sweeps so drift affects every arm equally.
It is not device time: where the host cannot keep the stream fed, the
measured interval holds host stalls as well. Every arm is timed in its own
process, and the correctness pass is the exception: it gates the whole
roster in one process, because many arms name another arm as their
reference. Correctness gates run first and fail the run loudly. Each
scenario writes a manifest and a results file, with the raw per-replicate
samples, under `out/<timestamp>/kernels/`.

**One cross-engine number exists so far**, and it covers two arms of one
scenario: `attention_core`'s titan arm against `mcore/base` at sequence
length 16384, where titan is about 1.12x slower forward and 1.14x slower
forward plus backward. Fifteen of the sixteen cross-engine scenarios have
never had an arm built. Read those as declarations until a run says
otherwise.

## Tests

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests
```

GPU kernel tests skip themselves when CUDA is unavailable. The same command
runs as a pre-push hook, and `.github/workflows/tests.yml` runs the
CPU-only modules on every push.

## Licenses and provenance

The TE fused cross-entropy code is derived from TransformerEngine and is
covered by `LICENSE.transformer-engine`. The local grouped-expert SwiGLU
implementation is derived from TorchTitan and is covered by
`LICENSE.torchtitan`.
