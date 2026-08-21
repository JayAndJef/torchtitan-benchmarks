# torchtitan-benchmarks

Declarative end-to-end and kernel microbenchmarks for the Piper Qwen3-1B
TorchTitan port. The repository is out-of-tree: it registers the
`benchmarks.models.piper_qwen3` module and experiment overrides without
modifying the TorchTitan checkout.

TorchTitan is pinned as a submodule at `third_party/torchtitan`. The port imports
private TorchTitan Qwen3 helpers, so bumping the submodule means revalidating
`benchmarks/models/piper_qwen3/config_registry.py`. Every run records the
revision it used in `manifest.json`.

## Layout

Everything importable lives in the single `benchmarks/` package; `tools/` holds
the argv-driven operator scripts.

| path | purpose |
|---|---|
| `benchmarks/cli/` | Click CLI and the `python -m benchmarks.cli` entry point. |
| `benchmarks/e2e/` | End-to-end system: scenario registry, runner, subprocess launch, validation, results. Includes the Megatron-LM training driver under `e2e/megatron/`. |
| `benchmarks/kernel/` | Kernel-isolation system: scenario registry, span registry, runner/worker, timing engine, arm builders, results. |
| `benchmarks/models/piper_qwen3/` | Piper Qwen3-1B config port, model shape, and benchmark-local kernel overrides under `components/`. Also the Megatron model builder and submodule bootstrap. |
| `benchmarks/traces/` | Chrome-trace parsing and per-region pooling. |
| `benchmarks/artifacts/` | Manifest and run-state IO, output layout, shared sample summaries. |
| `benchmarks/execution/` | Subprocess environment: repo paths, CPU pinning, hardware metadata. |
| `tools/` | Cross-engine parity check, the shared-box matrix supervisor, and standalone trace diagnostics. |
| `tests/` | CLI, runner, artifact, metric, and kernel correctness tests. |
| `third_party/torchtitan/` | Pinned TorchTitan submodule; installed editable into `.venv`. |
| `third_party/Megatron-LM/` | Pinned Megatron-LM submodule; placed on `sys.path`, not pip-installed. |

The model configuration is registered as `qwen3_piper_1b`. The port represents
Piper's 1B routed-MoE Qwen3 variant; all scenarios use the same model except
where an arm explicitly selects an alternate trainer configuration.

## Scenarios

| scenario | arms | comparison |
|---|---|---|
| `piper1b_rope` | `baseline`, `helion`, `te` | Stock, Helion, and TE-derived RoPE. |
| `piper1b_swiglu` | `baseline`, `piper_optimized_triton`, `piper_optimized_inductor` | Stock routed experts and the two Piper fused-w13 grouped-expert variants. |
| `piper1b_qkv` | `baseline`, `fused_qkv` | Separate Q/K/V projections and a fused QKV projection. |
| `piper1b_lm_head` | `baseline`, `fused_linear_ce`, `te_fused_ce`, `piper_optimized_te_ce` | Full-logits, PyTorch fused linear-CE, TE fused CE, and Piper-optimized TE CE. |
| `piper1b_attention` | `baseline`, `flash_attention_3`, `flex_flash` | FlexAttention, FlashAttention-3 varlen, and FlexAttention lowered to FlashAttention-4. |
| `piper1b_megatron` | `baseline`, `titan_stock`, `titan_swiglu`, `titan_lm_head`, `titan_swiglu_lm_head` | Megatron-LM + TransformerEngine against the best-improved TorchTitan configurations. |

Scenario definitions live in `benchmarks/e2e/registry.py`. They declare the
workload, arm-specific trainer config, overrides, and expected trace markers.
Scenario and arm names are stable identifiers and are unaffected by the package
layout.

## Requirements

- An NVIDIA driver reporting CUDA 13.0 or newer in the `nvidia-smi` header. The
  locked wheels are cu130 builds covering `sm_75` through `sm_120`, so Ampere
  (A6000, A100) and Hopper (H100, H200) work without changes. On a driver that
  reports less, `run_bench.sh` sources `cuda_compat.sh`, which stages NVIDIA's
  forward-compat userspace driver under `.cuda-compat/` and prepends it to
  `LD_LIBRARY_PATH`; it is a no-op on drivers that already report 13.0+.
- `uv`, and a host compiler satisfying C++20 for the standalone TE CUDA
  extensions.

torch is pinned to an exact nightly. The PyTorch nightly index retains roughly
sixty days of builds, so the pin eventually stops resolving and has to be bumped;
a bump changes the numbers, so rerun the baselines rather than comparing across
it.

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

`sync.sh` wraps `uv sync` in two passes: the first installs torch and the NVIDIA
header wheels, the second builds the dependency groups that compile without
build isolation against the pinned nightly. A plain `uv sync` works only once
those wheels are already in uv's cache.

The result is a `.venv` with the pinned torch, the TorchTitan submodule installed
editable, and the benchmark dependencies. `run_bench.sh` uses that interpreter,
and the same one launches training, so the CLI and the training process cannot
diverge.

## Run

```bash
# List available scenarios and arms.
./run_bench.sh scenarios

# Run, validate, and evaluate every arm in one scenario.
./run_bench.sh run-all <gpu-index> --scenario piper1b_rope

# Sweep every scenario in sequence, grouped under one out/<timestamp>/.
./run_bench.sh run-all <gpu-index> --all-scenarios

# Run one arm only.
./run_bench.sh run <gpu-index> --scenario piper1b_qkv --arm fused_qkv

# Run at the huge model shape (1 layer, dim 12288;
# see benchmarks/models/piper_qwen3/shape.py).
# Results are only comparable within one --model-size.
./run_bench.sh run-all <gpu-index> --scenario piper1b_megatron \
    --ac none --compile-mode cuda-graph --model-size huge

# Evaluate a completed run without training again.
./run_bench.sh evaluate out/<timestamp>/<scenario>/<hardware>

# Resume an interrupted all-arm run. Valid arms are retained.
./run_bench.sh run-all <gpu-index> --resume \
    out/<timestamp>/<scenario>/<hardware>
```

`run-all` runs every selected arm, validates completion and traces, then writes
`results.json` automatically. `run` performs only execution and validation.
Pass additional TorchTitan arguments after `--`.

`--all-scenarios` runs every scenario in sequence and stops at the first
failure. It cannot be combined with `--scenario`, `--out`, `--resume`, or
`--results`.

The runner accepts `--batch`, `--seq-len`, `--steps`, `--out`, `--cache-root`,
and `--compiler-env`, plus the three global comparability axes `--compile-mode`,
`--ac`, and `--model-size`; `--help` shows their environment-variable
equivalents. The default workload is batch 4, sequence length 1024, and 40
steps. It requires at least two profiler windows, so use at least 40 steps
unless the scenario schedule is also changed.

## Outputs and evaluation

Each run writes:

```text
out/<timestamp>/<scenario>/<hardware>/
  manifest.json          # workload, commands, source revisions, and hardware metadata (including the two cuDNN fields)
  run_state.json         # resumable arm and evaluation status
  results.json           # throughput, memory, gpu kernel time, region timings
  <arm>.log              # training output
  <arm>/profiling/traces/iteration_*/rank0_trace.json.gz
```

Training runs are bound to the GPU's NUMA node with `numactl` when available,
so host scheduling does not decide throughput; the manifest records the
binding as `cpu_pinning`.

The manifest also records which cuDNN the run used, as `cudnn_torch_build`
and `cudnn_loader_resolves`. TransformerEngine binds the host's cuDNN rather
than the pinned wheel's, so which cuDNN a Megatron arm ran is a property of
the host. The version changes no value -- measured on 2026-08-21, bitwise
identical outputs under 9.23.2 and 9.24.0 -- so cite both fields beside a
timing number, where the effect is unmeasured, and not as a reason two runs
disagree numerically.


Run directories written before the package restructure record the module names
this repository used at the time. They still load and evaluate, but they cannot
be resumed across it: `--resume` requires a matching `benchmarks_git_rev`, and
the restructure changed it. No measurement changed -- a file move is not a
benchmark result.

Evaluation reports stable tokens/s, peak allocated memory, per-step GPU kernel
time (the host-speed-immune way to compare kernel implementations), and each
compiled transformer-block region measured both as a GPU span and as summed
kernel time. It also reports Welch's t-test, Mann-Whitney U, and Cohen's d for
pooled per-layer span distributions. Those values diagnose differences within
a trace; they are not
independent repeated-run significance tests.

Keep local benchmark reports and investigation notes under `/reports/`; the
directory is intentionally ignored. Do not put hardware-specific results or
research conclusions in this README.

## Kernel-isolation benchmarks

`kernel-bench` times competing implementations of one kernel family
head-to-head on synthetic tensors at Piper-1B shapes. These are not
end-to-end results: a kernel that wins in isolation can be irrelevant once
the compiler fuses the graph around it.

```bash
./run_bench.sh kernel-bench <gpu>                      # all scenarios
./run_bench.sh kernel-bench <gpu> --scenario expert_mlp # one scenario
./run_bench.sh kernel-bench <gpu> --burst              # add the dispatch-cost diagnostic
```

A **span** is a third kind of unit. A scenario cuts the model at one boundary
and ranks the implementations there; a span is an implementation that fuses
*across* a cut, so it belongs to no scenario and is declared over an ordered
scenario range. Its claim is the span against the sum of the scenarios it
replaces, and a span result holds both totals. `--span NAME` measures one,
plus every scenario it replaces, in the same run; it defaults to none and it
does not combine with `--out` or `--arm`. A span writes to
`out/<timestamp>/kernels/spans/<span>/<hardware>/`, one directory deeper than
a scenario, so a shallow glob of the scenarios cannot sweep one up. The
roster lives in `benchmarks/kernel/spans.py`, `./run_bench.sh scenarios`
prints it, and no span has been measured yet.

Two properties of a span number are not properties of a scenario number, and
both must be said next to it. The interval is an **unpaired** bootstrap
published as `unpaired_ratio_ci_*`, because nothing pairs a span's replicate
with a part's. And the parts side pays one host dispatch chain per enclosed
scenario against the span's one, so a span ratio is biased in the span's own
favour, by more as the range grows.

`--arm NAME` measures a subset of one scenario's arms; repeat it per arm, and
pair it with a single `--scenario`. The selection must name the scenario's
anchor arm, because every comparison is a ratio against it, and every
correctness reference the selected arms use. `kernel-bench` refuses a
selection that omits one of those rather than adding the arm for you. Every
arm the selection leaves out reaches `results.json` and the manifest as
`skipped`, with a reason that names the flag, so a reader can tell the
operator's choice from a host that could not run the arm.

The registry declares 17 scenarios and 71 arms. 16 are cross-engine: they cut
the model at one component and put megatron-core beside TorchTitan there.
`./run_bench.sh scenarios` prints every one with its description, and
`benchmarks/kernel/registry.py` is the authority. Two of the 17:

| scenario | compares |
|---|---|
| `rope` | cross-engine: megatron's THD rotary path against three titan modules, stock vs Helion vs TransformerEngine |
| `lm_head` | full logits vs fused linear-CE vs TE and Piper-optimized cross entropy |

`lm_head` is the only single-engine scenario left. Three others were deleted
once a cross-engine scenario re-homed every one of their arms: `swiglu` into
`expert_mlp`, `qkv` into `qkv_prep`, and `attention` into `attention_core`.
A re-homed titan arm is the same code under an `engine/profile` name, so
`swiglu/piper_optimized_triton` is now
`expert_mlp/titan/piper_optimized_triton`. Two of the successors changed
what the number means, and both say so in their own description: `qkv_prep`
puts the attention-input norm inside the cut, because megatron fuses it into
the GEMM prologue and runs neither half alone; `expert_mlp` puts each Piper
arm against TorchTitan's own w13 fusion rather than against unfused experts.
`lm_head` stays because its `fused_linear_ce` arm has no successor *scenario*:
`FusedLinearCrossEntropyLoss` owns the LM head, so the arm fits neither
`lm_head_projection` nor `cross_entropy`. Its successor is the
`fused_linear_ce` span over both, and `lm_head` stays until that span can be
measured.

`--span NAME` measures a **span** instead of a scenario: an implementation
that fuses across a scenario cut, compared against the SUM of the scenarios
it replaces. A span is never a default, because asking for one adds every
scenario it encloses to the run. The registry declares five --
`expert_combine`, `attn_residual_norm`, `ffn_norm_to_moe_residual`,
`fused_linear_ce` and `chunked_ce` -- and `./run_bench.sh scenarios` prints
each one with the arms it replaces. Two cautions belong beside every span
number. The parts side pays one host dispatch chain per enclosed scenario
and the span pays one, so a ratio below 1.0 holds chains no fusion removed,
and the effect grows with the length of the range. And the span-against-parts
interval is **unpaired**, unlike every scenario ratio: it is published as
`unpaired_ratio_ci_*` and must not be read as a scenario interval.

**No span can be measured yet.** Every span arm names a builder module under
`benchmarks/kernel/operations/` that nobody has written, so `--span` measures
the enclosed scenarios and then fails in the span's own worker. The
declarations are the specification those builders must meet.

Each number is the burst-amortized per-call cost under back-to-back dispatch,
repeated over replicate sweeps so drift affects every arm equally. It is not
device time: where the host cannot keep the stream fed, the measured interval
holds host stalls as well, and such an arm's ratio moves with the burst size.
Run `--burst` to see which arms those are. Every arm is *timed* in its own
process, so one arm's dependencies never reach another's during measurement;
the parent merges the per-worker fragments into the results file. The
correctness pass is the exception: it gates the whole roster in one process,
because 36 of the 71 arms name another arm as their reference and a check
needs both sides at once. It holds one arm resident at a time, building each
and dropping it before the next. Correctness gates run first and fail the run
loudly, and no arm is timed after a failed gate. Every
declared arm reaches the results file with a status of `ok`, `skipped` or
`failed`, so a host that cannot build one arm still measures the rest and
says which one it dropped. Each scenario writes a manifest and results JSON
(with the raw per-replicate samples) under `out/<timestamp>/kernels/`.
`./run_bench.sh scenarios` lists every scenario and arm.

**One cross-engine number exists so far**, and it covers two arms of one
scenario: `attention_core`'s `titan` against `mcore/base` at sequence length
16384, where titan is about 1.12x slower forward and 1.14x slower forward
plus backward. Fifteen of the sixteen cross-engine scenarios have never had
an arm built. Read those as declarations until a run says otherwise.


## Tests

```bash
.venv/bin/python -m unittest discover -s tests
```

GPU kernel tests are skipped when CUDA is unavailable. The runner validates
expected overrides, completion, absence of fallback operations, and trace
windows for every executed arm.

## Licenses and provenance

The TE fused cross-entropy code is derived from TransformerEngine and is covered
by `LICENSE.transformer-engine`. The local grouped-expert SwiGLU implementation
is derived from TorchTitan and is covered by `LICENSE.torchtitan`.
