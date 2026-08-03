# torchtitan-benchmarks

Declarative end-to-end and kernel microbenchmarks for the Piper Qwen3-1B
TorchTitan port. The repository is out-of-tree: it registers the `piper1b`
module and experiment overrides without modifying the TorchTitan checkout.

The port currently targets TorchTitan revision `b5eb9d92` and imports private
TorchTitan Qwen3 helpers. Revalidate `piper1b/config_registry.py` when
updating TorchTitan.

## Layout

| path | purpose |
|---|---|
| `piper1b/` | Piper Qwen3-1B config port and benchmark-local kernel overrides. |
| `benchmarks/` | Declarative scenarios, Click CLI, runner, artifacts, metrics, and reporting. |
| `analysis/` | Compatibility entry point for evaluation. |
| `microbench/` | Standalone kernel measurements. |
| `tests/` | CLI, runner, artifact, metric, and kernel correctness tests. |

The model configuration is registered as `qwen3_piper_1b`. The port represents
Piper's 1B routed-MoE Qwen3 variant; all scenarios use the same model except
where an arm explicitly selects an alternate trainer configuration.

## Scenarios

| scenario | arms | comparison |
|---|---|---|
| `piper1b_rope` | `baseline`, `helion`, `te` | Stock, Helion, and TE-derived RoPE. |
| `piper1b_swiglu` | `baseline`, `fused_grouped_experts` | Stock routed experts and local combined-gradient grouped experts. |
| `piper1b_qkv` | `baseline`, `fused_qkv` | Separate Q/K/V projections and a fused QKV projection. |
| `piper1b_lm_head` | `baseline`, `fused_linear_ce`, `te_fused_ce`, `piper_optimized_te_ce` | Full-logits, PyTorch fused linear-CE, TE fused CE, and Piper-optimized TE CE. |

Scenario definitions live in `benchmarks/scenarios.py`. They declare the
workload, arm-specific trainer config, overrides, and expected trace markers.

## Run

Install benchmark dependencies into the TorchTitan environment:

```bash
pip install -r requirements.txt
```

```bash
# List available scenarios and arms.
./run_bench.sh scenarios

# Run, validate, and evaluate every arm in one scenario.
./run_bench.sh run-all <gpu-index> --scenario piper1b_rope

# Run one arm only.
./run_bench.sh run <gpu-index> --scenario piper1b_qkv --arm fused_qkv

# Evaluate a completed run without training again.
./run_bench.sh evaluate out/<timestamp>/<scenario>/<hardware>

# Resume an interrupted all-arm run. Valid arms are retained.
./run_bench.sh run-all <gpu-index> --resume \
    out/<timestamp>/<scenario>/<hardware>
```

`run-all` runs every selected arm, validates completion and traces, then writes
`results.json` automatically. `run` performs only execution and validation.
Pass additional TorchTitan arguments after `--`.

The runner accepts `--batch`, `--seq-len`, `--steps`, `--out`, `--titan-dir`,
`--cache-root`, and `--compiler-env`; `--help` shows their environment-variable
equivalents. The default workload is batch 4, sequence length 1024, and 40
steps. It requires at least two profiler windows, so use at least 40 steps
unless the scenario schedule is also changed.

## Outputs and evaluation

Each run writes:

```text
out/<timestamp>/<scenario>/<hardware>/
  manifest.json          # workload, commands, source revisions, and hardware metadata
  run_state.json         # resumable arm and evaluation status
  results.json           # throughput, memory, region timings, and diagnostics
  <arm>.log              # training output
  <arm>/profiling/traces/iteration_*/rank0_trace.json.gz
```

Evaluation reports stable tokens/s, peak allocated memory, and the GPU span of
the compiled forward and backward transformer-block regions. It also reports
Welch's t-test, Mann-Whitney U, and Cohen's d for pooled per-layer invocation
durations. Those values diagnose differences within a trace; they are not
independent repeated-run significance tests.

Keep local benchmark reports and investigation notes under `/reports/`; the
directory is intentionally ignored. Do not put hardware-specific results or
research conclusions in this README.

## Kernel microbenchmarks

Standalone tools measure isolated CUDA kernels and are not end-to-end results:

- `piper1b/rope/` contains RoPE correctness, timing, significance, accuracy,
  and ablation tools.
- `piper1b/swiglu/benchmark.py` compares upstream and local SwiGLU kernels.
- `piper1b/lm_head/benchmark.py` measures full-token LM-head/loss paths.

## Tests

```bash
python3 -m unittest discover -s tests
```

GPU kernel tests are skipped when CUDA is unavailable. The runner validates
expected overrides, completion, absence of fallback operations, and trace
windows for every executed arm.

## Licenses and provenance

The TE fused cross-entropy code is derived from TransformerEngine and is covered
by `LICENSE.transformer-engine`. The local grouped-expert SwiGLU implementation
is derived from TorchTitan and is covered by `LICENSE.torchtitan`.
