# torchtitan-benchmarks

Benchmarks for the Piper Qwen3 MoE port of TorchTitan, measured against
stock Megatron-LM.

The repository holds two measurement systems. Never mix their numbers.

| system | command | measures |
|---|---|---|
| End-to-end | `run` | Tokens/s, step time and peak memory of full training runs. |
| Kernel isolation | `kernel-bench` | One model component, implementation against implementation. |

`AGENTS.md` is the full operating guide: flag tables, validation rules,
parallelism rules and reporting rules.

## Setup

You need `uv`, a C++20 host compiler, and an NVIDIA GPU.

```bash
git clone --recurse-submodules https://github.com/JayAndJef/torchtitan-benchmarks
cd torchtitan-benchmarks
./sync.sh
```

`sync.sh` builds `.venv` and installs the pre-push hook. The first sync
compiles FlashAttention 3 from source and takes 15 to 40 minutes. To skip
that build, use `./sync.sh --no-group flash3`.

On a driver older than CUDA 13.0, `run_bench.sh` stages NVIDIA's
forward-compat driver under `.cuda-compat/`. No action is necessary.

## Run

```bash
./run_bench.sh scenarios                            # list scenarios and arms
./run_bench.sh run 0 --model-size 1b                # one GPU
./run_bench.sh run 0,1 --pp 2 --pp-schedule 1F1B    # two-GPU pipeline
./run_bench.sh run 0 --model-size 1b --scenario engines --arm titan_eager
./run_bench.sh evaluate out/<timestamp>/<scenario>/<hardware>
./run_bench.sh run 0 --resume out/<timestamp>/<scenario>/<hardware>
```

`run` trains, validates and evaluates the arms of each scenario. The
`engines` scenario has three arms:

| arm | treatment |
|---|---|
| `titan_compiled` | TorchTitan, whole-block `torch.compile` |
| `titan_eager` | TorchTitan, eager |
| `megatron_stock` | Stock Megatron-LM `pretrain` |

The `attention` scenario keeps `titan_compiled` and `megatron_stock`, and
adds `titan_compiled_fa3`, which replaces TorchTitan's FlexAttention with
FA3 varlen.

Read these points before you publish a number:

- **The default shape needs a mesh.** `30b-a3b` does not fit one GPU. Use
  `--model-size 1b` on one GPU.
- **The Megatron arm differs by design.** It keeps fp32 optimizer state and
  unfused kernels. `AGENTS.md` lists the four differences. State them
  beside each cross-engine number.
- **Compare like with like.** Two runs are comparable only when their
  `manifest.json` files agree on the `run` block, every arm config and every
  source revision.
- **Set an arm's config with `--set`.** Use `--set <arm>.<field>=<value>`,
  or `--set <arm>.extra_flags+=<flags>` for engine flags. A perf flag
  passes. A flag that a harness option owns is refused.

`./run_bench.sh run --help` shows every option.

## Outputs

```text
out/<timestamp>/<scenario>/<hardware>/
  manifest.json     # run block, arm configs, commands, revisions, hardware
  run_state.json    # resumable status
  results.json      # schema 7: tokens/s, step time, peak memory, per-step tables, trajectories
  <arm>.log         # training output, every rank
```

The evaluation reads the logs alone. It publishes the median and the mean
of each figure, each at the slowest rank for that statistic. The mean
tokens/s is the total tokens over the total time. It gives no baseline and
no ratio.

Put reports and notes in `reports/`, which git ignores. Keep results out of
this file.

## Kernel-isolation benchmarks

```bash
./run_bench.sh kernel-bench 0                         # all scenarios
./run_bench.sh kernel-bench 0 --scenario expert_mlp   # one scenario
./run_bench.sh kernel-bench 0 --burst                 # dispatch-cost check
```

Each scenario cuts the model at one component and times each implementation
in its own process, after a correctness gate. A number is the per-call cost
under back-to-back dispatch. It is not device time.

A kernel number is not an end-to-end number. A kernel that wins in
isolation can lose once the compiler fuses the graph around it. Most
cross-engine scenarios have no built arms yet. `benchmarks/kernel/registry.py`
is the authority.

## Tests

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m unittest discover -s tests
```

The suite runs on the CPU in about a minute. The pre-push hook runs it.

## Layout

| path | contents |
|---|---|
| `benchmarks/e2e/` | End-to-end runner, checks, validation and results |
| `benchmarks/e2e/engines/` | One package per training engine, behind one interface |
| `benchmarks/kernel/` | Kernel-isolation system |
| `benchmarks/models/piper_qwen3/` | The model port and its shapes |
| `benchmarks/cli/` | The command line |
| `tools/` | Matrix runner, matrix collector, pre-push hook |
| `third_party/` | Pinned TorchTitan fork and Megatron-LM |

## Licenses

The TE fused cross-entropy code comes from TransformerEngine
(`LICENSE.transformer-engine`). The grouped-expert SwiGLU code comes from
TorchTitan (`LICENSE.torchtitan`).
