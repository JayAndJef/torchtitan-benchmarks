# Piper-inspired stock Megatron against TorchTitan

Revision 2. An earlier draft recorded the decisions. This revision verifies
every claim against the code, replaces each guess with a decision, and adds
the detail an implementer needs. Read section "What revision 1 got wrong"
before you reuse anything from memory.

Repository state when this plan was written: branch `parallelism/pp-first`,
HEAD `5e6474e`. Megatron-LM submodule `59b72fa57` (core 0.20.0, reported by
`git describe` as `core_v0.15.0rc7-2120-g59b72fa57`).

## 0. Status of the launch gate

Revision 1 said "stop and request fresh approval before any GPU run".

**The operator gave that approval in writing on 2026-08-26.** The approval
covers the full run matrix in section 11, plus any additional model size or
configuration that is useful and that fits the hardware. The gate is
therefore satisfied. This section keeps the record that a gate existed,
because a later reader must not read the matrix as self-authorised.

Two conditions still apply, and they come from `CLAUDE.md`, not from the
gate:

- Check `nvidia-smi` for eight free GPUs before each cell. A shared GPU
  invalidates the timing.
- Do not run a GPU cell until the CPU test suite passes at the commit you
  run.

## 1. Terms

| term | meaning |
|---|---|
| stock arm | The new `baseline` arm. It runs `megatron.training.pretrain`. |
| tuned arm | The existing `piper1b_megatron/baseline` arm. It runs `benchmarks/e2e/megatron/train.py`. |
| titan arm | `titan_stock`, TorchTitan on the replay data stream. |
| spec rule N | A numbered check in `validate_parallelism`. |
| arm rule N | A numbered check in `validate_arm`. |
| shim | A first-party substitution installed in our own process. It edits no file. |
| DP | Data parallelism. PP: pipeline parallelism. EP: expert parallelism. |

The two rule families use the same numbers for different checks. Always
write "spec rule N" or "arm rule N". Never write a bare "rule N".

## 2. Goal, and the one claim this suite makes

Add one scenario that measures **stock Megatron-LM** against **stock
TorchTitan** on the same data, at `--dp 2 --pp 4` on eight GPUs.

The claim is a **systems-throughput** claim about two configured engines. It
is not a numerical-equivalence claim, and it is not a claim about the
Megatron engine as such. Four differences below are deliberate and each one
changes the number:

1. The stock arm keeps **fp32 master weights and reduces gradients in fp32**.
   The titan arm and the tuned arm run plain bf16. Section 6.1 gives the
   evidence.
2. The stock arm runs Megatron's **unfused native cross entropy**. The tuned
   arm runs the TransformerEngine cross entropy.
3. The stock arm keeps Megatron's `--init-method-std 0.01`. The titan arm
   keeps TorchTitan's own initialization. No weight transfer happens.
4. The stock arm applies **no permutation fusion**, because stock Megatron
   defaults it off.

The report must state all four beside every number.

This scenario does not replace the tuned arm. The tuned arm stays the
"Megatron at its own best" comparison.

## 3. What revision 1 got wrong

Each item below is a verified error, not an opinion.

| revision 1 said | the truth | evidence |
|---|---|---|
| "Add the minimal replay-data hook to vendored `pretrain_gpt.py`" | That edits a submodule, which the operator forbids. It is also unnecessary. | Section 5. |
| "Models: Piper 1B and Piper 9B" without naming registry keys | `PIPER_SHAPES` holds **six** shapes: `1b`, `large`, `9b`, `huge`, `giant`, `48b`. `9b` exists and matches Piper's own `qwen3_9b` config field for field. | `benchmarks/models/piper_qwen3/shape.py`; `/m-coriander/coriander/jayden/piper/artifact/run_megatron.py` `MODEL_CONFIGS`. |
| "one stock-comparison scenario per model size" | The model size is a **run axis** (`--model-size`), not part of a scenario. `CLAUDE.md` rejects a per-size scenario copy by name. | `benchmarks/cli/e2e.py`; `CLAUDE.md`, "Model sizes". |
| `e2e run --local-batch-size ... --microbatch-size ... --activation-checkpointing none` | The real spellings are `./run_bench.sh run <gpu> ... --batch N --pp-microbatch-size N --ac none`. | `benchmarks/cli/e2e.py:104-217`. |
| "no auxiliary router loss (instead of Piper's aux-loss .001)" with no flag | The flag pair is `--moe-router-load-balancing-type none --moe-aux-loss-coeff 0.0`. | `arguments.py:3370-3375`. |
| Silent about the profiler | Stock Megatron writes **one** trace per rank to `{tensorboard_dir}/../torch_profile/rank-<n>.json.gz`, with `repeat=1`. The harness needs **two** windows at `<arm>/profiling/traces/iteration_*/rank<n>_trace.json.gz`. | `training.py:3702-3730`; `benchmarks/artifacts/layout.py:82-91`. |
| Silent about arm rule 11 | Stock Megatron prints no whole-model parameter total. It prints a per-stage count in another format. | `training.py:1815-1829`; `benchmarks/e2e/validation.py:425`. |
| Silent about Python | `megatron.training` needs `typing.override`, which is Python 3.12. The venv is Python **3.10.20**. Without a shim the whole stock path fails at import. | Section 5.2. |
| Silent about NUMA | Eight GPUs on this box span two NUMA nodes, so every cell runs **unpinned**. | Section 9.4. |
| "Prefer local batch 32 / microbatch 4 ... fallback 8 / 1" | Both are legal and both give exactly 8 microbatches. The arithmetic was never shown. | Section 10.1. |

## 4. Verified facts

Every row was read at HEAD `5e6474e`. Nothing here is assumed.

| fact | evidence |
|---|---|
| `pretrain_gpt.py`'s providers are module-level functions. | `pretrain_gpt.py:282` `forward_step`, `:444` `train_valid_test_datasets_provider`, `:480` `get_embedding_ranks`. |
| `pretrain()` takes the provider, the model type and the forward step as arguments. | `training.py:1015-1030`. |
| The model comes from `cfg_container.model`, whose `builder` is a dotted string. | `megatron/training/models/gpt.py:154`. |
| `--dataloader-type external` passes our object through as the dataloader. | `data_samplers.py:39-42`. |
| The external iterator must implement `__next__`. | `rerun_state_machine.py:1146` calls `next(self.iterable)`. |
| Megatron merges per-sample `cu_seqlens` and strips trailing `seq_length` padding itself. | `megatron/core/utils.py:2517-2609`. |
| `NullTokenizer` takes `--vocab-size` and adds no token. | `null_tokenizer.py:16-20`. |
| `151936` is already a multiple of 128, so no vocab padding occurs. | `build_tokenizer.py:121-134`. |
| Stock Megatron refuses `cross_entropy_fusion_impl='te'` only when `cross_entropy_loss_fusion` is on. | `arguments.py:1630-1635`. |
| `--bf16` alone forces fp32 master parameters and fp32 gradient reduction. | `arguments.py:1168-1182`; `optimizer_config.py:187-203`. |
| Both pipeline-split accounting flags default `False`, so Megatron divides `num_layers`. | `transformer_config.py:130-136`. |
| The harness sends TorchTitan `pipeline-parallel-first/last-stage-less-layers 0`. | `benchmarks/e2e/launch.py:68-106`. |
| Arm rule 11 matches `f"size: {shape.param_count:,} total parameters"`. | `benchmarks/e2e/validation.py:425`. |
| Arm rule 13's marker is `ncclDevKernel_AllReduce`, read on **every** rank. | `benchmarks/e2e/validation.py:107, 621-632`. |
| `MAX_WORLD_SIZE = 4` and `MAX_PP = 2` are the only caps in the repository. | `benchmarks/e2e/parallelism.py:109-110`. |
| A run declares no regions whenever `pp != 1`. | `benchmarks/e2e/runner.py:353-357`. |
| A titan-only `--arm` subset may use `--compile-mode none` even when the scenario declines it. | `benchmarks/e2e/runner.py:288-302`. |
| Both engines run with `cwd=third_party/torchtitan`. | `benchmarks/e2e/runner.py:560`. |
| `add_megatron_to_path()` inserts the Megatron root at `sys.path[0]`. | `benchmarks/models/piper_qwen3/megatron_bootstrap.py:38-42`. |
| GPUs 0 to 3 sit on NUMA node 0. GPUs 4 to 7 sit on node 1. | `/sys/bus/pci/devices/0000:*:00.0/numa_node`. |
| The CPU suite passes today: 1568 tests, 11 skipped. `TEST_CENSUS_TOTAL` is 1481. | `tests/test_migration_contract.py:1780`. |

## 5. The three blocking problems, and how each is solved

### 5.1 The `third_party/` rule

**Verdict: no submodule edit is needed. Revision 1's step 4 is withdrawn.**

`pretrain_gpt.py` builds nothing in its `__main__` block that a first-party
module cannot build. The block calls `parse_and_validate_args`,
`gpt_config_from_args`, `pretrain_cfg_container_from_args` and then
`pretrain(...)`. Every one of those is importable, and every provider it
passes is a module-level function.

So `benchmarks/e2e/megatron_stock/train.py` reproduces that block and
substitutes **one** argument: the dataset provider. The stock model builder,
optimizer, scheduler, distributed setup, forward step, embedding-rank rule
and training loop all stay Megatron's.

Proof that the import chain works on this box, run on 2026-08-26:

```
OK megatron.training
OK megatron.training.training
OK megatron.training.arguments
OK model_provider
OK gpt_builders
OK pretrain_gpt
```

### 5.2 Python 3.10 against a Python 3.12 import

`megatron/training/models/hybrid.py:5` reads
`from typing import Any, Callable, ClassVar, Literal, override`.
`typing.override` arrived in Python 3.12. The venv runs Python 3.10.20, so
the import fails with `ImportError: cannot import name 'override'`.

The existing tuned driver never met this, because it imports `megatron.core`
only. The stock path imports `megatron.training`, so it does meet it.

**Decision: install a one-name shim before the first Megatron import.**

```python
import typing
import typing_extensions

if not hasattr(typing, "override"):            # Python 3.10 and 3.11
    typing.override = typing_extensions.override
```

Measured: with that one line, `megatron.training`,
`megatron.training.training`, `model_provider`, `gpt_builders` and
`pretrain_gpt` all import, and the argument parser builds **847** flags.

Rules for the shim, and they are not optional:

- It lives in one function in `benchmarks/e2e/megatron_stock/bootstrap.py`.
- It runs **before** `add_megatron_to_path()` returns to the caller.
- It sets the name only when `typing` lacks it, so a Python 3.12 venv is
  untouched.
- If `typing_extensions` has no `override`, it raises. It never continues
  silently.

The alternative was to move the venv to Python 3.12. That rebuilds every
wheel, rebuilds FlashAttention-3 from source, and makes every published
number incomparable. Reject it for this work.

### 5.3 The profiler layout

Stock Megatron writes one trace per rank, one window, at
`{args.tensorboard_dir}/../torch_profile/rank-<rank>.json.gz`
(`training.py:3714-3717`). Its schedule carries `repeat=1`
(`training.py:3724`).

The harness reads `<arm_dir>/profiling/traces/iteration_*/rank<n>_trace.json.gz`
and arm rule 5 needs `min_trace_windows` files per rank, which is 2.

**Decision: shim `torch.profiler.profile` in the driver process.**

`training.py:3719` calls `torch.profiler.profile(...)` by attribute at run
time. The driver replaces that attribute before it calls `pretrain()`. The
replacement:

- keeps every keyword the caller passed, except two;
- replaces `schedule` with
  `torch.profiler.schedule(wait=profile_freq - warmup - active, warmup=warmup, active=active, repeat=0)`;
- replaces `on_trace_ready` with a handler that writes
  `<arm_dir>/profiling/traces/iteration_<prof.step_num>/rank<rank>_trace.json.gz`;
- counts its own calls.

Loud guards, all required:

- The driver raises if the shim was called a number of times other than 1.
- The driver raises if this rank wrote fewer than `min_trace_windows` trace
  files after `pretrain()` returns.
- The command line sets `--profile-step-end` equal to `--train-iters`, so
  Megatron calls `prof.stop()` at the last iteration and the second window
  flushes. `prof.stop()` at `training.py:3241` emits the pending trace.

Do **not** lower `min_trace_windows` for this scenario. Two windows are what
every published figure in this repository rests on.

## 6. Model, data and precision decisions

### 6.1 Precision: stock Megatron is not plain bf16

With `--bf16` and no `--use-precision-aware-optimizer`, Megatron sets
`params_dtype=bfloat16`, keeps `main_params_dtype=float32`,
`exp_avg_dtype=float32`, `exp_avg_sq_dtype=float32`, and forces
`accumulate_allreduce_grads_in_fp32=True` (`arguments.py:1174-1182`,
`optimizer_config.py:187-203`).

So the stock arm holds about **18 bytes per parameter** of state, against
titan's 8, and it all-reduces gradients in fp32 rather than bf16.

**Decision: keep the stock defaults.** Piper ran them, a stock user gets
them, and changing them would make the arm neither Piper nor stock.

Two consequences the report must carry:

- `execution_model` in the manifest reads `plain-bf16`, because
  `benchmarks/e2e/parallelism.py` composes it from the **spec**. That string
  describes the titan arm's parameter treatment and **not** the stock arm's.
  Do not change `execution_model`; the trivial-spec value is a fixed point
  that a test pins. State the difference in the arm description, in the
  scenario description and in the report.
- The driver prints the four precision fields on its mode line, so the log
  records them per run.

### 6.2 The model shapes this suite runs

**`1b` and `9b`.** Both are registered, and both are transcribed from
Piper's own configs.

| | `1b` | `9b` |
|---|---|---|
| dim | 1024 | 2048 |
| n_layers | 16 | 24 |
| n_heads / n_kv_heads | 16 / 8 | 32 / 8 |
| head_dim | 64 | 64 |
| MoE width | 3584 | 7168 |
| experts / top_k | 4 / 2 | 8 / 2 |
| param_count | 1,066,241,024 | 9,330,201,600 |
| layers per stage at pp 4 | 4 | 6 |

Piper's `run_megatron.py` `MODEL_CONFIGS["qwen3_1b"]` and `["qwen3_9b"]`
match those two rows field for field. No new shape entry is needed, and none
is in scope.

**`9b` has never run in either engine.** Its `parity_gate` of 2e-2 is a
fitted prediction that no parity check has tested, and arm rule 7 has never
seen a 24-layer trace. Neither matters for this suite: a `pp 4` run declares
no regions, so arm rule 7 does not judge it, and this suite makes no parity
claim. Run `tools/megatron_parity_check.py --model-size 9b` before you make
any numerical claim at `9b`.

### 6.3 Data

Both engines consume the same `c4_test` stream, rank for rank.

The stock provider calls
`benchmarks.e2e.megatron.data.materialize_titan_samples(seq_len=..., num_samples=steps * local_batch_size, dp_rank=mpu.get_data_parallel_rank(), dp_world_size=mpu.get_data_parallel_world_size())`.
That function drains TorchTitan's own dataset class with TorchTitan's own
tokenizer, and `tests/test_megatron_data.py` already asserts that its stream
is bit-identical to the titan replay loader's.

**Decision: `--dataloader-type external`.** Megatron's own
`MegatronPretrainingSampler` would shard a global stream by a different
rule than TorchTitan's `split_dataset_by_node`, so the two engines would put
different tokens on the same rank. The external path removes that risk and
depends on no sampler internals.

**Decision: `--dataloader-inter-document-masking`.** TorchTitan applies a
block-diagonal packed-document mask. Without this flag, stock Megatron
applies a plain causal mask over the whole sequence, which is strictly more
attention work, and the throughput comparison would then be biased against
Megatron. The flag is stock, and `get_batch` handles the packed keys itself
(`pretrain_gpt.py:104`, `:150`).

The iterator yields one dict per microbatch, on CPU:

| key | dtype | shape | value |
|---|---|---|---|
| `tokens` | int64 | `(m, S)` | titan's `input` rows |
| `labels` | int64 | `(m, S)` | titan's `label` rows |
| `loss_mask` | float32 | `(m, S)` | all ones |
| `position_ids` | int64 | `(m, S)` | titan's per-document `positions` |
| `cu_seqlens` | int32 | `(m, P)` | each row: that sample's document offsets in `[0, S]`, right-padded with copies of `S` |
| `max_seqlen` | int32 | `(m,)` | that sample's longest document |
| `attention_mask` | -- | -- | `None` |
| `cu_seqlens_padded` | -- | -- | `None` |

`m` is `--micro-batch-size`. `S` is `--seq-length`. `P` is one constant per
run, taken over that rank's own packs.

Megatron merges the rows and strips the padding itself
(`megatron/core/utils.py:2541-2560`), so the padding convention above is
Megatron's own, not ours.

Three rules the provider must obey:

1. Build on **every** rank. With `dataloader_inter_document_masking` on,
   middle pipeline stages also call `next(data_iterator)`
   (`pretrain_gpt.py:116-121` skips the early return).
2. Key on the **data-parallel** rank, never on the global rank. The four
   stages of one pipeline must read the same tokens in the same order.
3. Raise on exhaustion. Never wrap. A wrap changes the workload silently.

**Known difference against the tuned arm:** the merged `cu_seqlens` length
varies per step, because the document count per pack varies. The tuned
driver pads it to a constant length for CUDA-graph capture. This scenario
runs no graphs, so a varying length is legal. If TransformerEngine shows a
per-step planning cost, supply `cu_seqlens_padded` as well. This is
untested.

### 6.4 The tokenizer

**Decision: `--tokenizer-type NullTokenizer --vocab-size 151936
--padded-vocab-size 151936`.**

`NullTokenizer` needs no file. It adds no token: its `vocab_size` is exactly
the value passed (`null_tokenizer.py:16-20`). `--padded-vocab-size` sets
`should_pad_vocab=False` (`argument_utils.py:473-476`), which pins the
embedding table at 151,936 rows whatever `--make-vocab-size-divisible-by`
holds. Arm rule 11 then has a deterministic target.

The tokenizer never tokenizes anything, because the provider supplies
already-tokenized rows.

### 6.5 The parameter-count proof, for arm rule 11

Stock Megatron prints a per-stage count in its own format
(`training.py:1821-1826`) and no whole-model total. Arm rule 11 needs
`size: <N> total parameters` on **every** rank.

**Decision: a first-party model builder subclass prints it.**

`GPTModelConfig.builder` is a `ClassVar[str]` dotted path
(`megatron/training/models/gpt.py:154`). The driver calls
`gpt_config_from_args(args, model_config_cls=BenchGPTModelConfig)`, whose
only change is
`builder = "benchmarks.e2e.megatron_stock.model_builder.CountingGPTModelBuilder"`.
That is the same extension point the ModelOpt branch of `pretrain_gpt.py`
already uses.

The builder subclasses `GPTModelBuilder`, builds the model through `super()`,
and then, on every rank:

1. counts `sum(p.numel() for p in model.parameters())`;
2. asserts the count equals
   `shape.stage_param_count(pipeline_degree=pp, stage_index=stage)`, and
   raises with both numbers when it does not;
3. prints `stock-megatron stage <i>/<pp> local size: <n> parameters`;
4. prints `Model qwen3_piper_<size> stock-megatron size: <shape.param_count:,> total parameters`.

Step 4 runs only after step 2 passes, so the printed total is backed by a
real per-rank check. The identity
"the stage counts sum to `param_count`" is arithmetic that
`tests/test_model_shape.py` already pins.

**No collective runs inside the builder.** An all-reduce there would deadlock
if any rank ever built a different number of model chunks. The tuned driver
takes the collective route; this one does not. Say so in the commit message.

The builder must raise if `virtual_pipeline_model_parallel_size` is not
`None`, because a virtual pipeline breaks the one-chunk assumption.

## 7. The stock flag list

The flags below were checked against a real parser. The check enumerated all
847 option strings that `add_megatron_arguments` builds and compared them
against this list. **Re-run that check after any submodule bump**:

```bash
.venv/bin/python - <<'PY'
import sys, typing, typing_extensions, argparse
sys.path.insert(0, "third_party/Megatron-LM")
if not hasattr(typing, "override"):
    typing.override = typing_extensions.override
from megatron.training.arguments import add_megatron_arguments
p = argparse.ArgumentParser(allow_abbrev=False)
add_megatron_arguments(p)
names = {s for a in p._actions for s in a.option_strings}
print(sorted(n for n in FLAGS if n not in names))   # FLAGS = the list below
PY
```

### Geometry, from `PiperShape`

```
--num-layers <n_layers>            --hidden-size <dim>
--num-attention-heads <n_heads>    --group-query-attention
--num-query-groups <n_kv_heads>    --kv-channels <head_dim>
--ffn-hidden-size <moe_hidden_dim> --moe-ffn-hidden-size <moe_hidden_dim>
--num-experts <num_experts>        --moe-router-topk <top_k>
--moe-layer-freq 1                 --seq-length <seq_len>
--max-position-embeddings <max_seq_len>
--position-embedding-type rope     --use-rotary-position-embeddings
--rotary-percent 1.0               --rotary-base <int(rope_theta)>
--normalization RMSNorm            --norm-epsilon 1e-6
--swiglu                           --disable-bias-linear
--untie-embeddings-and-output-weights
--qk-layernorm
--attention-dropout 0.0            --hidden-dropout 0.0
--init-method-std 0.01
```

`--rotary-base` is `type=int` (`arguments.py:2332`). Pass `1000000`, never
`1e6`.

### Engine and precision

```
--bf16
--transformer-impl transformer_engine
--use-mcore-models
--no-gradient-accumulation-fusion
```

Attention backend: **pass no backend flag.** TransformerEngine then resolves
to cuDNN FusedAttention on Hopper, which is what the tuned arm runs, so the
two Megatron arms differ in nothing they do not mean to. Piper passes
`--use-flash-attn`; this plan declines it, and section 15 names the reversal.

### MoE

```
--moe-token-dispatcher-type alltoall
--moe-grouped-gemm
--moe-router-load-balancing-type none
--moe-aux-loss-coeff 0.0
--moe-router-dtype fp32
```

`--moe-router-dtype fp32` is a deliberate deviation from Piper. TorchTitan
runs its router gate in fp32 through `torch.autocast`
(`models/common/moe.py:292`), so an unset dtype could make the router a
precision difference that no rule catches.

`--moe-permute-fusion` is **omitted**, because stock Megatron defaults it
off. The tuned arm sets it on, so `_permute_kernel` is **not** a trace marker
for the stock arm.

`--cross-entropy-loss-fusion` is **omitted**, so the stock arm runs
Megatron's unfused native cross entropy. That follows Piper. It is the
slowest cross-entropy path Megatron offers, and the report must say so.

### Optimizer and schedule, matched to TorchTitan

```
--micro-batch-size <pp_microbatch_size>
--global-batch-size <local_batch_size * dp>
--train-iters <steps>
--lr 8e-4        --lr-decay-style linear   --lr-decay-iters <steps>
--lr-warmup-iters 2                        --min-lr 0.0
--adam-beta1 0.9 --adam-beta2 0.95         --adam-eps 1e-8
--weight-decay 0.1                         --clip-grad 1.0
```

Those values come from `benchmarks/e2e/megatron/train.py:629-632` and
`:240-252`, which replicate TorchTitan's own optimizer and its LR lambda.

**Unverified:** whether Megatron's `--lr-decay-iters 40` decays over the 38
steps after the warmup, as TorchTitan does, or over all 40. Read
`OptimizerParamScheduler` before the first run and record the answer. The
learning rate does not change the throughput, so a mismatch is a
reporting defect and not a measurement defect.

### Mesh

```
--tensor-model-parallel-size 1
--context-parallel-size 1
--expert-model-parallel-size 1
--pipeline-model-parallel-size <pp>
```

> **Superseded on 2026-08-27 by the `--dense-sharding` value.**
> `--expert-model-parallel-size` now carries `spec.ep`, not a fixed 1.
> `_mesh_flags` in `benchmarks/e2e/megatron_stock/flags.py` is the
> authority. An expert degree is legal only under `--dense-sharding shard`,
> which spec rule 14 enforces.

Both `--account-for-embedding-in-pipeline-split` and
`--account-for-loss-in-pipeline-split` stay **off**, which is their default.
Megatron then divides `config.num_layers`.

### Data, logging and the profiler

```
--tokenizer-type NullTokenizer  --vocab-size 151936  --padded-vocab-size 151936
--dataloader-type external
--dataloader-inter-document-masking
--no-create-attention-mask-in-dataloader
--num-workers 0
--eval-iters 0                  --eval-interval 1000000
--seed <workload.seed>
--rerun-mode disabled
--log-interval 1                --log-throughput
--profile                       --use-pytorch-profiler
--profile-step-start 1          --profile-step-end <steps>
```

`--mock-data` and `--data-path` are both **omitted**. Megatron allows zero
data sources (`arguments.py:1611-1615`), and the external provider never
builds a Megatron dataset.

`--tensorboard-dir` is **omitted**. The profiler shim owns the trace path, so
Megatron's own path is never used.

`--profile-ranks` is **omitted**, and an empty list means every rank
(`training.py:3704`).

`--rerun-mode disabled` stops `RerunDataIterator` copying every microbatch
(`rerun_state_machine.py:1147`).

### Deliberately omitted, and why

```
--use-distributed-optimizer   --overlap-grad-reduce   --overlap-param-gather
--data-parallel-sharding-strategy   --use-megatron-fsdp
```

Piper passes the first four whenever `dp > 1`. This suite omits all five, so
that both engines replicate their parameters and the DP axis carries one
change and not two. A sharding study is separate work and must not be pooled
with this result.

> **Superseded on 2026-08-27 by the `--dense-sharding` value**, which is
> that sharding study, expressed so the two parities cannot be pooled.
> Under `replicate` this paragraph still holds and all five stay omitted.
> Under `shard` the arm sends `--use-megatron-fsdp`,
> `--megatron-fsdp-version`, `--data-parallel-sharding-strategy`,
> `--use-distributed-optimizer` and `--ckpt-format fsdp_dtensor`, and both
> engines shard. `--overlap-grad-reduce` and `--overlap-param-gather` stay
> omitted under **both** values, for a reason `ALWAYS_OMITTED_FLAGS` gives.
> `SHARDING_FLAGS` and `omitted_flags` in
> `benchmarks/e2e/megatron_stock/flags.py` are the authority. The manifest
> records the parity, and `--resume` refuses to cross it.

`NCCL_P2P_DISABLE` is never set.

### Harness-only flags

The driver adds its own group through Megatron's own `extra_args_provider`
hook (`arguments.py:124-125`), so Megatron parses them:

```
--bench-arm-dir PATH        --bench-model-size NAME
--bench-local-batch-size N  --bench-profile-freq N
--bench-profiler-warmup N   --bench-profiler-active N
--bench-mode MODE           --bench-pp-schedule NAME
```

The driver reads the DP and PP degrees from `args.data_parallel_size` and
`args.pipeline_model_parallel_size`, which Megatron itself resolved. It does
not take them as harness flags. A degree the harness asserts is weaker
evidence than a degree the engine resolved.

## 8. The files to write

### 8.1 New package `benchmarks/e2e/megatron_stock/`

| file | contents |
|---|---|
| `__init__.py` | Empty. Import-light and torch-free. |
| `bootstrap.py` | `install_typing_override()`, then `prepare()` which calls it, `configure_te_environment()` and `add_megatron_to_path()` in that order. |
| `data.py` | `StockReplayIterator`: builds the per-rank sample list, groups it into microbatches, yields the dicts of section 6.3, raises on exhaustion. `provider(train_val_test_num_samples)` returns `(iterator, None, None)` and carries `is_distributed = True`. |
| `model_builder.py` | `BenchGPTModelConfig` and `CountingGPTModelBuilder` of section 6.5. |
| `profiling.py` | `install_profiler_shim(...)` and `assert_windows_written(...)` of section 5.3. |
| `flags.py` | `stock_megatron_flags(shape, workload, spec, ...) -> list[str]`. Parent-side and torch-free, so a test can read the whole command line without a GPU. |
| `train.py` | The driver. Section 8.2. |

`flags.py` must import nothing but `dataclasses`, `PiperShape`,
`ParallelismSpec` and `Workload`. `benchmarks/e2e/launch.py` imports it, and
`launch.py` runs in the parent process.

### 8.2 `train.py`, in order

1. `bootstrap.prepare()`.
2. Import `pretrain_gpt`, `megatron.training`, `megatron.core.enums`.
3. `args = parse_and_validate_args(extra_args_provider=add_bench_args)`.
4. Refuse `--bench-mode` other than `default`. Refuse a
   `--bench-pp-schedule` other than `1F1B`. Refuse
   `virtual_pipeline_model_parallel_size` that is not `None`.
5. Print the mode line and the parallelism line. Section 8.4 fixes both.
6. `install_profiler_shim(arm_dir, profile_freq, warmup, active)`.
7. `model_cfg = gpt_config_from_args(args, model_config_cls=BenchGPTModelConfig)`.
8. `full_config = pretrain_cfg_container_from_args(args, model_cfg)`.
9. `pretrain(full_config, provider, ModelType.encoder_or_decoder, pretrain_gpt.forward_step, get_embedding_ranks=pretrain_gpt.get_embedding_ranks)`.
10. `assert_windows_written(...)`.
11. Print `Training completed` on every rank.

Step 11 runs on every rank on purpose. Arm rule 1 runs per rank, and
Megatron's own `after training is done` line is rank 0 only
(`training.py:1440`).

The driver does **not** call `inprocess_restart.maybe_wrap_for_inprocess_restart`.
A restart wrapper would re-enter the training loop and write a second set of
traces under one label.

### 8.3 Changes to existing files

| file | change |
|---|---|
| `benchmarks/e2e/parallelism.py` | `MAX_WORLD_SIZE = 8`, `MAX_PP = 4`, and the comment that records why. |
| `benchmarks/e2e/launch.py` | A `megatron_stock` branch in `command_for_arm`, plus `_megatron_stock_command`. It reuses `_megatron_launcher(spec)` unchanged. |
| `benchmarks/e2e/validation.py` | A `megatron_stock` `ValidationProfile`, plus `_megatron_stock_parallelism_markers`. |
| `benchmarks/e2e/registry.py` | The new `Scenario` and its two `Arm`s. |
| `tests/test_parallelism.py` | Fix the two cap tests. Add pp 4 cases. |
| `tests/test_migration_contract.py` | `TEST_CENSUS["test_parallelism"]` and `TEST_CENSUS_TOTAL`. |
| `CLAUDE.md` | The new scenario, the new launcher, the new caps, and the four caveats of section 2. |

### 8.4 The exact marker strings

Both the driver and the validation profile must carry these, character for
character. A test compares the two.

Mode line, printed by `train.py` on every rank:

```
Megatron-LM stock training loop (mode=default, main_params_dtype=torch.float32,
main_grads_dtype=torch.float32, accumulate_allreduce_grads_in_fp32=True,
cross_entropy_loss_fusion=False, moe_token_dispatcher_type=alltoall)
```

Print it as one line. The profile's `mode_line` is
`lambda mode: f"Megatron-LM stock training loop (mode={mode},"`.

Parallelism line, printed on every rank above world size 1:

```
Megatron-LM stock parallelism: dp=<dp> pp=<pp> schedule=1F1B microbatches=<m> stages=<pp>
```

`dp` and `pp` come from `args`. `m` is `get_num_microbatches()`.

Data-parallel line, printed on every rank when `dp > 1`:

```
Megatron-LM stock data parallel: DistributedDataParallel over <dp> ranks (overlap_grad_reduce=False, grad_reduce_in_fp32=True)
```

**This line is weaker evidence than the tuned arm's.** It is derived from
`args`, before `pretrain()` wraps the model, so it declares the mesh rather
than observing the wrap. Arm rule 13 carries the mechanism proof: it needs
`ncclDevKernel_AllReduce` in every rank's traces. Cite the two together, and
never the line alone.

Parameter lines, printed by `CountingGPTModelBuilder` on every rank:

```
stock-megatron stage <i>/<pp> local size: <n> parameters
Model qwen3_piper_<size> stock-megatron size: <N> total parameters
```

`<N>` uses a thousands separator, because arm rule 11 builds its target with
`f"{shape.param_count:,}"`.

### 8.5 The validation profile

```python
"megatron_stock": ValidationProfile(
    completion_marker="Training completed",
    mode_line=lambda mode: f"Megatron-LM stock training loop (mode={mode},",
    compiled_marker=None,
    failure_markers=(),
    check_ac_line=False,
    check_regions=False,
    parallelism_markers=_megatron_stock_parallelism_markers,
    pipelined_pattern=re.compile(
        r"Megatron-LM stock parallelism: dp=\d+ pp=(?!1\b)\d+"
    ),
    data_parallel_pattern=re.compile(
        r"Megatron-LM stock parallelism: dp=(?!1\b)\d+"
        r"|Megatron-LM stock data parallel:"
    ),
),
```

### 8.6 The scenario

```python
PIPER_MEGATRON_STOCK = Scenario(
    name="piper_megatron_stock",
    description=(
        "Piper-inspired stock Megatron-LM against stock TorchTitan on one "
        "c4_test stream. The Megatron arm keeps fp32 master weights, an "
        "fp32 gradient reduction and the unfused native cross entropy, so "
        "it is not plain bf16 and its number is a configured-engine number."
    ),
    workload=PIPER_1B_MEGATRON_WORKLOAD,
    regions=(),
    supported_ac_modes=("none",),
    supported_compile_modes=("default",),
    arms=(
        Arm(
            name="baseline",
            description=(
                "stock megatron.training.pretrain through pretrain_gpt's own "
                "providers: alltoall dispatcher, grouped GEMM, no aux router "
                "loss, no cross-entropy fusion, no permute fusion, no "
                "distributed optimizer, --init-method-std 0.01, fp32 master "
                "weights"
            ),
            launcher="megatron_stock",
            validation="megatron_stock",
            trace_kernel_markers=(
                "cudnn_generated_fort_native_sdpa",
                "_mul_silu_split",
            ),
        ),
        Arm(
            name="titan_stock",
            description=(
                "TorchTitan qwen3_piper_1b on the pre-tokenized replay "
                "stream, identical to the arm of the same name in "
                "piper1b_megatron"
            ),
        ),
    ),
)
```

The scenario name carries no size, because `--model-size` is a run axis.
The workload is reused, so the seed and the replay loader come from one
place.

**The new arm does not join `piper1b_megatron`.** `--resume` compares the
selected arm names, so adding an arm there would refuse a resume of every
`piper1b_megatron` directory already on disk.

**The two trace markers are expected, not measured.** Both come from the
tuned arm, which shares the attention backend and the fused SwiGLU. Confirm
them on the first run. If `_mul_silu_split` is absent, read the trace before
you change the declaration.

## 9. Lifting PP 4 and world size 8

An audit of `benchmarks/`, `tools/` and `tests/` found **one** cap.

### 9.1 The cap

- `benchmarks/e2e/parallelism.py:109` -- `MAX_WORLD_SIZE = 4`
- `benchmarks/e2e/parallelism.py:110` -- `MAX_PP = 2`

Set them to 8 and 4. Rewrite the comment block at `:90-108`: it currently
says the two engines agree only at `pp <= 2`, and that is no longer the
reason, because `launch.py:68-106` already sends the two `less-layers 0`
flags that make them agree at any degree.

### 9.2 Everything else is degree-generic, verified

| module | why it is already correct |
|---|---|
| `benchmarks/execution/devices.py:61` | `^\d+(,\d+)*$` accepts `0,1,2,3,4,5,6,7`. No length bound. |
| `benchmarks/execution/environment.py:80-86, :116` | `NGPU` and `LOG_RANK` both derive from `world_size`. |
| `benchmarks/execution/provenance.py:153-156` | `_device_names` compares against `gpu.count(",") + 1`, which is 8. |
| `benchmarks/e2e/launch.py:273-291` | `--nproc-per-node` and the rank filter derive from `spec.world_size`. |
| `benchmarks/e2e/megatron/train.py:97, :214-219, :677-681` | One schedule, a generic mesh check, and a broadcast source taken from `get_pipeline_model_parallel_last_rank()`. |
| `benchmarks/e2e/validation.py:179-278` | Both marker builders interpolate the spec. |
| `benchmarks/e2e/results.py:331-372` | `loss_visible_rank` is `(world_size // pp) * (pp - 1)`, which is rank 6 at world 8 and pp 4. |
| `benchmarks/artifacts/layout.py:79, :94-110` | The rank regex is multi-digit. |
| `benchmarks/models/piper_qwen3/shape.py:366-415` | `stage_param_count` refuses an uneven split and is otherwise generic. |
| `benchmarks/models/piper_qwen3/parallelize.py:97-122` | The refusals are per axis. A pipeline rank passes at any degree. |

### 9.3 Spec rule 12 is a second gate on pp 4

`benchmarks/e2e/parallelism.py:659` demands
`microbatches >= 2 * pp * stages_per_rank`. At pp 4 with 1F1B that is 8. The
`Workload` default `local_batch_size` is 4
(`benchmarks/e2e/registry.py:175`), so **a bare `--pp 4` fails even after the
cap lifts.** Section 10.1 gives the two batch settings that pass.

### 9.4 Eight GPUs run unpinned on this box

`benchmarks/execution/affinity.py:121-128` returns an empty `numactl` prefix
whenever the requested devices span NUMA nodes. Measured on this box:

```
GPU 0,1,2,3 -> NUMA node 0
GPU 4,5,6,7 -> NUMA node 1
```

So every cell of this suite runs unpinned, and `cpu_pinning` records
`none: devices 0,1,2,3,4,5,6,7 span NUMA nodes 0,0,0,0,1,1,1,1`.

**This is the largest measurement caveat of the suite.** `CLAUDE.md` states
that the training step is host-bound at benchmark sizes, so an unpinned run
measures scheduler placement as well as kernels. Both arms pay it, so the
cross-engine ratio stays usable, but no figure here compares to a pinned
single-GPU figure. `--resume` also refuses to mix pinned and unpinned runs.

Do not build a per-rank NUMA shim for this work. It is deferred, and an
unpinned run with a recorded reason is honest.

### 9.5 Arm rule 6 loses coverage at pp 4

`benchmarks/e2e/validation.py:600-604` reads the trace markers over every
rank's files flattened, so **one** stage satisfies a marker for the whole
arm. At pp 2 that is one of two. At pp 4 it is one of four.

So a silent TransformerEngine fallback to unfused attention on three of four
stages passes arm rule 6.

**Do not widen or tighten the rule.** Instead: before you publish any pp 4
number that rests on a marker, open each rank's trace by hand, check the
marker, and say in the report that you did.

### 9.6 Tests that fail as written

- `tests/test_parallelism.py:447-449` `test_the_largest_budgeted_mesh_passes`
  builds `dp=2, pp=MAX_PP` at the default batch 4. At `MAX_PP=4` spec rule 12
  raises. Give the call `local_batch_size=8`.
- `tests/test_parallelism.py:455-459`
  `test_a_pipeline_deeper_than_two_is_refused` expects the message
  `"pipeline degree"`. At `MAX_PP=4` a `pp=4` spec no longer raises that.
  Rewrite it as "a pipeline deeper than the maximum is refused" with
  `pp=MAX_PP * 2`.
- `tests/test_migration_contract.py:1725` and `:1780` pin
  `"test_parallelism": 85` and `TEST_CENSUS_TOTAL = 1481`.

### 9.7 `tools/run_matrix.sh` cannot drive this suite

`tools/run_matrix.sh:85` requires one PCI index, and `:291` passes no
`--dp` or `--pp`. Its idle check and its watchdog query one device.

**Decision: do not extend it in this work.** Run the four cells by hand, one
at a time, and check `nvidia-smi` before each. Record that the matrix
supervisor did not guard these cells, so no cell carries a `CONTAMINATED`
verdict either way.

### 9.8 `titan_stock` at pp 4

Nothing changes for it beyond the degree. `launch.py:68-106` already sends
`--parallelism.pipeline-parallel-degree 4` and the two `less-layers 0` flags,
and `third_party/torchtitan/torchtitan/distributed/pipeline_parallel.py:215-216`
reads those two weights. With both at 0 the divisor is `num_layers`, so 16
layers split `[4, 4, 4, 4]` and 24 layers split `[6, 6, 6, 6]`.

With TorchTitan's own defaults of 1 and 1, the divisor would be
`num_layers + 2`, and 16 layers would split `[4, 5, 4, 3]` against Megatron's
`[4, 4, 4, 4]`. The flags are what stop that. A test must pin them.

`parallelize_piper1b` lets a pipeline rank through at any degree
(`benchmarks/models/piper_qwen3/parallelize.py:97-122`), and it applies
`fully_shard` at `dp 2` with `dp_replicate=2, dp_shard=1`.

## 10. Arithmetic

### 10.1 Spec rule 12 at pp 4

`n_microbatches = local_batch_size // pp_microbatch_size`
(`benchmarks/e2e/parallelism.py:330-337`).
`total_stages = pp * stages_per_rank`, and 1F1B has `stages_per_rank = 1`, so
`total_stages = 4`.

| batch | microbatch | microbatches | spec rule 10 | spec rule 11 | spec rule 12 |
|---|---|---|---|---|---|
| 32 | 4 | 8 | `32 % 4 == 0` | `8 % 4 == 0` | `8 >= 8` |
| 8 | 1 | 8 | `8 % 1 == 0` | `8 % 4 == 0` | `8 >= 8` |

Both pass. Both sit **exactly** on the spec rule 12 boundary, so any lower
batch or any larger microbatch fails.

The Megatron global batch is `local_batch_size * dp`: 64 and 16.
`num_microbatches` is then `global / (micro * dp)`, which is 8 in both cases.
The two engines therefore run the same number of microbatches per step.

### 10.2 The bubble

`bubble = (pp - 1) / (n_microbatches + pp - 1) = 3 / 11 = 27.3%`.
Both engines pay it. A pp 4 number never compares to a single-GPU number.

### 10.3 Stage parameters and memory

| shape | pp 4 stage counts | max stage |
|---|---|---|
| `1b` | 344,351,232 / 188,768,768 / 188,768,768 / 344,352,256 | 344,352,256 |
| `9b` | 2,488,132,352 / 2,176,967,424 / 2,176,967,424 / 2,488,134,400 | 2,488,134,400 |

Each row sums to the shape's `param_count`.

State memory, from the byte counts of section 6.1, on the busiest stage:

| shape | titan at 8 B/param | stock megatron at 18 B/param |
|---|---|---|
| `1b` | 2.57 GiB | 5.77 GiB |
| `9b` | 18.54 GiB | 41.7 GiB |

The device holds 139.81 GiB. Both fit with room for activations. **The
activation term is not in this table**, and 1F1B holds `pp` microbatches on
rank 0, so `9b` at batch 32 is the cell most likely to run out of memory. If
it does, fall back to batch 8 and microbatch 1 and treat it as a different
workload.

## 11. The run matrix and the exact commands

Approved on 2026-08-26. Eight GPUs, `0,1,2,3,4,5,6,7`.

| cell | shape | arms | compile mode | batch / microbatch |
|---|---|---|---|---|
| 1 | `1b` | `baseline`, `titan_stock` | `default` | 32 / 4 |
| 2 | `1b` | `titan_stock` | `none` | 32 / 4 |
| 3 | `9b` | `baseline`, `titan_stock` | `default` | 32 / 4 |
| 4 | `9b` | `titan_stock` | `none` | 32 / 4 |

Cell 1, the paired comparison. `run-all` runs every arm and evaluates:

```bash
./run_bench.sh run-all 0,1,2,3,4,5,6,7 \
  --scenario piper_megatron_stock \
  --model-size 1b \
  --dp 2 --pp 4 --pp-schedule 1F1B --pp-microbatch-size 4 \
  --batch 32 --steps 40 --seq-len 1024 \
  --compile-mode default --ac none
```

Cell 2, the eager TorchTitan reference. It selects one arm, so it uses `run`
and a separate `evaluate`:

```bash
./run_bench.sh run 0,1,2,3,4,5,6,7 \
  --scenario piper_megatron_stock --arm titan_stock \
  --model-size 1b \
  --dp 2 --pp 4 --pp-schedule 1F1B --pp-microbatch-size 4 \
  --batch 32 --steps 40 --seq-len 1024 \
  --compile-mode none --ac none

./run_bench.sh evaluate out/<timestamp>/piper_megatron_stock/nvidia-h200
```

Cells 3 and 4 repeat cells 1 and 2 with `--model-size 9b`.

`--compile-mode none` is legal in cell 2 and illegal in cell 1. The scenario
declares `supported_compile_modes=("default",)`, and
`benchmarks/e2e/runner.py:288-302` admits an uncompiled run only when every
**selected** arm is TorchTitan.

Cell 2's `results.json` carries no ratio against a baseline, because no
baseline arm ran. The report may place it beside cell 1 as a labelled
cross-subrun reference. **It must never carry a fabricated same-run ratio.**

If a `9b` cell runs out of memory, rerun it at `--batch 8
--pp-microbatch-size 1`, and report it as a different workload. Never
compare its raw throughput against a batch-32 row without saying so.

## 12. Test obligations

Run the whole suite at every commit:

```bash
HF_DATASETS_CACHE="$HOME/.cache/hf-datasets" \
  .venv/bin/python -m unittest discover -s tests
```

**Export `HF_DATASETS_CACHE`.** `HF_HOME` on this box points at a directory
another user owns, and three tests in `tests/test_megatron_data.py` skip
without a writable cache. `run_bench.sh` exports it; a bare `unittest` run
does not. Those three tests hold the per-rank slice, the global padding
target and the stream parity that this scenario's data contract rests on.

Baseline at HEAD `5e6474e`: **1568 tests, 11 skipped, about 52 seconds.**

`TEST_CENSUS` in `tests/test_migration_contract.py` pins a count per censused
module, and `TEST_CENSUS_TOTAL` must equal `sum(TEST_CENSUS.values())`.
**Recompute the total as the sum of the dict. Never add a delta.** A new test
module does not need a census entry; the census is deliberately partial.

### New test modules

`tests/test_megatron_stock_driver.py`, CPU only, imports no torch:

- `stock_megatron_flags` produces every flag of section 7 for `1b` and `9b`.
- It produces no omitted flag: assert the absence of
  `--use-distributed-optimizer`, `--overlap-grad-reduce`,
  `--overlap-param-gather`, `--data-parallel-sharding-strategy`,
  `--use-megatron-fsdp`, `--moe-permute-fusion`,
  `--cross-entropy-loss-fusion`, `--mock-data` and `--tensorboard-dir`.
- `--global-batch-size` equals `local_batch_size * dp`, and
  `--micro-batch-size` equals `pp_microbatch_size`.
- The geometry flags equal the `PiperShape` fields, for every registered
  shape.
- `--rotary-base` is an integer string.
- The microbatch dicts hold the eight keys of section 6.3, with the right
  dtypes and shapes.
- Each `cu_seqlens` row starts at 0, ends at `seq_len`, and pads with copies
  of `seq_len`.
- The iterator raises on exhaustion rather than wrapping.
- The driver refuses a mode other than `default`, a schedule other than
  `1F1B`, and a virtual pipeline degree.
- `install_typing_override` is idempotent, and it raises when
  `typing_extensions` lacks `override`.

`tests/test_megatron_stock_launch.py`, CPU only:

- `command_for_arm` on the stock arm produces the whole argv, frozen against
  a golden list, at `dp 2 x pp 4` and at the trivial spec.
- The argv begins with `python -m torch.distributed.run
  --nproc-per-node=8 ... -m benchmarks.e2e.megatron_stock.train`.
- The `titan_stock` argv in this scenario carries
  `--parallelism.pipeline-parallel-first-stage-less-layers 0` and its last
  twin. **This test must break if an upstream default changes.**
- The `megatron_stock` validation profile's marker strings equal the strings
  the driver module declares. Import both and compare.
- `_megatron_stock_parallelism_markers` returns a non-empty tuple above world
  size 1 and interpolates the spec.
- The scenario declines `--ac sac` and declines `--compile-mode cuda-graph`.
- A run selecting `baseline` at `--compile-mode none` is refused.
- A run selecting only `titan_stock` at `--compile-mode none` is accepted.

### Additions to `tests/test_parallelism.py`

- The two repairs of section 9.6.
- `dp 2 x pp 4` at world 8 passes for `1b` and for `9b`, at batch 32 with
  microbatch 4, and at batch 8 with microbatch 1.
- The same mesh at batch 4 is refused by spec rule 12, and the message names
  8 microbatches.
- `pp 4` at `huge` is refused by spec rule 7.
- `execution_model` at `dp 2, pp 4` reads `8-gpu-plain-bf16-dp2-pp4-1F1B`.
- The trivial spec still reads `single-gpu-plain-bf16-no-fsdp`.

## 13. Hazards, one per step

Give each reviewer exactly one hazard. A general request produces a general
answer.

| step | the hazard the reviewer must hunt |
|---|---|
| 1. Lift the caps | A test that now passes for a different reason than it used to. |
| 2. `bootstrap.py` | A shim that changes behaviour on a Python 3.12 interpreter, or that hides a missing `typing_extensions`. |
| 3. `flags.py` | A flag that does not exist in this Megatron rev, or a geometry value that does not come from `PiperShape`. |
| 4. `data.py` | Two ranks of one pipeline that read different tokens, or a `cu_seqlens` row that Megatron's merge misreads. |
| 5. `model_builder.py` | A printed parameter total that is not backed by a real count on that rank. |
| 6. `profiling.py` | A run that writes one window and still passes, or a shim that silently did not install. |
| 7. `train.py` | A marker string that differs by one character from the validation profile's. |
| 8. `launch.py` | A `--parallelism.*` or Megatron token in the argv at the trivial spec. |
| 9. `validation.py` | A profile that proves nothing about the mesh, or a pattern that matches `dp=1`. |
| 10. `registry.py` | A scenario that admits a compile mode or an AC mode its Megatron arm cannot honour. |
| every | A validation rule relaxed to make a test pass. |

The last row is this repository's own rule. Never work around a validation
rule.

## 14. Parallelization

**Verdict: three agents in one wave, then one integrator. The wave is real,
not forced.**

The three sets of files are disjoint, and this plan fixes every string they
must agree on, so no agent has to negotiate with another.

### Wave A, three agents, each in its own git worktree

| agent | owns | must not touch |
|---|---|---|
| **P** (caps) | `benchmarks/e2e/parallelism.py`, `tests/test_parallelism.py`, `tests/test_migration_contract.py` | anything else |
| **D** (driver) | the whole new `benchmarks/e2e/megatron_stock/` package, new `tests/test_megatron_stock_driver.py` | `tests/test_migration_contract.py` |
| **W** (wiring) | `benchmarks/e2e/launch.py`, `benchmarks/e2e/validation.py`, `benchmarks/e2e/registry.py`, new `tests/test_megatron_stock_launch.py` | `tests/test_migration_contract.py` |

`tests/test_migration_contract.py` holds both the census and the golden
command lines. **Only P edits it.** W puts its golden argv test in its own
new module for that reason.

The contract between D and W is fixed by this plan and by nothing else:

- the module path `benchmarks.e2e.megatron_stock.train`;
- the harness flag names of section 7;
- the four marker strings of section 8.4;
- `stock_megatron_flags`'s signature, in `benchmarks/e2e/megatron_stock/flags.py`.

W imports `flags.py`, which D writes. So **D must land `flags.py` first**, as
its own commit, before W's own work merges. Either D pushes that one file
early, or W writes against the signature in this plan and the integrator
resolves the single import. Prefer the first.

W's tests will fail until D's package exists. That is expected. W runs the
rest of the suite and states which of its tests are pending.

### Wave B, one integrator, sequential

1. Merge P, then D, then W.
2. Recompute `TEST_CENSUS_TOTAL` as the sum of the dict.
3. Run the whole suite. Record the test and skip counts.
4. Update `CLAUDE.md`.
5. Run the four GPU cells of section 11, one at a time.

### Each agent's loop

1. Implement.
2. Run the suite.
3. Spawn a reviewer. Give it the diff, its one hazard from section 13, and
   the rule that no validation check may be relaxed.
4. Fix what the reviewer confirms. Run the suite again.
5. Stop after two rounds.
6. Check every factual claim in the commit message against the code, then
   commit.

Commit in single, self-contained steps. Do not commit to `master`. Do not
push. Do not open a pull request. Do not edit `third_party/`. Do not edit
`BENCHMARK_IMPROVEMENT_SPEC.md`.

## 15. Decisions that are cheap to reverse

Each row names the one place to edit.

| decision | reverse it here |
|---|---|
| No cross-entropy fusion | Add `--cross-entropy-loss-fusion` to the fusion block in `flags.py`. Native fused CE then runs; `te` stays refused by `arguments.py:1630`. |
| No permutation fusion | Add `--moe-permute-fusion` in `flags.py`, and add `_permute_kernel` to the arm's `trace_kernel_markers`. |
| No attention-backend flag | Add `--use-flash-attn` in `flags.py`, and replace the `cudnn_generated_fort_native_sdpa` marker with the FlashAttention name the first trace shows. |
| fp32 master weights and fp32 gradient reduction | Add `--grad-reduce-in-bf16` for the reduction alone (`arguments.py:1179`), or `--use-precision-aware-optimizer` with the four dtype flags for the whole optimizer. Both leave "stock". |
| `--dataloader-type external` | Change the one flag to `single` in `flags.py`, and return a map-style `Dataset` over the **global** stream from `data.py`'s provider. Megatron's own sampler then shards it, and the two engines stop agreeing rank for rank. |
| Packed-document masking | Remove `--dataloader-inter-document-masking` from `flags.py`, and drop the three THD keys from `data.py`. Megatron then runs a plain causal mask and does more attention work than titan. |
| The world-size and pipeline caps | `MAX_WORLD_SIZE` and `MAX_PP`, `benchmarks/e2e/parallelism.py:109-110`. |
| Batch 32 and microbatch 4 | The `--batch` and `--pp-microbatch-size` values on the command line. They are run options, not registry values. |
| A new scenario rather than a third arm on `piper1b_megatron` | Move the `Arm` into `PIPER_1B_MEGATRON.arms` and delete the scenario. This breaks `--resume` for every existing `piper1b_megatron` directory. |
| The scenario name `piper_megatron_stock` | One `name=` in `benchmarks/e2e/registry.py`. Change it before the first run; afterwards it names directories under `out/`. |
| No `tools/run_matrix.sh` support | Section 9.7. Extending it is separate work. |

## 16. Unverified, and what would settle each

Nothing below is claimed as true. Each is a thing to check on the first run.

| item | how to settle it |
|---|---|
| `_mul_silu_split` appears in a stock-arm trace. | Read one rank's trace. If it is absent, read why before you change the declaration. |
| `cudnn_generated_fort_native_sdpa` appears in a stock-arm trace. | The same. TransformerEngine chooses the backend at run time. |
| Megatron's `--lr-decay-iters 40` decays over 38 post-warmup steps. | Read `OptimizerParamScheduler`. |
| The `9b` shape fits at batch 32 on this box. | Run cell 3. Fall back to batch 8 and microbatch 1. |
| The merged `cu_seqlens` length varies per step without a TransformerEngine cost. | Compare the per-step host time of cells 1 and 2. |
| Both engines map rank to pipeline stage the same way. | Read one log per engine. Neither engine's mapping is set by the harness, and the benchmark does not need them to agree. |
| Arm rule 13's marker survives Megatron's DDP bucketing at world 8. | Read every rank's trace. A grouped launch can surface as `ncclDevKernel_Generic`. If it does, read the trace before you widen the marker, and never widen it to a bare `nccl`. |
| Megatron's `9b` parity gate of 2e-2. | `tools/megatron_parity_check.py --model-size 9b`. |
| The pp 4 noise floor. | Run one cell twice, one hour apart, on an idle box. |

## 17. Required report contents

- The environment, every recorded revision, and the exact commands of
  section 11.
- The four deliberate differences of section 2, stated before any number.
- `cpu_pinning`, and the statement that every cell ran unpinned.
- The layer split each engine produced, read from its own log.
- The full Megatron flag table: enabled, deliberately omitted, and
  unavailable.
- Evidence that arm rule 6 was checked by hand on every rank.
- Per-rank throughput, the published minimum, the global figure, and the
  1.15x spread check.
- The paired ratio from cell 1 and cell 3 only. The eager TorchTitan rows are
  labelled cross-subrun references and carry no ratio.
- The memory result, and any fallback to batch 8.
- Conclusions limited to what the measurements support.
