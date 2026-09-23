# Piper's Megatron baseline loses ZeRO-1 when ep equals dp

Read `SKILL.md` first. This file holds the mechanism and the evidence.

**The paper's TorchTitan version string does not identify the code they
ran.** The paper says TorchTitan 0.2.2. `artifact/Dockerfile:7` pins
`TORCHTITAN_COMMIT=b01adfb544b4331ecab090ebdb50b2296cd8eb6a`, dated
2026-03-30, which is **140 commits past** the `v0.2.2` tag (`73a0e697`,
2026-02-20). The layouts differ, so the difference is not cosmetic: the
tag holds the reshard policy in `torchtitan/models/llama3/infra/
parallelize.py` and has no `torchtitan/distributed/fsdp.py`, where the
pinned commit has that file and reads the policy from it. Every
TorchTitan claim in this section is read at the **pinned commit**, which
is in this repository's own object store. Anyone reproducing from the
paper's version string alone reads different code.

## The chain

1. `artifact/e2e_eval.py:391` computes
   `dp_megatron = exp.dp // exp.ep if exp.ep > 1 else exp.dp`. At
   `dp 4, ep 4` that is **1**.
2. `artifact/e2e_eval.py:401` passes it as `--dp 1`.
3. `artifact/run_megatron.py:280` reads it: `if dp > 1:`. This is the
   **only** read of `dp` in that program, and the sole guard on
   `--use-distributed-optimizer`, `--overlap-grad-reduce`,
   `--overlap-param-gather` and `--data-parallel-sharding-strategy`.
4. The block is skipped. Megatron derives its real data-parallel degree
   from the world size and gets **4**, so the dense parameters are
   replicated 4 times with full fp32 Adam state on each rank.

**The value is correct and the question is wrong.** `dp // ep` is the
number of copies of each expert, which Megatron calls
`expert_data_parallel_size`. The gate decides whether to shard the **dense**
optimizer state, and the right quantity there is `exp.dp`, undivided.

## Why the division is there

`artifact/scripts/run-qwen-megatron.sh:40` asserts
`tp * pp * dp * cp * ep == nnode * ngpu`. At `pp 8, dp 4, ep 4` on 32 GPUs
that product is 128. The assert fails. `dp // ep` makes it pass, and kills
the gate as a side effect. Megatron is the only system with that assert,
and the only one whose `dp` is divided: DeepSpeed (`:428`) and Piper
(`:443`) both take `exp.dp` undivided, and both receive `--zero-stage`
with no gate at all.

## The evidence

Megatron's own argument dump, for the paper's Figure 7a config
(Qwen3 1B, 32 GPUs, PP-8 x DP/EP-4):

```
... --pp 8 --dp 1 --cp 1 --ep 4 ... --zero-level zero1
  data_parallel_size .............................. 4
  expert_model_parallel_size ...................... 4
  use_distributed_optimizer ....................... False
  data_parallel_sharding_strategy ................. optim_grads_params
```

`optim_grads_params` is Megatron's default, not the `optim` that `zero1`
maps to, so the block never ran. Across 52 dumps in that archive, 37 read
`True` and 15 read `False`; the 15 are exactly the `ep > 1` runs plus the
genuine one-rank cases. Six hours earlier the same config ran with an
undivided `--dp 4` and reported `use_distributed_optimizer True`; it did
not complete, and every later expert run passes `--dp 1`.

## Which experiments this reaches

Only the `schedule` sweep sets `ep > 1` (`e2e_eval.py:174`), and it sets
`ep = dp`. The paper's Figure 7 is that sweep: Qwen3 1B at `pp 8 x dp/ep 4`
on 32 GPUs, and Qwen3 9B at `pp 4 x dp/ep 4` on 16 GPUs. Sections 6.1 and
6.3 both read it, so it is the paper's main comparison against Megatron.
Table 2 (`ep 1`) is unaffected and its Megatron arm really does run ZeRO-1.

**The `scalability` sweep produces no Megatron result at all.** It sets
`zero_level="zero0"` (`:152`), which `run_megatron.py`'s argparse rejects.

## What TorchTitan does in the same runs, and why

**TorchTitan moves the opposite way on the same switch, and it has no
choice.** Two meshes carry the two regions, and `apply_fsdp`
(`torchtitan/models/llama4/parallelize.py` at Piper's pinned commit) hands
each parameter to one of them through `shard_placement_fn`:

| region | mesh | degree at `tp=cp=etp=1` | placement |
|---|---|---|---|
| routed experts | `edp_mesh` (`efsdp`, plus `dp_replicate`) | `dp_shard // ep` | `Shard(0)` or `Shard(1)` |
| everything else | `dp_mesh` (`fsdp`, plus `dp_replicate`) | `dp_shard` | `Shard(0)` |

**`dp_shard_degree` does not shard the experts, and saying it "shards
everything" is wrong.** It sizes the `fsdp` mesh, which shards the dense
region. The experts are split by the `ep` mesh and then FSDP-sharded again
over `efsdp = dp_shard * cp * tp // (etp * ep)`. At Piper's configuration
that second factor is 1, so each region is split `dp_shard` ways in total
by two different mechanisms over two different meshes. Expert parallelism
is not an extra factor on the world size either: the assert at
`parallel_dims.py:53` is `dp_replicate * dp_shard * cp * tp * pp ==
world_size`, with `ep` absent.

**An expert degree therefore forces the dense region to be sharded.**
`efsdp` must be at least 1, so `dp_shard * cp * tp >= etp * ep`; at
`tp=cp=etp=1` that is `dp_shard >= ep > 1`, which makes `fsdp_enabled`
true and takes `apply_fsdp` rather than `apply_replicate`. The replicate
branch exists (`qwen3/parallelize.py`, the `elif
parallel_dims.dp_replicate_enabled` arm) and is simply unreachable under
`ep > 1`. So a titan arm cannot split experts while keeping the dense
parameters replicated -- not by configuration, and not by a flag we have
declined to set.

**What follows is the fair-comparison rule, and it is the opposite of
replicated parity.** Under an expert degree TorchTitan shards the dense
region's parameters, gradients and optimizer state. A Megatron arm beside
it must therefore shard too -- ZeRO-2 (`optim_grads`) or ZeRO-3
(`optim_grads_params`) under `--use-megatron-fsdp` -- or the row compares
two memory strategies rather than two engines. **Replicated-on-both is the
right parity only while `ep` is 1.** This repository's `piper_megatron_stock`
scenario is built on replicated parity and its
`benchmarks/models/piper_qwen3/parallelize.py` refuses `dp_shard > 1`, so
an expert cell there needs a decision about which parity it measures before
it needs any code.

**Piper's own harness cannot reach that fair comparison.**
`zero_levels_by_system` (`e2e_eval.py:163-168`) allows Megatron `("zero1",)`
alone, and `run_megatron.py:292-293` appends `--use-megatron-fsdp` only for
`zero2`/`zero3`, so that branch is dead for Megatron. Their Megatron arm can
reach ZeRO-1 at best, and under `ep == dp` the gate above denies it even
that. Note also that `--data-parallel-sharding-strategy` is read **only** by
the Megatron-FSDP path (`megatron/core/distributed/fsdp/`); with the classic
DDP plus distributed optimizer it is inert, so the `optim_grads_params` in
their argument dump is a default string that changes nothing.

**What pins Piper to `ep == dp` is their own line, not TorchTitan.**
`backend_layout` (`e2e_eval.py:332-336`) makes `world = pp * dp` on both
backends, and TorchTitan asserts
`dp_replicate * dp_shard * cp * tp * pp == world_size`, so with
`dp_replicate = 1` the shard degree has to be `dp`. `e2e_eval.py:348`
writes `dp_shard_degree = exp.ep` instead. The two are equal in every
configuration Piper runs, so it never fails -- but any `ep < dp` would die
at mesh construction, and `exp.dp` would have been correct in general.
TorchTitan itself accepts `ep < dp_shard`. **That line is what makes
`ep == dp` inevitable, which is the exact condition that disables the
Megatron gate above.** The two defects are linked.

The archive proves the flip. Two runs of the same sweep, same `dp 4`,
both labelled `zero1`, neither passing
`--parallelism.fsdp_reshard_after_forward`:

| | `data_parallel_replicate_degree` | `data_parallel_shard_degree` | effect |
|---|---|---|---|
| `ep 1` | 4 | 1 | pure replication -- **nothing sharded** |
| `ep 4` | 1 | 4 | the dense region's parameters, gradients and optimizer state sharded 4 ways |

So one flag moves **TorchTitan up** and **Megatron down** from the same
label. Figure 7 puts three treatments under one `zero1`: Megatron shards
nothing, Piper and DeepSpeed shard the optimizer state, and TorchTitan
shards parameters, gradients and optimizer state. That is ZeRO-0 against
ZeRO-1 against ZeRO-3, under one label, in one figure.

**Do not put one ZeRO number on that arm. Two facts hold at once, and
they point at different levels.** What is *sharded* is ZeRO-3: FSDP2 holds
the dense parameters, their gradients and the optimizer state split
`dp_shard` ways. What is *resident during a step* is closer to ZeRO-2,
because Piper sends no `--parallelism.fsdp_reshard_after_forward` for an
expert run (`e2e_eval.py:366-369` fires for `zero2`/`zero3` only), the
policy is therefore `"default"`, and TorchTitan resolves that to
`not pp_enabled` (`torchtitan/distributed/fsdp.py` at their pinned commit).
Every one of these runs carries `pp 8` or `pp 4`, so
`reshard_after_forward` is **False**: the parameters are all-gathered
before a block's forward and **kept full** through backward and across
every microbatch. The comment there states the reason -- a per-microbatch
all-gather does not overlap.

So the paper's own complaint that TorchTitan "maintains a full copy of
parameters and gradients across multiple microbatches" describes the
documented default in Figure 7, whatever it describes in Figure 8. Say
"parameters, gradients and optimizer state sharded, gathered parameters
held through the step" and give no ZeRO number at all.

**TorchTitan is never at ZeRO-1 anywhere in that suite.** At `ep 1`,
`zero1` maps to `dp_replicate = dp, dp_shard = 1`, which shards nothing;
TorchTitan has no optimizer-only mode to map to. That reaches Table 2 as
well, which is titled "DP ZeRO-1 throughput on all systems".

**Provenance of the TorchTitan bars is settled.** The run directory
`out/e2e-eval/20260422_164251` under the archive reproduces both of the
paper's own percentages for Figure 7a: interleaved 1F1B at 2.971 s
against 1F1B at 2.614 s is 13.6% worse ("14% worse than its own 1F1B"),
and DualPipeV at 2.535 s is 3.0% better ("improves only 3%"). Every one
of those cells carries `--compile.no-enable`.

