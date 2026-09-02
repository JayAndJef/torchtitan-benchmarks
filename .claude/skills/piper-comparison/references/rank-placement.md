# Piper puts the two engines on different rank layouts

Read `SKILL.md` first. This file holds the second asymmetry, and it is the
one that decides every multi-node number in the archive. It was diagnosed
on 2026-08-28 and verified by four independent audits on 2026-09-02
(`reports/20260902-multinode-verification/REPORT.md`).

**Piper's harness puts one data-parallel replica on each node. Piper's own
system and its patched TorchTitan obey that layout. Megatron was left on its
own default, which is the opposite. The two sides therefore put different
axes across the machine boundary, and their multi-node rows measure that
difference, not the two engines.**

## What the patch does

Commit `c330f40a` ("training scripts", 2026-04-16) in
`/m-coriander/coriander/mfris/torchtitan` changes
`torchtitan/distributed/parallel_dims.py`. The commit touches 16 files and
886 lines; the mesh change is 18 of them, and the message does not mention
it. It is the only commit that ever touched that file in the checkout, so
every April run is patched.

```
upstream:  ("pp", "dp_replicate", "fsdp", "tp")
Piper:     ("dp_replicate", "fsdp", "pp", "tp")
```

`DeviceMesh._unflatten` makes the **last** name vary fastest, so the leftmost
name carries the largest stride. The same swap is applied to the dataloading
mesh and to the sparse mesh.

## Piper's own system is on the same layout, by its own code

Piper does not use `ParallelDims` at all. `src/actor.py:34-35` computes
`rank = pp_rank + dp_rank * pp_degree`, and `src/coordinator.py:91-105`
places one Ray bundle of `pp` GPUs per data-parallel replica. Its
multi-node log shows `Actor 0..7` on one host and `Actor 8..15` on the
next. So the patch did not single TorchTitan out. It moved TorchTitan onto
Piper's own layout. Three of the four arms share it; Megatron alone does
not. **Say "two rank layouts", never "a handicap aimed at TorchTitan".**

## The two rank equations

At `pp 8, dp 2` on 16 GPUs, with 8 GPUs in a node:

| arm | rank equation | a node holds | data-parallel group | pipeline group |
|---|---|---|---|---|
| upstream TorchTitan | `dp + 2*pp` | stages 0-3 of both replicas | **inside one node** | crosses nodes |
| Piper's TorchTitan | `pp + 8*dp` | a **whole** 8-stage pipeline | **crosses nodes** | inside one node |
| Piper's own system | `pp + 8*dp` | a **whole** 8-stage pipeline | **crosses nodes** | inside one node |
| Megatron, default | `dp + 2*pp` | stages 0-3 of both replicas | **inside one node** | crosses nodes |

Megatron's default order is `tp-cp-ep-dp-pp`
(`megatron/core/parallel_state.py:617`), and in Megatron's convention the
**first** name varies fastest. Both equations were run, not only read:
Piper's own `ParallelDims` under a fake process group gives data-parallel
groups `(0,8)...(7,15)`, upstream gives `(0,1)...(14,15)`, and Megatron's
`RankGenerator` keeps the group inside a node at dp 2 and at dp 4.

**Both placements were observed in the logs.** In
`.../20260421_052821/torchtitan/logs/`, local rank *i* is pipeline stage *i*
on every node. In the Megatron log of the same run the `[rankN]` labels
come from `ProcessGroupNCCL` warnings, which print global ranks, and the
emitting set is 8 ranks of stride exactly `dp` at both dp 2 and dp 4. (The
parameter-count lines carry no rank prefix; an earlier version of this file
said they did.)

## Why it decides the result

The data-parallel axis moves gradients, which are gigabytes per step. The
pipeline axis moves activations, which are megabytes per step. Under the
patched layout each node holds a whole pipeline, so it all-reduces the
whole fp32 gradient, 37.32 GB at `qwen3_9b`, across the fabric every step.

| | per step | rate |
|---|---|---|
| TorchTitan all-reduce, one rank, 2 nodes | 2.46 s for 5.60 GB | 2.3 GB/s per rank |
| TorchTitan all-reduce, whole node, 2 nodes | 2.755 s union for 37.32 GB egress | **13.5 GB/s = 108 Gbps per node** |
| Megatron reduce-scatter, 4 nodes, four single-bucket ranks | 15.5 ms for 3.27 GB | **209 to 212 GB/s**, three routes agree |

209 GB/s exceeds the node's whole 400 Gbps link by 4x, so that transfer
cannot be crossing it. This is a physical argument and needs no assumption
about the fabric. Every one of the 16 TorchTitan ranks spends 63% to 70%
of its step inside a gradient collective.

**The control.** Same sweep (`csv/april/20260421_052821`), added seconds
per step over each system's own one-node cell:

| system | DP placement | added at 2 nodes | added at 4 nodes |
|---|---|---|---|
| Megatron | inside a node | +0.081 | +0.102 |
| TorchTitan | across nodes | +2.660 | +3.764 |
| Piper's own system | across nodes | +1.362 | +1.797 |

One link constant, 2.3 GB/s per rank, applied to each system's own volume
predicts all four cross-node deltas within 11%. About 0.45 to 0.49 s of
each cross-node delta is a fixed term that does not scale with the degree,
most likely the data-parallel wrapper that exists only above dp 1. It is
17% of TorchTitan's added cost.

**At one machine the layout is invisible**, because no data-parallel group
spans anything. That is the crossover: TorchTitan wins on one node on raw
compute (1.100 s against 1.498 s per step) and loses by about 3x on two.

## The harness knows about this and never acts on it

The April driver sets `node_count(exp) = exp.dp`
(`scripts/run_qwen_e2e_eval.py:314-315`) and launches every system with
`--nnode <dp> --ngpu <pp>`. Megatron is sent `--pp`, `--dp` and `--ep` and
**no rank-order flag**.

Both harnesses define the flag that would equalise the arms, and no caller
passes either:

```python
# run_megatron.py:151
parser.add_argument("--use-tp-pp-dp-mapping", action="store_true", default=False,
                    help="Use tp-cp-ep-pp-dp rank ordering (PP intra-node) "
                         "instead of default tp-cp-ep-dp-pp (PP cross-node).")
```

Piper's own `--pp-outer` (`src/coordinator.py:96-101`) is defaulted off at
`artifact/scripts/run_piper.sh:24`. The recorded Megatron argument dump
reads `use_tp_pp_dp_mapping .... False`.

## The fabric was EFA, and the shortfall is 3.5x, not 20x

An earlier version of this file called the inter-node rate a 20x shortfall
with an unknown cause. Both halves are corrected.

- **The transport was EFA.** The `nsys` report of the profiled 2-node run
  holds the process environment and the loaded-module table:
  `FI_PROVIDER=efa`, `FI_EFA_USE_DEVICE_RDMA=1`,
  `/opt/aws-ofi-nccl/lib/libnccl-net.so`, `libfabric.so.1.30.0`,
  `libefa.so.1.4.61.0` and `libgdrapi.so.2.5` loaded, four EFA devices
  listed. NCCL's socket transport loads none of those. Both engines ran
  NCCL at defaults with identical fabric settings.
- **The 20x divided a per-rank rate by a per-node link.** Eight ranks share
  the node's NICs. On the node basis the fabric delivered 108 to 119 Gbps
  of a nominal 400, a stable 28%, at both dp 2 and dp 4. Eight independent
  2-rank cross-node communicators delivering that fraction is ordinary.
- **Placement alone flips the winner, even at the nominal link.** At a
  perfect 400 Gbps the 37.32 GB costs 0.746 s, so TorchTitan would run
  1.846 s against Megatron's 1.579 s. Megatron still wins by 1.17x. The
  only counterfactual that removes the crossover moves the gradients off
  the fabric.
- **The `_LL` kernel-name suffix names no protocol.** NCCL's specialised
  kernel entry points are `RING_LL` and `TREE_LL` only. Piper's own run
  sets `NCCL_PROTO=simple` and still emits only `_RING_LL` kernels. The
  archive records no wire protocol at all.

## What may be said, and what may not

- **May**: their multi-node rows compare two rank layouts as well as two
  engines, so they do not support an engine claim.
- **May**: at one machine the placement is irrelevant, so their single-node
  rows are a fair engine comparison.
- **May**: Piper's own system pays the same placement cost, 17x Megatron's
  added cost at 2 nodes.
- **May not**: "Megatron beats TorchTitan across machines." Nothing in the
  archive separates the engine from the layout.
- **May not**: "a sixth handicap on their TorchTitan arm." Three of four
  arms share the layout.
- **May not**: any multi-node claim about `qwen3_1b`. **The archive holds
  no same-session multi-node `1b` TorchTitan-against-Megatron pair.**
  Every such comparison is `9b`.

**Do not treat `csv/april/20260420_055305` as data.** Its Megatron row is a
single warm-up iteration, 19x low. Three other single-warm-up rows exist
and are excluded the same way.

**The published TorchTitan multi-node figure is one arbitrary node's local
rank 0.** `run_qwen_e2e_eval.py:731-732` reads `iter_by_rank[0]` and the
log prefix is the local rank, so every node writes `[rank0]` and the last
one wins. The value survives, because every rank in that cell reports 3.7
to 3.8 s. The provenance does not.

## What would settle it

Add `--use-tp-pp-dp-mapping` to one Megatron `pp8 dp2` run. The flag exists
and is plumbed, so this needs no code change. Megatron's own fp32
reduce-scatter plus all-gather over stage 0's 1.4e9 parameters is about
4.2 GB per rank at d = 2, so its 1.579 s step should move to roughly 3.4 s.
**This needs two machines. This box has one.**

The full analysis is `reports/20260828-multinode-crossover/HYPOTHESES.md`
(13 hypotheses, each graded) with its corrections block, and the four
verification reports under `reports/20260902-multinode-verification/`.
