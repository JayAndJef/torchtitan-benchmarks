# Piper puts the two engines on different rank layouts

Read `SKILL.md` first. This file holds the second defect, and it is the one
that decides every multi-node number in the archive.

**Piper patched TorchTitan's rank placement and left Megatron on the default.
The two arms therefore put different axes across the machine boundary. Their
multi-node rows measure that difference, not the two engines.**

## What the patch does

Commit `c330f40a` ("training scripts", 2026-04-16) in
`/m-coriander/coriander/mfris/torchtitan` changes
`torchtitan/distributed/parallel_dims.py`. The commit touches 16 files and
886 lines; the mesh change is 18 of them, and the message does not mention
it.

```
upstream:  ("pp", "dp_replicate", "fsdp", "tp")
Piper:     ("dp_replicate", "fsdp", "pp", "tp")
```

`DeviceMesh._unflatten` makes the **last** name vary fastest, so the leftmost
name carries the largest stride. The same swap is applied to the dataloading
mesh and to the sparse mesh.

## The two rank equations

At `pp 8, dp 2` on 16 GPUs, with 8 GPUs in a node:

| arm | rank equation | a node holds | data-parallel group | pipeline group |
|---|---|---|---|---|
| upstream TorchTitan | `dp + 2*pp` | stages 0-3 of both replicas | **inside one node** | crosses nodes |
| Piper's TorchTitan | `pp + 8*dp` | a **whole** 8-stage pipeline | **crosses nodes** | inside one node |
| Megatron, default | `dp + 2*pp` | stages 0-3 of both replicas | **inside one node** | crosses nodes |

Megatron's default order is `tp-cp-ep-dp-pp`
(`megatron/core/parallel_state.py:617`), and in Megatron's convention the
**first** name varies fastest (`:355` sets the stride product, `:514-517`
fills the sizes in order). So Megatron agrees with **upstream** TorchTitan
and disagrees with Piper's patched TorchTitan.

**Both placements were observed, not only derived.** In
`.../20260421_052821/torchtitan/logs/`, local rank *i* is pipeline stage *i*
on every node, so each node holds a complete pipeline. In the Megatron log of
the same run the ranks that print the parameter counts are
`0, 2, 4, 6, 8, 10, 12, 14`, which is `rank = dp_idx + 2*pp_idx` observed.

## Why it decides the result

The data-parallel axis moves gradients, which are gigabytes per step. The
pipeline axis moves activations, which are megabytes per step. Piper's patch
puts the gigabyte axis on the slow link for one arm only.

| | per rank per step | rate |
|---|---|---|
| TorchTitan all-reduce, 2 nodes | 2.455 s for 5.60 GB | **2.3 GB/s** |
| Megatron reduce-scatter, 4 nodes, fastest rank | 0.0156 s for 3.27 GB | **209 GB/s** |

209 GB/s is 4.2x the node's whole 400 Gbps link, so that transfer cannot be
crossing it. **This is a physical argument and it needs no assumption about
the fabric.**

**At one machine the patch is invisible**, because no data-parallel group
spans anything. That is the crossover: TorchTitan wins on one node on raw
compute (1.100 s against 1.498 s per step) and loses by about 3x on two.

## The harness knows about this and never acts on it

The April driver sets `node_count(exp) = exp.dp`
(`scripts/run_qwen_e2e_eval.py:314-315`) and launches every system with
`--nnode <dp> --ngpu <pp>`. So the harness's own model is one node per
data-parallel replica, and the mesh patch makes TorchTitan obey it. Megatron
is sent `--pp`, `--dp` and `--ep` and **no rank-order flag**.

Piper's own harness defines the flag that would equalise the two arms:

```python
# run_megatron.py:151
parser.add_argument("--use-tp-pp-dp-mapping", action="store_true", default=False,
                    help="Use tp-cp-ep-pp-dp rank ordering (PP intra-node) "
                         "instead of default tp-cp-ep-dp-pp (PP cross-node).")
```

It is plumbed at `:294-295`. **No caller passes it**, and the recorded
argument dump reads `use_tp_pp_dp_mapping .... False`. The help text names
the exact distinction that decides these numbers.

## What may be said, and what may not

- **May**: their multi-node rows compare two rank placements as well as two
  engines, so they do not support an engine claim.
- **May**: at one machine the placement is irrelevant, so their single-node
  rows are a fair engine comparison.
- **May not**: "Megatron beats TorchTitan across machines." Nothing in the
  archive separates the engine from the layout.
- **May not**: any multi-node claim about `qwen3_1b`. **The archive holds no
  multi-node `1b` pair at all.** Every multi-node comparison is `9b`.

**Do not treat `csv/april/20260420_055305` as data.** Its Megatron row reads
139 tokens/s per device against about 2,600 in every other 2-node cell, 19x
low. It is the one multi-node row that favours TorchTitan.

## What would settle it

Add `--use-tp-pp-dp-mapping` to one Megatron `pp8 dp2` run. The flag exists
and is plumbed, so this needs no code change. If Megatron's step moves from
1.579 s toward about 3.9 s, the gap is placement and nothing else.
**This needs two machines. This box has one.**

## What is not settled

The inter-node path delivered about 2.3 GB/s where the nominal link is
50 GB/s, a 20x shortfall. The shortfall is measured by three independent
routes. **Its cause is unknown**: no log holds `NCCL INFO`, so the transport
is not recorded. Placement alone would not flip the winner if the link ran at
its nominal rate, so read this as a second necessary condition rather than as
a detail.

The full analysis, with 13 hypotheses each graded, is
`reports/20260828-multinode-crossover/HYPOTHESES.md`. Nine of the thirteen
are refuted there, including the ZeRO-1 defect, the C4 dataloader and the
1F1B bubble.
