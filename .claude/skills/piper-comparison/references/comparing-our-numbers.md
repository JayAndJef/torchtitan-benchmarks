# Comparing our numbers against Piper's

Read `SKILL.md` first. This file holds the two arithmetic traps, the
coverage gap and the handicaps.

## Two traps that already produced a withdrawn claim

**`ngpu` in Piper's CSVs is GPUs PER NODE, not total.** Verified:
`pp * dp * ep == nnode * ngpu` holds for 120 rows and `== ngpu` for none. So

    tokens/s per device = global_batch_size * seq_len / (nnode * ngpu) / iter_time_mean

A divisor of `ngpu` alone makes a 4-node figure 4x too high. Ratios survive
the error; absolute values do not.

**Pair their runs inside ONE run directory.** Their repeats of one identical
cell span up to 5.6x, so a median over sessions mixes machine states months
apart and can invert the order. A same-session pair holds the machine, the
date and the sweep constant.
`reports/piper-analysis/same_session_pairs.py` finds them. **That script
still divides by `ngpu` alone**, and is correct only because it filters
`nnode == 1`.

## Which of their numbers we can compare against

**Their single-node cells are a fair engine comparison. Their multi-node
cells are not**, because Piper puts the two engines on different rank
layouts. Read `references/rank-placement.md` before you cite any multi-node
row. It is a defect in their harness, not a coverage gap in ours, and it was
diagnosed on 2026-08-28.

Piper's AWS layout sets `nnode = dp` (`e2e_eval.py:332-336`), so their
data-parallel degree **is** their node count. Sorted that way, their April
archive splits:

| layout | who wins | is it an engine result? |
|---|---|---|
| 1 node, any `pp` | mixed; see below | **yes** |
| 2 or 4 nodes | Megatron, about 3x | **no** -- placement, see `rank-placement.md` |

**On one machine the winner is not settled.** Matched same-session pairs
split **4-4** between the engines. The `1b` figure of "TorchTitan 3.4x
ahead", which earlier versions of this file stated beside our own 3.6x,
**has no shown derivation**. It traces to one session,
`csv/june/schedule_local` (3.39x), which is a legitimate same-session pair
and one draw from a wide spread. Do not quote it as their headline result.

**Their archive holds no multi-node `qwen3_1b` pair at all.** Every
multi-node comparison is `9b`.

**It is not a sharding difference.** Their `dp > 1` TorchTitan runs carry
`data_parallel_replicate_degree = dp, data_parallel_shard_degree = 1` --
replication, exactly what `parallelize.py` builds here. That hypothesis was
checked against their command lines and refused, twice.

## Five handicaps sit on their TorchTitan arm and on none of ours

State them beside any comparison against a Piper figure:

1. Live C4 streamed over HTTP **inside** the timed region; the other three
   systems get synthetic tensors, and Megatron gets `--mock-data`. Ours uses
   the replay loader. **This cost is flat in the node count**, so it does not
   explain their multi-node result.
2. `GroupedExperts.forward` monkeypatched off `torch._grouped_mm` onto a
   Python BMM loop, on by default (`sitecustomize.py`), which calls
   `int(counts.max().item())` -- a device-to-host sync per MoE layer per
   microbatch.
3. `--compile.no-enable`: compile off. Ours defaults to per-block compile.
   Our own `--compile-mode none` cell still beats **our** megatron arm,
   2.2x at `1b` (25,568 against 11,630 tok/s/device), so compile is not
   what decides the single-node winner. That is a comparison inside our
   own matrix; it says nothing about their Megatron, which ran on other
   hardware.
4. `load_balance_coeff=None`, so their router is unbalanced while Megatron
   gets `aux_loss` balancing. This compounds item 2.
5. cuDNN removed from the SDPA backend priority list, in their patch.

**Two handicaps run the other way**, so a corrected single-node comparison
would favour TorchTitan by more than we measure, not less: Megatron alone
gets `NCCL_P2P_DISABLE=1` (`scripts/run_megatron.sh:45`), and Megatron
alone gets 12 virtual pipeline stages on the interleaved schedule against
TorchTitan's and Piper's 2.

**A sixth handicap applies to the multi-node rows only**, and it is the
rank placement. It is larger than the other five together.

## What may be said

- **May**: our single-node numbers stand beside their single-node numbers.
- **May**: their multi-node rows compare two layouts as well as two engines.
- **May not**: "TorchTitan wins on one machine" as a settled result. The
  matched pairs are 4-4.
- **May not**: any engine claim from their multi-node rows.

A second host would let us test the placement directly. This box has one
(`MAX_WORLD_SIZE = 8`).

Evidence copied to `reports/20260826-stock-megatron/piper-evidence/` --
812 result CSVs and four decisive logs, with a manifest recording that
they come from another user's directory on this shared box.
