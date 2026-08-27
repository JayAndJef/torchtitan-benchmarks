# Comparing our numbers against Piper's

Read `SKILL.md` first. This file holds the coverage gap and the handicaps.

**Our stock-Megatron results agree with Piper's single-node results and
disagree with their multi-node ones. The disagreement is a coverage gap,
not a contradiction.** Their published claim that TorchTitan trails
Megatron holds only where the run crosses nodes, and this box cannot
cross nodes (`MAX_WORLD_SIZE = 8`, one host).

Piper's AWS layout sets `nnode = dp` (`e2e_eval.py:332-336`), so their
data-parallel degree **is** their node count. Sorted that way, their own
April archive splits cleanly:

| layout | who wins |
|---|---|
| 1 node, any `pp` | **TorchTitan**, 1.4x to 10x |
| 2 or 4 nodes | **Megatron**, 1.3x to 3.8x |

Their single-node cells and ours land in the same place. At `1b` they
measured TorchTitan 3.4x ahead; we measured 3.6x.

**It is not a sharding difference.** Their `dp > 1` TorchTitan runs carry
`data_parallel_replicate_degree = dp, data_parallel_shard_degree = 1` --
replication, exactly what `parallelize.py` builds here. That hypothesis
was checked against their command lines and refused.

**Five handicaps sit on their TorchTitan arm and on none of ours.** State
them beside any comparison against a Piper figure:

1. Live C4 streamed over HTTP **inside** the timed region; the other three
   systems get synthetic tensors. Ours uses the replay loader.
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

**What may be said**: our numbers reproduce Piper's own single-node
numbers. **What may not**: any statement about the cross-node case. That
needs a second host.

Evidence copied to `reports/20260826-stock-megatron/piper-evidence/` --
812 result CSVs and four decisive logs, with a manifest recording that
they come from another user's directory on this shared box.

