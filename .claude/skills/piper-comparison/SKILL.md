---
name: piper-comparison
description: "Piper's published Megatron baseline loses ZeRO-1 whenever the expert degree equals the data-parallel degree, and five handicaps sit on its TorchTitan arm. Use when: comparing any number of ours against Piper's paper (arXiv 2606.11169), reading the piper checkout or its run archive, or citing Figure 7, Figure 8, or Table 2."
---

# Comparing against Piper's artifact

**Read this before you compare any number of ours against Piper's paper**
(arXiv 2606.11169). Piper's Megatron baseline is labelled `zero1` and runs
with **no optimizer-state sharding** whenever the expert degree equals the
data-parallel degree. Every Megatron bar in the paper's Figure 7 is such a
run.

This is read from a checkout outside this repository:
`/m-coriander/coriander/jayden/piper`, branch `main`, rev `439e960`. The
run archive is `/m-coriander/coriander/mfris/` -- another user's directory
on this shared box. **Say where the evidence comes from if you cite it.**

## How to read it

- **Do not copy Piper's configuration.** Their Megatron arm lands on
  replicated dense parameters by accident, their TorchTitan arm shards
  them because it must, and no flag of theirs can bring the two together.
  Under an expert degree the target to copy is **sharded on both sides**,
  which their harness cannot express.
- **The throughput effect is small.** A ring all-reduce already is a
  reduce-scatter plus an all-gather, so ZeRO-1 adds no communication and
  removes some optimizer work. Missing it is roughly neutral, and if
  anything costs Megatron a little. Do not report it as a speed advantage.
- **The memory effect is real.** One local 4-GPU cell in the archive puts
  Megatron at 49.0 GiB against TorchTitan's 34.5 GiB on the same
  configuration. The paper reports no Megatron memory figure anywhere.
- **The paper states no ZeRO level for Figure 7**, for any system. So the
  runs contradict no printed claim. What they contradict is the authors'
  own `--zero-level zero1` on every command line.

## The evidence

- `references/zero1-defect.md` -- the causal chain, Megatron's own argument
  dump, which experiments it reaches, and what TorchTitan does in the same
  runs. Read it before you dispute or extend the claim.
- `references/comparing-our-numbers.md` -- our stock-Megatron results against
  theirs, the five handicaps on their TorchTitan arm, and the two that run
  the other way. Read it before you cite a Piper figure.
- `reports/20260826-stock-megatron/piper-evidence/` -- 812 result CSVs and
  four decisive logs, with a manifest recording their origin.
