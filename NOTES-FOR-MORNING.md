# Notes for the morning

Branch: **`huge-model-cuda-graph`** (off master `6301158`). Master is
untouched and still at `6301158`. Nothing was merged.

Read this, then `reports/20260809/huge-1layer-cuda-graph-matrix.md`.

---

## TL;DR

| | status |
|---|---|
| Code: `--model-size` axis, huge shape, schema 9, rule 11, supervisor | **done, committed, 160 tests green** |
| Memory-ceiling probe | **done** -- `huge` = dim 12288, 1 layer, 10,528,837,760 params |
| Parity at normal | **PASSED**, rel_l2 5.465e-03 (was ~6e-3; the refactor is clean) |
| Parity at huge | **PASSED**, rel_l2 2.027e-02, with fp32 evidence that it is accumulation |
| Smokes: titan huge, megatron huge, override arm huge, normal replay @80 | **all validated** |
| Overnight sweep, 26 cells | **COMPLETE: 26/26 OK, 0 failed, 0 contaminated** |
| Reports | `reports/20260809/` -- both matrices written, tables final |

**The single most important thing in this file** is the watchdog incident in
"Two false starts" below. The supervisor's first discriminator threw away a
completely valid five-arm result. It is fixed and the fix is verified live,
but you should know it happened.

---

## Commits on this branch

| rev | what |
|---|---|
| `b00d2b3` | the `--model-size` axis, huge shape, schema 9, rule 11, `tools/run_matrix.sh`, docs, tests |
| `3a4e81b` | supervisor: sustained-idle gate; `tools/collect_matrix.py` |
| `44cd39b` | supervisor: **identify our own GPU processes by ancestry, not session id** (see below) |

`44cd39b` is the rev the real sweep runs at, recorded in every cell's
`manifest.json` as `benchmarks_git_rev`. **Do not commit while the sweep is
running** -- that field is a `--resume` gate, so a new commit makes every
in-flight cell unresumable. `NOTES-FOR-MORNING.md` is deliberately left
uncommitted for that reason; commit it when the sweep is done.

---

## Sweep status: DONE

`cells not OK: 0 / 26`. Every cell clean: empty watchdog file, no
`CONTAMINATED` marker, all 26 `results.json` present, single
`benchmarks_git_rev 44cd39b`, 80 steps, GPU 4 throughout.
Ran 17:31-19:06 UTC.

Authoritative record: `out/matrix-20260809b/sweep.log`.

### Headline results

1. **cuda-graph is a 19-26% tok/s win at the normal shape and ~0% at the
   huge shape**, with ~0 change in GPU kernel time in both cases. That is
   the question the two matrices were built to answer. The mechanism is
   measured, not assumed: cuda-graph removes per-launch host cost, which
   binds at the normal shape and does not bind at the huge one.
2. **The huge shape is GPU-bound** (~50-54% MFU vs ~15%), with the launch
   decomposition to prove it (item 3 below). Its tok/s column is
   trustworthy, unlike the 1B one.
3. **FlexAttention's Triton template is the fastest of the three attention
   backends at both shapes**, by `attention_core`: baseline < flex_flash
   (1.19-1.21x) < flash_attention_3 (1.67-1.78x), stable across a 12x dim
   change and both ac modes. Two total-kernel-time "wins" for the other
   backends were published during the session and are retracted -- see
   item 0 under "Things you may want to act on".
4. **At the huge shape the engine gap nearly closes**: titan's best arm is
   0.936-0.943x megatron's step cost, against a much larger win at 1B.
5. **Megatron pays 37% more memory under cuda-graph for no throughput** at
   the huge shape (87.82 -> 120.14 GiB: 19.61 of `main_grad`, 12.71 of graph
   pools).

Root: `out/matrix-20260809b/<size>/ac-<ac>/<mode>/<scenario>/`
(the earlier `out/matrix-20260809/` -- no `b` -- holds the two false starts
and is kept only as evidence; do not read numbers out of it).

26 cells, huge first, all at `--steps 80`, batch 4, seq 1024, GPU 4:

- huge x {default, cuda-graph} x {piper1b_megatron, piper1b_attention}, ac=none = 4 cells (16 arm-runs)
- normal x {sac, none} x {default, cuda-graph} x 6 scenarios, minus megatron's 2 sac cells = 22 cells (70 arm-runs)

### Relaunching

```bash
cd /m-coriander/coriander/jayden/torchtitan-benchmarks

# Continue (skips cells that already have results.json and no CONTAMINATED marker).
ROOT=out/matrix-20260809b GPU=4 STEPS=80 PASSES=8 \
HF_DATASETS_CACHE=/m-coriander/coriander/jayden/.cache/hf-datasets \
  setsid nohup ./tools/run_matrix.sh >> out/run_matrix_b.nohup 2>&1 &

# Half at a time:
CELLS=huge   ROOT=out/matrix-20260809b GPU=4 STEPS=80 ./tools/run_matrix.sh
CELLS=normal ROOT=out/matrix-20260809b GPU=4 STEPS=80 ./tools/run_matrix.sh

# One cell by hand (exactly what the supervisor runs):
HF_DATASETS_CACHE=/m-coriander/coriander/jayden/.cache/hf-datasets \
./run_bench.sh run-all 4 --scenario piper1b_megatron --ac none \
    --compile-mode cuda-graph --model-size huge --steps 80 \
    --out out/matrix-20260809b/huge/ac-none/cuda-graph/piper1b_megatron
```

The supervisor refuses to start on a dirty tree and holds a `flock` on
`$ROOT/.lock`, so a second invocation cannot race the first.

### Collecting and rendering

```bash
.venv/bin/python tools/collect_matrix.py out/matrix-20260809b \
    --size huge --launch-counts --out reports/20260809/huge-matrix.json
.venv/bin/python tools/collect_matrix.py out/matrix-20260809b \
    --size normal --out reports/20260809/normal-matrix.json
.venv/bin/python reports/20260809/render_matrix.py reports/20260809/huge-matrix.json
```

`render_matrix.py` lives in `reports/` (gitignored) rather than `tools/`
because it exists only to regenerate this report's markdown. Both were
smoke-tested against real output before the sweep.

---

## Two false starts, and what they cost

### 1. A foreign job took the GPU 28 s after the idle check (16:50)

The GPU was empty at 16:49:40, a foreign 93.8 GiB process appeared at
16:50:08, and megatron OOMed allocating `exp_avg_sq`. The supervisor caught
it, marked it `CONTAMINATED` and scheduled a re-run -- correct behaviour.

Fix (`3a4e81b`): `wait_for_idle` now requires **three consecutive idle
samples 20 s apart** instead of one instantaneous sample, and the watchdog
samples every 15 s instead of 30. No cell had completed, so this cost
nothing.

### 2. The watchdog condemned our own arms as foreign (17:26) -- the serious one

`huge|none|default|piper1b_megatron` **completed successfully**: all five
arms ran, validated and evaluated, `results.json` written, `rc=0`. The
watchdog flagged it anyway and it was moved aside.

The discriminator was "a compute PID whose session id differs from the
supervisor's is foreign". **Every training process reports its own pid as
its session id** -- torch elastic spawns workers with
`start_new_session=True` and the megatron driver ends up detached the same
way -- so every arm of both engines looked foreign, with `ours=0MiB` on
every single sample.

The giveaway was in the verdict line itself: `rc=0 (gpu after=4MiB)`. If a
90 GiB foreign process had genuinely been resident 16 seconds earlier, it
would still have been there when the cell ended. It was our own megatron
arm.

Fix (`44cd39b`): `is_ours()` walks the **ppid chain** up to the supervisor.
`setsid` changes a process's session but never its parent, so ancestry
survives exactly the case that broke the old check.
`tools/test_watchdog_attribution.sh` reproduces the failure and asserts the
fix (a descendant spawned with `start_new_session=True` reads OURS and is
confirmed to be its own session leader; pid 1 still reads FOREIGN). Two
smaller hardenings came with it: an already-exited compute pid is skipped
rather than flagged, and `FOREIGN_MEM` needs two consecutive samples.

**Verified live, not just in a unit test.** On the first cell of the real
sweep the log carries `watchdog: attributing 518MiB on gpu4 to our own arms`
and the `.watch` file stayed **empty** while our megatron arm held
90,700 MiB. Under the old code that would have been two flags per sample.

**Known residual limitation.** Ancestry breaks if an intermediate parent
exits and the CUDA process reparents to pid 1. That cannot happen during
normal operation -- `subprocess.run` blocks, so every parent in the chain
outlives the child -- and when it *does* happen the cell is already being
torn down, so flagging it is correct anyway. If you want belt-and-braces
later, add a cmdline match (`megatron_baseline.train|run_train|torchtitan`)
restricted to our own username as a second "ours" signal. I deliberately did
not add it mid-sweep: it would mean another commit, and `benchmarks_git_rev`
is a resume gate.

### The discarded-but-valid result (evidence the pipeline works at huge)

`out/matrix-20260809/huge/ac-none/default/piper1b_megatron.contaminated-20260809T172631Z/`
is kept on purpose. It is a complete, clean, five-arm huge-shape result --
discarded **only** by the watchdog bug, and measured at `3a4e81b`, so it is
not comparable to the post-fix sweep and is **not** published in the report.
Cited here as proof the whole path works end to end at the huge shape:

| arm | tok/s | n | kern ms/step | vs base | launch us | peak GiB |
|---|---:|---:|---:|---:|---:|---:|
| baseline (megatron) | 15,472 | 36 | 276.10 | 1.000 | 8.6 | 87.82 |
| titan_stock | 15,428 | 36 | 260.20 | 0.942 | 478.1 | 86.39 |
| titan_swiglu | 14,676 | 36 | 272.50 | 0.987 | 516.9 | 86.86 |
| titan_lm_head | 15,498 | 36 | 258.58 | 0.937 | 479.0 | 86.39 |
| titan_swiglu_lm_head | 14,794 | 36 | 271.65 | 0.984 | 517.9 | 86.86 |

`stable_sample_count = 36` on every arm confirms the 80-step change works
(four cycles instead of two). Treat the numbers as indicative only; the
sweep's own cells supersede them. `results.json` carried this warning, which
will matter for the real cells too:

> host launch latency varies 60.48x across arms (baseline 8.56us ..
> titan_swiglu_lm_head 517.87us); tokens/s and span metrics are
> host-speed-confounded -- compare kernel time

That 60x is not contention: it is megatron issuing far fewer, far larger
kernel launches than titan at this shape. Compare `kern ms/step`.

### Also: the coordinating session intervened at ~17:28-17:29

The main session independently diagnosed the same watchdog bug, killed the
supervisor, and killed the in-flight `huge|none|default|piper1b_attention`
cell mid-arm (on `qwen3_piper_1b_flex_flash_huge`). That half-written cell is
in the **old** root `out/matrix-20260809/`, which the current sweep does not
use, so it is inert. Nothing needs cleaning up for the sweep to be correct.

---

## What was built, and the decisions inside it

### `--model-size` is a third global run axis

Exactly parallel to `--compile-mode` and `--ac`: one shape per run, recorded
in the manifest, gated by `--resume`, a hard comparability boundary.
`piper1b/model_shape.py` holds frozen `PiperShape` dataclasses and is the
single source of geometry for **both** engines, so `config_registry.py` and
`megatron_baseline/model.py` cannot drift. It deleted the two hand-copied
`NUM_FLOPS_PER_TOKEN` constants and the `VOCAB_SIZE` literal.

Config naming is `<config>` at normal, `<config>_<size>` otherwise; ten
explicit `_huge` `def`s, with a test asserting the closure over every
(scenario, arm, size) triple.

### Three things that would have silently produced wrong numbers

1. **Regions at one layer.** A region is identified by its invocations per
   window, and that count *is* the identity -- on a real 16-layer trace the
   forward graphs run {5, 80, 5} times per window and the backward {5, 80},
   so 80 is unique to the block graphs. At one layer the block graph also
   runs 5 times and collides with three other partitions. Rather than add a
   tiebreak (which would be relaxing validation rule 7), `PiperShape.huge`
   sets `supports_block_regions=False` and the run declares `regions: []`,
   exactly as `piper1b_megatron` already does. **Rule 7 does not guard huge
   runs; rules 8, 9 and the new rule 11 do.**
2. **Override counts at one layer.** `Arm.expected_override_count = 16`
   became `Arm.overrides_per_block = 1`, which `validate_arm` multiplies by
   the shape's layer count. Without it, `titan_swiglu` and
   `titan_swiglu_lm_head` would have failed rule 2 at huge.
3. **New validation rule 11.** The log must contain
   `size: <N> total parameters` for the requested shape's exact count. Both
   engines print it. Without it, a run whose `--config` mapping or
   `--model-size` silently fell back would pass every other rule and be
   published under the wrong size.

### The 80-step blocker

`piper1b/pretokenized_data.py` pinned `replay_steps = 40` and hard-fails when
the run asks for more, so every titan arm in `piper1b_megatron` would have
died at step 41. Rather than bump the constant to 80 (which just moves the
cliff), the count now **tracks `--training.steps`**:
`Workload.replay_dataloader` marks the scenario and `command_for_arm` emits
`--dataloader.replay-steps <steps>`. Verified end to end at 80 steps at both
sizes. `stable_tps` needed no change -- it derives its window modularly from
`profile_freq`, so 80 steps give four cycles and 36 samples.

### Deviations from the plan

| plan said | what I did | why |
|---|---|---|
| `--steps 40` (section 8.2) | `--steps 80` everywhere | the later instruction; verified `stable_tps` and rule 5 handle it, and fixed the `replay_steps` blocker it exposed |
| bump `replay_steps` to 80 | made it track `--steps` | a bumped constant breaks again at 160; tyro accepts `--dataloader.replay-steps` (verified before relying on it) |
| pick the largest dim under 125 GiB | 12288 | 13312 and 14336 both peak at ~136.7 GiB (97.7%) during graph capture -- 3 GiB of headroom is not survivable overnight |
| "do not touch the 2e-2 parity gate yet" | per-shape gate: 2e-2 normal, 5e-2 huge | huge measured 2.027e-02. Followed the plan's section-11 ladder and gathered the fp32 evidence it demands first (below) |
| one commit before the sweep | three | each extra commit followed a supervisor bug found before any cell had been *kept*, so none invalidated data. Everything since `44cd39b` is uncommitted |
| supervisor: single idle sample, 30 s watchdog, session-id attribution | 3 consecutive idle samples, 15 s watchdog, **ancestry** attribution | the two false starts above |

### The huge parity gate: what was actually checked

Diagnosed in the plan's order before touching anything:

1. Normal-size parity re-run as the regression gate on the refactor:
   **5.465e-03**, unchanged. The parameterization is not the problem.
2. The QKV grouped interleave -- the only place a dim change could plausibly
   break the transfer -- is asserted **bitwise** by `_assert_qkv_roundtrip`,
   which reconstructs `wq/wk/wv` out of the packing and requires
   `torch.equal`. Passes at both shapes. (A wrong interleave also gives
   rel_l2 of order 1, not 2e-2.)
3. `sqrt` scaling predicts it: `5.465e-3 * sqrt(12288/1024) = 1.89e-2`
   against 2.027e-2 measured.
4. Direct evidence via the new `--fp32-reference`: the **same weights** in
   fp32 put titan's own bf16 output **3.251e-02** from the reference and
   megatron's **3.286e-02** -- ratio **1.011**. The engine-to-engine distance
   (2.027e-02) is *smaller* than either engine's distance to fp32, and
   megatron's loss (12.327709) is closer to the fp32 loss (12.327950) than
   titan's (12.327404). Both engines are equally correct; the residual is
   reduction order.

Only then was the gate widened, per shape, with the reasoning recorded at
`_PARITY_GATE` in `tools/megatron_parity_check.py` and in CLAUDE.md.

---

## Things you may want to act on

0. **RANK ARMS ON `components.py`, NEVER ON TOTAL KERNEL TIME.** This is the
   most important thing in this file after the watchdog incident, and it
   produced a wrong published claim twice in one session -- once at the
   normal shape (caught by the coordinating session) and once by me at the
   huge shape.

   Arms within a scenario differ in exactly one thing; everything else is
   byte-identical code. But Inductor autotuning and allocator placement move
   the *unrelated* components by several ms/step between arms, and that drift
   was **larger than the effect being measured**:

   | cell | unrelated-component drift | real effect |
   |---|---:|---:|
   | huge / default / piper1b_attention (FA3 vs baseline) | **11.14 ms/step** (4.2% of step) | 1.64 ms/step |
   | huge / default / piper1b_megatron (swiglu vs stock) | 3.96 ms/step (1.5%) | 9.16 ms/step |

   In the attention cell the MoE expert GEMM alone swung 6.65 ms/step and the
   optimizer 3.35 ms/step -- components no attention backend touches.

   **Corrected attention ranking at huge** (`attention_core` ms/step):
   baseline **2.109** < flex_flash **2.503** (1.19x) < flash_attention_3
   **3.750** (1.78x). The total column said the reverse (FA3 "best" at
   0.9645). **I published the total-based ranking in an earlier message; it
   is retracted** and section 5 of the report carries the retraction.

   The genuinely valuable result is that this ordering is *identical* at the
   normal shape (3.032 / 3.552 = 1.17x / 4.969 = 1.64x). Across a 12x dim and
   16x layer-count change the FlexAttention Triton template stays fastest and
   the ratios barely move. The attention verdict does **not** invert with
   shape.

   Which earlier claims survive this:
   - "titan_swiglu is a regression at huge" -- **survives**, direction
     confirmed, but the honest magnitude is ~9.2 ms/step, not 13.1: +1.10 in
     the SwiGLU activation and +8.06 in the fused-w13 expert GEMM (both
     legitimately changed by the override), with +3.96 of unrelated drift.
   - "the engine gap nearly closes at huge" (titan best 0.943x megatron) --
     **survives**, and unlike the arm comparisons the total is the *right*
     metric here, because the two engines share no code and every component
     legitimately differs. Treat ~1.5-4% as the single-run error bar, so a
     ~6% gap is real but not precise.
   - "flex_flash improved at huge (0.9915 vs clearly behind at normal)" --
     **retracted**. Its `attention_core` penalty is unchanged (1.17x ->
     1.19x); the sm90 per-lane mask path costs a fixed amount and the step
     got ~5x longer, so the same penalty dilutes in the total.

1. **`analysis/components.py` cannot break down a cuda-graph titan arm.**
   Graph replay erases the per-op CPU frames the frame-first classifier
   reads, so the whole captured block lands in `other_elementwise` (measured
   on the huge smoke: `moe_expert_gemm 0.019` / `other_elementwise 123.948`
   for the cuda-graph arm, versus `moe_expert_gemm 115.424` /
   `other_elementwise 2.626` for the same model in default mode). Attribution
   health still prints 100% agreement, so the tool does **not** warn you.
   **Only run `components.py` on default-mode cells.** Not currently
   documented in CLAUDE.md; it should be.
2. **At the huge shape the engine gap nearly closes**, and the SwiGLU
   override becomes a regression. Confirmed on the sweep's own clean cell:
   titan's best arm is 0.943x megatron's kernel time, against a much larger
   win at the normal shape. See item 0 for the error bar and for what the
   swiglu regression really decomposes into.
3. **The huge shape is GPU-bound, with evidence.** Titan's launches/step
   *fell* 5.6x going to huge (2,233 -> 401) while the mean time inside each
   launch call *rose* 83x (5.9us -> 493.6us). Host dispatch cost cannot behave
   that way; a 493us launch call is the host blocking on a full queue. With
   ~50-54% MFU (vs ~15% at normal), huge tok/s is trustworthy in a way the 1B
   numbers never were -- the confound that has dogged every previous
   conclusion here. Two corollaries: titan's `launch us` at huge is
   back-pressure, not host overhead; and the "launch latency varies 59x
   across arms" warning on the megatron cell is a **false alarm** (megatron
   8.8us never fills the queue, titan 477us does, yet both land within 6% on
   step cost). Report section 6b has the table.
4. **A contaminated cell currently discards its clean arms too.** If a
   foreign job appears during arm 5 of 5, arms 1-4 are thrown away with it,
   because the watchdog does not attribute flags to arm boundaries. That is
   the conservative choice and it is what was asked for, but on a badly
   cycling night it could stop a 5-arm cell from ever completing. Timestamping
   flags against arm start/end times would let a re-run redo only the
   affected arm.

---

## The CLAUDE.md amendment (uncommitted on purpose -- commit it!)

`CLAUDE.md` now carries a new subsection under Evaluation, **"Total kernel
time cannot rank arms that differ in one component"**, with the three
measured cases, the noise-floor table, the cross-engine exception, and the
`components.py`/cuda-graph limitation. It is the correction to CLAUDE.md's
own stated methodology ("This is the host-speed-immune metric; compare
kernels with it"), which is true about host speed but not about Inductor
autotuning.

It is **edited but not committed**, deliberately. `benchmarks_git_rev` is
`git rev-parse HEAD`, which ignores uncommitted edits, so every cell of the
matrix records `44cd39b`. Committing mid-sweep would have split the 26 cells
across two revs and broken `--resume` for anything in flight. Commit it with
`NOTES-FOR-MORNING.md` once the sweep is done; the resulting rev will be one
commit ahead of what the manifests record, which is correct and harmless
because the commit changes no code.

## Uncommitted at hand-off

This file, plus everything under `reports/` (gitignored by design). Commit
this file once the sweep has finished:

```bash
git add NOTES-FOR-MORNING.md && \
git -c user.name="Jayden" -c user.email="zejiaqi1203@gmail.com" \
    commit -m "Add overnight run notes"
```

## Housekeeping

- `out/smoke-huge-titan`, `out/smoke-huge-megatron`, `out/smoke-huge-swiglu`,
  `out/smoke-normal-replay`, `out/probe/` are pre-sweep verification runs,
  safe to delete.
- `out/matrix-20260809/` (no `b`) is the two false starts. Keep the
  `piper1b_megatron.contaminated-*` directory as evidence; the rest is inert.
- `out/run_matrix_b.nohup` mirrors `sweep.log`.
- The sweep's datasets cache is
  `/m-coriander/coriander/jayden/.cache/hf-datasets`. The shared
  `HF_HOME=/m-coriander/coriander/hf` is owned by another user; its
  `builder.lock` is the failure the memory note refers to.
