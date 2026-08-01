"""Compare the three run_bench.sh arms from their profiler traces.

Usage:
    python compare_arms.py <out-dir>          # e.g. torchtitan-benchmarks/out/<ts>
    python compare_arms.py <out-dir> --arms baseline helion

For each arm it expects <out-dir>/<arm>/profiling/traces*/iteration_*/
rank0_trace.json.gz. Reports:
  1. per-arm device totals (kernel sum, busy union, wall span) per window
  2. per-compiled-region span stats, arm vs baseline, with Welch/MWU/Cohen d
  3. RoPE-attributable kernels per arm (standalone kernels by name; for the
     baseline the RoPE cost is fused into inductor kernels and NOT separable
     -- the region-level comparison is the honest signal there)
  4. loss trajectory per arm from the run logs (baseline vs helion should
     track bitwise-closely; the te arm ignores positions and diverges)

Traces are decompressed into /data/zejiaqi/tmp/titan_bench_scratch/ (never /tmp).
"""

import glob
import gzip
import json
import os
import re
import shutil
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stats import describe, rank, spans  # noqa: E402

from scipy import stats as sps  # noqa: E402

SCRATCH = Path("/data/zejiaqi/tmp/titan_bench_scratch")

# Kernel-name substrings that identify standalone RoPE work per arm. The
# baseline has none: inductor fuses RoPE into neighboring kernels.
ROPE_KERNEL_MARKERS = ("helion__rope", "fused_rope_forward", "fused_rope_backward")


def decompress(out_dir: Path, arm: str) -> list[Path]:
    """Decompress all iteration traces for one arm into the scratch dir."""
    srcs = sorted(
        glob.glob(str(out_dir / arm / "profiling" / "traces*" / "iteration_*" / "*.json.gz"))
    )
    dsts = []
    for src in srcs:
        it = Path(src).parent.name  # iteration_NN
        dst = SCRATCH / f"{out_dir.name}_{arm}_{it}.json"
        if not dst.exists() or os.path.getmtime(src) > os.path.getmtime(dst):
            with gzip.open(src, "rb") as f_in, open(dst, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        dsts.append(dst)
    return dsts


def device_totals(path: Path) -> dict:
    ev = json.load(open(path))["traceEvents"]
    dev = [
        e
        for e in ev
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    kern = [e for e in dev if e["cat"] == "kernel"]
    iv = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in dev)
    busy, cur_s, cur_e = 0.0, None, None
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                busy += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        busy += cur_e - cur_s
    lo = min(e["ts"] for e in dev)
    hi = max(e["ts"] + e.get("dur", 0) for e in dev)
    return {
        "kernel_ms": sum(e["dur"] for e in kern) / 1e3,
        "n_kernels": len(kern),
        "busy_ms": busy / 1e3,
        "wall_ms": (hi - lo) / 1e3,
    }


def rope_kernels(path: Path) -> dict[str, tuple[float, int]]:
    ev = json.load(open(path))["traceEvents"]
    out = defaultdict(lambda: [0.0, 0])
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") == "kernel":
            n = e["name"]
            if any(m in n for m in ROPE_KERNEL_MARKERS):
                out[n][0] += e["dur"]
                out[n][1] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}


def losses(log_path: Path) -> list[tuple[int, float]]:
    if not log_path.exists():
        return []
    out = []
    pat = re.compile(r"step:\s*(\d+).*?loss:\s*([0-9.]+)")
    for line in open(log_path, errors="replace"):
        m = pat.search(line)
        if m:
            out.append((int(m.group(1)), float(m.group(2))))
    return out


def compare_regions(base_files: list[Path], arm_files: list[Path], arm: str) -> None:
    A = rank(spans([str(p) for p in base_files]))
    B = rank(spans([str(p) for p in arm_files]))
    labels = ["backward block", "forward block"]
    print(f"\n  compiled-region span us/call, baseline vs {arm} "
          f"(pooled over {len(base_files)} windows; regions paired by size rank):")
    hdr = (f"    {'region':16s} {'n':>4} | {'base mean':>10} {'sd':>7} | "
           f"{'arm mean':>10} {'sd':>7} | {'delta':>8} {'ratio':>7} | "
           f"{'Welch p':>10} {'MWU p':>10} {'d':>6}")
    print(hdr)
    for i, ((_, av), (_, bv)) in enumerate(zip(A, B)):
        label = labels[i] if i < len(labels) else f"region{i}"
        na, ma, sa, _ = describe(av)
        nb, mb, sb, _ = describe(bv)
        t = sps.ttest_ind(av, bv, equal_var=False)
        u = sps.mannwhitneyu(av, bv, alternative="two-sided")
        pooled = (
            (((na - 1) * sa**2 + (nb - 1) * sb**2) / (na + nb - 2)) ** 0.5
            if na + nb > 2
            else 0.0
        )
        d = (mb - ma) / pooled if pooled else 0.0
        print(
            f"    {label:16s} {na:>4} | {ma:10.1f} {sa:7.1f} | "
            f"{mb:10.1f} {sb:7.1f} | {mb - ma:+8.1f} {mb / ma:7.4f} | "
            f"{t.pvalue:10.3g} {u.pvalue:10.3g} {d:6.2f}"
        )


def main() -> None:
    out_dir = Path(sys.argv[1]).resolve()
    arms = ["baseline", "helion", "te"]
    if "--arms" in sys.argv:
        arms = sys.argv[sys.argv.index("--arms") + 1 :]
    SCRATCH.mkdir(parents=True, exist_ok=True)

    files: dict[str, list[Path]] = {}
    for arm in arms:
        files[arm] = decompress(out_dir, arm)
        if not files[arm]:
            print(f"[{arm}] no traces found under {out_dir / arm}; skipping")

    print(f"== {out_dir} ==")

    print("\n1. Device totals per window:")
    for arm in arms:
        for p in files.get(arm, []):
            t = device_totals(p)
            print(
                f"  {arm:9s} {p.name.split('_')[-1]:13s} kernel {t['kernel_ms']:8.2f} ms "
                f"(n={t['n_kernels']})  busy {t['busy_ms']:8.2f}  wall {t['wall_ms']:8.2f}"
            )

    if "baseline" in files and files["baseline"]:
        for arm in arms:
            if arm == "baseline" or not files.get(arm):
                continue
            compare_regions(files["baseline"], files[arm], arm)

    print("\n3. Standalone RoPE kernels per arm (baseline: none -- fused by inductor):")
    for arm in arms:
        agg: dict[str, list[float]] = defaultdict(lambda: [0.0, 0])
        for p in files.get(arm, []):
            for k, (us, n) in rope_kernels(p).items():
                agg[k][0] += us
                agg[k][1] += n
        if not agg:
            print(f"  {arm:9s} (no standalone rope kernels)")
        for k, (us, n) in sorted(agg.items(), key=lambda kv: -kv[1][0]):
            print(f"  {arm:9s} {us / 1e3:8.3f} ms  n={n:<5} {us / n:8.1f} us/call  {k[:70]}")

    print("\n4. Loss trajectories (te arm ignores positions; divergence expected):")
    for arm in arms:
        ls = losses(out_dir / f"{arm}.log")
        if ls:
            pts = [ls[0]] + [ls[i] for i in (9, 19, 29, 39) if i < len(ls)]
            print(f"  {arm:9s} " + "  ".join(f"s{s}:{v:.5f}" for s, v in pts))
        else:
            print(f"  {arm:9s} (no log)")


if __name__ == "__main__":
    main()
