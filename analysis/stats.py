"""Per-invocation statistics for compiled transformer-block regions.

Each ``## Call CompiledFxGraph`` GPU annotation is one block-invocation, so a
5-step trace gives 40 samples per region (8 blocks x 5 steps), and the two
traces (iter20, iter40) give 80 independent samples per run.

Reports n / mean / sd / median per region, Welch's t-test and Mann-Whitney U
between runs, and Cohen's d.
"""
import json
import sys
from collections import defaultdict

from scipy import stats

TAG = "## Call CompiledFxGraph"


def spans(paths):
    """graph-hash -> list of GPU span durations (us), pooled across traces."""
    out = defaultdict(list)
    for p in paths:
        ev = json.load(open(p))["traceEvents"]
        for e in ev:
            if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation" \
                    and e["name"].startswith(TAG):
                out[e["name"]].append(e.get("dur", 0.0))
    return out


def rank(d):
    """Regions sorted by total time, largest first."""
    return sorted(d.items(), key=lambda kv: -sum(kv[1]))


def describe(v):
    import statistics as st
    return len(v), st.mean(v), (st.stdev(v) if len(v) > 1 else 0.0), st.median(v)


LABELS = ["backward block", "forward block", "small region A", "small region B", "non-block"]

if __name__ == "__main__":
    A = rank(spans(["baseline_iter20.json", "baseline_iter40.json"]))
    B = rank(spans(["helion_iter20.json", "helion_iter40.json"]))

    print("per-invocation GPU span duration (us), pooled over iter20 + iter40\n")
    hdr = f"{'region':16s} {'n':>4} | {'A mean':>9} {'A sd':>7} {'A med':>9} | " \
          f"{'B mean':>9} {'B sd':>7} {'B med':>9} | {'delta':>8} {'B/A':>7} | " \
          f"{'Welch p':>10} {'MWU p':>10} {'Cohen d':>8}"
    print(hdr)
    print("-" * len(hdr))

    for i, ((an, av), (bn, bv)) in enumerate(zip(A, B)):
        label = LABELS[i] if i < len(LABELS) else f"region{i}"
        na, ma, sa, mea = describe(av)
        nb, mb, sb, meb = describe(bv)
        t = stats.ttest_ind(av, bv, equal_var=False)
        u = stats.mannwhitneyu(av, bv, alternative="two-sided")
        pooled = (((na - 1) * sa ** 2 + (nb - 1) * sb ** 2) / (na + nb - 2)) ** 0.5 if na + nb > 2 else 0
        d = (mb - ma) / pooled if pooled else 0.0
        print(f"{label:16s} {na:>4} | {ma:9.1f} {sa:7.1f} {mea:9.1f} | "
              f"{mb:9.1f} {sb:7.1f} {meb:9.1f} | {mb-ma:+8.1f} {mb/ma:7.4f} | "
              f"{t.pvalue:10.3g} {u.pvalue:10.3g} {d:8.2f}")

    print("\nper-trace split (checks the two windows agree):")
    for tag, files in (("A", ["baseline_iter20.json", "baseline_iter40.json"]),
                       ("B", ["helion_iter20.json", "helion_iter40.json"])):
        for f in files:
            r = rank(spans([f]))
            bwd, fwd = describe(r[0][1]), describe(r[1][1])
            print(f"  {tag} {f:24s} bwd mean {bwd[1]:9.1f} (sd {bwd[2]:6.1f}, n={bwd[0]})   "
                  f"fwd mean {fwd[1]:8.1f} (sd {fwd[2]:6.1f}, n={fwd[0]})")
