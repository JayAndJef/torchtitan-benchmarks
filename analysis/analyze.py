"""Detailed comparison of two torch-profiler chrome traces.

Usage:
    python analyze.py baseline_iter40.json helion_iter40.json

Reports, for each trace and as a diff:
  - device totals: kernel busy, memcpy, memset, wall span, GPU utilization, gaps
  - host totals: cpu_op time, cuda_runtime launch time, launch counts
  - per-step breakdown from ProfilerStep annotations
  - top kernels by total device time, side by side
  - kernels present in only one run (the fusion changes)
"""
import json
import sys
from collections import defaultdict

DEVICE_CATS = ("kernel", "gpu_memcpy", "gpu_memset")


def load(path):
    with open(path) as f:
        return json.load(f)["traceEvents"]


def xs(events, cat):
    return [e for e in events if e.get("ph") == "X" and e.get("cat") == cat]


def span(evs):
    if not evs:
        return 0.0, 0.0, 0.0
    lo = min(e["ts"] for e in evs)
    hi = max(e["ts"] + e.get("dur", 0) for e in evs)
    return lo, hi, hi - lo


def busy_union(evs):
    """Total wall time during which at least one event is running (handles overlap)."""
    iv = sorted((e["ts"], e["ts"] + e.get("dur", 0)) for e in evs)
    total = 0.0
    cur_s, cur_e = None, None
    for s, e in iv:
        if cur_s is None:
            cur_s, cur_e = s, e
        elif s > cur_e:
            total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_s is not None:
        total += cur_e - cur_s
    return total


def by_name(evs):
    d = defaultdict(lambda: [0.0, 0])
    for e in evs:
        d[e["name"]][0] += e.get("dur", 0)
        d[e["name"]][1] += 1
    return d


def steps(events):
    """ProfilerStep spans on the host thread (dedup by name, take the longest)."""
    out = {}
    for e in events:
        n = e.get("name", "")
        if e.get("ph") == "X" and n.startswith("ProfilerStep#"):
            if n not in out or e.get("dur", 0) > out[n].get("dur", 0):
                out[n] = e
    return dict(sorted(out.items(), key=lambda kv: kv[1]["ts"]))


def report(tag, events):
    ker = xs(events, "kernel")
    cpy = xs(events, "gpu_memcpy")
    mst = xs(events, "gpu_memset")
    dev = ker + cpy + mst
    cpu = xs(events, "cpu_op")
    rt = xs(events, "cuda_runtime")

    _, _, wall = span(dev)
    busy = busy_union(dev)
    print(f"===== {tag} =====")
    print(f"  device wall span      {wall/1000:10.2f} ms")
    print(f"  device busy (union)   {busy/1000:10.2f} ms   ({100*busy/wall:5.1f}% utilization)")
    print(f"  device idle (gaps)    {(wall-busy)/1000:10.2f} ms")
    print(f"    kernel  sum         {sum(e.get('dur',0) for e in ker)/1000:10.2f} ms  n={len(ker)}")
    print(f"    memcpy  sum         {sum(e.get('dur',0) for e in cpy)/1000:10.2f} ms  n={len(cpy)}")
    print(f"    memset  sum         {sum(e.get('dur',0) for e in mst)/1000:10.2f} ms  n={len(mst)}")
    print(f"  host cpu_op sum       {sum(e.get('dur',0) for e in cpu)/1000:10.2f} ms  n={len(cpu)}")
    print(f"  host cuda_runtime sum {sum(e.get('dur',0) for e in rt)/1000:10.2f} ms  n={len(rt)}")

    st = steps(events)
    if st:
        print(f"  per-step (host ProfilerStep):")
        for n, e in st.items():
            print(f"    {n:18s} {e.get('dur',0)/1000:9.2f} ms")
    print()
    return {"kernel": by_name(ker), "busy": busy, "wall": wall, "n_ker": len(ker)}


def diff(a, b, topn=30):
    an, bn = a["kernel"], b["kernel"]
    print("===== per-kernel diff (B - A), sorted by |delta| =====")
    print(f"{'delta_ms':>10s} {'A_ms':>9s} {'A_n':>5s} {'A_us/call':>10s} "
          f"{'B_ms':>9s} {'B_n':>5s} {'B_us/call':>10s}  kernel")
    rows = []
    for n in set(an) | set(bn):
        at, ac = an.get(n, [0.0, 0])
        bt, bc = bn.get(n, [0.0, 0])
        rows.append((bt - at, n, at, ac, bt, bc))
    rows.sort(key=lambda r: -abs(r[0]))
    for d, n, at, ac, bt, bc in rows[:topn]:
        if abs(d) < 20:
            continue
        apc = at / ac if ac else 0
        bpc = bt / bc if bc else 0
        print(f"{d/1000:+10.3f} {at/1000:9.3f} {ac:5d} {apc:10.1f} "
              f"{bt/1000:9.3f} {bc:5d} {bpc:10.1f}  {n[:70]}")
    print()

    only_a = sorted(((v[0], k) for k, v in an.items() if k not in bn), reverse=True)
    only_b = sorted(((v[0], k) for k, v in bn.items() if k not in an), reverse=True)
    print(f"===== kernels only in A ({len(only_a)}), top 12 by time =====")
    for t, k in only_a[:12]:
        print(f"  {t/1000:9.3f} ms  {k[:88]}")
    print(f"\n===== kernels only in B ({len(only_b)}), top 12 by time =====")
    for t, k in only_b[:12]:
        print(f"  {t/1000:9.3f} ms  {k[:88]}")
    print()

    print("===== totals =====")
    print(f"  device busy   A {a['busy']/1000:9.2f} ms   B {b['busy']/1000:9.2f} ms   "
          f"{b['busy']/a['busy']:.4f}x")
    print(f"  device wall   A {a['wall']/1000:9.2f} ms   B {b['wall']/1000:9.2f} ms   "
          f"{b['wall']/a['wall']:.4f}x")
    print(f"  kernel count  A {a['n_ker']:9d}      B {b['n_ker']:9d}      "
          f"{b['n_ker']-a['n_ker']:+d}")


if __name__ == "__main__":
    ap, bp = sys.argv[1], sys.argv[2]
    A = report(f"A  {ap}", load(ap))
    B = report(f"B  {bp}", load(bp))
    diff(A, B)
