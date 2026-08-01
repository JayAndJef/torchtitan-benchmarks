"""Per-compiled-region (== per attention/FFN block) GPU time, averaged per invocation.

torchtitan has no record_function spans, but torch.compile emits
``## Call CompiledFxGraph <hash>`` annotations on BOTH the host thread and the
GPU stream. Each graph is invoked once per layer per step, so aggregating GPU
kernels inside each GPU-side annotation and dividing by the invocation count
gives a per-block average that is directly comparable between two runs.

Usage: python per_block.py A.json B.json
"""
import json
import sys
from collections import defaultdict

TAG = "## Call CompiledFxGraph"


def analyze(path):
    ev = json.load(open(path))["traceEvents"]

    # GPU-side compiled-region spans, per stream
    spans = defaultdict(list)  # stream tid -> [(ts, te, name)]
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") == "gpu_user_annotation" and e["name"].startswith(TAG):
            spans[e["tid"]].append((e["ts"], e["ts"] + e.get("dur", 0), e["name"]))
    for tid in spans:
        spans[tid].sort()

    # kernels per stream
    kernels = defaultdict(list)
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset"):
            kernels[e["tid"]].append((e["ts"], e.get("dur", 0), e["name"], e.get("cat")))

    # attribute each kernel to the enclosing span (same stream, time containment)
    per_graph = defaultdict(lambda: {"gpu_us": 0.0, "n_kernels": 0, "kern": defaultdict(lambda: [0.0, 0])})
    invocations = defaultdict(int)
    span_wall = defaultdict(float)
    for tid, sp in spans.items():
        for ts, te, name in sp:
            invocations[name] += 1
            span_wall[name] += te - ts
        # sweep kernels
        for kts, kdur, kname, kcat in kernels.get(tid, []):
            for ts, te, name in sp:
                if ts <= kts < te:
                    g = per_graph[name]
                    g["gpu_us"] += kdur
                    g["n_kernels"] += 1
                    g["kern"][kname][0] += kdur
                    g["kern"][kname][1] += 1
                    break
    return per_graph, invocations, span_wall


def fmt(path):
    pg, inv, wall = analyze(path)
    rows = []
    for name, d in pg.items():
        n = inv[name]
        rows.append((d["gpu_us"], name, n, d["gpu_us"] / n, wall[name] / n, d["n_kernels"] / n, d["kern"]))
    rows.sort(reverse=True)
    return rows


if __name__ == "__main__":
    a_rows, b_rows = fmt(sys.argv[1]), fmt(sys.argv[2])
    print(f"A = {sys.argv[1]}")
    print(f"B = {sys.argv[2]}\n")

    def show(tag, rows):
        print(f"----- {tag}: GPU time attributed to each compiled region -----")
        print(f"{'total_ms':>10} {'calls':>6} {'us/call':>10} {'span_us/call':>13} {'kern/call':>10}  graph")
        for tot, name, n, per, wper, kper, _ in rows:
            print(f"{tot/1000:10.3f} {n:6d} {per:10.1f} {wper:13.1f} {kper:10.1f}  {name[-12:]}")
        print()

    show("A", a_rows)
    show("B", b_rows)

    # pair graphs by rank (largest first) since hashes differ between runs
    print("----- paired by size rank (hashes differ between runs) -----")
    print(f"{'rank':>4} {'A us/call':>11} {'B us/call':>11} {'delta':>10} {'B/A':>8} {'A kern':>8} {'B kern':>8}")
    for i, (a, b) in enumerate(zip(a_rows, b_rows)):
        print(f"{i:4d} {a[3]:11.1f} {b[3]:11.1f} {b[3]-a[3]:+10.1f} {b[3]/a[3]:8.4f} {a[5]:8.1f} {b[5]:8.1f}")
    print()

    ta = sum(r[0] for r in a_rows) / a_rows[0][2]
    tb = sum(r[0] for r in b_rows) / b_rows[0][2]
    print(f"sum of all compiled regions, per layer-step:  A {ta:9.1f} us   B {tb:9.1f} us   {tb/ta:.4f}x")

    # biggest kernel-level movers inside the largest region
    print(f"\n----- kernels inside the LARGEST region (rank 0), per call -----")
    ak, bk = a_rows[0][6], b_rows[0][6]
    na, nb = a_rows[0][2], b_rows[0][2]
    names = set(ak) | set(bk)
    diffs = []
    for n in names:
        av = ak.get(n, [0.0, 0])[0] / na
        bv = bk.get(n, [0.0, 0])[0] / nb
        diffs.append((bv - av, n, av, bv))
    diffs.sort(key=lambda r: -abs(r[0]))
    for d, n, av, bv in diffs[:10]:
        if abs(d) < 1:
            continue
        print(f"  {d:+9.1f} us   A {av:9.1f}   B {bv:9.1f}   {n[:64]}")
