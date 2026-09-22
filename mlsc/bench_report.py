#!/usr/bin/env python3
"""bench_report.py — pick benchmark volumes, then compare partitions.

    bench_report.py --work W --select [--n 3]
        prints a comma-separated list of representative volume ids from
        manifest/volumes.csv (smallest, median, largest by voxel count, plus
        evenly spaced extras when --n > 3). run_pipeline.sh bench feeds this
        to predict_batch.py on each partition.

    bench_report.py --work W
        reads manifest/bench/*.json (written by predict_batch.py --bench-out)
        and prints, per partition x model: seconds/volume, peak GPU MB, peak
        RSS MB, CPU-fallback count; then recommends the lightest partition
        whose peak GPU allocation stays under 90 % of the card and a
        PREDICT_MEM value (1.3 x peak RSS, rounded up to 8 GB).
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import sys

# GB per card, lightest first (fairshare weight: A100 > RTX8000 > RTX6000).
PARTITION_GPU_GB = [("rtx6000", 24), ("rtx8000", 48), ("dgx-a100", 40), ("pubgpu", 40)]


def select_volumes(work, n):
    path = os.path.join(work, "manifest", "volumes.csv")
    with open(path, newline="") as f:
        rows = [r for r in csv.DictReader(f) if r.get("n_voxels")]
    if not rows:
        raise SystemExit("no staged volumes in manifest/volumes.csv")
    rows.sort(key=lambda r: int(r["n_voxels"]))
    if len(rows) <= n:
        picks = rows
    else:
        idx = sorted({0, len(rows) // 2, len(rows) - 1} |
                     {round(i * (len(rows) - 1) / (n - 1)) for i in range(n)})
        picks = [rows[i] for i in idx][:n] if n >= 3 else [rows[i] for i in idx[:n]]
    return picks


def load_bench(work):
    out = []
    for path in sorted(glob.glob(os.path.join(work, "manifest", "bench", "*.json"))):
        with open(path) as f:
            s = json.load(f)
        s["_file"] = os.path.basename(path)
        out.append(s)
    return out


def summarize(s):
    ok = [v for v in s.get("volumes", []) if v.get("status") == "ok"]
    secs = [v["total_seconds"] for v in ok]
    gpu = [v["gpu_peak_alloc_mb"] for v in ok if v.get("gpu_peak_alloc_mb") is not None]
    rss = max([s.get("rss_self_mb") or 0] + [v.get("rss_self_mb") or 0 for v in ok])
    rss_kids = max([s.get("rss_children_mb") or 0] + [v.get("rss_children_mb") or 0 for v in ok])
    return {
        "partition": s.get("partition") or s["_file"].split("-")[0],
        "gpu": s.get("gpu"),
        "model": s["model"],
        "n_ok": len(ok), "n_failed": s.get("n_failed", 0),
        "sec_mean": sum(secs) / len(secs) if secs else None,
        "sec_max": max(secs) if secs else None,
        "gpu_peak_mb": max(gpu) if gpu else None,
        "rss_peak_mb": rss + rss_kids,
        "cpu_fallbacks": sum(1 for v in ok if v.get("cpu_fallback")),
        "load_seconds": s.get("load_seconds"),
    }


def _fmt(v, nd=0):
    if v is None:
        return "-"
    return f"{v:.{nd}f}"


def report(work):
    rows = [summarize(s) for s in load_bench(work)]
    if not rows:
        print("no bench results under manifest/bench/")
        return 1
    rows.sort(key=lambda r: ([p for p, _ in PARTITION_GPU_GB].index(r["partition"])
                             if r["partition"] in dict(PARTITION_GPU_GB) else 99, r["model"]))
    print(f"{'partition':10} {'gpu':22} {'model':18} {'ok':>3} {'fail':>4} {'s/vol':>7} {'s max':>7} "
          f"{'gpu MB':>7} {'rss MB':>7} {'cpuFB':>5} {'load s':>6}")
    for r in rows:
        print(f"{r['partition']:10} {(r['gpu'] or '-')[:22]:22} {r['model']:18} {r['n_ok']:>3} "
              f"{r['n_failed']:>4} {_fmt(r['sec_mean']):>7} {_fmt(r['sec_max']):>7} "
              f"{_fmt(r['gpu_peak_mb']):>7} {_fmt(r['rss_peak_mb']):>7} {r['cpu_fallbacks']:>5} "
              f"{_fmt(r['load_seconds']):>6}")

    print()
    by_part = {}
    for r in rows:
        by_part.setdefault(r["partition"], []).append(r)
    chosen = None
    for part, gb in PARTITION_GPU_GB:
        rs = by_part.get(part)
        if not rs:
            continue
        peak = max((r["gpu_peak_mb"] or 0) for r in rs)
        fails = sum(r["n_failed"] for r in rs)
        fallbacks = sum(r["cpu_fallbacks"] for r in rs)
        fits = peak < 0.9 * gb * 1024 and fails == 0 and fallbacks == 0
        print(f"{part}: peak GPU {peak:.0f} MB of {gb} GB, failures={fails}, cpu fallbacks={fallbacks}"
              f" → {'fits' if fits else 'does not fit cleanly'}")
        if fits and chosen is None:
            chosen = part
    rss = max(r["rss_peak_mb"] for r in rows)
    mem_gb = int(math.ceil(rss * 1.3 / 1024 / 8) * 8)
    secs = max((r["sec_max"] or 0) for r in rows)
    print()
    if chosen:
        print(f"recommend: GPU_PARTITION={chosen}")
    else:
        print("recommend: no partition fit cleanly — inspect failures / fallbacks above")
    print(f"recommend: PREDICT_MEM={mem_gb}G   (1.3 x peak RSS {rss:.0f} MB, rounded up to 8 GB)")
    print(f"note: slowest volume took {secs:.0f}s; size PREDICT_TIME as CHUNK_SIZE x that + model load.")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--select", action="store_true")
    ap.add_argument("--n", type=int, default=3)
    args = ap.parse_args(argv)
    if args.select:
        picks = select_volumes(args.work, args.n)
        print(",".join(p["volume_id"] for p in picks))
        for p in picks:
            print(f"  {p['volume_id']}  {p['rows']}x{p['cols']}x{p['n_slices']}  "
                  f"{int(p['n_voxels']) / 1e6:.0f} Mvox", file=sys.stderr)
        return 0
    return report(args.work)


if __name__ == "__main__":
    sys.exit(main())
