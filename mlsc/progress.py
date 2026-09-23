#!/usr/bin/env python3
"""progress.py — where the run stands and when it will finish.

Reads manifest/volumes.csv, the per-volume JSONL records under logs/, and the
outputs on disk, then prints per-model done / remaining / failed counts, any
error messages (deduplicated), staging decisions, seconds per volume observed
so far, and an ETA computed from the remaining work, the observed mean, and
the number of GPU tasks running (--running, which run_pipeline.sh fills from
squeue). Read-only; safe on the login node.
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_cohort import output_paths  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("--models", required=True)
    ap.add_argument("--running", type=int, default=0, help="GPU tasks currently running.")
    ap.add_argument("--concurrency", type=int, default=6, help="GPU_CONCURRENCY (used when --running is 0).")
    args = ap.parse_args(argv)
    models = args.models.replace(",", " ").split()
    work = args.work

    # staging
    idx_path = os.path.join(work, "manifest", "series_index.csv")
    if os.path.isfile(idx_path):
        with open(idx_path, newline="") as f:
            series = list(csv.DictReader(f))
        dec = collections.Counter(r["decision"] for r in series)
        reasons = collections.Counter(r["reasons"].split("=")[0] for r in series if r["decision"] != "staged")
        cases = len({r["case_id"] for r in series})
        print(f"staging: {cases} cases, {len(series)} series → "
              + ", ".join(f"{k}={v}" for k, v in sorted(dec.items())))
        print("   non-staged reasons: " + ", ".join(f"{k}={v}" for k, v in reasons.most_common()))
        failed_stage = [r for r in series if r["decision"] == "failed"]
        for r in failed_stage[:10]:
            print(f"   FAILED {r['case_id']} {r['series_number']} {r['series_description']}: {r['reasons']}")
    vol_path = os.path.join(work, "manifest", "volumes.csv")
    if not os.path.isfile(vol_path):
        print("no manifest/volumes.csv yet (build not run)")
        return 0
    with open(vol_path, newline="") as f:
        volumes = list(csv.DictReader(f))

    # per-volume records
    recs = []
    for p in glob.glob(os.path.join(work, "logs", "predict-*.jsonl")):
        with open(p) as f:
            for line in f:
                try:
                    recs.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    ok = [r for r in recs if r.get("status") == "ok"]
    failed = [r for r in recs if r.get("status") == "failed"]
    secs_by_model = collections.defaultdict(list)
    for r in ok:
        secs_by_model[r["model"]].append(r["total_seconds"])
    all_secs = [r["total_seconds"] for r in ok]

    print(f"\n{'model':20} {'done':>5} {'remain':>6} {'failed':>6} {'s/vol':>6}")
    remaining_total = 0
    for m in models:
        done = sum(1 for v in volumes if all(os.path.isfile(p) for p in output_paths(v, m)))
        remaining = len(volumes) - done
        remaining_total += remaining
        nf = sum(1 for r in failed if r["model"] == m)
        s = secs_by_model.get(m)
        mean = sum(s) / len(s) if s else None
        print(f"{m:20} {done:>5} {remaining:>6} {nf:>6} {'-' if mean is None else round(mean):>6}")
    print(f"{'total':20} {len(volumes) * len(models) - remaining_total:>5} {remaining_total:>6} {len(failed):>6}")

    if failed:
        # Only count a failure if the volume still lacks outputs (retries may have fixed it).
        errs = collections.Counter()
        for r in failed:
            v = next((v for v in volumes if v["volume_id"] == r["volume_id"]), None)
            if v and all(os.path.isfile(p) for p in output_paths(v, r["model"])):
                continue
            errs[(r["model"], r.get("error", "")[:160])] += 1
        if errs:
            print("\nunresolved failures:")
            for (m, e), n in errs.most_common():
                print(f"   {n}x {m}: {e}")
        else:
            print("\nall logged failures were resolved by later retries")

    if all_secs:
        mean = sum(all_secs) / len(all_secs)
        slots = args.running or args.concurrency
        eta_s = remaining_total * mean / max(1, slots)
        last = max(r["ts"] for r in recs)
        print(f"\nobserved: {len(ok)} volumes done, mean {mean:.0f}s/vol, max {max(all_secs):.0f}s, "
              f"last record {last}")
        print(f"ETA for {remaining_total} remaining at {slots} GPU task(s): "
              f"~{datetime.timedelta(seconds=int(eta_s))} "
              f"({'from squeue' if args.running else 'assuming GPU_CONCURRENCY'})"
              + (" — nothing running; is the predict array queued?" if not args.running and remaining_total else ""))
    else:
        print("\nno completed volumes logged yet")
    return 0


if __name__ == "__main__":
    sys.exit(main())
