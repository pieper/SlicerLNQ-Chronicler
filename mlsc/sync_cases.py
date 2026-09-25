#!/usr/bin/env python3
"""sync_cases.py — copy the pipeline output case by case with rclone.

Whole cases are transferred in E-number order so the first ones are complete
(and viewable) while the rest are still in flight. Idempotent: every case is
handed to `rclone copy`, which sends only what is missing or changed, so a
re-run after an interruption resumes where it stopped. Only the real files
move: E*/ trees, manifest/, cohort/qc/, logs/, mlsc.conf. The cohort/nrrd and
cohort/predictions symlink views are never copied; in --pull mode they are
rebuilt locally (build_cohort.py --refresh-links) after every case.

    # on Martinos (upload)
    sync_cases.py --work /vast/lnq/pdac-processing --remote dropbox:PDAC --push
    # on the Mac (download)
    sync_cases.py --work /Volumes/enc/pdac-processing --remote dropbox:PDAC --pull
    # what is complete so far (no transfer)
    sync_cases.py --work ... --remote ... --push --status

Prints, after each case, the cases completed so far, GB moved, measured
throughput and an ETA for the full set. State (per-case bytes / seconds /
completion time) is kept in <work>/manifest/sync-<push|pull>.json.
Standard library + rclone only.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import subprocess
import sys
import time

CASE_RE = re.compile(r"^E\d{8}$")
SHARED = ["manifest", "cohort/qc", "logs", "mlsc.conf"]
SHARED_EXCLUDES = ["manifest/chunks/**", "manifest/sync-*.json"]


def sh(args, stream=True):
    """Run rclone; stream its (stats) lines with a prefix; return exit code."""
    proc = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    last = ""
    for line in proc.stdout:
        line = line.rstrip()
        if not line:
            continue
        last = line
        if stream:
            print("   rclone: " + line, flush=True)
    proc.wait()
    return proc.returncode, last


def rclone_copy(src, dst, extra=(), transfers=8, stats="30s"):
    args = ["rclone", "copy", "--skip-links", "--transfers", str(transfers), "--checkers", "16",
            "--dropbox-chunk-size", "128M", "--stats", stats, "--stats-one-line",
            "--stats-log-level", "NOTICE", *extra, src, dst]
    return sh(args)


def local_cases(work):
    return sorted(d for d in os.listdir(work) if CASE_RE.match(d) and os.path.isdir(os.path.join(work, d)))


def remote_cases(remote):
    out = subprocess.run(["rclone", "lsjson", "--dirs-only", remote], capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(f"rclone lsjson {remote} failed: {out.stderr.strip()}")
    return sorted(e["Name"] for e in json.loads(out.stdout) if CASE_RE.match(e["Name"]))


def local_bytes(path):
    total = 0
    for dirpath, _, files in os.walk(path):
        for f in files:
            p = os.path.join(dirpath, f)
            if not os.path.islink(p):
                total += os.stat(p).st_size
    return total


def remote_bytes(remote):
    out = subprocess.run(["rclone", "size", "--json", remote], capture_output=True, text=True)
    if out.returncode != 0:
        return 0
    return int(json.loads(out.stdout).get("bytes", 0))


def gb(n):
    return f"{n / 1e9:.1f} GB"


def hms(seconds):
    return str(datetime.timedelta(seconds=int(max(0, seconds))))


def load_state(path):
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {"cases": {}}


def save_state(path, state):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".partial", "w") as f:
        json.dump(state, f, indent=1)
    os.replace(path + ".partial", path)


def print_table(cases, sizes, state, current=None):
    done_b = sum(sizes[c] for c in cases if c in state["cases"])
    total_b = sum(sizes.values())
    print(f"\n{'case':10} {'size':>9} {'status':10} {'took':>8}")
    for c in cases:
        st = state["cases"].get(c)
        status = "complete" if st else ("copying" if c == current else "pending")
        took = hms(st["seconds"]) if st else ""
        print(f"{c:10} {gb(sizes[c]):>9} {status:10} {took:>8}")
    n_done = sum(1 for c in cases if c in state["cases"])
    print(f"complete: {n_done}/{len(cases)} cases, {gb(done_b)} of {gb(total_b)}")
    return done_b, total_b


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True, help="Local pipeline tree (source for --push, destination for --pull).")
    ap.add_argument("--remote", required=True, help="rclone remote path holding the tree, e.g. dropbox:PDAC")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--push", action="store_true", help="local → remote")
    mode.add_argument("--pull", action="store_true", help="remote → local")
    ap.add_argument("--cases", default=None, help="Comma-separated subset / explicit order.")
    ap.add_argument("--first", type=int, default=None, help="Only the first N cases (in order).")
    ap.add_argument("--transfers", type=int, default=8)
    ap.add_argument("--status", action="store_true", help="Print the table and exit; no transfer.")
    ap.add_argument("--models", default="mediastinal-v1 axillary-v1 inguinal-v1 abdominopelvic-v1",
                    help="For rebuilding cohort links after each pulled case.")
    args = ap.parse_args(argv)

    push = args.push
    remote = args.remote.rstrip("/")
    state_path = os.path.join(args.work, "manifest", f"sync-{'push' if push else 'pull'}.json")
    state = load_state(state_path)

    cases = local_cases(args.work) if push else remote_cases(remote)
    if args.cases:
        wanted = [c.strip() for c in args.cases.split(",") if c.strip()]
        missing = [c for c in wanted if c not in cases]
        if missing:
            raise SystemExit(f"not in source: {missing}")
        cases = wanted
    if args.first:
        cases = cases[: args.first]
    if not cases:
        raise SystemExit("no E######## case directories found in the source")

    print(f"{'push' if push else 'pull'}: {len(cases)} cases  {args.work} {'→' if push else '←'} {remote}")
    print("measuring case sizes …", flush=True)
    sizes = {}
    for c in cases:
        sizes[c] = local_bytes(os.path.join(args.work, c)) if push else remote_bytes(f"{remote}/{c}")
    print_table(cases, sizes, state)
    if args.status:
        return 0

    def shared_copy(label):
        print(f"\n== {label}: manifest / qc / logs", flush=True)
        excl = []
        for e in SHARED_EXCLUDES:
            excl += ["--exclude", e]
        for item in SHARED:
            src = os.path.join(args.work, item) if push else f"{remote}/{item}"
            dst = f"{remote}/{item}" if push else os.path.join(args.work, item)
            if push and not os.path.exists(src):
                continue
            if item.endswith(".conf"):
                dst = os.path.dirname(dst)
                rc, _ = sh(["rclone", "copy", src, dst])
            else:
                rc, _ = rclone_copy(src, dst, extra=excl, transfers=4, stats="0")
            if rc == 3:                      # source directory/file not present: fine
                continue
            if rc != 0:
                print(f"   WARN: copying {item} returned {rc}", flush=True)

    def refresh_links():
        if push:
            return
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        try:
            import build_cohort
            build_cohort.main(["--work", args.work, "--models", args.models, "--refresh-links"])
        except Exception as exc:  # noqa: BLE001
            print(f"   WARN: cohort link refresh failed: {exc}", flush=True)

    shared_copy("first")
    session_bytes = 0
    session_seconds = 0.0
    t_start = time.time()
    for i, c in enumerate(cases, 1):
        src = os.path.join(args.work, c) if push else f"{remote}/{c}"
        dst = f"{remote}/{c}" if push else os.path.join(args.work, c)
        already = c in state["cases"]
        print(f"\n== [{i}/{len(cases)}] {c} ({gb(sizes[c])}){' — verifying, marked complete earlier' if already else ''}",
              flush=True)
        t0 = time.time()
        rc, last = rclone_copy(src, dst, transfers=args.transfers)
        dt = time.time() - t0
        if rc != 0:
            print(f"   FAILED (rclone exit {rc}): {last}\n   re-run to resume; stopping here.", flush=True)
            save_state(state_path, state)
            return 1
        if push:
            complete = remote_bytes(dst) >= sizes[c]
        else:
            complete = local_bytes(dst) >= sizes[c]
        if not complete:
            print("   WARN: size on destination smaller than source after copy; will retry on next run", flush=True)
            continue
        if not already or dt > 15:
            state["cases"][c] = {"bytes": sizes[c], "seconds": round(dt, 1),
                                 "done_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
            if dt > 15:                      # a real transfer, not a verify pass
                session_bytes += sizes[c]
                session_seconds += dt
        save_state(state_path, state)
        refresh_links()
        done_b, total_b = print_table(cases, sizes, state)
        if session_seconds > 0:
            rate = session_bytes / session_seconds
            remaining = total_b - done_b
            print(f"throughput this session: {rate / 1e6:.0f} MB/s over {hms(session_seconds)}   "
                  f"remaining: {gb(remaining)}   ETA full set: {hms(remaining / rate)} "
                  f"(~{(datetime.datetime.now() + datetime.timedelta(seconds=remaining / rate)).strftime('%H:%M')})",
                  flush=True)
    shared_copy("final")
    refresh_links()
    print(f"\nall done in {hms(time.time() - t_start)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
