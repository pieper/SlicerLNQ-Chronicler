#!/usr/bin/env python3
"""inventory.py — what is in the input tree, before or without staging.

Reads every DICOM header under every E######## case directory and reports
cases, distinct patients, studies, series, files and candidate axial CT
volumes, plus a per-case table with an anonymous patient index (P01, P02, …
in order of first appearance) and the number of days since that patient's
earliest study — enough to see which patients have follow-up (e.g.
post-surgery) studies without writing any identifier or date.

PatientID / PatientName / dates are used in memory only; the written
manifest/inventory.csv contains case id, patient index, day offsets and
counts. Run via `run_pipeline.sh inventory` (a basic-partition job; it
touches every file once).
"""
from __future__ import annotations

import argparse
import csv
import datetime
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stage_dicom import (classify_series, find_case_dirs, read_header,  # noqa: E402
                         series_meta, slices_from_dataset, split_stacks)


def _date(s):
    try:
        return datetime.datetime.strptime(str(s)[:8], "%Y%m%d").date()
    except (TypeError, ValueError):
        return None


def scan_case_for_inventory(case_dir):
    """Like stage_dicom.scan_case but also keeps patient/study keys (in memory)."""
    series = {}
    n_files = 0
    patient_keys = set()
    studies = {}
    for dirpath, dirnames, filenames in os.walk(case_dir):
        dirnames.sort()
        for name in sorted(filenames):
            if name.upper() == "DICOMDIR" or name.startswith("."):
                continue
            ds = read_header(os.path.join(dirpath, name))
            if ds is None or not hasattr(ds, "SOPClassUID"):
                continue
            n_files += 1
            uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
            if not uid:
                continue
            pid = str(getattr(ds, "PatientID", "") or "").strip()
            if pid:
                patient_keys.add(pid)
            study_uid = str(getattr(ds, "StudyInstanceUID", "") or "")
            if study_uid and study_uid not in studies:
                studies[study_uid] = _date(getattr(ds, "StudyDate", ""))
            entry = series.get(uid)
            if entry is None:
                entry = series[uid] = {"meta": series_meta(ds), "slices": [], "n_files": 0}
            entry["n_files"] += 1
            entry["slices"].extend(slices_from_dataset(ds, os.path.join(dirpath, name)))
    return series, n_files, patient_keys, studies


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--min-slices", type=int, default=20)
    args = ap.parse_args(argv)

    case_dirs = find_case_dirs(args.input)
    patient_index = {}          # PatientID -> P01 (memory only)
    patient_first = {}          # PatientID -> earliest study date (memory only)
    rows = []
    totals = {"files": 0, "series": 0, "studies": set(), "candidates": 0, "volumes": 0}
    per_case = []
    for case_dir in case_dirs:
        case_id = os.path.basename(case_dir)
        series, n_files, pids, studies = scan_case_for_inventory(case_dir)
        n_cand = 0
        n_vol = 0
        n_by_kind = {}
        for entry in series.values():
            decision, reasons, tags = classify_series(entry, args.min_slices)
            if decision != "candidate":
                continue
            n_cand += 1
            n_vol += len(split_stacks(entry["slices"]))
            kind = next((t.split("=", 1)[1] for t in tags if t.startswith("spectral=")), "conventional")
            n_by_kind[kind] = n_by_kind.get(kind, 0) + 1
        if len(pids) > 1:
            print(f"WARNING: {case_id} contains {len(pids)} different PatientIDs", file=sys.stderr)
        pid = sorted(pids)[0] if pids else f"?{case_id}"
        if pid not in patient_index:
            patient_index[pid] = f"P{len(patient_index) + 1:02d}"
        dates = [d for d in studies.values() if d]
        if dates:
            first = min(dates)
            patient_first[pid] = min(patient_first.get(pid, first), first)
        per_case.append((case_id, pid, studies, n_files, len(series), n_cand, n_vol, n_by_kind))
        totals["files"] += n_files
        totals["series"] += len(series)
        totals["studies"].update(studies.keys())
        totals["candidates"] += n_cand
        totals["volumes"] += n_vol
        print(f"{case_id}: {n_files} files, {len(series)} series, {n_cand} axial CT series "
              f"({', '.join(f'{k}={v}' for k, v in sorted(n_by_kind.items()))})", flush=True)

    for case_id, pid, studies, n_files, n_series, n_cand, n_vol, n_by_kind in per_case:
        dates = [d for d in studies.values() if d]
        offset = ""
        if dates and pid in patient_first:
            offset = (min(dates) - patient_first[pid]).days
        rows.append({"case_id": case_id, "patient": patient_index[pid],
                     "days_since_patient_first_study": offset,
                     "n_studies": len(studies), "n_series": n_series, "n_files": n_files,
                     "n_axial_ct_series": n_cand, "n_volumes": n_vol,
                     "spectral_kinds": ";".join(f"{k}={v}" for k, v in sorted(n_by_kind.items()))})
    rows.sort(key=lambda r: (r["patient"], r["days_since_patient_first_study"] or 0))

    out = os.path.join(args.work, "manifest", "inventory.csv")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["case_id"])
        wr.writeheader()
        wr.writerows(rows)

    studies_per_patient = {}
    for r in rows:
        studies_per_patient[r["patient"]] = studies_per_patient.get(r["patient"], 0) + r["n_studies"]
    hist = {}
    for n in studies_per_patient.values():
        hist[n] = hist.get(n, 0) + 1
    print()
    print(f"cases (E-dirs): {len(case_dirs)}   patients: {len(patient_index)}   "
          f"studies: {len(totals['studies'])}   series: {totals['series']}   files: {totals['files']}")
    print(f"axial CT series (candidates): {totals['candidates']}   volumes after stack split: {totals['volumes']}")
    print("studies per patient: " + ", ".join(f"{k} {'study' if k == 1 else 'studies'}: {v} patient{'s' if v != 1 else ''}"
                                             for k, v in sorted(hist.items())))
    print()
    print(f"{'case':10} {'patient':7} {'day':>5} {'studies':>7} {'series':>6} {'axialCT':>7} {'volumes':>7}  kinds")
    for r in rows:
        print(f"{r['case_id']:10} {r['patient']:7} {str(r['days_since_patient_first_study']):>5} "
              f"{r['n_studies']:>7} {r['n_series']:>6} {r['n_axial_ct_series']:>7} {r['n_volumes']:>7}  "
              f"{r['spectral_kinds']}")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
