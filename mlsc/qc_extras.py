#!/usr/bin/env python3
"""qc_extras.py — fold staging metadata into idc-batch-qc.py's qc.csv.

idc-batch-qc.py knows nothing about where a volume came from. This joins
manifest/volumes.csv (by volume_id == qc case_id) onto cohort/qc/<model>/qc.csv
and appends columns a reviewer wants to sort on: series description, kernel,
slice thickness, z spacing, matrix, spectral kind, geometry flags, tags.
LNQReview ignores columns it doesn't list, so this is safe to run every time.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

EXTRA = [("series_description", "series_description"), ("kernel", "kernel"),
         ("slice_thickness_mm", "slice_thickness_mm"), ("z_spacing_mm", "z_spacing_mm"),
         ("rows", "matrix"), ("n_slices", "n_slices"), ("spectral", "spectral"),
         ("flags", "geometry_flags"), ("tags", "series_tags"), ("case_id", "case")]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", required=True)
    ap.add_argument("--models", required=True)
    args = ap.parse_args(argv)

    with open(os.path.join(args.work, "manifest", "volumes.csv"), newline="") as f:
        volumes = {r["volume_id"]: r for r in csv.DictReader(f)}

    for model in args.models.replace(",", " ").split():
        path = os.path.join(args.work, "cohort", "qc", model, "qc.csv")
        if not os.path.isfile(path):
            print(f"{model}: no qc.csv yet, skipping")
            continue
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
            fields = list(reader.fieldnames or [])
        for _, out_name in EXTRA:
            if out_name not in fields:
                fields.append(out_name)
        n = 0
        for r in rows:
            v = volumes.get(r.get("case_id", ""))
            if not v:
                continue
            n += 1
            for src, dst in EXTRA:
                r[dst] = v.get(src, "")
        with open(path + ".partial", "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
        os.replace(path + ".partial", path)
        print(f"{model}: {n}/{len(rows)} rows annotated → {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
