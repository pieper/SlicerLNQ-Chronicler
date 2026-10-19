#!/usr/bin/env python3
"""register-equivalent-volumes.py — register Manila volumes as voxel-equivalent
to chronicle Blobs.

For each NRRD file in --dir whose name yields a case_id, finds the case in the
project's latest CohortResolution, looks up its ct_ref.blob_id, and appends a
location entry to that Blob:

  {"kind": "equivalent-volume",
   "value": "file://<absolute path>",
   "host": "<--host>",
   "file_sha256": "<sha256 of the file on disk>",
   "voxel_sha256": "<sha256 of the decompressed voxel array, C-order>",
   "verified_at": "<iso utc now>"}

The Blob's `_id` is `blob:<file_sha256_of_canonical_upload>`. When an
upstream pipeline re-saves the file (different writer toolchain, different
float-string formatter, different gzip-stream OS byte) the on-disk sha256
stops matching `_id` even though the scientific content is unchanged.
`kind: equivalent-volume` records the file's actual hash AND the voxel-array
hash so the file→blob link captures "different bytes, same volume" honestly,
and lets `resolve_local_path` find it for training.

Idempotent — re-runs skip files already registered at the same path.
"""

import argparse
import datetime
import fnmatch
import hashlib
import os
import re
import sys

import requests

try:
    import SimpleITK as sitk
except ImportError:
    print("Missing SimpleITK (needs /opt/lnq/nnenv/bin/python or similar).",
          file=sys.stderr)
    sys.exit(2)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def voxel_sha256(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)
    return hashlib.sha256(arr.tobytes(order="C")).hexdigest()


def load_conf(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-id", required=True,
                    help="Project _id whose CohortResolution maps case_id -> ct_ref.")
    ap.add_argument("--dir", required=True,
                    help="Directory containing the equivalent volumes.")
    ap.add_argument("--pattern", default="*.nrrd",
                    help="Glob filter for files in --dir (default: *.nrrd).")
    ap.add_argument("--case-id-regex", default=r"(case_\d+)",
                    help="Regex with one group; extracts case_id from each filename.")
    ap.add_argument("--ref-field", default="ct_ref",
                    choices=("ct_ref", "seg_ref"),
                    help="Which case ref field to register against (default ct_ref).")
    ap.add_argument("--conf", default="/opt/lnq/chronicle.conf")
    ap.add_argument("--host", default="manila",
                    help="Tag for the new location entry's 'host' field.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    conf = load_conf(args.conf)
    base = conf.get("CHRONICLE_URL")
    if not base and conf.get("DOMAIN_NAME"):
        base = "https://" + conf["DOMAIN_NAME"]
    if not base:
        print("Missing CHRONICLE_URL / DOMAIN_NAME", file=sys.stderr); sys.exit(2)
    db = conf.get("CHRONICLE_DB", "lnq")
    s = requests.Session()
    s.auth = ("admin", conf["COUCHDB_ADMIN_PASSWORD"])
    iso_now = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ")

    # Build case_id -> blob_id via the project's latest CohortResolution.
    proj = s.get(f"{base}/{db}/{args.project_id}", timeout=15)
    proj.raise_for_status()
    proj = proj.json()
    cohort_id = proj.get("cohort_id")
    if not cohort_id:
        print(f"Project {args.project_id} has no cohort_id", file=sys.stderr)
        sys.exit(2)

    rows = s.get(f"{base}/{db}/_all_docs?include_docs=true", timeout=60).json()["rows"]
    resolutions = [r["doc"] for r in rows
                   if r.get("doc") and r["doc"].get("type") == "CohortResolution"
                   and r["doc"].get("cohort_id") == cohort_id]
    resolutions.sort(key=lambda r: (r.get("resolved_at") or r.get("created_at") or ""))
    if not resolutions:
        print(f"No CohortResolution for cohort {cohort_id}", file=sys.stderr)
        sys.exit(2)
    cases = resolutions[-1].get("cases") or []
    blob_by_case = {c["case_id"]: (c.get(args.ref_field) or {}).get("blob_id")
                    for c in cases}

    re_case = re.compile(args.case_id_regex)
    stats = {"files": 0, "no_case_id": 0, "no_blob_id": 0,
             "matched": 0, "already": 0, "updated": 0,
             "blob_404": 0, "errors": 0}

    for name in sorted(os.listdir(args.dir)):
        if not fnmatch.fnmatch(name, args.pattern):
            continue
        path = os.path.join(args.dir, name)
        if not os.path.isfile(path):
            continue
        stats["files"] += 1
        m = re_case.search(name)
        if not m:
            stats["no_case_id"] += 1
            continue
        case_id = m.group(1)
        blob_id = blob_by_case.get(case_id)
        if not blob_id:
            stats["no_blob_id"] += 1
            continue
        stats["matched"] += 1

        r = s.get(f"{base}/{db}/{blob_id}", timeout=15)
        if r.status_code == 404:
            stats["blob_404"] += 1
            continue
        r.raise_for_status()
        doc = r.json()

        target = "file://" + path
        # Skip if any existing location already points at this path (any kind).
        if any(l.get("value") == target for l in (doc.get("locations") or [])):
            stats["already"] += 1
            continue

        fsha = file_sha256(path)
        vsha = voxel_sha256(path)
        new_loc = {
            "kind": "equivalent-volume",
            "value": target,
            "host": args.host,
            "file_sha256": fsha,
            "voxel_sha256": vsha,
            "verified_at": iso_now,
        }

        if args.dry_run:
            stats["updated"] += 1
            print(f"  WOULD APPEND  {case_id}  vsha={vsha[:12]}  fsha={fsha[:12]}")
            continue

        locs = list(doc.get("locations") or [])
        locs.append(new_loc)
        doc["locations"] = locs
        p = s.put(f"{base}/{db}/{blob_id}", json=doc,
                  headers={"Content-Type": "application/json"}, timeout=15)
        if p.ok:
            stats["updated"] += 1
            print(f"  appended      {case_id}  vsha={vsha[:12]}")
        else:
            stats["errors"] += 1
            print(f"  FAILED {blob_id}: {p.status_code} {p.text[:200]}")

    print(("DRY RUN" if args.dry_run else "APPLIED"))
    for k, v in stats.items():
        print(f"  {k:12s} {v}")


if __name__ == "__main__":
    main()
