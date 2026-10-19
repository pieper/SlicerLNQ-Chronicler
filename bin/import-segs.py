#!/usr/bin/env python3
"""Bulk-import a directory of segmentation files as Annotations on a Project.

For each file matching `case_*.nrrd` or `case_*.seg.nrrd` under --segs-dir, create
one new Annotation on --project-id with seg_ref pointing at the file. The case_id
is parsed from the filename (`case_0042.nrrd` -> `case_0042`). Each new
annotation chains off the existing head (predecessor = current latest), so the
discipline is preserved and prior history is not lost.

Typical use — landing 20 manually-segmented inguinal LN cases as the training
truth on a new project:

  bin/import-segs.py \\
      --project-id project:<uuid> \\
      --segs-dir /private/tmp/media/share/LNQ-data/Inguinal-LN/manual \\
      --label manual-v1 \\
      --producer-kind manual \\
      --status approved

Reads chronicle.conf for DOMAIN_NAME and COUCHDB_ADMIN_PASSWORD.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import re
import sys
import uuid

try:
    import requests
except ImportError:
    print("Need 'requests'. Run from a venv: "
          "python3 -m venv /tmp/v && /tmp/v/bin/pip install requests",
          file=sys.stderr)
    sys.exit(1)


VALID_STATUSES = {
    "todo", "in_progress", "submitted_for_review",
    "needs_changes", "approved", "needs_consultation",
}
VALID_PRODUCER_KINDS = {"manual", "model", "review"}
CASE_ID_RE = re.compile(r"^(case_\d+)")


def iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_conf(path: str) -> dict:
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


class Chronicle:
    def __init__(self, base_url: str, db: str, password: str):
        self.base = base_url.rstrip("/")
        self.db = db
        self.s = requests.Session()
        self.s.auth = ("admin", password)

    def get(self, doc_id: str) -> dict:
        r = self.s.get(f"{self.base}/{self.db}/{doc_id}")
        r.raise_for_status()
        return r.json()

    def put(self, doc: dict) -> dict:
        r = self.s.put(
            f"{self.base}/{self.db}/{doc['_id']}",
            json=doc, headers={"Content-Type": "application/json"})
        if not r.ok:
            try:
                msg = r.json().get("reason", r.text)
            except Exception:
                msg = r.text
            raise RuntimeError(f"PUT {doc['_id']} failed: {r.status_code} {msg}")
        return r.json()

    def list_by_type(self, type_name: str) -> list:
        prefix = {
            "Cohort": "cohort:", "Protocol": "protocol:", "Project": "project:",
            "CohortResolution": "cohortresolution:", "Annotation": "annotation:",
            "ModelGeneration": "modelgeneration:", "InferenceRun": "inferencerun:",
        }[type_name]
        r = self.s.get(
            f"{self.base}/{self.db}/_all_docs",
            params={
                "startkey": json.dumps(prefix),
                "endkey": json.dumps(prefix + "￰"),
                "include_docs": "true",
            })
        r.raise_for_status()
        return [row["doc"] for row in r.json().get("rows", [])
                if row.get("doc") and not row["id"].startswith("_design")]


def parse_case_id(filename: str) -> str | None:
    """Pull `case_NNNN` out of a filename. Returns None if the name doesn't match."""
    base = os.path.basename(filename)
    # Strip extension(s); accept .nrrd and .seg.nrrd both.
    stem = base
    for ext in (".seg.nrrd", ".nrrd"):
        if stem.endswith(ext):
            stem = stem[:-len(ext)]
            break
    m = CASE_ID_RE.match(stem)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-id", required=True,
                    help="Target Project doc _id (project:...)")
    ap.add_argument("--segs-dir", required=True,
                    help="Directory containing case_*.nrrd / case_*.seg.nrrd files")
    ap.add_argument("--label", required=True,
                    help="Producer label, e.g. 'manual-v1', 'inguinal-retro1'")
    ap.add_argument("--status", default="approved", choices=sorted(VALID_STATUSES),
                    help="Status to record on each new Annotation (default: approved)")
    ap.add_argument("--producer-kind", default="manual", choices=sorted(VALID_PRODUCER_KINDS),
                    help="producer.kind on each new Annotation (default: manual)")
    ap.add_argument("--model-generation-id", default=None,
                    help="Optional ModelGeneration ref to record in producer.model_generation_id (use with --producer-kind=model)")
    ap.add_argument("--actor", default=None,
                    help="created_by user (default: $USER or 'admin')")
    ap.add_argument("--conf", default="chronicle.conf",
                    help="Path to chronicle.conf (default: chronicle.conf)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Show what would be created; do not write")
    args = ap.parse_args()

    if not os.path.isfile(args.conf):
        print(f"Missing {args.conf}; run from SlicerLNQ-Chronicler/", file=sys.stderr)
        sys.exit(1)
    conf = load_conf(args.conf)
    domain = conf.get("DOMAIN_NAME") or "lnq-chronicle.isomics.dev"
    password = conf["COUCHDB_ADMIN_PASSWORD"]

    actor = args.actor or os.environ.get("USER") or "admin"
    chronicle = Chronicle(f"https://{domain}", "lnq", password)

    # Verify project exists, gather valid case_ids from its CohortResolution.
    project = chronicle.get(args.project_id)
    if project.get("type") != "Project":
        print(f"--project-id {args.project_id} is not a Project (got type={project.get('type')})", file=sys.stderr)
        sys.exit(1)
    cohort_id = project["cohort_id"]
    resolutions = [r for r in chronicle.list_by_type("CohortResolution")
                   if r.get("cohort_id") == cohort_id]
    if not resolutions:
        print(f"No CohortResolution found for cohort {cohort_id}; create one before importing.",
              file=sys.stderr)
        sys.exit(1)
    resolutions.sort(key=lambda r: (r.get("resolved_at") or r.get("created_at") or ""))
    resolution = resolutions[-1]
    cases_by_id = {c["case_id"]: c for c in resolution.get("cases", [])}

    # Latest annotation per case in this project (for predecessor chain).
    annotations = [a for a in chronicle.list_by_type("Annotation")
                   if a.get("project_id") == args.project_id]
    by_case_chain = {}
    for ann in annotations:
        by_case_chain.setdefault(ann["case_id"], []).append(ann)
    for k in by_case_chain:
        by_case_chain[k].sort(key=lambda a: a.get("version", 0))

    # Find files.
    patterns = [f"{args.segs_dir}/case_*.seg.nrrd", f"{args.segs_dir}/case_*.nrrd"]
    files = sorted({p for pat in patterns for p in glob.glob(pat)})
    if not files:
        print(f"No case_*.nrrd or case_*.seg.nrrd files under {args.segs_dir}", file=sys.stderr)
        sys.exit(1)

    # Build + write.
    created = 0
    skipped_unknown_case = 0
    skipped_dup = 0
    for f in files:
        case_id = parse_case_id(f)
        if not case_id:
            print(f"  ? cannot parse case_id from {f}; skipping")
            continue
        if case_id not in cases_by_id:
            print(f"  ! case_id {case_id} not in cohort resolution; skipping {f}")
            skipped_unknown_case += 1
            continue
        chain = by_case_chain.get(case_id, [])
        head = chain[-1] if chain else None
        # Idempotency: skip if head already references this exact path with this label.
        if head and head.get("seg_ref") and head["seg_ref"].get("value") == f"file://{f}" \
                and head.get("producer", {}).get("label") == args.label:
            print(f"  = head already matches for {case_id}; skipping")
            skipped_dup += 1
            continue
        new_doc = {
            "_id": f"annotation:{uuid.uuid4()}",
            "type": "Annotation",
            "name": f"{case_id} / {args.label}",
            "created_at": iso_now(),
            "created_by": actor,
            "version": (head.get("version", 0) + 1) if head else 1,
            "predecessor": head["_id"] if head else None,
            "project_id": args.project_id,
            "case_id": case_id,
            "study_uid": cases_by_id[case_id].get("study_uid"),
            "status": args.status,
            "notes": "",
            "producer": {
                "kind": args.producer_kind,
                "label": args.label,
                "model_generation_id": args.model_generation_id,
            },
            "seg_ref": {
                "kind": "local-uri",
                "value": f"file://{os.path.abspath(f)}",
                "sha256": None,
                "size": os.path.getsize(f),
            },
        }
        if args.dry_run:
            print(f"  + would write {new_doc['_id']} for {case_id}  pred={head['_id'] if head else None}")
            continue
        chronicle.put(new_doc)
        print(f"  + {case_id}  v{new_doc['version']}  -> {new_doc['_id']}")
        created += 1

    print()
    print(f"Created: {created}, dup-skipped: {skipped_dup}, "
          f"unknown-case-skipped: {skipped_unknown_case}, files-seen: {len(files)}")


if __name__ == "__main__":
    main()
