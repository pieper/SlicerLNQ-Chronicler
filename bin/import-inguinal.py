#!/usr/bin/env python3
"""Import Syed's inguinal lymph-node annotations into the Chronicle.

Reads:
  --segs-dir : directory of case_NNNN.nrrd (CT) + case_NNNN-inguinal.seg.nrrd (SEG)
  --xlsx     : Seg-retro6-notes-clinical.xlsx with the per-case review state
               (columns: caseID, sex, primaryCondition, trainingCase, notes,
                Inguinal LN Seg, Checked by Tagwa, Second review by Tagwa)

Writes:
  Blob doc per CT + SEG file (sha256-keyed).
  Cohort      : 'LNQ-retro (inguinal Syed)'.
  CohortResolution v1 of that cohort, with 69 cases (one per spreadsheet row that has a CT on disk).
  Protocol    : 'Inguinal lymph node segmentation v1'.
  Project     : 'Inguinal LN retro1', binding new cohort + new protocol; admin=admin, pieper as admin/annotator/reviewer, syed as annotator, tagwa as reviewer.
  Annotation per row that has a SEG on disk, status mapped from the spreadsheet:
    Second review = OK                       -> approved
    Checked = OK                             -> approved
    Checked = 'Updated by tagwa'             -> approved
    Checked = specific feedback              -> needs_changes (Tagwa note in .notes)
    Inguinal LN Seg = Done, no Checked       -> submitted_for_review
    Inguinal LN Seg empty / **               -> in_progress (file exists) or todo (no file)

Re-running is mostly idempotent: Blob docs upserts add the same location, but
Cohort/Protocol/Project/Annotation docs would duplicate. Use --reset to wipe and
re-create those project-scoped docs. The retro Cohort + Annotations remain untouched.

Reads chronicle.conf for DOMAIN_NAME and COUCHDB_ADMIN_PASSWORD.
"""

from __future__ import annotations

import argparse
import datetime
import glob
import hashlib
import json
import os
import re
import socket
import sys
import uuid

try:
    import requests
except ImportError:
    print("Need 'requests'.", file=sys.stderr); sys.exit(1)
try:
    import openpyxl
except ImportError:
    print("Need 'openpyxl' to parse the XLSX.", file=sys.stderr); sys.exit(1)


PROJECT_NAME = "Inguinal LN retro1"
PROTOCOL_NAME = "Inguinal lymph node segmentation v1"
COHORT_NAME = "LNQ-retro (inguinal Syed)"
RESOLUTION_NAME = "Inguinal Syed import"
PRODUCER_LABEL = "syed-inguinal-v1"


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


def hash_file(path, chunk=4 * 1024 * 1024):
    h = hashlib.sha256(); size = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b); size += len(b)
    return h.hexdigest(), size


class Chronicle:
    def __init__(self, base_url, db, password, host_label=None):
        self.base = base_url.rstrip("/")
        self.db = db
        self.s = requests.Session()
        self.s.auth = ("admin", password)
        self.host_label = host_label or socket.gethostname()

    def get(self, doc_id):
        r = self.s.get(f"{self.base}/{self.db}/{doc_id}")
        if r.status_code == 404:
            return None
        r.raise_for_status()
        return r.json()

    def put(self, doc):
        r = self.s.put(f"{self.base}/{self.db}/{doc['_id']}", json=doc,
                       headers={"Content-Type": "application/json"})
        if not r.ok:
            try:
                msg = r.json().get("reason", r.text)
            except Exception:
                msg = r.text
            raise RuntimeError(f"PUT {doc['_id']} failed: {r.status_code} {msg}")
        return r.json()

    def list_by_type(self, type_name):
        prefix = {
            "Cohort": "cohort:", "Protocol": "protocol:", "Project": "project:",
            "CohortResolution": "cohortresolution:", "Annotation": "annotation:",
            "Blob": "blob:", "ModelGeneration": "modelgeneration:", "InferenceRun": "inferencerun:",
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

    def register_blob(self, local_path, actor, mime_type=None):
        sha256, size = hash_file(local_path)
        blob_id = f"blob:{sha256}"
        location = {
            "kind": "local-uri",
            "value": f"file://{os.path.abspath(local_path)}",
            "host": self.host_label,
            "verified_at": iso_now(),
        }
        existing = self.get(blob_id)
        if existing is None:
            doc = {
                "_id": blob_id, "type": "Blob",
                "name": f"{sha256[:8]}… ({size} bytes)",
                "created_at": iso_now(), "created_by": actor,
                "version": 1, "predecessor": None,
                "sha256": sha256, "size": size,
                "mime_type": mime_type or _guess_mime(local_path),
                "locations": [location],
            }
            self.put(doc)
            return doc, "created"
        # Update locations if not already present.
        locs = list(existing.get("locations") or [])
        for i, l in enumerate(locs):
            if l.get("kind") == location["kind"] and l.get("value") == location["value"]:
                locs[i] = location
                existing["locations"] = locs
                self.put(existing)
                return existing, "refreshed"
        locs.append(location)
        existing["locations"] = locs
        self.put(existing)
        return existing, "appended"


def _guess_mime(path):
    low = path.lower()
    if low.endswith(".seg.nrrd"):
        return "application/octet-stream+seg.nrrd"
    if low.endswith(".nrrd"):
        return "application/octet-stream+nrrd"
    return None


def map_status(row) -> tuple[str, str]:
    """Returns (status, notes). row is a dict keyed by header text.
    Notes are formatted so the dashboard's _parse_reviewer_from_notes can
    extract the reviewer identity — always include a 'Tagwa: ...' prefix
    when Tagwa has reviewed, even if the verdict is just OK."""
    second = (row.get("Second review by Tagwa") or "").strip()
    checked = (row.get("Checked by Tagwa") or "").strip()
    done = (row.get("Inguinal LN Seg") or "").strip()
    note_parts = []
    if checked:
        note_parts.append(f"Tagwa: {checked}")
    if second:
        note_parts.append(f"2nd review: {second}")
    notes = "\n".join(note_parts)

    if second.upper() == "OK":
        return "approved", notes
    if checked.upper() == "OK":
        return "approved", notes
    if checked.lower().startswith("updated by tagwa"):
        return "approved", notes
    if checked:  # any other non-empty text = specific feedback
        return "needs_changes", notes
    if done.lower() == "done":
        return "submitted_for_review", notes
    return "in_progress", notes


def parse_xlsx(path):
    wb = openpyxl.load_workbook(path, data_only=True)
    ws = wb.active
    headers = [c.value for c in ws[1]]
    rows = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        if not row[0]:
            continue
        d = {headers[i]: (row[i] if row[i] is None else str(row[i]).strip().strip('"'))
             for i in range(len(headers))}
        d["caseID"] = d.get("caseID")
        rows.append(d)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--segs-dir", required=True)
    ap.add_argument("--xlsx", required=True)
    ap.add_argument("--actor", default="pieper")
    ap.add_argument("--conf", default="chronicle.conf")
    ap.add_argument("--host-label", default=None,
                    help="Label written into Blob locations (default: hostname).")
    ap.add_argument("--reset", action="store_true",
                    help="Delete any existing inguinal Cohort/Protocol/Project/CohortResolution before importing. Blobs are kept.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not os.path.isfile(args.conf):
        print(f"Missing {args.conf}", file=sys.stderr); sys.exit(1)
    conf = load_conf(args.conf)
    chronicle = Chronicle(
        f"https://{conf.get('DOMAIN_NAME', 'lnq-chronicle.isomics.dev')}",
        "lnq", conf["COUCHDB_ADMIN_PASSWORD"],
        host_label=args.host_label,
    )

    rows = parse_xlsx(args.xlsx)
    print(f"spreadsheet rows: {len(rows)}")

    # Pair each row with its on-disk CT + SEG (if present).
    case_files = {}
    for r in rows:
        cid = r["caseID"]
        ct = os.path.join(args.segs_dir, f"{cid}.nrrd")
        seg = os.path.join(args.segs_dir, f"{cid}-inguinal.seg.nrrd")
        case_files[cid] = {
            "ct": ct if os.path.isfile(ct) else None,
            "seg": seg if os.path.isfile(seg) else None,
        }
    have_ct = sum(1 for v in case_files.values() if v["ct"])
    have_seg = sum(1 for v in case_files.values() if v["seg"])
    print(f"on disk: {have_ct}/{len(rows)} CTs, {have_seg}/{len(rows)} SEGs")

    if args.reset and not args.dry_run:
        # Delete existing project-scoped docs (cohort/resolution/protocol/project + their annotations).
        # Blobs are kept (they're the catalog).
        # Find by name.
        for tname, name in (("Project", PROJECT_NAME), ("Protocol", PROTOCOL_NAME),
                            ("Cohort", COHORT_NAME)):
            for d in chronicle.list_by_type(tname):
                if d.get("name") == name:
                    r = chronicle.s.delete(f"{chronicle.base}/{chronicle.db}/{d['_id']}",
                                            params={"rev": d["_rev"]})
                    print(f"  deleted {tname} {d['_id']}")
        # Annotations + CohortResolutions referencing the about-to-be-deleted project/cohort.
        proj_ids = []
        for p in chronicle.list_by_type("Project"):
            if p.get("name") == PROJECT_NAME:
                proj_ids.append(p["_id"])
        # Delete annotations belonging to those projects (legacy after deletion).
        for ann in chronicle.list_by_type("Annotation"):
            if ann.get("project_id") in proj_ids:
                chronicle.s.delete(f"{chronicle.base}/{chronicle.db}/{ann['_id']}",
                                    params={"rev": ann["_rev"]})
        for res in chronicle.list_by_type("CohortResolution"):
            if res.get("name") == RESOLUTION_NAME:
                chronicle.s.delete(f"{chronicle.base}/{chronicle.db}/{res['_id']}",
                                    params={"rev": res["_rev"]})

    # ---- Register blobs ----
    blob_for_ct = {}    # case_id -> blob_id
    blob_for_seg = {}   # case_id -> blob_id
    ct_new = ct_app = 0
    seg_new = seg_app = 0
    for cid, paths in case_files.items():
        if paths["ct"]:
            if args.dry_run:
                blob_for_ct[cid] = "blob:" + "0" * 64
            else:
                doc, action = chronicle.register_blob(paths["ct"], actor=args.actor)
                blob_for_ct[cid] = doc["_id"]
                ct_new += (action == "created"); ct_app += (action != "created")
        if paths["seg"]:
            if args.dry_run:
                blob_for_seg[cid] = "blob:" + "0" * 64
            else:
                doc, action = chronicle.register_blob(paths["seg"], actor=args.actor)
                blob_for_seg[cid] = doc["_id"]
                seg_new += (action == "created"); seg_app += (action != "created")
    if not args.dry_run:
        print(f"blobs — CT: {ct_new} new, {ct_app} location updated; "
              f"SEG: {seg_new} new, {seg_app} location updated")

    # ---- Cohort ----
    cohort_id = f"cohort:{uuid.uuid4()}"
    cohort = {
        "_id": cohort_id, "type": "Cohort", "name": COHORT_NAME,
        "created_at": iso_now(), "created_by": args.actor,
        "version": 1, "predecessor": None,
        "description": "Inguinal-LN cohort sourced from Syed's annotation working set; same case IDs as the retro6 cohort but distinct file backing.",
        "sources": [{
            "kind": "local-uri",
            "value": f"file://{os.path.abspath(args.segs_dir)}",
        }],
    }
    if not args.dry_run:
        chronicle.put(cohort)
    print(f"Cohort: {cohort_id}")

    # ---- CohortResolution ----
    resolution_id = f"cohortresolution:{uuid.uuid4()}"
    cases = []
    for r in rows:
        cid = r["caseID"]
        case_entry = {
            "case_id": cid,
            "study_uid": None,
            "patient_id": None,
            "primary_site": r.get("primaryCondition"),
            "sex": r.get("sex"),
            "ct_ref": ({"blob_id": blob_for_ct[cid]} if blob_for_ct.get(cid) else None),
            "metadata": {
                "trainingCase": (r.get("trainingCase") or "").lower() == "true",
                "spreadsheet_inguinal_status": r.get("Inguinal LN Seg"),
                "spreadsheet_tagwa": r.get("Checked by Tagwa"),
                "spreadsheet_tagwa_2": r.get("Second review by Tagwa"),
            },
        }
        cases.append(case_entry)
    resolution = {
        "_id": resolution_id, "type": "CohortResolution", "name": RESOLUTION_NAME,
        "created_at": iso_now(), "created_by": args.actor,
        "version": 1, "predecessor": None,
        "cohort_id": cohort_id,
        "resolved_at": iso_now(),
        "resolver": "import-inguinal-v1",
        "cases": cases,
    }
    if not args.dry_run:
        chronicle.put(resolution)
    print(f"CohortResolution: {resolution_id} ({len(cases)} cases)")

    # ---- Protocol ----
    protocol_id = f"protocol:{uuid.uuid4()}"
    protocol = {
        "_id": protocol_id, "type": "Protocol", "name": PROTOCOL_NAME,
        "created_at": iso_now(), "created_by": args.actor,
        "version": 1, "predecessor": None,
        "description": "Single foreground label: inguinal lymph node. SOP: include bilateral inguinal LNs; exclude external iliac, vessels, and other non-target structures.",
        "color_table": [
            {"label": 1, "name": "Inguinal lymph node",
             "color": [240, 220, 60],
             "code": {"scheme": "SCT", "value": "39937001", "meaning": "Inguinal lymph node"}},
        ],
        "rules": ("Inclusion: bilateral inguinal LNs (deep + superficial groups). "
                  "Exclusion: external iliac chain, vessels, fat lobules, scar. "
                  "Ambiguity: prefer under-call to over-call; flag for consultation."),
    }
    if not args.dry_run:
        chronicle.put(protocol)
    print(f"Protocol: {protocol_id}")

    # ---- Project ----
    project_id = f"project:{uuid.uuid4()}"
    project = {
        "_id": project_id, "type": "Project", "name": PROJECT_NAME,
        "created_at": iso_now(), "created_by": args.actor,
        "version": 1, "predecessor": None,
        "description": "Inguinal LN training/review project. Source annotations from Syed; reviewed by Tagwa per Seg-retro6-notes-clinical.xlsx.",
        "cohort_id": cohort_id,
        "protocol_id": protocol_id,
        "members": [
            {"user": "admin",  "role": "admin"},
            {"user": "pieper", "role": "admin"},
            {"user": "pieper", "role": "annotator"},
            {"user": "pieper", "role": "reviewer"},
            {"user": "syed",   "role": "annotator"},
            {"user": "tagwa",  "role": "reviewer"},
        ],
    }
    if not args.dry_run:
        chronicle.put(project)
    print(f"Project: {project_id}")

    # ---- Annotations ----
    status_counts = {}
    skipped_no_seg = 0
    for r in rows:
        cid = r["caseID"]
        if not blob_for_seg.get(cid):
            skipped_no_seg += 1
            continue
        status, notes = map_status(r)
        status_counts[status] = status_counts.get(status, 0) + 1
        ann = {
            "_id": f"annotation:{uuid.uuid4()}",
            "type": "Annotation",
            "name": f"{cid} / {PRODUCER_LABEL}",
            "created_at": iso_now(), "created_by": args.actor,
            "version": 1, "predecessor": None,
            "project_id": project_id,
            "case_id": cid,
            "study_uid": None,
            "status": status,
            "notes": notes,
            "producer": {
                "kind": "manual",
                "label": PRODUCER_LABEL,
                "model_generation_id": None,
            },
            "seg_ref": {"blob_id": blob_for_seg[cid]},
        }
        if not args.dry_run:
            chronicle.put(ann)
    print(f"Annotations: {sum(status_counts.values())} written, {skipped_no_seg} skipped (no SEG on disk)")
    for s, n in sorted(status_counts.items()):
        print(f"  {s}: {n}")

    print()
    print("Done.")
    if args.dry_run:
        print("(dry-run; no docs written)")


if __name__ == "__main__":
    main()
