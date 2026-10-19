#!/usr/bin/env python3
"""Bootstrap importer: ingest the LNQ-retro on-disk dataset into the Chronicle.

Reads:
  <data_root>/LNQ-retro/retro6/inputs/case_*_0000.nrrd     -> the cohort
  <data_root>/LNQ-retro/segmentations/case_*.nrrd          -> manual (retro1)
  <data_root>/LNQ-retro/retroN/retroN-segmentations/...    -> model output for each N
  <data_root>/Mediastinal-Lymph-Node-SEG-VER1-CLINICAL-1.csv  -> clinical metadata
  <data_root>/LNQ-retro/retro6/Seg-retro6-notes.json (optional) -> review state

Writes one Cohort, one CohortResolution, one Protocol, one Project, and one
Annotation per (case, retro-generation) pair, chained via predecessor in the
order [manual, retro2, retro3, ...] so the chain reflects the actual evolution.

Run:
  bin/import-lnq-retro.py [--base /private/tmp/media/share/LNQ-data] \
                          [--reset] [--annotator pieper]

--reset wipes the Chronicle's `lnq` database and re-deploys the design doc
before importing. Use during development; do NOT use against shared data.

Reads chronicle.conf for DOMAIN_NAME and COUCHDB_ADMIN_PASSWORD.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import glob
import json
import os
import subprocess
import sys
import uuid

try:
    import requests
except ImportError:
    print("Need 'requests'. Run from a venv: python3 -m venv /tmp/v && /tmp/v/bin/pip install requests", file=sys.stderr)
    sys.exit(1)


def iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_conf(path: str) -> dict:
    """Crude .conf parser — just enough to get DOMAIN_NAME and COUCHDB_ADMIN_PASSWORD."""
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

    def database_exists(self) -> bool:
        r = self.s.head(f"{self.base}/{self.db}")
        return r.status_code == 200

    def ensure_db(self):
        r = self.s.put(f"{self.base}/{self.db}")
        if r.status_code not in (201, 202, 412):
            r.raise_for_status()

    def delete_db(self):
        r = self.s.delete(f"{self.base}/{self.db}")
        if r.status_code not in (200, 404):
            r.raise_for_status()

    def put(self, doc: dict) -> dict:
        r = self.s.put(
            f"{self.base}/{self.db}/{doc['_id']}",
            json=doc,
            headers={"Content-Type": "application/json"},
        )
        if not r.ok:
            try:
                msg = r.json().get("reason", r.text)
            except Exception:
                msg = r.text
            raise RuntimeError(f"PUT {doc['_id']} failed: {r.status_code} {msg}")
        return r.json()


def base_doc(type_name: str, prefix: str, name: str, actor: str,
             version: int = 1, predecessor=None, doc_id=None) -> dict:
    return {
        "_id": doc_id or f"{prefix}{uuid.uuid4()}",
        "type": type_name,
        "name": name,
        "created_at": iso_now(),
        "created_by": actor,
        "version": version,
        "predecessor": predecessor,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", default="/private/tmp/media/share/LNQ-data",
                    help="Local mount point of the LNQ-data share.")
    ap.add_argument("--annotator", default=os.environ.get("USER", "anonymous"),
                    help="User to assign as annotator + reviewer on the project (created_by on import docs is always 'admin').")
    ap.add_argument("--reset", action="store_true",
                    help="Drop and recreate the Chronicle database before importing. DEV ONLY.")
    ap.add_argument("--conf", default="./chronicle.conf",
                    help="Path to chronicle.conf.")
    args = ap.parse_args()

    conf = load_conf(args.conf)
    domain = conf["DOMAIN_NAME"]
    password = conf["COUCHDB_ADMIN_PASSWORD"]
    chronicle = Chronicle(f"https://{domain}", "lnq", password)

    if args.reset:
        print(f"--reset: dropping existing 'lnq' database...")
        chronicle.delete_db()
        chronicle.ensure_db()
        # Re-deploy the design doc.
        subprocess.run(["bin/deploy-design.sh"], check=True)
    else:
        chronicle.ensure_db()

    base = args.base
    inputs_dir = f"{base}/LNQ-retro/retro6/inputs"
    seg_manual_dir = f"{base}/LNQ-retro/segmentations"
    retro_dirs = {
        # generation -> seg dir
        "retro2": f"{base}/LNQ-retro/retro2/retro2-segmentations",
        "retro3": f"{base}/LNQ-retro/retro3/retro3-segmentations",
        "retro4": f"{base}/LNQ-retro/retro4/retro4-segmentations",
        "retro5": f"{base}/LNQ-retro/retro5/retro5-segmentations",
        "retro6": f"{base}/LNQ-retro/retro6/retro6-segmentations",
    }
    clinical_csv = f"{base}/Mediastinal-Lymph-Node-SEG-VER1-CLINICAL-1.csv"

    if not os.path.isdir(inputs_dir):
        sys.exit(f"inputs dir not found: {inputs_dir} (use --base to override)")

    # ----- enumerate cases -----
    case_ids = sorted({
        os.path.basename(p).split("_0000.nrrd")[0]
        for p in glob.glob(f"{inputs_dir}/case_*_0000.nrrd")
    })
    print(f"found {len(case_ids)} cases under {inputs_dir}")

    # ----- read clinical metadata -----
    clinical = {}
    if os.path.exists(clinical_csv):
        with open(clinical_csv) as f:
            for row in csv.DictReader(f):
                clinical[row["case"]] = row
        print(f"  + clinical metadata: {len(clinical)} rows")
    else:
        print(f"  ! no clinical CSV at {clinical_csv}")

    # ----- read review notes if present -----
    notes_path = f"{base}/LNQ-retro/retro6/Seg-retro6-notes.json"
    notes = {}
    if os.path.exists(notes_path):
        notes = json.loads(open(notes_path).read())
        print(f"  + review notes: {len(notes)} entries")
    else:
        print(f"  - no review notes at {notes_path} (will not seed reviews)")

    # ----- create Cohort -----
    cohort = base_doc("Cohort", "cohort:", "LNQ retro mediastinal cohort", "admin")
    cohort.update({
        "description": (
            "Imported from /media/share/LNQ-data/LNQ-retro on "
            f"{iso_now()}. Single-source local-uri pointing at the inputs "
            "directory; resolved cases live in the matching CohortResolution."
        ),
        "sources": [{
            "kind": "local-uri",
            "value": inputs_dir,
            "description": "Mediastinal LN cases, nnU-Net naming convention (case_NNNN_0000.nrrd).",
        }],
    })
    chronicle.put(cohort)
    print(f"created Cohort: {cohort['_id']}")

    # ----- create CohortResolution -----
    cases_payload = []
    for case_id in case_ids:
        clin = clinical.get(case_id, {})
        cases_payload.append({
            "case_id": case_id,
            "study_uid": clin.get("StudyInstanceUID") or None,
            "series_uid": clin.get("SeriesInstanceUID") or None,
            "patient_id": None,
            "primary_site": clin.get("PrimaryCondition") or None,
            "sex": clin.get("PatientSex") or None,
            "ct_ref": {
                "kind": "local-uri",
                "value": f"file://{inputs_dir}/{case_id}_0000.nrrd",
                "sha256": None,
                "size": None,
            },
            "metadata": {},
        })
    resolution = base_doc("CohortResolution", "cohortresolution:",
                          f"Resolution of LNQ retro cohort ({len(case_ids)} cases)", "admin")
    resolution.update({
        "cohort_id": cohort["_id"],
        "resolved_at": iso_now(),
        "resolver": "lnq-retro-import-v1",
        "cases": cases_payload,
    })
    chronicle.put(resolution)
    print(f"created CohortResolution: {resolution['_id']}  ({len(cases_payload)} cases)")

    # ----- create Protocol -----
    protocol = base_doc("Protocol", "protocol:", "LNQ retro mediastinal LN segmentation", "admin")
    protocol.update({
        "description": (
            "Single-class segmentation of mediastinal lymph nodes for the LNQ "
            "retro series. Annotators outline lymph nodes per the retro SOP; "
            "all stations collapsed into a single label."
        ),
        "color_table": [{
            "label": 1,
            "name": "Mediastinal lymph node",
            "color": [230, 230, 50],
            "code": {
                "scheme": "SCT",
                "value": "59441001",
                "meaning": "Lymph node",
            },
        }],
        "rules": (
            "Outline the boundary of each enlarged or pathologic mediastinal "
            "lymph node. Combine all nodes into the single 'Mediastinal lymph "
            "node' label. See LNQ-retro SOP for ambiguous-boundary guidance."
        ),
    })
    chronicle.put(protocol)
    print(f"created Protocol: {protocol['_id']}")

    # ----- create Project -----
    members = [
        {"user": "admin", "role": "admin"},
        {"user": args.annotator, "role": "annotator"},
        {"user": args.annotator, "role": "reviewer"},
    ]
    # de-dup if annotator == "admin"
    seen = set()
    members = [m for m in members if (m["user"], m["role"]) not in seen and not seen.add((m["user"], m["role"]))]
    project = base_doc("Project", "project:", "LNQ retro (imported)", "admin")
    project.update({
        "description": (
            "Imported lnq-retro working project. Cohort, Protocol, and case "
            "history all derived from the on-disk LNQ-retro tree."
        ),
        "cohort_id": cohort["_id"],
        "protocol_id": protocol["_id"],
        "members": members,
    })
    chronicle.put(project)
    print(f"created Project: {project['_id']}  members={[m['user']+':'+m['role'] for m in members]}")

    # ----- generation order -----
    # Manual first (when present), then retro2..retro6.
    generations: list[tuple[str, str, str]] = []  # (label, kind, dir)
    generations.append(("manual", "manual", seg_manual_dir))
    for label in ("retro2", "retro3", "retro4", "retro5", "retro6"):
        if os.path.isdir(retro_dirs[label]):
            generations.append((label, "model", retro_dirs[label]))

    # ----- per-case Annotation chain -----
    total_annotations = 0
    skipped_cases_with_no_segs = 0
    for case_id in case_ids:
        prev_id = None
        version = 0
        wrote_any = False
        for label, kind, segdir in generations:
            seg_path = f"{segdir}/{case_id}.nrrd"
            if not os.path.exists(seg_path):
                continue
            version += 1
            # Status discipline: model outputs are "submitted_for_review"; manual seed is "approved"
            # so the chain has a sensible starting point. Reviewer then writes new Annotations
            # to flip status.
            status = "approved" if kind == "manual" else "submitted_for_review"
            ann = base_doc(
                "Annotation",
                "annotation:",
                f"{case_id} / {label}",
                "admin",
                version=version,
                predecessor=prev_id,
            )
            ann.update({
                "project_id": project["_id"],
                "case_id": case_id,
                "study_uid": clinical.get(case_id, {}).get("StudyInstanceUID") or None,
                "status": status,
                "notes": "",
                "producer": {
                    "kind": kind,
                    "label": label,
                    "model_generation_id": None,
                },
                "seg_ref": {
                    "kind": "local-uri",
                    "value": f"file://{seg_path}",
                    "sha256": None,
                    "size": None,
                },
            })
            chronicle.put(ann)
            prev_id = ann["_id"]
            total_annotations += 1
            wrote_any = True

        if not wrote_any:
            skipped_cases_with_no_segs += 1

        # Layer review state on top of the chain (when notes are present).
        review_entry = notes.get(case_id)
        if wrote_any and review_entry:
            state = (review_entry.get("state") or "").strip()
            text = (review_entry.get("text") or "").strip()
            status_map = {
                "good": "approved",
                "bad": "needs_changes",
                "not reviewed": "submitted_for_review",
            }
            if state in status_map:
                version += 1
                review = base_doc(
                    "Annotation",
                    "annotation:",
                    f"{case_id} / review",
                    "admin",
                    version=version,
                    predecessor=prev_id,
                )
                review.update({
                    "project_id": project["_id"],
                    "case_id": case_id,
                    "study_uid": clinical.get(case_id, {}).get("StudyInstanceUID") or None,
                    "status": status_map[state],
                    "notes": text,
                    "producer": {"kind": "review", "label": "import-review", "model_generation_id": None},
                    "seg_ref": None,
                })
                chronicle.put(review)
                total_annotations += 1

    print(f"created {total_annotations} Annotations across {len(case_ids) - skipped_cases_with_no_segs} cases")
    if skipped_cases_with_no_segs:
        print(f"  ({skipped_cases_with_no_segs} cases had no segmentations — skipped)")

    # ----- summary -----
    print()
    print("Done. Summary:")
    print(f"  Cohort:             {cohort['_id']}")
    print(f"  CohortResolution:   {resolution['_id']}")
    print(f"  Protocol:           {protocol['_id']}")
    print(f"  Project:            {project['_id']}")
    print(f"  Annotations:        {total_annotations}")


if __name__ == "__main__":
    main()
