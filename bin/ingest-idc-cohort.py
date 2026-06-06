#!/usr/bin/env python3
"""ingest-idc-cohort.py — IDC collection → Chronicle cohort + batch inference.

Pipeline (all stages idempotent — re-runs pick up where they left off):

  1. Query idc-index for CT + (optional) SEG series in the collection,
     filtered by body part.
  2. Download every series to Manila staging via idc_index.download_from_selection.
  3. Convert each CT DICOM directory to <case_id>_0000.nrrd (SimpleITK GDCM).
  4. Convert each DICOM SEG to <case_id>.nrrd resampled onto the CT grid
     (pydicom_seg + sitk.Resample with nearest neighbour).
  5. Chronicle:
       - Register CT NRRD + GT SEG NRRD as Blobs.
       - Create / find Cohort + CohortResolution by name.
       - For each case with a GT, emit an Annotation with producer.kind="manual"
         and producer.label="NIH-IDC".
  6. Inference (skippable):
       - For each case in the cohort:
         - Run lnq_segmenter.predict(model, ct_nrrd, out_seg, probability_output=...)
         - Register SEG + probability NRRDs as Blobs.
         - Emit a model Annotation linked to a single ModelGeneration created for
           this run (producer.kind="model", producer.label=model_spec).

Typical invocation on lnq-inferencer:

  /opt/lnq/nnenv/bin/python /opt/lnq/ingest-idc-cohort.py \
      --collection ct_lymph_nodes \
      --body-part MEDIASTINUM \
      --model mediastinal-v1 \
      --stage-root /media/share/LNQ-data/idc \
      --conf /opt/lnq/chronicle.conf

Useful flags:
  --limit N           stop after N cases (smoke testing)
  --skip-inference    do stages 1-5 only
  --no-chronicle      do stages 1-4 only (file staging dry-run)
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import shutil
import socket
import sys
import time
import traceback
import uuid


# ----- conf / chronicle helpers -----

def iso_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_kv_conf(path):
    out = {}
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln: continue
            k, _, v = ln.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def hash_file(path, chunk=1 << 20):
    h = hashlib.sha256(); size = 0
    with open(path, "rb") as f:
        for buf in iter(lambda: f.read(chunk), b""):
            h.update(buf); size += len(buf)
    return h.hexdigest(), size


class Chronicle:
    def __init__(self, base_url, db, password, host_label=None):
        import requests
        self.base = base_url.rstrip("/"); self.db = db
        self.s = requests.Session(); self.s.auth = ("admin", password)
        self.host_label = host_label or socket.gethostname()

    def get(self, doc_id):
        r = self.s.get(f"{self.base}/{self.db}/{doc_id}", timeout=30)
        return r.json() if r.status_code == 200 else None

    def put(self, doc):
        r = self.s.put(f"{self.base}/{self.db}/{doc['_id']}", json=doc,
                       headers={"Content-Type": "application/json"}, timeout=30)
        if not r.ok:
            try: msg = r.json().get("reason", r.text)
            except Exception: msg = r.text
            raise RuntimeError(f"PUT {doc['_id']} failed: {r.status_code} {msg}")
        return r.json()

    def list_by_type(self, type_name):
        prefix = {
            "Cohort": "cohort:", "Protocol": "protocol:", "Project": "project:",
            "CohortResolution": "cohortresolution:", "Annotation": "annotation:",
            "Blob": "blob:", "ModelGeneration": "modelgeneration:",
        }[type_name]
        r = self.s.get(
            f"{self.base}/{self.db}/_all_docs",
            params={"startkey": json.dumps(prefix),
                    "endkey": json.dumps(prefix + "￰"),
                    "include_docs": "true"},
            timeout=60)
        r.raise_for_status()
        return [row["doc"] for row in r.json().get("rows", [])
                if row.get("doc") and not row["id"].startswith("_design")]

    def register_blob(self, local_path, actor, mime_type=None):
        sha256, size = hash_file(local_path)
        blob_id = f"blob:{sha256}"
        location = {"kind": "local-uri",
                    "value": f"file://{os.path.abspath(local_path)}",
                    "host": self.host_label, "verified_at": iso_now()}
        existing = self.get(blob_id)
        if existing is None:
            doc = {"_id": blob_id, "type": "Blob",
                   "name": f"{sha256[:8]}… ({size} bytes)",
                   "created_at": iso_now(), "created_by": actor,
                   "version": 1, "predecessor": None,
                   "sha256": sha256, "size": size,
                   "mime_type": mime_type or "application/octet-stream",
                   "locations": [location]}
            self.put(doc)
            return doc, "created"
        # Append location if not already present.
        locs = list(existing.get("locations") or [])
        if not any(l.get("value") == location["value"] for l in locs):
            locs.append(location)
            existing["locations"] = locs
            self.put(existing)
            return existing, "appended"
        return existing, "exists"


# ----- IDC discovery + download -----

def discover_cases(collection_id, body_part=None, limit=None):
    """Return [(patient_id, ct_seriesUID, [seg_seriesUID, ...]), ...]."""
    import idc_index
    client = idc_index.IDCClient()
    idx = client.index
    sub = idx[idx["collection_id"] == collection_id]
    if body_part:
        sub = sub[sub["BodyPartExamined"] == body_part.upper()]
    cases = []
    for patient_id, group in sub.groupby("PatientID"):
        cts = group[group["Modality"] == "CT"]
        segs = group[group["Modality"] == "SEG"]
        if cts.empty:
            continue
        # Pick the CT with the most slices (in case multiple).
        ct_row = cts.sort_values("instanceCount", ascending=False).iloc[0]
        cases.append((patient_id,
                      ct_row["SeriesInstanceUID"],
                      list(segs["SeriesInstanceUID"]),
                      ct_row["StudyInstanceUID"]))
    cases.sort(key=lambda t: t[0])
    return cases[:limit] if limit else cases


def download_series(client, series_uids, dest_dir):
    """Wrap idc_index.download_from_selection. Idempotent — s5cmd skips
    series whose target tree already exists."""
    os.makedirs(dest_dir, exist_ok=True)
    client.download_from_selection(
        downloadDir=dest_dir,
        seriesInstanceUID=list(series_uids),
        quiet=True, show_progress_bar=False)


# ----- DICOM → NRRD conversion -----

def dicom_dir_to_nrrd(dicom_dir, out_nrrd):
    """Convert a CT DICOM directory to NRRD via SimpleITK's GDCM series reader."""
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    files = reader.GetGDCMSeriesFileNames(dicom_dir)
    if not files:
        raise RuntimeError(f"no DICOM series in {dicom_dir}")
    reader.SetFileNames(files)
    img = reader.Execute()
    sitk.WriteImage(img, out_nrrd, useCompression=True)
    return img


def seg_dcm_to_nrrd_on_ct(seg_dcm, ct_img, out_nrrd):
    """Convert a DICOM SEG → binary NRRD resampled onto the CT grid.
    Combines all segments via OR (single-label foreground)."""
    import pydicom, pydicom_seg, numpy as np, SimpleITK as sitk
    ds = pydicom.dcmread(seg_dcm)
    res = pydicom_seg.SegmentReader().read(ds)
    combined = None; ref = None
    for n in res.available_segments:
        img = res.segment_image(n)
        arr = sitk.GetArrayFromImage(img)
        if combined is None:
            combined = arr.astype(np.uint8); ref = img
        elif arr.shape == combined.shape:
            combined |= arr.astype(np.uint8)
    if combined is None:
        return None
    native = sitk.GetImageFromArray((combined > 0).astype(np.uint8))
    native.CopyInformation(ref)
    resampled = sitk.Resample(native, ct_img, sitk.Transform(),
                              sitk.sitkNearestNeighbor, 0, native.GetPixelID())
    sitk.WriteImage(resampled, out_nrrd, useCompression=True)
    return out_nrrd


def case_id_from_patient(patient_id):
    """Normalise IDC PatientID to a case_id usable as a filename / chronicle key.
    NIH MED_LYMPH_021 passes through; other collections may need sanitising."""
    return patient_id


# ----- main pipeline -----

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--collection", required=True,
                    help="IDC collection_id (e.g. ct_lymph_nodes).")
    ap.add_argument("--body-part", default=None,
                    help="Optional BodyPartExamined filter (MEDIASTINUM, ABDOMEN, ...).")
    ap.add_argument("--cohort-name", default=None,
                    help="Chronicle Cohort name. Default: '<collection> <body_part>'.")
    ap.add_argument("--stage-root", default="/media/share/LNQ-data/idc",
                    help="Manila path where per-case dirs are written.")
    ap.add_argument("--conf", default="/opt/lnq/chronicle.conf")
    ap.add_argument("--actor", default="ingest-idc")
    ap.add_argument("--model", default="mediastinal-v1",
                    help="lnq-segmenter model name for batch inference.")
    ap.add_argument("--protocol-name", default=None,
                    help="Chronicle Protocol name to reuse. Required to write "
                    "Annotations (schema rejects null project_id, which forces "
                    "Project, which references a Protocol).")
    ap.add_argument("--project-name", default=None,
                    help="Chronicle Project name. Default: cohort_name + ' review'.")
    ap.add_argument("--model-version", default=None,
                    help="lnq-segmenter model version. Defaults to registry latest.")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many cases (smoke testing).")
    ap.add_argument("--skip-inference", action="store_true",
                    help="Do stages 1-5; skip the batch predict.")
    ap.add_argument("--no-chronicle", action="store_true",
                    help="Do stages 1-4; skip chronicle writes.")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    conf = load_kv_conf(args.conf)
    base_url = conf.get("CHRONICLE_URL") or ("https://" + conf["DOMAIN_NAME"])
    chronicle = None if args.no_chronicle else Chronicle(
        base_url, conf.get("CHRONICLE_DB", "lnq"), conf["COUCHDB_ADMIN_PASSWORD"],
        host_label="lnq-inferencer")

    # ----- Stage 1: discover -----
    cohort_name = args.cohort_name or (
        f"IDC {args.collection}" + (f" {args.body_part.lower()}" if args.body_part else ""))
    stage_dir = os.path.join(args.stage_root, args.collection)
    nrrd_dir = os.path.join(stage_dir, "nrrd")
    os.makedirs(nrrd_dir, exist_ok=True)

    print(f"[stage 1] discovering IDC cases in {args.collection}"
          + (f" ({args.body_part})" if args.body_part else ""), flush=True)
    cases = discover_cases(args.collection, args.body_part, args.limit)
    print(f"[stage 1] {len(cases)} cases", flush=True)

    # ----- Stage 2-4 per case: download + convert -----
    import idc_index
    idc_client = idc_index.IDCClient()
    per_case = []
    for i, (patient_id, ct_uid, seg_uids, study_uid) in enumerate(cases, 1):
        cid = case_id_from_patient(patient_id)
        ct_nrrd  = os.path.join(nrrd_dir, f"{cid}_0000.nrrd")
        seg_nrrd = os.path.join(nrrd_dir, f"{cid}.nrrd")
        case_dl = os.path.join(stage_dir, patient_id)

        if not os.path.isfile(ct_nrrd):
            print(f"[{i:3d}/{len(cases)}] {cid}: download + convert CT", flush=True)
            # idc-index applies dirTemplate
            # "%collection_id/%PatientID/%StudyInstanceUID/%Modality_%SeriesInstanceUID"
            # under the downloadDir, so let it create that tree from stage_root.
            download_series(idc_client, [ct_uid] + seg_uids, args.stage_root)
            # Recursive glob: any depth under stage_root finds the CT/SEG dirs.
            import glob
            ct_dirs = glob.glob(
                f"{args.stage_root}/**/CT_{ct_uid}", recursive=True)
            if not ct_dirs:
                print(f"  WARN no CT dir matching CT_{ct_uid} under "
                      f"{args.stage_root}", flush=True)
                continue
            ct_img = dicom_dir_to_nrrd(ct_dirs[0], ct_nrrd)
        else:
            print(f"[{i:3d}/{len(cases)}] {cid}: CT NRRD cached", flush=True)
            import SimpleITK as sitk
            ct_img = sitk.ReadImage(ct_nrrd)

        # GT SEG conversion (optional — collection may not have one)
        if seg_uids and not os.path.isfile(seg_nrrd):
            import glob
            seg_dirs = []
            for u in seg_uids:
                seg_dirs += glob.glob(
                    f"{args.stage_root}/**/SEG_{u}", recursive=True)
            seg_dirs = sorted(set(seg_dirs))
            if seg_dirs:
                seg_dcms = sorted(glob.glob(f"{seg_dirs[0]}/*.dcm"))
                if seg_dcms:
                    try:
                        seg_dcm_to_nrrd_on_ct(seg_dcms[0], ct_img, seg_nrrd)
                    except Exception as exc:
                        print(f"  WARN SEG conversion failed: {exc}", flush=True)

        per_case.append({
            "case_id": cid, "patient_id": patient_id,
            "study_uid": study_uid, "ct_series_uid": ct_uid,
            "ct_nrrd": ct_nrrd, "seg_nrrd": seg_nrrd if os.path.isfile(seg_nrrd) else None,
        })

    # ----- Stage 5: chronicle Cohort + CohortResolution + GT Annotations -----
    cohort_id = resolution_id = None
    if chronicle is not None:
        print(f"\n[stage 5] chronicle writes (Cohort + CohortResolution + GT)", flush=True)
        existing_cohorts = {c.get("name"): c for c in chronicle.list_by_type("Cohort")}
        if cohort_name in existing_cohorts:
            cohort_id = existing_cohorts[cohort_name]["_id"]
            print(f"  Cohort exists: {cohort_id}", flush=True)
        else:
            cohort_id = f"cohort:{uuid.uuid4()}"
            chronicle.put({
                "_id": cohort_id, "type": "Cohort", "name": cohort_name,
                "created_at": iso_now(), "created_by": args.actor,
                "version": 1, "predecessor": None,
                "description": (f"IDC collection {args.collection}"
                                + (f", body part {args.body_part}" if args.body_part else "")
                                + f". Ingested via ingest-idc-cohort.py at {iso_now()}."),
                # Cohort schema requires kind in (idc-query|dicomweb|local-uri).
                "sources": [{"kind": "idc-query",
                             "value": args.collection,
                             "body_part": args.body_part}],
            })
            print(f"  Cohort: {cohort_id}", flush=True)

        # Register CT + GT Blobs, build CohortResolution cases list.
        cases_payload = []
        for c in per_case:
            ct_blob, _ = chronicle.register_blob(c["ct_nrrd"], actor=args.actor,
                                                  mime_type="application/x-nrrd")
            gt_blob = None
            if c["seg_nrrd"]:
                gt_blob, _ = chronicle.register_blob(c["seg_nrrd"], actor=args.actor,
                                                     mime_type="application/x-nrrd")
            cases_payload.append({
                "case_id": c["case_id"],
                "study_uid": c["study_uid"],
                "series_uid": c["ct_series_uid"],
                "patient_id": c["patient_id"],
                "ct_ref": {"blob_id": ct_blob["_id"]},
                "metadata": {"source": f"IDC {args.collection}",
                             "body_part": args.body_part,
                             "has_idc_seg": bool(gt_blob)},
            })
            c["ct_blob_id"] = ct_blob["_id"]
            c["gt_blob_id"] = gt_blob["_id"] if gt_blob else None

        # Find or create CohortResolution.
        existing_res = [r for r in chronicle.list_by_type("CohortResolution")
                        if r.get("cohort_id") == cohort_id]
        if existing_res:
            resolution_id = max(existing_res,
                                key=lambda r: r.get("created_at") or "")["_id"]
            print(f"  CohortResolution exists: {resolution_id}", flush=True)
        else:
            resolution_id = f"cohortresolution:{uuid.uuid4()}"
            chronicle.put({
                "_id": resolution_id, "type": "CohortResolution",
                "name": f"Resolution of {cohort_name} ({len(cases_payload)} cases)",
                "created_at": iso_now(), "created_by": args.actor,
                "version": 1, "predecessor": None,
                "cohort_id": cohort_id, "resolved_at": iso_now(),
                "resolver": "ingest-idc-cohort-v1",
                "cases": cases_payload,
            })
            print(f"  CohortResolution: {resolution_id} ({len(cases_payload)} cases)",
                  flush=True)

        # Find/create Protocol — Annotations require a Project, which requires
        # a Protocol. Reuse an existing Protocol by name if --protocol-name was
        # passed; otherwise we mint a minimal one named after the cohort.
        protocols = {p.get("name"): p for p in chronicle.list_by_type("Protocol")}
        if args.protocol_name and args.protocol_name in protocols:
            protocol_id = protocols[args.protocol_name]["_id"]
            print(f"  Protocol exists: {protocol_id} ({args.protocol_name!r})", flush=True)
        else:
            protocol_name = args.protocol_name or f"{cohort_name} protocol"
            if protocol_name in protocols:
                protocol_id = protocols[protocol_name]["_id"]
                print(f"  Protocol exists: {protocol_id}", flush=True)
            else:
                protocol_id = f"protocol:{uuid.uuid4()}"
                chronicle.put({
                    "_id": protocol_id, "type": "Protocol", "name": protocol_name,
                    "created_at": iso_now(), "created_by": args.actor,
                    "version": 1, "predecessor": None,
                    "description": (f"Auto-created by ingest-idc-cohort.py for the "
                                    f"{cohort_name} review pipeline. "
                                    f"Single foreground label: lymph node."),
                    "color_table": [{"label": 1, "name": "Lymph node",
                                     "color": [200, 100, 230],
                                     "code": {"scheme": "SCT", "value": "59441001",
                                              "meaning": "Lymph node"}}],
                    "rules": ("Inclusion: clinically-detectable lymph nodes per "
                              "the source dataset's inclusion criteria. "
                              "Reviewer-corrected annotations are training-eligible "
                              "iff quality_flag != 'rejected'."),
                })
                print(f"  Protocol: {protocol_id} ({protocol_name!r})", flush=True)

        # Find/create Project — annotation.project_id must point at one.
        project_name = args.project_name or f"{cohort_name} review"
        projects = {p.get("name"): p for p in chronicle.list_by_type("Project")}
        if project_name in projects:
            project_id = projects[project_name]["_id"]
            print(f"  Project exists: {project_id}", flush=True)
        else:
            project_id = f"project:{uuid.uuid4()}"
            chronicle.put({
                "_id": project_id, "type": "Project", "name": project_name,
                "created_at": iso_now(), "created_by": args.actor,
                "version": 1, "predecessor": None,
                "description": (f"Review project against the IDC {args.collection} "
                                f"cohort. Pre-computed model annotations seed the "
                                f"reviewer's per-case starting point; corrected "
                                f"annotations flow back here with "
                                f"producer.kind == 'review'."),
                "cohort_id": cohort_id,
                "protocol_id": protocol_id,
                "members": [
                    {"user": "admin",  "role": "admin"},
                    {"user": "pieper", "role": "admin"},
                    {"user": "pieper", "role": "reviewer"},
                    {"user": "tagwa",  "role": "reviewer"},
                ],
            })
            print(f"  Project: {project_id} ({project_name!r})", flush=True)

        # GT Annotations (per case with a SEG).
        existing_anns = chronicle.list_by_type("Annotation")
        gt_keys = {(a.get("case_id"), (a.get("producer") or {}).get("label")): a
                    for a in existing_anns
                    if a.get("project_id") == project_id}
        gt_new = gt_existing = 0
        for c in per_case:
            if not c["gt_blob_id"]: continue
            key = (c["case_id"], "NIH-IDC")
            if key in gt_keys:
                gt_existing += 1; continue
            ann_id = f"annotation:{uuid.uuid4()}"
            chronicle.put({
                "_id": ann_id, "type": "Annotation",
                "name": f"{c['case_id']} / NIH-IDC",
                "created_at": iso_now(), "created_by": args.actor,
                "version": 1, "predecessor": None,
                "project_id": project_id,
                "case_id": c["case_id"], "study_uid": c["study_uid"],
                "status": "submitted_for_review",
                "producer": {"kind": "manual", "label": "NIH-IDC",
                             "model_generation_id": None},
                "seg_ref": {"blob_id": c["gt_blob_id"]},
                "notes": "IDC-distributed ground-truth segmentation.",
            })
            gt_new += 1
        print(f"  GT Annotations: {gt_new} new, {gt_existing} pre-existing", flush=True)

    # ----- Stage 6: batch inference -----
    if args.skip_inference:
        print("\n[stage 6] skipped (--skip-inference)", flush=True)
        return

    print(f"\n[stage 6] batch inference {args.model}"
          + (f"@{args.model_version}" if args.model_version else ""), flush=True)

    # Need a ModelGeneration to anchor model annotations.
    mg_id = None
    if chronicle is not None:
        from lnq_segmenter import registry as _reg
        entry = _reg.get_model(args.model, args.model_version)
        mg_name = f"{args.model}@{entry['version']} on {cohort_name}"
        existing_mgs = {m.get("name"): m for m in chronicle.list_by_type("ModelGeneration")}
        if mg_name in existing_mgs:
            mg_id = existing_mgs[mg_name]["_id"]
            print(f"  ModelGeneration exists: {mg_id}", flush=True)
        else:
            mg_id = f"modelgeneration:{uuid.uuid4()}"
            # ModelGeneration validator requires non-empty training_annotation_ids;
            # for a published lnq-segmenter model that wasn't trained here, point
            # at the GT annotations we just registered (they're its evaluation
            # set, not its training set, but the field expects ann refs and this
            # is the closest semantically-truthful option).
            ga = [a["_id"] for a in chronicle.list_by_type("Annotation")
                  if a.get("project_id") == project_id
                  and (a.get("producer") or {}).get("label") == "NIH-IDC"]
            # GitHub release URL for the meta zip; the validator accepts kind=https.
            weights_url = entry["weights_url_template"].format(
                name=entry["name"], version=entry["version"],
                filename=f"{entry['name']}-{entry['version']}-meta.zip")
            chronicle.put({
                "_id": mg_id, "type": "ModelGeneration",
                "name": mg_name,
                "created_at": iso_now(), "created_by": args.actor,
                "version": 1, "predecessor": None,
                "project_id": project_id,
                "label": args.model,
                "training_annotation_ids": ga or [],
                "weights_ref": {"kind": "https", "value": weights_url},
                "training_started_at": iso_now(),
                "training_finished_at": iso_now(),
                "framework": "lnq-segmenter",
                "notes": (f"Batch-inference run via ingest-idc-cohort.py on cohort "
                          f"{cohort_id}. Source: pieper/lnq-segmenter registry "
                          f"entry {entry['name']}@{entry['version']}."),
            })
            print(f"  ModelGeneration: {mg_id}", flush=True)

    from lnq_segmenter import predict as _predict

    # Per-case predict.
    pred_dir = os.path.join(stage_dir, "predictions", args.model)
    os.makedirs(pred_dir, exist_ok=True)
    succ = skip = fail = 0
    for i, c in enumerate(per_case, 1):
        out_seg  = os.path.join(pred_dir, f"{c['case_id']}.nrrd")
        out_prob = os.path.join(pred_dir, f"{c['case_id']}-prob.nrrd")
        if os.path.isfile(out_seg) and os.path.isfile(out_prob):
            skip += 1
            print(f"[{i:3d}/{len(per_case)}] {c['case_id']}: cached", flush=True)
        else:
            t0 = time.time()
            try:
                _predict.predict(
                    args.model, c["ct_nrrd"], out_seg,
                    version=args.model_version,
                    device=args.device,
                    probability_output=out_prob,
                )
                succ += 1
                dt = round(time.time() - t0, 1)
                print(f"[{i:3d}/{len(per_case)}] {c['case_id']}: predict ok ({dt}s)",
                      flush=True)
            except Exception as exc:
                fail += 1
                print(f"[{i:3d}/{len(per_case)}] {c['case_id']}: FAIL {exc}",
                      flush=True)
                traceback.print_exc()
                continue

        if chronicle is None: continue
        # Register output Blobs + model Annotation.
        seg_blob,  _ = chronicle.register_blob(out_seg,  actor=args.actor,
                                                mime_type="application/x-nrrd")
        prob_blob, _ = chronicle.register_blob(out_prob, actor=args.actor,
                                                mime_type="application/x-nrrd")
        ann_id = f"annotation:{uuid.uuid4()}"
        chronicle.put({
            "_id": ann_id, "type": "Annotation",
            "name": f"{c['case_id']} / {args.model}",
            "created_at": iso_now(), "created_by": args.actor,
            "version": 1, "predecessor": None,
            "project_id": project_id,
            "case_id": c["case_id"], "study_uid": c["study_uid"],
            "status": "submitted_for_review",
            "producer": {"kind": "model", "label": args.model,
                         "model_generation_id": mg_id},
            "seg_ref": {"blob_id": seg_blob["_id"]},
            "probability_ref": {"blob_id": prob_blob["_id"]},
            "notes": (f"Batch inference output. Seed for the review tab "
                      f"(producer.model_generation_id pins the seed model)."),
        })

    print(f"\n[done] predicts: {succ} ok, {skip} cached, {fail} failed", flush=True)


if __name__ == "__main__":
    main()
