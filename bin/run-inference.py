#!/usr/bin/env python3
"""run-inference.py — 4-fold ensemble inference on a Project's unreviewed cases.

For each case in the Project's latest CohortResolution whose head Annotation is
NOT in --status-not (default: 'approved'), resolves the CT via the Blob's
locations array, symlinks it into an inputs dir under nnU-Net's naming
convention, then runs `nnUNetv2_predict -f 0 1 2 3` against the checkpoints
already on the share. For each output SEG: hashes the bytes, registers a Blob
(or reuses an existing one with that sha), and writes a new Annotation with
producer.kind=model + producer.model_generation_id pointing at the
ModelGeneration doc this run creates. Two distinct --label values (e.g.
'preview-2026-05-31-ep550' now and 'final-2026-06-01' Monday) produce
distinguishable Annotations that the Review tab can compare per case.

Designed to run on a fresh trainer instance (g3.xl or similar) — needs:
  - /media/share/LNQ-data/...  (Manila mounted)
  - /opt/lnq/nnenv  (SimpleITK + numpy + nnunetv2)
  - chronicle reachable (/etc/hosts hairpin, etc.)

Usage:
  bin/run-inference.py \\
    --project-id project:6df... \\
    --label preview-2026-05-31-ep550 \\
    --conf /opt/lnq/chronicle.conf

Idempotent within a single run — re-running with the same --label just
re-creates Annotations referencing the same ModelGeneration (a new MG is
created per invocation; pass --model-generation-id to reuse one).

Stdlib for chronicle; SimpleITK for SEG hashing only at end.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.error
import urllib.request
import uuid


# ----- chronicle HTTP client (stdlib) -----

class Chronicle:
    def __init__(self, base_url, db, password):
        self.base = base_url.rstrip("/"); self.db = db
        token = base64.b64encode(f"admin:{password}".encode()).decode()
        self.auth = f"Basic {token}"

    def _req(self, method, doc_id, body=None, params=""):
        url = f"{self.base}/{self.db}/{doc_id}{params}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Authorization": self.auth,
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            detail = ""
            try: detail = json.loads(e.read()).get("reason", "")
            except Exception: pass
            raise RuntimeError(f"{method} {doc_id}: HTTP {e.code} {detail}") from None

    def get(self, doc_id): return self._req("GET", doc_id)
    def put(self, doc):    return self._req("PUT", doc["_id"], body=doc)

    def list_by_type(self, type_name):
        rows = self._req("GET", "_all_docs", params="?include_docs=true").get("rows", [])
        out = []
        for r in rows:
            d = r.get("doc")
            if d and d.get("type") == type_name:
                out.append(d)
        return out

    def has(self, doc_id):
        try:
            self.get(doc_id); return True
        except RuntimeError as exc:
            if "HTTP 404" in str(exc): return False
            raise


# ----- conf -----

def load_conf(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln:
                continue
            k, _, v = ln.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def iso_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----- CT resolution -----

_LOC_KINDS = ("local-uri", "equivalent-volume")


def resolve_ct_local(blob, base_canonical="/media/share/LNQ-data",
                     base_local="/media/share/LNQ-data"):
    """Walk a Blob's locations, return the first reachable path on this host.
    Mirrors export-nnunet-dataset.py's resolve_local_path (which accepts both
    local-uri and equivalent-volume kinds)."""
    for loc in (blob.get("locations") or []):
        if loc.get("kind") not in _LOC_KINDS:
            continue
        v = loc.get("value") or ""
        if v.startswith("file://"):
            v = v[len("file://"):]
        if base_canonical and base_local and v.startswith(base_canonical):
            cand = base_local + v[len(base_canonical):]
            if os.path.lexists(cand):
                return cand
        if os.path.lexists(v):
            return v
    return None


# ----- per-fold training log inspection (snapshot the moment of inference) -----

EMA_DICE_RE = re.compile(r"New best EMA pseudo Dice:\s*([\d.]+)")
EPOCH_HEAD_RE = re.compile(r":\s*Epoch (\d+)\s*$")


def fold_snapshot(results_root, dataset_dirname, plans_dirname, fold):
    """Return {epoch, best_ema_dice, training_log_path} for the most recent
    training_log_*.txt under fold_X. Best-effort; missing data is None."""
    fold_dir = os.path.join(results_root, dataset_dirname, plans_dirname,
                            f"fold_{fold}")
    logs = sorted([p for p in os.listdir(fold_dir)
                   if p.startswith("training_log_") and p.endswith(".txt")]) \
        if os.path.isdir(fold_dir) else []
    if not logs:
        return {"fold": fold, "epoch": None, "best_ema_dice": None,
                "training_log_path": None}
    log_path = os.path.join(fold_dir, logs[-1])
    epoch = None; best = None
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END); end = f.tell()
            f.seek(max(0, end - 262144))
            data = f.read().decode("utf-8", errors="replace")
        for ln in data.splitlines():
            m = EPOCH_HEAD_RE.search(ln)  # search not match: timestamp precedes ":"
            if m: epoch = int(m.group(1))
            m = EMA_DICE_RE.search(ln)
            if m:
                try: best = float(m.group(1))
                except ValueError: pass
    except OSError:
        pass
    return {"fold": fold, "epoch": epoch, "best_ema_dice": best,
            "training_log_path": log_path}


# ----- Blob registration -----

def ensure_blob(chronicle, abs_path, kind="local-uri", host="manila",
                created_by="run-inference.py"):
    """If a Blob with the file's sha256 already exists, append the Manila
    location if missing and return its _id. Otherwise create a new Blob.
    Returns blob_id ('blob:<sha>')."""
    sha = file_sha256(abs_path)
    blob_id = "blob:" + sha
    size = os.path.getsize(abs_path)
    target = "file://" + abs_path
    if chronicle.has(blob_id):
        doc = chronicle.get(blob_id)
        locs = list(doc.get("locations") or [])
        if not any(l.get("value") == target for l in locs):
            locs.append({"kind": kind, "value": target, "host": host,
                         "verified_at": iso_now()})
            doc["locations"] = locs
            chronicle.put(doc)
        return blob_id
    # Fresh.
    doc = {
        "_id": blob_id,
        "type": "Blob",
        "name": os.path.basename(abs_path),
        "created_at": iso_now(),
        "created_by": created_by,
        "version": 1,
        "predecessor": None,
        "sha256": sha,
        "size": size,
        "mime_type": "application/octet-stream+nrrd",
        "locations": [{
            "kind": kind, "value": target, "host": host,
            "verified_at": iso_now(),
        }],
    }
    chronicle.put(doc)
    return blob_id


# ----- main -----

def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--project-id", required=True)
    ap.add_argument("--label", required=True,
                    help="Identifier baked into producer.label + ModelGeneration.label, "
                         "e.g. 'preview-2026-05-31-ep550' or 'final-2026-06-01'.")
    ap.add_argument("--conf", default="/opt/lnq/chronicle.conf")
    ap.add_argument("--status-not", action="append", default=None,
                    help="Annotation statuses to EXCLUDE (repeatable). "
                         "Default: 'approved' (we don't re-infer the training set).")
    ap.add_argument("--dataset-id", type=int, default=1)
    ap.add_argument("--task-name", default="LNQinguinal")
    ap.add_argument("--folds", default="0,1,2,3",
                    help="Comma-separated fold ids for the ensemble.")
    ap.add_argument("--plans-identifier", default="nnUNetResEncUNetMPlans")
    ap.add_argument("--config-name", default="3d_fullres")
    ap.add_argument("--trainer-class", default="nnUNetTrainer")
    ap.add_argument("--nnunet-base",
                    default="/media/share/LNQ-data/Inguinal_Syed",
                    help="Parent containing nnUNet_raw|preprocessed|results.")
    ap.add_argument("--inputs-dir", default=None,
                    help="If set, use this dir for the symlinked CTs. "
                         "Default: /tmp/lnq-inference-<run-id>/inputs")
    ap.add_argument("--outputs-base",
                    default="/media/share/LNQ-data/Inguinal_Syed/inference_runs",
                    help="Outputs land at <outputs-base>/<label>/outputs/.")
    ap.add_argument("--model-generation-id", default=None,
                    help="If set, reuse this MG; otherwise create one.")
    ap.add_argument("--checkpoint", default="checkpoint_best.pth",
                    help="Which checkpoint file to load per fold. nnU-Net "
                         "default is checkpoint_best.pth (EMA-best). Use "
                         "checkpoint_final.pth for the end-of-training state — "
                         "useful when checkpoint_best.pth got clobbered by a "
                         "post-training restart.")
    ap.add_argument("--per-fold-disagreement", action="store_true",
                    help="Run each fold as a separate predict pass and capture "
                         "ensemble-disagreement statistics: per-case agreement "
                         "Dice (mean pairwise Dice across all C(4,2) fold pairs), "
                         "per-voxel agreement count map (uint8 [0..n_folds]), "
                         "and per-fold predicted volume. The ensemble SEG saved "
                         "as the Annotation's seg_ref is the majority-vote of "
                         "the per-fold SEGs (>=ceil(n_folds/2) agree). Costs "
                         "n_folds times the wall-clock vs. a single ensemble "
                         "predict, but produces a sortable uncertainty scalar "
                         "(low Dice = high disagreement → triage to top).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Plan + symlink only; don't actually call nnUNetv2_predict.")
    args = ap.parse_args()

    status_not = set(args.status_not or ["approved"])
    folds = [int(s) for s in args.folds.split(",") if s.strip()]

    # ----- connect -----
    conf = load_conf(args.conf)
    base = conf.get("CHRONICLE_URL")
    if not base and conf.get("DOMAIN_NAME"):
        base = "https://" + conf["DOMAIN_NAME"]
    if not base:
        print("Missing CHRONICLE_URL / DOMAIN_NAME", file=sys.stderr); sys.exit(2)
    chronicle = Chronicle(base, conf.get("CHRONICLE_DB", "lnq"),
                          conf["COUCHDB_ADMIN_PASSWORD"])

    # ----- target cases -----
    project = chronicle.get(args.project_id)
    if project.get("type") != "Project":
        print(f"{args.project_id} is not a Project", file=sys.stderr); sys.exit(2)

    resolutions = [r for r in chronicle.list_by_type("CohortResolution")
                   if r.get("cohort_id") == project["cohort_id"]]
    if not resolutions:
        print("No CohortResolution for cohort", file=sys.stderr); sys.exit(2)
    resolutions.sort(key=lambda r: (r.get("resolved_at") or r.get("created_at") or ""))
    cases_by_id = {c["case_id"]: c for c in resolutions[-1].get("cases") or []}

    annotations = [a for a in chronicle.list_by_type("Annotation")
                   if a.get("project_id") == args.project_id]
    annotations.sort(key=lambda a: (a.get("case_id") or "", a.get("version") or 0))
    head = {}
    for a in annotations: head[a["case_id"]] = a

    targets = []
    for cid, c in sorted(cases_by_id.items()):
        h = head.get(cid)
        status = (h or {}).get("status")
        if status in status_not:
            continue
        targets.append((cid, c, h))
    print(f"[infer] target cases: {len(targets)}", flush=True)

    # ----- run dir -----
    run_id = str(uuid.uuid4())
    work_dir = args.inputs_dir or f"/tmp/lnq-inference-{run_id}/inputs"
    out_dir = os.path.join(args.outputs_base, args.label, "outputs")
    os.makedirs(work_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    print(f"[infer] run_id={run_id}", flush=True)
    print(f"[infer] inputs:  {work_dir}", flush=True)
    print(f"[infer] outputs: {out_dir}", flush=True)

    # ----- symlink CTs -----
    skipped_no_blob = []; skipped_no_path = []; linked = []
    for cid, c, _h in targets:
        ct_ref = c.get("ct_ref") or {}
        blob_id = ct_ref.get("blob_id")
        if not blob_id:
            skipped_no_blob.append(cid); continue
        try:
            blob = chronicle.get(blob_id)
        except Exception:
            skipped_no_blob.append(cid); continue
        ct_path = resolve_ct_local(blob)
        if not ct_path:
            skipped_no_path.append(cid); continue
        dst = os.path.join(work_dir, f"{cid}_0000.nrrd")
        if os.path.lexists(dst):
            os.unlink(dst)
        os.symlink(ct_path, dst)
        linked.append(cid)
    print(f"[infer] linked {len(linked)}; skipped no_blob={len(skipped_no_blob)} "
          f"no_path={len(skipped_no_path)}", flush=True)
    if skipped_no_path:
        print(f"[infer] skipped (CT not on this host): {skipped_no_path[:8]}",
              flush=True)
    if not linked:
        print("[infer] nothing to do; exiting", file=sys.stderr); sys.exit(2)

    # ----- fold snapshots for provenance -----
    dataset_dirname = f"Dataset{args.dataset_id:03d}_{args.task_name}"
    plans_dirname = f"{args.trainer_class}__{args.plans_identifier}__{args.config_name}"
    results_root = os.path.join(args.nnunet_base, "nnUNet_results")
    fold_snapshots = [fold_snapshot(results_root, dataset_dirname, plans_dirname, f)
                      for f in folds]
    for s in fold_snapshots:
        print(f"[infer] fold {s['fold']}: epoch={s['epoch']} "
              f"best_ema_dice={s['best_ema_dice']}", flush=True)

    # nnU-Net reads its three input/output dirs from os.environ — set them
    # in-process for the Python API to pick up.
    os.environ.update({
        "nnUNet_raw":          os.path.join(args.nnunet_base, "nnUNet_raw"),
        "nnUNet_preprocessed": os.path.join(args.nnunet_base, "nnUNet_preprocessed"),
        "nnUNet_results":      results_root,
    })

    # ----- ModelGeneration created BEFORE the predict loop so per-case
    # Annotations can reference it as they stream in. -----
    if args.model_generation_id:
        mg_id = args.model_generation_id
        print(f"[infer] reusing ModelGeneration {mg_id}", flush=True)
    else:
        mg_id = f"modelgeneration:{uuid.uuid4()}"
        train_set = [head[cid]["_id"] for cid in head
                     if head[cid].get("status") == "approved"]
        mg_doc = {
            "_id": mg_id,
            "type": "ModelGeneration",
            "name": args.label,
            "created_at": iso_now(),
            "created_by": "run-inference.py",
            "version": 1,
            "predecessor": None,
            "project_id": args.project_id,
            "label": args.label,
            "training_annotation_ids": train_set,
            "weights_ref": {
                "kind": "local-uri",
                "value": "file://" + os.path.join(results_root, dataset_dirname,
                                                  plans_dirname),
            },
            "framework": "nnUNetv2",
            "training_log_ref": None,
            "hardware": None,
            "software": None,
            "training_started_at": None,
            "training_finished_at": iso_now(),
            "notes": (f"{len(folds)}-fold ensemble; checkpoint={args.checkpoint}. "
                      "Per-fold log heads at run time: "
                      + "; ".join(f"fold{s['fold']}: epoch={s['epoch']} "
                                   f"ema_dice={s['best_ema_dice']}"
                                   for s in fold_snapshots)
                      + (" (NOTE: log heads are the *current* training_log, which "
                         "may be from a fresh restart after a post-training crash. "
                         "The checkpoint loaded is from a previously completed run.)"
                         if any(s["epoch"] is not None and s["epoch"] < 50 for s in fold_snapshots)
                         else "")),
        }
        chronicle.put(mg_doc)
        print(f"[infer] created {mg_id}", flush=True)

    per_case_disagreement = {}
    annotations_created = []

    # ----- per-case predict + post-process + Annotation (streaming) -----
    if args.per_fold_disagreement:
        if args.dry_run:
            print("[infer] DRY RUN — would init 4 predictors + per-case loop "
                  "(each case lands in chronicle as soon as its 4 fold predicts "
                  "+ ensemble math finish, ~2-3 min/case)", flush=True)
            return

        # nnUNetPredictor amortizes model loading: 4 fold checkpoints stay in
        # VRAM, predict() is called per case per fold from the same process.
        # Total time same as a CLI fold-by-fold pass; key win is each case's
        # Annotation lands the moment it's ready so the reviewer can begin
        # working without waiting for all 49.
        from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
        import torch, SimpleITK as sitk
        import numpy as np
        from itertools import combinations

        per_fold_root = os.path.join(out_dir, "_per_fold")
        agree_map_dir = os.path.join(out_dir, "_agreement_maps")
        os.makedirs(per_fold_root, exist_ok=True)
        os.makedirs(agree_map_dir, exist_ok=True)

        model_dir = os.path.join(results_root, dataset_dirname, plans_dirname)
        predictors = {}
        for f in folds:
            print(f"[infer] loading fold {f} ({args.checkpoint})", flush=True)
            p = nnUNetPredictor(
                tile_step_size=0.5,
                use_gaussian=True,
                use_mirroring=True,
                perform_everything_on_device=True,
                device=torch.device("cuda", 0),
                verbose=False,
                verbose_preprocessing=False,
                allow_tqdm=False,
            )
            p.initialize_from_trained_model_folder(
                model_dir,
                use_folds=(f,),
                checkpoint_name=args.checkpoint,
            )
            predictors[f] = p
            os.makedirs(os.path.join(per_fold_root, f"fold_{f}"), exist_ok=True)
        print(f"[infer] {len(predictors)} predictors in VRAM; streaming cases",
              flush=True)
        threshold = (len(folds) // 2) + 1  # strict majority for ensemble

        for cid in linked:
            case_input = os.path.join(work_dir, f"{cid}_0000.nrrd")
            if not os.path.isfile(case_input):
                print(f"[infer] {cid}: input missing; skip", flush=True); continue

            # Predict each fold; load the resulting NRRD into a numpy array.
            fold_arrs = {}; ref_img = None; ok = True
            for f in folds:
                fold_outdir = os.path.join(per_fold_root, f"fold_{f}")
                try:
                    predictors[f].predict_from_files(
                        list_of_lists_or_source_folder=[[case_input]],
                        output_folder_or_list_of_truncated_output_files=fold_outdir,
                        save_probabilities=False,
                        overwrite=True,
                        num_processes_preprocessing=2,
                        num_processes_segmentation_export=2,
                    )
                except Exception as exc:
                    print(f"[infer] {cid} fold {f}: {exc}", flush=True)
                    ok = False; break
                seg_path = os.path.join(fold_outdir, f"{cid}.nrrd")
                if not os.path.isfile(seg_path):
                    print(f"[infer] {cid} fold {f}: output missing", flush=True)
                    ok = False; break
                img = sitk.ReadImage(seg_path)
                fold_arrs[f] = (sitk.GetArrayFromImage(img) > 0).astype("uint8")
                if ref_img is None:
                    ref_img = img
            if not ok or ref_img is None:
                continue

            # Ensemble (majority vote) + per-voxel agreement count map.
            arrs = [fold_arrs[f] for f in folds]
            agree_count = sum(arrs)
            ensemble = (agree_count >= threshold).astype("uint8")

            # Per-case agreement scalar: mean pairwise Dice across fold pairs.
            dices = []
            for a, b in combinations(arrs, 2):
                p_sum = int(a.sum()); g_sum = int(b.sum())
                denom = p_sum + g_sum
                if denom == 0:
                    dices.append(1.0)
                else:
                    inter = int(np.logical_and(a, b).sum())
                    dices.append(2.0 * inter / denom)
            agreement_dice = float(np.mean(dices)) if dices else 1.0
            sx, sy, sz = ref_img.GetSpacing()
            voxel_ml = sx * sy * sz / 1000.0
            per_fold_volumes_ml = [float(a.sum()) * voxel_ml for a in arrs]
            disagreement = {
                "agreement_dice": agreement_dice,
                "per_fold_volume_ml": per_fold_volumes_ml,
                "majority_vote_volume_ml": float(ensemble.sum()) * voxel_ml,
                "voxel_count_disputed": int(((agree_count > 0)
                                             & (agree_count < len(folds))).sum()),
            }
            per_case_disagreement[cid] = disagreement

            # Persist ensemble SEG + agreement count map alongside.
            ensemble_path = os.path.join(out_dir, f"{cid}.nrrd")
            agree_path = os.path.join(agree_map_dir, f"{cid}.nrrd")
            ens_img = sitk.GetImageFromArray(ensemble); ens_img.CopyInformation(ref_img)
            sitk.WriteImage(ens_img, ensemble_path, useCompression=True)
            ag_img = sitk.GetImageFromArray(agree_count); ag_img.CopyInformation(ref_img)
            sitk.WriteImage(ag_img, agree_path, useCompression=True)

            # Register Blobs + create Annotation IMMEDIATELY so the reviewer
            # can begin work on this case without waiting for the rest.
            try:
                seg_blob_id = ensure_blob(chronicle, ensemble_path)
                ag_blob_id = ensure_blob(chronicle, agree_path)
            except Exception as exc:
                print(f"[infer] {cid}: blob registration failed: {exc}", flush=True)
                continue
            ann_id = f"annotation:{uuid.uuid4()}"
            notes_lines = [
                f"{len(folds)}-fold ensemble inference (label={args.label}, "
                f"checkpoint={args.checkpoint}).",
                f"agreement_dice={agreement_dice:.3f} (mean pairwise Dice across "
                f"{len(folds)} folds; lower = more disagreement).",
                f"per_fold_volume_ml={[round(v, 2) for v in per_fold_volumes_ml]}; "
                f"majority_vote_volume_ml={disagreement['majority_vote_volume_ml']:.2f}; "
                f"voxels_disputed={disagreement['voxel_count_disputed']}",
            ]
            ann_doc = {
                "_id": ann_id,
                "type": "Annotation",
                "name": f"{cid} / {args.label}",
                "created_at": iso_now(),
                "created_by": "run-inference.py",
                # Validator: version=1 requires predecessor=null; when chained,
                # bump version off the head's version so the chain stays linear.
                "version": (head[cid].get("version", 0) + 1) if cid in head else 1,
                "predecessor": (head[cid]["_id"] if cid in head else None),
                "project_id": args.project_id,
                "case_id": cid,
                "study_uid": (head[cid] or {}).get("study_uid") if cid in head else None,
                "status": "submitted_for_review",
                "notes": "\n".join(notes_lines),
                "producer": {
                    "kind": "model",
                    "label": args.label,
                    "model_generation_id": mg_id,
                    "ensemble_agreement_dice": agreement_dice,
                    "agreement_map_blob_id": ag_blob_id,
                },
                "seg_ref": {"blob_id": seg_blob_id},
            }
            try:
                chronicle.put(ann_doc); annotations_created.append(ann_id)
                print(f"[infer] ONLINE {cid}  agreement_dice={agreement_dice:.3f} "
                      f"per_fold_ml={[round(v, 1) for v in per_fold_volumes_ml]} "
                      f"ann={ann_id[-12:]}", flush=True)
            except Exception as exc:
                print(f"[infer] {cid} Annotation PUT failed: {exc}", flush=True)
    else:
        # Single-pass ensemble predict (averages softmax across folds internally),
        # then register/annotate after the entire batch finishes. No per-case
        # streaming and no uncertainty metric — kept for backwards compat.
        predict_cmd = [
            "/opt/lnq/nnenv/bin/nnUNetv2_predict",
            "-i", work_dir, "-o", out_dir,
            "-d", str(args.dataset_id),
            "-c", args.config_name,
            "-p", args.plans_identifier,
            "-f", *[str(f) for f in folds],
            "-tr", args.trainer_class,
            "-chk", args.checkpoint,
        ]
        print(f"[infer] running: {' '.join(predict_cmd)}", flush=True)
        if args.dry_run:
            print("[infer] DRY RUN — exiting before predict", flush=True); return
        rc = subprocess.call(predict_cmd)
        if rc != 0:
            print(f"[infer] nnUNetv2_predict exited {rc}", file=sys.stderr); sys.exit(rc)
        for cid in linked:
            seg_path = os.path.join(out_dir, f"{cid}.nrrd")
            if not os.path.isfile(seg_path):
                print(f"[infer] WARNING: no output for {cid}", flush=True); continue
            blob_id = ensure_blob(chronicle, seg_path)
            ann_id = f"annotation:{uuid.uuid4()}"
            ann_doc = {
                "_id": ann_id,
                "type": "Annotation",
                "name": f"{cid} / {args.label}",
                "created_at": iso_now(),
                "created_by": "run-inference.py",
                # Validator: version=1 requires predecessor=null; when chained,
                # bump version off the head's version so the chain stays linear.
                "version": (head[cid].get("version", 0) + 1) if cid in head else 1,
                "predecessor": (head[cid]["_id"] if cid in head else None),
                "project_id": args.project_id,
                "case_id": cid,
                "study_uid": (head[cid] or {}).get("study_uid") if cid in head else None,
                "status": "submitted_for_review",
                "notes": f"{len(folds)}-fold ensemble inference (label={args.label}).",
                "producer": {"kind": "model", "label": args.label,
                             "model_generation_id": mg_id},
                "seg_ref": {"blob_id": blob_id},
            }
            try:
                chronicle.put(ann_doc); annotations_created.append(ann_id)
            except Exception as exc:
                print(f"[infer] PUT annotation {cid} failed: {exc}", flush=True)
    print(f"[infer] created {len(annotations_created)} Annotations", flush=True)

    # ----- InferenceRun -----
    ir_id = f"inferencerun:{uuid.uuid4()}"
    ir_doc = {
        "_id": ir_id,
        "type": "InferenceRun",
        "created_at": iso_now(),
        "created_by": "run-inference.py",
        "version": 1,
        "predecessor": None,
        "project_id": args.project_id,
        "model_generation_id": mg_id,
        "case_ids": linked,
        "outputs_ref": {
            "kind": "local-uri",
            "value": "file://" + out_dir,
        },
        "notes": (f"{len(folds)}-fold ensemble / {args.label}."
                  + (f" Per-fold disagreement captured for {len(per_case_disagreement)} "
                     f"cases (-chk={args.checkpoint})." if per_case_disagreement else "")),
        # Roll-up of per-case agreement so the worklist can sort by uncertainty
        # without round-tripping each Annotation doc.
        "per_case_agreement": {
            cid: {"dice": d["agreement_dice"],
                  "majority_vote_volume_ml": d["majority_vote_volume_ml"]}
            for cid, d in per_case_disagreement.items()
        } if per_case_disagreement else None,
    }
    try:
        chronicle.put(ir_doc); print(f"[infer] created {ir_id}", flush=True)
    except Exception as exc:
        print(f"[infer] InferenceRun PUT failed (non-fatal): {exc}", flush=True)

    print(f"[infer] DONE — {len(annotations_created)} annotations submitted "
          f"for review under model={mg_id}", flush=True)


if __name__ == "__main__":
    main()
