#!/usr/bin/env python3
"""trainer-runner.py — one-fold nnU-Net v2 driver that reports to the Chronicle.

Reads /opt/lnq/chronicle.conf for credentials and /etc/lnq/job.conf for the
TrainingJob ID it's responsible for. PUTs status/heartbeat/metrics onto that
doc as it runs; on success writes a Blob for the weights + a ModelGeneration
doc; on failure records the tail of the log into TrainingJob.notes.

Runs as a systemd unit (lnq-trainer-runner.service) launched at boot after
cloud-init completes. Designed so that multiple folds running on different
machines against the same Manila share don't race on plan_and_preprocess
(file-locked) or on a shared nnUNet_results directory (per-fold subdirs).

This file is vendored verbatim into the trainer instance by the cloud-init
template. Keep it standalone (no LNQStudio imports).
"""

from __future__ import annotations

import argparse
import datetime
import fcntl
import glob
import hashlib
import json
import logging
import os
import re
import socket
import subprocess
import sys
import threading
import time
import uuid

import requests


CONF_PATH_DEFAULT = "/opt/lnq/chronicle.conf"
JOB_CONF_PATH_DEFAULT = "/etc/lnq/job.conf"
LOG_FORMAT = "%(asctime)s %(levelname)s %(message)s"


def iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_kv_conf(path):
    out = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


# ----- minimal Chronicle client (self-contained, no LNQStudioLib import) -----

class Chronicle:
    def __init__(self, base_url, db, username, password):
        self.base = base_url.rstrip("/")
        self.db = db
        self.s = requests.Session()
        self.s.auth = (username, password)
        self.host_label = socket.gethostname()

    def get(self, doc_id):
        r = self.s.get(f"{self.base}/{self.db}/{doc_id}")
        r.raise_for_status()
        return r.json()

    def put(self, doc):
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

    def register_blob(self, local_path, mime_type=None, actor=None):
        sha256, size = self._hash_file(local_path)
        blob_id = f"blob:{sha256}"
        location = {
            "kind": "local-uri",
            "value": f"file://{os.path.abspath(local_path)}",
            "host": self.host_label,
            "verified_at": iso_now(),
        }
        try:
            existing = self.get(blob_id)
        except requests.exceptions.HTTPError as exc:
            if exc.response.status_code != 404:
                raise
            doc = {
                "_id": blob_id, "type": "Blob",
                "name": f"{sha256[:8]} ({size} bytes)",
                "created_at": iso_now(), "created_by": actor or "trainer",
                "version": 1, "predecessor": None,
                "sha256": sha256, "size": size,
                "mime_type": mime_type,
                "locations": [location],
            }
            self.put(doc)
            return doc
        locs = list(existing.get("locations") or [])
        for i, l in enumerate(locs):
            if l.get("kind") == location["kind"] and l.get("value") == location["value"]:
                locs[i] = location; break
        else:
            locs.append(location)
        existing["locations"] = locs
        self.put(existing)
        return existing

    @staticmethod
    def _hash_file(path, chunk=4 * 1024 * 1024):
        h = hashlib.sha256(); size = 0
        with open(path, "rb") as f:
            while True:
                b = f.read(chunk)
                if not b: break
                h.update(b); size += len(b)
        return h.hexdigest(), size


# ----- nnU-Net helpers -----

EPOCH_RE = re.compile(r"^[\d\-: .]+:\s*Epoch (\d+)\s*$")
TRAIN_LOSS_RE = re.compile(r":\s*train_loss\s+([-+\d.]+)")
VAL_LOSS_RE = re.compile(r":\s*val_loss\s+([-+\d.]+)")
MEAN_DICE_RE = re.compile(r":\s*Mean validation Dice:\s*([\d.]+)")
PSEUDO_DICE_RE = re.compile(r":\s*Pseudo dice\s*\[([\d., ]+)\]")


def parse_training_log(log_path):
    """Read the nnU-Net training_log_*.txt and return the most recent
    (epoch, train_loss, val_loss, mean_dice) tuple. Any of the trailing
    fields may be None if not yet emitted for that epoch."""
    if not os.path.isfile(log_path):
        return None, None, None, None
    epoch = train_loss = val_loss = mean_dice = None
    try:
        with open(log_path, "rb") as f:
            # Bounded read: only the last ~256 KB. For a 1000-epoch log this
            # is plenty to find the latest epoch's metric lines.
            f.seek(0, os.SEEK_END)
            end = f.tell()
            f.seek(max(0, end - 262144))
            data = f.read().decode("utf-8", errors="replace")
    except OSError:
        return None, None, None, None
    for line in data.splitlines():
        m = EPOCH_RE.match(line)
        if m:
            epoch = int(m.group(1))
            train_loss = val_loss = mean_dice = None
            continue
        m = TRAIN_LOSS_RE.search(line)
        if m: train_loss = float(m.group(1)); continue
        m = VAL_LOSS_RE.search(line)
        if m: val_loss = float(m.group(1)); continue
        m = MEAN_DICE_RE.search(line)
        if m: mean_dice = float(m.group(1)); continue
        m = PSEUDO_DICE_RE.search(line)
        if m and mean_dice is None:
            try:
                vals = [float(x) for x in m.group(1).replace(" ", "").split(",") if x]
                if vals:
                    mean_dice = sum(vals) / len(vals)
            except ValueError:
                pass
    return epoch, train_loss, val_loss, mean_dice


def find_training_log(results_root, dataset_dirname, plans_subdir, fold):
    pattern = os.path.join(results_root, dataset_dirname, plans_subdir,
                            f"fold_{fold}", "training_log_*.txt")
    matches = sorted(glob.glob(pattern))
    return matches[-1] if matches else None


def dataset_dirname(dataset_id, task_name):
    return f"Dataset{int(dataset_id):03d}_{task_name}"


def plans_subdir(plans_identifier, config_name, trainer_class="nnUNetTrainer"):
    # nnU-Net v2 results layout is <trainer_class>__<plans_identifier>__<config>/
    # e.g. nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres/fold_X/
    # The default trainer class is nnUNetTrainer; passing a custom subclass
    # (e.g. nnUNetTrainerNoMirroring) would change this segment.
    return f"{trainer_class}__{plans_identifier}__{config_name}"


# ----- preprocess lock -----

def with_preprocess_lock(base_dir, fn):
    """Single-writer file lock for plan_and_preprocess. Multiple trainer
    instances on the same share will serialize through this; only the first
    holder actually runs the preprocessing — the others find it already
    done and return immediately."""
    lock_path = os.path.join(base_dir, ".preprocess.lock")
    os.makedirs(base_dir, exist_ok=True)
    f = open(lock_path, "a+")
    try:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
        return fn()
    finally:
        try:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        f.close()


# ----- runner -----

def run_subprocess_streaming(cmd, env=None, cwd=None):
    """Run a subprocess, streaming stdout/stderr to our stdout (which systemd
    captures to the journal)."""
    logging.info("running: %s", " ".join(cmd))
    proc = subprocess.Popen(cmd, env=env, cwd=cwd,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            bufsize=1, universal_newlines=True)
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
    finally:
        proc.wait()
    return proc.returncode


def update_job(chronicle, job_id, **fields):
    """GET, update fields, PUT. CouchDB's _rev is preserved across the round-trip."""
    job = chronicle.get(job_id)
    job.update(fields)
    if "last_heartbeat_at" not in fields:
        job["last_heartbeat_at"] = iso_now()
    chronicle.put(job)
    return job


def heartbeat_loop(stop_event, chronicle, job_id, log_finder, total_epochs):
    """Background thread: every 30 s, parse the active training_log and PUT
    progress fields onto the TrainingJob. Cheap and robust to log rotation."""
    last_epoch_reported = -1
    while not stop_event.is_set():
        try:
            log_path = log_finder()
            if log_path:
                ep, tl, vl, md = parse_training_log(log_path)
                if ep is not None and ep != last_epoch_reported:
                    update_job(chronicle, job_id,
                               current_epoch=ep,
                               latest_metrics={
                                   "train_loss": tl,
                                   "val_loss": vl,
                                   "mean_dice": md,
                               })
                    last_epoch_reported = ep
                else:
                    # Just bump the heartbeat so the dashboard knows we're alive.
                    update_job(chronicle, job_id)
            else:
                update_job(chronicle, job_id)
        except Exception:
            logging.exception("heartbeat tick failed")
        stop_event.wait(timeout=30)


def main():
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    ap = argparse.ArgumentParser()
    ap.add_argument("--training-job-id", default=None,
                    help="If omitted, read TRAINING_JOB_ID from --job-conf.")
    ap.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    ap.add_argument("--chronicle-conf", default=CONF_PATH_DEFAULT)
    ap.add_argument("--job-conf", default=JOB_CONF_PATH_DEFAULT)
    args = ap.parse_args()

    conf = load_kv_conf(args.chronicle_conf)
    job_conf = load_kv_conf(args.job_conf) if os.path.isfile(args.job_conf) else {}
    training_job_id = args.training_job_id or job_conf.get("TRAINING_JOB_ID")
    if not training_job_id:
        print("Required: --training-job-id or TRAINING_JOB_ID in job.conf", file=sys.stderr)
        sys.exit(2)

    chronicle = Chronicle(
        conf["CHRONICLE_URL"], conf.get("CHRONICLE_DB", "lnq"),
        username="admin", password=conf["COUCHDB_ADMIN_PASSWORD"],
    )

    smoke_test = (job_conf.get("SMOKE_TEST") or "").strip() == "1"

    job = chronicle.get(training_job_id)
    cfg = job.get("config") or {}

    # Resolve nnunet_base — database-first, conf-fallback. Order of preference:
    #   1. TrainingJob.config.training_workspace  (per-job override; what we use)
    #   2. Project.training_workspace             (anatomy-wide default if set)
    #   3. /opt/lnq/chronicle.conf NNUNET_BASE    (legacy conf-on-disk)
    #   4. /etc/lnq/job.conf NNUNET_BASE          (legacy cloud-init template)
    nnunet_base = cfg.get("training_workspace")
    if nnunet_base:
        print(f"nnunet_base from TrainingJob.config.training_workspace: {nnunet_base}")
    if not nnunet_base and job.get("project_id"):
        try:
            proj = chronicle.get(job["project_id"])
            nnunet_base = proj.get("training_workspace")
            if nnunet_base:
                print(f"nnunet_base from Project.training_workspace: {nnunet_base}")
        except Exception as exc:
            print(f"could not fetch project {job['project_id']}: {exc}",
                  file=sys.stderr)
    if not nnunet_base:
        nnunet_base = conf.get("NNUNET_BASE") or job_conf.get("NNUNET_BASE")
        if nnunet_base:
            print(f"nnunet_base from conf fallback: {nnunet_base}")
    if not nnunet_base:
        print("Required: TrainingJob.config.training_workspace, "
              "Project.training_workspace, or NNUNET_BASE in "
              "/opt/lnq/chronicle.conf or job.conf", file=sys.stderr)
        sys.exit(2)

    raw_dir   = os.path.join(nnunet_base, "nnUNet_raw")
    prep_dir  = os.path.join(nnunet_base, "nnUNet_preprocessed")
    res_dir   = os.path.join(nnunet_base, "nnUNet_results")
    for d in (raw_dir, prep_dir, res_dir):
        os.makedirs(d, exist_ok=True)

    dataset_id  = int(cfg.get("dataset_id", 1))
    task_name   = cfg.get("task_name") or "LNQinguinal"
    config_name = cfg.get("config_name") or "3d_fullres"
    planner     = cfg.get("planner") or "nnUNetPlannerResEncM"
    plans_id    = cfg.get("plans_identifier") or "nnUNetResEncUNetMPlans"
    fold        = int(cfg.get("fold", 0))
    total_epochs = int(cfg.get("total_epochs") or 1000)

    ds_name = dataset_dirname(dataset_id, task_name)
    ps_name = plans_subdir(plans_id, config_name)

    # Mark running.
    update_job(
        chronicle, training_job_id,
        status="running",
        started_at=iso_now(),
        host=socket.gethostname(),
        log_ref={
            "kind": "local-uri",
            "value": f"file://{res_dir}/{ds_name}/{ps_name}/fold_{fold}",
            "host": socket.gethostname(),
        },
        notes="SMOKE TEST in progress" if smoke_test else "",
    )

    if smoke_test:
        try:
            _smoke_test(chronicle, training_job_id, job, cfg, fold, ds_name, ps_name, res_dir)
        except Exception as exc:
            logging.exception("smoke test failed")
            try:
                update_job(chronicle, training_job_id,
                           status="failed", finished_at=iso_now(),
                           notes=f"SMOKE TEST failed: {exc}")
            except Exception:
                pass
            raise
        return

    env = os.environ.copy()
    env["nnUNet_raw"] = raw_dir
    env["nnUNet_preprocessed"] = prep_dir
    env["nnUNet_results"] = res_dir
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    stop_event = threading.Event()
    log_finder = lambda: find_training_log(res_dir, ds_name, ps_name, fold)

    try:
        # 0. Regenerate the dataset locally on the share. Idempotent.
        export_cmd = [
            "python3", "/opt/lnq/export-nnunet-dataset.py",
            "--project-id", job["project_id"],
            "--output-root", raw_dir,
            "--dataset-id", str(dataset_id),
            "--task-name", task_name,
            "--label-name", task_name.lower().replace("lnq", "").strip("_") + "_ln" or "lesion",
            "--status", "approved",
            "--link-mode", "symlink",
            "--conf", args.chronicle_conf,
        ]
        rc = run_subprocess_streaming(export_cmd, env=env, cwd="/opt/lnq")
        if rc != 0:
            raise RuntimeError(f"export-nnunet-dataset.py exited {rc}")

        # 1. plan_and_preprocess (file-locked across folds on the same share).
        def _maybe_preprocess():
            preprocessed_marker = os.path.join(prep_dir, ds_name, "dataset_fingerprint.json")
            if os.path.exists(preprocessed_marker):
                logging.info("preprocessing already done for %s; skipping.", ds_name)
                return
            cmd = ["nnUNetv2_plan_and_preprocess",
                   "-d", str(dataset_id), "-pl", planner, "-np", "8",
                   "-c", config_name, "--verify_dataset_integrity"]
            rc = run_subprocess_streaming(cmd, env=env)
            if rc != 0:
                raise RuntimeError(f"plan_and_preprocess exited {rc}")
        with_preprocess_lock(prep_dir, _maybe_preprocess)

        # 2. Start heartbeat thread, then train.
        hb = threading.Thread(target=heartbeat_loop, daemon=True,
                              args=(stop_event, chronicle, training_job_id, log_finder, total_epochs))
        hb.start()
        # --c: resume from checkpoint_latest.pth if it exists, else start
        # fresh. Cheap insurance against systemd restarts or trainer-health
        # auto-revives wiping hours of progress by re-running from epoch 0.
        train_cmd = ["nnUNetv2_train", str(dataset_id), config_name,
                     str(fold), "-p", plans_id, "--c"]
        rc = run_subprocess_streaming(train_cmd, env=env)
        if rc != 0:
            raise RuntimeError(f"nnUNetv2_train exited {rc}")

        # Final metric update.
        ep, tl, vl, md = parse_training_log(log_finder())
        update_job(chronicle, training_job_id,
                   current_epoch=ep,
                   latest_metrics={"train_loss": tl, "val_loss": vl, "mean_dice": md})

        # 3. Register weights + write ModelGeneration.
        checkpoint = os.path.join(res_dir, ds_name, ps_name, f"fold_{fold}",
                                   "checkpoint_final.pth")
        if not os.path.isfile(checkpoint):
            # nnU-Net writes checkpoint_best.pth and checkpoint_latest.pth too; fall back.
            for cand in ("checkpoint_best.pth", "checkpoint_latest.pth"):
                alt = os.path.join(res_dir, ds_name, ps_name, f"fold_{fold}", cand)
                if os.path.isfile(alt):
                    checkpoint = alt; break
        if not os.path.isfile(checkpoint):
            raise RuntimeError(f"no checkpoint found under fold_{fold}/")
        weights_blob = chronicle.register_blob(checkpoint,
                                                mime_type="application/octet-stream+pytorch",
                                                actor=job.get("created_by") or "trainer")

        modelgen_id = f"modelgeneration:{uuid.uuid4()}"
        modelgen = {
            "_id": modelgen_id, "type": "ModelGeneration",
            "name": f"{job.get('label', '?')} / fold {fold}",
            "created_at": iso_now(), "created_by": job.get("created_by") or "trainer",
            "version": 1, "predecessor": None,
            "project_id": job["project_id"],
            "label": job.get("label", task_name.lower()),
            "training_annotation_ids": job.get("training_annotation_ids", []),
            "weights_ref": {"blob_id": weights_blob["_id"]},
            "framework": "nnUNetv2",
            "hardware": {
                "host": socket.gethostname(),
                "gpu_visible": str(args.gpu),
            },
            "software": {
                "config_name": config_name,
                "plans_identifier": plans_id,
                "planner": planner,
                "dataset_id": dataset_id,
                "fold": fold,
            },
            "training_log_ref": {
                "kind": "local-uri",
                "value": f"file://{res_dir}/{ds_name}/{ps_name}/fold_{fold}",
                "host": socket.gethostname(),
            },
            "training_started_at": (chronicle.get(training_job_id).get("started_at")),
            "training_finished_at": iso_now(),
            "notes": "",
        }
        chronicle.put(modelgen)

        # 4. Mark converged.
        stop_event.set()
        update_job(chronicle, training_job_id,
                   status="converged",
                   finished_at=iso_now(),
                   model_generation_id=modelgen_id)
        logging.info("converged; ModelGeneration=%s", modelgen_id)

    except Exception as exc:
        stop_event.set()
        logging.exception("training failed")
        # Pull last 50 log lines into notes for triage from the dashboard.
        tail = ""
        try:
            log_path = log_finder()
            if log_path and os.path.isfile(log_path):
                with open(log_path) as f:
                    tail = "".join(f.readlines()[-50:])
        except Exception:
            pass
        try:
            update_job(chronicle, training_job_id,
                       status="failed",
                       finished_at=iso_now(),
                       notes=f"{exc}\n\n--- last 50 log lines ---\n{tail}")
        except Exception:
            logging.exception("could not record failure on TrainingJob")
        raise


def _smoke_test(chronicle, training_job_id, job, cfg, fold, ds_name, ps_name, res_dir):
    """Validate the trainer environment without actually training:
      - nvidia-smi works (GPU passed through to the instance)
      - PyTorch sees CUDA and can allocate on the device
      - nnunetv2 imports
    Then write a dummy checkpoint, register it as a Blob, and write a
    ModelGeneration doc + flip the TrainingJob to converged. Total wall-clock
    on the instance: ~30 seconds once the runner starts."""
    import torch
    import nnunetv2

    # 1. nvidia-smi
    try:
        smi_out = subprocess.check_output(["nvidia-smi"], text=True, timeout=20)
    except (subprocess.SubprocessError, FileNotFoundError) as exc:
        raise RuntimeError(f"nvidia-smi failed: {exc}")

    # 2. torch + CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("torch.cuda.is_available() == False")
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    # Allocate + simple op on the device to exercise the driver.
    x = torch.zeros(2048, 2048, device="cuda")
    x.add_(1.0)
    gpu_sum = float(x.sum().item())
    if not (gpu_sum > 0):
        raise RuntimeError(f"GPU smoke-allocate gave unexpected sum: {gpu_sum}")

    # 3. nnunetv2 importability — just having the module loaded is enough.
    nnunet_version = getattr(nnunetv2, "__version__", "unknown")

    # 4. Manila mount visible? (the runner's working dir + nnUNet_results
    #    should already be on it; sanity-check by writing a marker)
    marker_path = os.path.join(res_dir, ds_name, ps_name, f"fold_{fold}", "_smoke_marker")
    os.makedirs(os.path.dirname(marker_path), exist_ok=True)
    with open(marker_path, "w") as f:
        f.write(f"{socket.gethostname()} {iso_now()}\n")

    # 5. Dummy checkpoint, written to /tmp so the real nnUNet_results path
    #    stays clean for a future real run.
    dummy_path = f"/tmp/smoke_checkpoint_fold{fold}.pth"
    torch.save({
        "smoke": True,
        "fold": fold,
        "host": socket.gethostname(),
        "gpu": gpu_name,
        "gpu_memory_gb": gpu_mem_gb,
        "nnunetv2_version": nnunet_version,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "_padding": torch.zeros(64, 64),
    }, dummy_path)

    weights_blob = chronicle.register_blob(
        dummy_path,
        mime_type="application/octet-stream+pytorch",
        actor=job.get("created_by") or "trainer-smoke",
    )

    # 6. ModelGeneration with notes flagging it as smoke-only.
    modelgen_id = f"modelgeneration:{uuid.uuid4()}"
    modelgen = {
        "_id": modelgen_id, "type": "ModelGeneration",
        "name": f"{job.get('label', '?')} / fold {fold} (smoke)",
        "created_at": iso_now(), "created_by": job.get("created_by") or "trainer-smoke",
        "version": 1, "predecessor": None,
        "project_id": job["project_id"],
        "label": job.get("label", "smoke") + "-smoke",
        "training_annotation_ids": job.get("training_annotation_ids", []),
        "weights_ref": {"blob_id": weights_blob["_id"]},
        "framework": "nnUNetv2",
        "hardware": {
            "host": socket.gethostname(),
            "gpu": gpu_name,
            "gpu_memory_gb": round(gpu_mem_gb, 1),
        },
        "software": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "nnunetv2": nnunet_version,
            "fold": fold,
        },
        "training_log_ref": None,
        "training_started_at": chronicle.get(training_job_id).get("started_at"),
        "training_finished_at": iso_now(),
        "notes": (f"SMOKE TEST ONLY — no nnUNetv2_plan_and_preprocess or "
                  f"nnUNetv2_train was run. This ModelGeneration represents a "
                  f"validated environment, not a usable model. Do not feed to "
                  f"inference.\n\nGPU: {gpu_name} ({gpu_mem_gb:.1f} GB)\n"
                  f"CUDA: {torch.version.cuda}\nnnunetv2: {nnunet_version}"),
    }
    chronicle.put(modelgen)

    notes_summary = (
        f"SMOKE TEST passed.\n"
        f"GPU: {gpu_name} ({gpu_mem_gb:.1f} GB), torch {torch.__version__} / "
        f"CUDA {torch.version.cuda}, nnunetv2 {nnunet_version}.\n"
        f"Manila marker: {marker_path}\n"
        f"Dummy checkpoint blob: {weights_blob['_id']}\n\n"
        f"nvidia-smi (head):\n" + "\n".join(smi_out.splitlines()[:18])
    )
    update_job(
        chronicle, training_job_id,
        status="converged",
        finished_at=iso_now(),
        model_generation_id=modelgen_id,
        latest_metrics={
            "smoke_test": True,
            "gpu_name": gpu_name,
            "gpu_memory_gb": round(gpu_mem_gb, 1),
            "nnunetv2_version": nnunet_version,
        },
        notes=notes_summary,
    )
    logging.info("smoke test passed; ModelGeneration=%s", modelgen_id)


if __name__ == "__main__":
    main()
