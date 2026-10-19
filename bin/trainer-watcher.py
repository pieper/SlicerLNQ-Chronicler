#!/usr/bin/env python3
"""Watcher daemon for the mediastinal+axillary training run on Js2.

Polls chronicle every 60s and handles three things:

  1. **Failure-shelver.** If a TrainingJob is `status=failed` and its last
     heartbeat is more than 5 min stale, the trainer-runner has given up
     (systemd's Restart=on-failure has burned its budget). Shelve the host
     so we stop paying SU on a dead trainer.

  2. **Axillary auto-trigger.** When all 5 `mediastinal-v1` TrainingJobs
     report `status=converged`, repurpose the same 5 instances for axillary:
     SSH each, rewrite /etc/lnq/job.conf with the new axillary TJ ID, reset
     and restart the systemd service. Same g3.xl instances; no re-launch.

  3. **Final shelver.** When all 5 `axillary-v1` TrainingJobs converge,
     shelve all 5 instances so no more SU is consumed.

Runs in foreground; recommended invocation:
  nohup python3 bin/trainer-watcher.py > /tmp/trainer-watcher.log 2>&1 &

Stop with: pkill -f trainer-watcher.py
"""
from __future__ import annotations

import argparse
import base64
import datetime
import json
import logging
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


# ----- chronicle -----

class Chronicle:
    def __init__(self, base, db, password):
        self.base = base.rstrip("/"); self.db = db
        tok = base64.b64encode(f"admin:{password}".encode()).decode()
        self.auth = f"Basic {tok}"

    def _req(self, method, path, body=None):
        url = f"{self.base}/{path}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
            headers={"Authorization": self.auth, "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read())

    def list_by_type(self, type_name):
        rows = self._req("GET", f"{self.db}/_all_docs?include_docs=true").get("rows", [])
        return [r["doc"] for r in rows if r.get("doc") and r["doc"].get("type") == type_name]


def parse_iso(ts):
    if not ts: return None
    try:
        return datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ").replace(
            tzinfo=datetime.timezone.utc)
    except ValueError:
        return None


# ----- openstack -----

def openstack(*args, os_cloud="MED250016_IU"):
    cmd = ["openstack", "--os-cloud", os_cloud] + list(args)
    try:
        out = subprocess.run(cmd, capture_output=True, timeout=120, check=False)
        return out.returncode, out.stdout.decode(), out.stderr.decode()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"


def shelve_instance(host):
    """Shelve a Js2 instance by name. Idempotent — if already shelved, no-op."""
    rc, out, err = openstack("server", "show", host, "-f", "json")
    if rc != 0:
        logging.warning("show %s failed: %s", host, err.strip())
        return False
    try:
        status = json.loads(out)["status"]
    except Exception:
        status = "?"
    if status.startswith("SHELVED"):
        logging.info("  already shelved: %s", host); return True
    logging.info("  shelving %s (status=%s)", host, status)
    rc, _, err = openstack("server", "shelve", host)
    if rc != 0:
        logging.warning("shelve %s failed: %s", host, err.strip())
        return False
    return True


# ----- axillary trigger -----

SSH_KEY = "/Users/pieper/.ssh/lnq-chronicler-deploy"
TRAINER_REPO = "/Users/pieper/slicer/latest/SlicerLNQ-Chronicler"


def trainer_ssh(host, cmd):
    """SSH into a trainer via the existing trainer.sh ssh wrapper."""
    full = ["bash", "-lc",
            f'cd "{TRAINER_REPO}" && ./bin/trainer.sh ssh "{host}" -- {json.dumps(cmd)}']
    res = subprocess.run(full, capture_output=True, timeout=180)
    return res.returncode, res.stdout.decode(), res.stderr.decode()


def push_file(host, local_path, remote_path):
    """Copy a small local file into place on the trainer (sudo tee).
    Bypasses gist CDN caching by sending the bytes directly over SSH."""
    with open(local_path, "rb") as f:
        content = f.read()
    cmd = f"sudo tee {remote_path} > /dev/null && sudo chmod +x {remote_path}"
    full = ["bash", "-lc",
            f'cd "{TRAINER_REPO}" && ./bin/trainer.sh ssh "{host}" -- {json.dumps(cmd)}']
    res = subprocess.run(full, input=content, capture_output=True, timeout=60)
    return res.returncode == 0


def start_snapshotter(host):
    """Push lnq-val-snapshotter.py + start-val-snapshotter.sh to the trainer
    (overwriting any prior copy), kill any running snapshotter, and launch
    a fresh one bound to whatever TrainingJob /etc/lnq/job.conf now points at.

    Called twice per trainer in this run:
      1. once now, to attach snapshotters to the live mediastinal jobs
      2. again from repurpose_for_axillary, after job.conf has been swapped
    """
    snap_local = os.path.join(TRAINER_REPO, "bin", "lnq-val-snapshotter.py")
    wrap_local = os.path.join(TRAINER_REPO, "bin", "start-val-snapshotter.sh")
    if not push_file(host, snap_local, "/opt/lnq/lnq-val-snapshotter.py"):
        logging.warning("  push snapshotter -> %s failed", host); return False
    if not push_file(host, wrap_local, "/opt/lnq/start-val-snapshotter.sh"):
        logging.warning("  push wrapper -> %s failed", host); return False
    # `< /dev/null` so SSH does not hang on the backgrounded snapshotter's stdin.
    rc, out, err = trainer_ssh(host,
        "sudo pkill -f lnq-val-snapshotter.py || true; "
        "sleep 1; sudo /opt/lnq/start-val-snapshotter.sh < /dev/null")
    ok = rc == 0 and "snapshotter pid=" in out
    if ok:
        logging.info("  snapshotter started on %s: %s", host, out.strip())
    else:
        logging.warning("  snapshotter start failed on %s: rc=%s err=%s",
                        host, rc, (err or out).strip()[:200])
    return ok


def repurpose_for_axillary(trainer_to_tj):
    """For each (trainer_name, axillary_tj_id) pair, rewrite job.conf, restart
    the trainer service, and re-launch the validation snapshotter against the
    new TJ."""
    success = []
    for trainer, tj_id in trainer_to_tj.items():
        cmd = (f"sudo sed -i 's|^TRAINING_JOB_ID=.*|TRAINING_JOB_ID={tj_id}|' "
               f"/etc/lnq/job.conf && "
               f"sudo systemctl reset-failed lnq-trainer-runner.service && "
               f"sudo systemctl start lnq-trainer-runner.service && "
               f"echo started-{tj_id[-8:]}")
        rc, out, err = trainer_ssh(trainer, cmd)
        if rc == 0 and "started" in out:
            success.append(trainer)
            logging.info("  repurposed %s -> %s", trainer, tj_id[-12:])
            # Re-attach snapshotter to the new (axillary) TJ. Best-effort —
            # snapshotter exiting does not fail the repurpose.
            start_snapshotter(trainer)
        else:
            logging.warning("  failed to repurpose %s: rc=%s err=%s",
                            trainer, rc, err.strip()[:200])
    return success


# ----- main loop -----

def now_utc():
    return datetime.datetime.now(datetime.timezone.utc)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--chronicle-conf",
                    default=f"{TRAINER_REPO}/chronicle.conf")
    ap.add_argument("--mediastinal-jobs",
                    default="/tmp/mediastinal-training-jobs-v2.json")
    ap.add_argument("--axillary-jobs",
                    default="/tmp/axillary-training-jobs.json")
    ap.add_argument("--poll-interval", type=int, default=60)
    ap.add_argument("--stale-fail-minutes", type=int, default=5,
                    help="A TJ failed with no heartbeat for this long → shelve host.")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [watcher] %(message)s",
                        datefmt="%Y-%m-%d %H:%M:%S")

    # Read chronicle conf.
    conf = {}
    with open(args.chronicle_conf) as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith("#") or "=" not in ln: continue
            k, _, v = ln.partition("=")
            conf[k.strip()] = v.strip().strip('"').strip("'")
    chr = Chronicle("https://" + conf["DOMAIN_NAME"],
                    conf.get("CHRONICLE_DB", "lnq"),
                    conf["COUCHDB_ADMIN_PASSWORD"])

    med_ids = set(json.load(open(args.mediastinal_jobs))["training_job_ids"])
    ax_ids  = set(json.load(open(args.axillary_jobs))["training_job_ids"])
    logging.info("watching %d mediastinal + %d axillary TJs",
                 len(med_ids), len(ax_ids))

    axillary_launched = False
    shelved_hosts = set()

    while True:
        try:
            tjs = chr.list_by_type("TrainingJob")
            tjs_by_id = {t["_id"]: t for t in tjs}

            # ----- 1. Failure-shelver -----
            for tj in tjs:
                if tj["_id"] not in (med_ids | ax_ids):
                    continue
                if tj.get("status") != "failed":
                    continue
                host = tj.get("host")
                if not host or host in shelved_hosts:
                    continue
                hb = parse_iso(tj.get("last_heartbeat_at"))
                if hb is None: continue
                age_min = (now_utc() - hb).total_seconds() / 60
                if age_min < args.stale_fail_minutes:
                    continue
                logging.info("FAIL: %s (%s); heartbeat %.1f min stale",
                             tj["name"], host, age_min)
                if shelve_instance(host):
                    shelved_hosts.add(host)

            # ----- 2. Axillary auto-trigger -----
            if not axillary_launched:
                med_tjs = [tjs_by_id[i] for i in med_ids if i in tjs_by_id]
                converged = sum(1 for t in med_tjs if t.get("status") == "converged")
                if converged == 5:
                    logging.info("ALL 5 MEDIASTINAL CONVERGED — launching axillary")
                    # Map each trainer host to one axillary TJ. Use mediastinal
                    # hosts in fold order so trainer hostnames are stable.
                    sorted_med = sorted(med_tjs, key=lambda t: (t.get("config") or {}).get("fold", 0))
                    ax_id_list = json.load(open(args.axillary_jobs))["training_job_ids"]
                    trainer_to_tj = {
                        t["host"]: ax_id_list[i]
                        for i, t in enumerate(sorted_med)
                        if t.get("host")
                    }
                    if len(trainer_to_tj) == 5:
                        repurpose_for_axillary(trainer_to_tj)
                        axillary_launched = True
                    else:
                        logging.warning("only %d hosts known, can't launch axillary yet",
                                        len(trainer_to_tj))

            # ----- 3. Final shelver after axillary done -----
            if axillary_launched:
                ax_tjs = [tjs_by_id[i] for i in ax_ids if i in tjs_by_id]
                converged = sum(1 for t in ax_tjs if t.get("status") == "converged")
                if converged == 5:
                    logging.info("ALL 5 AXILLARY CONVERGED — final shelve")
                    for t in ax_tjs:
                        host = t.get("host")
                        if host and host not in shelved_hosts:
                            if shelve_instance(host):
                                shelved_hosts.add(host)
                    logging.info("DONE. Watcher exiting.")
                    sys.exit(0)

        except Exception as exc:
            logging.warning("loop iteration failed: %s", exc)

        time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
