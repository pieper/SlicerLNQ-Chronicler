#!/usr/bin/env python3
"""trainer-metrics-publisher.py — Mac-side publisher of per-epoch metrics.

Side-channel fix for an in-instance heartbeat bug where the runner's
PSEUDO_DICE_RE doesn't match the `np.float32(...)` format current nnU-Net
emits, and the per-epoch reset clobbers the previous epoch's metrics with
None when the parser races the next "Epoch N" line. Rather than restart the
runners (which would forfeit training progress; nnUNetv2_train doesn't get
the --c flag), we tail each fold's training_log_*.txt via SSH, parse with
fixed regexes, and PUT current_epoch + latest_metrics to chronicle on a
30 s cadence. Only non-None values are merged in, so we never clobber a
known-good metric with a None just because the tail caught a mid-epoch
window.

The runner's own heartbeat keeps updating last_heartbeat_at + current_epoch,
so there's no contention on those; we just keep latest_metrics honest.

Stdlib-only — runs with /usr/bin/env python3, no pip deps.
"""
from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request


# Fixed regexes vs the trainer-runner's in-tree copy.
EPOCH_RE       = re.compile(r"^[\d\-: .]+:\s*Epoch (\d+)\s*$")
TRAIN_LOSS_RE  = re.compile(r":\s*train_loss\s+([-+\d.eE]+)")
VAL_LOSS_RE    = re.compile(r":\s*val_loss\s+([-+\d.eE]+)")
# Accept both legacy `[0.34, 0.21]` and current `[np.float32(0.34)]` shapes.
PSEUDO_DICE_RE = re.compile(
    r":\s*Pseudo dice\s*\[(.+?)\]\s*$")
FLOAT_INSIDE_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def parse_log(text):
    """Walk forward; track the most recent epoch and the most recent metric
    values seen *at or after* that epoch. Does NOT reset to None on the next
    Epoch line — we want last-good values to survive a tail that races into a
    mid-epoch window."""
    epoch = train_loss = val_loss = mean_dice = None
    for line in text.splitlines():
        m = EPOCH_RE.match(line)
        if m:
            epoch = int(m.group(1)); continue
        m = TRAIN_LOSS_RE.search(line)
        if m:
            try: train_loss = float(m.group(1))
            except ValueError: pass
            continue
        m = VAL_LOSS_RE.search(line)
        if m:
            try: val_loss = float(m.group(1))
            except ValueError: pass
            continue
        m = PSEUDO_DICE_RE.search(line)
        if m:
            # Strip numpy type wrappers before extracting floats so the
            # FLOAT regex doesn't pick up the "32" / "64" in "np.float32".
            inner = re.sub(r"np\.float\d+", "", m.group(1))
            vals = [float(v) for v in FLOAT_INSIDE_RE.findall(inner)]
            if vals:
                mean_dice = sum(vals) / len(vals)
    return epoch, train_loss, val_loss, mean_dice


def iso_now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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


class Chronicle:
    def __init__(self, base_url, db, password):
        self.base = base_url.rstrip("/"); self.db = db
        token = base64.b64encode(f"admin:{password}".encode()).decode()
        self.auth_header = f"Basic {token}"

    def _request(self, method, doc_id, body=None):
        url = f"{self.base}/{self.db}/{doc_id}"
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            url, data=data, method=method,
            headers={"Authorization": self.auth_header,
                     "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = ""
            try: detail = json.loads(exc.read()).get("reason", "")
            except Exception: pass
            raise RuntimeError(f"{method} {doc_id}: HTTP {exc.code} {detail}") from None

    def get(self, doc_id): return self._request("GET", doc_id)
    def put(self, doc):    return self._request("PUT", doc["_id"], body=doc)


def ssh_tail(ssh_key, jump_host, jump_user, ssh_user, internal_ip, path, nbytes):
    """Run `tail -c <nbytes> <path>` on the trainer via ssh + ProxyCommand.
    Returns the bytes (decoded utf-8) or '' on any error."""
    proxy = (f"ssh -i {ssh_key} -W %h:%p -o StrictHostKeyChecking=accept-new "
             f"-o ConnectTimeout=10 {jump_user}@{jump_host}")
    cmd = ["ssh", "-i", ssh_key,
           "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ConnectTimeout=10",
           "-o", f"ProxyCommand={proxy}",
           f"{ssh_user}@{internal_ip}",
           f"tail -c {nbytes} {path} 2>/dev/null"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        return r.stdout if r.returncode == 0 else ""
    except subprocess.TimeoutExpired:
        return ""
    except Exception:
        return ""


def find_log_via_ssh(ssh_key, jump_host, jump_user, ssh_user, internal_ip,
                     base_dir):
    """Resolve the most recent training_log_*.txt under any plans subdir
    matching fold_X. Returns absolute path or None."""
    cmd_str = (f"ls -1t {base_dir}/training_log_*.txt 2>/dev/null | head -1")
    proxy = (f"ssh -i {ssh_key} -W %h:%p -o StrictHostKeyChecking=accept-new "
             f"-o ConnectTimeout=10 {jump_user}@{jump_host}")
    cmd = ["ssh", "-i", ssh_key,
           "-o", "StrictHostKeyChecking=accept-new",
           "-o", "ConnectTimeout=10",
           "-o", f"ProxyCommand={proxy}",
           f"{ssh_user}@{internal_ip}",
           cmd_str]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        path = (r.stdout or "").strip()
        return path or None
    except Exception:
        return None


def internal_ip_from_doc(doc):
    """Pull the 10.x.x.x address out of the observer's openstack_state."""
    addrs = ((doc.get("openstack_state") or {}).get("server") or {}).get("addresses") or {}
    for ips in addrs.values():
        for ip in ips:
            if ip.startswith("10."):
                return ip
    return None


def merge_metrics(existing, new):
    """Keep last-good values; only overwrite when the new value is non-None."""
    merged = dict(existing or {})
    for k, v in (new or {}).items():
        if v is not None:
            merged[k] = v
    return merged


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pair", nargs=2, action="append", default=[],
                    metavar=("TRAINING_JOB_ID", "INSTANCE_NAME"),
                    help="Repeatable; matches trainer-observe.py.")
    ap.add_argument("--results-base",
                    default="/media/share/LNQ-data/Inguinal_Syed/nnUNet_results",
                    help="Manila path where Dataset<id>_<task>/<plans>/fold_X/ live.")
    ap.add_argument("--conf", default="chronicle.conf")
    ap.add_argument("--trainer-conf", default="trainer.conf")
    ap.add_argument("--interval", type=int, default=30,
                    help="cadence s (default 30)")
    ap.add_argument("--tail-bytes", type=int, default=8192)
    args = ap.parse_args()

    if not args.pair:
        ap.error("at least one --pair required")
    conf = load_conf(args.conf)
    tconf = load_conf(args.trainer_conf)
    ssh_key = tconf.get("SSH_KEY"); ssh_user = tconf.get("SSH_USER", "ubuntu")
    jump_host = tconf.get("JUMP_HOST"); jump_user = tconf.get("JUMP_USER", "ubuntu")
    if not (ssh_key and jump_host):
        print("Missing SSH_KEY / JUMP_HOST in trainer.conf", file=sys.stderr); sys.exit(1)
    # trainer.conf often uses $HOME/.ssh/... — expanduser only handles ~, not
    # $VAR, so we have to do both.
    ssh_key = os.path.expanduser(os.path.expandvars(ssh_key))

    chronicle_url = conf.get("CHRONICLE_URL")
    if not chronicle_url and conf.get("DOMAIN_NAME"):
        chronicle_url = "https://" + conf["DOMAIN_NAME"]
    if not chronicle_url:
        print("Missing CHRONICLE_URL / DOMAIN_NAME", file=sys.stderr); sys.exit(1)
    chronicle = Chronicle(chronicle_url, conf.get("CHRONICLE_DB", "lnq"),
                          conf["COUCHDB_ADMIN_PASSWORD"])

    print(f"[metrics] interval={args.interval}s pairs={len(args.pair)}",
          flush=True)

    while True:
        for jid, inst in args.pair:
            try:
                doc = chronicle.get(jid)
            except Exception as exc:
                print(f"[metrics] GET {jid}: {exc}", flush=True); continue
            cfg = doc.get("config") or {}
            ds_name = f"Dataset{int(cfg.get('dataset_id', 1)):03d}_{cfg.get('task_name', 'unknown')}"
            fold = cfg.get("fold", 0)
            ip = internal_ip_from_doc(doc)
            if not ip:
                print(f"[metrics] {inst} skip: no internal IP in openstack_state",
                      flush=True)
                continue
            # Walk all candidate plans subdirs (we don't know the trainer class).
            base = (f"{args.results_base}/{ds_name}/*/fold_{fold}")
            log_path = find_log_via_ssh(ssh_key, jump_host, jump_user, ssh_user,
                                        ip, base)
            if not log_path:
                print(f"[metrics] {inst} skip: no log at {base}", flush=True)
                continue
            data = ssh_tail(ssh_key, jump_host, jump_user, ssh_user, ip,
                            log_path, args.tail_bytes)
            if not data:
                print(f"[metrics] {inst} skip: tail returned empty for {log_path}",
                      flush=True)
                continue
            ep, tl, vl, md = parse_log(data)
            new_metrics = merge_metrics(
                doc.get("latest_metrics"),
                {"train_loss": tl, "val_loss": vl, "mean_dice": md})
            updates = {}
            if ep is not None and ep != doc.get("current_epoch"):
                updates["current_epoch"] = ep
            if new_metrics != (doc.get("latest_metrics") or {}):
                updates["latest_metrics"] = new_metrics
            if not updates:
                continue
            try:
                fresh = chronicle.get(jid)
                fresh.update(updates)
                chronicle.put(fresh)
                print(f"[metrics] {inst} ep={ep} td={updates.get('latest_metrics', {}).get('mean_dice')}",
                      flush=True)
            except Exception as exc:
                print(f"[metrics] PUT {jid}: {exc}", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
