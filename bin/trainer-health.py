#!/usr/bin/env python3
"""trainer-health.py — instance health publisher for the LNQ dashboard.

Runs every 60 s as a systemd timer (lnq-trainer-health.timer). Gathers:
  - cloud-init phase + last log lines
  - Manila mount state
  - GPU state (nvidia-smi output OR explicit failure)
  - lnq-trainer-runner.service state + journal tail
  - host info (uptime, free memory)

Then PUTs the whole snapshot into TrainingJob.runtime_telemetry, an
opportunistic mutable field. The dashboard's Selected-job panel renders it
live via the existing CouchDB _changes subscription, so a stuck instance
shows its real state in Slicer without anyone SSHing in.

This script is INDEPENDENT of the trainer runner: it starts at first boot
and keeps reporting even if the runner can't launch (e.g. GPU driver
didn't bind). That's the whole point.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import shutil
import socket
import subprocess
import sys

import requests


CONF_PATH = "/opt/lnq/chronicle.conf"
JOB_CONF_PATH = "/etc/lnq/job.conf"


def iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_kv(path):
    out = {}
    if not os.path.isfile(path):
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def run_capture(cmd, timeout=10):
    """Return (rc, stdout) — no exceptions raised."""
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, (r.stdout or "") + (r.stderr or "")
    except FileNotFoundError as e:
        return 127, f"FileNotFoundError: {e}"
    except subprocess.TimeoutExpired:
        return 124, "timeout"
    except Exception as e:
        return 1, f"{type(e).__name__}: {e}"


# ----- collectors -----

def collect_cloud_init():
    rc, out = run_capture(["cloud-init", "status", "--format", "json"])
    parsed = {}
    try:
        parsed = json.loads(out)
    except Exception:
        pass
    tail = []
    log_path = "/var/log/cloud-init-output.log"
    try:
        if os.path.isfile(log_path):
            with open(log_path, "rb") as f:
                f.seek(0, os.SEEK_END)
                end = f.tell()
                f.seek(max(0, end - 8192))
                data = f.read().decode("utf-8", errors="replace")
            tail = [ln for ln in data.splitlines() if ln.strip()][-15:]
    except OSError:
        pass
    return {
        "status": parsed.get("status") or ("done" if rc == 0 else "?"),
        "errors": parsed.get("errors") or [],
        "datasource": parsed.get("datasource") or "",
        "tail": tail,
    }


def collect_manila(mount_point):
    rc, out = run_capture(["mount"])
    mounted = False
    detail = ""
    for ln in out.splitlines():
        if mount_point and mount_point in ln:
            mounted = True
            # CephFS path is huge; truncate.
            detail = ln if len(ln) <= 240 else (ln[:240] + "…")
            break
    return {"mount_point": mount_point, "mounted": mounted, "detail": detail}


def collect_gpu():
    # nvidia-smi presence
    if not shutil.which("nvidia-smi"):
        return {"nvidia_smi_present": False, "ok": False,
                "error": "nvidia-smi binary not in PATH"}
    # Structured query first (clean rows of driver/name/mem)
    rc, out = run_capture(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used,utilization.gpu",
         "--format=csv,noheader,nounits"], timeout=15)
    if rc != 0:
        # Surface the human-readable failure
        rc2, out2 = run_capture(["nvidia-smi"], timeout=15)
        first_err = next((ln for ln in (out + out2).splitlines() if ln.strip()), "")
        # Loaded modules + /dev/nvidia* state are helpful diagnostics
        lsmod_rc, lsmod_out = run_capture(["sh", "-c", "lsmod | grep -E '^nvidia' || true"])
        dev_rc, dev_out = run_capture(["sh", "-c", "ls /dev/nvidia* 2>/dev/null || echo '(none)'"])
        return {
            "nvidia_smi_present": True, "ok": False,
            "error": first_err[:240],
            "lsmod": (lsmod_out or "").strip().splitlines(),
            "dev_nodes": (dev_out or "").strip().splitlines(),
        }
    devices = []
    for ln in out.strip().splitlines():
        parts = [p.strip() for p in ln.split(",")]
        if len(parts) >= 5:
            devices.append({
                "name": parts[0],
                "driver_version": parts[1],
                "memory_total_mb": int(parts[2]) if parts[2].isdigit() else None,
                "memory_used_mb": int(parts[3]) if parts[3].isdigit() else None,
                "gpu_util_pct": int(parts[4]) if parts[4].isdigit() else None,
            })
    return {"nvidia_smi_present": True, "ok": True, "devices": devices}


def collect_system_journal(lines=40):
    """System-wide journal tail. Catches anything systemd, kernel, apt,
    networkd, etc. produces — useful when no specific service is reporting
    yet, e.g. during cloud-init's runcmd."""
    rc, out = run_capture(
        ["journalctl", "--no-pager", "-n", str(lines), "-o", "short-iso",
         "--system"], timeout=15)
    if rc != 0:
        return {"available": False, "error": (out or "")[:200]}
    keep = [ln for ln in (out or "").splitlines() if ln.strip()][-lines:]
    return {"available": True, "tail": keep}


def collect_apt(lines=20):
    """Recent apt activity from /var/log/apt/history.log + term.log tails.
    Lets the dashboard tell whether a long cuda-drivers install is still in
    progress."""
    out = {}
    for label, path in (("history", "/var/log/apt/history.log"),
                        ("term", "/var/log/apt/term.log")):
        try:
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    f.seek(0, os.SEEK_END)
                    end = f.tell()
                    f.seek(max(0, end - 8192))
                    data = f.read().decode("utf-8", errors="replace")
                out[label] = [ln for ln in data.splitlines() if ln.strip()][-lines:]
        except OSError:
            pass
    return out


def collect_service(unit="lnq-trainer-runner.service"):
    rc_a, out_a = run_capture(["systemctl", "is-active", unit])
    rc_s, out_s = run_capture(["systemctl", "status", unit, "--no-pager", "-n", "0"])
    state = (out_a or "").strip()
    # journal tail — most useful when failed or never started.
    rc_j, out_j = run_capture(
        ["journalctl", "-u", unit, "--no-pager", "-n", "20", "-o", "short-iso"], timeout=15)
    journal_lines = [ln for ln in (out_j or "").splitlines() if ln.strip()][-15:]
    # condition checks from status output (ConditionPathExists etc.)
    cond = []
    for ln in (out_s or "").splitlines():
        if "Condition" in ln and "=" in ln:
            cond.append(ln.strip())
    return {
        "unit": unit,
        "active": state or "?",
        "journal_tail": journal_lines,
        "condition_lines": cond[:5],
    }


def collect_host():
    out = {}
    try:
        with open("/proc/uptime") as f:
            out["uptime_seconds"] = int(float(f.read().split()[0]))
    except OSError:
        pass
    try:
        with open("/proc/meminfo") as f:
            mem = {ln.split(":")[0].strip(): ln.split(":")[1].strip()
                   for ln in f if ":" in ln}
        out["mem_total"] = mem.get("MemTotal", "")
        out["mem_available"] = mem.get("MemAvailable", "")
    except OSError:
        pass
    out["hostname"] = socket.gethostname()
    out["kernel"] = os.uname().release
    return out


# ----- chronicle update -----

def get_doc(conf, doc_id):
    """GET a chronicle doc; raise on error. Used to consult TJ.status before
    the service-autostart heuristic so we don't revive a converged job."""
    base = conf["CHRONICLE_URL"].rstrip("/")
    db = conf.get("CHRONICLE_DB", "lnq")
    r = requests.get(f"{base}/{db}/{doc_id}",
                     auth=("admin", conf["COUCHDB_ADMIN_PASSWORD"]),
                     timeout=15)
    r.raise_for_status()
    return r.json()


def put_telemetry(conf, training_job_id, telemetry):
    base = conf["CHRONICLE_URL"].rstrip("/")
    db = conf.get("CHRONICLE_DB", "lnq")
    s = requests.Session(); s.auth = ("admin", conf["COUCHDB_ADMIN_PASSWORD"])
    r = s.get(f"{base}/{db}/{training_job_id}", timeout=15)
    r.raise_for_status()
    doc = r.json()
    doc["runtime_telemetry"] = telemetry
    # Bump heartbeat so the dashboard's "last_heartbeat_at" stays fresh too.
    doc["last_heartbeat_at"] = telemetry["at"]
    r = s.put(f"{base}/{db}/{training_job_id}",
              json=doc, headers={"Content-Type": "application/json"},
              timeout=15)
    if not r.ok:
        try:
            reason = r.json().get("reason", r.text)
        except Exception:
            reason = r.text
        raise RuntimeError(f"PUT failed: {r.status_code} {reason}")


def main():
    conf = load_kv(CONF_PATH)
    job_conf = load_kv(JOB_CONF_PATH)
    if "CHRONICLE_URL" not in conf or "COUCHDB_ADMIN_PASSWORD" not in conf:
        print("missing chronicle config", file=sys.stderr); sys.exit(2)
    training_job_id = job_conf.get("TRAINING_JOB_ID")
    if not training_job_id:
        print("missing TRAINING_JOB_ID in job.conf", file=sys.stderr); sys.exit(2)
    mount_point = (conf.get("MOUNT_POINT") or
                   job_conf.get("MOUNT_POINT") or
                   "/media/share/LNQ-data")
    telemetry = {
        "at": iso_now(),
        "host": collect_host(),
        "cloud_init": collect_cloud_init(),
        "manila": collect_manila(mount_point),
        "gpu": collect_gpu(),
        "service": collect_service(),
        "system_journal": collect_system_journal(),
        "apt": collect_apt(),
    }

    # If the GPU is healthy AND the trainer-runner is sitting inactive,
    # kick it off. Works around an observed systemd-enable-vs-boot ordering
    # quirk where the runner is `Loaded: enabled` but doesn't actually start
    # at multi-user.target. We'd rather burn one stray systemctl call than
    # ask the user to SSH and bring it up by hand.
    #
    # SKIP the auto-start when the TrainingJob has already reached a terminal
    # state. Otherwise we re-launch nnUNetv2_train from epoch 0 right after
    # the trainer-runner exits cleanly on convergence, which silently destroys
    # the just-written checkpoint_best.pth.
    terminal_status = None
    try:
        tj = get_doc(conf, training_job_id)
        terminal_status = tj.get("status") if isinstance(tj, dict) else None
    except Exception as exc:
        telemetry["service_autostart_lookup_error"] = str(exc)[:200]
    is_terminal = terminal_status in ("converged", "shelved", "failed-final", "cancelled")
    if (telemetry["gpu"].get("ok")
            and telemetry["service"].get("active") == "inactive"
            and not is_terminal):
        run_capture(["systemctl", "start", "lnq-trainer-runner.service"], timeout=20)
        telemetry["service_autostart"] = "triggered (gpu.ok + svc.inactive)"
    elif is_terminal:
        telemetry["service_autostart"] = f"skipped (TJ status={terminal_status})"
    put_telemetry(conf, training_job_id, telemetry)


if __name__ == "__main__":
    main()
