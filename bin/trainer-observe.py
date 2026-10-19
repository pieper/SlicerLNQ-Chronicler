#!/usr/bin/env python3
"""trainer-observe.py — Mac-side openstack observer for one or more TrainingJobs.

Polls openstack every 15 s for each instance backing a TrainingJob: server
status, addresses, FIP attachment + status, console log tail. Writes the
state to TrainingJob.openstack_state on every tick and appends a new entry
to TrainingJob.runtime_events whenever something notable changes
(`status: BUILD->ACTIVE`, `fip: DOWN->ACTIVE`, etc.). The dashboard renders
both; the user never has to run openstack CLI to ask "is anything happening".

Pair with bin/trainer.sh create — single-fold:

    bin/trainer.sh create --training-job-id <id> --name lnq-trainer-fold0
    bin/trainer-observe.py --pair <id> lnq-trainer-fold0 &

Or multi-fold (one process polls all of them):

    bin/trainer-observe.py \\
      --pair trainingjob:abc... lnq-trainer-fold0 \\
      --pair trainingjob:def... lnq-trainer-fold1 \\
      --pair trainingjob:ghi... lnq-trainer-fold2 &

The legacy two-flag form (--training-job-id + --name) is still accepted as
shorthand for a single pair.

The observer exits when every pair's TrainingJob status transitions to one
of {converged, failed, cancelled} AND each saw a non-terminal status during
this run.

Reads chronicle.conf for credentials. Uses OS_CLOUD from trainer.conf for
the openstack CLI. Stdlib-only — no pip deps, so plain python3 works.
"""

from __future__ import annotations

import argparse
import base64
import datetime
import json
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request


def iso_now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_conf(path):
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


def run(cmd, timeout=20):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"
    except FileNotFoundError as e:
        return 127, "", str(e)


def os_show(os_cloud, kind, name_or_id, timeout=20):
    """openstack <kind> show <name> -f json. Returns dict or None."""
    rc, out, err = run(
        ["openstack", "--os-cloud", os_cloud, kind, "show", name_or_id, "-f", "json"],
        timeout=timeout)
    if rc != 0:
        return None
    try:
        return json.loads(out)
    except json.JSONDecodeError:
        return None


def console_tail(os_cloud, name, max_chars=4096):
    rc, out, _err = run(
        ["openstack", "--os-cloud", os_cloud, "console", "log", "show", name],
        timeout=30)
    if rc != 0 or not out:
        return None
    if len(out) > max_chars:
        return out[-max_chars:]
    return out


def parse_addresses(addr_field):
    """Normalize openstack `addresses` to {network: [ips]}.

    `openstack server show -f json` returns this as a dict already (recent
    osc); older builds return a string like
    'auto_allocated_network=10.2.87.107, 203.0.113.17'. Handle both."""
    if isinstance(addr_field, dict):
        return {net: list(ips) for net, ips in addr_field.items() if ips}
    out = {}
    if not isinstance(addr_field, str):
        return out
    for entry in addr_field.split(";"):
        if "=" not in entry:
            continue
        net, ips = entry.split("=", 1)
        out[net.strip()] = [ip.strip() for ip in ips.split(",")]
    return out


def find_floating_ip(addresses):
    """Return the first public IP across all networks, or None."""
    for ips in addresses.values():
        for ip in ips:
            if ip.startswith("149.165."):
                return ip
    return None


def gather_state(os_cloud, instance_name):
    """Snapshot openstack-side state. None fields signal 'not available'."""
    snapshot = {"polled_at": iso_now()}
    server = os_show(os_cloud, "server", instance_name)
    if server is None:
        snapshot["server"] = {"error": "instance not found"}
        return snapshot
    snapshot["server"] = {
        "id": server.get("id"),
        "name": server.get("name"),
        "status": server.get("status"),
        "vm_state": server.get("OS-EXT-STS:vm_state"),
        "power_state": server.get("OS-EXT-STS:power_state"),
        "task_state": server.get("OS-EXT-STS:task_state"),
        "created": server.get("created"),
        "launched_at": server.get("OS-SRV-USG:launched_at"),
        "flavor": (server.get("flavor") or {}).get("original_name") or server.get("flavor"),
        "image": server.get("image"),
        "addresses": parse_addresses(server.get("addresses") or ""),
        "security_groups": server.get("security_groups"),
    }
    fip = find_floating_ip(snapshot["server"]["addresses"])
    snapshot["server"]["floating_ip"] = fip
    if fip:
        fip_obj = os_show(os_cloud, "floating ip", fip)
        if fip_obj:
            snapshot["fip"] = {
                "address": fip,
                "status": fip_obj.get("status"),
                "port_id": fip_obj.get("port_id"),
                "fixed_ip_address": fip_obj.get("fixed_ip_address"),
            }
    # Console log: last ~4KB. Only if server is ACTIVE (otherwise console returns nothing).
    if snapshot["server"]["status"] == "ACTIVE":
        tail = console_tail(os_cloud, instance_name)
        if tail:
            lines = [ln for ln in tail.splitlines() if ln.strip()]
            snapshot["console_log_tail"] = lines[-25:]
    return snapshot


# ---- chronicle helpers (mutable update on TrainingJob) ----

class Chronicle:
    """Stdlib-only HTTP basic-auth client. No requests dependency so plain
    /usr/bin/env python3 works on the Mac without an extra pip install."""

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
            try:
                detail = json.loads(exc.read()).get("reason", "")
            except Exception:
                detail = ""
            raise RuntimeError(f"{method} {doc_id}: HTTP {exc.code} {detail}") from None

    def get(self, doc_id):
        return self._request("GET", doc_id)

    def put(self, doc):
        return self._request("PUT", doc["_id"], body=doc)


def diff_events(prev, cur):
    """Walk a few interesting fields; emit an event string per change."""
    events = []
    def g(d, *path):
        for p in path:
            if not isinstance(d, dict): return None
            d = d.get(p)
        return d
    ps = g(prev or {}, "server", "status")
    cs = g(cur, "server", "status")
    if ps != cs:
        events.append(f"server.status: {ps or '∅'} -> {cs or '∅'}")
    pf = g(prev or {}, "fip", "status")
    cf = g(cur, "fip", "status")
    if pf != cf:
        events.append(f"fip.status: {pf or '∅'} -> {cf or '∅'}")
    pa = g(prev or {}, "server", "addresses")
    ca = g(cur, "server", "addresses") or {}
    p_has_fip = any(any(ip.startswith("149.165.") for ip in ips) for ips in (pa or {}).values())
    c_has_fip = any(any(ip.startswith("149.165.") for ip in ips) for ips in ca.values())
    if p_has_fip != c_has_fip:
        events.append(f"server.fip_attached: {p_has_fip} -> {c_has_fip}")
    pt = g(prev or {}, "server", "task_state")
    ct = g(cur, "server", "task_state")
    if pt != ct and (pt or ct):
        events.append(f"server.task_state: {pt or '∅'} -> {ct or '∅'}")
    return events


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    # Multi-instance form (preferred).
    ap.add_argument("--pair", nargs=2, action="append", default=[],
                    metavar=("TRAINING_JOB_ID", "INSTANCE_NAME"),
                    help="Repeatable. Each --pair adds one (job, instance) target.")
    # Legacy single-instance form — keeps prior cmdlines working.
    ap.add_argument("--training-job-id")
    ap.add_argument("--name", help="openstack server name (legacy form)")
    ap.add_argument("--conf", default="chronicle.conf")
    ap.add_argument("--trainer-conf", default="trainer.conf")
    ap.add_argument("--interval", type=int, default=15, help="poll interval s (default 15)")
    ap.add_argument("--max-events", type=int, default=200,
                    help="cap on runtime_events array length")
    args = ap.parse_args()

    pairs = list(args.pair)
    if args.training_job_id and args.name:
        pairs.append([args.training_job_id, args.name])
    if not pairs:
        ap.error("at least one --pair (or --training-job-id + --name) is required")

    if not os.path.isfile(args.conf):
        print(f"Missing {args.conf}", file=sys.stderr); sys.exit(1)
    conf = load_conf(args.conf)
    tconf = load_conf(args.trainer_conf)
    os_cloud = tconf.get("OS_CLOUD")
    if not os_cloud:
        print("Missing OS_CLOUD in trainer.conf", file=sys.stderr); sys.exit(1)

    chronicle_url = conf.get("CHRONICLE_URL")
    if not chronicle_url and conf.get("DOMAIN_NAME"):
        chronicle_url = f"https://{conf['DOMAIN_NAME']}"
    if not chronicle_url:
        print("Missing CHRONICLE_URL or DOMAIN_NAME in chronicle.conf", file=sys.stderr)
        sys.exit(1)
    chronicle = Chronicle(
        chronicle_url, conf.get("CHRONICLE_DB", "lnq"),
        conf["COUCHDB_ADMIN_PASSWORD"])

    # Per-pair state.
    pair_state = {jid: {"prev": None, "saw_non_terminal": False, "done": False,
                        "instance": inst}
                  for jid, inst in pairs}
    print(f"[observer] os_cloud={os_cloud} interval={args.interval}s "
          f"pairs={len(pairs)}", flush=True)

    # Seed one start event on each so the user can tell the observer is alive.
    for jid, inst in pairs:
        try:
            doc = chronicle.get(jid)
            ev = list(doc.get("runtime_events") or [])
            ev.append({"at": iso_now(), "source": "observer",
                       "msg": f"observer started for instance {inst!r}"})
            doc["runtime_events"] = ev[-args.max_events:]
            chronicle.put(doc)
            print(f"[observer] seeded {inst} ({jid})", flush=True)
        except Exception as exc:
            print(f"[observer] seed {inst} failed: {exc}", flush=True)

    while True:
        active = [(jid, inst) for jid, inst in pairs if not pair_state[jid]["done"]]
        if not active:
            print("[observer] all pairs terminal; exiting.", flush=True)
            return
        for jid, inst in active:
            s = pair_state[jid]
            try:
                cur = gather_state(os_cloud, inst)
            except Exception as exc:
                print(f"[observer] gather {inst}: {exc}", flush=True)
                continue
            deltas = diff_events(s["prev"], cur)
            try:
                doc = chronicle.get(jid)
            except Exception as exc:
                print(f"[observer] GET {jid}: {exc}", flush=True)
                continue
            doc["openstack_state"] = cur
            ev = list(doc.get("runtime_events") or [])
            for d in deltas:
                ev.append({"at": iso_now(), "source": "observer", "msg": d})
                print(f"[observer] {inst} event: {d}", flush=True)
            doc["runtime_events"] = ev[-args.max_events:]
            try:
                chronicle.put(doc)
            except Exception as exc:
                print(f"[observer] PUT {jid}: {exc}", flush=True)
            s["prev"] = cur
            st = doc.get("status")
            if st in ("pending", "running"):
                s["saw_non_terminal"] = True
            elif st in ("converged", "failed", "cancelled") and s["saw_non_terminal"]:
                s["done"] = True
                print(f"[observer] {inst} terminal ({st}); dropping from rotation.", flush=True)
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
