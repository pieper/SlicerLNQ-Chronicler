#!/usr/bin/env python3
"""jetstream-status.py — pre-flight Js2 status check.

Hits https://api.status.io/1.0/status/<jetstream2-page-id> and prints a tight
summary of:
  - overall status (operational / degraded / outage)
  - active incidents (optionally filtered to keywords like flavor names)
  - upcoming maintenance within the next N hours

Exit codes:
  0 — all clear (or warnings only)
  3 — active incident matching the filter (e.g. the flavor you're about to
      launch is named in an open incident)
  4 — outage / partial outage on the primary cloud
  5 — couldn't reach the status API

The dashboard's header indicator reads the same JSON output via --json.
"""

from __future__ import annotations

import argparse
import datetime
import json
import re
import sys

try:
    import requests
except ImportError:
    print("Need 'requests'.", file=sys.stderr); sys.exit(5)


# Jetstream2's status.io page id (from jetstream.status.io HTML).
JS2_PAGE_ID = "61dc808a7e9a82053ce739d2"
STATUS_API = f"https://api.status.io/1.0/status/{JS2_PAGE_ID}"
STATUS_DOCS_URL = "https://docs.jetstream-cloud.org/overview/status/"
STATUS_HTML_URL = "https://jetstream.status.io/"


def fetch(timeout=10):
    r = requests.get(STATUS_API, timeout=timeout)
    r.raise_for_status()
    return r.json().get("result") or {}


def summarize(result, flavor_keywords=None, upcoming_hours=24):
    """Return a structured summary suitable for shell rendering or JSON."""
    overall = result.get("status_overall") or {}
    incidents = result.get("incidents") or []
    maint_blk = result.get("maintenance") or {}
    upcoming_maint = maint_blk.get("upcoming") if isinstance(maint_blk, dict) else []

    # Active incident filter: any keyword (case-insensitive) appearing in
    # incident name or first message details counts as a "matched" incident.
    keywords = [k.lower() for k in (flavor_keywords or [])]
    matched = []
    other = []
    for inc in incidents:
        text = (inc.get("name") or "") + " "
        msgs = inc.get("messages") or []
        if msgs:
            text += msgs[0].get("details", "") or ""
        text_l = text.lower()
        item = {
            "name": inc.get("name", "?"),
            "status": (inc.get("current_status") or {}).get("status", "?"),
            "details": (msgs[0].get("details", "") if msgs else "")[:240],
            "shortlink": inc.get("shortlink") or STATUS_HTML_URL,
        }
        (matched if keywords and any(k in text_l for k in keywords) else other).append(item)

    # Upcoming maintenance within the window.
    now = datetime.datetime.now(datetime.timezone.utc)
    horizon = now + datetime.timedelta(hours=upcoming_hours)
    soon_maint = []
    for m in (upcoming_maint or []):
        scheduled = m.get("datetime_planned_start", "")
        try:
            t = datetime.datetime.fromisoformat(scheduled.replace("Z", "+00:00"))
            if now <= t <= horizon:
                soon_maint.append({"name": m.get("name", "?"),
                                   "starts_at": scheduled})
        except Exception:
            # Couldn't parse; include conservatively if the announced window
            # is anywhere in the future.
            soon_maint.append({"name": m.get("name", "?"),
                               "starts_at": scheduled or "?"})

    return {
        "overall_status": overall.get("status") or "?",
        "overall_status_code": overall.get("status_code") or 0,
        "is_operational": (overall.get("status_code") == 100),
        "incidents_matched": matched,
        "incidents_other": other,
        "maintenance_soon": soon_maint,
        "maintenance_upcoming_total": len(upcoming_maint or []),
        "checked_at": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def render_text(summary, flavor_keywords):
    lines = []
    op = summary["is_operational"]
    overall = summary["overall_status"]
    lines.append(f"Jetstream2 overall: {overall}")
    if not op:
        lines.append(f"  !!! NOT operational — review {STATUS_HTML_URL} before launching.")
    if summary["incidents_matched"]:
        lines.append("")
        lines.append(f"Open incidents matching keywords {flavor_keywords or '(none)'}:")
        for inc in summary["incidents_matched"]:
            lines.append(f"  * [{inc['status']}] {inc['name']}")
            if inc["details"]:
                lines.append(f"      {inc['details'][:200]}")
            lines.append(f"      {inc['shortlink']}")
    if summary["incidents_other"]:
        lines.append("")
        lines.append(f"Other open incidents ({len(summary['incidents_other'])}):")
        for inc in summary["incidents_other"][:3]:
            lines.append(f"  - [{inc['status']}] {inc['name']}")
    if summary["maintenance_soon"]:
        lines.append("")
        lines.append("Upcoming maintenance (within window):")
        for m in summary["maintenance_soon"]:
            lines.append(f"  ! {m['name']}  starts {m['starts_at']}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--flavor", default=None,
                    help="Flavor name to match against open incidents (e.g. g3.xl). "
                         "Listed multiple times accepted; first ',' splits.")
    ap.add_argument("--keyword", action="append", default=[],
                    help="Additional keyword to match (repeatable).")
    ap.add_argument("--upcoming-hours", type=int, default=24,
                    help="Window for upcoming-maintenance warning (default 24).")
    ap.add_argument("--json", action="store_true",
                    help="Emit the summary as JSON instead of text.")
    args = ap.parse_args()

    keywords = list(args.keyword)
    if args.flavor:
        keywords.extend(args.flavor.split(","))

    try:
        result = fetch()
    except requests.RequestException as exc:
        msg = f"Could not reach status.io: {exc}"
        if args.json:
            print(json.dumps({"error": msg}))
        else:
            print(msg, file=sys.stderr)
        sys.exit(5)

    summary = summarize(result, flavor_keywords=keywords)

    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        print(render_text(summary, keywords))

    if not summary["is_operational"]:
        sys.exit(4)
    if summary["incidents_matched"]:
        sys.exit(3)
    sys.exit(0)


if __name__ == "__main__":
    main()
