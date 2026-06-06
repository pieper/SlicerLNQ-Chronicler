#!/usr/bin/env bash
# poll-and-shelve.sh — wait for a remote ingest/inference process to finish,
# then shelve the host. Designed to run on the Mac so we don't need to put
# openstack credentials on the GPU box.
#
# Defaults match the current IDC mediastinal batch on lnq-inferencer.
# Pass --instance <name> --pattern <pgrep substring> to repurpose.

set -eu

INSTANCE="lnq-inferencer"
PATTERN="ingest-idc-cohort.py"
POLL_SECONDS=300                  # 5 min between probes
CONFIRM_TICKS=3                   # require 3 consecutive empties before shelving
LOG="/tmp/poll-and-shelve-${INSTANCE}.log"

while [ $# -gt 0 ]; do
  case "$1" in
    --instance)        INSTANCE="$2"; shift 2;;
    --pattern)         PATTERN="$2";  shift 2;;
    --poll-seconds)    POLL_SECONDS="$2"; shift 2;;
    --confirm-ticks)   CONFIRM_TICKS="$2"; shift 2;;
    --log)             LOG="$2"; shift 2;;
    -h|--help)
      sed -n '2,/^$/p' "$0" | sed 's/^# //;s/^#//'
      exit 0;;
    *)
      echo "unknown arg: $1" >&2; exit 2;;
  esac
done

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
# shellcheck disable=SC1091
. trainer.conf

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
log() { echo "$(ts) $*" | tee -a "$LOG"; }

log "watcher start instance=$INSTANCE pattern=$PATTERN poll=${POLL_SECONDS}s confirm=$CONFIRM_TICKS"

empties=0
while true; do
  sleep "$POLL_SECONDS"
  # `|| true` so we don't exit on transient ssh failures.
  running=$(./bin/trainer.sh ssh "$INSTANCE" "pgrep -af '$PATTERN' | grep -v 'pgrep -af' || true" 2>/dev/null || true)
  if [ -n "$running" ]; then
    empties=0
    log "still running on $INSTANCE"
    continue
  fi
  empties=$((empties + 1))
  log "no $PATTERN match on $INSTANCE ($empties/$CONFIRM_TICKS confirmations)"
  if [ "$empties" -ge "$CONFIRM_TICKS" ]; then
    log "shelving $INSTANCE"
    openstack --os-cloud "$OS_CLOUD" server shelve "$INSTANCE" \
      2>&1 | tee -a "$LOG" || true
    log "watcher done"
    exit 0
  fi
done
