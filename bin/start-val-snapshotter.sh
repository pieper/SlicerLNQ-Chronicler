#!/usr/bin/env bash
# Start lnq-val-snapshotter.py for the TrainingJob bound to this trainer.
#
# Reads /etc/lnq/job.conf for TRAINING_JOB_ID, /opt/lnq/chronicle.conf for the
# chronicle creds, GETs the TJ doc, pulls dataset_id / task_name / fold /
# training_workspace, then nohups the snapshotter so it survives this shell.
# Idempotent — kills any prior snapshotter on this host first.
#
# Expects /opt/lnq/lnq-val-snapshotter.py to already be present (curl'd from
# BIN_BASE_URL by cloud-init or by trainer-watcher.py).

set -eu

CONF_CHRONICLE=/opt/lnq/chronicle.conf
CONF_JOB=/etc/lnq/job.conf
SNAP_PY=/opt/lnq/lnq-val-snapshotter.py
PY=/opt/lnq/nnenv/bin/python
LOG=/var/log/lnq-val-snapshotter.log
EVERY_N=${EVERY_N_EPOCHS:-10}
MAX_VAL_CASES=${MAX_VAL_CASES:-4}

[ -r "$CONF_CHRONICLE" ] || { echo "missing $CONF_CHRONICLE"; exit 2; }
[ -r "$CONF_JOB" ]       || { echo "missing $CONF_JOB"; exit 2; }
[ -x "$SNAP_PY" ]        || { echo "missing $SNAP_PY"; exit 2; }
[ -x "$PY" ]             || { echo "missing $PY"; exit 2; }

# shellcheck disable=SC1090
. "$CONF_CHRONICLE"
# shellcheck disable=SC1090
. "$CONF_JOB"

: "${CHRONICLE_URL:?}"; : "${COUCHDB_ADMIN_PASSWORD:?}"
: "${CHRONICLE_DB:=lnq}"
: "${TRAINING_JOB_ID:?}"

doc=$(curl -fsS -u "admin:${COUCHDB_ADMIN_PASSWORD}" \
       "${CHRONICLE_URL}/${CHRONICLE_DB}/${TRAINING_JOB_ID}")

# Tiny jq-free extractor. Inputs are well-formed JSON written by us.
get() { python3 -c "import json,sys; d=json.loads(sys.stdin.read()); v=(d.get('config') or {}).get('$1'); print('' if v is None else v)" <<<"$doc"; }
DATASET_ID=$(get dataset_id)
TASK=$(get task_name)
FOLD=$(get fold)
WS=$(get training_workspace)
PLANS=$(get plans_identifier); [ -n "$PLANS" ] || PLANS=nnUNetResEncUNetMPlans
CFG=$(get config_name);       [ -n "$CFG" ]   || CFG=3d_fullres
TRAINER=$(get trainer_class); [ -n "$TRAINER" ] || TRAINER=nnUNetTrainer
[ -n "$WS" ] || WS=${NNUNET_BASE:-}

if [ -z "$DATASET_ID" ] || [ -z "$TASK" ] || [ -z "$FOLD" ] || [ -z "$WS" ]; then
  echo "could not derive snapshotter args from TJ ${TRAINING_JOB_ID}"
  exit 3
fi

# Kill any existing snapshotter on this host so we don't end up with stale
# args after an axillary repurpose.
pkill -f lnq-val-snapshotter.py || true
sleep 1

echo "=== $(date -u +%FT%TZ) start-val-snapshotter ds=$DATASET_ID task=$TASK fold=$FOLD ws=$WS every=$EVERY_N ===" >> "$LOG"
nohup "$PY" "$SNAP_PY" \
  --nnunet-base "$WS" \
  --dataset-id "$DATASET_ID" \
  --task "$TASK" \
  --plans-identifier "$PLANS" \
  --config "$CFG" \
  --trainer-class "$TRAINER" \
  --fold "$FOLD" \
  --every-n-epochs "$EVERY_N" \
  --max-val-cases "$MAX_VAL_CASES" \
  --pause-train \
  >> "$LOG" 2>&1 < /dev/null &
disown || true
echo "snapshotter pid=$! tj=${TRAINING_JOB_ID##*:} fold=$FOLD task=$TASK"
