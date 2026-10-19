#!/usr/bin/env bash
# Trainer lifecycle for one Js2 GPU instance running one nnU-Net fold.
# Reads ./trainer.conf, talks to OpenStack via the openstack CLI.
#
# Usage:
#   bin/trainer.sh create  --training-job-id <id> [--name <instance>] [--no-fip]
#                          [--flavor <flavor>]
#   bin/trainer.sh status  [<name>]
#   bin/trainer.sh ssh     <name> [args]
#   bin/trainer.sh logs    <name>                # tails cloud-init, then training_log
#   bin/trainer.sh shelve  <name>
#   bin/trainer.sh unshelve <name>
#   bin/trainer.sh destroy <name>
#   bin/trainer.sh list                          # all trainers in the project

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_FILE="${TRAINER_CONF:-./trainer.conf}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Missing $CONFIG_FILE" >&2
  echo "Copy trainer.conf.example to trainer.conf and fill in your values." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

# Required vars.
: "${OS_CLOUD:?}" "${IMAGE:?}" "${FLAVOR:?}" "${NETWORK:?}"
: "${KEY_NAME:?}" "${SSH_KEY:?}" "${SSH_USER:?}"
: "${CHRONICLE_URL:?}" "${CHRONICLE_DB:?}" "${COUCHDB_ADMIN_PASSWORD:?}"
: "${MANILA_SHARE_NAME:?}" "${MANILA_SHARE_PATH:?}"
: "${MANILA_ACCESS_RULE_NAME:?}" "${MANILA_ACCESS_RULE_KEY:?}"
: "${BIN_BASE_URL:?Set BIN_BASE_URL in trainer.conf to the raw URL prefix where trainer-runner.py + export-nnunet-dataset.py can be fetched (gist or repo).}"
: "${SECURITY_GROUPS:=default,lnq-trainer}"
: "${PUBLIC_SECGROUP:=lnq-trainer}"
: "${MOUNT_POINT:=/media/share/LNQ-data}"
: "${NNUNET_BASE:=$MOUNT_POINT/Inguinal_Syed}"

OS=(openstack --os-cloud "$OS_CLOUD")

# --- helpers ---------------------------------------------------------
get_addresses() {
  "${OS[@]}" server show "$1" -f value -c addresses 2>/dev/null \
    | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' || true
}

# First public-looking IP attached to the server, if any.
get_fip_for() {
  local addrs
  addrs=$(get_addresses "$1")
  # FIP from config is most reliable; check if it's actually attached.
  if [ -n "${FLOATING_IP:-}" ] && echo "$addrs" | grep -qx "$FLOATING_IP"; then
    echo "$FLOATING_IP"; return
  fi
  # Otherwise: any 149.165.* (Js2 public range)
  echo "$addrs" | grep -E '^149\.165\.' | head -1 || true
}

# Internal IP (RFC 1918) for the server.
get_internal_for() {
  get_addresses "$1" | grep -vE '^149\.165\.' | head -1 || true
}

# Idempotent SSH-22 security group.
ensure_secgroup() {
  if "${OS[@]}" security group show "$PUBLIC_SECGROUP" >/dev/null 2>&1; then
    return 0
  fi
  echo "Creating $PUBLIC_SECGROUP security group (TCP 22 from 0.0.0.0/0)..."
  "${OS[@]}" security group create \
    --description "SSH only — LNQ trainer" \
    "$PUBLIC_SECGROUP" >/dev/null
  "${OS[@]}" security group rule create \
    --protocol tcp --dst-port 22 --remote-ip 0.0.0.0/0 \
    "$PUBLIC_SECGROUP" >/dev/null
}

# --- commands --------------------------------------------------------
cmd_create() {
  local training_job_id="" name="" want_fip=1 flavor="$FLAVOR" smoke_test=0
  local skip_status=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --training-job-id)   training_job_id="$2"; shift 2 ;;
      --name)              name="$2"; shift 2 ;;
      --no-fip)            want_fip=0; shift ;;
      --flavor)            flavor="$2"; shift 2 ;;
      --smoke-test)        smoke_test=1; shift ;;
      --skip-status-check) skip_status=1; shift ;;
      *) echo "Unknown create arg: $1" >&2; exit 1 ;;
    esac
  done

  # --- Js2 status pre-flight --------------------------------------------
  if [ "$skip_status" -eq 0 ] && [ -x "bin/jetstream-status.py" ]; then
    local py="${LNQ_PYTHON:-python3}"
    set +e
    "$py" bin/jetstream-status.py --flavor "$flavor" --keyword GPU
    local rc=$?
    set -e
    case "$rc" in
      0) ;;  # all clear
      3) echo
         echo "*** Jetstream2 has an OPEN INCIDENT matching '$flavor' / GPU above. ***"
         echo "    Launching anyway is likely to fail. To proceed: rerun with --skip-status-check."
         exit 3 ;;
      4) echo
         echo "*** Jetstream2 reports a non-operational status. ***"
         echo "    Review https://jetstream.status.io/ before proceeding."
         echo "    To proceed anyway: rerun with --skip-status-check."
         exit 4 ;;
      5) echo "(could not reach status.io — continuing without pre-flight)" ;;
      *) echo "(jetstream-status.py exited $rc — continuing)" ;;
    esac
  fi

  if [ -z "$training_job_id" ]; then
    echo "Required: --training-job-id trainingjob:..." >&2; exit 1
  fi
  if [ -z "$name" ]; then
    # Derive from the job id's tail + fold (if visible in the id, otherwise short uuid)
    name="lnq-trainer-${training_job_id##*:}"
    name="${name:0:48}"
  fi

  if "${OS[@]}" server show "$name" >/dev/null 2>&1; then
    echo "Instance '$name' already exists. Destroy it or pick a new --name." >&2
    exit 1
  fi

  ensure_secgroup

  # Look up the chronicle's internal IP so the trainer's cloud-init can
  # pin lnq-chronicle.isomics.dev to it in /etc/hosts. Working around Js2's
  # FIP-hairpin issue (trainer SNAT to the chronicle's FIP gets dropped on
  # this project's L2). Empty string is fine; cloud-init skips the rewrite.
  local chronicle_internal_ip=""
  if "${OS[@]}" server show lnq-chronicle >/dev/null 2>&1; then
    chronicle_internal_ip=$("${OS[@]}" server show lnq-chronicle -f json 2>/dev/null \
      | python3 -c "
import sys, json
d = json.load(sys.stdin); addrs = d.get('addresses') or {}
for ips in addrs.values():
    for ip in ips:
        if ip.startswith('10.'): print(ip); sys.exit(0)
")
    [ -n "$chronicle_internal_ip" ] && echo "Chronicle internal IP: $chronicle_internal_ip"
  fi

  # Render cloud-init from the template.
  local userdata
  userdata=$(mktemp)
  CHRONICLE_URL="$CHRONICLE_URL" \
  CHRONICLE_DB="$CHRONICLE_DB" \
  COUCHDB_ADMIN_PASSWORD="$COUCHDB_ADMIN_PASSWORD" \
  CHRONICLE_INTERNAL_IP="$chronicle_internal_ip" \
  MANILA_SHARE_NAME="$MANILA_SHARE_NAME" \
  MANILA_SHARE_PATH="$MANILA_SHARE_PATH" \
  MANILA_ACCESS_RULE_NAME="$MANILA_ACCESS_RULE_NAME" \
  MANILA_ACCESS_RULE_KEY="$MANILA_ACCESS_RULE_KEY" \
  MOUNT_POINT="$MOUNT_POINT" \
  NNUNET_BASE="$NNUNET_BASE" \
  TRAINING_JOB_ID="$training_job_id" \
  SMOKE_TEST="$smoke_test" \
  BIN_BASE_URL="$BIN_BASE_URL" \
    envsubst '${CHRONICLE_URL} ${CHRONICLE_DB} ${COUCHDB_ADMIN_PASSWORD} ${CHRONICLE_INTERNAL_IP} ${MANILA_SHARE_NAME} ${MANILA_SHARE_PATH} ${MANILA_ACCESS_RULE_NAME} ${MANILA_ACCESS_RULE_KEY} ${MOUNT_POINT} ${NNUNET_BASE} ${TRAINING_JOB_ID} ${SMOKE_TEST} ${BIN_BASE_URL}' \
    < ops/cloud-init/trainer.yml.tmpl \
    > "$userdata"

  local sg_args=()
  local IFS=','
  for sg in $SECURITY_GROUPS; do
    sg_args+=(--security-group "$sg")
  done
  unset IFS

  local mode_str
  if [ "$smoke_test" -eq 1 ]; then mode_str="SMOKE TEST"; else mode_str="full training"; fi
  echo "Creating $name ($flavor / $IMAGE) for TrainingJob $training_job_id [$mode_str]..."
  "${OS[@]}" server create \
    --image "$IMAGE" \
    --flavor "$flavor" \
    --network "$NETWORK" \
    --key-name "$KEY_NAME" \
    "${sg_args[@]}" \
    --user-data "$userdata" \
    --wait \
    "$name" >/dev/null

  rm -f "$userdata"

  if [ "$want_fip" -eq 1 ] && [ -n "${FLOATING_IP:-}" ]; then
    # Attach only if FIP isn't already in use elsewhere.
    local fip_status
    fip_status=$("${OS[@]}" floating ip show "$FLOATING_IP" -f value -c status 2>/dev/null || true)
    if [ "$fip_status" = "DOWN" ] || [ -z "$fip_status" ]; then
      echo "Attaching $FLOATING_IP..."
      "${OS[@]}" server add floating ip "$name" "$FLOATING_IP"
    else
      echo "FIP $FLOATING_IP is already in use (status=$fip_status); skipping. Use --no-fip to silence."
    fi
  fi

  echo
  echo "Created. First boot installs CUDA + reboots; second boot launches the runner."
  echo "  Watch:    $0 logs $name"
  echo "  SSH:      $0 ssh $name"
  echo "  Dashboard reflects training progress via the Chronicle changes feed."
}

cmd_status() {
  if [ $# -eq 0 ]; then
    "${OS[@]}" server list --name '^lnq-trainer-' -c Name -c Status -c Networks -c Image -c Flavor
    return
  fi
  "${OS[@]}" server show "$1" -c name -c status -c addresses -c image -c flavor
}

cmd_ssh() {
  local name="${1:?Usage: $0 ssh <name> [args]}"; shift
  local fip
  fip=$(get_fip_for "$name")
  local ssh_args=(-i "$SSH_KEY" -o StrictHostKeyChecking=accept-new)
  if [ -n "$fip" ]; then
    exec ssh "${ssh_args[@]}" "$SSH_USER@$fip" "$@"
  fi
  if [ -z "${JUMP_HOST:-}" ]; then
    echo "$name has no FIP and JUMP_HOST is not set in trainer.conf." >&2
    exit 1
  fi
  local internal
  internal=$(get_internal_for "$name")
  if [ -z "$internal" ]; then
    echo "Could not determine internal IP for $name." >&2; exit 1
  fi
  exec ssh "${ssh_args[@]}" -J "${JUMP_USER:-ubuntu}@$JUMP_HOST" "$SSH_USER@$internal" "$@"
}

cmd_logs() {
  local name="${1:?Usage: $0 logs <name>}"
  # Tail cloud-init output until 'finished' marker appears, then switch to
  # the trainer service's journal + nnU-Net training log.
  cmd_ssh "$name" '
    set -e
    if ! sudo test -e /run/cloud-init/result.json; then
      sudo tail -F /var/log/cloud-init-output.log
    else
      echo "[cloud-init finished. switching to lnq-trainer-runner.service]"
      sudo journalctl -u lnq-trainer-runner.service --no-pager -n 100 -f
    fi
  '
}

cmd_shelve() {
  local name="${1:?Usage: $0 shelve <name>}"
  echo "Shelving $name..."
  "${OS[@]}" server shelve "$name"
}

cmd_unshelve() {
  local name="${1:?Usage: $0 unshelve <name>}"
  echo "Unshelving $name..."
  "${OS[@]}" server unshelve "$name"
}

cmd_destroy() {
  local name="${1:?Usage: $0 destroy <name>}"
  if ! "${OS[@]}" server show "$name" >/dev/null 2>&1; then
    echo "Instance '$name' not found; nothing to destroy."
    return
  fi
  echo "Destroying $name..."
  "${OS[@]}" server delete --wait "$name"
}

cmd_list() {
  "${OS[@]}" server list --name '^lnq-trainer-' -c Name -c Status -c Networks -c Flavor
}

cmd_shelve_all() {
  local names
  names=$("${OS[@]}" server list --name '^lnq-trainer-' -f value -c Name -c Status \
           | awk '$2=="ACTIVE"{print $1}')
  if [ -z "$names" ]; then
    echo "No active lnq-trainer-* instances."
    return
  fi
  for n in $names; do
    echo "Shelving $n..."
    "${OS[@]}" server shelve "$n"
  done
}

case "${1:-}" in
  create)     shift; cmd_create     "$@" ;;
  status)     shift; cmd_status     "$@" ;;
  ssh)        shift; cmd_ssh        "$@" ;;
  logs)       shift; cmd_logs       "$@" ;;
  shelve)     shift; cmd_shelve     "$@" ;;
  shelve-all) shift; cmd_shelve_all "$@" ;;
  unshelve)   shift; cmd_unshelve   "$@" ;;
  destroy)    shift; cmd_destroy    "$@" ;;
  list)       shift; cmd_list       "$@" ;;
  ""|-h|--help)
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    ;;
  *)
    echo "Unknown command: $1" >&2
    echo "Usage: $0 {create|status|ssh|logs|shelve|unshelve|destroy|list}" >&2
    exit 1
    ;;
esac
