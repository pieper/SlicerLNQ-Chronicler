#!/usr/bin/env bash
# SlicerLNQ-Chronicler — single-instance lifecycle for the Chronicle backend on Js2.
# Reads ./chronicle.conf, talks to OpenStack via the openstack CLI.
#
# Usage:
#   bin/chronicle.sh create       # create the instance + attach FIP
#   bin/chronicle.sh destroy      # delete the instance (FIP stays allocated)
#   bin/chronicle.sh ssh [args]   # ssh in (forwards extra args to ssh)
#   bin/chronicle.sh status       # show instance state
#   bin/chronicle.sh logs         # tail cloud-init-output.log on the instance

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_FILE="${CHRONICLE_CONF:-./chronicle.conf}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Missing $CONFIG_FILE" >&2
  echo "Copy chronicle.conf.example to chronicle.conf and fill in your values." >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

# Required vars; trip early if any are missing.
: "${OS_CLOUD:?}" "${INSTANCE_NAME:?}" "${IMAGE:?}" "${FLAVOR:?}" "${NETWORK:?}"
: "${KEY_NAME:?}" "${SSH_KEY:?}" "${SSH_USER:?}"
: "${DOMAIN_NAME:?}" "${FLOATING_IP:?}" "${LETSENCRYPT_EMAIL:?}" "${COUCHDB_ADMIN_PASSWORD:?}"
: "${DICOMWEB_REF:=master}"

OS=(openstack --os-cloud "$OS_CLOUD")

# --- helpers ---------------------------------------------------------
get_ip() {
  "${OS[@]}" server show "$INSTANCE_NAME" -f value -c addresses 2>/dev/null \
    | grep -oE '[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' | tail -1
}

# --- commands --------------------------------------------------------
cmd_create() {
  if "${OS[@]}" server show "$INSTANCE_NAME" >/dev/null 2>&1; then
    echo "Instance '$INSTANCE_NAME' already exists. Run '$0 destroy' first, or '$0 ssh'." >&2
    exit 1
  fi

  local userdata
  userdata=$(mktemp)
  trap 'rm -f "$userdata"' EXIT

  # envsubst with a whitelist — only these names are substituted; other
  # $VAR uses inside the cloud-init heredocs are left alone.
  DOMAIN_NAME="$DOMAIN_NAME" \
  LETSENCRYPT_EMAIL="$LETSENCRYPT_EMAIL" \
  COUCHDB_ADMIN_PASSWORD="$COUCHDB_ADMIN_PASSWORD" \
  DICOMWEB_REF="$DICOMWEB_REF" \
    envsubst '${DOMAIN_NAME} ${LETSENCRYPT_EMAIL} ${COUCHDB_ADMIN_PASSWORD} ${DICOMWEB_REF}' \
    < ops/cloud-init/chronicle.yml.tmpl \
    > "$userdata"

  echo "Creating $INSTANCE_NAME ($FLAVOR / $IMAGE)..."
  "${OS[@]}" server create \
    --image "$IMAGE" \
    --flavor "$FLAVOR" \
    --network "$NETWORK" \
    --key-name "$KEY_NAME" \
    --user-data "$userdata" \
    --wait \
    "$INSTANCE_NAME" >/dev/null

  echo "Attaching $FLOATING_IP..."
  "${OS[@]}" server add floating ip "$INSTANCE_NAME" "$FLOATING_IP"

  echo
  echo "Created. Cloud-init still running on the instance (~5-10 min)."
  echo "  Watch:    $0 logs"
  echo "  Probe:    curl -i https://$DOMAIN_NAME/"
  echo "  Public:   https://$DOMAIN_NAME/"
}

cmd_destroy() {
  if ! "${OS[@]}" server show "$INSTANCE_NAME" >/dev/null 2>&1; then
    echo "Instance '$INSTANCE_NAME' not found; nothing to destroy."
    return
  fi
  echo "Destroying $INSTANCE_NAME..."
  "${OS[@]}" server delete --wait "$INSTANCE_NAME"
  echo "Done. FIP $FLOATING_IP remains allocated."
}

cmd_ssh() {
  local ip
  ip=$(get_ip) || true
  if [ -z "${ip:-}" ]; then
    echo "Could not find IP for $INSTANCE_NAME. Is it running?" >&2
    exit 1
  fi
  exec ssh -i "$SSH_KEY" "$SSH_USER@$ip" "$@"
}

cmd_status() {
  if ! "${OS[@]}" server show "$INSTANCE_NAME" \
      -c name -c status -c addresses -c image -c flavor 2>/dev/null; then
    echo "Instance '$INSTANCE_NAME' not found."
    exit 1
  fi
}

cmd_logs() {
  cmd_ssh sudo tail -f /var/log/cloud-init-output.log
}

case "${1:-}" in
  create)  shift; cmd_create  "$@" ;;
  destroy) shift; cmd_destroy "$@" ;;
  ssh)     shift; cmd_ssh     "$@" ;;
  status)  shift; cmd_status  "$@" ;;
  logs)    shift; cmd_logs    "$@" ;;
  ""|-h|--help)
    sed -n '2,/^$/p' "$0" | sed 's/^# \?//'
    ;;
  *)
    echo "Unknown command: $1" >&2
    echo "Usage: $0 {create|destroy|ssh|status|logs}" >&2
    exit 1
    ;;
esac
