#!/usr/bin/env bash
# SlicerLNQ-Chronicler — push the _design/lnq document to CouchDB.
# Reads ./chronicle.conf for DOMAIN_NAME and COUCHDB_ADMIN_PASSWORD.
# Inlines design/validate_doc_update.js as the validate_doc_update field.
#
# Usage:
#   bin/deploy-design.sh             # deploy to the default 'lnq' database
#   bin/deploy-design.sh somedb      # deploy to 'somedb' (also creates it if missing)

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_FILE="${CHRONICLE_CONF:-./chronicle.conf}"
if [ ! -f "$CONFIG_FILE" ]; then
  echo "Missing $CONFIG_FILE" >&2
  exit 1
fi
# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${DOMAIN_NAME:?}" "${COUCHDB_ADMIN_PASSWORD:?}"

command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }

DB_NAME="${1:-lnq}"
VALIDATOR_SRC="design/validate_doc_update.js"
[ -f "$VALIDATOR_SRC" ] || { echo "Missing $VALIDATOR_SRC" >&2; exit 1; }

BASE="https://admin:${COUCHDB_ADMIN_PASSWORD}@${DOMAIN_NAME}"

# Create the database if it doesn't already exist. 412 means "already there"; that's fine.
# No -f: we want to inspect the status code rather than have curl bail on 4xx.
echo "Ensuring database '$DB_NAME' exists..."
status=$(curl -sS -o /dev/null -w '%{http_code}' -X PUT "$BASE/$DB_NAME")
case "$status" in
  201|202) echo "  created" ;;
  412)     echo "  already exists" ;;
  *)       echo "  unexpected HTTP $status while creating '$DB_NAME'" >&2; exit 1 ;;
esac

# Build the design doc with the validator function inlined as a string.
# jq's --rawfile handles JSON-string escaping correctly.
echo "Building _design/lnq..."
DESIGN_DOC=$(jq -n \
  --rawfile validator "$VALIDATOR_SRC" \
  '{
    _id: "_design/lnq",
    language: "javascript",
    validate_doc_update: $validator
  }')

# If a previous _design/lnq exists, fetch its _rev so we can replace it.
REV=$(curl -fsS "$BASE/$DB_NAME/_design/lnq" 2>/dev/null | jq -r '._rev // empty' || true)
if [ -n "$REV" ]; then
  echo "  replacing existing _design/lnq (rev $REV)"
  DESIGN_DOC=$(echo "$DESIGN_DOC" | jq --arg rev "$REV" '. + {_rev: $rev}')
else
  echo "  installing new _design/lnq"
fi

# Push it.
response=$(curl -fsS -X PUT "$BASE/$DB_NAME/_design/lnq" \
  -H 'Content-Type: application/json' \
  --data-binary "$DESIGN_DOC")
echo "Response: $response"
echo "Done."
