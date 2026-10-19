#!/usr/bin/env bash
# SlicerLNQ-Chronicler — smoke-test the validator against a live CouchDB.
# Creates a temp database, deploys _design/lnq into it, runs a sequence of
# accept/reject cases, and tears the database down. Prints OK on success.
#
# Reads ./chronicle.conf for DOMAIN_NAME and COUCHDB_ADMIN_PASSWORD.

set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG_FILE="${CHRONICLE_CONF:-./chronicle.conf}"
[ -f "$CONFIG_FILE" ] || { echo "Missing $CONFIG_FILE" >&2; exit 1; }
# shellcheck disable=SC1090
source "$CONFIG_FILE"

: "${DOMAIN_NAME:?}" "${COUCHDB_ADMIN_PASSWORD:?}"
command -v jq >/dev/null || { echo "jq is required" >&2; exit 1; }

BASE="https://admin:${COUCHDB_ADMIN_PASSWORD}@${DOMAIN_NAME}"
DB="lnq-validator-test-$$"

cleanup() {
  curl -fsS -o /dev/null -X DELETE "$BASE/$DB" 2>/dev/null || true
}
trap cleanup EXIT

echo "Creating temp database '$DB'..."
curl -fsS -X PUT "$BASE/$DB" >/dev/null

echo "Deploying _design/lnq..."
bin/deploy-design.sh "$DB" >/dev/null

# Helpers ----------------------------------------------------------------
fails=0
ok()    { printf '  \xE2\x9C\x93 %s\n' "$1"; }
fail()  { printf '  \xE2\x9C\x97 %s\n' "$1"; fails=$((fails+1)); }

# put DOC_JSON DOCID — returns the HTTP status code (no -f: we want the code on 4xx/5xx).
put_status() {
  local doc="$1" id="$2"
  curl -s -o /dev/null -w '%{http_code}' \
    -X PUT "$BASE/$DB/$id" \
    -H 'Content-Type: application/json' \
    --data-binary "$doc"
}

expect_accept() {
  local doc="$1" id="$2" desc="$3"
  local s; s=$(put_status "$doc" "$id")
  if [ "$s" = "201" ]; then ok "accept: $desc"
  else fail "accept: $desc — expected 201, got $s"
  fi
}

expect_reject() {
  local doc="$1" id="$2" desc="$3"
  local s; s=$(put_status "$doc" "$id")
  if [ "$s" = "403" ]; then ok "reject: $desc"
  else fail "reject: $desc — expected 403, got $s"
  fi
}

now() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }

# Test cases -------------------------------------------------------------
echo "Running cases..."

# Valid Cohort v1
COHORT1_ID="cohort:11111111-1111-4111-a111-111111111111"
expect_accept "$(jq -n --arg id "$COHORT1_ID" --arg now "$(now)" '{
  _id: $id, type: "Cohort", name: "Test cohort",
  created_at: $now, created_by: "admin", version: 1, predecessor: null,
  sources: [{kind: "idc-query", value: "collection_id = \"nsclc_radiogenomics\""}]
}')" "$COHORT1_ID" "Cohort v1 (valid)"

# _id prefix mismatch
expect_reject "$(jq -n --arg now "$(now)" '{
  _id: "wrong:prefix", type: "Cohort", name: "X",
  created_at: $now, created_by: "admin", version: 1, predecessor: null,
  sources: [{kind: "idc-query", value: "x"}]
}')" "wrong:prefix" "Cohort with non-cohort _id"

# Missing required field (sources)
expect_reject "$(jq -n --arg now "$(now)" '{
  _id: "cohort:22222222-2222-4222-a222-222222222222", type: "Cohort", name: "X",
  created_at: $now, created_by: "admin", version: 1, predecessor: null
}')" "cohort:22222222-2222-4222-a222-222222222222" "Cohort missing sources"

# v1 with non-null predecessor
expect_reject "$(jq -n --arg now "$(now)" '{
  _id: "cohort:33333333-3333-4333-a333-333333333333", type: "Cohort", name: "X",
  created_at: $now, created_by: "admin", version: 1, predecessor: "cohort:zzz",
  sources: [{kind: "idc-query", value: "x"}]
}')" "cohort:33333333-3333-4333-a333-333333333333" "Cohort v1 with predecessor"

# Mutation attempt: PUT to existing _id (no _rev → 409 conflict, with _rev → 403 immutable)
existing_rev=$(curl -fsS "$BASE/$DB/$COHORT1_ID" | jq -r '._rev')
mutate_doc=$(jq -n --arg id "$COHORT1_ID" --arg rev "$existing_rev" --arg now "$(now)" '{
  _id: $id, _rev: $rev, type: "Cohort", name: "MUTATED",
  created_at: $now, created_by: "admin", version: 1, predecessor: null,
  sources: [{kind: "idc-query", value: "x"}]
}')
expect_reject "$mutate_doc" "$COHORT1_ID" "Cohort mutation (PUT with _rev)"

# Delete attempt
delete_doc=$(jq -n --arg id "$COHORT1_ID" --arg rev "$existing_rev" '{
  _id: $id, _rev: $rev, _deleted: true, type: "Cohort"
}')
expect_reject "$delete_doc" "$COHORT1_ID" "Cohort delete via _deleted"

# v2 with valid predecessor
COHORT2_ID="cohort:44444444-4444-4444-a444-444444444444"
expect_accept "$(jq -n --arg id "$COHORT2_ID" --arg pred "$COHORT1_ID" --arg now "$(now)" '{
  _id: $id, type: "Cohort", name: "Test cohort v2",
  created_at: $now, created_by: "admin", version: 2, predecessor: $pred,
  sources: [{kind: "idc-query", value: "x"}]
}')" "$COHORT2_ID" "Cohort v2 with predecessor"

# Valid Protocol
PROTO_ID="protocol:55555555-5555-4555-a555-555555555555"
expect_accept "$(jq -n --arg id "$PROTO_ID" --arg now "$(now)" '{
  _id: $id, type: "Protocol", name: "LN seg v1",
  created_at: $now, created_by: "admin", version: 1, predecessor: null,
  color_table: [
    {label: 1, name: "Right paratracheal", color: [255, 100, 100],
     code: {scheme: "SCT", value: "59441001", meaning: "Lymph node"}}
  ]
}')" "$PROTO_ID" "Protocol v1 (valid)"

# Protocol with duplicate label
expect_reject "$(jq -n --arg now "$(now)" '{
  _id: "protocol:66666666-6666-4666-a666-666666666666", type: "Protocol", name: "X",
  created_at: $now, created_by: "admin", version: 1, predecessor: null,
  color_table: [
    {label: 1, name: "A", color: [255,0,0], code: {scheme:"SCT",value:"1",meaning:"a"}},
    {label: 1, name: "B", color: [0,255,0], code: {scheme:"SCT",value:"2",meaning:"b"}}
  ]
}')" "protocol:66666666-6666-4666-a666-666666666666" "Protocol with duplicate label"

# Valid Project referencing the Cohort + Protocol
PROJ_ID="project:77777777-7777-4777-a777-777777777777"
expect_accept "$(jq -n --arg id "$PROJ_ID" --arg cid "$COHORT1_ID" --arg pid "$PROTO_ID" --arg now "$(now)" '{
  _id: $id, type: "Project", name: "Test project",
  created_at: $now, created_by: "admin", version: 1, predecessor: null,
  cohort_id: $cid, protocol_id: $pid,
  members: [{user: "admin", role: "admin"}]
}')" "$PROJ_ID" "Project v1 (valid)"

# Project with no admin member
expect_reject "$(jq -n --arg cid "$COHORT1_ID" --arg pid "$PROTO_ID" --arg now "$(now)" '{
  _id: "project:88888888-8888-4888-a888-888888888888", type: "Project", name: "X",
  created_at: $now, created_by: "admin", version: 1, predecessor: null,
  cohort_id: $cid, protocol_id: $pid,
  members: [{user: "bob", role: "annotator"}]
}')" "project:88888888-8888-4888-a888-888888888888" "Project without admin"

# Project with bad role
expect_reject "$(jq -n --arg cid "$COHORT1_ID" --arg pid "$PROTO_ID" --arg now "$(now)" '{
  _id: "project:99999999-9999-4999-a999-999999999999", type: "Project", name: "X",
  created_at: $now, created_by: "admin", version: 1, predecessor: null,
  cohort_id: $cid, protocol_id: $pid,
  members: [{user: "bob", role: "bystander"}]
}')" "project:99999999-9999-4999-a999-999999999999" "Project with bad role"

# Unknown type
expect_reject "$(jq -n --arg now "$(now)" '{
  _id: "weird:1", type: "Weird", name: "X",
  created_at: $now, created_by: "admin", version: 1, predecessor: null
}')" "weird:1" "Unknown type"

# Result -----------------------------------------------------------------
if [ "$fails" -eq 0 ]; then
  echo "OK ($((0)) failures)"
  exit 0
else
  echo "FAILED: $fails case(s)"
  exit 1
fi
