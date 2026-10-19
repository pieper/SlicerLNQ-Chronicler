# CouchDB design documents

Server-side enforcement of the Chronicle document discipline. The validator runs inside CouchDB on every write; rejected writes return HTTP 403 with the validator's message.

## Files

- [`validate_doc_update.js`](validate_doc_update.js) — the validator function as readable JavaScript. Hand-rolled to mirror the rules in [`../schemas/`](../schemas/). Edit this when the schemas change.
- The `_design/lnq` document itself is *generated* by [`../bin/deploy-design.sh`](../bin/deploy-design.sh) at deploy time — it inlines `validate_doc_update.js` as a string under `validate_doc_update`. There's no committed JSON for the design doc; the JS file is the source of truth.

## What the validator enforces

1. **Type-prefix discipline.** `Cohort` → `cohort:…`, `Protocol` → `protocol:…`, `Project` → `project:…`. Mismatch → 403.
2. **Immutability.** Once a typed doc exists, all subsequent `PUT`s against the same `_id` are rejected (whether or not `_rev` matches). `_deleted: true` is also rejected. New versions go in *new documents* with `predecessor` pointing at the old `_id`.
3. **Required common fields:** `type`, `name`, `created_at`, `created_by`, `version`, `predecessor`. `version: 1` requires `predecessor: null`; `version > 1` requires `predecessor` matching the type prefix.
4. **Anti-forgery.** `created_by` must equal `userCtx.name` for non-admins.
5. **Per-type structural rules** that mirror the JSON Schemas: required arrays non-empty, enum values, project membership includes at least one admin, etc.

## What it does NOT enforce

- **Cross-document referential integrity.** A Project's `cohort_id` is required to match the `cohort:` prefix, but the validator does not look up whether that Cohort exists — `validate_doc_update` runs against the writing database only and cannot do queries. Clients (Slicer extension) and agents are responsible for not writing dangling references; the changes feed lets us detect them post-hoc if we care.
- **Full JSON Schema conformance.** The schemas allow more nuance than the validator checks (e.g., `format: date-time` on `created_at`). The validator covers the rules that matter for write-time invariants; the schemas cover everything for client-side validation.

## Deploying

```sh
bin/deploy-design.sh           # deploys to the default 'lnq' database
bin/deploy-design.sh somedb    # deploys to a specific database
```

Reads `chronicle.conf` for `DOMAIN_NAME` and `COUCHDB_ADMIN_PASSWORD`. Creates the database if it doesn't exist. Idempotent — re-running deploys the current `validate_doc_update.js`.

## Smoke testing

```sh
bin/test-design.sh
```

Creates a temporary database, deploys the design doc, exercises a handful of accept/reject cases, and tears down. Should print `OK` and exit 0 if everything passes.
