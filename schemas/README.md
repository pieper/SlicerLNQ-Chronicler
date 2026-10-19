# Schemas

JSON Schema (draft 2020-12) definitions for the Chronicle document types stored in CouchDB. These are the **authoritative shape** of a document; clients (the Slicer extension, agents, scripts) validate against them locally before writing.

The CouchDB-side enforcement lives in [`design/validate_doc_update.js`](../design/validate_doc_update.js), which is hand-rolled to mirror the rules below. When a schema changes, the validator changes too.

## Common discipline (Chronicle pamphlet §8)

Every typed document is **immutable** once written. Updates are *new documents* that reference the predecessor; corrections are also new documents (as a `Correction` type, when introduced). The validator rejects:

- Any `PUT` against an existing typed `_id` (whether or not `_rev` matches).
- Any `_deleted: true` against a typed document.
- A `_id` whose prefix doesn't match its `type` field.
- A document whose `created_by` doesn't equal the writer's user (admins exempt).

## Common fields (every typed document)

| Field          | Type                  | Notes                                                              |
|----------------|-----------------------|--------------------------------------------------------------------|
| `_id`          | `<type-prefix>:<uuid>`| e.g. `cohort:550e8400-e29b-41d4-a716-446655440000`                 |
| `type`         | string                | one of `Cohort`, `Protocol`, `Project`, … (per-type schemas extend)|
| `name`         | string                | human-readable label                                               |
| `created_at`   | string (ISO 8601)     | UTC, with timezone, e.g. `2026-05-06T15:23:11Z`                    |
| `created_by`   | string                | writer's CouchDB username (admins may override)                    |
| `version`      | integer ≥ 1           | starts at 1; new versions of a logical entity bump this            |
| `predecessor`  | `_id` or `null`       | `null` when `version == 1`; `_id` of previous version otherwise    |

Versioning convention: a Cohort that needs to be updated produces a new doc with `version: 2`, `predecessor: cohort:<old-id>`. The old doc is never touched. "The current version of cohort X" is found by walking forward from `predecessor`-pointers until you hit a doc with no successor (or by Mango query: `predecessor == cohort:<id>`).

## First batch (phase 2)

- [`cohort.schema.json`](cohort.schema.json) — cohort definition: selection criteria + sources.
- [`protocol.schema.json`](protocol.schema.json) — annotation SOP, color table with controlled terminology.
- [`project.schema.json`](project.schema.json) — binds a Cohort to a Protocol with member roles.
- [`common.schema.json`](common.schema.json) — `$defs` reused by the above (terminology codes, the base-document fields, etc.).

Later batches: `Annotation`, `CohortResolution`, `ModelGeneration`, `InferenceRun`, `ComparisonReport`, `Comment`, `Correction`.

## ID prefixes

| `type`     | prefix       |
|------------|--------------|
| `Cohort`   | `cohort:`    |
| `Protocol` | `protocol:`  |
| `Project`  | `project:`   |
