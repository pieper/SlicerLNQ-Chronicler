# SlicerLNQ-Chronicler

Operational repository for **SlicerLNQ-Chronicle**, the CouchDB + dicomweb-server backend for the [SlicerLNQ](https://github.com/pieper/SlicerLNQ) project, hosted on [Jetstream2](https://jetstream-cloud.org/).

This repo defines the full lifecycle of the Chronicle instance: Terraform/OpenTofu for the VMs and storage, cloud-init for first-boot provisioning, Docker Compose for the running services, and GitHub Actions workflows that drive lifecycle and routine maintenance.

Architectural context lives in [SlicerLNQ/docs/architecture.md](https://github.com/pieper/SlicerLNQ/blob/main/docs/architecture.md). This README covers the operating side only.

## Topology

```
Internet → doorman (m3.tiny, public FIP, Caddy + TLS)
           └→ core (m3.medium, internal-only, CouchDB + dicomweb-server)
```

The doorman is the only public entry point. The core is reachable only over the Js2 tenant network. Activity-driven shelve/unshelve is added in a later commit.

## First-time setup

Before the first `apply`, see [docs/first-time-setup.md](docs/first-time-setup.md) for the one-shot tasks: creating the Swift container that holds Terraform state, confirming GitHub Actions secrets are set, and verifying DNS.

## Running

Lifecycle is driven by GitHub Actions workflows under `.github/workflows/`:

- `plan.yml` — runs on PRs touching `ops/terraform/`. Reports the plan as a comment.
- `apply.yml` — manual `workflow_dispatch`. Brings up or updates the topology.
- `destroy.yml` — manual `workflow_dispatch` with explicit confirmation. Tears everything down except the floating IP, the Swift state container, and the DNS record (those are intentionally outside Terraform's blast radius).

You should not need to run `tofu` locally for routine operations. Local runs are useful for development of the Terraform itself.

## Repository layout

```
ops/
  terraform/        # OpenTofu sources for Js2 resources
  cloud-init/       # First-boot provisioning templates for both instances
                    # (services run as native systemd units; no Docker)
  runbooks/         # Operator scripts (backup, restore, health) — added later
.github/workflows/  # GitHub Actions for lifecycle and maintenance
docs/               # First-time setup, runbook references
```

## Status

Phase 1 — initial stand-up. Watchman / auto-shelve, OIDC, runbooks, and the LNQ schema/views land in subsequent commits per the architecture doc's phase plan.
