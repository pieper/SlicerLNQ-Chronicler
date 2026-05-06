# SlicerLNQ-Chronicler

Lifecycle scripts for **SlicerLNQ-Chronicle**, the CouchDB + dicomweb-server backend for the [SlicerLNQ](https://github.com/pieper/SlicerLNQ) project on [Jetstream2](https://jetstream-cloud.org/).

One Js2 instance, one shell script, no automation surface to fight. Public TLS via Caddy + Let's Encrypt. Designed for a single operator running a public-data research project — generous on simplicity, sparing on abstractions.

## Topology

```
Internet → lnq-chronicle.isomics.dev (FIP)
            │
            ▼
        ┌─────────────────────────────┐
        │   m3.medium (Featured-      │
        │   Ubuntu24)                 │
        │                             │
        │   Caddy (80/443)            │
        │     ↓                       │
        │     ├─ /          → CouchDB         (127.0.0.1:5984)
        │     ├─ /dicomweb/ → dicomweb-server (127.0.0.1:5985)
        │     └─ /health    → 200 ok
        └─────────────────────────────┘
```

All services run as native systemd units. Caddy fronts both APIs over TLS. CouchDB and dicomweb-server bind to localhost only — Caddy is the only public surface.

## Usage

```sh
# Once: copy and fill in your config
cp chronicle.conf.example chronicle.conf
$EDITOR chronicle.conf

# Create the instance (server create + cloud-init + FIP attach)
bin/chronicle.sh create

# Watch first-boot install (~5–10 min)
bin/chronicle.sh logs

# When ready:
curl -i https://lnq-chronicle.isomics.dev/
curl -i https://lnq-chronicle.isomics.dev/dicomweb/studies

# Other commands
bin/chronicle.sh status
bin/chronicle.sh ssh
bin/chronicle.sh destroy
```

That's the whole interface.

## Repository layout

```
bin/
  chronicle.sh             # the lifecycle script (~150 lines)
ops/
  cloud-init/
    chronicle.yml.tmpl     # first-boot setup; envsubst'd by chronicle.sh
chronicle.conf.example     # copy to chronicle.conf (gitignored) and fill in
docs/
  setup.md                 # one-time prerequisites (FIP, DNS, key, secret)
  phase1-postmortem.md     # historical: bugs hit during the original
                           # Terraform-based attempt; lessons that motivated
                           # the simplification you're looking at
```

The previous Terraform/CI-based architecture lives on the [`terraform-experiment`](https://github.com/pieper/SlicerLNQ-Chronicler/tree/terraform-experiment) branch in case it's useful later.

## Architectural context

Design intent for the broader SlicerLNQ project — Chronicle data model, document schemas, agent pattern, phase plan — lives in the sibling repo: [SlicerLNQ/docs/architecture.md](https://github.com/pieper/SlicerLNQ/blob/main/docs/architecture.md). This repo is purely the operational layer.
