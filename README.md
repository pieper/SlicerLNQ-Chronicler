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

After the instance is up, push the schemas and validator into CouchDB:

```sh
bin/deploy-design.sh         # creates 'lnq' database + installs _design/lnq
bin/test-design.sh           # optional: smoke-test the validator
```

## Repository layout

```
bin/
  chronicle.sh             # the lifecycle script (~150 lines)
  deploy-design.sh         # push _design/lnq (validators) into CouchDB
  test-design.sh           # smoke-test the validator against a temp database

  # --- ingest: on-disk and public data -> Chronicle documents ---
  import-lnq-retro.py      # the LNQ-retro tree (cases + retro2..retro6 segs)
  import-inguinal.py       # the inguinal cohort
  import-segs.py           # segmentations for an existing cohort
  ingest-idc-cohort.py     # an IDC collection -> Cohort + batch inference
  idc-batch-qc.py          # QC pass over an ingested IDC cohort
  register-equivalent-volumes.py  # link volumes that are the same series
  characterize-nrrd.py     # geometry/intensity summary for a NRRD

  # --- training on Jetstream2 ---
  trainer.sh               # lifecycle for the fold instances
  trainer-runner.py        # runs on the instance; drives nnU-Net, reports status
  export-nnunet-dataset.py # approved Annotations -> nnU-Net raw dataset
  trainer-watcher.py       # waits for convergence, repurposes instances
  trainer-health.py        # health probe across the fold instances
  trainer-observe.py       # read-only view of instance + job state
  trainer-metrics-publisher.py  # training curves -> Chronicle
  lnq-val-snapshotter.py   # periodic validation snapshots
  start-val-snapshotter.sh # attach the snapshotter to a TrainingJob

  # --- publish + consume ---
  publish-model.py         # trained folds -> release assets + registry entry
  run-inference.py         # predict into Chronicle as model Annotations
  run-extra-anatomies.py   # batch predict extra anatomies over a cohort

  # --- ops ---
  jetstream-status.py      # Js2 status.io + allocation check
ops/
  cloud-init/
    chronicle.yml.tmpl     # first-boot setup; envsubst'd by chronicle.sh
    trainer.yml.tmpl       # first-boot setup for a trainer fold instance
schemas/
  *.schema.json            # JSON Schemas for the Chronicle document types
                           # (Cohort, Protocol, Project; more to come).
                           # Authoritative shape for client-side validation.
design/
  validate_doc_update.js   # CouchDB validator that mirrors the schemas at
                           # the write boundary; deployed by deploy-design.sh.
chronicle.conf.example     # copy to chronicle.conf (gitignored) and fill in
trainer.conf.example       # copy to trainer.conf (gitignored) and fill in
docs/
  setup.md                 # one-time prerequisites (FIP, DNS, key, secret)
  training.md              # end-to-end training run
  next-training-improvements.md  # queued changes for the next round
  phase1-postmortem.md     # historical: bugs hit during the original
                           # Terraform-based attempt; lessons that motivated
                           # the simplification you're looking at
```

The previous Terraform/CI-based architecture lives on the [`terraform-experiment`](https://github.com/pieper/SlicerLNQ-Chronicler/tree/terraform-experiment) branch in case it's useful later.

## Architectural context

Design intent for the broader SlicerLNQ project — Chronicle data model, document schemas, agent pattern, phase plan — lives in the sibling repo: [SlicerLNQ/docs/architecture.md](https://github.com/pieper/SlicerLNQ/blob/main/docs/architecture.md). This repo is purely the operational layer.
