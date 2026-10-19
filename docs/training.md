# Training a model

End-to-end flow: dashboard → 5 TrainingJob docs → 5 Js2 A100 instances → 5 ModelGeneration docs landing in the Chronicle as folds converge.

## Prereqs (one-time)

1. `openstack` CLI configured with the LNQ cloud (`clouds.yaml` for `MED250016_IU`).
2. `lnq-chronicler-deploy` SSH keypair registered in the project (`openstack --os-cloud MED250016_IU keypair list`).
3. A floating IP reserved for the SSH-accessible fold (`openstack ... floating ip create public`).
4. `trainer.conf` filled in from `trainer.conf.example`. The Manila access key and `COUCHDB_ADMIN_PASSWORD` come from `chronicle.conf`; the FIP from step 3.

## How the trainer instance gets the runner + exporter scripts

`ops/cloud-init/trainer.yml.tmpl` fetches `bin/trainer-runner.py` and `bin/export-nnunet-dataset.py` from `$BIN_BASE_URL` (set in `trainer.conf`) at first boot. Today this is a public gist; when `SlicerLNQ-Chronicler` is published the URL switches to `raw.githubusercontent.com/pieper/SlicerLNQ-Chronicler/<tag>/bin` and nothing else changes.

**Update the gist after local edits:**

```sh
# from the SlicerLNQ-Chronicler repo root:
gh gist edit ea740fee37abde0f4aa2612338b410a7 bin/trainer-runner.py
gh gist edit ea740fee37abde0f4aa2612338b410a7 bin/export-nnunet-dataset.py
```

The cloud-init pulls `${BIN_BASE_URL}/trainer-runner.py` (no version pin) on first boot, so a fresh `bin/trainer.sh create` always gets the latest. Existing instances already have the scripts on disk; SSH in and refetch if you're patching mid-run:

```sh
bin/trainer.sh ssh lnq-trainer-fold0
# on the box:
sudo curl -fsSL https://gist.githubusercontent.com/pieper/ea740fee37abde0f4aa2612338b410a7/raw/trainer-runner.py -o /opt/lnq/trainer-runner.py
sudo systemctl restart lnq-trainer-runner.service
```

## Workflow

1. **Approve the training set in LNQ Studio.** The dashboard's *Ready to train* row shows the project's approved annotation count.
2. **Click *Plan training*.** This writes five TrainingJob docs (folds 0–4) in `status=pending`, all sharing the same frozen `training_annotation_ids` list. They appear in the *Training jobs* table immediately via the `_changes` feed.
3. **Note each job's `_id`** — copy from the dashboard's drill-down or by listing in the Chronicle.
4. **Launch one instance per fold:**

   ```sh
   cd SlicerLNQ-Chronicler

   # Fold 0 gets the FIP (only one is needed; the rest are reached via the chronicle box as jump host).
   bin/trainer.sh create --training-job-id trainingjob:<id-for-fold-0> --name lnq-trainer-fold0
   bin/trainer.sh create --training-job-id trainingjob:<id-for-fold-1> --name lnq-trainer-fold1 --no-fip
   bin/trainer.sh create --training-job-id trainingjob:<id-for-fold-2> --name lnq-trainer-fold2 --no-fip
   bin/trainer.sh create --training-job-id trainingjob:<id-for-fold-3> --name lnq-trainer-fold3 --no-fip
   bin/trainer.sh create --training-job-id trainingjob:<id-for-fold-4> --name lnq-trainer-fold4 --no-fip
   ```

5. **Watch the dashboard.** Each fold transitions `pending → running → converged` (or `failed`), with epoch + mean Dice updating roughly every 30 s while training is live.

6. **Shelve / destroy when done.** After all folds converge:

   ```sh
   bin/trainer.sh shelve-all
   # or, per-instance:
   for i in 0 1 2 3 4; do bin/trainer.sh shelve lnq-trainer-fold$i; done
   ```

   `shelve` stops the SU clock without destroying the disk. `destroy` is permanent.

## Smoke-test mode (recommended for first run)

Before burning hours on a real 5-fold training, validate the orchestration end-to-end with `--smoke-test`:

```sh
bin/trainer.sh create --training-job-id trainingjob:<id-for-fold-0> --name lnq-trainer-fold0 --smoke-test
```

What the smoke test does on the instance, after cloud-init + reboot:

1. Verifies `nvidia-smi` returns a GPU.
2. Verifies `torch.cuda.is_available()` and that an allocation + add on the device gives the expected result.
3. Verifies `nnunetv2` imports.
4. Writes a sentinel under `${NNUNET_BASE}/nnUNet_results/<Dataset>/<plans>/fold_N/_smoke_marker` to confirm the Manila mount is writeable.
5. Writes a dummy 4 KB checkpoint to `/tmp` (NOT to the real results dir), registers it as a Blob, writes a `ModelGeneration` doc whose `notes` field explicitly flags it as a smoke artifact.
6. Flips the TrainingJob to `status=converged` with `latest_metrics.smoke_test=true` and a notes blob containing `nvidia-smi` header + version pins.

Total runner-side time: ~30 s once boot completes. End-to-end (boot + cloud-init CUDA install + reboot + runner): ~15 min.

A passed smoke test means: instance launches, Manila mounts, CUDA + driver land, PyTorch + nnunetv2 are healthy on the GPU, the runner can read job config + talk to Chronicle + write Blob/ModelGeneration docs. Everything except the actual training compute is exercised.

The dashboard shows the smoke run as `converged` with `gpu_name`, `gpu_memory_gb`, and `nnunetv2_version` in the metrics panel. The `ModelGeneration.notes` text makes the smoke artifact unmistakable so it doesn't get picked up by a future inference run by accident.

After a smoke pass, **shelve everything** before launching the real run:

```sh
bin/trainer.sh shelve-all
```

Then create fresh instances without `--smoke-test` for the real training.

## What's running where

| Component | Lives on | Why |
|---|---|---|
| `trainer.conf` | your laptop | secrets + topology |
| `trainer.sh` | your laptop | openstack CLI wrapper |
| Cloud-init template | rendered locally, sent as user-data | first-boot setup |
| `trainer-runner.py` | each instance, fetched from GitHub at first boot | per-fold driver |
| `nnUNet_raw / preprocessed / results` | `/media/share/LNQ-data/Inguinal_Syed/` (Manila) | shared across folds; survives instance teardown |
| `TrainingJob` doc | Chronicle | dashboard's source of truth |
| `ModelGeneration` doc + weights blob | Chronicle (+ Manila for the checkpoint file) | post-convergence record |

Two folds racing to run `nnUNetv2_plan_and_preprocess` on the same dataset are serialized via `fcntl.flock` on `${NNUNET_BASE}/nnUNet_preprocessed/.preprocess.lock` — only the first to win the lock actually runs preprocessing; the others see the `dataset_fingerprint.json` marker and skip.

## SSH access to non-FIP instances

```sh
# Direct (fold 0 only):
bin/trainer.sh ssh lnq-trainer-fold0

# Via jump host (folds 1-4): the script auto-detects no FIP and uses
# `ssh -J $JUMP_USER@$JUMP_HOST` to reach the internal IP.
bin/trainer.sh ssh lnq-trainer-fold3
```

The Manila share is also reachable from the chronicle box (it has the same project access), so progress.png + training_log files for any fold can be `scp`'d from `<chronicle-fip>:/media/share/LNQ-data/Inguinal_Syed/nnUNet_results/...` without SSH'ing the trainer at all.

## What goes wrong, and how to triage

| Symptom | Where to look |
|---|---|
| Instance created but `pending` never moves to `running` | `bin/trainer.sh logs lnq-trainer-fold0` — cloud-init is still installing CUDA. First boot takes ~15 min before the reboot. |
| Status flips to `failed` | dashboard → Selected job → notes field has the last 50 lines of the training log. SSH in for the full log under `/media/share/LNQ-data/Inguinal_Syed/nnUNet_results/.../fold_N/`. |
| Training underway but no metric updates | heartbeat thread fell over. SSH in and check `journalctl -u lnq-trainer-runner.service`. |
| `nnUNetv2_plan_and_preprocess` failing | a single fold ran it under the lock; check that fold's journal. |
| Weights not registered as Blob | check the `ModelGeneration` step in the journal; the runner wrote `model_generation_id` only after a successful Blob upload. |

## When training is done

Each fold's `ModelGeneration` doc carries `weights_ref` (blob) + `training_log_ref` (path on Manila). For inference, pick the fold(s) you want to ensemble; `bin/run-inference.py` (next iteration) will take a `--model-generation-id` and a project, predict, and write back `Annotation` docs with `producer.kind=model`.
