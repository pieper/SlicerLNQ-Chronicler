# Improvements to fold into the next training run

Based on the inguinal-retro1 weekend run, here's what to change before the next big training.

## 1. Local-data copy per trainer (~3× wall-clock speedup)

The 4-fold parallel run hit ~100 s/epoch instead of the solo 35 s/epoch — Manila CephFS contention from 4 GPUs × ~4 dataloader workers each.

**Fix:** copy `nnUNet_preprocessed/Dataset<NNN>_<task>/` to local disk before training. Then export `nnUNet_preprocessed` pointing at the local copy. Each instance has ~100 GB local SSD on g3.xl which fits the preprocessed dataset many times over.

Sketch (cloud-init `runcmd`, just before `systemctl enable lnq-trainer-runner.service`):

```yaml
- |
  mkdir -p /opt/lnq/nnUNet_preprocessed
  # Wait for plan_and_preprocess (run on the first fold) to deposit data.
  # All other folds wait on the file-lock that with_preprocess_lock holds.
  # Then rsync the preprocessed subdir to local SSD.
- |
  # Add an EnvironmentFile override so the runner sees the local path.
  echo "nnUNet_preprocessed=/opt/lnq/nnUNet_preprocessed" >> /etc/lnq/job.conf
```

The runner needs to:
- Detect this override and pass it via the env to nnUNetv2_train + plan_and_preprocess.
- Sync the preprocessed dir locally **after** plan_and_preprocess (which still writes to Manila), **before** nnUNetv2_train.

Keep `nnUNet_results` on Manila so checkpoints + logs remain shared/visible.

Estimated savings: ~50–60 % wall-clock on parallel runs.

## 2. Per-epoch validation predictions for convergence visualization

The user wants to scrub through the segmentation evolution per validation case across epochs (2D + 3D) and compare across folds.

nnU-Net v2's default trainer does NOT save predictions per epoch — it just computes Pseudo dice numerically. To capture the visual sequence, subclass `nnUNetTrainer`:

```python
# bin/nnunet_extras/nnUNetTrainerLNQValidationSeq.py
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
import SimpleITK as sitk, os

class nnUNetTrainerLNQValidationSeq(nnUNetTrainer):
    """Subclass that saves the predicted SEG of each held-out validation case
    every EVERY_N_EPOCHS into validation_raw/<case>_epoch_NNNNN.nrrd. The
    Slicer module reads those as a sequence keyed by epoch so the reviewer
    can scrub convergence visually."""
    EVERY_N_EPOCHS = 10  # keep storage bounded; 100 frames/fold

    def on_validation_epoch_end(self, val_outputs):
        super().on_validation_epoch_end(val_outputs)
        if (self.current_epoch + 1) % self.EVERY_N_EPOCHS:
            return
        out_dir = os.path.join(self.output_folder, "validation_raw")
        os.makedirs(out_dir, exist_ok=True)
        for case_id, pred in zip(self.val_case_ids, val_outputs):
            img = sitk.GetImageFromArray(pred.argmax(0).cpu().numpy())
            # Copy geometry from the source CT.
            ref = sitk.ReadImage(self.raw_paths[case_id])
            img.CopyInformation(ref)
            sitk.WriteImage(img, os.path.join(out_dir,
                f"{case_id}_epoch_{self.current_epoch:05d}.nrrd"),
                useCompression=True)
```

Launch via `nnUNetv2_train ... -tr nnUNetTrainerLNQValidationSeq`. The runner accepts `--trainer-class` already.

In LNQ Studio, add a "Load validation sequence" button on the Selected Job panel that:
- Discovers `validation_raw/<case>_epoch_*.nrrd` for the selected (fold, case).
- Loads each as a SEG node into a Slicer Sequence.
- User scrubs the timeline to see segmentation evolution.

Storage cost: 4 val cases × ~5 MB each × 100 frames × 5 folds = ~10 GB on Manila. Fine.

## 3. Cloud-init template fixes already in trainer.yml.tmpl

- PATH on systemd unit so `nnUNetv2_*` shell-outs resolve.
- `/etc/hosts` override for chronicle internal-IP (Js2 FIP hairpin workaround).
  - Variable interpolation: shell vars use `${VAR}` (envsubst's allowlist leaves them alone), NOT `$${VAR}` which envsubst would keep literal and the shell would mis-read as `$$VAR` (PID).
- `bin/trainer.sh` looks up chronicle's internal IP via `openstack server show lnq-chronicle` and passes as `CHRONICLE_INTERNAL_IP`.

## 4. Trainer-runner improvements

- `plans_subdir()` was wrong; the real nnU-Net v2 results layout is
  `<trainer_class>__<plans_identifier>__<config>`. Fix is in `bin/trainer-runner.py`
  but the BIN_BASE_URL gist needs to be updated before the next launch.
- `nnUNetv2_train` should be invoked with `--c` so restarts after transient
  failures resume from the latest checkpoint instead of failing. Today's
  workaround was a symlink so the heartbeat finds the log; the restart-resume
  story still needs `--c`.
- `parse_training_log` resets metrics on every `Epoch N` line and so wipes
  good values on a mid-epoch parse window. Either don't reset, or only
  overwrite when the new value is non-None. `bin/trainer-metrics-publisher.py`
  already does the right thing; fold it back into trainer-runner.py too.
- The Pseudo dice regex didn't accept `np.float32(...)` syntax; fixed in the
  publisher, port back into trainer-runner.

## 5. Capacity / flavor preflight

Yesterday: fold4 hit g3.xl scheduler scarcity, then g3.large gave us a GPU
whose PCI ID the bundled cuda-drivers 610.43.02 didn't support, then g5.xl
(H100) hung at the sshd-banner stage. End state: 4-fold instead of 5-fold.

Before next launch: smoke a single instance of the target flavor first; if
the runner reaches `Epoch 1` within 15 min, only then schedule the rest.

## 6. Pin BIN_BASE_URL to the chronicler repo

Currently a gist. Move it to a stable raw GitHub URL referencing the
chronicler repo's `main` branch so the cloud-init can re-fetch the runner +
exporter at any time and bug-fixes ship cleanly.
