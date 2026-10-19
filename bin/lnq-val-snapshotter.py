#!/usr/bin/env python3
"""lnq-val-snapshotter.py — save per-epoch validation predictions for scrubbing.

Sidecar to nnUNetv2_train: polls the fold's training_log_*.txt for
"Epoch N" lines, and every --every-n-epochs runs nnUNetv2_predict on the
fold's validation cases using `checkpoint_latest.pth`. Outputs land at
<fold_dir>/validation_raw/<case_id>_epoch_NNNNN.nrrd so the LNQ Studio
review module can load them as a Slicer Sequence — scrubbing the timeline
shows the segmentation evolving epoch by epoch, both in slice views and
3D rendering.

For each new snapshot the snapshotter ALSO computes:
  - Dice against the canonical labelsTr/<case>.nrrd (the same GT nnU-Net
    used for its internal Pseudo dice)
  - predicted lymph-node volume (mL) + GT volume (mL)
  - contextual metrics from the training log at that epoch: learning rate,
    train_loss, val_loss, Pseudo dice, EMA pseudo Dice, epoch time
and appends one JSONL record per (epoch, case) to validation_raw/metrics.jsonl
so the LNQ Studio dashboard can render Dice/volume/LR curves without having
to do its own arithmetic. A retroactive pass at startup fills in metrics
for any snapshots that landed before this version of the snapshotter ran.

Designed to run on the same instance as the trainer, alongside the
lnq-trainer-runner.service. Needs SimpleITK + numpy (in /opt/lnq/nnenv).

Usage:
  lnq-val-snapshotter.py \\
    --nnunet-base /media/share/LNQ-data/Inguinal_Syed \\
    --dataset-id 1 --task LNQinguinal \\
    --plans-identifier nnUNetResEncUNetMPlans --config 3d_fullres \\
    --trainer-class nnUNetTrainer \\
    --fold 0 \\
    --every-n-epochs 10

The list of val case_ids is read from splits_final.json which nnU-Net
deposits in nnUNet_preprocessed/Dataset<NNN>_<task>/. The CT files come
from nnUNet_raw/Dataset<NNN>_<task>/imagesTr/<case>_0000.nrrd which the
trainer already symlinked at export time.
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import re
import shutil
import subprocess
import sys
import time


EPOCH_RE = re.compile(r":\s*Epoch (\d+)\s*$")
TRAIN_LOSS_RE  = re.compile(r":\s*train_loss\s+([-+\d.eE]+)")
VAL_LOSS_RE    = re.compile(r":\s*val_loss\s+([-+\d.eE]+)")
LR_RE          = re.compile(r":\s*Current learning rate:\s*([\d.eE+-]+)")
EPOCH_TIME_RE  = re.compile(r":\s*Epoch time:\s*([\d.]+)")
PSEUDO_DICE_RE = re.compile(r":\s*Pseudo dice\s*\[(.+?)\]\s*$")
EMA_DICE_RE    = re.compile(r"New best EMA pseudo Dice:\s*([\d.]+)")
FLOAT_INSIDE_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def latest_epoch(log_path):
    """Return the most recent epoch number in the training log (None if no
    log or no Epoch line). Bounded read so we don't choke on big logs."""
    if not log_path or not os.path.isfile(log_path):
        return None
    ep = None
    try:
        with open(log_path, "rb") as f:
            f.seek(0, os.SEEK_END); end = f.tell()
            f.seek(max(0, end - 65536))
            data = f.read().decode("utf-8", errors="replace")
        for ln in data.splitlines():
            m = EPOCH_RE.search(ln)
            if m:
                ep = int(m.group(1))
    except OSError:
        pass
    return ep


def parse_full_training_log(log_path):
    """Walk the log and return {epoch: {lr, train_loss, val_loss, pseudo_dice,
    epoch_time_s, ema_dice}}. ema_dice carries forward across epochs once a
    'New best' line lands; the rest reset at each Epoch header. Designed for
    bulk parsing — fine for 1000-epoch logs (a few hundred kB)."""
    out = {}
    if not log_path or not os.path.isfile(log_path):
        return out
    cur = None
    last_ema = None
    try:
        with open(log_path, encoding="utf-8", errors="replace") as f:
            for ln in f:
                m = EPOCH_RE.search(ln)
                if m:
                    cur = int(m.group(1))
                    out[cur] = {"lr": None, "train_loss": None, "val_loss": None,
                                "pseudo_dice": None, "epoch_time_s": None,
                                "ema_dice": last_ema}
                    continue
                if cur is None:
                    continue
                rec = out[cur]
                if rec["lr"] is None:
                    m = LR_RE.search(ln)
                    if m:
                        try: rec["lr"] = float(m.group(1))
                        except ValueError: pass
                if rec["train_loss"] is None:
                    m = TRAIN_LOSS_RE.search(ln)
                    if m:
                        try: rec["train_loss"] = float(m.group(1))
                        except ValueError: pass
                if rec["val_loss"] is None:
                    m = VAL_LOSS_RE.search(ln)
                    if m:
                        try: rec["val_loss"] = float(m.group(1))
                        except ValueError: pass
                if rec["pseudo_dice"] is None:
                    m = PSEUDO_DICE_RE.search(ln)
                    if m:
                        inner = re.sub(r"np\.float\d+", "", m.group(1))
                        vals = [float(v) for v in FLOAT_INSIDE_RE.findall(inner)]
                        if vals:
                            rec["pseudo_dice"] = sum(vals) / len(vals)
                if rec["epoch_time_s"] is None:
                    m = EPOCH_TIME_RE.search(ln)
                    if m:
                        try: rec["epoch_time_s"] = float(m.group(1))
                        except ValueError: pass
                m = EMA_DICE_RE.search(ln)
                if m:
                    try:
                        last_ema = float(m.group(1))
                        rec["ema_dice"] = last_ema
                    except ValueError: pass
    except OSError:
        pass
    return out


def per_case_metrics(pred_path, gt_path):
    """Dice + voxel/mL volumes for a (prediction, ground-truth) NRRD pair.
    Treats any non-zero label as foreground (single-class LN segmentation)."""
    import SimpleITK as sitk
    import numpy as np
    try:
        pred_img = sitk.ReadImage(pred_path)
        gt_img = sitk.ReadImage(gt_path)
    except Exception as exc:
        return {"error": f"read: {exc}"}
    pred = sitk.GetArrayFromImage(pred_img)
    gt = sitk.GetArrayFromImage(gt_img)
    if pred.shape != gt.shape:
        return {"error": f"shape mismatch pred={pred.shape} gt={gt.shape}"}
    pred_b = pred != 0
    gt_b = gt != 0
    inter = int(np.logical_and(pred_b, gt_b).sum())
    p_sum = int(pred_b.sum()); g_sum = int(gt_b.sum())
    denom = p_sum + g_sum
    dice = (2.0 * inter / denom) if denom else 0.0
    sx, sy, sz = pred_img.GetSpacing()
    voxel_ml = (sx * sy * sz) / 1000.0
    return {
        "dice": float(dice),
        "pred_voxel_count": p_sum,
        "gt_voxel_count":   g_sum,
        "pred_volume_ml":   p_sum * voxel_ml,
        "gt_volume_ml":     g_sum * voxel_ml,
    }


def existing_metrics_keys(metrics_path):
    """{(epoch, case_id), ...} for records already in metrics.jsonl so the
    retroactive pass doesn't re-compute or double-write."""
    seen = set()
    if not os.path.isfile(metrics_path):
        return seen
    try:
        with open(metrics_path) as f:
            for ln in f:
                try:
                    r = json.loads(ln)
                    seen.add((int(r["epoch"]), r["case_id"]))
                except Exception:
                    continue
    except OSError:
        pass
    return seen


def write_metrics(metrics_path, fold, epoch, case_id, case_stats, ctx):
    """Append one JSONL record. Atomicity is line-level (write+flush)."""
    rec = {
        "epoch": int(epoch),
        "fold":  int(fold),
        "case_id": case_id,
        **case_stats,
        **ctx,
        "wrote_at": datetime.datetime.now(datetime.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
    }
    line = json.dumps(rec, separators=(",", ":"))
    with open(metrics_path, "a") as f:
        f.write(line + "\n")
        f.flush()


def newest_log(fold_dir):
    """Return the most recently created training_log_*.txt under fold_dir."""
    if not os.path.isdir(fold_dir):
        return None
    logs = [p for p in os.listdir(fold_dir)
            if p.startswith("training_log_") and p.endswith(".txt")]
    if not logs:
        return None
    return os.path.join(fold_dir, sorted(logs)[-1])


def load_val_case_ids(prep_dir, fold):
    """Read splits_final.json under nnUNet_preprocessed/<dataset>/ and return
    the val case_ids for the requested fold."""
    splits_path = os.path.join(prep_dir, "splits_final.json")
    if not os.path.isfile(splits_path):
        return []
    with open(splits_path) as f:
        splits = json.load(f)
    if fold >= len(splits):
        return []
    return list(splits[fold].get("val", []))


def stage_inputs(case_ids, raw_dir, work_inputs):
    """Symlink each case's CT (<case>_0000.nrrd) into a fresh work_inputs dir
    so nnUNetv2_predict only sees this fold's val subset."""
    os.makedirs(work_inputs, exist_ok=True)
    # Clear any prior symlinks so removed cases don't haunt us.
    for f in os.listdir(work_inputs):
        os.unlink(os.path.join(work_inputs, f))
    missing = []
    for cid in case_ids:
        src = os.path.join(raw_dir, f"{cid}_0000.nrrd")
        if not os.path.lexists(src):
            missing.append(cid); continue
        os.symlink(src, os.path.join(work_inputs, f"{cid}_0000.nrrd"))
    return missing


def find_train_pids():
    """Return the PIDs of any running nnUNetv2_train processes on this host.
    Used to pause/resume the trainer around predict so the two don't fight
    for the GPU. pgrep -f matches against the full command line."""
    try:
        out = subprocess.check_output(["pgrep", "-f", "nnUNetv2_train"],
                                      text=True).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []
    pids = []
    for ln in out.splitlines():
        ln = ln.strip()
        if ln.isdigit():
            pids.append(int(ln))
    return pids


def pause_train(pids):
    import signal
    for p in pids:
        try: os.kill(p, signal.SIGSTOP)
        except ProcessLookupError: pass


def resume_train(pids):
    import signal
    for p in pids:
        try: os.kill(p, signal.SIGCONT)
        except ProcessLookupError: pass


def run_predict_snapshot(args, fold_dir, epoch, work_inputs):
    """Run nnUNetv2_predict with checkpoint_latest.pth and stash the results
    under validation_raw/ keyed by epoch.

    If --pause-train is set, SIGSTOP any nnUNetv2_train on this host before
    starting predict and SIGCONT after — keeps the GPU from being shared.
    The trainer keeps its CUDA context (memory not freed), so the box still
    needs roughly 2× the working set in VRAM; A100 80GB has plenty for
    3d_fullres at our patch sizes."""
    snap_dir = os.path.join(fold_dir, "validation_raw")
    os.makedirs(snap_dir, exist_ok=True)
    work_outputs = os.path.join(args.work_dir,
                                f"predict_fold{args.fold}_epoch{epoch:05d}")
    os.makedirs(work_outputs, exist_ok=True)

    env = dict(os.environ)
    env.update({
        "nnUNet_raw":          os.path.join(args.nnunet_base, "nnUNet_raw"),
        "nnUNet_preprocessed": os.path.join(args.nnunet_base, "nnUNet_preprocessed"),
        "nnUNet_results":      os.path.join(args.nnunet_base, "nnUNet_results"),
    })
    cmd = [
        os.path.join(args.venv_bin, "nnUNetv2_predict"),
        "-i", work_inputs,
        "-o", work_outputs,
        "-d", str(args.dataset_id),
        "-c", args.config,
        "-p", args.plans_identifier,
        "-tr", args.trainer_class,
        "-chk", "checkpoint_latest.pth",
        "-f", str(args.fold),
        "--disable_progress_bar",
    ]
    paused_pids = []
    if args.pause_train:
        paused_pids = find_train_pids()
        if paused_pids:
            print(f"[snapshot] SIGSTOPing nnUNetv2_train pids {paused_pids}",
                  flush=True)
            pause_train(paused_pids)
    print(f"[snapshot] epoch={epoch} cmd={' '.join(cmd)}", flush=True)
    try:
        try:
            rc = subprocess.call(cmd, env=env)
        except FileNotFoundError as exc:
            print(f"[snapshot] cannot exec nnUNetv2_predict: {exc}", flush=True)
            return False
        if rc != 0:
            print(f"[snapshot] predict exited {rc}; skipping snapshot",
                  flush=True)
            return False
    finally:
        if paused_pids:
            resume_train(paused_pids)
            print(f"[snapshot] SIGCONTed nnUNetv2_train pids {paused_pids}",
                  flush=True)

    # Rename outputs to validation_raw/<case>_epoch_NNNNN.nrrd
    count = 0
    for fn in os.listdir(work_outputs):
        if not (fn.endswith(".nrrd") or fn.endswith(".nii.gz")):
            continue
        case_id = fn.split(".")[0]
        # nnU-Net's filename is <case_id>.nrrd or <case_id>.nii.gz (no _0000)
        src = os.path.join(work_outputs, fn)
        dst = os.path.join(snap_dir, f"{case_id}_epoch_{epoch:05d}.nrrd")
        if fn.endswith(".nii.gz"):
            # nnU-Net's default is nii.gz; convert to nrrd via SimpleITK
            # so all snapshots are uniform for the Slicer sequence loader.
            try:
                import SimpleITK as sitk
                img = sitk.ReadImage(src)
                sitk.WriteImage(img, dst, useCompression=True)
            except Exception as exc:
                print(f"[snapshot] {case_id} convert: {exc}", flush=True)
                continue
        else:
            shutil.copy2(src, dst)
        count += 1
    shutil.rmtree(work_outputs, ignore_errors=True)
    print(f"[snapshot] wrote {count} files to {snap_dir}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--nnunet-base", required=True,
                    help="Parent of nnUNet_raw|preprocessed|results.")
    ap.add_argument("--dataset-id", type=int, required=True)
    ap.add_argument("--task", required=True, help="Task name segment, e.g. LNQinguinal.")
    ap.add_argument("--plans-identifier", default="nnUNetResEncUNetMPlans")
    ap.add_argument("--config", default="3d_fullres")
    ap.add_argument("--trainer-class", default="nnUNetTrainer")
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--every-n-epochs", type=int, default=10,
                    help="Snapshot frequency (epochs). Default 10 → 100 frames/fold.")
    ap.add_argument("--poll-seconds", type=int, default=30)
    ap.add_argument("--venv-bin", default="/opt/lnq/nnenv/bin",
                    help="Where nnUNetv2_predict lives.")
    ap.add_argument("--work-dir", default="/tmp/lnq-val-snapshots")
    ap.add_argument("--max-epoch", type=int, default=10000,
                    help="Exit once this many epochs have been observed.")
    ap.add_argument("--max-val-cases", type=int, default=0,
                    help="If >0, only snapshot this many val cases per fold "
                    "(deterministic by sorted case_id). Caps GPU contention; "
                    "still enough material for visual convergence scrub.")
    ap.add_argument("--pause-train", action="store_true",
                    help="SIGSTOP local nnUNetv2_train processes before "
                    "predict and SIGCONT after, so the predict step gets the "
                    "GPU to itself. Recommended when sharing one A100 between "
                    "training and validation predict.")
    args = ap.parse_args()

    os.makedirs(args.work_dir, exist_ok=True)
    ds_name = f"Dataset{args.dataset_id:03d}_{args.task}"
    plans_subdir = f"{args.trainer_class}__{args.plans_identifier}__{args.config}"
    fold_dir = os.path.join(args.nnunet_base, "nnUNet_results",
                            ds_name, plans_subdir, f"fold_{args.fold}")
    raw_dir = os.path.join(args.nnunet_base, "nnUNet_raw", ds_name, "imagesTr")
    prep_dir = os.path.join(args.nnunet_base, "nnUNet_preprocessed", ds_name)

    val_case_ids = load_val_case_ids(prep_dir, args.fold)
    if not val_case_ids:
        print(f"[snapshot] no val case_ids in {prep_dir}/splits_final.json fold={args.fold}",
              file=sys.stderr)
        sys.exit(2)
    if args.max_val_cases and len(val_case_ids) > args.max_val_cases:
        val_case_ids = sorted(val_case_ids)[:args.max_val_cases]
        print(f"[snapshot] capped to first {args.max_val_cases} val cases",
              flush=True)
    print(f"[snapshot] fold {args.fold} val cases: {val_case_ids}", flush=True)

    work_inputs = os.path.join(args.work_dir, f"inputs_fold{args.fold}")
    missing = stage_inputs(val_case_ids, raw_dir, work_inputs)
    if missing:
        print(f"[snapshot] WARN: missing raw CTs for {missing}", flush=True)

    # Path to ground-truth segs for this fold's val cases. Identical to what
    # nnU-Net uses internally for its Pseudo dice.
    labels_dir = os.path.join(args.nnunet_base, "nnUNet_raw", ds_name, "labelsTr")
    snap_dir = os.path.join(fold_dir, "validation_raw")
    os.makedirs(snap_dir, exist_ok=True)
    metrics_path = os.path.join(snap_dir, "metrics.jsonl")

    def collect_metrics_for_epoch(epoch_int, log_full):
        """Read each <case>_epoch_<epoch>.nrrd in snap_dir, compute per-case
        Dice + volume vs labelsTr/<case>.nrrd, append to metrics.jsonl. Skip
        any (epoch, case) already present in the file. Robust to partial
        snapshot completions."""
        seen = existing_metrics_keys(metrics_path)
        ctx = log_full.get(epoch_int, {})
        for case_id in val_case_ids:
            if (epoch_int, case_id) in seen:
                continue
            pred_path = os.path.join(snap_dir,
                                     f"{case_id}_epoch_{epoch_int:05d}.nrrd")
            gt_path = os.path.join(labels_dir, f"{case_id}.nrrd")
            if not os.path.isfile(pred_path) or not os.path.isfile(gt_path):
                continue
            try:
                stats = per_case_metrics(pred_path, gt_path)
            except Exception as exc:
                stats = {"error": f"{type(exc).__name__}: {exc}"}
            try:
                write_metrics(metrics_path, args.fold, epoch_int, case_id,
                              stats, ctx)
                print(f"[metrics] e{epoch_int:05d} {case_id} "
                      f"dice={stats.get('dice')} pred_ml={stats.get('pred_volume_ml')}",
                      flush=True)
            except OSError as exc:
                print(f"[metrics] WRITE FAILED {case_id} e{epoch_int}: {exc}",
                      flush=True)

    # ----- retroactive metrics pass -----
    # Any snapshots that landed before this version of the snapshotter ran
    # get their metrics filled in now. Idempotent — re-running is a no-op
    # because existing_metrics_keys() deduplicates.
    log_full = parse_full_training_log(newest_log(fold_dir))
    discovered = {}  # epoch -> True (snapshot exists for at least one case)
    for fn in os.listdir(snap_dir) if os.path.isdir(snap_dir) else []:
        m = re.search(r"_epoch_(\d+)\.nrrd$", fn)
        if m: discovered[int(m.group(1))] = True
    if discovered:
        print(f"[metrics] retroactive: {sorted(discovered.keys())}", flush=True)
        for ep in sorted(discovered):
            collect_metrics_for_epoch(ep, log_full)

    last_snapshot_epoch = -args.every_n_epochs
    while True:
        log = newest_log(fold_dir)
        ep = latest_epoch(log)
        if ep is None:
            time.sleep(args.poll_seconds); continue
        if ep >= args.max_epoch:
            print(f"[snapshot] reached max_epoch={args.max_epoch}; exiting",
                  flush=True)
            return
        # Only snapshot once per N-epoch boundary, and require the next
        # checkpoint to have been written (checkpoint_latest.pth touched
        # after we last looked).
        target = ((ep // args.every_n_epochs) * args.every_n_epochs)
        if target > last_snapshot_epoch:
            ckpt = os.path.join(fold_dir, "checkpoint_latest.pth")
            if os.path.isfile(ckpt):
                ok = run_predict_snapshot(args, fold_dir, target, work_inputs)
                if ok:
                    # Re-parse the log so the new epoch's metrics (lr,
                    # losses, ema_dice, etc.) are picked up.
                    log_full = parse_full_training_log(newest_log(fold_dir))
                    collect_metrics_for_epoch(target, log_full)
                    last_snapshot_epoch = target
            else:
                print(f"[snapshot] no checkpoint_latest.pth yet at epoch {ep}",
                      flush=True)
        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    main()
