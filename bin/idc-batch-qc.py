#!/usr/bin/env python3
"""idc-batch-qc.py — per-case QC for an ingest-idc-cohort batch.

Walks the predictions tree the ingest pipeline produces and emits:

  * A CSV (one row per case) with Dice / sensitivity / precision at the
    default argmax cutoff p>=0.5 AND at the log-scale cutoff p>=0.001
    (the calibrated low-end threshold that lifted MED_LYMPH_021 Dice
    0.45 -> 0.78 in the earlier study).
  * Volumes in mL: ground truth, predicted at p>=0.5, predicted at
    p>=0.001.
  * Probability-distribution stats over the GT-only "missed" voxels —
    the bimodality vs. log-enrichment story we want to be able to
    sort-by at the Monday meeting.
  * Per-case axial PNG: CT background, Inferno overlay of the
    probability map (clamped to [0, 0.05] so the low-end signal is
    visible), GT outline in red, predicted SEG outline in cyan.
    Frame is the slice with the largest combined (GT + pred) foot-
    print so the screenshot lands somewhere informative.

Designed to run on lnq-inferencer once the batch has produced a few
cases. Re-runnable — skips cases whose PNG already exists, and works
fine partway through a still-running batch."""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time


def case_paths(predictions_root, nrrd_root, case_id):
    """Return the four file paths we expect per case. Some may not
    exist yet (batch in progress) or for the entire collection (no GT
    in IDC, etc.)."""
    return {
        "ct":   os.path.join(nrrd_root,        f"{case_id}_0000.nrrd"),
        "gt":   os.path.join(nrrd_root,        f"{case_id}.nrrd"),
        "pred": os.path.join(predictions_root, f"{case_id}.nrrd"),
        "prob": os.path.join(predictions_root, f"{case_id}-prob.nrrd"),
    }


def discover_cases(predictions_root):
    """Find every case_id that has a predicted SEG (and therefore is
    worth QC'ing)."""
    out = []
    for p in sorted(glob.glob(os.path.join(predictions_root, "*.nrrd"))):
        base = os.path.basename(p)
        if base.endswith("-prob.nrrd"):
            continue
        out.append(base[:-len(".nrrd")])
    return out


def voxel_volume_mL(image):
    sx, sy, sz = image.GetSpacing()
    return sx * sy * sz / 1000.0


def compute_case_stats(paths):
    """Open the volumes lazily and return a dict of per-case numbers
    suitable for one CSV row."""
    import numpy as np
    import SimpleITK as sitk

    if not os.path.isfile(paths["ct"]) or not os.path.isfile(paths["prob"]):
        return None

    pred_img = sitk.ReadImage(paths["pred"]) if os.path.isfile(paths["pred"]) else None
    prob_img = sitk.ReadImage(paths["prob"])
    ct_img   = sitk.ReadImage(paths["ct"])
    gt_img   = sitk.ReadImage(paths["gt"]) if os.path.isfile(paths["gt"]) else None

    prob = sitk.GetArrayFromImage(prob_img)
    pred = (sitk.GetArrayFromImage(pred_img) > 0) if pred_img is not None else (prob >= 0.5)
    gt   = (sitk.GetArrayFromImage(gt_img)   > 0) if gt_img   is not None else None

    vml = voxel_volume_mL(ct_img)
    out = {
        "case_id": paths.get("case_id", ""),
        "pred_volume_mL_p0.5":    round(int(pred.sum())                * vml, 2),
        "pred_volume_mL_p0.001":  round(int((prob >= 0.001).sum())     * vml, 2),
        "pred_volume_mL_p0.0001": round(int((prob >= 0.0001).sum())    * vml, 2),
        "voxels_above_p0.5":      int(pred.sum()),
        "voxels_above_p0.1":      int((prob >= 0.1).sum()),
        "voxels_above_p0.01":     int((prob >= 0.01).sum()),
        "voxels_above_p0.001":    int((prob >= 0.001).sum()),
        "prob_max":               round(float(prob.max()), 4),
        "prob_p99":               round(float(__import__("numpy").percentile(prob, 99)), 4),
    }

    if gt is None:
        out.update({"has_gt": False,
                     "gt_volume_mL": "", "dice_p0.5": "", "dice_p0.001": "",
                     "sensitivity_p0.5": "", "precision_p0.5": "",
                     "missed_prob_mean": "", "missed_prob_median": "",
                     "missed_above_p0.001_frac": "", "missed_above_p0.1_frac": ""})
    else:
        gs = int(gt.sum())
        ps = int(pred.sum())
        inter = int((gt & pred).sum())
        denom = ps + gs
        dice  = 2.0 * inter / denom if denom else 0.0
        sens  = inter / gs if gs else 0
        prec  = inter / ps if ps else 0

        # Re-threshold the probability map at p>=0.001 (the log-scale
        # calibrated cutoff) and recompute Dice.
        pred_lo = (prob >= 0.001)
        inter_lo = int((gt & pred_lo).sum())
        ps_lo = int(pred_lo.sum())
        denom_lo = ps_lo + gs
        dice_lo = 2.0 * inter_lo / denom_lo if denom_lo else 0.0

        missed = (~pred) & gt
        missed_vals = prob[missed]
        if missed_vals.size:
            import numpy as np
            m_mean   = float(missed_vals.mean())
            m_median = float(np.median(missed_vals))
            m_above_0001 = float((missed_vals >= 0.001).mean())
            m_above_01   = float((missed_vals >= 0.1).mean())
        else:
            m_mean = m_median = m_above_0001 = m_above_01 = 0.0

        out.update({
            "has_gt": True,
            "gt_volume_mL":               round(gs * vml, 2),
            "dice_p0.5":                  round(dice, 4),
            "dice_p0.001":                round(dice_lo, 4),
            "sensitivity_p0.5":           round(sens, 4),
            "precision_p0.5":             round(prec, 4),
            "missed_prob_mean":           round(m_mean,   4),
            "missed_prob_median":         round(m_median, 4),
            "missed_above_p0.001_frac":   round(m_above_0001, 4),
            "missed_above_p0.1_frac":     round(m_above_01,   4),
        })
    return out


def render_overlay_png(paths, out_png, low_max=0.05):
    """Pick the slice with the largest combined GT + pred footprint,
    render an axial overlay PNG. low_max sets the upper end of the
    Inferno colormap so faint signal is visible without burning out
    on the high-confidence core."""
    import numpy as np
    import SimpleITK as sitk
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    ct_img   = sitk.ReadImage(paths["ct"])
    prob_img = sitk.ReadImage(paths["prob"])
    pred_img = sitk.ReadImage(paths["pred"]) if os.path.isfile(paths["pred"]) else None
    gt_img   = sitk.ReadImage(paths["gt"])   if os.path.isfile(paths["gt"])   else None

    ct   = sitk.GetArrayFromImage(ct_img)
    prob = sitk.GetArrayFromImage(prob_img)
    pred = (sitk.GetArrayFromImage(pred_img) > 0) if pred_img else (prob >= 0.5)
    gt   = (sitk.GetArrayFromImage(gt_img)   > 0) if gt_img   else None

    # Slice with the largest combined GT + pred footprint, or fall back
    # to peak predicted-foreground slice if there's no GT.
    if gt is not None:
        per_slice = (gt | pred).sum(axis=(1, 2))
    else:
        per_slice = pred.sum(axis=(1, 2))
    if per_slice.sum() == 0:
        z = ct.shape[0] // 2
    else:
        z = int(per_slice.argmax())

    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    ax.imshow(ct[z], cmap="gray",
              vmin=-200, vmax=400, interpolation="nearest", aspect="equal")
    # Mask the heatmap below 0.001 to leave the CT clean elsewhere.
    prob_slice = prob[z]
    masked = np.ma.masked_less(prob_slice, 0.001)
    ax.imshow(masked, cmap="inferno", vmin=0, vmax=low_max,
              alpha=0.55, interpolation="nearest")
    if gt is not None and gt[z].any():
        ax.contour(gt[z].astype(float), levels=[0.5], colors=["red"],
                    linewidths=1.0)
    if pred[z].any():
        ax.contour(pred[z].astype(float), levels=[0.5], colors=["cyan"],
                    linewidths=0.8)
    ax.set_title(
        f"{os.path.basename(paths['ct']).replace('_0000.nrrd', '')}  "
        f"slice z={z}  (prob windowed [0, {low_max}])\n"
        f"red = GT  ·  cyan = pred (p>=0.5)",
        fontsize=9)
    ax.set_axis_off()
    plt.tight_layout()
    fig.savefig(out_png, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                  formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-root",
                    default="/media/share/LNQ-data/idc/ct_lymph_nodes",
                    help="Stage root used by ingest-idc-cohort.py.")
    ap.add_argument("--model", default="mediastinal-v1",
                    help="Model subdir under <stage>/predictions/.")
    ap.add_argument("--out-dir", default=None,
                    help="Where CSV + PNGs land. Default: <stage>/qc/<model>/.")
    ap.add_argument("--csv-name", default="qc.csv")
    ap.add_argument("--no-png", action="store_true",
                    help="Skip PNG rendering (CSV only).")
    ap.add_argument("--png-low-max", type=float, default=0.05,
                    help="Upper end of the Inferno colormap window. "
                    "Lower = brighter faint signal.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    nrrd_root = os.path.join(args.stage_root, "nrrd")
    predictions_root = os.path.join(args.stage_root, "predictions", args.model)
    out_dir = args.out_dir or os.path.join(args.stage_root, "qc", args.model)
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, args.csv_name)

    if not os.path.isdir(predictions_root):
        print(f"no predictions dir: {predictions_root}", file=sys.stderr)
        sys.exit(2)

    cases = discover_cases(predictions_root)
    if args.limit:
        cases = cases[:args.limit]
    print(f"QC over {len(cases)} cases", flush=True)
    if not cases:
        print("nothing to QC yet.")
        return

    rows = []
    for i, case_id in enumerate(cases, 1):
        paths = case_paths(predictions_root, nrrd_root, case_id)
        paths["case_id"] = case_id
        t0 = time.time()
        stats = compute_case_stats(paths)
        if stats is None:
            print(f"[{i:3d}/{len(cases)}] {case_id}: skip (missing CT or prob)",
                  flush=True)
            continue
        rows.append(stats)

        if not args.no_png:
            out_png = os.path.join(out_dir, f"{case_id}.png")
            if os.path.isfile(out_png):
                continue
            try:
                render_overlay_png(paths, out_png, low_max=args.png_low_max)
            except Exception as exc:
                print(f"  WARN PNG failed for {case_id}: {exc}", flush=True)
        print(f"[{i:3d}/{len(cases)}] {case_id}: stats + png "
              f"in {time.time() - t0:.1f}s", flush=True)

    if rows:
        fieldnames = list(rows[0].keys())
        # Carry every case's keys in case some had GT and others didn't.
        for r in rows:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=fieldnames)
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
        print(f"\nwrote {csv_path} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
