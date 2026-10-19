#!/usr/bin/env python3
"""characterize-nrrd.py — describe / compare NRRD volumes.

Usage:
  characterize-nrrd.py <path>                # one file → JSON report on stdout
  characterize-nrrd.py <left> <right>        # two files → both reports + diff

The "voxel_sha256" hashes the decompressed numpy array's raw bytes in canonical
C-order. Two files with identical voxel_sha256 but different file_sha256 are
just recompressed/repackaged versions of the same volume — for training, that
is functionally equivalent (data parity holds). Anything else (shape, spacing,
origin, direction, dtype, or pixel-wise diff) characterizes what specifically
changed: recompression vs. crop vs. re-orient vs. wrong-patient.

Designed to run on both Mac (via Slicer's PythonSlicer / SimpleITK) and the
Js2 trainer venv (/opt/lnq/nnenv/bin/python), so you can produce reports on
each side and compare without needing both files in the same place.
"""

import argparse
import hashlib
import json
import os
import sys

try:
    import SimpleITK as sitk
    import numpy as np
except ImportError as exc:
    print(f"Missing dependency: {exc}. Use SimpleITK + numpy "
          "(Slicer's PythonSlicer or /opt/lnq/nnenv/bin/python).", file=sys.stderr)
    sys.exit(2)


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def characterize(path):
    img = sitk.ReadImage(path)
    arr = sitk.GetArrayFromImage(img)  # (z, y, x) per ITK convention
    voxel_sha = hashlib.sha256(arr.tobytes(order="C")).hexdigest()
    # 16-bin histogram covering the full intensity range so two volumes' shape
    # distributions can be eyeballed (good "wrong patient" smoke-test).
    hist, edges = np.histogram(arr, bins=16)
    return {
        "path":            path,
        "file_size":       os.path.getsize(path),
        "file_sha256":     file_sha256(path),
        "shape_zyx":       list(arr.shape),
        "spacing_xyz":     [round(v, 6) for v in img.GetSpacing()],
        "origin_xyz":      [round(v, 6) for v in img.GetOrigin()],
        "direction":       [round(v, 6) for v in img.GetDirection()],
        "dtype":           str(arr.dtype),
        "voxel_sha256":    voxel_sha,
        "intensity_min":   float(arr.min()),
        "intensity_max":   float(arr.max()),
        "intensity_mean":  float(arr.mean()),
        "intensity_std":   float(arr.std()),
        "histogram_bins":  [round(float(e), 3) for e in edges],
        "histogram_counts": hist.tolist(),
    }, arr


def diff_reports(a, b, arr_a=None, arr_b=None):
    """Compare two characterize() outputs, optionally with full voxel arrays
    for byte-level statistics when shapes agree."""
    out = {}
    same_shape = (a["shape_zyx"] == b["shape_zyx"])
    same_geom = (a["spacing_xyz"] == b["spacing_xyz"]
                 and a["origin_xyz"] == b["origin_xyz"]
                 and a["direction"] == b["direction"])
    out["same_shape"] = same_shape
    out["same_geometry"] = same_geom
    out["same_dtype"] = (a["dtype"] == b["dtype"])
    out["voxel_sha256_match"] = (a["voxel_sha256"] == b["voxel_sha256"])
    out["file_sha256_match"] = (a["file_sha256"] == b["file_sha256"])
    if arr_a is not None and arr_b is not None and same_shape and same_geom \
            and a["dtype"] == b["dtype"] and a["voxel_sha256"] != b["voxel_sha256"]:
        d = arr_a.astype("float64") - arr_b.astype("float64")
        absd = np.abs(d)
        out["voxel_diff"] = {
            "max_abs":   float(absd.max()),
            "mean_abs":  float(absd.mean()),
            "n_differing": int((d != 0).sum()),
            "n_total":   int(d.size),
            "frac_differing": float((d != 0).mean()),
            "intensity_correlation": float(
                np.corrcoef(arr_a.ravel().astype("float64"),
                            arr_b.ravel().astype("float64"))[0, 1]),
        }
    # Plain-language characterization.
    if out["file_sha256_match"]:
        out["interpretation"] = "identical files"
    elif out["voxel_sha256_match"]:
        out["interpretation"] = ("identical voxel data — different file "
                                 "encoding (compression / header style)")
    elif not same_shape:
        out["interpretation"] = ("different shape → cropped, padded, or "
                                 "resampled")
    elif not same_geom:
        out["interpretation"] = ("same shape, different spacing/origin/direction "
                                 "→ re-oriented or coordinate-system change")
    else:
        # Same shape + geometry but voxels differ.
        vd = out.get("voxel_diff", {})
        corr = vd.get("intensity_correlation")
        if corr is not None and corr > 0.999:
            out["interpretation"] = ("same scan, minor numeric drift "
                                     "(re-encoded, rescaled, or precision change)")
        elif corr is not None and corr > 0.9:
            out["interpretation"] = ("same patient, intensity-rescaled "
                                     "(e.g. window/level baked in)")
        elif corr is not None and corr < 0.5:
            out["interpretation"] = ("LIKELY DIFFERENT PATIENT — intensity "
                                     "correlation very low")
        else:
            out["interpretation"] = ("same shape/geometry but voxels differ "
                                     "non-trivially")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paths", nargs="+", help="One or two NRRD paths")
    ap.add_argument("--no-voxel-stats", action="store_true",
                    help="Skip voxel-diff statistics (faster for batch use).")
    args = ap.parse_args()
    if len(args.paths) > 2:
        ap.error("at most two paths")

    chars, arrs = [], []
    for p in args.paths:
        c, a = characterize(p)
        chars.append(c)
        arrs.append(a)

    if len(chars) == 1:
        print(json.dumps(chars[0], indent=2))
        return
    arg_a = None if args.no_voxel_stats else arrs[0]
    arg_b = None if args.no_voxel_stats else arrs[1]
    report = {
        "left":  chars[0],
        "right": chars[1],
        "diff":  diff_reports(chars[0], chars[1], arg_a, arg_b),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
