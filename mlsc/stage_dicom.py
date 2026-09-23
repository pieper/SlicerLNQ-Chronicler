#!/usr/bin/env python3
"""stage_dicom.py — DICOM study → per-series NRRD volumes for LNQ inference.

One case (an `E########` directory under --input) per invocation. Scans every
file's header, groups by SeriesInstanceUID, decides which series are axial CT
volumes worth running the models on, checks their geometry, and writes

    <work>/<case>/<SeriesNumber:03d>_<sanitized SeriesDescription>[_stkN]/
        ct.nrrd          int16 HU, gzip, LPS
        geometry.json    decision, flags, spacing stats, key tags, source files
    <work>/<case>/series.csv   one row per series (staged AND rejected)

Rejected: non-CT modality, non-image SOP classes (dose reports, SR, secondary
capture), ImageType LOCALIZER/SCOUT/TOPOGRAM, non-axial orientation (coronal /
sagittal MPRs), fewer than --min-slices slices.

Kept but flagged (in `flags`): duplicate slice positions, missing slices
(filled by linear interpolation on the true grid and counted in
`interpolated_slices=N`; optionally rejected as `incomplete_series` via
--max-missing-frac), irregular spacing, gantry tilt / sheared stacks, mixed
rescale, multiple stacks in one series, multiple studies in one case.
Volumes are written best effort (median z-spacing when irregular) unless
--strict is given. Re-running reconverts any volume whose source file count
changed since it was staged. Note that a series uniformly thinned (every
other slice absent) is indistinguishable from a coarser acquisition and is
not flagged.

Kept and tagged (informational, in `tags`): spectral-derived series (VNC,
iodine, monoenergetic, Z-effective), reconstruction kernel, matrix size,
slice thickness, kVp.

PHI: nothing patient-identifying is written. Manifests carry the case id
(directory name), series number/description/UID, geometry and scanner
settings only. NRRD files carry no DICOM tags.

Usage (see run_pipeline.sh for the sbatch wrapper):

    stage_dicom.py --input /vast/lnq/PCCT_exemplary_cases \\
                   --work  /vast/lnq/pdac-processing --list-cases
    stage_dicom.py --input ... --work ... --case E12345678 [--dry-run]
    stage_dicom.py --input ... --work ... --case-index $SLURM_ARRAY_TASK_ID
"""
from __future__ import annotations

import argparse
import csv
import datetime
import json
import logging
import os
import re
import sys
import time

CASE_RE = re.compile(r"^E\d{8}$")

CT_IMAGE = "1.2.840.10008.5.1.4.1.1.2"
ENHANCED_CT = "1.2.840.10008.5.1.4.1.1.2.1"
LEGACY_ENHANCED_CT = "1.2.840.10008.5.1.4.1.1.2.2"
CT_SOP_CLASSES = {CT_IMAGE, ENHANCED_CT, LEGACY_ENHANCED_CT}
MULTIFRAME_SOP_CLASSES = {ENHANCED_CT, LEGACY_ENHANCED_CT}

SCOUT_TOKENS = {"LOCALIZER", "SCOUT", "TOPOGRAM", "TOPO"}
SPECTRAL_PATTERNS = [
    ("vnc", re.compile(r"\bVNC\b|VIRTUAL[ _]?NON[ _]?CONTRAST", re.I)),
    ("iodine", re.compile(r"IODINE|\bIMAP\b|I-?MAP", re.I)),
    ("monoenergetic", re.compile(r"MONO|\bkeV\b|\d{2,3}\s*keV", re.I)),
    ("zeff", re.compile(r"Z[ _-]?EFF", re.I)),
    ("spectral", re.compile(r"SPECTRAL|\bSPP\b|PURE[ _]?LUMEN|CALCIUM[ _]?REMOVAL", re.I)),
]
KERNEL_RE = re.compile(r"\b([BQHS][rv]?\d{2}[a-z]?)\b")

# Series CSV columns (also what build_cohort.py aggregates). Keep in one place.
SERIES_COLUMNS = [
    "case_id", "study_index", "series_number", "series_description", "series_uid",
    "volume_id", "series_dir", "decision", "reasons", "flags", "tags",
    "modality", "sop_class", "image_type", "n_files", "n_frames", "n_slices",
    "rows", "cols", "pixel_spacing_mm", "slice_thickness_mm",
    "z_spacing_mm", "z_spacing_min_mm", "z_spacing_max_mm", "extent_z_mm",
    "kernel", "kvp", "spectral", "n_voxels", "ct_path", "staged_at",
]

log = logging.getLogger("stage_dicom")


# --------------------------------------------------------------------------- utils

def sanitize(text, maxlen=48):
    """Filesystem / nnU-Net safe identifier chunk: [A-Za-z0-9-] only."""
    s = str(text or "").strip().replace(".", "-")
    s = re.sub(r"[^A-Za-z0-9-]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    s = s[:maxlen].rstrip("_-")
    return s or "series"


def find_case_dirs(input_root):
    """Directories whose basename is E######## anywhere under input_root.
    Does not descend into a case dir once found."""
    found = []
    for dirpath, dirnames, _ in os.walk(input_root):
        base = os.path.basename(dirpath)
        if CASE_RE.match(base):
            found.append(dirpath)
            dirnames[:] = []
            continue
        dirnames.sort()
    return sorted(found, key=os.path.basename)


def _float_list(value, n):
    try:
        out = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    return out if len(out) == n else None


def _cross(a, b):
    return [a[1] * b[2] - a[2] * b[1],
            a[2] * b[0] - a[0] * b[2],
            a[0] * b[1] - a[1] * b[0]]


def _dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def _norm(a):
    return _dot(a, a) ** 0.5


def _unit(a):
    n = _norm(a)
    return [x / n for x in a] if n else a


# --------------------------------------------------------------------------- header scan

def _is_probably_dicom(path):
    from pydicom.misc import is_dicom
    try:
        if is_dicom(path):
            return True
    except Exception:
        return False
    ext = os.path.splitext(path)[1].lower()
    # Raw (preamble-less) files are rare from clinical exports; only try them
    # when the name suggests DICOM.
    return ext in ("", ".dcm", ".ima", ".dicom")


def read_header(path):
    """Header-only read. Returns a pydicom Dataset or None."""
    import pydicom
    if not _is_probably_dicom(path):
        return None
    try:
        return pydicom.dcmread(path, stop_before_pixels=True, force=True)
    except Exception as exc:  # noqa: BLE001 — anything unreadable is "not dicom"
        log.debug("unreadable %s: %s", path, exc)
        return None


def _seq_item(ds, name):
    seq = getattr(ds, name, None)
    return seq[0] if seq else None


def _fg_value(frame, shared, seq_name, attr):
    """Look up a per-frame functional-group attribute, falling back to the
    shared functional groups."""
    for group in (frame, shared):
        if group is None:
            continue
        item = _seq_item(group, seq_name)
        if item is not None and hasattr(item, attr):
            return getattr(item, attr)
    return None


def slices_from_dataset(ds, path):
    """Return a list of slice records (one per image frame) for a header."""
    sop_class = str(getattr(ds, "SOPClassUID", ""))
    base = {
        "path": path,
        "frame": None,
        "instance_number": _int_or(getattr(ds, "InstanceNumber", None), 0),
        "acquisition_number": _int_or(getattr(ds, "AcquisitionNumber", None), None),
        "for_uid": str(getattr(ds, "FrameOfReferenceUID", "")),
        "rows": _int_or(getattr(ds, "Rows", None), 0),
        "cols": _int_or(getattr(ds, "Columns", None), 0),
        "temporal_position": _int_or(getattr(ds, "TemporalPositionIndex", None), None),
        "stack_id": str(getattr(ds, "StackID", "") or ""),
    }
    if sop_class in MULTIFRAME_SOP_CLASSES or getattr(ds, "NumberOfFrames", 1) not in (1, None, ""):
        n_frames = _int_or(getattr(ds, "NumberOfFrames", 1), 1)
        shared = _seq_item(ds, "SharedFunctionalGroupsSequence")
        per_frame = list(getattr(ds, "PerFrameFunctionalGroupsSequence", []) or [])
        out = []
        for i in range(n_frames):
            frame = per_frame[i] if i < len(per_frame) else None
            ipp = _fg_value(frame, shared, "PlanePositionSequence", "ImagePositionPatient")
            iop = _fg_value(frame, shared, "PlaneOrientationSequence", "ImageOrientationPatient")
            ps = _fg_value(frame, shared, "PixelMeasuresSequence", "PixelSpacing")
            st = _fg_value(frame, shared, "PixelMeasuresSequence", "SliceThickness")
            slope = _fg_value(frame, shared, "PixelValueTransformationSequence", "RescaleSlope")
            icept = _fg_value(frame, shared, "PixelValueTransformationSequence", "RescaleIntercept")
            fc = _seq_item(frame, "FrameContentSequence") if frame is not None else None
            rec = dict(base)
            rec.update({
                "frame": i,
                "ipp": _float_list(ipp, 3) if ipp is not None else None,
                "iop": _float_list(iop, 6) if iop is not None else None,
                "pixel_spacing": _float_list(ps, 2) if ps is not None else None,
                "slice_thickness": _float_or(st, None),
                "rescale_slope": _float_or(slope, 1.0),
                "rescale_intercept": _float_or(icept, 0.0),
                "stack_id": str(getattr(fc, "StackID", "") or "") if fc is not None else base["stack_id"],
                "temporal_position": _int_or(getattr(fc, "TemporalPositionIndex", None), None)
                if fc is not None else base["temporal_position"],
                "instance_number": base["instance_number"] * 100000 + i,
            })
            out.append(rec)
        return out

    rec = dict(base)
    rec.update({
        "ipp": _float_list(getattr(ds, "ImagePositionPatient", None) or [], 3),
        "iop": _float_list(getattr(ds, "ImageOrientationPatient", None) or [], 6),
        "pixel_spacing": _float_list(getattr(ds, "PixelSpacing", None) or [], 2),
        "slice_thickness": _float_or(getattr(ds, "SliceThickness", None), None),
        "rescale_slope": _float_or(getattr(ds, "RescaleSlope", None), 1.0),
        "rescale_intercept": _float_or(getattr(ds, "RescaleIntercept", None), 0.0),
    })
    return [rec]


def _int_or(v, default):
    try:
        return int(v)
    except (TypeError, ValueError):
        return default


def _float_or(v, default):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def scan_case(case_dir):
    """Walk case_dir, read every DICOM header, group by SeriesInstanceUID.

    Returns dict series_uid -> {"meta": {...}, "slices": [...]} and a count of
    files skipped as non-DICOM."""
    series = {}
    n_files = 0
    n_skipped = 0
    for dirpath, dirnames, filenames in os.walk(case_dir):
        dirnames.sort()
        for name in sorted(filenames):
            if name.upper() == "DICOMDIR" or name.startswith("."):
                continue
            path = os.path.join(dirpath, name)
            ds = read_header(path)
            if ds is None or not hasattr(ds, "SOPClassUID"):
                n_skipped += 1
                continue
            n_files += 1
            uid = str(getattr(ds, "SeriesInstanceUID", "") or "")
            if not uid:
                n_skipped += 1
                continue
            entry = series.get(uid)
            if entry is None:
                entry = series[uid] = {"meta": series_meta(ds), "slices": [], "n_files": 0}
            entry["n_files"] += 1
            entry["slices"].extend(slices_from_dataset(ds, path))
    log.info("scanned %d dicom files (%d skipped) in %d series", n_files, n_skipped, len(series))
    return series, n_skipped


def series_meta(ds):
    image_type = getattr(ds, "ImageType", None)
    if image_type is None:
        image_type = []
    elif isinstance(image_type, str):
        image_type = [image_type]
    kernel = getattr(ds, "ConvolutionKernel", None)
    if kernel is not None and not isinstance(kernel, str):
        try:
            kernel = "/".join(str(k) for k in kernel)
        except TypeError:
            kernel = str(kernel)
    return {
        "modality": str(getattr(ds, "Modality", "") or ""),
        "sop_class": str(getattr(ds, "SOPClassUID", "") or ""),
        "series_uid": str(getattr(ds, "SeriesInstanceUID", "") or ""),
        "study_uid": str(getattr(ds, "StudyInstanceUID", "") or ""),
        "study_date": str(getattr(ds, "StudyDate", "") or ""),
        "study_time": str(getattr(ds, "StudyTime", "") or ""),
        "series_number": _int_or(getattr(ds, "SeriesNumber", None), 0),
        "series_description": str(getattr(ds, "SeriesDescription", "") or ""),
        "image_type": [str(t).upper() for t in image_type],
        "kernel": str(kernel or ""),
        "kvp": _float_or(getattr(ds, "KVP", None), None),
        "reconstruction_diameter": _float_or(getattr(ds, "ReconstructionDiameter", None), None),
        "protocol_name": str(getattr(ds, "ProtocolName", "") or ""),
        "body_part": str(getattr(ds, "BodyPartExamined", "") or ""),
        "gantry_tilt": _float_or(getattr(ds, "GantryDetectorTilt", None), None),
        "manufacturer": str(getattr(ds, "Manufacturer", "") or ""),
        "model_name": str(getattr(ds, "ManufacturerModelName", "") or ""),
    }


# --------------------------------------------------------------------------- classification

def spectral_kind(meta):
    """Description first (what the tech named it), ImageType as fallback."""
    for text in (meta.get("series_description", ""), " ".join(meta.get("image_type", []))):
        for kind, pattern in SPECTRAL_PATTERNS:
            if pattern.search(text):
                return kind
    return ""


def orientation_class(iop, axial_min_cos=0.8):
    """'axial' | 'coronal' | 'sagittal' | 'oblique' from the slice normal."""
    if iop is None:
        return "unknown"
    row, col = _unit(iop[:3]), _unit(iop[3:])
    normal = _unit(_cross(row, col))
    ax = [abs(c) for c in normal]
    if ax[2] >= axial_min_cos:
        return "axial"
    if ax[1] >= axial_min_cos:
        return "coronal"
    if ax[0] >= axial_min_cos:
        return "sagittal"
    return "oblique"


def classify_series(entry, min_slices):
    """Return (decision, reasons, tags) for a scanned series. decision is
    'candidate' (go on to geometry) or 'rejected'."""
    meta, slices = entry["meta"], entry["slices"]
    reasons = []
    tags = []
    if meta["modality"] != "CT":
        reasons.append(f"modality={meta['modality'] or 'none'}")
    if meta["sop_class"] not in CT_SOP_CLASSES:
        reasons.append(f"sop_class={sop_class_name(meta['sop_class'])}")
    if any(t in SCOUT_TOKENS for t in meta["image_type"]):
        reasons.append("localizer")
    if reasons:
        return "rejected", reasons, tags

    iops = [s["iop"] for s in slices if s.get("iop")]
    if not iops:
        return "rejected", ["no_orientation"], tags
    classes = {orientation_class(iop) for iop in iops}
    if "axial" not in classes:
        return "rejected", ["orientation=" + "/".join(sorted(classes))], tags
    if len(classes) > 1:
        tags.append("mixed_orientation")
    n_axial = sum(1 for iop in iops if orientation_class(iop) == "axial")
    if n_axial < min_slices:
        return "rejected", [f"too_few_slices={n_axial}"], tags

    kind = spectral_kind(meta)
    if kind:
        tags.append(f"spectral={kind}")
    if "DERIVED" in meta["image_type"] and "SECONDARY" in meta["image_type"]:
        tags.append("derived_secondary")
    m = KERNEL_RE.search(meta.get("kernel") or "") or KERNEL_RE.search(meta.get("series_description", ""))
    if m:
        tags.append(f"kernel={m.group(1)}")
    return "candidate", reasons, tags


def sop_class_name(uid):
    try:
        from pydicom.uid import UID
        return UID(uid).name.replace(" ", "")
    except Exception:  # noqa: BLE001
        return uid or "none"


# --------------------------------------------------------------------------- geometry

def _stack_key(s):
    iop = tuple(round(v, 3) for v in (s.get("iop") or []))
    ps = tuple(round(v, 4) for v in (s.get("pixel_spacing") or []))
    return (iop, s.get("for_uid", ""), s.get("rows"), s.get("cols"), ps)


def split_stacks(slices):
    """Group slices into geometrically homogeneous stacks (same orientation,
    frame of reference, matrix, pixel spacing). Only axial stacks are kept."""
    groups = {}
    for s in slices:
        if orientation_class(s.get("iop")) != "axial" or not s.get("ipp"):
            continue
        groups.setdefault(_stack_key(s), []).append(s)
    return [g for _, g in sorted(groups.items(), key=lambda kv: -len(kv[1]))]


def _positions(stack):
    iop = stack[0]["iop"]
    normal = _unit(_cross(_unit(iop[:3]), _unit(iop[3:])))
    return normal, [_dot(s["ipp"], normal) for s in stack]


def _dedupe_or_split(stack, tol=0.01):
    """Handle repeated slice positions. Returns (list_of_stacks, flags)."""
    normal, pos = _positions(stack)
    order = sorted(range(len(stack)), key=lambda i: (pos[i], stack[i]["instance_number"]))
    dup = 0
    for a, b in zip(order, order[1:]):
        if abs(pos[b] - pos[a]) <= tol:
            dup += 1
    if dup == 0:
        return [[stack[i] for i in order]], []
    # Try to split into interleaved sub-stacks that are each duplicate-free.
    for key in ("acquisition_number", "temporal_position", "stack_id"):
        buckets = {}
        for s in stack:
            buckets.setdefault(s.get(key), []).append(s)
        if len(buckets) < 2:
            continue
        ok = True
        subs = []
        for _, sub in sorted(buckets.items(), key=lambda kv: str(kv[0])):
            _, p = _positions(sub)
            p_sorted = sorted(p)
            if any(abs(b - a) <= tol for a, b in zip(p_sorted, p_sorted[1:])):
                ok = False
                break
            subs.append(sorted(sub, key=lambda s: (_dot(s["ipp"], normal), s["instance_number"])))
        if ok:
            return subs, [f"split_by_{key}={len(subs)}"]
    kept = []
    last = None
    for i in order:
        if last is not None and abs(pos[i] - last) <= tol:
            continue
        kept.append(stack[i])
        last = pos[i]
    return [kept], [f"duplicate_positions={dup}"]


def analyze_stack(stack, meta, rel_tol=0.01, abs_tol=0.01):
    """Sort a stack along its normal and compute spacing statistics + flags."""
    normal, pos = _positions(stack)
    order = sorted(range(len(stack)), key=lambda i: (pos[i], stack[i]["instance_number"]))
    stack = [stack[i] for i in order]
    pos = [pos[i] for i in order]
    gaps = [b - a for a, b in zip(pos, pos[1:])]
    flags = []
    info = {"n_slices": len(stack)}
    if not gaps:
        return stack, {"n_slices": len(stack), "z_spacing": None}, ["single_slice"]
    gaps_sorted = sorted(gaps)
    median = gaps_sorted[len(gaps_sorted) // 2]
    tol = max(abs_tol, rel_tol * median)
    irregular = [g for g in gaps if abs(g - median) > tol]
    info.update({
        "z_spacing": median,
        "z_spacing_min": min(gaps),
        "z_spacing_max": max(gaps),
        "extent_z": pos[-1] - pos[0],
        "n_irregular_gaps": len(irregular),
    })
    if irregular:
        missing = 0
        unexplained = 0
        for g in irregular:
            k = round(g / median) if median > 0 else 0
            if k >= 2 and abs(g - k * median) <= tol * k:
                missing += k - 1
            else:
                unexplained += 1
        info["n_missing"] = missing
        info["n_expected"] = len(stack) + missing
        if missing:
            flags.append(f"missing_slices={missing}")
        if unexplained:
            flags.append(f"irregular_spacing={unexplained}")
    # Gantry tilt: header says so, or the stack is sheared (in-plane drift).
    if meta.get("gantry_tilt"):
        flags.append(f"gantry_tilt={meta['gantry_tilt']:g}")
    delta = [b - a for a, b in zip(stack[0]["ipp"], stack[-1]["ipp"])]
    along = _dot(delta, normal)
    inplane = [d - along * n for d, n in zip(delta, normal)]
    drift = _norm(inplane)
    info["inplane_drift_mm"] = drift
    if drift > 1.0:
        flags.append(f"sheared_stack={drift:.1f}mm")
    slopes = {(s["rescale_slope"], s["rescale_intercept"]) for s in stack}
    if len(slopes) > 1:
        flags.append("mixed_rescale")
    thick = stack[0].get("slice_thickness")
    if thick and abs(thick - median) > tol:
        info["thickness_vs_spacing"] = "overlap" if thick > median else "gap"
    return stack, info, flags


# --------------------------------------------------------------------------- volume writing

def _sitk_direction(iop, normal):
    row, col = _unit(iop[:3]), _unit(iop[3:])
    return (row[0], col[0], normal[0],
            row[1], col[1], normal[1],
            row[2], col[2], normal[2])


def _expected_geometry(stack, info):
    iop = stack[0]["iop"]
    normal = _unit(_cross(_unit(iop[:3]), _unit(iop[3:])))
    ps = stack[0]["pixel_spacing"] or [1.0, 1.0]
    dz = info.get("z_spacing") or stack[0].get("slice_thickness") or 1.0
    return {
        "origin": tuple(stack[0]["ipp"]),
        "spacing": (ps[1], ps[0], dz),      # sitk: (x=col, y=row, z)
        "direction": _sitk_direction(iop, normal),
        "size": (stack[0]["cols"], stack[0]["rows"], info.get("n_grid") or len(stack)),
    }


def _geometry_matches(img, exp, tol=0.05):
    diffs = []
    for name, got, want in (("origin", img.GetOrigin(), exp["origin"]),
                            ("spacing", img.GetSpacing(), exp["spacing"]),
                            ("direction", img.GetDirection(), exp["direction"])):
        if any(abs(g - w) > tol for g, w in zip(got, want)):
            diffs.append(name)
    if tuple(img.GetSize()) != tuple(exp["size"]):
        diffs.append("size")
    return diffs


def read_stack_sitk(stack):
    """Legacy single-frame series via SimpleITK/GDCM (handles compressed
    transfer syntaxes and per-file rescale)."""
    import SimpleITK as sitk
    reader = sitk.ImageSeriesReader()
    reader.SetFileNames([s["path"] for s in stack])
    reader.MetaDataDictionaryArrayUpdateOff()
    reader.LoadPrivateTagsOff()
    return reader.Execute()


def read_stack_numpy(stack, grid=None):
    """Assemble a volume from pydicom pixel data; used for multi-frame
    (Enhanced CT) objects, for gap filling, and as a fallback for legacy
    series.

    grid=(z_spacing, n_grid): place each slice at its true index on a
    regular grid starting at the first slice and fill missing indices by
    linear interpolation between the nearest present neighbours. Without
    grid, slices are simply stacked in order."""
    import numpy as np
    import pydicom
    import SimpleITK as sitk
    rows, cols = stack[0]["rows"], stack[0]["cols"]
    if grid:
        dz, n = grid
        normal, pos = _positions(stack)
        idx = [int(round((p - pos[0]) / dz)) for p in pos]
    else:
        n = len(stack)
        idx = list(range(n))
    vol = np.zeros((n, rows, cols), dtype=np.float32)
    present = np.zeros(n, dtype=bool)
    cache = {}
    for s, i in zip(stack, idx):
        ds = cache.get(s["path"])
        if ds is None:
            ds = pydicom.dcmread(s["path"])
            cache = {s["path"]: ds}          # one file resident at a time
        arr = ds.pixel_array
        plane = arr[s["frame"]] if (s["frame"] is not None and arr.ndim == 3) else arr
        vol[i] = plane.astype(np.float32) * s["rescale_slope"] + s["rescale_intercept"]
        present[i] = True
    pres = np.flatnonzero(present)
    for j in np.flatnonzero(~present):
        lo = pres[pres < j].max()
        hi = pres[pres > j].min()
        w = (j - lo) / float(hi - lo)
        vol[j] = (1.0 - w) * vol[lo] + w * vol[hi]
    vol = np.clip(np.rint(vol), -32768, 32767).astype(np.int16)
    return sitk.GetImageFromArray(vol)


def write_volume(stack, info, out_path, force_numpy=False):
    """Write the stack as int16 NRRD. Returns list of geometry flags."""
    import SimpleITK as sitk
    grid = None
    if info.get("n_missing"):
        # Fill gaps on the true grid rather than packing slices together.
        info["n_grid"] = info["n_expected"]
        grid = (info["z_spacing"], info["n_expected"])
        flags_extra = [f"interpolated_slices={info['n_missing']}"]
    else:
        flags_extra = []
    exp = _expected_geometry(stack, info)
    flags = list(flags_extra)
    img = None
    multiframe = any(s["frame"] is not None for s in stack)
    if not multiframe and not force_numpy and grid is None:
        try:
            img = read_stack_sitk(stack)
        except Exception as exc:  # noqa: BLE001
            log.warning("SimpleITK series read failed (%s); falling back to pydicom", exc)
            flags.append("sitk_read_failed")
    if img is None:
        img = read_stack_numpy(stack, grid=grid)
        img.SetOrigin(exp["origin"])
        img.SetSpacing(exp["spacing"])
        img.SetDirection(exp["direction"])
    else:
        mismatch = _geometry_matches(img, exp)
        if "size" in mismatch:
            raise RuntimeError(f"reader size {img.GetSize()} != expected {exp['size']}")
        if mismatch:
            flags.append("reader_geometry_adjusted=" + "+".join(mismatch))
            img.SetOrigin(exp["origin"])
            img.SetSpacing(exp["spacing"])
            img.SetDirection(exp["direction"])
    if img.GetPixelID() != sitk.sitkInt16:
        img = sitk.Clamp(img, sitk.sitkInt16, -32768, 32767)
    tmp = out_path + ".partial.nrrd"
    sitk.WriteImage(img, tmp, useCompression=True)
    os.replace(tmp, out_path)
    return flags, img.GetSize()


# --------------------------------------------------------------------------- per-case driver

def plan_case(case_id, series, min_slices, max_missing_frac=1.0):
    """Turn scanned series into a list of planned volumes / rejections.
    Pure function of the headers (no pixel IO) so --dry-run can print it."""
    studies = sorted({(e["meta"]["study_date"], e["meta"]["study_time"], e["meta"]["study_uid"])
                      for e in series.values()})
    study_index = {uid: i + 1 for i, (_, _, uid) in enumerate(studies)}
    multi_study = len(studies) > 1
    used_dirs = {}
    rows = []
    ordered = sorted(series.values(),
                     key=lambda e: (study_index[e["meta"]["study_uid"]],
                                    e["meta"]["series_number"], e["meta"]["series_uid"]))
    for entry in ordered:
        meta = entry["meta"]
        decision, reasons, tags = classify_series(entry, min_slices)
        base_row = {
            "case_id": case_id,
            "study_index": study_index[meta["study_uid"]],
            "series_number": meta["series_number"],
            "series_description": meta["series_description"],
            "series_uid": meta["series_uid"],
            "modality": meta["modality"],
            "sop_class": sop_class_name(meta["sop_class"]),
            "image_type": "\\".join(meta["image_type"]),
            "n_files": entry["n_files"],
            "n_frames": len(entry["slices"]),
            "kernel": meta["kernel"],
            "kvp": meta["kvp"] if meta["kvp"] is not None else "",
            "spectral": spectral_kind(meta),
        }
        if multi_study:
            tags = tags + ["multi_study"]
        if decision == "rejected":
            rows.append(dict(base_row, decision="rejected", reasons=";".join(reasons),
                             flags="", tags=";".join(tags), volume_id="", series_dir="",
                             n_slices="", ct_path="", _stack=None))
            continue
        stacks = split_stacks(entry["slices"])
        stack_tags = list(tags)
        if len(stacks) > 1:
            stack_tags.append(f"multi_stack={len(stacks)}")
        produced = 0
        for k, stack in enumerate(stacks):
            substacks, dflags = _dedupe_or_split(stack)
            for j, sub in enumerate(substacks):
                if len(sub) < min_slices:
                    rows.append(dict(base_row, decision="rejected",
                                     reasons=f"stack_too_few_slices={len(sub)}",
                                     flags=";".join(dflags), tags=";".join(stack_tags),
                                     volume_id="", series_dir="", n_slices=len(sub),
                                     ct_path="", _stack=None))
                    continue
                sub, info, gflags = analyze_stack(sub, meta)
                flags = dflags + gflags
                n_missing = info.get("n_missing", 0)
                if n_missing and n_missing / float(info["n_expected"]) > max_missing_frac:
                    rows.append(dict(base_row, decision="rejected",
                                     reasons=f"incomplete_series={len(sub)}/{info['n_expected']}",
                                     flags=";".join(flags), tags=";".join(stack_tags),
                                     volume_id="", series_dir="", n_slices=len(sub),
                                     ct_path="", _stack=None))
                    continue
                suffix = ""
                if len(stacks) > 1 or len(substacks) > 1:
                    suffix = f"_stk{produced + 1}"
                desc = sanitize(meta["series_description"])
                if multi_study:
                    desc = f"S{study_index[meta['study_uid']]}_{desc}"
                series_dir = f"{meta['series_number']:03d}_{desc}{suffix}"
                # Collision guard: different series with same number+description.
                n_prev = used_dirs.get(series_dir)
                if n_prev:
                    used_dirs[series_dir] = n_prev + 1
                    series_dir = f"{series_dir}_{n_prev + 1}"
                else:
                    used_dirs[series_dir] = 1
                produced += 1
                ps = sub[0].get("pixel_spacing") or ["", ""]
                rows.append(dict(
                    base_row,
                    decision="staged", reasons="", flags=";".join(flags),
                    tags=";".join(stack_tags),
                    volume_id=f"{case_id}__{series_dir}", series_dir=series_dir,
                    n_slices=len(sub), rows=sub[0]["rows"], cols=sub[0]["cols"],
                    pixel_spacing_mm=f"{ps[0]:.4g}x{ps[1]:.4g}" if ps[0] != "" else "",
                    slice_thickness_mm=sub[0].get("slice_thickness") or "",
                    z_spacing_mm=_fmt(info.get("z_spacing")),
                    z_spacing_min_mm=_fmt(info.get("z_spacing_min")),
                    z_spacing_max_mm=_fmt(info.get("z_spacing_max")),
                    extent_z_mm=_fmt(info.get("extent_z")),
                    n_voxels=sub[0]["rows"] * sub[0]["cols"] * len(sub),
                    ct_path="", _stack=sub, _info=info,
                ))
    return rows


def _fmt(v):
    return "" if v is None else f"{v:.4f}"


def stage_case(case_id, case_dir, work, min_slices, dry_run=False, force=False,
               strict=False, max_missing_frac=1.0):
    t0 = time.time()
    series, _ = scan_case(case_dir)
    rows = plan_case(case_id, series, min_slices, max_missing_frac=max_missing_frac)
    case_out = os.path.join(work, case_id)
    if dry_run:
        print_plan(rows)
        return rows
    os.makedirs(case_out, exist_ok=True)
    for row in rows:
        stack = row.pop("_stack", None)
        info = row.pop("_info", None)
        if row["decision"] != "staged":
            continue
        vol_dir = os.path.join(case_out, row["series_dir"])
        ct_path = os.path.join(vol_dir, "ct.nrrd")
        geom_path = os.path.join(vol_dir, "geometry.json")
        row["ct_path"] = ct_path
        if strict and row["flags"]:
            row["decision"] = "rejected"
            row["reasons"] = "strict:" + row["flags"]
            row["ct_path"] = ""
            log.info("%s: strict mode rejects (%s)", row["series_dir"], row["flags"])
            continue
        if os.path.isfile(ct_path) and os.path.isfile(geom_path) and not force:
            prev = _json_load(geom_path)
            if prev.get("n_source_frames") == len(stack):
                log.info("%s: exists, skip", row["series_dir"])
                row["staged_at"] = prev.get("staged_at", "")
                continue
            log.info("%s: source changed (%s → %d frames), reconverting",
                     row["series_dir"], prev.get("n_source_frames"), len(stack))
        os.makedirs(vol_dir, exist_ok=True)
        t1 = time.time()
        try:
            wflags, size = write_volume(stack, info, ct_path)
        except Exception as exc:  # noqa: BLE001
            log.error("%s: conversion FAILED: %s", row["series_dir"], exc)
            row["decision"] = "failed"
            row["reasons"] = f"conversion_error={type(exc).__name__}"
            row["ct_path"] = ""
            continue
        if wflags:
            row["flags"] = ";".join([f for f in [row["flags"]] + wflags if f])
        row["staged_at"] = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        geometry = {
            "case_id": case_id,
            "volume_id": row["volume_id"],
            "series_uid": row["series_uid"],
            "series_number": row["series_number"],
            "series_description": row["series_description"],
            "decision": row["decision"],
            "flags": row["flags"].split(";") if row["flags"] else [],
            "tags": row["tags"].split(";") if row["tags"] else [],
            "size_xyz": list(size),
            "spacing_xyz": list(_expected_geometry(stack, info)["spacing"]),
            "origin_lps": list(stack[0]["ipp"]),
            "direction": list(_expected_geometry(stack, info)["direction"]),
            "z_spacing_stats": {k: info.get(k) for k in
                                ("z_spacing", "z_spacing_min", "z_spacing_max", "extent_z",
                                 "n_irregular_gaps", "inplane_drift_mm", "thickness_vs_spacing")},
            "scanner": {k: series[row["series_uid"]]["meta"].get(k) for k in
                        ("manufacturer", "model_name", "kernel", "kvp", "reconstruction_diameter",
                         "protocol_name", "body_part", "gantry_tilt", "image_type")},
            "source_files": sorted({s["path"] for s in stack}),
            "n_source_frames": len(stack),
            "n_missing_interpolated": info.get("n_missing", 0),
            "staged_at": row["staged_at"],
            "conversion_seconds": round(time.time() - t1, 1),
        }
        with open(geom_path + ".partial", "w") as f:
            json.dump(geometry, f, indent=1)
        os.replace(geom_path + ".partial", geom_path)
        log.info("%s: wrote %s size=%s flags=%s (%.1fs)", row["series_dir"], ct_path,
                 size, row["flags"] or "-", time.time() - t1)
    write_series_csv(os.path.join(case_out, "series.csv"), rows)
    n_staged = sum(1 for r in rows if r["decision"] == "staged")
    log.info("%s: %d series → %d volumes staged in %.1fs", case_id, len(rows), n_staged,
             time.time() - t0)
    return rows


def _json_load(path):
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def write_series_csv(path, rows):
    with open(path + ".partial", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=SERIES_COLUMNS, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow({k: r.get(k, "") for k in SERIES_COLUMNS})
    os.replace(path + ".partial", path)


def print_plan(rows):
    print(f"{'ser':>5} {'description':34} {'decision':9} {'slices':>7} {'files':>8}  reasons / flags / tags")
    for r in rows:
        extra = " | ".join(x for x in (r.get("reasons"), r.get("flags"), r.get("tags")) if x)
        print(f"{r['series_number']:>5} {r['series_description'][:34]:34} {r['decision']:9} "
              f"{str(r.get('n_slices', '')):>7} {r['n_files']:>8}  {extra}")


# --------------------------------------------------------------------------- main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True, help="Root of the rclone'd DICOM tree.")
    ap.add_argument("--work", required=True, help="Pipeline output root.")
    sel = ap.add_mutually_exclusive_group(required=True)
    sel.add_argument("--case", help="E######## case id to stage.")
    sel.add_argument("--case-index", type=int, help="0-based index into --case-list.")
    sel.add_argument("--list-cases", action="store_true",
                     help="Write <work>/manifest/case_list.txt and exit.")
    ap.add_argument("--case-list", default=None,
                    help="Default: <work>/manifest/case_list.txt")
    ap.add_argument("--min-slices", type=int, default=20)
    ap.add_argument("--max-missing-frac", type=float, default=1.0,
                    help="Reject a series when more than this fraction of its slice positions "
                         "is missing. Default 1.0 = never reject: gaps are filled by linear "
                         "interpolation on the true grid and flagged (the PCCT syngo.via "
                         "exports are missing slices we cannot recover).")
    ap.add_argument("--dry-run", action="store_true", help="Classify only; no conversion.")
    ap.add_argument("--force", action="store_true", help="Re-convert existing volumes.")
    ap.add_argument("--strict", action="store_true",
                    help="Reject any series with geometry flags instead of best effort.")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    case_list = args.case_list or os.path.join(args.work, "manifest", "case_list.txt")

    if args.list_cases:
        dirs = find_case_dirs(args.input)
        os.makedirs(os.path.dirname(case_list), exist_ok=True)
        with open(case_list, "w") as f:
            for d in dirs:
                f.write(d + "\n")
        print(f"{len(dirs)} cases → {case_list}")
        return 0

    if args.case:
        matches = [d for d in find_case_dirs(args.input) if os.path.basename(d) == args.case]
        if not matches:
            log.error("no directory named %s under %s", args.case, args.input)
            return 2
        case_dir = matches[0]
    else:
        with open(case_list) as f:
            dirs = [line.strip() for line in f if line.strip()]
        if args.case_index < 0 or args.case_index >= len(dirs):
            log.error("case index %d out of range (0..%d)", args.case_index, len(dirs) - 1)
            return 2
        case_dir = dirs[args.case_index]
    case_id = os.path.basename(case_dir)
    rows = stage_case(case_id, case_dir, args.work, args.min_slices,
                      dry_run=args.dry_run, force=args.force, strict=args.strict,
                      max_missing_frac=args.max_missing_frac)
    return 0 if any(r["decision"] == "staged" for r in rows) or args.dry_run else 1


if __name__ == "__main__":
    sys.exit(main())
