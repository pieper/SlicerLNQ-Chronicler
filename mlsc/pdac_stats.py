#!/usr/bin/env python3
"""pdac_stats.py — per-series lymph-node segment statistics for the PDAC
photon-counting CT cohort, feeding the PDACReview Slicer dashboard.

For every staged volume (manifest/volumes.csv) and every model with a
<model>-seg.nrrd next to the CT, computes:

  * total segmented volume (mL) and voxel count
  * connected components ("nodes", 26-connected) with per-node volume (mL),
    centroid (LPS, mm), equivalent-ellipsoid short / long axis (mm) and
    bounding box; nodes smaller than --min-node-ml are counted separately as
    specks and excluded from node counts
  * node count, largest node, sum of node volumes

and joins series metadata (patient index + day offset from
manifest/inventory.csv when present, phase parsed from the series
description, spectral kind, kernel, thickness, geometry flags, fraction of
interpolated slices) and the probability-map numbers already in
cohort/qc/<model>/qc.csv.

Per-study "agreement across series" is volume-only: for each model the
median / min / max / coefficient of variation of total mL and node count
across the study's series.

Outputs (under <root>/manifest/): pdac_stats.json (what the dashboard
reads), pdac_stats.csv (one row per volume x model), pdac_nodes.csv (one row
per node). Per-volume results are cached in <vol_dir>/pdac-stats.json and
recomputed only when a SEG is newer or --force is given.

Runs anywhere SimpleITK + numpy are importable: Slicer's Python (the
PDACReview module calls it in-process), or a venv. No PHI is read or
written: only the NRRDs and the manifests the pipeline produced.

    pdac_stats.py --root /Volumes/12T/PHI/PDAC [--min-node-ml 0.05] [--force]
"""
import argparse
import csv
import datetime
import json
import os
import re
import statistics
import sys
import time

PHASE_RE = re.compile(r"(\d+)\s*(sec|min)\b", re.I)
DEFAULT_MIN_NODE_ML = 0.05          # ~4.6 mm sphere; below that is a speck


def parse_phase(description):
    m = PHASE_RE.search(description or "")
    if not m:
        return "", 0
    n, unit = int(m.group(1)), m.group(2).lower()
    seconds = n * (60 if unit == "min" else 1)
    return f"{n} {unit}", seconds


def missing_fraction(flags, n_slices):
    m = re.search(r"missing_slices=(\d+)", flags or "")
    try:
        n = int(n_slices)
    except (TypeError, ValueError):
        return 0.0
    if not m or not n:
        return 0.0
    return round(int(m.group(1)) / float(n), 3)


def read_csv(path):
    if not os.path.isfile(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def discover_models(root):
    pred = os.path.join(root, "cohort", "predictions")
    if not os.path.isdir(pred):
        return []
    return sorted(d for d in os.listdir(pred) if os.path.isdir(os.path.join(pred, d)))


def seg_stats(seg_path, min_node_ml):
    """Segment statistics for one binary SEG NRRD via SimpleITK."""
    import SimpleITK as sitk
    img = sitk.ReadImage(seg_path)
    sx, sy, sz = img.GetSpacing()
    voxel_ml = sx * sy * sz / 1000.0
    binary = sitk.Cast(img > 0, sitk.sitkUInt8)
    stats = sitk.StatisticsImageFilter()
    stats.Execute(binary)
    n_voxels = int(round(stats.GetSum()))
    out = {"total_ml": round(n_voxels * voxel_ml, 3), "n_voxels": n_voxels,
           "voxel_ml": voxel_ml, "n_components": 0, "n_specks": 0, "n_nodes": 0,
           "nodes_ml": 0.0, "largest_ml": 0.0, "nodes": []}
    if n_voxels == 0:
        return out
    cc = sitk.ConnectedComponent(binary, True)
    shape = sitk.LabelShapeStatisticsImageFilter()
    shape.ComputeOrientedBoundingBoxOff()
    shape.Execute(cc)
    nodes = []
    n_specks = 0
    for label in shape.GetLabels():
        ml = shape.GetPhysicalSize(label) / 1000.0
        if ml < min_node_ml:
            n_specks += 1
            continue
        d = sorted(shape.GetEquivalentEllipsoidDiameter(label))
        bb = shape.GetBoundingBox(label)          # (x, y, z, sx, sy, sz) in voxels
        cx, cy, cz = shape.GetCentroid(label)
        nodes.append({"label": int(label), "ml": round(ml, 3),
                      "short_mm": round(d[0], 1), "mid_mm": round(d[1], 1), "long_mm": round(d[2], 1),
                      "centroid_lps": [round(cx, 1), round(cy, 1), round(cz, 1)],
                      "bbox_ijk": [int(v) for v in bb]})
    nodes.sort(key=lambda n: -n["ml"])
    out.update({"n_components": len(shape.GetLabels()), "n_specks": n_specks,
                "n_nodes": len(nodes), "nodes_ml": round(sum(n["ml"] for n in nodes), 3),
                "largest_ml": nodes[0]["ml"] if nodes else 0.0, "nodes": nodes})
    return out


def volume_record(root, vrow, models, min_node_ml, force, patients, qc_rows):
    """Compute (or load cached) stats for one staged volume across models."""
    ct_path = os.path.join(root, vrow["case_id"], vrow["series_dir"], "ct.nrrd")
    vol_dir = os.path.dirname(ct_path)
    cache_path = os.path.join(vol_dir, "pdac-stats.json")
    cached = {}
    if not force and os.path.isfile(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
        except (OSError, json.JSONDecodeError):
            cached = {}
        if cached.get("min_node_ml") != min_node_ml:
            cached = {}
    per_model = dict(cached.get("models", {}))
    changed = False
    for m in models:
        seg = os.path.join(vol_dir, f"{m}-seg.nrrd")
        prob = os.path.join(vol_dir, f"{m}-prob.nrrd")
        if not os.path.isfile(seg):
            per_model.pop(m, None)
            continue
        seg_mtime = os.path.getmtime(seg)
        entry = per_model.get(m)
        if entry and entry.get("seg_mtime") == seg_mtime:
            continue
        entry = seg_stats(seg, min_node_ml)
        entry["seg_mtime"] = seg_mtime
        entry["seg_path"] = seg
        entry["prob_path"] = prob if os.path.isfile(prob) else None
        per_model[m] = entry
        changed = True
    if changed or not cached:
        with open(cache_path + ".partial", "w") as f:
            json.dump({"volume_id": vrow["volume_id"], "min_node_ml": min_node_ml,
                       "models": per_model,
                       "computed_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}, f)
        os.replace(cache_path + ".partial", cache_path)
    # join qc.csv probability numbers (cheap, keeps the prob map out of memory)
    for m, entry in per_model.items():
        q = qc_rows.get(m, {}).get(vrow["volume_id"])
        if q:
            entry["qc"] = {k: q.get(k, "") for k in
                           ("pred_volume_mL_p0.5", "pred_volume_mL_p0.001", "prob_max", "prob_p99")}
    phase, phase_s = parse_phase(vrow["series_description"])
    pinfo = patients.get(vrow["case_id"], {})
    return {
        "volume_id": vrow["volume_id"], "case_id": vrow["case_id"],
        "patient": pinfo.get("patient", vrow["case_id"]),
        "day": pinfo.get("day", 0),
        "series_number": int(vrow["series_number"] or 0),
        "series_description": vrow["series_description"], "series_dir": vrow["series_dir"],
        "phase": phase, "phase_seconds": phase_s,
        "spectral": vrow.get("spectral", "") or "conventional",
        "kernel": vrow.get("kernel", ""),
        "slice_thickness_mm": float(vrow["slice_thickness_mm"] or 0),
        "z_spacing_mm": float(vrow["z_spacing_mm"] or 0),
        "n_slices": int(vrow["n_slices"] or 0),
        "matrix": int(vrow["rows"] or 0),
        "flags": vrow.get("flags", ""),
        "missing_frac": missing_fraction(vrow.get("flags", ""), vrow.get("n_slices")),
        "ct_path": ct_path,
        "models": per_model,
    }


def cv(values):
    if len(values) < 2:
        return 0.0
    mean = statistics.fmean(values)
    return round(statistics.pstdev(values) / mean, 3) if mean else 0.0


def study_summary(case_id, vols, models):
    out = {"case_id": case_id, "patient": vols[0]["patient"], "day": vols[0]["day"],
           "n_series": len(vols), "models": {}}
    for m in models:
        mls = [v["models"][m]["total_ml"] for v in vols if m in v["models"]]
        nodes = [v["models"][m]["n_nodes"] for v in vols if m in v["models"]]
        if not mls:
            continue
        out["models"][m] = {
            "n_series": len(mls),
            "median_ml": round(statistics.median(mls), 2), "min_ml": min(mls), "max_ml": max(mls),
            "cv_ml": cv(mls),
            "median_nodes": statistics.median(nodes), "min_nodes": min(nodes), "max_nodes": max(nodes),
            "cv_nodes": cv(nodes),
        }
    return out


def compute(root, models=None, min_node_ml=DEFAULT_MIN_NODE_ML, force=False, limit=None,
            progress=None):
    """Compute everything and write the manifest outputs. Returns the JSON dict."""
    t0 = time.time()
    models = models or discover_models(root)
    volumes = [r for r in read_csv(os.path.join(root, "manifest", "volumes.csv"))]
    if limit:
        volumes = volumes[:limit]
    inventory = read_csv(os.path.join(root, "manifest", "inventory.csv"))
    patients = {r["case_id"]: {"patient": r["patient"],
                               "day": int(r["days_since_patient_first_study"] or 0)}
                for r in inventory}
    qc_rows = {}
    for m in models:
        rows = read_csv(os.path.join(root, "cohort", "qc", m, "qc.csv"))
        qc_rows[m] = {r["case_id"]: r for r in rows}

    records = []
    for i, vrow in enumerate(volumes, 1):
        if progress:
            progress(i, len(volumes), vrow["volume_id"])
        try:
            records.append(volume_record(root, vrow, models, min_node_ml, force, patients, qc_rows))
        except Exception as exc:  # noqa: BLE001 — one bad volume shouldn't stop the report
            print(f"WARN {vrow['volume_id']}: {type(exc).__name__}: {exc}", file=sys.stderr)
    records.sort(key=lambda r: (r["patient"], r["day"], r["case_id"], r["series_number"]))

    by_case = {}
    for r in records:
        by_case.setdefault(r["case_id"], []).append(r)
    studies = [study_summary(c, v, models) for c, v in by_case.items()]
    studies.sort(key=lambda s: (s["patient"], s["day"], s["case_id"]))

    result = {"generated_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
              "root": root, "models": models, "min_node_ml": min_node_ml,
              "n_patients": len({r["patient"] for r in records}),
              "n_studies": len(studies), "n_volumes": len(records),
              "studies": studies, "volumes": records,
              "seconds": round(time.time() - t0, 1)}
    write_outputs(root, result)
    return result


def write_outputs(root, result):
    manifest = os.path.join(root, "manifest")
    os.makedirs(manifest, exist_ok=True)
    path = os.path.join(manifest, "pdac_stats.json")
    with open(path + ".partial", "w") as f:
        json.dump(result, f, indent=1)
    os.replace(path + ".partial", path)

    cols = ["patient", "day", "case_id", "volume_id", "series_number", "series_description", "phase",
            "spectral", "kernel", "slice_thickness_mm", "n_slices", "missing_frac", "flags", "model",
            "total_ml", "n_nodes", "n_specks", "nodes_ml", "largest_ml",
            "qc_pred_volume_mL_p0.001", "qc_prob_max"]
    with open(os.path.join(manifest, "pdac_stats.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for r in result["volumes"]:
            for m, e in r["models"].items():
                q = e.get("qc", {})
                wr.writerow([r["patient"], r["day"], r["case_id"], r["volume_id"], r["series_number"],
                             r["series_description"], r["phase"], r["spectral"], r["kernel"],
                             r["slice_thickness_mm"], r["n_slices"], r["missing_frac"], r["flags"], m,
                             e["total_ml"], e["n_nodes"], e["n_specks"], e["nodes_ml"], e["largest_ml"],
                             q.get("pred_volume_mL_p0.001", ""), q.get("prob_max", "")])
    with open(os.path.join(manifest, "pdac_nodes.csv"), "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["patient", "case_id", "volume_id", "model", "node", "ml", "short_mm", "long_mm",
                     "centroid_L", "centroid_P", "centroid_S"])
        for r in result["volumes"]:
            for m, e in r["models"].items():
                for k, n in enumerate(e["nodes"], 1):
                    wr.writerow([r["patient"], r["case_id"], r["volume_id"], m, k, n["ml"],
                                 n["short_mm"], n["long_mm"], *n["centroid_lps"]])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="Pipeline tree, e.g. /Volumes/12T/PHI/PDAC")
    ap.add_argument("--models", default=None, help="Space/comma-separated; default: all under cohort/predictions")
    ap.add_argument("--min-node-ml", type=float, default=DEFAULT_MIN_NODE_ML)
    ap.add_argument("--force", action="store_true", help="Ignore per-volume caches.")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args(argv)
    models = args.models.replace(",", " ").split() if args.models else None

    def progress(i, n, vid):
        print(f"[{i}/{n}] {vid}", flush=True)

    result = compute(args.root, models, args.min_node_ml, args.force, args.limit, progress)
    print(f"{result['n_patients']} patients, {result['n_studies']} studies, {result['n_volumes']} volumes, "
          f"models {result['models']} → {os.path.join(args.root, 'manifest', 'pdac_stats.json')} "
          f"({result['seconds']}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
