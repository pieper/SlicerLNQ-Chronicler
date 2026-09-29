#!/usr/bin/env python3
"""build_cohort.py — aggregate staged volumes into the flat cohort view and
plan the GPU work.

Reads every <work>/E########/series.csv that stage_dicom.py wrote and:

  * writes <work>/manifest/series_index.csv   (every series, staged or not)
  * writes <work>/manifest/volumes.csv        (staged volumes with a ct.nrrd)
  * (re)creates the flat symlink view LNQReview / idc-batch-qc.py expect:
        <work>/cohort/nrrd/<volume_id>_0000.nrrd          -> ../../E…/…/ct.nrrd
        <work>/cohort/predictions/<model>/<volume_id>.nrrd       -> …/<model>-seg.nrrd
        <work>/cohort/predictions/<model>/<volume_id>-prob.nrrd  -> …/<model>-prob.nrrd
    (prediction links are only made for outputs that already exist;
     predict_batch.py adds its own as it goes, and --refresh-links redoes
     the sweep)
  * writes <work>/manifest/predict_tasks.tsv: one row per (model, chunk) of
    volumes still missing a SEG or probability map for that model. Row N is
    what SLURM_ARRAY_TASK_ID=N of predict.sbatch runs. Volumes are dealt into
    chunks round-robin after sorting by voxel count so chunks are balanced.

Cheap (headers + symlinks only), but run it as a job anyway to stay off the
login node when the cohort is large.

Portable: ct paths are re-derived from <work>/<case>/<series_dir>/, so after
copying the real files (E*/ , manifest/, cohort/qc/) elsewhere,
`build_cohort.py --work <copy> --models ... --refresh-links` recreates the
cohort/nrrd + cohort/predictions symlinks there. Standard library only.
"""
from __future__ import annotations

import argparse
import csv
import glob
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_registry  # noqa: E402
from stage_dicom import SERIES_COLUMNS  # noqa: E402


def read_series_rows(work):
    rows = []
    for path in sorted(glob.glob(os.path.join(work, "E*", "series.csv"))):
        with open(path, newline="") as f:
            rows.extend(csv.DictReader(f))
    return rows


def write_csv(path, rows, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path + ".partial", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        wr.writeheader()
        for r in rows:
            wr.writerow(r)
    os.replace(path + ".partial", path)


def relink(link_path, target):
    """Create/refresh a relative symlink link_path -> target."""
    os.makedirs(os.path.dirname(link_path), exist_ok=True)
    rel = os.path.relpath(target, os.path.dirname(link_path))
    if os.path.islink(link_path):
        if os.readlink(link_path) == rel:
            return False
        os.unlink(link_path)
    elif os.path.exists(link_path):
        raise RuntimeError(f"{link_path} exists and is not a symlink; refusing to replace")
    os.symlink(rel, link_path)
    return True


def output_files(volume_row, model):
    """role -> path of every file `model` writes for this volume (model_registry)."""
    return model_registry.output_files(os.path.dirname(volume_row["ct_path"]), model)


def output_paths(volume_row, model):
    """(seg, prob) for callers that only need the classic pair."""
    files = output_files(volume_row, model)
    return files["seg"], files.get("prob")


def cohort_link_paths(work, volume_id, model):
    links = model_registry.cohort_link_files(work, volume_id, model)
    return links["seg"], links.get("prob")


def link_volume(work, volume_row, models):
    """Symlinks for one volume: CT always, every model output that exists."""
    vid = volume_row["volume_id"]
    n = 0
    n += relink(os.path.join(work, "cohort", "nrrd", f"{vid}_0000.nrrd"), volume_row["ct_path"])
    for model in models:
        files = output_files(volume_row, model)
        links = model_registry.cohort_link_files(work, vid, model)
        for role, path in files.items():
            if os.path.isfile(path) and role in links:
                n += relink(links[role], path)
    return n


def pending_volumes(volumes, model):
    """Volumes still missing at least one of the model's outputs."""
    out = []
    for v in volumes:
        if not all(os.path.isfile(p) for p in output_files(v, model).values()):
            out.append(v)
    return out


def balanced_chunks(volumes, chunk_size):
    """Deal volumes (largest first) round-robin into ceil(n/chunk_size) chunks."""
    if not volumes:
        return []
    ordered = sorted(volumes, key=lambda v: -int(v.get("n_voxels") or 0))
    n_chunks = max(1, math.ceil(len(ordered) / chunk_size))
    chunks = [[] for _ in range(n_chunks)]
    for i, v in enumerate(ordered):
        chunks[i % n_chunks].append(v)
    return chunks


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--models", required=True,
                    help="Space- or comma-separated model names (lnq-segmenter registry or models.json).")
    ap.add_argument("--chunk-size", type=int, default=8)
    ap.add_argument("--refresh-links", action="store_true",
                    help="Only redo the symlink sweep + manifests; don't rewrite predict_tasks.tsv.")
    ap.add_argument("--volume-ids", default=None,
                    help="Comma-separated subset of volume ids to plan (bench / smoke tests).")
    args = ap.parse_args(argv)

    models = [m for m in args.models.replace(",", " ").split() if m]
    manifest = os.path.join(args.work, "manifest")
    rows = read_series_rows(args.work)
    # The recorded ct_path is absolute on the machine that staged the volume.
    # Re-derive it from <work>/<case>/<series_dir>/ct.nrrd so a copy of the
    # tree (e.g. pulled to a laptop) can rebuild the cohort view in place.
    for r in rows:
        if r.get("decision") == "staged" and r.get("series_dir"):
            local = os.path.join(args.work, r["case_id"], r["series_dir"], "ct.nrrd")
            if os.path.isfile(local):
                r["ct_path"] = local
    write_csv(os.path.join(manifest, "series_index.csv"), rows, SERIES_COLUMNS)

    volumes = [r for r in rows if r["decision"] == "staged" and os.path.isfile(r.get("ct_path", ""))]
    missing_ct = [r for r in rows if r["decision"] == "staged" and not os.path.isfile(r.get("ct_path", ""))]
    write_csv(os.path.join(manifest, "volumes.csv"), volumes, SERIES_COLUMNS)
    print(f"series: {len(rows)}  staged volumes: {len(volumes)}  "
          f"rejected: {sum(1 for r in rows if r['decision'] == 'rejected')}  "
          f"failed: {sum(1 for r in rows if r['decision'] == 'failed')}"
          + (f"  staged-but-missing-ct: {len(missing_ct)}" if missing_ct else ""))

    n_links = 0
    for v in volumes:
        n_links += link_volume(args.work, v, models)
    for model in models:
        os.makedirs(os.path.join(args.work, "cohort", "predictions", model), exist_ok=True)
        # External models: keep dataset.json with the manifests so stats /
        # dashboards on another machine can resolve label names without the
        # model folder.
        folder = model_registry.model_folder(model, os.environ.get("MODELS_CACHE", ""))
        src = os.path.join(folder, "dataset.json") if folder else None
        if src and os.path.isfile(src):
            dst_dir = os.path.join(manifest, "models", model)
            os.makedirs(dst_dir, exist_ok=True)
            with open(src) as f_in, open(os.path.join(dst_dir, "dataset.json"), "w") as f_out:
                f_out.write(f_in.read())
    print(f"cohort links created/updated: {n_links}")

    if args.refresh_links:
        return 0

    if args.volume_ids:
        wanted = set(args.volume_ids.split(","))
        volumes = [v for v in volumes if v["volume_id"] in wanted]

    tasks = []
    chunk_dir = os.path.join(manifest, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)
    for old in glob.glob(os.path.join(chunk_dir, "*.txt")):
        os.remove(old)
    for model in models:
        pending = pending_volumes(volumes, model)
        for k, chunk in enumerate(balanced_chunks(pending, args.chunk_size)):
            chunk_file = os.path.join(chunk_dir, f"{model}-{k:04d}.txt")
            with open(chunk_file, "w") as f:
                for v in chunk:
                    f.write(f"{v['volume_id']}\t{v['ct_path']}\n")
            tasks.append({"task": len(tasks), "model": model, "chunk": k,
                          "n_volumes": len(chunk), "chunk_file": chunk_file})
        print(f"{model}: {len(pending)} volumes pending → "
              f"{sum(1 for t in tasks if t['model'] == model)} chunks")
    tasks_path = os.path.join(manifest, "predict_tasks.tsv")
    with open(tasks_path + ".partial", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=["task", "model", "chunk", "n_volumes", "chunk_file"],
                            delimiter="\t")
        wr.writeheader()
        for t in tasks:
            wr.writerow(t)
    os.replace(tasks_path + ".partial", tasks_path)
    print(f"{len(tasks)} predict tasks → {tasks_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
