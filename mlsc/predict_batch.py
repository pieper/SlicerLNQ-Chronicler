#!/usr/bin/env python3
"""predict_batch.py — GPU worker: one model over a chunk of staged volumes,
model loaded once, probability maps kept.

Two model kinds (see model_registry.py / models.json):
  * lnq-segmenter registry models — binary lymph-node SEG + one probability
    map via nnU-Net's stock export (`<model>-seg.nrrd`, `<model>-prob.nrrd`).
  * external nnU-Net v2 model folders (e.g. pants-v1, 28 labels, 1 mm iso):
    the volume is written as a temporary .nii.gz so nnU-Net's own reader
    reorients it exactly as in training, logits are computed at the model's
    working resolution, converted to labels there, and only the label map and
    the classes of interest (lesion, pancreas) are resampled back onto the CT
    grid — nnU-Net's stock export would resample all 28 channels to the
    original grid (~40 GB for the thin-slice series). Output is a Slicer
    segmentation NRRD (`<model>.seg.nrrd`, named/colored segments) plus
    `<model>-prob.nrrd` and `<model>-<class>-prob.nrrd`.

Normal use is one Slurm array task per row of <work>/manifest/predict_tasks.tsv
(written by build_cohort.py):

    predict_batch.py --work /vast/lnq/pdac-processing --task-index $SLURM_ARRAY_TASK_ID

Bench use (bench.sbatch) runs an explicit model + volume list and writes a
JSON summary that bench_report.py compares across partitions:

    predict_batch.py --work ... --model mediastinal-v1 \\
        --volumes E1__002_Body_1-0_Br40,E2__005_... --bench-out manifest/bench/rtx6000-mediastinal-v1.json

Per volume:
  * nnUNetPredictor.predict_from_files(save_probabilities=True) into the
    node-local temp dir ($TMPDIR = /scratch/$SLURM_JOB_ID on MLSC) — the
    softmax .npz for a PCCT volume is multi-GB and must not land on VAST.
  * lnq_segmenter.predict._write_probability_map turns the .npz into a
    float32 foreground-probability NRRD on the CT grid (same helper the
    Slicer module uses, so the maps are identical to what LymphNodeQuantifier
    would produce).
  * outputs moved to <vol_dir>/<model>-seg.nrrd and <model>-prob.nrrd, and the
    flat cohort/predictions/<model>/ symlinks are created.
  * skip-if-both-exist, per-volume try/except, one JSON line per volume with
    timing + peak GPU / RSS in <work>/logs/predict-<model>-<chunk>.jsonl.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import io
import json
import logging
import os
import platform
import resource
import shutil
import socket
import sys
import tempfile
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import model_registry  # noqa: E402
from build_cohort import relink  # noqa: E402

log = logging.getLogger("predict_batch")


def _set_nnunet_env(base):
    """nnunetv2 warns loudly (and some code paths fail) without these."""
    for name in ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results"):
        path = os.environ.setdefault(name, os.path.join(base, name.replace("nnUNet_", "")))
        os.makedirs(path, exist_ok=True)


def _rss_mb():
    scale = 1.0 / 1024 if platform.system() != "Darwin" else 1.0 / (1024 * 1024)
    me = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale
    kids = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss * scale
    return round(me, 1), round(kids, 1)


def _register_trainer_shims(shims):
    """Make a checkpoint's custom trainer name resolvable without installing
    the fork: define `Name(Base)` with no overrides and plant it on the
    nnunetv2 trainer module so nnU-Net's recursive class lookup finds it.
    Only valid when the custom trainer changes nothing inference-relevant
    (true for PanTS: it only overrides get_dataloaders)."""
    import nnunetv2
    import nnunetv2.training.nnUNetTrainer.nnUNetTrainer as base_mod
    from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
    folder = os.path.join(nnunetv2.__path__[0], "training", "nnUNetTrainer")
    for name, base_name in (shims or {}).items():
        if hasattr(base_mod, name):
            continue
        base = recursive_find_python_class(folder, base_name, "nnunetv2.training.nnUNetTrainer")
        if base is None:
            raise RuntimeError(f"trainer shim base {base_name!r} not found in nnunetv2")
        setattr(base_mod, name, type(name, (base,), {"__module__": base_mod.__name__}))
        log.info("registered trainer shim %s(%s)", name, base_name)


def load_folder_predictor(model_name, spec, device="cuda", folds=None):
    """External nnU-Net v2 model folder (dataset.json + plans.json + fold_*/)."""
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
    cache = os.environ.get("LNQ_SEGMENTER_CACHE") or os.path.expanduser("~/.cache/lnq-segmenter")
    folder = os.environ.get("MLSC_MODEL_DIR_" + model_name.upper().replace("-", "_")) \
        or model_registry.model_folder(model_name, os.environ.get("MODELS_CACHE", cache))
    use_folds = tuple(folds) if folds else tuple(spec.get("folds", ["all"]))
    checkpoint = spec.get("checkpoint", "checkpoint_final.pth")
    for f in use_folds:
        ck = os.path.join(folder, f"fold_{f}", checkpoint)
        if not os.path.isfile(ck):
            raise FileNotFoundError(f"{ck} missing — run setup-env (hf download {spec.get('hf_repo')})")
    _register_trainer_shims(spec.get("trainer_shims"))
    t0 = time.time()
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=True, device=torch.device(device),
        verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(folder, use_folds=use_folds, checkpoint_name=checkpoint)
    predictor._mlsc_model_folder = folder
    entry = {"name": model_name, "version": spec.get("version", "hf:" + spec.get("hf_repo", "?")),
             "folds": list(use_folds), "folder": folder}
    log.info("loaded %s (%s) folds=%s trainer=%s from %s in %.1fs", model_name, spec["kind"],
             list(use_folds), predictor.trainer_name, folder, time.time() - t0)
    return predictor, entry


def load_predictor(model_name, device="cuda", folds=None):
    spec = model_registry.spec(model_name)
    if spec is not None:
        return load_folder_predictor(model_name, spec, device=device, folds=folds)
    from lnq_segmenter import registry, cache as _cache
    from lnq_segmenter.predict import _ensure_nnunet_layout
    import torch
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    entry = registry.get_model(model_name)
    if not _cache.is_complete(entry["name"], entry["version"], entry):
        raise FileNotFoundError(
            f"weights for {entry['name']}@{entry['version']} missing under "
            f"{_cache.cache_root()} — run setup-env (lnq-segmenter download)")
    bundle = _cache.bundle_dir(entry["name"], entry["version"])
    plans_dir = os.path.join(bundle, entry["plans_subdir"])
    _ensure_nnunet_layout(bundle, plans_dir)
    use_folds = tuple(folds) if folds else tuple(entry["folds"])
    t0 = time.time()
    predictor = nnUNetPredictor(
        tile_step_size=0.5, use_gaussian=True, use_mirroring=True,
        perform_everything_on_device=True, device=torch.device(device),
        verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(
        plans_dir, use_folds=use_folds, checkpoint_name="checkpoint_final.pth")
    log.info("loaded %s@%s folds=%s from %s in %.1fs", entry["name"], entry["version"],
             list(use_folds), plans_dir, time.time() - t0)
    return predictor, entry


def _move(src, dst):
    """Atomic-ish move across filesystems: copy to <dst>.partial then rename."""
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".partial"
    shutil.move(src, tmp)
    os.replace(tmp, dst)


def predict_volume(predictor, ct_path, seg_path, prob_path, tmp_root, n_procs):
    """Run one volume through the loaded predictor. Returns dict of metrics."""
    import torch
    from lnq_segmenter.predict import _write_probability_map

    tmp_dir = tempfile.mkdtemp(prefix="lnq-", dir=tmp_root)
    file_ending = (predictor.dataset_json or {}).get("file_ending", ".nrrd")
    stem = os.path.join(tmp_dir, "pred")
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    captured = io.StringIO()
    t0 = time.time()
    try:
        with contextlib.redirect_stdout(captured):
            predictor.predict_from_files(
                [[ct_path]], [stem], save_probabilities=True, overwrite=True,
                num_processes_preprocessing=n_procs,
                num_processes_segmentation_export=n_procs)
        t_pred = time.time() - t0
        text = captured.getvalue()
        cpu_fallback = ("Moving results to CPU" in text) or ("unsuccessful" in text)
        produced = stem + file_ending
        if not os.path.isfile(produced):
            raise RuntimeError(f"nnU-Net produced no {produced}; stdout was:\n{text[-2000:]}")
        prob_tmp = os.path.join(tmp_dir, "prob" + file_ending)
        _write_probability_map(stem, ct_path, prob_tmp)
        _move(produced, seg_path)
        _move(prob_tmp, prob_path)
        gpu_alloc = gpu_reserved = None
        if torch.cuda.is_available():
            gpu_alloc = round(torch.cuda.max_memory_allocated() / 2**20)
            gpu_reserved = round(torch.cuda.max_memory_reserved() / 2**20)
        return {"predict_seconds": round(t_pred, 1),
                "total_seconds": round(time.time() - t0, 1),
                "gpu_peak_alloc_mb": gpu_alloc, "gpu_peak_reserved_mb": gpu_reserved,
                "cpu_fallback": cpu_fallback}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _channel_index(label_manager, values):
    """Index of the output channel for a class given its label values
    (region tuple or single label)."""
    wanted = tuple(sorted(int(v) for v in values))
    if label_manager.has_regions:
        for i, r in enumerate(label_manager.foreground_regions):
            rr = tuple(sorted(r)) if isinstance(r, (tuple, list)) else (int(r),)
            if rr == wanted:
                return i
        raise KeyError(f"no output region with labels {wanted}")
    if len(wanted) != 1:
        raise KeyError("softmax model: a class must be a single label")
    return list(label_manager.all_labels).index(wanted[0])


def predict_volume_multiclass(predictor, model, spec, ct_path, out_files, tmp_root):
    """External multi-label model: preprocess with nnU-Net's own reader
    (reorientation!), predict logits at the working resolution, convert to
    labels there, resample label map + selected probabilities to the CT grid."""
    import numpy as np
    import torch
    import SimpleITK as sitk
    import nibabel as nib

    tmp_dir = tempfile.mkdtemp(prefix="mlsc-", dir=tmp_root)
    try:
        t0 = time.time()
        ct = sitk.ReadImage(ct_path)
        nii = os.path.join(tmp_dir, "img_0000.nii.gz")
        sitk.WriteImage(ct, nii)
        pm, cm, dj = predictor.plans_manager, predictor.configuration_manager, predictor.dataset_json
        if list(pm.transpose_forward) != [0, 1, 2]:
            raise RuntimeError(f"transpose_forward {pm.transpose_forward} not supported")
        preprocessor = cm.preprocessor_class(verbose=False)
        data, _, props = preprocessor.run_case([nii], None, pm, cm, dj)
        t_pre = time.time() - t0
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        t1 = time.time()
        logits = predictor.predict_logits_from_preprocessed_data(torch.from_numpy(data))   # (C, z, y, x), cpu
        t_pred = time.time() - t1
        del data
        lm = predictor.label_manager
        # labels at the working resolution, channel by channel (no full softmax array)
        t2 = time.time()
        seg = np.zeros(tuple(logits.shape[1:]), dtype=np.uint8)
        if lm.has_regions:
            for i, c in enumerate(lm.regions_class_order):
                seg[(logits[i] > 0).numpy()] = int(c)          # sigmoid(x) > 0.5  <=>  x > 0
        else:
            probs = lm.apply_inference_nonlin(logits)
            seg = lm.convert_probabilities_to_segmentation(probs).numpy().astype(np.uint8)
            del probs
        dataset_labels = model_registry.read_dataset_labels(entry_folder(predictor))
        prob_arrays = {}
        for cls, suffix in spec["outputs"]["prob"].items():
            idx = _channel_index(lm, dataset_labels[cls])
            ch = logits[idx].float()
            prob_arrays[cls] = (torch.sigmoid(ch) if lm.has_regions else ch).numpy().astype(np.float32)
        del logits
        # geometry: nnU-Net array (z,y,x) of the cropped + resampled reoriented image
        affine = np.asarray(props["nibabel_stuff"]["reoriented_affine"], dtype=float)
        bbox = props["bbox_used_for_cropping"]
        shape_c = props["shape_after_cropping_and_before_resampling"]
        M = np.eye(4)
        for a in range(3):                      # a: nnU-Net axis (0=z,1=y,2=x) -> nibabel axis 2-a
            ratio = float(shape_c[a]) / float(seg.shape[a])
            na = 2 - a
            M[na, na] = ratio
            M[na, 3] = float(bbox[a][0]) - 0.5 + 0.5 * ratio     # align_corners=False / grid_mode=True
        affine_new = affine @ M

        def to_ct_grid(arr, interpolator, pixel_type):
            path = os.path.join(tmp_dir, "tmp.nii.gz")
            nib.save(nib.Nifti1Image(np.ascontiguousarray(arr.transpose(2, 1, 0)), affine_new), path)
            img = sitk.ReadImage(path)
            return sitk.Resample(img, ct, sitk.Transform(), interpolator, 0.0, pixel_type)

        seg_ct = to_ct_grid(seg, sitk.sitkNearestNeighbor, sitk.sitkUInt8)
        names = model_registry.label_value_names(dataset_labels)
        colors = spec.get("segment_colors", {})
        seg_tmp = os.path.join(tmp_dir, "seg.seg.nrrd")
        n_segments = model_registry.write_slicer_seg_nrrd(seg_ct, seg_tmp, names, colors)
        _move(seg_tmp, out_files["seg"])
        counts = {}
        seg_arr = sitk.GetArrayViewFromImage(seg_ct)
        for cls in spec["outputs"]["prob"]:
            counts[cls] = int(np.isin(seg_arr, dataset_labels[cls]).sum())
        for cls, suffix in spec["outputs"]["prob"].items():
            role = "prob" if suffix == "-prob.nrrd" else suffix[1:-len(".nrrd")]
            img = to_ct_grid(prob_arrays[cls], sitk.sitkLinear, sitk.sitkFloat32)
            p_tmp = os.path.join(tmp_dir, role + ".nrrd")
            sitk.WriteImage(img, p_tmp, useCompression=True)
            _move(p_tmp, out_files[role])
        t_export = time.time() - t2
        gpu_alloc = gpu_reserved = None
        if torch.cuda.is_available():
            gpu_alloc = round(torch.cuda.max_memory_allocated() / 2**20)
            gpu_reserved = round(torch.cuda.max_memory_reserved() / 2**20)
        return {"predict_seconds": round(t_pred, 1), "preprocess_seconds": round(t_pre, 1),
                "export_seconds": round(t_export, 1), "total_seconds": round(time.time() - t0, 1),
                "working_shape": list(seg.shape), "n_segments": n_segments,
                "class_voxels": counts,
                "gpu_peak_alloc_mb": gpu_alloc, "gpu_peak_reserved_mb": gpu_reserved,
                "cpu_fallback": False}
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def entry_folder(predictor):
    """Model folder the predictor was initialized from (stashed by load_folder_predictor)."""
    return getattr(predictor, "_mlsc_model_folder")


def _read_tasks(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _read_chunk(path):
    out = []
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            vid, ct = line.split("\t")
            out.append((vid, ct))
    return out


def _volumes_by_id(work):
    path = os.path.join(work, "manifest", "volumes.csv")
    with open(path, newline="") as f:
        return {r["volume_id"]: r for r in csv.DictReader(f)}


def run(work, model, volumes, device, n_procs, tmp_root, jsonl_path, bench_out=None,
        skip_existing=True, folds=None):
    os.makedirs(os.path.dirname(jsonl_path), exist_ok=True)
    host = socket.gethostname()
    job = {k: os.environ.get(k) for k in ("SLURM_JOB_ID", "SLURM_ARRAY_JOB_ID",
                                          "SLURM_ARRAY_TASK_ID", "SLURM_JOB_PARTITION")}
    gpu_name = None
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
    except Exception:  # noqa: BLE001
        pass
    log.info("host=%s gpu=%s job=%s model=%s volumes=%d device=%s tmp=%s",
             host, gpu_name, job, model, len(volumes), device, tmp_root)

    t_load = time.time()
    predictor, entry = load_predictor(model, device=device, folds=folds)
    load_seconds = round(time.time() - t_load, 1)

    spec = model_registry.spec(model)
    records = []
    n_ok = n_skip = n_fail = 0
    with open(jsonl_path, "a") as jf:
        for i, (vid, ct_path) in enumerate(volumes, 1):
            vol_dir = os.path.dirname(ct_path)
            out_files = model_registry.output_files(vol_dir, model)
            seg_path, prob_path = out_files["seg"], out_files.get("prob")
            rec = {"ts": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                   "host": host, "gpu": gpu_name, "partition": job.get("SLURM_JOB_PARTITION"),
                   "job_id": job.get("SLURM_JOB_ID"), "model": model,
                   "model_version": entry["version"], "volume_id": vid, "ct_path": ct_path}
            if skip_existing and all(os.path.isfile(p) for p in out_files.values()):
                rec["status"] = "skipped"
                n_skip += 1
                log.info("[%d/%d] %s: skip (outputs exist)", i, len(volumes), vid)
            elif not os.path.isfile(ct_path):
                rec.update(status="failed", error="missing ct.nrrd")
                n_fail += 1
                log.error("[%d/%d] %s: missing %s", i, len(volumes), vid, ct_path)
            else:
                try:
                    import SimpleITK as sitk
                    hdr = sitk.ImageFileReader()
                    hdr.SetFileName(ct_path)
                    hdr.ReadImageInformation()          # header only; no pixel IO
                    size = hdr.GetSize()
                    rec["size_xyz"] = list(size)
                    rec["n_voxels"] = int(size[0] * size[1] * size[2])
                    if spec is not None:
                        metrics = predict_volume_multiclass(predictor, model, spec, ct_path, out_files, tmp_root)
                    else:
                        metrics = predict_volume(predictor, ct_path, seg_path, prob_path, tmp_root, n_procs)
                    rec.update(metrics)
                    for role, link in model_registry.cohort_link_files(work, vid, model).items():
                        if role in out_files and os.path.isfile(out_files[role]):
                            relink(link, out_files[role])
                    rec["status"] = "ok"
                    n_ok += 1
                    log.info("[%d/%d] %s: ok %.0fs gpu_peak=%sMB fallback=%s", i, len(volumes), vid,
                             metrics["total_seconds"], metrics["gpu_peak_alloc_mb"],
                             metrics["cpu_fallback"])
                except Exception as exc:  # noqa: BLE001 — keep the chunk going
                    rec.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                               traceback=traceback.format_exc()[-4000:])
                    n_fail += 1
                    log.error("[%d/%d] %s: FAIL %s", i, len(volumes), vid, exc)
                    for p in out_files.values():
                        if os.path.exists(p + ".partial"):
                            os.remove(p + ".partial")
            rec["rss_self_mb"], rec["rss_children_mb"] = _rss_mb()
            records.append(rec)
            jf.write(json.dumps(rec) + "\n")
            jf.flush()

    summary = {"model": model, "model_version": entry["version"], "host": host, "gpu": gpu_name,
               "partition": job.get("SLURM_JOB_PARTITION"), "job_id": job.get("SLURM_JOB_ID"),
               "device": device, "n_procs": n_procs, "load_seconds": load_seconds,
               "n_ok": n_ok, "n_skipped": n_skip, "n_failed": n_fail,
               "rss_self_mb": _rss_mb()[0], "rss_children_mb": _rss_mb()[1],
               "volumes": records}
    log.info("done model=%s ok=%d skipped=%d failed=%d load=%.0fs", model, n_ok, n_skip, n_fail,
             load_seconds)
    if bench_out:
        os.makedirs(os.path.dirname(bench_out), exist_ok=True)
        with open(bench_out, "w") as f:
            json.dump(summary, f, indent=1)
        log.info("bench summary → %s", bench_out)
    return 0 if n_fail == 0 else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", required=True)
    ap.add_argument("--task-index", type=int, default=None,
                    help="Row of manifest/predict_tasks.tsv (default: $SLURM_ARRAY_TASK_ID).")
    ap.add_argument("--tasks-file", default=None)
    ap.add_argument("--model", default=None, help="Explicit model (bench / ad hoc).")
    ap.add_argument("--volumes", default=None, help="Comma-separated volume ids (with --model).")
    ap.add_argument("--volumes-file", default=None, help="Chunk file (volume_id<TAB>ct_path lines).")
    ap.add_argument("--folds", type=int, nargs="*", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--procs", type=int, default=None,
                    help="nnU-Net preprocessing/export workers (default: $SLURM_CPUS_PER_TASK or 2).")
    ap.add_argument("--tmp-dir", default=None,
                    help="Node-local scratch (default: $TMPDIR, else /scratch/$SLURM_JOB_ID, else mkdtemp).")
    ap.add_argument("--bench-out", default=None, help="Write a JSON summary here (bench mode).")
    ap.add_argument("--no-skip-existing", action="store_true")
    ap.add_argument("--nnunet-base", default=None,
                    help="Where to point nnUNet_raw/preprocessed/results (default: $LNQ_NNUNET_BASE or <work>/../nnunet).")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    _set_nnunet_env(args.nnunet_base or os.environ.get("LNQ_NNUNET_BASE")
                    or os.path.join(os.path.dirname(args.work.rstrip("/")), "nnunet"))

    n_procs = args.procs or int(os.environ.get("SLURM_CPUS_PER_TASK", "0") or 0) or 2
    tmp_root = args.tmp_dir or os.environ.get("TMPDIR")
    if not tmp_root and os.environ.get("SLURM_JOB_ID") and os.path.isdir("/scratch"):
        tmp_root = f"/scratch/{os.environ['SLURM_JOB_ID']}"
    if not tmp_root or not os.path.isdir(tmp_root):
        tmp_root = tempfile.mkdtemp(prefix="lnq-predict-")
    logs = os.path.join(args.work, "logs")

    if args.model:
        if args.volumes_file:
            volumes = _read_chunk(args.volumes_file)
        elif args.volumes:
            by_id = _volumes_by_id(args.work)
            missing = [v for v in args.volumes.split(",") if v not in by_id]
            if missing:
                log.error("unknown volume ids (not in manifest/volumes.csv): %s", missing)
                return 2
            volumes = [(v, by_id[v]["ct_path"]) for v in args.volumes.split(",")]
        else:
            log.error("--model needs --volumes or --volumes-file")
            return 2
        tag = f"bench-{os.environ.get('SLURM_JOB_PARTITION', 'local')}" if args.bench_out else "adhoc"
        jsonl = os.path.join(logs, f"predict-{args.model}-{tag}.jsonl")
        model = args.model
    else:
        idx = args.task_index
        if idx is None:
            idx = int(os.environ["SLURM_ARRAY_TASK_ID"])
        tasks = _read_tasks(args.tasks_file or os.path.join(args.work, "manifest", "predict_tasks.tsv"))
        if idx < 0 or idx >= len(tasks):
            log.error("task index %d out of range (0..%d)", idx, len(tasks) - 1)
            return 2
        task = tasks[idx]
        model = task["model"]
        volumes = _read_chunk(task["chunk_file"])
        jsonl = os.path.join(logs, f"predict-{model}-{int(task['chunk']):04d}.jsonl")

    return run(args.work, model, volumes, args.device, n_procs, tmp_root, jsonl,
               bench_out=args.bench_out, skip_existing=not args.no_skip_existing,
               folds=args.folds)


if __name__ == "__main__":
    sys.exit(main())
