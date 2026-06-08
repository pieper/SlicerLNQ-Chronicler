#!/usr/bin/env python3
"""run-extra-anatomies.py — predict additional anatomies on an already-staged
IDC cohort.

The mediastinal IDC batch (ingest-idc-cohort.py --model mediastinal-v1) leaves
per-case CT NRRDs under `<stage_root>/<collection>/nrrd/<case>_0000.nrrd`. We
want to run the abdominopelvic-v1 / axillary-v1 / inguinal-v1 models on the
same CTs so the LNQ Review cohort table can flag cases where a non-mediastinal
anatomy lit up.

This driver is intentionally thinner than ingest-idc-cohort.py: it does NOT
touch Chronicle. It iterates (model, case) pairs, calls lnq_segmenter.predict,
and lands SEG + probability NRRDs at the standard layout the LNQReview cohort
browser already consumes:

    <stage_root>/<collection>/predictions/<model>/<case>.nrrd
    <stage_root>/<collection>/predictions/<model>/<case>-prob.nrrd

Idempotent: skip a (model, case) if both outputs already exist. Run on
lnq-inferencer (CUDA), pointed at Manila.

Typical invocation:

  bin/run-extra-anatomies.py \\
      --stage-root /media/share/LNQ-data/idc \\
      --collection ct_lymph_nodes \\
      --models abdominopelvic-v1,axillary-v1,inguinal-v1 \\
      --device cuda
"""
from __future__ import annotations

import argparse
import datetime
import glob
import logging
import os
import sys
import time
import traceback


def list_cases(nrrd_dir):
    """Find every `<case>_0000.nrrd` under nrrd_dir and strip the suffix."""
    out = []
    for path in sorted(glob.glob(os.path.join(nrrd_dir, "*_0000.nrrd"))):
        name = os.path.basename(path)
        out.append((name[: -len("_0000.nrrd")], path))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--stage-root", default="/media/share/LNQ-data/idc",
                    help="Parent containing <collection>/nrrd/.")
    ap.add_argument("--collection", required=True,
                    help="IDC collection_id (e.g. ct_lymph_nodes).")
    ap.add_argument("--models", required=True,
                    help="Comma-separated lnq-segmenter model names.")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--limit", type=int, default=None,
                    help="Stop after this many cases per model (smoke testing).")
    ap.add_argument("--skip-existing", action="store_true", default=True,
                    help="Skip (model, case) pairs whose SEG + prob NRRDs are "
                         "already on disk. Default: on (idempotent).")
    ap.add_argument("--log", default=None,
                    help="Tee logs here in addition to stdout.")
    args = ap.parse_args()

    handlers = [logging.StreamHandler(sys.stdout)]
    if args.log:
        handlers.append(logging.FileHandler(args.log))
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=handlers)
    log = logging.getLogger("run-extra-anatomies")

    cohort_dir = os.path.join(args.stage_root, args.collection)
    nrrd_dir = os.path.join(cohort_dir, "nrrd")
    if not os.path.isdir(nrrd_dir):
        log.error("No CT NRRD dir at %s — has ingest-idc-cohort.py been run?",
                  nrrd_dir)
        sys.exit(2)
    cases = list_cases(nrrd_dir)
    if args.limit is not None:
        cases = cases[: args.limit]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    log.info("cohort=%s cases=%d models=%s device=%s",
             args.collection, len(cases), ",".join(models), args.device)

    # Import here so a missing nnunet/torch surfaces clearly with a non-empty
    # log file rather than at module-load time.
    from lnq_segmenter import predict as _predict

    total_start = time.time()
    for m_idx, model in enumerate(models):
        out_dir = os.path.join(cohort_dir, "predictions", model)
        os.makedirs(out_dir, exist_ok=True)
        log.info("[%d/%d] model=%s → %s", m_idx + 1, len(models), model, out_dir)
        for c_idx, (case_id, ct_path) in enumerate(cases):
            seg_path  = os.path.join(out_dir, f"{case_id}.nrrd")
            prob_path = os.path.join(out_dir, f"{case_id}-prob.nrrd")
            if args.skip_existing and os.path.isfile(seg_path) and os.path.isfile(prob_path):
                log.info("[%s][%3d/%d] %s: skip (SEG + prob already present)",
                         model, c_idx + 1, len(cases), case_id)
                continue
            t0 = time.time()
            try:
                _predict.predict(
                    model, ct_path, seg_path,
                    probability_output=prob_path,
                    device=args.device,
                )
                dt = time.time() - t0
                log.info("[%s][%3d/%d] %s: ok (%.1fs)",
                         model, c_idx + 1, len(cases), case_id, dt)
            except Exception as exc:
                log.error("[%s][%3d/%d] %s: FAIL %s\n%s",
                          model, c_idx + 1, len(cases), case_id, exc,
                          traceback.format_exc())
                # Keep going — one bad case shouldn't sink the rest of the run.

    total = time.time() - total_start
    log.info("done. total wall=%s",
             str(datetime.timedelta(seconds=int(total))))


if __name__ == "__main__":
    main()
