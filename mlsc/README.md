# mlsc/ — PCCT → LNQ inference on the Martinos MLSC cluster

Runs the four `lnq-segmenter` models (`mediastinal-v1`, `axillary-v1`,
`inguinal-v1`, `abdominopelvic-v1`) over every axial CT series of the PDAC
photon-counting-CT studies, keeping probability maps, and produces the
`qc.csv` + directory layout that LNQReview / LNQStudio already read.

It is the Slurm re-cut of the Jetstream2 flow (`bin/ingest-idc-cohort.py` →
`bin/run-extra-anatomies.py` → `bin/idc-batch-qc.py`), following
[MLSC cluster](https://it.martinos.org/mlsc-cluster/) and
[etiquette](https://it.martinos.org/mlsc-etiquette/) guidance: explicit
`--mem`/`--time`, CPU work on `basic`, job arrays with `%N` concurrency,
~3 CPUs per GPU, node-local `$TMPDIR` for big intermediates, nothing heavy on
the login node.

**Data stays on `/vast/lnq`.** These are institutional studies (possible PHI,
faces). Nothing here uploads anything; manifests and logs contain only the
`E########` case id, series number/description/UID, geometry and scanner
settings. NRRD files carry no DICOM tags. If you copy `cohort/` to Dropbox
(MGB-approved for PHI) remember it is a tree of **symlinks**: use
`rclone copy --copy-links` or `cp -rL`.

## Layout on /vast/lnq

```
/vast/lnq/
  PCCT_exemplary_cases/               input DICOM (rclone'd), E######## dirs anywhere below
  env/                                venv        (setup-env.sbatch)
  models/                             lnq-segmenter weights cache (~16 GB)
  nnunet/                             dummy nnUNet_raw/preprocessed/results
  pdac-processing/
    mlsc.conf                         your copy of mlsc.conf.example
    manifest/
      case_list.txt                   one case dir per line (array index for stage)
      series_index.csv                every series seen, staged or rejected, with reasons
      volumes.csv                     staged volumes: id, ct path, geometry, flags, tags
      predict_tasks.tsv, chunks/      (model, chunk) rows → predict array tasks
      bench/<partition>-<model>.json  benchmark results
    logs/                             sbatch output + predict-*.jsonl per-volume records
    E12345678/
      003_Body_1-0_Br40_3/            <SeriesNumber:03d>_<sanitized description>[_stkN]
        ct.nrrd                       int16 HU, gzip, LPS
        geometry.json                 decision, flags, spacing stats, tags, source files
        mediastinal-v1-seg.nrrd       + -prob.nrrd (float32 foreground softmax on the CT grid)
        axillary-v1-seg.nrrd ...      x4 models
      series.csv                      per-case manifest (aggregated into manifest/)
    cohort/                           flat symlink view = what LNQReview expects
      nrrd/<volume_id>_0000.nrrd  →   ../../E…/…/ct.nrrd
      predictions/<model>/<volume_id>.nrrd, <volume_id>-prob.nrrd
      qc/<model>/qc.csv, <volume_id>.png
```

`volume_id = <case>__<series_dir>`, e.g. `E12345678__003_Body_1-0_Br40_3`.
In Slicer: **LNQ Studio → Inference Review**, cohort root
`/vast/lnq/pdac-processing/cohort`, model = any of the four. Each row loads
the CT, that model's SEG + probability map, and the other three anatomies'
SEGs (`cohort_list.derive_case_paths`).

## First time

```sh
cd /vast/lnq/pdac-processing
cp ~/SlicerLNQ-Chronicler/mlsc/mlsc.conf.example mlsc.conf     # edit paths if needed
M=~/SlicerLNQ-Chronicler/mlsc/run_pipeline.sh

$M probe            # one 10-min job per partition: python, driver, scratch, network
                    # → check TORCH_INDEX in mlsc.conf (cu126 needs driver >= 525)
$M setup            # uv-managed CPython 3.11 under /vast/lnq/python, venv on
                    # /vast/lnq/env, model weights (basic partition, ~30-60 min)
```

## Dry run and benchmark

```sh
$M stage --limit 2                    # two cases through DICOM → NRRD
$M build                              # manifests + cohort links (after stage finishes)
less manifest/series_index.csv        # check decisions / flags against Slicer
$M bench                              # all models × 3 representative volumes on each
                                      # partition in BENCH_PARTITIONS (~1-3 h each)
$M report                             # table + recommended GPU_PARTITION / PREDICT_MEM
```

Put the recommendation into `mlsc.conf` (`GPU_PARTITION`, `PREDICT_MEM`, and
`PREDICT_TIME` ≈ `CHUNK_SIZE` × slowest volume + model load).

## Production

```sh
$M all              # stage array → build (which then submits predict array → qc)
$M status           # queue + counts of volumes / seg / prob / qc rows / failures
```

Re-running any stage is safe: staging skips volumes that already have
`ct.nrrd` + `geometry.json`, `build` only plans volumes still missing a SEG or
probability map, `predict` skips finished volumes, `qc` regenerates the CSVs
(PNGs are kept). To redo a failed chunk: `$M build && $M predict --then-qc`.

Every job passes `-A lnqmlsc`, `--mem`, `--time`, `--mail-type END,FAIL`; GPU
arrays run at most `GPU_CONCURRENCY` (default 6) at a time. Check
`showpending` / `shownodes` before a big run and drop `GPU_CONCURRENCY` if the
partition is busy.

## What staging does (stage_dicom.py)

Per case: read every header, group by series, then

* **reject**: modality ≠ CT; SOP class not CT / Enhanced CT (dose reports,
  SR, secondary capture); `ImageType` LOCALIZER/SCOUT/TOPOGRAM; non-axial
  orientation (coronal/sagittal MPRs); fewer than `MIN_SLICES` (20) slices.
* **flag** (still converted, best effort): `duplicate_positions`,
  `missing_slices` + `interpolated_slices` (gaps filled by linear interpolation
  on the true grid; more than `MAX_MISSING_FRAC` missing → rejected as
  `incomplete_series`, typically an unfinished rclone copy — re-run `stage`
  once the copy completes and the volume is reconverted automatically),
  `irregular_spacing` (median spacing used), `gantry_tilt`, `sheared_stack`,
  `mixed_rescale`, `split_by_acquisition_number`, `multi_stack`,
  `reader_geometry_adjusted`. `--strict` rejects flagged series.
* **tag** (informational): `spectral=vnc|iodine|monoenergetic|zeff|spectral`,
  `kernel=Br40`, `multi_study`, `derived_secondary`.

Legacy single-frame series go through SimpleITK/GDCM (compressed transfer
syntaxes OK); Enhanced multi-frame CT is assembled from the per-frame
functional groups with pydicom. Flags/tags end up in `series_index.csv`,
`geometry.json`, and (via `qc_extras.py`) as extra columns in each `qc.csv`.

## Files

| file | runs on | purpose |
|---|---|---|
| `run_pipeline.sh` | login node | `sbatch` wrapper; reads `mlsc.conf` |
| `probe.sbatch` | each partition | node facts before setup |
| `setup-env.sbatch` | basic | relocatable python (uv) + venv + `lnq-segmenter download` |
| `stage_dicom.py` / `stage.sbatch` | basic (array/case) | DICOM → `ct.nrrd` + `geometry.json` |
| `build_cohort.py` / `build-cohort.sbatch` | basic | manifests, cohort symlinks, `predict_tasks.tsv` |
| `predict_batch.py` / `predict.sbatch` | GPU (array/(model,chunk)) | model loaded once per chunk; SEG + prob; per-volume JSONL |
| `bench.sbatch` / `bench_report.py` | GPU partitions | pick the lightest partition that fits |
| `qc.sbatch` (+ `../bin/idc-batch-qc.py`, `qc_extras.py`) | basic | `qc.csv` + PNGs per model |
| `tests/` | laptop | synthetic-DICOM tests: `python -m pytest mlsc/tests -q` |

`predict_batch.py` reuses `lnq_segmenter.registry` / `cache` /
`predict._ensure_nnunet_layout` / `predict._write_probability_map`, so the
outputs are byte-for-byte what the LymphNodeQuantifier Slicer module would
produce, just batched.
