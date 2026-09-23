#!/usr/bin/env bash
# stage.sbatch — DICOM → NRRD for one case per array task (CPU, basic partition).
#SBATCH --job-name=lnq-stage
set -euo pipefail
: "${ENV:?}" "${INPUT:?}" "${WORK:?}" "${MLSC_DIR:?}"
# shellcheck disable=SC1090
. "$ENV/activate.sh"
echo "== $(date -u) host=$(hostname) task=${SLURM_ARRAY_TASK_ID:-?}"
python "$MLSC_DIR/stage_dicom.py" --input "$INPUT" --work "$WORK" \
  --case-index "${SLURM_ARRAY_TASK_ID:?}" --min-slices "${MIN_SLICES:-20}" \
  --max-missing-frac "${MAX_MISSING_FRAC:-1.0}" ${STAGE_EXTRA:-}
