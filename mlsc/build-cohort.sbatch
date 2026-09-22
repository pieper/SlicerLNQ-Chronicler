#!/usr/bin/env bash
# build-cohort.sbatch — aggregate manifests, make the flat cohort view, plan
# GPU chunks. With SUBMIT_NEXT=1 (run_pipeline.sh all) it then submits the
# predict array and the qc job itself, because the array size is only known
# once this step has run.
#SBATCH --job-name=lnq-build-cohort
#SBATCH --time=0-00:30:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
set -euo pipefail
: "${ENV:?}" "${WORK:?}" "${MLSC_DIR:?}" "${MODELS:?}"
# shellcheck disable=SC1090
. "$ENV/activate.sh"
echo "== $(date -u) host=$(hostname)"
python "$MLSC_DIR/build_cohort.py" --work "$WORK" --models "$MODELS" --chunk-size "${CHUNK_SIZE:-8}"
if [ "${SUBMIT_NEXT:-0}" = "1" ]; then
  echo "== submitting predict + qc"
  "$MLSC_DIR/run_pipeline.sh" --conf "${MLSC_CONF:?}" predict --then-qc
fi
