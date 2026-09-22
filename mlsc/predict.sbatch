#!/usr/bin/env bash
# predict.sbatch — one (model, chunk) per array task on a GPU.
#SBATCH --job-name=lnq-predict
set -euo pipefail
: "${ENV:?}" "${WORK:?}" "${MLSC_DIR:?}"
# shellcheck disable=SC1090
. "$ENV/activate.sh"
export TMPDIR=${TMPDIR:-/scratch/$SLURM_JOB_ID}
mkdir -p "$TMPDIR"
echo "== $(date -u) host=$(hostname) partition=${SLURM_JOB_PARTITION:-?} task=${SLURM_ARRAY_TASK_ID:-?} cpus=${SLURM_CPUS_PER_TASK:-?} tmp=$TMPDIR"
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader 2>/dev/null || true
python "$MLSC_DIR/predict_batch.py" --work "$WORK" --task-index "${SLURM_ARRAY_TASK_ID:?}" ${PREDICT_EXTRA:-}
