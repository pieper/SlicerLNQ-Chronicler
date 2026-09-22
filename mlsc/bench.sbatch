#!/usr/bin/env bash
# bench.sbatch — run every model over the same few volumes on this partition
# and record time / peak GPU / peak RSS. One submission per partition
# (run_pipeline.sh bench). Results: $WORK/manifest/bench/<partition>-<model>.json
#SBATCH --job-name=lnq-bench
set -euo pipefail
: "${ENV:?}" "${WORK:?}" "${MLSC_DIR:?}" "${MODELS:?}" "${BENCH_VOLUMES:?}"
# shellcheck disable=SC1090
. "$ENV/activate.sh"
export TMPDIR=${TMPDIR:-/scratch/$SLURM_JOB_ID}
mkdir -p "$TMPDIR" "$WORK/manifest/bench"
part=${SLURM_JOB_PARTITION:-unknown}
echo "== $(date -u) host=$(hostname) partition=$part volumes=$BENCH_VOLUMES"
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || true
rc=0
for m in $MODELS; do
  echo "== model $m"
  python "$MLSC_DIR/predict_batch.py" --work "$WORK" --model "$m" --volumes "$BENCH_VOLUMES" \
    --no-skip-existing --bench-out "$WORK/manifest/bench/$part-$m.json" || rc=1
done
echo "== sacct view of this job (MaxRSS is the number PREDICT_MEM must cover):"
sacct -j "$SLURM_JOB_ID" --format=JobID,Elapsed,MaxRSS,MaxVMSize,State -P 2>/dev/null || true
exit $rc
