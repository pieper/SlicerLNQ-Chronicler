#!/usr/bin/env bash
# sync.sbatch — run sync_cases.py (push or pull) as a basic-partition job.
# Submitted by `run_pipeline.sh push|pull --remote <rclone path>`.
#SBATCH --job-name=lnq-sync
#SBATCH --time=1-00:00:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=2
set -euo pipefail
: "${ENV:?}" "${WORK:?}" "${MLSC_DIR:?}" "${SYNC_MODE:?}" "${SYNC_REMOTE:?}"
# shellcheck disable=SC1090
. "$ENV/activate.sh"
echo "== $(date -u) host=$(hostname) mode=$SYNC_MODE remote=$SYNC_REMOTE"
command -v rclone >/dev/null || { echo "rclone not on PATH on this node" >&2; exit 2; }
python "$MLSC_DIR/sync_cases.py" --work "$WORK" --remote "$SYNC_REMOTE" "--$SYNC_MODE" ${SYNC_EXTRA:-}
