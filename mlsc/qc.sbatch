#!/usr/bin/env bash
# qc.sbatch — refresh cohort links, then per-model qc.csv + preview PNGs via
# the existing bin/idc-batch-qc.py, plus staging metadata columns.
#SBATCH --job-name=lnq-qc
set -euo pipefail
: "${ENV:?}" "${WORK:?}" "${MLSC_DIR:?}" "${MODELS:?}"
# shellcheck disable=SC1090
. "$ENV/activate.sh"
echo "== $(date -u) host=$(hostname)"
python "$MLSC_DIR/build_cohort.py" --work "$WORK" --models "$MODELS" --refresh-links
for m in $MODELS; do
  others=$(for o in $MODELS; do [ "$o" != "$m" ] && printf '%s,' "$o"; done | sed 's/,$//')
  echo "== qc for $m (extra anatomies: $others)"
  python "$MLSC_DIR/../bin/idc-batch-qc.py" --stage-root "$WORK/cohort" --model "$m" \
    --extra-anatomies "$others" ${QC_EXTRA:-}
done
python "$MLSC_DIR/qc_extras.py" --work "$WORK" --models "$MODELS"
echo "== done $(date -u)"
