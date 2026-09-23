#!/usr/bin/env bash
# inventory.sbatch — header-only census of the input tree (basic partition).
#SBATCH --job-name=lnq-inventory
#SBATCH --time=0-02:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
set -euo pipefail
: "${ENV:?}" "${INPUT:?}" "${WORK:?}" "${MLSC_DIR:?}"
# shellcheck disable=SC1090
. "$ENV/activate.sh"
echo "== $(date -u) host=$(hostname)"
python "$MLSC_DIR/inventory.py" --input "$INPUT" --work "$WORK" --min-slices "${MIN_SLICES:-20}"
