#!/usr/bin/env bash
# run_pipeline.sh — submit the PCCT → LNQ pipeline stages to Slurm on MLSC.
#
#   run_pipeline.sh [--conf FILE] [--dry-run] <command> [options]
#
#   probe                 10-min probe job on every partition in $BENCH_PARTITIONS + $CPU_PARTITION
#   setup                 venv + model weights on the basic partition
#   list-cases            write manifest/case_list.txt (directory walk only; fine on the login node)
#   inventory             header-only census: cases, patients, studies, series, candidate volumes
#   stage [--limit N] [--force] [--strict]
#                         DICOM → NRRD, one array task per case
#   build [--after JOB]   manifests + cohort symlinks + predict_tasks.tsv
#   bench [--volumes a,b] every model over 3 representative volumes on each bench partition
#   report                bench_report.py table + recommendation
#   predict [--then-qc]   GPU array over predict_tasks.tsv (needs build first)
#   qc [--after JOB]      qc.csv + PNGs per model
#   all [--limit N]       stage → build → (build submits predict → qc)
#   status                queue, on-disk counts, failures, seconds/volume, ETA
#   push --remote R [args]   copy results to rclone remote R (e.g. dropbox:PDAC) case by case, as a job
#   pull --remote R [args]   the reverse (remote → $WORK), as a job
#   sync-log              tail the log of the most recent push/pull job
#   sync-status --remote R   which cases are complete on the remote (no transfer)
#
# Only sbatch/squeue/sacct and a directory walk run here; everything else is a job.
set -euo pipefail

MLSC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DRY_RUN=0
CONF="${MLSC_CONF:-}"
while [ $# -gt 0 ]; do
  case "$1" in
    --conf) CONF="$2"; shift 2;;
    --dry-run) DRY_RUN=1; shift;;
    -h|--help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0;;
    *) break;;
  esac
done
CMD="${1:-}"; shift || true
[ -n "$CMD" ] || { sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 2; }

for candidate in "$CONF" "$PWD/mlsc.conf" /vast/lnq/pdac-processing/mlsc.conf "$MLSC_DIR/mlsc.conf"; do
  if [ -n "$candidate" ] && [ -f "$candidate" ]; then CONF="$candidate"; break; fi
done
[ -f "${CONF:-}" ] || { echo "no mlsc.conf found (copy mlsc.conf.example and pass --conf)" >&2; exit 2; }
set -a
# shellcheck disable=SC1090
. "$CONF"
set +a
export MLSC_DIR MLSC_CONF="$CONF"
: "${ACCOUNT:?}" "${WORK:?}" "${INPUT:?}" "${ENV:?}" "${MODELS:?}" "${CPU_PARTITION:?}" "${GPU_PARTITION:?}"
mkdir -p "$WORK/logs" "$WORK/manifest"
PY="$ENV/bin/python"; [ -x "$PY" ] || PY=python3

ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
common=(-A "$ACCOUNT" --mail-type "${MAIL_TYPE:-END,FAIL}")
[ -n "${MAIL_USER:-}" ] && common+=(--mail-user "$MAIL_USER")

submit() {  # submit <sbatch flags...> -- script ; echoes job id
  local flags=() script
  while [ "$1" != "--" ]; do flags+=("$1"); shift; done
  shift; script="$1"
  echo "$(ts) sbatch ${common[*]} ${flags[*]} $script" >&2
  if [ "$DRY_RUN" = 1 ]; then echo "DRY"; return; fi
  sbatch --parsable "${common[@]}" "${flags[@]}" "$script" | cut -d';' -f1
}

count_lines() { [ -f "$1" ] && grep -c . "$1" || echo 0; }

cmd_probe() {
  for p in $CPU_PARTITION ${BENCH_PARTITIONS:-}; do
    submit -p "$p" $( [ "$p" = "$CPU_PARTITION" ] || echo "--gpus=1" ) \
      --output "$WORK/logs/probe-$p-%j.out" -- "$MLSC_DIR/probe.sbatch"
  done
  echo "read $WORK/logs/probe-*.out, then set PYTHON / TORCH_INDEX in $CONF"
}

cmd_setup() {
  submit -p "$CPU_PARTITION" --output "$WORK/logs/setup-env-%j.out" -- "$MLSC_DIR/setup-env.sbatch"
}

cmd_list_cases() {
  "$PY" "$MLSC_DIR/stage_dicom.py" --input "$INPUT" --work "$WORK" --list-cases
}

cmd_inventory() {
  submit -p "$CPU_PARTITION" --output "$WORK/logs/inventory-%j.out" -- "$MLSC_DIR/inventory.sbatch"
  echo "result: $WORK/logs/inventory-<jobid>.out and $WORK/manifest/inventory.csv" >&2
}

cmd_stage() {
  local limit="" extra=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --limit) limit="$2"; shift 2;;
      --force|--strict) extra="$extra $1"; shift;;
      *) echo "stage: unknown option $1" >&2; exit 2;;
    esac
  done
  cmd_list_cases >&2
  local n; n=$(count_lines "$WORK/manifest/case_list.txt")
  [ "$n" -gt 0 ] || { echo "no E######## case directories under $INPUT" >&2; exit 1; }
  [ -n "$limit" ] && [ "$limit" -lt "$n" ] && n="$limit"
  STAGE_EXTRA="$extra" submit -p "$CPU_PARTITION" --cpus-per-task "${STAGE_CPUS:-4}" \
    --mem "${STAGE_MEM:-24G}" --time "${STAGE_TIME:-0-02:00:00}" \
    --array "0-$((n - 1))%${STAGE_CONCURRENCY:-20}" \
    --output "$WORK/logs/stage-%A_%a.out" -- "$MLSC_DIR/stage.sbatch"
}

cmd_build() {
  local after="" flags=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --after) after="$2"; shift 2;;
      --submit-next) export SUBMIT_NEXT=1; shift;;
      *) echo "build: unknown option $1" >&2; exit 2;;
    esac
  done
  [ -n "$after" ] && flags+=(--dependency "afterany:$after")
  submit -p "$CPU_PARTITION" "${flags[@]}" --output "$WORK/logs/build-cohort-%j.out" \
    -- "$MLSC_DIR/build-cohort.sbatch"
}

cmd_bench() {
  local volumes=""
  while [ $# -gt 0 ]; do
    case "$1" in
      --volumes) volumes="$2"; shift 2;;
      *) echo "bench: unknown option $1" >&2; exit 2;;
    esac
  done
  [ -n "$volumes" ] || volumes=$("$PY" "$MLSC_DIR/bench_report.py" --work "$WORK" --select --n 3)
  echo "bench volumes: $volumes" >&2
  for p in ${BENCH_PARTITIONS:-$GPU_PARTITION}; do
    BENCH_VOLUMES="$volumes" submit -p "$p" --gpus=1 --cpus-per-task "${PREDICT_CPUS:-3}" \
      --mem "${BENCH_MEM:-128G}" --time "${BENCH_TIME:-0-04:00:00}" \
      --output "$WORK/logs/bench-$p-%j.out" -- "$MLSC_DIR/bench.sbatch"
  done
  echo "when done: run_pipeline.sh report" >&2
}

cmd_report() { "$PY" "$MLSC_DIR/bench_report.py" --work "$WORK"; }

cmd_predict() {
  local then_qc=0
  while [ $# -gt 0 ]; do
    case "$1" in
      --then-qc) then_qc=1; shift;;
      *) echo "predict: unknown option $1" >&2; exit 2;;
    esac
  done
  local tasks="$WORK/manifest/predict_tasks.tsv"
  [ -f "$tasks" ] || { echo "no $tasks — run build first" >&2; exit 1; }
  local t; t=$(( $(count_lines "$tasks") - 1 ))
  if [ "$t" -le 0 ]; then echo "nothing to predict (all outputs present)"; [ "$then_qc" = 1 ] && cmd_qc; return; fi
  local jid
  jid=$(submit -p "$GPU_PARTITION" --gpus=1 --cpus-per-task "${PREDICT_CPUS:-3}" \
    --mem "${PREDICT_MEM:-96G}" --time "${PREDICT_TIME:-0-08:00:00}" \
    --array "0-$((t - 1))%${GPU_CONCURRENCY:-6}" \
    --output "$WORK/logs/predict-%A_%a.out" -- "$MLSC_DIR/predict.sbatch")
  echo "predict array: $jid ($t tasks, ${GPU_CONCURRENCY:-6} concurrent on $GPU_PARTITION)"
  [ "$then_qc" = 1 ] && cmd_qc --after "$jid"
  return 0
}

cmd_qc() {
  local after="" flags=()
  while [ $# -gt 0 ]; do
    case "$1" in
      --after) after="$2"; shift 2;;
      *) echo "qc: unknown option $1" >&2; exit 2;;
    esac
  done
  [ -n "$after" ] && flags+=(--dependency "afterany:$after")
  submit -p "$CPU_PARTITION" "${flags[@]}" --cpus-per-task "${QC_CPUS:-2}" --mem "${QC_MEM:-32G}" \
    --time "${QC_TIME:-0-04:00:00}" --output "$WORK/logs/qc-%j.out" -- "$MLSC_DIR/qc.sbatch"
}

cmd_all() {
  local stage_jid build_jid
  stage_jid=$(cmd_stage "$@")
  echo "stage array: $stage_jid"
  build_jid=$(cmd_build --after "$stage_jid" --submit-next)
  echo "build-cohort: $build_jid (will submit predict + qc when it finishes)"
}

cmd_status() {
  echo "== queue"; squeue -u "$USER" -o "%.16i %.14j %.9P %.8T %.10M %.6D %R" 2>/dev/null || true
  echo "== recent jobs"; sacct -u "$USER" -S "$(date -d '-2 days' +%F 2>/dev/null || date -v-2d +%F)" \
    --format=JobID,JobName%16,Partition,State,Elapsed,MaxRSS -P 2>/dev/null | grep -v '\.batch\|\.extern' | tail -30 || true
  echo "== on disk ($WORK)"
  echo "   cases listed:   $(count_lines "$WORK/manifest/case_list.txt")"
  echo "   cases staged:   $(ls -d "$WORK"/E????????/series.csv 2>/dev/null | wc -l | tr -d ' ')"
  echo "   volumes (ct):   $(ls "$WORK"/cohort/nrrd/*_0000.nrrd 2>/dev/null | wc -l | tr -d ' ')"
  for m in $MODELS; do
    printf '   %-20s seg=%s prob=%s qc_rows=%s\n' "$m" \
      "$(ls "$WORK"/E????????/*/"$m"-seg.nrrd 2>/dev/null | wc -l | tr -d ' ')" \
      "$(ls "$WORK"/E????????/*/"$m"-prob.nrrd 2>/dev/null | wc -l | tr -d ' ')" \
      "$(( $(count_lines "$WORK/cohort/qc/$m/qc.csv") > 0 ? $(count_lines "$WORK/cohort/qc/$m/qc.csv") - 1 : 0 ))"
  done
  local running
  running=$(squeue -u "$USER" -h -t R -o "%j" 2>/dev/null | grep -c '^lnq-predict' || true)
  echo "== progress"
  "$PY" "$MLSC_DIR/progress.py" --work "$WORK" --models "$MODELS" --running "${running:-0}" \
    --concurrency "${GPU_CONCURRENCY:-6}"
}

cmd_sync() {   # cmd_sync push|pull --remote R [extra sync_cases.py args]
  local mode="$1" remote="" extra=""; shift
  while [ $# -gt 0 ]; do
    case "$1" in
      --remote) remote="$2"; shift 2;;
      *) extra="$extra $1"; shift;;
    esac
  done
  [ -n "$remote" ] || { echo "$mode: --remote <rclone path> required (e.g. dropbox:PDAC)" >&2; exit 2; }
  local jid
  jid=$(SYNC_MODE="$mode" SYNC_REMOTE="$remote" SYNC_EXTRA="$extra" \
        submit -p "$CPU_PARTITION" --job-name "lnq-sync-$mode" \
        --output "$WORK/logs/sync-$mode-%j.out" -- "$MLSC_DIR/sync.sbatch")
  echo "$jid" > "$WORK/manifest/sync-$mode.jobid"
  echo "sync-$mode job $jid → $WORK/logs/sync-$mode-$jid.out   (run_pipeline.sh sync-log to follow)"
}

cmd_sync_log() {
  # Follow the log of the most recently submitted push/pull job (from the
  # saved job id), waiting for the file if the job is still queued.
  local idfile mode jid f
  idfile=$(ls -t "$WORK"/manifest/sync-*.jobid 2>/dev/null | head -1)
  [ -n "$idfile" ] || { echo "no sync job submitted yet (run_pipeline.sh push|pull)" >&2; exit 1; }
  mode=$(basename "$idfile" .jobid | sed 's/^sync-//')
  jid=$(cat "$idfile")
  f="$WORK/logs/sync-$mode-$jid.out"
  while [ ! -f "$f" ]; do
    echo "job $jid not started yet: $(squeue -j "$jid" -h -o '%T %R' 2>/dev/null || echo 'not in queue')"
    sleep 15
  done
  echo "== $f"; tail -n 40 -f "$f"
}

cmd_sync_status() {
  local remote=""
  while [ $# -gt 0 ]; do case "$1" in --remote) remote="$2"; shift 2;; *) shift;; esac; done
  [ -n "$remote" ] || { echo "sync-status: --remote required" >&2; exit 2; }
  "$PY" "$MLSC_DIR/sync_cases.py" --work "$WORK" --remote "$remote" --push --status
}

case "$CMD" in
  probe) cmd_probe "$@";;
  setup) cmd_setup "$@";;
  list-cases) cmd_list_cases "$@";;
  inventory) cmd_inventory "$@";;
  stage) cmd_stage "$@";;
  build) cmd_build "$@";;
  bench) cmd_bench "$@";;
  report) cmd_report "$@";;
  predict) cmd_predict "$@";;
  qc) cmd_qc "$@";;
  all) cmd_all "$@";;
  status) cmd_status "$@";;
  push|pull) cmd_sync "$CMD" "$@";;
  sync-log) cmd_sync_log "$@";;
  sync-status) cmd_sync_status "$@";;
  *) echo "unknown command: $CMD" >&2; exit 2;;
esac
