#!/usr/bin/env bash
# probe.sbatch — 10-minute look at a node: OS, python, driver, scratch, network.
# Submitted once per partition by `run_pipeline.sh probe`. Read the output to
# pick PYTHON / TORCH_INDEX in mlsc.conf before running setup.
#SBATCH --job-name=lnq-probe
#SBATCH --time=0-00:10:00
#SBATCH --mem=4G
#SBATCH --cpus-per-task=1
set -u
PY=${PYTHON:-python3}
echo "== host: $(hostname)   partition: ${SLURM_JOB_PARTITION:-?}   job: ${SLURM_JOB_ID:-?}   $(date -u)"
echo "== os: $(grep PRETTY_NAME /etc/os-release 2>/dev/null | cut -d= -f2)   glibc: $(ldd --version 2>/dev/null | head -1)"
echo "== cpus: $(nproc)   mem: $(free -g 2>/dev/null | awk '/Mem/{print $2"G"}')"
echo "== python candidates:"
for p in "$PY" python3 python3.9 python3.10 python3.11 python3.12 python3.13; do
  command -v "$p" >/dev/null 2>&1 && echo "   $(command -v "$p"): $("$p" --version 2>&1)"
done | sort -u
ls -d /usr/pubsw/packages/python* /usr/pubsw/packages/anaconda* 2>/dev/null | sed 's/^/   /'
if command -v module >/dev/null 2>&1; then module avail python 2>&1 | head -20 | sed 's/^/   /'; fi
echo "== nvidia:"
nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv 2>&1 | sed 's/^/   /' || echo "   no nvidia-smi"
echo "== scratch: TMPDIR=${TMPDIR:-unset}   SINGULARITY_TMPDIR=${SINGULARITY_TMPDIR:-unset}"
df -h /scratch "${TMPDIR:-/scratch}" 2>&1 | tail -n +2 | sed 's/^/   /'
echo "== vast:"
df -h /vast/lnq 2>&1 | tail -1 | sed 's/^/   /'
ls -ld /vast/lnq /vast/lnq/* 2>&1 | sed 's/^/   /'
echo "== network from node:"
tmp=$(mktemp -d "${TMPDIR:-/tmp}/probe.XXXXXX")
if "$PY" -m pip download --no-deps -q -d "$tmp" six >/dev/null 2>&1; then echo "   pip index: OK"; else echo "   pip index: FAILED (pip missing or no outbound network)"; fi
curl -sSI --max-time 20 https://github.com -o /dev/null -w "   github.com: HTTP %{http_code}\n" 2>&1 || echo "   github.com: unreachable"
curl -sSI --max-time 20 https://download.pytorch.org/whl/ -o /dev/null -w "   download.pytorch.org: HTTP %{http_code}\n" 2>&1 || echo "   download.pytorch.org: unreachable"
rm -rf "$tmp"
echo "== done"
