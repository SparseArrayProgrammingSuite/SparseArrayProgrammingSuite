#!/bin/bash
# Submit a parallel full-suite ASV run as a Slurm task array, then (optionally)
# compile visualization reports after all chunks finish.
#
# The suite is split into CHUNK_COUNT independent array tasks so each piece can
# fit into short/harvester openings instead of one long wall-clock job.
#
# Usage:
#   ACCOUNT=gts-wahrens6 SAPS_VENV=$PWD/.venv scripts/submit_full_suite.sh
#
# Harvester / scavenger-friendly example (many short tasks):
#   ACCOUNT=gts-wahrens6 PARTITION=cpu-harvester CHUNK_COUNT=64 \
#     TIME_LIMIT=01:00:00 MEM=16G CPUS=2 SAPS_VENV=$PWD/.venv \
#     scripts/submit_full_suite.sh
#
# Prerequisites:
#   - Working SAPS env (Poetry or .venv) with test deps / torch installed
#   - Read access to the configured dataset storage backend
#   - A PACE Slurm charge account (see: pace-quota)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p logs

# Prefer many short array tasks so Slurm can pack them into small openings.
CHUNK_COUNT="${CHUNK_COUNT:-64}"
PARTITION="${PARTITION:-cpu-harvester}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"
MEM="${MEM:-16G}"
CPUS="${CPUS:-2}"
ACCOUNT="${ACCOUNT:-${SLURM_ACCOUNT:-}}"
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$$}"
RUN_ROOT="${SAPS_RUN_ROOT:-${REPO_ROOT}/.saps/runs/${RUN_ID}}"

if [[ -z "${ACCOUNT}" ]]; then
  echo "ACCOUNT is required (PACE Slurm charge account)." >&2
  echo "Find yours with: pace-quota" >&2
  echo "Then resubmit, e.g.:" >&2
  echo "  ACCOUNT=gts-wahrens6 SAPS_VENV=\$PWD/.venv scripts/submit_full_suite.sh" >&2
  exit 1
fi

if [[ "${CHUNK_COUNT}" -lt 1 ]]; then
  echo "CHUNK_COUNT must be >= 1" >&2
  exit 1
fi

ARRAY_MAX=$((CHUNK_COUNT - 1))

module load python/3.12.5
if [[ -n "${SAPS_VENV:-}" ]]; then
  SAPS_VENV="$(cd "${SAPS_VENV}" && pwd)"
elif [[ -f "${REPO_ROOT}/.venv/bin/activate" ]]; then
  SAPS_VENV="${REPO_ROOT}/.venv"
elif command -v poetry >/dev/null 2>&1; then
  SAPS_VENV="$(poetry env info -p 2>/dev/null || true)"
fi

if [[ -z "${SAPS_VENV:-}" || ! -f "${SAPS_VENV}/bin/activate" ]]; then
  echo "A SAPS virtual environment is required." >&2
  echo "Run 'poetry install --with test' and set SAPS_VENV to its path." >&2
  exit 1
fi

# shellcheck source=/dev/null
source "${SAPS_VENV}/bin/activate"
if ! python -c "import asv, numpy, saps, scipy, sparse, torch"; then
  echo "The selected environment is missing SAPS benchmark dependencies." >&2
  echo "Run 'poetry install --with test' before submitting." >&2
  exit 1
fi

mkdir -p "${RUN_ROOT}/logs" "${RUN_ROOT}/results"

echo "Submitting full suite as Slurm array: ${CHUNK_COUNT} tasks on ${PARTITION}"
echo "Each task walltime=${TIME_LIMIT} mem=${MEM} cpus=${CPUS}"
echo "Slurm account: ${ACCOUNT}"
echo "Isolated run folder: ${RUN_ROOT}"
RUN_JOB_ID="$(
  sbatch --parsable \
    --account="${ACCOUNT}" \
    --partition="${PARTITION}" \
    --time="${TIME_LIMIT}" \
    --mem="${MEM}" \
    --cpus-per-task="${CPUS}" \
    --array="0-${ARRAY_MAX}%${ARRAY_THROTTLE:-${CHUNK_COUNT}}" \
    --output="${RUN_ROOT}/logs/saps-full-%A_%a.out" \
    --error="${RUN_ROOT}/logs/saps-full-%A_%a.err" \
    --export=ALL,SAPS_REPO_ROOT="${REPO_ROOT}",SAPS_RUN_ROOT="${RUN_ROOT}",SAPS_VENV="${SAPS_VENV}",SAPS_CONFIG="${SAPS_CONFIG:-scripts/full_suite.conf.json}",SAPS_TIMEOUT="${SAPS_TIMEOUT:-30}",SAPS_WARMUP_TIME="${SAPS_WARMUP_TIME:-4}" \
    scripts/run_full_suite.sbatch
)"
echo "Benchmark array job: ${RUN_JOB_ID}  (tasks 0-${ARRAY_MAX})"

if [[ -f "${REPO_ROOT}/scripts/compile_visualizations.sbatch" ]]; then
  VIZ_JOB_ID="$(
    sbatch --parsable \
      --account="${ACCOUNT}" \
      --partition="${PARTITION}" \
      --time="${VIZ_TIME_LIMIT:-00:30:00}" \
      --mem="${VIZ_MEM:-8G}" \
      --cpus-per-task="${VIZ_CPUS:-1}" \
      --dependency="afterany:${RUN_JOB_ID}" \
      --output="${RUN_ROOT}/logs/saps-viz-%j.out" \
      --error="${RUN_ROOT}/logs/saps-viz-%j.err" \
      --export=ALL,SAPS_REPO_ROOT="${REPO_ROOT}",SAPS_RUN_ROOT="${RUN_ROOT}",SAPS_CHUNK_COUNT="${CHUNK_COUNT}",SAPS_VENV="${SAPS_VENV}",SAPS_VIZ_DIR="${SAPS_VIZ_DIR:-${RUN_ROOT}/visualizations}",SAPS_RESULTS_DIR="${SAPS_RESULTS_DIR:-${RUN_ROOT}/results}" \
      scripts/compile_visualizations.sbatch
  )"
  echo "Visualization job: ${VIZ_JOB_ID} (afterany:${RUN_JOB_ID})"
else
  echo "Skipping viz job: scripts/compile_visualizations.sbatch not found."
  echo "ASV JSONs will still land under ${RUN_ROOT}/results/"
  echo "You can build the Plotly profile later with:"
  echo "  python bin/build_asv_summary.py --results-dir ${RUN_ROOT}/results"
  echo "  python bin/build_plotly.py"
fi

echo
echo "Monitor with:  squeue -u \$USER"
echo "Run folder:    ${RUN_ROOT}"
echo "ASV results:   ${RUN_ROOT}/results/"
