#!/bin/bash
#SBATCH --job-name=saps-full
#SBATCH --output=logs/saps-full-%A_%a.out
#SBATCH --error=logs/saps-full-%A_%a.err
#SBATCH --array=0-63
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --partition=cpu-harvester
#
# One Slurm array task = one suite chunk.
# submit_full_suite.sh overrides --array/--time/--partition/--mem so many short
# tasks can fill harvester / small openings instead of one long job.
#
# Inside the task:
#   --chunk-count  = SLURM_ARRAY_TASK_COUNT
#   --chunk-index  = SLURM_ARRAY_TASK_ID
#
# Submit via: scripts/submit_full_suite.sh

set -euo pipefail

REPO_ROOT="${SAPS_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
mkdir -p logs

module load python/3.12.5

if [[ -n "${SAPS_VENV:-}" ]]; then
  # shellcheck source=/dev/null
  source "${SAPS_VENV}/bin/activate"
elif [[ -x "${REPO_ROOT}/.venv/bin/activate" ]]; then
  # shellcheck source=/dev/null
  source "${REPO_ROOT}/.venv/bin/activate"
elif command -v poetry >/dev/null 2>&1; then
  POETRY_ENV="$(poetry env info -p 2>/dev/null || true)"
  if [[ -n "${POETRY_ENV}" && -f "${POETRY_ENV}/bin/activate" ]]; then
    # shellcheck source=/dev/null
    source "${POETRY_ENV}/bin/activate"
  fi
fi

if ! python -c "import asv, numpy, saps, scipy, sparse, torch" >/dev/null 2>&1; then
  echo "ERROR: SAPS Python environment not available." >&2
  echo "Run 'poetry install --with test', then export SAPS_VENV to that env." >&2
  exit 1
fi

CHUNK_COUNT="${SLURM_ARRAY_TASK_COUNT:-1}"
CHUNK_INDEX="${SLURM_ARRAY_TASK_ID:-0}"
RUN_ROOT="${SAPS_RUN_ROOT:?SAPS_RUN_ROOT must identify this submission}"
TASK_ROOT="${RUN_ROOT}/tasks/chunk_${CHUNK_INDEX}"
MACHINE="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}-chunk${CHUNK_INDEX}"
CONFIG="${SAPS_CONFIG:-scripts/full_suite.conf.json}"
TIMEOUT="${SAPS_TIMEOUT:-30}"

if [[ "${CONFIG}" != /* ]]; then
  CONFIG="${REPO_ROOT}/${CONFIG}"
fi

mkdir -p "${TASK_ROOT}/home" "${RUN_ROOT}/results"
cp "${REPO_ROOT}/manifest.json" "${TASK_ROOT}/manifest.json"
export SAPS_DIR="${TASK_ROOT}/saps"
export SAPS_MANIFEST_PATH="${TASK_ROOT}/manifest.json"
export HOME="${TASK_ROOT}/home"
export XDG_CACHE_HOME="${TASK_ROOT}/xdg-cache"
export SAPS_WARMUP_TIME="${SAPS_WARMUP_TIME:-4}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

echo "Host=$(hostname) array_task=${CHUNK_INDEX}/${CHUNK_COUNT} machine=${MACHINE}"
echo "Python=$(command -v python) $(python --version 2>&1)"
echo "Run root=${RUN_ROOT}"
echo "Task SAPS_DIR=${SAPS_DIR}"
echo "Warmup=${SAPS_WARMUP_TIME}s timeout=${TIMEOUT}s"

python ./bin/run_benchmark.py \
  --config "${CONFIG}" \
  --machine "${MACHINE}" \
  --timeout "${TIMEOUT}" \
  --metric time peakmem \
  --chunk-count "${CHUNK_COUNT}" \
  --chunk-index "${CHUNK_INDEX}" \
  --show-stderr

# Each task has private ASV/cache state. Publish only its uniquely named result
# directory into the run-level collection consumed by the dependent report job.
for machine_dir in "${SAPS_DIR}/outputs/results"/*; do
  [[ -d "${machine_dir}" ]] || continue
  cp -a "${machine_dir}" "${RUN_ROOT}/results/"
done
touch "${TASK_ROOT}/complete"
