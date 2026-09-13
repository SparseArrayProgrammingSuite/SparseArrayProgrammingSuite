#!/usr/bin/env bash
#SBATCH --job-name=saps-competition
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --exclusive
#SBATCH --mem=128G
#SBATCH --tmp=50G
#SBATCH --time=04:00:00
#SBATCH --array=0-127
#SBATCH --output=competition-%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahrens@gatech.edu

set -euo pipefail

repo_directory="${SAPS_REPO_DIRECTORY:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_directory" ]]; then
  script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
  repo_directory=$(cd -- "$script_directory/.." && pwd)
fi
repo_directory=$(git -C "$repo_directory" rev-parse --show-toplevel)

export PATH="$HOME/.local/bin:$PATH"
export AWS_PROFILE="${AWS_PROFILE:-dataset-upload}"
export REMOTE_STORAGE_BACKEND="${REMOTE_STORAGE_BACKEND:-s3}"
export REMOTE_STORAGE_BUCKET="${REMOTE_STORAGE_BUCKET:-sparse-array-programming-suite}"

competition_config="${SAPS_COMPETITION_CONFIG:-$repo_directory/competition.config.json}"

job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
task_index="${SLURM_ARRAY_TASK_ID:-0}"
task_count="${SLURM_ARRAY_TASK_COUNT:-1}"
competition_directory="$repo_directory/competition"
competition_run_root="$competition_directory/run_$job_id"
if (( $# > 0 )); then
  if (( $# != 2 )) || [[ "$1" != "--resume" ]]; then
    echo "usage: sbatch $0 [--resume RUN_DIRECTORY]" >&2
    exit 2
  fi
  competition_run_root=$(cd -- "$2" && pwd -P)
fi
run_name="${competition_run_root##*/}"
competition_run_directory="$competition_run_root/task_$task_index"
task_scratch="${TMPDIR:?Slurm must provide TMPDIR for task-local storage}/saps-competition-$job_id-$task_index"
mkdir -p "$task_scratch"
export PIP_CACHE_DIR="$task_scratch/pip-cache"
export VIRTUALENV_OVERRIDE_APP_DATA="$task_scratch/virtualenv-cache"

benchmark_args=(
  --resume
  --machine "$run_name-task-$task_index"
  --saps-dir "$competition_run_directory"
  --env-dir "$task_scratch/env"
  --results-dir "$competition_run_directory/results"
  --remote-storage-backend "$REMOTE_STORAGE_BACKEND"
  --remote-storage-bucket "$REMOTE_STORAGE_BUCKET"
  --chunk-count "$task_count"
  --chunk-index "$task_index"
)

if [[ -n "${SAPS_COMPETITION_ARGS:-}" ]]; then
  read -r -a extra_args <<< "$SAPS_COMPETITION_ARGS"
  benchmark_args+=("${extra_args[@]}")
fi

cd "$repo_directory"

set +e
poetry run ./bin/run_benchmark.py \
  --config "$competition_config" \
  "${benchmark_args[@]}"
benchmark_status=$?
set -e

poetry run ./scripts/combine_competition_results.py \
  --run-directory "$competition_run_root"

exit "$benchmark_status"
