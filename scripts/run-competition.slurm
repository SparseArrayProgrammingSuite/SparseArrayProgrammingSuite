#!/usr/bin/env bash
#SBATCH --job-name=saps-competition
#SBATCH --nodes=1
#SBATCH --mem=450G
#SBATCH --time=24:00:00
#SBATCH --output=/dev/null

set -euo pipefail

repo_directory="${SAPS_REPO_DIRECTORY:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_directory" ]]; then
  script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
  repo_directory=$(cd -- "$script_directory/.." && pwd)
fi

export PATH="$HOME/.local/bin:$PATH"
export AWS_PROFILE="${AWS_PROFILE:-dataset-upload}"
export REMOTE_STORAGE_BACKEND="${REMOTE_STORAGE_BACKEND:-s3}"
export REMOTE_STORAGE_BUCKET="${REMOTE_STORAGE_BUCKET:-sparse-array-programming-suite}"

competition_config="${SAPS_COMPETITION_CONFIG:-$repo_directory/competition.config.json}"

job_id="${SLURM_JOB_ID:-local}"
competition_directory="$repo_directory/competition"
competition_run_directory="$competition_directory/run_$job_id"
combined_results="$competition_directory/results_$job_id.json"

mkdir -p "$competition_run_directory/logs"
exec > >(tee -a "$competition_run_directory/logs/competition-$job_id.log") 2>&1

benchmark_args=(
  --saps-dir "$competition_run_directory"
  --env-dir "$competition_run_directory/env"
  --results-dir "$competition_run_directory/results"
  --remote-storage-backend "$REMOTE_STORAGE_BACKEND"
  --remote-storage-bucket "$REMOTE_STORAGE_BUCKET"
  --tags standard
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

poetry run ./bin/combine_competition_results.py \
  --run-directory "$competition_run_directory" \
  --output "$combined_results"

exit "$benchmark_status"
