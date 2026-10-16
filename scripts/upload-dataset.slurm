#!/usr/bin/env bash
#SBATCH --job-name=dataset-upload
#SBATCH --nodes=1
#SBATCH --mem=350G
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --array=0-7
#SBATCH --output=upload-%A_%a.log

set -e

repo_directory="${SAPS_REPO_DIRECTORY:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_directory" ]]; then
  script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
  repo_directory=$(cd -- "$script_directory/.." && pwd)
fi
repo_directory=$(git -C "$repo_directory" rev-parse --show-toplevel)

export PATH="$HOME/.local/bin:$PATH"
export AWS_PROFILE=dataset-upload
export REMOTE_STORAGE_BACKEND=s3
export REMOTE_STORAGE_BUCKET=sparse-array-programming-suite

cd "$repo_directory"

poetry run ./bin/run_benchmark.py \
  --cache-datasets \
  --chunk-count "${SLURM_ARRAY_TASK_COUNT:-1}" \
  --chunk-index "${SLURM_ARRAY_TASK_ID:-0}"
