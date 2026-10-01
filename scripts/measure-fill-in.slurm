#!/usr/bin/env bash
#SBATCH --job-name=saps-fill-in
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --array=0-72
#SBATCH --output=fill-in-%A_%a.log

set -euo pipefail

repo_directory="${SAPS_REPO_DIRECTORY:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_directory" ]]; then
  script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
  repo_directory=$(cd -- "$script_directory/.." && pwd)
fi
repo_directory=$(git -C "$repo_directory" rev-parse --show-toplevel)

: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch scripts/measure-fill-in.slurm}"

export PATH="$HOME/.local/bin:$PATH"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$repo_directory"

# One task per GAP and SNAP matrix in the manifest (73 when this was written).
mapfile -t dataset_names < <(
  poetry run python ./scripts/measure_fill_in.py --datasets GAP SNAP --list-datasets
)
if (( SLURM_ARRAY_TASK_ID >= ${#dataset_names[@]} )); then
  echo "SLURM_ARRAY_TASK_ID must be below ${#dataset_names[@]}" >&2
  exit 2
fi
dataset_name="${dataset_names[$SLURM_ARRAY_TASK_ID]}"
job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
output_directory="${SAPS_FILL_IN_OUTPUT_DIR:-$repo_directory/.saps/outputs/fill-in/run_$job_id}"

# Extra CLI arguments can override sampling and traversal budgets. Keep the
# dataset and output fixed per task so array workers cannot overwrite each other.
poetry run python ./scripts/measure_fill_in.py \
  --samples 1024 \
  --seconds 5400 \
  "$@" \
  --datasets "$dataset_name" \
  --output "$output_directory/$dataset_name.json"
