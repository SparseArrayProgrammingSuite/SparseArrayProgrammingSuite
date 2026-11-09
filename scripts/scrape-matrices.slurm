#!/usr/bin/env bash
#SBATCH --job-name=saps-scrape-matrices
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --array=0-31
#SBATCH --output=scrape-matrices-%A_%a.log

set -euo pipefail

repo_directory="${SAPS_REPO_DIRECTORY:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_directory" ]]; then
  script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
  repo_directory=$(cd -- "$script_directory/.." && pwd)
fi
repo_directory=$(git -C "$repo_directory" rev-parse --show-toplevel)

: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch scripts/scrape-matrices.slurm}"

export PATH="$HOME/.pixi/bin:$PATH"
export PYTHONUNBUFFERED=1

cd "$repo_directory"

# Each task scrapes every chunk_count-th SuiteSparse entry. Set
# SAPS_SCRAPE_CHUNK_COUNT to rerun part of an array with its original chunking.
chunk_index=$((SLURM_ARRAY_TASK_ID - ${SLURM_ARRAY_TASK_MIN:-0}))
chunk_count="${SAPS_SCRAPE_CHUNK_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
output_directory="${SAPS_SCRAPE_OUTPUT_DIR:-$repo_directory/.saps/outputs/scrape-matrices/run_$job_id}"
mkdir -p "$output_directory"

# Extra CLI arguments (such as --solver amg) pass through. Keep the chunk and
# output fixed per task so array workers cannot overwrite each other.
pixi run python ./scripts/scrape_matrices.py \
  "$@" \
  --chunk-count "$chunk_count" \
  --chunk-index "$chunk_index" \
  --output "$output_directory/results-$chunk_index.jsonl"
