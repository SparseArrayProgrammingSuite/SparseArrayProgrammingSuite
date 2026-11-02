#!/usr/bin/env bash
#SBATCH --job-name=dataset-upload
#SBATCH --nodes=1
#SBATCH --partition=cpu-large
#SBATCH --mem=720G
#SBATCH --cpus-per-task=24
#SBATCH --time=3-00:00:00
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
# GraphBLAS references use OpenMP, which should match the CPUs Slurm grants.
if [[ -n "${SLURM_CPUS_PER_TASK:-}" ]]; then
  export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"
fi

cd "$repo_directory"

# A per-process cap makes oversized datasets fail cleanly instead of tripping
# the Slurm OOM killer, which would skip the rest of the chunk.
memory_limit_args=()
if [[ -n "${SAPS_MEMORY_LIMIT:-}" ]]; then
  memory_limit_args=(--memory-limit "$SAPS_MEMORY_LIMIT")
fi

poetry run ./bin/run_benchmark.py \
  --cache-datasets \
  --process-timeout 3600 \
  --chunk-count "${SLURM_ARRAY_TASK_COUNT:-1}" \
  --chunk-index "${SLURM_ARRAY_TASK_ID:-0}" \
  "${memory_limit_args[@]}"
