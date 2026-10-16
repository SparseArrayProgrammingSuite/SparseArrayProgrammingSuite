#!/usr/bin/env bash
#SBATCH --job-name=graph-counts
#SBATCH --nodes=1
#SBATCH --mem=350G
#SBATCH --cpus-per-task=24
#SBATCH --time=08:00:00
#SBATCH --array=0-5
#SBATCH --output=graph-counts-%A_%a.log

set -euo pipefail

repo_directory="${SAPS_REPO_DIRECTORY:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_directory" ]]; then
  script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
  repo_directory=$(cd -- "$script_directory/.." && pwd)
fi
repo_directory=$(git -C "$repo_directory" rev-parse --show-toplevel)

: "${SLURM_ARRAY_TASK_ID:?Submit with sbatch scripts/compute-graph-counts.slurm}"

export PATH="$HOME/.local/bin:$PATH"
export NUMBA_NUM_THREADS="${SLURM_CPUS_PER_TASK:-24}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1

cd "$repo_directory"

dataset_names=(com-Friendster twitter7 GAP-twitter GAP-web GAP-kron GAP-urand)
if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= ${#dataset_names[@]} )); then
  echo "SLURM_ARRAY_TASK_ID must be between 0 and $((${#dataset_names[@]} - 1))" >&2
  exit 2
fi
dataset_name="${dataset_names[$SLURM_ARRAY_TASK_ID]}"
output_directory="${SAPS_GRAPH_COUNTS_OUTPUT_DIR:-$repo_directory/.saps/outputs/graph-counts}"
mkdir -p "$output_directory"

# Require the compiled counter for these large graphs.
poetry run python -c 'import numba'

# Each task saves its own result after both counts finish. Reusing the output
# directory skips completed graphs; interrupted graphs restart from the beginning.
echo "Computing graph counts for $dataset_name"
exec poetry run python ./scripts/compute_graph_counts.py \
  --only "$dataset_name" \
  --output "$output_directory/$dataset_name.json"
