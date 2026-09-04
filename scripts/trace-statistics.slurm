#!/usr/bin/env bash
#SBATCH --job-name=statistics-trace
#SBATCH --nodes=1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=trace-%A_%a.log

set -euo pipefail

repo_directory="${SAPS_REPO_DIRECTORY:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_directory" ]]; then
  script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
  repo_directory=$(cd -- "$script_directory/.." && pwd)
fi

: "${SAPS_TRACE_CHUNK_COUNT:?SAPS_TRACE_CHUNK_COUNT must be set}"
: "${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set}"

export PATH="$HOME/.local/bin:$PATH"
export AWS_PROFILE=dataset-upload
export REMOTE_STORAGE_BACKEND=s3
export REMOTE_STORAGE_BUCKET=sparse-array-programming-suite

trace_output_dir="${SAPS_TRACE_OUTPUT_DIR:-$repo_directory/.saps/outputs/trace-chunks}"
statistics_path="$trace_output_dir/statistics-$SLURM_ARRAY_TASK_ID.json"
tagger_stats_dir="$trace_output_dir/tagger_stats-$SLURM_ARRAY_TASK_ID"

cd "$repo_directory"

mkdir -p "$trace_output_dir"
rm -f "$statistics_path"
rm -rf "$tagger_stats_dir"

export SAPS_STATISTICS_PATH="$statistics_path"
export SAPS_TAGGER_STATS_DIR="$tagger_stats_dir"

poetry run ./bin/run_benchmark.py \
  --trace-statistics \
  --tag trace \
  --chunk-count "$SAPS_TRACE_CHUNK_COUNT" \
  --chunk-index "$SLURM_ARRAY_TASK_ID" \
  --timeout "${SAPS_TRACE_TIMEOUT:-30}" \
  --show-stderr \
  --remote-storage-backend "$REMOTE_STORAGE_BACKEND" \
  --remote-storage-bucket "$REMOTE_STORAGE_BUCKET"
