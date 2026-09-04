#!/usr/bin/env bash
#SBATCH --job-name=metadata-finalize
#SBATCH --nodes=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00
#SBATCH --output=finalize-metadata-%j.log

set -euo pipefail

repo_directory="${SAPS_REPO_DIRECTORY:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_directory" ]]; then
  script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
  repo_directory=$(cd -- "$script_directory/.." && pwd)
fi

: "${SAPS_TRACE_CHUNK_COUNT:?SAPS_TRACE_CHUNK_COUNT must be set}"

trace_output_dir="${SAPS_TRACE_OUTPUT_DIR:-$repo_directory/.saps/outputs/trace-chunks}"
statistics_output="${SAPS_STATISTICS_OUTPUT:-$repo_directory/statistics.json}"
statistics_paths=()

for ((chunk = 0; chunk < SAPS_TRACE_CHUNK_COUNT; chunk++)); do
  statistics_path="$trace_output_dir/statistics-$chunk.json"
  if [[ -s "$statistics_path" ]]; then
    statistics_paths+=("$statistics_path")
  else
    echo "missing or empty trace statistics: $statistics_path" >&2
  fi
done

if [[ ${#statistics_paths[@]} -eq 0 ]]; then
  echo "no trace statistics files found in $trace_output_dir" >&2
  exit 1
fi

export PATH="$HOME/.local/bin:$PATH"
cd "$repo_directory"

poetry run ./bin/merge_statistics.py \
  --output "$statistics_output" \
  "${statistics_paths[@]}"
poetry run ./bin/generate_metadata.py --statistics "$statistics_output"
