#!/usr/bin/env bash
set -euo pipefail

script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
repo_directory=$(cd -- "$script_directory/.." && pwd)

export PATH="$HOME/.local/bin:$PATH"

if ! aws configure list-profiles | grep -Fxq dataset-upload; then
  aws configure sso --profile dataset-upload --use-device-code
fi

aws sso login --profile dataset-upload --use-device-code
cd "$repo_directory"

"$script_directory/ensure-poetry-env.sh"
poetry run ./bin/generate_metadata.py

account="${SAPS_SLURM_ACCOUNT:-gts-wahrens6}"
trace_chunk_count="${SAPS_TRACE_CHUNK_COUNT:-4}"
trace_array_end=$((trace_chunk_count - 1))

if ((trace_chunk_count < 1)); then
  echo "SAPS_TRACE_CHUNK_COUNT must be at least 1" >&2
  exit 1
fi

submit_job() {
  local job_id
  job_id=$(sbatch --parsable "$@")
  echo "${job_id%%;*}"
}

upload_job_id=$(
  submit_job \
    -A "$account" \
    -q embers \
    -C amd \
    --chdir "$repo_directory" \
    --export=ALL,SAPS_REPO_DIRECTORY="$repo_directory" \
    "$script_directory/upload-dataset.slurm"
)

trace_job_id=$(
  submit_job \
    -A "$account" \
    -p cpu-small \
    --dependency="afterok:$upload_job_id" \
    --array="0-$trace_array_end" \
    --chdir "$repo_directory" \
    --export=ALL,SAPS_TRACE_CHUNK_COUNT="$trace_chunk_count",SAPS_REPO_DIRECTORY="$repo_directory" \
    "$script_directory/trace-statistics.slurm"
)

merge_job_id=$(
  submit_job \
    -A "$account" \
    -p cpu-small \
    --dependency="afterok:$trace_job_id" \
    --chdir "$repo_directory" \
    --export=ALL,SAPS_TRACE_CHUNK_COUNT="$trace_chunk_count",SAPS_REPO_DIRECTORY="$repo_directory" \
    "$script_directory/merge-statistics.slurm"
)

cat <<EOF
submitted SAPS data refresh:
  upload:           $upload_job_id
  trace array:      $trace_job_id
  merge + metadata: $merge_job_id
EOF
