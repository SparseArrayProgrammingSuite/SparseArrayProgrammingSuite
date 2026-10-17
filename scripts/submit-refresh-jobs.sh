#!/usr/bin/env bash
set -euo pipefail

with_competition=false
while (($# > 0)); do
  case "$1" in
    --with-competition)
      with_competition=true
      shift
      ;;
    *)
      echo "usage: $0 [--with-competition]" >&2
      exit 2
      ;;
  esac
done

submission_directory=$(pwd -P)
# Escape literal percent signs in Slurm filename patterns.
log_directory="${submission_directory//%/%%}"

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
upload_chunk_count="${SAPS_UPLOAD_CHUNK_COUNT:-8}"
trace_chunk_count="${SAPS_TRACE_CHUNK_COUNT:-8}"
trace_array_end=$((trace_chunk_count - 1))

if ((upload_chunk_count < 1)); then
  echo "SAPS_UPLOAD_CHUNK_COUNT must be at least 1" >&2
  exit 1
fi

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
    --array="0-$((upload_chunk_count - 1))" \
    --output "$log_directory/upload-%A_%a.log" \
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
    --output "$log_directory/trace-%A_%a.log" \
    --chdir "$repo_directory" \
    --export=ALL,SAPS_TRACE_CHUNK_COUNT="$trace_chunk_count",SAPS_REPO_DIRECTORY="$repo_directory" \
    "$script_directory/trace-statistics.slurm"
)

merge_job_id=$(
  submit_job \
    -A "$account" \
    -p cpu-small \
    --dependency="afterok:$trace_job_id" \
    --output "$log_directory/finalize-metadata-%j.log" \
    --chdir "$repo_directory" \
    --export=ALL,SAPS_TRACE_CHUNK_COUNT="$trace_chunk_count",SAPS_REPO_DIRECTORY="$repo_directory" \
    "$script_directory/finalize-metadata.slurm"
)

cat <<EOF
submitted SAPS data refresh:
  upload array:     $upload_job_id
  trace array:      $trace_job_id
  merge + metadata: $merge_job_id
EOF

if $with_competition; then
  "$script_directory/submit-competition.sh" --after "$merge_job_id"
fi
