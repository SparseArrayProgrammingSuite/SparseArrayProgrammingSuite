#!/usr/bin/env bash
# Submit one competition run as two Slurm arrays that share a run directory:
# CPU frameworks on CPU nodes, and GPU frameworks (SAPS_DEVICE=gpu in the
# competition config) on GPU nodes. A final job combines both arrays' results.
#
# Arguments other than --resume and --after go to bin/run_benchmark.py in both
# arrays, for example: SAPS_CHUNK_COUNT=8 $0 --tag suite-train
set -euo pipefail

usage() {
  echo "usage: $0 [--resume RUN_DIRECTORY] [--after JOB_ID] [RUN_BENCHMARK_ARGS...]" >&2
  exit 2
}

run_directory=""
dependency=()
forwarded_args=()
while (($# > 0)); do
  case "$1" in
    --resume)
      (($# >= 2)) || usage
      run_directory=$(cd -- "$2" && pwd -P)
      shift 2
      ;;
    --after)
      (($# >= 2)) || usage
      dependency=(--dependency="afterok:$2")
      shift 2
      ;;
    *)
      forwarded_args+=("$1")
      shift
      ;;
  esac
done

submission_directory=$(pwd -P)
# Escape literal percent signs in Slurm filename patterns.
log_directory="${submission_directory//%/%%}"

script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
repo_directory=$(cd -- "$script_directory/.." && pwd)

account="${SAPS_SLURM_ACCOUNT:-gts-wahrens6}"
competition_config="${SAPS_COMPETITION_CONFIG:-$repo_directory/competition.config.json}"
chunk_count="${SAPS_CHUNK_COUNT:-64}"

if ((chunk_count < 1)); then
  echo "SAPS_CHUNK_COUNT must be at least 1" >&2
  exit 1
fi

# Skip the GPU array when no framework needs a GPU, rather than holding GPU
# nodes only for the runner to find nothing to do.
uses_gpu=$(
  python3 - "$competition_config" <<'EOF'
import json
import sys

with open(sys.argv[1]) as f:
    config = json.load(f)
print(
    "true"
    if any(
        include.get("env_nobuild", {}).get("SAPS_DEVICE") == "gpu"
        for include in config.get("include", [])
    )
    else "false"
)
EOF
)

submit_job() {
  local job_id
  job_id=$(sbatch --parsable "$@")
  echo "${job_id%%;*}"
}

submit_array() {
  local device=$1
  shift
  submit_job \
    -A "$account" \
    ${dependency[@]+"${dependency[@]}"} \
    --array="0-$((chunk_count - 1))" \
    --output "$log_directory/competition-$device-%A_%a.log" \
    --chdir "$repo_directory" \
    --export=ALL,SAPS_REPO_DIRECTORY="$repo_directory" \
    "$script_directory/competition-$device.slurm" \
    "$@" \
    ${forwarded_args[@]+"${forwarded_args[@]}"}
}

if [[ -n "$run_directory" ]]; then
  cpu_job_id=$(submit_array cpu --resume "$run_directory")
else
  # A new run is named after the CPU array; the GPU array joins it.
  cpu_job_id=$(submit_array cpu)
  run_directory="$repo_directory/competition/run_$cpu_job_id"
  mkdir -p "$run_directory"
fi

gpu_job_id=""
combine_after="afterany:$cpu_job_id"
if $uses_gpu; then
  gpu_job_id=$(submit_array gpu --resume "$run_directory")
  combine_after="$combine_after:$gpu_job_id"
fi

combine_command="export PATH=\"\$HOME/.local/bin:\$PATH\" && poetry run ./scripts/combine_competition_results.py --run-directory $(printf %q "$run_directory")"
combine_job_id=$(
  submit_job \
    -A "$account" \
    -p cpu-small \
    --job-name=saps-combine \
    --mem=16G \
    --time=00:30:00 \
    --dependency="$combine_after" \
    --output "$log_directory/competition-combine-%j.log" \
    --chdir "$repo_directory" \
    --wrap "$combine_command"
)

cat <<EOF
submitted SAPS competition:
  run directory: $run_directory
  cpu array:     $cpu_job_id
  gpu array:     ${gpu_job_id:-skipped (no SAPS_DEVICE=gpu frameworks in $competition_config)}
  combine:       $combine_job_id
EOF
