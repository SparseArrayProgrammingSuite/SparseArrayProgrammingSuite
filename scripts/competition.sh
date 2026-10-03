# Shared body of the competition Slurm scripts. Do not submit this file
# directly: submit scripts/competition-cpu.slurm or scripts/competition-gpu.slurm
# (or use scripts/submit-competition.sh to submit both). The wrappers carry the
# #SBATCH resources and source this file after setting:
#
#   repo_directory      repository root
#   competition_device  cpu or gpu; only frameworks for this device run
#
# Wrapper arguments:
#
#   --resume DIRECTORY  write into an existing run directory (also how the CPU
#                       and GPU arrays of one run share a directory)
#
# Every other argument goes to bin/run_benchmark.py, for example
# --tag suite-train to run the training set instead of the config's datasets.

export PATH="$HOME/.local/bin:$PATH"
export AWS_PROFILE="${AWS_PROFILE:-dataset-upload}"
export REMOTE_STORAGE_BACKEND="${REMOTE_STORAGE_BACKEND:-s3}"
export REMOTE_STORAGE_BUCKET="${REMOTE_STORAGE_BUCKET:-sparse-array-programming-suite}"

competition_config="${SAPS_COMPETITION_CONFIG:-$repo_directory/competition.config.json}"

job_id="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}"
task_index="${SLURM_ARRAY_TASK_ID:-0}"
# A partial rerun (e.g. --array=19-23) must keep the original chunk count, so
# SAPS_CHUNK_COUNT overrides the size of the submitted array.
task_count="${SAPS_CHUNK_COUNT:-${SLURM_ARRAY_TASK_COUNT:-1}}"
competition_run_root="$repo_directory/competition/run_$job_id"
forwarded_args=()
while (( $# > 0 )); do
  if [[ "$1" == --resume ]]; then
    (( $# >= 2 )) || { echo "--resume needs a run directory" >&2; exit 2; }
    competition_run_root=$(cd -- "$2" && pwd -P)
    shift 2
  else
    forwarded_args+=("$1")
    shift
  fi
done
run_name="${competition_run_root##*/}"
# CPU tasks keep the original task_<index> layout so older runs still resume.
task_label="$task_index"
if [[ "$competition_device" != cpu ]]; then
  task_label="${competition_device}_$task_index"
fi
competition_run_directory="$competition_run_root/task_$task_label"
task_scratch="${TMPDIR:?Slurm must provide TMPDIR for task-local storage}/saps-competition-$job_id-$task_index"
mkdir -p "$task_scratch"
export PIP_CACHE_DIR="$task_scratch/pip-cache"
export VIRTUALENV_OVERRIDE_APP_DATA="$task_scratch/virtualenv-cache"

benchmark_args=(
  --resume
  --machine "$run_name-task-$task_label"
  --saps-dir "$competition_run_directory"
  --env-dir "$task_scratch/env"
  --results-dir "$competition_run_directory/results"
  --memory-limit 128G
  --remote-storage-backend "$REMOTE_STORAGE_BACKEND"
  --remote-storage-bucket "$REMOTE_STORAGE_BUCKET"
  --chunk-count "$task_count"
  --chunk-index "$task_index"
  --device "$competition_device"
)
benchmark_args+=(${forwarded_args[@]+"${forwarded_args[@]}"})

if [[ -n "${SAPS_COMPETITION_ARGS:-}" ]]; then
  read -r -a extra_args <<< "$SAPS_COMPETITION_ARGS"
  benchmark_args+=("${extra_args[@]}")
fi

cd "$repo_directory"

set +e
poetry run ./bin/run_benchmark.py \
  --config "$competition_config" \
  "${benchmark_args[@]}"
benchmark_status=$?
set -e

poetry run ./scripts/combine_competition_results.py \
  --run-directory "$competition_run_root"

exit "$benchmark_status"
