#!/usr/bin/env bash

set -euo pipefail

JOBS="${JOBS:-8}"

if ! [[ "$JOBS" =~ ^[1-9][0-9]*$ ]]; then
  echo "JOBS must be a positive integer, got: $JOBS" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
SCRAPER="$REPO_ROOT/src/saps/utils/scrape_matrices.py"
LOG_DIR="${LOG_DIR:-$REPO_ROOT/src/saps/utils/scrape_logs}"
RUN_ID="$(date +%Y%m%d_%H%M%S)"

cd "$REPO_ROOT/src/saps/utils"
mkdir -p "$LOG_DIR"

echo "Starting $JOBS scrape_matrices workers"
echo "Writing logs to $LOG_DIR"

pids=()
log_files=()
for batch_index in $(seq 0 "$((JOBS - 1))"); do
  log_file="$LOG_DIR/scrape_matrices_batch_${batch_index}_${RUN_ID}.log"
  (
    echo "[batch ${batch_index}]: started"
    echo "[batch ${batch_index}]: log file: $log_file"
    poetry run python -u "$SCRAPER" \
      --num-batches "$JOBS" \
      --batch-index "$batch_index" \
      "$@"
  ) >"$log_file" 2>&1 &
  pids+=("$!")
  log_files+=("$log_file")
  echo "[batch ${batch_index}]: started; log: $log_file"
done

status=0
for batch_index in "${!pids[@]}"; do
  if wait "${pids[$batch_index]}"; then
    echo "[batch ${batch_index}]: finished; log: ${log_files[$batch_index]}"
  else
    echo "[batch ${batch_index}]: failed; log: ${log_files[$batch_index]}" >&2
    status=1
  fi
done

exit "$status"
