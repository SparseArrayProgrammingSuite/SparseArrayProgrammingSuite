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

cd "$REPO_ROOT"

echo "Starting $JOBS scrape_matrices workers"

pids=()
for batch_index in $(seq 0 "$((JOBS - 1))"); do
  (
    poetry run python -u "$SCRAPER" \
      --num-batches "$JOBS" \
      --batch-index "$batch_index" \
      "$@" 2>&1 |
      sed -u "s/^/[batch ${batch_index}]: /"
  ) &
  pids+=("$!")
  echo "[batch ${batch_index}]: started"
done

status=0
for batch_index in "${!pids[@]}"; do
  if wait "${pids[$batch_index]}"; then
    echo "[batch ${batch_index}]: finished"
  else
    echo "[batch ${batch_index}]: failed" >&2
    status=1
  fi
done

exit "$status"
