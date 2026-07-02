#!/usr/bin/env bash

if [ -z "${BASH_VERSION:-}" ]; then
  exec bash "$0" "$@"
fi

set -euo pipefail
shopt -s nullglob

usage() {
  echo "Usage: $0 [--output matrices.json]"
}

OUTPUT="matrices.json"
if [[ $# -gt 0 ]]; then
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --output)
      if [[ $# -ne 2 ]]; then
        usage >&2
        exit 2
      fi
      OUTPUT="$2"
      ;;
    *)
      if [[ $# -ne 1 ]]; then
        usage >&2
        exit 2
      fi
      OUTPUT="$1"
      ;;
  esac
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to merge batch outputs" >&2
  exit 127
fi

cd "$SCRIPT_DIR"

solvers=(jacobi cg jacobi_cg block_jacobi_cg lsqr)

for solver in "${solvers[@]}"; do
  files=("${solver}"_batch_*_"${OUTPUT}")
  if [[ "${#files[@]}" -eq 0 ]]; then
    echo "No batch files found for $solver"
    continue
  fi

  target="${solver}_${OUTPUT}"
  key="${solver} iterations"
  tmp="$(mktemp "${target}.tmp.XXXXXX")"

  jq -s --arg key "$key" '
    add
    | map(select(has($key)))
    | unique_by(.matrix_name)
    | sort_by(.[$key], .matrix_name)
  ' "${files[@]}" >"$tmp"
  count="$(jq length "$tmp")"
  if [[ "$count" -eq 0 ]]; then
    rm "$tmp"
    echo "No iteration-stat batch entries found for $solver"
    continue
  fi
  mv "$tmp" "$target"

  echo "Merged ${#files[@]} files into $target ($count entries)"
done
