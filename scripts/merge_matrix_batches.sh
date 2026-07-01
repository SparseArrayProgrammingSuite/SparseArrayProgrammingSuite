#!/usr/bin/env bash

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

if ! command -v jq >/dev/null 2>&1; then
  echo "jq is required to merge batch outputs" >&2
  exit 127
fi

solvers=(jacobi cg jacobi_cg block_jacobi_cg lsqr)

for solver in "${solvers[@]}"; do
  files=("${solver}"_batch_*_"${OUTPUT}")
  if [[ "${#files[@]}" -eq 0 ]]; then
    echo "No batch files found for $solver"
    continue
  fi

  target="${solver}_${OUTPUT}"
  key="${solver} convergence criteria"
  tmp="$(mktemp "${target}.tmp.XXXXXX")"

  jq -s --arg key "$key" '
    add
    | unique_by(.matrix_name)
    | sort_by(.[$key])
  ' "${files[@]}" >"$tmp"
  mv "$tmp" "$target"

  echo "Merged ${#files[@]} files into $target"
done
