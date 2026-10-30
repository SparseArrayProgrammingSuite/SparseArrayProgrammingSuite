#!/usr/bin/env bash
set -euo pipefail

script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
repo_directory=$(cd -- "$script_directory/.." && pwd)

export PATH="$HOME/.local/bin:$PATH"
cd "$repo_directory"

if ! command -v poetry >/dev/null 2>&1; then
  python3 -m pip install --user poetry
fi

install_poetry_environment() {
  # poetry.lock is not tracked; the generate group's exact pins make re-locking
  # reproducible.
  if ! poetry check --lock >/dev/null 2>&1; then
    poetry lock --no-interaction
  fi
  poetry sync --with generate --no-interaction
}

mkdir -p .saps/locks
if command -v flock >/dev/null 2>&1; then
  (
    flock 9
    install_poetry_environment
  ) 9>.saps/locks/poetry-install.lock
else
  install_poetry_environment
fi
