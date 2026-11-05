#!/usr/bin/env bash
set -euo pipefail

script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
repo_directory=$(cd -- "$script_directory/.." && pwd)

export PATH="$HOME/.pixi/bin:$PATH"
cd "$repo_directory"

install_pixi_environment() {
  if ! command -v pixi >/dev/null 2>&1; then
    # The official installer, into ~/.pixi/bin, without editing shell profiles.
    curl -fsSL https://pixi.sh/install.sh |
      PIXI_VERSION=v0.83.0 PIXI_NO_PATH_UPDATE=1 bash
  fi
  # pixi.lock is not tracked; the generate feature's exact pins make
  # re-locking reproducible. pixi install re-locks when pixi.toml changed.
  pixi install --environment default
}

mkdir -p .saps/locks
if command -v flock >/dev/null 2>&1; then
  (
    flock 9
    install_pixi_environment
  ) 9>.saps/locks/pixi-install.lock
else
  install_pixi_environment
fi
