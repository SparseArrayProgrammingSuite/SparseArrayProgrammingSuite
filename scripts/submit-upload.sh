#!/usr/bin/env bash
set -e

script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
repo_directory=$(cd -- "$script_directory/.." && pwd)

export PATH="$HOME/.local/bin:$PATH"

if ! aws configure list-profiles | grep -Fxq dataset-upload; then
  aws configure sso --profile dataset-upload --use-device-code
fi

aws sso login --profile dataset-upload --use-device-code
cd "$repo_directory"
sbatch -A gts-wahrens6 -q embers -C amd "$script_directory/upload-dataset.slurm"
