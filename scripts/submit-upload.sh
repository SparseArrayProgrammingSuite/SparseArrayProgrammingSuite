#!/usr/bin/env bash
set -e

script_directory=$(cd -- "$(dirname -- "$0")" && pwd)

aws sso login --profile dataset-upload --use-device-code
sbatch "$script_directory/upload-dataset.slurm" "$1"
