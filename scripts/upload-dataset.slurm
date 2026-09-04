#!/usr/bin/env bash
#SBATCH --job-name=dataset-upload
#SBATCH --nodes=1
#SBATCH --mem=450G
#SBATCH --time=04:00:00
#SBATCH --output=upload-%j.log

set -e

repo_directory="${SAPS_REPO_DIRECTORY:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_directory" ]]; then
  script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
  repo_directory=$(cd -- "$script_directory/.." && pwd)
fi

export PATH="$HOME/.local/bin:$PATH"
export AWS_PROFILE=dataset-upload
export REMOTE_STORAGE_BACKEND=s3
export REMOTE_STORAGE_BUCKET=sparse-array-programming-suite

cd "$repo_directory"

poetry run ./bin/run_benchmark.py --cache-datasets
