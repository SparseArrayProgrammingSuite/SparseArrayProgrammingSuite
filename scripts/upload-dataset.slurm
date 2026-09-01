#!/usr/bin/env bash
#SBATCH --job-name=dataset-upload
#SBATCH --nodes=1
#SBATCH --mem=700G
#SBATCH --time=04:00:00
#SBATCH --output=upload-%j.log

set -e

export PATH="$HOME/.local/bin:$PATH"
export AWS_PROFILE=dataset-upload
export REMOTE_STORAGE_BACKEND=s3
export REMOTE_STORAGE_BUCKET=sparse-array-programming-suite

poetry run ./bin/run_benchmark.py --cache-datasets
