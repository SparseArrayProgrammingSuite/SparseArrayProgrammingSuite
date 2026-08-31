#!/usr/bin/env bash
#SBATCH --job-name=dataset-upload
#SBATCH --nodes=1
#SBATCH --mem=450G
#SBATCH --time=04:00:00
#SBATCH --output=upload-%j.log

set -e

source_directory=$1
dataset_name=$(basename "$source_directory")
destination="s3://my-dataset-bucket/datasets/$dataset_name/"

export AWS_PROFILE=dataset-upload

aws s3 sync "$source_directory/" "$destination"
