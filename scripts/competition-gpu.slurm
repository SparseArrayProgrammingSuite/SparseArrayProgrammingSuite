#!/usr/bin/env bash
# Competition array for GPU frameworks (SAPS_DEVICE=gpu in the config). Shared
# logic: scripts/competition.sh. Submit together with the CPU array using
# scripts/submit-competition.sh.
#SBATCH --job-name=saps-competition-gpu
#SBATCH --nodes=1
# Take a whole V100 node (2 GPUs, 24 cores) so no other job shares it while
# timing, like the CPU array. Phoenix allows at most 12 CPUs per V100, so 24
# CPUs need both GPUs; frameworks only use the first one. Jobs die in the A100
# queue, so stay on V100s.
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:V100:2
#SBATCH --exclusive
#SBATCH --mem=160G
#SBATCH --tmp=50G
# 32 tasks rather than the CPU array's 128, to hold fewer V100 nodes, so each
# task runs ~200 standard datasets of up to 180 s.
#SBATCH --time=12:00:00
#SBATCH --array=0-31
#SBATCH --output=competition-gpu-%A_%a.log
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=ahrens@gatech.edu

set -euo pipefail

# Slurm runs a spooled copy of this script, so find the repository from the
# submission directory rather than from $0.
repo_directory="${SAPS_REPO_DIRECTORY:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_directory" ]]; then
  script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
  repo_directory=$(cd -- "$script_directory/.." && pwd)
fi
repo_directory=$(git -C "$repo_directory" rev-parse --show-toplevel)

# PyPI's torch is a CUDA 13 build, which has no V100 (sm_70) kernels
# (cudaErrorNoKernelImageForDevice); the cu126 build is the one that does, and
# pip prefers its 2.12.1+cu126 over 2.12.1.
export PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu126

competition_device=gpu
source "$repo_directory/scripts/competition.sh"
