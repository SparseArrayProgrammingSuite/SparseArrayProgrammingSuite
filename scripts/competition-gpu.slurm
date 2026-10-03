#!/usr/bin/env bash
# Competition array for GPU frameworks (SAPS_DEVICE=gpu in the config). Shared
# logic: scripts/competition.sh. Submit together with the CPU array using
# scripts/submit-competition.sh.
#SBATCH --job-name=saps-competition-gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:RTX_6000:1
#SBATCH --exclusive
#SBATCH --mem=160G
#SBATCH --tmp=50G
#SBATCH --time=08:00:00
#SBATCH --array=0-63
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

competition_device=gpu
source "$repo_directory/scripts/competition.sh"
