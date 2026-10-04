#!/usr/bin/env bash
# Competition array for GPU frameworks (SAPS_DEVICE=gpu in the config). Shared
# logic: scripts/competition.sh. Submit together with the CPU array using
# scripts/submit-competition.sh.
#SBATCH --job-name=saps-competition-gpu
#SBATCH --nodes=1
# Take a whole V100 node (2 GPUs, 24 cores) so no other job shares it while
# timing, like the CPU array. Phoenix allows at most 12 CPUs per V100, so 24
# CPUs need both GPUs; frameworks only use the first one.
#SBATCH --cpus-per-task=24
#SBATCH --gres=gpu:V100:2
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
