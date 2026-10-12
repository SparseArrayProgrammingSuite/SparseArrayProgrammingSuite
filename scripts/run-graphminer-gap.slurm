#!/usr/bin/env bash
#SBATCH --job-name=graphminer-gap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=350G
#SBATCH --cpus-per-task=1
#SBATCH --qos=embers
#SBATCH --time=8:00:00
#SBATCH --output=graphminer-gap-%j.log

set -euo pipefail

repo_directory="${SAPS_REPO_DIRECTORY:-${SLURM_SUBMIT_DIR:-}}"
if [[ -z "$repo_directory" ]]; then
  script_directory=$(cd -- "$(dirname -- "$0")" && pwd)
  repo_directory=$(cd -- "$script_directory/.." && pwd)
fi
repo_directory=$(git -C "$repo_directory" rev-parse --show-toplevel)
export PATH="$HOME/.local/bin:$PATH"
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export PYTHONUNBUFFERED=1
cd "$repo_directory"
# Load the toolkit in the batch shell, not just on the submission host.
cuda_module="${GRAPHMINER_CUDA_MODULE-cuda/11.8.0}"
if [[ -n "$cuda_module" ]]; then
  if ! type module >/dev/null 2>&1; then
    for module_init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
      if [[ -r "$module_init" ]]; then
        source "$module_init"
        break
      fi
    done
  fi
  if ! type module >/dev/null 2>&1; then
    echo "Cannot initialize environment modules; set GRAPHMINER_CUDA_MODULE='' and provide CUDA_HOME or nvcc on PATH." >&2
    exit 1
  fi
  module load "$cuda_module"
  # Derive the toolkit root from the newly loaded compiler, avoiding a stale
  # CUDA_HOME inherited from a different toolkit on the submission host.
  nvcc_path=$(command -v nvcc)
  export CUDA_HOME="$(dirname -- "$(dirname -- "$nvcc_path")")"
  export GRAPHMINER_CUDA_HOME="$CUDA_HOME"
fi
# Build and validate on the GPU node.
make graphminer-gpu-check
revision=2a76e3f612e40e46a821d603ca11d10fcbc63ddd
gpu_directory="${GRAPHMINER_GPU_DIR:-$repo_directory/.saps/tools/GraphMiner-$revision-a100}"
exec poetry run python scripts/run_graphminer_gap.py \
  --graphminer-bin "$gpu_directory/bin/clique_gpu_base" \
  --datasets GAP-web GAP-kron GAP-urand \
  --threads "${SLURM_CPUS_PER_TASK:-24}" \
  --output-dir "${SAPS_GRAPHMINER_OUTPUT_DIR:-$repo_directory/.saps/outputs/graphminer-gap}" \
  --resume "$@"
