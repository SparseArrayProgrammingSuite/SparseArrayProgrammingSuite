#!/usr/bin/env bash
#SBATCH --job-name=graphminer-gap
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=350G
#SBATCH --cpus-per-task=24
#SBATCH --time=24:00:00
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
gcc_module="${GRAPHMINER_GCC_MODULE-gcc/11.3.1-bbxqwp5}"
if [[ -n "$cuda_module" || -n "$gcc_module" ]]; then
  if ! type module >/dev/null 2>&1; then
    for module_init in /etc/profile.d/lmod.sh /etc/profile.d/modules.sh /usr/share/Modules/init/bash; do
      if [[ -r "$module_init" ]]; then
        source "$module_init"
        break
      fi
    done
  fi
  if ! type module >/dev/null 2>&1; then
    echo "Cannot initialize environment modules; set GRAPHMINER_GCC_MODULE='' and GRAPHMINER_CUDA_MODULE='' and provide compatible GCC and CUDA explicitly." >&2
    exit 1
  fi
  if [[ -n "$gcc_module" ]]; then
    module load "$gcc_module"
  fi
fi
if [[ -n "$cuda_module" ]]; then
  module load "$cuda_module"
  # Derive the toolkit root from the newly loaded compiler, avoiding a stale
  # CUDA_HOME inherited from a different toolkit on the submission host.
  nvcc_path=$(command -v nvcc)
  export CUDA_HOME="$(dirname -- "$(dirname -- "$nvcc_path")")"
  export GRAPHMINER_CUDA_HOME="$CUDA_HOME"
fi
# Use the same host compiler for ordinary C++ and nvcc, even if another GCC is
# installed on the node. Print its identity in the job log for diagnosis.
export GRAPHMINER_CXX="$(command -v "${GRAPHMINER_CXX:-g++}")"
"$GRAPHMINER_CXX" --version
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
