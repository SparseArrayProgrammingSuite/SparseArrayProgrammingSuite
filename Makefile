.DEFAULT_GOAL := help

GRAPHMINER_REV := 2a76e3f612e40e46a821d603ca11d10fcbc63ddd
GRAPHMINER_DIR ?= $(CURDIR)/.saps/tools/GraphMiner-$(GRAPHMINER_REV)
GRAPHMINER_BIN := $(GRAPHMINER_DIR)/bin/clique_omp_base
GRAPHMINER_CXX ?= g++
GRAPHMINER_CUDA_MODULE ?= cuda/11.8.0
GRAPHMINER_GPU_DIR ?= $(CURDIR)/.saps/tools/GraphMiner-$(GRAPHMINER_REV)-a100
GRAPHMINER_GPU_BIN := $(GRAPHMINER_GPU_DIR)/bin/clique_gpu_base
# Set CUDA_HOME explicitly, or load a CUDA module that puts nvcc on PATH.
GRAPHMINER_CUDA_HOME ?= $(if $(CUDA_HOME),$(CUDA_HOME),$(shell command -v nvcc 2>/dev/null | sed 's,/bin/nvcc$$,,'))
SBATCH_ARGS ?=
THREADS ?= 24
DATASETS ?= GAP-web GAP-kron GAP-urand
GRAPHMINER_ARGS ?=
POETRY ?= poetry

.PHONY: help graphminer graphminer-check graphminer-gap graphminer-gap-submit graphminer-gpu graphminer-gpu-check

help:
	@echo 'make graphminer             Download pinned GraphMiner and build its CPU counter'
	@echo 'make graphminer-check       Build and check small graphs against brute force'
	@echo 'make graphminer-gap         Build and compute missing GAP four-clique counts'
	@echo 'make graphminer-gpu         Build the A100 CUDA counter'
	@echo 'make graphminer-gpu-check   Build and check the CUDA counter (requires a GPU)'
	@echo 'make graphminer-gap-submit  Submit one A100 job for all three missing counts'
	@echo 'Overrides: GRAPHMINER_CXX=g++-15 THREADS=24 DATASETS=GAP-road GRAPHMINER_ARGS="..."'

$(GRAPHMINER_DIR)/.saps-ready $(GRAPHMINER_GPU_DIR)/.saps-ready:
	@mkdir -p "$(dir $(@D))"
	@if [ ! -d "$(@D)/.git" ]; then \
		git clone https://github.com/chenxuhao/GraphMiner.git "$(@D)"; \
	fi
	git -C "$(@D)" checkout --detach "$(GRAPHMINER_REV)"
	@touch "$@"

$(GRAPHMINER_BIN): $(GRAPHMINER_DIR)/.saps-ready
	$(MAKE) -C "$(GRAPHMINER_DIR)/src/clique" clique_omp_base CXX="$(GRAPHMINER_CXX)"

graphminer: $(GRAPHMINER_BIN)

graphminer-check: graphminer
	GRAPHMINER_BIN="$(GRAPHMINER_BIN)" $(POETRY) run pytest tests/test_graphminer_gap.py -o addopts= -q

graphminer-gap: graphminer
	$(POETRY) run python scripts/run_graphminer_gap.py \
		--graphminer-bin "$(GRAPHMINER_BIN)" --threads "$(THREADS)" \
		--datasets $(DATASETS) --resume $(GRAPHMINER_ARGS)

$(GRAPHMINER_GPU_BIN): $(GRAPHMINER_GPU_DIR)/.saps-ready
	@test -x "$(GRAPHMINER_CUDA_HOME)/bin/nvcc" || { echo 'Load CUDA or set CUDA_HOME before building GraphMiner GPU' >&2; exit 1; }
	$(MAKE) -C "$(GRAPHMINER_GPU_DIR)/src/clique" clique_gpu_base \
		CXX="$(GRAPHMINER_CXX)" CUDA_HOME="$(GRAPHMINER_CUDA_HOME)" \
		NVCC="$(GRAPHMINER_CUDA_HOME)/bin/nvcc" \
		CUDA_ARCH='-gencode arch=compute_80,code=sm_80 -gencode arch=compute_80,code=compute_80'

graphminer-gpu: $(GRAPHMINER_GPU_BIN)

graphminer-gpu-check: graphminer-gpu
	GRAPHMINER_BIN="$(GRAPHMINER_GPU_BIN)" $(POETRY) run pytest tests/test_graphminer_gap.py -o addopts= -q

# Build on the allocated node so CUDA and the CPU architecture match the job.
graphminer-gap-submit:
	GRAPHMINER_GPU_DIR="$(GRAPHMINER_GPU_DIR)" GRAPHMINER_CXX="$(GRAPHMINER_CXX)" \
		GRAPHMINER_CUDA_MODULE="$(GRAPHMINER_CUDA_MODULE)" \
		sbatch $(SBATCH_ARGS) scripts/run-graphminer-gap.slurm $(GRAPHMINER_ARGS)
