"""
Name: SUMMA Matrix Multiplication
Author: Kseniia Suleimanova
Email: suleimanovakr@gmail.com
Motivation:
"The Scalable Universal Matrix Multiplication Algorithm (SUMMA) is a
fundamental building block for distributed linear algebra. It achieves
near-optimal communication complexity for dense matrix multiplication and
underpins libraries such as ScaLAPACK and SLATE. Its sparse counterpart,
SpSUMMA, extends these communication bounds to sparse matrices and is
critical for large-scale graph analytics and scientific simulation."
R. A. Van De Geijn and J. Watts, "SUMMA: Scalable Universal Matrix
Multiplication Algorithm", Concurrency: Practice and Experience, vol. 9,
no. 4, pp. 255-274, 1997.
G. Ballard et al., "Communication Optimal Parallel Multiplication of Sparse
Random Matrices", UCB/EECS-2013-13, 2013.
Role of sparsity:
When the input matrices are sparse, the benchmark exercises SpGEMM kernels
that skip zero entries. For an n×n Erdos-Renyi matrix with degree parameter
d (each entry nonzero with probability d/n), the output has an expected
O(d^2·n) nonzeros instead of n^2, and the communication volume falls from
O(n^2/sqrt(P)) to O(d·n/sqrt(P)) per process. The data generators below
cover both regimes so implementations can be evaluated on sparse inputs
(ER random) and dense inputs (full random normal) with the same algorithm.
Implementation:
Handwritten code based on:
  Standard SpSUMMA  — Ballard et al. 2013, Section 4.2.1
  Improved SpSUMMA  — Ballard et al. 2013, Section 4.2.2
  Dense SUMMA       — Van De Geijn & Watts 1997
Each benchmark function accepts full matrices on rank 0, distributes blocks
across the sqrt(P) × sqrt(P) processor grid, runs the algorithm, and
reassembles the result on rank 0 so callers can verify correctness against
a serial reference. Works with P=1 (plain pytest) and P=4,16,64,… (mpirun).
Reference implementations used and translated:
https://github.com/SparseArrayProgrammingSuite/SparseArrayProgrammingSuite
Data Generation:
Sparse inputs: n×n Erdos-Renyi ER(d) matrices in scipy CSR format,
  each entry nonzero with probability d/n, values drawn from N(0,1).
Dense inputs: n×n numpy float32 matrices, entries drawn from N(0,1).
Statement on the use of Generative AI:
No generative AI was used to construct the benchmark function itself.
Generative AI was used to help restructure this file to match the
SparseArrayProgrammingSuite benchmark style. This statement was written
by hand.

SLURM script for running scaling experiments on PACE Phoenix (Georgia Tech):
-------------------------------------------------------------------------------
#!/bin/bash
#SBATCH --job-name=summa_bench
#SBATCH --output=summa_bench_%j.out
#SBATCH --error=summa_bench_%j.err
#SBATCH --time=04:00:00
#SBATCH --nodes=16               # enough for 256-proc runs (16 nodes x 16 cores)
#SBATCH --ntasks-per-node=16     # MPI ranks per node
#SBATCH --cpus-per-task=1        # pure MPI, no threading
#SBATCH --mem-per-cpu=4G
#SBATCH --partition=cpu-medium   # Phoenix: cpu-small / cpu-medium / cpu-large
#SBATCH --account=YOUR_ACCOUNT   # find with: pace-whoami
#
# One-time setup (run before submitting):
#   module load python/3.10.10 openmpi/4.1.5
#   python -m venv ~/envs/summa_env
#   source ~/envs/summa_env/bin/activate
#   pip install mpi4py scipy numpy
#
# Dense memory note: blocks are n_block^2 * 8 bytes.
#   P=1  -> n_block=20000 -> ~3.2 GB/matrix -> skip
#   P=4  -> n_block=10000 -> ~800 MB/matrix -> skip
#   P=16 -> n_block=5000  -> ~200 MB/matrix -> safe
#   Dense strong-scaling therefore starts at P=16.
#   Dense weak-scaling uses n0=1000/proc (blocks ~8 MB at all P).
#
# module purge
# module load python/3.10.10 openmpi/4.1.5
# source ~/envs/summa_env/bin/activate
#
# SCRIPT="$(dirname "$0")/summa_benchmark.py"
# RESULTS_DIR="/storage/scratch1/0/${USER}/summa_results_${SLURM_JOB_ID}"
# mkdir -p "${RESULTS_DIR}"
#
# run_sparse() {
#     local nprocs=$1 n=$2 d=$3 variant=$4 nreps=$5
#     srun -n "${nprocs}" python "${SCRIPT}" \
#         -n "${n}" -d "${d}" --nreps "${nreps}" \
#         --variant "${variant}" --matrix-type sparse \
#         2>&1 | tee -a "${RESULTS_DIR}/sparse_n${n}_d${d}_P${nprocs}.txt"
# }
#
# run_dense() {
#     local nprocs=$1 n=$2 nreps=$3
#     srun -n "${nprocs}" python "${SCRIPT}" \
#         -n "${n}" --nreps "${nreps}" --matrix-type dense \
#         2>&1 | tee -a "${RESULTS_DIR}/dense_n${n}_P${nprocs}.txt"
# }
#
# # Experiment 1: Strong scaling (n fixed, vary P)
# for P in 1 4 16 64 256; do run_sparse ${P} 20000 20 both 5; done
# for P in 16 64 256;      do run_dense  ${P} 20000       5; done
#
# # Experiment 2: Weak scaling (n grows with P, block size fixed)
# for P in 1 4 16 64 256; do
#     sqrtP=$(python3 -c "import math; print(int(math.sqrt(${P})))")
#     run_sparse ${P} $((2000 * sqrtP)) 20 both 5
#     run_dense  ${P} $((1000 * sqrtP))    5
# done
#
# # Experiment 3: Density sweep — sparse only (n=20000, P=64, vary d)
# for D in 5 10 20 40 80; do run_sparse 64 20000 ${D} both 5; done
-------------------------------------------------------------------------------
"""

import numpy as np
import scipy.sparse as sp
from mpi4py import MPI


# ---------------------------------------------------------------------------
# Sparse helpers
# ---------------------------------------------------------------------------

# converts scipy sparse martix to numpy so that mpi can be used
def _pack_csr(mat):
    mat = mat.tocsr()
    return {"data": mat.data, "indices": mat.indices,
            "indptr": mat.indptr, "shape": mat.shape}

# reverses the action of _pack_csr
def _unpack_csr(d):
    return sp.csr_matrix(
        (d["data"], d["indices"], d["indptr"]),
        shape=d["shape"], dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# SpSUMMA – standard (Ballard et al. 2013, Section 4.2.1)
# ---------------------------------------------------------------------------

def _spSUMMA_standard(A_local, B_local, comm_row, comm_col,
                      rank_row, rank_col, sqrtP):
    C_local = None
    for k in range(sqrtP):
        A_pkg = _pack_csr(A_local) if rank_col == k else None
        A_pkg = comm_row.bcast(A_pkg, root=k)
        A_bcast = _unpack_csr(A_pkg)

        B_pkg = _pack_csr(B_local) if rank_row == k else None
        B_pkg = comm_col.bcast(B_pkg, root=k)
        B_bcast = _unpack_csr(B_pkg)

        contrib = (A_bcast @ B_bcast).tocsr()
        C_local = contrib if C_local is None else C_local + contrib

    return C_local


# ---------------------------------------------------------------------------
# SpSUMMA – improved (Ballard et al. 2013, Section 4.2.2)
# ---------------------------------------------------------------------------

def _spSUMMA_improved(A_local, B_local, comm_row, comm_col,
                      rank_row, rank_col, sqrtP):
    # every processor sends its A block to all others in its row and receives all of theirs back
    all_A = comm_row.allgather(_pack_csr(A_local))
    all_B = comm_col.allgather(_pack_csr(B_local))

    # stack received blocks to one matrix
    A_row = sp.hstack([_unpack_csr(p) for p in all_A], format="csr")
    B_col = sp.vstack([_unpack_csr(p) for p in all_B], format="csr")
    return (A_row @ B_col).tocsr()


# ---------------------------------------------------------------------------
# Dense SUMMA (Van De Geijn & Watts 1997)
# ---------------------------------------------------------------------------

def _dense_SUMMA(A_local, B_local, comm_row, comm_col,
                 rank_row, rank_col, sqrtP):
    n_r, n_c = A_local.shape
    C_local = np.zeros((n_r, n_c), dtype=np.float64)
    for k in range(sqrtP):
        A_bcast = A_local.copy() if rank_col == k else np.empty((n_r, n_c), dtype=np.float64)
        comm_row.Bcast(A_bcast, root=k)
        B_bcast = B_local.copy() if rank_row == k else np.empty((n_r, n_c), dtype=np.float64)
        comm_col.Bcast(B_bcast, root=k)
        C_local += A_bcast @ B_bcast
    return C_local


# ---------------------------------------------------------------------------
# Internal: build 2-D communicator grid
# ---------------------------------------------------------------------------

def _make_grid(comm, sqrtP): # 
    rank = comm.Get_rank()
    rank_row = rank // sqrtP
    rank_col = rank % sqrtP
    comm_row = comm.Split(rank_row, rank_col)
    comm_col = comm.Split(rank_col, rank_row)
    return rank_row, rank_col, comm_row, comm_col


# ---------------------------------------------------------------------------
# Public benchmark functions
# ---------------------------------------------------------------------------

"""
benchmark_summa_sparse(A, B, variant)

Distributed sparse matrix multiply C = A @ B using SpSUMMA.
Distributes A and B across MPI_COMM_WORLD, runs the chosen SpSUMMA
variant, and returns the assembled result on rank 0 (None on other ranks).

Args:
----
A: scipy sparse matrix (n×n, CSR/CSC/COO), provided on rank 0
B: scipy sparse matrix (n×n, CSR/CSC/COO), provided on rank 0
variant: "standard" (sqrt(P) broadcast stages, O(sqrt(P)) latency) or
         "improved" (allgather upfront, O(log P) latency)

Returns:
-------
C: scipy CSR matrix (n×n) on rank 0, None on other ranks
"""


def benchmark_summa_sparse(A, B, variant="standard"):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    P = comm.Get_size()
    sqrtP = int(round(P ** 0.5))
    assert sqrtP * sqrtP == P, f"P={P} must be a perfect square"

    n = comm.bcast(A.shape[0] if rank == 0 else None, root=0)
    n_block = n // sqrtP

    # Scatter blocks: rank 0 slices the grid and sends one block per process
    # Initially rank 0 has all the values
    if rank == 0:
        A_csr = A.tocsr()
        B_csr = B.tocsr()
        A_blocks = [
            A_csr[i * n_block:(i + 1) * n_block, j * n_block:(j + 1) * n_block]
            for i in range(sqrtP) for j in range(sqrtP)
        ]
        B_blocks = [
            B_csr[i * n_block:(i + 1) * n_block, j * n_block:(j + 1) * n_block]
            for i in range(sqrtP) for j in range(sqrtP)
        ]
    else:
        A_blocks = None
        B_blocks = None

    A_local = comm.scatter(A_blocks, root=0)
    B_local = comm.scatter(B_blocks, root=0)

    rank_row, rank_col, comm_row, comm_col = _make_grid(comm, sqrtP)

    algo = _spSUMMA_standard if variant == "standard" else _spSUMMA_improved
    C_local = algo(A_local, B_local, comm_row, comm_col,
                   rank_row, rank_col, sqrtP)

    # Gather and reassemble on rank 0
    all_C = comm.gather(C_local, root=0)

    if rank == 0:
        rows = [
            sp.hstack([all_C[i * sqrtP + j] for j in range(sqrtP)], format="csr")
            for i in range(sqrtP)
        ]
        return sp.vstack(rows, format="csr")
    # now rank 0 has resulting matrix
    return None


"""
benchmark_summa_dense(A, B)

Distributed dense matrix multiply C = A @ B using SUMMA.
Distributes A and B across MPI_COMM_WORLD, runs dense SUMMA,
and returns the assembled result on rank 0 (None on other ranks).

Args:
----
A: numpy ndarray (n×n, float64), provided on rank 0
B: numpy ndarray (n×n, float64), provided on rank 0

Returns:
-------
C: numpy ndarray (n×n, float64) on rank 0, None on other ranks
"""


def benchmark_summa_dense(A, B):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    P = comm.Get_size()
    sqrtP = int(round(P ** 0.5))
    assert sqrtP * sqrtP == P, f"P={P} must be a perfect square"

    n = comm.bcast(A.shape[0] if rank == 0 else None, root=0)
    n_block = n // sqrtP

    # Scatter using buffer-based Scatter for efficiency
    if rank == 0:
        A_send = np.ascontiguousarray(
            A.reshape(sqrtP, n_block, sqrtP, n_block)
             .transpose(0, 2, 1, 3)
             .reshape(P, n_block, n_block),
            dtype=np.float64,
        )
        B_send = np.ascontiguousarray(
            B.reshape(sqrtP, n_block, sqrtP, n_block)
             .transpose(0, 2, 1, 3)
             .reshape(P, n_block, n_block),
            dtype=np.float64,
        )
    else:
        A_send = None
        B_send = None

    A_local = np.empty((n_block, n_block), dtype=np.float64)
    B_local = np.empty((n_block, n_block), dtype=np.float64)
    comm.Scatter(A_send, A_local, root=0)
    comm.Scatter(B_send, B_local, root=0)

    rank_row, rank_col, comm_row, comm_col = _make_grid(comm, sqrtP)

    C_local = _dense_SUMMA(A_local, B_local, comm_row, comm_col,
                           rank_row, rank_col, sqrtP)

    # Gather back to rank 0
    C_recv = np.empty((P, n_block, n_block), dtype=np.float64) if rank == 0 else None
    comm.Gather(np.ascontiguousarray(C_local), C_recv, root=0)

    if rank == 0:
        # Reverse the scatter permutation: (proc, block_row, block_col) -> (n, n)
        return (C_recv.reshape(sqrtP, sqrtP, n_block, n_block)
                      .transpose(0, 2, 1, 3)
                      .reshape(n, n))
    return None


# ---------------------------------------------------------------------------
# Data generators
# ---------------------------------------------------------------------------

def dg_summa_sparse_small():
    """Small sparse ER(d) matrices: n=500, d=5. Expected nnz ≈ 2500 (density ~1%)."""
    n, d = 500, 5

    def _er(seed):
        rng = np.random.default_rng(seed)
        nnz = rng.binomial(n * n, d / n)
        rows = rng.integers(0, n, size=nnz)
        cols = rng.integers(0, n, size=nnz)
        vals = rng.standard_normal(nnz)
        m = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
        m.sum_duplicates()
        return m

    return (_er(0), _er(1))


def dg_summa_sparse_medium():
    """Medium sparse ER(d) matrices: n=5000, d=10. Expected nnz ≈ 50000 (density ~0.2%)."""
    n, d = 5000, 10

    def _er(seed):
        rng = np.random.default_rng(seed)
        nnz = rng.binomial(n * n, d / n)
        rows = rng.integers(0, n, size=nnz)
        cols = rng.integers(0, n, size=nnz)
        vals = rng.standard_normal(nnz)
        m = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
        m.sum_duplicates()
        return m

    return (_er(0), _er(1))


def dg_summa_dense_small():
    """Small dense matrices: n=512, entries N(0,1). ~2 MB per matrix."""
    n = 512
    rng = np.random.default_rng(0)
    return (
        rng.standard_normal((n, n)),
        rng.standard_normal((n, n)),
    )


def dg_summa_dense_medium():
    """
    Medium dense matrices: n=4096, entries N(0,1). ~128 MB per matrix.
    Safe for P>=16; skip P<16 to avoid >4 GB/proc peak.
    """
    n = 4096
    rng = np.random.default_rng(0)
    return (
        rng.standard_normal((n, n)),
        rng.standard_normal((n, n)),
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import time

    parser = argparse.ArgumentParser(
        description="SUMMA / SpSUMMA distributed matrix multiply benchmark"
    )
    parser.add_argument("-n", type=int, default=500,
                        help="Global matrix dimension (default: 500)")
    parser.add_argument("-d", type=float, default=5.0,
                        help="ER degree param for sparse (prob=d/n, default: 5)")
    parser.add_argument("--nreps", type=int, default=3,
                        help="Timed repetitions (default: 3)")
    parser.add_argument("--matrix-type", choices=["sparse", "dense", "both"],
                        default="both")
    parser.add_argument("--variant", choices=["standard", "improved", "both"],
                        default="both",
                        help="SpSUMMA variant for sparse (default: both)")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    P = comm.Get_size()
    sqrtP = int(round(P ** 0.5))

    if sqrtP * sqrtP != P:
        if rank == 0:
            print(f"ERROR: P={P} must be a perfect square.", flush=True)
        MPI.Finalize()
        raise SystemExit(1)

    n = args.n
    if n % sqrtP != 0:
        n = ((n + sqrtP - 1) // sqrtP) * sqrtP
        if rank == 0:
            print(f"WARNING: n padded to {n} to be divisible by sqrt(P)={sqrtP}")

    if rank == 0:
        print("=" * 60)
        print(f"SUMMA Benchmark  |  n={n}  P={P}  sqrt(P)={sqrtP}")
        print("=" * 60, flush=True)

    def _time_run(fn, *fnargs):
        times = []
        for _ in range(args.nreps):
            comm.Barrier()
            t0 = MPI.Wtime()
            result = fn(*fnargs)
            comm.Barrier()
            times.append(MPI.Wtime() - t0)
        return result, times

    rng = np.random.default_rng(42)

    if args.matrix_type in ("sparse", "both"):
        variants = (["standard", "improved"] if args.variant == "both"
                    else [args.variant])
        A_sp, B_sp = None, None
        if rank == 0:
            def _er(seed):
                r = np.random.default_rng(seed)
                nnz = r.binomial(n * n, args.d / n)
                rows = r.integers(0, n, size=nnz)
                cols = r.integers(0, n, size=nnz)
                vals = r.standard_normal(nnz)
                m = sp.csr_matrix((vals, (rows, cols)), shape=(n, n), dtype=np.float64)
                m.sum_duplicates()
                return m
            A_sp, B_sp = _er(0), _er(1)
            print(f"\n[Sparse] nnz(A)={A_sp.nnz:,}  nnz(B)={B_sp.nnz:,}", flush=True)

        for v in variants:
            C, times = _time_run(benchmark_summa_sparse, A_sp, B_sp, v)
            if rank == 0:
                min_t, med_t, max_t = min(times), sorted(times)[len(times)//2], max(times)
                theory_bw = args.d * n / sqrtP
                print(f"\n=== SpSUMMA ({v}) ===")
                print(f"  nnz(C)             = {C.nnz:,}  (expected ~{int(args.d**2*n):,})")
                print(f"  time (min/med/max) = {min_t:.3f}s / {med_t:.3f}s / {max_t:.3f}s")
                print(f"  theory O(dn/√P)    ≈ {int(theory_bw):,} words/proc", flush=True)

    if args.matrix_type in ("dense", "both"):
        A_dn, B_dn = None, None
        if rank == 0:
            A_dn = rng.standard_normal((n, n))
            B_dn = rng.standard_normal((n, n))
            print(f"\n[Dense] block={n//sqrtP}×{n//sqrtP}  "
                  f"({(n//sqrtP)**2*8/1e6:.1f} MB/proc)", flush=True)

        C, times = _time_run(benchmark_summa_dense, A_dn, B_dn)
        if rank == 0:
            min_t, med_t, max_t = min(times), sorted(times)[len(times)//2], max(times)
            theory_bw = n**2 / sqrtP
            gflops = 2 * n**3 / min_t / 1e9
            print(f"\n=== Dense SUMMA ===")
            print(f"  time (min/med/max) = {min_t:.3f}s / {med_t:.3f}s / {max_t:.3f}s")
            print(f"  GFlops (best, 2n³) = {gflops:.4f}")
            print(f"  theory O(n²/√P)    ≈ {int(theory_bw):,} elems/proc", flush=True)


        # run on cluster with srun python3 summa_benchmark.py -n 500 -d 5 --matrix-type both --nreps 2