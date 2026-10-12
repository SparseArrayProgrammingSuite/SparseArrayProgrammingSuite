#!/usr/bin/env python
"""Compute reference counts for the triangle and 4-clique counting benchmarks.

The benchmarks use the strict upper triangle of the simple undirected 0-1
adjacency A. Each triangle or four-clique is counted once, without division:

    triangles     = sum A[i,j] A[j,k] A[i,k]
    four-cliques  = sum A[i,j] A[i,k] A[i,l] A[j,k] A[j,l] A[k,l]

This script computes these counts without the einsums, loading each graph
through the benchmark's own generator so that the input is identical. Results
are written to a JSON file, one entry per dataset. Only entries marked with
this input convention are reused when resuming; older results are recomputed.

    python scripts/compute_graph_counts.py --self-test
    python scripts/compute_graph_counts.py --output counts.json --max-edges 2e8
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import scipy.sparse as sps

from binsparse.conversions import to_scipy

# Bound on the nonzeros one row block of a sparse product may produce.
_BLOCK_WORK = 20_000_000


def masked_product_sum(L, R, M) -> int:
    """Return sum((L @ R) * M) for CSR arrays, without forming all of L @ R.

    Rows of L are processed in blocks whose product has at most about
    _BLOCK_WORK structural nonzeros, a single heavy row excepted.
    """
    L, R, M = L.tocsr(), R.tocsr(), M.tocsr()
    # Products formed by each row of L, as a running total.
    work = np.cumsum(L @ np.diff(R.indptr).astype(np.int64))
    total, start, n = 0, 0, L.shape[0]
    while start < n:
        done = work[start - 1] if start else 0
        stop = int(np.searchsorted(work, done + _BLOCK_WORK, side="right"))
        stop = max(stop, start + 1)
        total += int((L[start:stop] @ R).multiply(M[start:stop]).sum())
        start = stop
    return total


def triangle_sum(A) -> int:
    """sum_{i,j,k} A[i,j] A[j,k] A[i,k], counting triangles once for upper A."""
    A = A.tocsr()
    return masked_product_sum(A, A, A)


def four_clique_sum(A) -> int:
    """sum_{i,j,k,l} A[i,j] A[i,k] A[i,l] A[j,k] A[j,l] A[k,l].

    For fixed i, j, k and l range over the out-neighbours P of i, so the inner
    sum is sum_{j,k,l in P} S[j,k] S[j,l] S[k,l] = sum((S.T @ S) * S) for the
    submatrix S = A[P, P].
    """
    A = A.tocsr()
    total = 0
    for i in range(A.shape[0]):
        P = A.indices[A.indptr[i] : A.indptr[i + 1]]
        if P.size == 0:
            continue
        S = A[P][:, P]
        if S.nnz:
            total += masked_product_sum(S.T, S, S)
    return total


try:
    import numba
except ImportError:  # the scipy versions above still work, single threaded
    numba = None

# Nodes are split into interleaved chunks, so hubs spread across threads.
# Keep a few chunks per worker without repeatedly allocating large marker arrays.
_CHUNKS_PER_THREAD = 4

if numba is not None:
    # Each chunk runs serially in its own function; the parallel loops only
    # call them, which keeps numba from treating the counters as reductions.

    @numba.njit(cache=True)
    def _triangle_chunk(c, chunks, indptr, indices, n):
        in_i = np.zeros(n, np.bool_)
        total = 0
        for i in range(c, n, chunks):
            start, stop = indptr[i], indptr[i + 1]
            if start == stop:
                continue
            for p in range(start, stop):
                in_i[indices[p]] = True
            # Edges i -> j -> k together with i -> k.
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                low, high = indptr[j], indptr[j + 1]
                # For a much larger row, search for the outgoing neighbors of i
                # instead of scanning every outgoing neighbor of j.
                # The 16:1 crossover is a performance heuristic, also used below.
                if high - low > 16 * (stop - start):
                    row = indices[low:high]
                    for q in range(start, stop):
                        k = indices[q]
                        pos = np.searchsorted(row, k)
                        if pos < len(row) and row[pos] == k:
                            total += 1
                else:
                    for q in range(low, high):
                        if in_i[indices[q]]:
                            total += 1
            for p in range(start, stop):
                in_i[indices[p]] = False
        return total

    @numba.njit(cache=True)
    def _four_clique_chunk(c, chunks, indptr, indices, n):
        in_p = np.zeros(n, np.bool_)
        in_pj = np.zeros(n, np.bool_)
        max_degree = 0
        for i in range(c, n, chunks):
            max_degree = max(max_degree, indptr[i + 1] - indptr[i])
        common = np.empty(max_degree, dtype=indices.dtype)
        total = 0
        for i in range(c, n, chunks):
            for p in range(indptr[i], indptr[i + 1]):
                in_p[indices[p]] = True
            for p in range(indptr[i], indptr[i + 1]):
                j = indices[p]
                # Mark out(j) within P, then count edges k -> l inside it.
                size = 0
                if indptr[j + 1] - indptr[j] > 16 * (indptr[i + 1] - indptr[i]):
                    row = indices[indptr[j] : indptr[j + 1]]
                    for q in range(indptr[i], indptr[i + 1]):
                        k = indices[q]
                        pos = np.searchsorted(row, k)
                        if pos < len(row) and row[pos] == k:
                            common[size] = k
                            size += 1
                            in_pj[k] = True
                else:
                    for q in range(indptr[j], indptr[j + 1]):
                        k = indices[q]
                        if in_p[k]:
                            common[size] = k
                            size += 1
                            in_pj[k] = True
                # Reuse the common neighbors for counting and clearing markers.
                for q in range(size):
                    k = common[q]
                    start, stop = indptr[k], indptr[k + 1]
                    if stop - start > 16 * size:
                        row = indices[start:stop]
                        for r in range(size):
                            neighbor = common[r]
                            pos = np.searchsorted(row, neighbor)
                            if pos < len(row) and row[pos] == neighbor:
                                total += 1
                    else:
                        for r in range(start, stop):
                            if in_pj[indices[r]]:
                                total += 1
                for q in range(size):
                    in_pj[common[q]] = False
            for p in range(indptr[i], indptr[i + 1]):
                in_p[indices[p]] = False
        return total

    @numba.njit(parallel=True, cache=True)
    def _triangle_sum_numba(indptr, indices, n, chunks):
        totals = np.zeros(chunks, np.int64)
        for c in numba.prange(chunks):
            totals[c] = _triangle_chunk(c, chunks, indptr, indices, n)
        return totals.sum()

    @numba.njit(parallel=True, cache=True)
    def _four_clique_sum_numba(indptr, indices, n, chunks):
        totals = np.zeros(chunks, np.int64)
        for c in numba.prange(chunks):
            totals[c] = _four_clique_chunk(c, chunks, indptr, indices, n)
        return totals.sum()


def fast_triangle_sum(A) -> int:
    """triangle_sum, compiled and parallel when numba is available."""
    if numba is None:
        return triangle_sum(A)
    A = A.tocsr()
    A.sum_duplicates()
    A.sort_indices()
    chunks = min(A.shape[0], _CHUNKS_PER_THREAD * numba.get_num_threads())
    return int(_triangle_sum_numba(A.indptr, A.indices, A.shape[0], chunks))


def fast_four_clique_sum(A) -> int:
    """four_clique_sum, compiled and parallel when numba is available."""
    if numba is None:
        return four_clique_sum(A)
    A = A.tocsr()
    A.sum_duplicates()
    A.sort_indices()
    chunks = min(A.shape[0], _CHUNKS_PER_THREAD * numba.get_num_threads())
    return int(_four_clique_sum_numba(A.indptr, A.indices, A.shape[0], chunks))


def self_test(trials: int = 200, seed: int = 0) -> None:
    """Compare against dense einsums on small random directed graphs."""
    rng = np.random.default_rng(seed)
    for trial in range(trials):
        n = int(rng.integers(1, 14))
        A = (rng.random((n, n)) < rng.uniform(0.05, 0.9)).astype(np.int64)
        if trial % 2:
            A = A | A.T  # symmetric graphs, with and without self-loops
        if trial % 3 == 0:
            np.fill_diagonal(A, 0)
        tri = int(np.einsum("ij,jk,ik->", A, A, A))
        four = int(np.einsum("ij,ik,il,jk,jl,kl->", A, A, A, A, A, A))
        S = sps.csr_array(A)
        if trial % 2:
            S.indices = S.indices.astype(np.int64)
            S.indptr = S.indptr.astype(np.int64)
        for count, expected in (
            (triangle_sum, tri),
            (fast_triangle_sum, tri),
            (four_clique_sum, four),
            (fast_four_clique_sum, four),
        ):
            assert count(S) == expected, (trial, count.__name__, count(S), expected)
    # Skewed rows exercise the binary searches, including loops and direction.
    star = np.eye(80, dtype=np.int64)
    star[0, :] = star[:, 0] = 1
    star[1:5, 1:5] = 1
    for A in (star, np.triu(star), np.zeros((0, 0), dtype=np.int64)):
        S = sps.csr_array(A)
        assert fast_triangle_sum(S) == int(np.einsum("ij,jk,ik->", A, A, A))
        assert fast_four_clique_sum(S) == int(
            np.einsum("ij,ik,il,jk,jl,kl->", A, A, A, A, A, A)
        )
    # A block boundary inside the product must not change the result.
    global _BLOCK_WORK
    saved, _BLOCK_WORK = _BLOCK_WORK, 3
    try:
        A = (rng.random((40, 40)) < 0.2).astype(np.int64)
        S = sps.csr_array(A)
        assert triangle_sum(S) == int(np.einsum("ij,jk,ik->", A, A, A))
        assert four_clique_sum(S) == int(
            np.einsum("ij,ik,il,jk,jl,kl->", A, A, A, A, A, A)
        )
    finally:
        _BLOCK_WORK = saved
    print(f"self-test passed ({trials} random graphs)")


def _datasets():
    from saps.benchmarks.triangle_counting import (
        TriangleCountingGAPGenerator,
        TriangleCountingSNAPGenerator,
    )

    # The 4-clique generators use the same strict upper-triangular inputs.
    for generator in (TriangleCountingSNAPGenerator(), TriangleCountingGAPGenerator()):
        for dataset in generator.datasets:
            yield generator, dataset


# Published edge counts for the GAP graphs (see saps.benchmarks.gap).
_GAP_EDGES = {
    "GAP-road": 58.3e6,
    "GAP-twitter": 1468.4e6,
    "GAP-web": 1949.4e6,
    "GAP-kron": 2111.6e6,
    "GAP-urand": 2147.4e6,
}


def declared_edges(name: str) -> float:
    """Edge count declared for a graph, without downloading it; inf if unknown."""
    if name in _GAP_EDGES:
        return _GAP_EDGES[name]
    from saps.benchmarks.snap import snap_graph

    try:
        edges = snap_graph(name).edges
    except (KeyError, ValueError):
        return float("inf")
    return float(edges) if isinstance(edges, int) else float("inf")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--output", type=Path, default=Path("graph_counts.json"))
    parser.add_argument("--only", nargs="*", help="dataset names to compute")
    parser.add_argument(
        "--max-edges", type=float, default=float("inf"), help="skip larger graphs"
    )
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return

    results = json.loads(args.output.read_text()) if args.output.exists() else {}
    for generator, dataset in _datasets():
        if args.only and dataset.name not in args.only:
            continue
        if results.get(dataset.name, {}).get("adjacency") == "simple_undirected_upper":
            continue
        # Checked before loading, so large graphs are never downloaded.
        edges = declared_edges(dataset.name)
        if not args.only and edges > args.max_edges:
            print(f"skip {dataset.name}: {edges:,.0f} declared edges", flush=True)
            continue
        start = time.perf_counter()
        A = to_scipy(generator.generate(dataset).inputs[0]).tocsr().astype(np.int64)
        A.sum_duplicates()
        loaded = time.perf_counter()
        tri = fast_triangle_sum(A)
        tri_done = time.perf_counter()
        four = fast_four_clique_sum(A)
        done = time.perf_counter()
        results[dataset.name] = {
            "nnz": int(A.nnz),
            "adjacency": "simple_undirected_upper",
            "triangles": tri,
            "four_cliques": four,
            "seconds": {
                "load": round(loaded - start, 1),
                "triangle": round(tri_done - loaded, 1),
                "four_clique": round(done - tri_done, 1),
            },
        }
        args.output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
        print(
            f"{dataset.name}: nnz {A.nnz:,} triangles {tri:,} "
            f"4-cliques {four:,} ({done - start:.1f}s)",
            flush=True,
        )


if __name__ == "__main__":
    main()
