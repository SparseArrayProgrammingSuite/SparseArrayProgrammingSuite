import numpy as np
import pytest

import saps.benchmarks.subgraph_matching as sgm
from frameworks.saps_numpy import NumpyFramework
from frameworks.saps_sparse import PyDataSparseFramework
from saps_framework import BinsparseFormat

# SciPyFramework is excluded: its einsum() uses array_api_compat.array_namespace
# which does not recognise scipy CSR arrays.  SciPyFramework is intended for
# dense linear-algebra benchmarks (GMRES, CG, …), not graph einsum patterns.

FRAMEWORKS = [NumpyFramework(), PyDataSparseFramework()]
FRAMEWORK_IDS = ["numpy", "pydata_sparse"]

# ---------------------------------------------------------------------------
# Why the metadata looks the way it does
#
# Subgraph matching needs several *named* sparse matrices per query
# (e.g. VA, E0, VB for "count A→B edges").  The harness in benchmark.py
# iterates a flat list[BinsparseFormat] and calls xp.from_binsparse() on
# every item, so we cannot pass a list-of-dicts.  Instead:
#
#   flat list     = [mat_Q0_0, mat_Q0_1, mat_Q0_2, mat_Q1_0, …]
#   query_sizes   = [3, 1, …]            # how many matrices per query
#   matrix_names  = ["VA", "E0", "VB", "E0", …]  # name of each flat entry
#
# benchmark() zips these back into per-query {name: array} dicts before
# calling xp.einsum().
# ---------------------------------------------------------------------------


def _run(xp, flat_bsf: list[BinsparseFormat], meta: dict) -> np.ndarray:
    """Convert BinsparseFormat objects, run the benchmark, return counts as numpy."""
    sgm.xp = xp
    data = [xp.from_binsparse(m) for m in flat_bsf]
    (counts_arr,) = sgm.SubgraphMatching().benchmark(data, meta)
    if hasattr(counts_arr, "todense"):
        return np.asarray(counts_arr.todense()).ravel()
    return np.asarray(counts_arr).ravel()


# ---------------------------------------------------------------------------
# Small test graph
#
#   Nodes:  0 (label A), 1 (label B), 2 (label A)
#   Edges (label 0): 0→1, 0→2, 2→1
#
#   VA = indicator vector for label-A nodes  → indices [0, 2]
#   VB = indicator vector for label-B nodes  → indices [1]
#   E0 = adjacency matrix of label-0 edges  → (0,1), (0,2), (2,1)
# ---------------------------------------------------------------------------

N = 3

VA = BinsparseFormat.from_coo(
    (np.array([0, 2]),),
    np.ones(2, dtype=np.int64),
    (N,),
)
VB = BinsparseFormat.from_coo(
    (np.array([1]),),
    np.ones(1, dtype=np.int64),
    (N,),
)
E0 = BinsparseFormat.from_coo(
    (np.array([0, 0, 2]), np.array([1, 2, 1])),
    np.ones(3, dtype=np.int64),
    (N, N),
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("xp", FRAMEWORKS, ids=FRAMEWORK_IDS)
def test_count_all_edges(xp):
    # S[] += E0[i,j]  →  sum of all entries in E0 = 3
    meta = {
        "exprs": ["S[] += E0[i,j]"],
        "gts": [3],
        "names": ["all_edges"],
        "query_sizes": [1],
        "matrix_names": ["E0"],
    }
    counts = _run(xp, [E0], meta)
    assert counts[0] == 3


@pytest.mark.parametrize("xp", FRAMEWORKS, ids=FRAMEWORK_IDS)
def test_count_label_a_nodes(xp):
    # S[] += VA[v]  →  number of label-A nodes = 2
    meta = {
        "exprs": ["S[] += VA[v]"],
        "gts": [2],
        "names": ["label_a_count"],
        "query_sizes": [1],
        "matrix_names": ["VA"],
    }
    counts = _run(xp, [VA], meta)
    assert counts[0] == 2


@pytest.mark.parametrize("xp", FRAMEWORKS, ids=FRAMEWORK_IDS)
def test_count_edges_a_to_b(xp):
    # S[] += VA[u] * E0[u,v] * VB[v]
    # Matches (u=0,v=1) and (u=2,v=1)  →  2
    meta = {
        "exprs": ["S[] += VA[u] * E0[u,v] * VB[v]"],
        "gts": [2],
        "names": ["a_to_b"],
        "query_sizes": [3],
        "matrix_names": ["VA", "E0", "VB"],
    }
    counts = _run(xp, [VA, E0, VB], meta)
    assert counts[0] == 2


@pytest.mark.parametrize("xp", FRAMEWORKS, ids=FRAMEWORK_IDS)
def test_count_edges_b_to_a(xp):
    # S[] += VB[u] * E0[u,v] * VA[v]
    # No label-B node has an outgoing edge to a label-A node → 0
    meta = {
        "exprs": ["S[] += VB[u] * E0[u,v] * VA[v]"],
        "gts": [0],
        "names": ["b_to_a"],
        "query_sizes": [3],
        "matrix_names": ["VB", "E0", "VA"],
    }
    counts = _run(xp, [VB, E0, VA], meta)
    assert counts[0] == 0


@pytest.mark.parametrize("xp", FRAMEWORKS, ids=FRAMEWORK_IDS)
def test_multiple_queries_batched(xp):
    # Two queries in one benchmark() call.
    # Q0: count A→B edges = 2  (3 matrices: VA, E0, VB)
    # Q1: count all edges = 3  (1 matrix: E0)
    meta = {
        "exprs": ["S[] += VA[u] * E0[u,v] * VB[v]", "S[] += E0[i,j]"],
        "gts": [2, 3],
        "names": ["a_to_b", "all_edges"],
        "query_sizes": [3, 1],
        "matrix_names": ["VA", "E0", "VB", "E0"],
    }
    counts = _run(xp, [VA, E0, VB, E0], meta)
    assert counts[0] == 2
    assert counts[1] == 3
