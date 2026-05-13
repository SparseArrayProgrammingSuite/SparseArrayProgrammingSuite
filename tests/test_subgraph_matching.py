import numpy as np

import saps.benchmarks.subgraph_matching as subgraph_matching
from frameworks.saps_sparse import PyDataSparseFramework
from saps_framework import BinsparseFormat


def test_single_edge_label_count():
    xp = PyDataSparseFramework()
    subgraph_matching.xp = xp

    edge_matrix = BinsparseFormat.from_coo(
        (np.array([0, 1]), np.array([1, 2])),
        np.array([1, 1], dtype=np.int64),
        (3, 3),
    )
    matrices = [{"E0": edge_matrix}]
    meta = {
        "exprs": ["S[] += E0[i,j]"],
        "gts": [2],
        "names": ["local_edges"],
    }

    results = subgraph_matching.SubgraphMatching().benchmark(matrices, meta)
    results = xp.from_binsparse(results)

    assert results[0] == meta["gts"][0]
