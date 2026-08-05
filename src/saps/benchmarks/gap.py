import numpy as np

from saps.benchmark import Author, DataInstance, Ref
from saps.benchmarks.suitesparse import fetch_suitesparse_matrix
from saps_framework.binsparse_format import BinsparseFormat

GAP_REFERENCE = Ref(
    title="The GAP Benchmark Suite",
    authors=[
        Author("Scott Beamer"),
        Author("Krste Asanović"),
        Author("David Patterson"),
    ],
    url="https://arxiv.org/abs/1508.03619",
    year=2015,
)


GAP_SUITE = [
    "gap-twitter",
    "gap-web",
    "gap-road",
    "gap-kron",
]


def fetch_gap_graph(
    name: str,
    *,
    src: int = 0,
    directed: bool = True,
    drop_weights: bool = True,
) -> DataInstance:
    """Fetch a GAP graph through the shared SuiteSparse cache with SNAP-style meta."""
    key = name.lower()
    if key not in GAP_SUITE:
        raise ValueError(f"Unknown GAP dataset: {name}")
    raw = fetch_suitesparse_matrix(key)
    adjacency = raw.inputs[0]
    if drop_weights:
        coo = BinsparseFormat.to_coo(adjacency).data
        adjacency = BinsparseFormat.from_coo(
            (coo["indices_0"], coo["indices_1"]),
            np.ones(len(coo["values"]), dtype=bool),
            coo["shape"],
        )
    meta = {
        "directed": directed,
        "num_nodes": raw.meta["shape"][0],
        "num_edges": raw.meta["nnz"],
        "num_matrix_entries": raw.meta["nnz"],
        "remap_nodes": False,
        "src": src,
    }
    return DataInstance(inputs=[adjacency], meta=meta)
