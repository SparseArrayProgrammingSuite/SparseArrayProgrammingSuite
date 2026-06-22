import saps
from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.downloaders.snap import download_snap_dataset

xp = saps.xp


class TransitiveClosureDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Transitive closure input {name}."
        self._suites = suites or []

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name

    @property
    def description(self) -> str:
        return self._description

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"


# BEGIN COPIED TEST FILE: tests/test_transitive_closure.py
# import numpy as np
#
# import saps.benchmarks.transitive_closure as tc
# from frameworks.saps_numpy import NumpyFramework
# from saps.downloaders.snap import load_toy_dataset
# from saps_framework import BinsparseFormat
#
#
# def _run_tc(A):
#     xp = NumpyFramework()
#     tc.xp = xp
#     A_bin = A if isinstance(A, BinsparseFormat) else BinsparseFormat.from_numpy(A)
#     (res,) = tc.TransitiveClosureBenchmark().benchmark([A_bin], {})
#     return res
#
#
# def test_transitive_closure():
#     # 6-node DAG.
#     input_matrix = np.array(
#         [
#             [0, 1, 1, 0, 0, 0],
#             [0, 0, 1, 1, 0, 0],
#             [0, 0, 0, 0, 1, 0],
#             [0, 0, 0, 0, 0, 0],
#             [0, 0, 0, 0, 0, 1],
#             [0, 0, 0, 1, 0, 0],
#         ],
#         dtype=bool,
#     )
#
#     expected = np.array(
#         [
#             [1, 1, 1, 1, 1, 1],
#             [0, 1, 1, 1, 1, 1],
#             [0, 0, 1, 1, 1, 1],
#             [0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 1, 1, 1],
#             [0, 0, 0, 1, 0, 1],
#         ],
#         dtype=bool,
#     )
#
#     res = _run_tc(input_matrix)
#     assert np.array_equal(res, expected)
#
#
# def test_stc():
#     # 8 node graph with 4 Stc.
#     input_matrix = np.array(
#         [
#             [0, 1, 0, 0, 0, 0, 0, 0],
#             [0, 0, 1, 0, 1, 1, 0, 0],
#             [0, 0, 0, 1, 0, 0, 1, 0],
#             [0, 0, 1, 0, 0, 0, 0, 1],
#             [1, 0, 0, 0, 0, 1, 0, 0],
#             [0, 0, 0, 0, 0, 0, 1, 0],
#             [0, 0, 0, 0, 0, 1, 0, 1],
#             [0, 0, 0, 0, 0, 0, 0, 1],
#         ],
#         dtype=bool,
#     )
#
#     expected = 4
#
#     res = _run_tc(input_matrix)
#
#     # count stc.
#     visited_set = set()
#     count = 0
#     for i in range(res.shape[0]):
#         comp = tuple(res[i, :])
#         if comp not in visited_set:
#             count += 1
#             visited_set.add(comp)
#
#     assert count == expected
#
#
# def test_stc_cycle():
#     # one stc. one cycle
#     input_matrix = np.array(
#         [
#             [0, 1, 0],
#             [0, 0, 1],
#             [1, 0, 0],
#         ],
#         dtype=bool,
#     )
#
#     res = _run_tc(input_matrix)
#     # clique matrix
#     expected = np.ones((3, 3), dtype=bool)
#     assert np.array_equal(res, expected)
#
#
# def test_stc_one_node():
#     # one node
#     input_matrix = np.array([[0]], dtype=bool)
#
#     res = _run_tc(input_matrix)
#
#     # simple 1x1 matrix with 1
#     expected = np.array([[1]], dtype=bool)
#     assert np.array_equal(res, expected)
#
#
# def test_transitive_closure_one_node():
#     # one node
#     input_matrix = np.array([[0]], dtype=bool)
#
#     res = _run_tc(input_matrix)
#
#     # should be self loop, 1x1 matrix with 1
#     expected = np.array([[1]], dtype=bool)
#     assert np.array_equal(res, expected)
#
#
# def test_transitive_snap_toy():
#     data, _ = load_toy_dataset()
#     res = _run_tc(data[0])
#     expected = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=bool)
#     assert np.array_equal(res, expected)
# END COPIED TEST FILE: tests/test_transitive_closure.py

class TransitiveClosureGenerator(Generator[TransitiveClosureDataset]):
    @property
    def name(self) -> str:
        return "transitive_closure_inputs"

    @property
    def pretty_name(self) -> str:
        return "Transitive Closure Input Generator"

    @property
    def description(self) -> str:
        return "Input generator for transitive closure benchmarks."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self) -> list[Contributor]:
        return []

    @property
    def references(self) -> list[Ref]:
        return []

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to construct the generator and dataset structures."
            " This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Generate sparse directed graph inputs for transitive closure."

    @property
    def datasets(self) -> list[TransitiveClosureDataset]:
        return [
            TransitiveClosureDataset(
                name="snap-email-Eu-core",
                pretty_name="SNAP email-Eu-core",
                description=(
                    "Directed email communication network from a European research"
                    " institution, with 1,005 nodes and 25,571 edges."
                ),
                suites=[],
            ),
            TransitiveClosureDataset(
                name="snap-ca-GrQc",
                pretty_name="SNAP ca-GrQc",
                description=(
                    "Arxiv General Relativity and Quantum Cosmology collaboration"
                    " network, with 5,242 nodes and 14,496 edges."
                ),
                suites=[],
            ),
        ]

    def generate(self, dataset: TransitiveClosureDataset) -> DataInstance:
        if dataset.name.startswith("snap"):
            inputs, meta = download_snap_dataset(dataset.name)
            return DataInstance(inputs=inputs, meta=meta)
        raise ValueError(f"Unsupported transitive closure dataset: {dataset.name}")


class TransitiveClosureBenchmark(Benchmark):
    @property
    def name(self):
        return "transitive_closure"

    @property
    def pretty_name(self):
        return "Transitive Closure"

    @property
    def description(self):
        return (
            "Computes the transitive closure of a directed graph using fixed-point"
            " iteration. The algorithm initializes the adjacency matrix with the"
            " identity, then iteratively applies the closure operation using sparse"
            " matrix operations until convergence. This enables reachability queries."
        )

    @property
    def suites(self):
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def authors(self):
        return [
            Contributor("Rithvik Reddygari", "rreddygari3@gatech.edu"),
            Contributor("Joel Mathew Cherian", "jcherian32@gatech.edu"),
        ]

    @property
    def references(self):
        return [
            Ref(
                title=("Graph Algorithms in the Language of Linear Algebra"),
                authors=[
                    Author("Kepner, Jeremy"),
                    Author("Gilbert, John"),
                ],
                journal="Society for Industrial and Applied Mathematics (SIAM)",
                city="Philadelphia",
                year=2011,
            ),
        ]

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to construct the benchmark function itself. "
            "Generative AI might have been used to construct tests."
        )

    @property
    def motivation(self):
        return ""

    @property
    def generators(self) -> list[Generator[TransitiveClosureDataset]]:
        return [TransitiveClosureGenerator()]

    def benchmark(self, data, meta):
        edges = xp.from_binsparse(data[0])
        (n, m) = edges.shape
        assert m == n

        # create identity matrix with edges
        graph = xp.array(edges, dtype=bool)
        identity_matrix = xp.eye(n, dtype=bool)
        graph = xp.logical_or(identity_matrix, graph)

        # do fixed-point iteration
        max_iterations = n
        for _iteration in range(max_iterations):
            nextGraph = xp.einsum(
                "nextGraph[i,j] or= graph[i,k] & graph[k,j]", graph=graph
            )

            if xp.all(xp.equal(graph, nextGraph)):
                break
            graph = nextGraph
        return [graph]
