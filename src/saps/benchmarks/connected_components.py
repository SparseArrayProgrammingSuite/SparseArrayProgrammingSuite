# BEGIN COPIED TEST FILE: tests/test_connected_components.py
# import numpy as np
#
# import saps.benchmarks.connected_components as cc
# from frameworks.saps_numpy import NumpyFramework
# from saps.downloaders.snap import load_toy_dataset
# from saps_framework import BinsparseFormat
#
#
# def _run_cc(A):
#     xp = NumpyFramework()
#     cc.xp = xp
#     A_bin = A if isinstance(A, BinsparseFormat) else BinsparseFormat.from_numpy(A)
#     (labels,) = cc.SimplyConnectedComponentsBenchmark().benchmark([A_bin], {})
#     return labels.ravel()
#
#
# # ---------------------------------------------------------------------------
# # Algorithm correctness
# # ---------------------------------------------------------------------------
#
#
# def test_cc_fully_connected():
#     """All nodes in a clique should end up with the same label."""
#     A = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool)
#     labels = _run_cc(A)
#     assert len(set(labels.tolist())) == 1, "all nodes should share one label"
#
#
# def test_cc_two_disconnected_components():
#     """Nodes 0-1 and nodes 2-3 are in separate undirected components."""
#     A = np.array(
#         [
#             [0, 1, 0, 0],
#             [1, 0, 0, 0],
#             [0, 0, 0, 1],
#             [0, 0, 1, 0],
#         ],
#         dtype=bool,
#     )
#     labels = _run_cc(A)
#     # nodes in the same component share a label; different components differ
#     assert labels[0] == labels[1]
#     assert labels[2] == labels[3]
#     assert labels[0] != labels[2]
#
#
# def test_cc_isolated_nodes():
#     """With no edges, every node is its own component."""
#     A = np.zeros((4, 4), dtype=bool)
#     labels = _run_cc(A)
#     assert len(set(labels.tolist())) == 4, (
#         "each isolated node should have a unique label"
#     )
#
#
# def test_cc_directed_star_pointing_inward():
#     """
#     Edges 1→0, 2→0, 3→0.  Each node's outgoing neighbourhood includes node 0,
#     so all nodes propagate to node 0's label.
#     """
#     A = np.array(
#         [
#             [0, 0, 0, 0],
#             [1, 0, 0, 0],
#             [1, 0, 0, 0],
#             [1, 0, 0, 0],
#         ],
#         dtype=bool,
#     )
#     labels = _run_cc(A)
#     assert len(set(labels.tolist())) == 1, "all nodes should converge to the same label"
#
#
# def test_cc_single_node():
#     """Trivial one-node graph."""
#     A = np.zeros((1, 1), dtype=bool)
#     labels = _run_cc(A)
#     assert labels.shape == (1,)
#
#
# # ---------------------------------------------------------------------------
# # Generator / downloader wiring
# # ---------------------------------------------------------------------------
#
#
# def test_cc_snap_toy():
#     data, _ = load_toy_dataset()
#     labels = _run_cc(data[0])
#     assert len(set(labels.tolist())) == 1, "all nodes should converge to the same label"
# END COPIED TEST FILE: tests/test_connected_components.py

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


class ConnectedComponentsDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Connected components input {name}."
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


class ConnectedComponentsGenerator(Generator[ConnectedComponentsDataset]):
    @property
    def name(self) -> str:
        return "connected_components_inputs"

    @property
    def pretty_name(self) -> str:
        return "Connected Components Input Generator"

    @property
    def description(self) -> str:
        return "Input generator for connected components benchmarks."

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
        return [
            Ref(
                title=(
                    "SNAP: A General Purpose Network Analysis and Graph Mining Library"
                ),
                authors=[
                    Author("Leskovec, Jure"),
                    Author("Sosič, Rok"),
                ],
                journal="ACM Transactions on Intelligent Systems and Technology",
                volume=8,
                number=1,
                year=2016,
                url="https://snap.stanford.edu/index.html",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to construct the generator and dataset structures."
            " This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Generate sparse graph inputs for connected components."

    @property
    def datasets(self) -> list[ConnectedComponentsDataset]:
        return [
            ConnectedComponentsDataset(
                name="snap-email-Eu-core",
                pretty_name="SNAP email-Eu-core",
                description=(
                    "Directed email communication network from a European research"
                    " institution, with 1,005 nodes and 25,571 edges."
                ),
                suites=[],
            ),
            ConnectedComponentsDataset(
                name="snap-facebook_combined",
                pretty_name="SNAP facebook_combined",
                description=(
                    "Combined Facebook social-circle network, with 4,039 nodes and"
                    " 88,234 edges."
                ),
                suites=[],
            ),
            ConnectedComponentsDataset(
                name="snap-ca-GrQc",
                pretty_name="SNAP ca-GrQc",
                description=(
                    "Arxiv General Relativity and Quantum Cosmology collaboration"
                    " network, with 5,242 nodes and 14,496 edges."
                ),
                suites=[],
            ),
        ]

    def generate(self, dataset: ConnectedComponentsDataset) -> DataInstance:
        if dataset.name.startswith("snap"):
            inputs, meta = download_snap_dataset(dataset.name)
            return DataInstance(inputs=inputs, meta=meta)
        raise ValueError(f"Unsupported connected components dataset: {dataset.name}")


class SimplyConnectedComponentsBenchmark(Benchmark):
    @property
    def name(self):
        return "simply_connected_components"

    @property
    def pretty_name(self):
        return "Simply Connected Components"

    @property
    def description(self):
        return (
            "Computes the simply connected components of a directed graph using label"
            " propagation."
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
            Contributor("Willow Ahrens", "ahrens@gatech.edu"),
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
            )
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
    def generators(self) -> list[Generator[ConnectedComponentsDataset]]:
        return [ConnectedComponentsGenerator()]

    def benchmark(self, data, meta):
        edges = xp.from_binsparse(data[0])
        (n, m) = edges.shape
        assert m == n

        # create identity matrix with edges
        graph = xp.array(edges, dtype=bool)
        graph = xp.logical_or(graph, graph.T)
        identity_matrix = xp.eye(n, dtype=bool)
        graph = xp.logical_or(identity_matrix, graph)
        labels = xp.arange(n)
        int_max = xp.iinfo(labels.dtype).max

        # do fixed-point iteration
        max_iterations = n
        for _iteration in range(max_iterations):
            nextLabels = xp.min(xp.where(graph, labels, int_max), axis=1)

            if xp.all(xp.equal(labels, nextLabels)):
                break
            labels = nextLabels
        return [labels]
