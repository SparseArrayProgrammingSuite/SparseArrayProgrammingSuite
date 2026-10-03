# ruff: noqa: E501
import numpy as np
from scipy.sparse import coo_array

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, from_scipy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.benchmarks.adjacency import zero_one_adjacency
from saps.benchmarks.gap import fetch_gap_graph, gap_graph
from saps.benchmarks.snap import (
    fetch_snap_graph,
    with_source_vertex,
)
from saps_framework.binsparse_utils import binsparse_equal


class BFSDataset(Dataset):
    def __init__(
        self,
        name: str | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        src: int | None = None,
        expected: np.ndarray | None = None,
        source_seed: int | None = None,
        source_name: str | None = None,
    ):
        if source_name is not None and source_seed is not None:
            name = f"{source_name}_seed{source_seed}"
            pretty_name = f"{source_name} (Seed {source_seed})"
        elif source_name is not None:
            name = f"{source_name}_src{src}"
            pretty_name = f"{source_name} (Source {src})"
        if name is None:
            raise ValueError("Datasets without a source_name need a name.")
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Breadth-first search input {name}."
        self._suites = list(suites or [])
        self.A = A
        self.src = src
        self.source_seed = source_seed
        self.source_name = source_name or name
        self.expected = expected

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


class BFSTestGenerator(Generator[BFSDataset]):
    @property
    def name(self) -> str:
        return "bfs_test"

    @property
    def pretty_name(self) -> str:
        return "Breadth-First Search Test"

    @property
    def description(self) -> str:
        return "Small deterministic BFS examples with reference outputs."

    @property
    def suites(self) -> list[str]:
        return ["test"]

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
            "Generative AI might have been used to construct tests. This statement "
            "was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Provide small BFS examples for benchmark correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[BFSDataset]:
        return [
            BFSDataset(
                "basic",
                pretty_name="Basic",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 1, 0, 0, 0],
                        [0, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 1],
                        [0, 0, 0, 1, 0, 0],
                    ],
                    dtype=bool,
                ),
                src=0,
                expected=np.array([1, 2, 2, 3, 3, 4], dtype=int),
            ),
            BFSDataset(
                "single_node",
                pretty_name="Single Node",
                suites=["test"],
                A=np.array([[0]], dtype=bool),
                src=0,
                expected=np.array([1], dtype=int),
            ),
            BFSDataset(
                "disconnected",
                pretty_name="Disconnected",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 0, 0],
                        [0, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 0, 0],
                    ],
                    dtype=bool,
                ),
                src=0,
                expected=np.array([1, 2, 0, 0], dtype=int),
            ),
            BFSDataset(
                "undirected",
                pretty_name="Undirected",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 0, 0],
                        [1, 0, 1, 0],
                        [0, 1, 0, 1],
                        [0, 0, 1, 0],
                    ],
                    dtype=bool,
                ),
                src=0,
                expected=np.array([1, 2, 3, 4], dtype=int),
            ),
            BFSDataset(
                "cycle",
                pretty_name="Cycle",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1],
                        [1, 0, 0, 0],
                    ],
                    dtype=bool,
                ),
                src=0,
                expected=np.array([1, 2, 3, 4], dtype=int),
            ),
            BFSDataset(
                "snap_toy",
                pretty_name="SNAP Toy",
                suites=["test"],
                A=np.array(
                    [
                        [0, 1, 0],
                        [0, 0, 1],
                        [0, 0, 0],
                    ],
                    dtype=bool,
                ),
                src=0,
                expected=np.array([1, 2, 3], dtype=int),
            ),
            *[
                BFSDataset(
                    name=f"random_source_seed{seed}",
                    pretty_name=f"Random Source (Seed {seed})",
                    suites=["test"],
                    A=np.array(
                        [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 0]],
                        dtype=bool,
                    ),
                    source_seed=seed,
                )
                for seed in range(10)
            ],
        ]

    def generate(self, dataset: BFSDataset) -> DataInstance:
        if dataset.source_seed is not None:
            from scipy.sparse.csgraph import shortest_path

            if dataset.A is None:
                raise ValueError("Seeded BFS tests require an adjacency matrix.")
            raw = with_source_vertex(
                DataInstance(inputs=[from_scipy(coo_array(dataset.A))], meta={}),
                seed=dataset.source_seed,
            )
            distances = shortest_path(
                dataset.A, directed=True, unweighted=True, indices=raw.meta["src"]
            )
            levels = np.where(np.isfinite(distances), distances + 1, 0).astype(int)
            raw.ref_outputs = [from_numpy(levels)]
            return raw
        if dataset.A is None or dataset.src is None or dataset.expected is None:
            raise ValueError("BFS test datasets must define A, src, and expected.")
        return DataInstance(
            inputs=[from_numpy(dataset.A)],
            meta={"src": dataset.src},
            ref_outputs=[from_numpy(dataset.expected)],
        )


class BFSSNAPGenerator(Generator[BFSDataset]):
    @property
    def name(self) -> str:
        return "bfs_snap"

    @property
    def pretty_name(self) -> str:
        return "Breadth-First Search SNAP"

    @property
    def description(self) -> str:
        return "SNAP input generator for breadth-first search benchmarks."

    @property
    def suites(self) -> list[str]:
        return ["standard"]

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
        return "Generate sparse graph inputs for breadth-first search."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[BFSDataset]:
        # Trace selects successful Smart runs < 60s in competition/run_13803684.
        # fmt: off
        return [
            *[
                BFSDataset(source_name="soc-Epinions1", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="soc-LiveJournal1", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="soc-Pokec", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="soc-Slashdot0811", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="soc-Slashdot0902", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="wiki-Vote", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="wiki-RfA", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="soc-sign-bitcoin-otc", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="soc-sign-bitcoin-alpha", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="com-LiveJournal", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="com-Friendster", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="com-Orkut", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="com-Youtube", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="com-DBLP", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="com-Amazon", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="email-Eu-core", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="wiki-topcats", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="email-EuAll", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="email-Enron", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="wiki-Talk", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="cit-HepPh", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="cit-HepTh", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="cit-Patents", source_seed=seed, suites=["standard", "trace", "train"] if seed == 7 else ["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="ca-AstroPh", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="ca-CondMat", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="ca-GrQc", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="ca-HepPh", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="ca-HepTh", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="web-BerkStan", source_seed=seed, suites=["standard", "trace"] if seed in (0, 2, 5) else ["standard"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="web-Google", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="web-NotreDame", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="web-Stanford", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="amazon0302", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="amazon0312", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="amazon0505", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="amazon0601", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="p2p-Gnutella04", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="p2p-Gnutella05", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="p2p-Gnutella06", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="p2p-Gnutella08", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="p2p-Gnutella09", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="p2p-Gnutella24", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="p2p-Gnutella25", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="p2p-Gnutella30", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="p2p-Gnutella31", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="roadNet-CA", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="roadNet-PA", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="roadNet-TX", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="as-735", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="as-Skitter", source_seed=seed, suites=["standard", "trace"] if seed in (1, 4, 6, 9) else ["standard"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="as-caida", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="Oregon-1", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="Oregon-2", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="soc-sign-epinions", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="soc-sign-Slashdot081106", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="soc-sign-Slashdot090216", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="soc-sign-Slashdot090221", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="loc-Gowalla", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="loc-Brightkite", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="sx-stackoverflow", source_seed=seed, suites=["standard", "trace"] if seed in (1, 2, 3, 5, 6, 9) else ["standard"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="sx-mathoverflow", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="sx-superuser", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="sx-askubuntu", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="wiki-talk-temporal", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="email-Eu-core-temporal", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="CollegeMsg", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="twitter7", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BFSDataset(source_name="higgs-twitter", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
        ]
        # fmt: on

    def generate(self, dataset: BFSDataset) -> DataInstance:
        raw = fetch_snap_graph(dataset.source_name)
        seed = dataset.source_seed
        if seed is None or not 0 <= seed < len(raw.meta["sources"]):
            raise ValueError(
                f"Source seed is outside the graph's available sources: {seed}"
            )
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0])],
            meta={**raw.meta, "src": raw.meta["sources"][seed], "seed": seed},
        )


class BFSGAPGenerator(Generator[BFSDataset]):
    @property
    def name(self) -> str:
        return "bfs_gap"

    @property
    def pretty_name(self) -> str:
        return "Breadth-First Search GAP"

    @property
    def description(self) -> str:
        return "GAP Input generator for breadth-first search benchmarks."

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
                title="The GAP Benchmark Suite",
                authors=[
                    Author("Scott Beamer"),
                    Author("Krste Asanović"),
                    Author("David Patterson"),
                ],
                url="https://arxiv.org/abs/1508.03619",
                year=2015,
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "Generative AI was used to construct the generator and dataset structures."
            " This statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return "Generate GAP graph inputs for breadth-first search."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[BFSDataset]:
        # fmt: off
        return [
            *[
                BFSDataset(source_name="GAP-road", src=src, suites=["standard"])
                for src in gap_graph("GAP-road").sources
            ],
            *[
                BFSDataset(source_name="GAP-twitter", src=src, suites=["standard"])
                for src in gap_graph("GAP-twitter").sources
            ],
            *[
                BFSDataset(source_name="GAP-web", src=src, suites=["standard"])
                for src in gap_graph("GAP-web").sources
            ],
            *[
                BFSDataset(source_name="GAP-kron", src=src, suites=["standard"])
                for src in gap_graph("GAP-kron").sources
            ],
            *[
                BFSDataset(source_name="GAP-urand", src=src, suites=["standard"])
                for src in gap_graph("GAP-urand").sources
            ],
        ]
        # fmt: on

    def generate(self, dataset: BFSDataset) -> DataInstance:
        raw = fetch_gap_graph(dataset.source_name)
        if dataset.src not in raw.meta["sources"]:
            raise ValueError(
                f"{dataset.src} is not a published source of {dataset.source_name}"
            )
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0])],
            meta={**raw.meta, "src": dataset.src},
        )


class BFSBenchmark(Benchmark):
    @property
    def name(self):
        return "bfs"

    @property
    def pretty_name(self):
        return "Breadth-First Search"

    @property
    def description(self):
        return (
            "The Breadth-First Search algorithm is an important graph traversal"
            " technique used to explore vertices by layers. It is a fundamental"
            " building block for more complex graph algorithms, especially in areas"
            " like parallel processing and high-performance computing."
        )

    @property
    def suites(self) -> list[str]:
        return ["standard-graphs-iterative"]

    @property
    def concepts(self) -> str:
        return """
<ccs2012>
<concept>
<concept_id>10002950.10003705</concept_id>
<concept_desc>Mathematics of computing~Mathematical software</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003705.10011686</concept_id>
<concept_desc>Mathematics of computing~Mathematical software performance</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003624.10003633.10010917</concept_id>
<concept_desc>Mathematics of computing~Graph algorithms</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10002950.10003624.10003633.10003640</concept_id>
<concept_desc>Mathematics of computing~Paths and connectivity problems</concept_desc>
<concept_significance>500</concept_significance>
</concept>
</ccs2012>
"""

    @property
    def authors(self) -> list[Contributor]:
        return [
            Contributor("Aarav Joglekar", "ajoglekar32@gatech.edu"),
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
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI might have been used to construct tests. This statement was"
            " written by hand."
        )

    @property
    def motivation(self):
        return (
            "In standard BFS, algorithms on sparse graphs are faster because they"
            " process fewer edges, and specialized algebraic methods use sparsity to"
            " avoid unnecessary computations by focusing only on non-zero elements."
            " Optimizing the use of sparse data structures and algorithms is key to"
            " achieving high performance, as it reduces memory footprint and leads to"
            " faster traversals."
        )

    @property
    def generators(self):
        return [
            BFSSNAPGenerator(),
            BFSTestGenerator(),
            BFSGAPGenerator(),
        ]

    def benchmark(self, xp, data: list, meta: dict):
        edges = data[0]
        src = meta["src"]

        (n, m) = edges.shape
        assert n == m
        visited = xp.zeros((n,), dtype=bool)
        frontier = xp.zeros((n,), dtype=bool)
        frontier[src] = True
        level = xp.zeros((n,), dtype=int)
        level_idx = 1
        frontier_count = 1
        while frontier_count > 0:
            level = xp.where(frontier, level_idx, level)
            visited = xp.logical_or(visited, frontier)
            frontier = xp.einsum(
                "frontier[j] += edges[i,j] * frontier[i]",
                edges=edges,
                frontier=frontier,
            )
            frontier = xp.logical_and(frontier, xp.logical_not(visited))
            frontier_count = xp.sum(frontier)

            level_idx += 1

        return [level]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return
        assert binsparse_equal(self._output[0], self._ref_outputs[0]), (
            f"BFS output mismatch for {param.dataset.name}"
        )
