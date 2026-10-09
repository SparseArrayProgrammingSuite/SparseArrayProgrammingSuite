# ruff: noqa: E501
import numpy as np

import sparse as sp
from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, to_numpy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.benchmarks.adjacency import (
    DEFAULT_MAX_DENSITY,
    distance_matrix,
    squaring_count,
    zero_one_adjacency,
)
from saps.benchmarks.gap import fetch_gap_graph
from saps.benchmarks.snap import fetch_snap_graph


class FloydWarshallDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        A=None,
        expected=None,
        ref_meta: dict | None = None,
        max_density: float = DEFAULT_MAX_DENSITY,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Floyd-Warshall input {name}."
        self._suites = list(suites or [])
        self.A = A
        self.expected = expected
        self.ref_meta = ref_meta
        self.max_density = max_density

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


def floyd_warshall_reference(A):
    if isinstance(A, sp.SparseArray):
        A = A.todense()
    expected = A.copy()
    for k in range(expected.shape[0]):
        expected = np.minimum(
            expected,
            np.expand_dims(expected[:, k], axis=1)
            + np.expand_dims(expected[k, :], axis=0),
        )
    return expected


def floyd_warshall_input_from_edges(
    n: int, edges: list[tuple[int, int]], *, symmetric: bool = False
):
    rows = [*range(n)]
    cols = [*range(n)]
    values = [0.0] * n
    for u, v in edges:
        rows.append(u)
        cols.append(v)
        values.append(1.0)
        if symmetric:
            rows.append(v)
            cols.append(u)
            values.append(1.0)
    return sp.COO(
        coords=np.array([rows, cols], dtype=np.int64),
        data=np.array(values, dtype=np.float64),
        shape=(n, n),
        fill_value=np.inf,
    )


class FloydWarshallTestGenerator(Generator[FloydWarshallDataset]):
    @property
    def name(self) -> str:
        return "floyd_warshall_test"

    @property
    def pretty_name(self) -> str:
        return "Floyd-Warshall Test"

    @property
    def description(self) -> str:
        return "Small deterministic Floyd-Warshall examples."

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
        return "Provide small graph examples for shortest-path correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[FloydWarshallDataset]:
        return [
            FloydWarshallDataset(
                name="single_node",
                pretty_name="Single Node",
                description="Floyd-Warshall test case single-node.",
                suites=["test"],
                A=np.array([[0.0]]),
                expected=np.array([[0.0]]),
            ),
            FloydWarshallDataset(
                name="two_node_directed",
                pretty_name="Two Node Directed",
                description="Floyd-Warshall test case two-node-directed.",
                suites=["test"],
                A=np.array([[0.0, 1.0], [np.inf, 0.0]]),
                expected=np.array([[0.0, 1.0], [np.inf, 0.0]]),
            ),
            FloydWarshallDataset(
                name="three_node_chain",
                pretty_name="Three Node Chain",
                description="Floyd-Warshall test case three-node-chain.",
                suites=["test"],
                A=np.array(
                    [
                        [0.0, 1.0, np.inf],
                        [np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 0.0],
                    ]
                ),
                expected=np.array(
                    [
                        [0.0, 1.0, 2.0],
                        [np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 0.0],
                    ]
                ),
            ),
            FloydWarshallDataset(
                name="three_node_shortcut",
                pretty_name="Three Node Shortcut",
                description="Floyd-Warshall test case three-node-shortcut.",
                suites=["test"],
                A=np.array(
                    [
                        [0.0, 1.0, 5.0],
                        [np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 0.0],
                    ]
                ),
                expected=np.array(
                    [
                        [0.0, 1.0, 2.0],
                        [np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 0.0],
                    ]
                ),
            ),
            FloydWarshallDataset(
                name="two_components",
                pretty_name="Two Components",
                description="Floyd-Warshall test case two-components.",
                suites=["test"],
                A=np.array(
                    [
                        [0.0, 1.0, np.inf, np.inf],
                        [1.0, 0.0, np.inf, np.inf],
                        [np.inf, np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 1.0, 0.0],
                    ]
                ),
                expected=np.array(
                    [
                        [0.0, 1.0, np.inf, np.inf],
                        [1.0, 0.0, np.inf, np.inf],
                        [np.inf, np.inf, 0.0, 1.0],
                        [np.inf, np.inf, 1.0, 0.0],
                    ]
                ),
            ),
            FloydWarshallDataset(
                name="large_symmetric",
                pretty_name="Large Symmetric",
                description="Floyd-Warshall test case large-symmetric.",
                suites=["test"],
                A=floyd_warshall_input_from_edges(
                    39,
                    [
                        (0, 1),
                        (0, 2),
                        (0, 3),
                        (0, 4),
                        (0, 5),
                        (0, 6),
                        (0, 7),
                        (0, 8),
                        (1, 9),
                        (1, 10),
                        (1, 11),
                        (1, 12),
                        (1, 13),
                        (1, 14),
                        (1, 15),
                        (1, 16),
                        (2, 9),
                        (2, 10),
                        (2, 17),
                        (2, 18),
                        (3, 11),
                        (3, 12),
                        (3, 19),
                        (3, 20),
                        (3, 21),
                        (4, 13),
                        (4, 22),
                        (4, 23),
                        (4, 24),
                        (5, 14),
                        (5, 22),
                        (5, 25),
                        (5, 26),
                        (6, 15),
                        (6, 23),
                        (6, 27),
                        (6, 28),
                        (7, 16),
                        (7, 24),
                        (7, 29),
                        (7, 30),
                        (8, 17),
                        (8, 18),
                        (8, 19),
                        (8, 20),
                        (8, 21),
                        (9, 22),
                        (9, 31),
                        (9, 32),
                        (10, 23),
                        (10, 31),
                        (10, 33),
                        (11, 24),
                        (11, 32),
                        (11, 34),
                        (12, 25),
                        (12, 26),
                        (12, 35),
                        (13, 27),
                        (13, 36),
                        (14, 28),
                        (14, 37),
                        (15, 29),
                        (15, 38),
                        (16, 30),
                        (17, 31),
                        (18, 32),
                        (19, 33),
                        (20, 34),
                        (21, 35),
                        (22, 36),
                        (23, 37),
                        (24, 38),
                        (25, 27),
                        (25, 29),
                        (26, 28),
                        (26, 30),
                        (27, 31),
                        (27, 33),
                        (28, 32),
                        (28, 34),
                        (29, 35),
                        (30, 36),
                        (31, 37),
                        (32, 38),
                        (33, 35),
                        (34, 36),
                        (35, 37),
                        (36, 38),
                        (37, 38),
                    ],
                    symmetric=True,
                ),
                ref_meta={"large_symmetric": True},
            ),
        ]

    def generate(self, dataset: FloydWarshallDataset):
        inputs = (
            dataset.A.todense() if isinstance(dataset.A, sp.SparseArray) else dataset.A
        )
        expected = dataset.expected
        if expected is None:
            expected = floyd_warshall_reference(inputs)
        return DataInstance(
            inputs=[from_numpy(inputs)],
            # Correctness fixtures exercise full shortest paths, even when dense.
            meta={"max_squarings": max(0, len(inputs) - 2).bit_length()},
            ref_outputs=[from_numpy(expected)],
            ref_meta=dataset.ref_meta,
        )


class FloydWarshallSNAPGenerator(Generator[FloydWarshallDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "floyd_warshall_snap"

    @property
    def pretty_name(self) -> str:
        return "Floyd-Warshall SNAP"

    @property
    def description(self) -> str:
        return "SNAP graphs converted to unit-length shortest-path inputs for Floyd-Warshall."

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
        return "Generative AI was used to construct this generator."

    @property
    def motivation(self) -> str:
        return (
            "Generate unit-length shortest-path inputs with the stored edge directions."
        )

    @property
    def datasets(self) -> list[FloydWarshallDataset]:
        # fmt: off
        return [
            FloydWarshallDataset("soc-Epinions1", suites=["standard", "trace"]),
            FloydWarshallDataset("soc-LiveJournal1", suites=["standard"]),
            FloydWarshallDataset("soc-Pokec", suites=["standard", "trace"]),
            FloydWarshallDataset("soc-Slashdot0811", suites=["standard", "trace"]),
            FloydWarshallDataset("soc-Slashdot0902", suites=["standard", "trace"]),
            FloydWarshallDataset("wiki-Vote", suites=["standard", "trace"]),
            FloydWarshallDataset("wiki-RfA", suites=["standard", "trace"]),
            FloydWarshallDataset("soc-sign-bitcoin-otc", suites=["standard", "trace"]),
            FloydWarshallDataset("soc-sign-bitcoin-alpha", suites=["standard", "trace"]),
            FloydWarshallDataset("com-LiveJournal", suites=["standard"]),
            FloydWarshallDataset("com-Friendster", suites=["standard"]),
            FloydWarshallDataset("com-Orkut", suites=["standard"]),
            FloydWarshallDataset("com-Youtube", suites=["standard", "trace"]),
            FloydWarshallDataset("com-DBLP", suites=["standard", "trace"]),
            FloydWarshallDataset("com-Amazon", suites=["standard", "trace"]),
            FloydWarshallDataset("email-Eu-core", suites=["standard", "trace"]),
            FloydWarshallDataset("wiki-topcats", suites=["standard", "trace"]),
            FloydWarshallDataset("email-EuAll", suites=["standard", "trace"]),
            FloydWarshallDataset("email-Enron", suites=["standard", "trace"]),
            FloydWarshallDataset("wiki-Talk", suites=["standard", "trace"]),
            FloydWarshallDataset("cit-HepPh", suites=["standard", "trace"]),
            FloydWarshallDataset("cit-HepTh", suites=["standard", "trace"]),
            FloydWarshallDataset("cit-Patents", suites=["standard", "trace"]),
            FloydWarshallDataset("ca-AstroPh", suites=["standard", "trace"]),
            FloydWarshallDataset("ca-CondMat", suites=["standard", "trace"]),
            FloydWarshallDataset("ca-GrQc", suites=["standard", "trace"]),
            FloydWarshallDataset("ca-HepPh", suites=["standard", "trace"]),
            FloydWarshallDataset("ca-HepTh", suites=["standard", "trace"]),
            FloydWarshallDataset("web-BerkStan", suites=["standard", "trace"]),
            FloydWarshallDataset("web-Google", suites=["standard", "trace"]),
            FloydWarshallDataset("web-NotreDame", suites=["standard", "trace"]),
            FloydWarshallDataset("web-Stanford", suites=["standard", "trace"]),
            FloydWarshallDataset("amazon0302", suites=["standard"]),
            FloydWarshallDataset("amazon0312", suites=["standard"]),
            FloydWarshallDataset("amazon0505", suites=["standard"]),
            FloydWarshallDataset("amazon0601", suites=["standard"]),
            FloydWarshallDataset("p2p-Gnutella04", suites=["standard", "trace"]),
            FloydWarshallDataset("p2p-Gnutella05", suites=["standard", "trace"]),
            FloydWarshallDataset("p2p-Gnutella06", suites=["standard", "trace"]),
            FloydWarshallDataset("p2p-Gnutella08", suites=["standard", "trace"]),
            FloydWarshallDataset("p2p-Gnutella09", suites=["standard", "trace"]),
            FloydWarshallDataset("p2p-Gnutella24", suites=["standard", "trace"]),
            FloydWarshallDataset("p2p-Gnutella25", suites=["standard", "trace"]),
            FloydWarshallDataset("p2p-Gnutella30", suites=["standard", "trace"]),
            FloydWarshallDataset("p2p-Gnutella31", suites=["standard", "trace"]),
            FloydWarshallDataset("roadNet-CA", suites=["standard"]),
            FloydWarshallDataset("roadNet-PA", suites=["standard"]),
            FloydWarshallDataset("roadNet-TX", suites=["standard"]),
            FloydWarshallDataset("as-735", suites=["standard", "trace"]),
            FloydWarshallDataset("as-Skitter", suites=["standard", "trace"]),
            FloydWarshallDataset("as-caida", suites=["standard", "trace"]),
            FloydWarshallDataset("Oregon-1", suites=["standard", "trace"]),
            FloydWarshallDataset("Oregon-2", suites=["standard", "trace"]),
            FloydWarshallDataset("soc-sign-epinions", suites=["standard", "trace"]),
            FloydWarshallDataset("soc-sign-Slashdot081106", suites=["standard", "trace"]),
            FloydWarshallDataset("soc-sign-Slashdot090216", suites=["standard", "trace"]),
            FloydWarshallDataset("soc-sign-Slashdot090221", suites=["standard", "trace"]),
            FloydWarshallDataset("loc-Gowalla", suites=["standard", "trace"]),
            FloydWarshallDataset("loc-Brightkite", suites=["standard", "trace"]),
            FloydWarshallDataset("sx-stackoverflow", suites=["standard", "trace", "train"]),
            FloydWarshallDataset("sx-mathoverflow", suites=["standard", "trace"]),
            FloydWarshallDataset("sx-superuser", suites=["standard", "trace"]),
            FloydWarshallDataset("sx-askubuntu", suites=["standard", "trace"]),
            FloydWarshallDataset("wiki-talk-temporal", suites=["standard", "trace"]),
            FloydWarshallDataset("email-Eu-core-temporal", suites=["standard", "trace"]),
            FloydWarshallDataset("CollegeMsg", suites=["standard", "trace"]),
            FloydWarshallDataset("twitter7", suites=["standard"]),
            FloydWarshallDataset("higgs-twitter", suites=["standard", "trace"]),
        ]
        # fmt: on

    def generate(self, dataset: FloydWarshallDataset) -> DataInstance:
        raw = fetch_snap_graph(dataset.name)
        return DataInstance(
            inputs=[distance_matrix(zero_one_adjacency(raw.inputs[0]))],
            meta={
                **raw.meta,
                "max_squarings": squaring_count(
                    raw.inputs[0].shape[0], raw.meta["max_degree"], dataset.max_density
                ),
            },
        )


class FloydWarshallGAPGenerator(Generator[FloydWarshallDataset]):
    @property
    def name(self) -> str:
        return "floyd_warshall_gap"

    @property
    def pretty_name(self) -> str:
        return "Floyd-Warshall GAP"

    @property
    def description(self) -> str:
        return (
            "GAP benchmark graphs from the SuiteSparse Matrix Collection, converted"
            " into weighted all-pairs shortest path inputs that keep each graph's"
            " edge weights."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

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
                year=2011,
            ),
            Ref(
                title=(
                    "The Network Data Repository with Interactive"
                    " Graph Analytics and Visualization"
                ),
                authors=[
                    Author("Ryan A. Rossi"),
                    Author("Nesreen K. Ahmed"),
                ],
                journal="AAAI",
                url="https://networkrepository.com",
                year=2015,
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI might have been used to construct tests. This statement was"
            " written by hand."
        )

    @property
    def motivation(self) -> str:
        return ""

    @property
    def datasets(self) -> list[FloydWarshallDataset]:
        # fmt: off
        return [
            FloydWarshallDataset("GAP-road", suites=["standard"]),
            FloydWarshallDataset("GAP-twitter", suites=["standard"]),
            FloydWarshallDataset("GAP-web", suites=["standard"]),
            FloydWarshallDataset("GAP-kron", suites=["standard"]),
            FloydWarshallDataset("GAP-urand", suites=["standard"]),
        ]
        # fmt: on
        # fmt: on

    @property
    def cacheable(self) -> bool:
        return False

    def generate(self, dataset: FloydWarshallDataset):
        raw = fetch_gap_graph(dataset.name)
        # GAP graphs are weighted, so edges keep their lengths.
        G = distance_matrix(raw.inputs[0], keep_weights=True)
        degree = raw.meta["max_degree"]
        return DataInstance(
            inputs=[G],
            meta={
                **raw.meta,
                "max_squarings": squaring_count(
                    G.shape[0], degree, dataset.max_density
                ),
            },
        )


class FloydWarshallBenchmark(Benchmark):
    @property
    def name(self):
        return "floyd_warshall"

    @property
    def pretty_name(self):
        return "Floyd-Warshall"

    @property
    def description(self):
        return (
            "Computes bounded-hop all-pairs shortest paths by min-plus matrix"
            " squaring. A maximum-degree bound limits the squaring count to keep"
            " finite-entry density at most 1% by default. Stops earlier at a fixed"
            " point; a density-limited result may omit longer paths. Inputs already"
            " above the bound receive no squarings."
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
                publisher="SIAM",
                year=2011,
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function itself."
            " Generative AI might have been used to construct tests. This statement was"
            " written by hand."
        )

    @property
    def motivation(self):
        return (
            "Sparse graphs reduce unnecessary computation, as most entries in the"
            " adjacency matrix represent non-edges and begin as inifinity. Efficient"
            " sparse representations allow the backend framework to skip work and"
            " minimize memory movement during the relaxation steps of the algorithm."
        )

    @property
    def generators(self):
        return [
            FloydWarshallTestGenerator(),
            FloydWarshallSNAPGenerator(),
            FloydWarshallGAPGenerator(),
        ]

    def benchmark(self, xp, meta, G):
        """
        Return shortest paths using at most 2**max_squarings edges.

        Inputs have a zero diagonal and no negative cycles. With a sufficient
        squaring budget this computes all-pairs shortest paths.
        """
        n, m = G.shape
        assert n == m
        for _iteration in range(meta["max_squarings"]):
            next_G = xp.einsum("D[i,j] min= G[i,k] + G[k,j]", G=G)
            if xp.all(xp.equal(G, next_G)):
                break
            G = next_G
        return G

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        output = to_numpy(self._output[0])
        if self._ref_outputs is not None:
            expected = to_numpy(self._ref_outputs[0])
            both_inf = np.isinf(output) & np.isinf(expected)
            both_finite = np.isfinite(output) & np.isfinite(expected)
            assert np.all(both_inf | (both_finite & (output == expected))), (
                f"Floyd-Warshall output mismatch for {param.dataset.name}"
            )
        if self._ref_meta and self._ref_meta.get("large_symmetric"):
            assert output.shape[0] == output.shape[1]
            assert np.all(np.diag(output) == 0.0)
            assert np.all(output >= 0.0)
            assert np.all(output == output.T)
            rng = np.random.default_rng(0)
            for _ in range(50):
                i, j, k = rng.integers(0, output.shape[0], size=3)
                assert output[i, j] <= output[i, k] + output[k, j]
