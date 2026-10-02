# ruff: noqa: E501
import numpy as np
import scipy.sparse as sps

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_sparse

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.benchmarks.adjacency import distance_matrix
from saps.benchmarks.gap import fetch_gap_graph, gap_graph
from saps.benchmarks.snap import (
    fetch_snap_graph,
    with_source_vertex,
)


def _from_binsparse(array):
    try:
        return to_numpy(array)
    except TypeError:
        return to_sparse(array).todense()


class BellmanFordDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        src: int = 0,
        expected: np.ndarray | None = None,
        source_seed: int | None = None,
        source_name: str | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Bellman-Ford input {name}."
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


def bellman_ford_reference(A, src):
    n = A.shape[0]
    D = np.full((n,), np.inf)
    D[src] = 0
    for _ in range(n):
        for v in range(n):
            for u in range(n):
                if A[u, v] + D[u] < D[v]:
                    D[v] = A[u, v] + D[u]
    return D


def bellman_ford_matrix(n, edges, *, symmetric=False):
    A = np.full((n, n), np.inf)
    np.fill_diagonal(A, 0)
    for u, v in edges:
        A[u, v] = 1.0
        if symmetric:
            A[v, u] = 1.0
    return A


class BellmanFordTestGenerator(Generator[BellmanFordDataset]):
    @property
    def name(self) -> str:
        return "bellman_ford_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "Bellman-Ford Test Input Generator"

    @property
    def description(self) -> str:
        return "Small deterministic Bellman-Ford examples."

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
        return "Provide small graph examples for Bellman-Ford correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[BellmanFordDataset]:
        tribes = bellman_ford_matrix(
            16,
            [
                (0, 1),
                (1, 0),
                (0, 2),
                (2, 0),
                (1, 2),
                (2, 1),
                (0, 3),
                (3, 0),
                (2, 3),
                (3, 2),
                (0, 4),
                (4, 0),
                (1, 4),
                (4, 1),
                (0, 5),
                (5, 0),
                (1, 5),
                (5, 1),
                (2, 5),
                (5, 2),
                (2, 6),
                (6, 2),
                (4, 6),
                (6, 4),
                (5, 6),
                (6, 5),
                (2, 7),
                (7, 2),
                (3, 7),
                (7, 3),
                (5, 7),
                (7, 5),
                (6, 7),
                (7, 6),
                (1, 8),
                (8, 1),
                (4, 8),
                (8, 4),
                (7, 8),
                (8, 7),
                (3, 9),
                (9, 3),
                (8, 9),
                (9, 8),
                (9, 10),
                (10, 9),
                (10, 11),
                (11, 10),
                (8, 11),
                (11, 8),
                (11, 12),
                (12, 11),
                (7, 12),
                (12, 7),
                (12, 13),
                (13, 12),
                (13, 14),
                (14, 13),
                (9, 14),
                (14, 9),
                (14, 15),
                (15, 14),
                (12, 15),
                (15, 12),
            ],
        )
        chesapeake = bellman_ford_matrix(
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
        )
        return [
            BellmanFordDataset(
                name="test_bellman_ford_tribes_src_0",
                suites=["test"],
                A=tribes,
                src=0,
            ),
            BellmanFordDataset(
                name="test_bellman_ford_chesapeake_src_0",
                suites=["test"],
                A=chesapeake,
                src=0,
            ),
            BellmanFordDataset(
                name="test_bellman_ford_chesapeake_src_10",
                suites=["test"],
                A=chesapeake,
                src=10,
            ),
            BellmanFordDataset(
                name="test_bellman_ford_chesapeake_src_38",
                suites=["test"],
                A=chesapeake,
                src=38,
            ),
            BellmanFordDataset(
                name="test_bellman_ford_snap_toy",
                suites=["test"],
                A=bellman_ford_matrix(3, [(0, 1), (1, 2)]),
                src=0,
                expected=np.array([0.0, 1.0, 2.0]),
            ),
            *[
                BellmanFordDataset(
                    name=f"test_bellmanford_snap_source_seed{seed}",
                    suites=["test"],
                    A=bellman_ford_matrix(4, [(1, 2), (2, 3)]),
                    source_seed=seed,
                )
                for seed in range(10)
            ],
        ]

    def generate(self, dataset: BellmanFordDataset) -> DataInstance:
        if dataset.source_seed is not None:
            if dataset.A is None:
                raise ValueError(
                    "Seeded Bellman-Ford tests require an adjacency matrix."
                )
            adjacency = np.isfinite(dataset.A) & (dataset.A != 0)
            raw = with_source_vertex(
                DataInstance(inputs=[from_scipy(sps.coo_array(adjacency))], meta={}),
                seed=dataset.source_seed,
            )
            return DataInstance(
                inputs=[distance_matrix(raw.inputs[0])],
                meta=raw.meta,
                ref_outputs=[
                    from_numpy(bellman_ford_reference(dataset.A, raw.meta["src"]))
                ],
            )
        if dataset.A is None:
            raise ValueError("Bellman-Ford test datasets must define A.")
        expected = dataset.expected
        if expected is None:
            expected = bellman_ford_reference(dataset.A, dataset.src)
        return DataInstance(
            inputs=[from_numpy(dataset.A)],
            meta={"src": dataset.src},
            ref_outputs=[from_numpy(expected)],
        )


class BellmanFordSNAPGenerator(Generator[BellmanFordDataset]):
    @property
    def name(self) -> str:
        return "bellman_ford_snap_inputs"

    @property
    def pretty_name(self) -> str:
        return "Bellman-Ford SNAP Input Generator"

    @property
    def description(self) -> str:
        return "SNAP input generator for Bellman-Ford shortest-path benchmarks."

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
        return "Generate weighted graph inputs for Bellman-Ford."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[BellmanFordDataset]:
        # Trace selects successful Smart runs < 60s in competition/run_13803684.
        # fmt: off
        return [
            *[
                BellmanFordDataset(f"soc-Epinions1_seed{seed}", source_name="soc-Epinions1", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"soc-LiveJournal1_seed{seed}", source_name="soc-LiveJournal1", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"soc-Pokec_seed{seed}", source_name="soc-Pokec", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"soc-Slashdot0811_seed{seed}", source_name="soc-Slashdot0811", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"soc-Slashdot0902_seed{seed}", source_name="soc-Slashdot0902", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"wiki-Vote_seed{seed}", source_name="wiki-Vote", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"wiki-RfA_seed{seed}", source_name="wiki-RfA", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"soc-sign-bitcoin-otc_seed{seed}", source_name="soc-sign-bitcoin-otc", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"soc-sign-bitcoin-alpha_seed{seed}", source_name="soc-sign-bitcoin-alpha", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"com-LiveJournal_seed{seed}", source_name="com-LiveJournal", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"com-Friendster_seed{seed}", source_name="com-Friendster", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"com-Orkut_seed{seed}", source_name="com-Orkut", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"com-Youtube_seed{seed}", source_name="com-Youtube", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"com-DBLP_seed{seed}", source_name="com-DBLP", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"com-Amazon_seed{seed}", source_name="com-Amazon", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"email-Eu-core_seed{seed}", source_name="email-Eu-core", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"wiki-topcats_seed{seed}", source_name="wiki-topcats", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"email-EuAll_seed{seed}", source_name="email-EuAll", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"email-Enron_seed{seed}", source_name="email-Enron", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"wiki-Talk_seed{seed}", source_name="wiki-Talk", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"cit-HepPh_seed{seed}", source_name="cit-HepPh", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"cit-HepTh_seed{seed}", source_name="cit-HepTh", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"cit-Patents_seed{seed}", source_name="cit-Patents", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"ca-AstroPh_seed{seed}", source_name="ca-AstroPh", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"ca-CondMat_seed{seed}", source_name="ca-CondMat", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"ca-GrQc_seed{seed}", source_name="ca-GrQc", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"ca-HepPh_seed{seed}", source_name="ca-HepPh", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"ca-HepTh_seed{seed}", source_name="ca-HepTh", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"web-BerkStan_seed{seed}", source_name="web-BerkStan", source_seed=seed, suites=["standard", "trace"] if seed in (0, 2, 5) else ["standard"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"web-Google_seed{seed}", source_name="web-Google", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"web-NotreDame_seed{seed}", source_name="web-NotreDame", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"web-Stanford_seed{seed}", source_name="web-Stanford", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"amazon0302_seed{seed}", source_name="amazon0302", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"amazon0312_seed{seed}", source_name="amazon0312", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"amazon0505_seed{seed}", source_name="amazon0505", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"amazon0601_seed{seed}", source_name="amazon0601", source_seed=seed, suites=["standard", "trace", "train"] if seed == 4 else ["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"p2p-Gnutella04_seed{seed}", source_name="p2p-Gnutella04", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"p2p-Gnutella05_seed{seed}", source_name="p2p-Gnutella05", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"p2p-Gnutella06_seed{seed}", source_name="p2p-Gnutella06", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"p2p-Gnutella08_seed{seed}", source_name="p2p-Gnutella08", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"p2p-Gnutella09_seed{seed}", source_name="p2p-Gnutella09", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"p2p-Gnutella24_seed{seed}", source_name="p2p-Gnutella24", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"p2p-Gnutella25_seed{seed}", source_name="p2p-Gnutella25", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"p2p-Gnutella30_seed{seed}", source_name="p2p-Gnutella30", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"p2p-Gnutella31_seed{seed}", source_name="p2p-Gnutella31", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"roadNet-CA_seed{seed}", source_name="roadNet-CA", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"roadNet-PA_seed{seed}", source_name="roadNet-PA", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"roadNet-TX_seed{seed}", source_name="roadNet-TX", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"as-735_seed{seed}", source_name="as-735", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"as-Skitter_seed{seed}", source_name="as-Skitter", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"as-caida_seed{seed}", source_name="as-caida", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"Oregon-1_seed{seed}", source_name="Oregon-1", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"Oregon-2_seed{seed}", source_name="Oregon-2", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"soc-sign-epinions_seed{seed}", source_name="soc-sign-epinions", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"soc-sign-Slashdot081106_seed{seed}", source_name="soc-sign-Slashdot081106", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"soc-sign-Slashdot090216_seed{seed}", source_name="soc-sign-Slashdot090216", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"soc-sign-Slashdot090221_seed{seed}", source_name="soc-sign-Slashdot090221", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"loc-Gowalla_seed{seed}", source_name="loc-Gowalla", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"loc-Brightkite_seed{seed}", source_name="loc-Brightkite", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"sx-stackoverflow_seed{seed}", source_name="sx-stackoverflow", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"sx-mathoverflow_seed{seed}", source_name="sx-mathoverflow", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"sx-superuser_seed{seed}", source_name="sx-superuser", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"sx-askubuntu_seed{seed}", source_name="sx-askubuntu", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"wiki-talk-temporal_seed{seed}", source_name="wiki-talk-temporal", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"email-Eu-core-temporal_seed{seed}", source_name="email-Eu-core-temporal", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"CollegeMsg_seed{seed}", source_name="CollegeMsg", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"twitter7_seed{seed}", source_name="twitter7", source_seed=seed, suites=["standard"])
                for seed in range(10)
            ],
            *[
                BellmanFordDataset(f"higgs-twitter_seed{seed}", source_name="higgs-twitter", source_seed=seed, suites=["standard", "trace"])
                for seed in range(10)
            ],
        ]
        # fmt: on

    def generate(self, dataset: BellmanFordDataset) -> DataInstance:
        raw = fetch_snap_graph(dataset.source_name)
        seed = dataset.source_seed
        if seed is None or not 0 <= seed < len(raw.meta["sources"]):
            raise ValueError(
                f"Source seed is outside the graph's available sources: {seed}"
            )
        return DataInstance(
            inputs=[distance_matrix(raw.inputs[0])],
            meta={**raw.meta, "src": raw.meta["sources"][seed], "seed": seed},
        )


class BellmanFordGAPGenerator(Generator[BellmanFordDataset]):
    @property
    def name(self) -> str:
        return "bellman_ford_gap_inputs"

    @property
    def pretty_name(self) -> str:
        return "Bellman-Ford GAP Input Generator"

    @property
    def description(self) -> str:
        return "Input GAP generator for Bellman-Ford shortest-path benchmarks."

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
        return "Generate weighted GAP graph inputs for Bellman-Ford."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[BellmanFordDataset]:
        # fmt: off
        return [
            *[
                BellmanFordDataset(f"GAP/GAP-road_{src}", source_name="GAP-road", src=src, suites=["standard"])
                for src in gap_graph("GAP-road").sources
            ],
            *[
                BellmanFordDataset(f"GAP/GAP-twitter_{src}", source_name="GAP-twitter", src=src, suites=["standard"])
                for src in gap_graph("GAP-twitter").sources
            ],
            *[
                BellmanFordDataset(f"GAP/GAP-web_{src}", source_name="GAP-web", src=src, suites=["standard"])
                for src in gap_graph("GAP-web").sources
            ],
            *[
                BellmanFordDataset(f"GAP/GAP-kron_{src}", source_name="GAP-kron", src=src, suites=["standard"])
                for src in gap_graph("GAP-kron").sources
            ],
            *[
                BellmanFordDataset(f"GAP/GAP-urand_{src}", source_name="GAP-urand", src=src, suites=["standard"])
                for src in gap_graph("GAP-urand").sources
            ],
        ]
        # fmt: on

    def generate(self, dataset: BellmanFordDataset) -> DataInstance:
        raw = fetch_gap_graph(dataset.source_name)
        if dataset.src not in raw.meta["sources"]:
            raise ValueError(
                f"{dataset.src} is not a published source of {dataset.source_name}"
            )
        return DataInstance(
            inputs=[distance_matrix(raw.inputs[0], keep_weights=True)],
            meta={**raw.meta, "src": dataset.src},
        )


class BellmanFordBenchmark(Benchmark):
    @property
    def name(self):
        return "bellman_ford"

    @property
    def pretty_name(self):
        return "Bellman Ford Algorithm"

    @property
    def description(self):
        return (
            "This code implements an Array-API compatible version of Bellman Ford"
            " Algorithm to find the shortest distance from a src node to all edges"
            " across a graph. It takes in an adjacency matrix as an input and then"
            " slowly relaxes each vector by broadcasting it and then determining the"
            " minimum distances iteratively."
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
            Contributor("Ilisha Gupta", "igupta90@gatech.edu"),
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
        return (
            "Linear algebraic graph algorithms use sparsity to avoid unnecessary"
            " computations by focusing only on non-zero elements. Optimizing the use of"
            " sparse data structures and algorithms is key to achieving high"
            " performance, as it reduces memory footprint and leads to faster"
            " traversals."
        )

    @property
    def generators(self):
        return [
            BellmanFordTestGenerator(),
            BellmanFordSNAPGenerator(),
            BellmanFordGAPGenerator(),
        ]

    def benchmark(self, xp, data, meta):
        edges = data[0]
        src = meta["src"]

        n = edges.shape[0]

        G = xp.asarray(edges, dtype=float)
        D = xp.full((n,), xp.inf)
        D[src] = 0

        for _ in range(n):
            D_prev = D
            candidates = xp.expand_dims(D, 1) + G
            D = xp.minimum(D, candidates.min(axis=0))
            stop = xp.all(D_prev == D)
            if stop:
                break

        return [D]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return

        result = _from_binsparse(self._output[0])
        expected = _from_binsparse(self._ref_outputs[0])
        assert np.allclose(result, expected, equal_nan=True), (
            f"Bellman-Ford output mismatch for {param.dataset.name}"
        )
