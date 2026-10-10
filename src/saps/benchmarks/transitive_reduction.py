# ruff: noqa: E501
import numpy as np

from binsparse import BinsparseTensor, COORMatrix
from binsparse.conversions import from_numpy, to_numpy, to_scipy

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.util.adjacency import zero_one_adjacency
from saps.benchmarks.gap import fetch_gap_graph
from saps.benchmarks.snap import fetch_snap_graph


class TransitiveReductionDataset(Dataset):
    def __init__(
        self,
        name,
        edges=None,
        expected_edges=None,
        suites=None,
        pretty_name=None,
        description=None,
    ):
        self._name = name
        self.edges = edges
        self.expected_edges = expected_edges
        self._suites = list(suites or [])
        self._pretty_name = pretty_name or name
        self._description = (
            description
            or "Small overlap graph with expected transitive reduction output."
        )

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


class TransitiveReductionTestGenerator(Generator[TransitiveReductionDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "transitive_reduction_test"

    @property
    def pretty_name(self) -> str:
        return "Transitive Reduction Test"

    @property
    def description(self) -> str:
        return "Small test graphs with expected reduced edges."

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
        return "No generative AI was used to construct the benchmark function."

    @property
    def motivation(self) -> str:
        return ""

    @property
    def datasets(self) -> list[TransitiveReductionDataset]:
        return [
            TransitiveReductionDataset(
                "remove_long_direct_edge",
                pretty_name="Remove Long Direct Edge",
                edges=[(0, 1, 10.0), (1, 2, 10.0), (0, 2, 30.0)],
                expected_edges=[(0, 1, 10.0), (1, 2, 10.0)],
                suites=["test"],
            ),
            TransitiveReductionDataset(
                "keep_short_direct_edge",
                pretty_name="Keep Short Direct Edge",
                edges=[(0, 1, 10.0), (1, 2, 10.0), (0, 2, 15.0)],
                expected_edges=[(0, 1, 10.0), (1, 2, 10.0), (0, 2, 15.0)],
                suites=["test"],
            ),
            TransitiveReductionDataset(
                "keep_when_indirect_is_long",
                pretty_name="Keep When Indirect Is Long",
                edges=[(0, 1, 40.0), (1, 2, 40.0), (0, 2, 30.0)],
                expected_edges=[(0, 1, 40.0), (1, 2, 40.0), (0, 2, 30.0)],
                suites=["test"],
            ),
            TransitiveReductionDataset(
                "remove_equal_direct_edge",
                pretty_name="Remove Equal Direct Edge",
                edges=[(0, 1, 10.0), (1, 2, 10.0), (0, 2, 20.0)],
                expected_edges=[(0, 1, 10.0), (1, 2, 10.0)],
                suites=["test"],
            ),
        ]

    def generate(self, dataset: TransitiveReductionDataset):
        R = np.full((3, 3), np.inf)
        for i, j, value in dataset.edges:
            R[i, j] = value
        expected = np.full((3, 3), np.inf)
        for i, j, value in dataset.expected_edges:
            expected[i, j] = value
        return DataInstance(
            inputs=[from_numpy(R)],
            meta={"x": 1, "max_iters": 5},
            ref_outputs=[from_numpy(expected)],
        )


def _unweighted_distances(adjacency: BinsparseTensor) -> COORMatrix:
    """Distance 1 along each edge of the 0-1 adjacency, infinite elsewhere.

    Self-loops are dropped, since a vertex is never its own transitive edge.
    """
    edges = to_scipy(zero_one_adjacency(adjacency)).tocoo()
    keep = edges.row != edges.col
    values = np.ones(np.count_nonzero(keep), dtype=float)
    return COORMatrix(
        edges.shape,
        values.size,
        fill=True,
        fill_value=np.inf,
        indices_0=edges.row[keep],
        indices_1=edges.col[keep],
        values=values,
    )


class TransitiveReductionSNAPGenerator(Generator[TransitiveReductionDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "transitive_reduction_snap"

    @property
    def pretty_name(self) -> str:
        return "Transitive Reduction SNAP"

    @property
    def description(self) -> str:
        return "SNAP input generator for transitive reduction benchmarks."

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
        return "Generate sparse directed graph inputs for transitive reduction."

    @property
    def datasets(self) -> list[TransitiveReductionDataset]:
        # fmt: off
        return [
            TransitiveReductionDataset("soc-Epinions1", suites=["standard"]),
            TransitiveReductionDataset("soc-LiveJournal1", suites=["standard"]),
            TransitiveReductionDataset("soc-Pokec", suites=["standard"]),
            TransitiveReductionDataset("soc-Slashdot0811", suites=["standard"]),
            TransitiveReductionDataset("soc-Slashdot0902", suites=["standard"]),
            TransitiveReductionDataset("wiki-Vote", suites=["standard"]),
            TransitiveReductionDataset("wiki-RfA", suites=["standard"]),
            TransitiveReductionDataset("soc-sign-bitcoin-otc", suites=["standard"]),
            TransitiveReductionDataset("soc-sign-bitcoin-alpha", suites=["standard"]),
            TransitiveReductionDataset("com-LiveJournal", suites=["standard"]),
            TransitiveReductionDataset("com-Friendster", suites=["standard"]),
            TransitiveReductionDataset("com-Orkut", suites=["standard"]),
            TransitiveReductionDataset("com-Youtube", suites=["standard"]),
            TransitiveReductionDataset("com-DBLP", suites=["standard"]),
            TransitiveReductionDataset("com-Amazon", suites=["standard"]),
            TransitiveReductionDataset("email-Eu-core", suites=["standard", "trace"]),
            TransitiveReductionDataset("wiki-topcats", suites=["standard"]),
            TransitiveReductionDataset("email-EuAll", suites=["standard"]),
            TransitiveReductionDataset("email-Enron", suites=["standard"]),
            TransitiveReductionDataset("wiki-Talk", suites=["standard"]),
            TransitiveReductionDataset("cit-HepPh", suites=["standard"]),
            TransitiveReductionDataset("cit-HepTh", suites=["standard"]),
            TransitiveReductionDataset("cit-Patents", suites=["standard"]),
            TransitiveReductionDataset("ca-AstroPh", suites=["standard"]),
            TransitiveReductionDataset("ca-CondMat", suites=["standard"]),
            TransitiveReductionDataset("ca-GrQc", suites=["standard"]),
            TransitiveReductionDataset("ca-HepPh", suites=["standard"]),
            TransitiveReductionDataset("ca-HepTh", suites=["standard"]),
            TransitiveReductionDataset("web-BerkStan", suites=["standard"]),
            TransitiveReductionDataset("web-Google", suites=["standard"]),
            TransitiveReductionDataset("web-NotreDame", suites=["standard"]),
            TransitiveReductionDataset("web-Stanford", suites=["standard"]),
            TransitiveReductionDataset("amazon0302", suites=["standard"]),
            TransitiveReductionDataset("amazon0312", suites=["standard"]),
            TransitiveReductionDataset("amazon0505", suites=["standard"]),
            TransitiveReductionDataset("amazon0601", suites=["standard"]),
            TransitiveReductionDataset("p2p-Gnutella04", suites=["standard"]),
            TransitiveReductionDataset("p2p-Gnutella05", suites=["standard"]),
            TransitiveReductionDataset("p2p-Gnutella06", suites=["standard"]),
            TransitiveReductionDataset("p2p-Gnutella08", suites=["standard"]),
            TransitiveReductionDataset("p2p-Gnutella09", suites=["standard"]),
            TransitiveReductionDataset("p2p-Gnutella24", suites=["standard"]),
            TransitiveReductionDataset("p2p-Gnutella25", suites=["standard"]),
            TransitiveReductionDataset("p2p-Gnutella30", suites=["standard"]),
            TransitiveReductionDataset("p2p-Gnutella31", suites=["standard"]),
            TransitiveReductionDataset("roadNet-CA", suites=["standard"]),
            TransitiveReductionDataset("roadNet-PA", suites=["standard"]),
            TransitiveReductionDataset("roadNet-TX", suites=["standard"]),
            TransitiveReductionDataset("as-735", suites=["standard"]),
            TransitiveReductionDataset("as-Skitter", suites=["standard"]),
            TransitiveReductionDataset("as-caida", suites=["standard"]),
            TransitiveReductionDataset("Oregon-1", suites=["standard"]),
            TransitiveReductionDataset("Oregon-2", suites=["standard"]),
            TransitiveReductionDataset("soc-sign-epinions", suites=["standard"]),
            TransitiveReductionDataset("soc-sign-Slashdot081106", suites=["standard"]),
            TransitiveReductionDataset("soc-sign-Slashdot090216", suites=["standard"]),
            TransitiveReductionDataset("soc-sign-Slashdot090221", suites=["standard"]),
            TransitiveReductionDataset("loc-Gowalla", suites=["standard"]),
            TransitiveReductionDataset("loc-Brightkite", suites=["standard"]),
            TransitiveReductionDataset("sx-stackoverflow", suites=["standard"]),
            TransitiveReductionDataset("sx-mathoverflow", suites=["standard"]),
            TransitiveReductionDataset("sx-superuser", suites=["standard"]),
            TransitiveReductionDataset("sx-askubuntu", suites=["standard"]),
            TransitiveReductionDataset("wiki-talk-temporal", suites=["standard"]),
            TransitiveReductionDataset("email-Eu-core-temporal", suites=["standard", "trace"]),
            TransitiveReductionDataset("CollegeMsg", suites=["standard", "trace", "train"]),
            TransitiveReductionDataset("twitter7", suites=["standard"]),
            TransitiveReductionDataset("higgs-twitter", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: TransitiveReductionDataset) -> DataInstance:
        raw = fetch_snap_graph(dataset.name)
        return DataInstance(
            inputs=[_unweighted_distances(raw.inputs[0])], meta=dict(raw.meta)
        )


class TransitiveReductionGAPGenerator(Generator[TransitiveReductionDataset]):
    @property
    def name(self) -> str:
        return "transitive_reduction_gap"

    @property
    def pretty_name(self) -> str:
        return "Transitive Reduction GAP"

    @property
    def description(self) -> str:
        return "Input GAP generator for transitive reduction benchmarks."

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
        return "Generate GAP directed graph inputs for transitive reduction."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[TransitiveReductionDataset]:
        return [
            TransitiveReductionDataset(
                name="GAP-road",
                description=(
                    "Directed roads with weights in the US, with 23.9M nodes and"
                    " 58.3M edges."
                ),
                suites=["standard"],
            ),
            TransitiveReductionDataset(
                name="GAP-twitter",
                description=(
                    "Directed weighted social network topology of Twitter, with 61.6M"
                    " nodes and 1,468.4M edges."
                ),
                suites=["standard"],
            ),
            TransitiveReductionDataset(
                name="GAP-web",
                description=(
                    "A web-crawl of the .sk domain, directed and weighted, with 50.6M"
                    " nodes and 1,949.4M edges."
                ),
                suites=["standard"],
            ),
            TransitiveReductionDataset(
                name="GAP-kron",
                description=(
                    "Symmetric random undirected weighted graph generated by"
                    " Kronecker synthetic graph generator with parameters"
                    " (A=0.57, B=C=0.19, D=0.05). Has 134.2M nodes and 2,111.6M"
                    " edges."
                ),
                suites=["standard"],
            ),
            TransitiveReductionDataset(
                name="GAP-urand",
                description=(
                    "Symmetric random undirected weighted graph generated by"
                    " Erdos–Reyni model (Uniform Random) with 134.2M nodes and"
                    " 2,147.4M edges."
                ),
                suites=["standard"],
            ),
        ]

    def generate(self, dataset: TransitiveReductionDataset) -> DataInstance:
        raw = fetch_gap_graph(dataset.name)
        return DataInstance(
            inputs=[_unweighted_distances(raw.inputs[0])], meta=dict(raw.meta)
        )


class TransitiveReductionBenchmark(Benchmark):
    @property
    def name(self) -> str:
        return "transitive_reduction"

    @property
    def pretty_name(self) -> str:
        return "Transitive Reduction"

    @property
    def description(self) -> str:
        return (
            "Iterative transitive reduction on a sparse overlap graph, following "
            "the diBELLA reduction step."
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
        return [Contributor("Jaehun Baek", "jbaek90@gatech.edu")]

    @property
    def references(self) -> list[Ref]:
        return [
            Ref(
                title=(
                    "Parallel String Graph Construction and Transitive Reduction "
                    "for De Novo Genome Assembly"
                ),
                authors=[
                    Author("Giulia Guidi"),
                    Author("Oguz Selvitopi"),
                    Author("Marquita Ellis"),
                    Author("Leonid Oliker"),
                    Author("Katherine Yelick"),
                    Author("Aydin Buluc"),
                ],
                conference=(
                    "IEEE International Parallel and Distributed Processing "
                    "Symposium (IPDPS)"
                ),
                year=2021,
                pages="517-526",
                doi="10.1109/IPDPS49936.2021.00060",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to construct the benchmark function. This "
            "statement was written by hand."
        )

    @property
    def motivation(self) -> str:
        return (
            "This benchmark implements the iterative transitive reduction step from "
            "the diBELLA 2D paper. The overlap graph R is a sparse matrix where "
            "R[i, j] represents the suffix length of an overlap between read i and "
            "read j. The reduction is implemented as sparse (min, +) semiring "
            "SpGEMM to find shortest 2-hop paths."
        )

    @property
    def generators(self):
        return [
            TransitiveReductionTestGenerator(),
            TransitiveReductionSNAPGenerator(),
            TransitiveReductionGAPGenerator(),
        ]

    def benchmark(self, xp, meta, R):
        x = meta.get("x", 1)
        max_iters = meta.get("max_iters", 10)

        R_nnz_prev_tensor = xp.sum(np.inf != R)
        R_nnz_prev = R_nnz_prev_tensor[()]

        for _i in range(max_iters):
            N = xp.einsum("N[i, j] min= R[i, k] + R[k, j]", R=R)

            R_for_max = xp.where(np.inf == R, -1.0, R)
            v = xp.max(R_for_max, axis=1)
            v = v + x

            v_expanded = xp.expand_dims(v, axis=1)
            M = v_expanded

            is_transitive = M >= N
            common_sparsity = xp.logical_and(np.inf != R, np.inf != N)
            edges_to_remove = xp.logical_and(common_sparsity, is_transitive)

            R = xp.where(edges_to_remove, np.inf, R)
            R_nnz_new_tensor = xp.sum(np.inf != R)
            R_nnz_new = R_nnz_new_tensor[()]

            if R_nnz_new == R_nnz_prev:
                break

            R_nnz_prev = R_nnz_new

        return R

    def check(self, param):
        super().check(param)
        if self._ref_outputs is None:
            return
        expected = to_numpy(self._ref_outputs[0])
        actual = to_numpy(self._output[0])
        assert np.array_equal(actual, expected)
