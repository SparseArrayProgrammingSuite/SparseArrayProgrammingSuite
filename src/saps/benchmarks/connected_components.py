import numpy as np

from binsparse import BinsparseTensor

from saps.benchmark import (
    Author,
    Benchmark,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
)
from saps.benchmarks.suitesparse import fetch_suitesparse_matrix
from saps.downloaders.snap import download_snap_dataset


class ConnectedComponentsDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        A: np.ndarray | None = None,
        ref_meta: dict | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"Connected components input {name}."
        self._suites = suites or []
        self.A = A
        self.ref_meta = ref_meta

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


class ConnectedComponentsTestGenerator(Generator[ConnectedComponentsDataset]):
    @property
    def name(self) -> str:
        return "connected_components_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "Connected Components Test Input Generator"

    @property
    def description(self) -> str:
        return "Small deterministic connected-components examples."

    @property
    def suites(self) -> list[str]:
        return ["test", "trace"]

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
        return "Provide small graph examples for correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[ConnectedComponentsDataset]:
        return [
            ConnectedComponentsDataset(
                "test_cc_fully_connected",
                suites=["test", "trace"],
                A=np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool),
                ref_meta={"component_count": 1},
            ),
            ConnectedComponentsDataset(
                "test_cc_two_disconnected_components",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [0, 1, 0, 0],
                        [1, 0, 0, 0],
                        [0, 0, 0, 1],
                        [0, 0, 1, 0],
                    ],
                    dtype=bool,
                ),
                ref_meta={"same": [[0, 1], [2, 3]], "different": [[0, 2]]},
            ),
            ConnectedComponentsDataset(
                "test_cc_isolated_nodes",
                suites=["test", "trace"],
                A=np.zeros((4, 4), dtype=bool),
                ref_meta={"component_count": 4},
            ),
            ConnectedComponentsDataset(
                "test_cc_directed_star_pointing_inward",
                suites=["test", "trace"],
                A=np.array(
                    [
                        [0, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                        [1, 0, 0, 0],
                    ],
                    dtype=bool,
                ),
                ref_meta={"component_count": 1},
            ),
            ConnectedComponentsDataset(
                "test_cc_single_node",
                suites=["test", "trace"],
                A=np.zeros((1, 1), dtype=bool),
                ref_meta={"shape": [1]},
            ),
            ConnectedComponentsDataset(
                "test_cc_snap_toy",
                suites=["test", "trace"],
                A=np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool),
                ref_meta={"component_count": 1},
            ),
        ]

    def generate(self, dataset: ConnectedComponentsDataset) -> DataInstance:
        if dataset.A is None:
            raise ValueError("Connected-components test datasets must define A.")

        return DataInstance(
            inputs=[BinsparseTensor.from_numpy(dataset.A)],
            meta={},
            ref_meta=dataset.ref_meta,
        )


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


class ConnectedComponentsGAPGenerator(Generator[ConnectedComponentsDataset]):
    @property
    def name(self) -> str:
        return "connected_components_gap_inputs"

    @property
    def pretty_name(self) -> str:
        return "Connected Components GAP Input Generator"

    @property
    def description(self) -> str:
        return "Input GAP generator for connected components benchmarks."

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
        return "Generate GAP graph inputs for connected components."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[ConnectedComponentsDataset]:
        return [
            ConnectedComponentsDataset(
                name="gap-road",
                pretty_name="GAP Road",
                description=(
                    "Directed roads with weights in the US, with 23.9M nodes and"
                    " 58.3M edges."
                ),
                suites=[],
            ),
            ConnectedComponentsDataset(
                name="gap-twitter",
                pretty_name="GAP Twitter",
                description=(
                    "Directed weighted social network topology of Twitter, with 61.6M"
                    " nodes and 1,468.4M edges."
                ),
                suites=[],
            ),
            ConnectedComponentsDataset(
                name="gap-web",
                pretty_name="GAP Web",
                description=(
                    "A web-crawl of the .sk domain, directed and weighted, with 50.6M"
                    " nodes and 1,949.4M edges."
                ),
                suites=[],
            ),
            ConnectedComponentsDataset(
                name="gap-kron",
                pretty_name="GAP Kron",
                description=(
                    "Symmetric random undirected weighted graph generated by"
                    " Kronecker synthetic graph generator with parameters"
                    " (A=0.57, B=C=0.19, D=0.05). Has 134.2M nodes and 2,111.6M"
                    " edges."
                ),
                suites=[],
            ),
            ConnectedComponentsDataset(
                name="gap-urand",
                pretty_name="GAP Urand",
                description=(
                    "Symmetric random undirected weighted graph generated by"
                    " Erdos–Reyni model (Uniform Random) with 134.2M nodes and"
                    " 2,147.4M edges."
                ),
                suites=[],
            ),
        ]

    def generate(self, dataset: ConnectedComponentsDataset) -> DataInstance:
        if dataset.name.startswith("gap"):
            raw = fetch_suitesparse_matrix(dataset.name)
            return DataInstance(inputs=raw.inputs, meta=raw.meta)
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
        return [
            ConnectedComponentsTestGenerator(),
            ConnectedComponentsGenerator(),
            ConnectedComponentsGAPGenerator(),
        ]

    def benchmark(self, xp, data, meta):
        edges = data[0]
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

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if not self._ref_meta:
            return
        labels = self._output[0].data["values"].reshape(self._output[0].data["shape"])
        if "component_count" in self._ref_meta:
            assert len(set(labels.tolist())) == self._ref_meta["component_count"]
        if "shape" in self._ref_meta:
            assert list(labels.shape) == self._ref_meta["shape"]
        for left, right in self._ref_meta.get("same", []):
            assert labels[left] == labels[right]
        for left, right in self._ref_meta.get("different", []):
            assert labels[left] != labels[right]
