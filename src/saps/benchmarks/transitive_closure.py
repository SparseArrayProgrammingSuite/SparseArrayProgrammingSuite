import numpy as np

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
from saps_framework import BinsparseFormat


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


class TransitiveClosureTestGenerator(Generator[TransitiveClosureDataset]):
    @property
    def name(self) -> str:
        return "transitive_closure_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "Transitive Closure Test Input Generator"

    @property
    def description(self) -> str:
        return "Small deterministic transitive closure examples."

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
        return "Provide small reachability examples for correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[TransitiveClosureDataset]:
        return [
            TransitiveClosureDataset("dag", suites=["test", "trace"]),
            TransitiveClosureDataset(
                "strong-component-count", suites=["test", "trace"]
            ),
            TransitiveClosureDataset("cycle", suites=["test", "trace"]),
            TransitiveClosureDataset("one-node", suites=["test", "trace"]),
            TransitiveClosureDataset("snap-toy", suites=["test", "trace"]),
        ]

    def generate(self, dataset: TransitiveClosureDataset) -> DataInstance:
        if dataset.name == "dag":
            A = np.array(
                [
                    [0, 1, 1, 0, 0, 0],
                    [0, 0, 1, 1, 0, 0],
                    [0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 1],
                    [0, 0, 0, 1, 0, 0],
                ],
                dtype=bool,
            )
            expected = np.array(
                [
                    [1, 1, 1, 1, 1, 1],
                    [0, 1, 1, 1, 1, 1],
                    [0, 0, 1, 1, 1, 1],
                    [0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 1, 1, 1],
                    [0, 0, 0, 1, 0, 1],
                ],
                dtype=bool,
            )
            return DataInstance(
                inputs=[BinsparseFormat.from_numpy(A)],
                meta={},
                ref_outputs=[BinsparseFormat.from_numpy(expected)],
            )

        if dataset.name == "strong-component-count":
            A = np.array(
                [
                    [0, 1, 0, 0, 0, 0, 0, 0],
                    [0, 0, 1, 0, 1, 1, 0, 0],
                    [0, 0, 0, 1, 0, 0, 1, 0],
                    [0, 0, 1, 0, 0, 0, 0, 1],
                    [1, 0, 0, 0, 0, 1, 0, 0],
                    [0, 0, 0, 0, 0, 0, 1, 0],
                    [0, 0, 0, 0, 0, 1, 0, 1],
                    [0, 0, 0, 0, 0, 0, 0, 1],
                ],
                dtype=bool,
            )
            return DataInstance(
                inputs=[BinsparseFormat.from_numpy(A)],
                meta={},
                ref_meta={"strong_component_count": 4},
            )

        if dataset.name == "cycle":
            A = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=bool)
            expected = np.ones((3, 3), dtype=bool)
        elif dataset.name == "one-node":
            A = np.array([[0]], dtype=bool)
            expected = np.array([[1]], dtype=bool)
        elif dataset.name == "snap-toy":
            A = np.array([[0, 1, 0], [0, 0, 1], [0, 0, 0]], dtype=bool)
            expected = np.array([[1, 1, 1], [0, 1, 1], [0, 0, 1]], dtype=bool)
        else:
            raise ValueError(f"Unsupported test dataset: {dataset.name}")

        return DataInstance(
            inputs=[BinsparseFormat.from_numpy(A)],
            meta={},
            ref_outputs=[BinsparseFormat.from_numpy(expected)],
        )


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


class TransitiveClosureGAPGenerator(Generator[TransitiveClosureDataset]):
    @property
    def name(self) -> str:
        return "transitive_closure_gap_inputs"

    @property
    def pretty_name(self) -> str:
        return "Transitive Closure GAP Input Generator"

    @property
    def description(self) -> str:
        return "Input GAP generator for transitive closure benchmarks."

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
        return "Generate GAP directed graph inputs for transitive closure."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[TransitiveClosureDataset]:
        return [
            TransitiveClosureDataset(
                name="gap-road",
                pretty_name="GAP Road",
                description=(
                    "Directed roads with weights in the US, with 23.9M nodes and"
                    " 58.3M edges."
                ),
                suites=[],
           ),
            TransitiveClosureDataset(
                name="gap-twitter",
                pretty_name="GAP Twitter",
                description=(
                    "Directed weighted social network topology of Twitter, with 61.6M"
                    " nodes and 1,468.4M edges."
                ),
                suites=[],
            ),
            TransitiveClosureDataset(
                name="gap-web",
                pretty_name="GAP Web",
                description=(
                    "A web-crawl of the .sk domain, directed and weighted, with 50.6M"
                    " nodes and 1,949.4M edges."
                ),
                suites=[],
            ),
        ]

    def generate(self, dataset: TransitiveClosureDataset) -> DataInstance:
        if dataset.name.startswith("gap"):
            raw = fetch_suitesparse_matrix(dataset.name)
            return DataInstance(inputs=[raw.inputs[0]], meta=raw.meta)
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
        return [
            TransitiveClosureGenerator(),
            TransitiveClosureTestGenerator(),
            TransitiveClosureGAPGenerator(),
        ]

    def benchmark(self, xp, data, meta):
        edges = data[0]
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

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseFormat), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is not None:
            assert self._output[0] == self._ref_outputs[0], (
                f"Transitive closure mismatch for {param.dataset.name}"
            )
        if self._ref_meta and "strong_component_count" in self._ref_meta:
            output = self._output[0]
            matrix = output.data["values"].reshape(output.data["shape"])
            visited_set = set()
            count = 0
            for i in range(matrix.shape[0]):
                comp = tuple(matrix[i, :])
                if comp not in visited_set:
                    count += 1
                    visited_set.add(comp)
            assert count == self._ref_meta["strong_component_count"]
