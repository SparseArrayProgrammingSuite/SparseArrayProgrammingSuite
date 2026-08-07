from typing import Any

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
from saps.downloaders.snap import download_snap_dataset
from saps_framework import BinsparseFormat


class FastSVDataset(Dataset):
    def __init__(
        self,
        name: str,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
    ):
        self._name = name
        self._pretty_name = pretty_name or name
        self._description = description or f"FastSV input {name}."
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


class FastSVTestGenerator(Generator[FastSVDataset]):
    @property
    def name(self) -> str:
        return "fastsv_test_inputs"

    @property
    def pretty_name(self) -> str:
        return "FastSV Test Input Generator"

    @property
    def description(self) -> str:
        return "Small deterministic FastSV examples with reference labels."

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
        return "Provide small graph examples for FastSV correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[FastSVDataset]:
        return [
            FastSVDataset("no-edges", suites=["test", "trace"]),
            FastSVDataset("single-component", suites=["test", "trace"]),
            FastSVDataset("two-components", suites=["test", "trace"]),
            FastSVDataset("chain", suites=["test", "trace"]),
            FastSVDataset("star", suites=["test", "trace"]),
            FastSVDataset("isolated-and-connected", suites=["test", "trace"]),
        ]

    def generate(self, dataset: FastSVDataset) -> DataInstance:
        A: np.ndarray[Any, Any]
        expected: np.ndarray[Any, Any]
        if dataset.name == "no-edges":
            A = np.zeros((5, 5), dtype=bool)
            expected = np.arange(5)
        elif dataset.name == "single-component":
            A = np.array(
                [
                    [0, 1, 1, 1],
                    [1, 0, 1, 1],
                    [1, 1, 0, 1],
                    [1, 1, 1, 0],
                ],
                dtype=bool,
            )
            expected = np.array([0, 0, 0, 0])
        elif dataset.name == "two-components":
            A = np.array(
                [
                    [0, 1, 0, 0],
                    [1, 0, 0, 0],
                    [0, 0, 0, 1],
                    [0, 0, 1, 0],
                ],
                dtype=bool,
            )
            expected = np.array([0, 0, 2, 2])
        elif dataset.name == "chain":
            A = np.array(
                [
                    [0, 1, 0, 0, 0],
                    [1, 0, 1, 0, 0],
                    [0, 1, 0, 1, 0],
                    [0, 0, 1, 0, 1],
                    [0, 0, 0, 1, 0],
                ],
                dtype=bool,
            )
            expected = np.array([0, 0, 0, 0, 0])
        elif dataset.name == "star":
            A = np.array(
                [
                    [0, 1, 1, 1, 1],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                    [1, 0, 0, 0, 0],
                ],
                dtype=bool,
            )
            expected = np.array([0, 0, 0, 0, 0])
        elif dataset.name == "isolated-and-connected":
            A = np.array(
                [
                    [0, 1, 0, 0, 0],
                    [1, 0, 1, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=bool,
            )
            expected = np.array([0, 0, 0, 3, 4])
        else:
            raise ValueError(f"Unsupported test dataset: {dataset.name}")

        return DataInstance(
            inputs=[BinsparseFormat.from_numpy(A)],
            meta={},
            ref_outputs=[BinsparseFormat.from_numpy(expected)],
        )


class FastSVGenerator(Generator[FastSVDataset]):
    @property
    def name(self) -> str:
        return "fastsv_inputs"

    @property
    def pretty_name(self) -> str:
        return "FastSV Input Generator"

    @property
    def description(self) -> str:
        return "Input generator for FastSV connected-components benchmarks."

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
        return "Generate sparse graph inputs for FastSV."

    @property
    def datasets(self) -> list[FastSVDataset]:
        return [
            FastSVDataset(
                name="snap-email-Eu-core",
                pretty_name="SNAP email-Eu-core",
                description=(
                    "Directed email communication network from a European research"
                    " institution, with 1,005 nodes and 25,571 edges."
                ),
                suites=[],
            ),
            FastSVDataset(
                name="snap-facebook_combined",
                pretty_name="SNAP facebook_combined",
                description=(
                    "Combined Facebook social-circle network, with 4,039 nodes and"
                    " 88,234 edges."
                ),
                suites=[],
            ),
            FastSVDataset(
                name="snap-ca-GrQc",
                pretty_name="SNAP ca-GrQc",
                description=(
                    "Arxiv General Relativity and Quantum Cosmology collaboration"
                    " network, with 5,242 nodes and 14,496 edges."
                ),
                suites=[],
            ),
        ]

    def generate(self, dataset: FastSVDataset) -> DataInstance:
        if dataset.name.startswith("snap"):
            inputs, meta = download_snap_dataset(dataset.name)
            return DataInstance(inputs=inputs, meta=meta)
        raise ValueError(f"Unsupported FastSV dataset: {dataset.name}")


class FastSVBenchmark(Benchmark):
    @property
    def name(self):
        return "fastsv"

    @property
    def pretty_name(self):
        return "FastSV Algorithm"

    @property
    def description(self):
        return (
            "The FastSV algorithm is a graph algorithm used to find the connected"
            " components for a simple graph. This algorithm introduces several"
            " optimizations that allow for faster convergence to a solution compared to"
            " the SV algorithm it is based on, specifically through modifications to"
            " the tree hooking and termination condition."
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
            Contributor("Richard Wan", "rwan41@gatech.edu"),
        ]

    @property
    def references(self):
        return [
            Ref(
                title=(
                    "FastSV: A distributed-memory connected component"
                    " algorithm with fast convergence."
                ),
                authors=[
                    Author("Zhang, Y."),
                    Author("Azad, A."),
                    Author("Hu, Z."),
                ],
                journal=(
                    "Proceedings of the 2020 SIAM Conference on Parallel"
                    " Processing for Scientific Computing"
                ),
                pages="46-57",
                publisher="Society for Industrial and Applied Mathematics",
                year=2020,
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
    def generators(self) -> list[Generator[FastSVDataset]]:
        return [FastSVTestGenerator(), FastSVGenerator()]

    def benchmark(self, xp, data, meta):
        A = data[0]
        A = A != 0

        (n, m) = A.shape
        assert n == m

        f = xp.arange(n)
        gf = xp.asarray(f, copy=True)

        int_max = xp.iinfo(f.dtype).max

        while True:
            dup = gf

            # step 1: stochastic hooking
            mngf = xp.min(xp.where(A, xp.expand_dims(gf, 0), int_max), axis=1)
            B = xp.zeros((n, n), dtype=bool)
            B[f, xp.arange(n)] = True
            f = xp.min(xp.where(B, xp.expand_dims(mngf, 0), int_max), axis=1)

            # step 2: aggressive hooking
            f = xp.minimum(f, mngf)

            # step 3: shortcutting
            f = xp.minimum(f, gf)

            # step 4: calculate grandparents
            gf = xp.take(f, f)

            # step 5: check termination
            stop = xp.all(dup == gf)

            if stop:
                break

        return [f]

    def check(self, param):
        for item in self._output:
            assert isinstance(item, BinsparseFormat), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return
        assert self._output[0] == self._ref_outputs[0], (
            f"FastSV output mismatch for {param.dataset.name}"
        )
