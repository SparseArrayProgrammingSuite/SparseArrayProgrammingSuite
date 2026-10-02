# ruff: noqa: E501
from typing import Any

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy

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
from saps.benchmarks.gap import fetch_gap_graph
from saps.benchmarks.snap import fetch_snap_graph
from saps_framework.binsparse_utils import binsparse_equal


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
        self._suites = list(suites or [])

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
        return "Provide small graph examples for FastSV correctness checks."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[FastSVDataset]:
        return [
            FastSVDataset("no-edges", suites=["test"]),
            FastSVDataset("single-component", suites=["test"]),
            FastSVDataset("two-components", suites=["test"]),
            FastSVDataset("chain", suites=["test"]),
            FastSVDataset("star", suites=["test"]),
            FastSVDataset("isolated-and-connected", suites=["test"]),
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
            inputs=[from_numpy(A)],
            meta={},
            ref_outputs=[from_numpy(expected)],
        )


class FastSVSNAPGenerator(Generator[FastSVDataset]):
    @property
    def name(self) -> str:
        return "fastsv_snap_inputs"

    @property
    def pretty_name(self) -> str:
        return "FastSV SNAP Input Generator"

    @property
    def description(self) -> str:
        return "SNAP input generator for FastSV connected-components benchmarks."

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
        return "Generate sparse graph inputs for FastSV."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[FastSVDataset]:
        # Trace selects successful Smart runs < 60s in competition/run_13803684.
        # fmt: off
        return [
            FastSVDataset("soc-Epinions1", suites=["standard", "trace"]),
            FastSVDataset("soc-LiveJournal1", suites=["standard"]),
            FastSVDataset("soc-Pokec", suites=["standard"]),
            FastSVDataset("soc-Slashdot0811", suites=["standard", "trace"]),
            FastSVDataset("soc-Slashdot0902", suites=["standard", "trace"]),
            FastSVDataset("wiki-Vote", suites=["standard", "trace"]),
            FastSVDataset("wiki-RfA", suites=["standard", "trace"]),
            FastSVDataset("soc-sign-bitcoin-otc", suites=["standard", "trace"]),
            FastSVDataset("soc-sign-bitcoin-alpha", suites=["standard", "trace"]),
            FastSVDataset("com-LiveJournal", suites=["standard"]),
            FastSVDataset("com-Friendster", suites=["standard"]),
            FastSVDataset("com-Orkut", suites=["standard"]),
            FastSVDataset("com-Youtube", suites=["standard", "trace"]),
            FastSVDataset("com-DBLP", suites=["standard", "trace"]),
            FastSVDataset("com-Amazon", suites=["standard", "trace"]),
            FastSVDataset("email-Eu-core", suites=["standard", "trace"]),
            FastSVDataset("wiki-topcats", suites=["standard"]),
            FastSVDataset("email-EuAll", suites=["standard", "trace"]),
            FastSVDataset("email-Enron", suites=["standard", "trace"]),
            FastSVDataset("wiki-Talk", suites=["standard", "trace"]),
            FastSVDataset("cit-HepPh", suites=["standard", "trace"]),
            FastSVDataset("cit-HepTh", suites=["standard", "trace"]),
            FastSVDataset("cit-Patents", suites=["standard"]),
            FastSVDataset("ca-AstroPh", suites=["standard", "trace"]),
            FastSVDataset("ca-CondMat", suites=["standard", "trace"]),
            FastSVDataset("ca-GrQc", suites=["standard", "trace"]),
            FastSVDataset("ca-HepPh", suites=["standard", "trace"]),
            FastSVDataset("ca-HepTh", suites=["standard", "trace"]),
            FastSVDataset("web-BerkStan", suites=["standard"]),
            FastSVDataset("web-Google", suites=["standard", "trace"]),
            FastSVDataset("web-NotreDame", suites=["standard", "trace"]),
            FastSVDataset("web-Stanford", suites=["standard", "trace"]),
            FastSVDataset("amazon0302", suites=["standard", "trace"]),
            FastSVDataset("amazon0312", suites=["standard", "trace"]),
            FastSVDataset("amazon0505", suites=["standard", "trace"]),
            FastSVDataset("amazon0601", suites=["standard", "trace"]),
            FastSVDataset("p2p-Gnutella04", suites=["standard", "trace"]),
            FastSVDataset("p2p-Gnutella05", suites=["standard", "trace"]),
            FastSVDataset("p2p-Gnutella06", suites=["standard", "trace"]),
            FastSVDataset("p2p-Gnutella08", suites=["standard", "trace"]),
            FastSVDataset("p2p-Gnutella09", suites=["standard", "trace"]),
            FastSVDataset("p2p-Gnutella24", suites=["standard", "trace"]),
            FastSVDataset("p2p-Gnutella25", suites=["standard", "trace"]),
            FastSVDataset("p2p-Gnutella30", suites=["standard", "trace"]),
            FastSVDataset("p2p-Gnutella31", suites=["standard", "trace"]),
            FastSVDataset("roadNet-CA", suites=["standard", "trace"]),
            FastSVDataset("roadNet-PA", suites=["standard", "trace"]),
            FastSVDataset("roadNet-TX", suites=["standard", "trace"]),
            FastSVDataset("as-735", suites=["standard", "trace"]),
            FastSVDataset("as-Skitter", suites=["standard"]),
            FastSVDataset("as-caida", suites=["standard", "trace"]),
            FastSVDataset("Oregon-1", suites=["standard", "trace"]),
            FastSVDataset("Oregon-2", suites=["standard", "trace"]),
            FastSVDataset("soc-sign-epinions", suites=["standard", "trace"]),
            FastSVDataset("soc-sign-Slashdot081106", suites=["standard", "trace"]),
            FastSVDataset("soc-sign-Slashdot090216", suites=["standard", "trace"]),
            FastSVDataset("soc-sign-Slashdot090221", suites=["standard", "trace"]),
            FastSVDataset("loc-Gowalla", suites=["standard", "trace"]),
            FastSVDataset("loc-Brightkite", suites=["standard", "trace"]),
            FastSVDataset("sx-stackoverflow", suites=["standard"]),
            FastSVDataset("sx-mathoverflow", suites=["standard", "trace"]),
            FastSVDataset("sx-superuser", suites=["standard", "trace"]),
            FastSVDataset("sx-askubuntu", suites=["standard", "trace"]),
            FastSVDataset("wiki-talk-temporal", suites=["standard", "trace"]),
            FastSVDataset("email-Eu-core-temporal", suites=["standard", "trace"]),
            FastSVDataset("CollegeMsg", suites=["standard", "trace"]),
            FastSVDataset("twitter7", suites=["standard"]),
            FastSVDataset("higgs-twitter", suites=["standard", "trace"]),
        ]
        # fmt: on

    def generate(self, dataset: FastSVDataset) -> DataInstance:
        raw = fetch_snap_graph(dataset.name)
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0])], meta=dict(raw.meta)
        )


class FastSVGAPGenerator(Generator[FastSVDataset]):
    @property
    def name(self) -> str:
        return "fastsv_gap_inputs"

    @property
    def pretty_name(self) -> str:
        return "FastSV GAP Input Generator"

    @property
    def description(self) -> str:
        return "Input GAP generator for FastSV connected-components benchmarks."

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
        return "Generate GAP graph inputs for FastSV."

    @property
    def cacheable(self) -> bool:
        return False

    @property
    def datasets(self) -> list[FastSVDataset]:
        # fmt: off
        return [
            FastSVDataset("GAP-road", suites=["standard"]),
            FastSVDataset("GAP-twitter", suites=["standard"]),
            FastSVDataset("GAP-web", suites=["standard"]),
            FastSVDataset("GAP-kron", suites=["standard"]),
            FastSVDataset("GAP-urand", suites=["standard"]),
        ]
        # fmt: on

    def generate(self, dataset: FastSVDataset) -> DataInstance:
        raw = fetch_gap_graph(dataset.name)
        return DataInstance(
            inputs=[zero_one_adjacency(raw.inputs[0])], meta=dict(raw.meta)
        )


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
    def generators(self) -> list[Generator]:
        return [FastSVTestGenerator(), FastSVSNAPGenerator(), FastSVGAPGenerator()]

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
            assert isinstance(item, BinsparseTensor), (
                "Output must be in binsparse format"
            )
        if self._ref_outputs is None:
            return
        assert binsparse_equal(self._output[0], self._ref_outputs[0]), (
            f"FastSV output mismatch for {param.dataset.name}"
        )
