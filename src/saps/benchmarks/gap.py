"""Shared SuiteSparse GAP matrices for graph benchmarks."""

from copy import copy
from typing import Any

from saps.benchmark import (
    Author,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
    ShellBenchmark,
)
from saps.benchmarks.suitesparse import fetch_suitesparse_matrix

# Most off-diagonal nonzeros stored in any row (largest out-degree in the
# stored edge direction, excluding self-loops and explicit zeros), as printed
# by scripts/measure_fill_in.py for Slurm run 13794825.
_MAX_DEGREES: dict[str, int] = {
    "GAP/GAP-kron": 1572838,
    "GAP/GAP-road": 9,
    "GAP/GAP-twitter": 2997469,
    "GAP/GAP-urand": 68,
    "GAP/GAP-web": 12869,
}


class GAPDataset(Dataset):
    def __init__(
        self,
        name: str,
        sources: list[int],
        description: str,
        *,
        pretty_name: str | None = None,
        suites: list[str] | None = None,
    ):
        self._name = name
        self._sources = list(sources)
        self._description = description
        self._pretty_name = pretty_name or name
        self._suites = list(suites or [])

    @property
    def name(self) -> str:
        return self._name

    @property
    def source_name(self) -> str:
        return f"GAP/{self.name}"

    @property
    def max_degree(self) -> int:
        """Most off-diagonal nonzeros in any row of the stored matrix."""
        return _MAX_DEGREES[self.source_name]

    @property
    def sources(self) -> list[int]:
        """Source vertices published with the GAP Benchmark Suite."""
        return list(self._sources)

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

    def with_suites(self, suites: list[str]) -> "GAPDataset":
        """A copy of this shared graph listed under ``suites``."""
        dataset = copy(self)
        dataset._suites = list(suites)
        return dataset

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            **super().metadata,
            "source_name": self.source_name,
            "max_degree": self.max_degree,
            "sources": self.sources,
        }


_GRAPHS = [
    GAPDataset(
        "GAP-road",
        [
            4795720,
            21003853,
            417968,
            6496511,
            6648699,
            9811073,
            22247478,
            5720252,
            12366459,
            20413729,
            4217374,
            2674749,
            22085557,
            19445040,
            2360788,
            19115968,
            7758767,
            13468234,
            30367,
            18599547,
            7526108,
            16836280,
            12742067,
            7697995,
            5876443,
            9616340,
            2497673,
            10052290,
            12493057,
            1670855,
            2760679,
            2460941,
            8489650,
            5005225,
            8744645,
            8512023,
            21912165,
            1105390,
            15432163,
            1600177,
            19079469,
            16516637,
            20202566,
            21372803,
            2898009,
            8491277,
            18798317,
            23757560,
            17161819,
            23180739,
            10997085,
            3730630,
            1079068,
            15426822,
            12190925,
            1155218,
            10693488,
            14434835,
            19963339,
            3486185,
            18383269,
            20269908,
            12370764,
            7843140,
        ],
        "Directed roads with weights in the US, with 23.9M nodes and 58.3M edges.",
    ),
    GAPDataset(
        "GAP-twitter",
        [
            12441072,
            54488257,
            25451915,
            57714473,
            14839494,
            32081104,
            52957357,
            50444380,
            49590701,
            20127816,
            34939333,
            48251001,
            19524253,
            43676726,
            33055508,
            15244687,
            24946738,
            6479472,
            26077682,
            22023875,
            22081915,
            40034162,
            49496014,
            42847507,
            52409557,
            55445388,
            22028097,
            48766648,
            44521241,
            60135542,
            28528671,
            9678012,
            40020306,
            31625735,
            37446892,
            51788952,
            52584255,
            20346696,
            48387909,
            37337427,
            50501084,
            30130061,
            41185893,
            56495703,
            45663305,
            33359460,
            48143058,
            33291513,
            53461445,
            29340610,
            34148498,
            49171806,
            35550696,
            14521507,
            51633218,
            46823382,
            19396273,
            19871750,
            36862677,
            49539126,
            34016452,
            36567395,
            55487793,
            14391370,
        ],
        "Directed weighted social network topology of Twitter, with 61.6M nodes"
        " and 1,468.4M edges.",
    ),
    GAPDataset(
        "GAP-web",
        [
            10219452,
            44758211,
            890671,
            13843756,
            14168062,
            20906930,
            12189584,
            26352335,
            43500686,
            8987024,
            5699762,
            41436455,
            5030727,
            40735218,
            16533563,
            28700166,
            64711,
            39634750,
            16037779,
            27152739,
            16404061,
            20491963,
            5322423,
            21420953,
            26622109,
            5882875,
            18091040,
            10665896,
            18634422,
            18138715,
            2355535,
            32885205,
            40657440,
            35196167,
            45544426,
            6175519,
            40058318,
            50626230,
            36571019,
            49397052,
            23434265,
            2299444,
            32873823,
            25978282,
            2461715,
            22787314,
            30759947,
            7428894,
            39173870,
            43194209,
            26361509,
            39747211,
            30670029,
            41483033,
            9358666,
            9945008,
            3355244,
            33831269,
            45124744,
            16137877,
            11235448,
            37509144,
            27402414,
            39546083,
        ],
        "A web-crawl of the .sk domain, directed and weighted, with 50.6M nodes"
        " and 1,949.4M edges.",
    ),
    GAPDataset(
        "GAP-kron",
        [
            2338012,
            31997659,
            23590940,
            43400604,
            75337937,
            169867,
            104041220,
            94177942,
            32871357,
            56230002,
            69883037,
            9346345,
            48915358,
            122571173,
            6183279,
            86323663,
            106725780,
            92389938,
            16210738,
            59816700,
            111669929,
            102831411,
            113384800,
            43872564,
            80508827,
            26105648,
            8807516,
            118452455,
            121818859,
            42361928,
            29493053,
            98461503,
            71931337,
            103808468,
            4092345,
            115276241,
            4649343,
            76656189,
            31312001,
            111334127,
            100962918,
            41823215,
            22631240,
            42848461,
            79485148,
            106818742,
            73347974,
            78848445,
            109920510,
            121492133,
            101037296,
            15438600,
            4584784,
            124503845,
            87241743,
            108297008,
            33955082,
            79934823,
            8608481,
            82435063,
            46579271,
            515421,
            121530467,
            127978736,
        ],
        "Symmetric random undirected weighted graph generated by Kronecker"
        " synthetic graph generator with parameters (A=0.57, B=C=0.19, D=0.05)."
        " Has 134.2M nodes and 2,111.6M edges.",
    ),
    GAPDataset(
        "GAP-urand",
        [
            27691419,
            121280314,
            2413431,
            37512113,
            38390877,
            56651037,
            128461248,
            33029842,
            71406328,
            117872827,
            24351938,
            15444519,
            127526281,
            112279428,
            13631649,
            110379302,
            44800623,
            77768193,
            175347,
            107397389,
            43457209,
            97215940,
            73575165,
            44449715,
            33931724,
            55526610,
            14422051,
            58043873,
            72137329,
            9647840,
            15940695,
            14209952,
            49020883,
            28901138,
            50493273,
            49150069,
            126525082,
            6382740,
            89108297,
            9239735,
            110168548,
            95370259,
            116653530,
            123410703,
            16733665,
            49030282,
            108545121,
            99095665,
            133850077,
            63499301,
            21541382,
            6230751,
            89077456,
            70392765,
            6670455,
            61746271,
            83349535,
            115272184,
            20129908,
            106148553,
            117042375,
            71431187,
            45287808,
            107702120,
        ],
        "Symmetric random undirected weighted graph generated by Erdos–Reyni"
        " model (Uniform Random) with 134.2M nodes and 2,147.4M edges.",
    ),
]


class GAPGraphGenerator(Generator[GAPDataset]):
    @property
    def cacheable(self) -> bool:
        return False

    @property
    def name(self) -> str:
        return "gap_graph"

    @property
    def pretty_name(self) -> str:
        return "GAP Graphs"

    @property
    def description(self) -> str:
        return (
            "Shared GAP adjacency matrices from the SuiteSparse Matrix Collection,"
            " with their maximum degrees and published source vertices."
        )

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
        return "This shell generator was written with assistance from Claude."

    @property
    def motivation(self) -> str:
        return "Reuse prepared SuiteSparse GAP matrices across graph benchmarks."

    @property
    def datasets(self) -> list[GAPDataset]:
        return list(_GRAPHS)

    def generate(self, dataset: GAPDataset) -> DataInstance:
        raw = fetch_suitesparse_matrix(dataset.source_name)
        return DataInstance(
            inputs=[raw.inputs[0]],
            meta={"max_degree": dataset.max_degree, "sources": dataset.sources},
        )


class GAPGraphShellBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return GAPGraphGenerator()


def gap_graph(name: str) -> GAPDataset:
    """The declared GAP shell dataset called ``name``."""
    dataset = next((d for d in _GRAPHS if d.name == name), None)
    if dataset is None:
        raise ValueError(
            f"Dataset {name!r} is not listed in GAPGraphGenerator.datasets. "
            "Add it to the shell dataset list before using it."
        )
    return dataset


def fetch_gap_graph(name: str) -> DataInstance:
    """Read a declared SuiteSparse GAP matrix from prepared storage."""
    return GAPGraphGenerator().cached_generate(gap_graph(name))
