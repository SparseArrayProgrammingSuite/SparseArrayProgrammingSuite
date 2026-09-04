from typing import Any

import numpy as np

from binsparse import BinsparseTensor
from binsparse.conversions import from_numpy, from_scipy, to_numpy, to_scipy

from saps.benchmark import (
    Author,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
    ShellBenchmark,
)
from saps.downloaders.suitesparse import load_suitesparse_matrix, random_rhs_for_matrix


class SuiteSparseDataset(Dataset):
    """Base Dataset for benchmarks backed by a SuiteSparse Matrix Collection matrix."""

    def __init__(
        self,
        name: str,
        *,
        source_name: str | None = None,
        pretty_name: str | None = None,
        description: str | None = None,
        suites: list[str] | None = None,
        nnz: int | None = None,
    ):
        self._name = name
        self.source_name = source_name if source_name is not None else name
        self._pretty_name = pretty_name
        self._description = description
        self._suites = suites or []
        self.nnz = nnz

    @property
    def name(self) -> str:
        return self._name

    @property
    def pretty_name(self) -> str:
        return self._pretty_name or self._name

    @property
    def description(self) -> str:
        return self._description or f"SuiteSparse matrix {self.source_name}."

    @property
    def suites(self) -> list[str]:
        return self._suites

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["nnz"] = self.nnz
        return data


_MATRICES: list[SuiteSparseDataset] = [
    SuiteSparseDataset(name)
    for name in [
        "mesh3em5",
        "bcsstm02",
        "fv1",
        "Muu",
        "Chem97ZtZ",
        "Dubcova1",
        "t3dl_e",
        "bcsstk09",
        "Trefethen_200",
        "Trefethen_500",
        "Trefethen_700",
        "fv2",
        "Trefethen_20000",
        "abb313",
        "ash958",
        "well1033",
        "Maragal_5",
        "illc1850",
        "bayer06",
        "mhdb416",
        "lund_b",
        "bcsstm12",
        "mesh1em1",
        "bcsstk05",
        "nos1",
        "nos2",
        "nos3",
        "dwt_59",
        "bcspwr01",
        "bcspwr02",
        "bcspwr03",
        "chesapeake",
        "ash85",
        "arc130",
        "bcspwr04",
        "ash292",
        "karate",
        "dolphins",
        "ca-GrQc",
        "email",
        "Chebyshev3",
        "ca-HepPh",
        "bcsstk01",
        "email-Eu-core",
        "CollegeMsg",
        "wiki-vote",
        "fpga_dcop_17",
        "email-enron",
        "gap-road",
        "gap-twitter",
        "gap-web",
        "gap-kron",
        "gap-urand",
    ]
]

_GAP_ROAD_SOURCES: list[int] = [
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
]

_GAP_TWITTER_SOURCES: list[int] = [
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
]

_GAP_WEB_SOURCES: list[int] = [
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
]

_GAP_KRON_SOURCES: list[int] = [
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
]

_GAP_URAND_SOURCES: list[int] = [
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
]


class SuiteSparseMatrixGenerator(Generator[SuiteSparseDataset]):
    """Downloads and caches raw SuiteSparse matrices, shared across every benchmark."""

    @property
    def name(self) -> str:
        return "suitesparse_matrix"

    @property
    def pretty_name(self) -> str:
        return "SuiteSparse Matrix Collection"

    @property
    def description(self) -> str:
        return (
            "Downloads and caches raw matrices from the SuiteSparse Matrix Collection."
            " Benchmark-specific generators compose this generator instead of"
            " downloading matrices themselves, so a matrix used by multiple benchmarks"
            " is only downloaded, cached, and uploaded once."
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
                title="The university of Florida sparse matrix collection",
                authors=[
                    Author("Timothy A. Davis"),
                    Author("Yifan Hu"),
                ],
                journal="ACM Transactions on Mathematical Software",
                publisher="Association for Computing Machinery (ACM)",
                volume="38",
                number="1",
                pages="1-25",
                year=2011,
                url="https://doi.org/10.1145/2049662.2049663",
                doi="10.1145/2049662.2049663",
            )
        ]

    @property
    def ai_disclosure(self) -> str:
        return (
            "No generative AI was used to write the algorithms for the benchmark"
            " function. Generative AI might have been used to construct the framework,"
            " comments and helper functions."
        )

    @property
    def motivation(self) -> str:
        return (
            "Many benchmarks reuse the same SuiteSparse matrices. Sharing a single"
            " cacheable generator for the raw download avoids redundant downloads and"
            " redundant cached copies of the same matrix."
        )

    @property
    def datasets(self) -> list[SuiteSparseDataset]:
        return _MATRICES

    def generate(self, dataset: SuiteSparseDataset) -> DataInstance:
        A, b, meta = load_suitesparse_matrix(dataset.source_name)
        inputs = [from_scipy(A)]
        if b is not None:
            inputs.append(from_numpy(b))
        return DataInstance(inputs=inputs, meta=meta)


class SuiteSparseMatrixBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return SuiteSparseMatrixGenerator()


def fetch_suitesparse_matrix(source_name: str) -> DataInstance:
    """Fetch (and cache) the raw matrix via the shared `SuiteSparseMatrixGenerator`.

    `.inputs[0]` is the matrix; `.inputs[1]` is its real RHS vector when the
    SuiteSparse collection entry ships one (see `.meta["has_b_file"]`).
    `.meta["shape"]` and `.meta["nnz"]` give the matrix shape/nnz.
    """
    raw_generator = SuiteSparseMatrixGenerator()
    raw_dataset = next(d for d in raw_generator.datasets if d.name == source_name)
    return raw_generator.cached_generate(raw_dataset)


def fetch_suitesparse_linear_system(
    source_name: str,
) -> tuple[BinsparseTensor, np.ndarray, bool]:
    """Fetch a matrix paired with a right-hand-side vector `b` to solve against.

    Returns `(A, b, has_real_rhs)`. Every CG/Jacobi/GMRES/LSQR/PreconditionedCG
    generator synthesizes `b` from the matrix the same deterministic way (`b = A @ x`
    for a random sparse `x`, via `random_rhs_for_matrix`'s defaults) unless the raw
    fetch actually included a real RHS file, so this is shared in one place rather
    than re-derived per benchmark. `has_real_rhs` tells the caller which happened,
    since that's the raw fetch's own ground truth, not something the caller tracks.
    """
    raw = fetch_suitesparse_matrix(source_name)
    A_bin = raw.inputs[0]
    has_real_rhs = len(raw.inputs) > 1
    if has_real_rhs:
        rhs_bin = raw.inputs[1]
        b = to_numpy(rhs_bin)
    else:
        b = random_rhs_for_matrix(to_scipy(A_bin).tocoo())
    return A_bin, b, has_real_rhs
