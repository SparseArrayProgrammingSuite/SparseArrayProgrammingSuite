from typing import Any

import numpy as np

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
from saps_framework import BinsparseFormat


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
        "gap-road",
    ]
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
        inputs = [BinsparseFormat.from_coo((A.row, A.col), A.data, A.shape)]
        if b is not None:
            inputs.append(BinsparseFormat.from_numpy(b))
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
) -> tuple[BinsparseFormat, np.ndarray, bool]:
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
        b = rhs_bin.data["values"].reshape(rhs_bin.data["shape"])
    else:
        b = random_rhs_for_matrix(A_bin.to_scipy_coo())
    return A_bin, b, has_real_rhs
