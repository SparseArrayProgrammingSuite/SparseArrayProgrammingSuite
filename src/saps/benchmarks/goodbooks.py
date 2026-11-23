from __future__ import annotations

from io import BytesIO
from typing import Any
from urllib.request import urlopen

import numpy as np

from binsparse.conversions import from_scipy, to_scipy

from saps import licenses
from saps.benchmark import (
    Author,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    License,
    Ref,
    ShellBenchmark,
)

# ratings.csv at goodbooks-10k's latest commit, pinned so the download never changes.
_GOODBOOKS_COMMIT = "6dd165b555a7b47b2dd36743a425776e641ff50c"
_GOODBOOKS_RATINGS_URL = (
    "https://raw.githubusercontent.com/zygmuntz/goodbooks-10k/"
    f"{_GOODBOOKS_COMMIT}/ratings.csv"
)

GOODBOOKS_NUM_USERS = 53424
GOODBOOKS_NUM_BOOKS = 10000
GOODBOOKS_NUM_RATINGS = 5976479


class GoodbooksDataset(Dataset):
    """Prepared sparse ratings matrix for goodbooks-10k."""

    @property
    def name(self) -> str:
        return "goodbooks"

    @property
    def pretty_name(self) -> str:
        return "goodbooks-10k"

    @property
    def description(self) -> str:
        return "goodbooks-10k ratings parsed into a sparse user-by-book matrix."

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["github_commit"] = _GOODBOOKS_COMMIT
        return data


class GoodbooksRatingsGenerator(Generator[GoodbooksDataset]):
    """Downloads and caches the prepared goodbooks-10k ratings matrix."""

    @property
    def name(self) -> str:
        return "goodbooks_ratings"

    @property
    def pretty_name(self) -> str:
        return "goodbooks-10k Ratings"

    @property
    def description(self) -> str:
        return (
            "Downloads goodbooks-10k's ratings.csv from GitHub at a pinned commit, "
            "parses it, and caches the prepared sparse ratings matrix."
        )

    @property
    def suites(self) -> list[str]:
        # --cache-datasets prepares the shell suite.
        return ["shell"]

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
                title="goodbooks-10k",
                authors=[Author("Zygmunt Zając")],
                year=2017,
                url="https://github.com/zygmuntz/goodbooks-10k",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return "Generative AI was used to implement this generator."

    @property
    def notes(self) -> str:
        return (
            "Multiple benchmarks reuse the goodbooks-10k ratings matrix. Sharing a "
            "cacheable generator for the parsed matrix avoids redundant downloads, "
            "parsing, and cached copies."
        )

    @property
    def license(self) -> License | None:
        return licenses.GOODBOOKS

    @property
    def datasets(self) -> list[GoodbooksDataset]:
        return [GoodbooksDataset()]

    def generate(self, dataset: GoodbooksDataset) -> DataInstance:
        matrix = _load_goodbooks_matrix()
        _validate_goodbooks_matrix(matrix)
        return DataInstance(
            inputs=[from_scipy(matrix)],
            meta={
                "github_commit": _GOODBOOKS_COMMIT,
                "num_users": int(matrix.shape[0]),
                "num_books": int(matrix.shape[1]),
                "num_ratings": int(matrix.nnz),
            },
        )


def _load_goodbooks_matrix():
    import scipy.sparse

    with urlopen(_GOODBOOKS_RATINGS_URL, timeout=60) as response:
        ratings = np.loadtxt(
            BytesIO(response.read()),
            delimiter=",",
            skiprows=1,
            dtype=np.int64,
            ndmin=2,
        )
    # Each row is user_id,book_id,rating, and both IDs count from 1.
    users, books, values = ratings.T
    return scipy.sparse.csr_matrix(
        (
            values.astype(np.float32),
            ((users - 1).astype(np.int32), (books - 1).astype(np.int32)),
        ),
        shape=(int(users.max()), int(books.max())),
    )


def _validate_goodbooks_matrix(matrix) -> None:
    # Duplicate ratings would be summed into one entry, so they also fail this.
    expected_shape = (GOODBOOKS_NUM_USERS, GOODBOOKS_NUM_BOOKS)
    if matrix.shape != expected_shape or matrix.nnz != GOODBOOKS_NUM_RATINGS:
        raise ValueError(
            "Expected the full goodbooks-10k ratings matrix with "
            f"shape {expected_shape} and {GOODBOOKS_NUM_RATINGS} ratings; "
            f"got shape {matrix.shape} and {matrix.nnz} ratings."
        )


class GoodbooksRatingsShellBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return GoodbooksRatingsGenerator()


def fetch_goodbooks_dataset() -> DataInstance:
    """Fetch (and cache) the prepared goodbooks-10k ratings matrix."""
    raw_generator = GoodbooksRatingsGenerator()
    return raw_generator.cached_generate(raw_generator.datasets[0])


def fetch_goodbooks_matrix():
    """Fetch the prepared sparse user-by-book ratings matrix."""
    raw = fetch_goodbooks_dataset()
    return to_scipy(raw.inputs[0]).tocsr(), raw.meta
