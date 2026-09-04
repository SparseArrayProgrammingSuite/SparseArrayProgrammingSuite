from __future__ import annotations

from typing import Any

import numpy as np

from binsparse.conversions import from_scipy, to_scipy

from saps.benchmark import (
    Author,
    Contributor,
    DataInstance,
    Dataset,
    Generator,
    Ref,
    ShellBenchmark,
)

NETFLIXPRIZE_NUM_USERS = 480189
NETFLIXPRIZE_NUM_MOVIES = 17770
NETFLIXPRIZE_NUM_RATINGS = 100480507


class NetflixPrizeDataset(Dataset):
    """Prepared sparse ratings matrix for the Netflix Prize training set."""

    @property
    def name(self) -> str:
        return "netflix"

    @property
    def pretty_name(self) -> str:
        return "Netflix Prize"

    @property
    def description(self) -> str:
        return (
            "Netflix Prize ratings parsed into a sparse user-by-movie matrix."
        )

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return "<ccs2012></ccs2012>"

    @property
    def metadata(self) -> dict[str, Any]:
        data = super().metadata
        data["kaggle_dataset"] = "netflix-inc/netflix-prize-data"
        return data


class NetflixPrizeGenerator(Generator[NetflixPrizeDataset]):
    """Downloads and caches the prepared Netflix Prize ratings matrix."""

    @property
    def name(self) -> str:
        return "netflixprize"

    @property
    def pretty_name(self) -> str:
        return "Netflix Prize Dataset"

    @property
    def description(self) -> str:
        return (
            "Downloads the Netflix Prize ratings from Kaggle, parses the combined "
            "ratings text files, and caches the prepared sparse ratings matrix."
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
                title="The Netflix Prize",
                authors=[Author("James Bennett"), Author("Stan Lanning")],
                year=2007,
                url="https://www.cs.uic.edu/~liub/KDD-cup-2007/NetflixPrize-description.pdf",
            ),
        ]

    @property
    def ai_disclosure(self) -> str:
        return "Generative AI was used to implement this generator."

    @property
    def motivation(self) -> str:
        return (
            "Multiple benchmarks reuse the Netflix Prize ratings matrix. Sharing a "
            "cacheable generator for the parsed matrix avoids redundant downloads, "
            "parsing, and cached copies."
        )

    @property
    def datasets(self) -> list[NetflixPrizeDataset]:
        return [NetflixPrizeDataset()]

    def generate(self, dataset: NetflixPrizeDataset) -> DataInstance:
        matrix = _load_netflixprize_matrix()
        _validate_netflixprize_matrix(matrix)
        return DataInstance(
            inputs=[from_scipy(matrix)],
            meta={
                "kaggle_dataset": "netflix-inc/netflix-prize-data",
                "num_users": int(matrix.shape[0]),
                "num_movies": int(matrix.shape[1]),
                "num_ratings": int(matrix.nnz),
            },
        )


def _load_netflixprize_matrix():
    import os

    import scipy.sparse

    import kagglehub

    cache_path = kagglehub.dataset_download("netflix-inc/netflix-prize-data")
    row_list = []
    col_list = []
    val_list = []
    user_map: dict[int, int] = {}
    current_movie = 0
    data_files = sorted(
        f
        for f in os.listdir(cache_path)
        if f.startswith("combined_data") and f.endswith(".txt")
    )
    for fname in data_files:
        with open(os.path.join(cache_path, fname)) as f:
            for line in f:
                line = line.strip()
                if line.endswith(":"):
                    current_movie = int(line[:-1]) - 1
                else:
                    uid_str, rating_str, _ = line.split(",", 2)
                    uid = int(uid_str)
                    if uid not in user_map:
                        user_map[uid] = len(user_map)
                    row_list.append(user_map[uid])
                    col_list.append(current_movie)
                    val_list.append(float(rating_str))

    n_users = len(user_map)
    return scipy.sparse.csr_matrix(
        (
            np.array(val_list, dtype=np.float32),
            (np.array(row_list, dtype=np.int32), np.array(col_list, dtype=np.int32)),
        ),
        shape=(n_users, NETFLIXPRIZE_NUM_MOVIES),
    )


def _validate_netflixprize_matrix(matrix) -> None:
    expected_shape = (NETFLIXPRIZE_NUM_USERS, NETFLIXPRIZE_NUM_MOVIES)
    if matrix.shape != expected_shape or matrix.nnz != NETFLIXPRIZE_NUM_RATINGS:
        raise ValueError(
            "Expected the full Netflix Prize training matrix with "
            f"shape {expected_shape} and {NETFLIXPRIZE_NUM_RATINGS} ratings; "
            f"got shape {matrix.shape} and {matrix.nnz} ratings."
        )


class NetflixPrizeBenchmark(ShellBenchmark):
    @property
    def generator(self) -> Generator:
        return NetflixPrizeGenerator()


def fetch_netflixprize_dataset() -> DataInstance:
    """Fetch (and cache) the prepared Netflix Prize ratings matrix."""
    raw_generator = NetflixPrizeGenerator()
    return raw_generator.cached_generate(raw_generator.datasets[0])


def fetch_netflixprize_matrix():
    """Fetch the prepared sparse user-by-movie ratings matrix."""
    raw = fetch_netflixprize_dataset()
    return to_scipy(raw.inputs[0]).tocsr(), raw.meta
