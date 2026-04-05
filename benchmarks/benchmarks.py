import json
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

import saps

xp = saps.xp

#poetry run asv run --python=same -v --set-commit-hash $(git rev-parse HEAD) --record-samples

@dataclass
class Author:
    name: str
    email: str | None = None

    def __str__(self):
        if self.email is None:
            return self.name
        return f"{self.name} <{self.email}>"


@dataclass
class Ref:
    title: str
    authors: list[Author]
    volume: str | None = None
    number: str | None = None
    pages: str | None = None
    year: int | None = None
    url: str | None = None
    doi: str | None = None

    def __str__(self):
        author_str = ", ".join(author.name for author in self.authors)
        volume_str = f", Vol. {self.volume}" if self.volume else ""
        number_str = f", No. {self.number}" if self.number else ""
        pages_str = f", pp. {self.pages}" if self.pages else ""
        year_str = f", {self.year}" if self.year else ""
        url_str = f", URL: {self.url}" if self.url else ""
        doi_str = f", DOI: {self.doi}" if self.doi else ""
        return f"{author_str}. \"{self.title}\"{volume_str}{number_str}{pages_str}{year_str}{url_str}{doi_str}."


class Benchmark(ABC):
    @property
    def params(self):
        return (self.dataset_names,)

    @property
    @abstractmethod
    def dataset_names(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def pretty_name(self) -> str:
        pass

    @property
    @abstractmethod
    def authors(self) -> list[Author]:
        pass

    @property
    @abstractmethod
    def description(self) -> str:
        pass

    @property
    @abstractmethod
    def motivation(self) -> str:
        pass

    @property
    @abstractmethod
    def references(self) -> list[Ref]:
        pass

    @property
    @abstractmethod
    def ai_disclosure(self) -> str:
        pass

    @property
    def pretty_source(self) -> str:
        source = {
            "name": self.pretty_name,
            "description": self.description,
            "motivation": self.motivation,
            "references": [str(ref) for ref in self.references],
            "authors": [str(author) for author in self.authors],
            "ai_disclosure": self.ai_disclosure,
        }
        return json.dumps(source)

    param_names = ["dataset"]


class TimeSuite(Benchmark):
    @property
    def dataset_names(self):
        return [0, 2, 3]

    @property
    def pretty_name(self):
        return "Random Numerical Linear Algebra: JL Approximate NN"

    @property
    def authors(self):
        return [Author("Vilohith Gokarakonda", "vgokarakonda3@gatech.edu")]

    @property
    def description(self):
        return (
            "Benchmarks Johnson-Lindenstrauss random projection followed by "
            "k-nearest-neighbor distance ranking."
        )

    @property
    def motivation(self):
        return (
            "Evaluates a foundational RNLA workload used in graph algorithms, "
            "PDE methods, and scientific machine learning."
        )

    @property
    def references(self):
        return [
            Ref(
                title="Random projection implementation reference",
                authors=[Author("scikit-learn contributors")],
                url="https://github.com/scikit-learn/scikit-learn/blob/d3898d9d57aeb1e960d266613a2e31b07bca39d7/sklearn/random_projection.py#L615",
            ),
            Ref(
                title="Randomized numerical linear algebra: A perspective on the field with an eye to software",
                authors=[
                    Author("Murray, R."),
                    Author("Demmel, J."),
                    Author("Mahoney, M. W."),
                    Author("Erichson, N. B."),
                    Author("Melnichenko, M."),
                    Author("Malik, O. A."),
                    Author("Dongarra, J."),
                ],
                year=2023,
                url="https://arxiv.org/abs/2302.11474",
            ),
        ]

    @property
    def ai_disclosure(self):
        return (
            "No generative AI was used to construct the benchmark function itself. "
            "Generative AI might have been used to construct tests."
        )

    def setup(self, dataset):
        self.d = {}
        for x in range(dataset):
            self.d[x] = None

    def time_range(self, dataset):
        d = self.d
        for key in range(dataset):
            d[key]


class MemSuite:
    def mem_list(self):
        return [0] * 256
