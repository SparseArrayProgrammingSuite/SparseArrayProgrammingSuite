# Write the benchmarking functions here.
# See "Writing benchmarks" in the asv docs for more information.
import saps
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, String
import json


xp = saps.xp

#poetry run asv run --python=same -v --set-commit-hash $(git rev-parse HEAD) --record-samples

@dataclass
class Author:
    name: String
    email: String | None = None

    def __str__(self):
        return f"{self.name} <{self.email}>"

@dataclass
class Ref:
    title: String
    authors: List[Author]
    volume: String | None = None
    number: String | None = None
    pages: String | None = None
    year: int | None = None
    url: String | None = None
    doi: String | None = None

    def __str__(self):
        author_str = ", ".join(String(author.name) for author in self.authors)
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
        return (self.dataset_names)
    
    @property
    @abstractmethod
    def dataset_names(self) -> List[String]:
        pass

    @property
    @abstractmethod
    def pretty_name(self) -> String: ...

    @property
    @abstractmethod
    def authors(self) -> List[Author]: ...

    @property
    @abstractmethod
    def description(self) -> String:...

    @property
    @abstractmethod
    def motivation(self) -> String:...

    @property
    @abstractmethod
    def references(self) -> List[Ref]: ...

    @property
    @abstractmethod
    def ai_disclosure(self) -> String: ...

    @property
    def pretty_source(self) -> String:
        source = {
            "name": self.pretty_name,
            "description": self.description,
            "motivation": self.motivation,
            "references": map(str, self.references),
            "authors": map(str, self.authors),
            "ai_disclosure": self.ai_disclosure,
        }
        return json.dumps(source)

    param_names = ["dataset"]

class TimeSuite(Benchmark):
    """
    An example benchmark that times the performance of various kinds
    of iterating over dictionaries in Python.
    """
    @property
    def dataset_names(self):
        return [0, 1, 100]

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
