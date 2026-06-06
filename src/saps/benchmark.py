import inspect
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import wraps
from typing import Any, Generic, TypeAlias, TypeVar

from saps.framework import xp
from saps.storage import build_storage_backend
from saps_framework.binsparse_format import BinsparseFormat


def compile(func):
    compiled = None

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal compiled

        compiler = getattr(xp, "compile", None)
        if compiler is None:
            return func(*args, **kwargs)

        if compiled is None:
            compiled = compiler(func)

        return compiled(*args, **kwargs)

    return wrapper


@dataclass
class Author:
    name: str
    email: str | None = None

    def __str__(self):
        if self.email is None:
            return self.name
        return f"{self.name} <{self.email}>"


@dataclass
class Contributor:
    name: str
    email: str

    def __str__(self):
        return f"{self.name} <{self.email}>"


@dataclass
class Ref:
    title: str
    authors: list[Author]
    journal: str | None = None
    conference: str | None = None
    booktitle: str | None = None
    publisher: str | None = None
    institution: str | None = None
    volume: str | int | None = None
    number: str | int | None = None
    pages: str | None = None
    city: str | None = None
    year: int | None = None
    url: str | None = None
    doi: str | None = None

    def __str__(self):
        author_str = ", ".join(author.name for author in self.authors)
        journal_str = f", {self.journal}" if self.journal else ""
        conference_str = f", {self.conference}" if self.conference else ""
        booktitle_str = f", In {self.booktitle}" if self.booktitle else ""
        publisher_str = f", {self.publisher}" if self.publisher else ""
        institution_str = f", {self.institution}" if self.institution else ""
        volume_str = f", Vol. {self.volume}" if self.volume else ""
        number_str = f", No. {self.number}" if self.number else ""
        pages_str = f", pp. {self.pages}" if self.pages else ""
        city_str = f", {self.city}" if self.city else ""
        year_str = f", {self.year}" if self.year else ""
        url_str = f", URL: {self.url}" if self.url else ""
        doi_str = f", DOI: {self.doi}" if self.doi else ""
        return (
            f'{author_str}. "{self.title}"'
            f"{journal_str}{conference_str}{booktitle_str}{publisher_str}{institution_str}"
            f"{volume_str}{number_str}{pages_str}{city_str}{year_str}{url_str}{doi_str}."
        )


class Metadata(ABC):
    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]: ...


class Tagged(Metadata):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    @abstractmethod
    def pretty_name(self) -> str: ...

    @property
    @abstractmethod
    def description(self) -> str: ...

    @property
    @abstractmethod
    def tags(self) -> list[str]: ...


class Attributed(ABC):
    @property
    @abstractmethod
    def authors(self) -> list[Contributor]: ...

    @property
    @abstractmethod
    def references(self) -> list[Ref]: ...

    @property
    @abstractmethod
    def ai_disclosure(self) -> str: ...


class Motivated(ABC):
    @property
    @abstractmethod
    def motivation(self) -> str: ...


class Dataset(Tagged):
    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pretty_name": self.pretty_name,
            "description": self.description,
            "tags": self.tags,
        }


TDataset = TypeVar("TDataset", bound=Dataset)
# Every DataInstance is a tuple of a list of BinsparseFormat objects
# and a json serializable dictionary
DataInstance: TypeAlias = tuple[list[BinsparseFormat], dict[str, Any]]


class Generator(Tagged, Attributed, Motivated, Generic[TDataset]):
    @property
    @abstractmethod
    def datasets(self) -> list[TDataset]: ...

    @property
    def cacheable(self) -> bool:
        return True

    @property
    def dataset_names(self) -> list[str]:
        return [dataset.name for dataset in self.datasets]

    @property
    def backend(self):
        if getattr(self, "_backend", None) is None:
            backend_type = os.environ.get("REMOTE_STORAGE_BACKEND")
            backend_bucket = os.environ.get("REMOTE_STORAGE_BUCKET")
            self._backend = build_storage_backend(backend_type, backend_bucket)
        return self._backend

    def cached_generate(self, dataset: TDataset) -> DataInstance:
        cacheable = self.cacheable
        if not cacheable:
            return self.generate(dataset)

        return self.backend.retrieve_dataset(self, dataset)

    @abstractmethod
    def generate(self, dataset: TDataset) -> DataInstance: ...

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pretty_name": self.pretty_name,
            "description": self.description,
            "tags": self.tags,
            "authors": [str(a) for a in self.authors],
            "references": [str(r) for r in self.references],
            "ai_disclosure": self.ai_disclosure,
            "motivation": self.motivation,
            "datasets": [dataset.metadata for dataset in self.datasets],
        }


@dataclass
class Param(Generic[TDataset]):
    generator: Generator[TDataset]
    dataset: TDataset

    def __repr__(self):
        return f"{self.generator.name}.{self.dataset.name}"


class Benchmark(Tagged, Attributed, Motivated):
    @property
    @abstractmethod
    def generators(self) -> list[Generator[Any]]: ...

    @abstractmethod
    def benchmark(self, data: list[Any], meta: Any) -> Any: ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        try:
            instance = cls()
        except (TypeError, ValueError):
            return

        def _mem_run(self, param):
            self.run(param)

        def _time_run(self, param):
            self.run(param)

        setattr(cls, f"mem_{instance.name}", _mem_run)
        getattr(cls, f"mem_{instance.name}").pretty_source = inspect.getsource(
            cls.benchmark
        )

        setattr(cls, f"time_{instance.name}", _time_run)
        getattr(cls, f"time_{instance.name}").pretty_source = inspect.getsource(
            cls.benchmark
        )

        cls.setup.pretty_source = "\n".join(
            inspect.getsource(generator.generate) for generator in instance.generators
        )

    @property
    def params(self):
        return [
            Param(generator, dataset)
            for generator in self.generators
            for dataset in generator.datasets
        ]

    param_names = ["dataset"]

    def setup(self, param):
        import logging

        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(levelname)s %(name)s: %(message)s",
            )
        input, meta = param.generator.cached_generate(param.dataset)
        self._input = input
        self._meta = meta

    def run(self, param):
        input = [xp.from_binsparse(d) for d in self._input]
        output = self.benchmark(input, self._meta)
        output = [xp.to_binsparse(o) for o in output]
        self._output = output

    def teardown(self, param):
        if hasattr(self, "_output"):
            self.check_correct(param)
            del self._output
        if hasattr(self, "_meta"):
            del self._meta
        if hasattr(self, "_input"):
            del self._input

    def check_correct(self, param):
        # TODO do better than this
        for item in self._output:
            assert isinstance(item, BinsparseFormat), (
                "Output must be in binsparse format"
            )

    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pretty_name": self.pretty_name,
            "id": f"{self.__class__.__module__}.{self.__class__.__name__}.{self.name}",
            "description": self.description,
            "tags": self.tags,
            "authors": [str(a) for a in self.authors],
            "references": [str(r) for r in self.references],
            "ai_disclosure": self.ai_disclosure,
            "motivation": self.motivation,
            "generators": [generator.metadata for generator in self.generators],
        }
