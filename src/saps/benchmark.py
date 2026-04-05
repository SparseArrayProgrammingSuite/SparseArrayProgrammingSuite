import json
import os
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path    
from typing import Any

import saps
from saps.binsparse_format import BinsparseFormat

xp = saps.xp
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
    volume: str | None = None
    number: str | None = None
    pages: str | None = None
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
        year_str = f", {self.year}" if self.year else ""
        url_str = f", URL: {self.url}" if self.url else ""
        doi_str = f", DOI: {self.doi}" if self.doi else ""
        return (
            f"{author_str}. \"{self.title}\""
            f"{journal_str}{conference_str}{booktitle_str}{publisher_str}{institution_str}"
            f"{volume_str}{number_str}{pages_str}{year_str}{url_str}{doi_str}."
        )

saps_root = Path(os.getenv("SAPS_DATA_PATH", Path(__file__).parent.parent.parent))
benchmark_metadata = saps_root / "benchmark_metadata.json"
benchmark_metadata.parent.mkdir(parents=True, exist_ok=True)
benchmark_metadata.unlink(missing_ok=True)
benchmark_metadata.write_text(json.dumps({"benchmarks": []}, indent=2), encoding="utf-8")

class Metadata(ABC):
    @property
    @abstractmethod
    def metadata(self) -> dict[str, Any]:...

class Tagged(Metadata):
    @property
    @abstractmethod
    def name(self) -> str:...

    @property
    @abstractmethod
    def pretty_name(self) -> str:...

    @property
    @abstractmethod
    def description(self) -> str:...

    @property
    @abstractmethod
    def tags(self) -> list[str]:...

class Attributed(ABC):
    @property
    @abstractmethod
    def authors(self) -> list[Contributor]:...

    @property
    @abstractmethod
    def references(self) -> list[Ref]:...

    @property
    @abstractmethod
    def ai_disclosure(self) -> str:...

class Motivated(ABC):
    @property
    @abstractmethod
    def motivation(self) -> str:...

class Dataset(Tagged):
    @property
    def metadata(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "pretty_name": self.pretty_name,
            "description": self.description,
            "tags": self.tags,
        }

class Generator(Tagged, Attributed, Motivated):
    @property
    @abstractmethod
    def datasets(self) -> list[Dataset]:...

    @property
    def dataset_names(self) -> list[str]:
        return [dataset.name for dataset in self.datasets]
    
    @abstractmethod
    def generate(self, dataset: Dataset) -> tuple[list[BinsparseFormat], Any]:...

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
class Param:
    generator: Generator
    dataset: Dataset
    def __repr__(self):
        return f"{self.generator.name}.{self.dataset.name}"

    def generate(self):
        return self.generator.generate(self.dataset)

class Benchmark(Tagged, Attributed, Motivated):
    @property
    @abstractmethod
    def generators(self) -> list[Generator]:...

    @abstractmethod
    def benchmark(self, data: list[Any], meta:Any) -> Any:...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        try:
            instance = cls()
        except Exception:
            return
        
        entry = instance.metadata

        payload = json.loads(benchmark_metadata.read_text(encoding="utf-8"))
        existing = payload.get("benchmarks", [])
        by_id = {item.get("id"): item for item in existing if isinstance(item, dict)}
        by_id[entry["id"]] = entry
        payload["benchmarks"] = sorted(by_id.values(), key=lambda item: item["id"])

        tmp_path = benchmark_metadata.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(benchmark_metadata)

        def _mem_run(self, param):
            self.run(param)

        def _time_run(self, param):
            self.run(param)

        setattr(cls, f"mem_{instance.tag}", _mem_run)
        setattr(getattr(cls, f"mem_{instance.tag}"), "pretty_source", inspect.getsource(cls.benchmark))
        
        setattr(cls, f"time_{instance.tag}", _time_run)
        setattr(getattr(cls, f"time_{instance.tag}"), "pretty_source", inspect.getsource(cls.benchmark))

        setattr(cls.setup, "pretty_source", "\n".join(inspect.getsource(generator.generate) for generator in instance.generators))
    
    @property
    def params(self):
        return [Param(generator, dataset) for generator in self.generators for dataset in generator.datasets]

    param_names = ["dataset"]

    def setup(self, param):
        input, meta = param.generate()
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
        #TODO do better than this
        for item in self._output:
            assert isinstance(item, BinsparseFormat), "Output must be in binsparse format"

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