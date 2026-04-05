import json
import os
import inspect
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path    

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

class Benchmark(ABC):
    _ASV_METHOD_PREFIXES = ("time_", "mem_", "track_", "peakmem_", "timeraw_")

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        try:
            instance = cls()
        except Exception:
            return

        entry = {
            "module": cls.__module__,
            "class_name": cls.__name__,
            "tag": instance.tag,
            "id": f"{cls.__module__}.{cls.__name__}.{instance.tag}",
            "dataset_names": list(instance.dataset_names),
            "param_names": list(getattr(instance, "param_names", [])),
            "authors": [str(a) for a in instance.authors],
            "references": [str(r) for r in instance.references],
            "description": instance.description,
            "motivation": instance.motivation,
            "ai_disclosure": instance.ai_disclosure,
            "benchmark_methods": [
                name
                for name, _ in inspect.getmembers(cls, inspect.isfunction)
                if any(name.startswith(p) for p in cls._ASV_METHOD_PREFIXES)
            ],
        }

        payload = json.loads(benchmark_metadata.read_text(encoding="utf-8"))
        existing = payload.get("benchmarks", [])
        by_id = {item.get("id"): item for item in existing if isinstance(item, dict)}
        by_id[entry["id"]] = entry
        payload["benchmarks"] = sorted(by_id.values(), key=lambda item: item["id"])

        tmp_path = benchmark_metadata.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(benchmark_metadata)

        def _mem_run(self, dataset):
            self.run(dataset)

        def _time_run(self, dataset):
            self.run(dataset)

        setattr(cls, f"mem_{instance.tag}", _mem_run)
        setattr(getattr(cls, f"mem_{instance.tag}"), "pretty_source", inspect.getsource(cls.run))
        
        setattr(cls, f"time_{instance.tag}", _time_run)
        setattr(getattr(cls, f"time_{instance.tag}"), "pretty_source", inspect.getsource(cls.run))


    @property
    def params(self):
        return (self.dataset_names,)

    @property
    @abstractmethod
    def dataset_names(self) -> list[str]:
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    @abstractmethod
    def tag(self) -> str:
        pass

    @property
    @abstractmethod
    def authors(self) -> list[Contributor]:
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

    param_names = ["dataset"]

    @abstractmethod
    def setup(self, dataset): ...

    @abstractmethod
    def run(self, dataset): ...