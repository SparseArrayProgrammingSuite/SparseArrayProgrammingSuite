import inspect
import json
import os
import re
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from saps.framework import load_framework
from saps.storage import build_storage_backend
from saps_framework.binsparse_format import BinsparseFormat as BinsparseFormat
from saps_framework.framework import Framework


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


def _tag_slug(value: str) -> str:
    tag = re.sub(r"[^0-9A-Za-z]+", "-", value.lower())
    return re.sub(r"-+", "-", tag).strip("-")


def ccs_xml_to_tags(xml_text: str | None) -> list[str]:
    """Convert pasted ACM CCS XML into SAPS tag slugs."""
    if not xml_text:
        return []

    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return []

    tags: set[str] = set()
    for node in root.iter():
        if node.tag.rsplit("}", 1)[-1] != "concept_desc" or not node.text:
            continue
        for part in re.split(r"\s*(?:~|::)\s*", node.text.strip()):
            tag = _tag_slug(part)
            if tag:
                tags.add(tag)
    return sorted(tags)


def _get_or_add_record(records: list[dict], key: str, value: str, defaults: dict):
    for record in records:
        if record.get(key) == value:
            return record
    record = {key: value, **defaults}
    records.append(record)
    return record


def _write_statistics_tags(
    statistics_path: Path,
    benchmark_id: str,
    benchmark_name: str,
    generator_name: str,
    dataset_name: str,
    tags: list[str],
):
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    if statistics_path.exists():
        document = json.loads(statistics_path.read_text(encoding="utf-8"))
    else:
        document = {"benchmarks": []}

    benchmark = _get_or_add_record(
        document.setdefault("benchmarks", []),
        "id",
        benchmark_id,
        {"name": benchmark_name, "statistics": [], "generators": []},
    )
    benchmark.setdefault("name", benchmark_name)
    benchmark.setdefault("statistics", [])

    generator = _get_or_add_record(
        benchmark.setdefault("generators", []),
        "name",
        generator_name,
        {"statistics": [], "datasets": []},
    )
    generator.setdefault("statistics", [])

    dataset = _get_or_add_record(
        generator.setdefault("datasets", []),
        "name",
        dataset_name,
        {"statistics": []},
    )
    dataset["statistics"] = sorted({*dataset.get("statistics", []), *tags})

    document["benchmarks"] = sorted(
        document.get("benchmarks", []),
        key=lambda record: record.get("id", ""),
    )
    for record in document["benchmarks"]:
        record["generators"] = sorted(
            record.get("generators", []),
            key=lambda generator: generator["name"],
        )
        for generator in record["generators"]:
            generator["datasets"] = sorted(
                generator.get("datasets", []),
                key=lambda dataset: dataset["name"],
            )

    statistics_path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
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
    def suites(self) -> list[str]: ...

    @property
    @abstractmethod
    def concepts(self) -> str: ...

    @property
    def topics(self) -> list[str]:
        return ccs_xml_to_tags(self.concepts)


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
            "suites": self.suites,
            "concepts": self.concepts,
            "topics": self.topics,
        }


@dataclass
class DataInstance:
    inputs: list[Any]
    meta: dict[str, Any]
    ref_outputs: list[Any] | None = None
    ref_meta: dict[str, Any] | None = None


TDataset = TypeVar("TDataset", bound=Dataset)


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
            "suites": self.suites,
            "concepts": self.concepts,
            "topics": self.topics,
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
    def benchmark(self, xp: Framework, data: list[Any], meta: Any) -> Any: ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return

        try:
            instance = cls()
        except (TypeError, ValueError):
            return

        benchmark_source = inspect.getsource(cls.benchmark)

        def _peakmem_run(self, param):
            self.run(param)

        setattr(cls, f"peakmem_{instance.name}", _peakmem_run)
        getattr(cls, f"peakmem_{instance.name}").pretty_source = benchmark_source

        def _time_run(self, param):
            self.run(param)

        setattr(cls, f"time_{instance.name}", _time_run)
        getattr(cls, f"time_{instance.name}").pretty_source = benchmark_source

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

    def setup(
        self, param, *, use_cache: bool = True, xp: Framework | None = None
    ):
        import logging

        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(levelname)s %(name)s: %(message)s",
            )
        problem = (
            param.generator.cached_generate(param.dataset)
            if use_cache
            else param.generator.generate(param.dataset)
        )
        self._input = problem.inputs
        self._meta = problem.meta
        self._ref_outputs = problem.ref_outputs
        self._ref_meta = problem.ref_meta
        if xp is None:
            try:
                xp = load_framework()
            except RuntimeError:
                xp = None
        if xp is not None:
            self._xp = xp

            def benchmark(data, meta):
                return self.benchmark(xp, data, meta)

            self._compiled_benchmark = xp.compile(benchmark)

    def run(self, param):
        if not hasattr(self, "_xp") or not hasattr(self, "_compiled_benchmark"):
            raise RuntimeError(
                "Benchmark.setup must bind a framework before run. Pass xp to "
                "setup or set SAPS_FRAMEWORK."
            )
        xp = self._xp
        if hasattr(xp, "reset_stats"):
            xp.reset_stats()
        input = [xp.from_binsparse(d) for d in self._input]
        output = self._compiled_benchmark(input, self._meta)
        output = [xp.to_binsparse(o) for o in output]
        self._output = output
        self._write_tagger_stats(param, xp)

    def _write_tagger_stats(self, param, xp: Framework):
        stats_dir = os.environ.get("SAPS_TAGGER_STATS_DIR")
        statistics_path = os.environ.get("SAPS_STATISTICS_PATH")
        if not hasattr(xp, "tags"):
            return

        benchmark_id = (
            f"{self.__class__.__module__}.{self.__class__.__name__}.{self.name}"
        )
        tags = sorted(getattr(xp, "tags", []))
        data = {
            "benchmark_id": benchmark_id,
            "benchmark_name": self.name,
            "generator_name": param.generator.name,
            "dataset_name": param.dataset.name,
            "tags": tags,
        }
        if stats_dir:
            path = Path(stats_dir)
            path.mkdir(parents=True, exist_ok=True)
            record_id = (
                f"{benchmark_id}.{param.generator.name}.{param.dataset.name}"
            )
            safe_id = "".join(
                char if char.isalnum() or char in "._-" else "_"
                for char in record_id
            )
            output_path = path / f"{safe_id}.json"
            output_path.write_text(
                json.dumps(data, indent=2, default=str) + "\n",
                encoding="utf-8",
            )
        if statistics_path:
            _write_statistics_tags(
                Path(statistics_path),
                benchmark_id,
                self.name,
                param.generator.name,
                param.dataset.name,
                tags,
            )

    def teardown(self, param):
        if hasattr(self, "_output"):
            self.check(param)
            del self._output
        if hasattr(self, "_meta"):
            del self._meta
        if hasattr(self, "_ref_outputs"):
            del self._ref_outputs
        if hasattr(self, "_ref_meta"):
            del self._ref_meta
        if hasattr(self, "_input"):
            del self._input
        if hasattr(self, "_xp"):
            del self._xp
        if hasattr(self, "_compiled_benchmark"):
            del self._compiled_benchmark

    def check(self, param):
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
            "suites": self.suites,
            "concepts": self.concepts,
            "topics": self.topics,
            "authors": [str(a) for a in self.authors],
            "references": [str(r) for r in self.references],
            "ai_disclosure": self.ai_disclosure,
            "motivation": self.motivation,
            "generators": [generator.metadata for generator in self.generators],
        }
