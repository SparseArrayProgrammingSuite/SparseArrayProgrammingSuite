import ast
import hashlib
import importlib.util
import inspect
import json
import os
import re
import sys
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Any, Generic, TypeVar

from saps.dependencies import dependency_versions
from saps.framework import load_framework
from saps.storage import build_storage_backend
from saps_framework.binsparse_format import BinsparseFormat as BinsparseFormat
from saps_framework.framework import Framework


def _repo_root() -> Path:
    root = os.environ.get("SAPS_REPO_ROOT")
    if root:
        return Path(root).resolve()
    return Path(__file__).resolve().parents[2]


def _module_path(module_name: str) -> Path | None:
    try:
        spec = importlib.util.find_spec(module_name)
    except (ImportError, ModuleNotFoundError, ValueError):
        return None
    if spec is None or spec.origin is None:
        return None
    path = Path(spec.origin).resolve()
    if path.suffix != ".py":
        return None
    try:
        path.relative_to(_repo_root())
    except ValueError:
        return None
    return path


def _imported_modules(path: Path) -> set[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    except SyntaxError:
        return set()

    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise RuntimeError(
                    f"Relative import in {path}:{node.lineno}. "
                    "Use an absolute import so freshness hashes are stable."
            )
            if node.module:
                modules.add(node.module)
    return modules


@cache
def _freshness_inputs(
    module_name: str,
) -> tuple[tuple[Path, ...], tuple[str, ...]]:
    pending = [module_name]
    seen_modules: set[str] = set()
    seen_files: set[Path] = set()
    external_modules: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen_modules:
            continue
        seen_modules.add(current)

        path = _module_path(current)
        if path is None:
            if (
                current.split(".", 1)[0] not in sys.stdlib_module_names
                and _module_path(current.split(".", 1)[0]) is None
            ):
                external_modules.add(current)
            continue
        if path in seen_files:
            continue
        seen_files.add(path)

        pending.extend(
            dependency
            for dependency in _imported_modules(path)
            if dependency not in seen_modules
        )

    return tuple(sorted(seen_files)), tuple(sorted(external_modules))


@cache
def _file_content_hash(path: str, _mtime_ns: int, _size: int) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cached_file_content_hash(path: Path) -> str:
    stat = path.stat()
    return _file_content_hash(str(path), stat.st_mtime_ns, stat.st_size)


@cache
def _class_source_path(cls: type) -> Path:
    return Path(inspect.getfile(cls)).resolve()


@cache
def _source_file(cls: type) -> str:
    return _class_source_path(cls).relative_to(_repo_root()).as_posix()


@cache
def _source_dependencies(module_name: str) -> list[str]:
    _, external_modules = _freshness_inputs(module_name)
    return list(external_modules)


def _file_signature(path: Path) -> tuple[str, int, int]:
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@cache
def _source_freshness_for_files(
    root: str, file_signatures: tuple[tuple[str, int, int], ...]
) -> str:
    root_path = Path(root)
    digest = hashlib.sha256()
    for path, _, _ in file_signatures:
        file_path = Path(path)
        relative = file_path.relative_to(root_path).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(_cached_file_content_hash(file_path).encode("utf-8"))
        digest.update(b"\0")

    return digest.hexdigest()


def _source_freshness(module_name: str, source_path: Path) -> str:
    files, _ = _freshness_inputs(module_name)
    if not files:
        files = (source_path,)
    return _source_freshness_for_files(
        str(_repo_root()), tuple(_file_signature(path) for path in files)
    )


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
    benchmark_name: str,
    generator_name: str,
    dataset_name: str,
    dataset_file: str,
    dataset_freshness: str,
    dataset_dependencies: list[str],
    tags: list[str],
):
    version_records = dependency_versions(dataset_dependencies)
    statistics_path.parent.mkdir(parents=True, exist_ok=True)
    if statistics_path.exists():
        document = json.loads(statistics_path.read_text(encoding="utf-8"))
    else:
        document = {"benchmarks": []}

    benchmark = _get_or_add_record(
        document.setdefault("benchmarks", []),
        "name",
        benchmark_name,
        {"statistics": [], "generators": []},
    )
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
        {
            "dependencies": dataset_dependencies,
            "dependency_versions": version_records,
            "file": dataset_file,
            "freshness": dataset_freshness,
            "statistics": [],
        },
    )
    dataset["dependencies"] = dataset_dependencies
    dataset["dependency_versions"] = version_records
    dataset["file"] = dataset_file
    dataset["freshness"] = dataset_freshness
    dataset["statistics"] = sorted({*dataset.get("statistics", []), *tags})

    document["benchmarks"] = sorted(
        document.get("benchmarks", []),
        key=lambda record: record["name"],
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

    @property
    def file(self) -> str:
        return _source_file(self.__class__)

    @property
    def freshness(self) -> str:
        return _source_freshness(
            self.__class__.__module__, _class_source_path(self.__class__)
        )

    @property
    def dependencies(self) -> list[str]:
        return _source_dependencies(self.__class__.__module__)


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
            "dependencies": self.dependencies,
            "file": self.file,
            "freshness": self.freshness,
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
            "dependencies": self.dependencies,
            "file": self.file,
            "freshness": self.freshness,
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

        tags = sorted(getattr(xp, "tags", []))
        data = {
            "benchmark_name": self.name,
            "generator_name": param.generator.name,
            "dataset_name": param.dataset.name,
            "tags": tags,
        }
        if stats_dir:
            path = Path(stats_dir)
            path.mkdir(parents=True, exist_ok=True)
            record_id = (
                f"{self.name}.{param.generator.name}.{param.dataset.name}"
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
                self.name,
                param.generator.name,
                param.dataset.name,
                param.dataset.file,
                param.dataset.freshness,
                param.dataset.dependencies,
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
            "dependencies": self.dependencies,
            "file": self.file,
            "freshness": self.freshness,
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
            "generators": [generator.metadata for generator in self.generators],
        }
