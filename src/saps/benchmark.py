import inspect
import logging
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Generic, TypeVar

from binsparse import BinsparseTensor as BinsparseTensor

from saps.benchmark_status import MeasurementTimeout, report_status
from saps.framework import load_framework
from saps.freshness import repo_root, source_freshness
from saps.storage import build_storage_backend
from saps_framework.framework import Framework


def _apply_memory_limit() -> None:
    """Cap this process's address space to SAPS_MEMORY_LIMIT_BYTES if set, so a
    runaway benchmark raises MemoryError instead of OOM-killing the machine.

    Only enforced on Linux: macOS accepts setrlimit(RLIMIT_AS) but silently does
    not enforce it, and Windows has no resource module at all. On those platforms
    we warn rather than give a false sense of protection.
    """
    limit = os.environ.get("SAPS_MEMORY_LIMIT_BYTES")
    if not limit:
        return
    if sys.platform != "linux":
        logging.warning(
            "Memory limit requested (SAPS_MEMORY_LIMIT_BYTES=%s) but it is only "
            "enforced on Linux; benchmark memory is uncapped on %s.",
            limit,
            sys.platform,
        )
        return
    try:
        import resource
    except ImportError:  # non-Unix platforms
        return
    nbytes = int(limit)
    _soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    new_soft = nbytes if hard == resource.RLIM_INFINITY else min(nbytes, hard)
    resource.setrlimit(resource.RLIMIT_AS, (new_soft, hard))


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
                tags.add(f"concept-{tag}")
    return sorted(tags)


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
        source_path = Path(inspect.getfile(self.__class__)).resolve()
        try:
            package_path = source_path.relative_to(Path(__file__).resolve().parent)
        except ValueError:
            return source_path.relative_to(repo_root()).as_posix()
        return (Path("src/saps") / package_path).as_posix()

    @property
    def freshness(self) -> str:
        source_path = Path(inspect.getfile(self.__class__)).resolve()
        return source_freshness(self.__class__.__module__, source_path)


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

    def generate_benchmark_function(
        self, dataset: TDataset, problem: DataInstance, benchmark: Callable[..., Any]
    ) -> Callable[..., Any]:
        """Return the function that computes ``dataset`` with ``problem``'s inputs.

        The result is called as ``function(xp, meta, *inputs)``. By default it is
        ``benchmark``, the benchmark's bound ``benchmark`` method. Generators
        whose datasets each need their own fixed-arity signature (for example,
        one parameter per matrix of a query) override this to build that
        function at setup time, typically with ``saps.codegen.define_function``.
        """
        return benchmark

    @property
    def metadata(self) -> dict[str, Any]:
        return {
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
    def benchmark(
        self, xp: Framework, meta: Any, *data_args: Any, **kwargs: Any
    ) -> Any:
        """Run the benchmark on the generator's inputs.

        Declare one positional parameter per input array after ``meta``, in the
        order the generator produces them (``def benchmark(self, xp, meta, A, b)``),
        or ``*data_args`` when the number of inputs varies by dataset. Return a
        single output directly (``return x``) or several as a tuple
        (``return x, y``).

        The trailing ``*data_args``/``**kwargs`` let type checkers accept either
        style of override; SAPS itself only passes inputs positionally.
        """

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

    def setup(self, param, *, use_cache: bool = True, xp: Framework | None = None):
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format="%(levelname)s %(name)s: %(message)s",
            )
        log_path = os.environ.get("SAPS_LOG_PATH")
        work_log = logging.getLogger("saps.work")
        if log_path and not work_log.handlers:
            Path(log_path).parent.mkdir(parents=True, exist_ok=True)
            work_log.setLevel(logging.INFO)
            work_log.propagate = False
            work_log.addHandler(logging.FileHandler(log_path))
        report_status(self.name, repr(param), "setup")
        self._measurement_timeout = float(
            os.environ.get("SAPS_MEASUREMENT_TIMEOUT", "0")
        )
        _apply_memory_limit()
        if os.environ.get("SAPS_CACHE_DATASETS"):
            if param.generator.cacheable and not param.generator.backend.upload_dataset(
                param.generator, param.dataset
            ):
                raise RuntimeError(
                    "Failed to cache dataset "
                    f"{param.generator.name}.{param.dataset.name}"
                )
            raise NotImplementedError("dataset cached")
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
            function = param.generator.generate_benchmark_function(
                param.dataset, problem, self.benchmark
            )

            def benchmark(meta, *data_args):
                return function(xp, meta, *data_args)

            benchmark.saps_metadata = {  # type: ignore[attr-defined]
                "benchmark_name": self.name,
                "generator_name": param.generator.name,
                "dataset_name": param.dataset.name,
                "dataset_file": param.dataset.file,
                "dataset_freshness": param.dataset.freshness,
            }
            self._compiled_benchmark = xp.compile(benchmark)
        report_status(self.name, repr(param), "measurement")
        if self._measurement_timeout:
            # ASV can repeat setup/teardown between samples of one dataset, so
            # the budget accumulates until the dataset changes.
            if getattr(self, "_measurement_param", None) != repr(param):
                self._measurement_param = repr(param)
                self._measurement_elapsed = 0.0
            self._measurement_started = time.perf_counter()

    def run(self, param):
        if not hasattr(self, "_xp") or not hasattr(self, "_compiled_benchmark"):
            raise RuntimeError(
                "Benchmark.setup must bind a framework before run. Pass xp to "
                "setup or set SAPS_FRAMEWORK."
            )
        xp = self._xp
        input = [xp.from_binsparse(d) for d in self._input]
        output = self._compiled_benchmark(self._meta, *input)
        if not isinstance(output, tuple):
            output = (output,)
        output = [xp.to_binsparse(o) for o in output]
        self._output = output

    def teardown(self, param):
        measurement_timed_out = False
        started = self.__dict__.pop("_measurement_started", None)
        if started is not None:
            # Exclude each setup and correctness check from the total.
            self._measurement_elapsed += time.perf_counter() - started
            measurement_timed_out = (
                self._measurement_elapsed > self._measurement_timeout
            )
            if measurement_timed_out:
                report_status(
                    self.name,
                    repr(param),
                    "measurement",
                    status="measurement_timeout",
                    elapsed_seconds=self._measurement_elapsed,
                    timeout_seconds=self._measurement_timeout,
                )
        report_status(self.name, repr(param), "teardown")
        if hasattr(self, "_output"):
            if not measurement_timed_out:
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
        report_status(self.name, repr(param), "finished")
        if measurement_timed_out:
            raise MeasurementTimeout(
                f"measurement_timeout: {self._measurement_elapsed:.6g}s of measurement "
                f"exceeded {self._measurement_timeout:g}s"
            )

    @abstractmethod
    def check(self, param): ...

    @property
    def metadata(self) -> dict[str, Any]:
        return {
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


class ShellBenchmark(Benchmark, ABC):
    @property
    @abstractmethod
    def generator(self) -> Generator[Any]: ...

    @property
    def name(self) -> str:
        return f"{self.generator.name}_shell"

    @property
    def pretty_name(self) -> str:
        return self.generator.pretty_name

    @property
    def description(self) -> str:
        return ""

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
        return []

    @property
    def ai_disclosure(self) -> str:
        return ""

    @property
    def motivation(self) -> str:
        return ""

    @property
    def generators(self) -> list[Generator[Any]]:
        return [self.generator]

    def benchmark(self, xp: Framework, meta: Any, *data_args: Any) -> Any:
        return ()

    def check(self, param):
        pass


class ReferenceDataset(Dataset, Generic[TDataset]):
    """A dataset whose reference outputs are cached apart from its inputs.

    It keeps the dataset's name and freshness, so its cache key and staleness
    match the dataset's, but joins no suites: it exists only to prepare caches.
    """

    def __init__(self, dataset: TDataset):
        self.dataset = dataset

    @property
    def name(self) -> str:
        return self.dataset.name

    @property
    def pretty_name(self) -> str:
        return self.dataset.pretty_name

    @property
    def description(self) -> str:
        return self.dataset.description

    @property
    def suites(self) -> list[str]:
        return []

    @property
    def concepts(self) -> str:
        return self.dataset.concepts

    @property
    def file(self) -> str:
        return self.dataset.file

    @property
    def freshness(self) -> str:
        return self.dataset.freshness


class ReferenceGenerator(Generator[ReferenceDataset[TDataset]], ABC):
    """Caches the expensive parts of an uncached generator's instances.

    A generator whose inputs come from cached shell generators stays uncached,
    so each shared input is stored once. Its expensive parts, such as reference
    outputs, training targets, or a factored preconditioner, live in a sibling
    `ReferenceGenerator` whose `datasets` wrap the generator's own in
    `ReferenceDataset` and whose `generate` returns a `DataInstance` holding only
    those parts. The generator's `generate` merges in
    `XReferenceGenerator().cached_generate(ReferenceDataset(dataset))`. Wrap each
    reference generator in a `ShellBenchmark` so that dataset caching prepares
    it. Test runs prepare no references, so test cases belong in test generators
    that compute inline.
    """

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
        return []

    @property
    def ai_disclosure(self) -> str:
        return ""

    @property
    def motivation(self) -> str:
        return ""
