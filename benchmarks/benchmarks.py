import json
import os
import inspect
import textwrap
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path

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

        conf_dir = os.environ.get("ASV_CONF_DIR")
        if not conf_dir:
            return None

        config_path = Path(conf_dir) / "asv.conf.json"
        if not config_path.exists():
            return None

        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None

        results_dir = config.get("results_dir", "results")
        if not isinstance(results_dir, str) or not results_dir:
            return None

        sidecar = Path(conf_dir) / results_dir / "benchmarks_extra.json"

        entry = {
            "id": f"{cls.__module__}.{cls.__name__}",
            "module": cls.__module__,
            "class_name": cls.__name__,
            "dataset_names": list(instance.dataset_names),
            "param_names": list(getattr(instance, "param_names", [])),
            "authors": [str(a) for a in instance.authors],
            "references": [str(r) for r in instance.references],
            "pretty_name": instance.pretty_name,
            "description": instance.description,
            "motivation": instance.motivation,
            "ai_disclosure": instance.ai_disclosure,
            "pretty_source": instance.pretty_source,
            "benchmark_methods": [
                name
                for name, _ in inspect.getmembers(cls, inspect.isfunction)
                if any(name.startswith(p) for p in cls._ASV_METHOD_PREFIXES)
            ],
        }

        sidecar.parent.mkdir(parents=True, exist_ok=True)

        payload = {"benchmarks": []}
        if sidecar.exists():
            try:
                payload = json.loads(sidecar.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                payload = {"benchmarks": []}

        existing = payload.get("benchmarks", [])
        by_id = {item.get("id"): item for item in existing if isinstance(item, dict)}
        by_id[entry["id"]] = entry
        payload["benchmarks"] = sorted(by_id.values(), key=lambda item: item["id"])

        tmp_path = sidecar.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        tmp_path.replace(sidecar)

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

    param_names = ["dataset"]

    @abstractmethod
    def setup(self, dataset): ...

    @abstractmethod
    def run(self, dataset): ...

    def mem_run(self, dataset):
        self.run(dataset)

    def time_run(self, dataset):
        self.run(dataset)

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

    def run(self, dataset):
        d = self.d
        for key in range(dataset):
            d[key]


class MemSuite:
    def mem_list(self):
        return [0] * 256
