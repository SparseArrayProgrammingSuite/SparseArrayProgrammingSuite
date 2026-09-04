from __future__ import annotations

import importlib
import inspect
import pkgutil
import xml.etree.ElementTree as ET
from collections.abc import Iterator

import pytest

import saps
import saps.benchmarks
from saps.benchmark import Tagged


def _benchmark_instances() -> Iterator[saps.Benchmark]:
    seen = set()
    for module_info in pkgutil.iter_modules(saps.benchmarks.__path__):
        module = importlib.import_module(f"saps.benchmarks.{module_info.name}")
        for _, cls in inspect.getmembers(module, inspect.isclass):
            if cls in seen or inspect.isabstract(cls):
                continue
            if not issubclass(cls, saps.Benchmark) or cls is saps.Benchmark:
                continue
            seen.add(cls)
            try:
                yield cls()
            except (TypeError, ValueError):
                continue


def _tagged_objects() -> Iterator[tuple[str, Tagged]]:
    for benchmark in _benchmark_instances():
        yield benchmark.name, benchmark
        for generator in benchmark.generators:
            yield f"{benchmark.name} / {generator.name}", generator
            for dataset in generator.datasets:
                yield (
                    f"{benchmark.name} / {generator.name} / {dataset.name}",
                    dataset,
                )


@pytest.mark.parametrize(("owner", "tagged"), list(_tagged_objects()))
def test_concepts_are_valid_ccs_xml(owner: str, tagged: Tagged):
    try:
        root = ET.fromstring(tagged.concepts)
    except ET.ParseError as exc:
        pytest.fail(f"{owner} concepts are not valid XML: {exc}")

    assert root.tag == "ccs2012", f"{owner} concepts root must be <ccs2012>"

    for child in root:
        assert child.tag == "concept", (
            f"{owner} concepts may only contain <concept> children"
        )
        concept_desc = child.find("concept_desc")
        assert concept_desc is not None, (
            f"{owner} concept entries must include <concept_desc>"
        )
        assert concept_desc.text and concept_desc.text.strip(), (
            f"{owner} concept_desc entries must contain text"
        )
